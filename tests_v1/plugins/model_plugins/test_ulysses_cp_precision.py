# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precision test: CP-on vs CP-off must agree.

Ulysses CP only does data movement (all-to-all), so for a sequence whose length is
divisible by cp_size (no padding), the per-token log_probs are bitwise identical to
the full-sequence (CP-off) computation; only the final loss sum order differs.
Hence loss should match to ~1e-5 and gradients to ~1e-3 (bf16 backward).
"""

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.plugins.model_plugins.parallelization.sequence_parallel import (
    SequenceParallelModelPlugin,
    sequence_parallel_loss,
)
from llamafactory.v1.utils.env import find_available_port
from llamafactory.v1.utils.pytest import dist_env


def _full_seq_loss(model, model_inputs):
    """Reference loss on the FULL sequence (CP off), same formula as sequence_parallel_loss."""
    out = model(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        position_ids=model_inputs["position_ids"],
    )
    logits = out.logits.float()
    labels = model_inputs["labels"]
    loss_weights = model_inputs["loss_weights"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_loss_weights = loss_weights[..., 1:].contiguous()
    log_probs = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(labels.size(0), -1)
    loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)
    return loss


def _test_cp_precision(local_rank: int, world_size: int, master_port: int):
    with dist_env(local_rank, world_size, master_port):
        torch.manual_seed(0)
        model_args = ModelArguments(model="llamafactory/tiny-random-qwen3")
        dist_config = {"cp_mode": "ulysses", "cp_size": world_size, "dp_size": 1}
        DistributedInterface(dist_config)
        model_engine = ModelEngine(model_args=model_args)
        model = model_engine.model
        model.eval()  # disable dropout for a deterministic forward
        for p in model.parameters():
            p.requires_grad_(True)

        device = next(model.parameters()).device
        seq_len = 8  # MUST be divisible by cp_size (= world_size) to avoid padding
        input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).view(1, seq_len).to(device)
        full = {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(0, seq_len, dtype=torch.long).view(1, seq_len).to(device),
            "loss_weights": torch.ones(1, seq_len, device=device),
        }

        # pick a 2D weight to compare gradients on
        ref_param = next(p for p in model.parameters() if p.ndim == 2)

        # --- CP off reference (computed BEFORE patching attention) ---
        model.zero_grad(set_to_none=True)
        loss_ref = _full_seq_loss(model, {k: v.clone() for k, v in full.items()})
        loss_ref.backward()
        grad_ref = ref_param.grad.detach().clone()

        # --- CP on: patch attention, then run SP loss (it splits the sequence internally) ---
        SequenceParallelModelPlugin("ulysses")(model, dist_config)
        model.zero_grad(set_to_none=True)
        loss_cp = sequence_parallel_loss(model, {k: v.clone() for k, v in full.items()})
        loss_cp.backward()
        grad_cp = ref_param.grad.detach().clone()

        # Both loss_cp and loss_ref are identical across ranks (loss_cp is all-gathered).
        if local_rank == 0:
            print(
                f"[cp-precision] loss_ref={loss_ref.item():.8f} loss_cp={loss_cp.item():.8f} "
                f"loss_diff={abs(loss_ref.item() - loss_cp.item()):.3e} "
                f"grad_max_diff={(grad_ref - grad_cp).abs().max().item():.3e}"
            )

        assert torch.allclose(loss_ref, loss_cp, rtol=1e-4, atol=1e-5), (
            f"CP loss mismatch: ref={loss_ref.item()} cp={loss_cp.item()}"
        )
        assert torch.allclose(grad_ref, grad_cp, rtol=1e-3, atol=1e-4), (
            f"CP grad mismatch: max_diff={(grad_ref - grad_cp).abs().max().item()}"
        )


@pytest.mark.runs_on(["cuda", "npu"])
@pytest.mark.require_distributed(2)
def test_cp_precision():
    master_port = find_available_port()
    world_size = 2
    mp.spawn(_test_cp_precision, args=(world_size, master_port), nprocs=world_size)
