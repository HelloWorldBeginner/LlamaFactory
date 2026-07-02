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

import sys
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

from ....accelerator.helper import get_current_device
from ....accelerator.interface import Dim, DistributedInterface
from ....utils import logging
from ....utils.plugin import BasePlugin
from ....utils.types import ModelOutput
from .ulysses import (
    UlyssesAttention,
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)


logger = logging.get_logger(__name__)


class SequenceParallelModelPlugin(BasePlugin):
    def __call__(self, model, model_args):
        return super().__call__(model, model_args)


class SequenceParallelLossPlugin(BasePlugin):
    def __call__(self, model, inputs, *args, **kwargs):
        return super().__call__(model, inputs, *args, **kwargs)


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    sequence_parallel_size=1,
    dropout=0,
    deterministic=False,
    is_causal=True,
    group=None,
    mode="ulysses",
    attn_fn=None,
    target_dtype=None,
    **kwargs,
):
    if mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=query_states.shape[1] * sequence_parallel_size,
            deterministic=deterministic,
            dropout_p=dropout,
            causal=is_causal,
            position_ids=kwargs.get("position_ids", None),
            target_dtype=target_dtype,
        )
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, model_args):
    # Replace _flash_attention_forward with new_flash_attn_forward
    module = sys.modules[model.__module__]
    cp_size = model_args.get("cp_size", 1)

    set_ulysses_sequence_parallel_group(DistributedInterface().get_group(Dim.CP))

    try:
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert num_attention_heads % cp_size == 0, "num_attention_heads must be divisible by cp_size"
    assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
        "num_key_value_heads must be divisible by cp_size"
    )

    origin_attn = transformers.modeling_flash_attention_utils._flash_attention_forward
    new_flash_attention_forward = partial(
        new_flash_attn_forward,
        group=get_ulysses_sequence_parallel_group(),
        mode="ulysses",
        attn_fn=origin_attn,
        sequence_parallel_size=cp_size,
    )

    for module_name, module in list(sys.modules.items()):
        try:
            if (
                hasattr(module, "__file__")
                and "transformers" in module.__file__
                and getattr(module._flash_attention_forward, "__name__", "") == "_flash_attention_forward"
            ):
                module._flash_attention_forward = new_flash_attention_forward
                logger.info_rank0(
                    f"Replaced _flash_attention_forward in module {module_name} with new_flash_attn_forward for sequence parallel."
                )
        except (AttributeError, TypeError):
            continue


def padding_and_split_data(data, device_mesh=None):
    if device_mesh is not None:
        cp_size = device_mesh["cp"].size()
        cp_rank = device_mesh["cp"].get_local_rank()
        cp_group = device_mesh["cp"].get_group()
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.ndim > 1:
                data_len = torch.tensor(v.shape[-1], device=v.device, dtype=torch.int64)
                global_data_len = [torch.empty_like(data_len) for _ in range(cp_size)]
                dist.all_gather(global_data_len, data_len, group=cp_group)
                max_data_len = max(global_data_len)
                pad_size = max_data_len - v.shape[-1] + (cp_size - max_data_len % cp_size) % cp_size
                if k in ("labels", "shift_labels"):
                    pad_value = -100
                elif k in ("loss_weights", "shift_loss_weights"):
                    pad_value = 0.0
                else:
                    pad_value = 0
                pad_data = F.pad(v, (0, pad_size), value=pad_value)
                data[k] = torch.chunk(pad_data, chunks=cp_size, dim=-1)[cp_rank].contiguous()
    return data


@SequenceParallelLossPlugin("sequence_parallel_loss").register()
def sequence_parallel_loss(model, model_inputs):
    device_mesh = DistributedInterface().get_device_mesh(Dim.CP)

    # Move tensors to the current accelerator device (e.g. npu:local_rank).
    current_device = get_current_device()
    model_inputs = {
        k: v.to(current_device, non_blocking=True) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)
    }

    # Shift labels (and loss_weights) BEFORE splitting across CP ranks (MindSpeed style).
    # Shifting on the full sequence then splitting ensures each CP rank's shift_labels
    # includes the boundary token from the next rank, so no all-gather of labels/log_probs
    # is needed — only a scalar all-reduce of the loss sum and token count.
    labels = model_inputs["labels"]
    shift_labels = F.pad(labels[..., 1:], (0, 1), value=-100)
    model_inputs["shift_labels"] = shift_labels

    has_loss_weights = "loss_weights" in model_inputs
    if has_loss_weights:
        loss_weights = model_inputs["loss_weights"]
        shift_loss_weights = F.pad(loss_weights[..., 1:], (0, 1), value=0.0)
        model_inputs["shift_loss_weights"] = shift_loss_weights

    model_inputs = padding_and_split_data(model_inputs, device_mesh)

    # Pop shift_* keys — they are for the loss, not model inputs.
    shift_labels = model_inputs.pop("shift_labels")
    shift_loss_weights = model_inputs.pop("shift_loss_weights", None)

    # Model forward on the local sequence shard.
    outputs: ModelOutput = model(**model_inputs)
    logits = outputs.logits.float()

    # Local CE SUM (scalar) — MindSpeed style: compute per-token loss locally, sum to a scalar,
    # then all-reduce the scalar across CP (not gather the full per-token log_probs tensor).
    shift_logits = logits.view(-1, logits.size(-1))
    shift_labels_flat = shift_labels.view(-1)

    if shift_loss_weights is not None:
        per_token = F.cross_entropy(shift_logits, shift_labels_flat, ignore_index=-100, reduction="none")
        loss = (per_token * shift_loss_weights.view(-1)).sum()
        num_items = shift_loss_weights.sum()
    else:
        loss = F.cross_entropy(shift_logits, shift_labels_flat, ignore_index=-100, reduction="sum")
        num_items = (shift_labels_flat != -100).sum().to(loss.dtype)

    cp_group = get_ulysses_sequence_parallel_group()

    # All-reduce loss and token count across CP (two scalars — lightweight vs gathering log_probs).
    # Must use dist.nn.all_reduce (differentiable): backward of SUM is identity, so each rank gets
    # the correct local gradient d(local_loss)/d(logits) / num_items. dist.all_reduce (in-place)
    # is NOT in the autograd graph and breaks backward on NPU.
    loss = dist.nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=cp_group)
    num_items = dist.nn.all_reduce(num_items, op=dist.ReduceOp.SUM, group=cp_group)

    loss = loss / (num_items + 1e-6)  # global per-token mean

    return loss
