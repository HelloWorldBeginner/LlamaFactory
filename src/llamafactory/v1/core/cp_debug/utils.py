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
#
# Vendored from the cp-precision-debug skill (MIT) for CP1/CP2 precision debugging.

"""CP Debug utility functions."""

from typing import Any, Optional

import torch
import torch.distributed as dist


def detect_seq_dim(tensor: torch.Tensor, expected_seq_len: int, cp_size: int = 1) -> Optional[int]:
    """Auto-detect the sequence dimension by matching shape.

    Args:
        tensor: input tensor (forward activations; weights/grads must not call this)
        expected_seq_len: full sequence length
        cp_size: CP size

    Returns:
        Sequence dimension index, or None if not found.

    Note:
        Judged solely by "some dim length equals the full or sharded sequence
        length", which has a coincidence risk when hidden_size == seq_len. The
        caller already skips this function for weight / param_grad.
    """
    if expected_seq_len is None:
        return None

    targets = {expected_seq_len}
    if cp_size > 1:
        targets.add(expected_seq_len // cp_size)

    # Prefer the sharded length (CP>1 local tensor's seq dim is the sharded length)
    sharded = expected_seq_len // cp_size if cp_size > 1 else None
    for dim, size in enumerate(tensor.shape):
        if sharded is not None and size == sharded:
            return dim
    for dim, size in enumerate(tensor.shape):
        if size == expected_seq_len:
            return dim

    return None


def all_gather_seq(
    tensor: torch.Tensor,
    cp_group: Optional[Any],
    expected_seq_len: Optional[int] = None,
    tensor_name: str = "",
    default_seq_dim: int = 1,
) -> torch.Tensor:
    """All-gather a tensor along the sequence dimension.

    Args:
        tensor: input tensor
        cp_group: CP process group
        expected_seq_len: full sequence length (used to auto-detect seq_dim)
        tensor_name: tensor name (for logging)
        default_seq_dim: default sequence dimension (current impl only auto-detects)

    Returns:
        Gathered tensor.
    """
    del default_seq_dim  # auto-detection only

    # No CP or no gather needed
    if cp_group is None or not dist.is_initialized():
        return tensor

    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return tensor

    # Uneven split warning: differing per-rank seq lengths break all_gather shape
    if expected_seq_len is not None and expected_seq_len % cp_size != 0:
        if tensor_name:
            print(
                f"[WARN] {tensor_name}: expected_seq_len={expected_seq_len} "
                f"not divisible by cp_size={cp_size}; all-gather may fail, "
                f"falling back to local tensor"
            )
        return tensor

    seq_dim = detect_seq_dim(tensor, expected_seq_len, cp_size)

    if seq_dim is None:
        if tensor_name:
            print(f"[WARN] {tensor_name}: no seq_dim found, skipping all-gather")
        return tensor

    try:
        gathered = [torch.zeros_like(tensor) for _ in range(cp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=cp_group)
        return torch.cat(gathered, dim=seq_dim)
    except Exception as e:
        if tensor_name:
            print(f"[ERROR] {tensor_name}: all-gather failed: {e}")
        return tensor


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Compute tensor statistics."""
    with torch.no_grad():
        t = tensor.detach().float()
        flat = t.flatten()
        return {
            "shape": list(tensor.shape),
            "mean": t.mean().item(),
            "std": t.std().item() if flat.numel() > 1 else 0.0,
            "min": t.min().item(),
            "max": t.max().item(),
            "first5": flat[:5].tolist(),
            "last5": flat[-5:].tolist(),
        }


def format_stats(name: str, stats: dict[str, Any], step: int) -> str:
    """Format statistics output."""
    return (
        f"[STEP {step}] {name}: "
        f"shape={stats['shape']} "
        f"mean={stats['mean']:.6f} "
        f"std={stats['std']:.6f} "
        f"min={stats['min']:.6f} "
        f"max={stats['max']:.6f} "
        f"first5={stats['first5']} "
        f"last5={stats['last5']}"
    )
