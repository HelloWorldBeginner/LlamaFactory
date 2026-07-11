# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Diagnostic dump for CP precision debugging. Gated by env vars, no-op by default:
#   CP_DUMP_STEPS=356,373,383   -> steps to dump
#   CP_DUMP_DIR=cp_dumps        -> output dir (default "cp_dumps")
#
# Dumps, per target step:
#   shapes_step_<n>.txt  - raw micro-batch shape / valid tokens / round_pad
#   tokens_step_<n>.txt  - per-token CE distribution + logit stats (outlier check)
#
# Only global rank 0 writes files; all ranks print to stdout.

import os
from typing import Any

import torch

IGNORE_INDEX = -100

_cur_step: int = -1
_dump_dir: str = "cp_dumps"


def _targets() -> set[int]:
    raw = os.environ.get("CP_DUMP_STEPS", "")
    return {int(x) for x in raw.replace(" ", "").split(",") if x}


def _active() -> bool:
    return _cur_step in _targets()


def set_step(step: int, dump_dir: str | None = None) -> None:
    """Set the current global step (call from fit() each step)."""
    global _cur_step, _dump_dir
    _cur_step = step
    if dump_dir:
        _dump_dir = dump_dir


def _path(name: str) -> str:
    os.makedirs(_dump_dir, exist_ok=True)
    return os.path.join(_dump_dir, name)


def _rank() -> int:
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        return 0


def dump_batch_shapes(micro_batch: Any, cp_size: int, dp_size: int) -> None:
    """Dump raw micro-batch shape / valid / round_pad. Call from fit() per micro-batch."""
    if not _active():
        return
    ids = micro_batch.get("input_ids")
    labels = micro_batch.get("labels")
    if not isinstance(ids, torch.Tensor):
        return
    B, L = ids.shape[0], ids.shape[-1]
    valid = int((labels != IGNORE_INDEX).sum().item()) if isinstance(labels, torch.Tensor) else -1
    lens: list[int] = []
    if isinstance(labels, torch.Tensor):
        for b in range(B):
            nz = (labels[b] != IGNORE_INDEX).nonzero(as_tuple=False)
            lens.append(int(nz[-1]) + 1 if nz.numel() else 0)
    max_len = max(lens) if lens else int(L)
    round_pad = (cp_size - max_len % cp_size) % cp_size if cp_size > 1 else 0
    line = (
        f"[step {_cur_step} rank {_rank()}] B={B} L={L} valid={valid} "
        f"per_sample_len={lens} max_real_len={max_len} round_pad(cp={cp_size})={round_pad} dp={dp_size}"
    )
    print("[cp_dump] " + line)
    if _rank() == 0:
        with open(_path(f"shapes_step_{_cur_step}.txt"), "a") as f:
            f.write(line + "\n")


def dump_token_stats(
    log_probs: torch.Tensor,
    logits: torch.Tensor,
    cp_rank: int = 0,
    tag: str = "",
) -> None:
    """Dump per-token CE distribution + logit stats.

    log_probs: per-token -CE (shifted), shape [B, L-1] (already ignores pad via ignore_index=IGNORE_INDEX).
    """
    if not _active():
        return
    with torch.no_grad():
        ce = (-log_probs).float().reshape(-1)
        valid_ce = ce[ce > 0]
        lg = logits.detach().float().reshape(-1)
        stats: list[str] = []
        stats.append(f"logits_max_abs={float(lg.abs().max()):.3f}")
        stats.append(f"logits_mean_abs={float(lg.abs().mean()):.3f}")
        if valid_ce.numel() > 0:
            stats.append(f"ce_max={float(valid_ce.max()):.3f}")
            stats.append(f"ce_p99={float(torch.quantile(valid_ce, 0.99)):.3f}")
            stats.append(f"ce_p95={float(torch.quantile(valid_ce, 0.95)):.3f}")
            stats.append(f"ce_mean={float(valid_ce.mean()):.3f}")
            stats.append(f"ce_n={int(valid_ce.numel())}")
            k = min(10, valid_ce.numel())
            topk = torch.topk(valid_ce, k).values.tolist()
            stats.append("ce_top10=" + ",".join(f"{x:.3f}" for x in topk))
        else:
            stats.append("ce_n=0")
        line = f"[step {_cur_step} rank {_rank()} cp_rank {cp_rank} tag={tag}] " + " ".join(stats)
    print("[cp_dump] " + line)
    if _rank() == 0:
        with open(_path(f"tokens_step_{_cur_step}.txt"), "a") as f:
            f.write(line + "\n")
