#!/usr/bin/env bash
# Launch v1 Ulysses sequence-parallel SFT training.
#
# IMPORTANT:
#   - v1's launcher AUTO-SPAWNS torchrun when >1 GPU is visible, so do NOT wrap
#     this in torchrun yourself (that causes a nested launch / port conflict).
#   - The v1 SFT command is `sft`, NOT `train`.
#   - cp_size is configured in the YAML (dist_config.cp_size). This script only
#     controls how many GPUs are used via NPROC_PER_NODE.
#
# Usage:
#   bash run_ulysses.sh                            # use all visible GPUs
#   NPROC_PER_NODE=4 bash run_ulysses.sh           # limit to 4 GPUs
#   CONFIG=path/to/your.yaml bash run_ulysses.sh
#
# Constraints (cp_size lives in the YAML):
#   - NPROC_PER_NODE must be divisible by cp_size   (dp_size = NPROC / cp_size)
#   - cp_size must divide the model's num_attention_heads
#   - YAML must have flash_attn: flash_attention_2
set -e

export USE_V1=1
# Enable PyTorch allocator tuning (expandable_segments) to reduce fragmentation OOM.
# The v1 launcher translates this into PYTORCH_CUDA_ALLOC_CONF / PYTORCH_NPU_ALLOC_CONF.
export OPTIM_TORCH=1

# Make the local source tree importable as `import llamafactory` (src layout).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Default to all visible accelerators (CUDA or NPU); override with NPROC_PER_NODE.
# `torch.accelerator.device_count()` is accelerator-agnostic (torch>=2.7), unlike
# `torch.cuda.device_count()` which returns 0 on Ascend NPU and would silently fall back to 1.
if [ -z "${NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE=$(python -c "import torch; print(torch.accelerator.device_count() or 1)" 2>/dev/null || echo 1)
fi
export NPROC_PER_NODE

CONFIG=${CONFIG:-examples/v1/train_full/train_full_ulysses_cp.yaml}

echo "[INFO] USE_V1=1  NPROC_PER_NODE=$NPROC_PER_NODE  CONFIG=$CONFIG"
echo "[INFO] Reminder: NPROC_PER_NODE must be divisible by cp_size in the YAML."

# Prefer the local source tree (where the SP code lives). If llamafactory is not
# importable, try `uv run` (editable install) so the local code is used.
if python -c "import llamafactory" 2>/dev/null; then
    python -m llamafactory.cli sft "$CONFIG"
else
    echo "[INFO] llamafactory not importable on PATH; trying 'uv run' (editable install)..."
    uv run python -m llamafactory.cli sft "$CONFIG"
fi
