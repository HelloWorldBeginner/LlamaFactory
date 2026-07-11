#!/usr/bin/env bash
# Launch v1 Ulysses sequence-parallel training.
#
# Usage:
#   bash run_ulysses.sh                       # default: 4 GPUs, cp_size=2 (DP=2, CP=2)
#   NPROC=8 CP_SIZE=4 bash run_ulysses.sh     # 8 GPUs, cp_size=4 (DP=2, CP=4)
#   CONFIG=path/to/your.yaml bash run_ulysses.sh
#
# Constraints:
#   - NPROC must be divisible by CP_SIZE  (dp_size = NPROC / CP_SIZE)
#   - cp_size must divide the model's num_attention_heads
#   - requires flash_attn: flash_attention_2 in the YAML


ps -ef |grep -i python |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef |grep -i torchrun |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
source /home/cann/9.0.1.b020/cann/set_env.sh
source /home/cann/9.0.1.b020/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export CP_DEBUG_DUMP_DIR="cp_debug_dumps_cp1"
set -euo pipefail


export PYTHONPATH=/home/m00611744/lf_0629/LlamaFactory_0710/src/:$PYTHONPATH


# export CP_DEBUG_SEQ_LEN=4096
# export CP_DEBUG_MAX_STEPS=1
# export CP_DEBUG=0
# export CP_DEBUG_AUTO_SEQ_LEN=1
# export CP_DEBUG_PAD_TO_CUTOFF=0
# export CP_DEBUG_DP_RANK=0,1,2,3
#export CP_DEBUG_RAW_PRINT=1
# export CP_ALIGN_ROUND_PAD=2
# export DISABLE_SHUFFLE=1
# export CP_DEBUG_STEPS=1

#export CP_ALIGN_ROUND_PAD=2
mkdir -p logs_cp

USE_V1=1 python -m llamafactory.cli sft examples/v1/train_full/train_full_ulysses_cp_own_cp1.yaml 2>&1 | tee "logs_cp/cp1_$(date +%Y%m%d_%H%M).log"