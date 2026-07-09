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

import os
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
    get_ulysses_sequence_parallel_world_size,
    set_ulysses_sequence_parallel_group,
)
from .seq_comm import SeqAllToAll4D

# 尝试导入 MindSpeed 的 all-to-all 接口（NPU 环境可用）；不可用时回退 SeqAllToAll4D
try:
    from mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_context_parallel.utils import (
        gather_heads_scatter_seq as _ms_gather_heads_scatter_seq,
        gather_seq_scatter_heads as _ms_gather_seq_scatter_heads,
    )
    _HAS_MINDSPEED_A2A = True
except Exception:
    _HAS_MINDSPEED_A2A = False


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
    num_attention_heads=None,
    num_key_value_heads=None,
    **kwargs,
):
    if mode == "ulysses":
        if num_attention_heads is not None and num_key_value_heads is not None:
            num_groups = num_attention_heads // num_key_value_heads
            if num_groups > 1:
                key_states = torch.repeat_interleave(key_states, dim=2, repeats=num_groups)
                value_states = torch.repeat_interleave(value_states, dim=2, repeats=num_groups)

        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        # Pop kwargs that UlyssesAttention handles explicitly, forward the rest
        # (sliding_window, softcap, etc.) to attn_fn so CP attention matches non-CP.
        position_ids = kwargs.pop("position_ids", None)
        softmax_scale = kwargs.pop("softmax_scale", None)
        kwargs.pop("query_length", None)  # HF passes local length; we use global length below
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=query_states.shape[1] * sequence_parallel_size,
            deterministic=deterministic,
            dropout_p=dropout,
            causal=is_causal,
            position_ids=position_ids,
            softmax_scale=softmax_scale,
            target_dtype=target_dtype,
            **kwargs,
        )
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


def _rebuild_full_eager_mask(attention_mask, group, cp_size, full_seq, dtype, device):
    """从 CP 本地 4D mask 重建全长 4D causal+padding additive mask。

    eager 用 4D additive mask（masked=fininfo.min，unmasked=0），无 is_causal 标志。
    CP all-to-all 把 seq 拼回全长后，需要 [bs,1,full,full] 的 causal+padding mask。
    本地 mask 只覆盖本地 seq 的 causal，无法直接拼；故：
    1. 从本地 mask 抽出 padding key（对所有 query 都 mask 的 key）→ all-gather 拼全长 padding。
    2. 重建全长 causal（triu）+ padding → 4D additive mask。
    """
    min_val = torch.finfo(dtype).min
    if attention_mask is not None:
        # attention_mask: [bs,1,seq_local,seq_local]；padding key = 所有 query 都被 mask
        pad_local = (attention_mask[:, 0] < 0).all(dim=1).to(torch.int64)  # [bs, seq_local]
        bs = pad_local.shape[0]
        gathered = [torch.empty_like(pad_local) for _ in range(cp_size)]
        dist.all_gather(gathered, pad_local, group=group)
        pad_full = torch.cat(gathered, dim=-1).to(torch.bool)  # [bs, full_seq]
    else:
        bs = 1
        pad_full = torch.zeros((bs, full_seq), dtype=torch.bool, device=device)
    # causal: 上三角 = masked；padding key = masked。用 bool OR + where，不能用 maximum
    # （maximum(负数 min_val, 0) = 0，会冲掉 causal 掩码 → 无 causal → forward 全错）
    causal_masked = torch.triu(torch.ones((full_seq, full_seq), dtype=torch.bool, device=device), diagonal=1)
    pad_masked = pad_full[:, None, None, :].expand(bs, 1, full_seq, full_seq)
    masked = causal_masked[None, None, :, :].expand(bs, 1, full_seq, full_seq) | pad_masked
    full_mask = torch.where(masked, min_val, torch.zeros((), dtype=dtype, device=device))
    return full_mask


def new_eager_attn_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    attn_fn=None,
    group=None,
    **kwargs,
):
    """Ulysses CP 包装（eager 后端）。

    - all-to-all: 优先用 MindSpeed 的 gather_seq_scatter_heads / gather_heads_scatter_seq，
      回退 SeqAllToAll4D。
    - mask: 用传进来的 attention_mask 重建全长 4D causal+padding（_rebuild_full_eager_mask）。
    - GQA: 预复制 K/V（和 FA2 路径一致），all-to-all 后头数对齐。
    """
    cp_size = get_ulysses_sequence_parallel_world_size(group)
    if not getattr(new_eager_attn_forward, "_confirmed", False):
        new_eager_attn_forward._confirmed = True
        if dist.is_initialized() and dist.get_rank() == 0:
            a2a = "MindSpeed gather_seq_scatter_heads" if _HAS_MINDSPEED_A2A else "SeqAllToAll4D"
            print(f"[CP] new_eager_attn_forward 已被调用 —— eager CP 生效，通信算子: {a2a}", flush=True)

    # GQA 预复制（和 FA2 路径、MindSpeed 一致）
    num_attention_heads = module.config.num_attention_heads
    num_key_value_heads = module.config.num_key_value_heads
    num_groups = num_attention_heads // num_key_value_heads
    if num_groups > 1:
        key = torch.repeat_interleave(key, dim=1, repeats=num_groups)
        value = torch.repeat_interleave(value, dim=1, repeats=num_groups)

    # all-to-all: [bs, heads, seq_local, head_dim] -> [bs, heads/cp, full_seq, head_dim]
    full_seq = query.shape[2] * cp_size
    if _HAS_MINDSPEED_A2A:
        q = _ms_gather_seq_scatter_heads(query, seq_dim=2, head_dim=1, gather_size=full_seq, group=group)
        k = _ms_gather_seq_scatter_heads(key, seq_dim=2, head_dim=1, gather_size=full_seq, group=group)
        v = _ms_gather_seq_scatter_heads(value, seq_dim=2, head_dim=1, gather_size=full_seq, group=group)
    else:
        q = SeqAllToAll4D.apply(group, query, 1, 2)
        k = SeqAllToAll4D.apply(group, key, 1, 2)
        v = SeqAllToAll4D.apply(group, value, 1, 2)
    full_seq = q.shape[2]

    # mask: 用传进来的 attention_mask 重建全长 4D causal+padding（还原，不用纯 causal）
    mask_dtype = attention_mask.dtype if attention_mask is not None else q.dtype
    full_mask = _rebuild_full_eager_mask(attention_mask, group, cp_size, full_seq, mask_dtype, q.device)

    attn_output, _ = attn_fn(module, q, k, v, full_mask, scaling, dropout, **kwargs)
    # eager 内部 transpose(1,2) → [bs, full_seq, heads/cp, head_dim]；回程 scatter seq(dim1)、gather heads(dim2)
    if _HAS_MINDSPEED_A2A:
        output = _ms_gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1, gather_size=num_attention_heads, group=group)
    else:
        output = SeqAllToAll4D.apply(group, attn_output, 1, 2)
    return output, None


@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, model_args):
    # Replace attention forward with Ulysses CP wrapper, dispatched by _attn_implementation.
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

    attn_impl = getattr(model.config, "_attn_implementation", "flash_attention_2")
    group = get_ulysses_sequence_parallel_group()

    if attn_impl == "flash_attention_2":
        # FA2: patch _flash_attention_forward（integrations 的 flash_attention_forward 内部调它）
        origin_attn = transformers.modeling_flash_attention_utils._flash_attention_forward
        new_flash_attention_forward = partial(
            new_flash_attn_forward,
            group=group,
            mode="ulysses",
            attn_fn=origin_attn,
            sequence_parallel_size=cp_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        for module_name, mod in list(sys.modules.items()):
            try:
                if (
                    hasattr(mod, "__file__")
                    and "transformers" in mod.__file__
                    and getattr(mod._flash_attention_forward, "__name__", "") == "_flash_attention_forward"
                ):
                    mod._flash_attention_forward = new_flash_attention_forward
                    logger.info_rank0(
                        f"Replaced _flash_attention_forward in module {module_name} with new_flash_attn_forward for sequence parallel."
                    )
            except (AttributeError, TypeError):
                continue
    elif attn_impl in ("eager",):
        # eager: patch modeling 模块的 eager_attention_forward（get_interface("eager", default) 用它）。
        # FSDP2 包裹后 model.__module__ 是 fsdp 模块，要遍历子模块找真正的 modeling 模块。
        eager_mod = None
        origin_eager = None
        for sub in model.modules():
            mod = sys.modules.get(type(sub).__module__)
            if mod is not None and hasattr(mod, "eager_attention_forward"):
                eager_mod = mod
                origin_eager = getattr(mod, "eager_attention_forward")
                break
        if origin_eager is None:
            raise NotImplementedError(
                "CP eager needs `eager_attention_forward` in the model's modeling module; not found. "
                "确认模型支持 eager 后端。"
            )
        new_eager_attention_forward = partial(new_eager_attn_forward, attn_fn=origin_eager, group=group)
        eager_mod.eager_attention_forward = new_eager_attention_forward
        logger.info_rank0(
            f"Replaced eager_attention_forward in {eager_mod.__name__} with new_eager_attn_forward for sequence parallel."
        )
    else:
        raise NotImplementedError(
            f"Sequence parallel (Ulysses) 目前仅支持 flash_attention_2 / eager 后端，当前 _attn_implementation={attn_impl!r}"
        )


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
                real_pad = max_data_len - v.shape[-1]
                round_pad = (cp_size - max_data_len % cp_size) % cp_size

                if k in ("labels", "shift_labels"):
                    pad_data = F.pad(v, (0, real_pad + round_pad), value=-100)
                elif k in ("loss_weights", "shift_loss_weights"):
                    pad_data = F.pad(v, (0, real_pad + round_pad), value=0.0)
                elif k == "attention_mask":
                    pad_data = F.pad(v, (0, real_pad), value=0)
                    if round_pad > 0:
                        pad_data = F.pad(pad_data, (0, round_pad), value=1)
                elif k == "position_ids":
                    pad_data = F.pad(v, (0, real_pad), value=0)
                    if round_pad > 0:
                        last_pos = pad_data[..., -1:]
                        round_pos = last_pos + torch.arange(1, round_pad + 1, device=v.device, dtype=v.dtype)
                        pad_data = torch.cat([pad_data, round_pos], dim=-1)
                else:
                    pad_data = F.pad(v, (0, real_pad + round_pad), value=0)

                data[k] = torch.chunk(pad_data, chunks=cp_size, dim=-1)[cp_rank].contiguous()
    return data


def round_pad_data(data, align_size=1):
    """仅 round-pad 到 align_size 倍数（不切分），用于让 CP1（cp_size=1）数据形状对齐 CP2。

    pad 值与 padding_and_split_data 的 round_pad 段一致：attention_mask=1、labels=-100、
    position_ids 续位、loss_weights=0、其余 0。这些位置 causal 隔离、不参与 loss。
    诊断用：CP1 设 env CP_ALIGN_ROUND_PAD=2 调用此函数，验证 round_pad 是否为精度差异来源。
    """
    if align_size <= 1:
        return data
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.ndim > 1:
            round_pad = (align_size - v.shape[-1] % align_size) % align_size
            if round_pad == 0:
                continue
            if k in ("labels", "shift_labels"):
                data[k] = F.pad(v, (0, round_pad), value=-100)
            elif k in ("loss_weights", "shift_loss_weights"):
                data[k] = F.pad(v, (0, round_pad), value=0.0)
            elif k == "attention_mask":
                data[k] = F.pad(v, (0, round_pad), value=1)
            elif k == "position_ids":
                last_pos = v[..., -1:]
                cont = last_pos + torch.arange(1, round_pad + 1, device=v.device, dtype=v.dtype)
                data[k] = torch.cat([v, cont], dim=-1)
            else:
                data[k] = F.pad(v, (0, round_pad), value=0)
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

    # 诊断：写死 attention_mask=None，规避 mask 重建差异（mbs=1 无 padding 时安全）
    if os.environ.get("CP_NO_ATTENTION_MASK", "0") == "1":
        model_inputs.pop("attention_mask", None)

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
