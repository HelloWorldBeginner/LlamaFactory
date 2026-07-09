# Copyright 2025 Bytedance Ltd. and/or its affiliates. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's verl library.
# https://github.com/verl-project/verl/blob/77476af84cc074edf5a6437f8d5ea418d7a54916/verl/utils/ulysses.py
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

from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch import Tensor


# ============================================================================
# 原始 SeqAllToAll4D（保留，回退用）
# ============================================================================

def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
):
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None]:
        return (
            None,
            all_to_all_tensor(grad_output[0], ctx.gather_dim, ctx.scatter_dim, ctx.group),
            None,
            None,
        )


# ============================================================================
# MindSpeed all-to-all（搬运自 mindspeed.core...unaligned_cp.mapping）
# 支持 aligned/unaligned，用 all_to_all_single（比 list-based all_to_all 更高效）
# ============================================================================

_PERMUTE_DIMS1 = {4: (1, 2, 3, 0), 5: (1, 2, 3, 0, 4)}
_PERMUTE_DIMS2 = {4: (1, 2, 0, 3), 5: (1, 2, 0, 3, 4)}


def _cal_split_sizes(dim_size, world_size):
    split_size = dim_size // world_size
    remainder = dim_size % world_size
    return [split_size + (1 if i < remainder else 0) for i in range(world_size)]


def _adjust_tensor_dimensions(tensor, scatter_idx, gather_idx):
    dims = list(range(tensor.dim()))
    assert scatter_idx != gather_idx
    if gather_idx == 0:
        if scatter_idx != 1:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
        else:
            dims[scatter_idx], dims[gather_idx] = dims[gather_idx], dims[scatter_idx]
    elif gather_idx == 1:
        if scatter_idx != 0:
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
    else:
        if scatter_idx == 0:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
        else:
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
    return tensor.permute(dims).contiguous(), dims


def _unadjust_tensor_dimensions(tensor, adjusted_dims):
    inverse_dims = [0] * len(adjusted_dims)
    for new_pos, old_pos in enumerate(adjusted_dims):
        inverse_dims[old_pos] = new_pos
    return tensor.permute(inverse_dims).contiguous()


def _aligned_all_to_all(input_, group, scatter_dim, gather_dim):
    world_size = dist.get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // world_size
    if scatter_dim == 0:
        input_t = input_.reshape([world_size] + inp_shape).contiguous()
    else:
        input_t = input_.reshape([-1, world_size] + inp_shape[scatter_dim:]).transpose(0, 1).contiguous()
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)
    output = output.view([world_size] + inp_shape).contiguous()
    output_dim = output.dim()
    if gather_dim == 1:
        output = output.transpose(0, 1).contiguous()
    elif gather_dim == 2:
        output = output.permute(*_PERMUTE_DIMS2[output_dim]).contiguous()
    elif gather_dim == 3:
        output = output.permute(*_PERMUTE_DIMS1[output_dim]).contiguous()
    output = output.view(inp_shape[:gather_dim] + [inp_shape[gather_dim] * world_size] + inp_shape[gather_dim + 1:]).contiguous()
    return output


def _full_unaligned_all_to_all(input_, group, scatter_dim, gather_dim, gather_size):
    world_size = dist.get_world_size(group)
    scatter_sizes = _cal_split_sizes(input_.size(scatter_dim), world_size)
    input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
    gather_sizes = _cal_split_sizes(gather_size, world_size)
    output_list = []
    tensor_shape_base = input_list[0].size()
    for i in range(world_size):
        tensor_shape = list(tensor_shape_base)
        tensor_shape[gather_dim] = gather_sizes[i]
        output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _partial_unaligned_all_to_all(input_, group, scatter_dim, gather_dim, gather_size):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group=group)
    input_ = input_.contiguous()
    scatter_size = input_.size(scatter_dim)
    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size
    assert not (gather_size % world_size != 0 and scatter_size % world_size != 0)
    scatter_size_per_rank = scatter_size // world_size
    scatter_size_remainder = scatter_size % world_size
    input_split_sizes = [scatter_size_per_rank + (1 if i < scatter_size_remainder else 0) for i in range(world_size)]
    gather_size_per_rank = gather_size // world_size
    gather_size_remainder = gather_size % world_size
    output_split_sizes = [gather_size_per_rank + (1 if i < gather_size_remainder else 0) for i in range(world_size)]
    reshaped_input, reshaped_input_dims = _adjust_tensor_dimensions(input_, scatter_dim, gather_dim)
    reshaped_input_shape = list(reshaped_input.shape)
    if scatter_size % world_size == 0:
        reshaped_input = reshaped_input.view(
            [world_size, input_.size(scatter_dim) // world_size, input_.size(gather_dim)] + reshaped_input_shape[2:]
        ).transpose(1, 2).contiguous()
    output_dims = reshaped_input_dims
    output_dims[1], output_dims[0] = output_dims[0], output_dims[1]
    output = torch.empty((gather_size, input_split_sizes[rank], *reshaped_input_shape[2:]),
                         dtype=input_.dtype, device=input_.device)
    dist.all_to_all_single(
        output, reshaped_input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes if scatter_size % world_size != 0 else [1 for _ in range(world_size)],
        group=group,
    )
    if gather_size % world_size == 0 and scatter_size % world_size != 0:
        output = output.view(
            [world_size, input_split_sizes[rank], gather_size // world_size] + reshaped_input_shape[2:]
        ).transpose(1, 2).reshape(output.shape).contiguous()
    return _unadjust_tensor_dimensions(output, output_dims)


def _ms_all_to_all(input_, group, scatter_dim, gather_dim, gather_size=None):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_
    scatter_size = input_.size(scatter_dim)
    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size
    gather_mod = gather_size % world_size
    scatter_mod = scatter_size % world_size
    if gather_mod == 0 and scatter_mod == 0:
        return _aligned_all_to_all(input_, group, scatter_dim, gather_dim)
    elif gather_mod != 0 and scatter_mod != 0:
        return _full_unaligned_all_to_all(input_, group, scatter_dim, gather_dim, gather_size)
    else:
        return _partial_unaligned_all_to_all(input_, group, scatter_dim, gather_dim, gather_size)


class _MsAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, gather_size=None):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.scatter_size = input_.size(scatter_dim)
        ctx.gather_dim = gather_dim
        ctx.gather_size = gather_size
        return _ms_all_to_all(input_, process_group, scatter_dim, gather_dim, gather_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad = _ms_all_to_all(grad_output, ctx.process_group, ctx.gather_dim, ctx.scatter_dim, ctx.scatter_size)
        return (grad, None, None, None, None)


def ms_all_to_all(input_, group, scatter_dim, gather_dim, gather_size=None):
    return _MsAllToAll.apply(input_, group, scatter_dim, gather_dim, gather_size)


# ============================================================================
# MindSpeed gather_seq_scatter_heads / gather_heads_scatter_seq
# （搬运自 mindspeed_llm...ulysses_context_parallel.utils，去掉 ParallelState 依赖）
# ============================================================================

def gather_seq_scatter_heads(input_, seq_dim, head_dim, gather_size, group=None):
    """scatter heads (dim=head_dim), gather seq (dim=seq_dim)。
    [bs, heads, seq_local, head_dim] → [bs, heads/cp, full_seq, head_dim]"""
    if group is None or dist.get_world_size(group) == 1:
        return input_
    return ms_all_to_all(input_, group, scatter_dim=head_dim, gather_dim=seq_dim, gather_size=gather_size)


def gather_heads_scatter_seq(input_, head_dim, seq_dim, gather_size, group=None):
    """scatter seq (dim=seq_dim), gather heads (dim=head_dim)。
    [bs, full_seq, heads/cp, head_dim] → [bs, seq_local, heads, head_dim]"""
    if group is None or dist.get_world_size(group) == 1:
        return input_
    return ms_all_to_all(input_, group, scatter_dim=seq_dim, gather_dim=head_dim, gather_size=gather_size)
