"""
CP Debug 工具函数
"""
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any


def detect_seq_dim(
    tensor: torch.Tensor,
    expected_seq_len: int,
    cp_size: int = 1
) -> Optional[int]:
    """
    通过匹配 shape 自动检测序列维度

    Args:
        tensor: 输入 tensor（应为前向激活；权重/梯度不要走此函数）
        expected_seq_len: 完整序列长度
        cp_size: CP 大小

    Returns:
        序列维度索引，找不到返回 None

    注意：
        仅依据"某一维长度等于完整或分片后的序列长度"判定，存在 hidden_size
        与 seq_len 相等时的巧合风险。调用方对 weight / param_grad 已跳过本函数。
    """
    if expected_seq_len is None:
        return None

    # 构建目标集合：完整长度或 CP 分片后的长度
    targets = {expected_seq_len}
    if cp_size > 1:
        targets.add(expected_seq_len // cp_size)

    # 优先匹配分片后的长度（CP>1 时本地 tensor 的序列维就是分片长度）
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
    default_seq_dim: int = 1
) -> torch.Tensor:
    """
    沿序列维度 all-gather tensor

    Args:
        tensor: 输入 tensor
        cp_group: CP process group
        expected_seq_len: 完整序列长度（用于自动检测 seq_dim）
        tensor_name: tensor 名称（用于日志）
        default_seq_dim: 默认序列维度（当前实现仅做自动检测，未使用）

    Returns:
        all-gather 后的 tensor
    """
    # 没有 CP 或不需要 gather
    if cp_group is None or not dist.is_initialized():
        return tensor

    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return tensor

    # 不均分告警：各 rank 序列长度不同会导致 all_gather 形状不一致
    if expected_seq_len is not None and expected_seq_len % cp_size != 0:
        if tensor_name:
            print(f"[WARN] {tensor_name}: expected_seq_len={expected_seq_len} "
                  f"not divisible by cp_size={cp_size}; all-gather may fail, "
                  f"falling back to local tensor")
        return tensor

    # 自动检测 seq_dim
    seq_dim = detect_seq_dim(tensor, expected_seq_len, cp_size)

    if seq_dim is None:
        # 找不到序列维度，跳过 all-gather
        if tensor_name:
            print(f"[WARN] {tensor_name}: no seq_dim found, skipping all-gather")
        return tensor

    # 执行 all-gather
    try:
        gathered = [torch.zeros_like(tensor) for _ in range(cp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=cp_group)
        return torch.cat(gathered, dim=seq_dim)
    except Exception as e:
        if tensor_name:
            print(f"[ERROR] {tensor_name}: all-gather failed: {e}")
        return tensor


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    计算 tensor 统计信息

    Args:
        tensor: 输入 tensor

    Returns:
        统计信息字典
    """
    with torch.no_grad():
        t = tensor.detach().float()
        flat = t.flatten()
        abs_t = t.abs()
        absmax_flat = int(abs_t.argmax().item())
        absmax_loc = tuple(int(i) for i in torch.unravel_index(absmax_flat, t.shape))
        return {
            "shape": list(tensor.shape),
            "mean": t.mean().item(),
            "std": t.std().item() if flat.numel() > 1 else 0.0,
            "min": t.min().item(),
            "max": t.max().item(),
            "absmax": abs_t.max().item(),
            "absmax_loc": absmax_loc,
            "first5": flat[:5].tolist(),
            "last5": flat[-5:].tolist(),
        }


def format_stats(name: str, stats: Dict[str, Any], step: int) -> str:
    """
    格式化统计信息输出

    Args:
        name: tensor 名称
        stats: 统计信息
        step: 当前 step

    Returns:
        格式化字符串
    """
    return (
        f"[STEP {step}] {name}: "
        f"shape={stats['shape']} "
        f"mean={stats['mean']:.6f} "
        f"std={stats['std']:.6f} "
        f"min={stats['min']:.6f} "
        f"max={stats['max']:.6f} "
        f"absmax={stats['absmax']:.6f}@loc={stats['absmax_loc']} "
        f"first5={stats['first5']} "
        f"last5={stats['last5']}"
    )
