"""
CP Debug 对比脚本

用法:
    python -m cp_debug.compare ./cp1_dumps ./cp2_dumps [options]

选项:
    --step N          对比特定 step（默认 0）
    --threshold T     差异阈值（默认 1e-5）
    --diff-only       只显示不一致的模块
    --gradients       包含梯度对比
    --detail          显示完整 tensor diff（含最大误差位置 + 两端原始值）

每个 tensor 报告: Max Diff（|t1-t2| 全局最大）、Mean Diff、Max Loc（最大误差
出现的多维坐标）。--detail 下额外打印该位置上 CP1/CP2 的原始值与有符号差。
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch


class _Tee:
    """同时写多个流（stdout + 文件），让所有 print 自动双写。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data) if data else 0

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def _step_dir(dump_dir: Path, step: int, dp_rank: Optional[int] = None) -> Path:
    """定位 step 目录：多 rank 布局 {dump_dir}/dp_rank{D}/step{S}/，旧布局 {dump_dir}/step{S}/"""
    if dp_rank is None:
        return dump_dir / f"step{step}"
    return dump_dir / f"dp_rank{dp_rank}" / f"step{step}"


def discover_dp_ranks(dump_dir: Path) -> List[int]:
    """发现 dump 目录下所有 dp_rank 子目录（多 rank 布局）。旧布局返回 []。"""
    if not dump_dir.exists():
        return []
    ranks = []
    for p in dump_dir.iterdir():
        if p.is_dir() and p.name.startswith("dp_rank"):
            try:
                ranks.append(int(p.name.replace("dp_rank", "")))
            except ValueError:
                continue
    return sorted(ranks)


def load_tensors(dump_dir: Path, step: int, dp_rank: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """加载 step 目录下的所有 tensor"""
    step_dir = _step_dir(dump_dir, step, dp_rank)
    if not step_dir.exists():
        return {}

    tensors = {}
    for pt_file in step_dir.glob("*.pt"):
        name = pt_file.stem
        tensors[name] = torch.load(pt_file, map_location="cpu")

    return tensors


def load_execution_order(dump_dir: Path, step: int, dp_rank: Optional[int] = None) -> List[Dict[str, str]]:
    """加载执行顺序文件"""
    order_file = _step_dir(dump_dir, step, dp_rank) / "execution_order.txt"
    if not order_file.exists():
        return []
    
    records = []
    with open(order_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # 解析格式: [   0] fwd_in   | model.layers.0.self_attn.in0                 | [1, 4096, 4096]
            parts = line.split("|")
            if len(parts) >= 3:
                idx_hook = parts[0].strip().split("]", 1)
                if len(idx_hook) == 2:
                    idx = idx_hook[0].strip("[")
                    hook_type = idx_hook[1].strip()
                    name = parts[1].strip()
                    shape = parts[2].strip()
                    records.append({
                        "index": int(idx),
                        "hook_type": hook_type,
                        "name": name,
                        "shape": shape
                    })
    
    return records


def compare_execution_order(
    order1: List[Dict[str, str]],
    order2: List[Dict[str, str]]
) -> bool:
    """对比执行顺序"""
    if not order1 and not order2:
        return True
    
    if len(order1) != len(order2):
        print(f"\n--- Execution Order ---")
        print(f"CP1: {len(order1)} records")
        print(f"CP2: {len(order2)} records")
        print("MISMATCH: Different number of records")
        return False
    
    mismatches = []
    for i, (r1, r2) in enumerate(zip(order1, order2)):
        if r1["name"] != r2["name"] or r1["hook_type"] != r2["hook_type"]:
            mismatches.append((i, r1, r2))
    
    if mismatches:
        print(f"\n--- Execution Order ---")
        print(f"Total records: {len(order1)}")
        print(f"Mismatches: {len(mismatches)}")
        print("\nFirst 5 mismatches:")
        for idx, r1, r2 in mismatches[:5]:
            print(f"  [{idx:4d}] CP1: {r1['hook_type']:8s} | {r1['name']}")
            print(f"         CP2: {r2['hook_type']:8s} | {r2['name']}")
        return False
    
    print(f"\n--- Execution Order ---")
    print(f"Total records: {len(order1)}")
    print("OK: Execution order matches")
    return True


def compare_tensors(
    t1: torch.Tensor,
    t2: torch.Tensor,
    threshold: float = 1e-5
) -> Tuple[str, float, float, Optional[tuple], float, float]:
    """
    对比两个 tensor（逐元素相减取绝对值）

    Returns:
        (status, max_diff, mean_diff, max_loc, val1, val2)
        - max_diff: |t1 - t2| 的全局最大值
        - max_loc:  该最大值出现的多维坐标（SHAPE_MISMATCH 时为 None）
        - val1/val2: 该位置上 CP1 / CP2 的原始值
    """
    if t1.shape != t2.shape:
        return ("SHAPE_MISMATCH", float('inf'), float('inf'), None, float('nan'), float('nan'))

    t1f = t1.float()
    t2f = t2.float()
    diff = (t1f - t2f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # 逐元素相减后绝对值最大的位置 + 该位置两端的值
    # unravel_index 在部分 torch 版本要求 indices 为 tensor（不接受 int）
    flat_idx = diff.argmax()
    max_loc = tuple(int(i) for i in torch.unravel_index(flat_idx, diff.shape))
    val1 = t1f[max_loc].item()
    val2 = t2f[max_loc].item()

    status = "OK" if max_diff < threshold else "FAIL"
    return (status, max_diff, mean_diff, max_loc, val1, val2)


def categorize_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """将 tensor 分类为 forward/backward/weights"""
    categories = {
        "Forward": {},
        "Backward": {},
        "Weights": {}
    }
    
    for name, tensor in tensors.items():
        # 精确后缀匹配，避免误把含 "weight"/"bias" 的前向模块名归入权重
        if name.endswith(".grad") or "grad_in" in name or "grad_out" in name:
            categories["Backward"][name] = tensor
        elif name.endswith(".weight") or name.endswith(".bias"):
            categories["Weights"][name] = tensor
        else:
            categories["Forward"][name] = tensor
    
    return categories


def compare_category(
    name: str,
    tensors1: Dict[str, torch.Tensor],
    tensors2: Dict[str, torch.Tensor],
    threshold: float,
    diff_only: bool,
    detail: bool
) -> List[Tuple]:
    """对比一个类别的所有 tensor

    结果元组: (name, status, max_diff, mean_diff, max_loc, val1, val2)
    """
    results = []

    common = set(tensors1.keys()) & set(tensors2.keys())
    only_in_1 = set(tensors1.keys()) - set(tensors2.keys())
    only_in_2 = set(tensors2.keys()) - set(tensors1.keys())

    # 对比共同的 tensor
    for tensor_name in sorted(common):
        t1 = tensors1[tensor_name]
        t2 = tensors2[tensor_name]

        status, max_diff, mean_diff, max_loc, val1, val2 = compare_tensors(t1, t2, threshold)

        if not diff_only or status != "OK":
            results.append((tensor_name, status, max_diff, mean_diff, max_loc, val1, val2))

            if status == "FAIL":
                # FAIL 默认就打印最大误差位置 + 两端原始值（无需 --detail）
                loc_str = str(max_loc) if max_loc is not None else "N/A"
                print(f"  [FAIL] {tensor_name}: max|diff|={max_diff:.6e} at loc={loc_str}  "
                      f"CP1={val1:.6e}  CP2={val2:.6e}  |diff|={abs(val1 - val2):.6e}")
                if detail:
                    diff = (t1.float() - t2.float()).abs()
                    print(f"    CP1: shape={list(t1.shape)}, mean={t1.float().mean():.6f}")
                    print(f"    CP2: shape={list(t2.shape)}, mean={t2.float().mean():.6f}")
                    print(f"    Diff: mean={diff.mean():.6e}")

    # 只在一个目录中的 tensor
    for tensor_name in sorted(only_in_1):
        if not diff_only:
            results.append((tensor_name, "ONLY_IN_CP1", float('inf'), float('inf'), None, float('nan'), float('nan')))

    for tensor_name in sorted(only_in_2):
        if not diff_only:
            results.append((tensor_name, "ONLY_IN_CP2", float('inf'), float('inf'), None, float('nan'), float('nan')))

    return results


def print_results(
    category: str,
    results: List[Tuple]
):
    """打印对比结果"""
    if not results:
        return

    print(f"\n--- {category} ---")
    print(f"{'Module':<50} {'Status':<13} {'Max Diff':<11} {'Mean Diff':<11} {'Max Loc':<24}")
    print("-" * 112)

    for name, status, max_diff, mean_diff, max_loc, val1, val2 in results:
        if status == "SHAPE_MISMATCH":
            max_str = "SHAPE"
            mean_str = "MISMATCH"
            loc_str = "-"
        elif status in ("ONLY_IN_CP1", "ONLY_IN_CP2"):
            max_str = "-"
            mean_str = "-"
            loc_str = "-"
        else:
            max_str = f"{max_diff:.2e}"
            mean_str = f"{mean_diff:.2e}"
            loc_str = str(max_loc) if max_loc is not None else "-"

        print(f"{name:<50} {status:<13} {max_str:<11} {mean_str:<11} {loc_str:<24}")


def print_summary(all_results: Dict[str, List[Tuple[str, str, float, float]]]):
    """打印汇总"""
    print("\n" + "=" * 90)
    print("Summary:")
    
    for category, results in all_results.items():
        ok_count = sum(1 for _, status, *_ in results if status == "OK")
        fail_count = sum(1 for _, status, *_ in results if status == "FAIL")

        first_fail = None
        for name, status, *_ in results:
            if status == "FAIL":
                first_fail = name
                break
        
        summary = f"  {category}: {ok_count} OK, {fail_count} FAIL"
        if first_fail:
            summary += f"  |  First: {first_fail}"
        print(summary)


def run_comparison(dir1: Path, dir2: Path, step: int, args, dp_rank: Optional[int]) -> bool:
    """对单个 dp_rank（或旧布局 dp_rank=None）跑一次完整对比。返回是否有数据。"""
    tensors1 = load_tensors(dir1, step, dp_rank)
    tensors2 = load_tensors(dir2, step, dp_rank)

    where = f"dp_rank{dp_rank}/" if dp_rank is not None else ""
    if not tensors1 and not tensors2:
        return False
    if not tensors1:
        print(f"Error: No tensors in {dir1}/{where}step{step}")
        return False
    if not tensors2:
        print(f"Error: No tensors in {dir2}/{where}step{step}")
        return False

    header = f"=== CP Debug Comparison: step {step}"
    if dp_rank is not None:
        header += f" | dp_rank {dp_rank}"
    header += " ==="
    print("\n" + "=" * 90)
    print(header)
    print(f"CP1: {dir1}/{where}step{step} ({len(tensors1)} tensors)")
    print(f"CP2: {dir2}/{where}step{step} ({len(tensors2)} tensors)")

    order1 = load_execution_order(dir1, step, dp_rank)
    order2 = load_execution_order(dir2, step, dp_rank)
    if order1 or order2:
        compare_execution_order(order1, order2)

    categories1 = categorize_tensors(tensors1)
    categories2 = categorize_tensors(tensors2)

    all_results = {}
    for category in ["Forward", "Backward", "Weights"]:
        if category == "Backward" and not args.gradients:
            continue
        results = compare_category(
            category,
            categories1[category],
            categories2[category],
            args.threshold,
            args.diff_only,
            args.detail,
        )
        if results:
            all_results[category] = results
            print_results(category, results)

    print_summary(all_results)
    return True


def _run_with_log(dir1: Path, dir2: Path, step: int, args, dp_rank: Optional[int],
                  out_dir: Path, ts: str) -> bool:
    """跑单个 dp_rank 的对比：屏幕照常输出，同时写一份单独的 per-dp_rank 日志文件。"""
    log_fh = None
    log_path = None
    orig_stdout = sys.stdout
    if not args.no_out:
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"dp_rank{dp_rank}_" if dp_rank is not None else ""
        log_path = out_dir / f"compare_{suffix}{ts}.log"
        log_fh = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(orig_stdout, log_fh)
        where = f"dp_rank{dp_rank}/" if dp_rank is not None else ""
        print(f"# CP Debug compare log\n# CP1: {dir1}\n# CP2: {dir2}\n# step: {step}  threshold: {args.threshold}  {where}\n# written: {log_path}\n")
    try:
        ok = run_comparison(dir1, dir2, step, args, dp_rank)
    finally:
        sys.stdout = orig_stdout
        if log_fh is not None:
            log_fh.close()
            print(f"[compare] dp_rank{dp_rank if dp_rank is not None else '-'} 结果已写入: {log_path}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="CP Debug 对比工具")
    parser.add_argument("dir1", type=Path, help="CP1 dump 目录")
    parser.add_argument("dir2", type=Path, help="CP2 dump 目录")
    parser.add_argument("--step", type=int, default=0, help="对比特定 step（默认 0）")
    parser.add_argument("--threshold", type=float, default=1e-5, help="差异阈值（默认 1e-5）")
    parser.add_argument("--diff-only", action="store_true", help="只显示不一致的模块")
    parser.add_argument("--gradients", action="store_true", help="包含梯度对比")
    parser.add_argument("--detail", action="store_true", help="显示完整 tensor diff")
    parser.add_argument("--out-dir", type=Path, default=Path("./cp_compare"),
                        help="结果日志目录（自动创建），默认 ./cp_compare")
    parser.add_argument("--no-out", action="store_true", help="不写文件，只打屏")

    args = parser.parse_args()

    if not args.dir1.exists():
        print(f"Error: {args.dir1} does not exist")
        sys.exit(1)
    if not args.dir2.exists():
        print(f"Error: {args.dir2} does not exist")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 屏幕先打一个总头（不写进 per-dp_rank 文件）
    print(f"######## CP Debug compare: CP1={args.dir1}  CP2={args.dir2}  step={args.step}  threshold={args.threshold} ########")

    ranks1 = discover_dp_ranks(args.dir1)
    ranks2 = discover_dp_ranks(args.dir2)

    if ranks1 and ranks2:
        common = sorted(set(ranks1) & set(ranks2))
        only1 = sorted(set(ranks1) - set(ranks2))
        only2 = sorted(set(ranks2) - set(ranks1))
        print(f"CP1 dp_ranks: {ranks1}  CP2 dp_ranks: {ranks2}  common: {common}")
        if only1:
            print(f"  only in CP1: {only1}")
        if only2:
            print(f"  only in CP2: {only2}")
        if not common:
            print("Error: no common dp_rank to compare")
            sys.exit(1)
        # 每个 dp_rank 单独写一个日志文件；屏幕合在一起输出
        for r in common:
            _run_with_log(args.dir1, args.dir2, args.step, args, dp_rank=r, out_dir=args.out_dir, ts=ts)
        return

    # 旧布局（单 rank，step{S}/ 在根）：直接比；一边多 rank 一边旧布局则报错
    if ranks1 and not ranks2:
        print(f"Error: {args.dir1} is multi-rank layout (dp_rank*/), but {args.dir2} is flat. 重新用同版本 dump。")
        sys.exit(1)
    if ranks2 and not ranks1:
        print(f"Error: {args.dir2} is multi-rank layout (dp_rank*/), but {args.dir1} is flat. 重新用同版本 dump。")
        sys.exit(1)
    _run_with_log(args.dir1, args.dir2, args.step, args, dp_rank=None, out_dir=args.out_dir, ts=ts)


if __name__ == "__main__":
    main()
