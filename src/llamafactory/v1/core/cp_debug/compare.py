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

"""CP Debug comparison script.

Usage:
    python -m llamafactory.v1.core.cp_debug.compare ./cp1_dumps ./cp2_dumps [options]

Options:
    --step N        Compare a specific step (default 0)
    --threshold T   Diff threshold (default 1e-5)
    --diff-only     Only show mismatched modules
    --gradients     Include gradient comparison
    --detail        Show full tensor diff
"""

import argparse
import sys
from pathlib import Path

import torch


def load_tensors(dump_dir: Path, step: int) -> dict[str, torch.Tensor]:
    """Load all tensors under a step directory."""
    step_dir = dump_dir / f"step{step}"
    if not step_dir.exists():
        return {}

    tensors = {}
    for pt_file in step_dir.glob("*.pt"):
        name = pt_file.stem
        tensors[name] = torch.load(pt_file, map_location="cpu")

    return tensors


def load_execution_order(dump_dir: Path, step: int) -> list[dict[str, str]]:
    """Load the execution order file."""
    order_file = dump_dir / f"step{step}" / "execution_order.txt"
    if not order_file.exists():
        return []

    records = []
    with open(order_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse: [   0] fwd_in   | model.layers.0.self_attn.in0                 | [1, 4096, 4096]
            parts = line.split("|")
            if len(parts) >= 3:
                idx_hook = parts[0].strip().split("]", 1)
                if len(idx_hook) == 2:
                    idx = idx_hook[0].strip("[")
                    hook_type = idx_hook[1].strip()
                    name = parts[1].strip()
                    shape = parts[2].strip()
                    records.append({"index": int(idx), "hook_type": hook_type, "name": name, "shape": shape})

    return records


def compare_execution_order(order1: list[dict[str, str]], order2: list[dict[str, str]]) -> bool:
    """Compare execution order."""
    if not order1 and not order2:
        return True

    if len(order1) != len(order2):
        print("\n--- Execution Order ---")
        print(f"CP1: {len(order1)} records")
        print(f"CP2: {len(order2)} records")
        print("MISMATCH: Different number of records")
        return False

    mismatches = []
    for i, (r1, r2) in enumerate(zip(order1, order2)):
        if r1["name"] != r2["name"] or r1["hook_type"] != r2["hook_type"]:
            mismatches.append((i, r1, r2))

    if mismatches:
        print("\n--- Execution Order ---")
        print(f"Total records: {len(order1)}")
        print(f"Mismatches: {len(mismatches)}")
        print("\nFirst 5 mismatches:")
        for idx, r1, r2 in mismatches[:5]:
            print(f"  [{idx:4d}] CP1: {r1['hook_type']:8s} | {r1['name']}")
            print(f"         CP2: {r2['hook_type']:8s} | {r2['name']}")
        return False

    print("\n--- Execution Order ---")
    print(f"Total records: {len(order1)}")
    print("OK: Execution order matches")
    return True


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, threshold: float = 1e-5) -> tuple[str, float, float]:
    """Compare two tensors.

    Returns:
        (status, max_diff, mean_diff)
    """
    if t1.shape != t2.shape:
        return ("SHAPE_MISMATCH", float("inf"), float("inf"))

    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    status = "OK" if max_diff < threshold else "FAIL"
    return (status, max_diff, mean_diff)


def categorize_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Categorize tensors into forward/backward/weights."""
    categories = {"Forward": {}, "Backward": {}, "Weights": {}}

    for name, tensor in tensors.items():
        # Exact suffix match to avoid misclassifying forward module names containing "weight"/"bias"
        if name.endswith(".grad") or "grad_in" in name or "grad_out" in name:
            categories["Backward"][name] = tensor
        elif name.endswith(".weight") or name.endswith(".bias"):
            categories["Weights"][name] = tensor
        else:
            categories["Forward"][name] = tensor

    return categories


def compare_category(
    name: str,
    tensors1: dict[str, torch.Tensor],
    tensors2: dict[str, torch.Tensor],
    threshold: float,
    diff_only: bool,
    detail: bool,
) -> list[tuple[str, str, float, float]]:
    """Compare all tensors in one category."""
    del name
    results = []

    common = set(tensors1.keys()) & set(tensors2.keys())
    only_in_1 = set(tensors1.keys()) - set(tensors2.keys())
    only_in_2 = set(tensors2.keys()) - set(tensors1.keys())

    for tensor_name in sorted(common):
        t1 = tensors1[tensor_name]
        t2 = tensors2[tensor_name]

        status, max_diff, mean_diff = compare_tensors(t1, t2, threshold)

        if not diff_only or status != "OK":
            results.append((tensor_name, status, max_diff, mean_diff))

            if detail and status == "FAIL":
                diff = (t1.float() - t2.float()).abs()
                print(f"\n  Detail for {tensor_name}:")
                print(f"    CP1: shape={list(t1.shape)}, mean={t1.float().mean():.6f}")
                print(f"    CP2: shape={list(t2.shape)}, mean={t2.float().mean():.6f}")
                print(f"    Diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

    for tensor_name in sorted(only_in_1):
        if not diff_only:
            results.append((tensor_name, "ONLY_IN_CP1", float("inf"), float("inf")))

    for tensor_name in sorted(only_in_2):
        if not diff_only:
            results.append((tensor_name, "ONLY_IN_CP2", float("inf"), float("inf")))

    return results


def print_results(category: str, results: list[tuple[str, str, float, float]]):
    """Print comparison results."""
    if not results:
        return

    print(f"\n--- {category} ---")
    print(f"{'Module':<60} {'Status':<15} {'Max Diff':<12} {'Mean Diff':<12}")
    print("-" * 100)

    for name, status, max_diff, mean_diff in results:
        if status == "SHAPE_MISMATCH":
            max_str = "SHAPE"
            mean_str = "MISMATCH"
        elif status in ("ONLY_IN_CP1", "ONLY_IN_CP2"):
            max_str = "-"
            mean_str = "-"
        else:
            max_str = f"{max_diff:.2e}"
            mean_str = f"{mean_diff:.2e}"

        print(f"{name:<60} {status:<15} {max_str:<12} {mean_str:<12}")


def print_summary(all_results: dict[str, list[tuple[str, str, float, float]]]):
    """Print summary."""
    print("\n" + "=" * 90)
    print("Summary:")

    for category, results in all_results.items():
        ok_count = sum(1 for _, status, _, _ in results if status == "OK")
        fail_count = sum(1 for _, status, _, _ in results if status == "FAIL")

        first_fail = None
        for name, status, _, _ in results:
            if status == "FAIL":
                first_fail = name
                break

        summary = f"  {category}: {ok_count} OK, {fail_count} FAIL"
        if first_fail:
            summary += f"  |  First: {first_fail}"
        print(summary)


def main():
    parser = argparse.ArgumentParser(description="CP Debug comparison tool")
    parser.add_argument("dir1", type=Path, help="CP1 dump directory")
    parser.add_argument("dir2", type=Path, help="CP2 dump directory")
    parser.add_argument("--step", type=int, default=0, help="Compare a specific step (default 0)")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Diff threshold (default 1e-5)")
    parser.add_argument("--diff-only", action="store_true", help="Only show mismatched modules")
    parser.add_argument("--gradients", action="store_true", help="Include gradient comparison")
    parser.add_argument("--detail", action="store_true", help="Show full tensor diff")

    args = parser.parse_args()

    if not args.dir1.exists():
        print(f"Error: {args.dir1} does not exist")
        sys.exit(1)
    if not args.dir2.exists():
        print(f"Error: {args.dir2} does not exist")
        sys.exit(1)

    tensors1 = load_tensors(args.dir1, args.step)
    tensors2 = load_tensors(args.dir2, args.step)

    if not tensors1:
        print(f"Error: No tensors found in {args.dir1}/step{args.step}")
        sys.exit(1)
    if not tensors2:
        print(f"Error: No tensors found in {args.dir2}/step{args.step}")
        sys.exit(1)

    print(f"=== CP Debug Comparison: step {args.step} ===")
    print(f"CP1: {args.dir1} ({len(tensors1)} tensors)")
    print(f"CP2: {args.dir2} ({len(tensors2)} tensors)")

    order1 = load_execution_order(args.dir1, args.step)
    order2 = load_execution_order(args.dir2, args.step)
    if order1 or order2:
        compare_execution_order(order1, order2)

    categories1 = categorize_tensors(tensors1)
    categories2 = categorize_tensors(tensors2)

    all_results = {}

    for category in ["Forward", "Backward", "Weights"]:
        # Skip gradients if not requested
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


if __name__ == "__main__":
    main()
