#!/usr/bin/env python
"""Compare CP1 vs CP2 batch/token dumps produced by llamafactory.v1.utils.cp_dump.

Usage:
    python compare_cp_dumps.py <cp1_dump_dir> <cp2_dump_dir> [step ...]

Prints shapes + per-token CE / logit stats side by side for each requested step
(default: all steps present in both dirs). Highlights outlier tokens (ce_top10)
and logit saturation -- the signal for whether a spiky batch is data-driven.
"""
import os
import sys


def read_step(d: str, kind: str, step: int) -> str:
    p = os.path.join(d, f"{kind}_step_{step}.txt")
    if not os.path.isfile(p):
        return f"<missing: {p}>"
    with open(p) as f:
        return f.read().rstrip()


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    d1, d2 = sys.argv[1], sys.argv[2]
    steps = [int(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else None

    if steps is None:
        s1 = {int(f.split("_")[-1].split(".")[0]) for f in os.listdir(d1) if f.startswith("tokens_step_")}
        s2 = {int(f.split("_")[-1].split(".")[0]) for f in os.listdir(d2) if f.startswith("tokens_step_")}
        steps = sorted(s1 & s2)

    for s in steps:
        print(f"\n================= step {s} =================")
        for kind in ("shapes", "tokens"):
            print(f"\n--- {kind} ---")
            a = read_step(d1, kind, s)
            b = read_step(d2, kind, s)
            print(f"[CP1]\n{a}")
            print(f"[CP2]\n{b}")


if __name__ == "__main__":
    main()
