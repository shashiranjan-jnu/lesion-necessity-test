"""A6j — resolution sweep with 3 seeds, matching the seed count used everywhere
else. A6d/E5 used 2 seeds, which made the 128px occluded value (0.762) differ
from Table 1's (0.769) for no substantive reason."""
import numpy as np
from scripts.a6d_shortcut_rigour import run_cv

SEEDS = [1, 2, 3]

if __name__ == "__main__":
    print("A6j | occluded models at decreasing resolution | 5 folds x 3 seeds")
    print("      (A6d/E5 used 2 seeds; this supersedes it)\n")
    print(f"  {'res':>5} {'macro-F1':>10} {'sd':>8}   per-fold")
    print("  " + "-" * 62)
    out = {}
    for size in [128, 64, 32, 16, 8]:
        v = run_cv("occlude", size=size, seeds=SEEDS)
        out[size] = v
        print(f"  {size:>5} {np.mean(v):>10.4f} {np.std(v):>8.4f}   "
              + " ".join(f"{x:.3f}" for x in v))
    print("\n  paste these into make_figures.py RES and Table 3")
