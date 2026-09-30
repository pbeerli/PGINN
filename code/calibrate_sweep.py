#!/usr/bin/env python3
"""
Calibration diagnostic (not part of the training pipeline): simulate a
handful of genealogies across a candidate range of s and check the sweep
model visibly interpolates from Kingman-like (s=0) to star-like (large s)
trees, before committing to the full train/val/predict simulation budget.

Usage: python calibrate_sweep.py [--nind N [N ...]] [--out-tex FILE]
"""
import argparse
import subprocess
import json
import tempfile
from pathlib import Path
import numpy as np
import sweep

HERE = Path(__file__).resolve().parent

THETA = 0.02
N_REPLICATES = 200
N_SEGMENTS = 40

S_VALUES = [0.0, 1.0, 4.0, 20.0, 100.0, 400.0, 2000.0]  # population-scaled s = c*s_g


def run_batch(theta, s, n_replicates, nind, seed=None):
    tmp = tempfile.mkdtemp(dir=HERE / "data")
    tp_path = f"{tmp}/calib_tp.txt"
    if s > 0:
        sweep.make_timepoints_file(tp_path, theta, s, N_SEGMENTS)
        tp_args = ["-tp", tp_path]
    else:
        tp_args = []  # s=0: theta_eff is constant == theta*x0 ~= theta, skip -tp entirely
    out_path = f"{tmp}/calib_out.json"
    cmd = ["python", str(HERE / "simtree.py"), "-t", str(theta), "-i", str(nind), "-a", "1.0",
           "-l", str(n_replicates), "-s", "200", "--json", "-sel", str(s),
           "-f", out_path] + tp_args + ([] if seed is None else ["--seed", str(seed)])
    subprocess.run(cmd, check=True, capture_output=True)
    data = json.load(open(out_path))
    for p in Path(tmp).iterdir():
        p.unlink()
    Path(tmp).rmdir()
    ages = np.array([d["ages"] for d in data])  # (n_replicates, n-1) sorted descending
    return ages


def summarize(ages):
    tmrca = ages[:, 0]
    # star-likeness: fraction of TMRCA NOT covered by the second-deepest
    # split (ages[:, 1], since ages is sorted descending: index 0 = TMRCA,
    # index 1 = second-deepest) -> small value means almost all mergers
    # happen near the same (deep) time, i.e. a star-like burst; large value
    # means a spread-out, Kingman-like ladder of merge times.
    star_ratio = (ages[:, 0] - ages[:, 1]) / ages[:, 0]
    cv_within = ages.std(axis=1) / ages.mean(axis=1)
    return {
        "TMRCA mean": tmrca.mean(),
        "TMRCA std": tmrca.std(),
        "star_ratio mean": star_ratio.mean(),
        "within-tree CV mean": cv_within.mean(),
    }


def sig(x, k):
    """k significant figures, fixed-point (sig(0.0114, 2) -> 0.011, sig(1.3e-05, 2) -> 0.000013)."""
    return f"{x:.{max(1, k - 1 - int(np.floor(np.log10(x))))}f}"


def s_tex(s):
    return f"{s:,g}".replace(",", "{,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nind", type=int, nargs="+", default=[10])
    ap.add_argument("--seed", type=int, default=1, help="simtree seed; cell k (in print order) uses seed+k")
    ap.add_argument("--out-tex", default=None, help="write the table body (tab:calib) to this file")
    args = ap.parse_args()
    rows = []
    k = 0
    for nind in args.nind:
        print(f"nind={nind}")
        print(f"{'s':>8} | {'TMRCA mean':>11} | {'TMRCA std':>10} | {'star_ratio':>10} | {'within CV':>10}")
        print("-" * 62)
        for i, s in enumerate(S_VALUES):
            ages = run_batch(THETA, s, N_REPLICATES, nind, args.seed + k)
            k += 1
            stats = summarize(ages)
            print(f"{s:8.1f} | {stats['TMRCA mean']:11.5f} | {stats['TMRCA std']:10.5f} | "
                  f"{stats['star_ratio mean']:10.4f} | {stats['within-tree CV mean']:10.4f}")
            rows.append(f"{nind if i == 0 else ''} & {s_tex(s)} & {sig(stats['TMRCA mean'], 3)} & "
                        f"{sig(stats['TMRCA std'], 2)} & {stats['star_ratio mean']:.3f} & "
                        f"{stats['within-tree CV mean']:.3f} \\\\")
    if args.out_tex:
        body = ["\\begin{tabular}{crrrrr}", "\\toprule",
                "Sample size & $s$ & TMRCA mean & TMRCA std & star ratio & within-tree CV \\\\",
                "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
        Path(args.out_tex).write_text("\n".join(body) + "\n")
        print("wrote", args.out_tex)
