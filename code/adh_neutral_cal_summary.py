#!/usr/bin/env python3
"""Summary of the neutral calibration of the Adh pipeline
(adh_neutral_cal.sh, inputs/adh/adh_neutral_cal.slurm): every replicate is
a neutral, recombining data set shaped like the real alignment (true theta
0.006 per site) that went through four-gamete segmentation, migrate-n and
the n=176 networks exactly like the real data. Reported, per network, over
all simulated segments and per data set, next to the real Adh values
(figures/real_adh_migrate_scan.csv):
  - segments whose 95% interval of s_hat has a lower end >= 0.5 (excludes 0
    at the integer rounding of the Adh table), with median s_hat > 5, > 10;
  - the share of posterior genealogies with s_hat < 1 (PGINN);
  - per data set: the largest median s_hat and the number of segments with
    an interval excluding 0, and the fraction of neutral data sets at least
    as extreme as the real data;
  - theta: relative error of migrate's Theta and of the networks' median
    theta_hat against the true 0.006, and how often the networks' 95%
    interval (over posterior genealogies) contains it.
With --net-seeds (several training runs), every segment's network values
are first averaged over the runs (average_runs.py), for the simulations and
for the real data, and the fractions of neutral data sets at least as
extreme as the real data are also given as the range over single runs.
Writes figures/tab_adh_neutral_cal.tex."""
import argparse
import csv
from pathlib import Path

import numpy as np

from average_runs import average, run_tables

TRUE_THETA = 0.006
FIG = Path(__file__).resolve().parent.parent / "figures"


def read(path):
    with open(path) as f:
        return [{k: (float(v) if k != "locus" else int(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def seg_stats(rows, net):
    lo = np.array([r[f"{net}_s_q025"] for r in rows])
    med = np.array([r[f"{net}_s"] for r in rows])
    return {"excl0": lo >= 0.5, "gt5": med > 5, "gt10": med > 10, "max_med": med.max()}


def p_values(data, real, net):
    """Fractions of neutral data sets with at least as many segments excluding 0, and as large a largest median s_hat."""
    rs = seg_stats(real, net)
    per_ds_excl = np.array([seg_stats(d, net)["excl0"].sum() for d in data])
    per_ds_max = np.array([seg_stats(d, net)["max_med"] for d in data])
    return np.mean(per_ds_excl >= rs["excl0"].sum()), np.mean(per_ds_max >= rs["max_med"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/adh-neutral-cal")
    ap.add_argument("--real", default=str(FIG / "real_adh_migrate_scan.csv"))
    ap.add_argument("--scan", default="scan.csv", help="per-replicate table, e.g. scan-featlog.csv (log-scaled KC features)")
    ap.add_argument("--net-seeds", type=int, nargs="+", default=[67],
                    help="training runs; tables of runs other than 67 carry -seed<N> (real_adh_migrate_scan.py --net-seeds)")
    args = ap.parse_args()
    reps = sorted(p for p in Path(args.dir).glob(f"rep*/{args.scan}"))
    seeds = args.net_seeds
    if len(seeds) > 1:
        data = [average(run_tables(str(p), seeds)) for p in reps]
        real = average(run_tables(args.real, seeds))
        runs = {s: ([read(t) for t in [run_tables(str(p), [s])[0] for p in reps]], read(run_tables(args.real, [s])[0]))
                for s in seeds}
        print(f"network values averaged over training runs {seeds}")
    else:
        data = [read(p) for p in reps]
        real = read(args.real)
        runs = {}
    print(f"{len(data)} neutral data sets, {sum(len(d) for d in data)} segments "
          f"(segments per data set {min(len(d) for d in data)}-{max(len(d) for d in data)}); real Adh: {len(real)} segments")
    tex = ["\\begin{tabular}{lrrr}", "\\toprule",
           " & neutral simulations & real \\emph{Adh} & fraction of neutral data sets $\\ge$ real \\\\", "\\midrule"]
    for net in ["MSE", "PGINN"]:
        segs = {k: np.concatenate([seg_stats(d, net)[k] for d in data]) for k in ["excl0", "gt5", "gt10"]}
        rs = seg_stats(real, net)
        per_ds_excl = np.array([seg_stats(d, net)["excl0"].sum() for d in data])
        per_ds_max = np.array([seg_stats(d, net)["max_med"] for d in data])
        p_excl, p_max = p_values(data, real, net)
        q_excl = q_max = ""
        if runs:
            pr = np.array([p_values(d, r, net) for d, r in runs.values()])
            print(f"   {net}, single training runs: P(segments excluding 0) {pr[:, 0].min():.2f}-{pr[:, 0].max():.2f}, "
                  f"P(largest median s_hat) {pr[:, 1].min():.2f}-{pr[:, 1].max():.2f}")
            q_excl = f" ({pr[:, 0].min():.2f}--{pr[:, 0].max():.2f})"
            q_max = f" ({pr[:, 1].min():.2f}--{pr[:, 1].max():.2f})"
        print(f"{net}: segments with 95% interval excluding 0: {segs['excl0'].mean():.3f} (real {rs['excl0'].mean():.3f}); "
              f"median s_hat > 5: {segs['gt5'].mean():.3f} (real {rs['gt5'].mean():.3f}); > 10: {segs['gt10'].mean():.3f} "
              f"(real {rs['gt10'].mean():.3f})")
        print(f"   per data set: segments excluding 0 median {np.median(per_ds_excl):.0f} (range {per_ds_excl.min()}-{per_ds_excl.max()}; "
              f"real {rs['excl0'].sum()}, P(neutral >= real) = {p_excl:.2f}); largest median s_hat median {np.median(per_ds_max):.2f} "
              f"(range {per_ds_max.min():.2f}-{per_ds_max.max():.2f}; real {rs['max_med']:.2f}, P = {p_max:.2f})")
        tex += [f"{net}: segments with 95\\% interval excluding 0 & {segs['excl0'].mean():.3f} & {rs['excl0'].mean():.3f} & -- \\\\",
                f"{net}: segments with median $\\hat s>5$ & {segs['gt5'].mean():.3f} & {rs['gt5'].mean():.3f} & -- \\\\",
                f"{net}: segments excluding 0 per data set & {np.median(per_ds_excl):.0f} ({per_ds_excl.min()}--{per_ds_excl.max()}) & {rs['excl0'].sum()} & {p_excl:.2f}{q_excl} \\\\",
                f"{net}: largest median $\\hat s$ per data set & {np.median(per_ds_max):.2f} ({per_ds_max.min():.2f}--{per_ds_max.max():.2f}) & {rs['max_med']:.2f} & {p_max:.2f}{q_max} \\\\"]
    pct = np.concatenate([[r["PGINN_pct_s0"] for r in d] for d in data])
    print(f"PGINN share of posterior genealogies with s_hat < 1: median {np.median(pct):.1f}% "
          f"(IQR {np.percentile(pct, 25):.1f}-{np.percentile(pct, 75):.1f}); real median {np.median([r['PGINN_pct_s0'] for r in real]):.1f}%")
    tex.append(f"PGINN: share of genealogies with $\\hat s<1$ (median) & {np.median(pct):.0f}\\% & "
               f"{np.median([r['PGINN_pct_s0'] for r in real]):.0f}\\% & -- \\\\")
    tex.append("\\midrule")
    allrows = [r for d in data for r in d]
    mig = np.array([r["migrate_theta_mode"] for r in allrows])
    print(f"theta (true {TRUE_THETA}): migrate Theta median {np.median(mig):.4f}, relative error median "
          f"{np.median(mig / TRUE_THETA - 1):+.2f} (IQR {np.percentile(mig / TRUE_THETA - 1, 25):+.2f} to {np.percentile(mig / TRUE_THETA - 1, 75):+.2f})")
    tex.append(f"migrate $\\Theta$, median relative error & {np.median(mig / TRUE_THETA - 1):+.2f} & -- & -- \\\\")
    for net in ["MSE", "PGINN"]:
        th = np.array([r[f"{net}_theta"] for r in allrows])
        cov = np.mean([(r[f"{net}_theta_q025"] <= TRUE_THETA <= r[f"{net}_theta_q975"]) for r in allrows])
        print(f"   {net}: median theta_hat {np.median(th):.4f}, relative error median {np.median(th / TRUE_THETA - 1):+.2f}; "
              f"95% interval (posterior genealogies) contains the true theta in {cov:.2f} of segments")
        tex.append(f"{net} $\\hat\\theta$: median relative error; interval coverage & {np.median(th / TRUE_THETA - 1):+.2f}; {cov:.2f} & -- & -- \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    out = FIG / ("tab_adh_neutral_cal" + ("_featlog" if "featlog" in args.scan else "") + ".tex")
    out.write_text("\n".join(tex) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
