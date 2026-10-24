#!/usr/bin/env python3
"""Summary of the end-to-end test on simulated chromosomes
(chrom_pipeline.sh): for every setting (sweep chromosome with s = 200, or
sweep-s<S> with s = S, or neutral; constant-size or growth prior in
migrate-n) and replicate, per window the
number of segregating sites, migrate's Theta and the networks' s_hat; and,
over replicates, the median s_hat and Theta of the window at the sweep site
against the other windows, and how often the sweep window has the largest
s_hat or the smallest Theta of its chromosome.

--scan picks the network summary per window: scan-featlog.csv (each
posterior tree scored, median over trees) or scan-featlog-avg.csv (one
input, the features averaged over 100 posterior trees).

--set-sizes: the replicate chromosomes of a setting are split, in order,
into disjoint sets of k independent chromosomes (each an independent
realization of the same history: the same sweep at the same site, or the
same neutral history), and each window's value is averaged over the k
chromosomes of a set; k=1 is a single chromosome, the case of real data.
Per k: how often the middle window has the largest averaged s_hat (the
smallest averaged Theta), and the fraction of sweep sets whose middle
window exceeds the 95% quantile of the neutral sets' middle windows.

--net-seeds: with more than one training run of the networks
(scan-featlog-seed<N>.csv, no suffix for 67), the same two numbers for
PGINN and MSE per run, summarized over runs (mean and range), and the
number of runs in which PGINN exceeds MSE; only replicates scored with
every run are used."""
import argparse
import csv
from pathlib import Path

import numpy as np


def load(d, scan):
    sc = list(csv.DictReader(open(d / scan)))
    w = list(csv.DictReader(open(d / "sim-windows.csv")))
    return {"S": np.array([int(x["segregating_sites"]) for x in w]),
            "theta": np.array([float(x["migrate_theta_mode"]) for x in sc]),
            "MSE": np.array([float(x["MSE_s"]) for x in sc]),
            "MSE_theta": np.array([float(x["MSE_theta"]) for x in sc]),
            "PGINN_theta": np.array([float(x["PGINN_theta"]) for x in sc]),
            "PGINN": np.array([float(x["PGINN_s"]) for x in sc])}


def run_scan(scan, seed):
    return scan if seed == 67 else scan.replace(".csv", f"-seed{seed}.csv")


def k_stats(sw, ne, key, k, mid):
    """Fraction of sweep sets with the middle window on top, and the sensitivity at the neutral sets' 95% quantile."""
    a, b = set_means(sw, key, k), set_means(ne, key, k)
    sign = -1 if key == "theta" else 1  # small Theta is the sweep signal
    top = np.mean(sign * a[:, mid] == (sign * a).max(axis=1))  # ties count as top
    thr = np.quantile(sign * b[:, mid], 0.95)
    return top, np.mean(sign * a[:, mid] > thr)


def set_means(reps, key, k):
    """Window values averaged over disjoint sets of k replicates: (sets, windows)."""
    x = np.array([r[key] for r in reps])
    m = len(x) // k
    return x[:m * k].reshape(m, k, -1).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/chrom-pipeline")
    ap.add_argument("--scan", default="scan-featlog.csv")
    ap.add_argument("--set-sizes", type=int, nargs="+", default=[1])
    ap.add_argument("--per-replicate", action="store_true", help="print every replicate's windows")
    ap.add_argument("--net-seeds", type=int, nargs="+", default=[67])
    args = ap.parse_args()
    print(f"network summary per window: {args.scan}")
    settings = {}
    for setting in sorted(p for p in Path(args.dir).iterdir() if p.is_dir()):
        reps = [load(d, args.scan) for d in sorted(setting.glob("rep*")) if (d / args.scan).exists()]
        if not reps:
            continue
        settings[setting.name] = reps
        print(f"== {setting.name}: {len(reps)} replicates")
        if args.per_replicate:
            for i, r in enumerate(reps, 1):
                print(f"   rep {i}: S " + " ".join(f"{v:3d}" for v in r["S"]))
                print(f"          Theta " + " ".join(f"{v:.3f}" for v in r["theta"]))
                print(f"          PGINN s_hat " + " ".join(f"{v:.1f}" for v in r["PGINN"]))
        mid = len(reps[0]["S"]) // 2
        for key in ["PGINN", "MSE", "theta"]:
            at = np.array([r[key][mid] for r in reps])
            rest = np.concatenate([np.delete(r[key], mid) for r in reps])
            top = np.mean([(r[key][mid] == (r[key].min() if key == "theta" else r[key].max())) for r in reps])
            print(f"   {key}: middle window median {np.median(at):.3g}, other windows median {np.median(rest):.3g}; "
                  f"middle window is the {'smallest' if key == 'theta' else 'largest'} in {top:.2f} of replicates")

    for name in sorted(n for n in settings if n.startswith("sweep")):  # sweep[-s<S>]-<prior> against neutral-<prior>
        prior = name.rsplit("-", 1)[1]
        sw, ne = settings[name], settings.get(f"neutral-{prior}")
        if not ne:
            continue
        mid = len(sw[0]["S"]) // 2
        print(f"== sets of k chromosomes, {name} against neutral-{prior} ({len(sw)} sweep, {len(ne)} neutral replicates)")
        for k in args.set_sizes:
            if len(sw) < k or len(ne) < k:
                continue
            cells = []
            for key in ["PGINN", "MSE", "theta"]:
                top, sens = k_stats(sw, ne, key, k, mid)
                cells.append(f"{key} middle top {top:.2f}, sensitivity {sens:.2f}")
            print(f"   k={k:2d} ({len(sw) // k} sweep sets, {len(ne) // k} neutral sets): " + "; ".join(cells))

    if len(args.net_seeds) < 2:
        return
    print(f"== over training runs {' '.join(map(str, args.net_seeds))}: mean [min, max]; "
          "PGINN > MSE = runs in which PGINN exceeds MSE (ties not counted)")
    runs = {}  # setting -> list over runs of the replicates scored with every run
    for setting in sorted(p for p in Path(args.dir).iterdir() if p.is_dir()):
        dirs = [d for d in sorted(setting.glob("rep*"))
                if all((d / run_scan(args.scan, seed)).exists() for seed in args.net_seeds)]
        if dirs:
            runs[setting.name] = [[load(d, run_scan(args.scan, seed)) for d in dirs] for seed in args.net_seeds]
    for name in sorted(n for n in runs if n.startswith("sweep")):
        prior = name.rsplit("-", 1)[1]
        if f"neutral-{prior}" not in runs:
            continue
        sw_runs, ne_runs = runs[name], runs[f"neutral-{prior}"]
        mid = len(sw_runs[0][0]["S"]) // 2
        print(f"== {name} against neutral-{prior} ({len(sw_runs[0])} sweep, {len(ne_runs[0])} neutral replicates)")
        for k in args.set_sizes:
            if len(sw_runs[0]) < k or len(ne_runs[0]) < k:
                continue
            st = {key: np.array([k_stats(sw, ne, key, k, mid) for sw, ne in zip(sw_runs, ne_runs)])
                  for key in ["PGINN", "MSE"]}
            cells = []
            for j, label in enumerate(["middle top", "sensitivity"]):
                for key in ["PGINN", "MSE"]:
                    x = st[key][:, j]
                    cells.append(f"{key} {label} {x.mean():.2f} [{x.min():.2f}, {x.max():.2f}]")
                cells.append(f"PGINN > MSE {int(np.sum(st['PGINN'][:, j] > st['MSE'][:, j]))}/{len(args.net_seeds)}")
            print(f"   k={k:2d}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
