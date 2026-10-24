#!/usr/bin/env python3
"""False-positive rates of the selection detectors under neutral models the
networks were not trained on. msprime simulates independent, non-recombining
genealogies (haploid, present size N0=1000, mu=1e-5 so theta=0.02 at the
present, branch lengths times mu) under:
  constant   standard neutral coalescent (reference)
  growth10   exponential growth to N0 from N0/10 over the last 0.5*N0 generations
  growth100  the same from N0/100
  bottleneck size N0/10 from 0.05*N0 to 0.15*N0 generations ago
  structure  two demes of N0 with symmetric migration 2*N0*m = 1, sample from one
Each example is 1 or 10 genealogies (averaged as in training), n = 10 or 20,
400 examples per model and regime. Every statistic is thresholded at its
95% quantile on the regime's standard neutral predict set (the 5%
false-positive threshold of the paper); the table gives the fraction of
examples above it (network statistics: mean over the 10 training seeds) and
the mean predicted s. MLE = mle_baseline.py under the constant-size sweep
model. Writes figures/tab_robustness_demography.tex."""
import json
import os
import shutil
from pathlib import Path

import msprime
import numpy as np

import mle_baseline
from compare import load_and_predict
from msprime_scan import window_tree_to_entry
from prepare import AveragedThetaDataset

N0, MU, N_EX = 1000, 1e-5, 400
SEEDS = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
REGIMES = [("10l", 10, 10, "10 loci, $n{=}10$"), ("10l-n20", 10, 20, "10 loci, $n{=}20$"),
           ("1l", 1, 10, "1 locus, $n{=}10$"), ("1l-n20", 1, 20, "1 locus, $n{=}20$")]
MODELS = ["constant", "growth10", "growth100", "bottleneck", "structure"]
FIG = Path(__file__).resolve().parent.parent / "figures"


def demography(model):
    d = msprime.Demography()
    if model == "structure":
        d.add_population(name="A", initial_size=N0)
        d.add_population(name="B", initial_size=N0)
        d.set_symmetric_migration_rate(["A", "B"], 1.0 / (2 * N0))
        return d
    d.add_population(name="A", initial_size=N0)
    if model.startswith("growth"):
        fold = int(model[len("growth"):])
        t = 0.5 * N0
        d.populations[0].growth_rate = np.log(fold) / t
        d.add_population_parameters_change(time=t, initial_size=N0 / fold, growth_rate=0, population="A")
    elif model == "bottleneck":
        d.add_population_parameters_change(time=0.05 * N0, initial_size=N0 / 10, population="A")
        d.add_population_parameters_change(time=0.15 * N0, initial_size=N0, population="A")
    return d


def simulate(model, n, loci, seed, workdir):
    """features, mean ages, per-locus ages for N_EX examples."""
    dem = demography(model)
    entries, locus_ages = [], []
    reps = msprime.sim_ancestry(samples={"A": n}, ploidy=1, demography=dem, random_seed=seed,
                                num_replicates=N_EX * loci)
    for i, ts in enumerate(reps):
        e = window_tree_to_entry(ts.first(), MU, i // loci, n)  # group key = example index
        e["ID"] = i
        entries.append(e)
    for ex in range(N_EX):
        locus_ages.append([entries[ex * loci + j]["ages"] for j in range(loci)])
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    with open(os.path.join(workdir, "sim.json"), "w") as f:
        json.dump(entries, f)
    thetas, features, ages = AveragedThetaDataset(workdir, [0, 2]).__getitems__()
    order = np.argsort(np.array(thetas)[:, 1])
    return np.array(features)[order], np.array(ages)[order], np.array(locus_ages)


def seed_paths(stub, seed):
    sfx = "" if seed == 67 else f"-seed{seed}"
    return f"model/model-{stub}{sfx}.pt", f"model/scaler-{stub}{sfx}.npz"


def net_s(stub, seed, x):
    m, s = seed_paths(stub, seed)
    return load_and_predict(m, s, x)[:, 1]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-suffix", default="", help="e.g. -featlog: networks trained on log-scaled KC features")
    args = ap.parse_args()
    sfx = args.model_suffix
    tex = ["\\begin{tabular}{ll" + "r" * len(MODELS) + "}", "\\toprule",
           "Regime & Statistic & " + " & ".join(MODELS) + " \\\\", "\\midrule"]
    for r, loci, n, rlabel in REGIMES:
        neutral_x = f"data/predict-neutral-{r}-features.npy"
        neu_ages = np.load(f"data/predict-neutral-{r}-ages.npy")
        thr = {}
        for net, stub in [("MSE", f"mse-{r}{sfx}"), ("PGINN", f"pginn-{r}-0.1{sfx}")]:
            thr[net] = {sd: np.quantile(net_s(stub, sd, np.load(neutral_x)), 0.95) for sd in SEEDS}
        thr["Star ratio"] = np.quantile(-(neu_ages[:, 0] - neu_ages[:, 1]) / neu_ages[:, 0], 0.95)
        thr["MLE"] = np.quantile(mle_baseline.fit(np.load(f"data/predict-neutral-{r}-locusages.npy"))[:, 1], 0.95)
        rows = {k: [] for k in ["MSE", "PGINN", "Star ratio", "MLE"]}
        for mi, model in enumerate(MODELS):
            x, ages, la = simulate(model, n, loci, 900000 + 1000 * mi + 10 * n + loci, f"data/rev/demography-{r}-{model}")
            for net, stub in [("MSE", f"mse-{r}{sfx}"), ("PGINN", f"pginn-{r}-0.1{sfx}")]:
                s = {sd: net_s(stub, sd, x) for sd in SEEDS}
                fpr = np.mean([np.mean(s[sd] > thr[net][sd]) for sd in SEEDS])
                rows[net].append((fpr, np.mean([s[sd].mean() for sd in SEEDS])))
            star = -(ages[:, 0] - ages[:, 1]) / ages[:, 0]
            rows["Star ratio"].append((np.mean(star > thr["Star ratio"]), np.nan))
            s_mle = mle_baseline.fit(la)[:, 1]
            rows["MLE"].append((np.mean(s_mle > thr["MLE"]), s_mle.mean()))
            print(f"{r} {model:10s} " + "  ".join(f"{k}: FPR {v[-1][0]:.3f} mean s {v[-1][1]:.1f}" for k, v in rows.items()),
                  flush=True)
        for k, v in rows.items():
            tex.append(f"{rlabel if k == 'MSE' else ''} & {k} & " + " & ".join(
                f"{f:.2f}" + ("" if np.isnan(m) else f" ({m:.0f})") for f, m in v) + " \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    out = FIG / f"tab_robustness_demography{sfx.replace('-', '_')}.tex"
    out.write_text("\n".join(tex) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
