#!/usr/bin/env python3
"""
Does the star_ratio + PGINN combined detector (combined_detector.py),
which clearly beats star ratio alone on this project's own simulated
1-locus predict set, also improve rank-based detection on msprime's
external, real-recombination single-chromosome test
(msprime_scan_detection_summary.py)?

The combined detector there is fit fresh per training seed on the
project's own [star_ratio, PGINN s_hat] -> selected/neutral labels (a
StandardScaler + LogisticRegression pipeline, full data this time, not
cross-validated -- we need a deployable classifier here, not an
unbiased AUC estimate). Each of the 10 seed-specific classifiers is
then applied AS-IS to every window of every msprime replicate
chromosome (transferred, not refit -- msprime windows have no
independent neutral/selected labels to refit against, only one true
sweep site per chromosome among otherwise-linked windows), and the 10
seeds' predicted probabilities are averaged into one combined score per
window. Rank-based detection (report_rank, same convention as
msprime_scan_detection_summary.py) is then compared for star ratio
alone, PGINN alone, and this combined score.
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare import load_and_predict
from detect_selection import star_ratio_score, seed_paths, SEEDS
from msprime_scan_detection import run_single_chrom_detection, report_rank


def fit_combined_classifiers(x_neutral, x_selected, ages_neutral, ages_selected, pginn_stub):
    """One StandardScaler+LogisticRegression per seed, fit on the full
    [star_ratio, PGINN s_hat] -> label predict set (no CV split: this
    classifier is meant to be transferred to msprime windows, not scored
    for its own AUC here -- that was already done in combined_detector.py)."""
    n_neutral = np.load(x_neutral).shape[0]
    n_selected = np.load(x_selected).shape[0]
    labels = np.concatenate([np.zeros(n_neutral), np.ones(n_selected)])
    star = np.concatenate([star_ratio_score(ages_neutral), star_ratio_score(ages_selected)])

    classifiers = []
    for seed in SEEDS:
        model, scaler = seed_paths(pginn_stub, seed)
        s_hat = np.concatenate([
            load_and_predict(model, scaler, x_neutral)[:, 1],
            load_and_predict(model, scaler, x_selected)[:, 1],
        ])
        X = np.column_stack([star, s_hat])
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        clf.fit(X, labels)
        classifiers.append(clf)
    return classifiers


def combined_score(classifiers, star_scores, pginn_s_hat):
    X = np.column_stack([star_scores, pginn_s_hat])
    probs = np.mean([clf.predict_proba(X)[:, 1] for clf in classifiers], axis=0)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ne", type=float, default=1000.0)
    ap.add_argument("--length", type=float, default=500_000.0)
    ap.add_argument("--s", type=float, default=0.01)
    ap.add_argument("--recomb-rate", type=float, default=2e-8)
    ap.add_argument("--theta-target", type=float, default=0.02)
    ap.add_argument("--n-replicates", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--model-mse", default="model/model-mse-1l.pt")
    ap.add_argument("--scaler-mse", default="model/scaler-mse-1l.npz")
    ap.add_argument("--model-pginn", default="model/model-pginn-1l-0.1.pt")
    ap.add_argument("--scaler-pginn", default="model/scaler-pginn-1l-0.1.npz")
    ap.add_argument("--json-dir", default="data/msprime-scan-detection-combined")
    args = ap.parse_args()

    print("Fitting 10 seed-specific combined classifiers on the project's own "
          "1-locus, n=10 predict set...")
    classifiers = fit_combined_classifiers(
        "data/predict-neutral-1l-features.npy", "data/predict-sweep-1l-features.npy",
        "data/predict-neutral-1l-ages.npy", "data/predict-sweep-1l-ages.npy",
        "pginn-1l-0.1")

    labels = ["star ratio", "MSE predicted s", "PGINN predicted s", "star+PGINN combined"]
    fractions = {label: [] for label in labels}
    n_trees_per_rep = []

    for j in range(args.n_replicates):
        seed = args.seed0 + 1000 * j
        sweep_pos, positions, results, star_scores = run_single_chrom_detection(
            args.n, args.ne, args.length, args.s, args.recomb_rate, None, args.theta_target,
            seed, args.model_mse, args.scaler_mse, args.model_pginn, args.scaler_pginn, args.json_dir)
        combo_scores = combined_score(classifiers, star_scores, results["PGINN"])
        n_trees_per_rep.append(len(positions))
        for label, scores in zip(labels, [star_scores, results["MSE"], results["PGINN"], combo_scores]):
            _, _, frac = report_rank(label, positions, scores, sweep_pos, None, verbose=False)
            fractions[label].append(frac)
        print(f"[{j+1}/{args.n_replicates}] seed {seed}: {len(positions)} trees, "
              f"top_fraction star={fractions['star ratio'][-1]:.2f} "
              f"PGINN={fractions['PGINN predicted s'][-1]:.2f} "
              f"combined={fractions['star+PGINN combined'][-1]:.2f}")

    print(f"\nSummary over {args.n_replicates} independent single-chromosome replicates "
          f"(median {int(np.median(n_trees_per_rep))} real trees/chromosome):")
    print(f"{'detector':22s} {'median top%':>12s} {'mean top%':>10s} {'top10%':>8s} {'top20%':>8s} {'top25%':>8s}")
    for label in labels:
        f = np.array(fractions[label])
        print(f"{label:22s} {100*np.median(f):11.1f}% {100*np.mean(f):9.1f}% "
              f"{100*np.mean(f<=0.10):7.1f}% {100*np.mean(f<=0.20):7.1f}% {100*np.mean(f<=0.25):7.1f}%")


if __name__ == "__main__":
    main()
