#!/usr/bin/env python3
"""
Does combining the classical star-ratio statistic with a network's
predicted s_hat improve detection (neutral vs. selected) beyond either
alone? detect_selection.py already showed star ratio alone is a strong
detector and beats both networks at 10 loci, so the practical question
is whether star_ratio + s_hat, as two inputs to a simple classifier,
dominates star_ratio by itself.

Five detectors per regime, fit as a 2-feature (or 1-feature) logistic
regression, evaluated by 5-fold stratified cross-validation (out-of-fold
predicted probabilities, so AUC/PR-AUC are not fit on their own
training data):
  1. star_ratio alone (deterministic, no seed dependence)
  2. MSE s_hat alone
  3. PGINN s_hat alone
  4. star_ratio + MSE s_hat
  5. star_ratio + PGINN s_hat

Variants 2-5 depend on the trained network, so they are evaluated at
all 10 training seeds (67-76), mean +/- SD, with paired t-tests between
the variants that share the same seed (e.g. PGINN alone vs.
star_ratio+PGINN). star_ratio alone has no seed variance and is reported
as a single reference value.
"""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare import load_and_predict
from detect_selection import star_ratio_score, seed_paths, SEEDS

N_FOLDS = 5
CV_SEED = 0  # fixed fold assignment so seeds/variants are compared on the same splits


def cv_auc_prauc(X, labels):
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros(len(labels))
    for train_idx, test_idx in kf.split(X, labels):
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        clf.fit(X[train_idx], labels[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return roc_auc_score(labels, oof), average_precision_score(labels, oof)


def evaluate_regime(name, x_neutral, x_selected, ages_neutral, ages_selected, mse_stub, pginn_stub, rows):
    n_neutral = np.load(x_neutral).shape[0]
    n_selected = np.load(x_selected).shape[0]
    labels = np.concatenate([np.zeros(n_neutral), np.ones(n_selected)])

    star = np.concatenate([star_ratio_score(ages_neutral), star_ratio_score(ages_selected)])
    star_auc, star_pr = cv_auc_prauc(star.reshape(-1, 1), labels)
    rows.append(dict(regime=name, variant="star_ratio", seed="", auc=star_auc, pr_auc=star_pr))
    print(f"\n=== {name} ===")
    print(f"star_ratio alone: AUC={star_auc:.4f} PR-AUC={star_pr:.4f} (deterministic)")

    per_variant = {"MSE": [], "PGINN": [], "star+MSE": [], "star+PGINN": [], "star+MSE+PGINN": []}
    for seed in SEEDS:
        s_hat = {}
        for tag, stub in [("MSE", mse_stub), ("PGINN", pginn_stub)]:
            model, scaler = seed_paths(stub, seed)
            s_hat_neutral = load_and_predict(model, scaler, x_neutral)[:, 1]
            s_hat_selected = load_and_predict(model, scaler, x_selected)[:, 1]
            s_hat[tag] = np.concatenate([s_hat_neutral, s_hat_selected])

        for tag, X in [
            ("MSE", s_hat["MSE"].reshape(-1, 1)),
            ("PGINN", s_hat["PGINN"].reshape(-1, 1)),
            ("star+MSE", np.column_stack([star, s_hat["MSE"]])),
            ("star+PGINN", np.column_stack([star, s_hat["PGINN"]])),
            ("star+MSE+PGINN", np.column_stack([star, s_hat["MSE"], s_hat["PGINN"]])),
        ]:
            a, pr = cv_auc_prauc(X, labels)
            per_variant[tag].append((a, pr))
            rows.append(dict(regime=name, variant=tag, seed=seed, auc=a, pr_auc=pr))

    for tag, vals in per_variant.items():
        aucs = np.array([v[0] for v in vals])
        prs = np.array([v[1] for v in vals])
        print(f"{tag:12s}: AUC={aucs.mean():.4f}+/-{aucs.std(ddof=1):.4f}  "
              f"PR-AUC={prs.mean():.4f}+/-{prs.std(ddof=1):.4f}")

    print("Paired t-tests (AUC, PGINN-vs-MSE variants, across the 10 seeds):")
    for a_tag, b_tag in [("PGINN", "MSE"), ("star+PGINN", "star+MSE"),
                         ("star+PGINN", "PGINN"), ("star+MSE", "MSE"),
                         ("star+MSE+PGINN", "star+PGINN"), ("star+MSE+PGINN", "star+MSE")]:
        a = np.array([v[0] for v in per_variant[a_tag]])
        b = np.array([v[0] for v in per_variant[b_tag]])
        t, p = stats.ttest_rel(a, b)
        print(f"  {a_tag} - {b_tag}: mean diff={a.mean()-b.mean():+.4f}  t={t:+.3f}  p={p:.4f}")

    n_exceed = sum(v[0] > star_auc for v in per_variant["star+PGINN"])
    print(f"star+PGINN AUC exceeds star_ratio-alone ({star_auc:.4f}) in {n_exceed}/10 seeds")


def plot_bar_chart(rows, out_fig):
    regimes = ["10 loci, n=10", "10 loci, n=20", "1 locus, n=10", "1 locus, n=20"]
    variants = ["star_ratio", "MSE", "PGINN", "star+MSE", "star+PGINN", "star+MSE+PGINN"]
    colors = {"star_ratio": "tab:green", "MSE": "tab:blue", "PGINN": "tab:orange",
              "star+MSE": "tab:blue", "star+PGINN": "tab:orange", "star+MSE+PGINN": "tab:red"}
    hatches = {"star+MSE": "//", "star+PGINN": "//", "star+MSE+PGINN": "//"}
    labels = {"star_ratio": "star ratio alone", "MSE": "MSE $\\hat s$ alone", "PGINN": "PGINN $\\hat s$ alone",
              "star+MSE": "star ratio + MSE", "star+PGINN": "star ratio + PGINN",
              "star+MSE+PGINN": "star ratio + MSE + PGINN"}

    means, sds = {}, {}
    for regime in regimes:
        for v in variants:
            aucs = np.array([r["auc"] for r in rows if r["regime"] == regime and r["variant"] == v])
            means[(regime, v)] = aucs.mean()
            sds[(regime, v)] = aucs.std(ddof=1) if len(aucs) > 1 else 0.0

    x = np.arange(len(regimes))
    width = 0.13
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, v in enumerate(variants):
        vals = [means[(r, v)] for r in regimes]
        errs = [sds[(r, v)] for r in regimes]
        ax.bar(x + (i - 2.5) * width, vals, width, yerr=errs, capsize=2, label=labels[v], color=colors[v],
               hatch=hatches.get(v), edgecolor="white" if v in hatches else None)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("AUC")
    lo = min(means[k] - sds[k] for k in means)
    ax.set_ylim(np.floor((lo - 0.02) * 20) / 20, 1.0)  # from the data, rounded down to 0.05
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig)
    print(f"Saved figure to {out_fig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-table", default="../figures/combined_detector.csv")
    ap.add_argument("--out-fig", default="../figures/combined_detector.pdf")
    ap.add_argument("--out-tex", default="../figures/tab_combined_detector.tex")
    args = ap.parse_args()

    rows = []
    evaluate_regime(
        "10 loci, n=10",
        "data/predict-neutral-10l-features.npy", "data/predict-sweep-10l-features.npy",
        "data/predict-neutral-10l-ages.npy", "data/predict-sweep-10l-ages.npy",
        "mse-10l", "pginn-10l-0.1", rows)

    evaluate_regime(
        "10 loci, n=20",
        "data/predict-neutral-10l-n20-features.npy", "data/predict-sweep-10l-n20-features.npy",
        "data/predict-neutral-10l-n20-ages.npy", "data/predict-sweep-10l-n20-ages.npy",
        "mse-10l-n20", "pginn-10l-n20-0.1", rows)

    evaluate_regime(
        "1 locus, n=10",
        "data/predict-neutral-1l-features.npy", "data/predict-sweep-1l-features.npy",
        "data/predict-neutral-1l-ages.npy", "data/predict-sweep-1l-ages.npy",
        "mse-1l", "pginn-1l-0.1", rows)

    evaluate_regime(
        "1 locus, n=20",
        "data/predict-neutral-1l-n20-features.npy", "data/predict-sweep-1l-n20-features.npy",
        "data/predict-neutral-1l-n20-ages.npy", "data/predict-sweep-1l-n20-ages.npy",
        "mse-1l-n20", "pginn-1l-n20-0.1", rows)

    with open(args.out_table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "variant", "seed", "auc", "pr_auc"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "auc": f"{r['auc']:.6f}", "pr_auc": f"{r['pr_auc']:.6f}"})
    print(f"\nSaved table to {args.out_table}")

    plot_bar_chart(rows, args.out_fig)
    write_tex(rows, args.out_tex)


def write_tex(rows, path):
    """tab:combined-detector body: AUC/PR-AUC, mean +- SD across seeds for the
    combinations; bold = best (mean) AUC per regime, star ratio included (ties all bold at 3 decimals)."""
    variants = [("star+MSE", "$+$ MSE $\\hat s$"), ("star+PGINN", "$+$ \\pginn $\\hat s$"),
                ("star+MSE+PGINN", "$+$ MSE $+$ \\pginn $\\hat s$")]
    lines = ["\\begin{tabularx}{\\textwidth}{@{}cc@{\\hspace{3mm}}c@{\\hspace{3mm}}c@{\\hspace{3mm}}c@{\\hspace{3mm}}c}",
             "\\toprule", "Loci/$n$ & Star ratio & " + " & ".join(v[1] for v in variants) + " \\\\", "\\midrule"]
    for regime in dict.fromkeys(r["regime"] for r in rows):
        rr = [r for r in rows if r["regime"] == regime]
        loci, n = regime.split(" loc")[0], regime.split("n=")[1]
        star = next(r for r in rr if r["variant"] == "star_ratio")
        stats = {}
        for v, _ in variants:
            a = np.array([r["auc"] for r in rr if r["variant"] == v])
            p = np.array([r["pr_auc"] for r in rr if r["variant"] == v])
            stats[v] = (a.mean(), a.std(ddof=1), p.mean(), p.std(ddof=1))
        best = max([round(x[0], 3) for x in stats.values()] + [round(star["auc"], 3)])
        sb1, sb2 = ("\\textbf{", "}") if round(star["auc"], 3) == best else ("", "")
        cells = []
        for v, _ in variants:
            am, asd, pm, psd = stats[v]
            b1, b2 = ("\\textbf{", "}") if round(am, 3) == best else ("", "")
            cells.append(f"{b1}{am:.3f}{b2}$\\pm${asd:.3f}/{b1}{pm:.3f}{b2}$\\pm${psd:.3f}")
        lines.append(f"{loci}/{n} & {sb1}{star['auc']:.3f}{sb2}/{sb1}{star['pr_auc']:.3f}{sb2} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}"]
    open(path, "w").write("\n".join(lines) + "\n")
    print("Saved LaTeX table to", path)


if __name__ == "__main__":
    main()
