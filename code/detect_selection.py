#!/usr/bin/env python3
"""
Can the already-trained (theta, s) regression models tell whether a locus is
under selection at all, not just estimate s given that it is? We combine each
regime's held-out selected predict set (true s > 0, log s ~ U(log 1, log 1000)) with a
matched held-out neutral predict set (true s == 0), score both with the
existing MSE and PGINN models' predicted s_hat, and treat s_hat as a
detection statistic: ROC/AUC for "true s > 0" vs. "true s == 0", plus a
sensitivity breakdown by true-s magnitude (weak selection should be harder to
detect than strong selection).

Scored across all 10 training seeds (67-76) per regime/loss, reporting
mean +/- SD, the same standard used for the continuous-recovery tables
(review round 2, issue #1: detection was previously seed-67-only).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score

from compare import load_and_predict

SEEDS = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
FPR = 0.05  # sensitivity is reported at this false-positive rate on the neutral predict set


def roc_curve_manual(labels, scores):
    """Standard rank-based ROC: labels in {0,1}, higher score = more 'positive'."""
    order = np.argsort(-scores)
    labels = labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = np.concatenate(([0.0], tps / tps[-1]))
    fpr = np.concatenate(([0.0], fps / fps[-1]))
    return fpr, tpr


def sensitivity_by_bin(s_true, s_hat, threshold, bins):
    """Fraction of true-selected examples with s_hat > threshold, per true-s bin."""
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s_true >= lo) & (s_true < hi)
        n = mask.sum()
        if n == 0:
            continue
        detected = (s_hat[mask] > threshold).sum()
        rows.append((lo, hi, n, detected / n))
    return rows


def star_ratio_score(ages_path):
    """Classical, non-network detection statistic: star-likeness of the
    (Monte-Carlo-averaged) genealogy, same formula as calibrate_sweep.py's
    summarize(). Smaller star_ratio = more star-like = more likely selected,
    so we return -star_ratio (higher score = more positive-for-selection),
    matching the convention roc_curve_manual expects."""
    ages = np.load(ages_path)  # (n_examples, K), sorted descending
    star_ratio = (ages[:, 0] - ages[:, 1]) / ages[:, 0]
    return -star_ratio


def seed_paths(stub, seed):
    """model-{stub}.pt / scaler-{stub}.npz for seed 67 (canonical, no suffix);
    model-{stub}-seed{N}.pt / scaler-{stub}-seed{N}.npz otherwise."""
    suffix = "" if seed == 67 else f"-seed{seed}"
    return f"model/model-{stub}{suffix}.pt", f"model/scaler-{stub}{suffix}.npz"


def evaluate_regime(name, x_neutral, x_selected, y_selected, ages_neutral, ages_selected,
                     mse_stub, pginn_stub, ax, table_rows):
    n_neutral = np.load(x_neutral).shape[0]
    s_true_selected = np.load(y_selected)[:, 1]
    labels = np.concatenate([np.zeros(n_neutral), np.ones(len(s_true_selected))])
    bins = [1, 10, 50, 200, 1000.0001]

    # Per-seed AUC and per-bin sensitivity for each network loss.
    per_loss = {}
    for tag, stub in [("MSE", mse_stub), ("PGINN", pginn_stub)]:
        seed_aucs, seed_sens = [], {}
        rep_fpr = rep_tpr = None  # seed-67 curve, plotted as representative
        for seed in SEEDS:
            model, scaler = seed_paths(stub, seed)
            s_hat_neutral = load_and_predict(model, scaler, x_neutral)[:, 1]
            s_hat_selected = load_and_predict(model, scaler, x_selected)[:, 1]
            scores = np.concatenate([s_hat_neutral, s_hat_selected])
            fpr, tpr = roc_curve_manual(labels, scores)
            seed_aucs.append(roc_auc_score(labels, scores))  # handles tied scores (s_hat bounded at 0)
            if seed == 67:
                rep_fpr, rep_tpr = fpr, tpr
            threshold = np.quantile(s_hat_neutral, 1.0 - FPR)  # fixed false-positive rate on the neutral set
            for lo, hi, n, sens in sensitivity_by_bin(s_true_selected, s_hat_selected, threshold, bins):
                seed_sens.setdefault((lo, hi, n), []).append(sens)
        mean_auc, sd_auc = float(np.mean(seed_aucs)), float(np.std(seed_aucs, ddof=1))
        ax.plot(rep_fpr, rep_tpr, label=f"{tag} (AUC={mean_auc:.3f}$\\pm${sd_auc:.3f})")
        per_loss[tag] = {"mean_auc": mean_auc, "sd_auc": sd_auc, "seed_aucs": seed_aucs, "sens": seed_sens}

    # Classical (non-network) baseline: star ratio of the same averaged genealogy.
    # Deterministic given the data (no trained model), so no seed variance.
    star_neutral = star_ratio_score(ages_neutral)
    star_selected = star_ratio_score(ages_selected)
    star_scores = np.concatenate([star_neutral, star_selected])
    fpr_c, tpr_c = roc_curve_manual(labels, star_scores)
    a_c = roc_auc_score(labels, star_scores)
    ax.plot(fpr_c, tpr_c, label=f"Classical: star ratio (AUC={a_c:.3f})", ls=":", color="tab:green")
    table_rows.append({"regime": name, "loss": "Classical:star_ratio", "metric": "AUC", "bin": "", "n": "", "value": f"{a_c:.6f}"})
    threshold_c = np.quantile(star_neutral, 1.0 - FPR)
    for lo, hi, n, sens in sensitivity_by_bin(s_true_selected, star_selected, threshold_c, bins):
        table_rows.append({"regime": name, "loss": "Classical:star_ratio", "metric": "sensitivity",
                            "bin": f"[{lo},{hi})", "n": n, "value": f"{sens:.3f}"})

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(name)
    ax.legend(fontsize=8, loc="lower right")

    print(f"\n=== {name}: 10-seed mean+/-SD AUC, sensitivity by true-s bin at {100 * FPR:.0f}% false positives ===")
    t, p = ttest_rel(per_loss["PGINN"]["seed_aucs"], per_loss["MSE"]["seed_aucs"])
    print(f"paired t-test PGINN vs MSE AUC across seeds: t={t:+.3f} p={p:.4f}; star ratio AUC {a_c:.4f}")
    for tag in ("MSE", "PGINN"):
        r = per_loss[tag]
        print(f"-- {tag}: AUC {r['mean_auc']:.4f} +/- {r['sd_auc']:.4f} (seeds: {[f'{a:.4f}' for a in r['seed_aucs']]}) --")
        table_rows.append({"regime": name, "loss": tag, "metric": "AUC", "bin": "", "n": "",
                            "value": f"{r['mean_auc']:.4f}+/-{r['sd_auc']:.4f}"})
        for (lo, hi, n), vals in r["sens"].items():
            mean_s, sd_s = float(np.mean(vals)), float(np.std(vals, ddof=1))
            print(f"  s in [{lo:>6},{hi:>6}): n={n:3d}  sensitivity={mean_s:.3f}+/-{sd_s:.3f}")
            table_rows.append({"regime": name, "loss": tag, "metric": "sensitivity",
                                "bin": f"[{lo},{hi})", "n": n, "value": f"{mean_s:.3f}+/-{sd_s:.3f}"})
    return per_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-fig", default="../figures/detection_roc.pdf")
    ap.add_argument("--out-table", default="../figures/detection_table.csv")
    ap.add_argument("--out-tex", default="../figures/tab_detection.tex")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    table_rows = []

    evaluate_regime(
        "10 loci, n=10",
        "data/predict-neutral-10l-features.npy",
        "data/predict-sweep-10l-features.npy", "data/predict-sweep-10l-params.npy",
        "data/predict-neutral-10l-ages.npy", "data/predict-sweep-10l-ages.npy",
        "mse-10l", "pginn-10l-0.1",
        axes[0][0], table_rows)

    evaluate_regime(
        "10 loci, n=20",
        "data/predict-neutral-10l-n20-features.npy",
        "data/predict-sweep-10l-n20-features.npy", "data/predict-sweep-10l-n20-params.npy",
        "data/predict-neutral-10l-n20-ages.npy", "data/predict-sweep-10l-n20-ages.npy",
        "mse-10l-n20", "pginn-10l-n20-0.1",
        axes[0][1], table_rows)

    evaluate_regime(
        "1 locus, n=10",
        "data/predict-neutral-1l-features.npy",
        "data/predict-sweep-1l-features.npy", "data/predict-sweep-1l-params.npy",
        "data/predict-neutral-1l-ages.npy", "data/predict-sweep-1l-ages.npy",
        "mse-1l", "pginn-1l-0.1",
        axes[1][0], table_rows)

    evaluate_regime(
        "1 locus, n=20",
        "data/predict-neutral-1l-n20-features.npy",
        "data/predict-sweep-1l-n20-features.npy", "data/predict-sweep-1l-n20-params.npy",
        "data/predict-neutral-1l-n20-ages.npy", "data/predict-sweep-1l-n20-ages.npy",
        "mse-1l-n20", "pginn-1l-n20-0.1",
        axes[1][1], table_rows)

    fig.tight_layout()
    fig.savefig(args.out_fig)
    print("\nSaved figure to", args.out_fig)

    with open(args.out_table, "w") as f:
        f.write("regime,loss,metric,bin,n,value\n")
        for r in table_rows:
            f.write(f"{r['regime']},{r['loss']},{r['metric']},{r['bin']},{r['n']},{r['value']}\n")
    print("Saved table to", args.out_table)
    write_tex(table_rows, args.out_tex)


def write_tex(table_rows, path):
    """tab:detection body: AUC (bold = best mean AUC per regime, ties all bold at 3 decimals) and per-bin
    sensitivity at threshold s_hat>0 (mean across seeds, subscript = bin size)."""
    lines = ["\\begin{tabular}{lllrrrrr}", "\\toprule",
             "Loci & $n$ & Detector & AUC (mean$\\pm$SD) & \\multicolumn{4}{c}{Sensitivity at 5\\% false positives, by true $s$, mean$_{(n)}$} \\\\",
             "& & & & $[1,10)$ & $[10,50)$ & $[50,200)$ & $[200,1000]$ \\\\", "\\midrule"]
    for regime in dict.fromkeys(r["regime"] for r in table_rows):
        rows = [r for r in table_rows if r["regime"] == regime]
        loci, n = regime.split(" loc")[0], regime.split("n=")[1]
        aucs = {r["loss"]: float(r["value"].split("+/-")[0]) for r in rows if r["metric"] == "AUC"}
        best = max(round(v, 3) for v in aucs.values())
        for loss, label in [("Classical:star_ratio", "Classical: star ratio"), ("MSE", "MSE"), ("PGINN", "\\pginn")]:
            auc_val = next(r["value"] for r in rows if r["loss"] == loss and r["metric"] == "AUC")
            if "+/-" in auc_val:
                m, sd = (float(v) for v in auc_val.split("+/-"))
                cell = f"{m:.3f}\\pm{sd:.3f}"
                cell = f"$\\mathbf{{{cell}}}$" if round(m, 3) == best else f"${cell}$"
                sens = [f"${float(r['value'].split('+/-')[0]):.2f}_{{({r['n']})}}$"
                        for r in rows if r["loss"] == loss and r["metric"] == "sensitivity"]
            else:
                cell = f"{float(auc_val):.3f}"
                cell = f"\\textbf{{{cell}}}" if round(float(auc_val), 3) == best else cell
                sens = [f"${float(r['value']):.2f}_{{({r['n']})}}$"
                        for r in rows if r["loss"] == loss and r["metric"] == "sensitivity"]
            sens += ["---"] * (4 - len(sens))
            lines.append(f"{loci} & {n} & {label} & {cell} & " + " & ".join(sens) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    print("Saved LaTeX table to", path)


if __name__ == "__main__":
    main()
