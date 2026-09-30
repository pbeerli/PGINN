#!/usr/bin/env python3
"""
Does PGINN's seed-to-seed variance reduction (Sec. ultra-scarce, Sec. detection)
concentrate on genealogies that are "ambiguous" under the sweep model (star
ratio close to the neutral expectation), or is it uniform / driven by one
globally-bad MSE seed? Uses the 10 already-trained seed checkpoints per
loss/regime (code/model/); no new training. Writes the summary table behind
Table~\\ref{tab:variance-mechanism} to ../figures/variance_mechanism_table.csv.
"""
import numpy as np
from scipy.stats import pearsonr

from compare import load_and_predict, log1p_s

SEEDS = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76]


def seed_paths(stub, seed):
    suffix = "" if seed == 67 else f"-seed{seed}"
    return f"model/model-{stub}{suffix}.pt", f"model/scaler-{stub}{suffix}.npz"


REGIMES = {
    "10 loci, n=10": ("10l", "mse-10l", "pginn-10l-0.1"),
    "10 loci, n=20": ("10l-n20", "mse-10l-n20", "pginn-10l-n20-0.1"),
    "1 locus, n=10": ("1l", "mse-1l", "pginn-1l-0.1"),
    "1 locus, n=20": ("1l-n20", "mse-1l-n20", "pginn-1l-n20-0.1"),
}


def run_regime(regime_name, data_stub, mse_stub, pginn_stub):
    print(f"\n{'='*70}\nREGIME: {regime_name}\n{'='*70}")
    X = f"data/predict-sweep-{data_stub}-features.npy"
    y_true = np.load(f"data/predict-sweep-{data_stub}-params.npy")  # [log_theta, s]
    ages = np.load(f"data/predict-sweep-{data_stub}-ages.npy")  # sorted descending
    s_true = y_true[:, 1]
    ls_true = log1p_s(s_true)  # s is compared on the log(1+s) scale
    star_ratio = (ages[:, 0] - ages[:, 1]) / ages[:, 0]

    preds = {}  # tag -> (10 seeds, N) log(1+s_hat)
    for tag, stub in [("MSE", mse_stub), ("PGINN", pginn_stub)]:
        rows = []
        for seed in SEEDS:
            model, scaler = seed_paths(stub, seed)
            s_hat = log1p_s(load_and_predict(model, scaler, X)[:, 1])
            rows.append(s_hat)
        preds[tag] = np.stack(rows)  # (10, N)

    # per-seed R^2 for s, to identify MSE's known bad seed(s)
    print("=== per-seed s R^2 (sanity check against paper Table oneloc n=20) ===")
    ss_tot = np.sum((ls_true - ls_true.mean()) ** 2)
    for tag in ("MSE", "PGINN"):
        r2s = []
        for i, seed in enumerate(SEEDS):
            ss_res = np.sum((ls_true - preds[tag][i]) ** 2)
            r2s.append(1 - ss_res / ss_tot)
        print(f"{tag}: " + ", ".join(f"{seed}:{r2:.3f}" for seed, r2 in zip(SEEDS, r2s)))

    # per-example std across the 10 seeds (ensemble disagreement)
    std_per_example = {tag: preds[tag].std(axis=0) for tag in ("MSE", "PGINN")}
    print("\n=== per-example seed-std of s_hat, mean across 400 examples ===")
    for tag in ("MSE", "PGINN"):
        print(f"{tag}: mean={std_per_example[tag].mean():.1f}  median={np.median(std_per_example[tag]):.1f}")

    # ambiguity proxies: star_ratio (smaller => more star-like => less ambiguous
    # about "selected"), and true s itself (smaller => weaker, more ambiguous signal)
    print("\n=== correlation: per-example seed-std(s_hat) vs ambiguity proxies ===")
    for tag in ("MSE", "PGINN"):
        r_star, p_star = pearsonr(star_ratio, std_per_example[tag])
        r_strue, p_strue = pearsonr(s_true, std_per_example[tag])
        print(f"{tag}: corr(std, star_ratio) = {r_star:+.3f} (p={p_star:.2g})   "
              f"corr(std, true_s) = {r_strue:+.3f} (p={p_strue:.2g})")

    # quartile breakdown by star_ratio (ambiguous = high star_ratio, closer to
    # neutral expectation ~0.42-0.44 at n=20 per appendix) and by true s
    print("\n=== mean per-example seed-std(s_hat), by star_ratio quartile (Q1=most star-like/least ambiguous) ===")
    q = np.quantile(star_ratio, [0.25, 0.5, 0.75])
    bins = np.digitize(star_ratio, q)
    for b in range(4):
        mask = bins == b
        row = " ".join(f"{tag}={std_per_example[tag][mask].mean():7.1f}" for tag in ("MSE", "PGINN"))
        print(f"  Q{b+1} (n={mask.sum():3d}, star_ratio {star_ratio[mask].min():.3f}-{star_ratio[mask].max():.3f}): {row}")

    print("\n=== mean per-example seed-std(s_hat), by true-s bin (paper's detection bins) ===")
    bin_edges = [1, 10, 50, 200, 1000.0001]
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (s_true >= lo) & (s_true < hi)
        if mask.sum() == 0:
            continue
        row = " ".join(f"{tag}={std_per_example[tag][mask].mean():7.1f}" for tag in ("MSE", "PGINN"))
        print(f"  s in [{lo:>6},{hi:>6}) n={mask.sum():3d}: {row}")

    # redo the headline stat leaving out MSE's single worst seed, to separate
    # "one globally bad seed" from "specific hard examples across all seeds"
    print("\n=== same, MSE with its single worst seed (lowest R^2) excluded ===")
    ss_res_per_seed = np.array([np.sum((ls_true - preds["MSE"][i]) ** 2) for i in range(10)])
    worst = np.argmax(ss_res_per_seed)
    print(f"excluding seed {SEEDS[worst]}")
    mse_trimmed = np.delete(preds["MSE"], worst, axis=0)
    std_trimmed = mse_trimmed.std(axis=0)
    print(f"MSE (9 seeds) mean std={std_trimmed.mean():.1f}  median={np.median(std_trimmed):.1f}")
    r_star_trimmed, p_star_trimmed = pearsonr(star_ratio, std_trimmed)
    print(f"corr(std, star_ratio) = {r_star_trimmed:+.3f} (p={p_star_trimmed:.2g})")

    q4_gap = std_per_example["MSE"][bins == 3].mean() - std_per_example["PGINN"][bins == 3].mean()
    q1_gap = std_per_example["MSE"][bins == 0].mean() - std_per_example["PGINN"][bins == 0].mean()
    return {
        "regime": regime_name,
        "mse_mean_std": std_per_example["MSE"].mean(),
        "pginn_mean_std": std_per_example["PGINN"].mean(),
        "reduction_pct": 100 * (1 - std_per_example["PGINN"].mean() / std_per_example["MSE"].mean()),
        "corr_mse": pearsonr(star_ratio, std_per_example["MSE"])[0],
        "corr_pginn": pearsonr(star_ratio, std_per_example["PGINN"])[0],
        "q4_over_q1_gap_ratio": q4_gap / q1_gap,
        "mse_mean_std_worst_seed_excluded": std_trimmed.mean(),
        "mse_pct_change_worst_seed_excluded": 100 * (std_trimmed.mean() / std_per_example["MSE"].mean() - 1),
    }


def main():
    rows = [run_regime(name, *stubs) for name, stubs in REGIMES.items()]
    out_path = "../figures/variance_mechanism_table.csv"
    with open(out_path, "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else f'"{v}"' for v in r.values()) + "\n")
    print(f"\nSaved summary table to {out_path}")

    tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Regime & MSE $\\bar\\sigma$ & \\pginn $\\bar\\sigma$ & Reduction & corr($\\sigma$, star ratio) & MSE $\\bar\\sigma$ \\\\",
           " & & & & MSE\\ /\\ \\pginn & (worst replicate out) \\\\", "\\midrule"]
    for r in rows:
        loci, n = r["regime"].split(", n=")
        tex.append(f"{loci}, $n{{=}}{n}$ & {r['mse_mean_std']:.2f} & {r['pginn_mean_std']:.2f} & "
                   f"{r['reduction_pct']:.0f}\\% & {r['corr_mse']:.2f} / {r['corr_pginn']:.2f} & "
                   f"{r['mse_mean_std_worst_seed_excluded']:.2f} "
                   f"(${r['mse_pct_change_worst_seed_excluded']:+.0f}\\%$) \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    tex_path = "../figures/tab_variance_mechanism.tex"
    open(tex_path, "w").write("\n".join(tex) + "\n")
    print(f"Saved LaTeX table to {tex_path}")


if __name__ == "__main__":
    main()
