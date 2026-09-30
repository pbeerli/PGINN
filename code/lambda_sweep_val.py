#!/usr/bin/env python3
"""PGINN weight (w) sweep, seed 67, scored on the 100-example validation
split (test-*), never on the independently simulated predict set. Prints a
markdown table and writes figures/tab_lambda_sweep.tex (bold: best
validation log(1+s) R^2 per regime)."""
import os
from pathlib import Path
import numpy as np
from compare import load_and_predict, to_natural, r2_bias_rmse, log1p_s

REGIMES = [("10l", "10 loci, $n{=}10$"), ("10l-n20", "10 loci, $n{=}20$"),
           ("1l", "1 locus, $n{=}10$"), ("1l-n20", "1 locus, $n{=}20$")]
WS = [0.05, 0.1, 0.2, 0.3]
OUT_TEX = Path(__file__).resolve().parent.parent / "figures" / "tab_lambda_sweep.tex"

tex = ["\\begin{tabular}{clrrr}", "\\toprule",
       "Regime & $w$ & $\\theta$ $R^2$ (val) & $\\log(1{+}s)$ $R^2$ (val) & $\\log(1{+}s)$ RMSE (val) \\\\", "\\midrule"]
print("| regime | w | theta R2 (val) | log(1+s) R2 (val) | log(1+s) RMSE (val) |")
print("|---|---|---|---|---|")
for regime, label in REGIMES:
    y_true = to_natural(np.load(f"data/test-sweep-{regime}-params.npy"))
    rows = []
    for w in WS:
        pred = to_natural(load_and_predict(
            f"model/model-pginn-{regime}-{w}.pt", f"model/scaler-pginn-{regime}-{w}.npz",
            f"data/test-sweep-{regime}-features.npy"))
        _, _, r2_t = r2_bias_rmse(y_true[:, 0], pred[:, 0])
        _, rmse_s, r2_s = r2_bias_rmse(log1p_s(y_true[:, 1]), log1p_s(pred[:, 1]))
        rows.append((w, r2_t, r2_s, rmse_s))
        print(f"| {regime} | {w} | {r2_t:.4f} | {r2_s:.4f} | {rmse_s:.3f} |")
    best = max(rows, key=lambda r: r[2])[0]
    for w, r2_t, r2_s, rmse_s in rows:
        b = (lambda x: f"\\textbf{{{x}}}") if w == best else (lambda x: x)
        tex.append(f"{label} & {b(f'{w:.2f}')} & {r2_t:.3f} & {b(f'{r2_s:.3f}')} & {b(f'{rmse_s:.2f}')} \\\\")
    # MSE baseline for the same regime, for reference (printed only)
    pred = to_natural(load_and_predict(
        f"model/model-mse-{regime}.pt", f"model/scaler-mse-{regime}.npz",
        f"data/test-sweep-{regime}-features.npy"))
    _, _, r2_t = r2_bias_rmse(y_true[:, 0], pred[:, 0])
    _, rmse_s, r2_s = r2_bias_rmse(log1p_s(y_true[:, 1]), log1p_s(pred[:, 1]))
    print(f"| {regime} | MSE | {r2_t:.4f} | {r2_s:.4f} | {rmse_s:.3f} |")
tex += ["\\bottomrule", "\\end{tabular}"]
OUT_TEX.write_text("\n".join(tex) + "\n")
print("wrote", os.path.relpath(OUT_TEX))
