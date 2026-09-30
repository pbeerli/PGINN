#!/usr/bin/env python3
"""
Evaluate the MSE-only and PGINN-loss models on the held-out predict set:
produces a 2-panel (theta, s) recovery/calibration figure and a bias/RMSE/R2
table for the two-parameter (theta, s) selection case.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from train_pginn import param_labels, predict


def load_and_predict(model_path, scaler_path, x_path):
    """[log_theta_hat, s_hat] (the parameter file's own layout) for the rows of x_path."""
    pred = predict(model_path, scaler_path, x_path)
    return np.column_stack([np.log(pred[:, 0]), pred[:, 1]])


def to_natural(y):
    """[log_theta, s] -> [theta, s]."""
    out = y.copy()
    out[:, 0] = np.exp(out[:, 0])
    return out


def log1p_s(s):
    """log(1+s), the scale s is reported on; s < 0 (outside the support) counts as 0."""
    return np.log1p(np.maximum(s, 0.0))


def r2_bias_rmse(true, pred):
    bias = np.mean(pred - true)
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return bias, rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--model-mse", required=True)
    ap.add_argument("--scaler-mse", required=True)
    ap.add_argument("--model-pginn", required=True)
    ap.add_argument("--scaler-pginn", required=True)
    ap.add_argument("--out-fig", default="recovery_theta_s.pdf")
    ap.add_argument("--out-table", default="recovery_table.csv")
    args = ap.parse_args()

    y_true = to_natural(np.load(args.y))
    pred_mse = to_natural(load_and_predict(args.model_mse, args.scaler_mse, args.x))
    pred_pginn = to_natural(load_and_predict(args.model_pginn, args.scaler_pginn, args.x))

    labels = [param_labels(2)[0], r"$\log(1{+}s)$"]
    variants = [("MSE", pred_mse, "tab:blue", "o"), ("PGINN", pred_pginn, "tab:orange", "^")]

    rows = []
    for name, pred, _, _ in variants:  # raw-scale s, for reference (CSV only)
        bias, rmse, r2 = r2_bias_rmse(y_true[:, 1], pred[:, 1])
        rows.append({"param": "$s$ (raw)", "loss": name, "bias": bias, "rmse": rmse, "r2": r2})
    # s is reported on the log(1+s) scale: the prior is log-uniform and the
    # network is trained on log(1+s); predictions below 0 are outside the
    # support and count as 0
    y_true[:, 1] = log1p_s(y_true[:, 1])
    for _, pred, _, _ in variants:
        pred[:, 1] = log1p_s(pred[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for j, ax in enumerate(axes):
        t = y_true[:, j]
        for name, pred, color, marker in variants:
            p = pred[:, j]
            bias, rmse, r2 = r2_bias_rmse(t, p)
            rows.append({"param": labels[j], "loss": name, "bias": bias, "rmse": rmse, "r2": r2})
            ax.scatter(t, p, s=14, alpha=0.6, color=color, marker=marker,
                       label=f"{name} ($R^2$={r2:.3f})")
        lo, hi = t.min(), t.max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"True {labels[j]}")
        ax.set_ylabel(f"Predicted {labels[j]}")
        # axes scaled to the true-value range, so an occasional wild
        # prediction does not squeeze every other point into a corner
        if j == 0:  # theta spans 1.7 decades (log-uniform prior): log axes, plain tick labels
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo / 1.3, hi * 1.3)
            ax.set_ylim(lo / 1.3, hi * 1.3)
            plain_fmt = mticker.FuncFormatter(lambda x, _: f"{x:g}")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))  # 1-2-5 ticks per decade
                axis.set_major_formatter(plain_fmt)
                axis.set_minor_formatter(mticker.NullFormatter())
        else:
            margin = 0.1 * (hi - lo)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out_fig)
    print("Saved figure to", args.out_fig)

    with open(args.out_table, "w") as f:
        f.write("param,loss,bias,rmse,r2\n")
        for r in rows:
            f.write(f"{r['param']},{r['loss']},{r['bias']:.6g},{r['rmse']:.6g},{r['r2']:.6g}\n")
    print("Saved table to", args.out_table)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
