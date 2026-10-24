#!/usr/bin/env python3
"""Evaluation of the nested design (reproduce.sh revision-simulate /
revision-train): per regime, D independent data sets (own training,
validation, predict-sweep and predict-neutral sets), each with networks
trained from 3 seeds, for MSE, PGINN (w=0.1), the exact-hazard loss and
the log(1+s)-weighted MSE; plus the maximum-likelihood baseline
(mle_baseline.py) and the star ratio, which need no training.

The independent data set is the unit of replication: per data set, each
network metric is averaged over the seeds; differences to MSE are then
summarized over the D data sets (mean, t-based 95% CI, number of data sets
in which the method is better). The seed SD within data sets and the SD
between data sets are both reported.

Detection: AUC on predict-sweep vs predict-neutral of the same data set.
The 5% threshold for each statistic is taken from the neutral set of a
different data set (d+1, cyclically), so the realized false-positive rate
and the sensitivities are measured on data not used for the threshold.

Errors by range: bias and RMSE of log(1+s) and of theta in bins of the
true values, pooled over data sets and seeds, including the s=0 neutral
sets; also how often predictions sit on the prior bounds (clipped).

Writes figures/tab_nested_recovery.tex, tab_nested_detection.tex,
tab_binned_errors.tex, and the main-text extracts tab_nested_main.tex and
tab_nested_detection_main.tex (MAIN_REC, MAIN_DET), and tab_nested_main_v3.tex
(MAIN_REC_V3: log-scaled features only, for version 3 of the manuscript).
tab_nested_recovery.tex and tab_nested_main_v3.tex report the error reduction
relative to the MSE network with the same input, 1 - (1-R^2)/(1-R^2_MSE),
instead of the difference in R^2."""
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from compare import load_and_predict, to_natural, r2_bias_rmse, log1p_s
import mle_baseline

REGIMES = [("10l", "10 loci, $n{=}10$"), ("10l-n20", "10 loci, $n{=}20$"),
           ("1l", "1 locus, $n{=}10$"), ("1l-n20", "1 locus, $n{=}20$")]
D = list(range(1, 11))
SEEDS = [67, 68, 69]
NETS = [("mse", "MSE"), ("pginn", "PGINN"), ("exact", "Exact log-likelihood ratio"), ("wmse", "Weighted MSE"),
        ("mse-featages", "MSE, +ages"), ("pginn-featages", "PGINN, +ages"),
        ("mse-logages", "MSE, ages only"), ("pginn-logages", "PGINN, ages only"),
        ("mse-featlog", "MSE, log-scaled"), ("pginn-featlog", "PGINN, log-scaled"),
        ("fisher-features", "Fisher penalty"), ("fisher-featlog", "Fisher penalty, log-scaled"),
        ("process-features", "Process divergence, pred.$\\|$true"), ("process-featlog", "Process divergence, pred.$\\|$true, log-scaled"),
        ("processfwd-features", "Process divergence, true$\\|$pred."),
        ("processfwd-featlog", "Process divergence, true$\\|$pred., log-scaled"),
        ("exact-featlog", "Exact log-likelihood ratio, log-scaled"), ("wmse-featlog", "Weighted MSE, log-scaled")]
# network input file per method (feature_ages.py); the rest use the KC features
NET_INPUT = {"mse-featages": "featages", "pginn-featages": "featages",
             "mse-logages": "logages", "pginn-logages": "logages",
             "mse-featlog": "featlog", "pginn-featlog": "featlog",
             "fisher-features": "features", "fisher-featlog": "featlog",
             "process-features": "features", "process-featlog": "featlog",
             "processfwd-features": "features", "processfwd-featlog": "featlog",
             "exact-featlog": "featlog", "wmse-featlog": "featlog"}
# reference for the "vs" column: the MSE network with the same input
REFERENCE = {"pginn-featages": "mse-featages", "pginn-logages": "mse-logages", "pginn-featlog": "mse-featlog",
             "fisher-featlog": "mse-featlog", "process-featlog": "mse-featlog",
             "processfwd-featlog": "mse-featlog", "exact-featlog": "mse-featlog", "wmse-featlog": "mse-featlog",
             "mle": "mse-featlog"}
# main-text extracts: (method, group label); the log-scaled KC features are the main input
MAIN_REC = [("mse-featlog", "log-scaled"), ("pginn-featlog", "log-scaled"), ("process-featlog", "log-scaled"),
            ("processfwd-featlog", "log-scaled"),
            ("mse", "linear"), ("pginn", "linear"), ("exact", "linear"), ("wmse", "linear"), ("mle", "")]
MAIN_DET = ["mse-featlog", "pginn-featlog", "mle", "star"]
# version 3 of the manuscript: log-scaled KC features only
MAIN_REC_V3 = ["mse-featlog", "pginn-featlog", "exact-featlog", "process-featlog", "processfwd-featlog",
               "wmse-featlog", "mle"]
V3_LABELS = {"mse-featlog": "MSE", "pginn-featlog": "Local rate divergence (PGINN)", "exact-featlog": "Exact log-likelihood ratio",
             "process-featlog": "Process divergence, pred.$\\|$true", "processfwd-featlog": "Process divergence, true$\\|$pred.",
             "wmse-featlog": "Weighted MSE", "mle": "MLE"}
FPR = 0.05
S_BINS = [1, 10, 50, 200, 1000.0001]
THETA_BINS = np.geomspace(0.001, 0.05, 5)
FIG = Path(__file__).resolve().parent.parent / "figures"
CACHE = Path("data/nested/mle-cache")


def nested_dir(r, d):
    return f"data/nested/{r}-d{d}"


def net_predict(r, d, net, seed, which):
    D_ = nested_dir(r, d)
    return to_natural(load_and_predict(f"{D_}/model/{net}-{seed}.pt", f"{D_}/model/{net}-{seed}.npz",
                                       f"{D_}/{which}-{NET_INPUT.get(net, 'features')}.npy"))


def mle_predict(r, d, which):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{r}-d{d}-{which}.npy"
    if not f.exists():
        np.save(f, mle_baseline.fit(np.load(f"{nested_dir(r, d)}/{which}-locusages.npy")))
    return np.load(f)


def star(r, d, which):
    a = np.load(f"{nested_dir(r, d)}/{which}-ages.npy")
    return -(a[:, 0] - a[:, 1]) / a[:, 0]


def metrics(y, p):
    _, rt, r2t = r2_bias_rmse(y[:, 0], p[:, 0])
    bs, rs, r2s = r2_bias_rmse(log1p_s(y[:, 1]), log1p_s(p[:, 1]))
    return {"theta_r2": r2t, "theta_rmse": rt, "s_r2": r2s, "s_bias": bs, "s_rmse": rs,
            "s_rmse_nat": float(np.sqrt(np.mean((p[:, 1] - y[:, 1]) ** 2))),
            "s_mae_nat": float(np.median(np.abs(p[:, 1] - y[:, 1])))}


def ci(x):
    x = np.asarray(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def recovery(r):
    """per data set: seed-mean metrics per method; also seed SDs."""
    out = {m: [] for m, _ in NETS + [("mle", "MLE")]}
    seed_sd = {m: [] for m, _ in NETS}
    for d in D:
        y = to_natural(np.load(f"{nested_dir(r, d)}/predict-sweep-params.npy"))
        for net, _ in NETS:
            ms = [metrics(y, net_predict(r, d, net, s, "predict-sweep")) for s in SEEDS]
            out[net].append({k: np.mean([m[k] for m in ms]) for k in ms[0]})
            seed_sd[net].append(np.std([m["s_r2"] for m in ms], ddof=1))
        out["mle"].append(metrics(y, mle_predict(r, d, "predict-sweep")))
    return out, seed_sd


def detection(r):
    """per data set: AUC per statistic; realized FPR and sensitivity per bin
    with the threshold from the neutral set of data set d+1."""
    scores = {}  # (stat, d, which) -> score array
    stats_ = [n for n, _ in NETS] + ["mle", "star"]
    for d in D:
        for which in ["predict-sweep", "predict-neutral"]:
            for net, _ in NETS:
                scores[(net, d, which)] = np.mean([net_predict(r, d, net, s, which)[:, 1] for s in SEEDS], axis=0)
            scores[("mle", d, which)] = mle_predict(r, d, which)[:, 1]
            scores[("star", d, which)] = star(r, d, which)
    res = {st: {"auc": [], "fpr": [], "sens": []} for st in stats_}
    for i, d in enumerate(D):
        dcal = D[(i + 1) % len(D)]
        s_true = np.load(f"{nested_dir(r, d)}/predict-sweep-params.npy")[:, 1]
        for st in stats_:
            sel, neu = scores[(st, d, "predict-sweep")], scores[(st, d, "predict-neutral")]
            labels = np.r_[np.ones(len(sel)), np.zeros(len(neu))]
            res[st]["auc"].append(roc_auc_score(labels, np.r_[sel, neu]))
            thr = np.quantile(scores[(st, dcal, "predict-neutral")], 1 - FPR)
            res[st]["fpr"].append(np.mean(neu > thr))
            res[st]["sens"].append([np.mean(sel[(s_true >= lo) & (s_true < hi)] > thr)
                                    for lo, hi in zip(S_BINS[:-1], S_BINS[1:])])
    return res


def binned(r):
    rows = []
    for net, label in NETS + [("mle", "MLE")]:
        ys, ps = [], []
        for d in D:
            for which in ["predict-sweep", "predict-neutral"]:
                y = to_natural(np.load(f"{nested_dir(r, d)}/{which}-params.npy"))
                preds = ([net_predict(r, d, net, s, which) for s in SEEDS] if net != "mle"
                         else [mle_predict(r, d, which)])
                for p in preds:
                    ys.append(y)
                    ps.append(p)
        y, p = np.vstack(ys), np.vstack(ps)
        cells = []
        s_edges = [(0, 0)] + list(zip(S_BINS[:-1], S_BINS[1:]))
        for lo, hi in s_edges:
            m = (y[:, 1] == 0) if hi == 0 else (y[:, 1] >= lo) & (y[:, 1] < hi)
            e = log1p_s(p[m, 1]) - log1p_s(y[m, 1])
            cells.append((e.mean(), np.sqrt(np.mean(e ** 2))))
        t_cells = []
        for lo, hi in zip(THETA_BINS[:-1], THETA_BINS[1:]):
            m = (y[:, 0] >= lo) & (y[:, 0] < hi * 1.0000001)
            e = (p[m, 0] - y[m, 0]) / y[m, 0]
            t_cells.append((e.mean(), np.sqrt(np.mean(e ** 2))))
        clip = {"s=0": np.mean(p[:, 1] <= 0), "s=1000": np.mean(p[:, 1] >= 1000),
                "theta=0.001": np.mean(p[:, 0] <= 0.001 + 1e-12), "theta=0.05": np.mean(p[:, 0] >= 0.05 - 1e-12)}
        rows.append((label, cells, t_cells, clip))
    return rows


def main():
    rec_tex = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
               "Regime & Method & $\\theta$ $R^2$ & $\\theta$ error reduction, \\% [95\\% CI] & $\\log(1{+}s)$ $R^2$ & "
               "$\\log(1{+}s)$ error reduction, \\% [95\\% CI] & better & $s$ med.\\ abs.\\ err. \\\\",
               "\\midrule"]
    det_tex = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
               "Regime & Statistic & AUC [95\\% CI] & realized FPR & \\multicolumn{4}{c}{sensitivity, true $s$ in} \\\\",
               " & & & & $[1,10)$ & $[10,50)$ & $[50,200)$ & $[200,1000]$ \\\\", "\\midrule"]
    bin_tex = ["\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
               "Regime & Method & \\multicolumn{5}{c}{$\\log(1{+}s)$ bias/RMSE, true $s$} & \\multicolumn{4}{c}{$\\theta$ relative bias/RMSE, true $\\theta$} \\\\",
               " & & $0$ & $[1,10)$ & $[10,50)$ & $[50,200)$ & $[200,1000]$ & " +
               " & ".join(f"$[{lo:.3g},{hi:.3g})$" for lo, hi in zip(THETA_BINS[:-1], THETA_BINS[1:])) + " \\\\",
               "\\midrule"]
    labels = dict(NETS + [("mle", "MLE"), ("star", "Star ratio")])
    main_rec, main_det = {}, {}
    for r, rlabel in REGIMES:
        out, seed_sd = recovery(r)
        print(f"== {r}: recovery over {len(D)} independent data sets (network metrics: mean of {len(SEEDS)} seeds)")
        for m in [n for n, _ in NETS] + ["mle"]:
            t = ci([x["theta_r2"] for x in out[m]])
            s = ci([x["s_r2"] for x in out[m]])
            line = f"  {labels[m]:13s} theta R2 {t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]  log(1+s) R2 {s[0]:.3f} [{s[1]:.3f},{s[2]:.3f}]"
            line += f"  s median|err| {np.mean([x['s_mae_nat'] for x in out[m]]):.1f}  s RMSE {np.mean([x['s_rmse_nat'] for x in out[m]]):.0f}"
            ref = REFERENCE.get(m, "mse")
            if m not in ("mse", "mse-featages", "mse-logages", "mse-featlog"):
                diff = np.array([a["s_r2"] - b["s_r2"] for a, b in zip(out[m], out[ref])])
                dt = np.array([a["theta_r2"] - b["theta_r2"] for a, b in zip(out[m], out[ref])])
                dc, dtc = ci(diff), ci(dt)
                line += (f"  vs {ref}: dS {dc[0]:+.3f} [{dc[1]:+.3f},{dc[2]:+.3f}] better {np.sum(diff > 0)}/{len(D)}"
                         f"  dTheta {dtc[0]:+.3f} [{dtc[1]:+.3f},{dtc[2]:+.3f}] better {np.sum(dt > 0)}/{len(D)}")
                cell_d = f"${dc[0]:+.3f}$ [${dc[1]:+.3f}$, ${dc[2]:+.3f}$]"
                cell_b = f"{np.sum(diff > 0)}/{len(D)}"
                cell_dt = f"${dtc[0]:+.3f}$ [${dtc[1]:+.3f}$, ${dtc[2]:+.3f}$]"
                # error reduction: share of the reference's unexplained variance (1 - R^2, i.e. its
                # squared error on the same prediction set) that the method removes, in %
                red = lambda k: ci([100 * (1 - (1 - a[k]) / (1 - b[k])) for a, b in zip(out[m], out[ref])])
                rs, rt = red("s_r2"), red("theta_r2")
                line += f"  error reduction S {rs[0]:+.1f}% [{rs[1]:+.1f},{rs[2]:+.1f}]  Theta {rt[0]:+.1f}% [{rt[1]:+.1f},{rt[2]:+.1f}]"
                pct = lambda x: f"${x:+.0f}$" if abs(x) >= 0.5 else f"${x:+.1f}$"  # no -0 or +0
                cell_rs = f"{pct(rs[0])} [{pct(rs[1])}, {pct(rs[2])}]"
                cell_rt = f"{pct(rt[0])} [{pct(rt[1])}, {pct(rt[2])}]"
            else:
                cell_d, cell_b, cell_dt, cell_rs, cell_rt = "--", "--", "--", "--", "--"
            main_rec[(r, m)] = (t[0], cell_dt, s[0], cell_d, cell_b, np.mean([x['s_mae_nat'] for x in out[m]]),
                                cell_rt, cell_rs)
            if m in seed_sd:
                line += f"  seed SD of s R2 {np.mean(seed_sd[m]):.3f}, between-data-set SD {np.std([x['s_r2'] for x in out[m]], ddof=1):.3f}"
            print(line)
            rec_tex.append(f"{rlabel if m == 'mse' else ''} & {labels[m]} & {t[0]:.3f} & {cell_rt} & {s[0]:.3f} & {cell_rs} & {cell_b} & "
                           f"{np.mean([x['s_mae_nat'] for x in out[m]]):.1f} \\\\")
        rec_tex.append("\\midrule")

        det = detection(r)
        print(f"== {r}: detection")
        for st in [n for n, _ in NETS] + ["mle", "star"]:
            a = ci(det[st]["auc"])
            sens = np.mean(det[st]["sens"], axis=0)
            fpr = np.array(det[st]["fpr"])
            print(f"  {labels[st]:13s} AUC {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]  realized FPR {fpr.mean():.3f} "
                  f"(range {fpr.min():.3f}-{fpr.max():.3f})  sensitivity " + " ".join(f"{x:.2f}" for x in sens))
            det_tex.append(f"{rlabel if st == 'mse' else ''} & {labels[st]} & {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] & "
                           f"{fpr.mean():.3f} & " + " & ".join(f"{x:.2f}" for x in sens) + " \\\\")
            main_det[(r, st)] = f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] & {fpr.mean():.3f} & " + " & ".join(f"{x:.2f}" for x in sens)
        det_tex.append("\\midrule")

        print(f"== {r}: errors by range (log(1+s) bias/RMSE; theta relative bias/RMSE; fraction on prior bounds)")
        for label, cells, t_cells, clip in binned(r):
            print(f"  {label:13s} s: " + "  ".join(f"{b:+.2f}/{e:.2f}" for b, e in cells)
                  + "   theta: " + "  ".join(f"{b:+.2f}/{e:.2f}" for b, e in t_cells)
                  + "   bounds: " + " ".join(f"{k} {v:.3f}" for k, v in clip.items()))
            bin_tex.append(f"{rlabel if label == 'MSE' else ''} & {label} & "
                           + " & ".join(f"${b:+.2f}$/{e:.2f}" for b, e in cells) + " & "
                           + " & ".join(f"${b:+.2f}$/{e:.2f}" for b, e in t_cells) + " \\\\")
        bin_tex.append("\\midrule")
    group = {"log-scaled": "PGINN family, log-scaled KC features", "linear": "KC features (linear ages)", "": "No network"}
    mrec = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
            "Regime & Method & $\\theta$ $R^2$ & $\\Delta\\theta$ $R^2$ [95\\% CI] & $\\log(1{+}s)$ $R^2$ & "
            "$\\Delta\\log(1{+}s)$ $R^2$ [95\\% CI] & better & $s$ med.\\ abs.\\ err. \\\\", "\\midrule"]
    mdet = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
            "Regime & Statistic & AUC [95\\% CI] & realized FPR & \\multicolumn{4}{c}{sensitivity, true $s$ in} \\\\",
            " & & & & $[1,10)$ & $[10,50)$ & $[50,200)$ & $[200,1000]$ \\\\", "\\midrule"]
    for r, rlabel in REGIMES:
        prev = None
        for i, (m, g) in enumerate(MAIN_REC):
            if g != prev:
                mrec.append(f"{rlabel if i == 0 else ''} & \\emph{{{group[g]}}} & & & & & & \\\\")
                prev = g
            tr, dt, sr, ds, b, mae, _, _ = main_rec[(r, m)]
            mrec.append(f" & {labels[m]} & {tr:.3f} & {dt} & {sr:.3f} & {ds} & {b} & {mae:.1f} \\\\")
        mrec.append("\\midrule")
        for i, st in enumerate(MAIN_DET):
            mdet.append(f"{rlabel if i == 0 else ''} & {labels[st]} & {main_det[(r, st)]} \\\\")
        mdet.append("\\midrule")
    mrec3 = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Regime & Method & $\\theta$ $R^2$ & $\\theta$ error reduction, \\% [95\\% CI] & $\\log(1{+}s)$ $R^2$ & "
             "$\\log(1{+}s)$ error reduction, \\% [95\\% CI] & better & $s$ med.\\ abs.\\ err. \\\\", "\\midrule"]
    for r, rlabel in REGIMES:
        for i, m in enumerate(MAIN_REC_V3):
            tr, _, sr, _, b, mae, rt, rs = main_rec[(r, m)]
            mrec3.append(f"{rlabel if i == 0 else ''} & {V3_LABELS[m]} & {tr:.3f} & {rt} & {sr:.3f} & {rs} & {b} & {mae:.1f} \\\\")
        mrec3.append("\\midrule")
    for tex, name in [(rec_tex, "tab_nested_recovery.tex"), (det_tex, "tab_nested_detection.tex"),
                      (bin_tex, "tab_binned_errors.tex"), (mrec, "tab_nested_main.tex"),
                      (mdet, "tab_nested_detection_main.tex"), (mrec3, "tab_nested_main_v3.tex")]:
        tex[-1] = "\\bottomrule"
        tex.append("\\end{tabular}")
        (FIG / name).write_text("\n".join(tex) + "\n")
        print("wrote", FIG / name)


if __name__ == "__main__":
    main()
