#!/usr/bin/env python3
"""Weight selection for the two loss variants added in revision, seed 67,
scored on the 100-example validation split (test-*), never on the predict
set, by the rule used for the PGINN weight w (lambda_sweep_val.py): the
value with the best validation log(1+s) R^2 in the most regimes (ties:
higher mean R^2).
  exact: train_pginn.py --loss exact (exact-hazard genealogy likelihood), w
  wmse:  train_pginn.py --pginnratio 0 --s-weight (log(1+s)-weighted MSE)
  fisher, process, process-fwd: train_pginn.py --loss fisher / process /
         process-fwd, w, on the KC features and on the log-scaled features
  exact and wmse are also run on the log-scaled features
Writes figures/tab_variant_sweep.tex; prints the chosen values."""
from collections import Counter
from pathlib import Path

import numpy as np

from compare import load_and_predict, to_natural, r2_bias_rmse, log1p_s

REGIMES = [("10l", "10 loci, $n{=}10$"), ("10l-n20", "10 loci, $n{=}20$"),
           ("1l", "1 locus, $n{=}10$"), ("1l-n20", "1 locus, $n{=}20$")]
VARIANTS = [("exact", "Exact log-likelihood ratio, $w$", ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3"]),
            ("wmse", "Weighted MSE, $\\log(1{+}s)$ weight", ["1.5", "2", "3", "5"]),
            ("fisher", "Fisher penalty, $w$", ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0"]),
            ("fisher-featlog", "Fisher penalty, log-scaled features, $w$", ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0"])]
PW = ["0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2"]
VARIANTS += [("process", "Process divergence, predicted $\\|$ true, $w$", PW),
             ("process-featlog", "Process divergence, predicted $\\|$ true, log-scaled features, $w$", PW),
             ("process-fwd", "Process divergence, true $\\|$ predicted, $w$", PW),
             ("process-fwd-featlog", "Process divergence, true $\\|$ predicted, log-scaled features, $w$", PW)]
VARIANTS += [("exact-featlog", "Exact log-likelihood ratio, log-scaled features, $w$", ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3"]),
             ("wmse-featlog", "Weighted MSE, log-scaled features, $\\log(1{+}s)$ weight", ["1.5", "2", "3", "5"])]
INPUT = {"fisher-featlog": "featlog", "process-featlog": "featlog", "process-fwd-featlog": "featlog",
         "exact-featlog": "featlog", "wmse-featlog": "featlog"}  # network input per variant; default: the KC features
OUT_TEX = Path(__file__).resolve().parent.parent / "figures" / "tab_variant_sweep.tex"


def val_scores(stub, regime, inp="features"):
    y = to_natural(np.load(f"data/test-sweep-{regime}-params.npy"))
    p = to_natural(load_and_predict(f"model/rev/model-{stub}.pt", f"model/rev/scaler-{stub}.npz",
                                    f"data/test-sweep-{regime}-{inp}.npy"))
    _, _, r2_t = r2_bias_rmse(y[:, 0], p[:, 0])
    _, _, r2_s = r2_bias_rmse(log1p_s(y[:, 1]), log1p_s(p[:, 1]))
    return r2_t, r2_s


def main():
    ncol = max(len(v) for _, _, v in VARIANTS)
    tex = ["\\begin{tabular}{ll" + "r" * ncol + "}", "\\toprule",
           f"Variant & Regime & \\multicolumn{{{ncol}}}{{c}}{{validation $\\theta$ $R^2$ / $\\log(1{{+}}s)$ $R^2$}} \\\\"]
    for var, label, values in VARIANTS:
        pad = " & " * (ncol - len(values))
        tex += ["\\midrule", f"{label} & & " + " & ".join(values) + pad + " \\\\", "\\midrule"]
        wins, means = Counter(), {v: [] for v in values}
        for regime, rlabel in REGIMES:
            sc = {v: val_scores(f"{var}-{regime}-{v}", regime, INPUT.get(var, "features")) for v in values}
            best = max(values, key=lambda v: sc[v][1])
            wins[best] += 1
            for v in values:
                means[v].append(sc[v][1])
            cells = [(f"\\textbf{{{sc[v][0]:.3f}/{sc[v][1]:.3f}}}" if v == best else f"{sc[v][0]:.3f}/{sc[v][1]:.3f}")
                     for v in values]
            tex.append(f" & {rlabel} & " + " & ".join(cells) + pad + " \\\\")
            print(f"{var} {regime}: " + "  ".join(f"{v}: {sc[v][0]:.3f}/{sc[v][1]:.3f}" for v in values) + f"  best {best}")
        for ref, rstub in [("MSE", "mse"), ("PGINN w=0.1", "pginn")]:  # same input, for reference
            if var == "fisher-featlog":
                print(f"   reference {ref}, log-scaled features: " + "  ".join(
                    f"{rg}: {val_scores(f'{rstub}-featlog-{rg}', rg, 'featlog')[1]:.3f}" for rg, _ in REGIMES))
        chosen = max(values, key=lambda v: (wins[v], np.mean(means[v])))
        print(f"{var}: chosen {chosen} (best in {wins[chosen]} of {len(REGIMES)} regimes)")
    tex += ["\\bottomrule", "\\end{tabular}"]
    OUT_TEX.write_text("\n".join(tex) + "\n")
    print("wrote", OUT_TEX)


if __name__ == "__main__":
    main()
