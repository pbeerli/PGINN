#!/usr/bin/env python3
"""
Score migrate-n genealogies for the DGRP Adh locus with the trained
(theta, s) PGINN: a single best tree per locus (print-tree=BEST), or a
posterior sample (print-tree=LASTCHAIN), summarized per locus as median
and 95% interval over trees. Loci are the 15 variable-length
segments cut at the four-gamete breakpoints (Hudson-Kaplan Rm=14; no
four-gamete violation inside any segment; make_migrate_windows.py
--four-gamete), no pooling.

Compatibility problem this solves: migrate's best trees have all 176
DGRP lines as tips, but Mm.py's MetricsVectors requires tip names of
the form "<int>_<index>" (it does int(name.split('_')[0])) to track which
simulated deme a tip came from -- real "RAL-105"-style names don't parse.
So each locus's tree is pruned (dendropy, branch-length preserving) down
to --n individuals and those tips relabelled to "0_i", exactly the
convention real_lct_scan.py/msprime_scan.py use; --n 176 keeps every
tip (a no-op prune) and scores against the n=176-trained model instead
of subsampling down to a model trained at n=10/20.

Branch-length scale: migrate's genealogies are in mutation-scaled time,
the same units simtree.py uses for training (a neutral tree's height is
about Theta), so by default they are scored as they are, with no rescale.
The training prior theta ~ log-uniform(0.001, 0.05) covers the Adh
segments' Theta. --theta-target restores the older behaviour: one global
constant for all trees, so their mean height is 0.9*theta_target.
"""
import argparse
import json
import re
from pathlib import Path

import dendropy
import numpy as np
from scipy.stats import spearmanr

import tree as T
import Mm as mm
from prepare import AveragedThetaDataset
from train_pginn import predict
from msprime_scan import collect_internal_ages

HERE = Path(__file__).resolve().parent
# migrate-15loci-fixed/posterior: migrate 6.1.11 (large-n Theta bias and
# print-tree=LASTCHAIN fixed), 10 replicates, every 100th sampled
# genealogy -> 1000 posterior trees per locus. A single best tree per
# locus proved unstable between runs (see CHANGELOG.md), so each locus is
# scored on its posterior trees and summarized by median and 95% interval.
# Theta comes from the same run's outfile; it's an independent comparison
# column only, not a model input.
DEFAULT_BESTTREE = HERE / "data" / "real-adh-cache" / "migrate-15loci-fixed" / "posterior" / "trees.tre"
DEFAULT_OUTFILE = HERE / "data" / "real-adh-cache" / "migrate-15loci-fixed" / "posterior" / "outfile"
ADH_START = 14_615_552

# Locus -> (chr2L start, chr2L end), the 15 four-gamete segments (see
# four_gamete_test.compatible_segments / make_migrate_windows.py); no pooling.
LOCUS_REGIONS = {
    1:  (14615552, 14615609),
    2:  (14615610, 14615789),
    3:  (14615790, 14615945),
    4:  (14615946, 14616039),
    5:  (14616040, 14616185),
    6:  (14616186, 14616416),
    7:  (14616417, 14616637),
    8:  (14616638, 14616901),
    9:  (14616902, 14617137),
    10: (14617138, 14617371),
    11: (14617372, 14617804),
    12: (14617805, 14618637),
    13: (14618638, 14618793),
    14: (14618794, 14618851),
    15: (14618852, 14618902),
}


def parse_trees(path):
    """Return {locus_number: [newick, ...]} from migrate's print-tree output:
    one tree per locus for BEST, many for LASTCHAIN/ALL (posterior sample)."""
    text = Path(path).read_text()
    trees = {}
    for m in re.finditer(r"\[& Locus (\d+), (?:best )?ln\(L\) = ([\-0-9.]+).*?\]\s*\n(.*?;)", text, re.S):
        trees.setdefault(int(m.group(1)), []).append(m.group(3).strip())
    return trees


def parse_besttree(path):
    """Return {locus_number: newick_string} from migrate's print-tree=BEST output."""
    return {locus: nwks[-1] for locus, nwks in parse_trees(path).items()}


def parse_migrate_thetas(path):
    """Pull the per-locus Theta_1 mode from migrate's outfile results table."""
    text = Path(path).read_text()
    thetas = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 9 and parts[1] == "Theta_1" and parts[0].isdigit():
            # columns: Locus Parameter HPD95lo HPD50lo mode HPD50hi HPD95hi median mean
            thetas[int(parts[0])] = float(parts[4])
    return thetas


def prune_and_relabel(newick, keep_names, name_to_index):
    t = dendropy.Tree.get(data=newick, schema="newick")
    t.retain_taxa_with_labels(keep_names)
    for leaf in t.leaf_node_iter():
        leaf.taxon.label = f"0_{name_to_index[leaf.taxon.label]}"
    return t.as_string(schema="newick", suppress_rooting=True,
                        suppress_internal_node_labels=True,
                        unquoted_underscores=True).strip()


def raw_tree_height(newick):
    t = T.Tree()
    T.Tree.i = 0
    t.myread(newick, t.root)
    return T.set_age1(t.root, 0.0)


def newick_to_entry(newick, rescale, locus_index):
    newick = re.sub(r":([0-9.eE+-]+)", lambda m: f":{float(m.group(1)) * rescale:.10g}", newick)
    t = T.Tree()
    T.Tree.i = 0
    t.myread(newick, t.root)
    max_depth = T.set_age1(t.root, 0.0)
    T.reset_age1(t.root, max_depth)
    internal_ages = []
    collect_internal_ages(t.root, internal_ages)
    internal_ages = sorted(internal_ages, reverse=True)
    mmt = mm.mmtree(t.root)
    tips, tipslength, pairlist = [], [], []
    Mv, mv, age, inter = mmt.MetricsVectors(newick, tips, tipslength, pairlist)
    return {
        "ID": locus_index,
        "params": [[0.02], [1.0], [locus_index]],  # theta placeholder, alpha=1.0, group key
        "M": Mv, "m": mv, "age": age,
        "inter": [list(p) for p in inter],
        "ages": internal_ages,
    }


S_ZERO = 1.0  # a posterior tree counts as "s_hat ~ 0" when PGINN s_hat < S_ZERO


def write_tex(rows, path):
    """tab:real-data body: medians over posterior trees, 95% interval for s."""
    def s_cell(r, tag):
        return f"{r[tag + '_s']:.0f} ({r[tag + '_s_q025']:.0f}--{r[tag + '_s_q975']:.0f})"
    tex = ["\\begin{tabular}{rrrrrrrr}", "\\toprule",
           "Seg. & bp & migrate $\\hat\\Theta$ & MSE $\\hat\\theta$ & MSE $\\hat s$ & \\pginn $\\hat\\theta$ & \\pginn $\\hat s$ & \\% $\\hat s{\\approx}0$ \\\\",
           "\\midrule"]
    for r in rows:
        tex.append(f"{r['locus']:2d} & {r['bp']:3d} & {r['migrate_theta_mode']:.4f} & {r['MSE_theta']:.4f} & {s_cell(r, 'MSE')} & "
                   f"{r['PGINN_theta']:.4f} & {s_cell(r, 'PGINN')} & {r['PGINN_pct_s0']:.0f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(tex) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--besttree", default=str(DEFAULT_BESTTREE))
    ap.add_argument("--outfile", default=str(DEFAULT_OUTFILE))
    ap.add_argument("--n", type=int, default=176, help="subsample size; a model trained at this n must exist (model/model-*-1l-n{N}*.pt, no -n suffix for n=10); 176 (default) = full DGRP sample, no pruning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--theta-target", type=float, default=None,
                    help="rescale all trees by one constant so their mean height is 0.9*theta_target; default: no rescale (migrate trees are already in Theta units)")
    ap.add_argument("--model-mse", default=None)
    ap.add_argument("--scaler-mse", default=None)
    ap.add_argument("--model-pginn", default=None)
    ap.add_argument("--scaler-pginn", default=None)
    ap.add_argument("--json-dir", default="data/real-adh-cache/migrate-scan-15loci")
    ap.add_argument("--out-table", default="../figures/real_adh_migrate_scan.csv")
    ap.add_argument("--max-trees", type=int, default=None, help="posterior tree files: score at most this many trees per locus (random, --seed)")
    ap.add_argument("--out-tex", default="../figures/tab_real_data.tex", help="posterior tree files: write the tab:real-data body here")
    ap.add_argument("--per-tree-out", default=None, help="posterior tree files: also write every tree's prediction to this csv")
    args = ap.parse_args()

    suffix = "" if args.n == 10 else f"-n{args.n}"
    model_mse = args.model_mse or f"model/model-mse-1l{suffix}.pt"
    scaler_mse = args.scaler_mse or f"model/scaler-mse-1l{suffix}.npz"
    model_pginn = args.model_pginn or f"model/model-pginn-1l{suffix}-0.1.pt"
    scaler_pginn = args.scaler_pginn or f"model/scaler-pginn-1l{suffix}-0.1.npz"

    trees = parse_trees(args.besttree)
    migrate_thetas = parse_migrate_thetas(args.outfile)
    rng_trees = np.random.default_rng(args.seed)
    for locus, nwks in trees.items():
        if args.max_trees and len(nwks) > args.max_trees:
            keep = sorted(rng_trees.choice(len(nwks), size=args.max_trees, replace=False))
            trees[locus] = [nwks[k] for k in keep]
    print(f"parsed {len(trees)} loci from {args.besttree}; trees per locus: {sorted({len(v) for v in trees.values()})}")

    first_newick = trees[sorted(trees)[0]][0]
    first_tree = dendropy.Tree.get(data=first_newick, schema="newick")
    all_names = sorted(leaf.taxon.label for leaf in first_tree.leaf_node_iter())
    rng = np.random.default_rng(args.seed)
    keep_names = sorted(rng.choice(all_names, size=args.n, replace=False).tolist())
    name_to_index = {name: i for i, name in enumerate(keep_names)}
    print(f"subsampled {args.n}/{len(all_names)} individuals (seed {args.seed}): {keep_names}")

    # (locus, tree index) -> pruned newick; group key is the locus for a single
    # best tree (unchanged behaviour) and locus*100000+index for a posterior
    # sample, so AveragedThetaDataset keeps every tree separate.
    single = all(len(v) == 1 for v in trees.values())
    pruned = {}
    for locus, nwks in trees.items():
        for j, nwk in enumerate(nwks):
            key = locus if single else locus * 100000 + j
            pruned[key] = prune_and_relabel(nwk, keep_names, name_to_index)

    keys_sorted = sorted(pruned)
    raw_heights = np.array([raw_tree_height(pruned[k]) for k in keys_sorted])
    mean_height = raw_heights.mean()
    rescale = 1.0 if args.theta_target is None else (0.9 * args.theta_target) / mean_height
    print(f"mean raw tree height across {len(keys_sorted)} trees: {mean_height:.6g}; rescale: {rescale:.6g}")

    entries = [newick_to_entry(pruned[k], rescale, k) for k in keys_sorted]

    json_dir = Path(args.json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    for f in json_dir.glob("*.json"):
        f.unlink()
    with open(json_dir / "scan.json", "w") as f:
        json.dump(entries, f)

    dataset = AveragedThetaDataset(str(json_dir), [0, 2])
    thetas_dummy, features, _ages = dataset.__getitems__()
    features = np.array(features)
    group_key = np.array(thetas_dummy)[:, 1].astype(int)
    order = np.argsort(group_key)
    features = features[order]
    keys_ordered = group_key[order]
    loci_of_row = keys_ordered if single else keys_ordered // 100000

    predictions = {}
    for tag, model_path, scaler_path in [("MSE", model_mse, scaler_mse), ("PGINN", model_pginn, scaler_pginn)]:
        predictions[tag] = predict(HERE / model_path, HERE / scaler_path, features)  # columns: [theta, s]
        predictions[tag][:, 1] = np.maximum(predictions[tag][:, 1], 0.0)  # s < 0 is outside the support

    if args.per_tree_out:
        with open(args.per_tree_out, "w") as f:
            f.write("locus,tree,MSE_theta,MSE_s,PGINN_theta,PGINN_s\n")
            for i, (loc, key) in enumerate(zip(loci_of_row, keys_ordered)):
                f.write(f"{loc},{key % 100000},{predictions['MSE'][i, 0]:.5f},{predictions['MSE'][i, 1]:.5f},"
                        f"{predictions['PGINN'][i, 0]:.5f},{predictions['PGINN'][i, 1]:.5f}\n")
        print(f"wrote {args.per_tree_out}")

    # one row per locus: the value itself for a single best tree; for a
    # posterior sample the median, with a 95% interval in the _q025/_q975 columns
    cols = ["MSE_theta", "MSE_s", "PGINN_theta", "PGINN_s"]
    rows = []
    for loc in sorted(set(loci_of_row.tolist())):
        sel = loci_of_row == loc
        vals = {"MSE_theta": predictions["MSE"][sel, 0], "MSE_s": predictions["MSE"][sel, 1],
                "PGINN_theta": predictions["PGINN"][sel, 0], "PGINN_s": predictions["PGINN"][sel, 1]}
        start, end = LOCUS_REGIONS[loc]
        r = {"locus": loc, "chr2L_start": start, "chr2L_end": end, "bp": end - start + 1,
              "migrate_theta_mode": migrate_thetas.get(loc, float("nan")), "ntrees": int(sel.sum()),
             "PGINN_pct_s0": 100 * float(np.mean(vals["PGINN_s"] < S_ZERO))}
        for c in cols:
            r[c] = float(np.median(vals[c]))
            r[c + "_q025"], r[c + "_q975"] = (float(q) for q in np.quantile(vals[c], [0.025, 0.975]))
        rows.append(r)

    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_table, "w") as f:
        if single:
            f.write("locus,chr2L_start,chr2L_end,bp,migrate_theta_mode,MSE_theta,MSE_s,PGINN_theta,PGINN_s\n")
        else:
            f.write("locus,chr2L_start,chr2L_end,bp,migrate_theta_mode,ntrees,"
                    + ",".join(f"{c},{c}_q025,{c}_q975" for c in cols) + ",PGINN_pct_s0\n")
        for r in rows:
            f.write(f"{r['locus']},{r['chr2L_start']},{r['chr2L_end']},{r['bp']},{r['migrate_theta_mode']:.5f},")
            if single:
                f.write(",".join(f"{r[c]:.5f}" for c in cols) + "\n")
            else:
                f.write(f"{r['ntrees']}," + ",".join(f"{r[c]:.5f},{r[c + '_q025']:.5f},{r[c + '_q975']:.5f}" for c in cols)
                        + f",{r['PGINN_pct_s0']:.1f}\n")
    print(f"wrote {args.out_table}")
    if args.out_tex and not single:
        write_tex(rows, args.out_tex)
    mig = [r["migrate_theta_mode"] for r in rows]
    for c in cols:
        rho, p = spearmanr(mig, [r[c] for r in rows])
        print(f"Spearman rho(migrate Theta, {c} median) = {rho:+.3f} (p={p:.2g}, {len(rows)} segments)")
    for r in rows:
        print(f"  locus {r['locus']} (chr2L:{r['chr2L_start']}-{r['chr2L_end']}, {r['bp']}bp): "
              f"migrate_theta={r['migrate_theta_mode']:.4f}  "
              f"MSE(theta={r['MSE_theta']:.4f}, s={r['MSE_s']:.4f})  "
              f"PGINN(theta={r['PGINN_theta']:.4f}, s={r['PGINN_s']:.4f})")


if __name__ == "__main__":
    main()
