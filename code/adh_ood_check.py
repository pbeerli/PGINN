#!/usr/bin/env python3
"""Out-of-distribution check for the Adh scan: for each segment, compare the
coalescent-event ages of --per-segment randomly chosen posterior genealogies
(migrate trees, native Theta units, as scored) with --n-ref neutral simtree
genealogies simulated at the same sample size and at that segment's own
migrate Theta. The ages (n-1 values, sorted) are non-redundant, unlike the
all-pairs feature vector, and are what the PGINN loss is built on. Reports per
segment how many genealogies have any age with |z| > 5 and the largest |z|."""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import tree as T
from real_adh_migrate_scan import (parse_trees, parse_migrate_thetas, prune_and_relabel,
                                   DEFAULT_BESTTREE, DEFAULT_OUTFILE)
from msprime_scan import collect_internal_ages

HERE = Path(__file__).resolve().parent


def sorted_ages(newick):
    t = T.Tree()
    T.Tree.i = 0
    t.myread(newick, t.root)
    max_depth = T.set_age1(t.root, 0.0)
    T.reset_age1(t.root, max_depth)
    ages = []
    collect_internal_ages(t.root, ages)
    return np.sort(ages)[::-1]


def neutral_reference(theta, n, n_ref, seed):
    tmp = Path(tempfile.mkdtemp(dir=HERE / "data"))
    out = tmp / "ref.tre"
    subprocess.run(["python", str(HERE / "simtree.py"), "-t", str(theta), "-i", str(n), "-a", "1.0",
                    "-sel", "0", "-l", str(n_ref), "-s", "1000", "-f", str(out), "--seed", str(seed)],
                   check=True, capture_output=True)
    nwks = [l.strip() for l in out.read_text().splitlines() if l.strip().endswith(";")]
    out.unlink()
    tmp.rmdir()
    return np.array([sorted_ages(nwk) for nwk in nwks])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trees", default=str(DEFAULT_BESTTREE))
    ap.add_argument("--outfile", default=str(DEFAULT_OUTFILE))
    ap.add_argument("--per-segment", type=int, default=20)
    ap.add_argument("--n-ref", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    trees = parse_trees(args.trees)
    thetas = parse_migrate_thetas(args.outfile)
    rng = np.random.default_rng(args.seed)
    print(f"{'seg':>3} {'Theta':>7} {'trees |z|>5':>12} {'max|z|':>7}")
    for locus in sorted(trees):
        nwks = trees[locus]
        pick = rng.choice(len(nwks), size=min(args.per_segment, len(nwks)), replace=False)
        names = sorted({tok for tok in __import__("re").findall(r"[(,]\s*([^():,\s]+)\s*:", nwks[0])})
        index = {name: i for i, name in enumerate(names)}
        ages = np.array([sorted_ages(prune_and_relabel(nwks[i], names, index)) for i in pick])
        ref = neutral_reference(thetas[locus], len(names), args.n_ref, args.seed + locus)
        z = (ages - ref.mean(axis=0)) / ref.std(axis=0, ddof=1)
        flagged = int((np.abs(z).max(axis=1) > 5).sum())
        print(f"{locus:3d} {thetas[locus]:7.4f} {flagged:>7d}/{len(pick):<4d} {np.abs(z).max():7.2f}")


if __name__ == "__main__":
    main()
