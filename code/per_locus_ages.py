#!/usr/bin/env python3
"""Write <prefix>-locusages.npy, shape (examples, loci, n-1): the sorted
internal-node ages of every locus genealogy of every example, in the same
example order as prepare.py (sorted file names in the simulation folder).
prepare.py's <prefix>-ages.npy is their mean over loci; this is checked.
Used by train_pginn.py --loss exact and by mle_baseline.py."""
import argparse
import json
import os

import numpy as np


def locus_ages(folder):
    rows = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name)) as f:
            rows.append([np.sort(np.asarray(e["ages"], dtype=np.float64))[::-1] for e in json.load(f)])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefixes", nargs="+", help="data/<set>-<regime> (folder and npy prefix)")
    args = ap.parse_args()
    for p in args.prefixes:
        a = locus_ages(p)
        mean_ages = np.load(f"{p}-ages.npy")
        assert np.allclose(a.mean(axis=1), mean_ages, rtol=1e-6, atol=1e-12), f"{p}: order or ages differ from prepare.py"
        np.save(f"{p}-locusages.npy", a)
        print(f"{p}-locusages.npy {a.shape}")


if __name__ == "__main__":
    main()
