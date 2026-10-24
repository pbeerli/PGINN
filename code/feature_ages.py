#!/usr/bin/env python3
"""Network inputs that contain the event ages directly: for each prefix,
write <prefix>-featages.npy (the Kendall-Colijn feature vector with the log
of the example's mean sorted internal-node ages appended) and
<prefix>-logages.npy (the log ages alone), and <prefix>-featlog.npy (the
Kendall-Colijn feature vector with its age-valued blocks on a log scale:
for P = n(n-1)/2 + n entries per block, blocks 0 (branch-length depths,
'm' in the code) and 2 (pair MRCA ages) and the age column of the final
block of node ages (max(20, n) x 3) become log(x + 1e-5); block 1, the topological
counts ('M' in the code), stays linear). Same example order as
prepare.py's -features.npy and -ages.npy."""
import argparse

import numpy as np


def layout(dim):
    """n, entries per pair block, rows of the node-age block, from the width
    of a one-population feature vector (prepare.py)."""
    for n in range(2, 2000):
        P, rows = n * (n - 1) // 2 + n, max(20, n)
        if 3 * P + 3 * rows == dim:
            return n, P, rows
    raise ValueError(f"no one-population feature layout has width {dim}")


def featlog(x):
    """Log-scale the age-valued blocks of KC feature rows x (see the module
    docstring); the topological-count block stays linear."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    _, P, rows = layout(x.shape[1])
    fl = x.copy()
    for b in (0, 2):
        fl[:, b * P:(b + 1) * P] = np.log(x[:, b * P:(b + 1) * P] + 1e-5)
    tail = fl[:, 3 * P:].reshape(len(x), rows, 3)
    tail[:, :, 0] = np.log(tail[:, :, 0] + 1e-5)
    fl[:, 3 * P:] = tail.reshape(len(x), 3 * rows)
    return fl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefixes", nargs="+")
    args = ap.parse_args()
    for p in args.prefixes:
        x = np.load(f"{p}-features.npy")
        la = np.log(np.load(f"{p}-ages.npy"))
        np.save(f"{p}-featages.npy", np.hstack([x, la]))
        np.save(f"{p}-logages.npy", la)
        assert layout(x.shape[1])[0] == la.shape[1] + 1, p
        fl = featlog(x)
        np.save(f"{p}-featlog.npy", fl)
        print(f"{p}: featages {x.shape[1]}+{la.shape[1]}, logages {la.shape[1]}")


if __name__ == "__main__":
    main()
