#!/usr/bin/env python3
"""
Four-gamete test (Hudson & Kaplan 1985) on the DGRP Adh alignment.

Before treating the whole ~3.3 kb Adh locus as a single non-recombining
unit for one ML tree + PGINN estimate, check whether the data itself
supports that assumption. Kreitman (1983) found two independent
intragenic recombination events among just 11 Adh sequences over 2.7 kb;
Schaeffer & Miller (1992) avoided genealogical methods on Adh entirely
because it is a nuclear gene subject to recombination. With 174 real
DGRP sequences instead of 11, that signal should be at least as strong.

For every pair of biallelic segregating sites, check whether all four
gametes (00, 01, 10, 11) are observed among lines with non-missing calls
at both sites. A pair showing all four is incompatible with any single
tree (under infinite sites) -- a recombination event must have occurred
between them. Hudson & Kaplan's R_m is the minimum number of such events
needed to explain the data: sort incompatible pairs by right endpoint,
greedily take non-overlapping ones.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_FASTA = HERE / "inputs" / "adh" / "dgrp_adh_205lines.fasta"
ADH_START = 14_615_552  # 1-based, chr2L, dm6/BDGP6.54 (FlyBase FBgn0000055)


def read_fasta(path):
    names, seqs = [], []
    name, chunks = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs.append("".join(chunks))
            name = line[1:].strip()
            names.append(name)
            chunks = []
        else:
            chunks.append(line.strip())
    if name is not None:
        seqs.append("".join(chunks))
    return names, seqs


def biallelic_segregating_sites(names, seqs):
    """Return (positions, columns) for sites with exactly 2 non-N alleles,
    dropping lines that are entirely N first."""
    keep = [i for i, s in enumerate(seqs) if s.count("N") < len(s)]
    names = [names[i] for i in keep]
    seqs = [seqs[i] for i in keep]
    L = len(seqs[0])
    positions, columns = [], []
    for pos in range(L):
        col = [s[pos] for s in seqs]
        alleles = set(c for c in col if c != "N")
        if len(alleles) == 2:
            positions.append(pos)
            columns.append(col)
    return names, positions, columns


def four_gamete_incompatible(col_i, col_j):
    gametes = set()
    for a, b in zip(col_i, col_j):
        if a == "N" or b == "N":
            continue
        gametes.add((a, b))
        if len(gametes) == 4:
            return True
    return False


def minimum_recombination_events(incompatible_pairs):
    """Hudson & Kaplan (1985) R_m: greedy interval selection."""
    intervals = sorted(incompatible_pairs, key=lambda p: p[1])
    rm = 0
    last_right = -1
    breakpoints = []
    for i, j in intervals:
        if i >= last_right:  # intervals sharing an endpoint site need separate events
            rm += 1
            last_right = j
            breakpoints.append((i, j))
    return rm, breakpoints


def compatible_segments(positions, incompatible_pairs, seq_len):
    """Split [0, seq_len) into R_m + 1 segments with no incompatible pair
    inside any segment. Each cut is only localized to a gap between two
    segregating sites: the forward greedy gives its latest feasible site
    index, the reverse greedy its earliest; the cut goes at the bp midpoint
    of that allowed range (moving it inside the range only moves invariant
    sites between neighbouring segments). Returns (bounds, cut_ranges),
    bounds as 0-based half-open (start, end), cut_ranges as bp (lo, hi)."""
    n = len(positions)
    _, fwd = minimum_recombination_events(incompatible_pairs)
    latest = [j for _, j in fwd]
    _, rev = minimum_recombination_events([(n - 1 - j, n - 1 - i) for i, j in incompatible_pairs])
    earliest = sorted(n - j for _, j in rev)
    cut_ranges = [(positions[e - 1] + 1, positions[l]) for e, l in zip(earliest, latest)]
    cuts = [(lo + hi) // 2 for lo, hi in cut_ranges]
    edges = [0] + cuts + [seq_len]
    bounds = list(zip(edges[:-1], edges[1:]))
    for start, end in bounds:
        idx = [k for k, p in enumerate(positions) if start <= p < end]
        assert not any((a in idx and b in idx) for a, b in incompatible_pairs), (start, end)
    return bounds, cut_ranges


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", default=str(DEFAULT_FASTA))
    ap.add_argument("--out-prefix", default=str(HERE / "data" / "real-adh-cache" / "four_gamete"))
    args = ap.parse_args()

    names, seqs = read_fasta(args.fasta)
    print(f"read {len(names)} lines, length {len(seqs[0])} bp")

    names, positions, columns = biallelic_segregating_sites(names, seqs)
    n_sites = len(positions)
    print(f"{len(names)} non-fully-missing lines; {n_sites} biallelic segregating sites")

    n_pairs = n_sites * (n_sites - 1) // 2
    incompatible = []
    compat_matrix = np.zeros((n_sites, n_sites), dtype=bool)
    for a in range(n_sites):
        for b in range(a + 1, n_sites):
            if four_gamete_incompatible(columns[a], columns[b]):
                incompatible.append((a, b))
                compat_matrix[a, b] = compat_matrix[b, a] = True

    print(f"{len(incompatible)} / {n_pairs} site pairs incompatible with a single tree "
          f"({100 * len(incompatible) / n_pairs:.1f}%)")

    rm, breakpoints = minimum_recombination_events(incompatible)
    print(f"Hudson-Kaplan minimum number of recombination events, R_m = {rm}")
    if breakpoints:
        print("breakpoint-defining site-index pairs (genomic positions in parens):")
        for i, j in breakpoints:
            gi, gj = ADH_START + positions[i], ADH_START + positions[j]
            print(f"  sites {i},{j}  (chr2L:{gi}-{gj})")

    # Largest run of consecutive segregating sites with no incompatibility
    # against any other site in the run -- a candidate recombination-clean block.
    clean_runs = []
    run_start = 0
    for a in range(n_sites):
        if any(compat_matrix[a, b] for b in range(run_start, a)):
            if a - run_start > 1:
                clean_runs.append((run_start, a - 1))
            run_start = a
    if n_sites - run_start > 1:
        clean_runs.append((run_start, n_sites - 1))
    if clean_runs:
        best = max(clean_runs, key=lambda r: r[1] - r[0])
        gi, gj = ADH_START + positions[best[0]], ADH_START + positions[best[1]]
        print(f"largest internally-consistent run of segregating sites: "
              f"sites {best[0]}-{best[1]} ({best[1]-best[0]+1} sites, chr2L:{gi}-{gj})")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(compat_matrix, cmap="Greys", origin="lower")
    ax.set_xlabel("segregating site index")
    ax.set_ylabel("segregating site index")
    ax.set_title(f"Four-gamete incompatibility, DGRP Adh (n={len(names)} lines)\n"
                 f"R_m = {rm} minimum recombination events")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_matrix.png", dpi=150)
    print(f"wrote {out_prefix}_matrix.png")


if __name__ == "__main__":
    main()
