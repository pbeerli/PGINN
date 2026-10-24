#!/usr/bin/env python3
"""
Neutral data sets shaped like the DGRP Adh alignment, for calibrating the
whole Adh pipeline (four-gamete segmentation -> migrate-n -> network).

msprime simulates the 3,351 bp region for the 176 lines as haploid
sequences (ploidy 1, so the per-site theta = 2*N*mu is the expected
pairwise difference per site, migrate's Theta scale) under a neutral,
constant-size coalescent with recombination, and JC69 mutations on a
random ancestral sequence with the real base composition. Each simulated
sequence then gets the missing-data pattern (N positions) of one real
line, and the real line names, so make_migrate_windows.py and
four_gamete_test.py treat the file exactly like the real one.

--calibrate prints the mean number of biallelic segregating sites and
the Hudson-Kaplan R_m over replicates for a grid of (theta, rho) per
site, used once to choose values that match the real data (108 and 14).
"""
import argparse
from pathlib import Path

import msprime
import numpy as np

import four_gamete_test as fgt

HERE = Path(__file__).resolve().parent
DEFAULT_FASTA = HERE / "inputs" / "adh" / "dgrp_adh_205lines.fasta"
N_E = 10_000  # only scales time; theta and rho per site are what matter


def real_template(path):
    names, seqs = fgt.read_fasta(path)
    keep = [(n, s) for n, s in zip(names, seqs) if s.count("N") < len(s)]
    names = [n for n, _ in keep]
    masks = np.array([[c == "N" for c in s] for _, s in keep])
    allseq = "".join(s for _, s in keep).replace("N", "")
    freqs = np.array([allseq.count(b) for b in "ACGT"], dtype=float)
    return names, masks, freqs / freqs.sum()


def simulate(theta_site, rho_site, seed, n, L, freqs, masks):
    rng = np.random.default_rng(seed)
    ts = msprime.sim_ancestry(samples=n, ploidy=1, population_size=N_E, sequence_length=L,
                              recombination_rate=rho_site / (2 * N_E), random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=theta_site / (2 * N_E), model=msprime.JC69(),
                               random_seed=seed + 1, discrete_genome=True)
    ancestral = rng.choice(list("ACGT"), size=L, p=freqs)
    seqs = np.tile(ancestral, (n, 1))
    for var in ts.variants():
        # mutated sites keep msprime's own JC69 ancestral and derived bases
        alleles = np.array(var.alleles)
        seqs[:, int(var.site.position)] = alleles[var.genotypes]
    seqs[masks] = "N"
    return ["".join(row) for row in seqs]


def write_fasta(path, names, seqs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(f">{n}\n{s}\n" for n, s in zip(names, seqs)))


def summary(names, seqs):
    _, positions, columns = fgt.biallelic_segregating_sites(names, seqs)
    incompatible = [(a, b) for a in range(len(positions)) for b in range(a + 1, len(positions))
                    if fgt.four_gamete_incompatible(columns[a], columns[b])]
    rm, _ = fgt.minimum_recombination_events(incompatible)
    return len(positions), rm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", default=str(DEFAULT_FASTA), help="real alignment (names, length, missing data)")
    ap.add_argument("--theta-site", type=float, default=None)
    ap.add_argument("--rho-site", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="output FASTA")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    names, masks, freqs = real_template(args.fasta)
    n, L = masks.shape
    if args.calibrate:
        for theta in [0.005, 0.006, 0.007]:
            for rho in [0.005, 0.01, 0.02, 0.04]:
                res = np.array([summary(names, simulate(theta, rho, 1000 + 7 * r, n, L, freqs, masks))
                                for r in range(args.reps)])
                print(f"theta/site {theta:.4f} rho/site {rho:.4f}: biallelic S {res[:, 0].mean():6.1f} "
                      f"(sd {res[:, 0].std():5.1f})  R_m {res[:, 1].mean():5.1f} (sd {res[:, 1].std():4.1f})",
                      flush=True)
        return
    seqs = simulate(args.theta_site, args.rho_site, args.seed, n, L, freqs, masks)
    write_fasta(args.out, names, seqs)
    S, rm = summary(names, seqs)
    print(f"seed {args.seed}: theta/site {args.theta_site} rho/site {args.rho_site}: "
          f"{S} biallelic segregating sites, R_m {rm}; wrote {args.out}")


if __name__ == "__main__":
    main()
