#!/usr/bin/env python3
"""
Sequence data along one recombining chromosome for the end-to-end test of
the pipeline (msprime -> migrate-n -> networks): n haploid samples of a
chromosome of length L with a hard sweep at its midpoint (msprime
SweepGenicSelection, completed just before sampling) or a neutral
control, constant size or exponential growth. JC69 mutations on a random
ancestral sequence with equal base frequencies.

Writes the sequences of W windows of `--window-bp` bp, evenly spaced (the
middle window is centred on the sweep site), concatenated into one FASTA
in window order, so that make_migrate_windows.py --window-size
<window-bp> turns each window into one migrate locus. Also writes a CSV
with each window's position, distance to the sweep site, number of
segregating sites, and the true genealogy at the window centre (TMRCA and
star ratio on the per-site branch-length scale).

Time scales: ploidy 1, population size N, so theta per site = 2 N mu as in
adh_neutral_sim.py; the population-scaled selection coefficient of the
networks is s = 2 N s_g (s_g: msprime's per-generation coefficient).
"""
import argparse
import csv
from pathlib import Path

import msprime
import numpy as np


def simulate(args):
    N = args.N
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=N)
    if args.growth > 1:
        t = args.growth_time * N
        dem.populations[0].growth_rate = np.log(args.growth) / t
        dem.add_population_parameters_change(time=t, initial_size=N / args.growth, growth_rate=0, population="A")
    model = None
    if args.s_g > 0:
        model = [msprime.SweepGenicSelection(position=args.length / 2, start_frequency=1.0 / (2 * N),
                                             end_frequency=1.0 - 1.0 / (2 * N), s=args.s_g, dt=1e-6),
                 "hudson"]
    ts = msprime.sim_ancestry(samples={"A": args.n}, demography=dem, ploidy=1, sequence_length=args.length,
                              recombination_rate=args.r, model=model, random_seed=args.seed)
    return msprime.sim_mutations(ts, rate=args.theta_site / (2 * N), model=msprime.JC69(),
                                 random_seed=args.seed + 1, discrete_genome=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--N", type=float, default=10_000)
    ap.add_argument("--length", type=float, default=1e6)
    ap.add_argument("--r", type=float, default=1e-8, help="recombination rate per bp and generation")
    ap.add_argument("--theta-site", type=float, default=0.005)
    ap.add_argument("--s-g", type=float, default=0.01, help="per-generation selection coefficient; 0 = neutral")
    ap.add_argument("--growth", type=float, default=1.0, help="fold growth (neutral demography); 1 = constant")
    ap.add_argument("--growth-time", type=float, default=0.5, help="duration of growth in units of N generations")
    ap.add_argument("--windows", type=int, default=11)
    ap.add_argument("--window-bp", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output prefix: <out>.fasta and <out>-windows.csv")
    args = ap.parse_args()
    if args.windows % 2 == 0:
        ap.error("--windows must be odd, so that one window is centred on the sweep site")

    ts = simulate(args)
    rng = np.random.default_rng(args.seed + 2)
    centres = (np.arange(args.windows) + 0.5) * args.length / args.windows
    starts = (centres - args.window_bp / 2).astype(int)
    seqs = np.empty((args.n, args.windows * args.window_bp), dtype="<U1")
    rows = []
    for w, (c, a) in enumerate(zip(centres, starts)):
        block = np.tile(rng.choice(list("ACGT"), size=args.window_bp), (args.n, 1))
        segsites = 0
        for var in ts.variants(left=a, right=a + args.window_bp):
            alleles = np.array(var.alleles)
            block[:, int(var.site.position) - a] = alleles[var.genotypes]
            segsites += len(set(var.genotypes)) > 1
        seqs[:, w * args.window_bp:(w + 1) * args.window_bp] = block
        tree = ts.at(c)
        ages = sorted((tree.time(u) for u in tree.nodes() if tree.is_internal(u)), reverse=True)
        scale = args.theta_site / (2 * args.N)  # generations -> expected mutations per site (mu)
        rows.append({"window": w + 1, "centre_bp": int(c), "distance_to_sweep_bp": int(abs(c - args.length / 2)),
                     "segregating_sites": segsites, "tmrca": ages[0] * scale,
                     "star_ratio": (ages[0] - ages[1]) / ages[0]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.fasta", "w") as f:
        for i in range(args.n):
            f.write(f">s{i + 1:03d}\n{''.join(seqs[i])}\n")
    with open(f"{out}-windows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"seed {args.seed}: s_g {args.s_g} (s = {2 * args.N * args.s_g:g}), growth {args.growth}; "
          f"{ts.num_trees} trees; segregating sites per window "
          + " ".join(str(r["segregating_sites"]) for r in rows)
          + "; star ratio at centres " + " ".join(f"{r['star_ratio']:.2f}" for r in rows))


if __name__ == "__main__":
    main()
