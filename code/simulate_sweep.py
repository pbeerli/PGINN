#!/usr/bin/env python
"""
Driver for the single-population coalescent-with-selection dataset.

Parallel calls to simtree.py, one per (theta, s) draw, for this paper's
model: a single population with theta and the population-scaled selection
coefficient s = c*s_g drawn from the priors set on the command line
(reproduce.sh uses theta ~ log-uniform(0.001, 0.05) and, for training
data, s = 0 with probability 0.25, otherwise log s ~ U(log 1, log 1000)).
For s > 0 a timepoints file encoding the hard-sweep rescaling (sweep.py)
is built and passed via -tp; s == 0 needs no timepoints file (theta_eff is
constant). At theta = 0.02 (calibrate_sweep.py), s up to about 4 looks
neutral, s of about 20-100 visibly compresses genealogies, and s of
400-2000 gives strongly star-like trees.
"""
import os
import math
import csv
import random
import argparse
import subprocess
from multiprocessing import Pool, cpu_count

import sweep

HERE = os.path.dirname(os.path.abspath(__file__))
SIMTREE = os.path.join(HERE, "simtree.py")

THETA_MIN, THETA_MAX = 0.005, 0.05
S_MAX = 20000.0
N_INDIVIDUALS = 10
N_SEGMENTS = 40


def pick_params(rng, s_max, theta_min=THETA_MIN, theta_max=THETA_MAX, theta_prior="uniform", s_zero_frac=0.0,
                s_prior="uniform", s_min=1.0):
    if theta_prior == "loguniform":
        theta = round(math.exp(rng.uniform(math.log(theta_min), math.log(theta_max))), 6)
    else:
        theta = round(rng.uniform(theta_min, theta_max), 6)
    # s_zero_frac > 0: a point mass at s=0 so neutral genealogies are well
    # represented in training, not just the ~s/s_max sliver a uniform prior gives
    if s_max > 0 and not (s_zero_frac > 0 and rng.random() < s_zero_frac):
        if s_prior == "loguniform":
            s = round(math.exp(rng.uniform(math.log(s_min), math.log(s_max))), 4)
        else:
            s = round(rng.uniform(0.0, s_max), 4)
    else:
        s = 0.0
    return theta, s


def run_one_sim(args):
    i, loci, sites, folder, s_max, nind, job_seed, prior = args
    rng = random.Random(job_seed)
    theta, s = pick_params(rng, s_max, *prior)
    out_path = f"{folder}/test-{i:06d}-theta-{theta:.6f}-s-{s:.4f}.json"
    cmd = ["python", SIMTREE, "-t", str(theta), "-i", str(nind), "-a", "1.0",
           "-sel", str(s), "-l", str(loci), "-s", str(sites), "--json", "-f", out_path]
    if job_seed is not None:
        cmd += ["--seed", str(job_seed)]
    if s > 0:
        tp_path = f"{folder}/sweep_tp_{os.getpid()}_{i}.txt"
        sweep.make_timepoints_file(tp_path, theta, s, N_SEGMENTS)
        cmd += ["-tp", tp_path]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    if s > 0:
        os.remove(tp_path)
    return {"index": i, "theta": theta, "s": s, "seed": job_seed, "path": out_path}


def main():
    parser = argparse.ArgumentParser(description="Simulate single-population coalescent-with-selection datasets.")
    parser.add_argument("-f", "--folder", required=True, help="Output folder for the per-draw JSON files")
    parser.add_argument("-n", "--number", type=int, default=100, help="Number of (theta, s) draws")
    parser.add_argument("-l", "--loci", type=int, default=1000, help="Number of loci (replicate genealogies) per draw")
    parser.add_argument("-s", "--sites", type=int, default=1000, help="Sites per locus (passed through to simtree.py)")
    parser.add_argument("--smax", type=float, default=S_MAX,
                         help="Upper bound of the s ~ U(0, smax) prior; smax=0 forces the pure-neutral (s=0) case")
    parser.add_argument("--nind", type=int, default=N_INDIVIDUALS, help="Number of haploid individuals sampled per locus")
    parser.add_argument("--seed", type=int, default=None,
                         help="Base RNG seed; draw i uses seed+i so the dataset is exactly reproducible. "
                              "Omit for nondeterministic draws.")
    parser.add_argument("--theta-min", type=float, default=THETA_MIN)
    parser.add_argument("--theta-max", type=float, default=THETA_MAX)
    parser.add_argument("--theta-prior", choices=["uniform", "loguniform"], default="uniform")
    parser.add_argument("--s-zero-frac", type=float, default=0.0,
                         help="Fraction of draws with s fixed at 0 (point mass); the rest follow --s-prior")
    parser.add_argument("--s-prior", choices=["uniform", "loguniform"], default="uniform",
                         help="uniform: s ~ U(0, smax); loguniform: log s ~ U(log s_min, log smax)")
    parser.add_argument("--s-min", type=float, default=1.0, help="lower bound of the log-uniform s prior")
    args = parser.parse_args()
    prior = (args.theta_min, args.theta_max, args.theta_prior, args.s_zero_frac, args.s_prior, args.s_min)

    os.makedirs(args.folder, exist_ok=True)
    jobs = [(i, args.loci, args.sites, args.folder, args.smax, args.nind,
              None if args.seed is None else args.seed + i, prior)
             for i in range(1, args.number + 1)]
    with Pool(processes=cpu_count()) as pool:
        records = pool.map(run_one_sim, jobs)

    manifest_path = os.path.join(args.folder, "manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "theta", "s", "seed", "path"])
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: r["index"]))


if __name__ == "__main__":
    main()
