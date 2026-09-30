# PGINN: Population-Genetics-Informed Neural Networks

<img alt="logo" src="logo-pginn.png" width=25% height=25%>

Code and preprint for

Peter Beerli. *Population-Genetics-Informed Neural Networks for a
Single-Population Coalescent Under Selection.* Preprint, bioRxiv
(DOI to be added).

PGINN adds a Kullback-Leibler term to the ordinary data-fitting loss when
training a neural network to estimate `(theta, s)` from coalescent
genealogies simulated under a hard-sweep model. The term compares the
coalescent rate implied by the network's predicted parameters with the
rate implied by the true parameters. `code/train_pginn.py` implements
the plain MSE loss (`--pginnratio 0`) and the PGINN loss
(`--pginnratio > 0`) in one script.

## Contents

| Path | What it is |
|---|---|
| `paper/PGINN-preprint.pdf` | the preprint |
| `paper/formulae.pdf` | supplementary derivation of the core equations |
| `code/` | simulation, training and analysis scripts; `reproduce.sh` runs them all |
| `code/inputs/adh/` | *Drosophila* Adh sequences (DGRP) and migrate-n settings |
| `figures/` | the paper's figures, the tables as LaTeX fragments (`tab_*.tex`), and `logs/` with each script's printed output |
| `reproducibility.txt` | maps every figure, table and number in the text to the script that makes it |
| `fetch_data.sh` | downloads the trained networks and the migrate-n Adh output from Zenodo |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

Optional: migrate-n 6.1.11 or
later (MPI build), only to regenerate the Adh posterior genealogies
yourself instead of downloading them.

## Trained networks and migrate-n output

The 234 trained networks and their scalers (about 1 GB) and the migrate-n
Adh genealogy sample (`trees.tre`, about 100 MB) are too large for git.
They are archived on Zenodo (DOI to be added):

```
./fetch_data.sh
```

This puts the networks in `code/model/` and the migrate-n output in
`code/data/adh-migrate/`. With these files, every analysis stage below
runs without retraining.

## Reproducing the paper

```
cd code
./reproduce.sh all          # everything, in order
./reproduce.sh <stage>      # one stage: simulate, train, calibration,
                            # recovery, detection, msprime, adh
```

`simulate` regenerates the simulated data sets in `code/data/` (about
14 GB). `train` retrains all networks; it runs `TRAIN_JOBS` (default 8)
jobs in parallel with `TRAIN_THREADS` (default 2) threads each. Training
reproduces the downloaded networks bit for bit only with the same seed,
thread count, CPU type and PyTorch version; otherwise the last digits of
the results can differ. The `adh` stage runs migrate-n when `MIGRATE_NP`
is set (`MIGRATE_NP=16 ./reproduce.sh adh`); otherwise it uses the files
in `code/data/adh-migrate/`. `code/inputs/adh/run_migrate.slurm` is the
same migrate-n run as a SLURM job. A full run takes a few hours on a
16-core workstation, most of it training the n=176 networks and running
migrate-n.

Outputs go to `figures/`. Comparing them with the copies in this
repository (for example with `git diff figures/`) checks the rebuild.

## License

MIT, see `LICENSE`.

## Citation

Please cite the preprint (citation to be added once it is posted).
