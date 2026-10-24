# PGINN: Population-Genetics-Informed Neural Networks

<img alt="logo" src="logo-pginn.png" width=25% height=25%>

Code for

Peter Beerli. *Population-Genetics-Informed Neural Networks for a
Single-Population Coalescent Under a Hard Sweep.*

PGINN adds a penalty from coalescent theory to the ordinary data-fitting
loss when training a neural network to estimate `(theta, s)` from
coalescent genealogies simulated under a hard-sweep model. The default
penalty is a Kullback-Leibler term that compares the coalescent rate
implied by the network's predicted parameters with the rate implied by
the true parameters. `code/train_pginn.py` implements the plain MSE loss
(`--pginnratio 0`), the PGINN loss (`--pginnratio > 0`) and the other
penalties (`--loss`) in one script.

## Contents

| Path | What it is |
|---|---|
| `tutorial/` | step-by-step tutorial: simulate, train and test the networks for one regime (`tutorial.pdf`, `run_tutorial.sh`, about 2 minutes) |
| `code/` | simulation, training and analysis scripts; `reproduce.sh` runs them all |
| `code/inputs/` | *Drosophila* Adh sequences (DGRP), migrate-n settings and SLURM scripts for the cluster steps |
| `docs/adh-data.md` | where the Adh data come from and how to rebuild them |
| `talk/growth_scan_talk.py` | the growth-scan figure (called by `reproduce.sh growth-scan`) |
| `reproducibility.txt` | the stages of `reproduce.sh`, the cluster steps and the output of each script |
| `fetch_data.sh` | downloads the trained networks, the migrate-n output and the score tables from Zenodo |

The earlier version of this repository, with the first preprint, is
tagged `v1-preprint`.

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

Optional: migrate-n 6.1.11 or later (MPI build), only to regenerate the
migrate-n genealogies yourself instead of downloading them.

## Tutorial

```
cd tutorial
./run_tutorial.sh        # all steps, about 70 s on a laptop
./build.sh               # tutorial.pdf from tutorial.tex
```

The tutorial runs the core of the pipeline for one regime (10 loci,
n = 10) with the settings of `reproduce.sh`; `tutorial.pdf` explains each
step and maps the other scripts.

## Trained networks and migrate-n output

The trained networks, the migrate-n output and the network score tables
are too large for git. They are archived on Zenodo,
[doi:10.5281/zenodo.23570566](https://doi.org/10.5281/zenodo.23570566):

```
./fetch_data.sh                 # FETCH_LARGE=1 adds the 5.9 GB of chromosome genealogies
```

This unpacks the files where `code/reproduce.sh` expects them
(`code/model/`, `code/data/`). With these files, the analysis stages run
without retraining and without migrate-n.

## Reproducing the figures and tables

```
cd code
./reproduce.sh all          # simulate ... revision, in order
./reproduce.sh <stage>      # one stage
```

`reproducibility.txt` lists the stages in order, the cluster steps
(migrate-n, simulated chromosomes) and which script writes which output.
Outputs go to `figures/`.

### How exact is the rebuild?

The simulated data sets are regenerated from fixed seeds, and with the
downloaded networks a rebuild reproduces every table exactly. Retraining
is exact only on the same hardware and software: the networks were
trained on an Apple M4 Max with PyTorch 2.12.1 and 2 threads per job. On
other CPUs or PyTorch versions, small rounding differences early in
training grow, and each retrained network behaves like a new training
run. Use the downloaded networks to reproduce the published numbers.

## License

MIT, see `LICENSE`.

## Citation

Please cite the Zenodo record,
[doi:10.5281/zenodo.23570566](https://doi.org/10.5281/zenodo.23570566).
