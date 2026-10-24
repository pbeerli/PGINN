#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/rescore-%j.out
# Workers pull finished replicates of data/chrom-n100 (one core each);
# several jobs on different partitions share the work through the lock
# directories of rescore_seeds.sh. Submit from the directory that holds
# real_adh_migrate_scan.py, model/ and data/chrom-n100, e.g.
#   sbatch --partition=<partition> --ntasks=40 rescore_seeds.slurm
cd $SLURM_SUBMIT_DIR
export PY=${PY:-$HOME/venv/pginn/bin/python} OMP_NUM_THREADS=1
ls -d data/chrom-n100/*/rep* | xargs -n 1 -P $SLURM_NTASKS bash rescore_seeds.sh
