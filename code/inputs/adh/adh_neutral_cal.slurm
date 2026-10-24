#!/bin/bash
#SBATCH --job-name=adh_neutral_cal
#SBATCH --partition=<your-partition>
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-50
#SBATCH --output=logs/cal-%A_%a.out
#SBATCH --error=logs/cal-%A_%a.err
# Neutral calibration of the Adh pipeline, one replicate per array task
# (code/adh_neutral_cal.sh). Submit from code/ (sbatch
# inputs/adh/adh_neutral_cal.slurm); results go to data/adh-neutral-cal/.
# PYTHON must have numpy, msprime, torch and dendropy (requirements.txt).
# Scoring 15 x 1000 posterior genealogies of 176 tips needs about 60 GB.
# load your site's compiler and MPI modules here, e.g.
# module load gcc openmpi
export MIGRATE_CMD="srun --mpi=pmix $HOME/bin/migrate-n-mpi"
export PYTHON=${PYTHON:-python}
cd "$SLURM_SUBMIT_DIR"
./adh_neutral_cal.sh "$SLURM_ARRAY_TASK_ID" data/adh-neutral-cal
