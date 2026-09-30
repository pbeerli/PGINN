#!/bin/bash
#SBATCH --job-name=adh_migrate_posterior
#SBATCH --partition=<your-partition>
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Adh posterior-genealogy run (migrate-n 6.1.11 or later), 16 MPI ranks, the
# same count as a local "mpirun -np 16 migrate-n-mpi parmfile -nomenu".
# Expects infile (make_migrate_windows.py --four-gamete) and parmfile here.
# load your site's compiler and MPI modules here, e.g.
# module load gcc openmpi
cd "$SLURM_SUBMIT_DIR"
srun --mpi=pmix "$HOME/bin/migrate-n-mpi" parmfile -nomenu
