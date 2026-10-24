#!/bin/bash
#SBATCH --job-name=chrom_pipeline
#SBATCH --partition=<your-partition>
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/chrom-%A_%a.out
#SBATCH --error=logs/chrom-%A_%a.err
# End-to-end test on simulated chromosomes (code/chrom_pipeline.sh). Array
# task i runs replicate REPS-block i: tasks 1..R sweep/constant,
# R+1..2R neutral/constant, 2R+1..3R sweep/growth, 3R+1..4R neutral/growth
# (R = REPS, default 2). Submit from code/:
#   sbatch --array=1-$((4*R)) --export=ALL,REPS=R inputs/chrom/chrom_pipeline.slurm
# (NSAMP, WINDOW_BP: see chrom_pipeline.sh; OUTDIR: default data/chrom-pipeline;
#  KIND=sweep-s20 [PRIOR=constant]: one setting, array task = replicate)
# load your site's compiler and MPI modules here, e.g.
# module load gcc openmpi
export MIGRATE_CMD=${MIGRATE_CMD:-"srun --mpi=pmix $HOME/bin/migrate-n-mpi"}
export PYTHON=${PYTHON:-python}
cd "$SLURM_SUBMIT_DIR"
R=${REPS:-2}
b=$(( (SLURM_ARRAY_TASK_ID - 1) / R )); rep=$(( (SLURM_ARRAY_TASK_ID - 1) % R + 1 ))
kinds=(sweep neutral sweep neutral); priors=(constant constant growth growth)
if [[ -n ${KIND:-} ]]; then  # one setting only, e.g. KIND=sweep-s20: task i is replicate i, constant prior
  ./chrom_pipeline.sh $SLURM_ARRAY_TASK_ID $KIND ${PRIOR:-constant} ${OUTDIR:-data/chrom-pipeline}
  exit
fi
./chrom_pipeline.sh $rep ${kinds[$b]} ${priors[$b]} ${OUTDIR:-data/chrom-pipeline}
