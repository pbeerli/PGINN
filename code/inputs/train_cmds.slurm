#!/bin/bash
# Run training commands from a file on a Slurm cluster, CHUNK lines per array
# task, one after the other with 2 threads (as reproduce.sh's parallel_cmds).
# The command file is a subset of `zsh reproduce.sh revision-list`, e.g.
#   zsh reproduce.sh revision-list | grep -- '--loss process' > cmds.txt
#   n=$(wc -l < cmds.txt); sbatch --array=1-$(( (n + 3) / 4 )) --export=ALL,CMDS=cmds.txt,CHUNK=4 inputs/train_cmds.slurm
# Run from manuscript-2/code with the data/ files the commands read.
#SBATCH --job-name=pginn_train
#SBATCH --partition=<partition>
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train-%A_%a.out

source ~/venv/pginn/bin/activate   # adjust to your Python environment
export OMP_NUM_THREADS=2
CHUNK=${CHUNK:-4}
first=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
last=$(( first + CHUNK - 1 ))
sed -n "${first},${last}p" "$CMDS" | while IFS= read -r cmd; do
  echo "$(date +%T) $cmd"
  bash -c "$cmd" || echo "FAILED: $cmd"
done
