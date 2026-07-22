#!/bin/bash
# Submit one model of the three-way comparison as a proper Slurm job.
# Slurm keeps it running independent of your SSH connection — no tmux needed.
#
# Usage (from the repo root):
#   sbatch --job-name=melstm submit_comparison.sbatch melstm
#   sbatch --job-name=felstm submit_comparison.sbatch felstm
#   sbatch --job-name=lstm   submit_comparison.sbatch lstm
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm_<job-name>_<jobid>.log
#
# Innovator's GPU partition caps walltime at 24h, likely well under a full
# ~50-epoch run (especially felstm). This script self-chains: if training
# isn't finished when the time limit hits, it resubmits itself so the run
# continues unattended across multiple 24h slices, picking up via
# run_comparison.sh's checkpoint auto-resume each time. If a job instead
# dies early (OOM, node failure), just resubmit the same sbatch command by
# hand — same auto-resume path.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

MODEL=${1:?usage: sbatch submit_comparison.sbatch lstm|felstm|melstm}
SAVE_DIR=./fernn/movmnist   # must match run_comparison.sh's SAVE_DIR

bash run_comparison.sh "$MODEL"

if ls ${SAVE_DIR}/DONE_${MODEL}_*.flag >/dev/null 2>&1; then
  echo ">>> Training complete for $MODEL — not resubmitting."
else
  echo ">>> Time limit reached before completion — resubmitting to continue."
  sbatch --job-name="$SLURM_JOB_NAME" "$0" "$MODEL"
fi
