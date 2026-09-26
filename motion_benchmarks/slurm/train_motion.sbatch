#!/bin/bash
# One train_motion.py run as a Slurm job. Every argument is passed through.
#
#   sbatch --job-name=radar_melstm motion_benchmarks/slurm/train_motion.sbatch \
#       --dataset radar_synthetic --model melstm --velocity_source frame_pair ...
#
# Run from the repo root. PYTHON overrides the interpreter (default: python on PATH).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

set -e
PYTHON=${PYTHON:-python}
echo ">>> $(date)  $SLURM_JOB_NAME  args: $*"
$PYTHON -m motion_benchmarks.train_motion --num_workers ${NUM_WORKERS:-6} "$@"
