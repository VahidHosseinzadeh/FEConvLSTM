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
# HARD_STOP (environment, e.g. 2026-10-01T23:58): kill the run at that time whatever happens,
# and do not start after it. Pair it with an earlier --stop_at, which ends training in time to
# write the results.
RUN=()
if [ -n "$HARD_STOP" ]; then
  LEFT=$(( $(date -d "$HARD_STOP" +%s) - $(date +%s) ))
  if [ "$LEFT" -le 60 ]; then echo "past HARD_STOP=$HARD_STOP, not starting"; exit 0; fi
  RUN=(timeout "$LEFT")
fi
"${RUN[@]}" $PYTHON -m motion_benchmarks.train_motion --num_workers ${NUM_WORKERS:-6} "$@"
