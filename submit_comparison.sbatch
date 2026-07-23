#!/bin/bash
# Submit one model of the three-way comparison as a proper Slurm job.
# Slurm keeps it running independent of your SSH connection — no tmux needed.
#
# Usage (from the repo root):
  # sbatch --job-name=melstm submit_comparison.sbatch melstm
  # sbatch --job-name=felstm submit_comparison.sbatch felstm
  # sbatch --job-name=lstm   submit_comparison.sbatch lstm
#
# Monitor:
#   squeue -u $USER
#   wandb — training progress, per-epoch prints, and the melstm velocity
#   report all show up there (Logs tab).
#   tail -f slurm_<job-name>_<jobid>.log — same content, locally.
#
# NOTE: training output is NOT redirected to /dev/null (a previous version
# of this script did that, reasoning wandb's console capture would still
# work independently -- it doesn't: wandb's stdout capture apparently
# depends on the underlying file descriptor still being live, so sending it
# to /dev/null before Python even starts silently broke the wandb Logs tab
# too, not just the local file). So slurm_%x_%j.log does contain the full
# training output again, not just this script's own orchestration messages.
#
# Innovator's GPU partition caps walltime at 24h, likely well under a full
# ~30-epoch run (especially felstm). This script self-chains ONLY when the
# run was still making progress and got cut off by that wall clock — judged
# by elapsed time, since Slurm's own exit-reason isn't reliably queryable
# from inside the job that just ended. A run that exits quickly (crash,
# bad config, stale/mismatched checkpoint) is NOT resubmitted — it prints
# an error and stops, so a broken config fails once instead of spamming the
# queue forever. A hard resubmission cap is a second, unconditional backstop.
#
# If you change any run_comparison.sh setting after a partial run, delete
# the stale checkpoints first:
#   rm -f experiments/run_state/checkpoint_*.pth experiments/run_state/DONE_*.flag
# — otherwise auto-resume will try to load an old, incompatible checkpoint
# and crash (caught by the logic below, but wastes a submission).

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

MODEL=${1:?usage: sbatch submit_comparison.sbatch lstm|felstm|melstm}
SAVE_DIR=./experiments   # must match run_comparison.sh's SAVE_DIR
RUN_STATE_DIR="${SAVE_DIR}/run_state"

TIME_LIMIT_SEC=$((24 * 3600))
NEAR_LIMIT_SEC=$((TIME_LIMIT_SEC - 15 * 60))   # within 15 min of the wall = plausible timeout
MAX_RESUBMITS=8                                # hard backstop regardless of the heuristic above

RESUBMIT_COUNT_FILE="${RUN_STATE_DIR}/.resubmit_count_${MODEL}"
RESUBMIT_COUNT=$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 0)

START_TS=$(date +%s)
bash run_comparison.sh "$MODEL"
ELAPSED=$(( $(date +%s) - START_TS ))

if ls ${RUN_STATE_DIR}/DONE_${MODEL}_*.flag >/dev/null 2>&1; then
  echo ">>> Training complete for $MODEL — not resubmitting."
  rm -f "$RESUBMIT_COUNT_FILE"

elif [ "$ELAPSED" -ge "$NEAR_LIMIT_SEC" ] && [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]; then
  echo ">>> Ran ${ELAPSED}s (near the ${TIME_LIMIT_SEC}s limit) without finishing — resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMITS)."
  echo $((RESUBMIT_COUNT + 1)) > "$RESUBMIT_COUNT_FILE"
  sbatch --job-name="$SLURM_JOB_NAME" "$0" "$MODEL"

elif [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
  echo ">>> Hit the $MAX_RESUBMITS-resubmission cap without finishing — stopping. Check this model's wandb run (or slurm_${SLURM_JOB_NAME}_*.log) for why it's stuck (not just slow)."

else
  echo ">>> Exited after only ${ELAPSED}s (well under the time limit) — this looks like a crash, not a timeout. NOT resubmitting."
  echo ">>> Check this model's wandb run (Logs tab) or slurm_${SLURM_JOB_NAME}_*.log for the real error before resubmitting by hand."
fi
