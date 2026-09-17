#!/bin/bash
# Submit one model of the Common-Fate classification comparison as a Slurm job.
# Slurm keeps it running independent of your SSH connection — no tmux needed.
#
# Usage (from the repo root):
#   sbatch --job-name=cf_lstm   submit_classification.sbatch lstm
#   sbatch --job-name=cf_felstm submit_classification.sbatch felstm
#   sbatch --job-name=cf_melstm submit_classification.sbatch melstm
#
# Anything after the model name is passed straight through to the training
# script, so a sweep needs no edits here:
#   sbatch --job-name=cf_k6 submit_classification.sbatch melstm --num_vel_modes 6
#
# Sweeping a SETTING rather than a seed: give each arm a TAG, or the arms share
# one checkpoint and the second job resumes the first one's weights.
#   TAG=head8 sbatch --job-name=cf_melstm_head8 submit_classification.sbatch \
#       melstm --head_channels 8 --head_blocks 3 --head_mlp_hidden 16
#
# For a seed sweep, set SEED in the environment (sbatch exports it by default):
#   for s in 0 1 2; do
#     SEED=$s sbatch --job-name=cf_melstm_s$s submit_classification.sbatch melstm
#   done
# Each seed gets its own run name and checkpoint. --data_seed stays fixed at 42
# inside run_classification.sh so every seed is scored on the same val set.
#
# NOTE: submit the SBATCH file, not run_classification.sh. The latter is a plain
# shell script holding the hyperparameters and carries no #SBATCH directives, so
# `sbatch run_classification.sh` fails with "No partition specified". This file
# is the Slurm wrapper; it calls run_classification.sh internally.
#
# The experiment: T context frames of a digit defined ONLY by moving differently
# from its background, then name the digit. READ THE lstm RUN FIRST — it has no
# transport structure and should sit at chance (10%). If it climbs meaningfully
# above chance the dataset is leaking a per-frame cue and neither of the other
# two numbers means anything.
#
# Also watch slot_hit_fig in the melstm log: it is the fraction of samples where
# some velocity slot actually held the figure's velocity. An accuracy number
# from slots that never transported at the figure's velocity measures nothing.
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm_<job-name>_<jobid>.log
#   wandb — per-epoch metrics and the velocity diagnostics land there too.
#
# Cost note: felstm carries (2*v_range+1)^2 = 25 transported copies of the
# hidden state at every timestep and keeps them all for BPTT, so it is roughly
# 25x melstm's memory at the same hidden size. run_classification.sh already
# gives it a smaller batch; it is still the run to watch for OOM.
#
# Self-chaining: this script resubmits ONLY when the run was still making
# progress and got cut off by the wall clock — judged by elapsed time, since
# Slurm's exit reason isn't reliably queryable from inside the job that just
# ended. A run that exits quickly (crash, bad config, stale checkpoint) is NOT
# resubmitted, so a broken config fails once instead of spamming the queue. A
# hard resubmission cap is a second, unconditional backstop.
#
# If you change any run_classification.sh setting after a partial run, delete
# the stale checkpoints first:
#   rm -f experiments_classification/run_state/checkpoint_*.pth \
#         experiments_classification/run_state/DONE_*.flag
# — otherwise auto-resume will load an old, incompatible checkpoint.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

MODEL=${1:?usage: sbatch submit_classification.sbatch lstm|felstm|melstm}
shift

SAVE_DIR=${SAVE_DIR:-./experiments_classification}   # run_classification.sh reads the same var
RUN_STATE_DIR="${SAVE_DIR}/run_state"

# TAG namespaces an ARM of a sweep. Everything this script remembers between
# submissions -- the checkpoint it auto-resumes, the DONE flag, the resubmit
# counter -- and the run name itself are keyed on it. Without that, two arms of
# a sweep share one checkpoint: the second job resumes the first one's weights,
# which for a different --head_channels is a shape mismatch and for a same-shape
# change is silent contamination. Set one per config:
#   TAG=head8 sbatch ... submit_classification.sbatch melstm --head_channels 8
TAG=${TAG:-}
RUN_NAME="cf_cls_${MODEL}_s${SEED:-0}${TAG:+_$TAG}"

TIME_LIMIT_SEC=$((24 * 3600))
NEAR_LIMIT_SEC=$((TIME_LIMIT_SEC - 15 * 60))   # within 15 min of the wall = plausible timeout
MAX_RESUBMITS=8                                # hard backstop regardless of the heuristic above

RESUBMIT_COUNT_FILE="${RUN_STATE_DIR}/.resubmit_count_${RUN_NAME}"
RESUBMIT_COUNT=$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 0)

# Resume automatically if a previous attempt was cut off.
SEED=${SEED:-0}
CKPT="${RUN_STATE_DIR}/checkpoint_${RUN_NAME}.pth"
RESUME=""
if [ -f "$CKPT" ]; then
  echo ">>> Resuming from $CKPT"
  RESUME="--resume $CKPT"
fi

START_TS=$(date +%s)
bash run_classification.sh "$MODEL" --use_wandb --run_name "$RUN_NAME" $RESUME "$@"
ELAPSED=$(( $(date +%s) - START_TS ))

if [ -f "${RUN_STATE_DIR}/DONE_${MODEL}_${RUN_NAME}.flag" ]; then
  echo ">>> Training complete for $RUN_NAME — not resubmitting."
  rm -f "$RESUBMIT_COUNT_FILE"

elif [ "$ELAPSED" -ge "$NEAR_LIMIT_SEC" ] && [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]; then
  echo ">>> Ran ${ELAPSED}s (near the ${TIME_LIMIT_SEC}s limit) without finishing — resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMITS)."
  echo $((RESUBMIT_COUNT + 1)) > "$RESUBMIT_COUNT_FILE"
  sbatch --job-name="$SLURM_JOB_NAME" "$0" "$MODEL" "$@"

elif [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
  echo ">>> Hit the $MAX_RESUBMITS-resubmission cap without finishing — stopping. Check this model's wandb run (or slurm_${SLURM_JOB_NAME}_*.log) for why it's stuck (not just slow)."

else
  echo ">>> Exited after only ${ELAPSED}s (well under the time limit) — this looks like a crash, not a timeout. NOT resubmitting."
  echo ">>> Check this model's wandb run (Logs tab) or slurm_${SLURM_JOB_NAME}_*.log for the real error before resubmitting by hand."
fi
