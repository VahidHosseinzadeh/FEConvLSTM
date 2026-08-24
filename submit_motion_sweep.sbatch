#!/bin/bash
# Motion-difficulty sweep over already-trained checkpoints (Task 3).
# Run AFTER submit_comparison.sbatch has finished all three models.
#
#   sbatch --job-name=motion_sweep submit_motion_sweep.sbatch
#   sbatch --job-name=motion_sweep submit_motion_sweep.sbatch 500      # fewer sequences
#   sbatch --job-name=motion_sweep submit_motion_sweep.sbatch 2000 a   # difficulty axis only
#
# Arguments are  [N_SEQ] [SWEEP] [REGIME ...]  -- pick ONE line, they are not a
# sequence to run in order. Every invocation evaluates all three models.
#
# Naming regimes after the sweep runs ONLY those cells, so you can ADD cells
# without recomputing the ones you already have (it implies SWEEP=b):
#
#   sbatch --job-name=motion_sweep submit_motion_sweep.sbatch 5000 b churn_mid churn_max
#
# Reads   $SAVE_DIR/models/   and  $SAVE_DIR/results/history_*.json
# Writes  $SAVE_DIR/motion_sweep/motion_sweep_<model>_<regime>_<runid>.npz
#         one file per (model, cell): 7 difficulty cells + 5 named regimes
#         x 3 models = 36 files.
#
# Evaluation only -- no training, no checkpointing, no self-resubmission.
# Unlike motion_generalization.py this writes each cell as it finishes rather
# than everything at the end, so a job that gets cut off leaves a USABLE but
# PARTIAL directory. Just run it again; cells are recomputed and overwritten
# deterministically (fixed --data_seed), so a rerun costs time but never
# corrupts anything. The notebook loader drops any regime that is missing for
# some model, so a partial directory still plots.
#
# Cost note: felstm carries 25 hidden-state copies through an 85-step rollout
# and dominates the runtime. Measured on CPU it is ~1.4 s per sequence for all
# three models, i.e. ~9 h for the full 12 x 2000 sweep -- hence the GPU.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

set -e

N_SEQ=${1:-2000}           # sequences per cell
SWEEP=${2:-all}            # all | a (difficulty axis) | b (named regimes)
REGIMES=("${@:3}")         # optional: only these named regimes (implies SWEEP=b)
SAVE_DIR=./experiments     # must match run_comparison.sh's SAVE_DIR
BATCH=100                  # rollout batch; lower if the GPU OOMs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=(python moving_mnist/motion_difficulty_sweep.py
     --model_save_dir "$SAVE_DIR"
     --models lstm felstm melstm
     --sweep "$SWEEP"
     --n_sequences "$N_SEQ"
     --batch_size "$BATCH"
     --root ./data
     --download
     --data_seed 42
     --gen_seq_len 100
     --gen_input_frames 15
     --image_size 36
     --max_speed 2)

if [ ${#REGIMES[@]} -gt 0 ]; then
  CMD+=(--regimes "${REGIMES[@]}")
  echo "running only these regimes: ${REGIMES[*]}"
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
