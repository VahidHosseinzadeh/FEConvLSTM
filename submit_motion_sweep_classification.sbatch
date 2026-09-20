#!/bin/bash
# Motion-regime sensitivity of the trained CLASSIFICATION models. Evaluation only.
# The classification counterpart of submit_motion_sweep.sbatch -- that one sweeps
# rollout MSE, this one sweeps accuracy on the motion-defined digit.
#
# Run AFTER the three classifiers have finished training.
#
#   sbatch --job-name=cls_sweep submit_motion_sweep_classification.sbatch
#   sbatch --job-name=cls_sweep submit_motion_sweep_classification.sbatch 5000
#   RUN_FILTER=s1_stress_test sbatch --job-name=cls_sweep submit_motion_sweep_classification.sbatch
#
# Arguments are  [N_SEQ] [SAVE_DIR] [BATCH]  -- pick ONE line, they are not a
# sequence to run in order.
#
# RUN_FILTER is how you choose WHICH arm to evaluate. A training save_dir
# accumulates every arm ever run (head2, head32_e70, s1_stress_test, ...), and
# the sweep refuses to guess: if more than one run matches a model it stops and
# lists them rather than silently evaluating the wrong checkpoints.
#
# Reads   $SAVE_DIR/models/*_best.pth  and  $SAVE_DIR/results/history_*.json
# Writes  $SAVE_DIR/motion_sweep_classification.npz   -- ONE file, all 21 cells
#         (1 reference + 7 rate + 6 jump + 7 named regimes) x 3 models.
#
# Copy that single .npz back to the repo beside the checkpoints and
# make_figures_common_fate.ipynb draws the figure from it -- the notebook runs
# the sweep itself only when the file is missing.
#
# The file is rewritten after EVERY cell (atomically), so a job cut off by the
# walltime still leaves a usable partial sweep rather than nothing. A rerun
# recomputes every cell deterministically (fixed --data_seed), so it costs time
# but never corrupts anything. The notebook plots whatever cells are present.
#
# What it measures. Only the motion kwargs change between cells; canvas, sequence
# length, texture and the figure/background rule all come from each checkpoint's
# own training config, so a cell differs from training in motion alone. All three
# models see the SAME sequences in a cell, which is what lets a small accuracy
# difference mean anything. BatchNorm statistics are NOT recomputed per regime --
# refitting them would let the model absorb the very shift being measured.
#
# Cost note: felstm carries 25 hidden-state copies and dominates the runtime, as
# in the prediction sweep. Measured on an M-series GPU it is ~2.5 min per cell at
# 500 sequences (~1 h for the full 21), which is why this exists. On a cluster GPU
# raise BATCH well above the MPS-safe default of 50 -- felstm is the only model
# that needs a small one, and only because of memory.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

set -e

N_SEQ=${1:-2000}                          # sequences per cell
SAVE_DIR=${2:-./experiments_classification}   # where run_classification.sh wrote them
BATCH=${3:-250}                           # lower if the GPU OOMs
RUN_FILTER=${RUN_FILTER:-}                # substring picking ONE arm per model

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=(python moving_mnist/motion_sweep_classification.py
     --save_dir "$SAVE_DIR"
     --models lstm felstm melstm
     --n_sequences "$N_SEQ"
     --batch_size "$BATCH"
     --data_seed 123
     --download)

if [ -n "$RUN_FILTER" ]; then
  CMD+=(--run_filter "$RUN_FILTER")
  echo "evaluating only runs matching: $RUN_FILTER"
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

echo ">>> done. Copy back:  $SAVE_DIR/motion_sweep_classification.npz"
