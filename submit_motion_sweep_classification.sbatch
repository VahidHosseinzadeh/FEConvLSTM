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
# Writes  $SAVE_DIR/motion_sweep_classification.npz   -- ONE file, 31 cells x 3 models:
#           1 reference + 7 rate + 6 jump + 8 named regimes   (the motion sweep)
#         + 6 context lengths + 3 figure speeds              (the generalization axes)
#
# WHAT EACH AXIS ANSWERS
#   rate / jump / named   how accuracy moves as the MOTION LAW changes.
#   length (D:T=...)      the encoder is recurrent and the head reads the final pooled
#                         state, so a model trained at T=15 runs at any T. Free at
#                         evaluation time, and the classification analogue of the
#                         length-generalization experiment on the prediction side.
#   speed  (E:v=...)      felstm's velocity lattice is frozen at training (v_range=2),
#                         so a figure faster than that is outside what it can represent
#                         at all, while melstm re-estimates its slots per frame pair.
#                         This is the lattice-coverage cost, at zero training cost.
#
# DIAGNOSTICS (--with_diagnostics, on by default here) additionally record, per cell,
# whether the STATE still holds the figure, which is the mechanism behind the accuracy:
#   slot_hit_fig / _bg    melstm only -- did some tracked slot end the encoder holding
#                         the figure's true velocity? Vacuous for felstm, whose lattice
#                         contains every representable velocity by construction.
#   shape_iou             area-matched IoU between the local variance of the transported
#                         state and the true figure mask, for the velocity-matched copy.
#                         Transport coherence, NOT task performance: felstm has reached
#                         0.976 accuracy with this at chance. Chance is the mask's area
#                         fraction (~0.09), reported as shape_iou_chance.
#   shape_iou_best        best over all copies. Biased toward models with more copies
#                         (best-of-25 vs best-of-2) -- compare within a model, not across.
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
# 500 sequences, which is why this exists. On a cluster GPU raise BATCH well above
# the MPS-safe default of 50 -- felstm is the only model that needs a small one, and
# only because of memory. The length cells scale with T, so D:T=30 costs about twice
# a T=15 cell; the walltime here allows for that.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

set -e

N_SEQ=${1:-5000}                          # sequences per cell
SAVE_DIR=${2:-./experiments_classification}   # where run_classification.sh wrote them
BATCH=${3:-250}                           # lower if the GPU OOMs
RUN_FILTER=${RUN_FILTER:-}                # substring picking ONE arm per model
DIAG=${DIAG:-1}                           # state diagnostics (slot hit, shape IoU)
GEN=${GEN:-1}                             # context-length and speed generalization
DIAG_SEQ=${DIAG_SEQ:-256}                 # diagnostics need far fewer sequences
DIAG_BATCH=${DIAG_BATCH:-32}              # the state tensor is (B, T, V, H, W)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=(python moving_mnist/motion_sweep_classification.py
     --save_dir "$SAVE_DIR"
     --models lstm felstm melstm
     --n_sequences "$N_SEQ"
     --batch_size "$BATCH"
     --data_seed 123
     --download)

if [ "$DIAG" != "0" ]; then
  CMD+=(--with_diagnostics --diag_sequences "$DIAG_SEQ" --diag_batch "$DIAG_BATCH")
fi
if [ "$GEN" != "0" ]; then
  CMD+=(--generalization)
fi

if [ -n "$RUN_FILTER" ]; then
  CMD+=(--run_filter "$RUN_FILTER")
  echo "evaluating only runs matching: $RUN_FILTER"
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

echo ">>> done. Copy back:  $SAVE_DIR/motion_sweep_classification.npz"
