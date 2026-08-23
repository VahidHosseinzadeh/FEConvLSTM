#!/bin/bash
# Three one-factor motion sweeps over already-trained checkpoints (Task 5).
# Run AFTER submit_comparison.sbatch has finished all three models.
#
#   sbatch --job-name=motion_factors submit_motion_factors.sbatch
#   sbatch --job-name=motion_factors submit_motion_factors.sbatch 2000     # cheaper
#   sbatch --job-name=motion_factors submit_motion_factors.sbatch 5000 b   # one axis
#
# Reads   $SAVE_DIR/models/  and  $SAVE_DIR/results/history_*.json
# Writes  $SAVE_DIR/motion_factors/motion_factors_<model>_<cell>_<runid>.npz
#         17 cells x 3 models = 51 files.
#
# This is SEPARATE from submit_motion_sweep.sbatch and writes to a different
# directory; neither overwrites the other.
#
#   axis A  switching rate   7 cells (d = 0 .. 1)
#   axis B  jump size        6 cells (s = 1.0 .. 0.0), p_change compensated
#   axis C  speed range      4 cells (max_speed = 1, 3, 4, 5), p_change compensated
#   + 1 shared centre cell = the training config, the centre of all three axes
#
# The script ASSERTS that the realised switching rate on axes B and C stays at
# the target (0.1630) within --rate_tol plus Monte-Carlo error. If the assertion
# fires, the compensation is wrong and the axis is confounded -- do not use the
# results, fix it first.
#
# Evaluation only -- no training, no checkpointing, no self-resubmission. Cells
# are written as they finish, so a job cut off leaves a usable but partial
# directory; the notebook loader drops cells missing for some model. Re-running
# recomputes deterministically (fixed --data_seed).
#
# Cost: 17 x 5000 x 3 = 255,000 model-rollouts, ~0.85x submit_motion_gen.sbatch
# at its default. felstm dominates (25 hidden-state copies through 85 steps).

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

set -e

N_SEQ=${1:-5000}           # sequences per cell (same for every cell)
SWEEP=${2:-all}            # all | a (rate) | b (jump) | c (speed)
SAVE_DIR=./experiments     # must match run_comparison.sh's SAVE_DIR
BATCH=100                  # rollout batch; lower if the GPU OOMs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python moving_mnist/motion_factors_sweep.py \
  --model_save_dir "$SAVE_DIR" \
  --models lstm felstm melstm \
  --sweep "$SWEEP" \
  --n_sequences "$N_SEQ" \
  --batch_size "$BATCH" \
  --root ./data \
  --download \
  --data_seed 42 \
  --gen_seq_len 100 \
  --gen_input_frames 15 \
  --image_size 36 \
  --rate_tol 0.005
