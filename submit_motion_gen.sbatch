#!/bin/bash
# Motion-generalization sweep over already-trained checkpoints.
# Run AFTER submit_comparison.sbatch has finished all three models.
#
#   sbatch --job-name=motion_gen submit_motion_gen.sbatch
#   sbatch --job-name=motion_gen submit_motion_gen.sbatch 2000     # fewer sequences
#
# Reads   $SAVE_DIR/models/   and  $SAVE_DIR/results/history_*.json
# Writes  $SAVE_DIR/motion_gen/motion_gen_<model>_<runid>.npz
#
# Evaluation only -- no training, no checkpointing, no self-resubmission. If it
# gets cut off, just run it again; nothing is left in a partial state except an
# unwritten .npz (results are only saved once every motion has been scored).

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log

set -e

N_SEQ=${1:-10000}          # sequences per motion preset
SAVE_DIR=./experiments     # must match run_comparison.sh's SAVE_DIR
BATCH=100                  # rollout batch; lower if the GPU OOMs

# felstm carries 25 hidden-state copies through an 85-step rollout, so it
# dominates the runtime; expandable_segments keeps the allocator from
# fragmenting across the eight presets.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python moving_mnist/motion_generalization.py \
  --model_save_dir "$SAVE_DIR" \
  --models lstm felstm melstm \
  --n_sequences "$N_SEQ" \
  --batch_size "$BATCH" \
  --root ./data \
  --data_seed 42
