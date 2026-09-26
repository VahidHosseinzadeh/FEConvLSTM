#!/bin/bash
# Submit the standard comparison on one dataset -- identical data and budget for every model:
#   lstm, felstm, melstm (frame-pair velocities), melstm (self-tracking), melstm with the TRUE
#   velocity (oracle), and a ConvLSTM in the frame co-moving with the true motion (oracle
#   stabiliser). Eulerian / Lagrangian persistence are evaluated inside every run.
#
#   bash motion_benchmarks/slurm/run_suite.sh radar_synthetic --radar_lifetime 6,48
#   bash motion_benchmarks/slurm/run_suite.sh swift_hohenberg --fluid_action regular
#   bash motion_benchmarks/slurm/run_suite.sh rbc3d --rbc_file rbc3d.h5 --fluid_action galilean
#
# Shared settings below; everything after the dataset name is passed to every run.
# --decoder_input warped: the rollout feeds each slot / velocity copy the previous prediction
# ALIGNED with its warped state, which is what makes the rollout exactly equivariant (see
# README, "decoder input"). The Moving MNIST runs used the original 'previous' input; pass
# --decoder_input previous to reproduce that protocol instead.
# MODELS / TAG / SEED can be overridden from the environment, e.g. MODELS="lstm melstm".
set -e
DATASET=${1:?usage: run_suite.sh <dataset> [extra train_motion args]}
shift
SEED=${SEED:-42}
TAG=${TAG:-s${SEED}}
FE_V=${FE_V:-3}
MODELS=${MODELS:-"lstm felstm melstm_fp melstm_track melstm_oracle lstm_stabilized"}

COMMON=(--dataset "$DATASET" --hidden_size 32 --decoder_hidden_size 64 --decoder_conv_layers 1
        --batch_size 16 --epochs 40 --lr 1e-3 --use_lr_scheduler --early_stop_patience 8
        --min_epochs 15 --data_seed "$SEED" --model_seed "$SEED" --input_frames 12
        --pred_frames 12 --gen_pred_frames 36 --decoder_input warped
        --save_dir ./experiments/motion_benchmarks)

for M in $MODELS; do
  case $M in
    lstm)            EXTRA=(--model lstm) ;;
    felstm)          EXTRA=(--model felstm --v_range "$FE_V") ;;
    melstm_fp)       EXTRA=(--model melstm --velocity_source frame_pair --num_vel_modes 1) ;;
    melstm_track)    EXTRA=(--model melstm --velocity_source track --num_vel_modes 2
                            --eval_velocity_mode both) ;;
    melstm_oracle)   EXTRA=(--model melstm_oracle --num_vel_modes 1) ;;
    lstm_stabilized) EXTRA=(--model lstm_stabilized) ;;
    *) echo "unknown model key $M"; exit 1 ;;
  esac
  NAME="${DATASET}_${M}_${TAG}"
  sbatch --job-name="$NAME" "$(dirname "$0")/train_motion.sbatch" \
      "${COMMON[@]}" "${EXTRA[@]}" --run_name "$NAME" "$@"
done
