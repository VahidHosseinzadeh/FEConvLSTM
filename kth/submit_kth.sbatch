#!/bin/bash
# Several seeds of one KTH configuration as ONE Slurm job, in parallel on one GPU.
#
#   source ~/.bashrc; conda activate torch2.2          # the job inherits this environment
#   SEEDS="1 2 3" TAG=v0 sbatch --job-name=kth_lstm_v0 kth/submit_kth.sbatch lstm --camera none
#   SEEDS="1 2 3" TAG=v0 sbatch --job-name=kth_melstm_v0 kth/submit_kth.sbatch melstm --camera none
#
# Run from the repo root. Everything after the model goes to kth/train_kth.py. Each seed is run
# kth_<model>_<TAG>_s<seed>, logs to logs/kth/<run>.log, and resumes from its last checkpoint if
# one exists (experiments_kth/run_state/), so resubmitting a killed job continues it.
#
# Why packed: at 32x32 a run uses a few GB and a fraction of an A100, and billing is per GPU-hour
# (plus CPUs). Packing also slows each run, so s/epoch from a packed job is NOT a cost-table
# number -- time a model with SEEDS=<one seed> on a GPU of its own.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=slurm_%x_%j.log

set -u
MODEL=${1:?usage: sbatch kth/submit_kth.sbatch lstm|felstm|melstm [train_kth args]}
shift
SEEDS=${SEEDS:-"1 2 3"}
TAG=${TAG:-run}
PYTHON=${PYTHON:-python}
NUM_WORKERS=${NUM_WORKERS:-2}
SAVE_DIR=${SAVE_DIR:-./experiments_kth}
mkdir -p logs/kth
echo ">>> $(date)  commit $(git rev-parse --short HEAD)  model $MODEL  seeds [$SEEDS]  tag $TAG"
echo ">>> args: $*"

pids=()
for s in $SEEDS; do
  RUN="kth_${MODEL}_${TAG}_s${s}"
  CKPT="${SAVE_DIR}/run_state/checkpoint_${RUN}.pth"
  RESUME=()
  [ -f "$CKPT" ] && RESUME=(--resume "$CKPT")
  $PYTHON -m kth.train_kth --model "$MODEL" --model_seed "$s" --run_name "$RUN" \
      --save_dir "$SAVE_DIR" --use_wandb --num_workers "$NUM_WORKERS" ${RESUME[@]+"${RESUME[@]}"} "$@" \
      > "logs/kth/${RUN}.log" 2>&1 &
  pids+=($!)
  echo ">>> started $RUN (pid $!)${RESUME:+ resuming}"
done

status=0
for p in "${pids[@]}"; do
  wait "$p" || status=1
done
for s in $SEEDS; do
  RUN="kth_${MODEL}_${TAG}_s${s}"
  echo "----- $RUN"; tail -n 8 "logs/kth/${RUN}.log"
done
exit $status
