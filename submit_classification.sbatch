#!/bin/bash
#SBATCH --job-name=cf_cls
#SBATCH --output=slurm_%x_%j.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Common-Fate motion-defined digit classification, one model per job:
#
#   sbatch --job-name=cf_lstm   submit_classification.sbatch lstm
#   sbatch --job-name=cf_felstm submit_classification.sbatch felstm
#   sbatch --job-name=cf_melstm submit_classification.sbatch melstm
#
# Anything after the model name is passed straight through to the training
# script, so a sweep needs no edits here:
#
#   sbatch submit_classification.sbatch melstm --num_vel_modes 6
#   sbatch submit_classification.sbatch felstm --velocity_pool max
#
# Cost note. felstm carries (2*v_range+1)^2 = 25 transported copies of the
# hidden state at every timestep and keeps them all for BPTT, so it is roughly
# 25x melstm's memory at the same hidden size -- it is the run to watch for OOM,
# and the one to drop --batch_size on first. This is also why local runs are not
# practical for it and this script exists.
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm_<job-name>_<jobid>.log
set -e

MODEL=${1:?usage: sbatch submit_classification.sbatch lstm|felstm|melstm [extra args]}
shift

echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "model   : $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Resume automatically if a previous attempt was cut off. Delete the checkpoint
# by hand if you CHANGED any setting -- otherwise the run silently continues
# under the old configuration.
CKPT="./experiments_classification/run_state/checkpoint_cf_cls_${MODEL}.pth"
RESUME=""
if [ -f "$CKPT" ]; then
    echo "resuming from $CKPT"
    RESUME="--resume $CKPT"
fi

bash run_classification.sh "$MODEL" --use_wandb $RESUME "$@"

echo "done: $MODEL"
