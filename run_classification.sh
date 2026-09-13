#!/bin/bash
# Common-Fate motion-defined digit classification: three-model comparison.
#
#   bash run_classification.sh lstm
#   bash run_classification.sh felstm
#   bash run_classification.sh melstm
#
# The task: T context frames of a digit defined ONLY by moving differently from
# its background, then name the digit. No single frame contains it.
#
# READ lstm FIRST. It has no transport structure and should sit at chance (10%).
# If it climbs meaningfully above chance the dataset is leaking a per-frame cue
# and neither of the other two numbers means anything.
#
# Settings below are shared and MUST stay identical across the three runs --
# hidden size and head are matched, so the models differ only in how they
# transport the hidden state.
set -e
MODEL=${1:?usage: bash run_classification.sh lstm|felstm|melstm}

# ---- shared ---------------------------------------------------------------
HIDDEN=32
HEAD_CH=64
HEAD_BLOCKS=3
BATCH=64
EPOCHS=40
LR=1e-3
SEQ_LEN=15         # context T
IMAGE=64
TRAIN_SAMPLES=20000
POOL=attention     # 'max' is the repo default elsewhere and is expected to fail
                   # here: every velocity copy carries equal-amplitude noise, so
                   # the informative one differs by spatial COHERENCE, not by
                   # magnitude. Worth running once as a contrast.

# ---- data -----------------------------------------------------------------
DATA_V=2           # figure max speed. felstm needs V_RANGE >= this, and its cost
                   # grows as (2R+1)^2, so raising it is expensive for felstm only.
BG_MIN=4           # background speed band, DISJOINT from the figure grid
BG_MAX=5           # (BG_MIN > DATA_V) => figure and background can never coincide
MOTION=piecewise   # figure velocity held 3-6 frames, then changes
CORR_LEN=1.0       # keep <= 1.0: above that the texture seam makes the digit
                   # visible in a single frame and the task stops being motion-defined

# ---- per-model ------------------------------------------------------------
V_RANGE=2          # felstm: (2*2+1)^2 = 25 transported copies, covers |v| <= 2
N_SLOTS=4          # melstm: slot_hit_fig 62%/78%/94% at K=2/4/6 on this data
VEL_SRC=frame_pair # melstm: 'tracked' measurably fails here -- see --help

SAVE_DIR=./experiments_classification

python moving_mnist/train_classification.py \
  --model "$MODEL" \
  --hidden_size $HIDDEN --head_channels $HEAD_CH --head_blocks $HEAD_BLOCKS \
  --velocity_pool $POOL --v_range $V_RANGE --num_vel_modes $N_SLOTS \
  --velocity_source $VEL_SRC \
  --seq_len $SEQ_LEN --image_size $IMAGE \
  --data_v_range $DATA_V --bg_speed_min $BG_MIN --bg_speed_max $BG_MAX \
  --motion_mode $MOTION --corr_len $CORR_LEN \
  --batch_size $BATCH --epochs $EPOCHS --lr $LR \
  --max_train_samples $TRAIN_SAMPLES \
  --use_lr_scheduler --early_stop_patience 10 \
  --save_dir $SAVE_DIR \
  --run_name "cf_cls_${MODEL}" \
  "${@:2}"
