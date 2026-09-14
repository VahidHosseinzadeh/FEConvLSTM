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
EPOCHS=40
LR=1e-3
SEQ_LEN=15         # context T
IMAGE=64
TRAIN_SAMPLES=20000
VAL_SAMPLES=2000   # --val_fraction of MNIST is 6000, paid every epoch; 2000 is
                   # plenty for a val estimate and meaningfully cheaper for felstm
POOL=attention     # 'max' is the repo default elsewhere and is expected to fail
                   # here: every velocity copy carries equal-amplitude noise, so
                   # the informative one differs by spatial COHERENCE, not by
                   # magnitude. Worth running once as a contrast.

# ---- data -----------------------------------------------------------------
DATA_V=2           # figure max speed. felstm needs V_RANGE >= this, and its cost
                   # grows as (2R+1)^2, so raising it is expensive for felstm only.
BG_MODE=opposite   # background stays ON the shared velocity grid and is separated by
                   # DIRECTION (opposing the digit at t=0) rather than by speed.
                   # 'disjoint' would guarantee separation by making the background
                   # faster than any figure -- but that puts it beyond every lattice
                   # copy felstm has, so felstm could not represent the background at
                   # all while melstm's tracked slots could: expressive power
                   # confounded with the effect being measured.
MOTION=piecewise   # figure velocity held 3-6 frames, then changes
CORR_LEN=1.0       # keep <= 1.0: above that the texture seam makes the digit
                   # visible in a single frame and the task stops being motion-defined

# ---- per-model ------------------------------------------------------------
V_RANGE=2          # felstm: (2*2+1)^2 = 25 transported copies, covers |v| <= 2,
                   # which now also covers the background since it shares the grid
N_SLOTS=2          # melstm: the scene has exactly two motions, digit and background.
                   # Measured slot_hit_fig with VEL_SRC=bootstrap: 97.9% at K=2, and
                   # identically 97.9% at K=3 and K=4 -- extra slots buy nothing.
VEL_SRC=bootstrap  # melstm: slot_hit_fig at K=2 is 97.9% (bootstrap) vs 68.8%
                   # (frame_pair) vs 0.0% (tracked, MEConvLSTM's own protocol)

# Batch size is NOT shared, because memory is not. The recurrent state is carried
# on every velocity copy at every timestep and kept for BPTT, so felstm's 25
# copies cost ~25x melstm's 4 at the same hidden size. Profiled per piece, the
# cost is the cell's CONVOLUTION (164 ms/step at B=16,K=4,64px on an M-series
# GPU) -- the warp is 5 ms and the phase correlation under 1 ms, so neither is
# worth optimising.
#
# Calibrated from run_comparison.sh, which measured ~62GB for felstm at
# hidden=32 / batch=32 / seq_len=25 on the 80GB A100. This task runs seq_len=15,
# so ~0.6x that: batch 32 lands near 37GB and fits, batch 64 would sit at the
# ceiling. Lower the felstm arm of the case below first if a run OOMs.
case "$MODEL" in
  felstm) BATCH=32 ;;
  *)      BATCH=64 ;;
esac

# Local runs: lstm is fine (~0.3 s/batch at B=16/64px on an M-series GPU),
# melstm is slow but possible (~14 s/batch), and felstm OOMs outright at B=16
# within a 20GB MPS budget. felstm is a cluster-only run -- use
# submit_classification.sbatch.

SAVE_DIR=./experiments_classification

python moving_mnist/train_classification.py \
  --model "$MODEL" \
  --hidden_size $HIDDEN --head_channels $HEAD_CH --head_blocks $HEAD_BLOCKS \
  --velocity_pool $POOL --v_range $V_RANGE --num_vel_modes $N_SLOTS \
  --velocity_source $VEL_SRC \
  --seq_len $SEQ_LEN --image_size $IMAGE \
  --data_v_range $DATA_V --bg_mode $BG_MODE \
  --motion_mode $MOTION --corr_len $CORR_LEN \
  --batch_size $BATCH --epochs $EPOCHS --lr $LR \
  --max_train_samples $TRAIN_SAMPLES --val_size $VAL_SAMPLES \
  --use_lr_scheduler --early_stop_patience 10 \
  --save_dir $SAVE_DIR \
  --run_name "cf_cls_${MODEL}" \
  "${@:2}"
