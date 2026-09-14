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
# NOTE on lstm: above chance is EXPECTED and is not by itself a leak. A plain
# ConvLSTM convolves the input with its previous hidden state, which is enough to
# build local motion detectors; it just cannot TRANSPORT its state to accumulate
# the figure coherently. The claim under test is melstm/felstm > lstm.
# The real leak test is a model with no motion access at all:
#   python moving_mnist/leak_probe.py
#
# Settings below are shared and MUST stay identical across the three runs --
# hidden size and head are matched, so the models differ only in how they
# transport the hidden state.
#
# Early stopping is OFF (--early_stop_patience 0): every model runs the full
# EPOCHS so the three curves are directly comparable end to end.
set -e
MODEL=${1:?usage: bash run_classification.sh lstm|felstm|melstm}

# ---- shared ---------------------------------------------------------------
HIDDEN=32
HEAD_CH=64
HEAD_BLOCKS=3
EPOCHS=50
LR=1e-3
SEQ_LEN=15         # context T
IMAGE=36           # a 28px digit still has room to travel on the torus, and felstm
                   # carries every one of its 25 copies at every timestep, so this is
                   # where its cost is actually decided
TRAIN_SAMPLES=50000
                   # NOT a dataset size. random=True renders a FRESH sequence on every
                   # access -- new digit, new textures, new velocities -- so this only
                   # sets how long an epoch is. Over 50 epochs that is 2.5M distinct
                   # sequences and there is no small-dataset overfitting to worry about;
                   # the only thing it buys is gradient steps (~39k at BATCH=64).
VAL_SAMPLES=2000   # --val_fraction of MNIST is 6000, paid every epoch; 2000 is
                   # plenty for a val estimate and meaningfully cheaper for felstm
CURVE_EVERY=25     # record the fixed-set val loss every N optimizer steps, for a
CURVE_SIZE=256     # loss-vs-steps curve far finer than one point per epoch
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
CORR_LEN=0.0       # LEAVE AT 0. Above 0 the texture seam marks the digit's outline in
                   # every single frame: a single-frame CNN with no temporal information
                   # scores 11%/13%/24%/36% at corr_len 0/0.5/1/2 against 10% chance.
                   # This is what let lstm -- the no-transport control -- reach high
                   # accuracy: it was reading the seam, not the motion.

# ---- per-model ------------------------------------------------------------
V_RANGE=2          # felstm: (2*2+1)^2 = 25 transported copies, covers |v| <= 2,
                   # which now also covers the background since it shares the grid
N_SLOTS=2          # melstm: the scene has exactly two motions, digit and background.
                   # Measured slot_hit_fig with VEL_SRC=bootstrap: 97.9% at K=2, and
                   # identically 97.9% at K=3 and K=4 -- extra slots buy nothing.
VEL_SRC=bootstrap  # melstm: slot_hit_fig at K=2 is 97.9% (bootstrap) vs 68.8%
                   # (frame_pair) vs 0.0% (tracked, MEConvLSTM's own protocol)

# Batch size is SHARED across the three on purpose: a different batch size means a
# different effective learning rate and gradient noise, which is a confound in a
# comparison whose whole point is the transport structure.
#
# It is affordable only because IMAGE=36 rather than 64. felstm carries all 25
# velocity copies of the state at every timestep and keeps them for BPTT, so it is
# the memory constraint; 36px is ~0.32x the pixels of 64px, which brought it back
# inside a shared batch. If felstm still OOMs on your GPU, drop BATCH for ALL
# THREE rather than for felstm alone, so the comparison stays matched.
#
# Profiled per piece: the cost is the recurrent cell's CONVOLUTION, not the warp
# (5 ms) or the velocity estimation (<1 ms), so hidden size, V and batch are the
# only levers.
BATCH=64

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
  --use_lr_scheduler --early_stop_patience 0 \
  --val_curve_interval $CURVE_EVERY --val_curve_size $CURVE_SIZE \
  --save_dir $SAVE_DIR \
  --run_name "cf_cls_${MODEL}" \
  "${@:2}"
