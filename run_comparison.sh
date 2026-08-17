#!/bin/bash
# Three-model comparison launcher. Run from the repo root (FEConvLSTM/):
#
#   tmux new -s melstm
#   bash run_comparison.sh melstm
#
# One model per invocation (one tmux session each): lstm | felstm | melstm.
# Auto-resumes from the newest matching checkpoint_*.pth if a previous
# attempt crashed. If you CHANGE any setting below, delete the stale
# checkpoints first (rm experiments/run_state/checkpoint_<model>_*.pth) so
# the run starts fresh instead of resuming an old configuration.
#
# EXCEPTION: changes to the motion block are detected automatically (see the
# motion stamp near the bottom) and suppress the resume by themselves, since
# those change the DATA and sweeping them is the normal way to use this
# script. Every other setting is still on you.
#
# Output layout under $SAVE_DIR (see train.py):
#   models/     the actual trained weights
#   results/    plot-script inputs (history/len_gen/vel_gen)
#   run_state/  internal recovery machinery (checkpoints, DONE flags)
#
# ---------------------------------------------------------------------------
# CURRENT DEFAULTS = the velocity-dynamics-head experiment, NOT the old
# baseline. Two data settings were changed from what this script used to run:
#
#     MOTION_MODE   piecewise -> accelerate
#     FREEZE_AFTER  (new)     -> 0
#
# Both are required for the experiment to mean anything. The dataset normally
# freezes the ground-truth velocity at the context/prediction boundary
# (FREEZE_AFTER=-1), which makes "decoder freezes its last encoder velocity"
# the EXACTLY correct policy and leaves a velocity predictor precisely zero
# headroom -- you would measure "no effect" and learn nothing. FREEZE_AFTER=0
# turns the freeze off so the velocity keeps ramping through the rollout, and
# accelerate makes it ramp systematically rather than random-walk.
#
# To get the old baseline back: MOTION_MODE=piecewise, FREEZE_AFTER=-1,
# USE_VEL_DYN=0. All three models still see identical data either way, so the
# lstm/felstm comparison remains valid at these settings -- it is just a
# harder, non-frozen version of the task.
# ---------------------------------------------------------------------------

set -e
MODEL=${1:?usage: bash run_comparison.sh lstm|felstm|melstm}

# ---- shared settings: MUST be identical across the three runs -------------
HIDDEN=32          # cheap: felstm's cost scales ~quadratically in hidden on top of its
                   # existing 25x multiplier, so this matters far more for wall-clock
                   # than 45 or 64 would. Already validated training (melstm) at this size.

BATCH=32           # ~62GB for felstm at the settings below, on the 80GB A100 (~75GB
                   # usable after CUDA context + fragmentation). Calibrated from the
                   # real 74.71GB measurement at hidden=64/batch=32/seq_len=30 --
                   # activations scale ~linearly in batch, hidden and seq_len, so that
                   # config sat right at the ceiling and OOM'd; seq_len=25 pulls it back.
DEC_LAYERS=1       # hidden decoder blocks (total convs = this + 1); ~37k extra params
                   # vs 1 layer, negligible next to the ~150k-param cell — not a real cost.
DEC_HIDDEN=32     # decoder conv width, independent of HIDDEN (which sets the recurrent
                   # cell width). Cheap to raise: the recurrent state is carried on every
                   # velocity copy at every timestep and kept for BPTT, while the decoder
                   # runs once per predicted frame on the already-pooled map -- 128 costs
                   # ~1GB here vs ~62GB for the encoder. Set to "$HIDDEN" for the old
                   # behavior (decoder width tied to the cell width).
SEQ_LEN=25
INPUT_FRAMES=15    # training context; pred = SEQ_LEN - INPUT_FRAMES = 10
GEN_INPUT=15       # = INPUT_FRAMES so len-gen isolates horizon only: evaluating at a
                   # context length the model never trained on is itself OOD (it hits
                   # felstm hardest -- every wrong-velocity copy drifts for the extra
                   # steps), which shows up as inflated error from the very first
                   # predicted frame rather than as a horizon effect.
GEN_SEQ_LEN=100    # 90 predicted frames in the len-gen benchmark (9x trained horizon)
IMAGE=36
EPOCHS=50          # generous shared ceiling; early stopping (below) ends lstm/melstm
                   # well before this once converged. felstm's real limit is wall-clock,
                   # not this number.
MIN_EPOCHS=40      # no early stop before this many epochs (gives the LR scheduler,
                   # patience=5, room to cut LR at least once first)
EARLY_STOP_PATIENCE=0   # ~2-3 LR reductions' worth of chances before giving up
SEED=42
SAVE_DIR=./experiments

# ---- motion settings: ALSO must be identical across the three runs --------
# These define the data, so a comparison is only meaningful if all three
# models saw the same motion. Only the parameters that apply to the chosen
# MOTION_MODE are passed through (see the case block below) -- the rest are
# left at train.py's defaults rather than silently pretending to matter.
MOTION_MODE=accelerate  # constant   : one velocity for the whole sequence
                        # piecewise  : held MIN_SEGMENT..MAX_SEGMENT frames, then changes
                        # stochastic : P_CHANGE chance of changing at every step
                        # accelerate : speed ramps by a per-digit constant sign every
                        #              MIN_SEGMENT..MAX_SEGMENT frames, clipped to
                        #              DATA_V_RANGE -- systematic, not a random walk,
                        #              so |v| changes over the context window
TRANSITION_MODE=smooth  # what a change jumps TO. uniform = anywhere on the grid;
                        # smooth = a neighbouring velocity (each component moves by
                        # at most 1) with probability SMOOTH_PROB. Applies to
                        # piecewise/stochastic only; accelerate defines its own step.
DATA_V_RANGE=2          # velocity grid is [-N..N]^2 minus (0,0), in pixels/frame.
                        # Raising this makes felstm markedly more expensive (see
                        # FE_V_RANGE below) -- (2N+1)^2-1 candidate slots: 24 at N=2,
                        # 48 at N=3.
MIN_SEGMENT=3           # piecewise/accelerate: frames held before a velocity change.
MAX_SEGMENT=6           # For accelerate, this is the ramp interval: smaller = faster
                        # acceleration. Ignored by constant/stochastic.
P_CHANGE=0.25           # stochastic only: per-step probability of a change.
SMOOTH_PROB=0.8         # TRANSITION_MODE=smooth only. 0.0 is exactly equivalent to
                        # TRANSITION_MODE=uniform.
FREEZE_AFTER=0          # When the dataset stops letting the velocity evolve.
                        #  -1 = freeze at the context/prediction boundary
                        #       (INPUT_FRAMES for train/val/test, GEN_INPUT for
                        #       len-gen) -- the historical behavior. Under it the
                        #       velocity that governs the ENTIRE rollout equals the
                        #       last transition visible in the context, so freezing
                        #       the last encoder velocity is exactly right and
                        #       nothing can beat it. Do not run the dynamics-head
                        #       experiment here.
                        #   0 = no freezing: the velocity keeps evolving through the
                        #       rollout. This is the ONLY regime where predicting
                        #       the velocity can beat freezing it.
                        #  >0 = freeze at that explicit step, for every split.

# Only the applicable knobs get passed, per the dataset's own applicability
# table (TDMovingMNISTDataset docstring, "Which parameters apply to which mode").
# FREEZE_AFTER lives in this array (not COMMON) because it is a DATA setting:
# that puts it in the motion stamp below, so changing it correctly refuses to
# resume a checkpoint trained on the frozen version of the data.
MOTION=(--motion_mode "$MOTION_MODE" --data_v_range "$DATA_V_RANGE"
        --freeze_after "$FREEZE_AFTER")
case $MOTION_MODE in
  constant)
    MOTION_TAG="const" ;;
  piecewise)
    MOTION+=(--transition_mode "$TRANSITION_MODE"
             --min_segment "$MIN_SEGMENT" --max_segment "$MAX_SEGMENT")
    MOTION_TAG="pw${MIN_SEGMENT}-${MAX_SEGMENT}${TRANSITION_MODE:0:1}" ;;
  stochastic)
    MOTION+=(--transition_mode "$TRANSITION_MODE" --p_change "$P_CHANGE")
    MOTION_TAG="st${P_CHANGE}${TRANSITION_MODE:0:1}" ;;
  accelerate)
    # transition_mode/smooth_probability/p_change are unused by this mode
    MOTION+=(--min_segment "$MIN_SEGMENT" --max_segment "$MAX_SEGMENT")
    MOTION_TAG="acc${MIN_SEGMENT}-${MAX_SEGMENT}" ;;
  *)
    echo "unknown MOTION_MODE: $MOTION_MODE (constant|piecewise|stochastic|accelerate)"; exit 1 ;;
esac
# smooth_probability only bites when a transition_mode is actually in play.
# Written as an if, not `[ ... ] && MOTION+=(...)`: a trailing test that fails
# is the kind of thing that interacts badly with `set -e`.
if [ "$MOTION_MODE" = piecewise ] || [ "$MOTION_MODE" = stochastic ]; then
  if [ "$TRANSITION_MODE" = smooth ]; then
    MOTION+=(--smooth_probability "$SMOOTH_PROB")
  fi
fi
MOTION_TAG="v${DATA_V_RANGE}${MOTION_TAG}"
# Freezing changes the task, not just the sample, so it belongs in the run
# name -- otherwise a frozen and an unfrozen run are indistinguishable in wandb.
case $FREEZE_AFTER in
  -1) : ;;                                   # historical default, left unmarked
   0) MOTION_TAG="${MOTION_TAG}_nofrz" ;;
   *) MOTION_TAG="${MOTION_TAG}_frz${FREEZE_AFTER}" ;;
esac

# ---- velocity dynamics head: melstm only ----------------------------------
# A small GRU over the velocity history that PREDICTS u_{t+1}, so the decoder
# can extrapolate the motion instead of freezing the last encoder velocity.
# Exactly motion-equivariant, and zero-initialized: with USE_VEL_DYN=1 but
# VEL_DYN_LOSS_W=0 the model is bitwise identical to USE_VEL_DYN=0, which is
# what makes the head safe to leave on.
#
# Ignored entirely by lstm/felstm (no velocity state to predict).
USE_VEL_DYN=1           # 0 = off, exactly the pre-head melstm. The ablation
                        # to run against this one, on the SAME data.
VEL_DYN_LOSS_W=1.0      # weight on smooth_l1(u_pred, v_measured). MUST be > 0
                        # or the head never trains and predicts the frozen
                        # velocity forever -- the "on but useless" trap.
VEL_DYN_STATE_DIM=32    # GRU hidden size, per slot. Tiny next to the cell
                        # (~4k params at 32): raise it freely if the head
                        # underfits the acceleration pattern.
VEL_DYN_GAIN=fixed      # fixed   : k=1, the phase-correlation measurement is
                        #           taken verbatim in the encoder and the head
                        #           is trained but does not steer it. Start here.
                        # learned : k=sigmoid(MLP(correlation peak score)), so a
                        #           weak/ambiguous peak defers to the head.
                        #           Second experiment, not the first.
VEL_DYN_USE_H=0         # 1 = also condition on a globally pooled (translation
                        # invariant) readout of the hidden state. Ablation only:
                        # it lets the head explain velocity by appearance and
                        # stop extrapolating, which is the failure mode this
                        # experiment is trying to avoid.
VEL_DYN_OPENLOOP_K=3    # extra multi-step supervision: for the last K encoder
                        # steps, replay the head open-loop (fed its own
                        # predictions) against velocities already measured for
                        # those steps. Trains exactly the extrapolation regime
                        # the decoder runs in, at no image-rollout cost. 0 = off.
                        # Keep well below INPUT_FRAMES.
EVAL_VEL_MODE=all       # frozen    : honest inference only (cheapest)
                        # both      : + oracle GT-tracked val
                        # all       : + head-predicted val. Logs val_loss,
                        #             val_predicted_loss and val_tracked_loss
                        #             every epoch, so the three-way comparison
                        #             lands in ONE run. Expected ordering:
                        #             tracked < predicted < frozen.
                        #             Costs two extra val passes per epoch.

# felstm carries one candidate slot per (vx,vy), so its grid must COVER the
# data's or the true velocity is simply not representable -- it can never be
# smaller than DATA_V_RANGE. Following it automatically is the safe default;
# override only if you know why. Cost scales with the slot count, so a bump
# from 2 to 3 nearly doubles felstm's memory and wall-clock.
FE_V_RANGE=$DATA_V_RANGE

# Bash array, not a backslash-continued string: a single stray trailing
# space after a "\" silently breaks string continuation (bash starts
# parsing the next line as a new command -- that's exactly what just
# happened: "run_comparison.sh: line 41: 15: command not found"). Array
# elements need no line-continuation character at all, so this class of
# corruption can't happen here.
COMMON=(
  "${MOTION[@]}"
  --hidden_size "$HIDDEN"
  --decoder_hidden_size "$DEC_HIDDEN"
  --decoder_conv_layers "$DEC_LAYERS"
  --batch_size "$BATCH"
  --grad_clip 1.0
  --data_seed "$SEED"
  --model_seed "$SEED"
  --image_size "$IMAGE"
  --seq_len "$SEQ_LEN"
  --input_frames "$INPUT_FRAMES"
  --gen_input_frames "$GEN_INPUT"
  --gen_seq_len "$GEN_SEQ_LEN"
  --lr 1e-3
  --use_lr_scheduler
  --epochs "$EPOCHS"
  --min_epochs "$MIN_EPOCHS"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --num_workers 4
  --check_velocity_predictor
  --len_gen_every 2
  --model_save_dir "$SAVE_DIR"
  --wandb_project FEConvLSTM
)

# Optional velocity-generalization heatmaps (FE-vs-ME extrapolation test).
# Expensive: full fixed-velocity test set per (vx,vy) pair, at every new best.
# COMMON+=(--run_velocity_generalization --gen_vel_min -3 --gen_vel_max 3)

RUN_TAG="h${HIDDEN}_${MOTION_TAG}_s${SEED}"   # motion is in the name so a sweep
                                           # gives distinguishable wandb runs
case $MODEL in
  lstm)
    EXTRA=(--model lstm --v_range 0 --wandb_name "lstm_${RUN_TAG}") ;;
  felstm)
    # --show_h_state: FELSTM's counterpart to melstm's --check_velocity_predictor
    # report — logs the per-(vx,vy) candidate h-slot maps to wandb.
    EXTRA=(--model felstm --v_range "$FE_V_RANGE" --show_h_state --wandb_name "felstm_${RUN_TAG}") ;;
  melstm)
    # eval_velocity_mode: honest (frozen) val always drives selection; the
    # oracle and head-predicted vals are logged alongside for the
    # velocity-vs-rendering decomposition. MELSTM-only effect.
    ME_EVAL_MODE=$EVAL_VEL_MODE
    ME_TAG=""
    EXTRA=(--model melstm --num_vel_modes 2)

    if [ "$USE_VEL_DYN" = 1 ]; then
      EXTRA+=(--use_velocity_dynamics
              --vel_dyn_state_dim "$VEL_DYN_STATE_DIM"
              --vel_dyn_gain "$VEL_DYN_GAIN"
              --vel_dyn_loss_weight "$VEL_DYN_LOSS_W"
              --vel_dyn_openloop_k "$VEL_DYN_OPENLOOP_K")
      ME_TAG="_vd${VEL_DYN_STATE_DIM}${VEL_DYN_GAIN:0:1}"
      if [ "$VEL_DYN_USE_H" = 1 ]; then
        EXTRA+=(--vel_dyn_use_h)
        ME_TAG="${ME_TAG}h"
      fi
      if [ "$VEL_DYN_OPENLOOP_K" != 0 ]; then
        ME_TAG="${ME_TAG}_ol${VEL_DYN_OPENLOOP_K}"
      fi
    else
      # 'predicted'/'all' need the head; without it they re-measure the frozen
      # rollout under a different name and quietly waste a val pass per epoch.
      if [ "$ME_EVAL_MODE" = all ]; then
        ME_EVAL_MODE=both
      elif [ "$ME_EVAL_MODE" = predicted ]; then
        ME_EVAL_MODE=frozen
      fi
      ME_TAG="_novd"
    fi

    EXTRA+=(--eval_velocity_mode "$ME_EVAL_MODE"
            --wandb_name "melstm_${RUN_TAG}${ME_TAG}") ;;
  *)
    echo "unknown model: $MODEL"; exit 1 ;;
esac

# ---- auto-resume after a crash --------------------------------------------
# Checkpoints are named checkpoint_<model>_<wandb_run_id>.pth (train.py), so
# the filename carries NO trace of the motion config. Resuming picks the
# newest match, which means changing a motion knob above and relaunching
# would silently continue a run trained on different data -- the model would
# keep its old weights and optimizer state while the dataset changed under
# it, and nothing would flag it. So stamp the motion config next to the
# checkpoints and refuse to resume across a change.
#
# A missing stamp is treated as "matches" so this doesn't disturb a run that
# is already in flight from before this guard existed.
#
# The same argument applies to the model-specific architecture flags (EXTRA):
# toggling USE_VEL_DYN changes the parameter set, but the checkpoint filename
# says only "melstm", so a relaunch would try to resume a head-less checkpoint
# into a model that has a head. train.py survives that (it detects the shape
# mismatch and starts fresh), but loudly and only after the fact -- better to
# not offer it the checkpoint at all. Kept as a SEPARATE stamp file so a run
# already in flight, which has a motion stamp but no arch stamp, still resumes.
MOTION_STAMP="$SAVE_DIR/run_state/motion_${MODEL}.cfg"
MOTION_CFG="${MOTION[*]}"
ARCH_STAMP="$SAVE_DIR/run_state/arch_${MODEL}.cfg"
ARCH_CFG="${EXTRA[*]}"

CKPT=$(ls -t "$SAVE_DIR"/run_state/checkpoint_${MODEL}_*.pth 2>/dev/null | head -1)
RESUME=()
if [ -n "$CKPT" ] && [ -f "$MOTION_STAMP" ] && [ "$(cat "$MOTION_STAMP")" != "$MOTION_CFG" ]; then
  echo ">>> Motion config CHANGED since $CKPT was written:"
  echo "      checkpoint: $(cat "$MOTION_STAMP")"
  echo "      requested : $MOTION_CFG"
  echo ">>> NOT resuming — starting a fresh run so the weights match the data."
  echo "    (old checkpoints are left in place; rm $SAVE_DIR/run_state/checkpoint_${MODEL}_*.pth to clean up)"
  CKPT=""
fi
if [ -n "$CKPT" ] && [ -f "$ARCH_STAMP" ] && [ "$(cat "$ARCH_STAMP")" != "$ARCH_CFG" ]; then
  echo ">>> Architecture config CHANGED since $CKPT was written:"
  echo "      checkpoint: $(cat "$ARCH_STAMP")"
  echo "      requested : $ARCH_CFG"
  echo ">>> NOT resuming — starting a fresh run so the weights match the model."
  echo "    (old checkpoints are left in place; rm $SAVE_DIR/run_state/checkpoint_${MODEL}_*.pth to clean up)"
  CKPT=""
fi

mkdir -p "$SAVE_DIR/run_state"
printf '%s\n' "$MOTION_CFG" > "$MOTION_STAMP"
printf '%s\n' "$ARCH_CFG" > "$ARCH_STAMP"

if [ -n "$CKPT" ]; then
  echo ">>> Found checkpoint $CKPT — resuming this run."
  RESUME=(--resume "$CKPT")
  # A DONE flag here would be stale: it means an EARLIER completion (e.g. at
  # a lower --epochs cap before you raised it), not that this resumed run is
  # done. Without clearing it, submit_comparison.sbatch's chain-check would
  # find it, wrongly conclude "already finished," and silently stop
  # resubmitting if this run later gets cut off by the 24h wall clock.
  # train.py writes a fresh one if/when this run genuinely completes again.
  rm -f "$SAVE_DIR"/run_state/DONE_${MODEL}_*.flag
fi

# expandable_segments: reduces allocator fragmentation on long runs (the
# "reserved but unallocated" growth in the OOM report)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python moving_mnist/train.py "${COMMON[@]}" "${EXTRA[@]}" "${RESUME[@]}"
