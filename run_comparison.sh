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
MOTION_MODE=piecewise   # constant   : one velocity for the whole sequence
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

# Only the applicable knobs get passed, per the dataset's own applicability
# table (TDMovingMNISTDataset docstring, "Which parameters apply to which mode").
MOTION=(--motion_mode "$MOTION_MODE" --data_v_range "$DATA_V_RANGE")
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
    # eval_velocity_mode both: honest val drives selection, oracle val logged
    # alongside (velocity-vs-rendering decomposition). MELSTM-only effect.
    EXTRA=(--model melstm --num_vel_modes 2 --eval_velocity_mode both --wandb_name "melstm_${RUN_TAG}") ;;
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
MOTION_STAMP="$SAVE_DIR/run_state/motion_${MODEL}.cfg"
MOTION_CFG="${MOTION[*]}"

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

mkdir -p "$SAVE_DIR/run_state"
printf '%s\n' "$MOTION_CFG" > "$MOTION_STAMP"

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
