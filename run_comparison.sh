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
# Output layout under $SAVE_DIR (see train.py):
#   models/     the actual trained weights
#   results/    plot-script inputs (history/len_gen/vel_gen)
#   run_state/  internal recovery machinery (checkpoints, DONE flags)

set -e
MODEL=${1:?usage: bash run_comparison.sh lstm|felstm|melstm}

# ---- shared settings: MUST be identical across the three runs -------------
HIDDEN=64          # cheap: felstm's cost scales ~quadratically in hidden on top of its
                   # existing 25x multiplier, so this matters far more for wall-clock
                   # than 45 or 64 would. Already validated training (melstm) at this size.
BATCH=32           # extrapolated ~25GB at hidden=32/seq_len=20, from the 74.71GB
                   # measurement at hidden=64/batch=32/seq_len=30 (activations scale
                   # ~linearly in both hidden and seq_len). Comfortable margin — still
                   # worth a quick probe once, not a real OOM risk at this size.
DEC_LAYERS=2       # hidden decoder blocks (total convs = this + 1); ~37k extra params
                   # vs 1 layer, negligible next to the ~150k-param cell — not a real cost.
SEQ_LEN=20
INPUT_FRAMES=10    # training context; pred = SEQ_LEN - INPUT_FRAMES = 10
GEN_INPUT=10       # = INPUT_FRAMES so len-gen isolates horizon only
GEN_SEQ_LEN=100    # 90 predicted frames in the len-gen benchmark (9x trained horizon)
IMAGE=36
EPOCHS=100          # generous shared ceiling; early stopping (below) ends lstm/melstm
                   # well before this once converged. felstm's real limit is wall-clock,
                   # not this number.
MIN_EPOCHS=30      # no early stop before this many epochs (gives the LR scheduler,
                   # patience=5, room to cut LR at least once first)
EARLY_STOP_PATIENCE=25   # ~2-3 LR reductions' worth of chances before giving up
SEED=42
SAVE_DIR=./experiments

# Bash array, not a backslash-continued string: a single stray trailing
# space after a "\" silently breaks string continuation (bash starts
# parsing the next line as a new command -- that's exactly what just
# happened: "run_comparison.sh: line 41: 15: command not found"). Array
# elements need no line-continuation character at all, so this class of
# corruption can't happen here.
COMMON=(
  --data_v_range 2
  --hidden_size "$HIDDEN"
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

case $MODEL in
  lstm)
    EXTRA=(--model lstm --v_range 0 --wandb_name "lstm_h${HIDDEN}_s${SEED}") ;;
  felstm)
    # --show_h_state: FELSTM's counterpart to melstm's --check_velocity_predictor
    # report — logs the per-(vx,vy) candidate h-slot maps to wandb.
    EXTRA=(--model felstm --v_range 2 --show_h_state --wandb_name "felstm_h${HIDDEN}_s${SEED}") ;;
  melstm)
    # eval_velocity_mode both: honest val drives selection, oracle val logged
    # alongside (velocity-vs-rendering decomposition). MELSTM-only effect.
    EXTRA=(--model melstm --num_vel_modes 2 --eval_velocity_mode both --wandb_name "melstm_h${HIDDEN}_s${SEED}") ;;
  *)
    echo "unknown model: $MODEL"; exit 1 ;;
esac

# ---- auto-resume after a crash --------------------------------------------
CKPT=$(ls -t "$SAVE_DIR"/run_state/checkpoint_${MODEL}_*.pth 2>/dev/null | head -1)
RESUME=()
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
