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
#     MOTION_MODE   piecewise -> harmonic
#     FREEZE_AFTER  (new)     -> 0
#
# Both are required for the experiment to mean anything.
#
# FREEZE_AFTER: the dataset normally freezes the ground-truth velocity at the
# context/prediction boundary (FREEZE_AFTER=-1), which makes "decoder freezes
# its last encoder velocity" the EXACTLY correct policy and leaves a velocity
# predictor precisely zero headroom. FREEZE_AFTER=0 turns that off.
#
# MOTION_MODE: turning the freeze off is not enough on its own -- the velocity
# also has to be PREDICTABLE. An exactly motion-equivariant dynamics head sees
# only the history of velocity DIFFERENCES, never the absolute velocity, so it
# can only learn dynamics whose next increment is a function of past
# increments. piecewise/stochastic have i.i.d. increments (no signal at all --
# the best possible prediction IS freeze) and accelerate reverses when |v|
# hits max_speed (a rule about the ABSOLUTE speed, which the head cannot see).
# harmonic is the family that satisfies the requirement. Measured end-of-
# rollout position error, 36 px frame, context 15, rollout 10:
#
#     shape        frozen   1-step-lagged
#     constant       0.0 px     0.0 px   <- freezing is exactly right, by design
#     orbit         38.6        5.8
#     axis          25.4        3.7
#     lissajous     37.2        5.5
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
GEN_SEQ_LEN=35     # = GEN_INPUT + 20 predicted frames, i.e. 2x the trained horizon.
                   # Was 100 (85 predicted). Long rollouts are dominated by
                   # compounding velocity error: past ~half a digit width a
                   # misplaced digit already scores worse than a blank frame,
                   # so most of an 85-frame curve measures how fast the model
                   # gives up, not how well it extrapolates.
NUM_DIGITS=1       # Digits per sequence, and NUM_VEL_MODES follows it below.
                   # 1 is the DEBUGGING configuration and the current default:
                   # with a single digit the correlation surface has one
                   # unambiguous peak, so the velocity measurement is exact
                   # (verified: 100% frame-to-frame) and any remaining failure
                   # belongs to the model rather than to the measurement. If the
                   # velocity head cannot be made to work at one digit it
                   # certainly cannot at two. Raise to 2 once it does.
IMAGE=64           # The MNIST digit is 28 px and is CENTRED, never scaled, at any
                   # image_size -- so this is really "how much room does the motion
                   # have relative to the digit". At 36 an orbit of radius ~13 px is
                   # smaller than the digit and reads as a wobble; at 64 the same
                   # motion is plainly an orbit. Costs ~3x the pixels.
EPOCHS=50          # generous shared ceiling; early stopping (below) ends lstm/melstm
                   # well before this once converged. felstm's real limit is wall-clock,
                   # not this number.
MIN_EPOCHS=40      # no early stop before this many epochs (gives the LR scheduler,
                   # patience=5, room to cut LR at least once first)
EARLY_STOP_PATIENCE=0   # ~2-3 LR reductions' worth of chances before giving up
SEED=42
SAVE_DIR=./experiments

# ---- reconstruction loss: shape / location split -------------------------
# Applies to ALL THREE models, so it stays a shared setting.
#
# Plain MSE on sparse bright-on-black frames has a degenerate minimum at
# "predict nothing": with a 28 px digit, a correctly drawn but misplaced digit
# costs the same as a blank frame at ~10 px of displacement and more beyond it.
#
#     displaced  2 px : 0.047      displaced 20 px : 0.469
#     displaced  5 px : 0.117      predict nothing : 0.234
#     displaced 10 px : 0.234  <-- identical to predicting nothing
#
# The first two harmonic runs both sat BEHIND that baseline for their whole
# length (best val 0.130 and 0.092, against 0.090 for all-zeros).
#
# FOURIER_LOSS=1 splits the squared-error half into a translation-invariant
# "shape" term and a "location" term (exact identity; they sum to the MSE).
# Raising SHAPE_WEIGHT prices blankness directly -- a blank prediction has zero
# Fourier magnitude everywhere, so it cannot hide behind roughly-right
# placement. At weights 1/1 this is EXACTLY the old MSE+L1, so the split is
# safe to leave on; only the weights change what is optimised.
FOURIER_LOSS=1
SHAPE_WEIGHT=3.0        # >1 punishes blur and blankness harder
LOCATION_WEIGHT=1.0     # where a pure translation puts ALL of its error

# ---- motion settings: ALSO must be identical across the three runs --------
# These define the data, so a comparison is only meaningful if all three
# models saw the same motion. Only the parameters that apply to the chosen
# MOTION_MODE are passed through (see the case block below) -- the rest are
# left at train.py's defaults rather than silently pretending to matter.
MOTION_MODE=harmonic    # constant   : one velocity for the whole sequence
                        # piecewise  : held MIN_SEGMENT..MAX_SEGMENT frames, then changes
                        # stochastic : P_CHANGE chance of changing at every step
                        # accelerate : speed ramps by a per-digit constant sign every
                        #              MIN_SEGMENT..MAX_SEGMENT frames, clipped to
                        #              DATA_V_RANGE -- systematic, not a random walk,
                        #              so |v| changes over the context window
                        # harmonic   : constant drift + a sinusoid per axis, drawn
                        #              per digit then DETERMINISTIC. The only mode
                        #              whose future velocity is predictable from
                        #              the velocity history alone, hence the only
                        #              one where USE_VEL_DYN can win. See the
                        #              HARMONIC block below.
TRANSITION_MODE=smooth  # what a change jumps TO. uniform = anywhere on the grid;
                        # smooth = a neighbouring velocity (each component moves by
                        # at most 1) with probability SMOOTH_PROB. Applies to
                        # piecewise/stochastic only; accelerate defines its own step.
DATA_V_RANGE=4          # velocity grid is [-N..N]^2 minus (0,0), in pixels/frame.
                        # In harmonic mode this ALSO sets the oscillation scale
                        # (amplitude = HARMONIC_AMP_* x this), and the orbit radius
                        # is roughly amplitude x period / 2pi -- so this is the main
                        # "how bold is the motion" knob. Velocities are integers, so
                        # a small range is destroyed by rounding: at 2 the sinusoid
                        # has only 5 levels.
                        #
                        # ONLY felstm pays for a large range ((2N+1)^2-1 candidate
                        # slots: 24 at N=2, 288 at N=8). melstm carries
                        # NUM_VEL_MODES slots regardless and lstm has none, so for
                        # the melstm-vs-lstm comparison this is free. Drop it back
                        # to 2-4 before running felstm.
                        #
                        # WHY 4 AND NOT 8. Reconstruction loss has a break-even
                        # displacement: past about half a digit width (~14 px
                        # here) a correctly drawn but misplaced digit costs MORE
                        # than a blank frame, so the model is rewarded for going
                        # silent. At v_range=8 the frozen decoder ends the
                        # rollout 70 px out and even a one-step-lagged oracle is
                        # at 11 px -- the whole comparison sits on the wrong side
                        # of that cliff, and the first run duly plateaued WORSE
                        # than the all-zeros predictor (0.130 vs 0.090). At
                        # v_range=4 frozen is ~24 px (clearly bad) while a good
                        # velocity model lands under 14 px (clearly good), which
                        # is the contrast the experiment is trying to measure.
                        # This is NOT about measurement accuracy: the bootstrap
                        # correlator is 100% at v_range 2, 4 and 8 alike.
MIN_SEGMENT=3           # piecewise/accelerate: frames held before a velocity change.
MAX_SEGMENT=6           # For accelerate, this is the ramp interval: smaller = faster
                        # acceleration. Ignored by constant/stochastic.
P_CHANGE=0.25           # stochastic only: per-step probability of a change.
SMOOTH_PROB=0.8         # TRANSITION_MODE=smooth only. 0.0 is exactly equivalent to
                        # TRANSITION_MODE=uniform.
# ---- harmonic motion: shape of the velocity signal (MOTION_MODE=harmonic) --
HARMONIC_SHAPES="orbit axis lissajous"
                        # Default is the OSCILLATING shapes only, so every digit
                        # actually moves in a non-trivial way. Add "constant" back
                        # to include the control member -- worth doing at least
                        # once, since it is the case where freezing the last
                        # velocity is exactly right and the head can therefore only
                        # cost something. Pass a single shape to isolate it.
                        #   constant  : pure constant flow. The degenerate
                        #               member, and the control: freezing is
                        #               already optimal, so the head must not
                        #               make things WORSE here.
                        #   orbit     : velocity rotates at a constant rate,
                        #               digit circles. Cleanest case -- one
                        #               parameter, identifiable from two
                        #               consecutive velocity differences.
                        #   axis      : constant flow along one axis,
                        #               sinusoidal along the other.
                        #   lissajous : both axes oscillate independently.
                        #               Direction AND magnitude change.
HARMONIC_PERIOD_MIN=12  # velocity period in frames. Two things pull against each
HARMONIC_PERIOD_MAX=30  # other here: the orbit RADIUS grows with the period
                        # (radius ~ amplitude x period / 2pi), but a period much
                        # longer than INPUT_FRAMES is not identifiable from the
                        # context and the head correctly falls back to freezing.
                        # 12-30 against a 15-frame context spans "one and a half
                        # cycles visible" to "half a cycle visible".
HARMONIC_AMP_MIN=0.7    # oscillation amplitude as a fraction of DATA_V_RANGE.
HARMONIC_AMP_MAX=1.0    # Velocities are integers, so a small amplitude is
                        # destroyed by rounding -- below ~0.5 the sinusoid stops
                        # being resolvable. Kept high by default so the motion is
                        # bold; lower the floor to mix gentle and violent digits.
HARMONIC_DRIFT=1        # 1 = add a constant velocity offset, so paths become
                        # looping trochoids. The offset is INTEGER and therefore
                        # cancels exactly in the velocity differences -- it is
                        # invisible to the equivariant head, so it costs that
                        # head nothing while making the motion strictly richer
                        # for any model carrying velocity implicitly. That is the
                        # inductive-bias argument, made visible in the data.
                        # 0 = oscillations centred on zero (isolation ablation).

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
  harmonic)
    # min/max_segment, transition_mode, smooth_probability and p_change are all
    # unused by this mode -- the trajectory is parametric, not step-by-step.
    MOTION+=(--harmonic_shapes $HARMONIC_SHAPES
             --harmonic_period_min "$HARMONIC_PERIOD_MIN"
             --harmonic_period_max "$HARMONIC_PERIOD_MAX"
             --harmonic_amp_min "$HARMONIC_AMP_MIN"
             --harmonic_amp_max "$HARMONIC_AMP_MAX")
    if [ "$HARMONIC_DRIFT" != 1 ]; then
      MOTION+=(--no_harmonic_drift)
    fi
    # shape initials, e.g. "coal" for all four; keeps run names short
    SHAPE_TAG=""
    for sh in $HARMONIC_SHAPES; do SHAPE_TAG="${SHAPE_TAG}${sh:0:1}"; done
    MOTION_TAG="hrm${SHAPE_TAG}${HARMONIC_PERIOD_MIN}-${HARMONIC_PERIOD_MAX}"
    if [ "$HARMONIC_DRIFT" != 1 ]; then MOTION_TAG="${MOTION_TAG}nod"; fi ;;
  *)
    echo "unknown MOTION_MODE: $MOTION_MODE (constant|piecewise|stochastic|accelerate|harmonic)"; exit 1 ;;
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
VEL_DYN_STATE_DIM=32    # GRU hidden size, per slot. Tiny next to the cell,
                        # so this is cheap: 3.5k params at 32/1 layer,
                        # 9.9k at 32/2 layers.
VEL_DYN_ARCH=gru        # gru        : emit the velocity increment directly.
                        # recurrence : emit the COEFFICIENTS of a stable
                        #              second-order linear recurrence
                        #              du_{t+1} = a1 du_t + a2 du_{t-1}, with
                        #              the poles forced inside the unit circle
                        #              so an open-loop rollout cannot diverge.
                        # A sinusoid satisfies that recurrence exactly, so this
                        # separates identification (hard, learned) from
                        # propagation (exact). It is aimed at a measured
                        # weakness: the gru head reached 0.1179 val_predicted
                        # against a ONE-STEP-LAGGED ceiling of 0.1149 -- i.e.
                        # it had learned to lag rather than to extrapolate.
VEL_DYN_LAYERS=1        # stacked GRU layers in the head
VEL_DYN_DEC_SUP=none    # What the head may take from the FUTURE frames while
                        # training (frames INPUT_FRAMES..SEQ_LEN-1, i.e. the
                        # rollout targets).
                        #   none     : nothing. The head is supervised only by
                        #              velocities measured inside the context,
                        #              which is all it ever has at inference.
                        #   teacher  : the ORIGINAL behaviour -- at each decoder
                        #              step the head is handed the ORACLE
                        #              previous velocity (measured against the
                        #              target frame) and scored one step ahead.
                        #              This is why it learned to LAG: repeating
                        #              a teacher-forced input is the optimal
                        #              answer to that task when the signal is
                        #              hard, and the head duly measured at
                        #              0.1179 against a 0.1149 one-step-lag
                        #              ceiling.
                        #   openloop : run the head on its OWN previous output
                        #              and score that against the measurement.
                        #              Future frames are then TARGETS but never
                        #              INPUTS, so it is trained to extrapolate.
                        #
                        # Note the head still gets multi-step supervision with
                        # 'none', from VEL_DYN_OPENLOOP_K -- that replay lives
                        # entirely inside the context.
                        #
                        # ARCH/LAYERS/STATE_DIM and DECODER_SAMPLING_P below are
                        # all back at the configuration that produced
                        # val_predicted_loss = 0.1179 (against a frozen ceiling
                        # of 0.1574 and a one-step-lag ceiling of 0.1149). The
                        # follow-up run changed all four AT ONCE plus the eval
                        # protocol, so nothing in it could be attributed. Change
                        # ONE of them per run from here.
VEL_DYN_V_MAX=$DATA_V_RANGE
                        # Hard clamp on the predicted speed. The data is bounded
                        # by |v| <= DATA_V_RANGE by construction, so anything
                        # outside is definitely wrong and clamping it is free
                        # information. It is also the ONLY real divergence guard
                        # for VEL_DYN_ARCH=recurrence: the pole radius bounds
                        # each individual step, but the coefficients are
                        # recomputed every step from a state the rollout drives,
                        # so a 200-step open-loop rollout can still blow up
                        # (measured: 1e16 at one seed). Costs exact equivariance
                        # only when it BINDS, which on in-range data it should
                        # never do. Empty = no clamp.
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
VEL_DYN_OPENLOOP_K=10   # extra multi-step supervision: for the last K encoder
                        # steps, replay the head open-loop (fed its own
                        # predictions) against velocities already measured for
                        # those steps. Trains exactly the extrapolation regime
                        # the decoder runs in, at no image-rollout cost. 0 = off.
                        #
                        # Set to the ROLLOUT length (SEQ_LEN - INPUT_FRAMES).
                        # Without it the head is only ever trained ONE step ahead,
                        # always restarted from a real measurement, while at
                        # evaluation it has to run the whole rollout on its own
                        # output -- the mismatch that open-loop supervision exists
                        # to close. The standalone test
                        # (motiond_test_velocity_rnn.ipynb) measured it mattering:
                        # 'axis' end-of-rollout drift 11.2 -> 8.6 px, 'constant'
                        # 5.8 -> 1.9. Internally the fork cannot start before the
                        # second measurement, so K is effectively capped at
                        # INPUT_FRAMES-2.
TRACK_CORR_ALPHA=1.0    # Whitening of the TRACKING correlator, track(h, X_t).
                        # The bootstrap correlator is untouched and stays at 1.0.
                        #   1.0 = classic phase correlation -- what every run
                        #         before this one used.
                        #   0.0 = plain cross-correlation.
                        # Phase correlation divides every frequency by its own
                        # magnitude, so bands the template barely occupies get
                        # amplified to unit gain. That is exact for a sharp
                        # template and catastrophic for a smooth one -- and
                        # h.mean(dim=2), the tracking template, is smooth.
                        # Measured on a 2-digit 64x64 scene:
                        #
                        #   template            alpha=1   alpha=0
                        #   sharp digit           100%       99%
                        #   3x3 blur                0%       92%
                        #   5x5 blur + tanh         0%       65%
                        #
                        # MEASURED, and the reason the default is back at 1.0:
                        # alpha=0 is bimodal on a real hidden state -- exactly
                        # right more often, but with a heavy tail of wildly
                        # wrong estimates (median error 0 px, MEAN 5.3 px,
                        # versus alpha=1's median 2.8 / mean 3.2). For a warp,
                        # a consistently small error is far better than an
                        # occasionally huge one: a 3 px mistake blurs h, a
                        # 20 px mistake scrambles it. The alpha=0 run collapsed
                        # to predicting nothing by epoch 14. Leave this at 1.0.
                        #
                        # NOTE this cannot affect motion equivariance: |R| is
                        # invariant to a shift of either input, so alpha changes
                        # how reliably the peak is found, never where it is.
DECODER_SAMPLING_P=0.0  # Scheduled sampling on the decoder velocity: fraction of
DECODER_SAMPLING_RAMP=10 # training rollouts that use the head's own predicted
                        # velocity instead of the tracked measurement, ramped in
                        # over this many epochs.
                        #
                        # Why it is on: training otherwise ALWAYS gives the
                        # decoder an oracle velocity, so the ConvLSTM learns to
                        # depend on one. Measured on a fixed held-out set during
                        # training, oracle-velocity loss improved 6x while
                        # frozen-velocity loss on the SAME data got 43% WORSE --
                        # the model was actively learning to need the oracle,
                        # and had no gradient path that could teach it otherwise.
                        # 0 = off.
EVAL_VEL_MODE=all       # frozen    : honest inference only (cheapest)
                        # both      : + oracle GT-tracked val
                        # all       : + head-predicted val. Logs val_loss,
                        #             val_predicted_loss and val_tracked_loss
                        #             every epoch, so the three-way comparison
                        #             lands in ONE run. Expected ordering:
                        #             tracked < predicted < frozen.
                        #             Costs two extra val passes per epoch.
                        #
                        # NOTE the mode changes what the VELOCITY TABLE reports
                        # for decoder steps: under 'tracked' it is the measured
                        # velocity, under 'predicted' it is the head's own
                        # output. Those are different quantities and are not
                        # comparable across runs -- and exact-match accuracy is
                        # a poor metric for a continuous predictor anyway, so
                        # read mean_l2 for the predicted arm.
                        #
                        # 'frozen' should never DRIVE selection, because it is a
                        # SATURATED metric here: a model with PERFECT appearance
                        # but a frozen velocity scores 0.1574, and the trained
                        # model already scores 0.1561. It cannot improve, yet it
                        # was driving the LR scheduler, checkpoint selection and
                        # early stopping. val_predicted_loss reached 0.1179 over
                        # the same run and was still falling at epoch 50.

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
RECON=()
if [ "$FOURIER_LOSS" = 1 ]; then
  RECON=(--fourier_loss --shape_weight "$SHAPE_WEIGHT"
         --location_weight "$LOCATION_WEIGHT")
fi

COMMON=(
  "${MOTION[@]}"
  --num_digits "$NUM_DIGITS"
  "${RECON[@]}"
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

RUN_TAG="h${HIDDEN}_n${NUM_DIGITS}_${MOTION_TAG}_s${SEED}"   # motion is in the name so a sweep
                                           # gives distinguishable wandb runs
# the objective is part of the run's identity too -- two runs with different
# loss weights are not comparable and should not look alike in wandb
if [ "$FOURIER_LOSS" = 1 ] && [ "$SHAPE_WEIGHT" != 1.0 ]; then
  RUN_TAG="${RUN_TAG}_sh${SHAPE_WEIGHT}"
fi
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
    EXTRA=(--model melstm --num_vel_modes "$NUM_DIGITS")

    if [ "$USE_VEL_DYN" = 1 ]; then
      EXTRA+=(--use_velocity_dynamics
              --vel_dyn_state_dim "$VEL_DYN_STATE_DIM"
              --vel_dyn_gain "$VEL_DYN_GAIN"
              --vel_dyn_loss_weight "$VEL_DYN_LOSS_W"
              --vel_dyn_openloop_k "$VEL_DYN_OPENLOOP_K"
              --vel_dyn_arch "$VEL_DYN_ARCH"
              --vel_dyn_decoder_supervision "$VEL_DYN_DEC_SUP"
              --vel_dyn_layers "$VEL_DYN_LAYERS"
              --vel_dyn_v_max "$VEL_DYN_V_MAX"
              --decoder_sampling_p "$DECODER_SAMPLING_P"
              --decoder_sampling_ramp "$DECODER_SAMPLING_RAMP")
      ME_TAG="_vd${VEL_DYN_STATE_DIM}x${VEL_DYN_LAYERS}${VEL_DYN_ARCH:0:1}${VEL_DYN_GAIN:0:1}"
      if [ "$VEL_DYN_DEC_SUP" != none ]; then ME_TAG="${ME_TAG}_${VEL_DYN_DEC_SUP}"; fi
      if [ "$DECODER_SAMPLING_P" != 0 ] && [ "$DECODER_SAMPLING_P" != 0.0 ]; then
        ME_TAG="${ME_TAG}_ss${DECODER_SAMPLING_P}"
      fi
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

    # Applies to melstm regardless of the dynamics head: it is the measurement,
    # not the predictor. Tagged so a run's name says which correlator it used.
    EXTRA+=(--track_corr_alpha "$TRACK_CORR_ALPHA")
    if [ "$TRACK_CORR_ALPHA" != 1.0 ]; then
      ME_TAG="${ME_TAG}_a${TRACK_CORR_ALPHA}"
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
