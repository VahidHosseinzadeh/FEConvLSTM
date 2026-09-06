import argparse
import copy
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM
from moving_mnist_dataset import MovingMNISTDataset
from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset
from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from train_eval_utils import (train_epoch, eval_epoch, eval_len_generalization,
                             eval_velocity_generalization, ValCurveRecorder, build_model)
from fourier_loss import FourierShapePhaseLoss
from velocity_metrics import VelocityMetrics

class MSEPlusL1Loss(nn.Module):
    def __init__(self, mse_weight=1.0, l1_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight

    def forward(self, input_seq, target_seq):
        return self.mse_weight * self.mse(input_seq, target_seq) + self.l1_weight * self.l1(input_seq, target_seq)


class FourierPlusL1Loss(nn.Module):
    """
    MSEPlusL1 with the squared-error half split into its translation-invariant
    ("shape") and remaining ("location") parts, so the two can be weighted
    separately. See fourier_loss.py for the identity and its caveats.

    At shape_weight == location_weight == 1.0 this is EXACTLY MSEPlusL1, so
    switching it on changes nothing until a weight is moved off 1.
    """

    def __init__(self, shape_weight=1.0, location_weight=1.0, l1_weight=1.0):
        super().__init__()
        self.fourier = FourierShapePhaseLoss(shape_weight, location_weight)
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.last_parts = {}

    def forward(self, input_seq, target_seq):
        f, parts = self.fourier(input_seq, target_seq)
        self.last_parts = parts
        return f + self.l1_weight * self.l1(input_seq, target_seq)


def log_length_generalization_curve(gen_mean, gen_std, gen_pred_frames, args):
    steps = np.arange(1, gen_pred_frames + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, gen_mean, label="MSE")
    ax.fill_between(
        steps,
        (gen_mean - gen_std).clip(min=0),
        gen_mean + gen_std,
        alpha=0.3,
        label="± std",
    )
    ax.set_xlabel("Prediction horizon t")
    ax.set_ylabel("MSE")
    ax.set_title(f"Length generalization (seq_len = {args.gen_seq_len})")
    ax.legend()
    wandb.log({"len_gen_curve_image": wandb.Image(fig)})
    plt.close(fig)

    table_data = [
        [int(step), float(m), max(0.0, float(m - s)), float(m + s)]
        for step, m, s in zip(steps, gen_mean, gen_std)
    ]
    table = wandb.Table(
        data=table_data,
        columns=["prediction_horizon", "mse", "mse_min", "mse_max"],
    )
    wandb.log({"len_gen_curve_table": table})
    wandb.log({
        "len_gen_curve_plot": wandb.plot.line(
            table,
            x="prediction_horizon",
            y="mse",
            title="Length generalization",
        )
    })


def save_len_gen_results(path, gen_mean, gen_std, details, args, epoch):
    """
    Persist one run's length-generalization results as .npz so
    plot_len_gen_comparison.py can combine several runs (models) into one
    figure with paired per-sequence statistics and qualitative strips.
    """
    run = wandb.run
    payload = dict(
        mean=gen_mean,
        std=gen_std,
        model=args.model,
        hidden_size=args.hidden_size,
        gen_input_frames=args.gen_input_frames,
        gen_seq_len=args.gen_seq_len,
        train_input_frames=args.input_frames,
        train_pred_frames=args.seq_len - args.input_frames,
        epoch=epoch,
        run_id=str(run.id) if run is not None else "none",
    )
    # per_seq_err, per_seq_err_copy_last, strip_*, and (with motion data)
    # boundary_age all come straight from eval_len_generalization.
    payload.update(details)
    np.savez_compressed(path, **payload)
    print(f"Saved length-generalization results to {path}")


def save_vel_gen_results(path, vx, vy, err, args):
    """Persist the velocity-generalization error grid for offline comparison
    (plot_vel_gen_comparison.py)."""
    run = wandb.run
    np.savez_compressed(
        path,
        vx=vx, vy=vy, err=err,
        model=args.model,
        hidden_size=args.hidden_size,
        data_v_range=args.data_v_range,
        run_id=str(run.id) if run is not None else "none",
    )
    print(f"Saved velocity-generalization results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate RNN models on Moving MNIST")
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--gen_input_frames', type=int, default=15, help='Encoder input frame count used **only** for the length-generalization experiment (separate from --input_frames, which is used for train/val/test)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes (0 = load on the main process, no parallelism)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_epochs', type=int, default=50, help='Minimum number of epochs before early stopping can trigger (no effect unless --early_stop_patience > 0)')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop once --min_epochs has passed and the selection val loss has not '
                             'improved for this many consecutive epochs (0 = disabled, always run to '
                             '--epochs). Lets each model in a comparison train to its own convergence '
                             'instead of everyone sharing one epoch count sized for the cheapest model.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Enable ReduceLROnPlateau: cuts LR when val_loss stops improving')
    parser.add_argument('--lr_patience', type=int, default=5, help='Epochs with no val_loss improvement before cutting LR (only with --use_lr_scheduler)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Multiplicative LR cut factor (only with --use_lr_scheduler)')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Floor below which LR will not be reduced further (only with --use_lr_scheduler)')
    parser.add_argument('--model', choices=['lstm', 'felstm', 'melstm'], default='felstm')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--decoder_hidden_size', type=int, default=None,
                        help='Width of the decoder conv stack, independent of --hidden_size (which sets the '
                             'recurrent cell width). Default: None = same as --hidden_size (previous behavior). '
                             'Much cheaper to raise than --hidden_size: the recurrent state is carried per '
                             'velocity slot/channel at every timestep and kept for BPTT, while the decoder runs '
                             'once per predicted frame on the already-pooled map.')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--decoder_conv_layers', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use (default: use all)')
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--v_range', type=int, default=2)
    parser.add_argument('--num_vel_modes', type=int, default=2, help='Number of velocity modes for MEConvLSTM')
    parser.add_argument('--track_corr_alpha', type=float, default=None,
                        help="MELSTM only: whitening exponent of the TRACKING correlator "
                             "(track(h, X_t)), leaving the bootstrap correlator untouched. "
                             "1.0 = classic phase correlation (the default, and what every "
                             "previous run used); 0.0 = plain cross-correlation. Phase "
                             "correlation divides each frequency by its own magnitude, so a "
                             "SMOOTH template -- which h.mean(dim=2) is -- has its empty "
                             "high-frequency bands amplified to unit gain and the peak "
                             "collapses. Measured on a 2-digit 64x64 scene with a blurred, "
                             "tanh-squashed template: 0%% at alpha=1 versus 65%% at alpha=0, "
                             "while a sharp template scores 100%% either way. Default None "
                             "= follow the bootstrap correlator, i.e. unchanged.")
    parser.add_argument('--data_v_range', type=int, default=2)
    parser.add_argument('--motion_mode', choices=['constant', 'piecewise', 'stochastic', 'accelerate', 'harmonic'], default='piecewise',
                        help="How digit velocity evolves over time: 'constant' = fixed for the whole "
                             "sequence; 'piecewise' = held for a random --min_segment..--max_segment window "
                             "then changes (--p_change has no effect in this mode); 'stochastic' = --p_change "
                             "chance of changing at every single step; 'accelerate' = every "
                             "--min_segment..--max_segment steps the speed is incremented by a per-digit "
                             "constant sign and clipped to --data_v_range, so |v| ramps systematically "
                             "instead of random-walking (--transition_mode, --smooth_probability and "
                             "--p_change are all unused in this mode); 'harmonic' = constant drift plus "
                             "a sinusoid per axis, with amplitude/period/phase drawn per digit and the "
                             "trajectory then deterministic. This is the only mode whose future velocity "
                             "is predictable from the velocity history alone, and therefore the only one "
                             "where --use_velocity_dynamics can beat a frozen decoder. See "
                             "--harmonic_shapes.")
    parser.add_argument('--transition_mode', choices=['uniform', 'smooth'], default='smooth',
                        help="What the new velocity is when a change happens: 'uniform' = uniformly random "
                             "pick from the whole velocity grid; 'smooth' = with probability "
                             "--smooth_probability, step to a neighboring velocity (vx and vy each change by "
                             "at most 1) instead of an unrelated jump.")
    parser.add_argument('--min_segment', type=int, default=3, help='motion_mode=piecewise/accelerate: minimum frames before a velocity change')
    parser.add_argument('--max_segment', type=int, default=6, help='motion_mode=piecewise/accelerate: maximum frames before a velocity change')
    parser.add_argument('--p_change', type=float, default=0.25, help='motion_mode=stochastic only: per-step probability of a velocity change (no effect in piecewise mode)')
    parser.add_argument('--smooth_probability', type=float, default=0.8, help='transition_mode=smooth only: probability a change is a neighbor step rather than a uniform jump')
    parser.add_argument('--freeze_after', type=int, default=-1,
                        help="Dataset velocity freezing (TDMovingMNISTDataset.freeze_after). "
                             "-1 (default) = freeze at the context/prediction boundary "
                             "(--input_frames for train/val/test, --gen_input_frames for "
                             "length-gen) — the historical behavior, under which a decoder that "
                             "freezes its last encoder velocity matches ground truth exactly. "
                             "0 = no freezing: the velocity keeps evolving through the rollout, "
                             "which is the only regime where predicting the velocity can beat "
                             "freezing it (see --use_velocity_dynamics). >0 = freeze at that "
                             "explicit step for every split.")
    # ---- motion_mode=harmonic only ----
    parser.add_argument('--harmonic_shapes', nargs='+',
                        choices=['constant', 'orbit', 'axis', 'lissajous'],
                        default=['constant', 'orbit', 'axis', 'lissajous'],
                        help="motion_mode=harmonic: which velocity shapes to mix, drawn uniformly per "
                             "digit. 'constant' = pure constant flow (the degenerate member; freezing "
                             "the last velocity is already optimal for it); 'orbit' = both axes share "
                             "one amplitude and period a quarter cycle apart, so the velocity rotates "
                             "and the digit circles; 'axis' = constant flow along one axis, sinusoidal "
                             "along the other; 'lissajous' = both axes oscillate with independent "
                             "amplitude, period and phase. Pass a single shape to isolate it.")
    parser.add_argument('--harmonic_period_min', type=int, default=12,
                        help='motion_mode=harmonic: shortest velocity period, in frames. Periods much '
                             'longer than --input_frames are not identifiable from the context, so the '
                             'dynamics head cannot beat freezing there -- keep this range comparable to '
                             'the context length.')
    parser.add_argument('--harmonic_period_max', type=int, default=30,
                        help='motion_mode=harmonic: longest velocity period, in frames')
    parser.add_argument('--harmonic_amp_min', type=float, default=0.6,
                        help='motion_mode=harmonic: minimum oscillation amplitude, as a fraction of '
                             '--data_v_range. Velocities are integers, so a small amplitude is destroyed '
                             'by rounding -- below ~0.5 the sinusoid stops being resolvable.')
    parser.add_argument('--harmonic_amp_max', type=float, default=1.0,
                        help='motion_mode=harmonic: maximum oscillation amplitude, as a fraction of '
                             '--data_v_range')
    parser.add_argument('--no_harmonic_drift', action='store_true',
                        help='motion_mode=harmonic: drop the constant velocity offset, so oscillations '
                             'are centred on zero. The drift is INTEGER and therefore cancels exactly in '
                             'the velocity differences, which is the only thing an equivariant dynamics '
                             'head sees -- so it costs that head nothing while making the paths looping '
                             'trochoids. Turn it off only to isolate the pure oscillation.')
    parser.add_argument('--gen_seq_len', type=int, default=40, help='Sequence length used **only** for length‑generalization evaluation (must be > seq_len)')
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for dataset splitting')
    parser.add_argument('--model_seed', type=int, default=None, help='Random seed for model initialization (default: random)')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the run')
    parser.add_argument('--model_save_dir', type=str, default='./fernn/movmnist/', help='Directory to save model checkpoints')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model checkpoint to load for evaluation')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate the loaded model without training')
    parser.add_argument('--gen_vel_min', type=int, default=-2, help='Minimum integer pixel velocity evaluated along each axis')
    parser.add_argument('--gen_vel_max', type=int, default= 2, help='Maximum integer pixel velocity evaluated along each axis')
    parser.add_argument('--gen_vel_step', type=int, default= 1, help='Step size (integer) for the velocity grid')
    parser.add_argument('--gen_vel_n_seq', type=int, default=128, help='How many sequences to sample **per (vx,vy)** for velocitygeneralisation test')
    parser.add_argument('--run_velocity_generalization', action='store_true', help='Run velocity generalization test (default: False)')
    # ---- reconstruction loss shape/location split (see fourier_loss.py) ----
    parser.add_argument('--fourier_loss', action='store_true',
                        help="Split the squared-error half of the reconstruction loss into a "
                             "translation-invariant 'shape' term and a 'location' term, weighted "
                             "separately. Motivation: plain MSE on sparse frames has a degenerate "
                             "minimum at 'predict nothing' -- a digit displaced by about half its "
                             "own width already costs exactly as much as a blank frame, and more "
                             "beyond that, so a rollout with imperfect velocity is rewarded for "
                             "going silent. The shape term prices blankness directly (a blank "
                             "prediction has zero magnitude at every frequency). At weights "
                             "(1, 1) this is EXACTLY the current MSE+L1, so it is safe to leave on.")
    parser.add_argument('--shape_weight', type=float, default=1.0,
                        help='--fourier_loss only: weight on the translation-invariant term. '
                             'Raise above 1 to punish blur and blankness harder.')
    parser.add_argument('--location_weight', type=float, default=1.0,
                        help='--fourier_loss only: weight on the location term, which is where a '
                             'pure translation puts ALL of its error.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (default: 1.0, None for no clipping)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default="FERNN", help='Wandb project')
    parser.add_argument('--wandb_dir', type=str, default='./tmp/', help='Wandb directory')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb name')
    parser.add_argument("--check_velocity_predictor",action="store_true",help="MELSTM only: debug the velocity predictor during training/evaluation.")
    parser.add_argument("--show_h_state", action="store_true",
                         help="FELSTM only: log the per-(vx,vy) candidate h-slot maps "
                              "(filtered to velocities actually observed in the input "
                              "window) to wandb. Independent of --check_velocity_predictor, "
                              "which is MELSTM's velocity-tracker debug flag.")
    parser.add_argument('--val_curve_interval', type=int, default=25, help='Record validation loss every N training batches for the loss-vs-steps curve (0 = off)')
    parser.add_argument('--val_curve_size', type=int, default=256, help='Number of fixed validation sequences used for the loss-vs-steps curve')
    parser.add_argument('--eval_velocity_mode',
                        choices=['frozen', 'tracked', 'both', 'predicted', 'all'], default='frozen',
                        help="MELSTM decoder velocity at evaluation: 'frozen' = honest inference (default, drives scheduler/selection); "
                             "'tracked' = oracle GT-tracked eval drives scheduler/selection; "
                             "'both' = select on frozen but also log val_tracked_loss each epoch; "
                             "'predicted' = velocity dynamics head rolls the velocity forward "
                             "(deployable, requires --use_velocity_dynamics) and drives selection; "
                             "'all' = select on frozen but log all three (val_loss, "
                             "val_predicted_loss, val_tracked_loss) so the three-way comparison "
                             "lands in one run")
    # ---- velocity dynamics head (process model for u) ----
    # All off by default: an existing command line produces exactly the model,
    # objective and metrics it produced before this feature existed.
    parser.add_argument('--use_velocity_dynamics', action='store_true',
                        help='MELSTM only: add the equivariant velocity dynamics head — a small '
                             'GRU over the velocity history that predicts u_{t+1}, so the decoder '
                             'can extrapolate the motion instead of freezing the last encoder '
                             'velocity. Zero-initialized output, so at init it predicts exactly '
                             'the frozen velocity.')
    parser.add_argument('--vel_dyn_state_dim', type=int, default=32,
                        help='GRU hidden size of the velocity dynamics head (per slot)')
    parser.add_argument('--vel_dyn_use_h', action='store_true',
                        help='Condition the dynamics head on a globally pooled (hence '
                             'translation-invariant) readout of the hidden state, not just on the '
                             'velocity history. Ablation, not a default: it lets the head explain '
                             'velocity by appearance and stop extrapolating.')
    parser.add_argument('--vel_dyn_gain', choices=['fixed', 'learned'], default='fixed',
                        help="How the prediction and the phase-correlation measurement are "
                             "combined in the encoder: 'fixed' (default) = k=1, the measurement is "
                             "taken verbatim and the head is trained but does not influence the "
                             "encoder; 'learned' = k=sigmoid(MLP(correlation peak score)), so a "
                             "weak peak defers to the process model.")
    parser.add_argument('--vel_dyn_loss_weight', type=float, default=0.0,
                        help="Weight on the dynamics head's one-step-ahead loss "
                             "smooth_l1(u_pred, v_measured.detach()). 0.0 (default) means the head "
                             "never trains — set it (e.g. 1.0) whenever --use_velocity_dynamics is on.")
    parser.add_argument('--vel_dyn_arch', choices=['gru', 'recurrence'], default='gru',
                        help="Structure of the dynamics head. 'gru' emits the velocity increment "
                             "directly. 'recurrence' emits the COEFFICIENTS of a stable "
                             "second-order linear recurrence du_{t+1} = a1 du_t + a2 du_{t-1}, "
                             "parametrised by a pole radius rho<1 and angle so an open-loop "
                             "rollout provably cannot diverge. A sinusoidal velocity satisfies "
                             "that recurrence exactly, so this splits identification (hard, "
                             "learned) from propagation (exact) instead of asking one GRU to do "
                             "both -- which is what makes a plain GRU degrade into a one-step-"
                             "LAGGED estimator. Both nest the frozen rollout exactly at init.")
    parser.add_argument('--vel_dyn_v_max', type=float, default=None,
                        help='Hard clamp on |predicted velocity| per component. The data is '
                             'bounded by --data_v_range by construction, so that is the natural '
                             'value and anything outside it is definitely wrong. This is the only '
                             'real divergence guard for --vel_dyn_arch recurrence (the pole radius '
                             'bounds each step, but the coefficients are re-estimated every step '
                             'from a state the rollout drives, so a long open-loop rollout can '
                             'still blow up). Costs exact equivariance only when it binds.')
    parser.add_argument('--vel_dyn_layers', type=int, default=1,
                        help='Number of stacked GRU layers in the dynamics head. 1 -> 3.5k '
                             'params, 2 -> 9.9k at state_dim 32.')
    parser.add_argument('--decoder_sampling_p', type=float, default=0.0,
                        help="Scheduled sampling on the decoder VELOCITY: probability that a "
                             "training rollout uses the head's own predicted velocity instead of "
                             "the tracked measurement. Training otherwise always hands the "
                             "decoder an oracle velocity, so the ConvLSTM learns to depend on one "
                             "and never gets a gradient teaching it to cope without -- measured "
                             "on a fixed set, oracle-velocity loss improved 6x while "
                             "frozen-velocity loss on the SAME data got 43%% worse. Ramped from 0 "
                             "over --decoder_sampling_ramp epochs. 0 = off.")
    parser.add_argument('--decoder_sampling_ramp', type=int, default=10,
                        help='Epochs over which --decoder_sampling_p ramps linearly from 0 to its '
                             'full value. Ramping matters: early on the head is untrained, so '
                             'sampling its output immediately would feed the ConvLSTM noise.')
    parser.add_argument('--vel_dyn_openloop_k', type=int, default=0,
                        help='Extra multi-step velocity supervision: for the last k encoder steps, '
                             'replay the head open-loop (fed its own predictions, no measurements) '
                             'against the velocities already measured for those steps. Trains the '
                             'extrapolation regime directly, with no image rollout. 0 = off.')
    parser.add_argument('--len_gen_every', type=int, default=0,
                        help='Also run length-generalization every N epochs regardless of val improvement '
                             '(0 = only on new best val). Saves to a separate *_latest.npz')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint_*.pth written during a previous run: restores model, '
                             'optimizer, scheduler, epoch, best-val, loss curves, and continues the same '
                             'wandb run. (Unlike --load_model, which loads best weights only.)')
    args = parser.parse_args()

    # --check_velocity_predictor only makes the datasets return motion
    # (return_motion=...): the shared train/eval functions detect it in the
    # batch and add velocity metrics / state visualizations on top of the
    # identical training procedure.
    train_fn = train_epoch
    eval_fn = eval_epoch
    len_gen_fn = eval_len_generalization


    # Set seeds for reproducibility
    # Data seed for consistent dataset splits
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    # Output layout under --model_save_dir:
    #   models/     the actual trained weights (*_best.pth, *_best_model_<id>.pth)
    #   results/    plot-script inputs (history_*.json, len_gen_*.npz, vel_gen_*.npz)
    #   run_state/  internal recovery machinery, not meant to be opened by hand
    #               (checkpoint_*.pth, DONE_*.flag, .resubmit_count_*)
    models_dir = os.path.join(args.model_save_dir, "models")
    results_dir = os.path.join(args.model_save_dir, "results")
    run_state_dir = os.path.join(args.model_save_dir, "run_state")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(run_state_dir, exist_ok=True)

    # Load the resume checkpoint before wandb.init so the original run
    # (and its id-based file names) can be continued.
    resume_ckpt = None
    if args.resume is not None:
        print(f"Loading resume checkpoint from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location='cpu')

    # Initialize wandb
    wandb.init(entity=args.wandb_entity,
               project=args.wandb_project,
               dir=args.wandb_dir,
               config=vars(args),
               name=args.wandb_name,
               id=resume_ckpt["run_id"] if resume_ckpt else None,
               resume="allow" if resume_ckpt else None)

    if args.use_velocity_dynamics and args.model != 'melstm':
        print(f"WARNING: --use_velocity_dynamics has no effect for --model {args.model} "
              f"(MELSTM only); the flag is ignored.")
    if args.use_velocity_dynamics and args.vel_dyn_loss_weight == 0.0:
        print("WARNING: --use_velocity_dynamics is on but --vel_dyn_loss_weight is 0, so the "
              "dynamics head never trains and predicts exactly the frozen velocity forever. "
              "Set --vel_dyn_loss_weight (e.g. 1.0).")
    if args.eval_velocity_mode in ('predicted', 'all') and not args.use_velocity_dynamics:
        print(f"WARNING: --eval_velocity_mode {args.eval_velocity_mode} needs "
              f"--use_velocity_dynamics; without it the 'predicted' evaluation degrades to "
              f"'frozen' and will report the same number as val_loss.")

    assert args.input_frames < args.seq_len, "input_frames must be less than seq_len"
    assert args.gen_input_frames < args.gen_seq_len, "gen_input_frames must be less than gen_seq_len"
    pred_frames = args.seq_len - args.input_frames
    gen_pred_frames = args.gen_seq_len - args.gen_input_frames

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset velocity freezing. Historically this was hardcoded to the
    # context/prediction boundary at all three construction sites, which makes
    # "freeze the last encoder velocity" the exactly-correct decoder policy and
    # therefore leaves a velocity *predictor* no headroom at all. --freeze_after
    # is what lets that be turned off for the experiments where predicting the
    # velocity can actually win. None = TDMovingMNISTDataset's "do not freeze".
    if args.freeze_after < 0:
        freeze_after, gen_freeze_after = args.input_frames, args.gen_input_frames
    elif args.freeze_after == 0:
        freeze_after = gen_freeze_after = None
    else:
        freeze_after = gen_freeze_after = args.freeze_after
    print(f"Dataset freeze_after: train/val/test={freeze_after}, len_gen={gen_freeze_after}")

    # motion_mode=harmonic knobs, shared by all three dataset constructions.
    # Inert for every other mode (the dataset only reads them in the harmonic
    # branch), so they can be passed unconditionally.
    harmonic_kwargs = dict(
        harmonic_shapes=tuple(args.harmonic_shapes),
        harmonic_period=(args.harmonic_period_min, args.harmonic_period_max),
        harmonic_amp=(args.harmonic_amp_min, args.harmonic_amp_max),
        harmonic_drift=not args.no_harmonic_drift,
    )
    if args.motion_mode == 'harmonic':
        print(f"Harmonic motion: shapes={args.harmonic_shapes} "
              f"period={harmonic_kwargs['harmonic_period']} frames "
              f"amp={harmonic_kwargs['harmonic_amp']} x {args.data_v_range} px/frame "
              f"drift={'on' if harmonic_kwargs['harmonic_drift'] else 'off'}")

    

    # # Dataset and splits
    # train_dataset = MovingMNISTDataset(
    #     root=args.root,
    #     train=True,
    #     seq_len=args.seq_len,
    #     image_size=args.image_size,
    #     velocity_range_x=(-args.data_v_range,args.data_v_range),
    #     velocity_range_y=(-args.data_v_range,args.data_v_range),
    #     num_digits=2
    # )
    
    # test_dataset = MovingMNISTDataset(
    #     root=args.root,
    #     train=False,
    #     seq_len=args.seq_len,
    #     image_size=args.image_size,
    #     velocity_range_x=(-args.data_v_range,args.data_v_range),
    #     velocity_range_y=(-args.data_v_range,args.data_v_range),
    #     num_digits=2
    # )

    # gen_test_dataset = MovingMNISTDataset(
    #         root=args.root,
    #         train=False,
    #         seq_len=args.gen_seq_len,
    #         image_size=args.image_size,
    #         velocity_range_x=(-args.data_v_range, args.data_v_range),
    #         velocity_range_y=(-args.data_v_range, args.data_v_range),
    #         num_digits=2,
    #         random=False
    # )

    train_dataset = TDMovingMNISTDataset(
        root=args.root,
        train=True,
        seq_len=args.seq_len,
        num_digits=2,
        image_size=args.image_size,
        max_speed=args.data_v_range,

        motion_mode=args.motion_mode,
        transition_mode=args.transition_mode,

        min_segment=args.min_segment,
        max_segment=args.max_segment,

        p_change=args.p_change,
        smooth_probability=args.smooth_probability,

        motion_difficulty=None,
        freeze_after=freeze_after,
        **harmonic_kwargs,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,

        return_motion=args.check_velocity_predictor or args.show_h_state,
        return_positions=False,

        transform=None,
        download=True,

        random=True,
        seed=42,

        max_tries=300,
    )

    test_dataset = TDMovingMNISTDataset(
        root=args.root,
        train=False,
        seq_len=args.seq_len,
        num_digits=2,
        image_size=args.image_size,
        max_speed=args.data_v_range,

        motion_mode=args.motion_mode,
        transition_mode=args.transition_mode,

        min_segment=args.min_segment,
        max_segment=args.max_segment,

        p_change=args.p_change,
        smooth_probability=args.smooth_probability,

        motion_difficulty=None,
        freeze_after=freeze_after,
        **harmonic_kwargs,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,

        return_motion=args.check_velocity_predictor or args.show_h_state,
        return_positions=False,

        transform=None,
        download=True,

        random=False,  # Fixed benchmark set (same idea as gen_test_dataset)
        seed=123,      # distinct seed so test and len-gen draw different streams

        max_tries=300,
    )

    gen_test_dataset = TDMovingMNISTDataset(
        root=args.root,
        train=False,
        seq_len=args.gen_seq_len,
        num_digits=2,
        image_size=args.image_size,
        max_speed=args.data_v_range,

        motion_mode=args.motion_mode,
        transition_mode=args.transition_mode,

        min_segment=args.min_segment,
        max_segment=args.max_segment,

        p_change=args.p_change,
        smooth_probability=args.smooth_probability,

        motion_difficulty=None,
        **harmonic_kwargs,

        freeze_after=gen_freeze_after,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,

        return_motion=args.check_velocity_predictor or args.show_h_state,
        return_positions=False,

        transform=None,
        download=True,

        random=False, 
        seed=42,
        max_tries=300
    )

    # Split training data into train and validation
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(args.data_seed))
    
    # Limit training samples if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.data_seed))[:args.max_train_samples]
        train_ds = Subset(train_ds, indices)
        print(f"Limited training to {args.max_train_samples} samples")
    
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_kwargs)
    # persistent_workers=False on purpose for the two fixed benchmark sets:
    # their seeded RNGs are stateful, and persistent workers would carry
    # advanced RNG state into the next evaluation (different sequences per
    # call). Non-persistent workers re-fork from the parent dataset — whose
    # RNG is reset via reset_rng() before each evaluation — so every pass
    # sees the identical set.
    fixed_loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, **fixed_loader_kwargs)
    gen_test_loader = DataLoader(gen_test_dataset,
                                batch_size=args.batch_size, **fixed_loader_kwargs)


    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_dataset)}")
    print(f"Length‑gen test size: {len(gen_test_dataset)} | Eval sequence length: {args.gen_seq_len}")
    print(f"Input frames: {args.input_frames}, Pred frames: {pred_frames}")
    print(f"Gen input frames: {args.gen_input_frames}, Gen pred frames: {gen_pred_frames} (gen_seq_len={args.gen_seq_len})")
    print(f"Model seed: {args.model_seed}")
    print(f"Data seed: {args.data_seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Min epochs: {args.min_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"LR scheduler: {'ReduceLROnPlateau' if args.use_lr_scheduler else 'none'}")
    if args.use_lr_scheduler:
        print(f"  patience={args.lr_patience}  factor={args.lr_factor}  min_lr={args.lr_min}")
    print(f"Model: {args.model}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num layers: {args.num_layers}")
    print(f"Decoder conv layers: {args.decoder_conv_layers}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Max train samples: {args.max_train_samples}")
    print(f"Image size: {args.image_size}")
    print(f"Velocity range: {args.v_range}")
    v_list = [(x, y) for x in range(-args.v_range, args.v_range + 1) for y in range(-args.v_range, args.v_range + 1)]
    num_v = len(v_list)
    print(f"Num v: {num_v}")

    # Set model initialization seed
    if args.model_seed is None:
        args.model_seed = int(time.time()) % 10000  # Use current time as seed if not provided
        wandb.config.update({"model_seed": args.model_seed}, allow_val_change=True)
    
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    random.seed(args.model_seed)
    
    # Single construction path shared with motion_generalization.py, so an
    # offline evaluation cannot rebuild a different architecture than was trained.
    model = build_model(args).to(device)

    # Parameter count: printed per top-level submodule and stored in the
    # wandb config so runs of different models are directly comparable.
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters ({args.model}): {num_params:,} total, {num_trainable:,} trainable")
    for name, module in model.named_children():
        sub = sum(p.numel() for p in module.parameters())
        if sub > 0:
            print(f"  {name:<20s} {sub:>12,}")
    wandb.config.update(
        {"num_parameters": num_params, "num_trainable_parameters": num_trainable},
        allow_val_change=True,
    )

    # Load model if specified
    if args.load_model is not None:
        print(f"Loading model from {args.load_model}")
        try:
            model.load_state_dict(torch.load(args.load_model))
            print(f"Successfully loaded model weights.")
        except Exception as e:
            print(f"Failed to load model weights {e}")

    # If evaluate_only is True, skip training and only run evaluation
    if args.evaluate_only:
        print("Running evaluation only...")
        criterion = (FourierPlusL1Loss(args.shape_weight, args.location_weight)
                     if args.fourier_loss else MSEPlusL1Loss())
        test_dataset.reset_rng()       # identical benchmark set every evaluation
        test_loss = eval_fn(model, test_loader, criterion, device, args.input_frames, 0, split_name="test",
                            show_h_state=args.show_h_state)
        print(f"Test Loss: {test_loss:.4f}")

        wandb.log({
            f"test_loss": test_loss,
            "epoch": 0
        })

        print("Running length generalization test...")
        gen_test_dataset.reset_rng()   # identical benchmark set every evaluation
        gen_mean, gen_std, gen_details = len_gen_fn(model, gen_test_loader, device, args.gen_input_frames, subsample_t=2,
                                                    show_h_state=args.show_h_state)
        print(f"Length generalization mean MSE: {gen_mean.mean():.4f}")
        save_len_gen_results(
            os.path.join(results_dir, f"len_gen_{args.model}_{wandb.run.id}.npz"),
            gen_mean, gen_std, gen_details, args, epoch=0,
        )

        wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
        wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
        wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})
        log_length_generalization_curve(gen_mean, gen_std, gen_pred_frames, args)

        if args.run_velocity_generalization:
            print("Running velocity generalization test...")
            vx, vy, err = eval_velocity_generalization(model, device, args)
            save_vel_gen_results(
                os.path.join(results_dir, f"vel_gen_{args.model}_{wandb.run.id}.npz"),
                vx, vy, err, args,
            )
            print("Velocity generalization results:")
            for i, vx_i in enumerate(vx):
                for j, vy_j in enumerate(vy):
                    print(f"Velocity (vx={vx_i}, vy={vy_j}): MSE {err[j, i]:.4f}")
            
            wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                    for i, vx_i in enumerate(vx)
                    for j, vy_j in enumerate(vy)})

            wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                    for i, vx_i in enumerate(vx)
                    for j, vy_j in enumerate(vy)})

            fig, ax = plt.subplots()
            im = ax.imshow(err[::-1],  # flip y so +vy is up
                        extent=[vx[0]-0.5, vx[-1]+0.5, vy[0]-0.5, vy[-1]+0.5],
                        origin='lower')
            ax.set_xlabel('$v_x$  (pixels / frame)')
            ax.set_ylabel('$v_y$')
            ax.set_title('Velocity-generalisation MSE')
            fig.colorbar(im, ax=ax)
            wandb.log({"vel_gen_heatmap": wandb.Image(fig)})
            plt.close(fig)

        return

    # Optimizers & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = (FourierPlusL1Loss(args.shape_weight, args.location_weight)
                 if args.fourier_loss else MSEPlusL1Loss())
    if args.fourier_loss:
        print(f"Reconstruction loss: fourier split, shape x{args.shape_weight} "
              f"location x{args.location_weight} (+ L1). "
              f"{'== plain MSE+L1' if args.shape_weight == args.location_weight == 1.0 else ''}")

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min,
        )

    # Fine-grained loss-vs-training-batches curves (fixed val subset,
    # materialized once — see ValCurveRecorder)
    curve_recorder = None
    if args.val_curve_interval > 0:
        print(f"Materializing {args.val_curve_size} fixed val sequences for the loss curve...")
        curve_recorder = ValCurveRecorder(
            val_ds, args.val_curve_size, args.val_curve_interval,
            args.input_frames, device,
        )

    history_path = os.path.join(results_dir, f"history_{args.model}_{wandb.run.id}.json")

    def dump_history(history):
        payload = {
            "config": vars(args),
            "history": history,
            "num_parameters": num_params,
            "num_trainable_parameters": num_trainable,
        }
        if curve_recorder is not None:
            payload.update(curve_recorder.as_dict())
        with open(history_path, "w") as f:
            json.dump(payload, f, indent=2)

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'test_loss': [], 'epoch_time_sec': []}
    best_val_losses = float('inf')
    epochs_since_improve = 0
    start_epoch = 1
    checkpoint_path = os.path.join(run_state_dir, f"checkpoint_{args.model}_{wandb.run.id}.pth")

    if resume_ckpt is not None:
        # A checkpoint from a run with different settings (hidden_size,
        # decoder_conv_layers, ...) has incompatible tensor shapes.
        # load_state_dict copies matching-shape params in place BEFORE
        # raising on the mismatched ones, so on failure the model can be
        # left partially mutated -- snapshot first and restore on error,
        # so an incompatible resume checkpoint can never crash the job or
        # silently corrupt a fresh model. It just starts fresh instead,
        # loudly, so you notice and can clean up the stale checkpoint.
        fresh_model_state = copy.deepcopy(model.state_dict())
        try:
            model.load_state_dict(resume_ckpt["model_state"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            if scheduler is not None and resume_ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            best_val_losses = resume_ckpt["best_val_losses"]
            epochs_since_improve = resume_ckpt.get("epochs_since_improve", 0)
            history = resume_ckpt["history"]
            if curve_recorder is not None and resume_ckpt.get("curve_recorder") is not None:
                curve_recorder.load_state_dict(resume_ckpt["curve_recorder"])
            start_epoch = resume_ckpt["epoch"] + 1
            print(f"Resumed from epoch {resume_ckpt['epoch']} "
                  f"(continuing at {start_epoch}) | best val so far: {best_val_losses:.4f}")
        except RuntimeError as e:
            model.load_state_dict(fresh_model_state)   # undo any partial copy
            epochs_since_improve = 0
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            if args.use_lr_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=args.lr_factor,
                    patience=args.lr_patience, min_lr=args.lr_min,
                )
            print(f"WARNING: --resume checkpoint at {args.resume} is incompatible "
                  f"with the current settings (hidden_size/decoder_conv_layers/...) "
                  f"and was ignored -- starting fresh instead:\n{e}")
            print(f"Delete the stale file to silence this: rm {args.resume}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        # Linear ramp: the head is useless at epoch 1, so sampling its output
        # then would just feed the ConvLSTM noise.
        samp_p = (args.decoder_sampling_p * min(1.0, epoch / max(1, args.decoder_sampling_ramp))
                  if args.decoder_sampling_p > 0 else 0.0)
        train_loss, dyn_loss = train_fn(model, train_loader, optimizer, criterion, device, args.input_frames,
                                        args.grad_clip, curve_recorder=curve_recorder,
                                        show_h_state=args.show_h_state,
                                        vel_dyn_loss_weight=args.vel_dyn_loss_weight,
                                        decoder_sampling_p=samp_p)
        epoch_time = time.time() - epoch_start
        val_loss = eval_fn(model, val_loader, criterion, device, args.input_frames, epoch, split_name="val",
                           show_h_state=args.show_h_state)

        # Oracle (GT-tracked decoder velocity) validation: measures the model
        # without velocity-estimation drift; the gap to val_loss isolates
        # velocity error from rendering error. MELSTM-only effect.
        val_tracked_loss = None
        if args.eval_velocity_mode in ('tracked', 'both', 'all'):
            val_tracked_loss = eval_fn(model, val_loader, criterion, device, args.input_frames, epoch,
                                       split_name="val_tracked", decoder_velocity_mode="tracked",
                                       show_h_state=args.show_h_state)

        # Deployable-but-extrapolating validation: the dynamics head rolls the
        # velocity forward with no measurement. Sits between val_loss (frozen,
        # deployable) and val_tracked_loss (oracle) -- 'all' logs the three
        # together so the whole comparison comes out of one run.
        val_predicted_loss = None
        if args.eval_velocity_mode in ('predicted', 'all'):
            val_predicted_loss = eval_fn(model, val_loader, criterion, device, args.input_frames, epoch,
                                         split_name="val_predicted", decoder_velocity_mode="predicted",
                                         show_h_state=args.show_h_state)

        # Metric driving the scheduler / best-model selection / len-gen gating.
        # 'both' and 'all' deliberately keep selecting on the honest frozen
        # metric: they add reporting, they do not change what is optimized for.
        if args.eval_velocity_mode == 'tracked':
            selection_val = val_tracked_loss
        elif args.eval_velocity_mode == 'predicted':
            selection_val = val_predicted_loss
        else:
            selection_val = val_loss

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time_sec'].append(epoch_time)
        if val_predicted_loss is not None:
            history.setdefault('val_predicted_loss', []).append(val_predicted_loss)
        if val_tracked_loss is not None:
            history.setdefault('val_tracked_loss', []).append(val_tracked_loss)
        if dyn_loss is not None:
            history.setdefault('dyn_loss', []).append(dyn_loss)
        extra_str = ""
        if val_predicted_loss is not None:
            extra_str += f" | Val (predicted): {val_predicted_loss:.4f}"
        if val_tracked_loss is not None:
            extra_str += f" | Val (tracked): {val_tracked_loss:.4f}"
        if dyn_loss is not None:
            extra_str += f" | Dyn: {dyn_loss:.4f}"
        print(f"Epoch {epoch}/{args.epochs} | {args.model:^8} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{extra_str} | {epoch_time:.1f}s")

        if scheduler is not None:
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(selection_val)
            lr_after = optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"  LR reduced: {lr_before:.2e} -> {lr_after:.2e}")

        # Log metrics to wandb
        log_payload = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch_time_sec": epoch_time,
            "epoch": epoch
        }
        if val_tracked_loss is not None:
            log_payload["val_tracked_loss"] = val_tracked_loss
        if val_predicted_loss is not None:
            log_payload["val_predicted_loss"] = val_predicted_loss
        if dyn_loss is not None:
            log_payload["dyn_loss"] = dyn_loss
        if args.decoder_sampling_p > 0:
            log_payload["decoder_sampling_p"] = samp_p
        wandb.log(log_payload)

        # Persist curves incrementally so a killed run still leaves data
        dump_history(history)
        
        # Save model if it's the best so far
        ran_len_gen_this_epoch = False
        if selection_val < best_val_losses:
            best_val_losses = selection_val
            epochs_since_improve = 0
            ran_len_gen_this_epoch = True
            model_filename = f"{args.model}_best_model_{wandb.run.id}.pth"
            model_path = os.path.join(models_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model for {args.model} at {model_path}")

            # Evaluate on test set with best model
            test_dataset.reset_rng()   # identical benchmark set every evaluation
            test_loss = eval_fn(model, test_loader, criterion, device, args.input_frames, epoch, split_name="test",
                                show_h_state=args.show_h_state)

            history['test_loss'].append(test_loss)
            print(f"Test Loss: {test_loss:.4f}")

            gen_test_dataset.reset_rng()   # identical benchmark set every evaluation
            gen_mean, gen_std, gen_details = len_gen_fn(model, gen_test_loader, device, args.gen_input_frames,
                                                        show_h_state=args.show_h_state)
            save_len_gen_results(
                os.path.join(results_dir, f"len_gen_{args.model}_{wandb.run.id}.npz"),
                gen_mean, gen_std, gen_details, args, epoch=epoch,
            )

            wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
            wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
            wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})
            log_length_generalization_curve(gen_mean, gen_std, gen_pred_frames, args)
            
            if args.run_velocity_generalization:
                vx, vy, err = eval_velocity_generalization(model, device, args)
                save_vel_gen_results(
                    os.path.join(results_dir, f"vel_gen_{args.model}_{wandb.run.id}.npz"),
                    vx, vy, err, args,
                )

                wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                        for i, vx_i in enumerate(vx)
                        for j, vy_j in enumerate(vy)})

                fig, ax = plt.subplots()
                im = ax.imshow(err[::-1],  # flip y so +vy is up
                            extent=[vx[0]-0.5, vx[-1]+0.5, vy[0]-0.5, vy[-1]+0.5],
                            origin='lower')
                ax.set_xlabel('$v_x$  (pixels / frame)')
                ax.set_ylabel('$v_y$')
                ax.set_title('Velocity-generalisation MSE')
                fig.colorbar(im, ax=ax)
                wandb.log({"vel_gen_heatmap": wandb.Image(fig)})
                plt.close(fig)

            # Log model path and metrics to wandb
            wandb.log({
                "best_model_path": model_path,
                "best_val_loss": val_loss,
                "test_loss": test_loss,
                "epoch": epoch
            })
        else:
            epochs_since_improve += 1

        # Periodic length generalization, decoupled from val improvement:
        # with honest (frozen-velocity) validation the selection metric can
        # saturate while the model keeps improving, which would otherwise
        # starve the len-gen monitoring. Saves to *_latest.npz so it never
        # overwrites the best-checkpoint results.
        if (args.len_gen_every > 0 and epoch % args.len_gen_every == 0
                and not ran_len_gen_this_epoch):
            gen_test_dataset.reset_rng()   # identical benchmark set every evaluation
            gen_mean, gen_std, gen_details = len_gen_fn(model, gen_test_loader, device, args.gen_input_frames,
                                                        show_h_state=args.show_h_state)
            save_len_gen_results(
                os.path.join(results_dir, f"len_gen_{args.model}_{wandb.run.id}_latest.npz"),
                gen_mean, gen_std, gen_details, args, epoch=epoch,
            )
            wandb.log({"len_gen_mean_over_time_latest": gen_mean.mean(), "epoch": epoch})

        # Full resume checkpoint (model + optimizer + scheduler + curves),
        # written atomically so a crash mid-save can't corrupt the file.
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val_losses": best_val_losses,
            "epochs_since_improve": epochs_since_improve,
            "history": history,
            "curve_recorder": curve_recorder.state_dict() if curve_recorder is not None else None,
            "run_id": wandb.run.id,
            "args": vars(args),
        }
        tmp_path = checkpoint_path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, checkpoint_path)

        if (args.early_stop_patience > 0 and epoch >= args.min_epochs
                and epochs_since_improve >= args.early_stop_patience):
            print(f"Early stopping at epoch {epoch}: no improvement for "
                  f"{epochs_since_improve} epochs (patience={args.early_stop_patience}, "
                  f"min_epochs={args.min_epochs}).")
            break

    # Final history dump (also written incrementally after every epoch)
    dump_history(history)
    print(f"Saved training history to {history_path}")

    # Three-way decoder-velocity comparison, side by side. The expected
    # ordering is tracked (oracle, has the true next frame) < predicted
    # (extrapolates) < frozen (holds the last encoder velocity) -- but ONLY on
    # data whose velocity keeps changing through the rollout. Under the default
    # --freeze_after the dataset freezes the velocity at the boundary, which
    # makes "frozen" exactly right and leaves the head nothing to win; run with
    # --motion_mode accelerate --freeze_after 0 to see the ordering above.
    if args.eval_velocity_mode == 'all' and history.get('val_predicted_loss'):
        best_i = int(np.argmin(history['val_loss']))
        print("\nDecoder velocity comparison (val loss at the best frozen-selection epoch "
              f"{best_i + 1}):")
        for label, key in (("frozen   ", 'val_loss'),
                           ("predicted", 'val_predicted_loss'),
                           ("tracked  ", 'val_tracked_loss')):
            series = history.get(key) or []
            if best_i < len(series):
                print(f"  {label} : {series[best_i]:.4f}   (final {series[-1]:.4f})")
        if args.freeze_after < 0:
            print("  NOTE: --freeze_after is at its default, so the dataset velocity is frozen at "
                  "the context boundary and 'frozen' is the exactly-correct policy. This comparison "
                  "only means something with --freeze_after 0.")

    # Print final best model paths and test results
    print("\nBest model paths and test results:")
    model_filename = f"{args.model}_best_model_{wandb.run.id}.pth"
    model_path = os.path.join(models_dir, model_filename)
    test_loss = history['test_loss'][-1] if history['test_loss'] else None
    print(f"{args.model}: {model_path} | Test Loss: {test_loss:.4f}" if test_loss is not None else f"{args.model}: {model_path} | Test Loss: N/A")

    # Stable, run-id-free copy of the best model's weights -- the one file to
    # point at for "the trained felstm/melstm/lstm parameters," without
    # needing to know this run's wandb id. Overwritten by any later run of
    # the same --model; the run-id-named file above is untouched and still
    # gives full per-run traceability if you ever need it.
    if os.path.exists(model_path):
        clean_path = os.path.join(models_dir, f"{args.model}_best.pth")
        shutil.copy(model_path, clean_path)
        print(f"Best model parameters (stable name): {clean_path}")
    else:
        print(f"WARNING: no best-model checkpoint found at {model_path} "
              f"(val loss never improved?) -- no stable-name copy made.")

    # The periodic len-gen snapshot (--len_gen_every) exists only to watch
    # progress mid-training when the honest val metric saturates -- once
    # training is truly done, the best-checkpoint len_gen_<model>_<id>.npz
    # above supersedes it. Removing it here keeps exactly one len-gen file
    # per finished run instead of two, so len_gen_<model>_*.npz globs match
    # cleanly for plot_len_gen_comparison.py.
    latest_len_gen_path = os.path.join(
        results_dir, f"len_gen_{args.model}_{wandb.run.id}_latest.npz")
    if os.path.exists(latest_len_gen_path):
        os.remove(latest_len_gen_path)
        print(f"Removed periodic len-gen snapshot (superseded): {latest_len_gen_path}")

    # All --epochs completed (not just this Slurm submission's walltime
    # slice): marks the run as finished so submit_comparison.sbatch's
    # self-chaining knows to stop resubmitting.
    done_path = os.path.join(run_state_dir, f"DONE_{args.model}_{wandb.run.id}.flag")
    open(done_path, "w").close()
    print(f"Training complete ({args.epochs} epochs) — wrote {done_path}")

if __name__ == '__main__':
    main()
