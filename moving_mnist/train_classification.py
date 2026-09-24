"""
Train a motion-defined-digit classifier on Common-Fate Moving MNIST.

The experiment
--------------
Give the model T context frames and ask which digit is moving. Nothing in any
single frame says: figure and background are drawn from the same band-limited
noise, so the digit exists only as a region whose velocity differs from its
surroundings. The claim under test is that the VELOCITY STRUCTURE of the
recurrent state exposes it -- felstm's fixed lattice of transported copies,
melstm's tracked slots -- while a plain ConvLSTM has nowhere to put it.

    lstm    is the control. If it climbs meaningfully above chance the dataset
            is leaking and the result means nothing, so it is the first number
            to read, not the last.

Data
----
The background is drawn from a velocity grid DISJOINT from the figures'
(--bg_speed_min > --data_v_range), so figure and background can never coincide
by accident -- no rejection sampling, no residual chance of a figureless
sequence. The figure's velocity is piecewise-constant by default.

For felstm to be able to represent the figure at all, --v_range must be at least
--data_v_range: its copies sit at FIXED lattice velocities and a figure moving
faster than any of them has no copy to accumulate in. The script refuses the
combination rather than reporting a mysterious failure.

Diagnostics that matter more than the loss curve
------------------------------------------------
`slot_hit_fig` : how often a melstm slot's tracked velocity actually equals the
                 ground-truth figure velocity. If this is near zero, melstm
                 never even represented the figure and its accuracy says nothing
                 about motion-defined classification.
`attn_on_fig`  : how much of the attention pool's mass sits on a velocity copy
                 matching the figure. The head can only use what it attends to.

Usage
-----
    python moving_mnist/train_classification.py --model melstm
    python moving_mnist/train_classification.py --model felstm --v_range 2
    python moving_mnist/train_classification.py --model lstm
"""
import argparse
import json
import math
import warnings
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common_fate_moving_mnist_dataset import CommonFateMovingMNISTDataset
from torchvision.datasets import MNIST
from motion_classification_model import build_classifier, recompute_bn_stats
from mps_integer_warp import enable_integer_shift_warp


# ------------------------------------------------------------------ arguments
def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- model
    p.add_argument('--model', choices=['lstm', 'felstm', 'melstm'], default='melstm')
    p.add_argument('--hidden_size', type=int, default=32,
                   help='Recurrent cell width. Carried on every velocity copy at every '
                        'timestep and kept for BPTT, so felstm pays for this (2R+1)^2 times.')
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--v_range', type=int, default=2,
                   help='felstm only: velocity lattice half-width, giving (2R+1)^2 copies. '
                        'Must be >= --data_v_range or no copy can represent the figure.')
    p.add_argument('--num_vel_modes', type=int, default=2,
                   help='melstm only: number of velocity slots. The scene contains exactly '
                        'TWO motions -- the digit and the background -- so 2 is the right '
                        'number, and measured slot_hit_fig confirms it: with '
                        '--velocity_source bootstrap it is 97.9%% at K=2 and identically '
                        '97.9%% at K=3 and K=4, so extra slots buy nothing and cost compute '
                        'linearly. (With the weaker frame_pair source, K=2 only reaches '
                        '69%% and extra slots DO help -- which is a defect of the peak '
                        'selection, not evidence that the scene has more motions.)')
    p.add_argument('--velocity_source', choices=['bootstrap', 'frame_pair', 'tracked'],
                   default='bootstrap',
                   help="melstm only: where slot velocities come from. Measured "
                        "slot_hit_fig at K=2 on this data: bootstrap 97.9%%, frame_pair "
                        "68.8%%, tracked 0.0%%. "
                        "'tracked' is MEConvLSTM's own protocol (correlate each slot's "
                        "hidden state against the next frame); it needs a clean template "
                        "and has none here, so no slot ends up on either motion and the "
                        "slots collapse onto each other. "
                        "'frame_pair' takes the top-K peaks of the raw frame pair, but the "
                        "background dominates the correlation surface so the figure's peak "
                        "often loses to the noise floor. "
                        "'bootstrap' (default) explains the dominant motion away first and "
                        "re-correlates the residual, which is what makes the MINORITY "
                        "motion reliably findable -- and therefore what makes K=2 correct.")
    p.add_argument('--velocity_pool', choices=['attention', 'max', 'mean', 'concat'],
                   default='attention',
                   help="How the velocity axis is reduced before the head. 'max' is the "
                        "default elsewhere in this repo and is expected to do BADLY here: "
                        "every copy carries equal-amplitude noise, so the informative one "
                        "is distinguished by spatial COHERENCE, not by magnitude.")
    p.add_argument('--pool_temperature', type=float, default=1.0)
    p.add_argument('--head_channels', type=int, default=64)
    p.add_argument('--head_blocks', type=int, default=3)
    p.add_argument('--head_mlp_hidden', type=int, default=128)
    p.add_argument('--head_dropout', type=float, default=0.0)
    p.add_argument('--precise_bn_batches', type=int, default=50,
                   help="Batches used to RE-ESTIMATE BatchNorm statistics before each "
                        "evaluation (0 = off, use the running EMA). BatchNorm's EMA is "
                        "collected while the weights are still moving, and for a head on a "
                        "recurrent state it never catches up -- val accuracy then bounces "
                        "between chance and its true value while training accuracy rises "
                        "smoothly. Recomputing gives the exact statistics for the CURRENT "
                        "weights. No effect when --head_norm is not 'batch'.")
    p.add_argument('--head_norm', choices=['batch', 'group', 'none'], default='batch',
                   help="Normalisation in the classifier head. 'batch' (default) is the "
                        "only one that LEARNS on this task -- measured over 14 epochs, "
                        "batch reaches 0.230 train accuracy while group (at 1e-3 and 3e-3) "
                        "and none all stay flat at ~0.11. Batch statistics remove the "
                        "component common to every sample, which here is the background "
                        "noise that dominates the figure's contribution. Its running "
                        "statistics do lag badly on this model, which is what "
                        "--precise_bn_batches exists to fix.")

    # --- data
    p.add_argument('--root', type=str, default=str(_HERE.parent / 'data'))
    p.add_argument('--seq_len', type=int, default=15, help='Context frames T')
    p.add_argument('--image_size', type=int, default=36,
                   help='Canvas size. 36 keeps a 28px digit on a torus with room to move '
                        'while staying affordable for felstm, whose cost carries every one '
                        'of its (2R+1)^2 copies at every timestep.')
    p.add_argument('--num_figures', type=int, default=1)
    p.add_argument('--variant', choices=['moving_mask', 'static_mask'], default='moving_mask')
    p.add_argument('--corr_len', type=float, default=0.0,
                   help='Texture correlation length. LEAVE AT 0. Above 0, pixels within a '
                        'region are correlated while pixels across the figure boundary are '
                        'not, so the outline is a local-statistics discontinuity present in '
                        'EVERY frame. A single-frame CNN with no temporal information at '
                        'all scores 11%% / 13%% / 24%% / 36%% at corr_len 0 / 0.5 / 1 / 2 '
                        '(chance 10%%) -- so at the old 1.0 default a per-frame model could '
                        'already do most of the job, and lstm reaching high accuracy meant '
                        'the dataset was leaking, not that it had learned motion.')
    p.add_argument('--data_v_range', type=int, default=2, help="Figure max speed")
    p.add_argument('--bg_mode', choices=['opposite', 'disjoint', 'constant', 'incoherent'],
                   default='opposite',
                   help="How the background moves. In 'opposite' and 'disjoint' it follows "
                        "the SAME motion law as the figure (--motion_mode etc.), drawn "
                        "independently. 'opposite' (default) keeps it on the SHARED velocity "
                        "grid and requires it to travel in an opposing direction at t=0 "
                        "only. 'disjoint' "
                        "gives it a faster grid of its own (--bg_speed_min/max), which "
                        "guarantees they never coincide but puts the background BEYOND "
                        "every lattice copy felstm has -- a motion felstm structurally "
                        "cannot represent while melstm's tracked slots can, which confounds "
                        "expressive power with the effect being measured. Use 'disjoint' "
                        "only when felstm is not in the comparison. 'constant' fixes the "
                        "background at --bg_velocity for every frame of every sequence, so "
                        "the motion law (and any motion sweep) applies to the figure alone; "
                        "the same felstm caveat holds when --bg_velocity is outside its "
                        "lattice (max(|vx|,|vy|) > --v_range). 'incoherent' draws FRESH "
                        "background noise every frame: no background motion for any model "
                        "to hold still, so the figure is the only coherent motion.")
    p.add_argument('--bg_velocity', type=int, nargs=2, default=None, metavar=('VX', 'VY'),
                   help="--bg_mode constant only: the background's velocity in px/frame. "
                        "Separated from every figure velocity by construction when "
                        "max(|VX|,|VY|) >= --data_v_range + 2, e.g. '4 0'; closer than that, "
                        "figure trajectories that come within 1 px/frame of it are redrawn.")
    p.add_argument('--bg_speed_min', type=int, default=4,
                   help='--bg_mode disjoint only: background min speed, must exceed '
                        '--data_v_range.')
    p.add_argument('--bg_speed_max', type=int, default=5,
                   help='--bg_mode disjoint only.')
    p.add_argument('--motion_mode', choices=['constant', 'piecewise', 'stochastic', 'accelerate'],
                   default='piecewise')
    p.add_argument('--transition_mode', choices=['uniform', 'smooth'], default='smooth')
    p.add_argument('--min_segment', type=int, default=3)
    p.add_argument('--max_segment', type=int, default=6)
    p.add_argument('--digit_scale', type=int, default=1)
    p.add_argument('--normalize', choices=['affine', 'minmax', 'none'], default='affine')

    # --- optimisation
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_train_samples', type=int, default=None,
                   help='Cap on training glyphs per epoch (None = all of them, 54000 at '
                        'the default --val_fraction). Each glyph is re-rendered with fresh '
                        'velocities and textures on every access, so lowering this reduces '
                        'the number of DISTINCT DIGITS seen, not just the epoch length.')
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--val_size', type=int, default=None,
                   help='Cap on validation glyphs per epoch (None = all 6000). The val '
                        'set is fixed, so this trades statistical precision for epoch time.')
    p.add_argument('--test_size', type=int, default=None,
                   help='Cap on test glyphs (None = all 10000, MNIST\'s entire test '
                        'split). With --val_fraction 0.1 the three pools are then '
                        '54000/6000/10000, i.e. MNIST\'s own split, re-rendered under '
                        'motion. Only the reported test number is affected; lowering '
                        'this trades statistical precision for time.')
    p.add_argument('--test_every', type=int, default=1,
                   help='Also evaluate the TEST set every N epochs, so wandb gets a '
                        'test_acc/test_loss curve rather than one final point (0 = off). '
                        'Model selection still reads val only -- the checkpoint, the '
                        'scheduler and early stopping never see this number, and the '
                        'reported test_acc is still the best-val checkpoint measured '
                        'once at the end. Treat the curve as a diagnostic: a human '
                        'watching it and stopping a run on it is selecting on test.')
    p.add_argument('--test_curve_size', type=int, default=2000,
                   help='How many test sequences each per-epoch curve point uses '
                        '(0 = all of them). The FINAL number always uses the full '
                        '--test_size; this only caps the diagnostic curve, which at '
                        '10000 sequences would otherwise add ~19%% to every epoch for '
                        'a standard error already below 0.2%%. The subset is the '
                        'loader\'s first N sequences, so it is the same set every '
                        'epoch rather than a fresh sample.')
    p.add_argument('--early_stop_patience', type=int, default=0,
                   help='Stop once the selection metric (val accuracy) has not improved for '
                        'this many consecutive epochs. 0 = DISABLED, always run the full '
                        '--epochs, which is the default.')
    p.add_argument('--val_curve_interval', type=int, default=25,
                   help='Record validation loss every N training batches, for a '
                        'loss-vs-steps curve far finer than one point per epoch (0 = off).')
    p.add_argument('--val_curve_size', type=int, default=256,
                   help='Number of FIXED validation sequences behind that curve. They are '
                        'materialised into a tensor once: this dataset resamples content on '
                        'every access, so holding indices fixed is not enough to keep the '
                        'measured set fixed.')
    p.add_argument('--val_curve_bn_batches', type=int, default=8,
                   help='Batches used to re-estimate BatchNorm before each point of that '
                        'curve (0 = off, use the running EMA). Without this the curve is '
                        'measured under a different BatchNorm regime than the per-epoch '
                        'val_acc and swings between chance and the true accuracy while the '
                        'model improves smoothly -- see --precise_bn_batches. Smaller than '
                        'that default because it runs every --val_curve_interval steps: 8 '
                        'batches is already ~500 sequences of statistics per channel.')
    p.add_argument('--use_lr_scheduler', action='store_true')
    p.add_argument('--lr_patience', type=int, default=4)
    p.add_argument('--lr_factor', type=float, default=0.5)

    # --- bookkeeping
    p.add_argument('--data_seed', type=int, default=42,
                   help='Governs the DATA and should be held FIXED across a seed sweep: '
                        'the 54000/6000 train/val glyph split, the seeded val and test '
                        'benchmarks, and which sequences the state images use. Changing it '
                        'changes the benchmark itself, so runs with different data_seeds '
                        'are not comparable to each other.')
    p.add_argument('--model_seed', type=int, default=None,
                   help='Governs the RUN and is what to vary across a seed sweep: weight '
                        'initialisation and the order training data is visited -- and '
                        'nothing else, as long as --train_stream_seed is held fixed. None = '
                        'follow --data_seed. Vary this alone and every run is scored on '
                        'the identical val set, so the spread you measure is run-to-run '
                        'variance rather than a different benchmark each time.')
    p.add_argument('--train_stream_seed', type=int, default=None,
                   help='Governs the TRAINING SEQUENCES: every training sample (motion of '
                        'both layers, textures, placement) is a function of (this seed, '
                        'epoch, glyph index) only. None = follow --data_seed, so a model_seed '
                        'sweep trains every seed on the identical sequences, only visited in '
                        'a different order. Vary this with model_seed held fixed to measure '
                        'the data stream\'s share of the seed-to-seed spread. (Before it '
                        'existed the samples came from np.random seeded per DataLoader worker '
                        'from the model seed, so every model_seed also saw different data.)')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--save_dir', type=str, default='./experiments_classification/')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--log_states_every', type=int, default=1,
                   help='Log the per-velocity-copy hidden state to wandb every N epochs '
                        '(0 = off; 1 = every epoch, the default, since the whole point is '
                        'watching it develop). Uses a FIXED set of sequences so the slider '
                        'shows the same sample developing as training proceeds. This is '
                        'the picture the experiment rests on: the frame row is noise, and '
                        'the question is whether the copy transported at the figure '
                        'velocity grows the digit while the others do not.')
    p.add_argument('--log_states_samples', type=int, default=30,
                   help='How many sequences to DRAW each time states are logged. Chosen at '
                        'random from the fixed benchmark set, then held fixed, so the '
                        'pictures span different digits and speeds while still showing the '
                        'SAME sequences developing across epochs.')
    p.add_argument('--log_states_readout_samples', type=int, default=3,
                   help='Additional DIAGNOSTIC panels, logged under val_readout_states_*, '
                        'that add a local-variance row under the figure copy -- the picture '
                        'form of val_state_shape_iou. Kept separate from the paper panels '
                        'because it is a diagnostic about the metric, not about the model. '
                        '0 = off.')
    p.add_argument('--states_fig_dir', type=str, default=None,
                   help='Write the state panels here as PNG and PDF, for the paper. Only '
                        'the FINAL epoch is written, however often --log_states_every logs '
                        'to wandb: the slider there is for watching training, the files are '
                        'for the paper and want one trained model, not fifty.')
    p.add_argument('--state_metric_samples', type=int, default=64,
                   help='How many sequences val_state_shape_iou averages over. Independent '
                        'of --log_states_samples: the scalar wants many for a readable '
                        'curve, the pictures want few to stay legible.')
    p.add_argument('--wandb_project', type=str, default='FERNN-common-fate')
    p.add_argument('--wandb_entity', type=str, default=None)
    p.add_argument('--wandb_dir', type=str, default='./tmp/')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--smoke_test', action='store_true',
                   help='Tiny run: a few batches per epoch, 2 epochs, no wandb. '
                        'For checking the plumbing end to end.')
    return p.parse_args(argv)


# ----------------------------------------------------------------------- data
def build_datasets(args):
    # The mask is only needed by the state visualisation (as the answer key
    # beside the hidden states); rendering it per sample otherwise is waste.
    want_mask = bool(args.use_wandb and args.log_states_every)
    common = dict(
        root=args.root, image_size=args.image_size, seq_len=args.seq_len,
        num_figures=args.num_figures, variant=args.variant, corr_len=args.corr_len,
        digit_scale=args.digit_scale, normalize=args.normalize,
        max_speed=args.data_v_range,
        bg_opposite_at_start=(args.bg_mode == "opposite"),
        bg_speed_range=((args.bg_speed_min, args.bg_speed_max)
                        if args.bg_mode == "disjoint" else None),
        bg_velocity=(tuple(args.bg_velocity) if args.bg_mode == "constant" else None),
        bg_incoherent=(args.bg_mode == "incoherent"),
        motion_mode=args.motion_mode, transition_mode=args.transition_mode,
        min_segment=args.min_segment, max_segment=args.max_segment,
        return_motion=True, return_mask=want_mask, download=True,
    )
    # Split the MNIST TRAIN split's glyphs into train / val, disjointly.
    #
    # This has to be done at the glyph level, not by dataset index: without
    # `digit_indices` the dataset ignores its index and draws a random glyph on
    # every access, so a random_split would hand both halves the same 60k pool
    # and the val number would be measured on digits already trained on. Fine for
    # next-frame prediction (what the parent class was built for), wrong for
    # classification.
    n_mnist = len(MNIST(root=args.root, train=True, download=True))
    perm = torch.randperm(n_mnist, generator=torch.Generator().manual_seed(args.data_seed))
    n_val = int(round(args.val_fraction * n_mnist))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    # Keyed per (stream seed, epoch, glyph index), so the training sequences do not
    # depend on --model_seed; make_loaders' sampler supplies the epoch.
    stream_seed = (args.data_seed if getattr(args, "train_stream_seed", None) is None
                   else args.train_stream_seed)
    train = CommonFateMovingMNISTDataset(train=True, random=True, seed=args.data_seed,
                                         stream_seed=stream_seed,
                                         digit_indices=train_idx, **common)
    # Val and test are both FIXED benchmarks: seeded and stateful, so reset_rng()
    # before every pass and never use persistent workers (they would carry
    # advanced RNG state forward). A fixed val set also makes the epoch-to-epoch
    # curve readable instead of mostly resampling noise.
    val = CommonFateMovingMNISTDataset(train=True, random=False, seed=777,
                                       digit_indices=val_idx, **common)
    # Test comes from MNIST's own TEST split -- a third disjoint glyph set.
    test = CommonFateMovingMNISTDataset(train=False, random=False, seed=123, **common)
    return train, val, test


class EpochShuffleSampler(torch.utils.data.Sampler):
    """
    Shuffled (epoch, index) pairs for a dataset keyed on stream_seed.

    The epoch travels WITH each index so the dataset can key every sample on
    (stream_seed, epoch, index): a fresh sequence per glyph per epoch, and the same
    one under every model seed. It cannot be set on the dataset instead -- with
    persistent workers each worker holds its own copy of the dataset, and an
    attribute set in the main process never reaches it. The sampler lives in the
    main process, so set_epoch() does.

    The ORDER is drawn from the global torch RNG, i.e. from --model_seed, exactly as
    shuffle=True drew it before.
    """

    def __init__(self, indices):
        self.indices = list(indices)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        perm = torch.randperm(len(self.indices)).tolist()
        return iter([(self.epoch, self.indices[j]) for j in perm])

    def __len__(self):
        return len(self.indices)


def make_loaders(args, train_ds, val_ds, test_ds):
    # The datasets already hold disjoint glyph pools; these only cap how many of
    # them an epoch visits. The training cap goes into the sampler rather than a
    # Subset, because the sampler hands the dataset (epoch, index) pairs.
    n_train = len(train_ds)
    va = val_ds
    if args.max_train_samples and args.max_train_samples < n_train:
        n_train = args.max_train_samples
    if args.val_size and args.val_size < len(va):
        va = Subset(va, list(range(args.val_size)))
    if args.smoke_test:
        n_train = min(n_train, 2 * args.batch_size)
        va = Subset(va, list(range(args.batch_size)))

    te = test_ds
    if args.test_size and args.test_size < len(te):
        te = Subset(te, list(range(args.test_size)))
    if args.smoke_test:
        te = Subset(test_ds, list(range(args.batch_size)))

    # Sequences for the state visualisation come from the FIXED benchmark set,
    # not from val. val is split off a random=True dataset, so every access
    # renders a fresh sequence -- fine for an unbiased val metric (and what
    # train.py does), useless for watching one sample develop across epochs,
    # which is the entire point of the picture. Paired with reset_rng() before
    # each logging pass, this yields the identical sequences every time.
    # One pool serves both the pictures and the shape-IoU scalar, so size it for
    # the larger of the two. Sizing it to log_states_samples silently capped the
    # metric to that many sequences, which made the curve mostly noise.
    #
    # The indices are drawn at RANDOM but then held fixed: random so the pictures
    # span different digits, speeds and placements instead of whatever happens to
    # sit at indices 0..n; fixed so the wandb slider shows the SAME sequences
    # developing across epochs, which is the point of logging them repeatedly.
    n_state = min(max(args.log_states_samples, args.state_metric_samples), len(test_ds))
    state_idx = torch.randperm(
        len(test_ds), generator=torch.Generator().manual_seed(args.data_seed + 1)
    )[:n_state].tolist()
    st = Subset(test_ds, state_idx)

    kw = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, batch_size=args.batch_size,
                       sampler=EpochShuffleSampler(range(n_train)),
                       persistent_workers=args.num_workers > 0, **kw),
            DataLoader(va, batch_size=args.batch_size, persistent_workers=False, **kw),
            DataLoader(te, batch_size=args.batch_size, persistent_workers=False, **kw),
            DataLoader(st, batch_size=n_state, shuffle=False, num_workers=0))


# ---------------------------------------------------------------- diagnostics
def velocity_diagnostics(aux, motion, n_figures):
    """
    Did the model's velocity machinery actually represent the figure?

    slot_hit_fig / slot_hit_bg : fraction of samples where SOME tracked slot
        ends the encoder holding the ground-truth figure / background velocity.
        melstm only -- felstm's copies are a fixed lattice that always contains
        every representable velocity, so the question is vacuous there.
    attn_on_fig : share of the attention pool's mass on a copy whose velocity
        matches the figure. Needs the velocities, so melstm only.

    motion is (B, T, N+1, 2); the figure velocity that h_T was last transported
    at is motion[:, -2, 0] -- the step from frame T-2 to T-1, the last one the
    encoder saw.
    """
    out = {}
    v = aux.get("velocities")
    if v is None or motion is None:
        return out

    v_last = v[:, -1]                                     # (B, K, 2) tracked
    gt_fig = motion[:, -2, 0]                             # (B, 2)
    gt_bg = motion[:, -2, n_figures]                      # (B, 2)

    hit_fig = (v_last.round().long() == gt_fig[:, None, :]).all(-1)   # (B, K)
    hit_bg = (v_last.round().long() == gt_bg[:, None, :]).all(-1)
    out["slot_hit_fig"] = hit_fig.any(-1).float().mean().item()
    out["slot_hit_bg"] = hit_bg.any(-1).float().mean().item()

    w = aux.get("pool_weights")
    if w is not None and w.shape[1] == v_last.shape[1]:
        out["attn_on_fig"] = (w * hit_fig.float()).sum(-1).mean().item()
        out["attn_on_bg"] = (w * hit_bg.float()).sum(-1).mean().item()
        # Entropy in nats: 0 = the head committed to one velocity copy,
        # log(K) = it is hedging uniformly and the pool is doing nothing.
        # Watch this FALL as attn_on_fig rises; flat at log(K) means the head
        # never learned to select, whatever the accuracy says.
        out["attn_entropy"] = (-(w.clamp_min(1e-9).log() * w).sum(-1)).mean().item()
        out["attn_entropy_max"] = float(np.log(w.shape[1]))
    return out


def _local_var(x, k=3):
    """Local variance in a kxk window, circular (the canvas is a torus). x: (B,H,W)."""
    x = x.unsqueeze(1)
    pad = k // 2
    mu = F.avg_pool2d(F.pad(x, (pad,) * 4, mode="circular"), k, stride=1)
    mu2 = F.avg_pool2d(F.pad(x * x, (pad,) * 4, mode="circular"), k, stride=1)
    return (mu2 - mu * mu).clamp(min=0).squeeze(1)


def _area_matched_iou(score, mask):
    """
    IoU after thresholding `score` at whatever level selects exactly as many
    pixels as the mask contains. Area-matching removes the threshold as a free
    parameter, so this measures the RANKING the score induces and nothing else.
    """
    B = score.shape[0]
    s = score.reshape(B, -1)
    t = mask.reshape(B, -1) > 0.5
    k = t.sum(1).clamp(min=1)
    thr = s.sort(dim=1, descending=True).values.gather(1, (k - 1).unsqueeze(1))
    p = s >= thr
    return ((p & t).sum(1).float() / (p | t).sum(1).clamp(min=1).float())


def state_shape_iou(model, states, velocities, motion, mask):
    """
    Does the hidden state CONTAIN the digit's shape, in the copy transported at
    the figure's velocity?

    Take that copy, take the local variance of its CHANNEL MEAN, and score it
    against the true mask (area-matched IoU). Returns two numbers: `matched`, for
    the velocity-matched copy, and `best`, the best over all copies.

    Read it as a measure of TRANSPORT COHERENCE, not of task performance. Three
    limits are worth knowing, all measured:

    * It is not a predictor of accuracy. felstm reached 0.976 val accuracy with a
      matched IoU at chance, because its information is spread across copies and
      over time rather than concentrated in one copy at the end.
    * It penalises FIXED velocity lattices under time-varying motion. Under
      constant motion melstm and felstm both score 0.521 (chance 0.095); under
      piecewise motion melstm holds 0.494 while felstm falls to 0.317, because
      felstm's copies cannot follow a velocity that changes mid-sequence while
      melstm re-estimates its slots every step. That is a real architectural
      difference, not a defect in either.
    * It reads the channel MEAN, so it measures raw accumulated texture. A
      trained cell may encode the figure in particular channels that cancel in
      the mean, which is why the number tends to fall rather than rise during
      training.

    `best` is the fairer cross-architecture number -- "is the digit anywhere in
    the state" -- but it is biased toward models with more copies, since taking
    the best of V gets more chances as V grows.
    """
    if mask is None or motion is None or states is None:
        return None

    h = states[:, -1]                                  # (B, V, H, W) channel-mean
    B, V = h.shape[:2]
    gt_fig = motion[:, -2, 0]                          # (B, 2) the last transported v
    target = mask[:, -1].amax(dim=1)                   # (B, H, W) figure at frame T-1

    # Pick, per sample, the copy transported at the figure's velocity.
    idx = torch.zeros(B, dtype=torch.long, device=h.device)
    valid = torch.ones(B, dtype=torch.bool, device=h.device)

    if model.model in ("lstm", "felstm"):
        grid = {tuple(v): k for k, v in enumerate(model.backbone.cell.v_list)}
        for b in range(B):
            key = tuple(int(x) for x in gt_fig[b])
            if key in grid:
                idx[b] = grid[key]
            elif V == 1:
                idx[b] = 0                              # lstm: the only state there is
            else:
                valid[b] = False
    else:
        if velocities is None:
            return None
        match = (velocities[:, -1].round().long() == gt_fig[:, None, :]).all(-1)  # (B, K)
        idx = match.float().argmax(-1)
        valid = match.any(-1)

    if not bool(valid.any()):
        return None
    sel = h[torch.arange(B, device=h.device), idx]      # (B, H, W)
    matched = _area_matched_iou(_local_var(sel), target)
    best = torch.stack([_area_matched_iou(_local_var(h[:, v]), target)
                        for v in range(V)]).max(dim=0).values
    return float(matched[valid].mean()), float(best.mean())


def log_states(model, loader, fixed_ds, device, args, epoch, step, save=False):
    """
    One small forward with return_states=True, on a FIXED set of sequences.

    `fixed_ds` is reset first so the sequences are byte-identical at every epoch
    -- otherwise the picture shows a different sample each time and cannot show
    anything developing.

    Kept separate from the training loop and capped at a couple of samples
    because the state tensor is (B, T, V, H, W) -- at felstm's 25 copies and
    64px that is gigabytes for a full batch, for a picture of two.
    """
    from visualization import log_motion_classification_states

    fixed_ds.reset_rng()
    batch = next(iter(loader))
    # More samples than the pictures use: the images need two, the IoU scalar
    # would be far too noisy on two.
    n = min(args.state_metric_samples, batch[0].shape[0])
    seq = batch[0][:n].to(device)
    motion = batch[2][:n].to(device) if len(batch) > 2 else None
    mask = batch[3][:n].to(device) if len(batch) > 3 else None

    was_training = model.training
    model.eval()
    with torch.no_grad():
        h_last, velocities, states = model.encode(seq, return_states=True)
    model.train(was_training)

    extra = {}
    iou = state_shape_iou(model, states, velocities, motion, mask)
    if iou is not None:
        extra["val_state_shape_iou"] = iou[0]          # velocity-matched copy
        extra["val_state_shape_iou_best"] = iou[1]     # best copy, fairer across V
        if mask is not None:
            # Chance for an area-matched IoU is the mask's area fraction.
            extra["val_state_shape_iou_chance"] = float(mask[:, -1].amax(1).mean())

    v_list = (model.backbone.cell.v_list
              if model.model in ("lstm", "felstm") else None)
    # Paper panels: no readout row, optionally written to disk as PNG/PDF.
    log_motion_classification_states(
        states, seq, mask_track=mask, velocities=velocities,
        v_list=v_list, gt_motion=motion, split_name="val", epoch=epoch,
        step=step, num_samples=min(args.log_states_samples, seq.shape[0]),
        save_dir=args.states_fig_dir if save else None)

    # A few diagnostic panels WITH the local-variance readout, under their own
    # key so they never end up in a paper figure by accident.
    if args.log_states_readout_samples:
        log_motion_classification_states(
            states, seq, mask_track=mask, velocities=velocities,
            v_list=v_list, gt_motion=motion, split_name="val_readout", epoch=epoch,
            step=step,
            num_samples=min(args.log_states_readout_samples, seq.shape[0]),
            show_shape_readout=True)
    return extra


class ValCurveRecorder:
    """
    Fine-grained loss-vs-training-batches curve, recorded during training.

    One point per epoch is a very coarse picture of a 50-epoch run; this samples
    the same fixed sequences every `interval` optimizer steps instead.

    The set is MATERIALISED into a tensor at construction rather than held by
    index. This dataset renders a fresh sequence on every access, so fixing the
    indices would still measure a different set each time and the curve would be
    mostly resampling noise. (Same reason the state images use the fixed
    benchmark split.)

    Measured under precise BatchNorm (see `bn_batches`), like the per-epoch val --
    the two series are meant to be read against each other.

    Points are logged LIVE, through `log_fn`, so the curve is watchable while a
    50-epoch run is still going rather than appearing only at the end. That is
    possible because this script's wandb step axis is the OPTIMIZER STEP, not the
    epoch -- epoch metrics are logged at whatever step the epoch ended on, so
    both series share one monotonically increasing counter and neither is
    rejected. (Mixing an epoch step axis with a per-batch one silently drops
    whichever is behind.)
    """

    def __init__(self, dataset, n_sequences, interval, device, batch_size=64,
                 bn_dataset=None, bn_batches=0):
        n = min(n_sequences, len(dataset))
        if hasattr(dataset, "reset_rng"):
            dataset.reset_rng()
        seqs, labels = [], []
        for i in range(n):
            item = dataset[i]
            seqs.append(item[0])
            labels.append(item[1])
        self.seq = torch.stack(seqs)
        self.label = torch.tensor(labels, dtype=torch.long)
        self.interval = interval
        self.device = device
        self.batch_size = batch_size
        # Training sequences for the precise-BN pass, MATERIALISED here rather than
        # taken by iterating the live train_loader. Iterating it would be a bug, not
        # a slow path: make_loaders sets persistent_workers=True whenever
        # num_workers > 0, and for such a loader DataLoader.__iter__ RESETS the one
        # shared iterator instead of returning a new one -- including the iterator
        # the training loop is in the middle of. The epoch then never reaches
        # StopIteration and never ends. (num_workers=0 makes a fresh iterator and
        # hides this entirely, so it must not be the only configuration tested.)
        # Fixed batches are also the better estimator here: identical statistics at
        # every curve point means the curve moves because the model moved.
        self._bn = []
        if bn_batches > 0 and bn_dataset is not None:
            want = min(bn_batches * batch_size, len(bn_dataset))
            pick = torch.randperm(
                len(bn_dataset), generator=torch.Generator().manual_seed(0)
            )[:want].tolist()
            xs = torch.stack([bn_dataset[i][0] for i in pick])
            self._bn = [(xs[i:i + batch_size],)
                        for i in range(0, len(xs), batch_size)]
        self.steps, self.val_loss, self.val_acc, self.train_loss = [], [], [], []

    def maybe_record(self, model, step, train_loss, criterion, log_fn=None):
        if self.interval <= 0 or step % self.interval:
            return
        was_training = model.training
        # Measure under the SAME BatchNorm regime as the per-epoch val, or the two
        # series are not comparable and this one is not interpretable. Straight
        # model.eval() uses the running EMA, which on this model lags the weights so
        # badly that the reading swings between chance and the true accuracy between
        # adjacent points while the model is in fact improving monotonically.
        # Measured on 64 overfit sequences, identical weights, felstm/max at step 40:
        # batch-stats 1.000, precise 1.000, EMA 0.062.
        # The BN buffers are snapshotted and put back, so this stays a pure
        # observation -- recompute_bn_stats() replaces running_mean/var by design and
        # would otherwise reset the EMA mid-epoch as a side effect of looking.
        saved_bn = None
        if self._bn:
            bns = [m for m in model.modules()
                   if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            if bns:
                saved_bn = [(bn, {k: v.clone() for k, v in bn.state_dict().items()})
                            for bn in bns]
                recompute_bn_stats(model, self._bn, self.device, len(self._bn))
        model.eval()
        tot_loss = correct = n = 0
        with torch.no_grad():
            for i in range(0, len(self.seq), self.batch_size):
                x = self.seq[i:i + self.batch_size].to(self.device)
                y = self.label[i:i + self.batch_size].to(self.device)
                logits = model(x)
                tot_loss += criterion(logits, y).item() * y.numel()
                correct += (logits.argmax(1) == y).sum().item()
                n += y.numel()
        if saved_bn is not None:
            for bn, state in saved_bn:
                bn.load_state_dict(state)
        model.train(was_training)
        vl, vacc = tot_loss / max(n, 1), correct / max(n, 1)
        self.steps.append(step)
        self.val_loss.append(vl)
        self.val_acc.append(vacc)
        self.train_loss.append(train_loss)
        if log_fn is not None:
            log_fn(step, {"curve/val_loss": vl, "curve/val_acc": vacc,
                          "curve/train_loss": train_loss})

    def as_dict(self):
        return {"step": self.steps, "val_loss": self.val_loss,
                "val_acc": self.val_acc, "train_loss": self.train_loss}


# -------------------------------------------------------------------- epochs
def run_epoch(model, loader, device, n_figures, criterion, optimizer=None,
              grad_clip=1.0, max_batches=None, collect_preds=False,
              curve=None, global_step=0, curve_log_fn=None):
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_correct = tot_n = 0
    diag_sum, diag_n = {}, 0
    y_true, y_pred = [], []

    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        seq, label = batch[0].to(device), batch[1].to(device)
        motion = batch[2].to(device) if len(batch) > 2 else None
        # batch may also carry the GT mask (index 3) for the state visualisation;
        # the training step itself never looks at it.

        with torch.set_grad_enabled(train):
            logits, aux = model(seq, return_aux=True)
            loss = criterion(logits, label)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            optimizer.step()
            global_step += 1
            if curve is not None:
                curve.maybe_record(model, global_step, loss.item(), criterion,
                                   log_fn=curve_log_fn)

        bs = label.size(0)
        pred = logits.argmax(1)
        tot_loss += loss.item() * bs
        tot_correct += (pred == label).sum().item()
        tot_n += bs
        if collect_preds:
            y_true.extend(label.tolist())
            y_pred.extend(pred.tolist())

        with torch.no_grad():
            d = velocity_diagnostics(aux, motion, n_figures)
        if d:
            for k, val in d.items():
                diag_sum[k] = diag_sum.get(k, 0.0) + val
            diag_n += 1

    stats = {"loss": tot_loss / max(tot_n, 1), "acc": tot_correct / max(tot_n, 1)}
    if diag_n:
        stats.update({k: v / diag_n for k, v in diag_sum.items()})
    if collect_preds:
        stats["_y_true"], stats["_y_pred"] = y_true, y_pred
    stats["_global_step"] = global_step
    return stats


# ---------------------------------------------------------------------- main
def main(argv=None):
    args = get_args(argv)

    if args.smoke_test:
        args.epochs, args.num_workers, args.use_wandb = 2, 0, False
        args.max_train_samples = 2 * args.batch_size

    if args.bg_mode == "disjoint" and args.bg_speed_min <= args.data_v_range:
        raise SystemExit(
            f"--bg_speed_min ({args.bg_speed_min}) must exceed --data_v_range "
            f"({args.data_v_range}); that is what makes the background grid disjoint "
            f"from the figure grid so the two can never coincide.")
    if args.bg_mode == "disjoint" and args.model == "felstm":
        warnings.warn(
            "--bg_mode disjoint puts the background outside felstm's velocity lattice, "
            "so felstm cannot represent the background motion at all while melstm can. "
            "That is a difference in expressive power confounded with the effect being "
            "measured. Prefer --bg_mode opposite for a three-way comparison.",
            UserWarning)
    if (args.bg_mode == "constant") != (args.bg_velocity is not None):
        raise SystemExit("--bg_velocity VX VY is required with --bg_mode constant, "
                         "and only meaningful with it.")
    if (args.bg_mode == "constant" and args.model == "felstm"
            and max(abs(c) for c in args.bg_velocity) > args.v_range):
        warnings.warn(
            f"--bg_velocity {args.bg_velocity} is outside felstm's lattice "
            f"(|v| <= {args.v_range}), so no felstm copy moves with the background "
            f"while melstm's slots can. Compare the models knowing that, or raise "
            f"--v_range (cost grows as (2R+1)^2).", UserWarning)
    # Resolved here, so the history json records the stream a run actually trained on.
    if args.train_stream_seed is None:
        args.train_stream_seed = args.data_seed
    if args.model == "felstm" and args.v_range < args.data_v_range:
        raise SystemExit(
            f"--v_range ({args.v_range}) < --data_v_range ({args.data_v_range}): "
            f"felstm's copies sit at fixed lattice velocities, so no copy could "
            f"represent a figure moving faster than {args.v_range}. Raise --v_range "
            f"(cost grows as (2R+1)^2) or lower --data_v_range.")

    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)
    if args.model_seed is not None:
        torch.manual_seed(args.model_seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu"))

    # MPS has no aten::grid_sampler_2d_backward, so melstm's warp cannot
    # backpropagate there at all. Swapping it for the exact integer-shift gather
    # makes local runs possible; it verifies itself against torch.roll first and
    # leaves fractional velocities on the original path.
    if device.type == "mps" and args.model == "melstm":
        res = enable_integer_shift_warp(device="mps")
        if res is not None:
            print(f"warp         : integer-shift gather enabled for MPS "
                  f"(exact vs torch.roll: {res[0]:.0e}, grid_sample differs by {res[1]:.1e})")

    run_name = args.run_name or f"cf_cls_{args.model}_{args.velocity_pool}_{int(time.time())}"
    models_dir = Path(args.save_dir) / "models"
    results_dir = Path(args.save_dir) / "results"
    state_dir = Path(args.save_dir) / "run_state"
    for d in (models_dir, results_dir, state_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(args)
    train_loader, val_loader, test_loader, state_loader = make_loaders(
        args, train_ds, val_ds, test_ds)

    model = build_classifier(args).to(device)
    report = model.parameter_report()
    print(f"run          : {run_name}")
    print(f"device       : {device}")
    print(f"pool         : {args.velocity_pool}")
    print(model.describe())
    print()
    bg_desc = {"opposite": "on the shared grid, opposing direction at t=0",
               "disjoint": f"|v| in [{args.bg_speed_min}, {args.bg_speed_max}] "
                           f"(disjoint grid)",
               "constant": f"fixed at v={tuple(args.bg_velocity or ())}",
               "incoherent": "fresh noise every frame (no motion)"}[args.bg_mode]
    print(f"data         : figure |v|<={args.data_v_range} ({args.motion_mode}), "
          f"background {bg_desc}, corr_len={args.corr_len}, T={args.seq_len}")
    print(f"seeds        : model {args.model_seed} (init + order), "
          f"train stream {args.train_stream_seed} (the sequences), data {args.data_seed} "
          f"(glyph split)")
    test_curve_batches = (math.ceil(args.test_curve_size / args.batch_size)
                          if args.test_curve_size else None)
    print(f"batches      : train {len(train_loader)}, val {len(val_loader)}, "
          f"test {len(test_loader)}  (chance accuracy = 10%)")
    print(f"glyphs       : train {len(train_loader.sampler)}, "
          f"val {len(val_loader.dataset)}, test {len(test_loader.dataset)} "
          f"(disjoint pools; test is MNIST's own split)")

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience)
        if args.use_lr_scheduler else None)
    criterion = nn.CrossEntropyLoss()

    wandb = None
    if args.use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   dir=args.wandb_dir, name=run_name, config=vars(args) | report)

    history = {"config": vars(args), "parameters": report, "epochs": []}
    best_val, best_epoch, since_improved = -1.0, -1, 0
    start_epoch, global_step = 0, 0

    curve_log_fn = None
    if wandb:
        # `epoch` as a selectable x axis in the UI; the logged step stays the
        # optimizer step so the per-batch curve and the per-epoch metrics agree.
        wandb.define_metric("epoch")
        curve_log_fn = lambda step, payload: wandb.log(payload, step=step)

    curve = None
    if args.val_curve_interval > 0:
        print(f"materialising {args.val_curve_size} fixed val sequences for the loss curve...")
        # val_ds, not test_ds: these points are logged as curve/val_* and are watched
        # live during a run, so a human reading them to judge progress is selecting on
        # whatever they measure. val is just as fixed a benchmark (random=False,
        # seed=777) and is the population the per-epoch number reports, so the two
        # series differ only in size and cadence.
        curve = ValCurveRecorder(val_ds, args.val_curve_size,
                                 args.val_curve_interval, device, args.batch_size,
                                 bn_dataset=Subset(train_loader.dataset,
                                                   train_loader.sampler.indices),
                                 bn_batches=args.val_curve_bn_batches)

    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_val, history = ck["best_val"], ck["history"]
        global_step = ck.get("global_step", 0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loader.sampler.set_epoch(epoch)     # fresh sequences, same for every model seed
        tr = run_epoch(model, train_loader, device, args.num_figures, criterion,
                       optimizer, args.grad_clip, curve=curve, global_step=global_step,
                       curve_log_fn=curve_log_fn)
        # BatchNorm's EMA lags the weights badly here, so re-estimate it from the
        # TRAINING distribution before measuring. Without this the val number
        # reflects statistics the model was never trained under.
        recompute_bn_stats(model, train_loader, device, args.precise_bn_batches)
        val_ds.reset_rng()          # fixed benchmark: identical sequences every epoch
        va = run_epoch(model, val_loader, device, args.num_figures, criterion)

        if scheduler:
            scheduler.step(va["acc"])

        # Test curve. Runs on the same weights and the same freshly recomputed
        # BatchNorm statistics as the val pass above -- nothing updates the model
        # between the two -- so the two curves are measured under one protocol and
        # differ only in which disjoint glyph pool they draw from. Capped at
        # test_curve_batches; the full test set is measured once at the end.
        te_row = {}
        if args.test_every and epoch % args.test_every == 0:
            test_ds.reset_rng()     # fixed benchmark, same sequences every epoch
            te_epoch = run_epoch(model, test_loader, device, args.num_figures, criterion,
                                 max_batches=test_curve_batches)
            te_epoch.pop("_global_step", None)
            te_row = {f"test_{k}": v for k, v in te_epoch.items()}

        global_step = tr.pop("_global_step", global_step)
        va.pop("_global_step", None)
        row = {"epoch": epoch, "time": time.time() - t0,
               "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()},
               **te_row}

        # Before the print and the log, so the shape IoU appears in both.
        if wandb and args.log_states_every and epoch % args.log_states_every == 0:
            row.update(log_states(model, state_loader, test_ds, device, args, epoch,
                                  global_step,
                                  save=(epoch == args.epochs - 1)) or {})
        history["epochs"].append(row)

        extra = "".join(f"  {k}={va[k]:.3f}" for k in
                        ("slot_hit_fig", "slot_hit_bg", "attn_on_fig") if k in va)
        if "val_state_shape_iou" in row:
            extra += (f"  shape_iou={row['val_state_shape_iou']:.3f}"
                      f"(chance {row['val_state_shape_iou_chance']:.3f})")
        te_desc = (f" | test loss {te_row['test_loss']:.4f} acc {te_row['test_acc']:.3f}"
                   if "test_acc" in te_row else "")
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} acc {tr['acc']:.3f} "
              f"| val loss {va['loss']:.4f} acc {va['acc']:.3f}{extra}{te_desc} "
              f"| {row['time']:.0f}s")

        if wandb:
            # step = optimizer step, shared with the fine-grained curve above.
            # `epoch` is in the payload, so the wandb UI can use it as the x axis.
            wandb.log(row, step=global_step)

        if va["acc"] > best_val:
            best_val, best_epoch, since_improved = va["acc"], epoch, 0
            torch.save({"model": model.state_dict(), "config": vars(args),
                        "epoch": epoch, "val_acc": best_val},
                       models_dir / f"{run_name}_best.pth")
        else:
            since_improved += 1

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_val": best_val, "history": history,
                    "global_step": global_step},
                   state_dir / f"checkpoint_{run_name}.pth")

        if args.early_stop_patience and since_improved >= args.early_stop_patience:
            print(f"early stop: no val improvement for {since_improved} epochs")
            break

    # --- final test on the best checkpoint
    ck = models_dir / f"{run_name}_best.pth"
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device)["model"])
    # The checkpoint stores the statistics that were current when it was saved,
    # which is what we want; recompute anyway so test matches val's protocol.
    recompute_bn_stats(model, train_loader, device, args.precise_bn_batches)
    test_ds.reset_rng()
    te = run_epoch(model, test_loader, device, args.num_figures, criterion,
                   collect_preds=bool(wandb))
    y_true, y_pred = te.pop("_y_true", None), te.pop("_y_pred", None)
    history["test"] = te
    history["best_val_acc"] = best_val
    history["best_epoch"] = best_epoch

    print(f"\nbest val acc {best_val:.3f} (epoch {best_epoch}) | "
          f"test acc {te['acc']:.3f} loss {te['loss']:.4f}")
    print("chance is 0.100 — read lstm's number first: if it is meaningfully "
          "above chance the dataset is leaking, not the model working.")

    if curve is not None and curve.steps:
        history["val_curve"] = curve.as_dict()

    with open(results_dir / f"history_{run_name}.json", "w") as f:
        json.dump(history, f, indent=2)

    if wandb:
        if y_true:
            # Which digits get confused with which. On this task a model reading
            # a per-frame cue tends to confuse by stroke thickness, while one
            # reading motion confuses by shape -- the structure of the errors
            # says more about the mechanism than the scalar does.
            last_step = global_step
            wandb.log({"test_confusion": wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred,
                class_names=[str(d) for d in range(10)])}, step=last_step)
            per_class = {}
            for t, pdt in zip(y_true, y_pred):
                a, b = per_class.setdefault(t, [0, 0])
                per_class[t] = [a + int(t == pdt), b + 1]
            for d, (c, n) in sorted(per_class.items()):
                wandb.summary[f"test_acc_digit{d}"] = c / max(n, 1)
        # Final numbers go in the summary, not the history: a step-less wandb.log
        # here would advance the counter past the last epoch for no benefit.
        #
        # This assignment lands AFTER every wandb.log above, so it overwrites the
        # value wandb mirrors into the summary from the per-epoch test curve. The
        # two are different quantities and the summary holds the one to quote: the
        # BEST-VAL CHECKPOINT's test accuracy, not the last epoch's.
        # report_test_accuracy.py reads exactly this key.
        wandb.summary["test_acc"] = te["acc"]
        wandb.summary["test_loss"] = te["loss"]
        wandb.summary["best_val_acc"] = best_val
        wandb.summary["best_epoch"] = best_epoch
        wandb.finish()

    # All --epochs completed (not just this Slurm submission's walltime slice):
    # marks the run finished so submit_classification.sbatch's self-chaining
    # knows to stop resubmitting. Same convention as train.py.
    if not args.smoke_test:
        done = state_dir / f"DONE_{args.model}_{run_name}.flag"
        done.touch()
        print(f"training complete ({args.epochs} epochs) — wrote {done}")

    return history


if __name__ == "__main__":
    main()
