#!/usr/bin/env python
"""
Motion-regime sensitivity for the Common-Fate CLASSIFICATION models. Evaluation only.

The classification counterpart of motion_factors_sweep.py / motion_difficulty_sweep.py.
Same three axes and the same one-factor construction -- the rate and jump grids and the
named regimes are IMPORTED from those scripts rather than restated, so the x positions
mean the same thing in both papers' figures -- but the metric is ACCURACY on the
motion-defined digit instead of rollout MSE.

What varies and what does not
-----------------------------
Only the motion kwargs change between cells. Canvas, sequence length, texture, the
figure/background separation rule and the glyph pool are taken from each checkpoint's
own training config, so a cell differs from the training distribution in motion alone.

All three models see the SAME sequences in a cell: each regime is materialised once and
the three are evaluated on that tensor. The comparison is therefore paired, which is
what lets a small accuracy difference mean anything at n=1000.

BatchNorm is NOT recomputed per regime. The checkpoint carries statistics re-estimated
from the TRAINING distribution (train_classification.py recomputes them before the val
pass that selects the checkpoint), and refitting them to a shifted regime would be
test-time adaptation -- the model would partly absorb the very shift being measured.

The reference cell
------------------
Not the prediction sweep's stochastic d=0.50 centre: these classifiers were trained on
a piecewise regime, so the reference is each checkpoint's own training motion, and it is
placed on every axis at its MEASURED rate / jump / u rather than at a nominal grid slot.

    python moving_mnist/motion_sweep_classification.py \
        --save_dir cluster_runs/experiments_classification_s1_leng10_head32
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common_fate_moving_mnist_dataset import CommonFateMovingMNISTDataset  # noqa: E402
from motion_classification_model import build_classifier                   # noqa: E402
from mps_integer_warp import enable_integer_shift_warp                     # noqa: E402
from motion_difficulty_sweep import NAMED_REGIMES                          # noqa: E402
from motion_factors_sweep import (D_GRID, S_GRID, TRAIN_D, TRAIN_S, TRAIN_N,  # noqa: E402
                                  TARGET_RATE, compensated_p)

# felstm carries (2R+1)^2 = 25 copies of the state, so it needs a smaller batch than
# the other two; 250 of them overruns a 20GB MPS budget outright.
BATCH = {"lstm": 250, "melstm": 250, "felstm": 50}
U_SHIFT_R = 4        # m(d) is needed for d in [-4, 4]^2, since velocities are +-2


# ------------------------------------------------------------------- the cells
def cells():
    """(regime, axis, level, motion kwargs) for every cell of the three axes.

    'reference' carries axis='reference' and no kwargs: it is the checkpoint's own
    training motion, filled in per model from its config.
    """
    base = dict(neighbor_kernel="symmetric")
    out = [("reference", "reference", float("nan"), None)]
    for d in D_GRID:
        out.append((f"A:d={d:.2f}", "rate", float(d),
                    dict(base, motion_mode="stochastic", transition_mode="smooth",
                         p_change=d ** 2.170, smooth_probability=TRAIN_S,
                         max_speed=TRAIN_N)))
    for s in S_GRID:
        out.append((f"B:s={s:.2f}", "jump", float(s),
                    dict(base, motion_mode="stochastic", transition_mode="smooth",
                         p_change=compensated_p(TARGET_RATE, s, TRAIN_N),
                         smooth_probability=s, max_speed=TRAIN_N)))
    for name, kw in NAMED_REGIMES.items():
        out.append((f"C:{name}", "named", float("nan"), dict(kw, **base)))
    return out


def slug(regime):
    return re.sub(r"[^A-Za-z0-9]+", "_", regime).strip("_")


# ------------------------------------------------------------------ the models
def find_runs(save_dir, models, run_filter=""):
    """One (config, checkpoint, run name) per model, from the history json beside it.

    Ambiguity is an ERROR, not a silent pick. A training save_dir accumulates every arm
    ever run (head2, head32_e70, ...), and quietly evaluating whichever one sorted first
    would attribute one arm's sensitivity to another. --run_filter narrows it.
    """
    out = {}
    for m in models:
        hits = sorted(g for g in glob.glob(
            os.path.join(save_dir, "results", f"history_*_{m}_*.json"))
            if run_filter in os.path.basename(g))
        if not hits:
            raise SystemExit(
                f"no history_*_{m}_*.json in {save_dir}/results"
                + (f" matching {run_filter!r}" if run_filter else ""))
        if len(hits) > 1:
            names = "\n  ".join(os.path.basename(h) for h in hits)
            raise SystemExit(
                f"{len(hits)} runs match model {m!r} in {save_dir}/results:\n  {names}\n"
                f"pass --run_filter with a substring that picks exactly one.")
        cfg = json.load(open(hits[0]))["config"]
        run = os.path.basename(hits[0])[len("history_"):-len(".json")]
        ck = os.path.join(save_dir, "models", f"{run}_best.pth")
        if not os.path.exists(ck):
            raise SystemExit(f"missing checkpoint {ck}")
        out[m] = (cfg, ck, run)
    return out


def load_model(cfg, ck, device):
    net = build_classifier(cfg).to(device)
    net.load_state_dict(torch.load(ck, map_location=device)["model"])
    net.eval()
    return net


# -------------------------------------------------------------------- the data
def make_dataset(cfg, motion_kw, seed, train_split, download=False):
    """The checkpoint's own data config, with only the motion knobs replaced."""
    common = dict(
        root=str(_HERE.parent / "data"), train=train_split, random=False, seed=seed,
        download=download, image_size=cfg["image_size"], seq_len=cfg["seq_len"],
        num_figures=cfg["num_figures"], variant=cfg["variant"], corr_len=cfg["corr_len"],
        digit_scale=cfg["digit_scale"], normalize=cfg["normalize"],
        max_speed=cfg["data_v_range"],
        bg_opposite_at_start=(cfg["bg_mode"] == "opposite"),
        bg_speed_range=((cfg["bg_speed_min"], cfg["bg_speed_max"])
                        if cfg["bg_mode"] == "disjoint" else None),
        return_motion=True,
    )
    if motion_kw is None:                      # the training motion itself
        common.update(motion_mode=cfg["motion_mode"],
                      transition_mode=cfg["transition_mode"],
                      min_segment=cfg["min_segment"], max_segment=cfg["max_segment"])
    else:
        common.update(motion_kw)
        common.setdefault("max_speed", cfg["data_v_range"])
    return CommonFateMovingMNISTDataset(**common)


def materialise(ds, n):
    """n sequences, their labels and their figure motion, from one pass.

    One pass matters: this dataset is stateful, and without `digit_indices` the GLYPH
    is drawn from the same RNG as the motion. Reading sequences and labels in separate
    passes silently pairs a sequence with another sample's label -- which looks exactly
    like a model at chance.
    """
    ds.reset_rng()
    items = [ds[i] for i in range(n)]
    seqs = torch.stack([it[0] for it in items])
    labels = torch.tensor([int(it[1]) for it in items], dtype=torch.long)
    motion = np.stack([np.asarray(it[2])[:, 0, :] for it in items])   # figure layer
    return seqs, labels, motion


# ------------------------------------------------------------------- the stats
def motion_stats(motion):
    """Realised switching rate and mean jump size, over the context transitions."""
    v = np.asarray(motion, np.int64)
    dv = v[:, 1:] - v[:, :-1]
    changed = np.abs(dv).max(-1) > 0
    rate = float(changed.mean())
    jump = float(np.abs(dv).max(-1)[changed].mean()) if changed.any() else 0.0
    travel = float(np.abs(v[:, :-1].sum(1)).sum(-1).mean())
    return rate, jump, travel


def shift_msd(frames, r=U_SHIFT_R):
    """m(d): mean squared difference between a frame and itself shifted by d."""
    F = np.asarray(frames, np.float64)
    M = np.zeros((2 * r + 1, 2 * r + 1))
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            M[dx + r, dy + r] = ((F - np.roll(F, (dy, dx), axis=(1, 2))) ** 2).mean()
    return M


def write_npz(dest, out, correct, models, n_sequences):
    """Atomic rewrite of the whole result file.

    Called after EVERY cell, not once at the end: a job cut off by the walltime then
    leaves a usable partial file rather than nothing at all. tmp + os.replace so a kill
    mid-write cannot truncate the previous version.
    """
    tmp = f"{dest}.tmp.npz"
    np.savez_compressed(
        tmp,
        regime=np.array(out["regime"]), axis=np.array(out["axis"]),
        level=np.array(out["level"], float), rate=np.array(out["rate"], float),
        jump=np.array(out["jump"], float), travel=np.array(out["travel"], float),
        u=np.array(out["u"], float), models=np.array(models),
        n_sequences=n_sequences,
        **{f"correct_{m}": np.stack(correct[m]) for m in models})
    os.replace(tmp, dest)


def u_of(motion, M, r=U_SHIFT_R):
    """Fraction of the frame-to-frame image change the PREVIOUS velocity misses.

    Same statistic as the prediction figure's u axis: the image-space cost of assuming
    v_t = v_{t-1}, over the image-space cost of the motion itself.
    """
    v = np.asarray(motion, np.int64)
    dv = v[:, 1:] - v[:, :-1]
    A = float(M[dv[..., 0] + r, dv[..., 1] + r].mean())
    B = float(M[v[..., 0] + r, v[..., 1] + r].mean())
    return A / B


# --------------------------------------------------------------------- the run
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--save_dir', default='cluster_runs/experiments_classification_s1_leng10_head32')
    p.add_argument('--models', nargs='+', default=['lstm', 'felstm', 'melstm'])
    p.add_argument('--run_filter', default='',
                   help='Substring the run name must contain, when a save_dir holds more '
                        'than one arm per model (e.g. "s1_stress_test").')
    p.add_argument('--n_sequences', type=int, default=1000)
    p.add_argument('--data_seed', type=int, default=123,
                   help='Seed of the evaluation benchmark. 123 is what train_classification.py '
                        'uses for its test set, so the reference cell reproduces the reported '
                        'test accuracy.')
    p.add_argument('--train_split', action='store_true',
                   help='Draw glyphs from MNIST train instead of test. Off by default: test '
                        'is the disjoint pool the reported numbers use.')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Override the per-model batch. The built-in felstm batch is 50, '
                        'which is an MPS-memory workaround; a cluster GPU wants far more '
                        'and is much faster for it.')
    p.add_argument('--download', action='store_true',
                   help='Let torchvision fetch MNIST if ./data is empty.')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--limit_cells', type=int, default=None, help='For a quick check.')
    args = p.parse_args()

    dev = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    if dev.type == "mps":
        enable_integer_shift_warp(device="mps")

    runs = find_runs(args.save_dir, args.models, args.run_filter)
    nets = {m: load_model(cfg, ck, dev) for m, (cfg, ck, _) in runs.items()}
    ref_cfg = runs[args.models[0]][0]
    for m, (cfg, _, run) in runs.items():
        print(f"{m:7s} {run}  seq_len={cfg['seq_len']} head={cfg['head_channels']} "
              f"trained motion={cfg['motion_mode']} {cfg['min_segment']}-{cfg['max_segment']} "
              f"{cfg['transition_mode']}")

    todo = cells()[:args.limit_cells] if args.limit_cells else cells()
    out = {"regime": [], "axis": [], "level": [], "rate": [], "jump": [], "travel": [], "u": []}
    correct = {m: [] for m in args.models}
    M = None

    dest = args.out or os.path.join(args.save_dir, "motion_sweep_classification.npz")
    for i, (regime, axis, level, kw) in enumerate(todo):
        t0 = time.time()
        ds = make_dataset(ref_cfg, kw, args.data_seed, args.train_split, args.download)
        seqs, labels, motion = materialise(ds, args.n_sequences)
        if M is None:                      # an image statistic: one regime's frames suffice
            M = shift_msd(seqs[:200, :, 0].numpy())
        rate, jump, travel = motion_stats(motion)
        out["regime"].append(regime); out["axis"].append(axis); out["level"].append(level)
        out["rate"].append(rate); out["jump"].append(jump); out["travel"].append(travel)
        out["u"].append(u_of(motion, M))
        accs = []
        for m in args.models:
            B = args.batch_size or BATCH.get(m, 100)
            ok = np.empty(len(labels), bool)
            with torch.no_grad():
                for b in range(0, len(labels), B):
                    pred = nets[m](seqs[b:b + B].to(dev)).argmax(1).cpu()
                    ok[b:b + B] = (pred == labels[b:b + B]).numpy()
            correct[m].append(ok)
            accs.append(f"{m} {ok.mean():.4f}")
        write_npz(dest, out, correct, args.models, args.n_sequences)
        print(f"[{i + 1:2d}/{len(todo)}] {regime:18s} rate={rate:.3f} jump={jump:.2f} "
              f"u={out['u'][-1]:.3f} travel={travel:4.1f}px | " + "  ".join(accs)
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nwrote {dest}  ({len(todo)} cells)")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
