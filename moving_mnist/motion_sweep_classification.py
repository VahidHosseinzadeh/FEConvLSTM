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
from train_classification import (velocity_diagnostics,                    # noqa: E402
                                  state_shape_iou)
from motion_factors_sweep import (D_GRID, S_GRID, TRAIN_D, TRAIN_S, TRAIN_N,  # noqa: E402
                                  TARGET_RATE, compensated_p)

# felstm carries (2R+1)^2 = 25 copies of the state, so it needs a smaller batch than
# the other two; 250 of them overruns a 20GB MPS budget outright.
BATCH = {"lstm": 250, "melstm": 250, "felstm": 50}


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


# Generalization axes. Both hold the TRAINING motion fixed and vary one thing the
# models never saw, so they measure extrapolation rather than a different motion law.
# The trained value is deliberately absent from each grid: the `reference` cell already
# measures it, and regenerating it would be a bit-identical duplicate.
LEN_GRID = (5, 8, 10,12,14,16,18, 20, 25, 30)          # trained at 15
SPEED_GRID = (1, 3, 4,5)                     # trained at 2

# m(d) is indexed by the velocity CHANGE dv = v_t - v_{t-1}, so the table must cover
# TWICE the largest speed in the sweep, not the speed itself. It was a hard-coded 4 --
# sized back when "velocities are +-2" -- which is why the E:v=3 and E:v=4 cells died
# with an out-of-bounds index. Derived from SPEED_GRID so widening that grid cannot
# reintroduce the crash.
U_SHIFT_R = 2 * max(SPEED_GRID + (TRAIN_N,))


def train_motion_kwargs(cfg):
    """The checkpoint's own motion law, as explicit kwargs."""
    return dict(motion_mode=cfg["motion_mode"], transition_mode=cfg["transition_mode"],
                min_segment=cfg["min_segment"], max_segment=cfg["max_segment"])


def generalization_cells(cfg, lengths=LEN_GRID, speeds=SPEED_GRID):
    """Context length and figure speed, one factor at a time.

    LENGTH is free at evaluation time: the encoder is recurrent and the head reads the
    final pooled state, so any T runs on a model trained at another one.

    SPEED is the lattice-coverage test. felstm's v_range is frozen at training, so at a
    figure speed above it part of the motion is literally outside what it can represent,
    while melstm re-estimates its slots per frame pair and is not bounded that way.
    """
    base = train_motion_kwargs(cfg)
    out = [(f"D:T={T}", "length", float(T), dict(base, seq_len=int(T)))
           for T in lengths if int(T) != int(cfg["seq_len"])]
    out += [(f"E:v={n}", "speed", float(n), dict(base, max_speed=int(n)))
            for n in speeds if int(n) != int(cfg["data_v_range"])]
    return out


def matched_rate_cell(train_rate, max_speed=TRAIN_N, s=TRAIN_S):
    """An axis-A cell whose realised switching rate equals the TRAINING rate.

    The reference cell is the model's own training motion, which for these classifiers
    is `piecewise` -- a different family from axis A's `stochastic` cells. Placing it on
    the axis by its measured rate still leaves an off-family point on the curve, and the
    step between it and its neighbours mixes a change of family with a change of rate.
    This cell is on the axis's OWN family at the same rate, so the curve has a legitimate
    point there and the training motion can be drawn as a separate marker.

    (motion_factors_sweep builds its centre this way from the start; it could, because
    the prediction models were trained on the stochastic family to begin with.)
    """
    p = compensated_p(train_rate, s, max_speed)
    if not 0.0 < p <= 1.0:
        raise SystemExit(
            f"cannot match a realised rate of {train_rate:.3f} in the stochastic family "
            f"at s={s}: it needs p_change={p:.3f}, outside (0, 1].")
    return ("A:matched", "rate", float("nan"),
            dict(neighbor_kernel="symmetric", motion_mode="stochastic",
                 transition_mode="smooth", p_change=p, smooth_probability=s,
                 max_speed=max_speed))


def measure_train_rate(cfg, seed, n, train_split=False, download=False):
    """Realised switching rate of the checkpoint's own training motion."""
    ds = make_dataset(cfg, None, seed, train_split, download)
    _, _, motion = materialise(ds, n)
    return motion_stats(motion)[0]


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
        # A constant background stays constant in every cell: the cells vary the
        # figure's motion law, and with a fixed background that is ALL they vary.
        bg_velocity=(tuple(cfg["bg_velocity"]) if cfg["bg_mode"] == "constant" else None),
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


def materialise_masked(ds, n):
    """Sequences, motion and the ground-truth figure mask, from one pass.

    The mask costs extra rendering, so it is drawn only for the diagnostic subset.
    Turning it on does NOT perturb the sample stream (verified: sequences and labels
    are identical with and without it), so these are the same first `n` sequences the
    accuracy pass scored.
    """
    ds.return_mask = True
    ds.reset_rng()
    items = [ds[i] for i in range(n)]
    seqs = torch.stack([it[0] for it in items])
    motion = torch.stack([torch.as_tensor(np.asarray(it[2])) for it in items])
    mask = torch.stack([torch.as_tensor(np.asarray(it[3])) for it in items])
    ds.return_mask = False
    return seqs, motion, mask


def diagnostics(net, seqs, motion, mask, n_figures, device, batch):
    """Per-cell mechanism diagnostics: does the state still hold the figure?

    slot_hit_fig / slot_hit_bg   melstm only -- did some tracked slot end the encoder
        holding the figure's (background's) true velocity? felstm's lattice contains
        every representable velocity by construction, so the question is vacuous there.
    shape_iou / _best / _chance  area-matched IoU between the local variance of the
        transported state and the true figure mask, for the velocity-MATCHED copy and
        for the best copy. Read it as transport coherence, not as task performance:
        felstm has reached 0.976 accuracy with a matched IoU at chance, because its
        information is spread over copies and time rather than concentrated in one.
        `_best` is biased toward models with more copies (best-of-25 vs best-of-2), so
        compare it WITHIN a model across regimes, never across models.
    """
    sums, n_batches = {}, 0
    for b in range(0, len(seqs), batch):
        s = seqs[b:b + batch].to(device)
        mo = motion[b:b + batch].to(device)
        mk = mask[b:b + batch].to(device)
        with torch.no_grad():
            _h, v, st = net.encode(s, return_states=True)
        d = dict(velocity_diagnostics({"velocities": v}, mo, n_figures))
        iou = state_shape_iou(net, st, v, mo, mk)
        if iou is not None:
            d["shape_iou"], d["shape_iou_best"] = iou
            d["shape_iou_chance"] = float(mk[:, -1].amax(1).mean())
        for k, val in d.items():
            sums[k] = sums.get(k, 0.0) + float(val)
        n_batches += 1
    return {k: v / n_batches for k, v in sums.items()} if n_batches else {}


DIAG_KEYS = ("slot_hit_fig", "slot_hit_bg", "shape_iou", "shape_iou_best",
             "shape_iou_chance")


def write_npz(dest, out, correct, models, n_sequences, diag=None,
              train_seq_len=0, train_max_speed=0):
    """Atomic rewrite of the whole result file.

    Called after EVERY cell, not once at the end: a job cut off by the walltime then
    leaves a usable partial file rather than nothing at all. tmp + os.replace so a kill
    mid-write cannot truncate the previous version.
    """
    tmp = f"{dest}.tmp.npz"
    extra = {}
    if diag:
        # One array per (model, metric), NaN where a metric does not apply -- slot_hit
        # is melstm-only, and shape_iou is undefined when no copy matches the figure.
        for m in models:
            for k in DIAG_KEYS:
                extra[f"diag_{m}_{k}"] = np.array(
                    [d.get(m, {}).get(k, np.nan) for d in diag], float)
    np.savez_compressed(
        tmp,
        regime=np.array(out["regime"]), axis=np.array(out["axis"]),
        level=np.array(out["level"], float), rate=np.array(out["rate"], float),
        jump=np.array(out["jump"], float), travel=np.array(out["travel"], float),
        u=np.array(out["u"], float), models=np.array(models),
        n_sequences=n_sequences, train_seq_len=train_seq_len,
        train_max_speed=train_max_speed,
        **{f"correct_{m}": np.stack(correct[m]) for m in models}, **extra)
    os.replace(tmp, dest)


def u_of(motion, M, r=U_SHIFT_R):
    """Fraction of the frame-to-frame image change the PREVIOUS velocity misses.

    Same statistic as the prediction figure's u axis: the image-space cost of assuming
    v_t = v_{t-1}, over the image-space cost of the motion itself.
    """
    v = np.asarray(motion, np.int64)
    dv = v[:, 1:] - v[:, :-1]
    lo, hi = -r, r
    if dv.min() < lo or dv.max() > hi or v.min() < lo or v.max() > hi:
        warnings.warn(f"u_of: velocities/changes exceed the m(d) table (|d|<={r}); "
                      f"clipping. Raise U_SHIFT_R if u matters for this run.")
    A = float(M[np.clip(dv[..., 0], lo, hi) + r, np.clip(dv[..., 1], lo, hi) + r].mean())
    B = float(M[np.clip(v[..., 0], lo, hi) + r, np.clip(v[..., 1], lo, hi) + r].mean())
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
    p.add_argument('--generalization', action='store_true',
                   help='Also sweep CONTEXT LENGTH and FIGURE SPEED, holding the training '
                        'motion fixed. Both are evaluation-only extrapolation tests: the '
                        'encoder is recurrent so any T runs, and felstm\'s velocity '
                        'lattice is frozen at training so speeds above it are outside '
                        'what it can represent at all.')
    p.add_argument('--lengths', type=int, nargs='+', default=list(LEN_GRID))
    p.add_argument('--speeds', type=int, nargs='+', default=list(SPEED_GRID))
    p.add_argument('--with_diagnostics', action='store_true',
                   help='Also record, per cell, whether the STATE still holds the figure: '
                        'melstm slot-hit rates and the area-matched shape IoU between the '
                        'transported state and the true mask. Answers WHY a model holds up '
                        'rather than only THAT it does. Needs the figure mask, so it runs '
                        'on a small subset.')
    p.add_argument('--diag_sequences', type=int, default=256,
                   help='Sequences per cell for the diagnostics. Far fewer than the '
                        'accuracy pass needs: these are means of per-sample rates, not '
                        'a 1-in-50 accuracy difference.')
    p.add_argument('--diag_batch', type=int, default=32,
                   help='The state tensor is (B, T, V, H, W); at felstm 25 copies this is '
                        'what keeps it in memory.')
    p.add_argument('--only', nargs='+', default=None,
                   help='Recompute ONLY these cells by name (e.g. E:v=3 E:v=4) and merge '
                        'them into the existing .npz, leaving every other cell alone. '
                        'For recovering from a crash without repeating the whole sweep.')
    p.add_argument('--append_matched_rate', action='store_true',
                   help="Evaluate ONE extra axis-A cell whose realised switching rate "
                        "equals the training motion's, in the axis's own stochastic "
                        "family, and merge it into the existing .npz. The other cells are "
                        'not recomputed. Gives the rate curve a legitimate point at the '
                        'training rate, since the reference cell is a piecewise motion '
                        'and so sits off that curve.')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--limit_cells', type=int, default=None, help='For a quick check.')
    args = p.parse_args()

    dev = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    if dev.type == "mps":
        enable_integer_shift_warp(device="mps")
    # Printed loudly: on CPU felstm's 25 transported copies make this hundreds of times
    # slower, which looks like a hang rather than a slow run. If this says "cpu" on a
    # cluster you are on the login node -- submit with sbatch instead.
    print(f"device       : {dev}"
          + ("   <-- NO GPU FOUND; felstm will be impractically slow"
             if dev.type == "cpu" else ""), flush=True)

    runs = find_runs(args.save_dir, args.models, args.run_filter)
    nets = {m: load_model(cfg, ck, dev) for m, (cfg, ck, _) in runs.items()}
    ref_cfg = runs[args.models[0]][0]
    for m, (cfg, _, run) in runs.items():
        print(f"{m:7s} {run}  seq_len={cfg['seq_len']} head={cfg['head_channels']} "
              f"trained motion={cfg['motion_mode']} {cfg['min_segment']}-{cfg['max_segment']} "
              f"{cfg['transition_mode']}")

    # Every cell is generated ONCE from ref_cfg and shown to all three models, which is
    # only legitimate if they were trained on the same data. A --run_filter that picks,
    # say, a seq_len=10 lstm beside a seq_len=15 felstm would otherwise be evaluated
    # silently on the wrong sequences.
    for key in ("seq_len", "image_size", "num_figures", "variant", "corr_len",
                "digit_scale", "normalize", "data_v_range", "bg_mode", "bg_velocity",
                "motion_mode", "transition_mode", "min_segment", "max_segment"):
        # .get: bg_velocity is absent from configs that predate --bg_mode constant.
        seen = {m: cfg.get(key) for m, (cfg, _, _) in runs.items()}
        if len(set(map(str, seen.values()))) > 1:
            raise SystemExit(
                f"the selected runs disagree on {key!r}: {seen}\n"
                f"they were not trained on the same data, so one sweep cannot compare "
                f"them. Narrow --run_filter to one matched set of arms.")

    todo = cells()
    if args.generalization:
        todo = todo + generalization_cells(ref_cfg, args.lengths, args.speeds)
    if args.limit_cells:
        todo = todo[:args.limit_cells]
    out = {"regime": [], "axis": [], "level": [], "rate": [], "jump": [], "travel": [], "u": []}
    correct = {m: [] for m in args.models}
    diag = [] if args.with_diagnostics else None
    M = None

    # The filter goes in the filename: sweeping several arms out of one save_dir
    # (s1/s2/s3, head16 vs head32) would otherwise have each run overwrite the last.
    stem = "motion_sweep_classification"
    if args.run_filter:
        stem += "_" + slug(args.run_filter)
    dest = args.out or os.path.join(args.save_dir, stem + ".npz")

    if args.append_matched_rate or args.only:
        # Recompute a FEW cells and merge them into the existing file, leaving the rest
        # untouched. This is how you recover from a crash in one cell without paying for
        # the whole sweep again.
        if not os.path.exists(dest):
            raise SystemExit(f"{dest} does not exist -- run the full sweep first.")
        prev = np.load(dest, allow_pickle=False)
        if int(prev["n_sequences"]) != args.n_sequences:
            raise SystemExit(
                f"{dest} holds {int(prev['n_sequences'])} sequences per cell but this run "
                f"asks for {args.n_sequences}; the per-sequence arrays would not line up.")
        if args.append_matched_rate:
            train_rate = measure_train_rate(ref_cfg, args.data_seed, args.n_sequences,
                                            args.train_split, args.download)
            todo = [matched_rate_cell(train_rate)]
        else:
            want = set(args.only)
            known = {c[0] for c in todo}
            if want - known:
                raise SystemExit(
                    f"unknown cell(s) {sorted(want - known)}.\nthis sweep defines:\n  "
                    + "\n  ".join(sorted(known)))
            todo = [c for c in todo if c[0] in want]
        # Whatever is being recomputed is dropped first, so a rerun replaces rather
        # than duplicates.
        recompute = {c[0] for c in todo}
        keep = [i for i, r in enumerate(prev["regime"]) if str(r) not in recompute]
        for k in ("regime", "axis", "level", "rate", "jump", "travel", "u"):
            out[k] = list(prev[k][keep])
        for m in args.models:
            correct[m] = list(prev[f"correct_{m}"][keep])
        if diag is not None:
            diag = [{m: {k: float(prev[f"diag_{m}_{k}"][i]) for k in DIAG_KEYS
                         if f"diag_{m}_{k}" in prev.files}
                     for m in args.models} for i in keep]
        print(f"merging into {dest}: keeping {len(keep)} cells, recomputing "
              + ", ".join(sorted(recompute)))
        if args.append_matched_rate:
            print(f"training motion's realised rate: {train_rate:.4f}  ->  matched "
                  f"stochastic cell p_change={todo[0][3]['p_change']:.4f}")

    for i, (regime, axis, level, kw) in enumerate(todo):
        t0 = time.time()
        # Announced BEFORE the work, not after: a cell at 5000 sequences takes minutes
        # (materialising alone is ~40s) and the result line only lands at the end, which
        # makes a healthy run look hung.
        print(f"[{i + 1:2d}/{len(todo)}] {regime:18s} generating {args.n_sequences} "
              f"sequences ...", end="", flush=True)
        ds = make_dataset(ref_cfg, kw, args.data_seed, args.train_split, args.download)
        seqs, labels, motion = materialise(ds, args.n_sequences)
        print(f" {time.time() - t0:.0f}s, evaluating ...", end="", flush=True)
        if M is None:
            # An image statistic. Flattened to (N, H, W) first: shift_msd rolls axes
            # (1, 2), so handing it (N, T, H, W) would roll along TIME, not height.
            f = seqs[:200, :, 0].numpy()
            M = shift_msd(f.reshape(-1, *f.shape[-2:]))
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
        if diag is not None:
            dseqs, dmot, dmask = materialise_masked(ds, min(args.diag_sequences,
                                                            args.n_sequences))
            cell_diag = {m: diagnostics(nets[m], dseqs, dmot, dmask, ref_cfg["num_figures"],
                                        dev, args.diag_batch) for m in args.models}
            diag.append(cell_diag)
            accs.append("| " + "  ".join(
                f"{m}:iou {cell_diag[m].get('shape_iou', float('nan')):.2f}"
                + (f" hit {cell_diag[m]['slot_hit_fig']:.2f}"
                   if "slot_hit_fig" in cell_diag[m] else "")
                for m in args.models))
        write_npz(dest, out, correct, args.models, args.n_sequences, diag,
                  ref_cfg["seq_len"], ref_cfg["data_v_range"])
        print(f"\r[{i + 1:2d}/{len(todo)}] {regime:18s} rate={rate:.3f} jump={jump:.2f} "
              f"u={out['u'][-1]:.3f} travel={travel:4.1f}px | " + "  ".join(accs)
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nwrote {dest}  ({len(todo)} cells)")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
