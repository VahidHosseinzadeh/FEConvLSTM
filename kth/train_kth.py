#!/usr/bin/env python
"""
KTH action recognition under camera motion: ConvLSTM vs FEConvLSTM vs MEConvLSTM.

    python -m kth.train_kth --model melstm --camera none --smoke_test --states_fig_dir /tmp/kth
    python -m kth.train_kth --model felstm --camera piecewise --use_wandb

The data protocol is Keller's FERNN KTH setup (kth_dataset.py); the training is the Motion-Only
MNIST classifier's (moving_mnist/train_classification.py): the same backbones, velocity pool and
conv head, cross-entropy, precise BatchNorm before every evaluation, and the test number read off
the best-val checkpoint (the last epoch's is reported too). The three models have the same trained
parameter count by construction -- one recurrent cell shared by all velocity copies / slots, one
head -- and the table is printed at the start of every run.

Camera conditions
-----------------
--camera is the TRAINING condition, and val uses it too (model selection in distribution).
--test_conditions lists the test sets the final checkpoints are scored on. They share clips and
windows, so they differ only in the camera: a constant-trained model scored on 'piecewise' is the
constant -> time-varying transfer. 'mode:R' sets that test's velocity range ('constant:2' =
Keller's V_2, outside FEConvLSTM's 9-copy lattice).

Defaults
--------
hidden 64 with a 64/128 head is 278,278 trained parameters for every model; Keller's G-RNN /
FERNN at this protocol is ~0.24 M. Batch 32, 500 epochs (~24k steps) and Adam 3e-4 are his; the
cosine decay to 1e-5 is ours (a fixed schedule, as in the recent MNIST arms). The head MLP is
LeakyReLU: a ReLU head MLP can die (every chance-level MNIST classifier run did), which here would
show as a train loss flat at ln 6 = 1.792.

Diagnostics (melstm, every step of every clip)
----------------------------------------------
cam_hit      some slot moves at the true camera velocity (with no camera motion: some slot at 0).
person_hit   some slot is within 0.5 px of the person's apparent velocity, over the steps where the
             person proxy is valid and the person visibly moves (> 0.5 px against the scene);
             also per locomotion class (walking / jogging / running).
pool_share   (felstm, melstm; from the logged state samples) the share of the head's max-pooled
             input each copy / slot supplies.
"""
import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from motion_benchmarks import _repo  # noqa: E402,F401
from motion_benchmarks.common.deadline import Deadline, add_deadline_args  # noqa: E402
from motion_classification_model import recompute_bn_stats  # noqa: E402
from mps_integer_warp import enable_integer_shift_warp  # noqa: E402
from velocity_predictor_model import PhaseCorrelation  # noqa: E402

from kth.camera_motion import MODES, CameraMotion  # noqa: E402
from kth.kth_dataset import KTH_ACTIONS, KTHClips, KTHStore  # noqa: E402
from kth.kth_model import build_kth_classifier, record_states  # noqa: E402

LOCOMOTION = ("walking", "jogging", "running")
PANEL_ORDER = ("running", "jogging", "walking", "boxing", "handwaving", "handclapping")


# ------------------------------------------------------------------ arguments
def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- model
    p.add_argument("--model", choices=["lstm", "felstm", "melstm"], default="melstm")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--v_range", type=int, default=1,
                   help="felstm: lattice half-width R, (2R+1)^2 copies; 1 = the 9-velocity V_1")
    p.add_argument("--num_vel_modes", type=int, default=4, help="melstm: K slots")
    p.add_argument("--velocity_source", choices=["frame_pair", "bootstrap", "tracked"],
                   default="frame_pair")
    p.add_argument("--slot_assign", choices=["nearest", "shift", "anchored"], default="nearest",
                   help="melstm frame-pair candidates -> slots: nearest previous velocity; "
                        "nearest after the best common shift (equivariant to time-varying camera "
                        "motion); or slot 0 = the top peak and the rest nearest (kth_model.py)")
    p.add_argument("--static_slot", type=int, default=0,
                   help="melstm (any velocity source): 1 = slot 0 pinned to velocity (0, 0), a "
                        "ConvLSTM state inside the MEConvLSTM; slots 1..K-1 behave exactly like "
                        "the (K-1)-slot model's. Use K+1 slots to keep K moving ones")
    p.add_argument("--readout_steps", type=int, default=1,
                   help="average the head's logits over the last N encoder steps (1 = h_T only, "
                        "the original). Every model and velocity source, the handover included")
    p.add_argument("--x_curriculum_epochs", type=int, default=0,
                   help="melstm --velocity_source tracked: the stochastic handover -- training "
                        "steps take the frame-pair velocity with probability p(epoch), 1 -> 0 "
                        "over this many epochs; evaluation is always pure tracking. 0 = off")
    p.add_argument("--x_curriculum_shape", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--pc_subpixel", type=int, default=1,
                   help="1 (default since set 2) = parabolic sub-pixel peaks, with the padded "
                        "(interpolating) warp; 0 = whole-pixel peaks and the exact integer warp. "
                        "Set 1 ran 0; at seed 1 the two were within 2.3 pt on every test set")
    p.add_argument("--pc_suppress_radius", type=int, default=0,
                   help="Chebyshev radius cleared around each phase-correlation peak before the "
                        "next is taken (0 = distinct pixels, the original top-K)")
    p.add_argument("--pc_search_radius", type=int, default=5,
                   help="only displacements with |v|_inf <= this (<= 0: the whole plane). Measured "
                        "on train, no camera (pc_analysis.py): the whole plane gives ~10%% of slot "
                        "velocities beyond 5 px/step -- noise peaks, nothing in KTH moves that fast "
                        "at 32x32 -- while 5 removes them and keeps runners (~2-3 px/step) plus a "
                        "V_2 camera in range")
    p.add_argument("--pc_alpha", type=float, default=1.0)
    p.add_argument("--forget_bias", type=float, default=None)
    p.add_argument("--velocity_pool", choices=["max", "attention", "mean"], default="max")
    p.add_argument("--head_channels", type=int, default=64)
    p.add_argument("--head_blocks", type=int, default=3)
    p.add_argument("--head_mlp_hidden", type=int, default=128)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--head_mlp_act", choices=["relu", "leaky"], default="leaky")
    p.add_argument("--head_norm", choices=["batch", "group", "none"], default="batch")
    p.add_argument("--precise_bn_batches", type=int, default=30)

    # --- data
    p.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent.parent
                                                    / "data" / "kth"))
    p.add_argument("--split_scheme", choices=["keller", "official"], default="keller")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--eval_windows", type=int, default=1,
                   help="fixed windows per val/test clip (Keller: one random window)")
    p.add_argument("--camera", choices=MODES, default="none",
                   help="TRAINING (and val) camera condition")
    p.add_argument("--camera_v_range", type=int, default=1)
    p.add_argument("--camera_keller_draws", type=int, default=1,
                   help="constant: Keller's own velocity per clip (RandomState(42))")
    p.add_argument("--camera_transition", choices=["uniform", "smooth"], default="uniform")
    p.add_argument("--camera_neighbor_kernel", choices=["legacy", "symmetric"], default="legacy")
    p.add_argument("--camera_min_segment", type=int, default=3)
    p.add_argument("--camera_max_segment", type=int, default=6)
    p.add_argument("--camera_p_change", type=float, default=0.25, help="stochastic mode")
    p.add_argument("--shake_amp", type=float, nargs=2, default=(1.0, 3.0),
                   help="shake amplitude range, px (capped by --shake_vmax)")
    p.add_argument("--shake_period", type=float, nargs=2, default=(6.0, 16.0),
                   help="shake period range, frames (12.5 / P Hz at 25 fps, step 2)")
    p.add_argument("--shake_axes", choices=["both", "x", "y"], default="both")
    p.add_argument("--shake_vmax", type=int, default=1,
                   help="largest whole-pixel shake step; 1 keeps the shake on V_1 (0 = no cap)")
    p.add_argument("--resample_camera", action="store_true",
                   help="draw a new trajectory on every training access (Keller fixes one per clip)")
    p.add_argument("--test_conditions", type=str,
                   default="none,constant,piecewise,shake,constant:2",
                   help="comma list of camera modes for the final test sets; 'mode:R' sets R")

    # --- optimisation
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--warmup_epochs", type=float, default=0.0)
    p.add_argument("--final_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--test_every", type=int, default=1,
                   help="in-distribution test curve every N epochs (diagnostic; selection is val)")

    # --- bookkeeping
    p.add_argument("--data_seed", type=int, default=42,
                   help="fixes the eval windows and the camera trajectories: hold it fixed")
    p.add_argument("--model_seed", type=int, default=None,
                   help="weights, training order, windows and flips; vary this across seeds")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="./experiments_kth")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="FERNN-kth")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_dir", type=str, default="./tmp/")
    p.add_argument("--log_states_every", type=int, default=25,
                   help="state panels every N epochs (and the last); few, for the home quota")
    p.add_argument("--log_states_samples", type=int, default=6,
                   help="panels per logging, one test clip per action")
    p.add_argument("--state_metric_samples", type=int, default=48,
                   help="test clips behind the pool-share numbers")
    p.add_argument("--states_fig_dir", type=str, default=None,
                   help="also write the panels here as PNG")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--smoke_test", action="store_true",
                   help="2 epochs on a few batches, panels every epoch; for the plumbing")
    add_deadline_args(p)
    args = p.parse_args(argv)
    if args.x_curriculum_epochs:
        if args.model != "melstm" or args.velocity_source != "tracked":
            p.error("--x_curriculum_epochs needs --model melstm --velocity_source tracked")
        if args.x_curriculum_epochs < 2:
            p.error("--x_curriculum_epochs must be >= 2")
    if args.smoke_test:
        args.epochs, args.num_workers, args.log_states_every = 2, 0, 1
        args.precise_bn_batches = 2
        args.state_metric_samples = min(args.state_metric_samples, 12)
    if args.run_name is None:
        args.run_name = f"kth_{args.model}_{args.camera}_{int(time.time())}"
    return args


def make_camera(args, mode, v_range=None):
    return CameraMotion(
        mode, v_range=args.camera_v_range if v_range is None else v_range,
        seq_len=args.seq_len, transition=args.camera_transition,
        min_segment=args.camera_min_segment, max_segment=args.camera_max_segment,
        neighbor_kernel=args.camera_neighbor_kernel, p_change=args.camera_p_change,
        shake_amp=args.shake_amp, shake_period=args.shake_period, shake_axes=args.shake_axes,
        shake_vmax=args.shake_vmax, keller_draws=bool(args.camera_keller_draws))


def parse_conditions(args):
    """'none,constant,constant:2' -> {name: CameraMotion}; the training condition is included."""
    out = {}
    for tok in [t.strip() for t in args.test_conditions.split(",") if t.strip()]:
        mode, _, r = tok.partition(":")
        R = int(r) if r else args.camera_v_range
        name = mode if R == args.camera_v_range or mode == "none" else f"{mode}_v{R}"
        out[name] = make_camera(args, mode, R)
    if args.camera not in out:
        out[args.camera] = make_camera(args, args.camera)
    return out


# ---------------------------------------------------------------- diagnostics
class VelocityStats:
    """
    melstm slot velocities against the truth, accumulated over every encoder step of every clip.
    vel[:, t-1] is the velocity a slot was transported by into frame t; its truth is the camera
    step motion[:, t-1] and the person proxy's step person[:, t-1].
    """

    def __init__(self):
        self.s = defaultdict(float)

    @torch.no_grad()
    def update(self, vel, motion, person, label):
        cam = motion[:, :-1, 0]                                           # (B, T-1, 2)
        hit = (vel.round() == cam[:, :, None]).all(-1)                    # (B, T-1, K)
        self.s["cam_hit"] += hit.any(-1).sum().item()
        self.s["cam_hit_slot0"] += hit[..., 0].sum().item()
        self.s["n_steps"] += hit.shape[0] * hit.shape[1]
        p = person[:, :-1]
        moving = torch.isfinite(p).all(-1) & ((p - cam).abs().amax(-1) > 0.5)
        near = ((vel - p[:, :, None]).abs().amax(-1) <= 0.5).any(-1)
        self.s["person_hit"] += (near & moving).sum().item()
        self.s["n_person"] += moving.sum().item()
        for name in LOCOMOTION:
            m = moving & (label == KTH_ACTIONS.index(name))[:, None]
            self.s[f"person_hit_{name}"] += (near & m).sum().item()
            self.s[f"n_person_{name}"] += m.sum().item()

    def result(self):
        if not self.s:
            return {}
        out = {"cam_hit": self.s["cam_hit"] / max(self.s["n_steps"], 1),
               "cam_hit_slot0": self.s["cam_hit_slot0"] / max(self.s["n_steps"], 1)}
        if self.s["n_person"]:
            out["person_hit"] = self.s["person_hit"] / self.s["n_person"]
        for name in LOCOMOTION:
            if self.s[f"n_person_{name}"]:
                out[f"person_hit_{name}"] = (self.s[f"person_hit_{name}"]
                                             / self.s[f"n_person_{name}"])
        return out


# -------------------------------------------------------------------- epochs
def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip=1.0, lr_fn=None,
              global_step=0, max_batches=None, collect=False):
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_correct = tot_n = 0
    vstats = VelocityStats()
    w_sum, w_n = None, 0
    y_true, y_pred = [], []
    for b, (seq, label, motion, person) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        seq, label = seq.to(device, non_blocking=True), label.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits, aux = model(seq, return_aux=True)
            loss = criterion(logits, label)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            if lr_fn is not None:
                for g in optimizer.param_groups:
                    g["lr"] = lr_fn(global_step)
            optimizer.step()
            global_step += 1
        bs = label.size(0)
        pred = logits.argmax(1)
        tot_loss += loss.item() * bs
        tot_correct += (pred == label).sum().item()
        tot_n += bs
        if aux.get("velocities") is not None:
            vstats.update(aux["velocities"].cpu(), motion, person, label.cpu())
        w = aux.get("pool_weights")
        if w is not None and w.dim() == 2:
            w = w.detach().sum(0).cpu()
            w_sum = w if w_sum is None else w_sum + w
            w_n += bs
        if collect:
            y_true += label.tolist()
            y_pred += pred.tolist()
    stats = {"loss": tot_loss / max(tot_n, 1), "acc": tot_correct / max(tot_n, 1),
             **vstats.result(), "_global_step": global_step}
    if w_sum is not None:                     # attention pool: mean weight on each copy / slot
        stats.update({f"attn_{k}": float(v) / w_n for k, v in enumerate(w_sum)})
    if collect:
        stats["_y_true"], stats["_y_pred"] = y_true, y_pred
    return stats


def class_report(y_true, y_pred):
    conf = np.zeros((len(KTH_ACTIONS), len(KTH_ACTIONS)), dtype=int)
    for t, q in zip(y_true, y_pred):
        conf[t, q] += 1
    per_class = {name: float(conf[i, i] / max(conf[i].sum(), 1))
                 for i, name in enumerate(KTH_ACTIONS)}
    loco = [KTH_ACTIONS.index(a) for a in LOCOMOTION]
    inplace = [i for i in range(len(KTH_ACTIONS)) if i not in loco]
    sub = lambda idx: float(conf[np.ix_(idx, idx)].trace() / max(conf[idx].sum(), 1))  # noqa: E731
    return {"per_class": per_class, "confusion": conf.tolist(),
            "acc_locomotion": sub(loco), "acc_inplace": sub(inplace)}


def evaluate_all(model, loaders, train_loader, device, criterion, args):
    """Precise BN once, then every test condition."""
    recompute_bn_stats(model, train_loader, device, args.precise_bn_batches)
    out = {}
    for name, loader in loaders.items():
        r = run_epoch(model, loader, device, criterion, collect=True)
        r.pop("_global_step", None)
        y_true, y_pred = r.pop("_y_true"), r.pop("_y_pred")
        out[name] = {**r, **class_report(y_true, y_pred)}
    return out


# --------------------------------------------------------------------- panels
def pick_state_items(ds, n_fig, n_total, seed):
    """One clip per action first (PANEL_ORDER), then random others; fixed for the whole run."""
    order = np.random.RandomState(seed).permutation(len(ds))
    labels = [ds.labels[ds.clip_index(i)] for i in order]
    chosen = []
    for name in PANEL_ORDER[:n_fig]:
        c = KTH_ACTIONS.index(name)
        for i, lab in zip(order, labels):
            if lab == c and i not in chosen:
                chosen.append(int(i))
                break
    for i in order:
        if len(chosen) >= n_total:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    items = [ds[i] for i in chosen]
    return (torch.stack([it[0] for it in items]), torch.tensor([it[1] for it in items]),
            torch.stack([it[2] for it in items]), torch.stack([it[3] for it in items]),
            min(n_fig, len(chosen)))


def log_panels(model, state, args, epoch, step, device, wandb, tag):
    from kth.kth_panels import state_figure, velocity_figure
    import matplotlib.pyplot as plt

    seq, label, motion, person, n_fig = state
    rec = record_states(model, seq.to(device))
    lattice = model.backbone.cell.v_list if model.model == "felstm" else None
    vel = rec["velocities"]
    pred = rec["logits"].argmax(1)
    picks = None
    if lattice is not None:
        pc = PhaseCorrelation(n_modes=args.num_vel_modes)
        x = seq[:n_fig]
        picks = torch.stack([pc(x[:, t - 1], x[:, t])[0] for t in range(1, x.shape[1])],
                            dim=1).numpy()
    fig_dir = Path(args.states_fig_dir) if args.states_fig_dir else None
    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)
    payload = {}

    def emit(fig, key):
        if fig_dir:
            fig.savefig(fig_dir / f"{key.replace('/', '_')}_ep{epoch}.png", dpi=130,
                        bbox_inches="tight")
        if wandb:
            payload[key] = wandb.Image(fig)
        plt.close(fig)

    for i in range(n_fig):
        name = KTH_ACTIONS[int(label[i])]
        title = (f"{tag} — {args.model}, {name} (pred {KTH_ACTIONS[int(pred[i])]}), "
                 f"camera {args.camera}, epoch {epoch}   [{args.run_name}]")
        fig = state_figure(rec["copies"][i].numpy(), rec["pooled"][i].numpy(),
                           rec["winner"][i].numpy(), seq[i, :, 0].numpy(), model.copy_labels(),
                           motion[i, :, 0].numpy(), person[i].numpy(),
                           slot_v=None if vel is None else vel[i].numpy(), lattice=lattice,
                           pc_picks=None if picks is None else picks[i], title=title,
                           pool_label=model.pool_label(),
                           pool_weights=(None if rec["weights"] is None
                                         else rec["weights"][i].numpy()))
        emit(fig, f"states_{tag}/{i}_{name}")
    if vel is not None:
        names = [KTH_ACTIONS[int(c)] for c in label[:n_fig]]
        fig = velocity_figure(vel[:n_fig].numpy(), motion[:n_fig, :, 0].numpy(),
                              person[:n_fig].numpy(), names,
                              title=f"{tag} — melstm slot velocities, epoch {epoch}")
        emit(fig, f"velocity_{tag}/slots")
    if wandb and payload:
        wandb.log(payload, step=step)
    scalars = {}
    labels = model.copy_labels()
    if len(labels) > 1:
        share = rec["share"].mean(0)
        scalars = {f"pool_share/{lab}": float(share[k]) for k, lab in enumerate(labels)}
    return scalars


# ---------------------------------------------------------------------- main
def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)
    if args.model_seed is not None:
        torch.manual_seed(args.model_seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "mps" and args.model == "melstm":
        enable_integer_shift_warp(device="mps")

    models_dir, results_dir, state_dir = (Path(args.save_dir) / d
                                          for d in ("models", "results", "run_state"))
    for d in (models_dir, results_dir, state_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- data
    t0 = time.time()
    store = KTHStore(args.root)
    conditions = parse_conditions(args)
    in_dist = args.camera if args.camera in conditions else next(iter(conditions))
    common = dict(scheme=args.split_scheme, seq_len=args.seq_len, step=args.step,
                  seed=args.data_seed)
    train_ds = KTHClips(store, "train", camera=conditions[args.camera], train=True,
                        resample_camera=args.resample_camera, **common)
    val_ds = KTHClips(store, "val", camera=conditions[args.camera],
                      eval_windows=args.eval_windows, **common)
    test_ds = {name: KTHClips(store, "test", camera=cam, eval_windows=args.eval_windows, **common)
               for name, cam in conditions.items()}
    print(f"data         : {len(store.frames)} videos loaded in {time.time() - t0:.1f}s; "
          f"split {args.split_scheme}: train {len(train_ds)} / val {len(val_ds)} / "
          f"test {len(test_ds[in_dist])} clips; T={args.seq_len} at step {args.step}")
    print(f"camera       : train/val {conditions[args.camera].describe()}")
    for name, ds in test_ds.items():
        print(f"  test {name:12s}: {ds.camera.describe()}")

    tr, va = train_ds, val_ds
    te = dict(test_ds)
    if args.smoke_test:
        tr = Subset(train_ds, list(range(2 * args.batch_size)))
        va = Subset(val_ds, list(range(args.batch_size)))
        te = {k: Subset(v, list(range(args.batch_size))) for k, v in test_ds.items()}
    kw = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                              persistent_workers=args.num_workers > 0, **kw)
    val_loader = DataLoader(va, batch_size=args.batch_size, **kw)
    test_loaders = {k: DataLoader(v, batch_size=args.batch_size, **kw) for k, v in te.items()}
    state = pick_state_items(test_ds[in_dist], args.log_states_samples,
                             args.state_metric_samples, args.data_seed + 1)

    # ---- model
    model = build_kth_classifier(args).to(device)
    report = model.parameter_report()
    print(f"run          : {args.run_name}   device {device}")
    print(model.describe())
    print(f"chance       : {1 / len(KTH_ACTIONS):.3f} (6 actions); a dead head sits at loss "
          f"{math.log(len(KTH_ACTIONS)):.3f}")

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    warm = int(round(args.warmup_epochs * len(train_loader)))

    def lr_fn(step):
        if args.lr_schedule == "constant":
            return args.lr
        if step < warm:
            return args.lr * (step + 1) / warm
        q = min(1.0, (step - warm) / max(1, total - warm))
        return args.final_lr + 0.5 * (args.lr - args.final_lr) * (1 + math.cos(math.pi * q))

    def x_track_p_at(epoch):
        E = args.x_curriculum_epochs
        if E <= 0 or epoch >= E - 1:
            return 0.0
        frac = epoch / (E - 1)
        return 1.0 - frac if args.x_curriculum_shape == "linear" else \
            0.5 * (1.0 + math.cos(math.pi * frac))

    criterion = nn.CrossEntropyLoss()
    wandb = None
    if args.use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, dir=args.wandb_dir,
                   name=args.run_name, config=vars(args) | {f"params_{k}": v
                                                           for k, v in report.items()})
        wandb.define_metric("epoch")
        wandb.summary["params_trained"] = report["trained"]

    history = {"config": vars(args), "parameters": report, "epochs": []}
    best_val, best_epoch, start_epoch, global_step = -1.0, -1, 0, 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, global_step = ck["epoch"] + 1, ck.get("global_step", 0)
        best_val, best_epoch, history = ck["best_val"], ck.get("best_epoch", -1), ck["history"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    deadline = Deadline(args.stop_at, args.eval_reserve_min)
    best_path = models_dir / f"{args.run_name}_best.pth"
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        if epoch > start_epoch and not deadline.room_for_epoch():
            print(f"--stop_at {args.stop_at}: stopping before epoch {epoch}")
            break
        if args.x_curriculum_epochs:
            model.x_track_p = x_track_p_at(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        tr_stats = run_epoch(model, train_loader, device, criterion, optimizer, args.grad_clip,
                             lr_fn, global_step)
        t_train = time.time() - t0
        global_step = tr_stats.pop("_global_step")
        recompute_bn_stats(model, train_loader, device, args.precise_bn_batches)
        va_stats = run_epoch(model, val_loader, device, criterion)
        va_stats.pop("_global_step")
        te_stats = {}
        if args.test_every and epoch % args.test_every == 0:
            te_stats = run_epoch(model, test_loaders[in_dist], device, criterion)
            te_stats.pop("_global_step")
        t_epoch = time.time() - t0
        deadline.epoch_done(t_epoch)

        row = {"epoch": epoch, "time": t_epoch, "time_train": t_train,
               "lr": optimizer.param_groups[0]["lr"],
               **({"x_track_p": model.x_track_p} if args.x_curriculum_epochs else {}),
               **({"gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
                  if device.type == "cuda" else {}),
               **{f"train_{k}": v for k, v in tr_stats.items()},
               **{f"val_{k}": v for k, v in va_stats.items()},
               **{f"test_{k}": v for k, v in te_stats.items()}}
        if args.log_states_every and (epoch % args.log_states_every == 0
                                      or epoch == args.epochs - 1):
            if wandb or args.states_fig_dir:
                row.update(log_panels(model, state, args, epoch, global_step, device, wandb,
                                      "test"))
        history["epochs"].append(row)

        diag = "".join(f"  {k}={va_stats[k]:.2f}" for k in ("cam_hit", "person_hit")
                       if k in va_stats)
        te_desc = f" | test acc {te_stats['acc']:.3f}" if te_stats else ""
        print(f"epoch {epoch:3d} | train {tr_stats['loss']:.4f} / {tr_stats['acc']:.3f} "
              f"| val {va_stats['loss']:.4f} / {va_stats['acc']:.3f}{diag}{te_desc} "
              f"| {t_train:.1f}s train, {t_epoch:.1f}s epoch")
        if wandb:
            wandb.log(row, step=global_step)

        if va_stats["acc"] > best_val:
            best_val, best_epoch = va_stats["acc"], epoch
            torch.save({"model": model.state_dict(), "config": vars(args), "epoch": epoch,
                        "val_acc": best_val}, best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch,
                    "history": history, "global_step": global_step},
                   state_dir / f"checkpoint_{args.run_name}.pth")

    # ---- final: last weights, then the best-val checkpoint, on every test condition
    model.x_track_p = 0.0
    final = {"last_epoch": epoch, "best_epoch": best_epoch, "best_val_acc": best_val}
    final["last"] = evaluate_all(model, test_loaders, train_loader, device, criterion, args)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False)["model"])
    final["best"] = evaluate_all(model, test_loaders, train_loader, device, criterion, args)
    times = [r["time_train"] for r in history["epochs"]]
    final["s_per_epoch_train_median"] = float(np.median(times)) if times else None
    history["final"] = final

    print(f"\nbest val acc {best_val:.3f} (epoch {best_epoch}); test acc, best ckpt | last epoch:")
    for name in final["best"]:
        b, last = final["best"][name], final["last"][name]
        extra = "".join(f"  {k}={b[k]:.2f}" for k in ("cam_hit", "person_hit") if k in b)
        print(f"  {name:12s} {b['acc']:.3f} | {last['acc']:.3f}   (locomotion "
              f"{b['acc_locomotion']:.3f}, in-place {b['acc_inplace']:.3f}){extra}")
    with open(Path(args.save_dir) / "results" / f"history_{args.run_name}.json", "w") as f:
        json.dump(history, f, indent=2)

    if wandb:
        if args.log_states_every:
            log_panels(model, state, args, epoch, global_step, device, wandb, "best")
        for name in final["best"]:
            wandb.summary[f"test_acc/{name}"] = final["best"][name]["acc"]
            wandb.summary[f"test_acc_last/{name}"] = final["last"][name]["acc"]
            for k in ("acc_locomotion", "acc_inplace", "cam_hit", "person_hit"):
                if k in final["best"][name]:
                    wandb.summary[f"test_{k}/{name}"] = final["best"][name][k]
        for cname, acc in final["best"][in_dist]["per_class"].items():
            wandb.summary[f"test_acc_class/{cname}"] = acc
        conf = np.array(final["best"][in_dist]["confusion"])
        y_true = [i for i in range(len(KTH_ACTIONS)) for j in range(len(KTH_ACTIONS))
                  for _ in range(conf[i, j])]
        y_pred = [j for i in range(len(KTH_ACTIONS)) for j in range(len(KTH_ACTIONS))
                  for _ in range(conf[i, j])]
        wandb.log({"test_confusion": wandb.plot.confusion_matrix(
            y_true=y_true, preds=y_pred, class_names=list(KTH_ACTIONS))}, step=global_step)
        wandb.summary["best_val_acc"] = best_val
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["s_per_epoch_train_median"] = final["s_per_epoch_train_median"]
        wandb.finish()

    if not args.smoke_test and epoch == args.epochs - 1:
        (state_dir / f"DONE_{args.run_name}.flag").touch()
    return history


if __name__ == "__main__":
    main()
