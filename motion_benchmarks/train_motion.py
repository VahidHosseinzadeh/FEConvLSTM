#!/usr/bin/env python
"""
Train and evaluate ConvLSTM / FEConvLSTM / MEConvLSTM -- and the oracle and persistence
baselines -- on any registered motion benchmark, with one protocol.

    python -m motion_benchmarks.train_motion --dataset radar_synthetic --model melstm \
        --velocity_source frame_pair --residual lagrangian --input_frames 12 --pred_frames 12

    python -m motion_benchmarks.train_motion --dataset radar_synthetic --model lstm ...
    python -m motion_benchmarks.train_motion --dataset radar_synthetic --model persistence_lagrangian

Run `--help` for every option; dataset options are prefixed with the dataset name.

Protocol (identical to moving_mnist/train.py where it overlaps)
----------------------------------------------------------------
* Training sequences are drawn fresh every epoch; val / test / gen_test are fixed benchmarks.
* MEConvLSTM is always given target_seq: during training its decoder velocity is tracked against
  the true next frame (what the repo's training protocol requires); at evaluation it is FROZEN
  at the last encoder estimate (honest inference) unless --eval_velocity_mode asks for the
  oracle. 'both' selects on frozen and logs tracked alongside.
* The rollout's motion freezes at the last transition in the context (freeze_after =
  input_frames) unless --no_freeze_future -- the length-generalisation protocol.
* The persistence baselines are evaluated on the SAME test sets, so every results.json carries
  its own Eulerian and Lagrangian reference (and the headroom ratio).

Outputs, under --save_dir/<run_name>/:
    checkpoint_best.pth, checkpoint_last.pth, results.json, examples.png
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from motion_benchmarks import _repo  # noqa: E402,F401
from motion_benchmarks.common.metrics import (FSS, Categorical, PerLeadError,  # noqa: E402
                                              ke_spectrum, log_spectral_distance, nusselt_volume)
from motion_benchmarks.common.deadline import Deadline, add_deadline_args  # noqa: E402
from motion_benchmarks.datasets.registry import REGISTRY  # noqa: E402
from motion_benchmarks.models.factory import (NEEDS_MOTION, TRAINABLE,  # noqa: E402
                                              add_model_args, build_model, count_parameters,
                                              pc_window, phase_corr_kwargs, run_model)
from motion_benchmarks.models.persistence import Persistence  # noqa: E402
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402

# make sure every dataset module registers itself
from motion_benchmarks.datasets import specs  # noqa: E402,F401


# ============================================================================ arguments
def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(REGISTRY))

    d = p.add_argument_group("data (shared)")
    d.add_argument("--input_frames", type=int, default=12)
    d.add_argument("--pred_frames", type=int, default=12)
    d.add_argument("--gen_pred_frames", type=int, default=0,
                   help="length generalisation: roll out this many frames on gen_test (0 = off)")
    d.add_argument("--image_size", type=int, default=None, help="dataset default if omitted")
    d.add_argument("--train_size", type=int, default=20000, help="sequences per epoch")
    d.add_argument("--val_size", type=int, default=1000)
    d.add_argument("--test_size", type=int, default=2000)
    d.add_argument("--gen_test_size", type=int, default=500)
    d.add_argument("--extra_test_size", type=int, default=500)
    d.add_argument("--no_freeze_future", action="store_true")
    d.add_argument("--data_seed", type=int, default=42)

    t = p.add_argument_group("training")
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--eval_batch_size", type=int, default=None)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--min_epochs", type=int, default=0)
    t.add_argument("--early_stop_patience", type=int, default=0)
    t.add_argument("--max_batches_per_epoch", type=int, default=None)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument("--loss", default="mse", choices=["mse", "mse_l1", "l1"])
    t.add_argument("--use_lr_scheduler", action="store_true")
    t.add_argument("--lr_patience", type=int, default=4)
    t.add_argument("--lr_factor", type=float, default=0.5)
    t.add_argument("--lr_min", type=float, default=1e-6)
    t.add_argument("--eval_velocity_mode", default="frozen", choices=["frozen", "tracked", "both"])
    t.add_argument("--model_seed", type=int, default=42)
    t.add_argument("--num_workers", type=int, default=4)
    t.add_argument("--device", type=str, default=None)
    t.add_argument("--no_baselines", action="store_true",
                   help="skip evaluating the persistence baselines on the test sets")
    add_deadline_args(t)

    o = p.add_argument_group("output")
    o.add_argument("--save_dir", type=str, default="./experiments/motion_benchmarks")
    o.add_argument("--run_name", type=str, default=None)
    o.add_argument("--resume", type=str, default=None, help="checkpoint to resume training from")
    o.add_argument("--evaluate_only", type=str, default=None, help="checkpoint to evaluate")
    o.add_argument("--save_examples", type=int, default=3)
    o.add_argument("--use_wandb", action="store_true")
    o.add_argument("--wandb_project", type=str, default="FEConvLSTM-motion-benchmarks")
    o.add_argument("--wandb_entity", type=str, default=None)
    o.add_argument("--wandb_dir", type=str, default="./tmp/")
    o.add_argument("--smoke_test", action="store_true",
                   help="tiny sizes, one short epoch, no workers: checks the pipeline end to end")

    add_model_args(p)
    for name in sorted(REGISTRY):
        REGISTRY[name].add_args(p)
    args = p.parse_args(argv)

    if args.smoke_test:
        args.train_size, args.val_size, args.test_size = 16, 8, 8
        args.gen_test_size, args.extra_test_size = 4, 4
        args.epochs, args.batch_size, args.num_workers = 1, 4, 0
        args.max_batches_per_epoch = 2
        args.input_frames = min(args.input_frames, 4)
        args.pred_frames = min(args.pred_frames, 3)
        if args.gen_pred_frames:
            args.gen_pred_frames = args.pred_frames + 2
        args.save_examples = min(args.save_examples, 1)
    if args.run_name is None:
        args.run_name = f"{args.dataset}_{args.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    return args


# ============================================================================ helpers
def make_criterion(kind):
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    if kind == "mse":
        return mse
    if kind == "l1":
        return l1
    return lambda a, b: mse(a, b) + l1(a, b)


def make_loader(ds, batch_size, train, args, device):
    if ds is None:
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                      persistent_workers=(train and args.num_workers > 0))


def has_motion(motion):
    return motion is not None and bool(torch.isfinite(motion).all())


def velocity_error(vel, motion, T_in):
    """
    Encoder velocities (B, T_in-1, K, 2) vs true motion[:, :T_in-1] (B, T_in-1, N, 2).
    Each true motion is scored against its best slot (slots are unordered).
    """
    true = motion[:, :vel.shape[1]].to(vel.dtype)
    d = torch.linalg.norm(vel[:, :, :, None, :] - true[:, :, None, :, :], dim=-1)  # B,T,K,N
    best = d.min(dim=2).values                                                     # B,T,N
    return float(best.mean()), float((best <= 0.5).float().mean())


class PhysicsMetrics:
    """Nusselt number and kinetic-energy spectra for fluid datasets (meta['physics'])."""

    def __init__(self, phys):
        self.p = phys
        self.mean = torch.tensor(phys["channel_mean"]).float()
        self.std = torch.tensor(phys["channel_std"]).float()
        self.nu_err = None
        self.lsd = None
        self.n = 0

    def _phys(self, x, idx):
        m = self.mean.to(x.device)[idx][:, None, None]
        s = self.std.to(x.device)[idx][:, None, None]
        return x[:, :, idx] * s + m

    @torch.no_grad()
    def update(self, pred, target):
        p = self.p
        B, T = pred.shape[:2]
        if p.get("T_channels") and p.get("w_channels"):
            nu_p = nusselt_volume(self._phys(pred, p["T_channels"]), self._phys(pred, p["w_channels"]),
                                  p["kappa"], p.get("delta_T", 1.0), p.get("Lz", 1.0))
            nu_t = nusselt_volume(self._phys(target, p["T_channels"]),
                                  self._phys(target, p["w_channels"]),
                                  p["kappa"], p.get("delta_T", 1.0), p.get("Lz", 1.0))
            e = (nu_p - nu_t).abs().sum(0)
            self.nu_err = e if self.nu_err is None else self.nu_err + e
        if p.get("u_channels") and p.get("v_channels"):
            sp = ke_spectrum(self._phys(pred, p["u_channels"]), self._phys(pred, p["v_channels"]))
            st = ke_spectrum(self._phys(target, p["u_channels"]),
                             self._phys(target, p["v_channels"]))
            lsd = log_spectral_distance(sp.mean(2), st.mean(2)).sum(0)      # (T,)
            self.lsd = lsd if self.lsd is None else self.lsd + lsd
        self.n += B

    def result(self):
        out = {}
        if self.nu_err is not None:
            out["nusselt_abs_err"] = (self.nu_err / self.n).cpu().tolist()
        if self.lsd is not None:
            out["ke_spectrum_lsd"] = (self.lsd / self.n).cpu().tolist()
        return out


@torch.no_grad()
def evaluate(model, loader, args, meta, device, criterion, pred_len=None, track=False,
             max_batches=None, full=True):
    model.eval()
    T_in = args.input_frames
    P = pred_len or args.pred_frames
    err = PerLeadError(groups=meta.get("channel_groups"))
    thr = meta.get("thresholds")
    cat = Categorical(thr) if (full and thr) else None
    fss = FSS(thr, meta.get("fss_scales", (1, 5, 9, 17))) if (full and thr) else None
    phys = PhysicsMetrics(meta["physics"]) if (full and meta.get("physics")) else None
    loss_sum, n, vel_e, vel_acc, vel_n = 0.0, 0, 0.0, 0.0, 0
    for b, (seq, _, motion) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        seq = seq.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        mo = motion if has_motion(motion) else None
        inp, tgt = seq[:, :T_in], seq[:, T_in:T_in + P]
        want_v = isinstance(model, Seq2SeqMEConvLSTM) and mo is not None and full
        out, vel = run_model(model, inp, P, target=tgt, motion=mo, track=track,
                             want_velocity=want_v)
        B = seq.shape[0]
        loss_sum += float(criterion(out, tgt)) * B
        n += B
        err.update(out, tgt)
        if cat is not None:
            cat.update(out, tgt)
            fss.update(out, tgt)
        if phys is not None:
            phys.update(out, tgt)
        if vel is not None:
            e, a = velocity_error(vel[:, :T_in - 1], mo, T_in)
            vel_e += e * B
            vel_acc += a * B
            vel_n += B
    res = {"loss": loss_sum / max(n, 1), "n": n}
    res.update(err.result())
    if cat is not None:
        res.update(cat.result())
        res.update(fss.result())
    if phys is not None:
        res.update(phys.result())
    if vel_n:
        res["velocity_epe"] = vel_e / vel_n
        res["velocity_within_0.5px"] = vel_acc / vel_n
    return res


def train_epoch(model, loader, optimizer, criterion, args, device, epoch):
    model.train()
    T_in, P = args.input_frames, args.pred_frames
    total, n, t0 = 0.0, 0, time.time()
    for b, (seq, _, motion) in enumerate(loader):
        if args.max_batches_per_epoch is not None and b >= args.max_batches_per_epoch:
            break
        seq = seq.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        mo = motion if has_motion(motion) else None
        inp, tgt = seq[:, :T_in], seq[:, T_in:T_in + P]
        out, _ = run_model(model, inp, P, target=tgt, motion=mo, track=None)
        loss = criterion(out, tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += loss.item() * seq.shape[0]
        n += seq.shape[0]
    return total / max(n, 1), time.time() - t0


def save_examples(model, ds, args, meta, device, path, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if ds is None or n <= 0:
        return
    model.eval()
    T_in, P = args.input_frames, args.pred_frames
    n = min(n, len(ds))
    ch = int(meta.get("display_channel", 0))
    leads = sorted(set([0, P // 2, P - 1]))
    fig, axes = plt.subplots(2 * n, len(leads) + 1, figsize=(2.2 * (len(leads) + 1), 4.4 * n),
                             squeeze=False)
    with torch.no_grad():
        for i in range(n):
            seq, _, motion = ds[i]
            seq = seq[None].to(device)
            motion = motion[None].to(device)
            mo = motion if has_motion(motion) else None
            out, _ = run_model(model, seq[:, :T_in], P, target=seq[:, T_in:T_in + P], motion=mo,
                               track=False)
            last = seq[0, T_in - 1, ch].cpu().numpy()
            vmin, vmax = float(np.percentile(last, 1)), float(np.percentile(last, 99.5))
            axes[2 * i, 0].imshow(last, vmin=vmin, vmax=vmax, cmap="viridis")
            axes[2 * i, 0].set_title("last input", fontsize=8)
            axes[2 * i + 1, 0].axis("off")
            for j, L in enumerate(leads):
                axes[2 * i, j + 1].imshow(seq[0, T_in + L, ch].cpu().numpy(), vmin=vmin, vmax=vmax,
                                          cmap="viridis")
                axes[2 * i, j + 1].set_title(f"target +{L + 1}", fontsize=8)
                axes[2 * i + 1, j + 1].imshow(out[0, L, ch].cpu().numpy(), vmin=vmin, vmax=vmax,
                                              cmap="viridis")
                axes[2 * i + 1, j + 1].set_title(f"{args.model} +{L + 1}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


# ============================================================================ main
def main(argv=None):
    args = get_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.save_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.data_seed)
    data = REGISTRY[args.dataset].build(args)
    meta = data["meta"]
    motion_known = bool(meta.get("has_motion", False))
    if args.model in NEEDS_MOTION and not motion_known:
        raise SystemExit(f"--model {args.model} needs ground-truth motion; "
                         f"dataset {args.dataset} has none")
    if args.model == "melstm" and args.velocity_source == "external" and not motion_known:
        raise SystemExit("velocity_source=external needs ground-truth motion")

    torch.manual_seed(args.model_seed)
    model = build_model(args, meta).to(device)
    n_params = count_parameters(model)
    print(f"[train_motion] dataset={args.dataset} model={args.model} params={n_params:,} "
          f"device={device}")
    print(f"[train_motion] meta: { {k: v for k, v in meta.items() if k not in ('physics', 'mean_flow')} }")

    criterion = make_criterion(args.loss)
    ebs = args.eval_batch_size or args.batch_size
    train_loader = make_loader(data["train"], args.batch_size, True, args, device)
    val_loader = make_loader(data["val"], ebs, False, args, device)
    test_loader = make_loader(data["test"], ebs, False, args, device)
    gen_loader = make_loader(data.get("gen_test"), ebs, False, args, device)
    extra_loaders = {k: make_loader(v, ebs, False, args, device)
                     for k, v in (data.get("extra_tests") or {}).items()}

    wb = None
    if args.use_wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, dir=args.wandb_dir,
                        name=args.run_name, config=vars(args))

    history, best_val, best_epoch, start_epoch = [], math.inf, -1, 0
    trainable = args.model in TRAINABLE and args.evaluate_only is None
    optimizer = scheduler = None
    if trainable:
        optimizer = torch.optim.Adam([q for q in model.parameters() if q.requires_grad],
                                     lr=args.lr, weight_decay=args.weight_decay)
        if args.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
                min_lr=args.lr_min)
        if args.resume:
            ck = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            if scheduler is not None and ck.get("scheduler"):
                scheduler.load_state_dict(ck["scheduler"])
            start_epoch = ck["epoch"] + 1
            best_val, best_epoch = ck.get("best_val", math.inf), ck.get("best_epoch", -1)
            history = ck.get("history", [])
            print(f"[train_motion] resumed from {args.resume} at epoch {start_epoch}")
    if args.evaluate_only:
        ck = torch.load(args.evaluate_only, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])

    select_track = args.eval_velocity_mode == "tracked"
    bad_epochs = 0
    # --stop_at: the final evaluation, in units of one validation pass (val_size sequences),
    # for the estimate of how long it takes
    deadline, stopped_at = Deadline(args.stop_at, args.eval_reserve_min), None
    n_modes = 2 if args.eval_velocity_mode == "both" and isinstance(model, Seq2SeqMEConvLSTM) else 1
    eval_units = (args.test_size
                  + (args.gen_test_size * max(1.0, args.gen_pred_frames / args.pred_frames)
                     if gen_loader is not None else 0)
                  + args.extra_test_size * len(extra_loaders)) / max(args.val_size, 1)
    eval_units *= n_modes + (0 if args.no_baselines else 2)
    if trainable and deadline.active and deadline.remaining() < deadline.reserve:
        raise SystemExit(f"[train_motion] --stop_at {args.stop_at}: no time left to train")
    final_eval_s = 0.0
    if trainable:
        for epoch in range(start_epoch, args.epochs):
            if epoch > start_epoch and not deadline.room_for_epoch(final_eval_s):
                stopped_at = epoch
                print(f"[train_motion] --stop_at {args.stop_at}: stopping before epoch {epoch} "
                      f"(epoch {deadline.longest_epoch:.0f}s, final evaluation ~{final_eval_s:.0f}s)")
                break
            t_epoch = time.time()
            tr_loss, tr_time = train_epoch(model, train_loader, optimizer, criterion, args,
                                           device, epoch)
            t_val = time.time()
            val = evaluate(model, val_loader, args, meta, device, criterion, track=select_track,
                           full=False)
            final_eval_s = 1.3 * (time.time() - t_val) * eval_units + 120.0
            row = dict(epoch=epoch, train_loss=tr_loss, val_loss=val["loss"],
                       val_mse=val.get("mse_mean"), lr=optimizer.param_groups[0]["lr"],
                       train_time=tr_time)
            if args.eval_velocity_mode == "both" and isinstance(model, Seq2SeqMEConvLSTM):
                row["val_tracked_loss"] = evaluate(model, val_loader, args, meta, device,
                                                   criterion, track=True, full=False)["loss"]
            history.append(row)
            print(f"[epoch {epoch:3d}] train {tr_loss:.5f}  val {val['loss']:.5f}"
                  + (f"  val_tracked {row['val_tracked_loss']:.5f}" if "val_tracked_loss" in row else "")
                  + f"  lr {row['lr']:.2e}  ({tr_time:.0f}s)")
            deadline.epoch_done(time.time() - t_epoch)
            if wb is not None:
                wb.log(row, step=epoch)
            if scheduler is not None:
                scheduler.step(val["loss"])
            improved = val["loss"] < best_val
            if improved:
                best_val, best_epoch, bad_epochs = val["loss"], epoch, 0
            else:
                bad_epochs += 1
            ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                      scheduler=scheduler.state_dict() if scheduler else None, epoch=epoch,
                      best_val=best_val, best_epoch=best_epoch, history=history, args=vars(args))
            torch.save(ck, out_dir / "checkpoint_last.pth")
            if improved:
                torch.save(ck, out_dir / "checkpoint_best.pth")
            if (args.early_stop_patience and epoch + 1 >= args.min_epochs
                    and bad_epochs >= args.early_stop_patience):
                print(f"[train_motion] early stop at epoch {epoch} (best {best_epoch})")
                break
        best = out_dir / "checkpoint_best.pth"
        if best.exists():
            model.load_state_dict(torch.load(best, map_location=device, weights_only=False)["model"])

    # -------------------------------------------------------------- final evaluation
    results = dict(args=vars(args), meta=meta, params=n_params, history=history,
                   best_epoch=best_epoch, best_val=best_val, stopped_by_deadline_at=stopped_at)
    track_modes = {"frozen": [False], "tracked": [True], "both": [False, True]}[args.eval_velocity_mode]
    if not isinstance(model, Seq2SeqMEConvLSTM):
        track_modes = [False]
    for tr in track_modes:
        tag = "tracked" if tr else "frozen"
        results[f"test_{tag}"] = evaluate(model, test_loader, args, meta, device, criterion,
                                          track=tr)
        if gen_loader is not None:
            results[f"gen_test_{tag}"] = evaluate(model, gen_loader, args, meta, device,
                                                  criterion, pred_len=args.gen_pred_frames,
                                                  track=tr)
        for k, ld in extra_loaders.items():
            results[f"extra_{k}_{tag}"] = evaluate(model, ld, args, meta, device, criterion,
                                                   track=tr)
    results["test"] = results[f"test_{'frozen' if False in track_modes else 'tracked'}"]

    if not args.no_baselines:
        base = {}
        for mode, vel in (("eulerian", "estimated"), ("lagrangian", "estimated"),
                          ("lagrangian", "oracle")):
            if vel == "oracle" and not motion_known:
                continue
            pkw = phase_corr_kwargs(args, meta)
            bm = Persistence(mode=mode, velocity=vel, alpha=pkw["alpha"],
                             subpixel=pkw["subpixel"], radius=pkw["search_radius"],
                             channels=meta.get("pc_channels"),
                             window=pc_window(args, meta)).to(device)
            name = f"{mode}" + ("_oracle" if vel == "oracle" else "")
            base[name] = {"test": evaluate(bm, test_loader, args, meta, device, criterion)}
            if gen_loader is not None:
                base[name]["gen_test"] = evaluate(bm, gen_loader, args, meta, device, criterion,
                                                  pred_len=args.gen_pred_frames)
            for k, ld in extra_loaders.items():
                base[name][f"extra_{k}"] = evaluate(bm, ld, args, meta, device, criterion)
        results["baselines"] = base
        if "eulerian" in base and "lagrangian" in base:
            e = np.array(base["eulerian"]["test"]["mse"])
            lg = np.array(base["lagrangian"]["test"]["mse"])
            results["headroom_eulerian_over_lagrangian"] = (e / np.maximum(lg, 1e-12)).tolist()

    t = results["test"]
    print(f"[train_motion] test MSE per lead: {np.round(t['mse'], 5).tolist()}")
    if "baselines" in results:
        for k, v in results["baselines"].items():
            print(f"[train_motion]   {k:>17s} persistence MSE: "
                  f"{np.round(v['test']['mse'], 5).tolist()}")
    if "velocity_epe" in t:
        print(f"[train_motion] encoder velocity EPE {t['velocity_epe']:.3f} px")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1, default=_json_default)
    try:
        save_examples(model, data["test"], args, meta, device, out_dir / "examples.png",
                      n=args.save_examples)
    except Exception as exc:     # plotting must never cost a finished run
        print(f"[train_motion] could not save examples: {exc}")
    if wb is not None:
        wb.summary.update({"test_mse_mean": t.get("mse_mean")})
        wb.finish()
    print(f"[train_motion] wrote {out_dir / 'results.json'}")
    return results


if __name__ == "__main__":
    main()
