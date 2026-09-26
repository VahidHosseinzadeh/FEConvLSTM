#!/usr/bin/env python
"""
Common-fate VIDEO classification: real silhouettes (Weizmann) or articulating MNIST digits,
rendered so that every frame is noise and the subject exists only by common fate.

    # Weizmann (download classification_masks.mat first; see datasets/common_fate_video.py)
    python -m motion_benchmarks.train_cf_video --dataset weizmann --weizmann_mat classification_masks.mat \
        --model melstm --num_vel_modes 4 --velocity_source frame_pair --readout_steps 6 \
        --forget_bias 0.44 --forget_bias_long 2.97 --test_subjects daria,denis --val_subjects eli

    # articulation sweep (how far outside G = R^2 transport keeps paying)
    for amp in 0 1 2 4 6; do
      python -m motion_benchmarks.train_cf_video --dataset deform_mnist --deform_amp $amp --dilate 2 ...
    done

Models: the three backbones of the CF-MNIST classifier (lstm / felstm / melstm) through
MotionVideoClassifier, with sub-pixel phase correlation, the two-timescale forget gate and the
trajectory readout (all opt-in). Controls built in: --control shuffle evaluates on
frame-shuffled clips (destroys common fate: a model that still scores is reading something
else), and lstm is the in-place control that should sit near chance.

Evaluation re-estimates BatchNorm statistics before every evaluation (precise BN), as the
CF-MNIST trainer does.
"""
import argparse
import json
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
from motion_benchmarks.datasets.common_fate_video import (WEIZMANN_ACTIONS,  # noqa: E402
                                                          DeformingCommonFateMNIST,
                                                          WeizmannCommonFate,
                                                          build_weizmann_sequences)
from motion_benchmarks.models.cf_classifier import MotionVideoClassifier  # noqa: E402
from motion_classification_model import recompute_bn_stats  # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=["weizmann", "deform_mnist"], default="weizmann")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--speed", type=float, default=2.0, help="layer speed, px per frame")
    p.add_argument("--corr_len", type=float, default=1.0)
    p.add_argument("--time_varying", type=int, default=1)
    p.add_argument("--train_size", type=int, default=8000, help="clips per epoch")
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--data_seed", type=int, default=42)
    # weizmann
    p.add_argument("--weizmann_mat", type=str, default=None,
                   help="classification_masks.mat; omitted -> synthetic stand-in silhouettes")
    p.add_argument("--mask_set", default="original", help="'original' or 'aligned'")
    p.add_argument("--target_area", type=float, default=0.30)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--test_subjects", type=str, default="daria,denis")
    p.add_argument("--val_subjects", type=str, default="eli")
    # deforming mnist
    p.add_argument("--mnist_npz", type=str, default=None, help="npz with X (N,28,28), y (N,)")
    p.add_argument("--mnist_root", type=str, default="./data")
    p.add_argument("--deform_amp", type=float, default=0.0)
    p.add_argument("--deform_period", type=float, default=8.0)
    p.add_argument("--dilate", type=int, default=0)
    p.add_argument("--digit_scale", type=int, default=2)
    # model
    p.add_argument("--model", choices=["lstm", "felstm", "melstm"], default="melstm")
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--v_range", type=int, default=2)
    p.add_argument("--num_vel_modes", type=int, default=4)
    p.add_argument("--velocity_source", choices=["bootstrap", "frame_pair", "tracked"],
                   default="frame_pair")
    p.add_argument("--velocity_pool", choices=["attention", "max", "mean", "concat"],
                   default="attention")
    p.add_argument("--head_channels", type=int, default=64)
    p.add_argument("--head_blocks", type=int, default=3)
    p.add_argument("--head_norm", choices=["batch", "group", "none"], default="batch")
    p.add_argument("--pc_alpha", type=float, default=1.0)
    p.add_argument("--pc_subpixel", type=int, default=1)
    p.add_argument("--pc_suppress_radius", type=int, default=2)
    p.add_argument("--pc_search_radius", type=int, default=None)
    p.add_argument("--forget_bias", type=float, default=None,
                   help="0.44 = tau 2 frames, 0.93 = tau 3, 1.26 = tau 4 (default: cell's 1.0)")
    p.add_argument("--forget_bias_long", type=float, default=None, help="2.97 = tau 20 frames")
    p.add_argument("--long_fraction", type=float, default=0.5)
    p.add_argument("--readout_steps", type=int, default=1)
    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--precise_bn_batches", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--model_seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--control", choices=["none", "shuffle"], default="none",
                   help="shuffle: also evaluate on frame-shuffled test clips")
    p.add_argument("--save_dir", type=str, default="./experiments/motion_benchmarks_cf")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--smoke_test", action="store_true")
    a = p.parse_args(argv)
    if a.smoke_test:
        a.train_size, a.val_size, a.test_size = 16, 8, 8
        a.epochs, a.batch_size, a.num_workers = 1, 4, 0
        a.seq_len = min(a.seq_len, 6)
        a.precise_bn_batches = 1
    if a.run_name is None:
        a.run_name = f"{a.dataset}_{a.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    return a


def load_mnist(a):
    if a.mnist_npz:
        z = np.load(a.mnist_npz)
        return z["X"], z["y"]
    from torchvision.datasets import MNIST
    tr = MNIST(a.mnist_root, train=True, download=True)
    return tr.data.numpy(), tr.targets.numpy()


def build(a):
    sd = a.data_seed
    common = dict(speed=a.speed, corr_len=a.corr_len, time_varying=bool(a.time_varying))
    if a.dataset == "weizmann":
        seqs = build_weizmann_sequences(a.weizmann_mat, a.image_size, a.target_area, a.mask_set,
                                        n_synthetic=12, seed=sd)
        test_s = set(a.test_subjects.split(",")) if a.test_subjects else set()
        val_s = set(a.val_subjects.split(",")) if a.val_subjects else set()
        if not a.weizmann_mat:                       # synthetic stand-in subjects
            subs = sorted({s for _, _, s in seqs})
            test_s, val_s = {subs[-1]}, {subs[-2]}
        pick = lambda keep: [(m, l) for m, l, s in seqs if keep(s)]  # noqa: E731
        tr = pick(lambda s: s not in test_s and s not in val_s)
        va = pick(lambda s: s in val_s) or tr
        te = pick(lambda s: s in test_s) or tr
        kw = dict(stride=a.frame_stride, **common)
        dsets = dict(train=WeizmannCommonFate(tr, a.train_size, a.seq_len, seed=sd, random=True, **kw),
                     val=WeizmannCommonFate(va, a.val_size, a.seq_len, seed=sd + 1, random=False, **kw),
                     test=WeizmannCommonFate(te, a.test_size, a.seq_len, seed=sd + 2, random=False, **kw))
        if a.control == "shuffle":
            dsets["test_shuffled"] = WeizmannCommonFate(te, a.test_size, a.seq_len, seed=sd + 2,
                                                        random=False, shuffle_time=True, **kw)
        n_classes = len(WEIZMANN_ACTIONS)
        info = dict(n_train_seq=len(tr), n_val_seq=len(va), n_test_seq=len(te),
                    test_subjects=sorted(test_s), val_subjects=sorted(val_s))
    else:
        X, y = load_mnist(a)
        n = len(X)
        idx = np.random.default_rng(sd).permutation(n)
        n_te, n_va = n // 6, n // 12
        parts = dict(test=idx[:n_te], val=idx[n_te:n_te + n_va], train=idx[n_te + n_va:])
        kw = dict(image_size=a.image_size, amp=a.deform_amp, period=a.deform_period,
                  scale=a.digit_scale, dilate=a.dilate, **common)
        size = dict(train=a.train_size, val=a.val_size, test=a.test_size)
        dsets = {k: DeformingCommonFateMNIST(X[v], y[v], size[k], a.seq_len,
                                             seed=sd + i, random=(k == "train"), **kw)
                 for i, (k, v) in enumerate(parts.items())}
        n_classes = 10
        info = dict(amp=a.deform_amp, dilate=a.dilate)
    return dsets, n_classes, info


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    correct, n = 0, 0
    conf = np.zeros((n_classes, n_classes), np.int64)
    for seq, y, _ in loader:
        seq, y = seq.to(device), y.to(device)
        pred = model(seq).argmax(dim=1)
        correct += int((pred == y).sum())
        n += len(y)
        for t, p_ in zip(y.cpu().numpy(), pred.cpu().numpy()):
            conf[t, p_] += 1
    per_class = (np.diag(conf) / np.maximum(conf.sum(1), 1)).tolist()
    return dict(accuracy=correct / max(n, 1), n=n, per_class=per_class, confusion=conf.tolist())


def main(argv=None):
    a = get_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(a.save_dir) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.data_seed)
    dsets, n_classes, info = build(a)
    print(f"[train_cf_video] {a.dataset}: {info}")

    torch.manual_seed(a.model_seed)
    pc = dict(alpha=a.pc_alpha, subpixel=bool(a.pc_subpixel),
              suppress_radius=a.pc_suppress_radius, search_radius=a.pc_search_radius)
    model = MotionVideoClassifier(
        model=a.model, hidden_channels=a.hidden_size, kernel_size=a.kernel_size,
        v_range=a.v_range, n_slots=a.num_vel_modes, n_classes=n_classes,
        velocity_pool=a.velocity_pool, velocity_source=a.velocity_source,
        head_channels=a.head_channels, head_blocks=a.head_blocks, head_norm=a.head_norm,
        phase_corr_kwargs=pc, forget_bias=a.forget_bias, forget_bias_long=a.forget_bias_long,
        long_fraction=a.long_fraction, readout_steps=a.readout_steps).to(device)
    params = model.trainable_parameters()
    print(f"[train_cf_video] model {a.model}: {sum(p.numel() for p in params):,} parameters")

    mk = lambda ds, train: DataLoader(ds, batch_size=a.batch_size, shuffle=train,  # noqa: E731
                                      drop_last=train, num_workers=a.num_workers,
                                      persistent_workers=train and a.num_workers > 0)
    loaders = {k: mk(v, k == "train") for k, v in dsets.items()}
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.weight_decay)
    crit = nn.CrossEntropyLoss()
    history, best = [], -1.0
    for epoch in range(a.epochs):
        model.train()
        tl, tc, tn, t0 = 0.0, 0, 0, time.time()
        for b, (seq, y, _) in enumerate(loaders["train"]):
            if a.smoke_test and b >= 2:
                break
            seq, y = seq.to(device), y.to(device)
            logits = model(seq)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
            opt.step()
            tl += loss.item() * len(y)
            tc += int((logits.argmax(1) == y).sum())
            tn += len(y)
        with torch.no_grad():
            recompute_bn_stats(model, loaders["train"], device, a.precise_bn_batches)
        val = evaluate(model, loaders["val"], device, n_classes)
        row = dict(epoch=epoch, train_loss=tl / max(tn, 1), train_acc=tc / max(tn, 1),
                   val_acc=val["accuracy"], time=time.time() - t0)
        history.append(row)
        print(f"[epoch {epoch:3d}] loss {row['train_loss']:.4f} train {row['train_acc']:.3f} "
              f"val {row['val_acc']:.3f} ({row['time']:.0f}s)")
        if val["accuracy"] > best:
            best = val["accuracy"]
            torch.save(dict(model=model.state_dict(), args=vars(a), epoch=epoch), out / "best.pth")
    model.load_state_dict(torch.load(out / "best.pth", map_location=device, weights_only=False)["model"])
    with torch.no_grad():
        recompute_bn_stats(model, loaders["train"], device, a.precise_bn_batches)
    res = dict(args=vars(a), info=info, history=history, best_val=best,
               test=evaluate(model, loaders["test"], device, n_classes))
    if "test_shuffled" in loaders:
        res["test_shuffled"] = evaluate(model, loaders["test_shuffled"], device, n_classes)
    print(f"[train_cf_video] test accuracy {res['test']['accuracy']:.3f}"
          + (f"  (frame-shuffled control {res['test_shuffled']['accuracy']:.3f})"
             if "test_shuffled" in res else "") + f"  chance {1 / n_classes:.3f}")
    with open(out / "results.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
