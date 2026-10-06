#!/usr/bin/env python
"""
Phase 0: what phase correlation sees on KTH, before any training.

    python -m kth.pc_analysis --split train --camera none

A frame-pair (or bootstrap) MEConvLSTM is transported by velocities computed from the raw frames
alone -- never from its weights -- so they can be measured exhaustively before training. This
script runs the classifier's own velocity code (_candidate_velocities + _match_to_slots, i.e.
_encode_melstm_frame_pair without the recurrent cell) over a grid of settings and reports, over
every step of every clip:

  bg_hit       some slot moves at the camera velocity -- (0, 0) with a static camera
  bg_hold      ...and a slot holding it at step t also held it at t-1 (the background keeps its slot)
  person_hit   some slot is within 0.5 px of the person proxy, over the steps where the proxy is
               valid and the person visibly moves (> 0.5 px against the scene); per locomotion
               class too. The proxy (kth_dataset.KTHStore) is a foreground centroid: noisy, and
               meaningless in d2 (zoom); read it as a ranking of settings, not as accuracy
  person_hold  ...and a slot near the person at t was also near it at t-1
  junk         slot velocities beyond 5 px/step (nothing in KTH moves that fast at 32x32)

and, for the number of motions (how many slots does a clip need?), how many peaks per frame pair
stand above z = 4, 5, 6 on the correlation surface (sub-pixel, suppression 1), per action. The
largest of 1024 noise cells on a 32x32 surface is ~3.3 sigma, so z >= 4 is already rare by chance.
Tune on train (or val); test is for the final numbers only.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from motion_benchmarks.common.phase_correlation import phase_correlate  # noqa: E402

from kth.camera_motion import MODES, CameraMotion  # noqa: E402
from kth.kth_dataset import KTH_ACTIONS, KTHClips, KTHStore  # noqa: E402
from kth.kth_model import build_kth_classifier  # noqa: E402

LOCOMOTION = ("walking", "jogging", "running")


@torch.no_grad()
def slot_velocities(model, seq, chunk=512):
    """The velocities _encode_melstm_frame_pair would transport its K slots by: (B, T-1, K, 2)."""
    out = []
    for i in range(0, seq.shape[0], chunk):
        x = seq[i:i + chunk]
        v = torch.zeros(x.shape[0], model.n_velocities, 2, dtype=x.dtype)
        steps = []
        for t in range(1, x.shape[1]):
            cand = model._candidate_velocities(x[:, t - 1], x[:, t]).to(x.dtype)
            v = cand if t == 1 else model._match_to_slots(cand, v)
            steps.append(v)
        out.append(torch.stack(steps, dim=1))
    return torch.cat(out)


def slot_metrics(vel, motion, person, labels):
    cam = motion[:, :-1, 0]                                          # (B, T-1, 2)
    on_bg = (vel.round() == cam[:, :, None]).all(-1)                 # (B, T-1, K)
    bg = on_bg.any(-1)
    bg_both = bg[:, 1:] & bg[:, :-1]
    bg_hold = (on_bg[:, 1:] & on_bg[:, :-1]).any(-1)

    p = person[:, :-1]
    moving = torch.isfinite(p).all(-1) & ((p - cam).abs().amax(-1) > 0.5)
    near = (vel - p[:, :, None]).abs().amax(-1) <= 0.5               # (B, T-1, K)
    hit = near.any(-1) & moving
    both = hit[:, 1:] & hit[:, :-1]
    hold = (near[:, 1:] & near[:, :-1]).any(-1)

    out = {"bg_hit": bg.float().mean().item(),
           "bg_hold": bg_hold[bg_both].float().mean().item() if bg_both.any() else float("nan"),
           "person_hit": hit.sum().item() / max(moving.sum().item(), 1),
           "person_hold": hold[both].float().mean().item() if both.any() else float("nan"),
           "junk": (vel.abs().amax(-1) > 5).float().mean().item()}
    for name in LOCOMOTION:
        m = moving & (labels == KTH_ACTIONS.index(name))[:, None]
        out[f"person_hit_{name[:4]}"] = (hit & m).sum().item() / max(m.sum().item(), 1)
    return out


@torch.no_grad()
def peak_counts(seq, labels, zs=(4.0, 5.0, 6.0), k=8):
    B, T, _, H, W = seq.shape
    a = seq[:, :-1, 0].reshape(-1, H, W)
    b = seq[:, 1:, 0].reshape(-1, H, W)
    _, _, conf = phase_correlate(a, b, k=k, subpixel=True, suppress=1)
    out = {}
    for z in zs:
        n = (conf >= z).sum(-1).reshape(B, T - 1).float()
        row = {"all": n.mean().item()}
        for c, name in enumerate(KTH_ACTIONS):
            row[name] = n[labels == c].mean().item()
        hist = [(n == j).float().mean().item() for j in range(4)] + [(n >= 4).float().mean().item()]
        row["hist_0_1_2_3_4+"] = [round(h, 3) for h in hist]
        out[f"z>={z:g}"] = row
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "data" / "kth"))
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--split_scheme", choices=["keller", "official"], default="keller")
    p.add_argument("--camera", choices=MODES, default="none")
    p.add_argument("--camera_v_range", type=int, default=1)
    p.add_argument("--ks", type=str, default="2,3,4")
    p.add_argument("--max_clips", type=int, default=None)
    p.add_argument("--out", type=str, default="./tmp/kth_pc_analysis")
    a = p.parse_args(argv)

    t0 = time.time()
    store = KTHStore(a.root)
    ds = KTHClips(store, a.split, a.split_scheme, camera=CameraMotion(a.camera,
                  v_range=a.camera_v_range), train=False, seed=42)
    n = len(ds) if a.max_clips is None else min(a.max_clips, len(ds))
    items = [ds[i] for i in range(n)]
    seq = torch.stack([it[0] for it in items])
    labels = torch.tensor([it[1] for it in items])
    motion = torch.stack([it[2] for it in items])
    person = torch.stack([it[3] for it in items])
    print(f"{a.split} ({a.split_scheme}), camera {a.camera}: {n} clips x {seq.shape[1] - 1} steps "
          f"(loaded in {time.time() - t0:.0f}s)")

    p_ok = torch.isfinite(person[:, :-1]).all(-1)
    speed = (person[:, :-1] - motion[:, :-1, 0]).abs().amax(-1)
    print("person proxy |v| (px/step, own motion) median per action:  " + "  ".join(
        f"{name} {speed[(labels == c)[:, None] & p_ok].median():.2f}"
        for c, name in enumerate(KTH_ACTIONS)))

    rows = []
    grid = []
    for K in [int(k) for k in a.ks.split(",")]:
        for source, sub, sup, rad, assign in itertools.product(
                ("frame_pair", "bootstrap"), (0, 1), (0, 1), (None, 5), ("nearest", "shift", "anchored")):
            if source == "bootstrap" and K < 2:
                continue
            grid.append((K, source, sub, sup, rad, assign))
    torch.manual_seed(0)
    for K, source, sub, sup, rad, assign in grid:
        model = build_kth_classifier(dict(model="melstm", num_vel_modes=K, velocity_source=source,
                                          pc_subpixel=sub, pc_suppress_radius=sup,
                                          pc_search_radius=rad, slot_assign=assign,
                                          hidden_size=8, head_channels=8, head_mlp_hidden=8))
        vel = slot_velocities(model, seq)
        m = slot_metrics(vel, motion, person, labels)
        rows.append({"K": K, "source": source, "subpixel": sub, "suppress": sup,
                     "search": rad if rad is not None else "-", "assign": assign, **m})

    cols = ["K", "source", "subpixel", "suppress", "search", "assign", "bg_hit", "bg_hold",
            "person_hit", "person_hit_walk", "person_hit_jogg", "person_hit_runn",
            "person_hold", "junk"]
    print("\n" + " ".join(f"{c[:11]:>11s}" for c in cols))
    for r in rows:
        print(" ".join(f"{r[c]:>11.3f}" if isinstance(r[c], float) else f"{str(r[c]):>11s}"
                       for c in cols))

    counts = peak_counts(seq, labels)
    print("\npeaks per frame pair above z (sub-pixel, suppression 1), mean and histogram 0/1/2/3/4+:")
    for z, row in counts.items():
        per = "  ".join(f"{name[:5]} {row[name]:.2f}" for name in KTH_ACTIONS)
        print(f"  {z}: all {row['all']:.2f}  [{per}]  hist {row['hist_0_1_2_3_4+']}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"pc_{a.split}_{a.camera}.json"
    with open(path, "w") as f:
        json.dump({"split": a.split, "camera": a.camera, "n_clips": n, "settings": rows,
                   "peak_counts": counts}, f, indent=2)
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
