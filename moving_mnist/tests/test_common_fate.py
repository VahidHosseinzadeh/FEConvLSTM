"""
Characterise Common-Fate Moving MNIST, and check the property the dataset
exists to create.

The claim under test is not "the frames look right" -- nothing in a single
frame looks like anything. It is:

  1. No single frame carries the figure. Intensity is a chance-level classifier
     of figure-vs-background (AUC ~ 0.5) in both variants, because both layers
     are drawn from the same texture.

  2. Under variant='moving_mask' the figure is STATIC in the frame co-moving at
     v_fg, so leaky accumulation there plus a local-variance readout recovers
     the SHAPE. Under variant='static_mask' nothing is static in any frame, and
     the same measurement recovers essentially nothing. That gap is the
     experiment: it is what a transport-equivariant memory can exploit and a
     per-frame model cannot.

  3. Phase correlation reads the dominant (background) velocity off two frames,
     and the residual bootstrap pulls the minority (figure) velocity out from
     under it -- in the moving_mask case. See test 3 for why static_mask is
     expected to fail there, which is a property of the aperture, not a bug.

Run directly to print the table:

    python moving_mnist/tests/test_common_fate.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from common_fate_moving_mnist_dataset import (  # noqa: E402
    CommonFateMovingMNISTDataset, make_sequence, _yx_to_xy,
)
from common_fate_diagnostics import (  # noqa: E402
    leaky_accumulate, local_var, pc_peaks, residual_bootstrap,
)

DATA_ROOT = os.environ.get("MNIST_ROOT", str(_PKG.parent / "data"))

N_TRIALS = int(os.environ.get("N_TRIALS", 12))
SEQ_LEN = 20
IMAGE_SIZE = 64
MAX_SPEED = 3
SEED = 20240912

# Observed over 12 sequences: moving_mask IoU 0.51 +- 0.04 (min 0.44),
# static_mask 0.012 +- 0.022 (max ~0.07). The thresholds sit well inside that
# gap so the test fails on a real regression, not on noise.
IOU_MOVING_MIN = 0.30
IOU_STATIC_MAX = 0.12


# ----------------------------------------------------------------------- helpers
def _mnist():
    from torchvision.datasets import MNIST
    return MNIST(root=DATA_ROOT, train=True, download=True)


def _glyphs(n, rng):
    mn = _mnist()
    return [np.asarray(mn[int(rng.integers(len(mn)))][0], dtype=np.uint8)
            for _ in range(n)]


def area_matched_iou(score, mask):
    """
    IoU after thresholding `score` at whatever level selects exactly as many
    pixels as the mask contains.

    Area-matching removes the threshold as a free parameter, so the two
    variants are compared on the ranking their score induces and nothing else.
    An arbitrary fixed threshold would flatter whichever variant happens to
    produce the larger dynamic range.
    """
    k = int(mask.sum())
    thr = np.partition(score.ravel(), -k)[-k]
    pred = score >= thr
    truth = mask > 0.5
    return float((pred & truth).sum() / (pred | truth).sum())


def intensity_auc(frame, mask):
    """AUC of raw intensity as a figure-vs-background classifier. 0.5 = no cue."""
    pos = frame[mask > 0.5].ravel()
    neg = frame[mask <= 0.5].ravel()
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    rank_pos = ranks[:len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def shape_recovery_iou(out, lam=0.9, k=3):
    """
    Accumulate the sequence in the frame co-moving at v_fg, then score the
    local-variance map against the mask.

    leaky_accumulate pulls frame t back by d_fg[t], so in 'moving_mask' the
    figure lands on the ORIGINAL mask m every step and adds coherently, while
    the background arrives at a different offset each step and averages itself
    down. Local variance turns "accumulated coherently" into a scalar per pixel.
    """
    h = leaky_accumulate(out["frames"], out["d_fg"], lam=lam)
    return area_matched_iou(local_var(h, k), out["mask"])


def sample(variant, rng, glyph, **kw):
    params = dict(T=SEQ_LEN, H=IMAGE_SIZE, W=IMAGE_SIZE, vmax=MAX_SPEED,
                  corr_len=1.0, variant=variant, time_varying=False,
                  min_dv=2, rng=rng)
    params.update(kw)
    return make_sequence(glyph, **params)


# ------------------------------------------------------------------------ tests
def test_no_single_frame_cue():
    """Both layers share texture statistics, so one frame says nothing."""
    rng = np.random.default_rng(SEED)
    glyphs = _glyphs(N_TRIALS, rng)
    for variant in ("moving_mask", "static_mask"):
        # Frame and mask must come from the SAME sequence -- scoring a frame
        # against an unrelated mask returns 0.5 whatever the data does.
        aucs = []
        for g in glyphs:
            out = sample(variant, rng, g)
            aucs.append(intensity_auc(out["frames"][0], out["mask"]))
        assert 0.45 <= np.mean(aucs) <= 0.55, \
            f"{variant}: frame-0 intensity separates figure from ground, AUC={np.mean(aucs):.3f}"


def test_transport_recovers_shape_only_for_moving_mask():
    """The decisive property. moving_mask: shape comes back. static_mask: it does not."""
    rng = np.random.default_rng(SEED)
    glyphs = _glyphs(N_TRIALS, rng)

    moving = [shape_recovery_iou(sample("moving_mask", rng, g)) for g in glyphs]
    static = [shape_recovery_iou(sample("static_mask", rng, g)) for g in glyphs]

    assert np.mean(moving) >= IOU_MOVING_MIN, \
        f"moving_mask transport should recover the shape, IoU={np.mean(moving):.3f}"
    assert np.mean(static) <= IOU_STATIC_MAX, \
        f"static_mask transport should NOT recover the shape, IoU={np.mean(static):.3f}"
    assert np.mean(moving) > 3 * np.mean(static), \
        "the two variants must be decisively separated, not merely ordered"


def test_phase_correlation_reads_the_velocities():
    """
    The dominant PC peak is the background motion (it owns most of the pixels);
    the residual bootstrap recovers the figure motion from what is left.

    Only asserted for moving_mask. In static_mask the figure's aperture is
    fixed while its texture scrolls, so the bootstrap's assumption -- that the
    unexplained region travels with the dominant velocity -- does not hold and
    the second peak is not v_fg. That is a property of a static aperture, not a
    defect: it is the same reason transport cannot recover the shape there.
    """
    rng = np.random.default_rng(SEED)
    glyphs = _glyphs(N_TRIALS, rng)

    bg_hits, fg_hits = 0, 0
    for g in glyphs:
        out = sample("moving_mask", rng, g)
        F0, F1 = out["frames"][0], out["frames"][1]
        v_bg_hat, v_fg_hat = residual_bootstrap(F0, F1)
        bg_hits += tuple(v_bg_hat) == tuple(out["v_bg"][0])
        fg_hits += tuple(v_fg_hat) == tuple(out["v_fg"][0])

    assert bg_hits >= 0.9 * N_TRIALS, f"background velocity recovered {bg_hits}/{N_TRIALS}"
    assert fg_hits >= 0.8 * N_TRIALS, f"figure velocity recovered {fg_hits}/{N_TRIALS}"


def test_dataset_contract():
    """Shapes, ranges and the (vx, vy) convention the velocity heads assume."""
    ds = CommonFateMovingMNISTDataset(
        root=DATA_ROOT, train=False, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
        max_speed=MAX_SPEED, return_motion=True, return_mask=True,
        random=False, seed=SEED, download=True)

    seq, label, motion, mask = ds[0]

    assert seq.shape == (SEQ_LEN, 1, IMAGE_SIZE, IMAGE_SIZE) and seq.dtype == torch.float32
    assert 0.0 <= float(seq.min()) and float(seq.max()) <= 1.0, "affine normalize must land in [0,1]"
    assert isinstance(label, int) and 0 <= label <= 9
    assert motion.shape == (SEQ_LEN, 2, 2) and motion.dtype == torch.int64
    assert mask.shape == (SEQ_LEN, IMAGE_SIZE, IMAGE_SIZE)
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}

    # motion[t, 0] = (vx, vy) of the figure. In moving_mask the mask travels
    # with it, so the frame-to-frame mask shift must equal (vy, vx) in (row,
    # col). This is the one place the numpy (dy, dx) world meets the repo's
    # (vx, vy) world, and a transpose here would be invisible everywhere else.
    vx, vy = int(motion[0, 0, 0]), int(motion[0, 0, 1])
    rolled = np.roll(mask[0].numpy(), (vy, vx), axis=(0, 1))
    assert np.array_equal(rolled, mask[1].numpy()), \
        "motion is not (vx, vy), or the mask track does not follow the figure"

    assert (motion[:, 0] != motion[:, 1]).any(), "figure and background share a velocity"

    # Fixed-benchmark contract: reset_rng rewinds the stream exactly. The
    # generator is stateful, so both passes must start from a reset -- the
    # draws above have already advanced it.
    ds.reset_rng()
    first = ds[0][0].clone()
    ds.reset_rng()
    assert torch.equal(first, ds[0][0]), "reset_rng did not restore the sequence"


def test_guards():
    """Configurations that would silently produce a figureless set are refused."""
    for kw, msg in [
        (dict(num_digits=2), "num_digits"),
        (dict(min_dv=7), "min_dv"),
        (dict(variant="nope"), "variant"),
        (dict(normalize="nope"), "normalize"),
    ]:
        try:
            CommonFateMovingMNISTDataset(root=DATA_ROOT, train=False,
                                         max_speed=MAX_SPEED, download=True, **kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {msg}={kw}")


# ------------------------------------------------------------------------- report
def main():
    rng = np.random.default_rng(SEED)
    glyphs = _glyphs(N_TRIALS, rng)

    print(f"Common-Fate Moving MNIST | {N_TRIALS} sequences, T={SEQ_LEN}, "
          f"{IMAGE_SIZE}x{IMAGE_SIZE}, vmax={MAX_SPEED}\n")
    print(f"{'variant':<14}{'IoU(shape|transport)':>22}{'AUC(1 frame)':>15}"
          f"{'v_bg hit':>10}{'v_fg hit':>10}")
    print("-" * 71)

    for variant in ("moving_mask", "static_mask"):
        ious, aucs, bg, fg = [], [], 0, 0
        for g in glyphs:
            out = sample(variant, rng, g)
            ious.append(shape_recovery_iou(out))
            aucs.append(intensity_auc(out["frames"][0], out["mask"]))
            v_bg_hat, v_fg_hat = residual_bootstrap(out["frames"][0], out["frames"][1])
            bg += tuple(v_bg_hat) == tuple(out["v_bg"][0])
            fg += tuple(v_fg_hat) == tuple(out["v_fg"][0])
        print(f"{variant:<14}{np.mean(ious):>12.3f} +-{np.std(ious):<7.3f}"
              f"{np.mean(aucs):>15.3f}{bg:>7}/{N_TRIALS}{fg:>7}/{N_TRIALS}")

    print("\nIoU is area-matched, so 'chance' is roughly the mask's area fraction "
          f"(~{np.mean([sample('moving_mask', rng, g)['mask'].mean() for g in glyphs[:4]]):.3f}).")
    print("moving_mask: the figure is static in the co-moving frame -> shape.")
    print("static_mask: nothing is static in any frame -> no shape, by construction.")


if __name__ == "__main__":
    main()
