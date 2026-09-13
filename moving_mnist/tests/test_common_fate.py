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
    CommonFateMovingMNISTDataset, make_sequence, cumulative_displacement,
    _xy_to_yx, _yx_to_xy,
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
    # N figures + 1 background layer, background last
    assert motion.shape == (SEQ_LEN, 2, 2) and motion.dtype == torch.int64
    assert mask.shape == (SEQ_LEN, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}

    assert (motion[:, 0] != motion[:, 1]).any(), "figure and background share a velocity"

    # Fixed-benchmark contract: reset_rng rewinds the stream exactly. The
    # generator is stateful, so both passes must start from a reset -- the
    # draws above have already advanced it.
    ds.reset_rng()
    first = ds[0][0].clone()
    ds.reset_rng()
    assert torch.equal(first, ds[0][0]), "reset_rng did not restore the sequence"


def test_motion_indexing_matches_the_rendered_displacement():
    """
    motion[t] must be the step taking frame t to frame t+1 -- TDMovingMNISTDataset's
    convention, and what every velocity head in this repo is trained against.

    The naive cumsum puts frame 0 at v[0] instead, which makes motion[t] the step
    from t-1 to t. A CONSTANT velocity hides that completely, so this test runs a
    time-varying mode on purpose: it is the only setting where the off-by-one is
    observable at all.
    """
    for mode in ("piecewise", "stochastic"):
        ds = CommonFateMovingMNISTDataset(
            root=DATA_ROOT, train=False, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
            max_speed=MAX_SPEED, motion_mode=mode, variant="moving_mask",
            return_motion=True, return_mask=True, random=False, seed=SEED,
            download=True)
        ds.reset_rng()
        changed = 0
        for i in range(4):
            _, _, motion, mask = ds[i]
            for t in range(SEQ_LEN - 1):
                vx, vy = int(motion[t, 0, 0]), int(motion[t, 0, 1])
                rolled = np.roll(mask[t, 0].numpy(), (vy, vx), axis=(0, 1))
                assert np.array_equal(rolled, mask[t + 1, 0].numpy()), (
                    f"{mode}: motion[{t}] is not the step from frame {t} to {t+1} "
                    f"(or motion is not (vx, vy))")
                changed += int(t > 0 and not torch.equal(motion[t, 0], motion[t - 1, 0]))
        assert changed > 0, f"{mode} produced no velocity change; the test proves nothing"


def test_cumulative_displacement_convention():
    """displacement[0] = 0 and displacement[t] = sum(v[0..t-1])."""
    v = np.array([[1, 2], [3, 4], [5, 6]])
    d = cumulative_displacement(v)
    assert np.array_equal(d, [[0, 0], [1, 2], [4, 6]])
    assert np.array_equal(np.diff(d, axis=0), v[:-1]), \
        "consecutive displacements must differ by the velocity of the EARLIER frame"


def test_inherits_the_parent_motion_vocabulary():
    """The point of subclassing: every TDMovingMNISTDataset motion mode works."""
    for mode in ("constant", "piecewise", "stochastic", "accelerate"):
        ds = CommonFateMovingMNISTDataset(
            root=DATA_ROOT, train=False, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
            max_speed=MAX_SPEED, motion_mode=mode, return_motion=True,
            random=False, seed=SEED, download=True)
        _, _, motion = ds[0]
        assert motion.shape == (SEQ_LEN, 2, 2)
        n_distinct = len({tuple(v) for v in motion[:, 0].tolist()})
        if mode == "constant":
            assert n_distinct == 1, "constant mode changed the figure velocity"
        assert (motion[:, 0] != motion[:, 1]).any()

    # motion_difficulty must not trip the parent's "family overridden" warning
    # just because this class defaults motion_mode to 'constant'.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        CommonFateMovingMNISTDataset(
            root=DATA_ROOT, train=False, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
            max_speed=MAX_SPEED, motion_difficulty=0.5, download=True)

    # freeze_after: the velocity is constant from freeze_after-1 onward
    ds = CommonFateMovingMNISTDataset(
        root=DATA_ROOT, train=False, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
        max_speed=MAX_SPEED, motion_mode="stochastic", freeze_after=8,
        return_motion=True, random=False, seed=SEED, download=True)
    _, _, motion = ds[0]
    assert (motion[7:] == motion[7]).all(), "freeze_after did not freeze the velocity"


def test_multiple_figures():
    """N figures + 1 background, with labels aligned to the figure motion slots."""
    N = 2
    ds = CommonFateMovingMNISTDataset(
        root=DATA_ROOT, train=False, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
        num_figures=N, max_speed=MAX_SPEED, motion_mode="piecewise",
        variant="moving_mask", return_motion=True, return_positions=True,
        return_mask=True, random=False, seed=SEED, download=True)

    seq, labels, motion, positions, mask = ds[0]
    assert seq.shape == (SEQ_LEN, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert labels.shape == (N,) and len(set(labels.tolist())) == N, \
        "require_distinct_digits should give a well-posed set-prediction target"
    assert motion.shape == (SEQ_LEN, N + 1, 2)
    assert positions.shape == (SEQ_LEN, N, 2)
    assert mask.shape == (SEQ_LEN, N, IMAGE_SIZE, IMAGE_SIZE)

    # Each figure's own mask track must follow its own motion slot -- this is
    # what makes labels[i], motion[:, i] and mask[:, i] refer to one object.
    for i in range(N):
        vx, vy = int(motion[0, i, 0]), int(motion[0, i, 1])
        rolled = np.roll(mask[0, i].numpy(), (vy, vx), axis=(0, 1))
        assert np.array_equal(rolled, mask[1, i].numpy()), \
            f"figure {i}'s mask does not follow motion slot {i}"

    # Every figure separated from the background at every step
    gap = (motion[:, :N] - motion[:, N:]).abs().amax(dim=2)
    assert int(gap.min()) >= 2, "a figure travelled with the background"


def test_still_no_single_frame_cue_with_two_figures():
    """Adding figures must not add a per-frame intensity cue."""
    ds = CommonFateMovingMNISTDataset(
        root=DATA_ROOT, train=False, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
        num_figures=2, max_speed=MAX_SPEED, normalize="none",
        return_mask=True, random=False, seed=SEED, download=True)
    ds.reset_rng()
    aucs = []
    for i in range(8):
        seq, _, _, mask = ds[i]
        any_fig = mask[0].amax(dim=0).numpy()
        aucs.append(intensity_auc(seq[0, 0].numpy(), any_fig))
    assert 0.45 <= np.mean(aucs) <= 0.55, \
        f"two-figure frames leak the figures through intensity, AUC={np.mean(aucs):.3f}"


def test_guards():
    """Configurations that would silently produce a figureless set are refused."""
    for kw, msg in [
        (dict(num_figures=0), "num_figures"),
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


def test_transport_selects_the_right_figure_with_two():
    """
    With two figures moving differently, transporting at figure j's velocity must
    recover figure j and not the other one.

    This is the property that makes the multi-figure setting worth having: the
    co-moving frame does not merely reveal "some shape", it SELECTS the object
    whose velocity you transported at. Without it, two figures would just be one
    noisier segmentation problem.
    """
    for variant, lo, hi in (("moving_mask", 0.30, None), ("static_mask", None, 0.12)):
        ds = CommonFateMovingMNISTDataset(
            root=DATA_ROOT, train=False, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
            num_figures=2, max_speed=MAX_SPEED, variant=variant, corr_len=1.0,
            min_dv=2, separate_figures=True, normalize="none",
            return_motion=True, return_mask=True, random=False, seed=99, download=True)
        ds.reset_rng()

        own, other = [], []
        for i in range(6):
            seq, _, motion, mask = ds[i]
            frames = seq[:, 0].numpy()
            disp = _xy_to_yx(cumulative_displacement(motion.numpy()))
            for j in range(2):
                score = local_var(leaky_accumulate(frames, disp[:, j], lam=0.9), 3)
                own.append(area_matched_iou(score, mask[0, j].numpy()))
                other.append(area_matched_iou(score, mask[0, 1 - j].numpy()))

        if lo is not None:
            assert np.mean(own) >= lo, \
                f"{variant}: transport did not recover the transported figure, IoU={np.mean(own):.3f}"
            assert np.mean(own) > 3 * np.mean(other), (
                f"{variant}: transport at figure j's velocity is not SELECTIVE -- "
                f"own={np.mean(own):.3f} vs other={np.mean(other):.3f}")
        if hi is not None:
            assert np.mean(own) <= hi, \
                f"{variant}: transport should recover nothing, IoU={np.mean(own):.3f}"


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
