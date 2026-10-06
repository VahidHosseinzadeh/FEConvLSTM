"""KTH pipeline tests: Keller parity, the camera laws, fixed benchmarks, equivariance, panels."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kth.camera_motion import CameraMotion, displacements, keller_constant_velocities
from kth.kth_dataset import KTH_ACTIONS, KTHClips, KTHStore
from kth.kth_model import build_kth_classifier, record_states


# ------------------------------------------------------------------ Keller parity
def keller_apply_velocity_shift(frame, vy, vx):
    """Verbatim from akandykeller/FERNN kth/train_kth_classification.py (KTHVideoClips)."""
    _, H, W = frame.shape
    vy = vy % H
    vx = vx % W
    if vy == 0 and vx == 0:
        return frame
    max_shift = max(vy, vx)
    kernel_size = max_shift * 2 + 1
    shift_kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=frame.device,
                               dtype=frame.dtype)
    center = kernel_size // 2
    shift_kernel[0, 0, center + vy, center + vx] = 1.0
    pad_y = max_shift % H or H
    pad_x = max_shift % W or W
    padded = F.pad(frame, (pad_x, pad_x, pad_y, pad_y), mode='circular')
    return F.conv2d(padded, shift_kernel, padding=0)


def test_keller_shift_is_a_circular_roll():
    """
    His conv shift == roll by minus the shift, for every shift a clip can produce. Not bit for
    bit: when the largest shift (mod 32) is 1 his one-hot kernel is 3x3, and the CPU convolution
    then takes a Winograd path that rounds (measured: <= 1.2e-7, and only there).
    """
    g = torch.Generator().manual_seed(0)
    frame = torch.rand(1, 32, 32, generator=g)
    for sy in range(-36, 37, 3):
        for sx in range(-36, 37, 5):
            ref = keller_apply_velocity_shift(frame, sy, sx)
            ours = torch.roll(frame, shifts=(-sy, -sx), dims=(1, 2))
            assert torch.allclose(ref, ours, rtol=0, atol=1e-6), (sy, sx)


def test_constant_camera_reproduces_keller_frames(store):
    """Our constant-camera test clips are his (same window; float rounding aside)."""
    ds = KTHClips(store, "test", camera=CameraMotion("constant", v_range=2), seed=42)
    rng = np.random.RandomState(42)                     # his draws, as he makes them
    his = [(rng.randint(-2, 3), rng.randint(-2, 3)) for _ in range(len(ds))]   # (vx, vy)
    for i in (0, 7, 100, len(ds) - 1):
        video, s, e = ds.clip_list[i]
        start = int(ds.starts[i])
        vx, vy = his[i]
        frames = []
        for t in range(16):
            f = torch.from_numpy(store.frames[video][start + 2 * t]).float().unsqueeze(0) / 255.0
            frames.append(keller_apply_velocity_shift(f, vy * t, vx * t))
        ref = torch.stack(frames)                       # (T, 1, H, W)
        clip, _, motion, _ = ds[i]
        assert torch.allclose(clip, ref, rtol=0, atol=1e-6), i       # see the test above
        assert motion[0, 0].tolist() == [-vx, -vy]      # content moves by -(vx, vy)


def test_split_counts(store):
    def n(scheme, split):
        return len(KTHClips(store, split, scheme))
    assert [n("keller", s) for s in ("train", "val", "test")] == [1517, 380, 472]
    assert [n("official", s) for s in ("train", "val", "test")] == [754, 758, 857]


# ------------------------------------------------------------------ camera laws
def _runs(v):
    """Lengths of the constant-velocity runs of a (T, 2) sequence."""
    change = np.any(v[1:] != v[:-1], axis=1)
    edges = np.flatnonzero(change) + 1
    return np.diff(np.concatenate([[0], edges, [len(v)]]))


def test_piecewise_camera_segments_and_grid():
    cam = CameraMotion("piecewise", v_range=1, seq_len=40)
    rng = np.random.RandomState(1)
    seen = set()
    for _ in range(300):
        v = cam.draw(rng)
        assert v.shape == (40, 2) and np.abs(v).max() <= 1
        seen |= {tuple(x) for x in v}
        runs = _runs(v)
        assert all(3 <= r <= 6 for r in runs[:-1]), runs     # 'uniform': every end is a change
    assert (0, 0) in seen and len(seen) == 9                  # V_1 with the camera at rest


def test_shake_respects_vmax_and_moves():
    cam = CameraMotion("shake", shake_amp=(1.0, 3.0), shake_period=(6.0, 16.0), shake_vmax=1)
    rng = np.random.RandomState(2)
    vs = np.stack([cam.draw(rng) for _ in range(500)])
    assert np.abs(vs).max() == 1
    assert (np.abs(vs).sum(axis=(1, 2)) > 0).mean() > 0.95    # nearly every clip shakes
    free = CameraMotion("shake", shake_amp=(3.0, 3.0), shake_period=(6.0, 6.0), shake_vmax=0)
    assert np.abs(free.draw(np.random.RandomState(3))).max() >= 2
    one_axis = CameraMotion("shake", shake_axes="y").draw(np.random.RandomState(4))
    assert np.all(one_axis[:, 0] == 0)


def test_keller_draws_sign_convention():
    v = keller_constant_velocities(5, 1)
    rng = np.random.RandomState(42)
    for i in range(5):
        vx, vy = rng.randint(-1, 2), rng.randint(-1, 2)
        assert v[i].tolist() == [-vx, -vy]
    d = displacements(np.tile(np.array([[2, -1]]), (4, 1)))
    assert d.tolist() == [[0, 0], [2, -1], [4, -2], [6, -3]]


# ------------------------------------------------------------------ fixed benchmarks
def _same(x, y):
    x, y = torch.as_tensor(x), torch.as_tensor(y)
    return x.shape == y.shape and torch.equal(torch.nan_to_num(x, 1e9), torch.nan_to_num(y, 1e9))


def test_eval_items_are_fixed(store):
    cam = CameraMotion("piecewise")
    a = KTHClips(store, "val", camera=cam, seed=42)
    b = KTHClips(store, "val", camera=cam, seed=42)
    for i in (0, 50, 200):
        for x, y in zip(a[i], b[i]):        # the person proxy holds NaN where it is unknown
            assert _same(x, y)
    # through DataLoader workers too (the old Moving MNIST benchmarks duplicated per worker)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(a, list(range(8))),
                                         batch_size=4, num_workers=2)
    got = torch.cat([batch[0] for batch in loader])
    ref = torch.stack([a[i][0] for i in range(8)])
    assert torch.equal(got, ref)
    # paired test conditions: same windows whatever the camera
    c = KTHClips(store, "val", camera=CameraMotion("shake"), seed=42)
    assert np.array_equal(a.starts, c.starts)


def _static_store(T_video=80, seed=0):
    """A fake store holding ONE static video: every frame the same random image."""
    st = object.__new__(KTHStore)
    img = (np.random.RandomState(seed).rand(32, 32) * 255).astype(np.uint8)
    st.frames = {"person01_running_d1": np.repeat(img[None], T_video, axis=0)}
    st.height = st.width = 32
    st.annotations = [("person01_running_d1", 1, T_video - 1)]
    st.fg_thr, st.min_area = 0.08, 8
    st.centroids = {"person01_running_d1": np.full((T_video, 2), np.nan, np.float32)}
    return st, img


@pytest.mark.parametrize("mode", ["constant", "piecewise", "shake"])
def test_frames_move_by_the_reported_camera_velocity(mode):
    """On a static scene, frame t+1 is frame t rolled by motion[t] -- flips included."""
    st, _ = _static_store()
    ds = KTHClips(st, "train", camera=CameraMotion(mode, v_range=1), train=True,
                  resample_camera=True)
    np.random.seed(5)
    for _ in range(20):
        clip, label, motion, _ = ds[0]
        assert KTH_ACTIONS[label] == "running"
        for t in range(15):
            vx, vy = (int(c) for c in motion[t, 0])
            assert torch.equal(clip[t + 1], torch.roll(clip[t], (vy, vx), dims=(1, 2)))


# ------------------------------------------------------------------ models
def test_trained_parameter_counts_match():
    counts = {m: build_kth_classifier(dict(model=m)).parameter_report()["trained"]
              for m in ("lstm", "felstm", "melstm")}
    assert len(set(counts.values())) == 1, counts


def _frame_pair_runs(store, mode, assign, every=8):
    """Encode the same test clips without and with a camera; return everything to compare."""
    torch.manual_seed(0)
    model = build_kth_classifier(dict(model="melstm", hidden_size=16, pc_search_radius=0,
                                      slot_assign=assign)).eval()
    base = KTHClips(store, "test", camera=CameraMotion("none"), seed=42)
    moved = KTHClips(store, "test", camera=CameraMotion(mode), seed=42)
    idx = list(range(0, len(base), every))
    x0 = torch.stack([base[i][0] for i in idx])
    x1 = torch.stack([moved[i][0] for i in idx])
    with torch.no_grad():
        h0, v0 = model.encode(x0)
        h1, v1 = model.encode(x1)
        c0 = torch.stack([model._candidate_velocities(x0[:, t - 1], x0[:, t])
                          for t in range(1, x0.shape[1])], dim=1)
        c1 = torch.stack([model._candidate_velocities(x1[:, t - 1], x1[:, t])
                          for t in range(1, x1.shape[1])], dim=1)
    cam = np.stack([moved.trajectories[i] for i in idx])                             # (B, T, 2)
    step = torch.from_numpy(cam[:, :-1, None, :]).float()
    cand_eq = (c1 == c0 + step).all(-1).all(-1)                                      # (B, T-1)
    near_wrap = (c0.abs().amax(-1) >= 12).any(-1) | (c1.abs().amax(-1) >= 12).any(-1)
    slot_eq = (v1 == v0 + step).all(-1).all(-1)
    return cam, h0, h1, cand_eq, near_wrap, slot_eq


@pytest.mark.parametrize("mode,assign", [("constant", "nearest"), ("piecewise", "shift"),
                                         ("shake", "shift")])
def test_frame_pair_melstm_is_equivariant(store, mode, assign):
    """
    A frame-pair MEConvLSTM follows the camera exactly: shifting frame t by d_t shifts every slot
    velocity by the camera step and the final state by d_{T-1}. 'nearest' slot assignment gives
    this for a constant camera, 'shift' for any camera trajectory. The one exception is phase
    correlation itself, at a junk peak near the +-16 px wrap of the 32-px plane, which flips sign
    under a shift (a search window centred at 0 would add its own edge, so it is off here):
      * candidates are equivariant at every step without a peak near the wrap;
      * every clip that loses equivariance loses it first at such a step, never in the
        assignment;
      * a clip whose velocities stay equivariant has an equivariant final state.
    """
    cam, h0, h1, cand_eq, near_wrap, slot_eq = _frame_pair_runs(store, mode, assign)
    assert (cand_eq | near_wrap).all()
    bad = ~slot_eq.all(-1)
    first = (~slot_eq).float().argmax(-1)
    rows = torch.arange(len(first))
    assert (~cand_eq[rows, first])[bad].all()
    assert slot_eq.all(-1).float().mean() > 0.8
    for b in torch.nonzero(~bad).flatten().tolist():
        d = displacements(cam[b])[-1]
        rolled = torch.roll(h0[b], (int(d[1]), int(d[0])), dims=(-2, -1))
        assert torch.allclose(h1[b], rolled, atol=1e-5), b


def test_nearest_assignment_breaks_under_a_moving_camera(store):
    """The documented limitation 'shift' exists for: the assignment itself loses the camera."""
    _, _, _, cand_eq, _, slot_eq = _frame_pair_runs(store, "shake", "nearest")
    assert cand_eq.float().mean() > 0.98
    assert slot_eq.float().mean() < 0.7


def test_pooled_panel_row_is_the_head_input(store):
    torch.manual_seed(0)
    model = build_kth_classifier(dict(model="felstm", hidden_size=16)).eval()
    ds = KTHClips(store, "test", seed=42)
    x = torch.stack([ds[i][0] for i in range(3)])
    rec = record_states(model, x)
    with torch.no_grad():
        h = model.encode(x)[0]
        feat, _ = model.pool(h)
    assert torch.allclose(rec["pooled"][:, -1], feat.mean(dim=1), atol=1e-6)
    assert rec["copies"].shape[2] == 9
    assert torch.allclose(rec["share"].sum(-1), torch.ones(3), atol=1e-5)
