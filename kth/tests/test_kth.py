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


def test_moving_mnist_laws_without_rest():
    """include_zero=False: Moving MNIST's grid (no (0, 0)); 'constant' then uses the generator."""
    rng = np.random.RandomState(5)
    for mode in ("constant", "stochastic", "piecewise"):
        cam = CameraMotion(mode, v_range=2, transition="smooth", neighbor_kernel="symmetric",
                           p_change=0.5, include_zero=False)
        vs = np.stack([cam.draw(rng) for _ in range(300)])
        assert vs.shape == (300, 16, 2) and np.abs(vs).max() == 2
        assert (np.abs(vs).max(-1) > 0).all()                     # never at rest
    const = CameraMotion("constant", v_range=2, include_zero=False)
    assert not const.uses_keller_draws
    c = np.stack([const.draw(rng) for _ in range(50)])
    assert (c == c[:, :1]).all()                                   # one velocity per clip
    assert len({tuple(x) for x in c[:, 0]}) > 10
    rest = {tuple(CameraMotion("constant", v_range=1).draw(rng)[0]) for _ in range(300)}
    assert (0, 0) in rest                                          # the default keeps it


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
    base = counts["melstm"]
    # a static slot is free (every slot shares the cell); attention adds its scoring MLP,
    # Linear(2C, 32) + Linear(32, 1) at C = 64
    assert build_kth_classifier(dict(model="melstm", num_vel_modes=5, static_slot=1)
                                ).parameter_report()["trained"] == base
    att = build_kth_classifier(dict(model="melstm", velocity_pool="attention"))
    assert att.parameter_report()["trained"] == base + (2 * 64 * 32 + 32) + (32 + 1)


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


@pytest.mark.parametrize("model_name,pool,V", [("felstm", "max", 9), ("melstm", "attention", 4)])
def test_pooled_panel_row_is_the_head_input(store, model_name, pool, V):
    torch.manual_seed(0)
    model = build_kth_classifier(dict(model=model_name, hidden_size=16, velocity_pool=pool)).eval()
    ds = KTHClips(store, "test", seed=42)
    x = torch.stack([ds[i][0] for i in range(3)])
    rec = record_states(model, x)
    with torch.no_grad():
        h = model.encode(x)[0]
        feat, w = model.pool(h)
    assert torch.allclose(rec["pooled"][:, -1], feat.mean(dim=1), atol=1e-6)
    assert rec["copies"].shape[2] == V
    assert torch.allclose(rec["share"].sum(-1), torch.ones(3), atol=1e-5)
    if pool == "attention":
        assert torch.allclose(rec["weights"][:, -1], w, atol=1e-6)
        assert torch.allclose(rec["weights"].sum(-1), torch.ones(3, x.shape[1]), atol=1e-5)
    else:
        assert rec["weights"] is None


def test_static_slot_is_the_smaller_model_plus_a_resting_slot(store):
    """K = 5 with a static slot moves slots 1..4 exactly like the 4-slot model, slot 0 at rest."""
    ds = KTHClips(store, "test", camera=CameraMotion("piecewise"), seed=42)
    x = torch.stack([ds[i][0] for i in range(0, 120, 4)])
    torch.manual_seed(0)
    static = build_kth_classifier(dict(model="melstm", hidden_size=16, num_vel_modes=5,
                                       static_slot=1, pc_search_radius=5)).eval()
    plain = build_kth_classifier(dict(model="melstm", hidden_size=16, num_vel_modes=4,
                                      pc_search_radius=5)).eval()
    with torch.no_grad():
        v5 = static.encode(x)[1]
        v4 = plain.encode(x)[1]
    assert torch.equal(v5[:, :, 0], torch.zeros_like(v5[:, :, 0]))
    assert torch.equal(v5[:, :, 1:], v4)
    assert static.copy_labels()[0] == "s0 (0,0)"
    for bad in (dict(model="lstm", static_slot=1), dict(model="melstm", num_vel_modes=1,
                                                         static_slot=1)):
        with pytest.raises(ValueError):
            build_kth_classifier(bad)


@pytest.mark.parametrize("model_name", ["lstm", "melstm"])
def test_readout_averages_the_last_steps(store, model_name):
    """readout_steps N: the logits are the mean of head(pool(h_t)) over the last N steps."""
    torch.manual_seed(0)
    model = build_kth_classifier(dict(model=model_name, hidden_size=16, readout_steps=3)).eval()
    ds = KTHClips(store, "test", seed=42)
    x = torch.stack([ds[i][0] for i in range(4)])
    states = []
    handle = model.backbone.cell.register_forward_hook(lambda m, i, o: states.append(o[0]))
    with torch.no_grad():
        model.encode(x)
        handle.remove()
        expected = torch.stack([model.head(model.pool(h)[0]) for h in states[-3:]]).mean(0)
        got = model(x)
    assert torch.allclose(got, expected, atol=1e-6)
    one = build_kth_classifier(dict(model=model_name, hidden_size=16, readout_steps=1)).eval()
    one.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.allclose(one(x), model.head(model.pool(states[-1])[0]), atol=1e-6)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_static_slot_with_tracking_and_handover(store, p):
    """K = 5 with a static slot = the 4-slot tracked (or handover) model plus a resting slot."""
    ds = KTHClips(store, "test", camera=CameraMotion("piecewise"), seed=42)
    x = torch.stack([ds[i][0] for i in range(0, 120, 8)])
    cfg = dict(model="melstm", hidden_size=16, velocity_source="tracked", pc_search_radius=0)
    torch.manual_seed(0)
    static = build_kth_classifier(dict(cfg, num_vel_modes=5, static_slot=1))
    torch.manual_seed(0)
    plain = build_kth_classifier(dict(cfg, num_vel_modes=4))
    for m in (static, plain):
        m.train(p > 0)                 # the handover only runs in training
        m.x_track_p = p
    with torch.no_grad():
        torch.manual_seed(1)
        h5, v5 = static.encode(x)
        torch.manual_seed(1)
        h4, v4 = plain.encode(x)
    assert torch.equal(v5[:, :, 0], torch.zeros_like(v5[:, :, 0]))
    assert torch.equal(v5[:, :, 1:], v4)
    assert torch.allclose(h5[:, 1:], h4, atol=1e-6)


def test_camera_mix(store):
    """A fixed, seeded half of the clips shakes, with the very trajectories of the all-shake set."""
    full = KTHClips(store, "train", camera=CameraMotion("shake"), seed=42)
    half = KTHClips(store, "train", camera=CameraMotion("shake"), seed=42, camera_mix=0.5)
    again = KTHClips(store, "train", camera=CameraMotion("shake"), seed=42, camera_mix=0.5)
    assert 0.45 < half.moving.mean() < 0.55
    assert np.array_equal(half.moving, again.moving)
    assert np.array_equal(half.trajectories[half.moving], full.trajectories[half.moving])
    assert not half.trajectories[~half.moving].any()
    windows = KTHClips(store, "val", camera=CameraMotion("shake"), seed=42, camera_mix=0.5,
                       eval_windows=3)
    per_window = np.abs(windows.trajectories).sum(axis=(1, 2)).reshape(-1, 3) > 0
    assert (per_window == per_window[:, :1]).all()          # a clip's windows share its fate
