"""
Datasets: determinism, ground-truth motion consistency, the physics each one promises, and the
file readers (SEVIR, GOES NetCDF, The Well, RBC HDF5, levelX CSV, Weizmann .mat) on tiny fakes.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from motion_benchmarks.common.phase_correlation import phase_correlate
from motion_benchmarks.datasets.base import GeneratedSequenceDataset


def pc_error(seq, motion, alpha=0.5, radius=None, window=None, slot=0):
    v, _, _ = phase_correlate(seq[:-1], seq[1:], alpha=alpha, radius=radius, window=window)
    return (v[:, 0] - motion[:-1, slot]).abs().amax(-1)


# ---------------------------------------------------------------------------- base
class _Toy(GeneratedSequenceDataset):
    def generate(self, rng, index):
        return self.pack(rng.standard_normal((3, 4, 4)), np.zeros((3, 2)), label=index)


def test_fixed_items_independent_of_workers():
    ds = _Toy(12, seed=3, random=False)
    a = torch.cat([b[0] for b in DataLoader(ds, batch_size=4, num_workers=0)])
    b = torch.cat([b[0] for b in DataLoader(ds, batch_size=4, num_workers=2)])
    c = torch.cat([b[0] for b in DataLoader(ds, batch_size=5, num_workers=3)])
    assert torch.equal(a, b) and torch.equal(a, c)
    assert len({tuple(x.flatten()[:3].tolist()) for x in a}) == 12


def test_random_items_differ_across_workers():
    ds = _Toy(16, random=True)
    x = torch.cat([b[0] for b in DataLoader(ds, batch_size=4, num_workers=2)])
    assert len({tuple(v.flatten()[:3].tolist()) for v in x}) == 16


# ---------------------------------------------------------------------------- radar
def test_synthetic_radar_properties():
    from motion_benchmarks.datasets.radar_synthetic import SyntheticRadarDataset
    ds = SyntheticRadarDataset(length=6, seq_len=16, image_size=64, lifetime=24.0,
                               freeze_after=8, random=False, seed=1)
    seq, lab, mot = ds[0]
    assert seq.shape == (16, 1, 64, 64) and mot.shape == (16, 1, 2) and lab == 0
    assert torch.equal(ds[0][0], seq) and not torch.equal(ds[1][0], seq)
    wet = float(np.mean([(ds[i][0] > 0).float().mean() for i in range(6)]))
    assert 0.28 < wet < 0.42
    assert (mot[7:] == mot[6]).all()                        # frozen after the context
    errs = torch.cat([pc_error(ds[i][0], ds[i][2]) for i in range(4)])
    assert float(errs.median()) < 0.2


def test_synthetic_radar_headroom_grows_with_lifetime():
    from motion_benchmarks.common.baselines import eulerian_persistence, lagrangian_persistence
    from motion_benchmarks.datasets.radar_synthetic import SyntheticRadarDataset
    ratio = {}
    for life in (3.0, 48.0):
        ds = SyntheticRadarDataset(length=12, seq_len=14, image_size=64, lifetime=life,
                                   freeze_after=10, random=False, seed=2)
        seq = torch.stack([ds[i][0] for i in range(12)])
        inp, tgt = seq[:, :10], seq[:, 10:]
        e = ((eulerian_persistence(inp, 1) - tgt[:, :1]) ** 2).mean()
        lg = ((lagrangian_persistence(inp, 1, alpha=0.5) - tgt[:, :1]) ** 2).mean()
        ratio[life] = float(e / lg)
    assert ratio[3.0] < 1.4 and ratio[48.0] > 3.0


def test_prepare_radar_sevir_and_archive(tmp_path):
    import h5py
    from motion_benchmarks.datasets.archive import FrameArchiveDataset
    from motion_benchmarks.scripts.prepare_radar import main as prep
    fake = tmp_path / "SEVIR_VIL_FAKE.h5"
    rng = np.random.default_rng(0)
    with h5py.File(fake, "w") as f:
        f.create_dataset("vil", data=rng.integers(0, 255, (3, 64, 64, 20), dtype=np.uint8))
        f.create_dataset("id", data=np.array([b"a", b"b", b"c"]))
    out = tmp_path / "vil.h5"
    prep(["--source", "sevir", "--inputs", str(fake), "--downsample", "4", "--out", str(out)])
    with h5py.File(out, "r") as f:
        assert f["frames"].shape == (3, 20, 16, 16) and f.attrs["quantity"] == "vil"
    ds = FrameArchiveDataset(out, 5, 10, crop=12, random=False, seed=0)
    seq, lab, mot = ds[0]
    assert seq.shape == (10, 1, 12, 12) and torch.isnan(mot).all()
    assert 0.0 <= float(seq.min()) and float(seq.max()) <= 1.0
    assert ds.meta["has_motion"] is False and ds.meta["periodic"] is False


def test_goes_reader_scales_and_masks(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    from motion_benchmarks.scripts.download_goes import block_mean, read_cmi, scan_start
    p = tmp_path / "OR_ABI-L2-CMIPC-M6C13_G19_s20251821801172_e20251821803545_c1.nc"
    with netCDF4.Dataset(p, "w") as nc:
        nc.createDimension("y", 8)
        nc.createDimension("x", 8)
        v = nc.createVariable("CMI", "i2", ("y", "x"), fill_value=-1)
        v.scale_factor = 0.01
        v.add_offset = 180.0
        raw = np.full((8, 8), 10000, np.int16)
        raw[0, 0] = -1
        v.set_auto_maskandscale(False)
        v[:] = raw
    a = read_cmi(p)
    assert np.isnan(a[0, 0]) and np.allclose(a[1:, 1:], 280.0)
    assert block_mean(np.ones((8, 8)), 4).shape == (2, 2)
    t = scan_start(p.name)
    assert (t.year, t.month, t.day, t.hour, t.minute) == (2025, 7, 1, 18, 1)


# ---------------------------------------------------------------------------- fluids
@pytest.fixture(scope="module")
def sh_source():
    from motion_benchmarks.datasets.fluids import swift_hohenberg_source
    return swift_hohenberg_source(4, 30, N=48, n_rolls=6.0, noise=0.01, g2=1.0, seed=0)


def test_swift_hohenberg_selects_wavelength():
    from motion_benchmarks.physics.swift_hohenberg import dominant_wavenumber, simulate
    b = simulate(3, 5, N=48, n_rolls=4, noise=0.0, seed=1)
    assert all(abs(dominant_wavenumber(b[i, -1]) - 4) < 0.7 for i in range(3))


def test_moving_frame_regular(sh_source):
    from motion_benchmarks.datasets.fluids import MovingFrameDataset
    ds = MovingFrameDataset(sh_source, 6, 12, action="regular", max_speed=1.5, random=False,
                            seed=0)
    errs = torch.cat([pc_error(ds[i][0], ds[i][2], radius=3) for i in range(4)])
    assert float(errs.median()) < 0.1
    none = MovingFrameDataset(sh_source, 2, 12, action="none", random=False, seed=0)
    assert float(none[0][2].abs().max()) == 0.0


def _fluid_source_with_velocity():
    from motion_benchmarks.datasets.fluids import ArraySource
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 20, 3, 16, 16)).astype(np.float32)
    a[:, :, 1:] -= a[:, :, 1:].mean(axis=(-2, -1), keepdims=True)       # zero lab-frame momentum
    return ArraySource(a, channel_names=["T", "u", "v"], spacing=(0.5, 0.25), dt=2.0,
                       ux=[1], uy=[2], T_ch=[0])


def test_moving_frame_galilean_connection_is_exact():
    from motion_benchmarks.datasets.fluids import MovingFrameDataset
    from motion_benchmarks.models.melstm_plus import MeanFlowConnection
    src = _fluid_source_with_velocity()
    ds = MovingFrameDataset(src, 3, 10, action="galilean", max_speed=2.0, random=False, seed=0)
    conn = MeanFlowConnection(**ds.meta["mean_flow"])
    for i in range(3):
        seq, _, mot = ds[i]
        v = conn.velocity(seq)                           # connection read from each snapshot
        assert torch.allclose(v, mot[:, 0], atol=1e-4)   # right-continuous: v(t) = motion[t]


def test_rbc3d_source_height_as_channels(tmp_path):
    import h5py
    from motion_benchmarks.datasets.fluids import MovingFrameDataset, RBC3DSource
    p = tmp_path / "rbc.h5"
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 12, 4, 8, 16, 16)).astype(np.float32)
    with h5py.File(p, "w") as f:
        d = f.create_dataset("fields", data=data)
        d.attrs["channel_names"] = np.array([b"T", b"u", b"v", b"w"])
        f.attrs.update(dict(Ra=2500.0, Pr=0.7, Lx=2 * np.pi, Ly=2 * np.pi, Lz=2.0, dt_snap=0.5))
    src = RBC3DSource(p, runs=[1, 2], z_stride=2)
    assert (src.n_traj, src.C, src.H, src.W) == (2, 16, 16, 16)
    w = src.window(0, 3, 4)
    assert np.allclose(w[:, 0:4], data[1, 3:7, 0, ::2])          # T at heights 0, 2, 4, 6
    assert np.allclose(w[:, 12:16], data[1, 3:7, 3, ::2])        # w
    assert src.ux == [4, 5, 6, 7] and src.uy == [8, 9, 10, 11] and src.uz == [12, 13, 14, 15]
    ds = MovingFrameDataset(src, 2, 6, action="galilean", random=False, seed=0)
    assert ds[0][0].shape == (6, 16, 16, 16) and "physics" in ds.meta


def test_the_well_source(tmp_path):
    import h5py
    from motion_benchmarks.datasets.fluids import MovingFrameDataset, WellSource
    p = tmp_path / "rayleigh_benard_fake.hdf5"
    rng = np.random.default_rng(0)
    b = rng.standard_normal((2, 10, 32, 8)).astype(np.float32)     # (traj, time, x, y)
    v = rng.standard_normal((2, 10, 32, 8, 2)).astype(np.float32)
    with h5py.File(p, "w") as f:
        f.create_dataset("t0_fields/buoyancy", data=b)
        f.create_dataset("t0_fields/pressure", data=b)
        f.create_dataset("t1_fields/velocity", data=v)
        f.create_dataset("dimensions/x", data=np.linspace(0, 4, 32, endpoint=False))
        f.create_dataset("dimensions/time", data=np.arange(10) * 0.25)
        f.attrs["Rayleigh"] = 1e6
        f.attrs["Prandtl"] = 1.0
    src = WellSource(p, downsample=(2, 2))
    assert (src.C, src.H, src.W) == (3, 4, 16) and src.periodic == (False, True)
    w = src.window(1, 2, 3)
    assert np.allclose(w[0, 0], b[1, 2].reshape(16, 2, 4, 2).mean(axis=(1, 3)).T)
    ds = MovingFrameDataset(src, 2, 5, action="regular", max_speed=1.0, random=False, seed=0)
    seq, _, mot = ds[0]
    assert seq.shape == (5, 3, 4, 16) and float(mot[:, 0, 1].abs().max()) == 0.0   # x only


# ---------------------------------------------------------------------------- trajectories
def test_synthetic_trajectories_render_and_motion():
    from motion_benchmarks.datasets.trajectories import TrajectoryVideoDataset, synthetic_tracks
    tr = synthetic_tracks("roundabout", 120, seed=1)
    ds = TrajectoryVideoDataset([tr], 6, 12, image_size=64, meters_per_px=0.5, fps_out=5,
                                sprite="gaussian", n_motions=1, random=False, seed=0)
    seq, _, mot = ds[0]
    assert seq.shape == (12, 1, 64, 64) and torch.isfinite(mot).all()
    speed = torch.linalg.norm(mot[:, 0], dim=-1)
    assert 0.2 < float(speed.mean()) < 5.0
    assert torch.equal(ds[0][0], seq)


def test_levelx_csv_loader(tmp_path):
    pd = pytest.importorskip("pandas")
    from motion_benchmarks.datasets.trajectories import Tracks, load_levelx
    pre = tmp_path / "00"
    rows = []
    for f in range(10):
        rows.append(dict(recordingId=0, trackId=1, frame=f, trackLifetime=f, xCenter=10 + f,
                         yCenter=-5.0, heading=0.0, width=1.8, length=4.5))
        rows.append(dict(recordingId=0, trackId=2, frame=f, trackLifetime=f, xCenter=0.0,
                         yCenter=-f * 0.5, heading=270.0, width=0.6, length=0.6))
    pd.DataFrame(rows).to_csv(f"{pre}_tracks.csv", index=False)
    pd.DataFrame([dict(trackId=1, **{"class": "car"}), dict(trackId=2, **{"class": "pedestrian"})]
                 ).to_csv(f"{pre}_tracksMeta.csv", index=False)
    pd.DataFrame([dict(recordingId=0, frameRate=25, orthoPxToMeter=0.1)]).to_csv(
        f"{pre}_recordingMeta.csv", index=False)
    tr = load_levelx(str(pre), "round")
    assert tr.frame_rate == 25 and tr.ortho == 0.1 and set(tr.cls.tolist()) == {0, 2}
    sl = tr.rows(3)
    assert sorted(tr.track_id[sl].tolist()) == [1, 2]
    tr.save(tmp_path / "rec.npz")
    tr2 = Tracks.load(tmp_path / "rec.npz")
    assert np.allclose(tr2.x, tr.x) and tr2.frame_rate == 25


# ---------------------------------------------------------------------------- calcium
def test_calcium_motion_injection():
    from motion_benchmarks.common.phase_correlation import hann2d
    from motion_benchmarks.datasets.calcium import CalciumMotionDataset, synthetic_calcium_movie
    mv = synthetic_calcium_movie(T=400, H=112, W=112, seed=1)
    ds = CalciumMotionDataset([mv], 6, 16, image_size=64, random=False, seed=0)
    seq, _, mot = ds[0]
    assert seq.shape == (16, 1, 64, 64)
    errs = torch.cat([pc_error(ds[i][0], ds[i][2], alpha=0.25, radius=6, window=hann2d(64, 64))
                      for i in range(6)])
    assert float(errs.median()) < 0.3


# ---------------------------------------------------------------------------- common fate video
def test_weizmann_names_and_mat_loader(tmp_path):
    from scipy.io import savemat
    from motion_benchmarks.datasets.common_fate_video import (load_weizmann_masks,
                                                              parse_weizmann_name)
    assert parse_weizmann_name("daria_walk") == ("daria", "walk")
    assert parse_weizmann_name("lena_walk2") == ("lena", "walk")
    assert parse_weizmann_name("ido_wave2") == ("ido", "wave2")
    m = np.zeros((20, 30, 7), bool)
    m[5:15, 10:20] = True
    savemat(tmp_path / "masks.mat", {"original_masks": {"daria_walk": m, "denis_jack": m},
                                     "aligned_masks": {"daria_walk": m[:10]}})
    d = load_weizmann_masks(tmp_path / "masks.mat")
    assert set(d) == {"daria_walk", "denis_jack"} and d["daria_walk"].shape == (7, 20, 30)


def test_common_fate_video_has_no_single_frame_leak():
    from motion_benchmarks.datasets.common_fate_video import (WeizmannCommonFate,
                                                              build_weizmann_sequences)
    seqs = build_weizmann_sequences(None, image_size=64, n_synthetic=3)
    ds = WeizmannCommonFate([(m, l) for m, l, _ in seqs], 4, 16, random=False, seed=0,
                            return_masks=True)
    seq, lab, mot, masks = ds[0]
    x = seq[:, 0].numpy()
    area = float((masks > 0.5).mean())
    assert 0.2 < area < 0.45
    assert abs(np.corrcoef((x ** 2).ravel(), masks.ravel())[0, 1]) < 0.12
    assert mot.shape == (16, 2, 2)
    speeds = torch.linalg.norm(mot, dim=-1)
    assert torch.allclose(speeds, torch.full_like(speeds, 2.0), atol=1e-5)
    opp = torch.linalg.norm(mot[:, 0] + mot[:, 1], dim=-1)
    assert float(opp.min()) >= 1.5                        # never (near-)opposite


def test_deforming_mnist():
    from motion_benchmarks.datasets.common_fate_video import DeformingCommonFateMNIST
    rng = np.random.default_rng(0)
    digits = (rng.random((10, 28, 28)) > 0.7).astype(np.uint8) * 255
    ds = DeformingCommonFateMNIST(digits, np.arange(10), 4, 8, amp=2.0, dilate=1, random=False)
    seq, lab, mot = ds[3]
    assert seq.shape == (8, 1, 64, 64) and 0 <= lab < 10 and mot.shape == (8, 2, 2)
