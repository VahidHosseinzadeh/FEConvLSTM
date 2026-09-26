"""End-to-end smoke runs of the trainers and scripts (tiny sizes, CPU)."""
import json

import pytest

from motion_benchmarks import train_cf_video, train_motion


@pytest.mark.parametrize("model,extra", [
    ("lstm", []),
    ("felstm", ["--v_range", "1"]),
    ("melstm", ["--velocity_source", "frame_pair", "--residual", "lagrangian",
                "--decoder_input", "warped"]),
    ("melstm", ["--velocity_source", "track", "--track_gate_radius", "3",
                "--eval_velocity_mode", "both"]),
    ("melstm_oracle", []),
    ("lstm_stabilized", []),
    ("persistence_lagrangian", []),
])
def test_train_motion_radar(tmp_path, model, extra):
    res = train_motion.main(["--dataset", "radar_synthetic", "--model", model, "--smoke_test",
                             "--image_size", "32", "--gen_pred_frames", "5",
                             "--radar_test_lifetimes", "6", "--save_dir", str(tmp_path)] + extra)
    assert len(res["test"]["mse"]) == 3
    assert "headroom_eulerian_over_lagrangian" in res
    out = next(tmp_path.glob("*/results.json"))
    assert json.load(open(out))["args"]["model"] == model


@pytest.mark.parametrize("dataset,extra", [
    ("swift_hohenberg", ["--sh_train_traj", "4", "--sh_eval_traj", "2", "--sh_snapshots", "20",
                         "--sh_cache", ""]),
    ("trajectories", ["--traj_n_motions", "2", "--num_vel_modes", "2"]),
    ("calcium", ["--ca_synthetic_movies", "1"]),
])
def test_train_motion_other_datasets(tmp_path, dataset, extra):
    res = train_motion.main(["--dataset", dataset, "--model", "melstm", "--velocity_source",
                             "frame_pair", "--smoke_test", "--save_dir", str(tmp_path)] + extra)
    assert "velocity_epe" in res["test"]


def test_train_motion_galilean_mean_flow(tmp_path):
    import h5py
    import numpy as np
    p = tmp_path / "rbc.h5"
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 16, 4, 4, 16, 16)).astype(np.float32)
    data[:, :, 1:3] -= data[:, :, 1:3].mean(axis=(-3, -2, -1), keepdims=True)
    with h5py.File(p, "w") as f:
        d = f.create_dataset("fields", data=data)
        d.attrs["channel_names"] = np.array([b"T", b"u", b"v", b"w"])
        f.attrs.update(dict(Ra=2500.0, Pr=0.7, Lx=2 * np.pi, Ly=2 * np.pi, Lz=2.0, dt_snap=0.5))
    res = train_motion.main(["--dataset", "rbc3d", "--rbc_file", str(p), "--rbc_split", "1,1,1",
                             "--rbc_z_stride", "1", "--fluid_action", "galilean", "--model",
                             "melstm", "--velocity_source", "mean_flow", "--smoke_test",
                             "--save_dir", str(tmp_path)])
    assert res["test"]["velocity_epe"] < 1e-3
    assert "nusselt_abs_err" in res["test"]


@pytest.mark.parametrize("model", ["melstm", "lstm"])
def test_train_cf_video(tmp_path, model):
    res = train_cf_video.main(["--dataset", "weizmann", "--model", model, "--smoke_test",
                               "--readout_steps", "2", "--control", "shuffle",
                               "--save_dir", str(tmp_path)])
    assert 0.0 <= res["test"]["accuracy"] <= 1.0 and "test_shuffled" in res


def test_headroom_script():
    from motion_benchmarks.scripts.headroom import main
    rows = main(["--dataset", "radar_synthetic", "--n", "8", "--input_frames", "6",
                 "--pred_frames", "3", "--image_size", "32", "--leads", "1,3"])
    assert rows[0][1]["headroom"][0] > 0.5
