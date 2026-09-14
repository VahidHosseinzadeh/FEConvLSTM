"""
The motion-defined-digit classification experiment: data, model, and the two
places it silently goes wrong.

The two failure modes these tests exist to catch:

1. The background accidentally sharing the figure's velocity, which erases the
   figure. `bg_speed_range` makes that impossible by construction, and
   test_background_grid_is_disjoint proves the construction rather than
   sampling and hoping.

2. MEConvLSTM's slots never holding the figure's velocity. Its own protocol
   tracks each slot's hidden state against the next frame, which needs a clean
   template and does not have one when every layer is the same noise. Measured
   here, 'tracked' collapses the slots and finds nothing; 'frame_pair' does not.
   A classifier whose slots never transported at the figure's velocity is
   measuring nothing, however good its loss curve looks.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from common_fate_moving_mnist_dataset import CommonFateMovingMNISTDataset  # noqa: E402
from motion_classification_model import (  # noqa: E402
    MotionDigitClassifier, VelocityPool, build_classifier,
)
from mps_integer_warp import (  # noqa: E402
    verify_equivalence, _integer_shift_warp, _ORIGINAL_WARP,
)
from velocity_model_based_MEConvLSTM_model import MEConvLSTMCell  # noqa: E402

DATA_ROOT = os.environ.get("MNIST_ROOT", str(_PKG.parent / "data"))
SEED = 7


def _ds(**kw):
    params = dict(root=DATA_ROOT, train=False, seq_len=10, image_size=48,
                  max_speed=2, bg_speed_range=(4, 5), motion_mode="piecewise",
                  corr_len=1.0, return_motion=True, random=False, seed=SEED,
                  download=True)
    params.update(kw)
    return CommonFateMovingMNISTDataset(**params)


# ------------------------------------------------------------------ the data
def test_background_grid_is_disjoint():
    """The background can never coincide with a figure -- by construction."""
    ds = _ds()
    fig_grid = set(ds.velocity_grid)
    bg_grid = set(ds.bg_velocity_grid)
    assert fig_grid and bg_grid
    assert not (fig_grid & bg_grid), "figure and background velocity grids overlap"
    assert ds.bg_separation_is_structural, \
        "bg_speed_min - max_speed should already cover min_dv, making the check a theorem"

    ds.reset_rng()
    for i in range(12):
        _, _, motion = ds[i]
        fig = {tuple(v) for v in motion[:, 0].tolist()}
        bg = {tuple(v) for v in motion[:, 1].tolist()}
        assert not (fig & bg), f"sample {i}: figure and background shared a velocity"
        assert all(max(abs(a), abs(b)) <= 2 for a, b in fig)
        assert all(4 <= max(abs(a), abs(b)) <= 5 for a, b in bg)


def test_background_grid_rejects_an_overlapping_range():
    with pytest.raises(ValueError, match="overlaps the figure grid"):
        _ds(max_speed=3, bg_speed_range=(3, 5))


# ----------------------------------------------------------------- the model
@pytest.mark.parametrize("model,kw", [("lstm", {}), ("felstm", dict(v_range=2)),
                                      ("melstm", dict(n_slots=3))])
@pytest.mark.parametrize("pool", ["attention", "max", "mean", "concat"])
def test_forward_shapes(model, kw, pool):
    net = MotionDigitClassifier(model=model, hidden_channels=8,
                                velocity_pool=pool, **kw)
    seq = torch.randn(2, 6, 1, 32, 32)
    logits, aux = net(seq, return_aux=True)
    assert logits.shape == (2, 10)
    if pool == "attention":
        w = aux["pool_weights"]
        assert w.shape == (2, net.n_velocities)
        assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-5), \
            "attention weights must be a distribution over the velocity axis"


def test_capacity_is_matched_across_backbones():
    """
    The three models must differ in TRANSPORT, not in parameter count, or the
    comparison measures capacity instead of structure.
    """
    counts = {}
    for model, kw in (("lstm", {}), ("felstm", dict(v_range=2)), ("melstm", dict(n_slots=4))):
        net = MotionDigitClassifier(model=model, hidden_channels=16, **kw)
        counts[model] = net.parameter_report()["trained"]
    assert len(set(counts.values())) == 1, f"trained parameter counts differ: {counts}"


def test_unused_decoder_is_excluded_from_training():
    net = MotionDigitClassifier(model="melstm", hidden_channels=8, n_slots=2)
    trained = {id(p) for p in net.trainable_parameters()}
    decoder = {id(p) for p in net.backbone.decoder.parameters()}
    assert not (trained & decoder), "the unused decoder leaked into the optimizer"
    assert net.parameter_report()["unused_decoder"] > 0


def test_gradients_reach_the_recurrent_cell():
    """A head that trains while the cell gets no gradient would be a per-frame model."""
    net = MotionDigitClassifier(model="melstm", hidden_channels=8, n_slots=2)
    net(torch.randn(2, 5, 1, 32, 32)).sum().backward()
    g = [p.grad for p in net.backbone.cell.parameters() if p.grad is not None]
    assert g and any(float(x.abs().sum()) > 0 for x in g), \
        "no gradient reached the recurrent cell"


def test_build_classifier_from_config_dict():
    cfg = dict(model="felstm", hidden_size=8, v_range=1, velocity_pool="max")
    net = build_classifier(cfg)
    assert net.model == "felstm" and net.n_velocities == 9
    assert net(torch.randn(1, 4, 1, 32, 32)).shape == (1, 10)


# ------------------------------------------------- the integer-shift MPS warp
def test_integer_shift_warp_is_an_exact_periodic_shift():
    err_roll, dev_gs = verify_equivalence()
    assert err_roll == 0.0, "the gather warp must be bit-exact against torch.roll"
    # grid_sample's own coordinate round-trip is the inexact one, ~1e-6 at unit scale
    assert dev_gs < 1e-4


def test_integer_shift_warp_backpropagates():
    cell = MEConvLSTMCell(1, 3)
    x = torch.randn(2, 2, 3, 8, 8, requires_grad=True)
    u = torch.randint(-3, 4, (2, 2, 2)).float()
    _integer_shift_warp(cell, x, u).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_fractional_velocities_still_use_grid_sample():
    """The gather warp is only valid for integers; anything else must fall through."""
    from mps_integer_warp import _dispatching_warp
    cell = MEConvLSTMCell(1, 3)
    x = torch.randn(2, 2, 3, 8, 8)
    u = torch.tensor([[[0.5, -1.5], [2.0, 1.0]], [[1.5, 0.5], [0.0, 2.0]]])
    assert torch.equal(_dispatching_warp(cell, x, u), _ORIGINAL_WARP(cell, x, u))


# --------------------------------------------------------- the velocity source
def test_frame_pair_keeps_the_slots_distinct_and_finds_the_figure():
    """
    The experiment's precondition. If no slot transports at the figure's
    velocity, the figure is never accumulated coherently and the classifier has
    nothing motion-defined to read -- whatever its accuracy turns out to be.

    'tracked' is not asserted against a threshold here because it is the
    behaviour under test, not a contract; what IS asserted is that frame_pair is
    decisively better, since that is the claim the default rests on.
    """
    ds = _ds(seq_len=12, image_size=64)
    ds.reset_rng()
    from torch.utils.data import DataLoader, Subset
    seq, _, motion = next(iter(DataLoader(Subset(ds, list(range(16))), batch_size=16)))

    def measure(source, K=4):
        torch.manual_seed(0)
        net = MotionDigitClassifier(model="melstm", hidden_channels=8, n_slots=K,
                                    velocity_source=source).eval()
        with torch.no_grad():
            _, v = net.encode(seq)
        v_last = v[:, -1].round().long()
        hit = (v_last == motion[:, -2, 0][:, None, :]).all(-1).any(-1).float().mean().item()
        distinct = np.mean([len({tuple(x) for x in v_last[b].tolist()})
                            for b in range(v_last.shape[0])])
        return hit, distinct

    hit_fp, distinct_fp = measure("frame_pair")
    hit_tr, distinct_tr = measure("tracked")

    assert distinct_fp == 4.0, \
        f"frame_pair slots collapsed ({distinct_fp:.2f}/4); slot identity matching failed"
    assert hit_fp >= 0.5, f"frame_pair held the figure velocity only {hit_fp:.1%} of the time"
    assert hit_fp > hit_tr, (
        f"frame_pair ({hit_fp:.1%}) should beat tracked ({hit_tr:.1%}) on this data -- "
        f"if tracked has caught up, re-check whether the default should change")


# ---------------------------------------------------------------- end to end
def test_training_script_runs_end_to_end(tmp_path):
    """Every model, plumbing included: data -> encoder -> pool -> head -> step."""
    from train_classification import main
    for model in ("lstm", "felstm", "melstm"):
        hist = main([
            "--model", model, "--smoke_test", "--hidden_size", "8",
            "--batch_size", "4", "--image_size", "32", "--seq_len", "5",
            "--num_vel_modes", "2", "--v_range", "2", "--data_v_range", "2",
            "--root", DATA_ROOT, "--save_dir", str(tmp_path / model),
        ])
        assert len(hist["epochs"]) == 2
        assert "test" in hist and 0.0 <= hist["test"]["acc"] <= 1.0
        assert hist["parameters"]["trained"] > 0


def test_training_script_refuses_impossible_configurations():
    from train_classification import main
    # background grid must be disjoint from the figure grid
    with pytest.raises(SystemExit, match="disjoint"):
        main(["--model", "lstm", "--smoke_test", "--data_v_range", "4",
              "--bg_speed_min", "3", "--root", DATA_ROOT])
    # felstm's lattice must be able to represent the figure at all
    with pytest.raises(SystemExit, match="no copy could"):
        main(["--model", "felstm", "--smoke_test", "--v_range", "1",
              "--data_v_range", "2", "--root", DATA_ROOT])


# ------------------------------------------------- states and the run header
@pytest.mark.parametrize("model,kw", [("lstm", {}), ("felstm", dict(v_range=1)),
                                      ("melstm", dict(n_slots=3))])
def test_encode_returns_per_step_states(model, kw):
    """(B, T, V, H, W): one channel-mean map per velocity copy per timestep."""
    net = MotionDigitClassifier(model=model, hidden_channels=8, **kw)
    seq = torch.randn(2, 7, 1, 32, 32)
    h, vel, states = net.encode(seq, return_states=True)
    assert states.shape == (2, 7, net.n_velocities, 32, 32)
    assert h.shape == (2, net.n_velocities, 8, 32, 32)
    # encode() without the flag must still return the 2-tuple forward() unpacks
    assert len(net.encode(seq)) == 2


def test_describe_reports_matching_trained_counts():
    """
    The run header is what catches an accidental capacity difference between the
    three models, so the table it prints has to actually agree with the counts.
    """
    totals = set()
    for model, kw in (("lstm", {}), ("felstm", dict(v_range=2)), ("melstm", dict(n_slots=4))):
        net = MotionDigitClassifier(model=model, hidden_channels=16, **kw)
        rows, trained = net.submodule_report()
        text = net.describe()
        assert "TRAINED" in text and "backbone.cell" in text
        assert f"{trained:,}" in text
        assert trained == net.parameter_report()["trained"]
        # the unused decoder must not be counted as trained
        assert dict((r[0], r[2]) for r in rows)["backbone.decoder"] == \
            net.parameter_report()["unused_decoder"]
        totals.add(trained)
    assert len(totals) == 1, f"models differ in trained parameters: {totals}"


def test_state_visualisation_runs_for_every_model(monkeypatch):
    """
    The figure is the experiment's main qualitative output, so a crash in it
    must fail here rather than 20 epochs into a cluster run.
    """
    import types
    import matplotlib
    matplotlib.use("Agg")

    logged = {}
    fake = types.ModuleType("wandb")
    fake.Image = lambda fig: fig
    fake.log = lambda d, **kw: logged.update(d)
    monkeypatch.setitem(sys.modules, "wandb", fake)

    import importlib
    import visualization
    importlib.reload(visualization)

    B, T, H, W = 2, 6, 32, 32
    frames = torch.rand(B, T, 1, H, W)
    mask = torch.zeros(B, T, 1, H, W)
    mask[:, :, :, 8:20, 8:20] = 1.0
    motion = torch.zeros(B, T, 2, 2, dtype=torch.long)
    motion[:, :, 0] = torch.tensor([1, 2])     # figure, on the lattice
    motion[:, :, 1] = torch.tensor([4, 5])     # background, deliberately off it

    for model, kw in (("lstm", {}), ("felstm", dict(v_range=2)), ("melstm", dict(n_slots=3))):
        net = MotionDigitClassifier(model=model, hidden_channels=4, **kw).eval()
        with torch.no_grad():
            _, vel, states = net.encode(frames, return_states=True)
        logged.clear()
        visualization.log_motion_classification_states(
            states, frames, mask_track=mask, velocities=vel,
            v_list=net.backbone.cell.v_list if model in ("lstm", "felstm") else None,
            gt_motion=motion, split_name="val", epoch=1, num_samples=1)
        assert logged, f"{model}: nothing was logged"
        # positional key, so wandb shows one slider per sample across epochs
        assert "val_velocity_states/sample0" in logged


def test_state_visualisation_uses_identical_sequences_every_epoch():
    """
    The picture is meant to show ONE sample developing as training proceeds, so
    the sequences it draws must be byte-identical at every epoch.

    They cannot come from val: val is split off a random=True dataset, which
    renders a fresh sequence on every access. That is correct for an unbiased
    val metric and useless here -- each epoch would show a different sample and
    nothing could be compared across them.
    """
    from train_classification import build_datasets, get_args, make_loaders

    args = get_args([
        "--root", DATA_ROOT, "--seq_len", "6", "--image_size", "32",
        "--batch_size", "4", "--num_workers", "0", "--use_wandb",
        "--log_states_every", "1", "--log_states_samples", "2",
        "--max_train_samples", "16", "--val_size", "8", "--test_size", "8",
    ])
    train_ds, test_ds = build_datasets(args)
    _, val_loader, _, state_loader = make_loaders(args, train_ds, test_ds)

    test_ds.reset_rng()
    first = next(iter(state_loader))[0].clone()
    test_ds.reset_rng()
    second = next(iter(state_loader))[0]
    assert torch.equal(first, second), \
        "state-visualisation sequences differ between passes; the picture cannot " \
        "show one sample developing"

    # and the contrast that motivates it: val really does resample
    a = next(iter(val_loader))[0].clone()
    b = next(iter(val_loader))[0]
    assert not torch.equal(a, b), \
        "val stopped resampling -- if this changed, re-check whether the separate " \
        "fixed state loader is still needed"

    # the mask must be present, since it is the answer key beside the states
    assert len(next(iter(state_loader))) > 3, \
        "state batch carries no GT mask; return_mask should be on when logging states"


def test_state_logging_does_not_advance_the_wandb_step(monkeypatch):
    """
    Regression: wandb.log() without `step` COMMITS and advances wandb's internal
    counter. Logging one image per sample therefore pushed the counter past the
    current epoch, and the next epoch's wandb.log(row, step=epoch) was rejected
    outright -- "Tried to log to step N that is less than the current step" --
    silently dropping that epoch's metrics as well as misplacing the images.

    So: exactly ONE log call per invocation, carrying every sample, at an
    explicit step.
    """
    import types
    import matplotlib
    matplotlib.use("Agg")

    calls = []
    fake = types.ModuleType("wandb")
    fake.Image = lambda fig: fig
    fake.log = lambda d, **kw: calls.append((set(d), kw.get("step", "MISSING")))
    monkeypatch.setitem(sys.modules, "wandb", fake)

    import importlib
    import visualization
    importlib.reload(visualization)

    B, T, H, W = 3, 5, 32, 32
    frames = torch.rand(B, T, 1, H, W)
    net = MotionDigitClassifier(model="melstm", hidden_channels=4, n_slots=2).eval()
    with torch.no_grad():
        _, vel, states = net.encode(frames, return_states=True)

    visualization.log_motion_classification_states(
        states, frames, velocities=vel, split_name="val", epoch=7, step=7,
        num_samples=3)

    assert len(calls) == 1, (
        f"{len(calls)} wandb.log calls for 3 samples; each step-less call advances "
        f"the counter and drops the following epoch's metrics")
    keys, step = calls[0]
    assert step == 7, f"state images logged at step {step!r}, not the epoch"
    assert keys == {f"val_velocity_states/sample{i}" for i in range(3)}


def test_training_script_has_no_step_less_wandb_logs():
    """
    Every wandb.log in the training script must pass an explicit step, or it
    desynchronises the x axis from the epoch and can drop later epochs. Final
    numbers belong in wandb.summary, which does not touch the step counter.
    """
    import re
    src = (_PKG / "train_classification.py").read_text()
    for m in re.finditer(r"wandb\.log\((.*?)\)\n", src, re.S):
        assert "step=" in m.group(1), \
            f"wandb.log without an explicit step:\n    wandb.log({m.group(1).strip()})"
