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
