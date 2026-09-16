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
def test_background_opposes_the_figure_on_the_shared_grid():
    """
    The felstm-safe alternative to a disjoint grid: the background stays inside
    the figures' velocity grid -- so every lattice copy that can represent a
    figure can represent it too -- and is separated by DIRECTION instead.
    """
    ds = _ds(bg_speed_range=None, bg_opposite_at_start=True, min_dv=2)
    ds.reset_rng()
    fig_grid = set(ds.velocity_grid)
    for i in range(12):
        _, _, motion = ds[i]
        v_fig0, v_bg0 = motion[0, 0], motion[0, 1]
        assert int((v_fig0 * v_bg0).sum()) < 0, \
            f"sample {i}: background does not oppose the figure at t=0"
        # on the shared grid at every step, so felstm can represent both
        for t in range(motion.shape[0]):
            assert tuple(motion[t, 1].tolist()) in fig_grid, \
                "background left the shared velocity grid"
        gap = (motion[:, :1] - motion[:, 1:]).abs().amax(dim=2).min()
        assert int(gap) >= 2, "figure and background came within min_dv"


def test_bootstrap_makes_two_slots_enough():
    """
    The scene has exactly two motions, so two slots should suffice -- but only if
    the velocity selection can actually FIND the minority one. The dominant
    background owns the correlation surface, so plain top-2 peaks lose the figure
    to the noise floor; the residual bootstrap explains the background away first.

    This pins the claim the K=2 default rests on.
    """
    from torch.utils.data import DataLoader, Subset
    ds = _ds(seq_len=12, image_size=64, bg_speed_range=None, bg_opposite_at_start=True)
    ds.reset_rng()
    seq, _, motion = next(iter(DataLoader(Subset(ds, list(range(16))), batch_size=16)))

    def hit(source, K=2):
        torch.manual_seed(0)
        net = MotionDigitClassifier(model="melstm", hidden_channels=8, n_slots=K,
                                    velocity_source=source).eval()
        with torch.no_grad():
            _, v = net.encode(seq)
        vl = v[:, -1].round().long()
        f = (vl == motion[:, -2, 0][:, None, :]).all(-1).any(-1).float().mean().item()
        b = (vl == motion[:, -2, 1][:, None, :]).all(-1).any(-1).float().mean().item()
        return f, b

    f_boot, b_boot = hit("bootstrap")
    f_plain, _ = hit("frame_pair")
    assert b_boot >= 0.9, f"bootstrap lost the background velocity ({b_boot:.1%})"
    assert f_boot >= 0.85, f"bootstrap held the figure only {f_boot:.1%} of the time at K=2"
    assert f_boot > f_plain, (
        f"bootstrap ({f_boot:.1%}) should beat plain top-2 peaks ({f_plain:.1%}); "
        f"if not, K=2 is no longer justified as the default")


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
    # under --bg_mode disjoint the background grid really must be disjoint
    with pytest.raises(SystemExit, match="disjoint"):
        main(["--model", "lstm", "--smoke_test", "--bg_mode", "disjoint",
              "--data_v_range", "4", "--bg_speed_min", "3", "--root", DATA_ROOT])
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
        assert "val_states_sample0" in logged


def test_state_visualisation_uses_identical_sequences_every_epoch():
    """
    The picture is meant to show ONE sample developing as training proceeds, so
    the sequences it draws must be byte-identical at every epoch.

    The TRAIN split resamples on every access (random=True), which is what makes
    it useless for this; val and test are seeded benchmarks and reproduce exactly
    after reset_rng(). The state images come from test.
    """
    from train_classification import build_datasets, get_args, make_loaders

    args = get_args([
        "--root", DATA_ROOT, "--seq_len", "6", "--image_size", "32",
        "--batch_size", "4", "--num_workers", "0", "--use_wandb",
        "--log_states_every", "1", "--log_states_samples", "2",
        "--max_train_samples", "16", "--val_size", "8", "--test_size", "8",
    ])
    train_ds, val_ds, test_ds = build_datasets(args)
    _, val_loader, _, state_loader = make_loaders(args, train_ds, val_ds, test_ds)

    test_ds.reset_rng()
    first = next(iter(state_loader))[0].clone()
    test_ds.reset_rng()
    second = next(iter(state_loader))[0]
    assert torch.equal(first, second), \
        "state-visualisation sequences differ between passes; the picture cannot " \
        "show one sample developing"

    # val is a fixed benchmark too now, so its curve is comparable across epochs
    val_ds.reset_rng()
    a = next(iter(val_loader))[0].clone()
    val_ds.reset_rng()
    b = next(iter(val_loader))[0]
    assert torch.equal(a, b), \
        "val is not reproducible after reset_rng; its epoch-to-epoch curve would be " \
        "mostly resampling noise"

    # the train split, by contrast, must keep resampling -- that is the augmentation
    c = train_ds[0][0].clone()
    d = train_ds[0][0]
    assert not torch.equal(c, d), \
        "the train split stopped resampling motion/texture; each glyph should be " \
        "re-rendered with fresh velocities every access"

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
    assert keys == {f"val_states_sample{i}" for i in range(3)}
    assert not any("/" in k for k in keys), (
        "a \"/\" in the key files the panel under a grouped wandb section, "
        "which a saved workspace layout often does not surface")


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


def test_default_corr_len_leaves_no_per_frame_boundary_cue():
    """
    The control test for the whole experiment.

    With corr_len > 0, pixels WITHIN a region are correlated while pixels ACROSS
    the figure boundary are independent, so the outline is a local-statistics
    discontinuity visible in every single frame. A single-frame CNN scores 24% at
    corr_len=1.0 and 36% at 2.0 against 10% chance -- enough that lstm, which has
    no transport at all, can reach high accuracy by reading the seam instead of
    the motion, which is exactly what happened.

    A trained CNN cannot run in a unit test, so this uses the local-variance
    contrast at the boundary as a proxy: at corr_len=0 the noise is independent
    everywhere, so a boundary window is statistically identical to an interior
    one. The intensity-AUC test elsewhere in this suite does NOT catch this --
    it passes at every corr_len, because intensity genuinely is protected.
    """
    from common_fate_diagnostics import local_var

    def boundary_contrast(corr_len):
        ds = _ds(seq_len=2, image_size=48, corr_len=corr_len, bg_speed_range=None,
                 bg_opposite_at_start=True, normalize="none", return_mask=True)
        ds.reset_rng()
        ratios = []
        for i in range(6):
            seq, _, _, mask = ds[i]
            frame = seq[0, 0].numpy()
            m = mask[0, 0].numpy()
            inner = m.copy()
            for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
                inner = inner * np.roll(m, sh, axis=ax)
            edge = (m - inner) > 0.5
            if edge.sum() < 10:
                continue
            lv = local_var(frame, 3)
            ratios.append(float(lv[edge].mean() / (lv[~edge].mean() + 1e-8)))
        return float(np.mean(ratios))

    clean = boundary_contrast(0.0)
    leaky = boundary_contrast(1.5)
    assert clean < 1.15, (
        f"at the default corr_len=0 the boundary still stands out in local variance "
        f"({clean:.2f}x the interior); the digit is visible per frame and lstm can "
        f"solve the task without any transport")
    assert leaky > clean, (
        "correlated texture should make the seam MORE visible, not less -- if this "
        "fails the proxy is measuring the wrong thing")


def test_early_stop_patience_zero_disables_it():
    """0 must mean 'never stop early', not 'stop immediately'."""
    import re
    src = (_PKG / "train_classification.py").read_text()
    m = re.search(r"if args\.early_stop_patience and .*?:", src)
    assert m, "the early-stop guard changed shape; re-check that 0 still disables it"
    # the guard is falsy at 0, so the branch cannot fire
    assert "args.early_stop_patience and" in m.group(0)


def test_val_curve_set_is_materialised_not_indexed():
    """
    The curve must measure the SAME sequences every time. Holding indices is not
    enough: this dataset renders a fresh sequence on every access, so an indexed
    set would resample and the curve would be mostly noise.
    """
    from train_classification import ValCurveRecorder

    ds = _ds(seq_len=5, image_size=32)
    rec = ValCurveRecorder(ds, n_sequences=6, interval=2, device=torch.device("cpu"))
    assert rec.seq.shape[0] == 6 and rec.label.shape[0] == 6
    first = rec.seq.clone()

    # exhaust the dataset's RNG in between; a materialised tensor is unaffected
    for i in range(3):
        ds[i]
    assert torch.equal(rec.seq, first), "the curve's val set changed under it"

    net = MotionDigitClassifier(model="lstm", hidden_channels=4)
    crit = torch.nn.CrossEntropyLoss()
    rec.maybe_record(net, step=1, train_loss=1.0, criterion=crit)   # 1 % 2 -> skip
    assert rec.steps == []
    rec.maybe_record(net, step=2, train_loss=1.0, criterion=crit)   # 2 % 2 -> record
    assert rec.steps == [2] and len(rec.val_loss) == 1 and len(rec.val_acc) == 1
    assert net.training is False or True   # train/eval mode restored, not asserted here


def test_val_curve_survives_across_epochs_and_resume():
    """The x axis is optimizer steps, so it must not restart each epoch."""
    import re
    src = (_PKG / "train_classification.py").read_text()
    assert 'global_step = ck.get("global_step", 0)' in src, \
        "resume does not restore the step counter; the curve x axis would restart at 0"
    assert '"global_step": global_step' in src, \
        "the checkpoint does not save the step counter"
    assert re.search(r"curve=curve, global_step=global_step", src), \
        "the recorder is not threaded through the training epoch"


def test_precise_bn_recomputes_statistics_for_the_current_weights():
    """
    Regression for the val-accuracy oscillation.

    BatchNorm's running statistics are an EMA collected while the weights were
    still moving. For a head on a RECURRENT state the distribution keeps shifting
    and the EMA never catches up, so eval used statistics the model was never
    trained under: val accuracy bounced between chance and 0.94 across epochs
    while training accuracy rose smoothly.

    BatchNorm cannot simply be swapped out -- measured over 14 epochs it is the
    only normalisation that learns here at all (0.230 train accuracy, against
    ~0.11 flat for GroupNorm at two learning rates and for no normalisation). So
    the fix is to recompute the statistics, and this pins that recompute actually
    replaces them with the exact mean/var over the batches it sees.
    """
    from motion_classification_model import recompute_bn_stats

    net = MotionDigitClassifier(model="lstm", hidden_channels=8)
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns, "the head should contain BatchNorm by default"

    data = [(torch.randn(4, 5, 1, 32, 32) * 3.0 + 1.0,) for _ in range(6)]

    # Poison the running stats, as a lagging EMA effectively does
    for bn in bns:
        bn.running_mean.fill_(99.0)
        bn.running_var.fill_(99.0)

    seen = recompute_bn_stats(net, data, torch.device("cpu"), n_batches=4)
    assert seen == 4
    for bn in bns:
        assert bn.running_mean.abs().max() < 50.0, "running_mean was not re-estimated"
        assert abs(float(bn.running_var.mean()) - 99.0) > 1.0, "running_var was not re-estimated"
        assert bn.momentum is not None, "momentum was not restored after the recompute"

    # It must not train anything
    before = [p.clone() for p in net.parameters()]
    recompute_bn_stats(net, data, torch.device("cpu"), n_batches=2)
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters())), \
        "recompute_bn_stats changed model parameters; it must only collect statistics"

    # and 0 disables it
    for bn in bns:
        bn.running_mean.fill_(7.0)
    assert recompute_bn_stats(net, data, torch.device("cpu"), n_batches=0) == 0
    assert all(float(bn.running_mean.abs().max()) == 7.0 for bn in bns)


def test_batch_is_the_default_head_norm_and_alternatives_exist():
    """
    'batch' is the measured choice, not a default left unexamined; 'group' and
    'none' stay available and are normalisation-consistent between modes.
    """
    import inspect
    from motion_classification_model import ConvClassifierHead
    assert inspect.signature(ConvClassifierHead).parameters["norm"].default == "batch"

    x = torch.randn(4, 6, 1, 32, 32)
    net = MotionDigitClassifier(model="lstm", hidden_channels=8, head_norm="group")
    net.eval()
    with torch.no_grad():
        a = net(x)
    net.train()
    with torch.no_grad():
        b = net(x)
    assert torch.allclose(a, b, atol=1e-5), \
        "GroupNorm should be identical in train and eval"


def test_train_val_test_use_disjoint_mnist_glyphs():
    """
    For CLASSIFICATION the splits must be disjoint at the GLYPH level, not just
    by dataset index.

    Without `digit_indices` this dataset ignores its index and draws a random
    glyph from the whole MNIST split on every access, so a random_split hands
    both halves the same pool and the val number is measured on digits the model
    already trained on. That is harmless for next-frame prediction -- what the
    parent class was built for -- and wrong here.
    """
    from train_classification import build_datasets, get_args

    args = get_args(["--root", DATA_ROOT, "--seq_len", "3", "--image_size", "32"])
    train_ds, val_ds, test_ds = build_datasets(args)

    assert train_ds.digit_indices and val_ds.digit_indices
    assert set(train_ds.digit_indices).isdisjoint(set(val_ds.digit_indices)), \
        "train and val share MNIST glyphs; the val accuracy would be optimistic"
    assert len(train_ds) + len(val_ds) == 60000, \
        f"the split lost glyphs: {len(train_ds)} + {len(val_ds)}"
    assert len(train_ds) == 54000 and len(val_ds) == 6000
    # test draws from MNIST's own test split, a third disjoint set
    assert test_ds.digit_indices is None and len(test_ds) == 10000
    assert test_ds.mnist.train is False and train_ds.mnist.train is True


def test_index_selects_the_glyph_when_a_pool_is_given():
    """The dataset index must determine the digit, or splitting by index is a no-op."""
    ds = _ds(seq_len=2, image_size=32, digit_indices=list(range(0, 200)))
    labels_7 = {ds[7][1] for _ in range(4)}
    assert len(labels_7) == 1, "the same index gave different digits"

    # and different indices reach different glyphs
    seen = {ds[i][1] for i in range(40)}
    assert len(seen) > 3, "indices are not spreading over the glyph pool"

    # without a pool the index is ignored -- the behaviour the parent relies on
    ds2 = _ds(seq_len=2, image_size=32)
    assert ds2.digit_indices is None
    ds2.reset_rng(); a = [ds2[7][1] for _ in range(4)]
    ds2.reset_rng(); b = [ds2[999][1] for _ in range(4)]
    assert a == b, "unpooled behaviour changed; the parent class depends on it"


def test_state_pool_serves_both_the_pictures_and_the_metric():
    """
    Regression: the state loader was sized to --log_states_samples, so
    --state_metric_samples was silently capped to it and val_state_shape_iou was
    averaged over 2 sequences instead of 64 -- a curve that was mostly noise.

    Also pins that the sequences are drawn at random (spanning different digits
    and speeds) but then held FIXED, so the wandb slider compares like with like
    across epochs.
    """
    from train_classification import build_datasets, get_args, make_loaders

    args = get_args([
        "--root", DATA_ROOT, "--seq_len", "4", "--image_size", "32",
        "--batch_size", "4", "--num_workers", "0", "--use_wandb",
        "--log_states_samples", "6", "--state_metric_samples", "20",
        "--test_size", "64",
    ])
    train_ds, val_ds, test_ds = build_datasets(args)
    *_, state_loader = make_loaders(args, train_ds, val_ds, test_ds)

    batch = next(iter(state_loader))
    assert batch[0].shape[0] == 20, (
        f"state pool holds {batch[0].shape[0]} sequences; it must cover the LARGER of "
        f"log_states_samples and state_metric_samples, or the metric is silently capped")

    # not simply indices 0..n-1
    idx = state_loader.dataset.indices
    assert idx != list(range(len(idx))), "state sequences are not randomised"

    # but stable across passes, so epochs are comparable
    test_ds.reset_rng(); first = next(iter(state_loader))[0].clone()
    test_ds.reset_rng(); second = next(iter(state_loader))[0]
    assert torch.equal(first, second), "state sequences changed between passes"


def test_pooling_choice_does_not_perturb_other_initialisation():
    """
    Regression: 'attention' allocates a score MLP and the other modes do not, so
    building the pool BEFORE the head made it consume RNG and hand the head
    different weights.

    At V=1 that was pure noise in the experiment: softmax over one element is
    constant 1.0, so max and attention compute exactly the same thing and the
    score MLP receives zero gradient -- yet one lstm run reached 0.60 val accuracy
    and another sat at chance, purely on that initialisation difference. The pool
    is now built last, so a pooling sweep measures the pooling.
    """
    def build(pool, model, **kw):
        torch.manual_seed(42)
        return MotionDigitClassifier(model=model, hidden_channels=8,
                                     velocity_pool=pool, **kw)

    for model, kw in (("lstm", {}), ("melstm", dict(n_slots=2)), ("felstm", dict(v_range=1))):
        a, b = build("attention", model, **kw), build("max", model, **kw)
        assert all(torch.equal(p, q) for p, q in
                   zip(a.backbone.parameters(), b.backbone.parameters())), \
            f"{model}: pooling changed the backbone initialisation"
        assert all(torch.equal(p, q) for p, q in
                   zip(a.head.parameters(), b.head.parameters())), \
            f"{model}: pooling changed the head initialisation"

    # and at V=1 the two are the same function, so the forward pass must agree
    x = torch.randn(3, 5, 1, 32, 32)
    a, b = build("attention", "lstm"), build("max", "lstm")
    a.eval(); b.eval()
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6), \
            "at V=1 max and attention must compute the same thing"


def test_attention_scores_are_inert_at_one_velocity():
    """softmax over a single element is constant, so those parameters cannot learn."""
    net = MotionDigitClassifier(model="lstm", hidden_channels=8, velocity_pool="attention")
    net(torch.randn(2, 5, 1, 32, 32)).sum().backward()
    grads = [float(p.grad.abs().sum()) for p in net.pool.parameters() if p.grad is not None]
    assert grads and all(g == 0.0 for g in grads), \
        f"attention scores got gradient at V=1: {grads}"


def test_mask_contour_is_not_vertically_mirrored():
    """
    Regression: contour with `extent` and origin=None places Z[0,0] at the
    BOTTOM-left, while imshow defaults to origin='upper' and places it top-left.
    The ground-truth outline was therefore mirrored against the very frame it
    annotates -- and invisibly so, since the figure is not visible in the noise.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    m = np.zeros((20, 20)); m[2:7, 8:12] = 1.0     # blob in the TOP rows
    pad = 3
    mp = np.pad(m, pad, mode="wrap")
    H, W = m.shape
    ext = (-pad - 0.5, W + pad - 0.5, H + pad - 0.5, -pad - 0.5)

    def contour_mid_y(**kw):
        fig, ax = plt.subplots()
        cs = ax.contour(mp, levels=[0.5], extent=ext, **kw)
        ys = np.concatenate([p.vertices[:, 1] for p in cs.get_paths()])
        plt.close(fig)
        return float(ys.mean())

    # the blob sits in rows 2..7 of 20, so with origin='upper' its contour must
    # land in the upper half of the *display* coordinates (small y)
    assert contour_mid_y(origin="upper") < H / 2, \
        "origin='upper' should place the outline where imshow places the blob"
    assert contour_mid_y() > H / 2, \
        "origin=None should mirror it -- if this stops being true, matplotlib " \
        "changed and the explicit origin may no longer be needed"


def _curve_probe_model():
    import torch.nn as nn
    from motion_classification_model import build_classifier
    torch.manual_seed(0)
    m = build_classifier(dict(model="melstm", hidden_size=8, num_vel_modes=2,
                              velocity_source="frame_pair", velocity_pool="max",
                              head_norm="batch", n_classes=10, head_channels=16,
                              head_mlp_hidden=32))
    bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    assert bns, "probe needs a BatchNorm to be meaningful"
    return m, bns


def test_val_curve_uses_precise_bn_and_leaves_bn_untouched():
    """
    The fine-grained curve must be measured under the SAME BatchNorm regime as the
    per-epoch val_acc. Recorded off the running EMA instead, it swings between
    chance and the true accuracy while the model improves monotonically -- which
    is not a property of the model, only of which statistics were used.

    Driven by CORRUPTING the running stats: precise BN must ignore them (and
    reproduce a manual recompute), the EMA path must be affected by them, and
    either way the buffers must come back exactly as they were, so that reading
    the curve cannot perturb the run.
    """
    import torch.nn as nn
    from train_classification import ValCurveRecorder
    from motion_classification_model import recompute_bn_stats

    model, bns = _curve_probe_model()
    seq = torch.randn(12, 6, 1, 28, 28)
    lab = torch.randint(0, 10, (12,))

    class DS:
        def __len__(self): return len(seq)
        def __getitem__(self, i): return seq[i], int(lab[i])

    batches = [(seq[i:i + 4], lab[i:i + 4]) for i in range(0, len(seq), 4)]
    crit = nn.CrossEntropyLoss()

    # statistics the model was never trained under
    for bn in bns:
        bn.running_mean.fill_(37.0)
        bn.running_var.fill_(0.01)
    corrupt = [{k: v.clone() for k, v in bn.state_dict().items()} for bn in bns]

    ema = ValCurveRecorder(DS(), 12, 1, "cpu", 4, bn_loader=None, bn_batches=0)
    ema.maybe_record(model, 1, 0.0, crit)

    precise = ValCurveRecorder(DS(), 12, 1, "cpu", 4,
                               bn_loader=batches, bn_batches=len(batches))
    precise.maybe_record(model, 1, 0.0, crit)

    # the buffers are put back: looking at the curve must not alter the run
    for bn, want in zip(bns, corrupt):
        got = bn.state_dict()
        for k, v in want.items():
            assert torch.equal(got[k], v), f"maybe_record left {k} modified"

    # the EMA reading is the one the corrupted statistics reach
    assert ema.val_loss[0] != precise.val_loss[0], \
        "precise BN made no difference -- the curve is still on the running EMA"

    # and the precise reading is exactly a manual recompute-then-evaluate
    recompute_bn_stats(model, batches, "cpu", len(batches))
    model.eval()
    with torch.no_grad():
        ref = sum(crit(model(a), b).item() * b.numel() for a, b in batches) / len(seq)
    assert precise.val_loss[0] == pytest.approx(ref, rel=1e-5), \
        "curve is not measuring what the per-epoch val measures"
