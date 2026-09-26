"""
Phase correlation: the extended legacy module must be bit-identical by default, and the new
estimators must get sign, sub-pixel accuracy, windows and multiple motions right.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from motion_benchmarks.common.phase_correlation import (estimate_peaks_np, estimate_sequence_np,
                                                        estimate_shift_np, gated_track, hann2d,
                                                        match_to_slots, phase_correlate,
                                                        residual_peaks)
from motion_benchmarks.common.shifts import bilinear_shift_np, fourier_shift_np, fourier_shift_torch
from velocity_predictor_model import PhaseCorrelation

from .conftest import band_limited


class LegacyPhaseCorrelation(nn.Module):
    """Verbatim copy of the ORIGINAL module (commit ec55db2), the reference for bit-identity."""

    def __init__(self, n_modes=2, periodic_bc=True, pad_factor=1, eps=1e-8):
        super().__init__()
        self.n_modes, self.periodic_bc, self.pad_factor, self.eps = n_modes, periodic_bc, pad_factor, eps

    def forward(self, frame1, frame2):
        B, _, H, W = frame1.shape
        H_pad, W_pad = H * self.pad_factor, W * self.pad_factor
        frame1, frame2 = frame1.mean(dim=1), frame2.mean(dim=1)
        F1 = torch.fft.rfft2(frame1, s=(H_pad, W_pad))
        F2 = torch.fft.rfft2(frame2, s=(H_pad, W_pad))
        R = F1 * torch.conj(F2)
        R = R / (R.abs() + self.eps)
        corr = torch.fft.irfft2(R, s=(H_pad, W_pad)).reshape(B, -1)
        scores, idx = torch.topk(corr, self.n_modes, dim=1)
        y, x = (idx // W_pad).float(), (idx % W_pad).float()
        if self.periodic_bc:
            x = torch.where(x > W_pad / 2, x - W_pad, x)
            y = torch.where(y > H_pad / 2, y - H_pad, y)
        return torch.stack((-x, -y), dim=-1), scores


@pytest.mark.parametrize("n_modes,pad", [(1, 1), (2, 1), (3, 2)])
def test_legacy_default_is_bit_identical(n_modes, pad):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 2, 36, 40, generator=g)
    y = torch.roll(x, (3, -5), (2, 3)) + 0.3 * torch.randn(6, 2, 36, 40, generator=g)
    a = LegacyPhaseCorrelation(n_modes=n_modes, pad_factor=pad)(x, y)
    b = PhaseCorrelation(n_modes=n_modes, pad_factor=pad)(x, y)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_legacy_sign_convention_matches_roll():
    x = torch.randn(1, 1, 32, 32)
    v, _ = PhaseCorrelation(n_modes=1)(x, torch.roll(x, (2, -3), (2, 3)))
    assert v[0, 0].tolist() == [-3.0, 2.0]          # (vx, vy) = displacement frame1 -> frame2


def test_extended_module_subpixel_and_alpha():
    rng = np.random.default_rng(0)
    z = torch.tensor(np.stack([band_limited(48, 48, 1.0, rng) for _ in range(24)]))[:, None]
    d = torch.tensor(rng.uniform(-3, 3, (24, 2)), dtype=torch.float32)
    zs = fourier_shift_torch(z, d)
    v_int, _ = PhaseCorrelation(n_modes=1)(z, zs)
    v_sub, _ = PhaseCorrelation(n_modes=1, subpixel=True)(z, zs)
    v_half, _ = PhaseCorrelation(n_modes=1, subpixel=True, alpha=0.5)(z, zs)
    assert (v_int[:, 0] - d).abs().max() <= 0.5 + 1e-4
    assert (v_sub[:, 0] - d).abs().median() < 0.15
    assert (v_half[:, 0] - d).abs().median() < 0.08


def test_extended_module_window_and_suppression():
    x = torch.randn(4, 1, 40, 40)
    y = torch.roll(x, (12, 11), (2, 3))
    v, _ = PhaseCorrelation(n_modes=1, search_radius=4)(x, y)
    assert (v[:, 0].abs() <= 4).all()                  # the true shift is outside the window
    v2, _ = PhaseCorrelation(n_modes=2, subpixel=True, suppress_radius=1)(x, y)
    assert (v2[:, 0] - v2[:, 1]).abs().amax(dim=-1).min() > 1.0


def test_common_matches_legacy_sign(noise_field):
    a = torch.tensor(noise_field)[None, None]
    for d in [(3, -2), (-5, 4), (0, 7)]:
        b = torch.roll(a, (d[1], d[0]), (2, 3))
        v_new, _, _ = phase_correlate(a, b, subpixel=False)
        v_old, _ = PhaseCorrelation(n_modes=1)(a, b)
        assert v_new[0, 0].tolist() == list(map(float, d)) == v_old[0, 0].tolist()


def test_subpixel_accuracy(rng):
    a = band_limited(64, 48, 1.0, rng)
    errs = {1.0: [], 0.5: []}
    for _ in range(60):
        d = rng.uniform(-4, 4, 2)
        b = fourier_shift_np(a, d)
        for al in errs:
            est, conf = estimate_shift_np(a, b, alpha=al)
            errs[al].append(np.abs(est - d).max())
    assert np.median(errs[1.0]) < 0.15 and max(errs[1.0]) < 0.3
    assert np.median(errs[0.5]) < 0.08


def test_window_around_previous_velocity(noise_field):
    a = torch.tensor(noise_field)[None]
    b = torch.roll(a, (14, -13), (1, 2))
    v, _, _ = phase_correlate(a, b, radius=2, center=torch.tensor([[-12.0, 13.0]]), subpixel=False)
    assert v[0, 0].tolist() == [-13.0, 14.0]
    v, _, _ = phase_correlate(a, b, radius=2, subpixel=False)      # window around 0 misses it
    assert v[0, 0].abs().max() <= 2


def two_layer_pair(rng, vf=(1.6, -0.7), vb=(-2.2, 1.1), size=64):
    A, B = band_limited(size, size, 1.0, rng), band_limited(size, size, 1.0, rng)
    m = np.zeros((size, size))
    m[18:46, 14:42] = 1
    m1 = bilinear_shift_np(m, vf)
    f0 = m * A + (1 - m) * B
    f1 = m1 * fourier_shift_np(A, vf) + (1 - m1) * fourier_shift_np(B, vb)
    return f0, f1


def test_two_motions_peaks_and_residual(rng):
    f0, f1 = two_layer_pair(rng)
    peaks, conf = estimate_peaks_np(f0, f1, k=2, suppress=2)
    truth = np.array([[-2.2, 1.1], [1.6, -0.7]])
    for t in truth:
        assert np.linalg.norm(peaks - t, axis=1).min() < 0.5
    v, _ = residual_peaks(torch.tensor(f0)[None].float(), torch.tensor(f1)[None].float(), k=2)
    v = v[0].numpy()
    assert np.linalg.norm(v[0] - truth[0]) < 0.3          # dominant = background (larger area)
    assert np.linalg.norm(v[1] - truth[1]) < 0.4          # minority recovered after explaining


def test_estimate_sequence_convention(noise_field):
    motion = np.array([[1.0, 0.0], [0.5, -1.5], [2.0, 1.0], [0.0, 0.0]])
    D = np.concatenate([[[0, 0]], np.cumsum(motion, 0)[:-1]])
    frames = np.stack([fourier_shift_np(noise_field, D[t]) for t in range(4)])
    est = estimate_sequence_np(frames)
    assert np.abs(est[:3] - motion[:3]).max() < 0.2        # est[t] = displacement t -> t+1


def test_gated_track_coasts_on_low_confidence(noise_field):
    t = torch.tensor(noise_field)[None]
    f = torch.randn(1, 1, 48, 48)                          # unrelated frame: no real peak
    v_prev = torch.tensor([[1.25, -0.5]])
    v, conf, acc = gated_track(t, f, v_prev, min_conf=1e6)
    assert not acc.item() and torch.equal(v, v_prev)
    f2 = torch.roll(t, (1, 2), (1, 2))[:, None]
    v, conf, acc = gated_track(t, f2, v_prev, min_conf=3.0, radius=3)
    assert acc.item() and torch.allclose(v, torch.tensor([[2.0, 1.0]]), atol=0.05)


def test_hann_window_helps_nonperiodic_crops(rng):
    big = band_limited(160, 160, 2.0, rng)
    errs = {False: [], True: []}
    for _ in range(30):
        d = rng.uniform(-2, 2, 2)
        a = big[40:104, 40:104]
        b = fourier_shift_np(big, d)[40:104, 40:104]
        for w in errs:
            est, _ = estimate_shift_np(a, b, alpha=0.5, window=w)
            errs[w].append(np.abs(est - d).max())
    assert np.median(errs[True]) < np.median(errs[False])


def test_match_to_slots_keeps_identity():
    prev = torch.tensor([[[1.0, 0.0], [-2.0, 1.0]]])
    cand = torch.tensor([[[-2.1, 1.1], [0.9, 0.1]]])
    out = match_to_slots(cand, prev)
    assert torch.allclose(out, torch.tensor([[[0.9, 0.1], [-2.1, 1.1]]]))


def test_hann2d_shape():
    w = hann2d(8, 10)
    assert w.shape == (8, 10) and float(w.max()) <= 1.0 and float(w[0].abs().max()) == 0.0
