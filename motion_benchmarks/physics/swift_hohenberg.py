#!/usr/bin/env python
"""
Swift-Hohenberg: the canonical reduced model of Rayleigh-Benard rolls near onset.

    d psi / dt = r psi - (1 + lap)^2 psi + g2 psi^2 - psi^3 + noise * xi

(critical wavenumber k_c = 1) on a periodic box of side L = 2 pi n_rolls, sampled on an N x N
grid, so the box holds n_rolls roll wavelengths. Semi-implicit pseudo-spectral stepping (linear
part implicit, nonlinearity explicit), batched over trajectories.

(Scaling k_c with the grid instead -- k_c = 2 pi n / N at unit spacing -- makes the unstable band
so wide relative to k_c that the domain-scale mode wins for small n; fixing k_c = 1 and sizing the
box keeps the wavelength properly selected.)

Why it is here: it is the cheapest physical pattern with a GOLDSTONE MODE -- a roll pattern breaks
continuous translation symmetry, so noise makes the whole pattern slide. That drift is real
motion even in the lab frame (claude/fluids_rbc_plan.md section 5, finding 2), and on top of it
the moving-frame datasets add a controlled, time-dependent frame motion. g2 != 0 favours
hexagons, noise > 0 drives intrinsic evolution.

Measured (48 x 48, alpha-whitened phase correlation, median per-step error in px):
    rolls noise   alpha=1   alpha=0.5   alpha=0
      4   0.00    0.102     0.011      0.048
      4   0.08    0.520     0.103      0.067
      8   0.08    0.950     0.336      0.410
Use alpha = 0.5 on these fields; keep per-step displacement well under half the roll wavelength
(wagon-wheel ambiguity).

CLI: pre-generate a bank of trajectories (the dataset can also build one on the fly and cache it)

    python -m motion_benchmarks.physics.swift_hohenberg --out sh_bank.npz --n_traj 512 \
        --n_snap 120 --n_rolls 2,4 --noise 0.02
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def simulate(n_traj, n_snap, N=48, n_rolls=4, r=0.3, g2=0.0, noise=0.0, dt=0.5,
             steps_between=20, spinup=600, seed=0, batch=128, dtype=np.float32, verbose=False):
    """
    Returns (n_traj, n_snap, N, N). `n_rolls` may be a number or a (lo, hi) range, drawn per
    trajectory (continuous: the pattern wavelength varies across the bank).
    """
    rng = np.random.default_rng(seed)
    my = np.fft.fftfreq(N) * N                                  # integer wavenumbers
    mx = np.fft.rfftfreq(N) * N
    M2 = my[:, None] ** 2 + mx[None, :] ** 2                    # (N, N//2+1)
    out = np.empty((n_traj, n_snap, N, N), dtype)
    n_steps = spinup + (n_snap - 1) * steps_between + 1
    for b0 in range(0, n_traj, batch):
        B = min(batch, n_traj - b0)
        if isinstance(n_rolls, (tuple, list)):
            nr = rng.uniform(n_rolls[0], n_rolls[1], size=B)
        else:
            nr = np.full(B, float(n_rolls))
        k2 = M2[None] / nr[:, None, None] ** 2                  # physical |k|^2, box 2 pi nr
        L = r - (1.0 - k2) ** 2                                 # (B, N, N//2+1)
        denom = 1.0 - dt * L
        psi = 0.1 * rng.standard_normal((B, N, N))
        k = 0
        for n in range(n_steps):
            nl = g2 * psi ** 2 - psi ** 3
            ph = (np.fft.rfft2(psi) + dt * np.fft.rfft2(nl)) / denom
            psi = np.fft.irfft2(ph, s=(N, N))
            if noise > 0:
                psi = psi + noise * np.sqrt(dt) * rng.standard_normal((B, N, N))
            if n >= spinup and (n - spinup) % steps_between == 0 and k < n_snap:
                out[b0:b0 + B, k] = psi
                k += 1
        if verbose:
            print(f"[swift_hohenberg] {b0 + B}/{n_traj} trajectories")
    return out


def dominant_wavenumber(psi):
    """|k| (cycles per domain) of the strongest Fourier mode of a (H, W) field."""
    P = np.abs(np.fft.fft2(psi - psi.mean())) ** 2
    i, j = np.unravel_index(np.argmax(P), P.shape)
    H, W = psi.shape
    return float(np.hypot(np.fft.fftfreq(H)[i] * H, np.fft.fftfreq(W)[j] * W))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_traj", type=int, default=512)
    ap.add_argument("--n_snap", type=int, default=120)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--n_rolls", type=str, default="2,4")
    ap.add_argument("--r", type=float, default=0.3)
    ap.add_argument("--g2", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--steps_between", type=int, default=20)
    ap.add_argument("--spinup", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    nr = [float(v) for v in a.n_rolls.split(",")]
    nr = nr[0] if len(nr) == 1 else tuple(nr)
    bank = simulate(a.n_traj, a.n_snap, N=a.N, n_rolls=nr, r=a.r, g2=a.g2, noise=a.noise,
                    dt=a.dt, steps_between=a.steps_between, spinup=a.spinup, seed=a.seed,
                    verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, psi=bank, dt_snap=a.dt * a.steps_between, N=a.N, r=a.r,
                        g2=a.g2, noise=a.noise, n_rolls=np.atleast_1d(nr))
    print(f"[swift_hohenberg] wrote {a.out}: {bank.shape}")


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
