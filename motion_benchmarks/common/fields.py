"""Random fields on the torus: band-limited noise and power-law (radar-like) fields."""
import numpy as np


def wavenumber_grid(H, W):
    """|k| in cycles per domain (integer wavenumbers), shape (H, W); k[0, 0] = 0."""
    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    return np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def band_limited_noise(H, W, corr_len, rng):
    """Gaussian noise low-passed with a Gaussian of `corr_len` pixels; zero mean, unit variance."""
    z = rng.standard_normal((H, W))
    if corr_len > 0:
        ky = np.fft.fftfreq(H)[:, None]
        kx = np.fft.fftfreq(W)[None, :]
        z = np.real(np.fft.ifft2(np.fft.fft2(z) *
                                 np.exp(-2 * (np.pi * corr_len) ** 2 * (ky ** 2 + kx ** 2))))
    return (z - z.mean()) / (z.std() + 1e-8)


def power_law_amplitude(H, W, beta, k_min=1.0):
    """Spectral amplitude |k|^(-beta/2) (power spectrum ~ |k|^-beta), DC removed."""
    K = wavenumber_grid(H, W)
    amp = np.maximum(K, k_min) ** (-beta / 2.0)
    amp[0, 0] = 0.0
    return amp


def complex_white(H, W, rng):
    return rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))


def power_law_field(H, W, beta, rng):
    """Gaussian random field with power spectrum ~ |k|^-beta, standardised."""
    z = np.real(np.fft.ifft2(power_law_amplitude(H, W, beta) * complex_white(H, W, rng)))
    return (z - z.mean()) / (z.std() + 1e-8)
