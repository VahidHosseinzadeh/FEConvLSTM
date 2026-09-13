"""
Phase-correlation diagnostics for Common-Fate Moving MNIST.

These are the measurement tools that make the two variants in
`common_fate_moving_mnist_dataset.py` decidable without training anything:

    pc_peaks             -- what velocities are present between two frames
    residual_bootstrap   -- the MINORITY velocity, recovered by explaining the
                            dominant one away first
    leaky_accumulate     -- what a transport-equivariant memory accumulates in
                            a frame co-moving at a given velocity
    local_var            -- the "is there structure here" readout that turns an
                            accumulator into a shape estimate

The decisive experiment: accumulate at v_fg, take local_var, compare with the
mask. Under 'moving_mask' the figure is static in that frame and the shape
comes back; under 'static_mask' nothing is static and it does not.

Everything here works in NUMPY AXIS ORDER: a displacement is (dy, dx), matching
`np.roll(x, d, axis=(0, 1))` and the `v_fg` / `d_fg` arrays the generator
returns. It is NOT the (vx, vy) order the Dataset class and the velocity heads
use -- convert with `_yx_to_xy` before comparing against a model's output.
"""
import numpy as np


# ------------------------------------------------------------------- phase correlation
def pc_surface(a, b, eps=1e-8):
    """Normalized cross-power spectrum surface. Peak = displacement taking a -> b."""
    Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
    R = Fb * np.conj(Fa)
    return np.real(np.fft.ifft2(R / (np.abs(R) + eps)))


def pc_peaks(a, b, k=1, radius=8, suppress=2):
    """Top-k peaks as signed integer displacements, restricted to |d|_inf <= radius."""
    s = pc_surface(a, b)
    H, W = s.shape
    win = np.full((H, W), -np.inf)
    idx = np.arange(-radius, radius + 1)
    win[np.ix_(idx % H, idx % W)] = s[np.ix_(idx % H, idx % W)]
    out = []
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(win), win.shape)
        dy = i - H if i > H // 2 else i
        dx = j - W if j > W // 2 else j
        out.append((np.array([dy, dx]), float(win[i, j])))
        for p in range(-suppress, suppress + 1):
            for q in range(-suppress, suppress + 1):
                win[(i + p) % H, (j + q) % W] = -np.inf
    return out


def leaky_accumulate(frames, disp, lam=0.9):
    """h_{t+1} = lam * T_v h_t + x_t, i.e. accumulate in the frame co-moving with disp."""
    T, H, W = frames.shape
    h = np.zeros((H, W), np.float32)
    for t in range(T):
        # pull every frame back into the co-moving frame, then accumulate in place
        h = lam * h + np.roll(frames[t], tuple(-disp[t]), (0, 1))
    return h


def local_var(x, k=3):
    """
    Local variance in a kxk window == a small conv + square + conv. The 'shape'
    feature.

    The kernel is rolled to sit at the origin BEFORE the FFT so the window is
    centred on its output pixel. Padding it at the top-left instead -- the
    obvious way to write this -- offsets the whole map by k//2 pixels, which is
    invisible in a picture and a systematic bias the moment you score the result
    against the true mask.
    """
    ker = np.zeros(x.shape, np.float32)
    ker[:k, :k] = 1.0 / (k * k)
    ker = np.roll(ker, (-(k // 2), -(k // 2)), axis=(0, 1))
    Fk = np.fft.fft2(ker)

    def conv(z):
        return np.real(np.fft.ifft2(np.fft.fft2(z) * Fk))

    mu = conv(x)
    return np.maximum(conv(x * x) - mu * mu, 0)


def residual_bootstrap(F0, F1, radius=8, blur=1.5):
    """Dominant PC peak = the majority (background) motion. Explain it away, then
    re-correlate the unexplained energy to recover the minority (figure) motion."""
    H, W = F0.shape
    v1 = pc_peaks(F0, F1, k=1, radius=radius)[0][0]
    R = np.abs(F0 - np.roll(F1, tuple(-v1), (0, 1)))
    ky = np.fft.fftfreq(H)[:, None]; kx = np.fft.fftfreq(W)[None, :]
    R = np.real(np.fft.ifft2(np.fft.fft2(R) * np.exp(-2 * (np.pi * blur) ** 2 * (ky**2 + kx**2))))
    R = R / (R.max() + 1e-8)
    W1m = np.roll(R, tuple(v1), (0, 1))
    v2 = pc_peaks(R * (F0 - F0.mean()), W1m * (F1 - F1.mean()), k=1, radius=radius)[0][0]
    return v1, v2
