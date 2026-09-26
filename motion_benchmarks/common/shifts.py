"""
Sub-pixel translations on the torus -- numpy and torch, ONE convention.

Convention (the repo's): a displacement is (vx, vy) = (columns, rows), and shifting a field by d
moves its content by +d:

    out[..., y, x] = in[..., y - dy, x - dx]      == torch.roll(in, (dy, dx), dims=(-2, -1))

for integer d. That is the convention of TDMovingMNISTDataset (frames rendered with
torch.roll(img, (cy, cx))), of MEConvLSTMCell.warp and of the FEConvLSTM shift index, so a velocity
produced anywhere in this package can be handed to the models unchanged.

Two interpolants, for two kinds of field:

  fourier  -- exact for band-limited periodic fields (noise textures, rain fields, convection).
              Integer shifts reproduce roll() to float precision. Rings on hard edges.
  bilinear -- for fields with hard edges (masks, sprites). The numpy twin of the grid_sample warp
              in MEConvLSTMCell, including its 1-px circular pad, so it is safe across the wrap.
"""
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================ numpy
def _as_xy(d):
    d = np.asarray(d, dtype=np.float64)
    if d.shape[-1] != 2:
        raise ValueError(f"displacement must end in a (vx, vy) axis, got shape {d.shape}")
    return d[..., 0], d[..., 1]


def fourier_shift_np(a, d):
    """
    Translate a (..., H, W) periodic field by d = (vx, vy) (broadcast over leading axes).

    Uses rfft2/irfft2 (same formulation as fourier_shift_torch, so the two agree to float
    precision). Exact for integer d; for fractional d the only approximation is at the Nyquist
    bins, which carry no energy in a band-limited field.
    """
    a = np.asarray(a)
    H, W = a.shape[-2:]
    dx, dy = _as_xy(d)
    dx = np.asarray(dx)[..., None, None]
    dy = np.asarray(dy)[..., None, None]
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    phase = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    out = np.fft.irfft2(np.fft.rfft2(a.astype(np.float64)) * phase, s=(H, W))
    return out.astype(a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64)


def bilinear_shift_np(a, d):
    """Periodic bilinear translation of a (H, W) field by d = (vx, vy)."""
    from scipy.ndimage import map_coordinates
    a = np.asarray(a, dtype=np.float64)
    H, W = a.shape
    dx, dy = (float(v) for v in _as_xy(d))
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    out = map_coordinates(a, [((Y - dy) % H).ravel(), ((X - dx) % W).ravel()],
                          order=1, mode="grid-wrap")
    return out.reshape(H, W)


def roll_np(a, d):
    """Integer periodic shift; d is rounded. (vx, vy) -> np.roll(..., (dy, dx), axes (-2, -1))."""
    dx, dy = (int(round(float(v))) for v in _as_xy(d))
    return np.roll(a, (dy, dx), axis=(-2, -1))


def cumulative_displacement(motion):
    """
    Per-step velocities -> displacement of every frame relative to frame 0.

        motion[t]       : displacement taking frame t to frame t+1   (repo convention)
        displacement[0] = 0,   displacement[t] = sum(motion[0 .. t-1])

    Works on numpy arrays or torch tensors of shape (T, ..., 2) or (B, T, ..., 2) with time_axis.
    """
    if isinstance(motion, torch.Tensor):
        z = torch.zeros_like(motion[:1])
        return torch.cat([z, torch.cumsum(motion, dim=0)[:-1]], dim=0)
    motion = np.asarray(motion, dtype=np.float64)
    z = np.zeros_like(motion[:1])
    return np.concatenate([z, np.cumsum(motion, axis=0)[:-1]], axis=0)


# ============================================================================ torch
def fourier_shift_torch(x, d):
    """
    Translate x (..., H, W) by d (..., 2) = (vx, vy), exactly on the torus.

    d broadcasts against the leading dims of x: x (B, C, H, W) with d (B, 2) is handled by
    inserting the channel axis automatically (d is expanded to x.ndim - 1 leading dims).
    Differentiable in x (and in d).
    """
    H, W = x.shape[-2:]
    d = d.to(dtype=torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype)
    # align d's leading dims with x's leading dims (right-pad with singleton axes)
    while d.dim() < x.dim() - 1:
        d = d.unsqueeze(-2)
    dx = d[..., 0, None, None]
    dy = d[..., 1, None, None]
    ky = torch.fft.fftfreq(H, device=x.device, dtype=d.dtype)[:, None]
    kx = torch.fft.rfftfreq(W, device=x.device, dtype=d.dtype)[None, :]
    phase = torch.exp(-2j * torch.pi * (ky * dy + kx * dx))
    X = torch.fft.rfft2(x.to(d.dtype))
    return torch.fft.irfft2(X * phase, s=(H, W)).to(x.dtype)


def bilinear_shift_torch(x, d):
    """
    Periodic bilinear translation, x (B, C, H, W), d (B, 2) = (vx, vy).

    Identical sampling to MEConvLSTMCell.warp: sample from a 1-px circular pad so the band
    straddling the wrap interpolates against the opposite edge instead of being clamped.
    """
    B, C, H, W = x.shape
    d = d.to(x.dtype)
    dx = d[:, 0, None, None]
    dy = d[:, 1, None, None]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                            torch.arange(W, device=x.device, dtype=x.dtype), indexing="ij")
    yy = yy.unsqueeze(0).expand(B, -1, -1)
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    xp = F.pad(x, (1, 1, 1, 1), mode="circular")
    gy = 2 * (torch.remainder(yy - dy, H) + 1) / (H + 1) - 1
    gx = 2 * (torch.remainder(xx - dx, W) + 1) / (W + 1) - 1
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(xp, grid, mode="bilinear", padding_mode="border", align_corners=True)


def shift_torch(x, d, method="fourier"):
    """Dispatch: x (B, C, H, W), d (B, 2)."""
    if method == "fourier":
        return fourier_shift_torch(x, d)
    if method == "bilinear":
        return bilinear_shift_torch(x, d)
    raise ValueError(f"unknown shift method {method!r}")


def roll_torch(x, d):
    """Per-sample integer periodic roll; x (B, C, H, W), d (B, 2) rounded."""
    B, C, H, W = x.shape
    dx = d[:, 0].round().long()
    dy = d[:, 1].round().long()
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing="ij")
    sy = (yy.unsqueeze(0) - dy[:, None, None]) % H
    sx = (xx.unsqueeze(0) - dx[:, None, None]) % W
    idx = (sy * W + sx).reshape(B, 1, H * W).expand(-1, C, -1)
    return x.reshape(B, C, H * W).gather(2, idx).view(B, C, H, W)
