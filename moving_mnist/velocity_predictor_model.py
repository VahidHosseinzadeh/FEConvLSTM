import torch
import torch.nn as nn


class PhaseCorrelation(nn.Module):
    def __init__(
        self,
        n_modes=2,
        periodic_bc=True,
        pad_factor=1,
        eps=1e-8,
        max_shift=None,
    ):
        super().__init__()

        self.n_modes = n_modes
        self.periodic_bc = periodic_bc
        self.pad_factor = pad_factor
        self.eps = eps
        # Search only displacements with max(|vx|, |vy|) <= max_shift. None = the
        # whole surface. A plain attribute, so a caller may switch it per call site.
        self.max_shift = max_shift
        self._window_cache = {}

    def _window(self, H, W, device):
        """(H*W,) bool: which surface cells lie within max_shift of zero shift."""
        key = (H, W, self.max_shift, device)
        if key not in self._window_cache:
            y = torch.arange(H, device=device)
            x = torch.arange(W, device=device)
            y = torch.where(y > H / 2, y - H, y).abs()
            x = torch.where(x > W / 2, x - W, x).abs()
            self._window_cache[key] = (
                torch.maximum(y[:, None], x[None, :]) <= self.max_shift).reshape(-1)
        return self._window_cache[key]

    def forward(self, frame1, frame2):
        """
        Parameters
        ----------
        frame1 : (B, C1, H, W)
        frame2 : (B, C2, H, W)

        Returns
        -------
        velocities : (B, n_modes, 2)
        scores     : (B, n_modes)
        """

        B, _, H, W = frame1.shape
        B2, _, H2, W2 = frame2.shape

        assert B == B2 and H == H2 and W == W2

        H_pad = H * self.pad_factor
        W_pad = W * self.pad_factor

        # collapse channels 
        frame1 = frame1.mean(dim=1)  # (B, H, W)
        frame2 = frame2.mean(dim=1)  # (B, H, W)

        # FFT
        F1 = torch.fft.rfft2(frame1, s=(H_pad, W_pad))
        F2 = torch.fft.rfft2(frame2, s=(H_pad, W_pad))

        # cross-power spectrum
        R = F1 * torch.conj(F2)
        R = R / (R.abs() + self.eps)

        # phase correlation
        corr = torch.fft.irfft2(R, s=(H_pad, W_pad))

        # pop peaks
        corr = corr.reshape(B, -1)
        if self.max_shift is not None:
            corr = corr.masked_fill(~self._window(H_pad, W_pad, corr.device), float("-inf"))
        scores, idx = torch.topk(corr, self.n_modes, dim=1)

        y = (idx // W_pad).float()
        x = (idx % W_pad).float()

        if self.periodic_bc:
            x = torch.where(x > W_pad / 2, x - W_pad, x)
            y = torch.where(y > H_pad / 2, y - H_pad, y)

        velocities = torch.stack((-x, -y), dim=-1)

        return velocities, scores
    
