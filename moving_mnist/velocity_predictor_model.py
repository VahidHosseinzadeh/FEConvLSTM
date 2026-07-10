import torch
import torch.nn as nn


class PhaseCorrelation(nn.Module):
    def __init__(
        self,
        n_modes=2,
        periodic_bc=True,
        pad_factor=1,
        eps=1e-8,
        subpixel=False,
    ):
        super().__init__()

        self.n_modes = n_modes
        self.periodic_bc = periodic_bc
        self.pad_factor = pad_factor
        self.eps = eps
        self.subpixel = subpixel

    def _parabolic_subpixel(self, corr, y, x, H, W):
        """
        3-tap parabolic interpolation around each integer peak, independently
        per axis. Refines the topk grid location to a fractional offset in
        (-0.5, 0.5) using the correlation values at the peak and its two
        immediate (periodic) neighbours.

        corr : (B, H, W)         unflattened correlation surface
        y, x : (B, n_modes) long integer peak locations (pre periodic_bc)

        Returns
        -------
        dy, dx : (B, n_modes) float subpixel offsets to add to y, x
        """
        B = corr.shape[0]
        batch_idx = torch.arange(B, device=corr.device)[:, None]

        xm1 = (x - 1) % W
        xp1 = (x + 1) % W
        ym1 = (y - 1) % H
        yp1 = (y + 1) % H

        c = corr[batch_idx, y, x]

        c_xm1 = corr[batch_idx, y, xm1]
        c_xp1 = corr[batch_idx, y, xp1]
        denom_x = c_xm1 - 2 * c + c_xp1
        dx = torch.where(
            torch.abs(denom_x) < 1e-12,
            torch.zeros_like(denom_x),
            0.5 * (c_xm1 - c_xp1) / denom_x,
        )

        c_ym1 = corr[batch_idx, ym1, x]
        c_yp1 = corr[batch_idx, yp1, x]
        denom_y = c_ym1 - 2 * c + c_yp1
        dy = torch.where(
            torch.abs(denom_y) < 1e-12,
            torch.zeros_like(denom_y),
            0.5 * (c_ym1 - c_yp1) / denom_y,
        )

        # A near-zero-but-not-quite-zero denominator gives a badly
        # conditioned parabola (fit is only valid close to the peak) —
        # clamp so refinement stays a "nudge" within the 3-tap neighborhood
        # instead of an occasional wild jump.
        dx = dx.clamp(-0.5, 0.5)
        dy = dy.clamp(-0.5, 0.5)

        return dy, dx

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
        corr = torch.fft.irfft2(R, s=(H_pad, W_pad))  # (B, H_pad, W_pad)

        # pop peaks (keep the unflattened corr around for subpixel lookups)
        corr_flat = corr.reshape(B, -1)
        scores, idx = torch.topk(corr_flat, self.n_modes, dim=1)

        y0 = idx // W_pad
        x0 = idx % W_pad

        y = y0.float()
        x = x0.float()

        if self.subpixel:
            dy, dx = self._parabolic_subpixel(corr, y0, x0, H_pad, W_pad)
            y = y + dy
            x = x + dx

        if self.periodic_bc:
            x = torch.where(x > W_pad / 2, x - W_pad, x)
            y = torch.where(y > H_pad / 2, y - H_pad, y)

        velocities = torch.stack((-x, -y), dim=-1)

        return velocities, scores

