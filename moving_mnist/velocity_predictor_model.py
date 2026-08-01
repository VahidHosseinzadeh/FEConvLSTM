import torch
import torch.nn as nn


class PhaseCorrelation(nn.Module):
    def __init__(
        self,
        n_modes=2,
        periodic_bc=True,
        pad_factor=1,
        eps=1e-8,
    ):
        super().__init__()

        self.n_modes = n_modes
        self.periodic_bc = periodic_bc
        self.pad_factor = pad_factor
        self.eps = eps

    def surface(self, frame1, frame2):
        """
        The phase-correlation surface itself, before peak extraction.

        Every operation here is differentiable w.r.t. both inputs (only the
        topk in forward() is not), so this is what a loss can be attached to
        when the goal is to shape a template into peaking at a known
        displacement.

        frame1 : (B, C1, H, W)
        frame2 : (B, C2, H, W)
        ->       (B, H_pad, W_pad)
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
        return torch.fft.irfft2(R, s=(H_pad, W_pad))

    @staticmethod
    def velocity_to_index(v, H_pad, W_pad):
        """
        Inverse of the peak -> velocity mapping below, for use as a target
        index into the flattened surface.

        forward() reads a flat peak index as y = idx // W_pad, x = idx % W_pad
        and reports velocity (-x, -y); so a velocity (vx, vy) sits at
        idx = ((-vy) mod H_pad) * W_pad + ((-vx) mod W_pad).

        v : (..., 2) -> (...) long
        """
        vx = torch.round(v[..., 0]).long()
        vy = torch.round(v[..., 1]).long()
        return ((-vy) % H_pad) * W_pad + ((-vx) % W_pad)

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
        H_pad = H * self.pad_factor
        W_pad = W * self.pad_factor

        corr = self.surface(frame1, frame2)

        # pop peaks
        corr = corr.reshape(B, -1)
        scores, idx = torch.topk(corr, self.n_modes, dim=1)

        y = (idx // W_pad).float()
        x = (idx % W_pad).float()

        if self.periodic_bc:
            x = torch.where(x > W_pad / 2, x - W_pad, x)
            y = torch.where(y > H_pad / 2, y - H_pad, y)

        velocities = torch.stack((-x, -y), dim=-1)

        return velocities, scores
    
