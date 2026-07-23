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
        max_disp=None,
    ):
        super().__init__()

        if max_disp is not None:
            assert (2 * max_disp + 1) ** 2 >= n_modes, (
                f"max_disp={max_disp} only offers {(2*max_disp+1)**2} candidate "
                f"offsets, fewer than n_modes={n_modes}"
            )

        self.n_modes = n_modes
        self.periodic_bc = periodic_bc
        self.pad_factor = pad_factor
        self.eps = eps
        # Refine each integer peak with parabolic (quadratic) interpolation
        # of its 4 grid neighbors -> sub-pixel velocity instead of raw
        # whole-pixel argmax.
        self.subpixel = subpixel
        # Restrict peak search to the known +/- max_disp window (e.g. the
        # data's v_range) instead of the full periodic correlation surface,
        # so noise can't produce an out-of-range velocity. None = unrestricted.
        self.max_disp = max_disp

        # (H_pad, W_pad, device) -> bool mask. Same rationale as
        # MEConvLSTMCell._meshgrid_cache: depends only on shape/device, not
        # on batch content, so build once instead of every forward call.
        self._disp_mask_cache = {}

    def _disp_mask(self, H_pad, W_pad, device):
        key = (H_pad, W_pad, device)
        cached = self._disp_mask_cache.get(key)
        if cached is None:
            yy, xx = torch.meshgrid(
                torch.arange(H_pad, device=device),
                torch.arange(W_pad, device=device),
                indexing="ij",
            )
            yy = torch.where(yy > H_pad // 2, yy - H_pad, yy)
            xx = torch.where(xx > W_pad // 2, xx - W_pad, xx)
            cached = (yy.abs() <= self.max_disp) & (xx.abs() <= self.max_disp)
            self._disp_mask_cache[key] = cached
        return cached

    def _parabolic_subpixel(self, corr, y, x):
        """
        corr : (B, H, W)
        y, x : (B, n_modes) integer peak indices
        returns dy, dx of shape (B, n_modes), each in [-0.5, 0.5]

        Neighbors of a max_disp-masked-out peak can be -inf; the quadratic
        fit is skipped (falls back to dx/dy=0) wherever any of the 3 samples
        it needs isn't finite, rather than propagating inf/nan.
        """
        B, H, W = corr.shape
        batch_idx = torch.arange(B, device=corr.device)[:, None]

        xm1 = (x - 1) % W
        xp1 = (x + 1) % W
        ym1 = (y - 1) % H
        yp1 = (y + 1) % H

        c = corr[batch_idx, y, x]

        c_xm1 = corr[batch_idx, y, xm1]
        c_xp1 = corr[batch_idx, y, xp1]
        denom_x = c_xm1 - 2 * c + c_xp1
        valid_x = torch.isfinite(c_xm1) & torch.isfinite(c) & torch.isfinite(c_xp1)
        dx = torch.where(
            valid_x & (denom_x.abs() >= 1e-12),
            0.5 * (c_xm1 - c_xp1) / denom_x,
            torch.zeros_like(denom_x),
        )

        c_ym1 = corr[batch_idx, ym1, x]
        c_yp1 = corr[batch_idx, yp1, x]
        denom_y = c_ym1 - 2 * c + c_yp1
        valid_y = torch.isfinite(c_ym1) & torch.isfinite(c) & torch.isfinite(c_yp1)
        dy = torch.where(
            valid_y & (denom_y.abs() >= 1e-12),
            0.5 * (c_ym1 - c_yp1) / denom_y,
            torch.zeros_like(denom_y),
        )

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

        if self.max_disp is not None:
            mask = self._disp_mask(H_pad, W_pad, corr.device)
            corr = corr.masked_fill(~mask.unsqueeze(0), float("-inf"))

        # pop peaks
        scores, idx = torch.topk(corr.reshape(B, -1), self.n_modes, dim=1)

        y = idx // W_pad
        x = idx % W_pad

        if self.subpixel:
            dy, dx = self._parabolic_subpixel(corr, y, x)
            y = y.float() + dy
            x = x.float() + dx
        else:
            y = y.float()
            x = x.float()

        if self.periodic_bc:
            x = torch.where(x > W_pad / 2, x - W_pad, x)
            y = torch.where(y > H_pad / 2, y - H_pad, y)

        velocities = torch.stack((-x, -y), dim=-1)

        return velocities, scores

