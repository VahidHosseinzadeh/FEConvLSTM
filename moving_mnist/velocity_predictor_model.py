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
        differentiable=False,
        temperature=0.1,
        grad_scale=1.0,
    ):
        super().__init__()

        self.n_modes = n_modes
        self.periodic_bc = periodic_bc
        self.pad_factor = pad_factor
        self.eps = eps
        self.subpixel = subpixel

        # Straight-through estimator: forward pass is unchanged (still the
        # exact hard topk peak, +subpixel if enabled); backward pass uses
        # the gradient of a soft-argmax (softmax-weighted expected position)
        # over the same correlation surface. Lets gradient reach whatever
        # produced frame1 (e.g. a learnable reduction of h) without changing
        # runtime behaviour at all. Only meaningful for a single peak per
        # correlation surface — n_modes=1 — since softmax-weighted expectation
        # collapses multiple peaks to one centroid, which isn't a sensible
        # target when n_modes>1 (e.g. bootstrap's K-peak case).
        #
        # temperature controls how sharply the soft-argmax concentrates
        # around the true peak (semantic accuracy of v_soft as an estimate).
        # grad_scale is a separate, independent multiplier on the resulting
        # gradient's magnitude. These two need to be decoupled: empirically,
        # a temperature that makes v_soft land close to the hard peak also
        # produces a gradient that can be comparable in magnitude to (or
        # larger than) the model's other gradients — dangerous, since
        # clip_grad_norm_ rescales the *whole* parameter vector by one
        # global norm, so a dominant/noisy path can drown out the legitimate
        # reconstruction-loss gradient for every other parameter too.
        # grad_scale lets this path be a gentle nudge instead.
        self.differentiable = differentiable
        self.temperature = temperature
        self.grad_scale = grad_scale
        if differentiable:
            assert n_modes == 1, (
                "PhaseCorrelation(differentiable=True) only supports n_modes=1: "
                "soft-argmax gives one centroid per correlation surface, which "
                "isn't a meaningful target when extracting >1 peak."
            )

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

    def _soft_argmax(self, corr, H, W):
        """
        Differentiable expected peak location: normalize the whole
        correlation surface into a distribution over grid positions via
        softmax, then take its mean. Unlike torch.topk, every step here has
        a well-defined gradient with respect to corr.

        corr : (B, H, W) unflattened correlation surface

        Returns
        -------
        y_soft, x_soft : (B,) float
        """
        B = corr.shape[0]
        corr_flat = corr.reshape(B, -1)
        # Scale by corr's own per-sample std, not an absolute value: corr's
        # numeric range depends on input statistics (and can drift over
        # training), so an absolute temperature would need re-tuning
        # whenever that scale changes. self.temperature is then a unitless
        # "sharpness in std units" knob instead of tied to a specific scale.
        corr_std = corr_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
        weights = torch.softmax(corr_flat / (self.temperature * corr_std), dim=1)  # (B, H*W)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=corr.device, dtype=corr.dtype),
            torch.arange(W, device=corr.device, dtype=corr.dtype),
            indexing="ij",
        )
        yy = yy.reshape(-1)  # (H*W,)
        xx = xx.reshape(-1)

        y_soft = (weights * yy[None, :]).sum(dim=1)  # (B,)
        x_soft = (weights * xx[None, :]).sum(dim=1)

        return y_soft, x_soft

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

        if self.differentiable:
            # Straight-through: value stays exactly the hard (+subpixel) peak
            # computed above; gradient becomes grad_scale times the gradient
            # of the soft-argmax below (see grad_scale note in __init__).
            y_soft, x_soft = self._soft_argmax(corr, H_pad, W_pad)
            y_soft = (self.grad_scale * y_soft).unsqueeze(1)  # (B, 1), n_modes == 1 (asserted in __init__)
            x_soft = (self.grad_scale * x_soft).unsqueeze(1)
            y = y_soft + (y - y_soft).detach()
            x = x_soft + (x - x_soft).detach()

        if self.periodic_bc:
            x = torch.where(x > W_pad / 2, x - W_pad, x)
            y = torch.where(y > H_pad / 2, y - H_pad, y)

        velocities = torch.stack((-x, -y), dim=-1)

        return velocities, scores

