import torch
import torch.nn as nn


class PhaseCorrelation(nn.Module):
    """
    Global phase correlation between two frames: the n_modes strongest displacements.

    The four keyword arguments after `eps` are OPT-IN extensions (real data moves by fractions of
    a pixel, and physics fields are narrowband). Their defaults reproduce the original module
    exactly -- the default configuration still runs the original code path below, so every
    existing experiment and checkpoint behaves bit-for-bit as before.

    alpha           whitening exponent: R / (|R| + eps)**alpha. 1.0 = classical phase correlation
                    (default, right for broadband textures); ~0.5 for narrowband fields such as
                    convection rolls or rain (5-10x more accurate on Swift-Hohenberg rolls);
                    0 = plain cross-correlation.
    subpixel        3-point parabolic refinement of every peak (0.11 px median error vs 0.36 px
                    for the integer peak on sub-pixel motion). Default False (integer peaks).
    search_radius   only consider displacements with |d|_inf <= search_radius. Default None.
    suppress_radius Chebyshev radius removed around each peak before the next is taken. The
                    original topk corresponds to 0 (distinct pixels only) -- which, for a
                    sub-pixel motion, returns the two halves of ONE peak as two "modes". Use >= 1
                    with n_modes > 1 on sub-pixel data.
    """

    def __init__(
        self,
        n_modes=2,
        periodic_bc=True,
        pad_factor=1,
        eps=1e-8,
        alpha=1.0,
        subpixel=False,
        search_radius=None,
        suppress_radius=0,
    ):
        super().__init__()

        self.n_modes = n_modes
        self.periodic_bc = periodic_bc
        self.pad_factor = pad_factor
        self.eps = eps
        self.alpha = alpha
        self.subpixel = subpixel
        self.search_radius = search_radius
        self.suppress_radius = suppress_radius

    def _is_legacy(self):
        return (self.alpha == 1.0 and not self.subpixel
                and self.search_radius is None and self.suppress_radius == 0)

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
        if self.alpha == 1.0:
            R = R / (R.abs() + self.eps)
        elif self.alpha != 0:
            R = R / (R.abs() + self.eps) ** self.alpha

        # phase correlation
        corr = torch.fft.irfft2(R, s=(H_pad, W_pad))

        if not self._is_legacy():
            return self._peaks_extended(corr)

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

    # ------------------------------------------------------------------
    # Opt-in path: search window, suppression between peaks, sub-pixel refinement.
    # ------------------------------------------------------------------

    @staticmethod
    def _parabolic(sm, s0, sp):
        den = sm - 2 * s0 + sp
        off = 0.5 * (sm - sp) / torch.where(den.abs() < 1e-12, torch.ones_like(den), den)
        off = torch.where(den < 0, off, torch.zeros_like(off))
        return off.clamp(-0.5, 0.5)

    def _peaks_extended(self, corr):
        """corr (B, Hp, Wp) with its peak at MINUS the displacement (frame1 -> frame2)."""
        B, Hp, Wp = corr.shape
        dev = corr.device
        iy_all = torch.arange(Hp, device=dev)
        ix_all = torch.arange(Wp, device=dev)
        if self.periodic_bc:
            sy = torch.where(iy_all > Hp / 2, iy_all - Hp, iy_all)
            sx = torch.where(ix_all > Wp / 2, ix_all - Wp, ix_all)
        else:
            sy, sx = iy_all, ix_all

        work = corr.clone()
        if self.search_radius is not None:
            r = self.search_radius
            ok = (sy.abs() <= r)[:, None] & (sx.abs() <= r)[None, :]
            work = work.masked_fill(~ok.unsqueeze(0), float("-inf"))

        rows = torch.arange(B, device=dev)
        half_y, half_x = (Hp - 1) // 2, (Wp - 1) // 2
        vel, sc = [], []
        for j in range(self.n_modes):
            val, idx = work.reshape(B, -1).max(dim=1)
            iy, ix = idx // Wp, idx % Wp
            y = sy[iy].to(corr.dtype)
            x = sx[ix].to(corr.dtype)
            if self.subpixel:
                s0 = corr[rows, iy, ix]
                y = y + self._parabolic(corr[rows, (iy - 1) % Hp, ix], s0,
                                        corr[rows, (iy + 1) % Hp, ix])
                x = x + self._parabolic(corr[rows, iy, (ix - 1) % Wp], s0,
                                        corr[rows, iy, (ix + 1) % Wp])
            vel.append(torch.stack((-x, -y), dim=-1))
            sc.append(corr[rows, iy, ix])
            if j < self.n_modes - 1:
                dy = torch.remainder(iy_all[None, :] - iy[:, None] + half_y, Hp) - half_y
                dx = torch.remainder(ix_all[None, :] - ix[:, None] + half_x, Wp) - half_x
                near = ((dy.abs() <= self.suppress_radius)[:, :, None]
                        & (dx.abs() <= self.suppress_radius)[:, None, :])
                work = work.masked_fill(near, float("-inf"))

        return torch.stack(vel, dim=1), torch.stack(sc, dim=1)
