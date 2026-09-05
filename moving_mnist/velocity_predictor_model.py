import torch
import torch.nn as nn


class PhaseCorrelation(nn.Module):
    """
    Displacement between two maps, from the peak of a (generalised) correlation
    surface computed in the Fourier domain.

    alpha -- how much the cross-power spectrum is whitened
    ------------------------------------------------------
    The surface is  irfft2( R / |R|^alpha )  with  R = F1 * conj(F2).

        alpha = 1  classic PHASE correlation: keep only the phase. Gives a
                   near-delta peak and is exact for a rigid shift of a SHARP
                   pattern -- the right choice when both inputs are frames.
        alpha = 0  plain cross-correlation: no whitening at all.
        0<a<1      generalised cross-correlation, interpolating between them.

    Why this is a knob and not a constant. Whitening divides each frequency by
    its own magnitude, so a band the template barely occupies gets amplified to
    unit gain -- it is pure noise, boosted to the same weight as real signal. A
    SMOOTH template has almost no high-frequency content, so alpha=1 amplifies
    mostly noise and the peak collapses. Measured on a 2-digit 64x64 scene,
    accuracy of the recovered displacement:

        template            alpha=1     alpha=0
        sharp digit           100%         99%
        3x3 blur                0%         92%
        5x5 blur                0%         77%
        5x5 blur + tanh         0%         65%

    That last row is what tracking a ConvLSTM hidden state actually looks like
    (Seq2SeqMEConvLSTM.track_velocities correlates h.mean(dim=2), a smooth
    tanh-saturated map, against a frame), which is why the tracking correlator
    and the bootstrap correlator want different values. Raising eps does NOT
    substitute: at alpha=1 the collapse survives eps=1e-2, because the problem
    is the normalisation itself, not the guard against dividing by zero.

    alpha is shift-covariant for every value: |R| is invariant to a shift of
    either input, so scaling by any function of it leaves the peak's LOCATION
    untouched. Changing alpha therefore cannot break the model's motion
    equivariance -- only how reliably the peak is found.

    Default stays alpha=1.0, so existing runs are bit-identical.
    """

    def __init__(
        self,
        n_modes=2,
        periodic_bc=True,
        pad_factor=1,
        eps=1e-8,
        alpha=1.0,
    ):
        super().__init__()

        self.n_modes = n_modes
        self.periodic_bc = periodic_bc
        self.pad_factor = pad_factor
        self.eps = eps
        self.alpha = alpha

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

        # cross-power spectrum, whitened by alpha (see the class docstring)
        R = F1 * torch.conj(F2)
        if self.alpha == 1.0:
            # written out separately, not as the general branch with an
            # exponent of 1: pow() is not bit-identical to a plain divide, and
            # this path has to reproduce existing runs exactly.
            R = R / (R.abs() + self.eps)
        elif self.alpha != 0.0:
            R = R / (R.abs() + self.eps) ** self.alpha

        # phase correlation
        corr = torch.fft.irfft2(R, s=(H_pad, W_pad))

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
    
