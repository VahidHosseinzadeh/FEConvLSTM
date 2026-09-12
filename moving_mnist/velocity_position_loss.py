r"""
A loss on WHERE THE DIGIT LANDS, not on the velocity and not on the pixels.

The problem with the two losses already in the repo
---------------------------------------------------
`smooth_l1(u_pred, v_measured)` scores each step independently. A systematic
bias of b px/frame costs the same at step 1 as at step 10, even though by step
10 the digit is 10b px away from where it belongs. It is the wrong shape for a
rollout.

The pixel loss is worse, because it SATURATES. Measured on this data (48 px
frame, one 28 px digit), a correctly drawn digit displaced by d px:

    d =  0 px : 0.000       d = 10 px : 0.060
    d =  2 px : 0.027       d = 14 px : 0.065
    d =  3 px : 0.039  <--- already equal to predicting a BLANK frame (0.037)
    d =  5 px : 0.049       d = 20 px : 0.073

Past ~3 px the pixel loss stops being a statement about distance at all: it
plateaus at roughly twice the blank-frame cost (it pays once for the digit that
is missing and once for the one that is spurious). Gradients from it cannot
tell "10 px out" from "20 px out", which is exactly the range a rollout lives
in. Raising SHAPE_WEIGHT moves the break-even to ~18 px but does not change the
shape of the curve.

What this module computes instead
---------------------------------
Read it outermost-first, as a composition:

    L = (1/H) sum_t w_t * rho_delta( d(e_t) ),   e_t = sum_{s<=t} (v_pred_s - v*_s)

    d_T2(e) = || e - S * round(e / S) ||_2       toroidal distance  (reporting)
    d(e)    = || e ||_2                          plain distance     (training)

    rho_delta(r) = 0.5 r^2 / delta        r <= delta      (smooth at 0)
                 = r - 0.5 delta          r >  delta      (linear, never saturates)

In words, and this is the whole statement:

    at every rollout step, work out where the predicted digit is relative to
    where the target digit should be, measure that positional distance, and
    penalise that distance.

Integrating the velocity error is merely HOW the positional discrepancy is
obtained when the model predicts velocities rather than positions. It is not
what the loss is about, and reading it as "a velocity loss with a cumsum in it"
gets the emphasis backwards.

The identity that licenses the reading: the motion here is a pure translation and
the dataset renders it with an integer `torch.roll`, so position IS the
cumulative velocity -- exactly, verified as roll(X_t, motions[t]) == X_{t+1} to
0.0. That is a PRECONDITION, not a general construction: it stops holding the
moment the content deforms, and with several digits there is no single "the
position" to speak of (which is what the K velocity slots are for).

Properties, which are the ones asked for:

  * L = 0  <=>  the predicted digit coincides with the target at every step;
  * L > 0 and strictly increasing in the separation, with no plateau;
  * differentiable in every predicted velocity, and dL/dv_s collects a term
    from every step t >= s -- an early error is charged for the whole horizon,
    which is the actual physics of a rollout.

Which distance, and when -- d or d_T2
--------------------------------------
Positions wrap: two digits whose displacement differs by exactly the frame size
are the SAME picture. So d_T2 is the honest PIXEL distance, and it is what should
be REPORTED.

It is the wrong thing to TRAIN on, for two reasons that both bite at these
settings. It is capped at S/sqrt(2), so over a long horizon every method piles up
against the ~18 px "two random positions" ceiling and the metric stops
discriminating between them. Worse, beyond S/2 it is NON-MONOTONE: the gradient
points the wrong way, pushing the prediction to wrap further rather than come
back. With max_speed 4 over 10 steps the cumulative error can reach ~80 px on a
48 px frame, so that regime is reachable early in training, not hypothetical.

Hence image_size=None (plain d, the default) is the training form, and
image_size=S (d_T2) the reporting form. Both are exercised in the tests.

Targets without labels
----------------------
`displacement_targets` measures the per-step displacement straight off the
frames with phase correlation, which is exact for a single digit (verified 100%
of steps). So this loss needs no ground-truth motion: the images supply the
target and the target is detached, exactly the arrangement the velocity head
already uses ("the head learns from the measurement; the measurement must never
learn from the head").
"""

import torch
import torch.nn as nn


def toroidal_wrap(d, size):
    """Wrap a displacement into [-size/2, size/2) per component.

    round() is detached so this is the piecewise-linear identity it looks like
    (gradient 1 almost everywhere) rather than something autograd tries to
    differentiate through the rounding.
    """
    return d - size * torch.round(d / size).detach()


def huber_on_norm(r, delta):
    """rho_delta applied to a NON-NEGATIVE radius r.

    Deliberately applied to the norm, not per component: the quantity that has
    to be zero is a DISTANCE, and a per-component Huber would make the loss
    depend on the orientation of the error (an error of (3,0) would score
    differently from one of (2.1,2.1) at the same distance).

    The r <= delta branch is quadratic, so it is smooth at r = 0 where the
    gradient of ||.|| itself is undefined.
    """
    return torch.where(r <= delta, 0.5 * r ** 2 / delta, r - 0.5 * delta)


class PositionRolloutLoss(nn.Module):
    """Distance between the predicted digit and the target digit, summed over a
    rollout.

    Parameters
    ----------
    image_size : int or None
        Frame size, for the toroidal wrap. None = plain Euclidean distance.
    delta : float
        Huber knee, in pixels. Below it the loss is quadratic (a fine-grained
        signal once the prediction is nearly right), above it linear (robust,
        and still informative at 30 px where a pixel loss is flat). 2 px is
        about a tenth of a digit.
    weighting : {'uniform', 'linear_decay', 'final'}
        'uniform'      every rollout step counts equally. Because e_t is a
                       CUMULATIVE sum, this already charges early errors more
                       (they appear in every later term) without any weighting.
        'linear_decay' down-weights late steps ~1/t, which removes that
                       compounding emphasis. Use it if long-horizon terms
                       dominate and destabilise training.
        'final'        only the last step. Cheapest statement of "end up in the
                       right place", but it leaves the path unconstrained.
    """

    def __init__(self, image_size=None, delta=2.0, weighting="uniform"):
        super().__init__()
        if weighting not in ("uniform", "linear_decay", "final"):
            raise ValueError(f"unknown weighting {weighting!r}")
        self.image_size = image_size
        self.delta = delta
        self.weighting = weighting

    def displacement_error(self, v_pred, v_target):
        """(B,H,2) velocities -> (B,H) toroidal distance between the digits."""
        if v_pred.shape != v_target.shape:
            raise ValueError(f"shape mismatch {tuple(v_pred.shape)} vs {tuple(v_target.shape)}")
        e = torch.cumsum(v_pred - v_target, dim=1)
        if self.image_size is not None:
            e = toroidal_wrap(e, self.image_size)
        return e.norm(dim=-1)

    def forward(self, v_pred, v_target, return_parts=False):
        """
        v_pred   : (B, H, 2) predicted rollout velocities, requires grad
        v_target : (B, H, 2) measured or ground-truth velocities (detached here)
        """
        r = self.displacement_error(v_pred, v_target.detach())
        per_step = huber_on_norm(r, self.delta)

        H = r.shape[1]
        if self.weighting == "uniform":
            w = torch.ones(H, device=r.device, dtype=r.dtype)
        elif self.weighting == "linear_decay":
            w = 1.0 / torch.arange(1, H + 1, device=r.device, dtype=r.dtype)
        else:
            w = torch.zeros(H, device=r.device, dtype=r.dtype)
            w[-1] = 1.0
        w = w / w.sum()

        loss = (per_step * w).sum(dim=1).mean()
        if not return_parts:
            return loss
        return loss, {"px_error_per_step": r.detach().mean(dim=0),
                      "final_px_error": r.detach()[:, -1].mean(),
                      "mean_px_error": r.detach().mean()}


@torch.no_grad()
def displacement_targets(frames, phase_corr):
    """Per-step displacement read off the FRAMES -- no ground-truth motion.

    frames     : (B, T, C, H, W)
    phase_corr : a PhaseCorrelation(n_modes=1) instance
    returns    : (B, T-1, 2) displacement carrying frame t into frame t+1

    Exact for a single digit. With several digits the single correlation peak
    belongs to whichever digit dominates, so use the ground-truth motions (or
    per-slot templates) instead.
    """
    B, T, C, H, W = frames.shape
    a = frames[:, :-1].reshape(-1, C, H, W)
    b = frames[:, 1:].reshape(-1, C, H, W)
    v, _ = phase_corr(a, b)
    return v[:, 0].reshape(B, T - 1, 2)
