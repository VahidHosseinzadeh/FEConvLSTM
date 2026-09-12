r"""
Velocity forecasters: context velocities in, a rollout of future velocities out.

Trained with PositionRolloutLoss (velocity_position_loss.py), i.e. on where the
digit ends up rather than on per-step velocity error.

Three architectures, and what they measured
-------------------------------------------
Final-step digit displacement, trained at H=10 on the position loss, evaluated
open loop; H=20 and H=30 are pure extrapolation:

    model                       H=10    H=20    H=30
    HarmonicForecaster           5.1    14.9    22.0   px
    GRUForecaster                5.3    12.6    21.9
    LTIForecaster                7.7    46.9    84.8
    frozen (baseline)           36.1    65.3    88.5

`GRUForecaster` ("predict the next increment") is the straightforward thing and
is what the repo's VelocityDynamicsHead already does. It rolls out
autoregressively, so a long horizon is a long chain of its own approximations.

`LTIForecaster` ("identify, then propagate") was the obvious improvement on it,
and IT DOES NOT WORK. It is kept as a measured negative result -- and as the
reason not to reach for the same idea again. The reasoning was that
v_t = c + A cos(w t + phi), which is what motion_mode="harmonic" generates,
satisfies for every t

    v_{t+1} - c = 2 cos(w) (v_t - c) - (v_{t-1} - c)

so the network need only recognise the oscillation and let a fixed recurrence
carry it forward. The flaw is in carrying it forward: the recurrence is SEEDED
with the last two velocities, which the dataset has rounded to integers, and it
then iterates. Fitting each family's parameters per sequence against the TRUE
future -- an oracle fit, identification removed entirely, so this is the ceiling
of the family itself -- gives

    family                            H=10    H=20    H=30
    closed-form harmonic              0.17    0.40    0.96   px
    2nd-order recurrence (rho,th,c)   3.10    9.64   13.27

The recurrence cannot express this data even when fitted to the answer: the seed
error and any small error in the pole angle dephase the oscillation, and since
position is the running sum of velocity, a dephased sinusoid integrates into a
large displacement. This is also the likely reason MEConvLSTM's own
--vel_dyn_arch recurrence never paid off. `HarmonicForecaster` evaluates the
closed form instead and never iterates -- see its docstring.

The general second-order form is parametrised through a pole radius and angle,

    a1 = 2 rho cos(theta),   a2 = -rho^2,      rho in [0, 1]

which keeps both roots inside the closed unit disc, so the rollout can decay or
sustain but never diverge -- unlike a freely-parametrised (a1, a2). rho = 1 is
a pure sustained sinusoid; rho < 1 lets it damp.

Read the first table together with the second: every learned model beats
freezing by ~7x at the trained horizon, the closed form and the GRU are within
noise of each other in practice, and both sit far above the 0.96 px ceiling
their family allows. The bottleneck is IDENTIFICATION from a 15-step quantised
context -- not propagation, and not capacity.

Equivariance -- read this before wiring any of these into MEConvLSTM
--------------------------------------------------------------------
The repo's VelocityDynamicsHead consumes ONLY velocity differences du, which
makes it exactly motion-equivariant: g(u + v) = g(u) + v. All three models here
consume the velocity itself (LTIForecaster emits an absolute offset c, and
HarmonicForecaster an offset d), so NONE is motion-equivariant by default, and
Theorem 2 stops applying to a model built on them.

That is a deliberate trade, not an oversight. Equivariance is what makes an
error in the measured velocity uncorrectable: the head can only add increments
to the velocity it is handed, so a bias b in the context survives the whole
rollout and contributes H*|b| px of drift (measured: predicted 16.76 px,
observed 16.76 px). A model that sees absolute velocity can average its context
and reject that bias -- but it must then learn the identity in v rather than
getting it for free.

`equivariant=True` restricts either model to du-only inputs and increments-only
outputs, recovering the exact guarantee, so the two can be compared on the same
loss and the same data.
"""

import torch
import torch.nn as nn


class _ContextEncoder(nn.Module):
    """GRU over the context velocity sequence -> a summary vector.

    Input per step is [v_t, dv_t] (or just dv_t when equivariant), so the
    increment structure the dynamics actually lives in is presented directly
    rather than left to be differenced internally.
    """

    def __init__(self, state_dim=64, equivariant=False, n_layers=1):
        super().__init__()
        self.equivariant = equivariant
        in_dim = 2 if equivariant else 4
        self.gru = nn.GRU(in_dim, state_dim, num_layers=n_layers, batch_first=True)
        self.state_dim = state_dim

    def forward(self, v_ctx):
        """v_ctx : (B, T, 2) -> (B, state_dim)"""
        dv = torch.zeros_like(v_ctx)
        dv[:, 1:] = v_ctx[:, 1:] - v_ctx[:, :-1]
        x = dv if self.equivariant else torch.cat([v_ctx, dv], dim=-1)
        out, _ = self.gru(x)
        return out[:, -1]


class LTIForecaster(nn.Module):
    """Identify a second-order linear recurrence from the context, then iterate it.

    Parameters
    ----------
    state_dim : int
        Width of the context encoder.
    v_max : float or None
        Clamp on the emitted velocity. The data is bounded by construction, so
        this is free information; it is also the only hard guard on the offset c.
    equivariant : bool
        Drop the absolute-velocity inputs and the offset, restoring exact motion
        equivariance (see the module docstring). The recurrence then runs on du.
    rho_max : float
        Upper bound on the pole radius. 1.0 allows a sustained oscillation, which
        is what the harmonic data actually is.
    """

    def __init__(self, state_dim=64, v_max=None, equivariant=False,
                 rho_max=1.0, n_layers=1):
        super().__init__()
        self.encoder = _ContextEncoder(state_dim, equivariant, n_layers)
        self.equivariant = equivariant
        self.v_max = v_max
        self.rho_max = rho_max
        # per axis: (rho_raw, theta_raw, c_raw); c is dropped when equivariant
        self.n_out = 2 if equivariant else 3
        self.head = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 2 * self.n_out),
        )
        # Start near rho = 1, theta small: a slowly-turning sustained oscillation
        # is a far better prior than rho = 0 (which is the frozen rollout and has
        # zero gradient through cos(theta)).
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            b = torch.zeros(2 * self.n_out)
            b[0::self.n_out] = 3.0        # sigmoid(3) ~ 0.95
            self.head[-1].bias.copy_(b)

    def coefficients(self, v_ctx):
        """-> rho (B,2), theta (B,2), c (B,2)   [c is 0 when equivariant]"""
        p = self.head(self.encoder(v_ctx)).view(-1, 2, self.n_out)
        rho = torch.sigmoid(p[..., 0]) * self.rho_max
        theta = torch.pi * torch.tanh(p[..., 1])
        if self.equivariant:
            c = torch.zeros_like(rho)
        else:
            # The offset is a velocity, so bound it like one.
            c = p[..., 2]
            if self.v_max is not None:
                c = self.v_max * torch.tanh(c / max(self.v_max, 1e-6))
        return rho, theta, c

    def forward(self, v_ctx, horizon):
        """
        v_ctx   : (B, T, 2) context velocities (T >= 2)
        horizon : int, how many steps to emit
        returns : (B, horizon, 2)

        The recurrence is SEEDED WITH THE LAST TWO CONTEXT VELOCITIES rather
        than with anything the network produces, so the rollout starts exactly
        where the data is and the network is responsible only for how it turns.
        """
        rho, theta, c = self.coefficients(v_ctx)
        a1 = 2.0 * rho * torch.cos(theta)
        a2 = -rho ** 2

        if self.equivariant:
            # run the recurrence on increments, then re-integrate
            prev2 = v_ctx[:, -2] - v_ctx[:, -3] if v_ctx.shape[1] >= 3 else torch.zeros_like(v_ctx[:, -1])
            prev1 = v_ctx[:, -1] - v_ctx[:, -2]
            v = v_ctx[:, -1]
            out = []
            for _ in range(horizon):
                d = a1 * prev1 + a2 * prev2
                v = v + d
                if self.v_max is not None:
                    v = v.clamp(-self.v_max, self.v_max)
                out.append(v)
                prev2, prev1 = prev1, d
            return torch.stack(out, dim=1)

        prev2 = v_ctx[:, -2] - c
        prev1 = v_ctx[:, -1] - c
        out = []
        for _ in range(horizon):
            nxt = a1 * prev1 + a2 * prev2
            v = nxt + c
            if self.v_max is not None:
                v = v.clamp(-self.v_max, self.v_max)
            out.append(v)
            prev2, prev1 = prev1, nxt
        return torch.stack(out, dim=1)


class GRUForecaster(nn.Module):
    """Autoregressive baseline: emit the next increment, feed it back.

    This is the same shape of model as the repo's VelocityDynamicsHead (and with
    equivariant=True, the same information); it is here so that "identify then
    propagate" is compared against "learn to roll out" under an identical loss
    and identical data, rather than against the repo's different objective.
    """

    def __init__(self, state_dim=64, v_max=None, equivariant=False, n_layers=1):
        super().__init__()
        self.equivariant = equivariant
        self.v_max = v_max
        in_dim = 2 if equivariant else 4
        self.cell = nn.GRUCell(in_dim, state_dim)
        self.readout = nn.Linear(state_dim, 2)
        self.state_dim = state_dim
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)      # delta = 0 at init == frozen rollout

    def _step_input(self, v, dv):
        return dv if self.equivariant else torch.cat([v, dv], dim=-1)

    def forward(self, v_ctx, horizon):
        B = v_ctx.shape[0]
        s = v_ctx.new_zeros(B, self.state_dim)
        dv = torch.zeros_like(v_ctx)
        dv[:, 1:] = v_ctx[:, 1:] - v_ctx[:, :-1]
        for t in range(v_ctx.shape[1]):                      # absorb the context
            s = self.cell(self._step_input(v_ctx[:, t], dv[:, t]), s)
        v, d = v_ctx[:, -1], dv[:, -1]
        out = []
        for _ in range(horizon):                             # roll out open loop
            s = self.cell(self._step_input(v, d), s)
            d = self.readout(s)
            v = v + d
            if self.v_max is not None:
                v = v.clamp(-self.v_max, self.v_max)
            out.append(v)
        return torch.stack(out, dim=1)


class HarmonicForecaster(nn.Module):
    r"""Predict the parameters of the velocity signal and evaluate it in CLOSED FORM.

    Per axis the network emits (omega, a_c, a_s, d) and the rollout is

        v_t = d + a_c cos(omega t) + a_s sin(omega t),      t = 1 .. H

    with t measured from the end of the context. No iteration, so nothing
    compounds and H = 30 is exactly as accurate as H = 10.

    Why this and not the recurrence
    -------------------------------
    Measured, by fitting each family's parameters per sequence with gradient
    descent on the position loss against the TRUE future -- an oracle fit, so
    this is the ceiling of the family itself, with identification removed:

        family                            H=10    H=20    H=30
        closed-form harmonic              0.17    0.40    0.96   px
        2nd-order recurrence (rho,th,c)   3.10    9.64   13.27   px
        (frozen, for reference)           35.9    65.6    88.9   px

    The recurrence cannot express this data even when fitted to the answer. It
    is seeded with the last two velocities, which the dataset has ROUNDED to
    integers, and it then iterates: the seed error and any small error in the
    pole angle dephase the oscillation, and since position is the running sum of
    velocity, a dephased sinusoid integrates into a large displacement. The
    closed form never iterates -- every step is evaluated from the parameters --
    so it inherits no error from its own past. Its <1 px residual is the
    dataset's integer rounding, which no velocity model can undo.

    Parametrisation
    ---------------
    (a_c, a_s) rather than (amplitude, phase): the signal is LINEAR in this
    pair, so there is no phase wrap-around for the optimiser to fall into and no
    atan2 in the graph. Only omega stays nonlinear, and it is confined to a band
    around the periods the data actually contains -- a frequency far outside it
    is not a hypothesis worth spending capacity on, and an unconstrained omega
    makes the loss surface violently multimodal.

    Note omega > 0 loses nothing: cos(-wt + phi) = cos(wt - phi), so a negative
    frequency is already covered by the (a_c, a_s) pair.

    Equivariance
    ------------
    With equivariant=False the head emits an absolute offset d, so it is NOT
    motion-equivariant. equivariant=True drops d and anchors the oscillation to
    the last context velocity instead (v_t = v_ctx[-1] + oscillation predicted
    from du), which restores exactness -- at the cost of pinning the level to a
    single measured velocity, which is precisely the sample that carries the
    measurement bias.
    """

    def __init__(self, state_dim=64, v_max=None, equivariant=False,
                 period_range=(8.0, 40.0), n_layers=1):
        super().__init__()
        self.encoder = _ContextEncoder(state_dim, equivariant, n_layers)
        self.equivariant = equivariant
        self.v_max = v_max
        self.w_min = 2.0 * torch.pi / period_range[1]
        self.w_max = 2.0 * torch.pi / period_range[0]
        self.n_out = 3 if equivariant else 4        # (w, a_c, a_s[, d])
        self.head = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 2 * self.n_out),
        )

    def parameters_for(self, v_ctx):
        """-> omega (B,2), a_c (B,2), a_s (B,2), d (B,2)"""
        p = self.head(self.encoder(v_ctx)).view(-1, 2, self.n_out)
        omega = self.w_min + (self.w_max - self.w_min) * torch.sigmoid(p[..., 0])
        a_c, a_s = p[..., 1], p[..., 2]
        if self.equivariant:
            d = v_ctx[:, -1]                     # anchored, not predicted
        else:
            d = p[..., 3]
            if self.v_max is not None:
                d = self.v_max * torch.tanh(d / max(self.v_max, 1e-6))
        return omega, a_c, a_s, d

    def forward(self, v_ctx, horizon):
        omega, a_c, a_s, d = self.parameters_for(v_ctx)
        t = torch.arange(1, horizon + 1, device=v_ctx.device, dtype=v_ctx.dtype)
        wt = omega.unsqueeze(1) * t.view(1, -1, 1)          # (B,H,2)
        v = (d.unsqueeze(1)
             + a_c.unsqueeze(1) * torch.cos(wt)
             + a_s.unsqueeze(1) * torch.sin(wt))
        if self.v_max is not None:
            v = v.clamp(-self.v_max, self.v_max)
        return v


# ---------------------------------------------------------------- baselines
def frozen_rollout(v_ctx, horizon):
    """Hold the last context velocity -- what the decoder does today."""
    return v_ctx[:, -1:].expand(-1, horizon, -1)


def momentum_rollout(v_ctx, horizon):
    """Linear extrapolation from the last increment."""
    last = v_ctx[:, -1:]
    step = v_ctx[:, -1:] - v_ctx[:, -2:-1]
    k = torch.arange(1, horizon + 1, device=v_ctx.device, dtype=v_ctx.dtype)
    return last + step * k.view(1, -1, 1)
