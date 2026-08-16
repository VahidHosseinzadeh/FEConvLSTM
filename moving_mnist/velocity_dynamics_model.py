"""
Equivariant velocity dynamics head.

The MEConvLSTM transports its hidden/cell state by a per-slot velocity u before
every recurrent update. Today u is *measured* from the data by phase
correlation: bootstrapped from (X_0, X_1), then re-tracked against every new
frame. That works while frames keep arriving, and stops working the moment
they don't -- during the decoder rollout there is no next frame, so the last
encoder velocity is frozen for the whole horizon.

This module adds the second half of a predict/correct filter. The correlation
head is the *measurement*; VelocityDynamicsHead is the *process model*: a small
recurrent network that predicts u_{t+1} from the velocity history, and can
therefore keep producing velocities after the measurements run out.

Why this lives in its own file: both Seq2SeqMEConvLSTM and its RNN twin need
it, and neither should own it.

TODO(MERNN): Seq2SeqMEConvRNN (velocity_model_based_MEConvRNN_model.py, on the
MERNN-and-Bootstrap branch -- it does not exist on main) is NOT wired up. Its
encoder velocity protocol is not the LSTM's: it bootstraps repeatedly until
B_until and resolves slots with assign_by_continuity, so "the step where a
measurement is produced" and "the previous velocity for this slot" mean
different things there. Porting the integration below without re-deriving that
correspondence would silently mis-pair predictions with targets, so it is
deliberately left undone rather than guessed at.

Equivariance (the constraint that dictates the architecture)
------------------------------------------------------------
The whole point of MEConvLSTM is motion equivariance: apply a global motion psi
with velocity v to the input sequence and the model's internals translate with
it. Concretely, every velocity estimate picks up the same offset,
u_s -> u_s + v for all s, and the hidden state translates spatially,
h -> psi |> h. Any module that consumes velocities and produces velocities has
to commute with that, *exactly*, or the guarantee is gone for the trained model
(not merely degraded -- Theorem 2 stops applying):

    g(u_{<=t} + v,  roll(h_t, s))  ==  g(u_{<=t}, h_t) + v      for all v, s

Two design rules follow, and they are not negotiable:

1. **The recurrence runs on velocity differences, never on velocities.**
   The GRU is fed du_t = u_t - u_{t-1}, which is invariant under u -> u + v
   (the offset cancels in the difference). Feeding raw u_t would make the GRU
   state a function of the global motion, and its output would then have to
   learn the identity map in v to stay equivariant -- i.e. approximately, on
   the training distribution only. This way it is exact by construction.

2. **h may only enter through a spatially translation-invariant readout.**
   Global pooling over (H, W) -- here concat(mean, max) per channel -- is
   invariant to a cyclic roll on the torus, which is exactly what the warp
   does. Raw h, spatial crops, coordinate channels, or anything else
   position-dependent would leak the absolute position and break rule 2.

The output is then an *invariant* increment added to an *equivariant* base:

    u_{t+1} = u_t + Delta_theta(gru_state, du_t, phi(h_t))

u_t carries the +v, Delta contributes 0 to it, so the sum carries exactly +v.

A consequence worth stating explicitly, because it is the opposite of how the
ConvLSTM's own state is handled: **the GRU state here is invariant, so it must
NOT be warped or transported between timesteps.** (h, c) live on the image grid
and have to be warped; this state lives in a velocity-difference space that the
global motion does not touch. Warping it would be a bug, not an improvement.

See test_velocity_dynamics.py::test_equivariance -- that test is the contract.
"""

import torch
import torch.nn as nn


class VelocityDynamicsHead(nn.Module):
    """
    Predicts u_{t+1} from the velocity history (and optionally an invariant
    readout of h). Exactly equivariant: g(u + v, roll(h, s)) == g(u, h) + v.

    Parameters
    ----------
    state_dim : int
        GRU hidden size, carried per slot.
    hidden_channels : int or None
        Channel count of h. Required only when use_h=True (it sizes the
        pooling projection); ignored otherwise.
    use_h : bool
        Condition the prediction on phi(h) as well as on the velocity history.
        Default OFF, deliberately: the pure velocity-history model is the one
        that has to work first. Conditioning on h adds a second failure mode
        (the head can start explaining velocity by appearance and stop
        extrapolating) and belongs in an ablation, not in the default path.
    h_embed_dim : int
        Width of the projection applied to the pooled h readout.
    v_max : float or None
        Optional symmetric clamp on |u_next| per component. Default None (no
        clamp). NOTE: clamping is NOT equivariant -- clamp(u + v) is not
        clamp(u) + v once the clamp binds -- so it trades the exactness
        guarantee for a stability guard. Leave it off unless a run actually
        diverges, and expect the equivariance test to be about the unclamped
        head.
    """

    def __init__(self,
                 state_dim=32,
                 hidden_channels=None,
                 use_h=False,
                 h_embed_dim=16,
                 v_max=None):
        super().__init__()

        self.state_dim = state_dim
        self.use_h = use_h
        self.v_max = v_max

        in_dim = 2  # du_t = (dvx, dvy)
        if use_h:
            if hidden_channels is None:
                raise ValueError("hidden_channels is required when use_h=True")
            # mean AND max per channel: mean alone throws away the peak
            # structure of a slot's occupancy map, max alone throws away its
            # mass. Both are exactly invariant under a cyclic roll.
            self.h_proj = nn.Linear(2 * hidden_channels, h_embed_dim)
            in_dim += h_embed_dim
        else:
            self.h_proj = None

        # One shared GRUCell over a flattened (B*K) batch: every slot uses the
        # same dynamics (they are the same kind of object -- a moving digit)
        # and carries its own state. A per-slot ParameterList would tie the
        # module to a fixed n_slots and give each slot its own, worse-sampled
        # copy of the identical function.
        self.gru = nn.GRUCell(in_dim, state_dim)

        self.out = nn.Linear(state_dim, 2)
        # Zero-initialized output => Delta == 0 at init => u_next == u_t
        # exactly, which IS the current frozen-velocity rollout. The new head
        # therefore strictly nests existing behaviour: an untrained head cannot
        # regress a checkpoint, and any change in the metrics is something the
        # head learned rather than something it perturbed.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ------------------------------------------------------------------

    def init_state(self, B, K, device, dtype):
        """Zero GRU state, (B, K, state_dim). Not warped, ever -- see module
        docstring: this state is invariant, unlike the ConvLSTM's (h, c)."""
        return torch.zeros(B, K, self.state_dim, device=device, dtype=dtype)

    def _phi(self, h_t):
        """
        (B, K, Ch, H, W) -> (B, K, h_embed_dim), invariant to any cyclic roll
        of the (H, W) axes.

        mean and max over the full spatial extent are both permutation
        invariant over pixels, hence invariant to the torus translation the
        warp applies. (mean can differ in the last ulp under reordered
        summation; max is bitwise identical. Both are far inside the 1e-5
        tolerance the equivariance test asserts.)
        """
        pooled = torch.cat([h_t.mean(dim=(-2, -1)),
                            h_t.amax(dim=(-2, -1))], dim=-1)   # (B, K, 2*Ch)
        return self.h_proj(pooled)

    def forward(self, u_t, du_t, h_t, state):
        """
        u_t   : (B, K, 2)  current velocity, [vx, vy] in pixel units
        du_t  : (B, K, 2)  u_t - u_{t-1}, zeros at the first step
        h_t   : (B, K, Ch, H, W) or None (unused unless use_h)
        state : (B, K, state_dim)

        Returns (u_next, new_state) with u_next : (B, K, 2).
        """
        B, K, _ = u_t.shape

        x = du_t.reshape(B * K, 2)
        if self.use_h:
            if h_t is None:
                raise ValueError("use_h=True but h_t is None")
            x = torch.cat([x, self._phi(h_t).reshape(B * K, -1)], dim=1)

        new_state = self.gru(x, state.reshape(B * K, self.state_dim))

        # Invariant increment on an equivariant base: this line is where the
        # equivariance is bought. u_t is never fed to the GRU, only added.
        u_next = u_t + self.out(new_state).view(B, K, 2)

        if self.v_max is not None:
            u_next = u_next.clamp(-self.v_max, self.v_max)

        return u_next, new_state.view(B, K, self.state_dim)
