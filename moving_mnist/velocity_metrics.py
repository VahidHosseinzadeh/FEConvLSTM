import torch


# class VelocityMetrics:
#     """
#     Tracks velocity prediction quality.

#     Assumes

#         pred_vel : (B, T, N, 2)
#         true_vel : (B, T, N, 2)

#     where
#         B = batch size
#         T = number of velocity estimates
#         N = number of digits
#     """

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.correct = None
#         self.total = None
#         self.l2_sum = None

#         self.examples = {}

#     def update(self, pred_vel, true_vel):

#         pred_vel = pred_vel.detach().clone().cpu()
#         true_vel = true_vel.detach().clone().cpu()

#         # In case lengths differ (e.g. length generalization)
#         T = min(pred_vel.size(1), true_vel.size(1))
#         pred_vel = pred_vel[:, :T]
#         true_vel = true_vel[:, :T]

#         B, T, N, _ = pred_vel.shape

#         if self.correct is None:
#             self.correct = torch.zeros(T, dtype=torch.long)
#             self.total = torch.zeros(T, dtype=torch.long)
#             self.l2_sum = torch.zeros(T)

#         pred_round = torch.round(pred_vel)

#         match = (pred_round == true_vel).all(dim=-1)  # (B,T,N)

#         l2 = torch.norm(pred_vel - true_vel.float(), dim=-1)  # (B,T,N)

#         for t in range(T):

#             self.correct[t] += match[:, t].sum()
#             self.total[t] += match[:, t].numel()
#             self.l2_sum[t] += l2[:, t].sum()

#             # Save first incorrect example
#             if t not in self.examples:

#                 idx = (~match[:, t]).nonzero(as_tuple=False)

#                 if len(idx) > 0:

#                     b = idx[0, 0].item()
#                     d = idx[0, 1].item()

#                     self.examples[t] = {
#                         "digit": d,
#                         "pred_float": tuple(
#                             round(x, 2) for x in pred_vel[b, t, d].tolist()
#                         ),
#                         "pred_round": tuple(
#                             pred_round[b, t, d].int().tolist()
#                         ),
#                         "true": tuple(
#                             true_vel[b, t, d].int().tolist()
#                         ),
#                     }

#     def report(self, title="Velocity Report"):

#         print()
#         print("=" * 90)
#         print(title)
#         print("=" * 90)

#         for t in range(len(self.correct)):

#             total = max(1, self.total[t].item())

#             acc = 100.0 * self.correct[t].item() / total
#             mean_l2 = self.l2_sum[t].item() / total

#             msg = (
#                 f"t={t+1:2d} | "
#                 f"acc={acc:6.2f}% "
#                 f"| mean_l2={mean_l2:.3f} "
#                 f"| ({self.correct[t].item()}/{total})"
#             )

#             if t in self.examples:

#                 e = self.examples[t]

#                 msg += (
#                     f"\n        first error:"
#                     f" digit={e['digit']}"
#                     f"  pred_float={e['pred_float']}"
#                     f"  pred_round={e['pred_round']}"
#                     f"  true={e['true']}"
#                 )

#             print(msg)

#         overall_acc = (
#             100.0 * self.correct.sum().item()
#             / max(1, self.total.sum().item())
#         )

#         overall_l2 = (
#             self.l2_sum.sum().item()
#             / max(1, self.total.sum().item())
#         )

#         print("-" * 90)
#         print(f"Overall accuracy : {overall_acc:.2f}%")
#         print(f"Overall mean L2  : {overall_l2:.3f}")
#         print("=" * 90)
#         print()




"""
VelocityMetrics — improved version
====================================

Key fix: best-assignment matching.
The K slots are ordered by correlation peak strength, not by digit index.
Comparing slot k directly against digit k is meaningless — this causes the
~50% baseline at t=1 (random assignment) and near-0% at t=2 (slot degeneracy).

Solution: for each batch item, find the permutation of K slots that maximises
matches across all time steps, then evaluate under that permutation. This asks
"does the model track the right velocities?" without penalising arbitrary slot
ordering.

Slot degeneracy at t=2
-----------------------
At encoder t=1 all K slots have identical hidden states (warp(0,v)=0 for
any v, so all slots receive the same cell input). _track_velocities at t=2
therefore uses the same template for all slots, returning the same velocity
for all of them. For N=2 with distinct velocities, both slots matching both
digits simultaneously is nearly impossible → near-zero accuracy at t=2 is
structural, not a parametrisation bug.

Timing convention
------------------
estimated_velocities[i]  corresponds to  motions[i]  (0-indexed):
  i=0         : bootstrap(X_0, X_1)           → motions[0]
  i=1..T_in-2 : track(h, X_{i+1})             → motions[i]
  i=T_in-1..  : track(h, target_seq[:,i-T_in+1]) → motions[i]
Printed as "t=i+1" in the report.
"""

from itertools import permutations as _permutations

import torch
import numpy as np


class VelocityMetrics:
    """
    Tracks velocity prediction quality with best-assignment slot matching.

    Expected shapes
    ---------------
    pred_vel : (B, T, K, 2)   estimated velocities, K slots
    true_vel : (B, T, N, 2)   ground-truth velocities, N digits
                               (pass motions[:, :T] from the dataset)

    K must be >= N. When K > N (more slots than digits, e.g. for extra
    tracking redundancy), the best-assignment search picks the best
    N-of-K slot subset (and its order) per batch item -- the "extra"
    K-N slots are simply never selected for that item. When K == N this
    reduces to exactly the original full-permutation behaviour.

    Two accuracies, and why both exist
    -----------------------------------
    The slots carry no inherent digit identity, so any score needs a
    slot->digit assignment. WHEN that assignment is chosen changes what is
    being measured:

    stepwise (per_t_acc_stepwise, the primary number)
        A separate assignment per (sequence, timestep). Answers "at time t,
        did the model produce the right SET of velocities?" — the estimator
        at time t, isolated. Nothing another timestep does can move it, so
        on a fixed evaluation set a parameter-free step (t=1, the phase
        correlation bootstrap) is CONSTANT across epochs. If this number
        moves, the estimator really changed.

    sequence-locked (per_t_acc, the previous behaviour)
        One assignment per sequence, chosen to maximise matches summed over
        all T timesteps, then applied to every step. Answers "given the
        model's overall slot->digit binding, was step t right?" A step that
        was estimated perfectly still scores 0 here if the other T-1 steps
        voted for the opposite binding — so this number moves with training
        even for steps that contain no learned parameters at all.

    per_t_binding_loss = stepwise - locked isolates that effect: it is the
    accuracy lost purely to the binding disagreeing across time. Large and
    growing means slots are swapping identity mid-sequence, which is a real
    finding about the tracker, just not one about the estimate at time t.

    mean_l2 is reported under each assignment (per_t_l2_stepwise, per_t_l2).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct    = None  # (T,) correct under the SEQUENCE-LOCKED assignment
        self.total      = None  # (T,) total predictions
        self.l2_sum     = None  # (T,) sum of L2 under the sequence-locked assignment
        self.correct_sw = None  # (T,) correct under the STEPWISE assignment
        self.l2_sum_sw  = None  # (T,) sum of L2 under the stepwise assignment
        self.examples   = {}    # first error per time step, for debugging
        self._T         = None

    # ------------------------------------------------------------------
    # Assignment matching
    # ------------------------------------------------------------------

    @staticmethod
    def _perm_matches(pred_vel, true_vel):
        """
        Match table for EVERY candidate slot->digit assignment at once.

        itertools.permutations(range(K), N) enumerates all injective maps
        from digit-position to slot-index (when K == N, the usual full
        permutations). Evaluating them all up front is what lets the two
        accuracies below (sequence-locked and stepwise) be derived from one
        shared computation instead of two searches.

        pred_vel : (B, T, K, 2) float
        true_vel : (B, T, N, 2) float
        returns  : match (P, B, T, N) bool -- match[p, b, t, n] is True when
                   the slot that permutation p assigns to digit n predicts
                   digit n's velocity exactly (after rounding),
                   perms (P, N) long
        """
        B, T, K, _ = pred_vel.shape
        N          = true_vel.shape[2]
        assert K >= N, f"n_slots ({K}) must be >= num_digits ({N})"

        perms  = torch.tensor(list(_permutations(range(K), N)), dtype=torch.long)
        pred_r = torch.round(pred_vel)

        match = torch.stack([
            (pred_r[:, :, p, :] == true_vel).all(dim=-1)      # (B, T, N)
            for p in perms.tolist()
        ], dim=0)
        return match, perms

    @staticmethod
    def _gather_perm(pred_vel, sel):
        """pred_vel (B,T,K,2) + sel (B,T,N) slot indices -> (B,T,N,2)."""
        B, T, N = sel.shape
        return torch.gather(pred_vel, 2, sel.unsqueeze(-1).expand(B, T, N, 2))

    @classmethod
    def _best_assignment(cls, pred_vel, true_vel):
        """
        Sequence-locked assignment: ONE permutation per batch item, the one
        maximising correct predictions summed over all T time steps.

        Kept as a standalone entry point (same signature and return values as
        before) for external callers; update() derives it from _perm_matches.

        returns : matched_pred (B, T, N, 2), best_perms (B, N) long
        """
        match, perms = cls._perm_matches(pred_vel, true_vel)
        B, T = pred_vel.shape[:2]
        N    = true_vel.shape[2]
        # argmax takes the FIRST maximum, matching the original loop's
        # strict `count > best_count` tie-break (identity permutation wins).
        best = match.sum(dim=(2, 3)).argmax(dim=0)            # (B,)
        sel  = perms[best]                                    # (B, N)
        matched = cls._gather_perm(pred_vel, sel.unsqueeze(1).expand(B, T, N))
        return matched, sel

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, pred_vel, true_vel):
        """
        pred_vel : (B, T, K, 2)
        true_vel : (B, T, N, 2)   e.g. motions[:, :T].unsqueeze(0).expand(B,...)
        """
        pred_vel = pred_vel.detach().float().cpu()
        true_vel = true_vel.detach().float().cpu()

        T = min(pred_vel.size(1), true_vel.size(1))
        pred_vel = pred_vel[:, :T]
        true_vel = true_vel[:, :T]
        B, T, K, _ = pred_vel.shape

        N = true_vel.shape[2]

        if self._T is None:
            self._T         = T
            self.correct    = torch.zeros(T, dtype=torch.long)
            self.total      = torch.zeros(T, dtype=torch.long)
            self.l2_sum     = torch.zeros(T)
            self.correct_sw = torch.zeros(T, dtype=torch.long)
            self.l2_sum_sw  = torch.zeros(T)

        match, perms = self._perm_matches(pred_vel, true_vel)     # (P,B,T,N)
        bx = torch.arange(B)

        # ---- sequence-locked: ONE permutation per sequence ------------
        # "Given the model's overall slot->digit binding, was step t right?"
        # A step can be scored wrong here purely because the binding chosen
        # by the OTHER T-1 steps disagrees with it.
        seq_best   = match.sum(dim=(2, 3)).argmax(dim=0)           # (B,)
        seq_sel    = perms[seq_best].unsqueeze(1).expand(B, T, N)  # (B,T,N)
        matched    = self._gather_perm(pred_vel, seq_sel)
        match_seq  = match[seq_best, bx]                           # (B,T,N)
        l2_seq     = torch.norm(matched - true_vel, dim=-1)

        # ---- stepwise: one permutation per (sequence, TIME STEP) ------
        # "At step t, taken on its own, was the right SET of velocities
        # produced?" This is the one that isolates the estimator at time t:
        # it cannot be dragged around by what other timesteps did, so a
        # parameter-free step (t=1 bootstrap) stays flat across epochs on a
        # fixed evaluation set.
        step_best  = match.sum(dim=3).argmax(dim=0)                # (B,T)
        step_sel   = perms[step_best]                              # (B,T,N)
        matched_sw = self._gather_perm(pred_vel, step_sel)
        match_sw   = match[step_best, bx[:, None], torch.arange(T)[None, :]]
        l2_sw      = torch.norm(matched_sw - true_vel, dim=-1)

        for t in range(T):
            self.correct[t]    += match_seq[:, t].sum().item()
            self.total[t]      += match_seq[:, t].numel()
            self.l2_sum[t]     += l2_seq[:, t].sum().item()
            self.correct_sw[t] += match_sw[:, t].sum().item()
            self.l2_sum_sw[t]  += l2_sw[:, t].sum().item()

            # First incorrect (b, n) entry for debugging. Keyed off the
            # STEPWISE match: that is the primary number in report(), so the
            # example shown has to be an example of *that* being wrong --
            # otherwise the printed line is a step the estimator got right
            # and only the sequence-wide binding disagreed with.
            if t not in self.examples:
                bad = (~match_sw[:, t]).nonzero(as_tuple=False)
                if len(bad) > 0:
                    b, n = bad[0, 0].item(), bad[0, 1].item()
                    self.examples[t] = {
                        "batch"       : b,
                        "digit"       : n,
                        "step_perm"   : step_sel[b, t].tolist(),
                        "seq_perm"    : seq_sel[b, t].tolist(),
                        "pred_float"  : tuple(round(x, 2) for x in
                                              matched_sw[b, t, n].tolist()),
                        "pred_round"  : tuple(int(x) for x in
                                              torch.round(matched_sw[b, t, n]).tolist()),
                        "true"        : tuple(int(x) for x in
                                              true_vel[b, t, n].tolist()),
                    }

    # ------------------------------------------------------------------
    # Summary / Report
    # ------------------------------------------------------------------

    def summary(self):
        """
        Structured per-timestep + overall metrics, computed once so both
        report() (console) and any external logger (e.g. wandb, via
        visualization.log_velocity_report) share the same numbers instead
        of recomputing them. Returns None if no data has been recorded yet.
        """
        if self._T is None:
            return None

        per_t_total = self.total.clamp(min=1)
        per_t_acc   = (100.0 * self.correct.float() / per_t_total).tolist()
        per_t_l2    = (self.l2_sum / per_t_total).tolist()

        overall_total = max(1, self.total.sum().item())
        overall_acc   = 100.0 * self.correct.sum().item() / overall_total
        overall_l2    = self.l2_sum.sum().item() / overall_total

        half      = self._T // 2
        enc_total = max(1, self.total[:half].sum().item())
        dec_total = max(1, self.total[half:].sum().item())
        encoder_acc = 100.0 * self.correct[:half].sum().item() / enc_total
        decoder_acc = 100.0 * self.correct[half:].sum().item() / dec_total

        # Stepwise (per-timestep assignment) counterparts. Same denominators.
        per_t_acc_sw = (100.0 * self.correct_sw.float() / per_t_total).tolist()
        per_t_l2_sw  = (self.l2_sum_sw / per_t_total).tolist()
        overall_acc_sw = 100.0 * self.correct_sw.sum().item() / overall_total
        overall_l2_sw  = self.l2_sum_sw.sum().item() / overall_total
        encoder_acc_sw = 100.0 * self.correct_sw[:half].sum().item() / enc_total
        decoder_acc_sw = 100.0 * self.correct_sw[half:].sum().item() / dec_total

        # How much of the sequence-locked score is lost purely to the
        # slot->digit binding disagreeing across time, rather than to a bad
        # velocity estimate. Non-negative by construction: the stepwise
        # assignment is free to pick the locked one at every step.
        per_t_binding_loss = [a - b for a, b in zip(per_t_acc_sw, per_t_acc)]

        return {
            "T"             : self._T,
            "per_t_acc"     : per_t_acc,
            "per_t_l2"      : per_t_l2,
            "per_t_correct" : self.correct.tolist(),
            "per_t_total"   : self.total.tolist(),
            "overall_acc"   : overall_acc,
            "overall_l2"    : overall_l2,
            "encoder_acc"   : encoder_acc,
            "decoder_acc"   : decoder_acc,
            "last_step_acc" : per_t_acc[-1],
            "last_step_l2"  : per_t_l2[-1],
            # ---- stepwise assignment (isolates the estimator at each t) --
            "per_t_acc_stepwise"     : per_t_acc_sw,
            "per_t_l2_stepwise"      : per_t_l2_sw,
            "per_t_correct_stepwise" : self.correct_sw.tolist(),
            "overall_acc_stepwise"   : overall_acc_sw,
            "overall_l2_stepwise"    : overall_l2_sw,
            "encoder_acc_stepwise"   : encoder_acc_sw,
            "decoder_acc_stepwise"   : decoder_acc_sw,
            "last_step_acc_stepwise" : per_t_acc_sw[-1],
            "last_step_l2_stepwise"  : per_t_l2_sw[-1],
            "per_t_binding_loss"     : per_t_binding_loss,
        }

    def report(self, title="Velocity Report"):
        s = self.summary()
        if s is None:
            print("No data recorded yet.")
            return

        print()
        print("=" * 100)
        print(title)
        print(
            "  acc%      : slot→digit assignment chosen AT EACH STEP — how good the\n"
            "              velocity estimate at time t is, on its own. Read this one.\n"
            "  acc%(seq) : one assignment per sequence, chosen across all timesteps.\n"
            "              Lower whenever a step's own binding disagrees with the rest\n"
            "              of the sequence; it mixes estimate quality with binding\n"
            "              consistency, so it can move even for a parameter-free step.\n"
            "  bind      : acc% - acc%(seq), i.e. how much is lost purely to the\n"
            "              binding disagreeing across time (never negative).\n"
            "  t=2 low is structural: both slots have identical hidden states after\n"
            "  t=1 (warp(0,v)=0), so they report one velocity and can match at most\n"
            "  one of the N digits."
        )
        print("=" * 100)
        print(f"{'t':>4}  {'acc%':>8}  {'acc%(seq)':>10}  {'bind':>6}  "
              f"{'mean_L2':>8}  {'L2(seq)':>8}  {'correct/total':>15}  note")
        print("-" * 100)

        for t in range(s["T"]):
            note = ""
            if t == 1:   # printed as t=2, first track call
                note = "← slot degeneracy (both slots identical at this step)"

            row = (f"{t+1:>4}  {s['per_t_acc_stepwise'][t]:>8.2f}  "
                   f"{s['per_t_acc'][t]:>10.2f}  "
                   f"{s['per_t_binding_loss'][t]:>6.2f}  "
                   f"{s['per_t_l2_stepwise'][t]:>8.3f}  {s['per_t_l2'][t]:>8.3f}"
                   f"  {s['per_t_correct_stepwise'][t]:>7}/{s['per_t_total'][t]:<7}  {note}")
            print(row)

            if t in self.examples:
                e = self.examples[t]
                print(f"       first error:  digit={e['digit']}"
                      f"  step_perm={e['step_perm']}"
                      f"  seq_perm={e['seq_perm']}"
                      f"  pred_float={e['pred_float']}"
                      f"  pred_round={e['pred_round']}"
                      f"  true={e['true']}")

        print("-" * 100)
        print(f"Overall accuracy (per-step assignment) : {s['overall_acc_stepwise']:.2f}%")
        print(f"Overall accuracy (sequence-locked)     : {s['overall_acc']:.2f}%")
        print(f"Overall mean L2  (per-step assignment) : {s['overall_l2_stepwise']:.3f}")
        print(f"Overall mean L2  (sequence-locked)     : {s['overall_l2']:.3f}")
        print("=" * 100)
        print()

        # ---- breakdown: encoder vs decoder -------------------------
        # The caller should know T_in to split; we report halves as proxy.
        half = s["T"] // 2
        print(f"  Encoder steps (t=1..{half})   accuracy: "
              f"{s['encoder_acc_stepwise']:.2f}%  (seq-locked {s['encoder_acc']:.2f}%)")
        print(f"  Decoder steps (t={half+1}..{s['T']}) accuracy: "
              f"{s['decoder_acc_stepwise']:.2f}%  (seq-locked {s['decoder_acc']:.2f}%)")
        print()


# ============================================================================
# Helper: build true_vel tensor from dataset motions for a batch
# ============================================================================

def motions_to_true_vel(motions, T_estimated):
    """
    Convert the motions tensor returned by the dataset to the true_vel
    format expected by VelocityMetrics.

    Dataset returns motions : (B, T_seq, N, 2)
    Model produces estimated_velocities : (B, T_estimated, K, 2)
    where T_estimated = T_in - 1 + pred_len

    The mapping is: estimated_velocities[:, i] ↔ motions[:, i]
    (estimated_velocities[0] = bootstrap(X_0,X_1) = motions[0], etc.)

    Parameters
    ----------
    motions       : (B, T_seq, N, 2)  from dataset (return_motion=True)
    T_estimated   : int               number of velocity estimates from model
                    = T_in - 1 + pred_len

    Returns
    -------
    true_vel : (B, T_estimated, N, 2)  ready to pass to VelocityMetrics.update
    """
    T = min(T_estimated, motions.shape[1])
    return motions[:, :T].float()