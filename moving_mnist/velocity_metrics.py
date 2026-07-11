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
                 (only exactly aligned when teacher_forcing_ratio=1)
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

    Metrics reported per time step
    --------------------------------
    acc_assigned  : % of (batch, digit) pairs correctly predicted,
                    under the best global permutation per batch item.
    acc_any       : % of time steps where at least one permutation
                    achieves a perfect (B-wide) match — upper bound.
    mean_l2       : mean L2 error under best-assignment permutation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct   = None   # (T,) correct predictions under best assignment
        self.total     = None   # (T,) total predictions
        self.l2_sum    = None   # (T,) sum of L2 errors under best assignment
        self.examples  = {}     # first error per time step, for debugging
        self._T        = None

    # ------------------------------------------------------------------
    # Assignment matching
    # ------------------------------------------------------------------

    @staticmethod
    def _best_assignment(pred_vel, true_vel):
        """
        For each batch item find the length-N ordered selection of K slots
        (which N of the K slots, and in what order they line up against the
        N digits) that maximises the number of correct velocity predictions
        across all T time steps. itertools.permutations(range(K), N) gives
        exactly this: all injective maps from digit-position to slot-index.
        When K == N it's the usual set of full permutations.

        pred_vel : (B, T, K, 2)  float
        true_vel : (B, T, N, 2)  long / float
        returns  : matched_pred (B, T, N, 2), best_perms (B, N) long
                   -- best_perms[b, n] is the slot index matched to digit n
        """
        B, T, K, _ = pred_vel.shape
        N           = true_vel.shape[2]
        assert K >= N, f"n_slots ({K}) must be >= num_digits ({N})"

        all_perms   = list(_permutations(range(K), N))
        matched     = torch.zeros(B, T, N, 2, dtype=pred_vel.dtype)
        best_perms  = torch.zeros(B, N, dtype=torch.long)

        pred_r = torch.round(pred_vel)   # (B, T, K, 2)
        true_f = true_vel.float()        # (B, T, N, 2)

        for b in range(B):
            best_count = -1
            best_p     = list(range(N))

            for perm in all_perms:
                # select+reorder N of the K slots: (T, N, 2)
                p_pred  = pred_r[b, :, perm, :]
                # count exact (vx, vy) matches across all (t, n)
                count   = (p_pred == true_vel[b].float()).all(dim=-1).sum().item()
                if count > best_count:
                    best_count = count
                    best_p     = list(perm)

            best_perms[b] = torch.tensor(best_p, dtype=torch.long)
            matched[b]    = pred_vel[b, :, best_p, :]

        return matched, best_perms

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

        if self._T is None:
            self._T     = T
            self.correct = torch.zeros(T, dtype=torch.long)
            self.total   = torch.zeros(T, dtype=torch.long)
            self.l2_sum  = torch.zeros(T)

        # ---- best-assignment matching --------------------------------
        matched, best_perms = self._best_assignment(pred_vel, true_vel)
        # matched: (B, T, N, 2) — slots reordered to best match digits

        pred_r = torch.round(matched)                            # (B, T, N, 2)
        match  = (pred_r == true_vel).all(dim=-1)               # (B, T, N) bool
        l2     = torch.norm(matched - true_vel, dim=-1)         # (B, T, N)

        for t in range(T):
            self.correct[t] += match[:, t].sum().item()
            self.total[t]   += match[:, t].numel()
            self.l2_sum[t]  += l2[:, t].sum().item()

            # Store first incorrect (B,N) entry for debugging
            if t not in self.examples:
                bad = (~match[:, t]).nonzero(as_tuple=False)
                if len(bad) > 0:
                    b, n = bad[0, 0].item(), bad[0, 1].item()
                    perm = best_perms[b].tolist()
                    self.examples[t] = {
                        "batch"       : b,
                        "digit"       : n,
                        "best_perm"   : perm,
                        "pred_float"  : tuple(round(x, 2) for x in
                                              matched[b, t, n].tolist()),
                        "pred_round"  : tuple(int(x) for x in
                                              pred_r[b, t, n].tolist()),
                        "true"        : tuple(int(x) for x in
                                              true_vel[b, t, n].tolist()),
                    }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, title="Velocity Report"):
        if self._T is None:
            print("No data recorded yet.")
            return

        print()
        print("=" * 100)
        print(title)
        print(
            "  Accuracy is computed under the best per-sequence slot→digit "
            "assignment (best N-of-K slot selection, K>=N).\n"
            "  t=2 near-zero is structural: both slots have identical hidden "
            "states after t=1 (warp(0,v)=0)."
        )
        print("=" * 100)
        print(f"{'t':>4}  {'acc%':>8}  {'mean_L2':>8}  {'correct/total':>15}  note")
        print("-" * 100)

        for t in range(self._T):
            total   = max(1, self.total[t].item())
            acc     = 100.0 * self.correct[t].item() / total
            mean_l2 = self.l2_sum[t].item() / total

            note = ""
            if t == 1:   # printed as t=2, first track call
                note = "← slot degeneracy (both slots identical at this step)"

            row = (f"{t+1:>4}  {acc:>8.2f}  {mean_l2:>8.3f}"
                   f"  {self.correct[t].item():>7}/{total:<7}  {note}")
            print(row)

            if t in self.examples:
                e = self.examples[t]
                print(f"       first error:  digit={e['digit']}"
                      f"  best_perm={e['best_perm']}"
                      f"  pred_float={e['pred_float']}"
                      f"  pred_round={e['pred_round']}"
                      f"  true={e['true']}")

        overall_acc = (100.0 * self.correct.sum().item()
                       / max(1, self.total.sum().item()))
        overall_l2  = (self.l2_sum.sum().item()
                       / max(1, self.total.sum().item()))

        print("-" * 100)
        print(f"Overall accuracy (best assignment) : {overall_acc:.2f}%")
        print(f"Overall mean L2  (best assignment) : {overall_l2:.3f}")
        print("=" * 100)
        print()

        # ---- breakdown: encoder vs decoder -------------------------
        # The caller should know T_in to split; we report halves as proxy.
        half = self._T // 2
        enc_acc = (100.0 * self.correct[:half].sum().item()
                   / max(1, self.total[:half].sum().item()))
        dec_acc = (100.0 * self.correct[half:].sum().item()
                   / max(1, self.total[half:].sum().item()))
        print(f"  Encoder steps (t=1..{half})   accuracy: {enc_acc:.2f}%")
        print(f"  Decoder steps (t={half+1}..{self._T}) accuracy: {dec_acc:.2f}%")
        print(
            "  Note: decoder accuracy is only meaningful when "
            "teacher_forcing_ratio=1\n"
            "        (with ratio<1, h tracks predictions not GT frames,\n"
            "        so estimated velocities don't align with motions)."
        )
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


# ============================================================================
# Usage example (pseudocode in comments)
# ============================================================================

"""
In train_eval_utils.py:

from velocity_metrics import VelocityMetrics, motions_to_true_vel

def eval_epoch_with_velocity(model, dataloader, criterion, device,
                              input_frames, epoch, split_name):
    model.eval()
    vm = VelocityMetrics()
    running_loss = 0.0

    with torch.no_grad():
        for seq, _, motions in dataloader:      # dataset with return_motion=True
            seq     = seq.to(device)
            motions = motions.to(device)        # (B, T_seq, N, 2)

            inp = seq[:, :input_frames]
            tgt = seq[:, input_frames:]

            # Always pass target_seq so decoder velocities are aligned with GT
            out, est_vel = model(
                inp,
                pred_len=tgt.size(1),
                teacher_forcing_ratio=1.0,     # required for decoder vel accuracy
                target_seq=tgt,
                return_velocity=True
            )
            # est_vel : (B, T_in-1+pred_len, K, 2)

            true_vel = motions_to_true_vel(motions, est_vel.size(1))
            vm.update(est_vel, true_vel.to(est_vel.device))

            loss = criterion(out, tgt)
            running_loss += loss.item() * seq.size(0)

    vm.report(title=f"Velocity Report — {split_name} epoch {epoch}")
    return running_loss / len(dataloader.dataset)
"""