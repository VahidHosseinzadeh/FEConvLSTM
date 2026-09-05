# The velocity model in MEConvLSTM — mechanism

Everything below is read off the implementation:
`velocity_model_based_MEConvLSTM_model.py`, `velocity_predictor_model.py`,
`velocity_dynamics_model.py`, `train_eval_utils.py`. Where a number appears it
was measured, not estimated.

The one-paragraph picture:

> The model keeps one hidden state **per velocity slot**. Before every
> recurrent update it **transports** that state by the slot's current velocity,
> so the ConvLSTM only ever has to model *appearance change*, never motion.
> The velocity itself is produced by a two-part filter: phase correlation
> **measures** it from the data, and a small GRU **predicts** it from the
> velocity history. During the context both run; during the rollout there is no
> data left to measure, so only the GRU remains. The GRU is built so that it
> cannot break the model's motion equivariance — it consumes velocity
> *differences*, never velocities.

---

## 1. Notation

| symbol | meaning |
|---|---|
| $X_t\in\mathbb{R}^{C\times H\times W}$ | input frame, $t=0\dots T_{\text{in}}-1$ |
| $Y_t$, $\hat X_t$ | ground-truth / predicted future frame |
| $K$ | number of velocity **slots** (`--num_vel_modes`, $=2$) |
| $h^k_t,c^k_t\in\mathbb{R}^{C_h\times H\times W}$ | hidden / cell state of slot $k$ |
| $u^k_t=(u_x,u_y)\in\mathbb{R}^2$ | velocity of slot $k$, pixels/frame |
| $z^k_t\in\mathbb{Z}^2$ | phase-correlation **measurement** |
| $\hat u^k_t$ | dynamics-head **prediction** |
| $s^k_t\in\mathbb{R}^{d}$ | GRU state of the dynamics head, $d=$ `--vel_dyn_state_dim` |
| $\mathcal{T}_u$ | warp (transport) operator |
| $\psi_a$ | rigid spatial shift by $a$ on the torus |

All spatial operations are on the **torus** $\mathbb{Z}_H\times\mathbb{Z}_W$:
convolutions use `padding_mode="circular"`, the warp wraps, and the data
generator places digits with an integer `torch.roll`. There are no boundaries.

---

## 2. The warp operator

$$
(\mathcal{T}_u\,x)(\mathbf{p}) \;=\; x\big(\mathbf{p}-u \bmod (H,W)\big),
\qquad \mathbf{p}=(i,j)
$$

so content sitting at $\mathbf p$ moves to $\mathbf p+u$. For non-integer $u$
the right-hand side is bilinear interpolation, implemented with `grid_sample`
on a **1-pixel circular pad** — without that pad the band straddling the wrap
normalises outside $[-1,1]$ and gets clamped instead of interpolated against
row $0$, which is what makes sub-pixel velocities safe.

$\mathcal{T}$ is a group action of $(\mathbb{R}^2,+)$:

$$
\mathcal{T}_u\mathcal{T}_w=\mathcal{T}_{u+w},\qquad
\mathcal{T}_0=\mathrm{Id},\qquad
\mathcal{T}_u^{-1}=\mathcal{T}_{-u}
$$

and a rigid shift is the same operator, $\psi_a=\mathcal{T}_a$. **This identity
is what every proof below rests on.**

---

## 3. The cell

Per slot $k$, given a velocity $u^k_t$:

$$
\tilde h^k_t=\mathcal{T}_{u^k_t}h^k_{t-1},
\qquad
\tilde c^k_t=\mathcal{T}_{u^k_t}c^k_{t-1}
$$

$$
\big[\,\mathbf i\;\mathbf f\;\mathbf o\;\mathbf g\,\big]
= W \circledast \big[\,X_t\,;\,\tilde h^k_t\,\big] + b
$$

$$
c^k_t=\sigma(\mathbf f)\odot\tilde c^k_t+\sigma(\mathbf i)\odot\tanh(\mathbf g),
\qquad
h^k_t=\sigma(\mathbf o)\odot\tanh(c^k_t)
$$

with $\circledast$ a **circular** convolution. Note that $X_t$ is broadcast
unchanged to all $K$ slots — only the state is transported. The frame readout
pools over slots:

$$
\hat X_t=\mathrm{Dec}\Big(\max_{k}\,h^k_t\Big)
$$

**Why transport at all.** Without $\mathcal{T}$, a ConvLSTM must encode
"this feature moved 3 px right" inside its weights, separately for every
velocity. With $\mathcal{T}$, motion is applied exactly and for free, and the
convolution only has to model what genuinely changed.

---

## 4. Motion equivariance

Let a **global motion** with velocity $v$ act on the input sequence:

$$
X'_t=\psi_{tv}\,X_t
$$

i.e. every frame is shifted by an extra $t\,v$ — the whole scene given a
constant drift $v$ on top of whatever it was doing.

**Claim.** If the velocities transform as $u'^k_t=u^k_t+v$, then

$$
h'^k_t=\psi_{tv}h^k_t,\qquad c'^k_t=\psi_{tv}c^k_t,\qquad
\hat X'_t=\psi_{tv}\hat X_t .
$$

**Proof (induction on $t$).** At $t=0$: $h_{-1}=c_{-1}=0$, $u_0=0$, and
$\psi_0=\mathrm{Id}$, so the base case is trivial. Assume it at $t-1$. Then

$$
\mathcal{T}_{u'^k_t}h'^k_{t-1}
=\mathcal{T}_{u^k_t+v}\,\psi_{(t-1)v}h^k_{t-1}
\;\overset{(\ast)}{=}\;\mathcal{T}_{u^k_t+v+(t-1)v}\,h^k_{t-1}
=\psi_{tv}\,\mathcal{T}_{u^k_t}h^k_{t-1}
=\psi_{tv}\,\tilde h^k_t
$$

where $(\ast)$ is exactly the group law of §2. The convolution is circular,
hence translation-equivariant, $W\circledast\psi_a y=\psi_a(W\circledast y)$;
the input also carries $\psi_{tv}$; and $\sigma,\tanh,\odot$ are pointwise so
they commute with $\psi$. Therefore every gate, $c^k_t$ and $h^k_t$ pick up
exactly $\psi_{tv}$. $\square$

**So the entire architecture is exactly motion-equivariant *provided the
velocity estimates shift by $+v$*.** That proviso is the whole design
constraint on everything in §5–§7: any module that touches velocity must
satisfy it, or the guarantee is void — not degraded, void.

---

## 5. The measurement: phase correlation

For two maps $a,b$ (channels averaged first):

$$
\hat R(\boldsymbol\xi)
=\frac{\mathcal{F}\{a\}(\boldsymbol\xi)\;\overline{\mathcal{F}\{b\}(\boldsymbol\xi)}}
{\big|\mathcal{F}\{a\}\overline{\mathcal{F}\{b\}}\big|+\epsilon},
\qquad
r=\mathcal{F}^{-1}\{\hat R\}
$$

$$
z=-\operatorname*{arg\,max}_{\mathbf p} r(\mathbf p),
\qquad\text{wrapped to } \big[-\tfrac H2,\tfrac H2\big)\times\big[-\tfrac W2,\tfrac W2\big)
$$

Normalising by the magnitude keeps only the **phase**, so $r$ is a sharp peak
at the displacement rather than a blurred correlation. If $b=\psi_d a$ then
$\mathcal{F}\{b\}=e^{-\mathrm i\langle\boldsymbol\xi,d\rangle}\mathcal{F}\{a\}$,
so $\hat R=e^{+\mathrm i\langle\boldsymbol\xi,d\rangle}$ and $r=\delta_{-d}$,
giving $z=d$ exactly.

**Two call sites, two roles.**

$$
\underbrace{z^{1..K}_1=\operatorname{top-}K\ \text{peaks of } r(X_0,X_1)}_{\textbf{bootstrap, } t=1}
\qquad
\underbrace{z^k_t=\mathrm{PC}\big(\bar h^k_{t-1},\,X_t\big)}_{\textbf{tracking, } t\ge2},
\quad \bar h^k=\tfrac1{C_h}\textstyle\sum_c h^k[c]
$$

Tracking asks, per slot, *"where did **my** content go in the new frame?"* —
each slot queries its own $\bar h^k$, so there is no assignment problem after
the bootstrap.

**Equivariance of the measurement.** If $a\mapsto\psi_\alpha a$ and
$b\mapsto\psi_\beta b$ then $r$ shifts by $\alpha-\beta$ and
$z\mapsto z+(\beta-\alpha)$. Under the global motion of §4:

- bootstrap: $\alpha=0$ (frame $X_0$), $\beta=v$ (frame $X_1$) $\Rightarrow z\mapsto z+v$ ✓
- tracking: $\alpha=(t-1)v$ (state), $\beta=tv$ (frame) $\Rightarrow z\mapsto z+v$ ✓

Both shift by exactly $+v$, which is the proviso §4 needs. The measurement is
computed under `no_grad` and comes from an `argmax`, so **$z$ is a constant
with respect to every parameter** — this matters in §11.

---

## 6. The velocity dynamics head (the new part)

### 6.1 Definition

Per slot, with $\delta u^k_t=u^k_t-u^k_{t-1}$:

$$
s^k_t=\mathrm{GRU}\Big(\big[\;\delta u^k_t\;;\;\phi(h^k_t)\;\big],\;s^k_{t-1}\Big)
$$

$$
\Delta^k_t=W_o\,s^k_t+b_o,
\qquad
\boxed{\;\hat u^k_{t+1}=u^k_t+\Delta^k_t\;}
$$

$$
\phi(h)=W_\phi\Big[\;\underbrace{\tfrac{1}{HW}\textstyle\sum_{ij}h}_{\text{mean}}\;;\;\underbrace{\max_{ij}h}_{\text{max}}\;\Big]
\quad\text{(only when \texttt{use\_h}; default off)}
$$

One `GRUCell` is **shared across slots** and applied to a flattened $(B\!\cdot\!K)$
batch, so every slot uses the same dynamics but carries its own state
$s^k$ — which is what makes "one global velocity per slot" come out.

### 6.2 Why $\delta u$ and not $u$ — the design is forced

The head must satisfy, **exactly**,

$$
g\big(u_{\le t}+v,\;\psi_a h_t\big)=g\big(u_{\le t},\,h_t\big)+v
\qquad\forall\,v,a .
$$

Two facts deliver it:

1. $\delta u$ is **invariant** under $u\mapsto u+v$, since
   $(u_t+v)-(u_{t-1}+v)=\delta u_t$. Feeding raw $u$ would make $s$ a function
   of the global motion, and the head would have to *learn* the identity in $v$
   — approximately, on the training distribution only.
2. $\phi$ is **invariant** under $\psi_a$: mean and max over the full spatial
   extent are permutation-invariant over pixels, and a cyclic roll is a
   permutation. Any position-dependent readout (raw $h$, crops, coordinate
   channels) would break this.

Hence $s^k_t$ and $\Delta^k_t$ are invariant, and

$$
\hat u^k_{t+1}\;=\;\underbrace{u^k_t}_{\text{equivariant, }\mapsto\,u+v}\;+\;\underbrace{\Delta^k_t}_{\text{invariant, }\mapsto\,\Delta}\;\longmapsto\;\hat u^k_{t+1}+v \;\;\checkmark
$$

*An invariant increment added to an equivariant base.* Verified numerically
through a full 10-step rollout of a **trained** head:
$\max|g(u+c)-g(u)-c| = 2.4\times10^{-6}$ (float32 round-off).

### 6.3 Zero initialisation

$W_o=0,\;b_o=0$ at init $\Rightarrow\Delta\equiv 0\Rightarrow\hat u_{t+1}=u_t$
**exactly** — which *is* the frozen-velocity rollout. The head therefore
strictly **nests** the previous behaviour: turning it on cannot move an
existing result by one ulp (there is a bitwise test), so anything that changes
afterwards is something the head learned.

### 6.4 The head's state must never be warped

$(h,c)$ live on the image grid and **must** be transported. $s^k$ lives in
velocity-difference space, which a global motion does not touch — it is
**invariant**. Warping it would be a bug. It has no spatial axes, so the type
system enforces this, and there is a test.

---

## 7. Coupling: a predict/correct filter

At any step where a measurement exists:

$$
\boxed{\;u^k_t=\hat u^k_t+\kappa^k_t\big(z^k_t-\hat u^k_t\big)\;}
$$

- `--vel_dyn_gain fixed` (default): $\kappa\equiv1$, so $u_t=z_t$ — the
  measurement is taken verbatim and the head is trained but does not steer the
  encoder. Implemented as literal assignment, not $\hat u+1\cdot(z-\hat u)$,
  because the latter is not bitwise equal to $z$ in floating point and §6.3
  must hold exactly.
- `--vel_dyn_gain learned`: $\kappa^k_t=\sigma\!\big(\mathrm{MLP}(\rho^k_t)\big)$
  with $\rho$ the correlation **peak height**. $\rho$ is invariant under global
  translation (a shift moves the peak, not its height), so $\kappa$ is
  invariant, $z-\hat u$ is invariant, and $u_t$ stays equivariant. Kalman-like:
  a weak peak defers to the process model.

---

## 8. The encoder

For $t=0,\dots,T_{\text{in}}-1$, with $n$ = number of measurements consumed:

$$
z_t=\begin{cases}
0 & t=0 \quad(h=c=0,\ \mathcal{T}_u0=0\ \forall u)\\[2pt]
\text{bootstrap}(X_0,X_1) & t=1\\[2pt]
\mathrm{PC}(\bar h^k_{t-1},X_t) & t\ge 2
\end{cases}
$$

then, for $t\ge1$,

$$
\hat u_t,\,s_t=\mathrm{head}(u_{t-1},\,\delta u_{t-1},\,h_{t-1},\,s_{t-1}),
\qquad
u_t=\hat u_t+\kappa_t(z_t-\hat u_t)
$$

$$
\delta u_t=\begin{cases}0 & n<1\\ u_t-u_{t-1}&n\ge1\end{cases}
\qquad\qquad
(h_t,c_t)=\mathrm{cell}(X_t,h_{t-1},c_{t-1},u_t)
$$

$\delta u$ is held at $0$ until two real measurements exist — feeding
$u_1-u_0=u_1-0$ would tell the GRU the digit accelerated from rest, which never
happened.

**Open-loop replay** (`--vel_dyn_openloop_k`$=\kappa$). A *side branch*, which
does not touch the states above: fork $(u,\delta u,s)$ at
$t_\star=\max(2,T_{\text{in}}-\kappa)$ and run the head on its **own** output,

$$
\hat u_{t+1}=\mathrm{head}(\hat u_t,\ \hat u_t-\hat u_{t-1},\ h_t,\ s_t),
\qquad t=t_\star\ldots T_{\text{in}}-1
$$

scoring each against the measurement already in hand. This is the only place
the head is trained in the regime it is *evaluated* in (§9), which is why
$\kappa$ should equal the rollout length.

---

## 9. The decoder — three modes

The cell input is always the model's own previous prediction (no teacher
forcing), $\hat X_{-1}:=X_{T_{\text{in}}-1}$. The three modes differ only in
where $u$ comes from:

$$
u_t=\begin{cases}
u_{T_{\text{in}}-1} & \textbf{frozen} \quad\text{— deployable; exact iff the velocity is constant}\\[4pt]
\mathrm{PC}(\bar h^k_{t-1},Y_t) & \textbf{tracked} \quad\text{— oracle: uses the true next frame}\\[4pt]
\hat u_t=\mathrm{head}(u_{t-1},\delta u_{t-1},h_{t-1},s_{t-1}) & \textbf{predicted} \quad\text{— deployable, and extrapolates}
\end{cases}
$$

`frozen` is the honest baseline, `tracked` is the upper bound, and the gap
between them isolates *velocity error* from *rendering error*. `predicted` is
the new option and should land between them. **During training the decoder is
always `tracked`** (`track_decoder_velocity = model.training`); the head is run
and supervised there too, but its output is not used.

---

## 10. The objective

$$
\mathcal{L}
=\underbrace{\frac{1}{|\Omega|}\sum_{\Omega}\Big[(\hat X-Y)^2+|\hat X-Y|\Big]}_{\mathcal{L}_{\text{rec}}}
\;+\;\lambda\,
\underbrace{\frac{1}{|\mathcal S|}\sum_{t\in\mathcal S}\mathrm{smooth}_{L_1}\!\big(\hat u_t,\ \mathrm{sg}[z_t]\big)}_{\mathcal{L}_{\text{dyn}}}
$$

with $\lambda=$`--vel_dyn_loss_weight` and $\mathcal S$ the supervised steps
(encoder $t\ge2$, all decoder steps under the tracked protocol, plus the
open-loop replay). $\mathrm{sg}[\cdot]$ is stop-gradient: **the measurement is
the target and the head learns from it; it must never learn from the head.**
The mean (not sum) makes $\lambda$ independent of $T_{\text{in}}$ and horizon.

---

## 11. Backpropagation — what actually receives gradient

Measured by backward passes on each loss separately (sum of $|\partial|$ over
each module's parameters). All rows are taken with $W_o\neq0$, i.e. *after* the
output layer has left its zero initialisation — see (c) for why that matters:

| loss | configuration | $\to$ head | $\to$ ConvLSTM cell | $\to$ decoder |
|---|---|---:|---:|---:|
| $\mathcal{L}_{\text{dyn}}$ | `use_h=False` (default) | $7.01$ | $\mathbf{0}$ | $\mathbf{0}$ |
| $\mathcal{L}_{\text{dyn}}$ | `use_h=True` | $7.54$ | $0.105$ | $\mathbf{0}$ |
| $\mathcal{L}_{\text{rec}}$ | fixed gain, decoder `tracked` | $\mathbf{0}$ | $2.507$ | $9.07$ |
| $\mathcal{L}_{\text{rec}}$ | learned gain, decoder `tracked` | $6.7\times10^{-5}$ | $2.508$ | $9.07$ |
| $\mathcal{L}_{\text{rec}}$ | fixed gain, decoder `predicted` | $8.7\times10^{-2}$ | $2.509$ | $9.04$ |

Read off:

**(a) In the default configuration the two losses are completely disjoint.**
$\mathcal{L}_{\text{dyn}}$ reaches only the head; $\mathcal{L}_{\text{rec}}$
reaches only the ConvLSTM and decoder. The velocity regulariser cannot perturb
the representation, and the image loss cannot train the velocity. The head is a
**bolt-on predictor trained purely by imitation of the phase-correlation
measurement.**

Why $\mathcal{L}_{\text{rec}}$ cannot reach the velocity: with $\kappa=1$,
$u_t=z_t$, and $z$ is an `argmax` index computed under `no_grad` — a constant.
$\partial u_t/\partial\theta=0$, so although $\mathcal{T}_u$ *is*
differentiable in $u$, there is nothing upstream to differentiate.

**(b) `use_h=True` breaks that isolation.** $\phi(h)$ is differentiable, so
$\mathcal{L}_{\text{dyn}}$ then flows back into the ConvLSTM ($0.105$, about
$4\%$ of what $\mathcal{L}_{\text{rec}}$ delivers there). The ablation is not
free: it lets the velocity loss reshape the representation.

**(c) The zero-init delays the gradient, and this is why the table fixes
$W_o\neq0$.** Since
$\partial\mathcal{L}/\partial s=W_o^{\!\top}\,\partial\mathcal{L}/\partial\Delta$
and $W_o=0$ at initialisation, the GRU receives **exactly zero** gradient on
the first step — only $W_o$ and $b_o$ move, and with `use_h` the cell also gets
nothing. Once $W_o\neq0$ the GRU picks up its share ($6.3$ of the $7.0$ above)
and the `use_h` path into the ConvLSTM opens. The head "switches on" its
recurrence only after the output layer leaves zero, so gradient magnitudes
measured at step 0 are not representative.

**(d) Gradient paths that exist inside the head.** With $u_{t-1}$ and
$\delta u_{t-1}$ constants (they are measurements), the only live path is the
GRU recurrence $s_{t-1}\!\to\!s_t$. So $\mathcal{L}_{\text{dyn}}$ is
backpropagated through time **within the head's own state**, and nowhere else.

**(e) The autoregressive image loop is cut.** The decoder does
`current_frame = prev_frame.detach()`, so no gradient flows across predicted
*frames*. It does flow through $(h,c)$, which are not detached — so BPTT runs
through the recurrent state, over the whole encoder and decoder, but not
through the pixel feedback.

### The train/evaluate asymmetry this creates

Two mismatches follow directly from the table, and both are worth keeping in
view:

1. **Horizon.** The head is trained one step ahead (restarted from a real
   measurement each time) but evaluated over a full open-loop rollout.
   `--vel_dyn_openloop_k` is the only thing that closes this, which is why it
   should equal the rollout length.
2. **Distribution.** During training the ConvLSTM *only ever* sees measured
   velocities. In `predicted` evaluation it is fed head velocities it has never
   been trained on. If the head is biased, the ConvLSTM is off-distribution —
   and, per (a), it was never given a gradient that would let it adapt. The
   available remedies are to train with the predicted decoder some fraction of
   the time (scheduled sampling on the velocity), or to let $\kappa<1$ so the
   head influences the encoder. Neither is currently enabled.

---

## 12. Why the data has to be `harmonic`

The head predicts $\Delta$ from $\delta u$ alone. So it can learn a dynamics
**iff the next increment is a function of past increments.** If $u$ obeys any
linear time-invariant recurrence

$$
u_{t+1}=\sum_{m=0}^{M}a_m\,u_{t-m},
$$

then applying the difference operator $D=1-S^{-1}$ (which commutes with every
LTI operator) gives

$$
D\,u_{t+1}=\sum_{m=0}^{M}a_m\,D\,u_{t-m}
\quad\Longleftrightarrow\quad
\delta u_{t+1}=\sum_{m=0}^{M}a_m\,\delta u_{t-m}.
$$

**$\delta u$ obeys the same recurrence as $u$.** Sinusoids are the canonical
such signals, hence:

$$
u(t)=d+\Big\lfloor A\odot\cos(\omega t+\varphi)\Big\rceil,
\qquad d\in\mathbb{Z}^2
$$

with $A,\omega,\varphi$ drawn per digit and the trajectory then deterministic.
For the pure `orbit` shape $u_{t+1}=R(\omega)u_t$, so
$\delta u_{t+1}=R(\omega)\,\delta u_t$ — one parameter, identifiable from two
consecutive increments.

The drift $d$ is an **integer** and added *after* rounding, so
$\lfloor d+A\cos\rceil=d+\lfloor A\cos\rceil$ and $d$ cancels in $\delta u$: it
is invisible to the head by construction, costs it nothing, and still makes the
path a looping trochoid.

Why the older modes gave the head nothing:

| mode | next increment depends on | learnable? |
|---|---|---|
| `constant` | nothing, $\delta u=0$ | yes — but freezing is already optimal |
| `piecewise`, `stochastic` | i.i.d. noise | no signal; the optimal prediction **is** freeze |
| `accelerate` | $\lvert u\rvert$ hitting `max_speed` | **no** — a rule about the *absolute* velocity, which an equivariant head cannot see |
| `harmonic` | past increments | **yes** |

---

## 13. Switch summary

| flag | default | effect |
|---|---|---|
| `--use_velocity_dynamics` | off | constructs the head at all |
| `--vel_dyn_loss_weight` | $0$ | $\lambda$; at $0$ the head never trains |
| `--vel_dyn_state_dim` | $32$ | $\dim s$; $3{,}522$ params total |
| `--vel_dyn_use_h` | off | adds $\phi(h)$ — and couples $\mathcal{L}_{\text{dyn}}$ into the ConvLSTM (§11b) |
| `--vel_dyn_gain` | `fixed` | $\kappa=1$ vs $\kappa=\sigma(\mathrm{MLP}(\rho))$ |
| `--vel_dyn_openloop_k` | $0$ | multi-step replay length (§8) |
| `--eval_velocity_mode` | `frozen` | `all` logs frozen / predicted / tracked together |

With everything off the forward pass is bit-for-bit the pre-existing model.
