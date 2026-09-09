# Overlap bench — handoff note

For whoever is implementing this. Same design as `DESIGN.labkit.md`,
explained as I would at a whiteboard: what we want to find out, why it
is set up this way, what to build first, and what would make me say we
were wrong. Numbers, thresholds and lock tables live in
`DESIGN.labkit.md` and are authoritative; if this note disagrees with
that file, that file wins and this note has a bug.

## Where this came from

We have a faithful port of a published oscillator-network image
segmenter (`07_posn` — the reference implementation, verified against
the authors' MATLAB). It works on the six toy images the paper ships.
We also have a folder of old experiments (`06_nb_mnist_phase_encoder`)
that is mostly dead ends, but contains one genuinely useful thing: a
proper benchmark. Three datasets of 32×32 images with *overlapping*
objects and per-pixel ground truth, 50k/10k/10k each, from the
`ComplexAutoEncoder` paper.

Somebody ran a quick probe: the reference segmenter, default
parameters, on 50 validation images from two of those datasets, against
the dumbest possible baseline — `scipy.ndimage.label`, connected
components on the thresholded image. Result:

| Dataset | Connected components | Reference segmenter |
|---|---:|---:|
| `2shapes` | 0.160 | 0.062 |
| `MNIST_shapes` | 0.016 | 0.143 |

On one dataset the oscillator network *loses to thresholding*. On the
other it wins by 9×. Same code, same parameters. That is either a real
and interesting failure mode, or a single-seed fluke — the probe drew
one random initial state per image, and the reference implementation's
own notes say the score is seed-sensitive.

Nobody knows which. That is the question.

## The idea in one paragraph

We want to know whether the fixed-weight, hand-tuned oscillator
approach *generalises* — and when it doesn't, whether the reason is the
parameters (tuned on the paper's toy shapes, wrong for this data) or
the mechanism (a fixed Gaussian coupling can't do overlap no matter how
you tune it). Then we ask the obvious follow-up: if you make the same
linear dynamics *trainable*, does that fix it? And only after both of
those have real numbers do we bring in a structurally different
nonlinear model as a third point of comparison.

Three models, one benchmark, one score, one protocol. No combined
"do oscillators work" verdict at the end — each model gets its own
answer.

## Why it is three experiments, and why in this order

**M0 is the reference dynamics, untouched.** We don't change a line of
`07_posn`. We call it, seed-average it, and sweep its two parameters.
This answers "does what exists transfer" and "is the probe real".

**M1 is the same maths, made trainable.** The reference model is
$x_{t+1} = (K + i\,\mathrm{diag}(\omega))\,x_t$ with $K$ a fixed
Gaussian sheet and $\omega$ the image. The continuous version
$x(t) = \exp((i\,\mathrm{diag}(\omega) + K)\,t)\,x(0)$ is exactly
computable with a matrix exponential, and there's a verified port of
that in `06` (`port/cv_rnn_exact.py` — the code is right, the
docstring's factorisation is wrong for per-node $\omega$, ignore the
docstring). Make $K$ and $\omega$ trainable, keep the reference's
readout so the readout isn't what changed, and train it. The important
control: at initialisation, before any gradient step, `M1` must give
*the same labels* as `M0`. If it doesn't, "trained" just means
"different code" and the comparison is meaningless.

**M2 is a different mechanism.** Real unit vectors per node, coupling
applied in the tangent space, a learned rotation — the AKOrN idea.
There's a notebook in `06` where it ran once on a fish picture. That
tells us the update executes and nothing else. Reimplement it from the
equations. It goes *last*, behind a gate, because comparing a third
model against two you haven't measured yet is comparing against
assumptions.

The ordering is the point. Arc 1a asks whether the probe survives
seeds. Arc 1b sweeps parameters *only if it does* — sweeping to explain
a fluke explains nothing. Arc 1c trains `M1` in parallel with 1b once
its harness passes. `M2` waits for both.

## Build order

**Arc 0, the harness — not science.**

For `M0`:
- A driver that loads a CAE `.npz` image, calls `run_2layer_torch` and
  `spatiotemporal_segmentation_torch` from `07_posn` at the pinned
  commit with *no argument overridden*, and returns foreground ARI.
- Check it reproduces `07`'s own recorded ARI on its three bundled
  images, same seed, to $10^{-12}$. It's the same code, so this is an
  identity — if it fails, the driver is wrong.
- The `cc` baseline beside it. Check it reproduces the probe's two
  numbers exactly (0.160, 0.016). It's deterministic, so exact.

For `M1`:
- The propagator, checked against an independent `scipy.linalg.expm`
  to $10^{-12}$ relative.
- A finite-difference gradient check that gradients actually reach
  both $\omega$ and $K$.
- The initialisation control: `M1` at init with `M0`'s readout gives
  `M0`'s labels on 50 `2shapes` images under seed 1.
- Pick the loss. It needs to be a differentiable surrogate for
  foreground ARI. Fix it, write it down, *before* Arc 1c runs — the
  written-down timestamp is what stops "we chose the loss after seeing
  validation".

**Arc 1a — does the probe survive seeds.** `M0` and `cc`, ten seeds,
both datasets, mean and standard deviation. Two conditions: the split
is real (`M0` below `cc` on `2shapes` and above on `MNIST_shapes`, both
by more than two seed-SDs), and seed noise isn't the story (SD less
than half the gap). If the split only holds on one dataset, the
question narrows to that one. If it holds on neither, we're done with
`M0`: the probe was a fluke, and the record says so.

**Arc 1b — parameters or data.** 27-cell sweep over `alpha` and
`sigma`$_1$, tuned on 50 *training* images, reported on the same 50
*validation* images. If some cell beats `cc` on `2shapes`, the default
parameters were the problem. If no cell does, the fixed mechanism
doesn't beat thresholding on this data at any tuning we tried — and
note the wording: "at any tuning tried", never "intrinsic". A second
condition checks the winning cell didn't buy `2shapes` by selling
`MNIST_shapes`.

**Arc 1c — does training help.** `M1`, fixed budget (20 epochs, no
early stopping — early stopping is a way of peeking), ten seeds. Two
conditions kept separate: does training beat `M0`'s best swept cell,
and does it beat `cc`. The first says whether training did anything.
The second is the one that matters. If training helps and still loses
to thresholding, that's a real finding — it's the one that makes `M2`
worth building.

**Arc 2 — the nonlinear model.** Its locks get written when its gate
opens, not now, because what it has to beat is what Arc 1 decides.

## What I want from you

- **Don't touch `07_posn` or `06`.** `M0` calls `07`'s functions at
  the pinned commit; it does not copy them. The datasets are *not*
  pinned by `06`'s commit — that directory is gitignored there — so
  they are pinned by sha256, and the harness checks the hashes before
  anything runs.
- **Take exactly three things from `06`**: the datasets, the exact
  propagator's factorisation, and the notebook's update equations for
  `M2`. Everything else in there is superseded, broken, or was never
  actually run — the `DESIGN.labkit.md` isolation note lists what and
  why. Don't rescue any of it.
- **Foreground ARI, one definition.** Pixels whose true label is
  positive (`truth > 0`). Overlap ($-1$) and background ($0$) excluded.
  Whole-image ARI gets reported as a diagnostic and decides nothing.
  This is the probe's own expression, confirmed by its author, and the
  harness reproduces the probe's 0.160 and 0.016 from it exactly.
- **`n_clusters = 2`, hardcoded.** The probe derived it from ground
  truth per image and got 2 every time; we hardcode the value so Arc
  1a changes one thing against the probe (seeds),
  not two. It happens that every image in both Arc 1 datasets has
  exactly two objects, so the default and the true count coincide;
  `3shapes`, where they wouldn't, is out of scope. Cluster count is
  not what we're measuring.
- **Ten seeds, all of them, every time.** No dropping a bad one. No
  drawing an eleventh.
- **Compare to the thresholds as written.** No "close enough".

## What would make me say we were wrong

- **The harness identity fails.** `M0`'s driver doesn't reproduce `07`'s
  own numbers, or `M1` at init doesn't match `M0`. Then nothing
  downstream means anything; fix the harness.
- **The probe was a fluke** (Arc 1a `P1` fails on both). Fine — that's
  an answer, and a cheap one. The record closes `M0`'s pursuit
  answered *no split to explain* and we didn't waste a sweep on it.
- **Parameters explain everything** (`P3` and `P4` pass). Also fine.
  Then the interesting question is why the paper's defaults are so
  brittle, and `M1` becomes less urgent.
- **Training helps but nothing linear beats thresholding.** That's the
  outcome I actually expect, and it's the one that justifies `M2`.

## What the record will and won't say

The probe's two numbers go on the record as an *observation marked
reconstructed* — we didn't run it under the protocol, we're recording
what was seen — with no analysis over them. They prompted the question;
they are not evidence for anything. The first real number is Arc 1a's.

The parameter tables are prose in a note. Nothing checks
`alpha = (0.5, 0.5)` as a number. What makes the locks binding is a
condition on every analysis — *was this run under the locks as
written* — that a person checks by reading the analysis against the
note. If you change a lock, that is a new pursuit, not an edit.

`M2` has no locks yet, deliberately. Writing them now would fix a
target that Arc 1 hasn't produced.

## Where things stand (added after Arc 1a)

The record in `./.labkit` is the authority now; `labkit now` tells you
what is ready, blocked, and why. This note is not updated below this
line. What has happened since it was written:

- **The probe's 2shapes result was a fluke.** Ten seeds put the fixed
  dynamics at 0.121 ± 0.040 against thresholding's 0.160 — inside one
  seed-SD. The 9× win on MNIST_shapes is real (0.135 ± 0.018 vs 0.016).
- **So the sweep never runs.** There is no 2shapes failure to explain
  with parameters. That is the design working, not a problem.
- **M2 is closed.** No clean failure for a third mechanism to fix. If
  training the linear model (M1) turns out to help and *still* lose to
  thresholding, that is the reason to reopen it — as a new pursuit.
- **What's left:** build M1, prove it equals M0 at initialisation, train
  it, and ask the one question still open — can a *trained* linear
  oscillator model beat thresholding on 2shapes? P5's comparator is now
  M0 at defaults, since the sweep cell it named never exists.

## After harness-M1

**My M1 spec was wrong, twice.** First: I wrote M1 as $x(t)=e^{At}x_0$
with $A$ the reference model's per-step matrix. But the reference is a
*discrete* map, $x_{n+1}=Ax_n$, and $e^{A}\neq A$ — so "equals M0 at
initialisation" could never hold, whatever the time convention. The
implementor built it as specified, watched it overflow by step 2, and
recorded the failure rather than patching around it. Correct.

Second: I'd missed that the reference runs *two* layers. Layer 1 runs 59
steps only to vote a background mask (an argmax — no gradients through
it). Layer 2 restarts from the masked initial state under a 29× gentler
coupling and runs to step 199, and *that* is the only trajectory the
readout ever looks at. The implementor caught this too.

**So M1 is now M1d:** the reference's layer-2 recurrence exactly, with its
coupling $K_2$ and frequencies $\omega_2$ trainable, and layer 1 plus the
mask kept fixed as the reference computes them. No exponential. It equals
M0 at initialisation by construction; the harness check is that it does,
to $10^{-12}$ on the trajectory and exactly on the labels.

**Then the last question:** train it, and ask whether a trained linear
oscillator model beats thresholding on 2shapes. That's all that is left.

## The result

**Training made it worse, and now we know exactly why.** The trainable
copy of the reference's layer-2 recurrence was verified to be the
reference exactly at initialisation, then 500 optimiser steps took it to
*chance* on 2shapes (−0.01, against the reference's 0.12 and plain
thresholding's 0.16). First explanation: the loss only looks at phases,
so amplitudes ran away and every image became one eigenvector. That is
a true description of the dynamics. The real reason, found by the
implementor after we had closed the record: the loss as I *specified*
it had a between-object term of $|z_o\bar z_{o'}|/(|z_o||z_{o'}|)$ —
which is identically 1 for any two complex numbers. The code
implemented the formula faithfully. So the loss was just
"maximise within-object coherence" with nothing whatsoever against
all objects sharing a phase, and gradient descent did precisely that.

The numbers stand. What they mean changed: they are evidence that we
never trained a discriminating objective, not that the model can't
learn one. The intended term is $\cos(\phi_o-\phi_{o'})$, and it has
not been tried.

**So where does the question stand?** Parked, on purpose. It asked
"when the fixed model fails, is it the parameters or the data?" — and at
ten seeds it doesn't fail: it ties thresholding on 2shapes and beats it
nine-fold on MNIST_shapes. The one thing that *did* fail was the trained
model, for a reason that is neither parameters nor data. The record
says *accepted as unresolved*, with two things written down that would
reopen it: a seed count or dataset where the fixed model actually loses,
or training under the loss that was actually intended — the one with
a real between-object term. That's the honest bucket — not answered, not abandoned.

One caution the implementor added after checking: the trained model
*sits* at the loss's trivial minimum, measured; whether training could
have escaped it is not established, and nobody should say "can't".

**What would come next, if anyone wants it:** a loss that penalises
amplitude growth (a spectral penalty, or normalise inside the surrogate).
That is a new pursuit under the same question — a different claim about
a different loss — not a tweak to this one. Nobody has started it.

Everything above is on the record with handles; `labkit why` on any of
them gives the chain. This note stops here.
