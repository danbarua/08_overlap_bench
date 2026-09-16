# Pursuit B: what the free-running attractor looks like past the trained horizon

This note reports `scripts/characterize_learned_attractor.py`'s findings on the
10,000-step checkpoint: 50 validation images x 10 seeds x 500-step free-running
`Layer2` rollouts, extending well past the 140-step window layer 2 is trained
and read out at.

The LabKit record remains authoritative for research standing; this is an
exploratory mini note. It does not supersede or restate `PURSUIT-B-MECHANISM.md`
-- it extends that note's readout-window (updates 80-120) findings to a much
longer horizon and independently reproduces its central claim by a different
method.

## Setup

- Checkpoint: `outputs/ckpt-10000.pt`.
- 50 `2shapes_val.npz` images, seeds 1-10, `TRAJECTORY_STEPS = 500`.
- Artifact: `outputs/attractor-10000.json`, sha256
  `3c437c1cbd37b00f85d73928c5f2fccfc4ea46a7c0f58eb9f3fcaf38297290db`.
- Reproduction:
  ```sh
  PYTHONPATH=src uv run python scripts/characterize_learned_attractor.py \
    --checkpoint outputs/ckpt-10000.pt --device cuda \
    --out outputs/attractor-10000.json
  ```

`Layer2._run` restarts from the layer-1 masked state and is affine-linear in
`x` for a fixed mask, so this is not a claim about a nonlinear chaotic system
in the colloquial sense -- see below.

## Finding 1: the "Lyapunov exponent" is exactly `log|top eigenvalue|`

For a fixed mask the recurrence is `x_{t+1} = A_i x_t` with
`A_i = P_i K_2 P_i + i diag(P_i(omega+delta_omega))` (same operator as
`PURSUIT-B-MECHANISM.md`). Eigendecomposing `A_i` for one sample (image 0,
seed 1) gives top eigenvalue `5.169 + 0.987j`, `log|lambda_top| =
1.6606609720421173`. The script's independently measured Lyapunov exponent
for that exact sample (perturbation-growth method, steps 0-200) is
`1.6606609720441858` -- agreement to 10 significant digits. This is expected
for a linear system and confirms the measurement code is behaving correctly,
not evidence of anything new by itself.

## Finding 2: instability strength is image-dependent, seed-independent

Across all 500 samples: mean Lyapunov exponent `1.985`, std `0.218`, range
`[1.425, 2.419]`. Decomposed by source: std **across images** accounts for
essentially all of it (`0.218`); std **across the 10 seeds within a fixed
image** is `~0.0000`. Different random initial phases converge to layer-1
masks whose induced operator has the same leading eigenvalue for a given
image. This is consistent with `PURSUIT-B-MECHANISM.md`'s finding that no
mask pixel differs across seeds at 10,000 steps, and sharpens it: not just
the mask but the *entire top-eigenvalue-driven growth rate* is a
seed-invariant property of the image.

## Finding 3: the phase pattern locks in almost immediately and never moves again

The order-parameter magnitude (global phase coherence, computed from
`angle(orbit)` only -- insensitive to the amplitude renormalization described
below) is `0.824` averaged over all 500 samples at step 0 and `0.825` at step
499: flat across the entire 500-step trajectory, no drift, while amplitude is
exploding through roughly 300 orders of magnitude over the same window.
Phase and amplitude are dynamically decoupled: amplitude is the unstable
direction: the Lyapunov exponent is its exponential growth rate; phase is
frozen close to whatever pattern it reaches almost immediately.

## Finding 4: that frozen phase pattern is exactly the top eigenvector, on the correct support

Comparing the final-step orbit's phase to the top eigenvector's phase, first
attempt (all 1024 pixels, including the ~840 masked-out background pixels
that the recurrence pins to exactly zero every step) gave a misleadingly weak
alignment of `0.739`: `angle()` of near-zero floating-point noise is
meaningless and corrupted the average. Restricted to the 172 real foreground
pixels for this image, the alignment (`|u^H v| / (||u|| ||v||)`, the same
projective metric `PURSUIT-B-MECHANISM.md` uses) is `1.0000000000000002` --
numerically exact. This directly reproduces that note's readout-window
finding (mean leading-eigenvector alignment `0.9999999996` at 10,000 steps,
measured at update 120) via an independent computation and a much longer
horizon, and shows the lock is not a transient near the readout window: it is
the trajectory's entire long-run behavior once the transient (a handful of
steps) has passed.

## What's new here versus `PURSUIT-B-MECHANISM.md`

That note characterizes the readout window (updates 80-120, inside training).
This note shows three things it did not: (1) the eigenvector lock, once
established, is stationary for at least 500 steps -- it is a genuine
attracting *direction*, not a window-specific artifact; (2) the standard
perturbation-based Lyapunov exponent exactly equals `log|lambda_top|`, an
independent confirmation of the spectral picture; (3) that exponent (hence
the whole instability character) is seed-invariant per image, extending the
existing seed-invariant-mask finding to the growth rate itself.

## What this does *not* show

- Nothing here is evidence of chaos in the informal sense: the operator is
  exactly linear given a fixed mask, so "Lyapunov exponent" here is a
  spectral-radius statement, not a statement about sensitive dependence in a
  genuinely nonlinear system.
- No claim about images/checkpoints outside this evaluation set.
- No re-derivation of why training reshapes `K2` this way -- see below, and
  `PURSUIT-B-MECHANISM.md`'s intervention section, which already locates the
  effect there.

## Why does ARI keep improving with training?

This is answered in `PURSUIT-B-MECHANISM.md`, not by this job; citing it
directly rather than re-deriving it. Two things move together across the
checkpoint trajectory (500 -> 10,000 Adam steps), both inside the trained
140-step / 80-120-readout window:

| steps | spectral modulus ratio `q` (mean/max) | within-object coherence | between-object cosine | mean FG-ARI |
|---:|---:|---:|---:|---:|
| 500 | 0.8139 / 0.9981 | 0.7654 | -0.8901 | 0.5897 |
| 4,000 | 0.7001 / 0.9733 | 0.9090 | -0.9980 | 0.8138 |
| 8,000 | 0.6286 / 0.9254 | 0.9504 | -0.99992 | 0.8984 |
| 10,000 | 0.6195 / 0.8883 | 0.9601 | -0.99997 | 0.9195 |

1. **The spectral gap widens.** `q = |lambda_2|/|lambda_1|` falls as training
   proceeds, so within a fixed 120-step readout window the state converges
   *more completely* onto a single dominant mode (mean alignment with that
   mode reaches `0.9999999996` by 8,000 steps). Early in training the gap is
   weak enough that different seeds' initial conditions haven't fully
   decayed onto the same mode by the readout time -- hence seed-dependent
   partitions at 500 steps (only 0/45 seed pairs agreeing) versus 45/45 at
   8,000-10,000 steps.
2. **The dominant mode itself becomes the right answer.** It's not just that
   convergence gets faster -- the leading eigenvector's own phase pattern
   becomes progressively more object-organized: within-object phase
   coherence rises `0.77 -> 0.96`, between-object phase goes from weakly
   anti-correlated (`-0.89`) to essentially exact antiphase (`-0.99997`).
   Faster convergence to a badly-organized mode would not raise ARI; both
   effects together do.
3. **Intervention evidence pins this to `K2`, not `delta_omega`.** Setting
   `delta_omega = 0` on the full 10,000-step model reproduces all 500
   partitions exactly (mean ARI `0.919518` vs `0.919518`... i.e. unchanged).
   Conversely, keeping the trained `delta_omega` but reverting `K2` to its
   untrained Gaussian-sheet initialization collapses mean ARI to `0.150613`
   and seed agreement to 0/45. Training's effect lives in `K2`'s broad,
   increasingly asymmetric structure (Frobenius relative change grows
   `0.49 -> 2.88` over 250 -> 10,000 steps, asymmetry ratio `0.53 -> 0.97`),
   not in the per-oscillator frequency offset.

The 20,000-step run's FG-ARI of `0.95104` (`RESULTS-20K.md`) is a continuation
of the same monotone trend past the mechanism note's last analyzed checkpoint
(10,000 steps, ARI `0.9195` in that note's table); it has not been run through
the same eigenvalue/alignment analysis, so I can report the trend but not
independently confirm the mechanism holds unchanged at 20,000 steps.

**One connection this job adds:** this note shows the mechanism improving
with training operates entirely *inside* the 140-step trained window. My
500-step probe shows that once you leave that window, more iteration buys
nothing further -- the phase pattern is already fixed by whatever mode it
locked onto near the start, and the only thing that keeps changing past
step ~140 is amplitude, unboundedly. Training improves ARI by shaping which
mode gets locked onto and how fast the lock completes within the readout
window, not by giving the system a genuinely stable long-run fixed point to
settle into.

## Does this generalize past 2shapes?

Ran the identical protocol (`--dataset` generalized into the script, not
hardcoded) on MNIST_shapes and 3shapes' 40,000-step checkpoints
(`PURSUIT-B-BEYOND-2SHAPES.md`). Zero NaN in coherence/spectrum for both,
500/500 samples each -- the renormalization fix generalizes cleanly.

| dataset (checkpoint) | Lyapunov mean/std | order param step0->499 | q | foreground alignment |
|---|---:|---:|---:|---:|
| 2shapes (10k) | 1.985 / 0.218 | 0.824 -> 0.825 | 0.7988 | 1.0000000000000002 |
| MNIST_shapes (40k) | 2.657 / 0.418 | 0.7703 -> 0.7678 | 0.6459 | 1.0000000000 |
| 3shapes (40k) | 2.394 / 0.094 | 0.7744 -> 0.7756 | 0.9919 | 0.9998278931 |

`figures/pursuit-b-attractor-three-dataset-mechanism.png`. The mechanism
itself generalizes: all three lock onto a dominant eigenvector almost
immediately and stay there, with the same order-of-magnitude Lyapunov
exponent and near-total (MNIST_shapes: machine-precision) to near-total
(3shapes) alignment.

**3shapes is a genuine outlier, not noise.** Its top two eigenvalues are
almost equal (`q=0.9919` vs `0.65-0.80` for the other two), and its
purification is correspondingly incomplete: `1 - alignment` is `~1.7e-4`
for 3shapes versus `~0` at machine precision for the other two.

**Correction: `q` does NOT narrow monotonically with training on either
dataset** -- checked directly rather than assumed. Computed `q` at all
three checkpoints per dataset:

| steps | MNIST_shapes `q` | 3shapes `q` |
|---:|---:|---:|
| 10,000 | 0.6535 | 0.6806 |
| 20,000 | 0.8169 | 0.8052 |
| 40,000 | 0.6459 | 0.9918 |

`figures/pursuit-b-mnist-3shapes-q-evolution.png`. Both rise from 10k to
20k, then diverge: MNIST_shapes falls back to roughly its 10k value,
3shapes climbs to near-degenerate. Meanwhile FG-ARI rises monotonically for
*both* datasets across all three checkpoints (MNIST_shapes 0.43->0.48->0.58;
3shapes 0.58->0.67->0.69). An earlier draft of this note assumed 2shapes'
500-to-10,000-step finding ("`q` narrows progressively with more training",
`PURSUIT-B-MECHANISM.md`) would extrapolate to 3shapes' own trajectory and
described the 40k snapshot as "consistent with needing proportionally more
training" -- that assumption is wrong on the actual data: 3shapes' `q` has
been *widening*, not narrowing, across its own most recent training, while
its ARI kept improving anyway. Whatever is driving continued ARI
improvement past 20k for either dataset, it is not "the spectral gap keeps
narrowing" the way it did for 2shapes' 500-10k window -- that mechanism
does not straightforwardly generalize across datasets or training regimes,
and a single q snapshot cannot stand in for its trajectory.

**This is not a cross-dataset q-vs-ARI predictor either.** MNIST_shapes has
the *tightest* gap of all three at 40k (`0.6459`) yet the *lowest* FG-ARI
(`0.5765`) -- confounded by its intrinsically harder image content (a real
grayscale, antialiased handwritten digit, not a binary shape --
`PURSUIT-B-BEYOND-2SHAPES.md`'s sample-image finding).

## What the learned weights themselves look like

`figures/pursuit-b-mnist-3shapes-k2-structure.png`. Same metrics
`PURSUIT-B-MECHANISM.md` used for 2shapes (`k2_frobenius_relative_change`,
`delta_omega_norm` -- both already computed by `run_pursuit_b.py` into
every run's own `diagnostics`; asymmetry `||K2-K2^T||/||K2||` computed
directly from the checkpoints), across all three datasets:

| dataset | steps | K2 relative change | delta_omega L2 norm | asymmetry |
|---|---:|---:|---:|---:|
| 2shapes | 10,000 | 2.8823 | 1.5679 | 0.9670 |
| 2shapes | 20,000 | 4.4046 | 1.6885 | (checkpoint not retained) |
| MNIST_shapes | 10,000 / 20,000 / 40,000 | 4.4968 / 7.4566 / 11.2813 | 4.0255 / 5.8033 / 6.7986 | 1.1375 / 1.1892 / 1.2273 |
| 3shapes | 10,000 / 20,000 / 40,000 | 2.7662 / 4.4145 / 6.9952 | 9.9406 / 16.4284 / 23.8304 | 1.1195 / 1.1811 / 1.2216 |

Three things stand out:

1. **MNIST_shapes' K2 moves far more than either other dataset's.**
   2shapes and 3shapes track each other almost exactly through 10k->20k
   (2.88->4.40 vs 2.77->4.41), then MNIST_shapes pulls well ahead (reaching
   11.28 by 40k) -- consistent with it needing a larger structural change
   to handle intrinsically harder (grayscale, antialiased) input.
2. **3shapes' `delta_omega` is 3-14x larger than the other two's, and
   growing fastest in absolute terms.** This is a real, novel, and
   deliberately *unresolved* observation: `PURSUIT-B-MECHANISM.md`
   established via an explicit zero-`delta_omega` ablation that
   `delta_omega` is causally irrelevant on 2shapes (zeroing it reproduces
   every one of 500 partitions exactly). Nobody has run that ablation on
   3shapes. Whether 3shapes' much larger `delta_omega` reflects genuine
   causal reliance, or is just a larger incidental value that would zero
   out just as harmlessly, is untested -- posed as `Q_10` rather than
   asserted either way.
3. **Asymmetry tracks nearly identically between MNIST_shapes and
   3shapes** (1.12-1.14 -> 1.22-1.23 for both), both already exceeding
   2shapes' 10k value (0.9670) at the same step count.

### What K2 actually looks like as a matrix

`figures/pursuit-b-mnist-k2-analysis.png`: the raw K2 matrix (init, trained,
difference), a spatial row-reshape, and the eigenvalue spectrum, for
MNIST_shapes' 40,000-step checkpoint.

First version of this figure was wrong in a specific, checkable way: the
row-reshape panels used `viridis` (a monotonic colormap) for genuinely
signed data, while the full-matrix panels correctly used a diverging
colormap (`RdBu_r`) centered at zero. On a monotonic colormap, spatially
smooth but sign-mixed structure can look like noise even when it isn't.
Caught by reproducing the identical row-reshape code on 2shapes' own 10k
checkpoint (which I have locally) and comparing to the established
`k2-evolution.png` story: with the colormap fixed, 2shapes' centre-oscillator
row shows a clear, dominant coupling peak at the trained location, matching
the known-good figure. Doing the *same* fixed-colormap comparison for
MNIST_shapes shows **no dominant peak** -- comparable-or-larger-magnitude
coupling values are scattered elsewhere in the 32x32 grid, not concentrated
near the trained oscillator's own neighborhood. This is now a verified
structural difference, not a plotting artifact: MNIST_shapes' training
does not preserve the "broadens but stays centered" character 2shapes'
K2 has; it moves to something closer to globally dense coupling, consistent
with its much larger Frobenius relative change (11.28 vs 2shapes' 2.88-4.40).

The eigenvalue spectrum (K2 alone, no `i*omega` contribution) shows the
trained top ~100 eigenvalues sitting roughly 5x above their initial values
throughout, not just at the very top -- consistent with a broadly
amplified operator rather than a few isolated modes being boosted.

### Does a dataset's learned K2 transfer to another dataset?

Untested as of this note. `Layer2._evaluate` takes `dataset` and `model`
independently (image size is 32x32=1024 oscillators for all three
datasets, so the dimensions are compatible), which makes a transplant
experiment -- load MNIST_shapes' trained K2/delta_omega, evaluate against
2shapes/3shapes' own val images and readout -- mechanically straightforward.
Not run yet: a naive local (CPU) attempt cost about 3 minutes of wasted
compute (36s/seed on CPU, 10 seeds x 3 dataset pairs would have been
~18 minutes) before being abandoned in favour of running it as a proper
GPU job, matching every other real evaluation in this project.
