# Pursuit B: what changed in the trained oscillator dynamics?

This mini note follows the mechanism question prompted by
`figures/pursuit-b-weights.png`: what changed while segmentation improved and
sensitivity to the ten tested initial phase seeds disappeared?

The LabKit record remains authoritative for research standing. This note carries
the mechanism analysis and its reproduction trail; if the two disagree, the
record wins. The analysis does not change the confirmatory Pursuit B result in
`PURSUIT-B.md`.

## Answer in one paragraph

The evidence supports a two-part working explanation, not one generic notion of
“convergence.” Along the checkpoint trajectory, readout-window phases become
increasingly coherent within each object and nearly antiphase between objects.
At 8,000 and 10,000 steps the largest observed spectral modulus ratios are
0.9254 and 0.8883, and tested states reach almost the same projective state at
update 120, the endpoint of the standard readout window. The phase measurements
track partition quality; the spectral and endpoint-alignment measurements track
tested-seed dependence.
The ablations show specifically that the learned frequency offset is not needed
to preserve 10,000-step ARI or tested-seed partition agreement: setting
`delta_omega = 0` reproduces all 500 full-model partitions. With that offset
fixed at zero, the $K_0+\Delta K\,\mathbf{1}[d>4]$ intervention retains most
of the aggregate ARI and all tested-seed agreement. Keeping the trained offset
does not improve either projected-operator summary. This is evidence for a
finite-time projective mechanism, not proof of a unique causal decomposition or
convergence for arbitrary initial states.

## The operator being diagnosed

For image $i$, after layer 1 has selected its foreground, layer 2 applies 140
raw discrete updates

$$
x_{t+1}=A_i x_t,
\qquad
A_i=P_i K_2 P_i
  + i\,\operatorname{diag}\!\left(P_i(I_i+\delta_\omega)\right),
$$

where $P_i$ removes the layer-1 background. Layer 2 restarts from the original
random unit-phase state; it does not inherit layer 1's terminal state. The
reference readout uses phase-only similarity over layer-2 updates 80 through
120 inclusive, then an eigendecomposition, a real projection, and deterministic
K-means.

The audit normalizes each seed state by its positive Euclidean norm after every
update to prevent overflow. In exact arithmetic this changes neither phase nor
projective state.

These reported trajectories are therefore norm-rescaled CPU
`float64`/`complex128` arithmetic, not bit-identical replays of the raw iterates.

Multiplying the initial state by any global phase multiplies every later state
by that phase, so there is no common seed-independent absolute phase. The
spectral diagnostic instead uses projective state: relative complex amplitudes
and phases up to one global complex scalar. The readout separately discards all
amplitudes and retains phase relations.

## First exclusion: the seed effect is not a changing mask

The exact layer-1 path was run for all 50 validation images and seeds 1–10. No
mask pixel differed across seeds. Foreground size ranged from 154 to 192 pixels,
with median 186. The initial complex states were nevertheless different: their
mean pairwise absolute difference was 1.2733 over the 1,024 pixels.

Thus training did not eliminate seed sensitivity by stabilizing the foreground
mask. For this evaluation set, seed dependence enters layer 2 through its
initial complex state.

## The weight image is a useful clue, not a global diagnosis

The lower panels of `pursuit-b-weights.png` show one incoming row of the
$1024\times1024$ matrix: the row for center pixel 528. Each panel also chooses
its own color scale. The upper panels show the complete 1,024-entry
`delta_omega` field. The figure establishes that structured changes occur; by
itself it does not establish the global geometry or which component matters.

The full matrices show that training does much more than rescale the initial
Gaussian sheet:

| Adam steps | $\|K-K_0\|_F/\|K_0\|_F$ | $\|K-K^T\|_F/\|K\|_F$ | $\|\delta_\omega\|_2$ |
|---:|---:|---:|---:|
| 250 | 0.4936 | 0.5308 | 0.4176 |
| 500 | 0.6932 | 0.6649 | 0.6062 |
| 6,000 | 2.1319 | 0.9325 | 1.3268 |
| 8,000 | 2.5508 | 0.9559 | 1.5362 |
| 10,000 | 2.8823 | 0.9670 | 1.5679 |

At 10,000 steps the best scalar fit is $1.1522K_0$, but its residual is
$0.9284\|K\|_F$. The trained matrix is consequently neither a scalar gain nor a
small symmetric perturbation.

Averaging per edge in spatial-distance bins gives a signed broad-scale pattern
at 10,000 steps:

| pixel distance | mean $\Delta K$ per edge | RMS $\Delta K$ per edge |
|---:|---:|---:|
| 0–1.01 | +0.04408 | 0.08538 |
| 1.01–2.01 | +0.03645 | 0.07358 |
| 2.01–4.01 | +0.01637 | 0.06583 |
| 4.01–8.01 | +0.00485 | 0.06312 |
| 8.01–16.01 | −0.01614 | 0.07876 |
| over 16.01 | −0.05000 | 0.08250 |

The bin means avoid the pair-count bias of reporting total energy in a dense
matrix. A negative real matrix entry is a pi-shifted contribution in this raw
complex linear recurrence; it should not be renamed “inhibition” without a
separate dynamical definition.

## Two properties appear together

For every image, the analysis formed the actual masked operator $A_i$. Its
spectral modulus ratio is

$$
q_i=\frac{|\lambda_2(A_i)|}{|\lambda_1(A_i)|}.
$$

Smaller $q_i$ means stronger asymptotic modal separation. Because these
operators are nonnormal, $q_i$ is only a modal diagnostic; direct orbit
agreement is the required finite-time check.
Projective agreement is the unsquared normalized magnitude
$|u^H v|/(\lVert u\rVert_2\lVert v\rVert_2)$. Its table column reports the
minimum over all 50 images and 45 seed pairs at update 120; mean and maximum
$q_i$ are over the 50 images. An unstable image has at least one disagreeing
pair among the 45; the final column counts pairs agreeing on all 50 images.

| steps | mean / max $q_i$ | minimum projective agreement between seed states at update 120 | unstable images | all-image-identical seed pairs |
|---:|---:|---:|---:|---:|
| 500 | 0.8139 / 0.9981 | 0.1052 | 8 | 0/45 |
| 4,000 | 0.7001 / 0.9733 | 0.9949 | 3 | 0/45 |
| 6,000 | 0.6668 / 0.9996 | 0.4565 | 3 | 0/45 |
| 8,000 | 0.6286 / 0.9254 | 0.99999980 | 0 | 45/45 |
| 10,000 | 0.6195 / 0.8883 | 0.99999999997 | 0 | 45/45 |

At 500 steps, each image's number of disagreeing seed pairs out of 45 has
Spearman correlation $+0.632$ with $q_i$. At 8,000 steps, mean alignment with
the leading right eigenvector is 0.9999999996 over the 500 image–seed states at
update 120—the end of the standard readout window—and every seed pair yields
the same partition across all 50 images.

The readout-window orbit also becomes more object-organized. For each seed and
sample, within-object coherence is the mean magnitude of the two ground-truth
object order parameters; between-object cosine compares their mean phases. The
first two columns average these quantities over ten seeds and the 41 samples in
updates 80–120 for each image, then report the mean over 50 images. Foreground
ARI is separately averaged over the 500 final readout partitions.

| steps | within-object phase coherence | between-object cosine | mean foreground ARI |
|---:|---:|---:|---:|
| 500 | 0.7654 | −0.8901 | 0.5897 |
| 4,000 | 0.9090 | −0.9980 | 0.8138 |
| 6,000 | 0.9357 | −0.9971 | 0.8738 |
| 8,000 | 0.9504 | −0.99992 | 0.8984 |
| 10,000 | 0.9601 | −0.99997 | 0.9195 |

Between-object antiphase is already almost saturated by 4,000 steps. The later
aggregate improvement accompanies continued within-object cleanup. These are
checkpoint associations along one deterministic training trajectory, not
independent training replicates.

## Interventions locate the effect in `K2`

The 10,000-step component interventions in this section use all 50 images, seeds
1–10, and the reference readout over updates 80–120. In this table and the
8,000-step readout-window table, seed SD is the population SD (`ddof=0`) across
ten per-seed mean FG-ARIs, each itself averaged over the 50 images.

Here $\Delta K=K_{10000}-K_0$. $S(\Delta K)$ assigns each ordered edge the
mean $\Delta K$ among edges with the same signed row and column displacement.

| 10,000-step intervention | mean FG ARI | seed SD | all-image-identical seed pairs | maps matching full model |
|---|---:|---:|---:|---:|
| full trained `K2`, `delta_omega = 0` | 0.919518 | $1.1\times10^{-16}$ | 45/45 | 500/500 |
| $K_0+S(\Delta K)$, `delta_omega = 0` | 0.909780 | 0 | 45/45 | 110/500 |
| $K_0+S(\Delta K)$, trained `delta_omega` | 0.908833 | $1.1\times10^{-16}$ | 45/45 | 110/500 |
| $K_0+\Delta K\,\mathbf{1}[d>4]$, `delta_omega = 0` | 0.913484 | $1.1\times10^{-16}$ | 45/45 | 160/500 |
| $K_0+\Delta K\,\mathbf{1}[d>4]$, trained `delta_omega` | 0.913484 | $1.1\times10^{-16}$ | 45/45 | 160/500 |
| untrained $K_0$, trained `delta_omega` | 0.150613 | 0.020916 | 0/45 | 34/500 |

Removing `delta_omega` from the full model changes none of the 500 partitions.
Conversely, retaining it with the untrained Gaussian matrix loses both high ARI
and seed agreement. Adding the trained offset to the stationary projection
changes mean ARI from 0.909780 to 0.908833; the far-only summary remains
0.913484. All four projected-operator runs retain 45/45 seed-pair agreement.

Both projected coupling structures start from $K_0$. The stationary variant
replaces every $\Delta K$ entry by its displacement-shared mean; the far-only
variant adds exact $\Delta K$ only for $d>4$ and leaves $d\leq4$ at $K_0$.
Both retain high aggregate ARI, although neither reproduces most individual
full-model partitions. Exact position-specific weights are therefore not
required for those two aggregate properties on this evaluation set; the
interventions do not show that one projected component is the unique mechanism.

## Same operator, different readout times

The 8,000-step operator was evaluated without changing its weights, using four
explicit 41-sample intervals. Every row uses all 50 images and seeds 1–10:

| inclusive layer-2 updates | mean FG ARI | seed SD | unstable images | all-image-identical seed pairs |
|---:|---:|---:|---:|---:|
| 1–41 | 0.898589 | 0.000951 | 7 | 0/45 |
| 20–60 | 0.898543 | 0.000229 | 5 | 4/45 |
| 40–80 | 0.898394 | $1.1\times10^{-16}$ | 0 | 45/45 |
| 80–120 | 0.898394 | $1.1\times10^{-16}$ | 0 | 45/45 |

The readout partitions vary across seeds in the early intervals and become
identical for all 45 seed pairs by updates 40–80, while mean ARI changes by less
than 0.0002. This shows that exact partition agreement can change while
aggregate ARI is nearly fixed; it does not show that projective collapse causes
segmentation quality or that segmentation quality causes collapse.

For the three images with nonzero seed-pair disagreement in the standard
80–120 window—zero-based indices 3, 32, and 34—the 6,000-step operator gave the
following numbers of identical seed pairs out of 45:

| inclusive updates | image 3 | image 32 | image 34 |
|---:|---:|---:|---:|
| 40–80 | 5 | 3 | 11 |
| 80–120 | 13 | 11 | 2 |
| 120–160 | 28 | 7 | 36 |
| 200–240 | 45 | 9 | 45 |
| 360–400 | 45 | 36 | 11 |
| 760–800 | 28 | 45 | 18 |
| 1560–1600 | 18 | 45 | 18 |

Agreement is therefore nonmonotone with window position through update 1,600
on these three images. The audit does not distinguish whether this comes from
near-degenerate rotating modes, nonnormal transients, downstream K-means
boundaries, or a combination; it also does not test precision dependence or
asymptotic behavior. The table supports no conclusion beyond the selected
images and windows.

## Interpretation and limits

The evidence supports this working account:

1. training changes `K2` into a broad, asymmetric operator;
2. readout-window phases become more coherent within objects and nearly
   antiphase between them along the checkpoint trajectory;
3. at 8,000 and 10,000 steps, the largest $q_i$ values are 0.9254 and 0.8883,
   and mean leading-eigenvector alignment at update 120 is 0.9999999996 and
   0.9999999999999 over the corresponding 500 image–seed states;
4. the phase readout removes amplitudes and is invariant to global phase; at
   those checkpoints all tested seeds yield the same partitions.

This does not establish convergence for all initial conditions, all images, or
other training trajectories. It does not turn $q_i$ into a contraction theorem:
the mean eigenbasis condition number is 132 at 8,000 steps and 136 at 10,000,
with maxima 223 and 234. Projective alignment is also not logically sufficient
for label identity: at earlier checkpoints, high alignment and label
disagreement coexist. The audit did not measure downstream decision margins.

## Reproduction

The deterministic audit reuses the exact Pursuit B layer-1 implementation and
the reference readout. It validates recomputed ARIs and partition counts for the
five full-dynamics checkpoints (500, 4,000, 6,000, 8,000, and 10,000) against
`outputs/seed-invariance-curve.json` and verifies the locked `2shapes_val.npz`
hash. All eight local checkpoints below are loaded; the other three contribute
weight geometry and checkpoint-hash provenance:
`outputs/ckpt-{250,500,1000,2000,4000,6000,8000,10000}.pt`.

```sh
PYTHONPATH=src uv run --locked python \
  scripts/verification/analyze_pursuit_b_mechanism.py \
  > /tmp/pursuit-b-mechanism.stdout
shasum -a 256 outputs/pursuit-b-mechanism.json
```

The output is `outputs/pursuit-b-mechanism.json`. Its CPU artifact SHA-256 is
`ed4849c272f9df6d8f14cea6fc1276122a29d7e66e9cb25e9c1034af73d7373a`.
The reproducibility claim is same code, input files, software versions, and CPU
execution; it is not a cross-device bit-identity claim.
