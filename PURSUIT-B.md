# Pursuit B: training the oscillator model with the intended phase-separation loss

This note records the successor pursuit that `REPORT.md` left as future work.
It is separate because the original programme measured a model trained under a
defective objective; Pursuit B changes that objective and therefore asks a new
question.

The LabKit record in `.labkit` is authoritative for current standing. This note
carries the sequence of the work and the relationship among the artifacts. If
the note and the record disagree, the record wins.

## Status

The three planned Pursuit B runs are complete in the record's work-state sense:
`TASK_11`, `TASK_12`, and `TASK_13` are all `carried-out` and ungated.

That does **not** mean every branch in the original design was completed:

- `Q_1`, about the fixed model and whether a genuine failure is caused by
  parameters or data, remains accepted as unresolved. Pursuit B trains a model
  under a different loss and does not answer that fixed-model question.
- The parameter sweep (`TASK_3`) remains blocked by `GATE_2`: no measured
  dataset makes the fixed model lose by the prespecified margin.
- The original continuous-exponential trainable-model task (`TASK_5`)
  remains blocked by `GATE_3` because its propagator failed the required
  equality with `M0` at initialization. Pursuit B uses the separately
  verified discrete `M1d` recurrence.
- `M2` (`TASK_6`) remains behind `GATE_4`. Pursuit B does not silently rewrite
  that gate's criteria.
- The only untested question in `labkit now` is a throwaway LabKit-behaviour
  probe (`Q_2`), not benchmark science.

So: all specified Pursuit B experiments are complete, and no planned science is
currently ready to run. The larger design is deliberately stopped at its gates,
not exhaustively completed.

## Question

The earlier trainable `M1d` run minimized a loss whose between-object term was

$$
\frac{|z_o\overline{z}_{o'}|}{|z_o||z_{o'}|}.
$$

For nonzero object order parameters this is identically one, independent of
phase. Training therefore had no term that penalized two objects for sharing a
phase. The near-chance result from that run was a result about the specified
loss, not evidence that a trainable oscillator layer cannot segment the data.

Pursuit B replaces only that term with

$$
\frac{\operatorname{Re}(z_o\overline{z}_{o'})}
{|z_o||z_{o'}| + 10^{-12}}
= \cos(\phi_o-\phi_{o'})
  \frac{|z_o||z_{o'}|}{|z_o||z_{o'}|+10^{-12}},
$$

while retaining the verified discrete recurrence, reference readout, datasets,
and foreground-ARI meter. It asks whether training under the intended phase
separation improves the model on `2shapes`, whether the result persists with a
larger budget, and whether the same corrected-loss procedure improves a model
trained separately from scratch on `MNIST_shapes`.

## Controls before the result

### Constructed loss cases

`loss_m1_cos.py` asserts known answers before any training or data load. The
expected values were derived by hand in `NOTE_49`; two earlier attempts at that
note were retracted because their arithmetic was wrong.

A fresh execution produced:

| constructed case | coded loss |
|---|---:|
| coherent, in phase | $1.11\times10^{-16}$ |
| coherent, antiphase | $-1.999999999998$ |
| coherent, fixed separation | $-0.349611082402$ |
| one half-coherent object | $0.249999999999$ |

The legacy formula returns $1.11\times10^{-16}$ on the antiphase case instead
of $-2$, so the check discriminates the corrected objective from the defective
one.

### Identity at initialization

Before training, the Pursuit B implementation compares its layer-2 orbit with
`07_posn`'s own `run_2layer_torch` result on three validation images and checks
that the reference readout gives byte-identical labels. The CUDA runs' maximum
relative orbit error was $8.40\times10^{-15}$; all three label maps matched.
This makes a trained result comparable with `M0` at initialization.

### Zero-step evaluation control

The same locked evaluation with `--steps 0` produced

$$
0.12085737825541296 \pm 0.04028502026304219,
$$

bit-for-bit equal to the earlier `M0`-at-defaults result. The 500-step increase
therefore does not come from a changed evaluation path. It comes after training
on that path. Evidence: `ART_17` / `EV_28`.

## Primary 2shapes result

At the prespecified 500-step budget, training on images 0–15999 with batch 32,
Adam learning rate $10^{-3}$, and the corrected loss produced

$$
\text{foreground ARI}=0.5896759222267839
\pm 0.0011881617347384494
$$

over ten evaluation seeds on 50 validation images. The connected-components
comparator is 0.16. `CLM_23` is supported and confirmatory under the
pursuit-specific `CRIT_23`.

The comparison with `M0` at defaults, 0.12085737825541296, is diagnostic rather
than the original P5 criterion: P5 names a best swept M0 cell, and the sweep was
correctly never run. The report does not relabel that diagnostic as a criterion
verdict.

## Training-budget characterization

The retained budget points are prefixes of one deterministic default training
trajectory, not eight independent training runs. At each checkpoint the model
was evaluated on the same 50 images and evaluation seeds 1–10. Ten-seed values
were rederived from the committed label artifacts because several run JSONs
contained only a two-seed live summary.

| Adam steps | mean FG ARI | seed SD | seed pairs identical on all 50 images | scored pixels differing across 45 pairs |
|---:|---:|---:|---:|---:|
| 250 | 0.5159 | 0.0009 | 0/45 | 1,936 |
| 500 | 0.5897 | 0.0012 | 0/45 | 4,802 |
| 1,000 | 0.6420 | 0.0014 | 0/45 | 4,023 |
| 2,000 | 0.7048 | 0.0030 | 0/45 | 5,309 |
| 4,000 | 0.8138 | 0.0036 | 0/45 | 2,770 |
| 6,000 | 0.8738 | 0.0005 | 0/45 | 302 |
| 8,000 | 0.8984 | 0.0000 | 45/45 | 0 |
| 10,000 | 0.9195 | 0.0000 | 45/45 | 0 |

Every adjacent rise in aggregate FG ARI exceeds two seed standard deviations
under the recorded comparison. Exact all-image partition agreement changes
between 6,000 and 8,000 steps.

The first interpretation of this curve was too strong. The all-or-nothing pair
count remained 0/45 through 6,000 and made convergence appear absent and then
abrupt. It is a threshold statistic: one mismatching image makes the whole pair
fail. The scored-pixel count and seed SD show an approach, especially the drop
from 2,770 differing pixels at 4,000 steps to 302 at 6,000. The exact pairwise
transition is real; the claim that there was no approach is not. `NOTE_62` and
`CLM_27` correct the earlier `CLM_25` wording.

At 10,000 steps the mean is 0.9195183743129258 and all ten per-seed means are
identical. The direct map comparison is stronger: every one of the 45 seed
pairs has the same partition on all 50 images at 8,000 and 10,000 steps. This
establishes seed-independent evaluation partitions for the ten tested initial
phase seeds on this checkpoint and device. It does not establish why the model
has that property. The proposed dominant-eigenvector mechanism remains an
unverified explanation, and `CLM_24` remains exploratory.

## A second training order

The default batching has no training random seed: without `--shuffle-batches`,
every budget follows the same index-derived trajectory. A second trajectory
using `--shuffle-batches 7` measured:

| Adam steps | mean FG ARI | all-image partition agreement | scored pixels differing |
|---:|---:|---:|---:|
| 6,000 | 0.8601 | 0/45 | 3,587 |
| 8,000 | 0.9099 | 45/45 | 0 |
| 10,000 | 0.9215 | 45/45 | 0 |

Thus both sampled training orders have the all-pair transition in
$(6000,8000]$. Two trajectories do not establish invariance over training
orders. The 8,000- and 10,000-step ARIs are retained only to four decimal
places, and their full remote JSONs were not downloaded before teardown;
`outputs/pursuit-b-shuffle7-summary.json` records that provenance limit.

## Cross-dataset procedure check on MNIST_shapes

`MNIST_shapes` is the clean two-object cross-dataset procedure check because
`n_clusters=2` is valid and the fixed model already has a measured comparator.
This is a separate model trained from scratch on MNIST_shapes, not a transfer
of the 2shapes weights. At 500 steps, the corrected-loss model produced

$$
0.2724268538807503 \pm 0.005783712026447483,
$$

against `M0` at defaults at
$0.13545693221331362\pm0.017840792268879264$. All ten trained per-seed means
are distinct. The margin is 0.1370, compared with the prespecified bar of two
trained-model SDs, 0.0116. `CLM_26` is supported and confirmatory under
`CRIT_25`.

Beating connected components is not the useful test here: its MNIST comparator
is only 0.01552888819008583, while fixed `M0` already exceeds it by a large
margin. Pursuit B was therefore held to improvement over `M0`, not the vacuous
thresholding comparison.

`3shapes` was not trained. Its images require three clusters, while the Pursuit
B training and readout path is intentionally restricted to the two-object
benchmarks. A fixed-model three-cluster baseline now exists, but generalizing
the trainable protocol to three clusters is a new method decision, not a data
file substitution.

## Per-image findings

These are post-primary descriptive analyses, not new criterion verdicts.

The aggregate 2shapes curve is monotone, but only 6 of the 50 per-image
mean-FG-ARI trajectories are nondecreasing across all eight retained budgets.
Individual images can regress even while the aggregate improves. Ground-truth
overlap fraction has a negative Spearman association with per-image score at
every retained budget; the plotted coefficients range from -0.70 to -0.56.
This is an association, not evidence that overlap causes a failure.

At 4,000 steps, seed disagreements are concentrated in three images:
image 6 differs in 43/45 seed pairs, image 43 in 42/45, and image 32 in 9/45.
By 10,000 steps no evaluated image differs across seed pairs. The corresponding
maps, distributions, trajectories, and weight diagnostics are in `figures/`;
`outputs/pursuit-b-figures-manifest.json` hashes their direct inputs, including
`2shapes_val.npz`.

## What the pursuit supports

- Under the corrected cosine phase-separation loss, trained `M1d` beats the
  connected-components comparator on 2shapes at the locked 500-step protocol
  (`CLM_23`, confirmatory).
- At 10,000 steps it reaches a higher score and exact partition agreement over
  the ten tested evaluation seeds (`CLM_24`, exploratory; direct maps add the
  partition result but do not prove its mechanism).
- On MNIST_shapes at 500 steps it improves over the fixed model by the
  prespecified margin (`CLM_26`, confirmatory).
- Aggregate quality continues to improve after tested-seed partitions become
  identical. Convergence to one partition is therefore not evidence that the
  best observed partition has already been reached.

## What it does not support

- A causal account of seed-invariance or a proof of a dominant-eigenvector
  attractor.
- Robustness over arbitrary training orders; only the default and shuffle-7
  orders were sampled.
- Performance on held-out test splits; the locked measurements are validation
  measurements.
- A claim about three-object segmentation.
- The original P5 best-swept-cell comparison, because no M0 sweep exists.
- Automatic satisfaction of legacy `GATE_4` or readiness to start `M2`.
- Resolution of `Q_1`'s fixed-model "parameters or data" clause.

## Reproduction

The known-answer check and the committed curve can be regenerated at current
`main`. `run_pursuit_b.py` also requires the sibling `07_posn` checkout at
commit `18a6064` in the location resolved by `ensure_on_path`:

```bash
uv sync --locked
PYTHONPATH=src uv run --locked python scripts/run_pursuit_b.py --check-only
PYTHONPATH=src uv run --locked python scripts/verification/seed_invariance_curve.py
```

Regenerating every figure additionally requires ignored runtime inputs that the
repository does not distribute: `data/cae/2shapes_val.npz` and
`outputs/ckpt-{500,2000,8000,10000}.pt`. Their expected hashes are in
`outputs/pursuit-b-figures-manifest.json`. With those files present:

```bash
PYTHONPATH=src uv run --locked python scripts/verification/plot_pursuit_b.py
```

The original GPU runs were made across the pursuit's historical commits. The
current runner preserves the arithmetic but has since gained fields and stricter
provenance checks, so byte-identical JSON is not promised. The locked protocol
for a new run is encoded in `scripts/run_pursuit_b.py`; direct partition checks
use `scripts/run_label_identity.py`.

Key committed artifacts:

| artifact | SHA-256 |
|---|---|
| `outputs/pursuit-b-500.json` | `266bd8618006f8e1b0ced195238def84e0ed707f7c3e2c89a6bf0c3f3afb82db` |
| `outputs/pursuit-b-null-control.json` | `d6d0505ccd951c2c5bc4ebd4468fafe0409e57b3edbd25bf3101f905b702bec5` |
| `outputs/pursuit-b-full.json` | `21053e672591f19603362f3b1adac3841cde226174df47b27ca240916fac583a` |
| `outputs/seed-invariance-curve.json` | `7684e2a29d2f8cc689f8c64699b1f605149d604833eca103b1c5fa188d674231` |
| `outputs/pursuit-b-run-mnist-500.json` | `de850bfdbe1b24aaaed95fb89fff8df8e99250e973bb3a9324431cb74c6ec790` |
| `outputs/pursuit-b-shuffle7-summary.json` | `6a8d14728f543a30e4275b0ba25c1c4143a401edd579b97fb96339cc1bb7d80f` |
| `outputs/pursuit-b-figures-manifest.json` | `232f6f3c7026105e6bca78f68b745603ae9642caa2c5de63d84f1c05cd7241c9` |

The primary result entered at `d4a80f7`; the reproducible eight-point curve at
`ea45697`; the second-order bracket at `5f3a80c`; and the per-image figures and
complete figure-input manifest at `d15f605` and `1e672e5`.