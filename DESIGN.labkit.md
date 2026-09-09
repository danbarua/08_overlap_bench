# Overlap bench — the design, as a record

Every heading below names the act that puts that section on the record
and the fields the act takes. Names in `code` — `M0`, `cc`, `P1` — are
this document's, for cross-reference; the record mints its own ids when
the act is recorded, and those are what later acts name.

**This document is the design as written; the record in `./.labkit` is
the book of record.** It was put on the record from this file (36 acts,
every one stamped `reconstructed-from` this document). Where the two
disagree, the record is right and this file is history. `labkit now`
answers what stands; `labkit why <handle>` answers why.

Handles the record minted for the names below: `Q_1` the question;
`LOE_1`/`LOE_2`/`LOE_3` = `M0`/`M1`/`M2`; `NOTE_1`–`NOTE_4` on `Q_1`
(isolation, benchmark and hashes, meter, protocol); `NOTE_5`/`NOTE_6`/
`NOTE_7` the `M0`/`M1`/`M2` locks; `NOTE_8` the probe; `CRIT_1`–`CRIT_3`
= `H0a`–`H0c`; `CRIT_4`–`CRIT_6` = `H1a`–`H1c`; `CRIT_7`/`CRIT_8` =
`L1`/`L2`; `CRIT_9`–`CRIT_12` = `P1`–`P4`; `CRIT_14` = `P6`; `CRIT_15` =
`P5` as amended (`CRIT_13` is the superseded wording); `TASK_1`–`TASK_6`
= `harness-M0`, `arc1a`, `arc1b`, `harness-M1`, `arc1c`, `arc2`;
`GATE_1`–`GATE_4` = `gate-arc1a`, `gate-arc1b`, `gate-arc1c`, `gate-M2`.

Decided on the record since this was written, each with its reason there:

- `arc1a` ran. `P1` and `P2` both failed (`CLM_4`, `CLM_5`): the probe's
  `2shapes` split was a single low seed; the `MNIST_shapes` split
  survives ten seeds. `GATE_2` is blocked and `arc1b` never runs.
- `P5`'s comparator was "`M0`'s best sweep cell", which now never
  exists. Amended to `M0` at defaults (`DEC_1`, citing `CLM_4`).
- `M2` closed as abandoned (`NOTE_15`): `P1` failing removed the premise
  for a third mechanism, not just the gate. Reopening is a new pursuit.
- Two locks the design left open were fixed before their numbers
  existed: `ddof=0` for every seed-SD (`NOTE_11`, `NOTE_12`), and the
  constraints on `M1`'s loss surrogate (`NOTE_13`).

Nothing below this line has been edited to reflect those. The sections
describe what was designed; the record describes what happened.

---

## The question — `pose`

> Does a fixed-weight, hand-tuned coupled-oscillator segmenter
> generalise from the images its parameters were tuned on to a
> benchmark with overlapping objects — and when it fails, is the cause
> the parameters or the data?

One question. It is *untested* until an analysis has been run against
it. The probe that prompted it does not count: it is recorded as an
observation with no analysis over it.

What prompted it, recorded as it was seen — `observe --reconstructed-from`
*"probe against 07 and 06, unmodified, single seed"*:

| Dataset (val, n=50) | Connected components, FG ARI | Fixed dynamics, FG ARI |
|---|---:|---:|
| `2shapes` | 0.160 | 0.062 |
| `MNIST_shapes` | 0.016 | 0.143 |

Same resolution, same code, same default parameters, opposite verdicts.
One `x0` draw per image. The seed-averaged gap is unmeasured, and the
reference implementation's own drift document says foreground ARI is
seed-sensitive on its own bundled images. That is the whole of what is
known, and it is why the question exists rather than an answer to it.

---

## Three lines of enquiry — `pursue <question> --approach`

Three models, one benchmark, one meter. They are three pursuits of one
question because each is a different mechanism, and no later act folds
them into a single "oscillator segmentation" verdict.

### `M0` — the fixed reference dynamics

**Approach.** The reference implementation's pipeline, unmodified:
dense Gaussian coupling, raw discrete recurrence
$x_{t+1} = (K + i\,\mathrm{diag}(\omega))\,x_t$, strict-majority phase
background mask, restart from the original initial state with
background disabled, windowed phase correlations, eigenvectors, real
projection, k-means. Published default parameters. This pursuit asks
only whether what already exists transfers.

### `M1` — the trainable linear propagator

**Approach.** A clean implementation of the exact propagator
$x(t) = \exp\bigl((i\,\mathrm{diag}(\omega) + K)\,t\bigr)\,x(0)$ with
$\omega$ and $K$ trainable, in the lineage of the exact-Fourier ring
propagator and the verified `ExactCVNN` port. No MATLAB indexing
conventions. Gradients reach both $\omega$ and $K$. Same readout as
`M0` so that the readout is not the variable.

### `M2` — the tangent-space rotation layer

**Approach.** Real unit-norm vectors per node, tangent-space projection
of the coupling update, learned rotation — the AKOrN-style mechanism.
Reimplemented from the update equations, not copied. A genuinely
different nonlinearity from the complex-phase family, which is why it
is a third pursuit and not a variant of `M1`. Behind `gate-M2`: it is
not started until `M0` and `M1` both have seed-averaged numbers on the
benchmark, so that a third mechanism is compared against two measured
ones rather than two assumed ones.

---

## What is fixed before anything runs — `note --on`

A note is dated and attributed and constrains nothing by itself. What
makes a lock binding is `L1`/`L2` below, which every Arc 1 analysis is
held to. **A changed lock is a new pursuit**, recorded with `pursue`
and a new approach — never an edit to the note.

### On the question: isolation

New project directory. The reference implementation (`07_posn`, commit
`18a6064`) is read, not modified, and is the oracle for `M0`. That
commit pins the code; the working tree at the time of this note was
not clean (`.gitignore` modified; untracked `datasets/*_ref.mat`,
`docs/DESIGN.labkit.md`, `docs/_archive/`), none of it under `src/`. The
lock is on the committed `src/`, and `H0a` is what checks it. The
exploration log (`06_nb_mnist_phase_encoder`, commit `86576fb`) is read
for exactly three things named below and nothing else; it is archived,
not built on. Disallowed: the Euler-step-plus-normalisation layers,
the coupled phase-field models, the modulo phase encoder (verified to
map distinct images to identical codes), the two incomplete demo
scripts, the classifier that trains against re-randomised labels, and
the demo scaffolding that downloads data at import.

### On the question: the benchmark

`{2shapes,3shapes,MNIST_shapes}_{train,val,test}.npz`, copied once from
the exploration log's `data/CAE/datasets/`. **That directory is
gitignored there, so its commit pins nothing**; the lock is the sha256
of each file, below, and `H0c` checks it. Labels per the upstream
`ComplexAutoEncoder` evaluation: $-1$ overlap, $0$ background, positive
integers per object. Arrays `images` $(n,1,32,32)$ and `labels`
$(n,32,32)$. Splits 50k / 10k / 10k. Every image in the first 50 of
`2shapes_val` and `MNIST_shapes_val` has exactly 2 objects; of
`3shapes_val`, exactly 3. The connected-components baseline scores FG
ARI 0.016–0.34 depending on split, so the benchmark is hard where the
reference implementation's six bundled images are not (CC scores 1.0
there).

| File | sha256 |
|---|---|
| `2shapes_train.npz` | `304b3224e453429e7fef1ee96fcff1c3c3cf138ce0804ca31f6dc2d135006730` |
| `2shapes_val.npz` | `9772425914944b31a07b40fdea967aab08d36d2461f202387eea6690c66fbb59` |
| `2shapes_test.npz` | `e9dbb9faa8acdf068ba9711347d31a44b96789962315031e7e2cc1af448c0693` |
| `3shapes_train.npz` | `f79e21ca86e151f010e51ed4c9ededcbc8f515d03bb2261ef209c9c3b6ea4064` |
| `3shapes_val.npz` | `9baec24f4e4dded964794e52f71a1dc934505533845d275f2d29aa35a6e24984` |
| `3shapes_test.npz` | `4abd3f666c58ec5d18cc313b92bfd492793569065dde20d7a14a51fb69013807` |
| `MNIST_shapes_train.npz` | `b0777f2803df09cf1dadfda9a84e093d8be139e2fcb6be6a4142ce2d180b718e` |
| `MNIST_shapes_val.npz` | `09900916adbd1d5011b285c5a1f3613c5969bb5cb87e76f43e33e545b2474d23` |
| `MNIST_shapes_test.npz` | `1ae2507abaf01254aa65a62e78bc3283881ca310e2b1bb35cc21430a9d4189b7` |

### On the question: the meter

**Foreground ARI**: adjusted Rand index between predicted and true
labels over pixels whose true label is positive. Overlap pixels
($-1$) and background ($0$) are excluded from the comparison. One
meter, fixed here, for every pursuit. Whole-image ARI is reported as a
diagnostic and enters no verdict.

**Read from the probe, not inferred.** The probe's author confirmed
the exact expressions it ran: `foreground = truth > 0` for the mask;
`label(im > 0)` for `cc`, with no `structure` argument, which is
scipy's default 4-connectivity. `H0b` then reproduced 0.160 and 0.016
by computation from those expressions. Both hold; nothing here is
provisional.

### On every pursuit: the protocol

| Quantity | Lock |
|---|---|
| Evaluation split | `val`, first 50 images in file order, per dataset |
| Datasets in Arc 1 | `2shapes`, `MNIST_shapes` |
| Seeds | integers $1,\ldots,10$ inclusive; one $x_0$ draw per (image, seed) |
| RNG | `torch.Generator().manual_seed(seed)` for $x_0$; `numpy.random.default_rng(seed)` elsewhere |
| Reported statistic | mean FG ARI over the 50 images, then mean and standard deviation over the 10 seeds |
| Baseline `cc` | `scipy.ndimage.label(im > 0)`, scipy default 4-connectivity, same 50 images, deterministic. The probe's own expression. |
| Precision | `float64` / `complex128` throughout; no downcasting |

### On `M0`: locks

| Quantity | Lock |
|---|---|
| Code | `07_posn` at `18a6064`: `run_2layer_torch` and `spatiotemporal_segmentation_torch`, called with no argument overridden |
| `alpha` | $(0.5, 0.5)$ |
| `sigma` | $(0.9, 0.0313)$ |
| `nt` | $(60, 200)$ |
| `n_clusters` | $2$, hardcoded — the published default, and the value the probe used on every image |
| `window_size`, `window_step` | $40, 40$ |
| Parameter sweep (Arc 1b only) | `alpha` $\in\{0.25, 0.5, 1.0\}\times\{0.25, 0.5, 1.0\}$; `sigma`$_1\in\{0.45, 0.9, 1.8\}$; `sigma`$_2$ fixed — 27 cells, tuned on `train` images 0–49, reported on `val` |

`n_clusters` is hardcoded so that Arc 1a changes exactly one thing
against the probe: the number of seeds. The probe's code derived it
per image from ground truth (`int(truth[truth > 0].max())`); the value
was $2$ on every image, because every image in both Arc 1 datasets has
exactly 2 objects. Hardcoding matches the probe's value and not its
mechanism, which is a choice made here on its own merits. The
`3shapes` split, where the two would differ, is
out of scope. Cluster count is not the variable.

### On `M1`: locks

| Quantity | Lock |
|---|---|
| Propagator | $\exp\bigl((i\,\mathrm{diag}(\omega)+K)\,t\bigr)$, dense, `torch.linalg.matrix_exp` |
| Initialisation | $K$ = the `M0` Gaussian sheet at `sigma`$_1$, `alpha`$_1$; $\omega$ = image intensity, as `M0` |
| Trainable | $\omega$ and $K$ |
| Loss | soft foreground-ARI surrogate on `train`; the exact surrogate is fixed by `harness-M1` and noted before Arc 1 |
| Training budget | 20 epochs on `train`, batch 32, Adam, lr $10^{-3}$; fixed, no early stopping |
| Readout | `M0`'s, unchanged |

The oracle check that `M1` is `M0` at initialisation, before any
training step, is `H1c`. It is what stops "trained" from meaning
"different code".

### On `M2`: locks

Written when `gate-M2` opens, as a note on `M2`. Not before, because
which comparison `M2` has to beat is what Arc 1 decides.

---

## The conditions results are held to — `criterion`

Stated now, before any number exists, so a check nobody ran still
counts against the finding it qualifies. Each is one sentence a check
can pass or fail. Numbers compare to the written thresholds as written;
no marginal band.

**Harness** — what each harness must show before its Arc 1 work runs:

- `H0a` — two halves, both required. (i) The reference implementation's
  own test suite passes at `18a6064` when run from the harness's
  environment. (ii) The driver, calling the reference unmodified,
  reproduces its foreground `ARI == 1.0` exactly on `2shapes` at seed 1
  and `3shapes` at seed 9 — the hardcoded seeds its demo uses, where
  that score is a property of those seeds and not of the method. The
  natural image has no foreground ARI (region map, whole-image only)
  and is not part of this check. An identity on the environment and
  the driver: the code is unchanged, so a miss is in one of those two.
  The reference masks foreground as `truth != 0`, which on its own
  images equals *positive labels* because they have no overlap class;
  on the benchmark it would not, which is why this record defines the
  meter separately.
- `H0b` — `cc`, computed from the probe's own expressions as noted on
  the meter, gives mean FG ARI over the 50 `val` images within
  $10^{-12}$ absolute of $0.16$ on `2shapes` and of
  $0.01552888819008583$ on `MNIST_shapes`. These are the locked
  values, produced in `float64` by the harness's first run; the probe
  reported them at 3 dp as $0.160$ and $0.016$. The tolerance is for
  last-ulp drift in a 50-term mean across BLAS builds, as `H1a` allows;
  it is not a marginal band.
- `H0c` — every benchmark file's sha256 matches the locked table.
- `H1a` — the `M1` propagator agrees with an independent
  `scipy.linalg.expm` to $10^{-12}$ relative on $N\in\{16, 1024\}$.
- `H1b` — gradients of a scalar loss reach both $\omega$ and $K$
  (finite-difference check to $10^{-6}$).
- `H1c` — `M1` at initialisation, with `M0`'s readout, gives the same
  labels as `M0` on the 50 `2shapes` images under seed 1.

**Locks** — what every Arc 1 analysis is held to:

- `L1` — the analysis was run under the protocol locks exactly as
  noted: 50 images, seeds $1\ldots 10$, none dropped, FG ARI as defined,
  `n_clusters` $=2$, dataset files matching the locked hashes.
- `L2` — the analysis for its pursuit was run under that pursuit's
  locks as noted; for `M0`, no argument overridden; for `M1`, the
  training budget as written.

**Arc 1a — does the probe survive seeds** (`M0`, default parameters):

- `P1` **The split is real.** The seed-averaged FG ARI of `M0` on
  `2shapes` is below `cc`'s by more than two seed-standard-deviations,
  *and* on `MNIST_shapes` is above `cc`'s by more than two. Pass if
  both hold. If only one holds, the probe's headline is wrong and the
  question narrows to the dataset where it held.
- `P2` **Seed noise is not the story.** On each dataset, the seed
  standard deviation of `M0`'s FG ARI is below half the gap between
  `M0` and `cc`.

**Arc 1b — parameters or data** (`M0`, swept):

- `P3` **Parameters explain `2shapes`.** Some cell of the sweep, tuned
  on `train`, gives `M0` a seed-averaged FG ARI on `2shapes` `val`
  above `cc`'s. Pass means the default parameters were the cause. Fail
  means the fixed mechanism does not beat thresholding on `2shapes` at
  any tuning tried, and the cause is in the data.
- `P4` **The tuning does not trade one dataset for the other.** The
  cell that passes `P3` still beats `cc` on `MNIST_shapes`. Evaluated
  only if `P3` passes.

**Arc 1c — trainable linear** (`M1`):

- `P5` **Training helps on the hard one.** `M1` after the training
  budget has seed-averaged FG ARI on `2shapes` above `M0`'s best sweep
  cell by more than two seed-standard-deviations.
- `P6` **Training beats thresholding.** `M1`'s seed-averaged FG ARI on
  `2shapes` is above `cc`'s.

`P5` and `P6` are never folded. `P5` can pass while `P6` fails: training
helps and nothing linear beats thresholding on this data. `P6` is the
claim that matters; `P5` is what says whether the training was real.

---

## What waits on what — `declare --governed-by --protecting --consequence`

A gate is satisfied when every condition governing it has a standing
pass; it holds its work while any is unchecked, and blocks it when one
has failed. Computed from the evaluations, never set.

- `gate-arc1a` — governed by `H0a H0b H0c`; protecting `arc1a`.
  Consequence: *no seed-averaged number is reported until the code is
  shown to be the reference's, the data is shown to be the locked
  files, and the baseline is shown to be the probe's.*
- `gate-arc1b` — governed by `P1 P2`; protecting `arc1b`.
  Consequence: *no parameter sweep is run until the split it would
  explain is shown to survive seed averaging; a sweep to explain a
  single-seed artefact explains nothing.*
- `gate-arc1c` — governed by `H1a H1b H1c`; protecting `arc1c`.
  Consequence: *no trained number is reported until the propagator is
  verified against an oracle and shown to equal `M0` at
  initialisation.*
- `gate-M2` — governed by `L1 L2 P1 P2 P3 P5 P6`; protecting `arc2`.
  Consequence: *the third mechanism is not started until the fixed
  dynamics and the trainable linear model both have seed-averaged
  numbers under the locks, so that it is compared against two
  measurements and not two assumptions.*

`P4` governs no gate. It qualifies the `P3` conclusion and enters no
verdict about `M2`.

---

## The work — `plan --objective --acceptance --may-read --enquiry`

- `harness-M0` — advances `M0`.
  **Objective:** a driver that calls the reference implementation
  unmodified on a CAE `.npz` image and returns FG ARI; the `cc`
  baseline beside it. No science.
  **Acceptance:** `H0a`, `H0b` evaluated and passed.
  **May read:** `07_posn` at `18a6064`; the benchmark note.

- `arc1a` — advances `M0`; behind `gate-arc1a`.
  **Objective:** `M0` default parameters and `cc`, seeds $1\ldots 10$,
  both datasets.
  **Acceptance:** one analysis held to `L1 L2 P1 P2`, reporting mean
  and seed-SD of FG ARI per (model, dataset), and whole-image ARI as a
  diagnostic.
  **May read:** `harness-M0`'s outputs; the protocol note.

- `arc1b` — advances `M0`; behind `gate-arc1b`.
  **Objective:** the 27-cell sweep, tuned on `train`, reported on `val`.
  **Acceptance:** one analysis held to `L1 L2 P3 P4`, reporting the
  full sweep table and naming the best cell before reporting its `val`
  number.
  **May read:** `arc1a`'s outputs; the `M0` locks note.

- `harness-M1` — advances `M1`.
  **Objective:** the propagator, its oracle check, gradient check, and
  the initialisation-equals-`M0` check; fix the loss surrogate and note
  it. No science.
  **Acceptance:** `H1a`, `H1b`, `H1c` evaluated and passed; the surrogate
  noted on `M1` before `arc1c` starts.
  **May read:** `06`'s `port/cv_rnn_exact.py` for the factorisation
  (its docstring is wrong for per-node $\omega$; the code is right);
  `07`'s `RingCVNN`; the `M1` locks note. No other file in `06`.

- `arc1c` — advances `M1`; behind `gate-arc1c`.
  **Objective:** train `M1` under the budget, evaluate on `val`, seeds
  $1\ldots 10$.
  **Acceptance:** one analysis held to `L1 L2 P5 P6`, reporting mean
  and seed-SD, the training curve as an observation, and `M1`'s
  learned $K$ against the initial sheet as a diagnostic.
  **May read:** `harness-M1`'s outputs; `arc1b`'s best cell.

- `arc2` — advances `M2`; behind `gate-M2`.
  **Objective:** the tangent-space rotation layer from its update
  equations, on the same benchmark, same meter, same protocol.
  **Acceptance:** written when `gate-M2` opens, with `M2`'s locks, and
  not before. **May read:** the exploration log's notebook cells 6–8
  for the update equations only; Nadasdy (2009) supplementary for the
  invariance claims.

Out of scope until the three arcs exist, and so not planned: the
`3shapes` split; whole-image or overlap-pixel ARI as a verdict;
learned cluster count; any claim about biological plausibility;
comparison to the upstream `ComplexAutoEncoder` numbers, which were
produced under a different protocol.

---

## What running it puts on the record

The acts, in the order the gates allow them. None has happened.

**Before anything.** `observe` the probe's two rows, above,
`--reconstructed-from` the probe, with no analysis over them. `note`
the benchmark's content hash on the question.

**Arc 0.** `observe` the harness numbers. `analyse --implementing
harness-M0 --held-to H0a H0b` and `conclude` what it found. `evaluate
H0a H0b --gate gate-arc1a --citing` the conclusion. Same for
`harness-M1` against `gate-arc1c`. When `gate-arc1a` is satisfied,
`arc1a` leaves *waiting* and becomes ready.

**Arc 1a.** One `analyse --implementing arc1a --held-to L1 L2 P1 P2`.
Two `conclude`s: whether the split survives, and whether seed noise is
the story. `evaluate P1 P2 --gate gate-arc1b --citing` each. If `P1`
fails on one dataset, `conclude` names which, and `reinterpret` the
probe observation: same evidence, narrower reading — a single-seed
artefact on that dataset. If `P1` fails on both, `close M0
--answered-by` that conclusion: the question as posed has no split to
explain, and `arc1b` stays *blocked* with its reason on the record.

**Arc 1b.** One `analyse --implementing arc1b --held-to L1 L2 P3 P4`.
`conclude` for `P3`: either *the default parameters were the cause* or
*the fixed mechanism does not beat thresholding on `2shapes` at any
tuning tried*. If `P3` passes, `conclude` for `P4`. `evaluate P3
--gate gate-M2 --citing` the conclusion; `evaluate P4` against no gate.

**Arc 1c.** One `analyse --implementing arc1c --held-to L1 L2 P5 P6`.
Two `conclude`s, never one. `evaluate P5 P6 --gate gate-M2`. If `P6`
fails while `P5` passes, `synthesise` **training improves the linear
model and nothing linear beats thresholding on `2shapes`**
`--resting-on` both — that is a finding, and it is the one that would
make `M2` worth running.

**No omnibus claim.** Nothing synthesises across `M0`, `M1`, `M2`. Three
mechanisms, three pursuits, three answers.

**Arc 2.** `gate-M2` satisfied. `note` `M2`'s locks on `M2`. Write
`arc2`'s acceptance with `amend`, citing the Arc 1 conclusions that
decided it.

---

## What the record cannot say, and what stands in

- **The probe is evidence of nothing and is on the record anyway.** As
  an observation, reconstructed, with no analysis. It is there so that
  a reader can see what prompted the question without mistaking it for
  a finding.
- **"Parameters or data" is two conditions, not one answer.** `P3`
  passing says parameters; `P3` failing says data *at the tunings
  tried*. The record cannot say "intrinsic to the data" — no sweep is
  exhaustive — and the `P3` conclusion is worded to say what was tried.
- **Lock values are prose.** A note holds the tables; nothing queries
  `alpha` as a number. What makes them binding is `L1`/`L2`.
- **`M2`'s locks do not exist yet, on purpose.** Writing them now would
  fix a comparison whose target Arc 1 has not yet produced.
- **The loss surrogate is a lock that cannot be written today.** It is
  fixed by `harness-M1` and noted before `arc1c`, which is the least
  violent way to keep "the loss was chosen after seeing `val`" off the
  record: the note's timestamp precedes the analysis.
