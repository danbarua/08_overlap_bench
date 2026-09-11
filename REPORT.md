# Overlap bench: what happened, and the four explanations that were wrong first

A report on one short research programme. The results are on the record in
`.labkit` and are better there than here — every number has a hash, every
condition was written before the number existed, every correction is dated.
What the record does not give, and what this document is for, is the
*sequence*: what was tried, what looked right, and what killed it.

Read this for the trail. Read `labkit now`, `labkit known` and
`labkit why <handle>` for the state. Handles below are that record's.

The corrected-loss successor named below has since been run. Its results and
their later corrections are kept separately in `PURSUIT-B.md` so this report
remains the trail of the original programme. The LabKit record wins if either
document disagrees with it.

---

## The question

Does a fixed-weight, hand-tuned coupled-oscillator segmenter generalise from
the images its parameters were tuned on to a benchmark with overlapping
objects — and when it fails, is the cause the parameters or the data? (`Q_1`)

It came from a throwaway probe: the reference implementation, published
defaults, 50 validation images, against `scipy.ndimage.label` on the
thresholded image. On `2shapes` the oscillator network *lost* to
thresholding (0.062 vs 0.160); on `MNIST_shapes` it won ninefold
(0.143 vs 0.016). One random initial state per image. That contrast is the
only reason the programme exists, and it is recorded as an observation with
no analysis over it (`NOTE_8`), because it is not evidence of anything.

Three pursuits: the fixed reference dynamics as they are (`M0`), the same
dynamics made trainable (`M1`), and a structurally different nonlinear
layer (`M2`), the last behind a gate so it would be compared against two
measurements rather than two assumptions.

---

## What was found

**One real result.** At ten seeds, `M0` at published defaults is *not*
distinguishable from thresholding on `2shapes` — 0.121 ± 0.040 against
0.160, inside one seed standard deviation — and beats it ninefold on
`MNIST_shapes`, 0.135 ± 0.018 against 0.0155. The probe's `2shapes` failure
was a single low draw; the per-seed spread runs 0.046 to 0.180.
(`CLM_4`, `CLM_5`, synthesised as `CLM_19`.)

**What the meter's numbers mean.** Foreground ARI is 1.0 for a perfect
labelling and 0.0 for chance; measured on this data, a random two-way split
of the foreground scores 0.000 and labelling everything one object scores
0.000. So every number in this report — 0.16, 0.121, 0.135 — sits in the
bottom sixth of the scale. Nothing here segments these images well.

And the aggregate hides a mixture. Decomposing `arc1a`'s own number — all
500 image-seed pairs, per-image, whose per-seed means reproduce `arc1a`
exactly — against whether an image actually contains overlap pixels:

| `2shapes` | images | pairs | `M0` mean | `M0` median | `M0` >0.5 | `cc` |
|---|---:|---:|---:|---:|---:|---:|
| overlap pixels present | 35 | 350 | 0.005 | −0.044 | 4.6% | **0.000** |
| no overlap pixels | 15 | 150 | 0.391 | 0.164 | 36.7% | 0.533 |

Per-image means over the ten seeds: nine images above 0.5 — **none of them
containing overlap** — eight between 0.02 and 0.5, and thirty-three at or
below 0.02, twenty-nine of those overlapping.

Neither aggregate describes uniform mediocrity. `M0`'s 500 pairs have mean
0.121 and median −0.037: most pairs score at or below chance, and 14.2% of
them exceed 0.5. Grouping images by their ten-seed mean gives nine above
0.5 (none containing overlap), eight between 0.02 and 0.5 (six with
overlap), and thirty-three at or below 0.02 (twenty-nine with overlap).

On overlapping images `M0`'s aggregate is near chance — mean 0.005, median
−0.044, 87% of pairs at or below 0.02 — while 4.6% of pairs still exceed
0.5. `cc` scores exactly 0.000 on all 35 of them, every time, because it
merges the two objects into one component. `M0` nonetheless falls *below*
`cc` on 80% of those pairs, because 80% of them are negative.

Pairwise, `M0` exceeds `cc` on 20.0% of pairs in the overlapping stratum
(70/350) and 20.0% in the non-overlapping one (30/150). The stratum means
order differently: 0.005 against 0.000 with overlap, 0.391 against 0.533
without, with 36% of `M0`'s non-overlapping pairs at or below 0.02. Those
are two facts about two different statistics; this programme fixed no
comparison statistic beyond `P1`'s means, so neither fact makes either
method the better one, and the report does not say which is.

`P1` compared the two by their means over images, which is what it said it
would do, and the comparison holds as written. What a mean cannot show is
that both sides of it are mixtures of near-perfect and near-chance images
rather than uniform mediocrity — the fact a successor pursuit would want,
and the reason to state the distribution beside the mean.
(`NOTE_52`–`NOTE_55`; the figures are `ART_15`, produced after closure by
`scripts/verification/stratify_arc1a_by_overlap.py` at commit `ba836f4`,
whose own assertion is that its per-seed means reproduce `arc1a.json` to
10⁻¹²; output `outputs/arc1a-stratified.json`, sha256 `78177c5cc27b634f…`.
`MNIST_shapes` was not stratified. `ART_14` records an earlier run of the
same script and its hash is stale.)

That killed the question as posed. "When it fails, is it the parameters or
the data" has no instance: at ten seeds the fixed mechanism does not fail on
either dataset. The parameter sweep that would have decided between
parameters and data never ran, because the gate protecting it was governed
by the finding that the split survives seed averaging, and it did not
(`GATE_2` blocked). `Q_1` is *accepted as unresolved* with two written
reopening conditions (`DEC_9`) — not answered, not abandoned.

**One result that turned out to be about our own code.** The trainable
model, verified identical to the reference at initialisation to zero error,
trained 500 steps, scored −0.010 ± 0.0001 on `2shapes`: chance. The reason
is that the loss we specified contained no term that did what we thought it
did. More below; it is the whole story of this programme.

---

## The trail: four explanations, in the order they were believed

### 0. Is the number even real?

The chance-level score arrived from a new evaluation pipeline. Before
attributing it to training, the implementor ran the *identical* pipeline
with the *untrained* parameters and reproduced `arc1a`'s ten independently
computed per-seed numbers exactly — 0.0828, 0.1208, 0.1802, and so on.
The pipeline was correct; the result needed an explanation, not a fix.
(`scripts/verification/validate_eval_path_with_oracle.py`.)

This is the step most likely to be skipped and the cheapest to run.

### 1. "Amplitude ran away and washed out the image."

Inspecting the trained model: the dominant eigenvalue of the recurrence had
grown from 2.94 to 5.07, and trajectory magnitudes reached ~10⁹⁷ — still
finite in float64, so no overflow check would have fired. The loss operates
on unit phases only, so it constrains nothing about amplitude.

**True, and the correct account of the dynamics** (`ART_12`). Incomplete as
a cause: it says how the model got somewhere degenerate, not why nothing in
the objective pushed back.

### 2. "Training found the loss's trivial fixed point and cannot leave it."

Measured directly on three validation images: every object internally
coherent (|z| ≈ 1.0) *and* the two objects' mean phases identical to six
decimals. Within = 1, between = 1, loss = 0 exactly — a genuine global
minimum of the loss as written, distinct from the intended optimum.
(`NOTE_28`.)

**Killed by its own author, before it hardened.** The claim implies the
parameter gradient vanishes there. Measured: ‖∇K₂‖ = 1.4 × 10⁻³ at the
trained point, *larger* than 1.0 × 10⁻⁶ at initialisation. The phase-space
argument was right; the parameter-space conclusion did not follow, because
140 steps of a matrix with |λ| ≈ 5 amplify a near-zero phase gradient.
(`NOTE_31`.) The mechanism survived; "cannot leave it" did not.

### 3. "Then it is noise-driven drift."

Proposed and checked; also incomplete. (`NOTE_38`.)

### 4. The actual cause: the between-object term was identically 1.

The check that found it inverted the question: *does the model we already
trust also look collapsed under this loss?* It did. The cross-object term
read 1.000000 on an image whose two objects were 0.86 radians apart, where
the correct cosine is 0.650.

Because $|z\,\overline{z'}| = |z||z'|$ for any complex pair. The specified
term was

$$\frac{|z_o \overline{z_{o'}}|}{|z_o||z_{o'}|} \equiv 1$$

independent of phase. So the loss minimised for 500 steps was
$1 - \text{within}$: it rewarded each object's internal coherence and
contained nothing whatsoever against different objects sharing a phase.
Gradient descent, working correctly, drove every object to one phase —
the most direct way to maximise every object's own coherence at once — and
amplitude growth (explanation 1) is the mechanism it used to get there.
(`NOTE_40`, `NOTE_41`; `scripts/verification/demonstrate_between_term_bug.py`.)

The intended quantity was
$\mathrm{Re}(z_o \overline{z_{o'}})/(|z_o||z_{o'}|) = \cos(\phi_o - \phi_{o'})$.
It has never been trained.

**So the M1 numbers are evidence that this programme never tested a
discriminating objective — not evidence that the model cannot learn under
one** (`CLM_21`). The numbers themselves stand; what they are evidence *of*
changed after closure.

---

## Why it took four tries, and the check that would have taken seconds

Three checks existed and none of them could catch it:

| check | what it verified | why it passed |
|---|---|---|
| `NOTE_13` | the surrogate's *form* and provenance — differentiable, no clustering, train-only | it constrained the shape of the formula, not its algebra |
| `H1b-d` | gradients reach both parameters, finite-difference verified | autograd differentiates whatever is coded, correctly, bug included |
| `arc1c-d` | the trained model's score under the meter | it measured the outcome, not whether the objective meant anything |

The missing check is one line of arithmetic: **evaluate the coded loss on
constructed inputs and compare against hand-derived values.** Two objects
perfectly coherent and in phase → 0. Antiphase → −2. Half-coherent object
against a coherent one → 0.25. The antiphase case alone fails the specified
formula, which returns 0 there.

It is now on the record as a prespecified criterion for any successor
pursuit (`NOTE_49`), and it took three attempts to write correctly — the
first two had the subtraction backwards and the wrong expected values,
which is the same defect class it exists to prevent. The third derives each
case by hand and shows the arithmetic, precisely so a reader can check it
without running anything; executing it afterwards changed only the
tolerance, from 10⁻¹² to 10⁻⁹, because the denominator guard sits at the
same scale as the tighter bound (`NOTE_50`). Derive first, then execute: a
known-answer test whose expected values came out of the code it tests is
not a control.

That is the transferable lesson of this programme, and it is not about
oscillators: **a loss is code, and code that has never been run against a
known answer is not verified.**

---

## What was recorded, and by whom

150 acts. 36 transcribed from the design document, 114 performed. Two
agents: one owning the design (questions, criteria, gates, locks), one
implementing and measuring.

Six defects in the design's own locks were found, four of them *before*
anything ran:

1. `H1a`'s oracle named a recorded ARI value the reference does not record —
   it asserts equality with 1.0 at two hardcoded seeds.
2. `M1`'s propagator was specified as $e^{At}$ with `A` the reference's
   *discrete* step matrix. $e^A \neq A$, so "identical at initialisation"
   was unpassable as written — and with |λ| ≈ 424, it overflowed by step 2.
3. The same lock missed that the reference runs *two* layers: layer 1 exists
   only to vote a background mask, and layer 2 — 29× gentler — produces the
   only trajectory the readout reads.
4. A single shared trainable frequency vector cannot also be each image's
   own intensity; caught before the training loop was written.
5. `H1b-d` demanded 10⁻⁶ *relative* agreement from a finite difference at
   ε = 10⁻⁶ on gradients of order 10⁻⁹ — below the method's own
   cancellation floor. The check was unmeasurable, not the gradient wrong.
6. The between-object term, above.

Defects 1–4 were caught by the implementor reading the lock before running
it. Defect 5 was caught by the implementor rechecking a criterion it had
already been recorded as passing — the recorded value was an *absolute*
error compared against a *relative* bar. Defect 6 was caught after the
programme had been closed and synthesised, by asking whether a model
already trusted also looked broken.

Every correction is on the record in the order it was found, with the
superseded reading left readable beside it. Nothing was silently rewritten.

---

## What came next, and what remains

The intended corrected-loss experiment was run as a separate successor
pursuit. See `PURSUIT-B.md` for its protocol, results, correction trail and
limits; importing those results here would erase the boundary between the
original failed objective and its replacement.

Two branches remain intentionally unrun:

- **The parameter sweep** remains blocked because no measured dataset makes
  the fixed model lose by the prespecified margin. There is still no failure
  instance for the sweep to explain (`GATE_2`).
- **`M2`** remains behind the legacy `GATE_4`. Pursuit B supplies a positive
  trainable-M1d result, but the gate's literal criteria refer to the original
  failed-loss work. Whether to replace that gate is a new design decision, not
  a conclusion licensed by the completed runs.

---

## Reproduction

```bash
uv sync --locked
PYTHONPATH=src uv run --locked python -m overlap_bench.run_arc1a        # the real result
PYTHONPATH=src uv run --locked python scripts/verification/demonstrate_between_term_bug.py
labkit now --db .        # what stands
labkit why Q_1 --db .    # and why
```

`scripts/verification/README.md` is the debugging trail as runnable
technique, written by the agent that walked it. Code at `f92ca38`.
