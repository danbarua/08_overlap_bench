# Verification scripts: a debugging trail

These scripts are cleaned-up, runnable versions of ad-hoc checks used while
troubleshooting `harness-M1d` and `arc1c-d` (see this project's LabKit
record: `NOTE_16` through `NOTE_45`, `ART_12`, `CLM_21`). Each demonstrates a
general technique, not just a one-off number. Run any of them directly:

```bash
PYTHONPATH=src uv run --locked python scripts/verification/<script>.py
```

## The narrative, in order

1. **A suspicious result arrived**: the trained `M1d` model scored at chance
   (foreground ARI ≈ -0.010) after training that visibly moved its
   parameters. Was this a bug in the new evaluation pipeline, or a real
   finding about training?

2. **`validate_eval_path_with_oracle.py`** — before trusting the number,
   re-run the *identical* new pipeline with the *old, already-verified*
   parameters (the untrained model). It reproduced `arc1a`'s independently-
   computed baseline exactly. The pipeline was correct; the chance-level
   result was real and needed an explanation, not a fix to the eval code.

3. **`diagnose_model_collapse.py`** — inspected the trained model directly:
   parameter magnitude and dominant eigenvalue grew substantially (2.94 →
   5.07), and the trajectory's magnitude blew up geometrically (to ~1e97,
   still finite in float64). What actually explains a chance-level
   *segmentation* score is the two objects' MEAN PHASES themselves — on
   several trained images they had converged to within a few thousandths
   of a radian of each other (e.g. 0.470 vs 0.470). The "cross-object
   cosine" the script also prints reads ≈1.000000 here too, but — as item 5
   found — that number reads ≈1.000000 unconditionally, whether phases
   actually match or not; the real evidence is the phases themselves.

4. A plausible mechanism ("training discovered the loss surrogate's trivial
   global-synchrony fixed point") was proposed and *checked before being
   recorded as fact*: does the parameter-space gradient actually vanish at
   the trained point, as "found a fixed point, can't leave it" would imply?
   It didn't — the gradient was *larger* there than at initialisation. The
   phase-space argument (within/between both individually maximized at
   exact synchrony) was real, but didn't fully explain the observation.

5. **`check_surrogate_at_baseline.py`** / **`demonstrate_between_term_bug.py`**
   — the check that actually found the root cause: does the *untrained,
   already-correct* baseline model *also* look collapsed under this same
   loss surrogate? It did — the "between" term reported cosine ≈ 1.000000
   even for two objects with mean phases 0.86 radians apart (image 2). That
   is not synchrony; it is an algebraic identity: `|z·conj(z')| = |z|·|z'|`
   for *any* complex `z`, `z'`, so `overlap.abs() / magnitude_product` is
   always ≈ 1, regardless of phase. The surrogate's between-object term
   never measured anything. This was a specification defect (in
   `loss_m1.py`, tracing back to the design note that first wrote the
   formula), not a training pathology.

6. **`check_gradient_conditioning.py`** — a separate thread, hit while
   re-verifying an earlier gradient check: a criterion asked for "1e-6
   relative" finite-difference agreement, but the analytic gradients being
   checked were themselves tiny (~1e-9), which makes a fixed-eps per-entry
   relative check ill-conditioned (its own floating-point-cancellation floor
   is comparable to the quantity being measured). What actually resolved
   it: scaling eps to the parameter's own magnitude (`1e-3*max(|param|,1)`)
   instead of fixing it at 1e-6 — this pushes the loss difference being
   measured far above the floor, and all entries then agree cleanly. A
   directional-derivative check and an eps-sweep are useful DIAGNOSTICS for
   telling "correct gradient, tested imprecisely" apart from "actually
   wrong gradient" (the V-shaped error curve — falling ~100x per decade,
   then rising again — is the signature of the former), but the scaled-eps
   method is what actually closed the check out.

## General lessons, reusable beyond this project

- **Suspicious result from new code?** Re-run the same code with old,
  already-trusted inputs first. If it doesn't reproduce the known-good
  answer, the bug is in the new code, not in whatever you were evaluating.
- **Trained model looks bad under your loss/metric?** Check whether a model
  you *already trust* also looks bad under that same loss/metric, before
  concluding the training procedure or model class is at fault. If yes,
  audit the loss/metric itself first — this is usually faster than trying
  to fix training.
- **A finite-difference gradient check gives an alarming relative error?**
  Check whether the analytic gradient itself is small. If so, a fixed eps
  is probably ill-conditioned; use a directional derivative or an eps-sweep
  before concluding the analytic gradient is wrong.
- **Before writing a causal explanation into a permanent record, check it
  numerically if you can.** Two explanations in this trail ("found a fixed
  point, can't leave it"; "training is a noise-driven random walk") were
  each individually checked, found to be incomplete, and corrected in the
  open rather than left standing — which is what let the actual root cause
  surface a step later instead of being buried under a plausible-sounding
  but wrong story.
