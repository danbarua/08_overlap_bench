Colab A100 task, from labkit-omp-claude. Dan has 100+ A100 hours expiring and wants a yes/no fast. Everything below is committed and smoke-tested on CPU.

## Before you start

1. Your pane is in a worktree (`~/.omp/wt/labkit-assistant-5cbb478`). The commits named below are in the main checkout at `~/Code/AI/08_overlap_bench`. Work there, or fetch from it; confirm `git rev-parse --short HEAD` is `3e8e4c1` or a later commit on `main` before running anything.
2. Register on agent-bus as `labkit-assistant` (you are not registered yet), then message `labkit-omp-claude` when you have a result or if you are blocked.

## What this is

Train a trainable oscillator segmenter under a corrected loss and report one number: foreground ARI on `2shapes` val. The previous run of this experiment scored at chance because the loss's between-object term was algebraically identically 1 (`|z conj z'| = |z||z'|` for any complex pair). This is the corrected term, `Re(z conj z')/(|z||z'|)` = cos of the phase difference.

## Repos — two, both needed

    ~/Code/AI/08_overlap_bench at 3e8e4c1   (the experiment)
    ~/Code/AI/07_posn         at 18a6064   (the reference model, imported via ensure_on_path)

The script imports `gaussian_sheet_torch` and `spatiotemporal_segmentation_torch` from `07_posn`. It is **not** standalone.

## Data — gitignored upstream

Copy into the directory `paths.CAE_DIR` names, which is `<repo root>/data/cae` — not `data/CAE/datasets/`:

    2shapes_train.npz  sha256 304b3224e453429e7fef1ee96fcff1c3c3cf138ce0804ca31f6dc2d135006730
    2shapes_val.npz    sha256 9772425914944b31a07b40fdea967aab08d36d2461f202387eea6690c66fbb59

`run_pursuit_b.py` calls `verify_locked_dataset_hashes()` with no argument, so it checks all nine locked `.npz` files, not just the `2shapes` pair. All nine must be present, and the script refuses to run on any mismatch.

## Run, in this order

    # 1. seconds, no data, no GPU. Four constructed loss cases whose expected
    #    values were derived by hand (recorded as NOTE_49), plus a control that
    #    the old broken formula still returns 0 on antiphase.
    PYTHONPATH=src uv run --locked python scripts/run_pursuit_b.py --check-only

    # 2. about 20 seconds. End to end on 32 images, 1 step, 1x1 eval. Prints
    #    "smoke only - no P6 verdict" by design.
    PYTHONPATH=src uv run --locked python scripts/run_pursuit_b.py \
        --train-images 32 --steps 1 --eval-images 1 --eval-seeds 1 --out /tmp/smoke.json

    # 3. THE RUN. Prespecified budget: 500 Adam steps, batch 32, lr 1e-3, train
    #    images 0..15999. Locked eval: 50 val images x seeds 1..10.
    PYTHONPATH=src uv run --locked python scripts/run_pursuit_b.py \
        --device cuda --steps 500 --out outputs/pursuit-b-500.json

    # 4. Only if 3 is promising. 20 epochs over the 16k subset = 10,000 steps.
    PYTHONPATH=src uv run --locked python scripts/run_pursuit_b.py \
        --device cuda --epochs 20 --out outputs/pursuit-b-full.json

## Compute, so you can sanity-check the wall clock

One Adam step is 140 recurrence steps of `(1024x1024)@(1024x32)` complex128, roughly 113 GFLOP forward+backward. 500 steps is about 57 TFLOP; 10,000 steps about 1.1 PFLOP. A100 fp64 peak is 9.7 TFLOPS, and these matmuls are only 32 wide, so expect low utilisation: minutes for step 3, tens of minutes to about an hour for step 4. If step 4 runs for many hours, something is wrong — stop and say so rather than burning credit.

`float64`/`complex128` is locked and must not change. The A100 is the one card where that is affordable. Do not switch to `complex64` to go faster: it would be a different model, and the at-init identity check below would then fail, correctly.

## Built-in checks

You do not need to add these. The script aborts before training if any fails:

- the four known-answer loss cases (step 1)
- the old-formula control
- dataset sha256s
- the at-init oracle: on 3 val images, this reimplementation's layer-2 orbit must match `07_posn`'s own `run_2layer_torch` to 1e-12 relative on unmasked pixels **and** give byte-identical labels through the reference readout. That is what makes the number comparable to the existing result. Verified on CPU: max relative error 5.2e-15, labels identical.

## Report back

The JSON from step 3 (and 4 if run), plus wall clock per phase. The three fields that matter are `seed_averaged_foreground_ari`, `p6_above_cc`, and `diagnostic_above_m0_at_defaults`. Comparators already on the record: M0 at published defaults 0.1209 ± 0.0403, connected components 0.16.

On honesty: `p6_above_cc` is criterion P6 as written. The M0 comparison is a **diagnostic, not P5**. P5 requires beating M0's best swept cell, and that cell does not exist — the sweep was gated off and never ran. Do not report the M0 comparison as a criterion verdict.

Do not fix `loss_m1.py` (the broken one). It is cited evidence for the previous result and must stay exactly as it is. The corrected term is a separate module, `loss_m1_cos.py`.

One CPU data point for comparison, not a verdict: 1 Adam step on 32 images, then the full 50×10 eval, gave 0.1809 ± 0.0253. That is one step on a 32-image subset, and it is above both comparators — which is why this is worth an A100 rather than another 40 minutes of laptop.
