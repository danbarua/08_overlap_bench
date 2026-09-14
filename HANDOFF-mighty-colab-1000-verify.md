# Handoff: `overlap-pursuit-b-1000-verify.yaml` — mighty-colab job spec

**From**: mighty-colab session (deps/orchestrator bugfix work)
**To**: labkit-assistant
**Date**: 2026-09-14

## Status: schema-valid, but re-sign before reuse

`mighty-colab job plan overlap-pursuit-b-1000-verify.yaml` passes clean right
now (`spec_hash=1c8bed1f93b0`, no diagnostics) and the spec **has already
been run successfully end-to-end** on a live A100 today (~618s wall clock,
all three artifacts uploaded, `workload: succeeded`, `ok: true`).

That said, "valid" needs one qualifier: **the signed GCS URLs embedded in
the YAML are time-boxed and were signed at 2026-09-14T01:58 UTC** (2h
duration for the `data[]` GET urls, 8h for `artifacts[]`/`control` PUT/GET).
By the time you pick this up they may already be stale. Symptom if they
are: `job apply` fails at `stage` with `staging failed: a declared input
could not be fetched, or failed its sha256 check`. This is not a
mighty-colab bug — signed URLs are inherently single-window by design.

**Before re-running**, re-sign fresh URLs. `scripts/gcs_upload_and_sign.py`
in this directory has the helper functions (`sign_url`, `create_placeholder`)
but its `main()` is hardcoded for two other job names
(`pursuit-b-20000`, `pursuit-b-mechanism`) — for this spec, either adapt
`main()`'s object lists to `pursuit-b-1000-verify` paths, or import the
module and call `sign_url()`/`create_placeholder()` directly (see today's
session transcript for the exact object list: 3 `data[]` GET urls under
`inputs/`, 3 `artifacts[]` PUT urls under `outputs/`, and one
`control/pursuit-b-1000-verify/result.json` PUT+GET pair — the control
object must exist before you can sign a GET for it, hence
`create_placeholder` first).

## Also required before every apply: `scripts/sync_bundle.sh`

New script, added today. `code.entry: run_repo_script.py` in the bundle
expects `bundle/08_overlap_bench/{src,scripts}/` to already contain this
repo's current source — it does **not** clone or fetch this repo at
runtime (it only extracts `07_posn` from a git bundle). The bundle
directory had gone stale/empty before today's run and the job failed at
`run` with a `FileNotFoundError` for `scripts/run_pursuit_b.py`, ~5 minutes
and one full provision+install+verify+stage cycle into a billed A100.

Run this before every `job plan`/`apply`, or whenever `src/`/`scripts/`
change:

```bash
./scripts/sync_bundle.sh
```

It's idempotent (`rsync -a --delete`), and re-running `job plan` afterward
will show a new `spec_hash` even with no source changes if bundle mtimes
shifted — that's expected, not a signal something broke.

## Deps: already fixed in this spec, don't need touching again

The original deps pins (`numpy==1.24.3`, `torch==2.2.2`, etc.) predated
Colab's current Python 3.13 base image and had no working wheels — some
failed outright (`torch==2.2.2` unavailable), others fell back to
building from source and then failed there too (`numpy==1.24.3` needs
`distutils`, removed in Python 3.12+). Already bumped in this file to:

```yaml
deps:
  - numpy==2.1.2
  - scipy==1.14.1
  - scikit-learn==1.5.2
  - torch==2.8.0
  - torchvision==0.23.0
```

These are confirmed working (today's successful run used them). If you're
authoring a *new* spec for this same codebase, copy these pins rather than
whatever older ones might be floating around other `.yaml` files in this
directory (e.g. `overlap-pursuit-b-20000.yaml` may still have stale pins —
not checked/touched in this session).

## What actually ran, for context

Real result from today's run (`pursuit-b-1000-verify.json`,
1000 training steps, 129.1s on `cuda`, `2shapes` dataset, 10 seeds × 50
eval images): `per_seed_mean_foreground_ari` clustered ~0.635 across all
10 seeds, `m0_at_defaults=0.1209` vs `cc_baseline=0.16`,
`p6_above_cc: true`. This was exploratory (per the requester), not a
rigorous claim — the `p6_above_cc`/`diagnostic_exceeds_m0_*` flags were
taken at face value from the artifact, not independently verified against
`run_pursuit_b.py`'s own definitions.

## Checklist for the next run

1. `./scripts/sync_bundle.sh`
2. Re-sign URLs (see above) — data GET 2h is usually enough if you plan
   immediately after signing; PUT/control get more headroom (8h).
3. `mighty-colab job plan overlap-pursuit-b-1000-verify.yaml --out plans/job.json`
4. `mighty-colab job apply plans/job.json`
5. If it fails, **`mighty-colab job destroy <job-id>` immediately** if the
   hint says `VM left running deliberately and is still billing` —
   confirm with `mighty-colab sessions` afterward.
