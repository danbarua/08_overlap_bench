# Pursuit B beyond 2shapes: MNIST_shapes and 3shapes at 10,000 steps

2shapes was the only dataset `run_pursuit_b.py` had ever been trained on, despite
`MNIST_shapes`/`3shapes` being locked, hash-verified, and sitting in `data/cae/`
the whole time. This note records the first real training runs on both, at the
same 10,000-step budget that made 2shapes seed-invariant and near-saturated.

The LabKit record remains authoritative for research standing; this is an
exploratory mini note.

## What was actually blocking this

Not difficulty avoidance. Two separate, narrow things:

1. **MNIST_shapes: nothing.** `run_pursuit_b.py --dataset MNIST_shapes` already
   worked -- comparators existed (`arc1a`, EV_8: M0 0.1355, cc 0.0155), the code
   path was generic. A single 500-step exploratory run existed
   (`outputs/pursuit-b-run-mnist-500.json`, FG-ARI 0.272) and nobody had pushed
   further.
2. **3shapes: one real blocker.** `run_pursuit_b.py` hardcoded `n_clusters=2` in
   three call sites (inherited from `arc1a`'s own scope note, which excluded
   3shapes because its images hold 3 objects against a 2-cluster readout). The
   underlying `spatiotemporal_segmentation_torch` already took `n_clusters` as a
   parameter -- `outputs/3shapes-baseline.json` (LabKit EV_34) had already
   validated `n_clusters=3` works, it just wasn't wired into the trainable path.
   Fixed by parameterizing `n_clusters` per dataset in `BASELINES`.

Both fixes are in `scripts/run_pursuit_b.py`. Smoke-tested at 5 steps on all
three datasets before spending GPU time: the M0-oracle check (this
reimplementation must equal 07_posn's own layer-1/layer-2 output before any
training happens) passed cleanly on all three, including 3shapes under
`n_clusters=3`.

## Results at 10,000 steps, same metric, same protocol

50 val images, seeds 1-10, mean FG-ARI per seed then mean/std over seeds
(`ddof=0`), M0-oracle passed before training on all three runs.

| dataset | seed-avg FG-ARI | seed SD | M0 at defaults | cc | above M0 by 2 seed-SDs | improvement over max(M0,cc) |
|---|---:|---:|---:|---:|:---:|---:|
| 2shapes | 0.9195 | ~0 | 0.1209 | 0.1600 | yes | 5.7x |
| MNIST_shapes | 0.4343 | 0.0011 | 0.1355 | 0.0155 | yes | 3.2x |
| 3shapes | 0.5755 | 0.0042 | 0.2395 | 0.2356 | yes | 2.4x |

Artifacts: `outputs/pursuit-b-mnist-10000.json` (sha256
`18010dee69fea45ea0f96c410550a4226353d607e0da8dcb532ad6b140eedede`),
`outputs/pursuit-b-3shapes-10000.json` (sha256
`ef6bbf41928fb31b3de7bdabc106e668febba7752d2433ff57166a462f815b56`), both
verified against the job envelope's own hash after download, and checkpoints
`outputs/ckpt-mnist-10000.pt` / `outputs/ckpt-3shapes-10000.pt` (gitignored,
same treatment as all other `.pt` checkpoints; sha256 in the run JSON).

Training clearly beats both comparators on all three datasets at this budget.
It does **not** reach the same absolute level: 2shapes is near-saturated at
10k steps, MNIST_shapes and 3shapes plateau far lower under the identical
budget, optimizer, and protocol. This is a real difficulty difference, not
noise -- seed SDs are small (`<=0.0042`) relative to the gap between datasets.
3shapes clearing its baseline is the more informative result of the two: M0
ties `cc` there (EV_34, gap `0.0039` inside 2 seed-SDs of noise), the same
"nothing to beat" starting point 2shapes had before training -- so 0.5755
is training producing a real result where the fixed-weight mechanism showed
none, not training padding an existing lead.

## What the plots show (`figures/pursuit-b-mnist-3shapes-*.png`)

`pursuit-b-mnist-3shapes-samples.png` shows what the two datasets actually
contain, not just their metadata. **MNIST_shapes is not a binary-shapes-only
task like 2shapes/3shapes**: its "image" channel is a real grayscale,
antialiased handwritten digit overlapping a clean binary square/triangle.
That is a strictly harder segmentation problem than separating two flat
binary regions -- continuous intensity and antialiased edges, not just a
foreground/background split -- and is a plausible large part of why its
ceiling (0.4343) sits below 3shapes' (0.5755) despite 3shapes asking the
harder combinatorial question (3 objects, not 2). 3shapes samples confirm
the object count and show frequent triple-object overlap.

`pursuit-b-mnist-3shapes-loss.png`: neither training loss curve has
plateaued by step 10,000 -- both are still declining steadily on a log-x
axis, unlike what 2shapes' saturated 10k-step ARI implies about its own loss
curve. This is visual evidence (not proof) that more steps would likely
still help both datasets, sharpening the open question below from "untested"
to "worth trying first."

`pursuit-b-mnist-3shapes-per-image.png`: per-image FG-ARI, sorted by the
same overlap-fraction ordering that cleanly separated 2shapes' hard and easy
images. Neither dataset shows that clean monotonic decline here -- low-ARI
images (some near or below the M0 line) are scattered across the sort order
rather than concentrated at the high-overlap end. **Overlap fraction, the
dominant difficulty driver identified for 2shapes, does not obviously
explain per-image difficulty on either of these two datasets.** What does
is an open question; for MNIST_shapes the antialiased-digit content is an
obvious candidate (e.g. digit stroke thickness or overlap-with-thin-strokes),
untested here.

`pursuit-b-three-dataset-comparison.png`: the three-way bar chart, trained
vs M0 vs cc, all at 10,000 steps, same protocol -- the summary view of the
results table above.

## 20,000 steps: the loss curve's prediction confirmed

The declining, unplateaued loss curves above (`pursuit-b-mnist-3shapes-loss.png`)
predicted more steps would help. Ran both to 20,000 steps, matching 2shapes'
own 10k-to-20k extension (`RESULTS-20K.md`):

| dataset | @10k | @20k | change |
|---|---:|---:|---:|
| MNIST_shapes | 0.4343 +/- 0.0011 | 0.4780 +/- 0.0015 | +0.0437 (+10%) |
| 3shapes | 0.5755 +/- 0.0042 | 0.6654 +/- 0.0050 | +0.0899 (+16%) |

Both artifacts hash-verified against their job envelopes:
`outputs/pursuit-b-mnist-20000.json`
(`e7098d094fd59bd7535d5aa0a9e84ec4b25290af496def75ab51aec0ab37fa3a`),
`outputs/pursuit-b-3shapes-20000.json`
(`1dd840f8958fe1ce30a2d17f25038bf5d9a68d629d8af4789171a1b1a2eff04a`).

`pursuit-b-mnist-3shapes-loss-20k.png` shows the loss still declining without
a clear plateau even through 20,000 steps on both datasets -- past the point
where the 10k run stopped (marked). Neither has reached whatever ceiling it
will eventually reach; 3shapes' larger relative gain (+16% vs MNIST_shapes'
+10%) is itself an unexplained difference between the two, not just "more
steps helps everywhere equally." `pursuit-b-mnist-3shapes-10k-vs-20k.png`
is the side-by-side bar chart.

Launched both with `mighty-colab job apply --job-id <id> --async`
(`0.8.1.dev64+g79439178b`, shipped mid-session in response to the keep-alive
failure below) -- spawns the blocking apply as a real detached child and
returns in ~2s with `{job_id, pid, log_path}`. No local process needed to
survive either job's ~45-50 minute run; both completed cleanly on the first
attempt this time.

## 40,000 steps via checkpoint resume: the datasets diverge

Neither loss curve had plateaued at 20k, so extended both further -- this
time via `--resume-from` (optimizer state now saved in the checkpoint),
continuing the *same* optimization trajectory rather than retraining from
scratch. This was also the first real end-to-end test of the resume
mechanism on GPU, not just the CPU smoke test that verified it earlier
(20-then-20-resumed bit-identical to 40-from-scratch, `+0.2870 +/- 0.0021`
both ways).

| dataset | @10k | @20k | @40k | 20k->40k change |
|---|---:|---:|---:|---:|
| MNIST_shapes | 0.4343 | 0.4780 | 0.5765 +/- 0.0007 | +0.0985 (+21%) |
| 3shapes | 0.5755 | 0.6654 | 0.6934 +/- 0.0058 | +0.0280 (+4.2%) |

The two datasets now behave differently, not just at different absolute
levels. MNIST_shapes' relative improvement **accelerated** (10k->20k: +10%;
20k->40k: +21%) -- still far from any visible ceiling. 3shapes'
**decelerated sharply** (10k->20k: +16%; 20k->40k: +4.2%) -- consistent with
approaching a slower-growth regime, though three points on one trajectory
don't prove an asymptote. Neither reaches 2shapes' 0.9195.

Artifacts hash-verified against their job envelopes:
`outputs/pursuit-b-mnist-40000.json`
(`1d787e532cdca58a72be87f1046909c3295dc3cc820af60ce987d50191c4c25e`),
`outputs/pursuit-b-3shapes-40000.json`
(`50e648fc8f54baf5126353e6265fd448c0e2b67a552d7087359931e95e95bd30`). Both
checkpoints carry `optimizer_state_dict` and a 1601-entry `training_curve`
(the full 0->40000 history, not just this run's increment) and are
themselves resumable further.

Hit one real process bug launching these: running `scripts/sync_bundle.sh`
to pick up two new helper scripts while the first pair of `--async` applies
was still staging from the same shared bundle directory raced mighty-colab's
source-lock check (`undeclared source file`, both jobs failed identically).
Not an infra flake -- fixed by leaving the bundle untouched between launch
and completion, then relaunching cleanly.

## What this does not show

- Whether 40,000 steps is enough for either dataset. MNIST_shapes visibly
  is not (still accelerating); 3shapes' deceleration is suggestive of an
  approaching ceiling but not established from three points. Whether the
  gap to 2shapes' 0.9195 eventually closes for either is untested past 40k.
- No mechanism analysis (eigenvalue/spectral-gap picture from
  `PURSUIT-B-MECHANISM.md`) has been run on any of these checkpoints yet.
- What actually drives per-image difficulty on these two datasets, now that
  overlap fraction (2shapes' answer) is shown not to obviously apply.

## A real infra failure hit along the way

Both jobs failed on the first attempt with the assignment disappearing from
the Colab server mid-run (`reason: "the assignment is gone from the server"`,
`cleanup: failed`, forced-teardown errors). Root cause: `mighty-colab job
apply` owns the TFE keep-alive daemon for the life of its own local process;
launching it under a short bash-tool timeout (300s) and letting the local
supervisor die abandons keep-alive for the rest of the run, and the VM gets
reclaimed well before a 10,000-step training run (~25-30 min) finishes. Fixed
by relaunching both with a foreground timeout matching the job's own
`wall_clock` budget so the local supervisor -- and the keep-alive daemon it
owns -- survives to completion. Both retries succeeded cleanly, sessions
confirmed empty afterward (no lingering billing). This is a real gap in
mighty-colab (no keep-alive resume mechanism once the owning process dies);
not fixed here since that project is owned separately.

## Reproduction

```sh
PYTHONPATH=src uv run python scripts/run_pursuit_b.py \
  --dataset MNIST_shapes --device cuda --steps 10000 \
  --out outputs/pursuit-b-mnist-10000.json --save-model outputs/ckpt-mnist-10000.pt

PYTHONPATH=src uv run python scripts/run_pursuit_b.py \
  --dataset 3shapes --device cuda --steps 10000 \
  --out outputs/pursuit-b-3shapes-10000.json --save-model outputs/ckpt-3shapes-10000.pt
```
