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

## What this does not show

- Whether 10,000 steps is enough for either dataset, or whether they'd keep
  climbing like 2shapes did to 20,000 (`RESULTS-20K.md`). Not tested here.
- No mechanism analysis (eigenvalue/spectral-gap picture from
  `PURSUIT-B-MECHANISM.md`) has been run on either checkpoint yet.
- No per-image breakdown akin to `pursuit-b-per-image-trajectories.png`
  exists for these datasets -- whether MNIST_shapes/3shapes' lower ceiling is
  a uniform shortfall or concentrated in specific hard images (the way
  2shapes' was, in overlap) is an open question this run doesn't answer.

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
