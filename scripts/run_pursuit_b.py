"""Pursuit (b): train M1d under the loss NOTE_16 intended, and answer whether
a discriminating objective does anything at all on this benchmark.

This is a NEW pursuit, not a re-run of arc1c-d. arc1c-d's numbers stand; they
are evidence about a loss whose between-object term was identically 1
(NOTE_40/NOTE_41). This script trains the same model under
`loss_m1_cos.cos_coherence_loss`, which is the term that was meant.

Order of operations, and the order matters:

  1. `assert_known_answers()` — NOTE_49's four constructed cases, expected
     values derived by hand in that note, tolerance from NOTE_50. Runs in
     milliseconds, before a single byte of data is touched. If the loss does
     not compute what the note says, the script exits non-zero and no GPU time
     is spent. This is the check whose absence cost the previous pursuit its
     entire result.
  2. `assert_old_formula_fails_case_two()` — the control on the control: the
     old modulus form must still return 0 for an antiphase pair. If it does
     not, `loss_m1` has been edited (it must not be) or the construction no
     longer builds antiphase, and (1) is not testing what it claims.
  3. Precompute M0's layer-1 output per training image: the background mask and
     the masked restart state. These are M0's, fixed, not trainable — layer 1
     is a mask-maker whose trajectory nothing reads (NOTE_20). Cached, because
     recomputing 60 dense steps per image per epoch is the dominant cost
     otherwise.
  4. Train layer 2 only: K_2's unmasked block and the shared delta_omega
     (NOTE_23). Adam, lr 1e-3, batch 32.
  5. Evaluate on val seeds 1..10 through M0's own readout, scoring foreground
     ARI exactly as the protocol note locks it, and report against the two
     numbers already on the record: M0 at defaults 0.1209 +/- 0.0403, and
     connected components 0.16.

Nothing here writes to the LabKit record. It emits JSON; whoever runs it
records the observation with its hash.

    # the check alone, no data, no GPU:
    PYTHONPATH=src uv run --locked python scripts/run_pursuit_b.py --check-only

    # a short run, minutes on an A100:
    PYTHONPATH=src uv run --locked python scripts/run_pursuit_b.py \
        --device cuda --steps 500 --out outputs/pursuit-b-500.json

    # the budget NOTE_6 originally named (20 epochs x 50k images):
    PYTHONPATH=src uv run --locked python scripts/run_pursuit_b.py \
        --device cuda --epochs 20 --out outputs/pursuit-b-full.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from overlap_bench.dataset_hashes import CAE_DIR, verify_locked_dataset_hashes
from overlap_bench.harness_m0 import foreground_ari
from overlap_bench.loss_m1_cos import (
    assert_known_answers,
    assert_old_formula_fails_case_two,
    cos_coherence_loss,
)
from overlap_bench.paths import ROOT_DIR
from overlap_bench.reference_repo import ensure_on_path

DATASET = "2shapes"  # default; --dataset selects, BASELINES gates what is allowed
LAYER1_STEPS = 60
LAYER2_STEPS = 140
BATCH = 32
LR = 1e-3
EVAL_IMAGES = 50
EVAL_SEEDS = tuple(range(1, 11))
TRAIN_SEED_OFFSET = 100_000  # NOTE_21: disjoint from the eval seeds

# Comparators are per-dataset and must travel with the dataset. Leaving them as
# 2shapes globals while parameterising the data would compute every non-2shapes
# verdict against the wrong baseline, and it would do it silently.
#
# m0_seed_sd is M0's own spread (ddof=0), not the trained model's. Both appear
# in the output: at a budget where the trained model's per-seed values collapse,
# two of ITS SDs is ~0 and that test degenerates into the plain above-M0
# comparison -- which is what happened to pursuit-b at 10,000 steps.
BASELINES: dict[str, dict[str, float]] = {
    "2shapes": {
        "m0_at_defaults": 0.12085737825541296,  # arc1a, ART_6
        "m0_seed_sd": 0.0403,  # arc1a, as reported on the record
        "cc": 0.16,  # H0b's locked value
    },
    "MNIST_shapes": {
        "m0_at_defaults": 0.13545693221331362,  # arc1a, EV_8, commit 31d80d3
        "m0_seed_sd": 0.017840792268879264,  # EV_8, ddof=0
        "cc": 0.01552888819008583,  # EV_8
    },
    # 3shapes is deliberately absent. It has no cc/M0 baseline in arc1a at all,
    # and its images hold three objects against a readout with n_clusters=2
    # hardcoded (here and in harness_m1d), which arc1a's own scope note put out
    # of scope. Two separate blockers; adding a row here would hide both.
}


def _progress(message: str) -> None:
    print(f"[pursuit-b] {message}", file=sys.stderr, flush=True)


def _sheets(nrow: int, ncol: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """M0's two Gaussian sheets, from 07_posn itself, at the locked parameters."""
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch

    k1 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9, dtype=torch.float64).real.clone()
    k2 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.0313, dtype=torch.float64).real.clone()
    return k1.to(device), k2.to(device)


def _layer1_masks(
    images: torch.Tensor, k1: torch.Tensor, seeds: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """M0's layer 1, batched: 59 steps under K_1, then the majority-phase mask.

    Returns (mask (B,N) bool, x0 (B,N) complex). Exactly M0's construction
    (cv_rnn_segmentation.py:143-155): iterate, take angles, threshold at the
    mean, keep whichever side has more pixels.
    """
    b, n = images.shape
    omega = images
    x0 = torch.empty((b, n), dtype=torch.complex128, device=device)
    for i in range(b):
        generator = torch.Generator(device="cpu").manual_seed(int(seeds[i]))
        phases = torch.rand(n, dtype=torch.float64, generator=generator)
        x0[i] = torch.exp(1j * (phases - 0.5) * (2 * torch.pi)).to(device)

    x = x0
    k1c = k1.to(torch.complex128).T
    omega_c = omega.to(torch.complex128)
    for _ in range(1, LAYER1_STEPS):
        x = x @ k1c + 1j * omega_c * x
    if not torch.isfinite(x).all():
        raise FloatingPointError("layer 1 overflowed; float64 is required here")

    angles = torch.angle(x)
    threshold = angles.mean(dim=1, keepdim=True)
    above = angles > threshold
    below = angles < threshold
    mask = torch.where(
        (above.sum(1, keepdim=True) > below.sum(1, keepdim=True)).expand_as(above), above, below
    )
    return mask, x0


class Layer2(torch.nn.Module):
    """M0's layer 2 with K_2 and a shared additive delta_omega trainable.

    NOTE_20: layer 2 is the only trajectory the readout reads. NOTE_23:
    omega_2(i) is image i's own intensity plus one shared delta_omega,
    initialised to zero, so that at initialisation this is M0 exactly.
    """

    def __init__(self, k2_init: torch.Tensor, n: int):
        super().__init__()
        self.K2 = torch.nn.Parameter(k2_init.clone())
        self.delta_omega = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64, device=k2_init.device))

    def _run(
        self, x0: torch.Tensor, omega: torch.Tensor, mask: torch.Tensor, keep_all: bool
    ) -> torch.Tensor:
        keep = (~mask).to(torch.float64)
        keep_c = keep.to(torch.complex128)
        omega2 = ((omega + self.delta_omega) * keep).to(torch.complex128)
        x = x0 * keep_c
        k2t = self.K2.to(torch.complex128).T
        states: list[torch.Tensor] = []
        for _ in range(LAYER2_STEPS):
            x = (x @ k2t) * keep_c + 1j * omega2 * x
            if keep_all:
                states.append(x)
        return torch.stack(states, dim=-1) if keep_all else x

    def orbit_final(
        self, x0: torch.Tensor, omega: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """x_199, batched. mask (B,N) bool; masked pixels contribute nothing."""
        return self._run(x0, omega, mask, keep_all=False)

    def orbit_all(
        self, x0: torch.Tensor, omega: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """The whole layer-2 orbit, (B, N, 140), for M0's readout."""
        return self._run(x0, omega, mask, keep_all=True)


def assert_matches_m0_at_init(
    k2: torch.Tensor, device: torch.device, n_images: int = 3, dataset: str = DATASET
) -> dict:
    """The oracle. Before training: this reimplementation must BE M0.

    Checks, per image, against 07's own `run_2layer_torch` output: the
    layer-2 orbit agrees to 1e-12 relative on unmasked pixels, and M0's
    readout gives byte-identical labels. This is CRIT_19's content, run here
    because a batched GPU reimplementation of layer 1 and layer 2 is new code
    and a preprocessing divergence would masquerade as an improvement.

    Raises AssertionError before any training happens. Seconds, not minutes.
    """
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import (
        run_2layer_torch,
        spatiotemporal_segmentation_torch,
    )

    with np.load(CAE_DIR / f"{dataset}_val.npz") as data:
        images_np = np.asarray(data["images"][:n_images, 0], dtype=np.float64)

    model = Layer2(k2, k2.shape[0]).to(device)
    results = []
    for i in range(n_images):
        image = torch.from_numpy(images_np[i])
        generator = torch.Generator().manual_seed(1)
        states_m0, mask = run_2layer_torch(image, generator=generator)
        x0 = states_m0[:, 0].masked_fill(mask, 0)
        omega = image.T.reshape(-1)
        with torch.no_grad():
            orbit = model.orbit_all(
                x0.to(device).unsqueeze(0),
                omega.to(device).unsqueeze(0),
                mask.to(device).unsqueeze(0),
            )[0].cpu()

        reference = states_m0[:, LAYER1_STEPS:200]
        unmasked = ~mask
        scale = reference[unmasked].abs().max().item()
        relative = (orbit[unmasked] - reference[unmasked]).abs().max().item() / scale

        states = states_m0.clone()
        states[:, LAYER1_STEPS:200] = orbit
        states[mask, LAYER1_STEPS:200] = torch.nan
        mine, *_ = spatiotemporal_segmentation_torch(
            states, image, mask, nt_mask=LAYER1_STEPS, n_clusters=2
        )
        theirs, *_ = spatiotemporal_segmentation_torch(
            states_m0, image, mask, nt_mask=LAYER1_STEPS, n_clusters=2
        )
        identical = bool(torch.equal(mine, theirs))
        results.append({"image": i, "orbit_relative_error": relative, "labels_identical": identical})
        assert relative < 1e-12, (
            f"image {i}: layer-2 orbit differs from M0 by {relative:.3e} relative at "
            "initialisation. This model is not M0, so nothing it scores is comparable "
            "to arc1a. Do not train."
        )
        assert identical, (
            f"image {i}: orbit matches M0 but the readout gives different labels. "
            "Do not train."
        )
    _progress(
        "oracle passed: at init this is M0 "
        f"(max orbit rel err {max(r['orbit_relative_error'] for r in results):.2e}, "
        f"labels identical on all {n_images})"
    )
    return {"n_images": n_images, "per_image": results}

def _train(
    model: Layer2,
    images: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    x0s: torch.Tensor,
    steps: int,
    device: torch.device,
) -> list[dict]:
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    n_train = images.shape[0]
    curve = []
    started = time.monotonic()
    for step in range(steps):
        lo = (step * BATCH) % n_train
        idx = torch.arange(lo, lo + BATCH, device=device) % n_train
        optimiser.zero_grad()
        x_T = model.orbit_final(x0s[idx], images[idx], masks[idx])
        losses = [cos_coherence_loss(x_T[b], labels[idx[b]]) for b in range(BATCH)]
        loss = torch.stack(losses).mean()
        if step == 0:
            # NOTE_21: the t=0 loss on the first batch, before any update.
            curve.append({"step": 0, "loss": float(loss), "before_first_update": True})
        loss.backward()
        optimiser.step()
        if (step + 1) % 25 == 0 or step + 1 == steps:
            # Timing goes to stderr and to the sidecar, never into the artefact:
            # run_arc1a.py's rule is that a re-run with the same code and data
            # must hash identically, and wall-clock makes that impossible.
            elapsed = time.monotonic() - started
            curve.append({"step": step + 1, "loss": float(loss)})
            _progress(
                f"step {step + 1}/{steps} loss {float(loss):+.6f} "
                f"({elapsed / (step + 1):.3f}s/step)"
            )
    return curve


def _evaluate(
    model: Layer2,
    device: torch.device,
    n_images: int = EVAL_IMAGES,
    n_seeds: int = len(EVAL_SEEDS),
    dataset: str = DATASET,
) -> dict:
    """Val through M0's own readout, foreground ARI as the protocol note locks it.

    `n_images`/`n_seeds` exist to smoke-test the path in seconds; a real run
    uses the locked 50 images and seeds 1..10, which are the defaults.
    """
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import run_2layer_torch, spatiotemporal_segmentation_torch

    with np.load(CAE_DIR / f"{dataset}_val.npz") as data:
        images_np = np.asarray(data["images"][:n_images, 0], dtype=np.float64)
        labels_np = np.asarray(data["labels"][:n_images], dtype=np.int64)

    # Per image, not just per seed. The mean is what the criteria are stated
    # against, but the distribution is what tells you whether a budget is
    # uniformly mediocre or excellent-except-for-three-images, and the two have
    # different implications. Computing it is free; only keeping it was missing.
    per_seed = []
    per_seed_per_image: list[list[float]] = []
    for seed in EVAL_SEEDS[:n_seeds]:
        scores = []
        for i in range(n_images):
            image = torch.from_numpy(images_np[i])
            generator = torch.Generator().manual_seed(seed)
            states_m0, mask = run_2layer_torch(image, generator=generator)
            x0 = states_m0[:, 0].masked_fill(mask, 0)
            omega = image.T.reshape(-1)

            with torch.no_grad():
                x = x0.to(device).unsqueeze(0)
                trained = model.orbit_all(
                    x, omega.to(device).unsqueeze(0), mask.to(device).unsqueeze(0)
                )
            states = states_m0.clone()
            states[:, LAYER1_STEPS:200] = trained[0].cpu()
            states[mask, LAYER1_STEPS:200] = torch.nan
            cluster_map, *_ = spatiotemporal_segmentation_torch(
                states, image, mask, nt_mask=LAYER1_STEPS, n_clusters=2
            )
            scores.append(foreground_ari(labels_np[i], cluster_map.numpy()))
        per_seed.append(float(np.mean(scores)))
        per_seed_per_image.append([float(s) for s in scores])
        _progress(f"eval seed {seed}: mean FG ARI {per_seed[-1]:+.4f}")

    mean = float(np.mean(per_seed))
    own_sd = float(np.std(per_seed, ddof=0))
    base = BASELINES[dataset]
    return {
        "dataset": dataset,
        "per_seed_mean_foreground_ari": per_seed,
        # [seed][image], seeds in EVAL_SEEDS order. Enables the per-image
        # distribution, and lets a later reader ask which images a budget fails
        # rather than only how it did on average.
        "per_seed_per_image_foreground_ari": per_seed_per_image,
        # Ground-truth overlap fraction per image: the obvious difficulty
        # covariate, and without it an unstable image cannot be told from a
        # hard one. -1 is the excluded overlap label.
        "per_image_overlap_fraction": [
            float(np.mean(labels_np[i] == -1)) for i in range(n_images)
        ],
        "seed_averaged_foreground_ari": mean,
        "seed_std_foreground_ari": own_sd,
        "m0_at_defaults": base["m0_at_defaults"],
        "m0_seed_sd": base["m0_seed_sd"],
        "cc_baseline": base["cc"],
        # P6 (CRIT_14) as written: above cc. This is the answer for 2shapes.
        # For MNIST_shapes it is close to vacuous -- M0 already beats cc there
        # ninefold -- so the informative prespecified bar on that dataset is the
        # M0-relative one below, not this.
        "p6_above_cc": mean > base["cc"],
        # NOT P5. CRIT_15 requires exceeding M0's best swept cell by more than
        # two seed SDs, and that cell does not exist -- GATE_2 blocked, arc1b
        # never ran, and DEC_1 amended the comparator to M0 at defaults for
        # exactly this reason. This is that amended comparison, and it is a
        # diagnostic here rather than a verdict: a verdict needs an evaluation
        # recorded against CRIT_15 on the record, not a boolean in a JSON file.
        "diagnostic_above_m0_at_defaults": mean > base["m0_at_defaults"],
        # Two SDs of the TRAINED model's own per-seed means. Degenerates when
        # those collapse: at pursuit-b's 10,000 steps the ten values were
        # bit-identical, so this reduces to the plain above-M0 test and must
        # not be cited as variance-aware there. Read it next to
        # seed_std_foreground_ari, never alone.
        "diagnostic_exceeds_m0_by_two_seed_sd": mean
        > base["m0_at_defaults"] + 2 * own_sd,
        # Two SDs of M0's own spread, which does not collapse with training.
        # Reported so the margin stays interpretable at every budget.
        "diagnostic_exceeds_m0_by_two_m0_seed_sd": mean
        > base["m0_at_defaults"] + 2 * base["m0_seed_sd"],
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dataset",
        default=DATASET,
        choices=sorted(BASELINES),
        help=(
            "which CAE benchmark to train and evaluate on. Restricted to the "
            "datasets with locked cc/M0 comparators on the record: a dataset "
            "without them cannot produce a verdict, only a number."
        ),
    )
    parser.add_argument("--steps", type=int, default=500, help="Adam steps (NOTE_21's budget)")
    parser.add_argument("--epochs", type=int, default=0, help="if set, overrides --steps")
    parser.add_argument("--train-images", type=int, default=16_000)
    parser.add_argument("--out", default=str(Path(ROOT_DIR) / "outputs" / "pursuit-b.json"))
    parser.add_argument("--eval-images", type=int, default=EVAL_IMAGES)
    parser.add_argument("--eval-seeds", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="skip the at-init identity check against M0. Do not use for a real run.",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help=(
            "write the trained K_2 and delta_omega here after training. Off by "
            "default, and the output JSON gains the model_checkpoint key only "
            "when it is used, so passing it is what changes the JSON rather than "
            "merely having the flag. Note this does not make a run byte-identical "
            "to ART_16/ART_17: the schema has since gained dataset, m0_seed_sd "
            "and a second M0-relative diagnostic. What reproduces bit-for-bit is "
            "the arithmetic -- verified on different hardware -- not the file. "
            "Without a checkpoint the trained weights die with the process, and "
            "any later question about them costs a full retrain."
        ),
    )
    args = parser.parse_args(argv)

    # 1 and 2. Before anything else, and before any GPU is touched.
    known = assert_known_answers()
    control = assert_old_formula_fails_case_two()
    _progress("known-answer check passed: " + json.dumps({k: v["got"] for k, v in known.items()}))
    _progress(f"old-formula control: antiphase gives {control['old_formula_antiphase']}")
    if args.check_only:
        return {"known_answers": known, "old_formula_control": control}

    # Only the files this run opens. Hashing the other benchmarks would be
    # evidence about nothing here, and costs 90 MB of transfer to a remote VM.
    verified = verify_locked_dataset_hashes(
        only=[f"{args.dataset}_train.npz", f"{args.dataset}_val.npz"]
    )
    device = torch.device(args.device)
    torch.manual_seed(0)

    with np.load(CAE_DIR / f"{args.dataset}_train.npz") as data:
        images_np = np.asarray(data["images"][: args.train_images, 0], dtype=np.float64)
        labels_np = np.asarray(data["labels"][: args.train_images], dtype=np.int64)
    n_train, nrow, ncol = images_np.shape
    n = nrow * ncol

    images = torch.from_numpy(images_np).transpose(1, 2).reshape(n_train, n).to(device)
    labels = torch.from_numpy(labels_np).transpose(1, 2).reshape(n_train, n).to(device)
    k1, k2 = _sheets(nrow, ncol, device)

    # The oracle, before any training: this reimplementation must be M0 at
    # initialisation, or nothing it produces is comparable to arc1a.
    oracle = (
        None
        if args.skip_oracle
        else assert_matches_m0_at_init(k2, device, dataset=args.dataset)
    )

    _progress(f"precomputing M0 layer-1 masks for {n_train} images on {device}")
    masks = torch.empty((n_train, n), dtype=torch.bool, device=device)
    x0s = torch.empty((n_train, n), dtype=torch.complex128, device=device)
    precompute_started = time.monotonic()
    for lo in range(0, n_train, BATCH):
        hi = min(lo + BATCH, n_train)
        seeds = torch.arange(lo, hi) + TRAIN_SEED_OFFSET
        masks[lo:hi], x0s[lo:hi] = _layer1_masks(images[lo:hi], k1, seeds, device)
        if lo % (BATCH * 50) == 0:
            _progress(f"  layer-1 precompute {hi}/{n_train}")
    precompute_s = time.monotonic() - precompute_started

    steps = args.steps if args.epochs == 0 else (args.epochs * n_train) // BATCH
    _progress(f"training {steps} steps, batch {BATCH}, lr {LR}, on {device}")
    model = Layer2(k2, n).to(device)
    train_started = time.monotonic()
    curve = _train(model, images, labels, masks, x0s, steps, device)
    train_s = time.monotonic() - train_started

    checkpoint_path = None
    if args.save_model:
        # Before the eval, not after: the eval is the long tail and a crash in
        # it would otherwise throw away the training it just spent.
        checkpoint_path = Path(args.save_model)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "steps": steps,
                "batch": BATCH,
                "lr": LR,
                "train_images": n_train,
                "dataset": args.dataset,
                "layer2_steps": LAYER2_STEPS,
            },
            checkpoint_path,
        )
        _progress(f"saved trained K_2 and delta_omega to {checkpoint_path}")

    _progress(f"evaluating on val: {args.eval_images} images, {args.eval_seeds} seeds")
    evaluation = _evaluate(
        model, device, args.eval_images, args.eval_seeds, dataset=args.dataset
    )

    with torch.no_grad():
        k2_change = float((model.K2 - k2).norm() / k2.norm())
        delta_omega_norm = float(model.delta_omega.norm())

    output = {
        "pursuit": "b: the loss NOTE_16 intended (cos between-object term)",
        "device": str(device),
        "dataset": args.dataset,
        "steps": steps,
        "batch": BATCH,
        "lr": LR,
        "train_images": n_train,
        "eval_images": args.eval_images,
        "eval_seeds": args.eval_seeds,
        "at_init_oracle": oracle,
        "dataset_hashes_verified": verified,
        "known_answers": known,
        "old_formula_control": control,
        "training_curve": curve,
        "diagnostics": {
            "k2_frobenius_relative_change": k2_change,
            "delta_omega_norm": delta_omega_norm,
        },
        "evaluation": evaluation,
    }
    # Present only when a checkpoint was written, and the basename rather than
    # the path: an absolute path makes the artefact's hash depend on which
    # directory it was produced in, which is the same reproducibility defect as
    # wall-clock wearing different clothes. The full path goes in the sidecar.
    if checkpoint_path is not None:
        output["model_checkpoint"] = checkpoint_path.name
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    # Wall-clock lives beside the artefact, not inside it, so the artefact's
    # hash is a reproducibility lock -- a re-run re-derives it -- rather than
    # merely an integrity lock on one file.
    #
    # The claim is precisely: same code, same data, same --device re-derives
    # this hash. `device` stays IN the artefact deliberately. It is not
    # incidental machine detail: CUDA and CPU differ in fp64 reduction order,
    # so a cuda number and a cpu number are not required to be equal and an
    # artefact that hid which one produced it would be claiming more than it
    # can. A hash that changes with the device is correct; one that changes
    # with the wall-clock or the working directory is not.
    timing = out.with_suffix(".timing.json")
    timing.write_text(
        json.dumps(
            {
                "artefact": out.name,
                "model_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "layer1_precompute_seconds": precompute_s,
                "training_seconds": train_s,
                "device": str(device),
                "steps": steps,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _progress(f"wrote {out}")
    return output


if __name__ == "__main__":
    result = main()
    if "evaluation" not in result:
        print("\nknown-answer check passed; no training requested (--check-only)")
        raise SystemExit(0)
    e = result["evaluation"]
    print(
        f"\nFG ARI {e['seed_averaged_foreground_ari']:+.4f} "
        f"+/- {e['seed_std_foreground_ari']:.4f}   "
        f"(M0 at defaults {e['m0_at_defaults']:.4f}, cc {e['cc_baseline']:.2f})"
    )
    locked_eval = (
        len(e["per_seed_mean_foreground_ari"]) == len(EVAL_SEEDS)
        and result["eval_images"] == EVAL_IMAGES
    )
    if not locked_eval:
        print("smoke only - no P6 verdict (eval was not the locked 50 images x 10 seeds)")
    else:
        print(
            f"P6 (above cc 0.16): {e['p6_above_cc']}\n"
            f"diagnostic, above M0 at defaults 0.1209: "
            f"{e['diagnostic_above_m0_at_defaults']}"
            f"  (by >2 seed SD: {e['diagnostic_exceeds_m0_by_two_seed_sd']})"
        )
