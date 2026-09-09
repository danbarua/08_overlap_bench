"""arc1c-d training loop: NOTE_21 (budget, in optimiser steps) + NOTE_23
(omega_2 parametrisation, resolving the shared-vs-per-image gap NOTE_20 left
open). TASK_8, implementing LOE_4.

Trainable: K_2 (shared, unmasked N x N real matrix, same for every image) and
delta_omega (shared, (N,) real vector, init zero). Per image: omega_2(i) =
intensity_i (that image's own pixel intensity, never overwritten) + delta_omega.
At delta_omega = 0, K_2 = its initial value, this is exactly M0's layer 2 for
every image (H1c-d's zero-error identity still holds).

Masking is applied per-image at forward time via the identity
(K o (m m^T)) x = m . (K (m . x)) -- shared K_2 needs no per-image copies;
only the batch's own mask vectors are multiplied in and out (NOTE_21's
batching note). mask/x0/intensity/labels come from 07's own unmodified
run_2layer_torch and the dataset's own ground truth, never recomputed.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from overlap_bench.dataset_hashes import CAE_DIR
from overlap_bench.loss_m1 import phase_coherence_loss
from overlap_bench.paths import ROOT_DIR
from overlap_bench.reference_repo import ensure_on_path

MODEL_OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "m1d_trained.pt"
TRAIN_LOG_PATH = Path(ROOT_DIR) / "outputs" / "harness-m1d-train.json"

BATCH_SIZE = 32
N_STEPS = 500
LR = 1e-3
SEED_OFFSET = 100_000  # NOTE_21: manual_seed(100000 + i), disjoint from eval seeds 1..10


def _progress(message: str) -> None:
    print(f"[train-m1d] {message}", file=sys.stderr, flush=True)


def _batched_layer2_step(k2: torch.Tensor, x: torch.Tensor, mask_float: torch.Tensor, omega2_eff: torch.Tensor) -> torch.Tensor:
    """x, mask_float, omega2_eff: (N,B). k2: (N,N) real, shared across the batch."""
    k2c = k2.to(torch.complex128)
    masked_input = mask_float * x
    kx = k2c @ masked_input
    kx = mask_float * kx
    return kx + 1j * omega2_eff * x


def _batched_layer1(images: np.ndarray, indices: range, k1: torch.Tensor, seed_offset: int = SEED_OFFSET):
    """Reproduce run_2layer_torch's layer 1 (mask, x0) for a batch of images,
    batched via the shared (image-independent) K_1 sheet -- verified
    bit-identical to the unmodified reference (see _verify_batched_layer1).
    No masking here: layer 1 has none; it only produces the mask layer 2 uses.
    """
    n = k1.shape[0]
    x0_list, omega_list = [], []
    for i in indices:
        image = torch.from_numpy(np.asarray(images[i, 0], dtype=np.float64))
        generator = torch.Generator().manual_seed(seed_offset + i)
        phases = torch.rand(n, dtype=torch.float64, generator=generator)
        x0_list.append(torch.exp(1j * (phases - 0.5) * (2 * math.pi)))
        omega_list.append(image.T.reshape(-1))
    x0_batch = torch.stack(x0_list, dim=1)
    omega_batch = torch.stack(omega_list, dim=1)

    k1c = k1.to(torch.complex128)
    x = x0_batch.clone()
    for _ in range(1, 60):
        x = k1c @ x + 1j * omega_batch * x

    phases_final = torch.angle(x)
    threshold = phases_final.mean(dim=0)
    above = phases_final > threshold.unsqueeze(0)
    below = phases_final < threshold.unsqueeze(0)
    mask_batch = torch.where(above.sum(dim=0, keepdim=True) > below.sum(dim=0, keepdim=True), above, below)
    return x0_batch, mask_batch, omega_batch


def _verify_batched_layer1(images: np.ndarray, k1: torch.Tensor, run_2layer_torch, n_check: int = 4) -> None:
    """Re-verify, every run, that the batched reimplementation matches the
    unmodified reference exactly, before trusting it for training data."""
    x0_batch, mask_batch, _ = _batched_layer1(images, range(n_check), k1)
    for idx in range(n_check):
        image = torch.from_numpy(np.asarray(images[idx, 0], dtype=np.float64))
        generator = torch.Generator().manual_seed(SEED_OFFSET + idx)
        states_m0, mask_ref = run_2layer_torch(image, generator=generator)
        diff = (x0_batch[:, idx] - states_m0[:, 0]).abs().max().item()
        if diff != 0.0 or not torch.equal(mask_batch[:, idx], mask_ref):
            raise AssertionError(
                f"_batched_layer1 diverges from run_2layer_torch at image {idx}: "
                f"x0 max abs diff {diff}, mask equal {torch.equal(mask_batch[:, idx], mask_ref)}"
            )
    _progress(f"_batched_layer1 verified bit-identical to run_2layer_torch on {n_check} images")


def _load_batch(images: np.ndarray, labels: np.ndarray, indices: range, k1: torch.Tensor):
    x0_batch, mask_batch, intensity_batch = _batched_layer1(images, indices, k1)
    labels_list = [
        torch.from_numpy(np.asarray(labels[i], dtype=np.int64)).T.reshape(-1) for i in indices
    ]
    return x0_batch, mask_batch, intensity_batch, torch.stack(labels_list, dim=1)


def main(n_steps: int = N_STEPS) -> dict:
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch, run_2layer_torch

    with np.load(CAE_DIR / "2shapes_train.npz") as data:
        images = data["images"][: n_steps * BATCH_SIZE]
        labels = data["labels"][: n_steps * BATCH_SIZE]

    nrow, ncol = images.shape[-2], images.shape[-1]
    n = nrow * ncol

    k1 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9, dtype=torch.float64).real.clone()
    _verify_batched_layer1(images, k1, run_2layer_torch)


    k2 = torch.nn.Parameter(gaussian_sheet_torch(nrow, ncol, 0.5, 0.0313, dtype=torch.float64).real.clone())
    delta_omega = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64))
    k2_initial = k2.detach().clone()

    optimizer = torch.optim.Adam([k2, delta_omega], lr=LR)

    t0_loss = None
    final_loss = None
    step_times = []

    for step in range(n_steps):
        start = step * BATCH_SIZE
        indices = range(start, start + BATCH_SIZE)
        x0_batch, mask_batch, intensity_batch, labels_batch = _load_batch(images, labels, indices, k1)

        mask_float = (~mask_batch).to(torch.float64)
        x0_masked = (x0_batch * mask_float).to(torch.complex128)

        step_start = time.perf_counter()

        omega2_eff = (intensity_batch + delta_omega.unsqueeze(1)) * mask_float

        x = x0_masked
        for _ in range(140):
            x = _batched_layer2_step(k2, x, mask_float, omega2_eff)

        losses = [phase_coherence_loss(x[:, b], labels_batch[:, b]) for b in range(BATCH_SIZE)]
        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        if step == 0:
            t0_loss = loss.item()
            _progress(f"t=0 loss (batch 1, pre-step-1): {t0_loss:.6f}")
        optimizer.step()

        step_elapsed = time.perf_counter() - step_start
        step_times.append(step_elapsed)
        final_loss = loss.item()

        if (step + 1) % 25 == 0 or step == 0:
            _progress(
                f"step {step + 1}/{n_steps} loss={loss.item():.6f} "
                f"step_time={step_elapsed:.3f}s"
            )

    k2_frobenius_relative_change = (k2.detach() - k2_initial).norm().item() / k2_initial.norm().item()
    delta_omega_final_norm = delta_omega.detach().norm().item()

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"K2": k2.detach(), "delta_omega": delta_omega.detach()}, MODEL_OUTPUT_PATH)

    result = {
        "t0_loss": t0_loss,
        "final_loss": final_loss,
        "n_steps": n_steps,
        "batch_size": BATCH_SIZE,
        "k2_frobenius_relative_change": k2_frobenius_relative_change,
        "delta_omega_final_norm": delta_omega_final_norm,
    }
    TRAIN_LOG_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _progress(f"done, mean_step_time={sum(step_times) / len(step_times):.4f}s, "
              f"total_train_wall={sum(step_times):.1f}s -- these are wall-clock, reported separately")
    return {**result, "mean_step_time_seconds": sum(step_times) / len(step_times), "total_train_wall_seconds": sum(step_times)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
