"""harness-M1d: H1a-d (step oracle), H1b-d (gradient check), H1c-d (init
equals M0), plus a per-image forward+backward timing observation. TASK_7,
implementing LOE_4/NOTE_20. No science.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from overlap_bench.dataset_hashes import CAE_DIR
from overlap_bench.loss_m1 import phase_coherence_loss
from overlap_bench.model_m1d import LinearRecurrence2
from overlap_bench.paths import ROOT_DIR
from overlap_bench.reference_repo import ensure_on_path

OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "harness-m1d.json"
TIMING_OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "harness-m1d-timing.json"


def _progress(message: str) -> None:
    print(f"[harness-m1d] {message}", file=sys.stderr, flush=True)


def _load_image(dataset: str, split: str, index: int) -> np.ndarray:
    with np.load(CAE_DIR / f"{dataset}_{split}.npz") as data:
        return np.asarray(data["images"][index, 0], dtype=np.float64)


def _build_m1d_at_init(image: torch.Tensor, mask: torch.Tensor, nrow: int, ncol: int):
    """K_2/omega_2 initialised exactly as M0's layer 2 (NOTE_20)."""
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch

    weights2 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.0313, dtype=torch.float64).real.clone()
    weights2[mask, :] = 0
    weights2[:, mask] = 0
    omega = image.T.reshape(-1)
    omega2 = omega.masked_fill(mask, 0)
    return LinearRecurrence2(weights2, omega2, mask)


def h1a_d_step_oracle() -> dict:
    """H1a-d / CRIT_20: one M1d layer-2 step from M0's masked x0 matches
    column 60 of 07's own run_2layer_torch output, on unmasked pixels."""
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import run_2layer_torch

    image = torch.from_numpy(_load_image("2shapes", "val", 0))
    nrow, ncol = image.shape
    generator = torch.Generator().manual_seed(1)
    states_m0, mask = run_2layer_torch(image, generator=generator)
    x0 = states_m0[:, 0]

    model = _build_m1d_at_init(image, mask, nrow, ncol)
    with torch.no_grad():
        x0_masked = x0.masked_fill(mask, 0)
        x_60 = model.step(x0_masked)

    reference_col60 = states_m0[:, 60]
    unmasked = ~mask
    diff = (x_60[unmasked] - reference_col60[unmasked]).abs().max().item()
    scale = reference_col60[unmasked].abs().max().item()
    relative_error = diff / scale if scale > 0 else diff
    _progress(f"H1a-d: relative_error={relative_error:.3e}")
    return {"relative_error": relative_error}


def h1b_d_gradient_check() -> dict:
    """H1b-d / CRIT_17: finite-difference gradient of the NOTE_16 loss
    reaches K_2 and omega_2, through the full 140-step layer-2 recurrence,
    at N=1024 on one image."""
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import run_2layer_torch

    image = torch.from_numpy(_load_image("2shapes", "train", 0))
    nrow, ncol = image.shape
    generator = torch.Generator().manual_seed(1)
    states_m0, mask = run_2layer_torch(image, generator=generator)
    x0 = states_m0[:, 0]
    x0_masked = x0.masked_fill(mask, 0)

    with np.load(CAE_DIR / "2shapes_train.npz") as data:
        truth = torch.from_numpy(np.asarray(data["labels"][0], dtype=np.int64)).T.reshape(-1)

    model = _build_m1d_at_init(image, mask, nrow, ncol)

    def loss_fn() -> torch.Tensor:
        x_T = model.orbit(x0_masked, steps=140)[:, -1]
        return phase_coherence_loss(x_T, truth)

    model.zero_grad()
    loss = loss_fn()
    loss.backward()
    analytic_grad_k = model.K2.grad.clone()
    analytic_grad_omega = model.omega2.grad.clone()

    unmasked_idx = (~mask).nonzero(as_tuple=True)[0]
    eps = 1e-6
    max_error_k = 0.0
    with torch.no_grad():
        for i, j in [
            (unmasked_idx[0].item(), unmasked_idx[0].item()),
            (unmasked_idx[0].item(), unmasked_idx[1].item()),
            (unmasked_idx[5].item(), unmasked_idx[7].item()),
            (unmasked_idx[-1].item(), unmasked_idx[-1].item()),
        ]:
            original = model.K2[i, j].item()
            model.K2[i, j] = original + eps
            plus = loss_fn().item()
            model.K2[i, j] = original - eps
            minus = loss_fn().item()
            model.K2[i, j] = original
            finite_diff = (plus - minus) / (2 * eps)
            max_error_k = max(max_error_k, abs(finite_diff - analytic_grad_k[i, j].item()))

    max_error_omega = 0.0
    with torch.no_grad():
        for i in unmasked_idx[:4].tolist():
            original = model.omega2[i].item()
            model.omega2[i] = original + eps
            plus = loss_fn().item()
            model.omega2[i] = original - eps
            minus = loss_fn().item()
            model.omega2[i] = original
            finite_diff = (plus - minus) / (2 * eps)
            max_error_omega = max(max_error_omega, abs(finite_diff - analytic_grad_omega[i].item()))

    _progress(f"H1b-d: max_error_K={max_error_k:.3e}, max_error_omega={max_error_omega:.3e}")
    return {
        "max_finite_difference_error_K": max_error_k,
        "max_finite_difference_error_omega": max_error_omega,
        "gradient_reaches_K": bool(analytic_grad_k.abs().sum() > 0),
        "gradient_reaches_omega": bool(analytic_grad_omega.abs().sum() > 0),
    }


def h1c_d_init_equals_m0() -> dict:
    """H1c-d / CRIT_19: M1d at initialisation gives M0's labels on 50
    2shapes val images (seed 1), and its layer-2 orbit matches M0's saved
    trajectory to 1e-12 relative on every unmasked pixel of every image."""
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import run_2layer_torch, spatiotemporal_segmentation_torch

    with np.load(CAE_DIR / "2shapes_val.npz") as data:
        images = data["images"][:50]

    seed = 1
    matches = 0
    mismatches = []
    max_orbit_relative_errors = []
    for i in range(50):
        image = torch.from_numpy(np.asarray(images[i, 0], dtype=np.float64))
        nrow, ncol = image.shape
        generator = torch.Generator().manual_seed(seed)
        states_m0, mask = run_2layer_torch(image, generator=generator)
        x0 = states_m0[:, 0]
        x0_masked = x0.masked_fill(mask, 0)

        model = _build_m1d_at_init(image, mask, nrow, ncol)
        with torch.no_grad():
            layer2_orbit = model.orbit(x0_masked, steps=140)

        reference_layer2 = states_m0[:, 60:200]
        unmasked = ~mask
        diff = (layer2_orbit[unmasked] - reference_layer2[unmasked]).abs().max().item()
        scale = reference_layer2[unmasked].abs().max().item()
        relative_error = diff / scale if scale > 0 else diff
        max_orbit_relative_errors.append(relative_error)

        states_m1d = states_m0.clone()
        states_m1d[:, 60:200] = layer2_orbit
        states_m1d[mask, 60:200] = torch.nan

        cluster_map_m0, *_ = spatiotemporal_segmentation_torch(states_m0, image, mask, nt_mask=60, n_clusters=2)
        cluster_map_m1d, *_ = spatiotemporal_segmentation_torch(states_m1d, image, mask, nt_mask=60, n_clusters=2)

        is_match = torch.equal(cluster_map_m0, cluster_map_m1d)
        if is_match:
            matches += 1
        else:
            mismatches.append(i)
        _progress(
            f"H1c-d: image {i + 1}/50 {'match' if is_match else 'MISMATCH'} "
            f"orbit_rel_err={relative_error:.3e}"
        )

    return {
        "n_images": 50,
        "matches": matches,
        "mismatches": mismatches,
        "max_orbit_relative_error": max(max_orbit_relative_errors),
        "seed": seed,
    }


def timing_observation() -> dict:
    """Per-image forward+backward time through the full 140-step recurrence,
    for extrapolating the 20-epoch x 50k-image training budget (NOTE_6)."""
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import run_2layer_torch

    image = torch.from_numpy(_load_image("2shapes", "train", 0))
    nrow, ncol = image.shape
    generator = torch.Generator().manual_seed(1)
    states_m0, mask = run_2layer_torch(image, generator=generator)
    x0 = states_m0[:, 0]
    x0_masked = x0.masked_fill(mask, 0)

    with np.load(CAE_DIR / "2shapes_train.npz") as data:
        truth = torch.from_numpy(np.asarray(data["labels"][0], dtype=np.int64)).T.reshape(-1)

    model = _build_m1d_at_init(image, mask, nrow, ncol)

    def one_step():
        model.zero_grad()
        x_T = model.orbit(x0_masked, steps=140)[:, -1]
        loss = phase_coherence_loss(x_T, truth)
        loss.backward()

    one_step()  # warm-up, excluded from timing
    n_reps = 5
    start = time.perf_counter()
    for _ in range(n_reps):
        one_step()
    elapsed = (time.perf_counter() - start) / n_reps

    n_train_images = 50_000
    n_epochs = 20
    total_seconds = elapsed * n_train_images * n_epochs
    _progress(
        f"timing: {elapsed:.4f}s/image (forward+backward), "
        f"extrapolated {n_epochs} epochs x {n_train_images} images = {total_seconds / 3600:.2f}h "
        f"(single-image, unbatched, single-core estimate)"
    )
    return {
        "seconds_per_image_forward_backward": elapsed,
        "n_train_images": n_train_images,
        "n_epochs": n_epochs,
        "extrapolated_total_seconds_unbatched": total_seconds,
    }


def main() -> dict:
    output = {
        "h1a_d": h1a_d_step_oracle(),
        "h1b_d": h1b_d_gradient_check(),
        "h1c_d": h1c_d_init_equals_m0(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _progress("done, written to outputs/harness-m1d.json")

    timing = timing_observation()
    TIMING_OUTPUT_PATH.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
    _progress("timing (non-deterministic wall-clock, kept out of the hashed artifact) "
              "written to outputs/harness-m1d-timing.json")
    return {**output, "timing": timing}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
