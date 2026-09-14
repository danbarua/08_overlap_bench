"""harness-M1: H1a (propagator oracle), H1b (gradient check), H1c (init
equals M0). No science -- this is TASK_4, not arc1c. NOTE_16 fixes the
loss surrogate, time convention, and mask-reuse choice this implements.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

from overlap_bench.dataset_hashes import CAE_DIR
from overlap_bench.loss_m1 import phase_coherence_loss
from overlap_bench.model_m1 import LinearPropagator
from overlap_bench.paths import ROOT_DIR
from overlap_bench.reference_repo import ensure_on_path

OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "harness-m1.json"


def _progress(message: str) -> None:
    print(f"[harness-m1] {message}", file=sys.stderr, flush=True)


def h1a_expm_oracle() -> dict:
    """H1a: torch.linalg.matrix_exp agrees with scipy.linalg.expm to 1e-12
    relative, at N=16 and N=1024 (the 32x32 image size). Times the N=1024
    call: 20 epochs x however many images/batch would call this repeatedly
    in arc1c, so a slow single call matters before committing to that budget.
    """
    results = {}
    rng = np.random.RandomState(0)
    for n in (16, 1024):
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        a = (a * 0.01).astype(np.complex128)  # small entries: keep exp well-scaled
        a_torch = torch.from_numpy(a)
        start = time.perf_counter()
        torch_result = torch.linalg.matrix_exp(a_torch).numpy()
        elapsed = time.perf_counter() - start
        scipy_result = scipy.linalg.expm(a)
        relative_error = float(
            np.max(np.abs(torch_result - scipy_result)) / np.max(np.abs(scipy_result))
        )
        results[str(n)] = {"relative_error": relative_error}
        _progress(f"H1a N={n}: relative_error={relative_error:.3e}, elapsed={elapsed:.4f}s")
    return results


def h1b_gradient_check() -> dict:
    """H1b: finite-difference gradient of phase_coherence_loss through the
    propagator reaches both K and omega, to 1e-6."""
    torch.manual_seed(0)
    n = 12
    k_init = (torch.rand(n, n, dtype=torch.float64) - 0.5) * 0.1
    omega_init = torch.rand(n, dtype=torch.float64)
    model = LinearPropagator(k_init, omega_init)
    x0 = torch.exp(1j * torch.rand(n, dtype=torch.float64) * 2 * torch.pi).to(torch.complex128)
    labels = torch.tensor([1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2], dtype=torch.long)
    assert labels.shape == (n,)

    def loss_fn() -> torch.Tensor:
        x_t = model.trajectory(x0, steps=5)[:, -1]
        return phase_coherence_loss(x_t, labels)

    model.zero_grad()
    loss = loss_fn()
    loss.backward()
    analytic_grad_k = model.K.grad.clone()
    analytic_grad_omega = model.omega.grad.clone()

    eps = 1e-6
    max_error_k = 0.0
    with torch.no_grad():
        for i, j in [(0, 0), (1, 3), (5, 7), (11, 11)]:
            original = model.K[i, j].item()
            model.K[i, j] = original + eps
            plus = loss_fn().item()
            model.K[i, j] = original - eps
            minus = loss_fn().item()
            model.K[i, j] = original
            finite_diff = (plus - minus) / (2 * eps)
            max_error_k = max(max_error_k, abs(finite_diff - analytic_grad_k[i, j].item()))

    max_error_omega = 0.0
    with torch.no_grad():
        for i in range(n):
            original = model.omega[i].item()
            model.omega[i] = original + eps
            plus = loss_fn().item()
            model.omega[i] = original - eps
            minus = loss_fn().item()
            model.omega[i] = original
            finite_diff = (plus - minus) / (2 * eps)
            max_error_omega = max(max_error_omega, abs(finite_diff - analytic_grad_omega[i].item()))

    _progress(f"H1b: max_error_K={max_error_k:.3e}, max_error_omega={max_error_omega:.3e}")
    return {
        "max_finite_difference_error_K": max_error_k,
        "max_finite_difference_error_omega": max_error_omega,
        "gradient_reaches_K": bool(analytic_grad_k.abs().sum() > 0),
        "gradient_reaches_omega": bool(analytic_grad_omega.abs().sum() > 0),
    }


def h1c_init_equals_m0() -> dict:
    """H1c: M1 at initialisation, M0's readout, M0's mask (NOTE_16(3)),
    gives M0's labels on the 50 2shapes val images under seed 1."""
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import (
        gaussian_sheet_torch,
        run_2layer_torch,
        spatiotemporal_segmentation_torch,
    )

    with np.load(CAE_DIR / "2shapes_val.npz") as data:
        images = data["images"][:50]

    seed = 1
    matches = 0
    mismatches = []
    diverged = []
    for i in range(50):
        image = torch.from_numpy(np.asarray(images[i, 0], dtype=np.float64))
        nrow, ncol = image.shape
        generator = torch.Generator().manual_seed(seed)
        states_m0, mask_m0 = run_2layer_torch(image, generator=generator)
        x0 = states_m0[:, 0]

        cluster_map_m0, *_ = spatiotemporal_segmentation_torch(
            states_m0, image, mask_m0, nt_mask=60, n_clusters=2
        )

        k_init = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9).real
        omega_init = image.T.reshape(-1)
        model = LinearPropagator(k_init, omega_init)

        max_real_eigenvalue = float(torch.linalg.eigvals(model.generator()).real.max())

        with torch.no_grad():
            trajectory = model.trajectory(x0, steps=199)
        states_m1 = torch.cat([x0.unsqueeze(1), trajectory], dim=1)

        if not torch.isfinite(states_m1).all():
            first_nonfinite_step = int((~torch.isfinite(states_m1).all(dim=0)).float().argmax())
            diverged.append({
                "image_index": i,
                "first_nonfinite_step": first_nonfinite_step,
                "max_real_eigenvalue_of_generator": max_real_eigenvalue,
            })
            _progress(
                f"H1c: image {i + 1}/50 DIVERGED at step {first_nonfinite_step} "
                f"(max Re(eigenvalue)={max_real_eigenvalue:.4f})"
            )
            continue

        cluster_map_m1, *_ = spatiotemporal_segmentation_torch(
            states_m1, image, mask_m0, nt_mask=60, n_clusters=2
        )

        is_match = torch.equal(cluster_map_m0, cluster_map_m1)
        if is_match:
            matches += 1
        else:
            mismatches.append(i)
        _progress(f"H1c: image {i + 1}/50 {'match' if is_match else 'MISMATCH'}")

    return {
        "n_images": 50,
        "matches": matches,
        "mismatches": mismatches,
        "diverged": diverged,
        "seed": seed,
    }


def main() -> dict:
    output = {
        "h1a": h1a_expm_oracle(),
        "h1b": h1b_gradient_check(),
        "h1c": h1c_init_equals_m0(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _progress("done, written to outputs/harness-m1.json")
    return output


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
