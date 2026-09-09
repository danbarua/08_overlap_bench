"""How-to: when a trained model scores far worse than expected, find out HOW
before guessing WHY.

Context: arc1c-d's trained M1d scored at chance (-0.010 FG ARI) after 500
Adam steps that visibly moved the parameters (K_2 Frobenius change 71.7%).
"Training made it worse" is an observation, not an explanation. This script
walks through the concrete inspections that turned the observation into a
mechanism (see ART_12, NOTE_28 in this project's LabKit record):

1. Compare parameter statistics before/after training (mean, std, and
   crucially the dominant eigenvalue of the actual dynamics matrix -- not
   just the trainable weight matrix in isolation; see the "concern" advisory
   in this project's history about checking the *masked* generator A_2 =
   K_2 + i*diag(omega), not K_2 alone).
2. Track the trajectory's magnitude step-by-step. A geometric growth rate
   matching the dominant eigenvalue's magnitude confirms the mechanism
   directly, rather than inferring it from the eigenvalue alone.
3. Check per-object phase statistics at the final timestep: internal
   coherence (|z_o|, should be near 1 for good within-object phase-locking)
   and cross-object alignment (does the model correctly keep DIFFERENT
   objects at DIFFERENT phases, or has everything converged to one shared
   phase?). This is what actually explains a chance-level segmentation
   score: if every foreground pixel (regardless of object) shares nearly
   the same phase, no downstream clustering method can tell objects apart.

Each check is cheap (seconds, a handful of images) and, taken together,
turns "the model failed" into "the model failed for this specific,
verifiable reason" -- which is what let this project's next question
("but is that the loss discovering a real fixed point, or something else?")
get asked precisely enough to answer (see check_surrogate_at_baseline.py).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import overlap_bench.run_arc1cd as arc1cd
from overlap_bench.reference_repo import ensure_on_path
from overlap_bench.paths import ROOT_DIR

ensure_on_path()
from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch  # noqa: E402


def load_images(n: int = 5):
    from overlap_bench.dataset_hashes import CAE_DIR
    with np.load(CAE_DIR / "2shapes_val.npz") as data:
        return data["images"][:n], data["labels"][:n]


def compare_parameter_statistics(k2_trained: torch.Tensor, k2_initial: torch.Tensor) -> None:
    """Step 1: coarse before/after comparison. Cheap, and often enough to
    notice something is off (here: the mean weight roughly quadrupled)."""
    print("--- Parameter statistics: initial vs trained ---")
    for name, k2 in [("initial", k2_initial), ("trained", k2_trained)]:
        print(f"  K2 {name}: mean={k2.mean().item():.4f} std={k2.std().item():.4f}")
    eig_initial = torch.linalg.eigvals(k2_initial.to(torch.complex128)).real.max().item()
    eig_trained = torch.linalg.eigvals(k2_trained.to(torch.complex128)).real.max().item()
    print(f"  K2 alone, max Re(eigenvalue): initial={eig_initial:.3f} trained={eig_trained:.3f}")
    print("  NOTE: check the *masked A_2* generator's eigenvalue too, not just K2 in isolation")
    print("        (a purely-real K2's eigenvalues don't include the i*diag(omega) contribution).")


def track_magnitude_growth(k2: torch.Tensor, delta_omega: torch.Tensor, images, seed: int = 1) -> None:
    """Step 2: step-by-step magnitude trajectory. A geometric growth rate
    matching the dominant |eigenvalue| of the masked generator A_2 confirms
    the mechanism directly."""
    nrow, ncol = images.shape[-2], images.shape[-1]
    k1 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9, dtype=torch.float64).real.clone()
    x0_batch, mask_batch, intensity_batch, _ = arc1cd._batched_layer1_shared_x0(images, seed, k1)
    mask_float = (~mask_batch).to(torch.float64)
    x0_masked = (x0_batch * mask_float).to(torch.complex128)
    omega2_eff = (intensity_batch + delta_omega.unsqueeze(1)) * mask_float

    print("--- Magnitude trajectory (image 0, unmasked mean |x|) ---")
    x = x0_masked
    mask0 = mask_batch[:, 0]
    with torch.no_grad():
        for t in range(140):
            x = arc1cd._batched_layer2_step(k2, x, mask_float, omega2_eff)
            if t < 3 or t % 40 == 0 or t >= 137:
                print(f"  step {t}: mean|x| = {x[~mask0, 0].abs().mean().item():.4e}")

    a2 = k2.to(torch.complex128) * mask_float[:, 0:1].to(torch.complex128) * mask_float[:, 0:1].T.to(torch.complex128)
    a2 = a2 + 1j * torch.diag(omega2_eff[:, 0]).to(torch.complex128)
    dominant = torch.linalg.eigvals(a2).abs().max().item()
    print(f"  masked A_2 dominant |eigenvalue| (image 0): {dominant:.4f}")
    print("  compare consecutive-step magnitude ratios above to this number -- they should match")


def check_object_synchrony(k2: torch.Tensor, delta_omega: torch.Tensor, images, labels, seed: int = 1) -> None:
    """Step 3: the check that actually explains chance-level ARI. Per
    object, |z_o| near 1 means good internal coherence (not itself a
    problem). The cross-object cosine near 1 means the objects have
    converged to the SAME phase -- that is what makes them indistinguishable
    to any downstream clustering, regardless of how coherent each is
    internally."""
    from overlap_bench.loss_m1 import phase_coherence_loss

    nrow, ncol = images.shape[-2], images.shape[-1]
    k1 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9, dtype=torch.float64).real.clone()
    x0_batch, mask_batch, intensity_batch, _ = arc1cd._batched_layer1_shared_x0(images, seed, k1)
    mask_float = (~mask_batch).to(torch.float64)
    x0_masked = (x0_batch * mask_float).to(torch.complex128)
    omega2_eff = (intensity_batch + delta_omega.unsqueeze(1)) * mask_float
    with torch.no_grad():
        x = x0_masked
        for _ in range(140):
            x = arc1cd._batched_layer2_step(k2, x, mask_float, omega2_eff)

    print("--- Per-object phase synchrony at t=199 ---")
    for i in range(images.shape[0]):
        x_T = x[:, i]
        truth = torch.from_numpy(np.asarray(labels[i], dtype=np.int64)).T.reshape(-1)
        object_ids = sorted(int(v) for v in torch.unique(truth).tolist() if v > 0)
        unit_phase = x_T / (x_T.abs() + 1e-12)
        zs = [unit_phase[truth == oid].mean() for oid in object_ids]
        print(f"  image {i}: " + ", ".join(f"|z_{oid}|={z.abs().item():.4f}" for oid, z in zip(object_ids, zs)))
        if len(zs) == 2:
            cross = (zs[0] * zs[1].conj()).abs().item() / (zs[0].abs().item() * zs[1].abs().item())
            loss = phase_coherence_loss(x_T, truth).item()
            print(f"    cross-object cosine={cross:.6f}  loss(between-within)={loss:.6f}")
            print("    cosine near 1.0 with DIFFERENT mean phases per object is itself worth a second look --")
            print("    see check_surrogate_at_baseline.py for what that turned out to mean here.")


if __name__ == "__main__":
    checkpoint = torch.load(Path(ROOT_DIR) / "outputs" / "m1d_trained.pt", weights_only=True)
    k2_trained, delta_omega_trained = checkpoint["K2"], checkpoint["delta_omega"]

    images, labels = load_images()
    nrow, ncol = images.shape[-2], images.shape[-1]
    k2_initial = gaussian_sheet_torch(nrow, ncol, 0.5, 0.0313, dtype=torch.float64).real.clone()

    compare_parameter_statistics(k2_trained, k2_initial)
    print()
    track_magnitude_growth(k2_trained, delta_omega_trained, images)
    print()
    check_object_synchrony(k2_trained, delta_omega_trained, images, labels)
