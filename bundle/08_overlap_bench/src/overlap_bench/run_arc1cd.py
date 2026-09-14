"""arc1c-d: trained M1d's FG ARI on 2shapes val, seeds 1..10. TASK_8,
implementing LOE_4, held to CRIT_7 (protocol lock), CRIT_8 (pursuit's own
locks -- NOTE_21/23's budget), CRIT_15 (P5), CRIT_14 (P6).

Reported statistic: mean FG ARI over the 50 val images per seed, then mean
and standard deviation of those 10 per-seed means (ddof=0, per NOTE_11/12),
matching arc1a's protocol exactly -- same seeds, same n_clusters=2, same
meter. Per-seed x0 is shared across all 50 images (arc1a's own convention:
one Generator().manual_seed(seed) per seed, not per image), verified
bit-identical to run_2layer_torch every run before use.

No wall-clock time in the output artefact.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from overlap_bench.dataset_hashes import CAE_DIR, verify_locked_dataset_hashes
from overlap_bench.harness_m0 import foreground_ari
from overlap_bench.paths import ROOT_DIR
from overlap_bench.reference_repo import ensure_on_path, verify_reference_commit
from overlap_bench.train_m1d import _batched_layer2_step

OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "arc1cd.json"
MODEL_PATH = Path(ROOT_DIR) / "outputs" / "m1d_trained.pt"
SEEDS = tuple(range(1, 11))
N_IMAGES = 50


def _progress(message: str) -> None:
    print(f"[arc1c-d] {message}", file=sys.stderr, flush=True)


def _batched_layer1_shared_x0(images: np.ndarray, seed: int, k1: torch.Tensor):
    """arc1a's own convention: one Generator().manual_seed(seed) per seed,
    reused (not re-seeded) across all N_IMAGES images in that seed's sweep."""
    n = k1.shape[0]
    generator = torch.Generator().manual_seed(seed)
    phases = torch.rand(n, dtype=torch.float64, generator=generator)
    x0_shared = torch.exp(1j * (phases - 0.5) * (2 * math.pi))
    b = images.shape[0]
    x0_batch = x0_shared.unsqueeze(1).repeat(1, b)
    omega_list = [torch.from_numpy(np.asarray(images[i, 0], dtype=np.float64)).T.reshape(-1) for i in range(b)]
    omega_batch = torch.stack(omega_list, dim=1)

    k1c = k1.to(torch.complex128)
    x = x0_batch.clone()
    layer1_orbit = torch.empty((n, b, 60), dtype=torch.complex128)
    layer1_orbit[:, :, 0] = x
    for t in range(1, 60):
        x = k1c @ x + 1j * omega_batch * x
        layer1_orbit[:, :, t] = x

    phases_final = torch.angle(x)
    threshold = phases_final.mean(dim=0)
    above = phases_final > threshold.unsqueeze(0)
    below = phases_final < threshold.unsqueeze(0)
    mask_batch = torch.where(above.sum(dim=0, keepdim=True) > below.sum(dim=0, keepdim=True), above, below)
    return x0_batch, mask_batch, omega_batch, layer1_orbit


def _verify_shared_x0_layer1(images: np.ndarray, k1: torch.Tensor, run_2layer_torch, seed: int, n_check: int = 4) -> None:
    x0_batch, mask_batch, _, _ = _batched_layer1_shared_x0(images[:n_check], seed, k1)
    for idx in range(n_check):
        image = torch.from_numpy(np.asarray(images[idx, 0], dtype=np.float64))
        generator = torch.Generator().manual_seed(seed)
        states_ref, mask_ref = run_2layer_torch(image, generator=generator)
        diff = (x0_batch[:, idx] - states_ref[:, 0]).abs().max().item()
        if diff != 0.0 or not torch.equal(mask_batch[:, idx], mask_ref):
            raise AssertionError(
                f"_batched_layer1_shared_x0 diverges from run_2layer_torch at seed={seed}, image {idx}"
            )
    _progress(f"_batched_layer1_shared_x0 verified bit-identical to run_2layer_torch at seed={seed}")


def main() -> dict:
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch, run_2layer_torch, spatiotemporal_segmentation_torch

    verify_reference_commit()
    hash_check = verify_locked_dataset_hashes()

    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    k2 = checkpoint["K2"]
    delta_omega = checkpoint["delta_omega"]

    with np.load(CAE_DIR / "2shapes_val.npz") as data:
        images = data["images"][:N_IMAGES]
        labels = data["labels"][:N_IMAGES]

    nrow, ncol = images.shape[-2], images.shape[-1]
    k1 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9, dtype=torch.float64).real.clone()
    _verify_shared_x0_layer1(images, k1, run_2layer_torch, seed=SEEDS[0])

    per_seed_mean_fg = []
    for seed_idx, seed in enumerate(SEEDS, start=1):
        x0_batch, mask_batch, intensity_batch, layer1_orbit = _batched_layer1_shared_x0(images, seed, k1)
        mask_float = (~mask_batch).to(torch.float64)
        x0_masked = (x0_batch * mask_float).to(torch.complex128)

        omega2_eff = (intensity_batch + delta_omega.unsqueeze(1)) * mask_float
        with torch.no_grad():
            x = x0_masked
            layer2_orbit = torch.empty((nrow * ncol, N_IMAGES, 140), dtype=torch.complex128)
            for t in range(140):
                x = _batched_layer2_step(k2, x, mask_float, omega2_eff)
                layer2_orbit[:, :, t] = x

        fg_scores = []
        for i in range(N_IMAGES):
            image_t = torch.from_numpy(np.asarray(images[i, 0], dtype=np.float64))
            states = torch.empty((nrow * ncol, 200), dtype=torch.complex128)
            states[:, 0:60] = layer1_orbit[:, i, :]
            states[:, 60:200] = layer2_orbit[:, i, :]
            states[mask_batch[:, i], 60:200] = torch.nan

            cluster_map, *_ = spatiotemporal_segmentation_torch(
                states, image_t, mask_batch[:, i], nt_mask=60, n_clusters=2
            )
            fg_scores.append(foreground_ari(labels[i], cluster_map.numpy()))

        assert not any(np.isnan(fg_scores)), f"seed={seed}: M1d foreground_ari produced NaN"
        per_seed_mean_fg.append(float(np.mean(fg_scores)))
        _progress(f"seed {seed_idx}/{len(SEEDS)} (seed={seed}) done, mean FG ARI = {per_seed_mean_fg[-1]:.4f}")

    result = {
        "n_images": N_IMAGES,
        "seeds": list(SEEDS),
        "n_files_hash_verified": len(hash_check),
        "m1d_per_seed_mean_foreground_ari": per_seed_mean_fg,
        "m1d_seed_averaged_foreground_ari": float(np.mean(per_seed_mean_fg)),
        "m1d_seed_std_foreground_ari": float(np.std(per_seed_mean_fg, ddof=0)),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _progress("done, written to outputs/arc1cd.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
