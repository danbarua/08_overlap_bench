"""How-to: validate a new evaluation pipeline before trusting a suspicious
result from it.

Context this came from: arc1c-d's first full run reported a chance-level
(slightly negative) foreground ARI for the trained M1d model. That result
was surprising enough to be alarming -- was the training genuinely bad, or
was the new splicing/readout/masking code (batched layer-1, layer-2 orbit,
NaN-fill, readout call) simply wrong? See NOTE_26 in this project's LabKit
record: "without [this check] the chance-level number was a bug until proven
otherwise; with it, it is a finding."

The technique, generalizable to any "new pipeline, suspicious result"
situation: before trusting the new result, run the IDENTICAL new pipeline
with OLD/KNOWN parameters (here: the untrained model, K_2 = initial sheet,
delta_omega = 0) and confirm it reproduces a result you already trust from
an independent, previously-verified source (here: arc1a's own recorded
per-seed numbers, computed by a completely different code path).

- If the oracle pass reproduces the known-good numbers exactly: the new
  pipeline's plumbing is correct, and the suspicious result is real --
  proceed to explain it, not to hunt for a bug in the pipeline.
- If the oracle pass does NOT reproduce the known-good numbers: the bug is
  in the new pipeline (splicing, masking, indexing, readout call, ...), not
  in whatever you were actually trying to evaluate. Fix that first.

Only after this passed did this project treat "M1d scores at chance" as a
finding worth investigating further (which, several corrections later,
turned out to trace to a loss-formula bug -- see NOTE_40 and
check_surrogate_at_baseline.py in this directory).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import overlap_bench.run_arc1cd as arc1cd
from overlap_bench.reference_repo import ensure_on_path

ensure_on_path()
from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch  # noqa: E402

# The "already trusted" reference: arc1a's own recorded per-seed 2shapes
# numbers (outputs/arc1a.json), computed by a completely independent code
# path (harness_m0.py's run_on_cae_image, not this project's batched
# layer-1/layer-2 reimplementation).
ARC1A_2SHAPES_PER_SEED = [
    0.08275986885554312, 0.12081303720127806, 0.18019217520344558,
    0.1592501000011282, 0.15676890081152461, 0.1536487441500005,
    0.04596555123311932, 0.08904092509627003, 0.12685934279401315,
    0.09327513720780711,
]


def run_oracle_pass(n_images: int = 50, n_seeds: int = 10) -> list[float]:
    """The new eval pipeline (batched layer1/layer2, splicing, readout),
    fed the UNTRAINED (oracle) parameters. Should reproduce ARC1A_2SHAPES_PER_SEED
    exactly, since untrained K_2/delta_omega=0 makes M1d identical to M0."""
    from src.cv_rnn.cv_rnn_segmentation import run_2layer_torch, spatiotemporal_segmentation_torch
    from overlap_bench.dataset_hashes import CAE_DIR
    from overlap_bench.harness_m0 import foreground_ari

    with np.load(CAE_DIR / "2shapes_val.npz") as data:
        images = data["images"][:n_images]
        labels = data["labels"][:n_images]

    nrow, ncol = images.shape[-2], images.shape[-1]
    k1 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9, dtype=torch.float64).real.clone()
    k2_oracle = gaussian_sheet_torch(nrow, ncol, 0.5, 0.0313, dtype=torch.float64).real.clone()
    delta_omega_oracle = torch.zeros(nrow * ncol, dtype=torch.float64)

    per_seed_mean_fg = []
    for seed in range(1, n_seeds + 1):
        x0_batch, mask_batch, intensity_batch, layer1_orbit = arc1cd._batched_layer1_shared_x0(images, seed, k1)
        mask_float = (~mask_batch).to(torch.float64)
        x0_masked = (x0_batch * mask_float).to(torch.complex128)
        omega2_eff = (intensity_batch + delta_omega_oracle.unsqueeze(1)) * mask_float
        with torch.no_grad():
            x = x0_masked
            layer2_orbit = torch.empty((nrow * ncol, n_images, 140), dtype=torch.complex128)
            for t in range(140):
                x = arc1cd._batched_layer2_step(k2_oracle, x, mask_float, omega2_eff)
                layer2_orbit[:, :, t] = x

        fg_scores = []
        for i in range(n_images):
            image_t = torch.from_numpy(np.asarray(images[i, 0], dtype=np.float64))
            states = torch.empty((nrow * ncol, 200), dtype=torch.complex128)
            states[:, 0:60] = layer1_orbit[:, i, :]
            states[:, 60:200] = layer2_orbit[:, i, :]
            states[mask_batch[:, i], 60:200] = torch.nan
            cluster_map, *_ = spatiotemporal_segmentation_torch(
                states, image_t, mask_batch[:, i], nt_mask=60, n_clusters=2
            )
            fg_scores.append(foreground_ari(labels[i], cluster_map.numpy()))
        per_seed_mean_fg.append(float(np.mean(fg_scores)))
        print(f"  seed {seed}: oracle mean FG ARI = {per_seed_mean_fg[-1]:.6f}")

    return per_seed_mean_fg


if __name__ == "__main__":
    print("Running the new eval pipeline with OLD/KNOWN (untrained) parameters...")
    oracle_result = run_oracle_pass()

    max_diff = max(abs(a - b) for a, b in zip(oracle_result, ARC1A_2SHAPES_PER_SEED))
    print(f"\nMax abs diff vs arc1a's independently-computed per-seed numbers: {max_diff:.2e}")
    if max_diff == 0.0:
        print("EXACT MATCH -- the new pipeline's plumbing is correct.")
        print("A suspicious result from this pipeline under DIFFERENT parameters is a real")
        print("finding, not a bug -- go explain it.")
    else:
        print("MISMATCH -- the bug is in the new pipeline, not in whatever you were evaluating.")
