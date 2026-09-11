"""3shapes baseline: M0 at default parameters vs cc, seeds 1..10.

Not part of arc1a: 3shapes has 3 objects per image (2shapes and MNIST_shapes
have 2), and arc1a's own scope note says "3shapes, where the two would
differ, is out of scope" -- arc1a hardcodes n_clusters=2 throughout.

This script is the minimum needed before any pursuit-b-style training on
3shapes is meaningful: it measures the comparators (cc, M0 at defaults)
under n_clusters=3, matching every image's actual object count, so a future
training result has a prespecified bar to be held to rather than one chosen
after the number lands. It does not touch run_arc1a.py or arc1a.json, which
stay locked and byte-identical.

Same protocol lock as arc1a otherwise: 50 val images, seeds 1..10, mean FG
ARI per seed then mean/std over seeds (ddof=0, NOTE_11/NOTE_12). cc is
seed-free. Valid-region ARI recorded as a diagnostic only, per DESIGN.labkit.md.

No wall-clock time in the output artefact: a re-run with the same code and
data must hash identically.
"""

import json
import sys
from pathlib import Path

import numpy as np

from overlap_bench.dataset_hashes import CAE_DIR, verify_locked_dataset_hashes
from overlap_bench.harness_m0 import (
    connected_components_baseline,
    foreground_ari,
    run_on_cae_image,
    valid_region_ari,
)
from overlap_bench.paths import ROOT_DIR
from overlap_bench.reference_repo import verify_reference_commit

OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "3shapes-baseline.json"
DATASET = "3shapes"
N_CLUSTERS = 3
SEEDS = tuple(range(1, 11))
N_IMAGES = 50


def _progress(message: str) -> None:
    print(f"[3shapes-baseline] {message}", file=sys.stderr, flush=True)


def main() -> dict:
    verify_reference_commit()
    hash_check = verify_locked_dataset_hashes(
        only=[f"{DATASET}_train.npz", f"{DATASET}_val.npz"]
    )

    with np.load(CAE_DIR / f"{DATASET}_val.npz") as data:
        images, labels = data["images"], data["labels"]
    images = images[:N_IMAGES]
    labels = labels[:N_IMAGES]

    object_counts = [int(labels[i][labels[i] > 0].max()) for i in range(N_IMAGES)]
    assert all(c == N_CLUSTERS for c in object_counts), (
        f"expected every image to have exactly {N_CLUSTERS} objects; "
        f"got counts {sorted(set(object_counts))}"
    )

    cc_fg = [foreground_ari(labels[i], connected_components_baseline(images[i, 0])) for i in range(N_IMAGES)]
    cc_valid = [valid_region_ari(labels[i], connected_components_baseline(images[i, 0])) for i in range(N_IMAGES)]
    assert not any(np.isnan(cc_fg)), "cc foreground_ari produced NaN (fewer than 2 foreground pixels)"

    per_seed_mean_fg = []
    per_seed_mean_valid = []
    for seed_idx, seed in enumerate(SEEDS, start=1):
        fg_scores = []
        valid_scores = []
        for i in range(N_IMAGES):
            result = run_on_cae_image(images[i, 0], seed=seed, n_clusters=N_CLUSTERS)
            fg_scores.append(foreground_ari(labels[i], result.predicted))
            valid_scores.append(valid_region_ari(labels[i], result.predicted))
        assert not any(np.isnan(fg_scores)), f"seed={seed}: M0 foreground_ari produced NaN"
        per_seed_mean_fg.append(float(np.mean(fg_scores)))
        per_seed_mean_valid.append(float(np.mean(valid_scores)))
        _progress(f"seed {seed_idx}/{len(SEEDS)} done ({seed_idx * N_IMAGES}/{len(SEEDS) * N_IMAGES} images)")

    output = {
        "dataset": DATASET,
        "n_clusters": N_CLUSTERS,
        "n_images": N_IMAGES,
        "seeds": list(SEEDS),
        "cc_mean_foreground_ari": float(np.mean(cc_fg)),
        "cc_mean_valid_region_ari_diagnostic": float(np.mean(cc_valid)),
        "m0_per_seed_mean_foreground_ari": per_seed_mean_fg,
        "m0_seed_averaged_foreground_ari": float(np.mean(per_seed_mean_fg)),
        "m0_seed_std_foreground_ari": float(np.std(per_seed_mean_fg, ddof=0)),
        "m0_per_seed_mean_valid_region_ari_diagnostic": per_seed_mean_valid,
        "m0_seed_averaged_valid_region_ari_diagnostic": float(np.mean(per_seed_mean_valid)),
        "n_files_hash_verified": len(hash_check),
        "dataset_hashes_verified": hash_check,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _progress("done, written to outputs/3shapes-baseline.json")
    return output


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
