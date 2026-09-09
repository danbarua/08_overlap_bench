"""arc1a: M0 at default parameters vs cc, seeds 1..10, both Arc-1 datasets.

Reported statistic (protocol lock): mean FG ARI over the 50 val images,
per seed, then mean and standard deviation of those 10 per-seed means
(ddof=0, per NOTE_11/NOTE_12 -- sharpened on the record before this ran).
cc is seed-free (deterministic); its number is a single scalar per dataset.

Valid-region ARI (truth != -1) recorded as a diagnostic only --
DESIGN.labkit.md: "enters no verdict".

No wall-clock time anywhere in the output artefact: a re-run with the
same code and data must hash identically. This run takes ~20 minutes;
progress prints to stderr every seed so liveness is checkable without
polling process CPU time.
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

OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "arc1a.json"
DATASETS = ("2shapes", "MNIST_shapes")
SEEDS = tuple(range(1, 11))
N_IMAGES = 50


def _progress(message: str) -> None:
    print(f"[arc1a] {message}", file=sys.stderr, flush=True)


def _dataset_result(dataset: str) -> dict:
    with np.load(CAE_DIR / f"{dataset}_val.npz") as data:
        images, labels = data["images"], data["labels"]
    images = images[:N_IMAGES]
    labels = labels[:N_IMAGES]

    cc_fg = [foreground_ari(labels[i], connected_components_baseline(images[i, 0])) for i in range(N_IMAGES)]
    cc_valid = [valid_region_ari(labels[i], connected_components_baseline(images[i, 0])) for i in range(N_IMAGES)]
    assert not any(np.isnan(cc_fg)), f"{dataset}: cc foreground_ari produced NaN (fewer than 2 foreground pixels)"

    per_seed_mean_fg = []
    per_seed_mean_valid = []
    for seed_idx, seed in enumerate(SEEDS, start=1):
        fg_scores = []
        valid_scores = []
        for i in range(N_IMAGES):
            result = run_on_cae_image(images[i, 0], seed=seed, n_clusters=2)
            fg_scores.append(foreground_ari(labels[i], result.predicted))
            valid_scores.append(valid_region_ari(labels[i], result.predicted))
        assert not any(np.isnan(fg_scores)), f"{dataset} seed={seed}: M0 foreground_ari produced NaN"
        per_seed_mean_fg.append(float(np.mean(fg_scores)))
        per_seed_mean_valid.append(float(np.mean(valid_scores)))
        _progress(f"{dataset}: seed {seed_idx}/{len(SEEDS)} done ({seed_idx * N_IMAGES}/{len(SEEDS) * N_IMAGES} images)")

    return {
        "cc_mean_foreground_ari": float(np.mean(cc_fg)),
        "cc_mean_valid_region_ari_diagnostic": float(np.mean(cc_valid)),
        "m0_per_seed_mean_foreground_ari": per_seed_mean_fg,
        "m0_seed_averaged_foreground_ari": float(np.mean(per_seed_mean_fg)),
        "m0_seed_std_foreground_ari": float(np.std(per_seed_mean_fg, ddof=0)),
        "m0_per_seed_mean_valid_region_ari_diagnostic": per_seed_mean_valid,
        "m0_seed_averaged_valid_region_ari_diagnostic": float(np.mean(per_seed_mean_valid)),
    }


def main() -> dict:
    verify_reference_commit()
    hash_check = verify_locked_dataset_hashes()
    output = {"n_images": N_IMAGES, "seeds": list(SEEDS), "datasets": {}}
    for dataset in DATASETS:
        _progress(f"starting {dataset}")
        output["datasets"][dataset] = _dataset_result(dataset)
    output["n_files_hash_verified"] = len(hash_check)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _progress("done, written to outputs/arc1a.json")
    return output


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
