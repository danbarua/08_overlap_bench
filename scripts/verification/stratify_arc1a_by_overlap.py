"""Decompose arc1a's aggregate score into per-image scores, split by whether
an image actually contains overlap pixels.

Why this exists: `arc1a` reports one mean per seed and then a mean and SD over
seeds. That number cannot distinguish "the method partly works on every image"
from "the method solves some images and never solves others". Those imply
different next experiments, so the distinction is worth measuring rather than
assuming.

The control that makes this trustworthy: the per-seed means this script
computes must reproduce `outputs/arc1a.json`'s own per-seed means exactly. If
they do, this is arc1a's number decomposed; if they do not, the decomposition
is measuring something else and its strata mean nothing. The check is asserted,
not printed.

Writes `outputs/arc1a-stratified.json`. Nothing here is on the LabKit record as
evidence for a criterion -- see NOTE_52 through NOTE_54, which are observations
recorded after the programme closed.

    PYTHONPATH=src uv run --locked python scripts/verification/stratify_arc1a_by_overlap.py
"""

import json
from pathlib import Path

import numpy as np

from overlap_bench.dataset_hashes import CAE_DIR, verify_locked_dataset_hashes
from overlap_bench.harness_m0 import (
    connected_components_baseline,
    foreground_ari,
    run_on_cae_image,
)
from overlap_bench.paths import ROOT_DIR

DATASET = "2shapes"
N_IMAGES = 50
SEEDS = tuple(range(1, 11))
N_CLUSTERS = 2
OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "arc1a-stratified.json"
ARC1A_PATH = Path(ROOT_DIR) / "outputs" / "arc1a.json"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [10, 25, 50, 75, 90])
    return {f"q{p}": float(v) for p, v in zip((10, 25, 50, 75, 90), q)}


def _summary(m0: np.ndarray, cc: np.ndarray) -> dict:
    """m0 is (images, seeds) for one stratum; cc is one score per image."""
    flat = m0.ravel()
    return {
        "n_images": int(m0.shape[0]),
        "n_pairs": int(flat.size),
        "m0_mean": float(flat.mean()),
        "m0_median": float(np.median(flat)),
        **{f"m0_{k}": v for k, v in _quantiles(flat).items()},
        "m0_frac_above_0.5": float((flat > 0.5).mean()),
        "m0_frac_above_0.2": float((flat > 0.2).mean()),
        "m0_frac_at_or_below_0.02": float((flat <= 0.02).mean()),
        "m0_frac_negative": float((flat < 0).mean()),
        "cc_mean": float(cc.mean()),
    }


def main() -> dict:
    verify_locked_dataset_hashes()
    with np.load(CAE_DIR / f"{DATASET}_val.npz") as data:
        images, labels = data["images"][:N_IMAGES], data["labels"][:N_IMAGES]

    # Per image: does it contain overlap-labelled pixels, how many components
    # does thresholding find on the foreground, and what does cc score.
    has_overlap = np.array([bool((labels[i] == -1).any()) for i in range(N_IMAGES)])
    cc_components = np.empty(N_IMAGES, dtype=int)
    cc_score = np.empty(N_IMAGES)
    for i in range(N_IMAGES):
        foreground = labels[i] > 0
        predicted = connected_components_baseline(images[i, 0])
        cc_components[i] = len(set(predicted[foreground].tolist()) - {0})
        cc_score[i] = foreground_ari(labels[i], predicted)

    m0 = np.empty((N_IMAGES, len(SEEDS)))
    for s, seed in enumerate(SEEDS):
        for i in range(N_IMAGES):
            result = run_on_cae_image(images[i, 0], seed=seed, n_clusters=N_CLUSTERS)
            m0[i, s] = foreground_ari(labels[i], result.predicted)

    # The control. arc1a computed these independently; if this decomposition
    # does not reproduce them, its strata are about some other computation.
    recorded = json.loads(ARC1A_PATH.read_text())["datasets"][DATASET][
        "m0_per_seed_mean_foreground_ari"
    ]
    computed = m0.mean(axis=0)
    for seed, (want, got) in zip(SEEDS, zip(recorded, computed)):
        assert abs(want - got) < 1e-12, (
            f"seed {seed}: this run gives {got!r}, arc1a.json recorded {want!r}; "
            "the decomposition is not of arc1a's number"
        )

    per_image_mean = m0.mean(axis=1)
    output = {
        "dataset": DATASET,
        "n_images": N_IMAGES,
        "seeds": list(SEEDS),
        "reproduces_arc1a_per_seed_means": True,
        "all_pairs": _summary(m0, cc_score),
        "overlap_pixels_present": _summary(m0[has_overlap], cc_score[has_overlap]),
        "no_overlap_pixels": _summary(m0[~has_overlap], cc_score[~has_overlap]),
        "cc_merged_one_component": _summary(
            m0[cc_components == 1], cc_score[cc_components == 1]
        ),
        "cc_found_two_components": _summary(
            m0[cc_components == 2], cc_score[cc_components == 2]
        ),
        "per_image_mean_bands": {
            "above_0.5": {
                "n": int((per_image_mean > 0.5).sum()),
                "n_with_overlap": int(has_overlap[per_image_mean > 0.5].sum()),
            },
            "0.02_to_0.5": {
                "n": int(((per_image_mean > 0.02) & (per_image_mean <= 0.5)).sum()),
                "n_with_overlap": int(
                    has_overlap[
                        (per_image_mean > 0.02) & (per_image_mean <= 0.5)
                    ].sum()
                ),
            },
            "at_or_below_0.02": {
                "n": int((per_image_mean <= 0.02).sum()),
                "n_with_overlap": int(has_overlap[per_image_mean <= 0.02].sum()),
            },
        },
        "per_image": [
            {
                "image": i,
                "has_overlap_pixels": bool(has_overlap[i]),
                "cc_components_on_foreground": int(cc_components[i]),
                "cc_foreground_ari": float(cc_score[i]),
                "m0_per_seed_foreground_ari": [float(v) for v in m0[i]],
                "m0_mean_over_seeds": float(per_image_mean[i]),
            }
            for i in range(N_IMAGES)
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    result = main()
    for name in (
        "all_pairs",
        "overlap_pixels_present",
        "no_overlap_pixels",
        "cc_merged_one_component",
        "cc_found_two_components",
    ):
        s = result[name]
        print(
            f"{name:26s} images={s['n_images']:2d} pairs={s['n_pairs']:3d} "
            f"m0_mean={s['m0_mean']:+.4f} m0_median={s['m0_median']:+.4f} "
            f"m0>0.5={s['m0_frac_above_0.5']:.3f} cc={s['cc_mean']:.4f}"
        )
    print(f"per-image bands: {json.dumps(result['per_image_mean_bands'])}")
