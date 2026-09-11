"""Does the trained layer 2 reach the same partition regardless of the phase seed?

pursuit-b's 10,000-step run returned bit-identical foreground ARI for all ten
eval seeds (ART_17: one distinct value, std 1.11e-16), where the 500-step run
returned ten distinct values. The seed demonstrably reaches the eval, so a dead
seed loop is ruled out -- but identical *scores* do not establish identical
*partitions*. Two different partitions can score the same.

This checks the partitions directly. For each val image it runs M0's readout
under two different phase seeds and compares the resulting cluster maps.

Two comparisons, because only one of them means what it looks like:

  raw:  np.array_equal on the cluster maps. Sensitive to cluster-id relabelling,
        which is not a difference in the partition -- spectral clustering is
        free to name the same two groups (0,1) or (1,0).
  partition: identical up to relabelling. This is the one that answers the
        question. A run can fail `raw` and pass `partition` and nothing is
        wrong; failing `partition` is the result that would falsify
        convergence-to-one-attractor.

For any pixel that differs, the breakdown says which ground-truth class it
belongs to. That matters: mismatches confined to background or to the excluded
overlap pixels would falsify strict identity while still explaining why the
foreground ARI came out the same, because foreground ARI does not score those
pixels. That outcome is more informative than either clean answer.

Every eval seed's cluster maps are kept rather than reduced to a score, and
every one of the C(10,2)=45 seed pairs is compared. A single pair only rules
out the trivial bug; the full pairwise picture at two budgets is what separates
gradual convergence from a step function.

    PYTHONPATH=src uv run --locked python scripts/run_label_identity.py \
        --checkpoint outputs/pursuit-b-full.pt --device cuda --seeds 10 \
        --out outputs/pursuit-b-label-identity-full.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pursuit_b import (  # noqa: E402
    BASELINES,
    DATASET,
    LAYER1_STEPS,
    Layer2,
    _progress,
    _sheets,
)

from overlap_bench.dataset_hashes import CAE_DIR, verify_locked_dataset_hashes  # noqa: E402
from overlap_bench.harness_m0 import foreground_ari  # noqa: E402
from overlap_bench.reference_repo import ensure_on_path  # noqa: E402


def _cluster_map(model: Layer2, image: torch.Tensor, seed: int, device: torch.device):
    """M0's readout on the trained layer-2 orbit, exactly as _evaluate does it."""
    from src.cv_rnn.cv_rnn_segmentation import run_2layer_torch, spatiotemporal_segmentation_torch

    generator = torch.Generator().manual_seed(seed)
    states_m0, mask = run_2layer_torch(image, generator=generator)
    x0 = states_m0[:, 0].masked_fill(mask, 0)
    omega = image.T.reshape(-1)
    with torch.no_grad():
        trained = model.orbit_all(
            x0.to(device).unsqueeze(0),
            omega.to(device).unsqueeze(0),
            mask.to(device).unsqueeze(0),
        )
    states = states_m0.clone()
    states[:, LAYER1_STEPS:200] = trained[0].cpu()
    states[mask, LAYER1_STEPS:200] = torch.nan
    cluster_map, *_ = spatiotemporal_segmentation_torch(
        states, image, mask, nt_mask=LAYER1_STEPS, n_clusters=2
    )
    return cluster_map.numpy()


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff a and b induce the same partition, ignoring cluster names.

    Equivalent to: the pixel-wise map a->b is a bijection. If any a-value maps
    to two different b-values (or vice versa) the groupings genuinely differ.
    """
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return False  # the masked-out sets themselves differ
    pairs = set(zip(a[finite].tolist(), b[finite].tolist()))
    forward = {p[0] for p in pairs}
    backward = {p[1] for p in pairs}
    return len(pairs) == len(forward) == len(backward)


def _mismatch_breakdown(a: np.ndarray, b: np.ndarray, truth: np.ndarray) -> dict:
    """Where the two cluster maps disagree, by ground-truth class.

    Labels follow the protocol convention: 0 background, >0 object, -1 the
    excluded overlap. Foreground ARI scores only the >0 pixels, so a
    disagreement living entirely outside them explains identical scores
    without identical partitions.
    """
    differs = ~((a == b) | (~np.isfinite(a) & ~np.isfinite(b)))
    n = int(differs.sum())
    if n == 0:
        return {"n_pixels_differing": 0}
    return {
        "n_pixels_differing": n,
        "on_background_label_0": int((differs & (truth == 0)).sum()),
        "on_object_labels_positive": int((differs & (truth > 0)).sum()),
        "on_excluded_overlap_label_minus_1": int((differs & (truth == -1)).sum()),
        "scored_by_foreground_ari": int((differs & (truth > 0)).sum()),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(BASELINES),
        help=(
            "defaults to the dataset recorded in the checkpoint, which is the "
            "only value that can be right. Pass it only to assert that value; a "
            "mismatch raises rather than quietly evaluating a model against "
            "another benchmark's images."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="compare eval seeds 1..N pairwise, all C(N,2) pairs (locked eval uses 10)",
    )
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[1] / "outputs" / "label-identity.json")
    )
    args = parser.parse_args(argv)

    # The checkpoint knows which benchmark it was trained on; read it there
    # rather than trusting a flag to agree with it.
    blob = torch.load(args.checkpoint, map_location="cpu")
    dataset = blob.get("dataset")
    if dataset not in BASELINES:
        raise ValueError(
            f"checkpoint does not record a known dataset (got {dataset!r}). It "
            "predates --save-model's dataset metadata, or was written by "
            "something else; either way what it should be evaluated against is "
            "not knowable from the file."
        )
    if args.dataset is not None and args.dataset != dataset:
        raise ValueError(
            f"checkpoint was trained on {dataset!r} but --dataset says "
            f"{args.dataset!r}. Comparing partitions across datasets would "
            "produce a number about nothing."
        )

    verified = verify_locked_dataset_hashes(only=[f"{dataset}_val.npz"])
    device = torch.device(args.device)
    ensure_on_path()

    with np.load(CAE_DIR / f"{dataset}_val.npz") as data:
        images_np = np.asarray(data["images"][: args.images, 0], dtype=np.float64)
        labels_np = np.asarray(data["labels"][: args.images], dtype=np.int64)
    n_images, nrow, ncol = images_np.shape

    _, k2 = _sheets(nrow, ncol, device)
    model = Layer2(k2, nrow * ncol).to(device)
    model.load_state_dict({k: v.to(device) for k, v in blob["state_dict"].items()})
    _progress(
        f"loaded checkpoint trained {blob['steps']} steps on {blob['train_images']} images"
    )

    seeds = list(range(1, args.seeds + 1))
    started = time.monotonic()

    # Every seed's cluster map for every image, kept rather than reduced to a
    # score. The eval loop already computes these; discarding them is what made
    # the question unanswerable from ART_17 alone.
    maps: dict[int, list[np.ndarray]] = {}
    ari: dict[int, list[float]] = {}
    for seed in seeds:
        maps[seed] = []
        ari[seed] = []
        for i in range(n_images):
            cluster_map = _cluster_map(model, torch.from_numpy(images_np[i]), seed, device)
            maps[seed].append(cluster_map)
            ari[seed].append(foreground_ari(labels_np[i], cluster_map))
        _progress(f"  seed {seed}: mean FG ARI {float(np.mean(ari[seed])):+.6f}")

    pairs = []
    for a_idx, seed_a in enumerate(seeds):
        for seed_b in seeds[a_idx + 1 :]:
            mismatches = []
            n_raw = n_part = 0
            for i in range(n_images):
                map_a, map_b = maps[seed_a][i], maps[seed_b][i]
                raw = bool(np.array_equal(map_a, map_b, equal_nan=True))
                part = _same_partition(map_a, map_b)
                n_raw += raw
                n_part += part
                if not part:
                    mismatches.append(
                        {"image": i, **_mismatch_breakdown(map_a, map_b, labels_np[i])}
                    )
            pairs.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "n_images_raw_identical": n_raw,
                    "n_images_same_partition": n_part,
                    "n_images_foreground_ari_identical": sum(
                        ari[seed_a][i] == ari[seed_b][i] for i in range(n_images)
                    ),
                    "partition_mismatches": mismatches,
                }
            )

    n_pairs = len(pairs)
    all_part = sum(p["n_images_same_partition"] == n_images for p in pairs)
    all_raw = sum(p["n_images_raw_identical"] == n_images for p in pairs)
    scored_diffs = sum(
        m.get("scored_by_foreground_ari", 0) for p in pairs for m in p["partition_mismatches"]
    )
    per_seed_mean = [float(np.mean(ari[s])) for s in seeds]

    output = {
        "question": (
            "does the trained layer 2 reach the same partition under different "
            "phase seeds, or merely the same score?"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_steps": blob["steps"],
        "device": str(device),
        "seeds": seeds,
        "n_pairs": n_pairs,
        "n_images": n_images,
        "dataset_hashes_verified": verified,
        "n_pairs_all_images_same_partition": all_part,
        "n_pairs_all_images_raw_identical": all_raw,
        "n_pixels_differing_that_foreground_ari_scores": scored_diffs,
        "per_seed_mean_foreground_ari": per_seed_mean,
        "n_distinct_per_seed_means": len(set(per_seed_mean)),
        "elapsed_seconds": time.monotonic() - started,
        "pairs": pairs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _progress(f"wrote {out}")
    return output


if __name__ == "__main__":
    result = main()
    print(
        f"\n{result['checkpoint_steps']} steps: partitions identical on all "
        f"{result['n_images']} images for {result['n_pairs_all_images_same_partition']}"
        f"/{result['n_pairs']} seed pairs "
        f"(raw, ignoring relabelling: {result['n_pairs_all_images_raw_identical']}"
        f"/{result['n_pairs']}); "
        f"{result['n_distinct_per_seed_means']} distinct per-seed mean ARI"
    )
    if (
        result["n_pixels_differing_that_foreground_ari_scores"] == 0
        and result["n_pairs_all_images_same_partition"] < result["n_pairs"]
    ):
        print(
            "partitions differ, but nowhere foreground ARI scores -- identical "
            "scores do NOT imply one attractor here"
        )
