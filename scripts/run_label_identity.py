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

    PYTHONPATH=src uv run --locked python scripts/run_label_identity.py \
        --checkpoint outputs/pursuit-b-full.pt --device cuda \
        --out outputs/pursuit-b-label-identity.json
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
    parser.add_argument("--seed-a", type=int, default=1)
    parser.add_argument("--seed-b", type=int, default=2)
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[1] / "outputs" / "label-identity.json")
    )
    args = parser.parse_args(argv)

    verified = verify_locked_dataset_hashes(only=[f"{DATASET}_val.npz"])
    device = torch.device(args.device)
    ensure_on_path()

    with np.load(CAE_DIR / f"{DATASET}_val.npz") as data:
        images_np = np.asarray(data["images"][: args.images, 0], dtype=np.float64)
        labels_np = np.asarray(data["labels"][: args.images], dtype=np.int64)
    n_images, nrow, ncol = images_np.shape

    blob = torch.load(args.checkpoint, map_location="cpu")
    _, k2 = _sheets(nrow, ncol, device)
    model = Layer2(k2, nrow * ncol).to(device)
    model.load_state_dict({k: v.to(device) for k, v in blob["state_dict"].items()})
    _progress(
        f"loaded checkpoint trained {blob['steps']} steps on {blob['train_images']} images"
    )

    started = time.monotonic()
    per_image = []
    for i in range(n_images):
        image = torch.from_numpy(images_np[i])
        map_a = _cluster_map(model, image, args.seed_a, device)
        map_b = _cluster_map(model, image, args.seed_b, device)
        row = {
            "image": i,
            "raw_labels_identical": bool(np.array_equal(map_a, map_b, equal_nan=True)),
            "same_partition_up_to_relabelling": _same_partition(map_a, map_b),
            "foreground_ari_seed_a": foreground_ari(labels_np[i], map_a),
            "foreground_ari_seed_b": foreground_ari(labels_np[i], map_b),
        }
        row["foreground_ari_identical"] = (
            row["foreground_ari_seed_a"] == row["foreground_ari_seed_b"]
        )
        row["mismatch"] = _mismatch_breakdown(map_a, map_b, labels_np[i])
        per_image.append(row)
        if (i + 1) % 10 == 0 or i + 1 == n_images:
            _progress(f"  compared {i + 1}/{n_images}")

    n_raw = sum(r["raw_labels_identical"] for r in per_image)
    n_part = sum(r["same_partition_up_to_relabelling"] for r in per_image)
    n_ari = sum(r["foreground_ari_identical"] for r in per_image)
    scored_diffs = sum(r["mismatch"].get("scored_by_foreground_ari", 0) for r in per_image)

    output = {
        "question": (
            "does the trained layer 2 reach the same partition under different "
            "phase seeds, or merely the same score?"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_steps": blob["steps"],
        "device": str(device),
        "seed_a": args.seed_a,
        "seed_b": args.seed_b,
        "n_images": n_images,
        "dataset_hashes_verified": verified,
        "n_raw_labels_identical": n_raw,
        "n_same_partition_up_to_relabelling": n_part,
        "n_foreground_ari_identical": n_ari,
        "n_pixels_differing_that_foreground_ari_scores": scored_diffs,
        "elapsed_seconds": time.monotonic() - started,
        "per_image": per_image,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _progress(f"wrote {out}")
    return output


if __name__ == "__main__":
    result = main()
    print(
        f"\npartition identical up to relabelling on "
        f"{result['n_same_partition_up_to_relabelling']}/{result['n_images']} images; "
        f"raw labels identical on {result['n_raw_labels_identical']}/{result['n_images']}; "
        f"FG ARI identical on {result['n_foreground_ari_identical']}/{result['n_images']}"
    )
    if result["n_pixels_differing_that_foreground_ari_scores"] == 0 and (
        result["n_same_partition_up_to_relabelling"] < result["n_images"]
    ):
        print(
            "partitions differ, but nowhere foreground ARI scores -- identical "
            "scores do NOT imply one attractor here"
        )
