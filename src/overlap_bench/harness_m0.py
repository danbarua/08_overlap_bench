"""M0: the reference implementation's pipeline, unmodified. Two RNG paths,
never conflated -- see DESIGN.labkit.md's protocol note and the H0a/arc1a
distinction it draws.

``run_reference_bundled_image`` calls 07_posn's own
``run_segmentation_example`` directly: no reimplementation, 07's own
``numpy.random.RandomState(seed)`` x0 construction, used only for H0a's
identity check against 07's own bundled images.

``run_on_cae_image`` calls ``run_2layer_torch`` /
``spatiotemporal_segmentation_torch`` directly with ``torch.Generator()
.manual_seed(seed)`` for x0, per DESIGN.labkit.md's locked protocol table.
This is the path Arc 1a uses on the CAE benchmark; it is a different x0
convention from ``run_reference_bundled_image`` on purpose, because it is
answering a different question (a new benchmark under a newly-locked
protocol, not reproducing 07's own certified numbers).
"""



from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import label
from sklearn.metrics import adjusted_rand_score

from overlap_bench.reference_repo import ensure_on_path


def probe_seed(image_index: int) -> int:
    """Seed mapping the original probe used: torch.Generator().manual_seed(1000 + i)
    per image index i, image-index-derived, NOT the protocol's seeds 1..10.

    Exists only to reconstruct the probe's fixed-dynamics numbers (0.062 on
    2shapes, 0.143 on MNIST_shapes) for traceability. arc1a uses seeds 1..10
    directly, one draw per (image, seed) -- a different, already-locked
    convention (DESIGN.labkit.md protocol table). Do not use this mapping
    for arc1a; do not use 1..10 to reconstruct the probe.
    """
    return 1000 + image_index


@dataclass(frozen=True)
class BundledImageResult:
    dataset: str
    image_index: int
    seed: int
    n_clusters: int
    scores: dict[str, float]


def run_reference_bundled_image(dataset: str, image_index: int = 0) -> BundledImageResult:
    """H0a path. Calls 07's own run_segmentation_example, unmodified."""
    ensure_on_path()
    from src.cv_rnn.segmentation_demo import run_segmentation_example

    example = run_segmentation_example(dataset, image_index)
    return BundledImageResult(
        dataset=dataset,
        image_index=image_index,
        seed=example.seed,
        n_clusters=example.n_clusters,
        scores=example.scores(),
    )


@dataclass(frozen=True)
class CaeRunResult:
    seed: int
    n_clusters: int
    predicted: np.ndarray  # image-shaped, -1 = background


def run_on_cae_image(image: np.ndarray, seed: int, n_clusters: int = 2) -> CaeRunResult:
    """Arc 1a path. torch.Generator x0, per DESIGN.labkit.md's protocol lock.

    ``n_clusters`` defaults to 2, the locked value -- not derived from
    ground truth here, matching the M0 locks note.
    """
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import (
        run_2layer_torch,
        spatiotemporal_segmentation_torch,
    )

    image_t = torch.from_numpy(np.asarray(image, dtype=np.float64))
    generator = torch.Generator().manual_seed(seed)
    states, mask = run_2layer_torch(image_t, generator=generator)
    cluster_map, *_ = spatiotemporal_segmentation_torch(
        states, image_t, mask, nt_mask=60, n_clusters=n_clusters
    )
    return CaeRunResult(seed=seed, n_clusters=n_clusters, predicted=cluster_map.numpy())


def connected_components_baseline(image: np.ndarray) -> np.ndarray:
    """``cc``: scipy.ndimage.label on the thresholded image, 4-connectivity.

    Pinned by H0b, not a guess to iterate on: this is exactly what the
    probe that motivated the question ran. Returns 1-labelled components
    (0 = background), the raw scipy.ndimage.label convention -- callers
    compare against ground truth using ``labels > 0`` / ``labels != -1``,
    same as foreground_ari below.
    """
    predicted, _ = label(np.asarray(image) > 0)
    return predicted


def foreground_ari(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Foreground ARI: ARI over pixels whose true label is positive.

    Overlap (-1) and background (0) pixels are excluded. DESIGN.labkit.md
    "On the question: the meter".
    """
    foreground = truth > 0
    if foreground.sum() < 2:
        return float("nan")
    return float(adjusted_rand_score(truth[foreground], predicted[foreground]))


def valid_region_ari(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Whole-image ARI over pixels that are not overlap. Diagnostic only;
    DESIGN.labkit.md: "enters no verdict"."""
    valid = truth != -1
    if valid.sum() < 2:
        return float("nan")
    return float(adjusted_rand_score(truth[valid], predicted[valid]))
