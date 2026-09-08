"""H0b: cc, computed from the probe's own expressions, gives mean FG ARI
over the 50 val images within 1e-12 absolute of the locked values --
0.16 on 2shapes, 0.01552888819008583 on MNIST_shapes (DESIGN.labkit.md,
amended). These are full-precision float64 means, produced by this
harness's first run; the probe reported them at 3 dp. The tolerance is
for last-ulp drift across BLAS builds, not a marginal band.
"""

import numpy as np

from overlap_bench.dataset_hashes import CAE_DIR
from overlap_bench.harness_m0 import connected_components_baseline, foreground_ari

LOCKED_FG_ARI = {
    "2shapes": 0.16,
    "MNIST_shapes": 0.01552888819008583,
}
TOLERANCE = 1e-12


def _mean_cc_foreground_ari(dataset: str, n: int = 50) -> float:
    with np.load(CAE_DIR / f"{dataset}_val.npz") as data:
        images, labels = data["images"], data["labels"]
    scores = []
    for i in range(n):
        predicted = connected_components_baseline(images[i, 0])
        scores.append(foreground_ari(labels[i], predicted))
    return float(np.mean(scores))


def test_h0b_2shapes_cc_matches_locked_value_exactly():
    mean = _mean_cc_foreground_ari("2shapes")
    assert abs(mean - LOCKED_FG_ARI["2shapes"]) < TOLERANCE, mean


def test_h0b_mnist_shapes_cc_matches_locked_value_exactly():
    mean = _mean_cc_foreground_ari("MNIST_shapes")
    assert abs(mean - LOCKED_FG_ARI["MNIST_shapes"]) < TOLERANCE, mean
