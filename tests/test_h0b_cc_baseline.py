"""H0b: cc reproduces the probe's two numbers (0.160, 0.016) exactly, on the
same 50 val images. This is the check on the FG-ARI meter, not on cc itself
-- see DESIGN.labkit.md's meter note. The convention pinned below (threshold
> 0, 4-connectivity, foreground_ari on labels > 0) is not a guess: it is
what the probe that motivated the question actually ran, in this session.
"""

import numpy as np

from overlap_bench.dataset_hashes import CAE_DIR
from overlap_bench.harness_m0 import connected_components_baseline, foreground_ari

PROBE_RECORDED_FG_ARI = {
    "2shapes": 0.160,
    "MNIST_shapes": 0.016,
}


def _mean_cc_foreground_ari(dataset: str, n: int = 50) -> float:
    with np.load(CAE_DIR / f"{dataset}_val.npz") as data:
        images, labels = data["images"], data["labels"]
    scores = []
    for i in range(n):
        predicted = connected_components_baseline(images[i, 0])
        scores.append(foreground_ari(labels[i], predicted))
    return float(np.mean(scores))


def test_h0b_2shapes_cc_reproduces_probe_number():
    assert round(_mean_cc_foreground_ari("2shapes"), 3) == PROBE_RECORDED_FG_ARI["2shapes"]


def test_h0b_mnist_shapes_cc_reproduces_probe_number():
    assert (
        round(_mean_cc_foreground_ari("MNIST_shapes"), 3)
        == PROBE_RECORDED_FG_ARI["MNIST_shapes"]
    )
