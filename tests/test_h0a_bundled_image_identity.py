"""H0a: M0 on the reference implementation's own bundled images reproduces
07's own recorded numbers, under 07's own seeding (numpy RandomState via
run_segmentation_example) -- not the CAE/torch.Generator path.

Locked reference numbers, established by running run_reference_bundled_image
directly (this session, before any Arc 1 work): 2shapes/0 and 3shapes/0 match
07's own test_segmentation_objects.py (foreground ARI == 1.0 at the hardcoded
demo seeds 1 and 9). natural/0 has no prior recorded value anywhere in 07;
its whole-image ARI is recorded here as the reference, not assumed.
"""

from overlap_bench.harness_m0 import run_reference_bundled_image

REFERENCE = {
    "2shapes": dict(image_index=0, seed=1, n_clusters=2, foreground_ari=1.0),
    "3shapes": dict(image_index=0, seed=9, n_clusters=3, foreground_ari=1.0),
}
REFERENCE_NATURAL_ARI = 0.7431811811234751


def test_h0a_2shapes_reproduces_recorded_foreground_ari():
    ref = REFERENCE["2shapes"]
    result = run_reference_bundled_image("2shapes", ref["image_index"])
    assert result.seed == ref["seed"]
    assert result.n_clusters == ref["n_clusters"]
    assert result.scores["foreground_ari"] == ref["foreground_ari"]


def test_h0a_3shapes_reproduces_recorded_foreground_ari():
    ref = REFERENCE["3shapes"]
    result = run_reference_bundled_image("3shapes", ref["image_index"])
    assert result.seed == ref["seed"]
    assert result.n_clusters == ref["n_clusters"]
    assert result.scores["foreground_ari"] == ref["foreground_ari"]


def test_h0a_natural_reproduces_recorded_whole_image_ari():
    result = run_reference_bundled_image("natural", 0)
    assert result.seed == 1
    assert abs(result.scores["ari"] - REFERENCE_NATURAL_ARI) < 1e-12
