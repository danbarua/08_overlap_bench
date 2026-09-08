"""H0a: two halves, both required (DESIGN.labkit.md, amended).

(i) 07's own test suite passes at 18a6064, run from this environment --
this project's own uv-managed venv, not 07's. An environment-parity check:
version specifiers are pinned to match 07's, but separately resolved.

(ii) The driver, calling the reference unmodified, reproduces foreground
ARI == 1.0 exactly on 2shapes at seed 1 and 3shapes at seed 9 -- the
demo's own hardcoded seeds, where that score is a property of those seeds,
not of the method (test_segmentation_objects.py's own docstring). The
natural image has no foreground ARI and is explicitly not part of this
check.
"""

import subprocess
import sys

from overlap_bench.harness_m0 import run_reference_bundled_image
from overlap_bench.reference_repo import REFERENCE_REPO, verify_reference_commit

EXACT_ARI_1 = {
    "2shapes": dict(image_index=0, seed=1, n_clusters=2),
    "3shapes": dict(image_index=0, seed=9, n_clusters=3),
}


def test_h0a_i_reference_test_suite_passes_in_harness_environment():
    verify_reference_commit()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REFERENCE_REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def test_h0a_ii_2shapes_reproduces_exact_ari_one():
    ref = EXACT_ARI_1["2shapes"]
    result = run_reference_bundled_image("2shapes", ref["image_index"])
    assert result.seed == ref["seed"]
    assert result.n_clusters == ref["n_clusters"]
    assert result.scores["foreground_ari"] == 1.0


def test_h0a_ii_3shapes_reproduces_exact_ari_one():
    ref = EXACT_ARI_1["3shapes"]
    result = run_reference_bundled_image("3shapes", ref["image_index"])
    assert result.seed == ref["seed"]
    assert result.n_clusters == ref["n_clusters"]
    assert result.scores["foreground_ari"] == 1.0
