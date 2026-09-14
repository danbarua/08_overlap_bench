"""Runs H0a/H0b/H0c and writes their real output to outputs/harness-m0.json.

Exists so labkit's content_hash points at an actual artefact -- a file a
future re-run can be checked against -- rather than a hash of an input or
a hand-typed proxy string. No wall-clock time anywhere in the output: a
re-run with the same code and data must hash identically.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from overlap_bench.dataset_hashes import CAE_DIR, verify_locked_dataset_hashes
from overlap_bench.harness_m0 import (
    connected_components_baseline,
    foreground_ari,
    run_reference_bundled_image,
)
from overlap_bench.paths import REFERENCE_REPO, ROOT_DIR
from overlap_bench.reference_repo import verify_reference_commit

OUTPUT_PATH = Path(ROOT_DIR) / "outputs" / "harness-m0.json"


def _h0a() -> dict:
    commit = verify_reference_commit()
    suite = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REFERENCE_REPO,
        capture_output=True,
        text=True,
    )
    passed_match = re.search(r"(\d+) passed", suite.stdout)
    failed_match = re.search(r"(\d+) failed", suite.stdout)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    two_shapes = run_reference_bundled_image("2shapes", 0)
    three_shapes = run_reference_bundled_image("3shapes", 0)
    return {
        "reference_commit": commit,
        "reference_suite_returncode": suite.returncode,
        "reference_suite_passed": passed,
        "reference_suite_failed": failed,
        "2shapes": {"seed": two_shapes.seed, "n_clusters": two_shapes.n_clusters, **two_shapes.scores},
        "3shapes": {"seed": three_shapes.seed, "n_clusters": three_shapes.n_clusters, **three_shapes.scores},
    }


def _h0b() -> dict:
    result = {}
    for dataset in ("2shapes", "MNIST_shapes"):
        with np.load(CAE_DIR / f"{dataset}_val.npz") as data:
            images, labels = data["images"], data["labels"]
        scores = [
            foreground_ari(labels[i], connected_components_baseline(images[i, 0]))
            for i in range(50)
        ]
        result[dataset] = {"n": 50, "mean_foreground_ari": float(np.mean(scores))}
    return result


def _h0c() -> dict:
    hashes = verify_locked_dataset_hashes()
    return {"files": hashes, "count": len(hashes)}


def main() -> dict:
    output = {"h0a": _h0a(), "h0b": _h0b(), "h0c": _h0c()}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
