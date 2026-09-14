"""Derive the seed-invariance curve from the committed pursuit-b artifacts.

Exists because the numbers were first quoted from an ad-hoc one-liner, which is
not re-derivable by a reader. This reads only committed files and emits one
artefact whose sha256 can be cited instead.

Two columns come from two different files per budget, and the reason matters:

- ARI: the run JSONs at 250/1000/2000/4000 report a 2-seed mean (they were run
  with --eval-seeds 2 to keep the sweep cheap). The labels JSONs carry
  `per_seed_mean_foreground_ari` with all TEN seeds at every budget, because the
  pairwise test needs every seed anyway. So the ten-seed ARI was computed and
  committed at every budget; it was simply not the number anyone read. This
  script takes ARI from the labels files, making the curve protocol-consistent
  with the locked 50 images x seeds 1..10 throughout.
- Partition agreement: the labels files, 45 pairs at every budget.

The 2-seed values are carried alongside as `ari_2seed_from_run` so the two can
be compared rather than one silently replacing the other.

No wall-clock in the output: a re-run against the same artefacts must hash
identically.
"""

import glob
import json
import re
import statistics
import sys
from pathlib import Path

from overlap_bench.paths import ROOT_DIR

OUT = Path(ROOT_DIR) / "outputs" / "seed-invariance-curve.json"
DATASET = "2shapes"


def _budget(path: str) -> int:
    found = re.search(r"labels-(\d+)\.json$", path)
    return int(found.group(1)) if found else -1


def main() -> dict:
    points = []
    for path in sorted(glob.glob(str(Path(ROOT_DIR) / "outputs" / "pursuit-b-labels-*.json"))):
        steps = _budget(path)
        if steps < 0:
            continue  # the mnist file, which is a different dataset
        labels = json.loads(Path(path).read_text())
        per_seed = labels["per_seed_mean_foreground_ari"]
        run = json.loads(
            (Path(ROOT_DIR) / "outputs" / f"pursuit-b-run-{steps}.json").read_text()
        )
        points.append(
            {
                "steps": steps,
                "ari_10seed": statistics.fmean(per_seed),
                "ari_10seed_sd": statistics.pstdev(per_seed),
                "n_seeds": len(per_seed),
                "ari_2seed_from_run": run["evaluation"]["seed_averaged_foreground_ari"],
                "n_seeds_in_run_json": run["eval_seeds"],
                "pairs": labels["n_pairs"],
                "pairs_all_images_same_partition": labels["n_pairs_all_images_same_partition"],
                "pixels_differing_that_foreground_ari_scores": labels[
                    "n_pixels_differing_that_foreground_ari_scores"
                ],
            }
        )
    points.sort(key=lambda p: p["steps"])

    # Every adjacent rise against two seed-SDs. The claim "climbs monotonically"
    # is only worth making if each step clears the noise it could be made of.
    rises = []
    for lower, upper in zip(points, points[1:]):
        bar = 2 * max(lower["ari_10seed_sd"], upper["ari_10seed_sd"])
        gap = upper["ari_10seed"] - lower["ari_10seed"]
        rises.append(
            {
                "from_steps": lower["steps"],
                "to_steps": upper["steps"],
                "gap": gap,
                "two_seed_sd": bar,
                "exceeds_two_sd": gap > bar,
            }
        )

    converged = [p["steps"] for p in points if p["pairs_all_images_same_partition"] == p["pairs"]]
    unconverged = [p["steps"] for p in points if p["pairs_all_images_same_partition"] == 0]
    bracket = (
        [max(unconverged), min(converged)] if converged and unconverged else None
    )

    output = {
        "dataset": DATASET,
        "eval_images": 50,
        "seeds": list(range(1, 11)),
        "points": points,
        "adjacent_rises": rises,
        "every_adjacent_rise_exceeds_two_sd": all(r["exceeds_two_sd"] for r in rises),
        # The transition is bracketed by the budgets actually run, not localised:
        # these points say it happened between them, not where.
        "seed_invariance_bracket_steps": bracket,
        "derived_from": sorted(
            Path(p).name for p in glob.glob(str(Path(ROOT_DIR) / "outputs" / "pursuit-b-labels-*.json"))
            if _budget(p) >= 0
        ),
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)
    return output


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
