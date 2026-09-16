#!/usr/bin/env python3
"""Two follow-up experiments from the K2/delta_omega structural comparison:

1. K2 transfer: does MNIST_shapes' trained K2/delta_omega do anything useful
   evaluated against 2shapes/3shapes' own val images and readout? (K2 is a
   1024x1024 operator regardless of dataset -- all three are 32x32 images --
   so the transplant is dimensionally valid; whether it's *meaningful* is
   the actual question.)

2. Q_10: is delta_omega causally irrelevant on 3shapes the way
   PURSUIT-B-MECHANISM.md established it is on 2shapes (zeroing it there
   reproduces every one of 500 partitions exactly)? Zero 3shapes' trained
   delta_omega and re-evaluate; compare to the un-ablated 0.6934.

Usage:
    PYTHONPATH=src python scripts/cross_dataset_experiments.py \
        --mnist-checkpoint outputs/ckpt-mnist-40000.pt \
        --3shapes-checkpoint outputs/ckpt-3shapes-40000.pt \
        --device cuda \
        --out outputs/cross-dataset-experiments.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_pursuit_b as rp


def _progress(msg: str) -> None:
    print(f"[cross-dataset] {msg}", file=sys.stderr, flush=True)


def load_model(ckpt_path: str, device: torch.device, n: int = 1024) -> rp.Layer2:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    k2_dummy = torch.zeros(n, n, dtype=torch.float64, device=device)
    model = rp.Layer2(k2_dummy, n).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-checkpoint", required=True)
    parser.add_argument("--3shapes-checkpoint", dest="shapes3_checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    results: dict = {"transfer": {}, "delta_omega_ablation": {}}

    _progress("=== Experiment 1: MNIST_shapes' K2 transplanted onto other datasets ===")
    model_mnist = load_model(args.mnist_checkpoint, device)
    for target_ds in ["MNIST_shapes", "2shapes", "3shapes"]:
        ev = rp._evaluate(model_mnist, device, n_images=50, n_seeds=10, dataset=target_ds)
        results["transfer"][target_ds] = ev
        _progress(
            f"  MNIST-trained model on {target_ds}: FG-ARI "
            f"{ev['seed_averaged_foreground_ari']:.4f} +/- {ev['seed_std_foreground_ari']:.4f} "
            f"(M0={ev['m0_at_defaults']:.4f}, cc={ev['cc_baseline']:.4f})"
        )
        Path(args.out).write_text(json.dumps(results, indent=2))

    _progress("=== Experiment 2 (Q_10): zero delta_omega on 3shapes' trained model ===")
    model_3s = load_model(args.shapes3_checkpoint, device)
    with torch.no_grad():
        model_3s.delta_omega.zero_()
    ev = rp._evaluate(model_3s, device, n_images=50, n_seeds=10, dataset="3shapes")
    results["delta_omega_ablation"]["3shapes_zero_delta_omega"] = ev
    _progress(
        f"  3shapes with delta_omega=0: FG-ARI {ev['seed_averaged_foreground_ari']:.4f} "
        f"+/- {ev['seed_std_foreground_ari']:.4f} "
        f"(full model was 0.6934 +/- 0.0058)"
    )
    Path(args.out).write_text(json.dumps(results, indent=2))
    _progress(f"done, wrote {args.out}")


if __name__ == "__main__":
    main()
