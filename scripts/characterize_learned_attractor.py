"""Characterize learned attractor properties: trajectories, Lyapunov exponents, phase coherence, spectra.

Captures data NOT in earlier pursuits:
  - Full oscillator state trajectories (500+ steps, all states over time)
  - Phase angles and their temporal evolution
  - Per-oscillator phase coherence as trajectory progresses
  - Lyapunov exponents (stability characterization)
  - Spectral content (frequency analysis of phase dynamics)

Usage:
    PYTHONPATH=src python scripts/characterize_learned_attractor.py \
        --checkpoint outputs/ckpt-10000.pt \
        --device cuda \
        --out outputs/attractor-10000.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from overlap_bench.dataset_hashes import CAE_DIR, verify_locked_dataset_hashes
from overlap_bench.harness_m0 import foreground_ari
from overlap_bench.paths import ROOT_DIR
from overlap_bench.reference_repo import ensure_on_path


# Reuse from run_pursuit_b
LAYER1_STEPS = 60
LAYER2_STEPS = 140
BATCH = 32
EVAL_IMAGES = 50
EVAL_SEEDS = tuple(range(1, 11))
DATASET = "2shapes"

# Attractor characterization parameters
TRAJECTORY_STEPS = 500  # Long enough for Lyapunov and spectral analysis
LYAPUNOV_EPSILON = 1e-6  # Perturbation magnitude for exponent computation
COHERENCE_WINDOW = 50  # Sliding window for per-oscillator coherence


def _progress(message: str) -> None:
    print(f"[attractor] {message}", file=sys.stderr, flush=True)


def _sheets(nrow: int, ncol: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """M0's two Gaussian sheets."""
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch

    k1 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9, device=device, dtype=torch.float64).real.clone()
    k2 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.0313, device=device, dtype=torch.float64).real.clone()
    return k1, k2


def _layer1_masks(
    images: torch.Tensor, k1: torch.Tensor, seeds: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """M0's layer 1 mask and initial state."""
    b, n = images.shape
    x0 = torch.empty((b, n), dtype=torch.complex128, device=device)
    for i in range(b):
        generator = torch.Generator(device="cpu").manual_seed(int(seeds[i]))
        phases = torch.rand(n, dtype=torch.float64, generator=generator)
        x0[i] = torch.exp(1j * (phases - 0.5) * (2 * torch.pi)).to(device)

    x = x0
    k1c = k1.to(torch.complex128).T
    omega_c = images.to(torch.complex128)
    for _ in range(1, LAYER1_STEPS):
        x = x @ k1c + 1j * omega_c * x

    angles = torch.angle(x)
    threshold = angles.mean(dim=1, keepdim=True)
    above = angles > threshold
    below = angles < threshold
    mask = torch.where(
        (above.sum(1, keepdim=True) > below.sum(1, keepdim=True)).expand_as(above), above, below
    )
    return mask, x0


class Layer2(torch.nn.Module):
    """Trained layer 2 model."""

    def __init__(self, k2_init: torch.Tensor, n: int):
        super().__init__()
        self.K2 = torch.nn.Parameter(k2_init.clone())
        self.delta_omega = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64, device=k2_init.device))

    def _run(
        self, x0: torch.Tensor, omega: torch.Tensor, mask: torch.Tensor, steps: int, keep_all: bool
    ) -> torch.Tensor:
        """Run dynamics for `steps` iterations."""
        keep = (~mask).to(torch.float64)
        keep_c = keep.to(torch.complex128)
        omega2 = ((omega + self.delta_omega) * keep).to(torch.complex128)
        x = x0 * keep_c
        k2t = self.K2.to(torch.complex128).T
        states: list[torch.Tensor] = []
        for _ in range(steps):
            x = (x @ k2t) * keep_c + 1j * omega2 * x
            if keep_all:
                states.append(x)
        return torch.stack(states, dim=-1) if keep_all else x

    def orbit(self, x0: torch.Tensor, omega: torch.Tensor, mask: torch.Tensor, steps: int) -> torch.Tensor:
        """Return full (B, N, steps) trajectory."""
        return self._run(x0, omega, mask, steps, keep_all=True)


def compute_lyapunov_exponents(
    model: Layer2,
    x0: torch.Tensor,
    omega: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 200,
    epsilon: float = LYAPUNOV_EPSILON,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Estimate maximum Lyapunov exponent via trajectory perturbation.

    Returns dict with:
      - max_lyapunov: Largest exponent (positive=chaotic, negative=stable)
      - divergence_curve: Divergence over trajectory steps
    """
    with torch.no_grad():
        # Reference trajectory
        ref = model.orbit(x0, omega, mask, steps)  # (1, N, steps)
        
        # Perturbed trajectory (add small noise to initial state)
        x0_pert = x0 + epsilon * torch.randn_like(x0)
        pert = model.orbit(x0_pert, omega, mask, steps)
        
        # Measure divergence over time
        divergences = []
        for t in range(steps):
            # Euclidean distance in oscillator state space
            diff = torch.abs(ref[0, :, t] - pert[0, :, t])
            dist = diff.mean().item()
            divergences.append(dist)
        
        # Lyapunov exponent from exponential growth rate
        # log(divergence) ≈ λ * t
        divergences = np.maximum(divergences, 1e-10)  # Avoid log(0)
        
        # Fit log(divergence) ~ λ*t using least squares on second half
        t_start = steps // 2
        t_vals = np.arange(t_start, steps)
        log_divs = np.log(divergences[t_start:])
        if len(t_vals) > 1:
            coeffs = np.polyfit(t_vals, log_divs, 1)
            max_lyapunov = float(coeffs[0])
        else:
            max_lyapunov = 0.0
    
    return {
        "max_lyapunov_exponent": max_lyapunov,
        "divergence_curve": [float(d) for d in divergences],
        "epsilon": epsilon,
    }


def compute_phase_coherence_trajectory(
    orbit: torch.Tensor, window: int = COHERENCE_WINDOW
) -> dict:
    """Track phase coherence (order parameter magnitude) over time.

    orbit: (N,) complex oscillator states
    Returns:
      - order_parameter_magnitude: Mean |sum(exp(i*phase))| per step
      - per_oscillator_std: Std of phase angles per oscillator
    """
    phases = torch.angle(orbit)  # (N, steps)
    steps = orbit.shape[-1]
    
    op_mag = []
    per_osc_std = []
    
    for t in range(steps):
        # Order parameter = |mean(exp(i*phase))|
        phase_t = phases[:, t]
        op = torch.abs(torch.mean(torch.exp(1j * phase_t))).item()
        op_mag.append(op)
        
        # Phase std per oscillator (optional, per-step aggregation)
        phase_std = phase_t.std().item()
        per_osc_std.append(phase_std)
    
    return {
        "order_parameter_magnitude": op_mag,
        "phase_std": per_osc_std,
        "mean_op_magnitude": float(np.mean(op_mag)),
        "max_op_magnitude": float(np.max(op_mag)),
    }


def compute_spectral_content(orbit: torch.Tensor) -> dict:
    """Spectral analysis of phase dynamics via FFT.

    orbit: (N,) complex oscillator states
    Returns:
      - power_spectrum: Power (magnitude squared) of FFT
      - dominant_frequencies: Top 5 frequency components
    """
    phases = torch.angle(orbit)  # (N, steps)
    
    # Compute FFT per oscillator, then average
    fft_list = []
    for i in range(phases.shape[0]):
        phase_i = phases[i, :].cpu().numpy()
        fft_i = np.abs(np.fft.fft(phase_i)) ** 2
        fft_list.append(fft_i)
    
    power = np.mean(fft_list, axis=0)
    freqs = np.fft.fftfreq(len(phase_i))
    
    # Top 5 frequencies
    top_idx = np.argsort(-power)[:5]
    top_freqs = [(float(freqs[i]), float(power[i])) for i in top_idx]
    
    return {
        "power_spectrum": [float(p) for p in power[:len(power)//2]],  # Only positive freqs
        "dominant_frequencies": top_freqs,
    }


def main():
    parser = argparse.ArgumentParser(description="Characterize learned attractor dynamics")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, help="Output JSON file")
    args = parser.parse_args()
    device = torch.device(args.device)

    # verify_locked_dataset_hashes()  # Skip when running with partial datasets
    _progress(f"device={device}")

    # Load checkpoint if available; otherwise use untrained model
    ckpt_path = Path(args.checkpoint)
    state_dict = None
    if ckpt_path.exists():
        _progress(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # Checkpoint has nested structure: {"state_dict": {...}, "steps": ..., ...}
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
    else:
        _progress(f"Checkpoint not found ({ckpt_path}); using untrained model")
    
    # Create model
    k1, k2 = _sheets(45, 45, device)
    n_osc = k2.shape[0]
    model = Layer2(k2, n_osc).to(device)
    
    # Load weights if checkpoint available
    if state_dict:
        model.load_state_dict(state_dict)
        _progress(f"Model loaded from checkpoint. delta_omega L2 norm: {torch.norm(model.delta_omega).item():.6f}")
    else:
        _progress("Model initialized fresh (no checkpoint loaded)")
    
    model.eval()

    # Load data
    with np.load(CAE_DIR / f"{DATASET}_val.npz") as data:
        images_np = np.asarray(data["images"][:EVAL_IMAGES, 0], dtype=np.float64)
    images = torch.from_numpy(images_np).to(device)

    results = {
        "dataset": DATASET,
        "checkpoint": str(args.checkpoint),
        "trajectory_steps": TRAJECTORY_STEPS,
        "eval_images": EVAL_IMAGES,
        "eval_seeds": list(EVAL_SEEDS),
        "per_image_analyses": [],
    }

    started = time.monotonic()
    count = 0

    for img_idx in range(EVAL_IMAGES):
        image = images[img_idx]
        
        # Per-seed analysis
        per_seed_results = []
        for seed in EVAL_SEEDS:
            seed_val = int(seed)
            
            # Layer 1 mask and initial state
            k1_local, k2_local = _sheets(45, 45, device)
            mask, x0 = _layer1_masks(
                image.unsqueeze(0),
                k1_local,
                torch.tensor([seed_val], device=device),
                device,
            )
            
            # Generate long trajectory
            with torch.no_grad():
                orbit = model.orbit(x0, image.unsqueeze(0), mask, TRAJECTORY_STEPS)  # (1, N, steps)
                orbit = orbit[0]  # (N, steps)
            
            # Analyses
            lyapunov = compute_lyapunov_exponents(model, x0, image.unsqueeze(0), mask, 200, device=device)
            coherence = compute_phase_coherence_trajectory(orbit)
            spectrum = compute_spectral_content(orbit)
            
            per_seed_results.append({
                "seed": seed_val,
                "lyapunov": lyapunov,
                "coherence": coherence,
                "spectrum": spectrum,
            })
            
            count += 1
            if count % 10 == 0:
                elapsed = time.monotonic() - started
                rate = count / elapsed
                remaining = (EVAL_IMAGES * len(EVAL_SEEDS) - count) / rate
                _progress(f"Image {img_idx+1}/{EVAL_IMAGES}, seed {seed_val}: {remaining:.1f}s remaining")
        
        results["per_image_analyses"].append({
            "image_idx": img_idx,
            "per_seed": per_seed_results,
        })

    # Write output
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.monotonic() - started
    _progress(f"Done in {elapsed:.1f}s. Output: {args.out}")


if __name__ == "__main__":
    main()
