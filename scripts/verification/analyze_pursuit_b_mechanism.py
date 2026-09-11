"""Reproduce the Pursuit B mechanism diagnostics used by the mini note.

The output contains no paths or wall-clock fields. On the same code, inputs,
CPU stack, and checkpoint files, two executions must produce identical JSON.
Progress goes to stderr so stdout remains machine-readable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.linalg
import torch
from scipy.stats import spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.metrics import adjusted_rand_score

from overlap_bench.dataset_hashes import CAE_DIR, verify_locked_dataset_hashes
from overlap_bench.harness_m0 import foreground_ari
from overlap_bench.paths import REFERENCE_COMMIT, ROOT_DIR
from overlap_bench.reference_repo import ensure_on_path

BUDGETS = (250, 500, 1000, 2000, 4000, 6000, 8000, 10000)
DYNAMICS_BUDGETS = (500, 4000, 6000, 8000, 10000)
SEEDS = tuple(range(1, 11))
EVAL_IMAGES = 50
STANDARD_WINDOW = (80, 120)
EARLY_WINDOWS = ((1, 41), (20, 60), (40, 80), (80, 120))
HARD_6000_IMAGES = (3, 32, 34)
HARD_6000_WINDOW_ENDS = (80, 120, 160, 240, 400, 800, 1600)
DISTANCE_BINS = ((0.0, 1.01), (1.01, 2.01), (2.01, 4.01), (4.01, 8.01), (8.01, 16.01), (16.01, float("inf")))
OUT = Path(ROOT_DIR) / "outputs" / "pursuit-b-mechanism.json"


def _progress(message: str) -> None:
    print(f"[pursuit-b-mechanism] {message}", file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    return state["K2"].numpy(), state["delta_omega"].numpy()


def _partition_equal(a: np.ndarray, b: np.ndarray) -> bool:
    scored = (a >= 0) & (b >= 0)
    return bool(np.array_equal(a < 0, b < 0) and adjusted_rand_score(a[scored], b[scored]) == 1.0)


def _pair_summary(maps: dict[tuple[int, int], np.ndarray]) -> dict[str, Any]:
    identical = []
    per_image_disagreement = np.zeros(EVAL_IMAGES, dtype=np.int64)
    for left, right in itertools.combinations(SEEDS, 2):
        same = []
        for image_index in range(EVAL_IMAGES):
            equal = _partition_equal(maps[(left, image_index)], maps[(right, image_index)])
            same.append(equal)
            per_image_disagreement[image_index] += not equal
        identical.append(all(same))
    return {
        "pairs": len(identical),
        "pairs_all_images_same_partition": int(sum(identical)),
        "unstable_images": int(np.count_nonzero(per_image_disagreement)),
        "per_image_disagreeing_pairs": per_image_disagreement.tolist(),
    }


def _stationary_projection(delta_k: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    grid = np.arange(nrow * ncol).reshape(nrow, ncol)
    projected = np.empty_like(delta_k)
    for dr in range(-(nrow - 1), nrow):
        for dc in range(-(ncol - 1), ncol):
            if dr >= 0:
                target_rows, source_rows = slice(dr, nrow), slice(0, nrow - dr)
            else:
                target_rows, source_rows = slice(0, nrow + dr), slice(-dr, nrow)
            if dc >= 0:
                target_cols, source_cols = slice(dc, ncol), slice(0, ncol - dc)
            else:
                target_cols, source_cols = slice(0, ncol + dc), slice(-dc, ncol)
            targets = grid[target_rows, target_cols].reshape(-1)
            sources = grid[source_rows, source_cols].reshape(-1)
            projected[targets, sources] = delta_k[targets, sources].mean()
    return projected


def _distance_matrix(nrow: int, ncol: int) -> np.ndarray:
    rows, cols = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
    positions = np.stack((rows.reshape(-1), cols.reshape(-1)), axis=1)
    return scipy.spatial.distance.cdist(positions, positions)


def _weight_diagnostics(
    k0: np.ndarray, checkpoints: dict[int, tuple[np.ndarray, np.ndarray]], distance: np.ndarray
) -> dict[str, Any]:
    k0_norm = np.linalg.norm(k0)
    rows = []
    for steps in BUDGETS:
        k, delta = checkpoints[steps]
        change = k - k0
        alpha = float(np.vdot(k0, k).real / np.vdot(k0, k0).real)
        bins = []
        for lower, upper in DISTANCE_BINS:
            selected = (distance >= lower) & (distance < upper)
            bins.append(
                {
                    "distance": [lower, None if np.isinf(upper) else upper],
                    "k0_mean": float(k0[selected].mean()),
                    "k0_rms": float(np.sqrt(np.mean(k0[selected] ** 2))),
                    "delta_k_mean": float(change[selected].mean()),
                    "delta_k_rms": float(np.sqrt(np.mean(change[selected] ** 2))),
                }
            )
        rows.append(
            {
                "steps": steps,
                "delta_k_over_k0_frobenius": float(np.linalg.norm(change) / k0_norm),
                "k_over_k0_frobenius": float(np.linalg.norm(k) / k0_norm),
                "asymmetry_over_k_frobenius": float(np.linalg.norm(k - k.T) / np.linalg.norm(k)),
                "best_scalar_multiple_of_k0": alpha,
                "scalar_fit_residual_over_k": float(np.linalg.norm(k - alpha * k0) / np.linalg.norm(k)),
                "delta_omega_l2": float(np.linalg.norm(delta)),
                "distance_bins_per_edge": bins,
            }
        )
    return {"k0_frobenius": float(k0_norm), "checkpoints": rows}


def _simulate(
    k: np.ndarray,
    delta: np.ndarray,
    omega: np.ndarray,
    active: np.ndarray,
    x0s: np.ndarray,
    maximum_update: int,
    capture_updates: set[int],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    matrix = k[np.ix_(active, active)].astype(np.complex128, copy=True)
    omega_active = omega[active] + delta[active]
    matrix[np.diag_indices_from(matrix)] += 1j * omega_active
    x = x0s[:, active].T.copy()
    captured = {}
    for update in range(1, maximum_update + 1):
        x = matrix @ x
        x /= np.linalg.norm(x, axis=0, keepdims=True)
        if update in capture_updates:
            captured[update] = x.copy()
    return captured, matrix, x


def _readout(
    window: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    spatiotemporal_segmentation_torch: Any,
) -> np.ndarray:
    # window is (foreground, 41, seeds), containing an inclusive update interval.
    n = mask.size
    save_x = torch.full((n, 41), torch.nan, dtype=torch.complex128)
    seed_maps = []
    for seed_index in range(window.shape[2]):
        save_x[~torch.from_numpy(mask)] = torch.from_numpy(window[:, :, seed_index])
        cluster_map, *_ = spatiotemporal_segmentation_torch(
            save_x, torch.from_numpy(image), torch.from_numpy(mask), nt_mask=1, n_clusters=2
        )
        seed_maps.append(cluster_map.numpy())
    return np.stack(seed_maps)


def _phase_organization(window: np.ndarray, labels_flat: np.ndarray, active: np.ndarray) -> tuple[float, float]:
    phases = np.exp(1j * np.angle(window))
    labels_active = labels_flat[active]
    object_ids = sorted(int(value) for value in np.unique(labels_active) if value > 0)
    if len(object_ids) != 2:
        raise AssertionError(f"expected two active objects, got {object_ids}")
    means = [phases[labels_active == object_id].mean(axis=0) for object_id in object_ids]
    within = float(np.mean([np.abs(value).mean() for value in means]))
    between = float(
        np.mean(np.real(means[0] * np.conj(means[1])) / (np.abs(means[0]) * np.abs(means[1])))
    )
    return within, between


def _projective_pair_mean(states: np.ndarray) -> tuple[float, float]:
    values = []
    for left, right in itertools.combinations(range(states.shape[1]), 2):
        numerator = abs(np.vdot(states[:, left], states[:, right]))
        denominator = np.linalg.norm(states[:, left]) * np.linalg.norm(states[:, right])
        values.append(float(numerator / denominator))
    return float(np.mean(values)), float(np.min(values))


def _dynamics_diagnostics(
    checkpoints: dict[int, tuple[np.ndarray, np.ndarray]],
    images: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    x0s: np.ndarray,
    curve: dict[str, Any],
    spatiotemporal_segmentation_torch: Any,
) -> tuple[list[dict[str, Any]], dict[int, dict[tuple[int, int], np.ndarray]]]:
    canonical = {row["steps"]: row for row in curve["points"]}
    rows = []
    maps_by_budget = {}
    for steps in DYNAMICS_BUDGETS:
        _progress(f"full dynamics at {steps} steps")
        k, delta = checkpoints[steps]
        q_values = []
        condition_numbers = []
        phase_pair_means = []
        phase_pair_minima = []
        right_eigenvector_alignments = []
        within_values = []
        between_values = []
        maps: dict[tuple[int, int], np.ndarray] = {}
        ari_per_seed = [[] for _ in SEEDS]
        for image_index in range(EVAL_IMAGES):
            mask = masks[image_index]
            active = ~mask
            omega = images[image_index].T.reshape(-1)
            captured, matrix, final = _simulate(
                k, delta, omega, active, x0s, STANDARD_WINDOW[1], set(range(80, 121))
            )
            window = np.stack([captured[update] for update in range(80, 121)], axis=1)
            seed_maps = _readout(window, images[image_index], mask, spatiotemporal_segmentation_torch)
            for seed_index, seed in enumerate(SEEDS):
                maps[(seed, image_index)] = seed_maps[seed_index]
                ari_per_seed[seed_index].append(
                    foreground_ari(labels[image_index], seed_maps[seed_index])
                )

            eigenvalues, eigenvectors = scipy.linalg.eig(matrix)
            order = np.argsort(np.abs(eigenvalues))[::-1]
            q_values.append(float(abs(eigenvalues[order[1]]) / abs(eigenvalues[order[0]])))
            condition_numbers.append(float(np.linalg.cond(eigenvectors)))
            mean_pair, min_pair = _projective_pair_mean(final)
            phase_pair_means.append(mean_pair)
            phase_pair_minima.append(min_pair)
            leading = eigenvectors[:, order[0]]
            right_eigenvector_alignments.extend(
                float(abs(np.vdot(leading, final[:, seed_index])) / (np.linalg.norm(leading) * np.linalg.norm(final[:, seed_index])))
                for seed_index in range(len(SEEDS))
            )
            within, between = _phase_organization(
                window, labels[image_index].T.reshape(-1), active
            )
            within_values.append(within)
            between_values.append(between)

            if image_index % 10 == 9:
                _progress(f"  {steps}: {image_index + 1}/{EVAL_IMAGES} images")

        pair_summary = _pair_summary(maps)
        per_seed_means = [float(np.mean(values)) for values in ari_per_seed]
        expected = canonical[steps]
        assert abs(float(np.mean(per_seed_means)) - expected["ari_10seed"]) < 1e-12
        assert pair_summary["pairs_all_images_same_partition"] == expected[
            "pairs_all_images_same_partition"
        ]
        correlation = (
            None
            if np.ptp(pair_summary["per_image_disagreeing_pairs"]) == 0
            else float(spearmanr(q_values, pair_summary["per_image_disagreeing_pairs"]).statistic)
        )
        rows.append(
            {
                "steps": steps,
                "canonical_mean_foreground_ari": expected["ari_10seed"],
                "canonical_seed_sd_foreground_ari": expected["ari_10seed_sd"],
                **pair_summary,
                "spectral_modulus_ratio": {
                    "minimum": float(np.min(q_values)),
                    "mean": float(np.mean(q_values)),
                    "maximum": float(np.max(q_values)),
                },
                "eigenbasis_condition_number": {
                    "mean": float(np.mean(condition_numbers)),
                    "maximum": float(np.max(condition_numbers)),
                },
                "update_120_projective_seed_agreement": {
                    "mean_of_image_pair_means": float(np.mean(phase_pair_means)),
                    "minimum_pair": float(np.min(phase_pair_minima)),
                },
                "update_120_right_leading_eigenvector_alignment_mean": float(
                    np.mean(right_eigenvector_alignments)
                ),
                "readout_window_phase_organization": {
                    "within_object_coherence_mean_over_images": float(np.mean(within_values)),
                    "between_object_cosine_mean_over_images": float(np.mean(between_values)),
                    "image_statistic": "mean over ten seeds and updates 80 through 120 inclusive",
                },
                "spearman_q_vs_image_disagreeing_pairs": correlation,
            }
        )
        maps_by_budget[steps] = maps
    return rows, maps_by_budget


def _component_ablation(
    name: str,
    k: np.ndarray,
    delta: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    x0s: np.ndarray,
    full_maps: dict[tuple[int, int], np.ndarray],
    spatiotemporal_segmentation_torch: Any,
) -> dict[str, Any]:
    _progress(f"component ablation: {name}")
    maps: dict[tuple[int, int], np.ndarray] = {}
    ari = [[] for _ in SEEDS]
    full_matches = 0
    for image_index in range(EVAL_IMAGES):
        mask = masks[image_index]
        active = ~mask
        captured, _, _ = _simulate(
            k,
            delta,
            images[image_index].T.reshape(-1),
            active,
            x0s,
            STANDARD_WINDOW[1],
            set(range(80, 121)),
        )
        window = np.stack([captured[update] for update in range(80, 121)], axis=1)
        seed_maps = _readout(window, images[image_index], mask, spatiotemporal_segmentation_torch)
        for seed_index, seed in enumerate(SEEDS):
            key = (seed, image_index)
            maps[key] = seed_maps[seed_index]
            ari[seed_index].append(foreground_ari(labels[image_index], seed_maps[seed_index]))
            full_matches += _partition_equal(seed_maps[seed_index], full_maps[key])
        if image_index % 10 == 9:
            _progress(f"  {name}: {image_index + 1}/{EVAL_IMAGES} images")
    per_seed = [float(np.mean(values)) for values in ari]
    return {
        "name": name,
        "delta_omega_l2": float(np.linalg.norm(delta)),
        "mean_foreground_ari": float(np.mean(per_seed)),
        "seed_sd_foreground_ari": float(np.std(per_seed, ddof=0)),
        **_pair_summary(maps),
        "maps_matching_full_10000_partition": int(full_matches),
        "maps_compared_to_full": EVAL_IMAGES * len(SEEDS),
    }


def _early_window_intervention(
    k: np.ndarray,
    delta: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    x0s: np.ndarray,
    spatiotemporal_segmentation_torch: Any,
) -> list[dict[str, Any]]:
    _progress("8000-step early-window intervention")
    maps = {window: {} for window in EARLY_WINDOWS}
    ari = {window: [[] for _ in SEEDS] for window in EARLY_WINDOWS}
    requested = {update for window in EARLY_WINDOWS for update in range(window[0], window[1] + 1)}
    for image_index in range(EVAL_IMAGES):
        mask = masks[image_index]
        active = ~mask
        captured, _, _ = _simulate(
            k, delta, images[image_index].T.reshape(-1), active, x0s, 120, requested
        )
        for window in EARLY_WINDOWS:
            start, end = window
            values = np.stack([captured[update] for update in range(start, end + 1)], axis=1)
            seed_maps = _readout(values, images[image_index], mask, spatiotemporal_segmentation_torch)
            for seed_index, seed in enumerate(SEEDS):
                maps[window][(seed, image_index)] = seed_maps[seed_index]
                ari[window][seed_index].append(
                    foreground_ari(labels[image_index], seed_maps[seed_index])
                )
        if image_index % 10 == 9:
            _progress(f"  early windows: {image_index + 1}/{EVAL_IMAGES} images")
    rows = []
    for window in EARLY_WINDOWS:
        per_seed = [float(np.mean(values)) for values in ari[window]]
        rows.append(
            {
                "inclusive_layer2_updates": list(window),
                "samples": 41,
                "mean_foreground_ari": float(np.mean(per_seed)),
                "seed_sd_foreground_ari": float(np.std(per_seed, ddof=0)),
                **_pair_summary(maps[window]),
            }
        )
    return rows


def _hard_image_long_horizon(
    k: np.ndarray,
    delta: np.ndarray,
    images: np.ndarray,
    masks: np.ndarray,
    x0s: np.ndarray,
    spatiotemporal_segmentation_torch: Any,
) -> list[dict[str, Any]]:
    _progress("6000-step hard-image long-horizon intervention")
    requested = {
        update
        for end in HARD_6000_WINDOW_ENDS
        for update in range(end - 40, end + 1)
    }
    rows = {end: [] for end in HARD_6000_WINDOW_ENDS}
    for image_index in HARD_6000_IMAGES:
        mask = masks[image_index]
        captured, _, _ = _simulate(
            k,
            delta,
            images[image_index].T.reshape(-1),
            ~mask,
            x0s,
            max(HARD_6000_WINDOW_ENDS),
            requested,
        )
        for end in HARD_6000_WINDOW_ENDS:
            window = np.stack(
                [captured[update] for update in range(end - 40, end + 1)], axis=1
            )
            seed_maps = _readout(window, images[image_index], mask, spatiotemporal_segmentation_torch)
            same = sum(
                _partition_equal(seed_maps[left], seed_maps[right])
                for left, right in itertools.combinations(range(len(SEEDS)), 2)
            )
            rows[end].append(int(same))
    return [
        {
            "inclusive_layer2_updates": [end - 40, end],
            "samples": 41,
            "image_indices": list(HARD_6000_IMAGES),
            "identical_seed_pairs_of_45_by_image": rows[end],
        }
        for end in HARD_6000_WINDOW_ENDS
    ]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    verified = verify_locked_dataset_hashes(only=["2shapes_val.npz"])
    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import spatiotemporal_segmentation_torch

    pursuit = runpy.run_path(str(Path(ROOT_DIR) / "scripts" / "run_pursuit_b.py"))
    device = torch.device("cpu")
    with np.load(CAE_DIR / "2shapes_val.npz") as data:
        images = np.asarray(data["images"][:EVAL_IMAGES, 0], dtype=np.float64)
        labels = np.asarray(data["labels"][:EVAL_IMAGES], dtype=np.int64)
    nrow, ncol = images.shape[1:]
    flattened_images = torch.from_numpy(images).transpose(1, 2).reshape(EVAL_IMAGES, -1)
    k1, k0_torch = pursuit["_sheets"](nrow, ncol, device)

    _progress("computing layer-1 masks for all 50 images and ten seeds")
    repeated_images = flattened_images.repeat(len(SEEDS), 1)
    repeated_seeds = torch.tensor(
        [seed for seed in SEEDS for _ in range(EVAL_IMAGES)], dtype=torch.int64
    )
    mask_rows, x0_rows = pursuit["_layer1_masks"](
        repeated_images, k1, repeated_seeds, device
    )
    masks_by_seed = mask_rows.reshape(len(SEEDS), EVAL_IMAGES, -1).numpy()
    differing_mask_pixels = int(
        np.count_nonzero(masks_by_seed != masks_by_seed[[0]])
    )
    assert differing_mask_pixels == 0
    masks = masks_by_seed[0]
    x0_all = x0_rows.reshape(len(SEEDS), EVAL_IMAGES, -1).numpy()
    assert np.all(x0_all == x0_all[:, [0]])
    x0s = x0_all[:, 0]

    output_dir = Path(ROOT_DIR) / "outputs"
    checkpoints = {
        steps: _load_checkpoint(output_dir / f"ckpt-{steps}.pt") for steps in BUDGETS
    }
    checkpoint_hashes = {
        f"ckpt-{steps}.pt": _sha256(output_dir / f"ckpt-{steps}.pt") for steps in BUDGETS
    }
    curve_path = output_dir / "seed-invariance-curve.json"
    curve = _load_json(curve_path)
    k0 = k0_torch.numpy()
    distance = _distance_matrix(nrow, ncol)

    dynamics, maps_by_budget = _dynamics_diagnostics(
        checkpoints,
        images,
        labels,
        masks,
        x0s,
        curve,
        spatiotemporal_segmentation_torch,
    )

    k10000, delta10000 = checkpoints[10000]
    change10000 = k10000 - k0
    stationary = k0 + _stationary_projection(change10000, nrow, ncol)
    far_only = k0 + change10000 * (distance > 4.0)
    zero_delta = np.zeros_like(delta10000)
    component_ablation = [
        _component_ablation(
            "full K, delta_omega zero",
            k10000,
            zero_delta,
            images,
            labels,
            masks,
            x0s,
            maps_by_budget[10000],
            spatiotemporal_segmentation_torch,
        ),
        _component_ablation(
            "displacement-stationary delta K, delta_omega zero",
            stationary,
            zero_delta,
            images,
            labels,
            masks,
            x0s,
            maps_by_budget[10000],
            spatiotemporal_segmentation_torch,
        ),
        _component_ablation(
            "displacement-stationary delta K, trained delta_omega",
            stationary,
            delta10000,
            images,
            labels,
            masks,
            x0s,
            maps_by_budget[10000],
            spatiotemporal_segmentation_torch,
        ),
        _component_ablation(
            "delta K beyond four pixels, delta_omega zero",
            far_only,
            zero_delta,
            images,
            labels,
            masks,
            x0s,
            maps_by_budget[10000],
            spatiotemporal_segmentation_torch,
        ),
        _component_ablation(
            "delta K beyond four pixels, trained delta_omega",
            far_only,
            delta10000,
            images,
            labels,
            masks,
            x0s,
            maps_by_budget[10000],
            spatiotemporal_segmentation_torch,
        ),
        _component_ablation(
            "untrained K, trained delta_omega",
            k0,
            delta10000,
            images,
            labels,
            masks,
            x0s,
            maps_by_budget[10000],
            spatiotemporal_segmentation_torch,
        ),
    ]

    trained_delta_l2 = float(np.linalg.norm(delta10000))

    stationary_result = component_ablation[1]
    assert round(stationary_result["mean_foreground_ari"], 5) == 0.90978
    assert stationary_result["pairs_all_images_same_partition"] == 45
    assert stationary_result["delta_omega_l2"] == 0.0

    stationary_trained_result = component_ablation[2]
    assert round(stationary_trained_result["mean_foreground_ari"], 5) == 0.90883
    assert stationary_trained_result["pairs_all_images_same_partition"] == 45
    assert stationary_trained_result["maps_matching_full_10000_partition"] == 110
    assert stationary_trained_result["delta_omega_l2"] == trained_delta_l2

    far_zero_result = component_ablation[3]
    assert far_zero_result["delta_omega_l2"] == 0.0

    far_trained_result = component_ablation[4]
    assert round(far_trained_result["mean_foreground_ari"], 5) == 0.91348
    assert far_trained_result["pairs_all_images_same_partition"] == 45
    assert far_trained_result["maps_matching_full_10000_partition"] == 160
    assert far_trained_result["delta_omega_l2"] == trained_delta_l2

    output = {
        "question": "what changed in oscillator dynamics while segmentation improved and tested-seed sensitivity vanished?",
        "scope": {
            "dataset": "2shapes",
            "images": EVAL_IMAGES,
            "seeds": list(SEEDS),
            "device": "cpu",
            "dtype": "float64/complex128",
            "layer2_updates": 140,
            "standard_readout_inclusive_layer2_updates": list(STANDARD_WINDOW),
        },
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
            "torch": torch.__version__,
        },
        "provenance": {
            "reference_commit": REFERENCE_COMMIT,
            "dataset_hashes_verified": verified,
            "checkpoint_sha256": checkpoint_hashes,
            "seed_invariance_curve": curve_path.name,
            "seed_invariance_curve_sha256": _sha256(curve_path),
        },
        "seed_path": {
            "layer1_masks_differing_pixels_across_seeds": differing_mask_pixels,
            "active_pixels": {
                "minimum": int((~masks).sum(axis=1).min()),
                "median": float(np.median((~masks).sum(axis=1))),
                "maximum": int((~masks).sum(axis=1).max()),
            },
            "mean_pairwise_initial_state_absolute_difference": float(
                np.mean(
                    [
                        np.mean(np.abs(x0s[left] - x0s[right]))
                        for left, right in itertools.combinations(range(len(SEEDS)), 2)
                    ]
                )
            ),
        },
        "weight_geometry": _weight_diagnostics(k0, checkpoints, distance),
        "full_checkpoint_dynamics": dynamics,
        "component_ablation_at_10000": component_ablation,
        "early_window_intervention_at_8000": _early_window_intervention(
            checkpoints[8000][0],
            checkpoints[8000][1],
            images,
            labels,
            masks,
            x0s,
            spatiotemporal_segmentation_torch,
        ),
        "long_horizon_intervention_at_6000": _hard_image_long_horizon(
            checkpoints[6000][0],
            checkpoints[6000][1],
            images,
            masks,
            x0s,
            spatiotemporal_segmentation_torch,
        ),
        "interpretation_limits": [
            "The recurrence is exactly equivariant to one global phase; only relative phase can converge.",
            "The masked operators are nonnormal, so eigenvalue ratios are modal diagnostics, not Euclidean contraction bounds.",
            "Exact KMeans partitions can change under small structured phase residuals.",
            "Checkpoint comparisons are one deterministic training trajectory, not independent training replicates.",
        ],
    }
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _progress(f"wrote {args.out}")
    return output


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
