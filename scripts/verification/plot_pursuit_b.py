"""Figures for the pursuit-b programme, from committed artefacts only.

Reads the run/labels JSONs and the trained checkpoints, writes PNGs to
figures/, and emits a manifest naming every input with its sha256 so a figure
can be traced to the data it came from.

On figure hashes: the manifest records them, but a PNG's bytes depend on the
matplotlib version that drew it. The reproducibility claim is on the INPUTS --
same artefacts in, same numbers plotted -- not on the image bytes.

Four figures, each answering something the tables state but do not show:

  1. curve     -- ARI against budget beside the two disagreement measures, so
                  the threshold statistic's flatness can be seen against the
                  continuous one's collapse. This is the programme's finding.
  2. instability -- per image, how many of the 45 seed pairs disagree, at every
                  budget. Shows whether the residual disagreement is spread
                  thinly over all images or concentrated in a stubborn few.
  3. weights   -- delta_omega as a 32x32 field per budget, and the learned
                  coupling for a centre pixel against its Gaussian-sheet
                  initialisation. Says whether training reshapes the kernel or
                  just scales it.
  4. loss      -- training loss per budget on one axis.

    PYTHONPATH=src uv run --locked python scripts/verification/plot_pursuit_b.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from overlap_bench.paths import ROOT_DIR

OUT_DIR = Path(ROOT_DIR) / "figures"
ARTEFACTS = Path(ROOT_DIR) / "outputs"
BUDGETS = (250, 500, 1000, 2000, 4000, 6000, 8000, 10000)
NROW = NCOL = 32
M0_AT_DEFAULTS = 0.12085737825541296
CC_BASELINE = 0.16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(name: str) -> dict:
    return json.loads((ARTEFACTS / name).read_text())


def _budget_data() -> tuple[list[dict], list[Path]]:
    rows, inputs = [], []
    for steps in BUDGETS:
        run_name, labels_name = f"pursuit-b-run-{steps}.json", f"pursuit-b-labels-{steps}.json"
        run, labels = _load(run_name), _load(labels_name)
        inputs += [ARTEFACTS / run_name, ARTEFACTS / labels_name]
        per_seed = labels["per_seed_mean_foreground_ari"]
        # Per image, how many of the 45 pairs disagree. Absent images agreed.
        per_image = np.zeros(labels["n_images"], dtype=int)
        for pair in labels["pairs"]:
            for mismatch in pair["partition_mismatches"]:
                per_image[mismatch["image"]] += 1
        rows.append(
            {
                "steps": steps,
                "ari": float(np.mean(per_seed)),
                "ari_min": float(np.min(per_seed)),
                "ari_max": float(np.max(per_seed)),
                "ari_sd": float(np.std(per_seed, ddof=0)),
                "pairs_converged": labels["n_pairs_all_images_same_partition"],
                "scored_px": labels["n_pixels_differing_that_foreground_ari_scores"],
                "mean_images_agreeing": float(
                    np.mean([p["n_images_same_partition"] for p in labels["pairs"]])
                ),
                "per_image_disagreeing_pairs": per_image,
                "curve": run["training_curve"],
                "k2_change": run["diagnostics"]["k2_frobenius_relative_change"],
                "delta_omega_norm": run["diagnostics"]["delta_omega_norm"],
            }
        )
    return rows, inputs


def _fig_curve(rows: list[dict]) -> Path:
    steps = [r["steps"] for r in rows]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    top.plot(steps, [r["ari"] for r in rows], "o-", color="#1f4e79", label="FG ARI (10 seeds)")
    top.fill_between(
        steps,
        [r["ari_min"] for r in rows],
        [r["ari_max"] for r in rows],
        color="#1f4e79",
        alpha=0.2,
        label="per-seed min-max",
    )
    top.axhline(M0_AT_DEFAULTS, ls="--", color="#999999", label=f"M0 at defaults {M0_AT_DEFAULTS:.4f}")
    top.axhline(CC_BASELINE, ls=":", color="#cc6600", label=f"connected components {CC_BASELINE}")
    top.set_ylabel("foreground ARI")
    top.set_title("Score improves smoothly; seed agreement does not")
    top.legend(fontsize=8, loc="lower right")
    top.grid(alpha=0.3)

    bottom.plot(
        steps,
        [max(r["scored_px"], 0.5) for r in rows],
        "s-",
        color="#a02020",
        label="scored pixels differing (all 45 pairs)",
    )
    bottom.set_yscale("log")
    bottom.set_ylabel("pixels where two seeds disagree")
    bottom.set_xscale("log")
    bottom.set_xlabel("Adam steps")
    bottom.grid(alpha=0.3)
    twin = bottom.twinx()
    twin.plot(
        steps,
        [r["pairs_converged"] for r in rows],
        "^--",
        color="#207020",
        label="seed pairs identical on all 50 images",
    )
    twin.set_ylabel("pairs converged (of 45)")
    twin.set_ylim(-2, 47)
    lines = bottom.get_lines() + twin.get_lines()
    bottom.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center left")
    bottom.annotate(
        "0.5 plotted for zero\n(log axis)",
        xy=(8000, 0.5),
        xytext=(2600, 1.6),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )

    fig.tight_layout()
    path = OUT_DIR / "pursuit-b-curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fig_instability(rows: list[dict]) -> Path:
    grid = np.vstack([r["per_image_disagreeing_pairs"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto", cmap="magma_r", vmin=0, vmax=45, interpolation="nearest")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["steps"] for r in rows])
    ax.set_ylabel("Adam steps")
    ax.set_xlabel("val image index")
    ax.set_title("Seed disagreement is concentrated in a few images, not spread thinly")
    fig.colorbar(im, ax=ax, label="pairs disagreeing (of 45)")
    fig.tight_layout()
    path = OUT_DIR / "pursuit-b-instability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fig_weights(rows: list[dict]) -> tuple[Path, list[Path]]:
    from overlap_bench.reference_repo import ensure_on_path

    ensure_on_path()
    from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch

    k2_init = gaussian_sheet_torch(NROW, NCOL, 0.5, 0.0313, dtype=torch.float64).real.numpy()
    shown = (500, 2000, 8000, 10000)
    used: list[Path] = []
    fig, axes = plt.subplots(2, len(shown) + 1, figsize=(3 * (len(shown) + 1), 6))

    centre = (NROW // 2) * NCOL + NCOL // 2
    axes[1][0].imshow(k2_init[centre].reshape(NROW, NCOL).T, cmap="viridis")
    axes[1][0].set_title("K2 row at init\n(Gaussian sheet)", fontsize=9)
    axes[0][0].axis("off")
    for ax in (axes[1][0],):
        ax.set_xticks([])
        ax.set_yticks([])

    for col, steps in enumerate(shown, start=1):
        ckpt = ARTEFACTS / f"ckpt-{steps}.pt"
        used.append(ckpt)
        blob = torch.load(ckpt, map_location="cpu")
        k2 = blob["state_dict"]["K2"].numpy()
        d_omega = blob["state_dict"]["delta_omega"].numpy()

        top = axes[0][col]
        img = top.imshow(d_omega.reshape(NROW, NCOL).T, cmap="coolwarm")
        top.set_title(f"delta_omega, {steps} steps", fontsize=9)
        top.set_xticks([])
        top.set_yticks([])
        fig.colorbar(img, ax=top, fraction=0.046)

        bottom = axes[1][col]
        img2 = bottom.imshow((k2 - k2_init)[centre].reshape(NROW, NCOL).T, cmap="RdBu_r")
        bottom.set_title(f"K2 row change, {steps}", fontsize=9)
        bottom.set_xticks([])
        bottom.set_yticks([])
        fig.colorbar(img2, ax=bottom, fraction=0.046)

    fig.suptitle("What training changes: the shared frequency offset and one pixel's coupling")
    fig.tight_layout()
    path = OUT_DIR / "pursuit-b-weights.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, used


def _fig_loss(rows: list[dict]) -> Path:
    """One trajectory, eight stopping points -- not eight independent runs.

    Drawing all eight overlaid produces one visible line, because they are
    bit-identical: same torch.manual_seed(0), same batch order, so the 250-step
    run is a prefix of the 10,000-step run. Verified, every shared step equal
    to the last bit across all seven shorter budgets. That is a determinism
    check worth having, and it is also a caveat on the whole curve: the
    seed-invariance transition is a property of THIS trajectory, not an average
    over training seeds.
    """
    longest = max(rows, key=lambda r: r["steps"])
    pts = [(e["step"], e["loss"]) for e in longest["curve"] if e["step"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=0.9, color="#1f4e79")
    for row in rows[:-1]:
        ax.axvline(row["steps"], color="#a02020", ls=":", lw=0.8)
        ax.text(
            row["steps"], ax.get_ylim()[1], f" {row['steps']}", rotation=90,
            va="top", ha="left", fontsize=6, color="#a02020",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("cos-coherence loss (batch of 32)")
    ax.set_title(
        "One training trajectory; dotted lines are where each budget stopped\n"
        "(all eight runs are bit-identical prefixes, verified step by step)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "pursuit-b-loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _val_overlap_fractions(n_images: int) -> np.ndarray:
    """Ground-truth overlap share per val image.

    A property of the benchmark, not of any run, so it is read from the locked
    npz rather than taken from an artefact. -1 is the excluded overlap label.
    """
    from overlap_bench.dataset_hashes import CAE_DIR

    with np.load(CAE_DIR / "2shapes_val.npz") as data:
        labels = np.asarray(data["labels"][:n_images], dtype=np.int64)
    return np.mean(labels == -1, axis=(1, 2))



def _per_image_budget_data() -> tuple[np.ndarray, np.ndarray, list[Path]] | None:
    """Mean FG-ARI for every image at every retained training budget."""
    paths = [ARTEFACTS / f"pursuit-b-labels2-{budget}.json" for budget in BUDGETS]
    if not all(path.exists() for path in paths):
        return None

    by_budget = []
    n_images = None
    for path in paths:
        labels = json.loads(path.read_text())
        values = np.asarray(labels["per_seed_per_image_foreground_ari"], dtype=float)
        if values.shape[0] != 10:
            raise ValueError(f"{path.name}: expected 10 eval seeds, got {values.shape[0]}")
        if n_images is None:
            n_images = values.shape[1]
        elif values.shape[1] != n_images:
            raise ValueError(f"{path.name}: inconsistent image count {values.shape[1]}")
        by_budget.append(values.mean(axis=0))

    scores = np.vstack(by_budget)
    overlap = _val_overlap_fractions(scores.shape[1])
    return scores, overlap, paths


def _fig_per_image_trajectories(scores: np.ndarray, overlap: np.ndarray) -> Path:
    """Show image-level progress hidden by the aggregate learning curve."""
    order = np.argsort(overlap)[::-1]
    ordered_scores = scores[:, order].T
    monotone = int(np.all(np.diff(scores, axis=0) >= -1e-12, axis=0).sum())

    fig, (heat, share) = plt.subplots(
        1,
        2,
        figsize=(10, 9),
        gridspec_kw={"width_ratios": (8, 1.4), "wspace": 0.08},
    )
    image = heat.imshow(
        ordered_scores,
        aspect="auto",
        cmap="viridis",
        vmin=-0.05,
        vmax=1.0,
        interpolation="nearest",
    )
    heat.set_xticks(range(len(BUDGETS)))
    heat.set_xticklabels(BUDGETS, rotation=35, ha="right")
    heat.set_yticks(range(len(order)))
    heat.set_yticklabels(order, fontsize=6)
    heat.set_xlabel("Adam steps")
    heat.set_ylabel("validation image index (highest overlap first)")
    heat.set_title(
        "Per-image mean FG-ARI over 10 eval seeds\n"
        f"only {monotone}/{scores.shape[1]} images improve monotonically"
    )
    fig.colorbar(image, ax=heat, label="mean foreground ARI")

    share.barh(range(len(order)), overlap[order], color="#a02020", alpha=0.8)
    share.set_ylim(len(order) - 0.5, -0.5)
    share.set_yticks([])
    share.set_xlabel("overlap\nfraction")
    share.set_title("Ground truth", fontsize=9)
    share.grid(axis="x", alpha=0.25)

    path = OUT_DIR / "pursuit-b-per-image-trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_per_image_distributions(scores: np.ndarray, overlap: np.ndarray) -> Path:
    """Distributional progress and its association with overlap."""
    from scipy.stats import spearmanr

    fig, (dist, corr) = plt.subplots(1, 2, figsize=(12, 4.8))
    positions = np.arange(len(BUDGETS))
    dist.boxplot(
        [scores[i] for i in positions],
        positions=positions,
        widths=0.62,
        whis=(10, 90),
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "#a02020", "markeredgecolor": "none"},
        medianprops={"color": "#1f4e79", "linewidth": 1.5},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
    )
    dist.set_xticks(positions)
    dist.set_xticklabels(BUDGETS, rotation=35, ha="right")
    dist.set_xlabel("Adam steps")
    dist.set_ylabel("per-image mean foreground ARI")
    dist.set_title("Image-level score distribution\nwhiskers: 10th–90th percentile")
    dist.set_ylim(-0.08, 1.05)
    dist.grid(axis="y", alpha=0.3)

    correlations = [spearmanr(overlap, scores[i]).statistic for i in positions]
    corr.plot(BUDGETS, correlations, "o-", color="#6a3d9a")
    corr.axhline(0.0, color="#999999", linewidth=0.8)
    corr.set_xscale("log")
    corr.set_ylim(-1.0, 0.05)
    corr.set_xlabel("Adam steps")
    corr.set_ylabel("Spearman rho")
    corr.set_title(
        "Ground-truth overlap vs per-image ARI\nnegative at every retained budget"
    )
    corr.grid(alpha=0.3)
    for budget, value in zip(BUDGETS, correlations, strict=True):
        corr.annotate(
            f"{value:.2f}",
            (budget, value),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    fig.tight_layout()
    path = OUT_DIR / "pursuit-b-per-image-distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _fig_hard_vs_unstable(budget: int) -> Path | None:
    """Is a seed-unstable image also a hard image, or an overlap-heavy one?

    Needs per-image ARI, which the first programme computed and discarded; this
    reads the recomputed labels2-* artefacts. Returns None if they are absent,
    so the script still runs against the original set.
    """
    path_in = ARTEFACTS / f"pursuit-b-labels2-{budget}.json"
    if not path_in.exists():
        return None
    labels = json.loads(path_in.read_text())
    per_image_ari = np.mean(np.array(labels["per_seed_per_image_foreground_ari"]), axis=0)
    disagreeing = np.zeros(labels["n_images"], dtype=int)
    for pair in labels["pairs"]:
        for mismatch in pair["partition_mismatches"]:
            disagreeing[mismatch["image"]] += 1
    overlap = _val_overlap_fractions(labels["n_images"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    scatter = left.scatter(
        overlap, per_image_ari, c=disagreeing, cmap="magma_r", vmin=0, vmax=45, s=45,
        edgecolor="#333333", linewidth=0.4,
    )
    left.set_xlabel("ground-truth overlap fraction")
    left.set_ylabel("mean foreground ARI over 10 seeds")
    left.set_title(f"Difficulty vs overlap, {budget} steps")
    left.grid(alpha=0.3)
    fig.colorbar(scatter, ax=left, label="pairs disagreeing (of 45)")

    unstable = disagreeing > 0
    right.scatter(
        per_image_ari[~unstable], disagreeing[~unstable], s=40, color="#1f4e79",
        label=f"seed-stable ({int((~unstable).sum())} images)",
    )
    right.scatter(
        per_image_ari[unstable], disagreeing[unstable], s=60, color="#a02020",
        label=f"seed-unstable ({int(unstable.sum())} images)",
    )
    right.set_xlabel("mean foreground ARI over 10 seeds")
    right.set_ylabel("pairs disagreeing (of 45)")
    right.set_title("Do the unstable images also score badly?")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / f"pursuit-b-hard-vs-unstable-{budget}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fig_maps(budget: int) -> Path | None:
    """What the model actually produces. Impossible before --save-maps existed."""
    npz = ARTEFACTS / f"pursuit-b-maps-{budget}.npz"
    if not npz.exists():
        return None
    with np.load(npz) as data:
        images, truth = data["images"], data["ground_truth"]
        seed_a = [data[f"map_s1_i{i}"] for i in range(len(images))]
        seed_b = [data[f"map_s2_i{i}"] for i in range(len(images))]

    n = len(images)
    fig, axes = plt.subplots(4, n, figsize=(2.1 * n, 8.8))
    rows = ("input", "ground truth", "prediction, seed 1", "prediction, seed 2")
    for col in range(n):
        for row, field in enumerate((images[col], truth[col], seed_a[col], seed_b[col])):
            ax = axes[row][col]
            if row == 0:
                ax.imshow(field.T, cmap="gray", interpolation="nearest")
            elif row == 1:
                # -1 excluded overlap, 0 background, 1..k objects. Fixed range
                # so a class reads the same way in every column.
                ax.imshow(field.T, cmap="tab10", vmin=-1, vmax=8, interpolation="nearest")
            else:
                # Cluster ids from k-means, arbitrary by construction: a colour
                # here means "same group", never "same class as the row above".
                ax.imshow(field.T, cmap="Set2", vmin=0, vmax=7, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=8)
            if row == 0:
                ax.set_title(f"val image {col}", fontsize=8)
    fig.suptitle(
        f"Segmentations at {budget} steps, two eval seeds\n"
        "prediction colours are arbitrary cluster ids: compare groupings, not colours",
        fontsize=10,
    )
    fig.tight_layout()
    path = OUT_DIR / f"pursuit-b-maps-{budget}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, inputs = _budget_data()

    figures = [_fig_curve(rows), _fig_instability(rows), _fig_loss(rows)]
    weights_fig, ckpts = _fig_weights(rows)
    figures.append(weights_fig)
    inputs += ckpts

    per_image = _per_image_budget_data()
    if per_image is not None:
        scores, overlap, per_image_inputs = per_image
        figures.extend(
            [
                _fig_per_image_trajectories(scores, overlap),
                _fig_per_image_distributions(scores, overlap),
            ]
        )
        inputs += per_image_inputs

    # These need the per-image data the first programme discarded, so they are
    # drawn only when the recomputed artefacts are present. Their absence is
    # not an error; it is the state the repo was in before the recompute.
    for budget in (4000, 10000):
        extra = _fig_hard_vs_unstable(budget)
        if extra is not None:
            figures.append(extra)
            inputs.append(ARTEFACTS / f"pursuit-b-labels2-{budget}.json")
    for budget in (500, 10000):
        extra = _fig_maps(budget)
        if extra is not None:
            figures.append(extra)
            inputs.append(ARTEFACTS / f"pursuit-b-maps-{budget}.npz")

    manifest = {
        "figures": {p.name: _sha256(p) for p in figures},
        "inputs": {p.name: _sha256(p) for p in inputs},
        "note": (
            "Figure hashes depend on the matplotlib version that drew them; the "
            "reproducibility claim is on the inputs, not the image bytes."
        ),
        "budgets": list(BUDGETS),
        "per_image_disagreeing_pairs": {
            str(r["steps"]): r["per_image_disagreeing_pairs"].tolist() for r in rows
        },
    }
    out = ARTEFACTS / "pursuit-b-figures-manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(figures)} figures to {OUT_DIR} and {out.name}", file=sys.stderr)
    for name, digest in manifest["figures"].items():
        print(f"  {name}  {digest[:16]}")
    return manifest


if __name__ == "__main__":
    result = main()
    worst = result["per_image_disagreeing_pairs"]["4000"]
    top = sorted(range(len(worst)), key=lambda i: -worst[i])[:5]
    print("\nmost seed-unstable val images at 4000 steps (image: pairs disagreeing):")
    print("  " + ", ".join(f"{i}: {worst[i]}" for i in top))
