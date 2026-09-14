#!/usr/bin/env python3
"""Regenerate figures/pursuit-b-k2-analysis.png from outputs/pursuit-b-mechanism.json.

Fixes a conflation in the original hand-built figure: the top-left panel's
"K0 estimate = 75.3" reference line was NOT the measured initial K2 norm.
It was `k2_norm_at_10k / best_scalar_multiple_of_k0`, i.e. what K0 would
have to be for the trained matrix to be exactly a scalar rescale of it --
a shape-preservation diagnostic from the mechanism's own scalar-fit check,
not a norm measurement. The actual measured K0 (`weight_geometry.k0_frobenius`
in the mechanism JSON) is a different, smaller number. Plotting both on the
same axis without distinguishing them let "1.15x growth" read as "raw norm
barely grew," when the real raw growth (measured K0 -> K_10k) is ~3.1x.

Usage:
    python scripts/verification/plot_k2_analysis.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MECH_JSON = ROOT / "outputs" / "pursuit-b-mechanism.json"
OUT_PNG = ROOT / "figures" / "pursuit-b-k2-analysis.png"


def main() -> None:
    mech = json.loads(MECH_JSON.read_text())
    wg = mech["weight_geometry"]
    k0 = wg["k0_frobenius"]
    records = wg["checkpoints"]

    steps = [r["steps"] for r in records]
    k2_norm = [r["k_over_k0_frobenius"] * k0 for r in records]
    delta_omega = [r["delta_omega_l2"] for r in records]
    ratio = [k / d for k, d in zip(k2_norm, delta_omega)]

    k2_10k = k2_norm[-1]
    scalar_fit_factor_10k = records[-1]["best_scalar_multiple_of_k0"]
    scalar_fit_implied_k0 = k2_10k / scalar_fit_factor_10k
    raw_growth = k2_10k / k0

    fig = plt.figure(figsize=(15.68, 11.22))
    gs = fig.add_gridspec(2, 2)

    # --- Top-left: corrected K2 norm growth ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(
        [0] + steps, [k0] + k2_norm, "o-", color="tab:blue",
        label="K2 L2 norm (measured)",
    )
    ax.axhline(
        k0, color="tab:green", linestyle="--", linewidth=1.5,
        label=f"Measured K\u2080 (step 0) = {k0:.1f}",
    )
    ax.axhline(
        scalar_fit_implied_k0, color="tab:red", linestyle=":", linewidth=1.5,
        label=(
            f"Scalar-fit-implied K\u2080 = {scalar_fit_implied_k0:.1f}\n"
            f"(backed out: K\u2081\u2080\u2096 / {scalar_fit_factor_10k:.3f}, not measured)"
        ),
    )
    ax.annotate(
        f"Raw growth (measured K\u2080 \u2192 K\u2081\u2080\u2096): {raw_growth:.2f}\u00d7\n"
        f"Scalar-fit residual growth (shape-preserving\n"
        f"component only, mechanism's own metric): {scalar_fit_factor_10k:.3f}\u00d7",
        xy=(0.03, 0.7), xycoords="axes fraction", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"),
    )
    ax.set_xlabel("Training Steps", fontweight="bold")
    ax.set_ylabel("K2 L2 Norm", fontweight="bold")
    ax.set_title("K2 Norm Growth: Measured vs Scalar-Fit Reference", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    # --- Top-right: delta_omega growth ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, delta_omega, "o-", color="tab:orange", label="delta_omega L2 norm")
    ax.set_xlabel("Training Steps", fontweight="bold")
    ax.set_ylabel("delta_omega L2 Norm", fontweight="bold")
    ax.set_title("delta_omega Growth", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    # --- Bottom-left: K2/delta_omega ratio ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, ratio, "o-", color="tab:red", label="K2 / delta_omega")
    ax.axhline(ratio[-1], color="salmon", linestyle="--", label=f"Final ratio = {ratio[-1]:.1f}\u00d7")
    ax.set_yscale("log")
    ax.set_xlabel("Training Steps", fontweight="bold")
    ax.set_ylabel("Norm Ratio", fontweight="bold")
    ax.set_title("K2 and delta_omega are Uncoupled", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    # --- Bottom-right: findings text ---
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    text = (
        "MECHANISM ANALYSIS FINDINGS:\n\n"
        "\u2713 K2 Norm Growth (two distinct quantities):\n"
        f"  \u2022 Measured K\u2080 (step 0): {k0:.2f} (Frobenius)\n"
        f"  \u2022 At 10k: {k2_10k:.2f} (Frobenius)\n"
        f"  \u2022 Raw growth: {raw_growth:.2f}\u00d7\n"
        f"  \u2022 Scalar-fit residual growth: {scalar_fit_factor_10k:.3f}\u00d7\n"
        "    (best \u03b1 s.t. K\u2081\u2080\u2096 \u2248 \u03b1\u00b7K\u2080; a shape-\n"
        "    preservation metric, NOT raw norm growth)\n\n"
        "\u2713 Norm-Rescaling (mechanism lines 49-51):\n"
        "  \u2022 Applied to oscillator STATE vectors\n"
        "  \u2022 Prevents iteration overflow\n"
        "  \u2022 Preserves phase & projective state\n"
        "  \u2022 NOT applied to K2 itself\n\n"
        "\u2717 K2 \u2194 delta_omega Asymmetry:\n"
        f"  \u2022 K2 L2 norm: {k2_10k:.2f}\n"
        f"  \u2022 delta_omega L2: {delta_omega[-1]:.2f}\n"
        f"  \u2022 Ratio: {ratio[-1]:.1f}\u00d7 (uncoupled; non-monotonic,\n"
        f"    dips to ~{min(ratio):.0f}\u00d7 mid-training)\n\n"
        "\u2713 Conclusion (mechanism lines 240-248):\n"
        "  \u2022 Training changes K2 into broad,\n"
        "    asymmetric operator\n"
        "  \u2022 Phase coherence increases\n"
        "  \u2022 At 10k: projective alignment\n"
        "    0.9999999999999 over 500 states\n"
        "  \u2022 All tested seeds yield same partitions\n"
    )
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    fig.suptitle(
        "K2 Weight Analysis: Norm Growth vs Mechanism Expectations (corrected)",
        fontweight="bold", fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    print(f"saved {OUT_PNG}")
    print(
        f"measured K0={k0:.2f}, K10k={k2_10k:.2f}, raw growth={raw_growth:.3f}x, "
        f"scalar-fit factor={scalar_fit_factor_10k:.4f}x, "
        f"scalar-fit-implied K0={scalar_fit_implied_k0:.2f}"
    )


if __name__ == "__main__":
    main()
