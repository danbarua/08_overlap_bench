"""How-to: the check that actually finds a loss/metric design bug, as opposed
to explaining a symptom of it.

Context: this project trained a model against a phase-coherence loss
surrogate and got a chance-level result. Several successive explanations
were tried and each was individually checked before being written down --
and each, once checked, turned out to be incomplete:

  1. "Amplitude blew up, washing out signal" (real, but doesn't explain WHY
     gradient descent moved that direction).
  2. "The surrogate has a trivial synchrony fixed point that training found"
     (the algebra is right, but checking the actual PARAMETER-space gradient
     at the trained point showed it was LARGER than at init, not vanishing
     -- so "found a fixed point, can't leave it" was not actually true).
  3. "Training is a noise-driven random walk since gradients are tiny"
     (plausible, but still didn't say WHY the walk drifted the same
     direction reliably).

The check that actually found the root cause: run the SAME surrogate on the
UNTRAINED baseline -- the known-good, already-verified M0 dynamics -- and
see whether it ALSO looks "collapsed" under this metric. If a model you
already trust (verified correct elsewhere, e.g. via arc1a's independently-
computed ARI) already scores near-zero loss and near-total synchrony under
your new surrogate, the surrogate isn't measuring what you think it's
measuring -- full stop, no need to keep debugging the training run. This is
what found NOTE_40 in this project's LabKit record: the surrogate's
between-object term was `|z_o . conj(z_o')| / (|z_o| |z_o'|)`, which is
`|z_o||z_o'| / (|z_o||z_o'|) == 1` identically for ANY nonzero z_o, z_o'
(modulus of a product is the product of moduli) -- it never measured phase
alignment at all, regardless of training.

General lesson: when a trained model looks bad under your loss/metric,
always ask "does a model I ALREADY trust ALSO look bad under this same
loss/metric?" before concluding the training procedure or model class is at
fault. If yes, audit the loss/metric itself first.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import overlap_bench.run_arc1cd as arc1cd
from overlap_bench.reference_repo import ensure_on_path

ensure_on_path()
from src.cv_rnn.cv_rnn_segmentation import gaussian_sheet_torch  # noqa: E402


def check_baseline_under_surrogate(images, labels, seed: int = 1) -> None:
    """Run the KNOWN-GOOD, untrained M0 dynamics through the surrogate.
    If this already reports near-zero loss / near-total synchrony, the
    surrogate is suspect -- not the (not-yet-existing) trained model."""
    from overlap_bench.loss_m1 import phase_coherence_loss

    nrow, ncol = images.shape[-2], images.shape[-1]
    k1 = gaussian_sheet_torch(nrow, ncol, 0.5, 0.9, dtype=torch.float64).real.clone()
    k2_oracle = gaussian_sheet_torch(nrow, ncol, 0.5, 0.0313, dtype=torch.float64).real.clone()
    delta_omega_oracle = torch.zeros(nrow * ncol, dtype=torch.float64)

    x0_batch, mask_batch, intensity_batch, _ = arc1cd._batched_layer1_shared_x0(images, seed, k1)
    mask_float = (~mask_batch).to(torch.float64)
    x0_masked = (x0_batch * mask_float).to(torch.complex128)
    omega2_eff = (intensity_batch + delta_omega_oracle.unsqueeze(1)) * mask_float
    with torch.no_grad():
        x = x0_masked
        for _ in range(140):
            x = arc1cd._batched_layer2_step(k2_oracle, x, mask_float, omega2_eff)

    print("--- Untrained (known-good) M0 dynamics under the surrogate ---")
    for i in range(images.shape[0]):
        x_T = x[:, i]
        truth = torch.from_numpy(np.asarray(labels[i], dtype=np.int64)).T.reshape(-1)
        object_ids = sorted(int(v) for v in torch.unique(truth).tolist() if v > 0)
        unit_phase = x_T / (x_T.abs() + 1e-12)
        zs, phases = [], []
        for oid in object_ids:
            z = unit_phase[truth == oid].mean()
            zs.append(z)
            phases.append(torch.angle(x_T[truth == oid]).mean().item())
        loss = phase_coherence_loss(x_T, truth).item()
        print(f"  image {i}: mean phases {[f'{p:.3f}' for p in phases]}, loss={loss:.3e}")
        if len(zs) == 2:
            cross = (zs[0] * zs[1].conj()).abs().item() / (zs[0].abs().item() * zs[1].abs().item())
            print(f"    'between' term reports: {cross:.8f}  <- watch this: does it depend on the")
            print(f"    phase DIFFERENCE above at all, or is it always ~1 regardless?")


def diagnose_the_formula() -> None:
    """Once the baseline looks suspicious, don't stop at 'phases differ but
    the between term is still ~1' -- go read the formula and check the
    algebra directly, on values with a genuinely large phase difference."""
    print("--- Direct algebra check on loss_m1.py's between-term formula ---")
    torch.manual_seed(0)
    for _ in range(3):
        # Two arbitrary nonzero complex numbers with a large phase difference.
        z_o = torch.polar(torch.tensor(1.0 + torch.rand(1).item()), torch.rand(1) * 2 * torch.pi)
        z_op = torch.polar(torch.tensor(1.0 + torch.rand(1).item()), torch.rand(1) * 2 * torch.pi)
        overlap = z_o * z_op.conj()
        magnitude_product = z_o.abs() * z_op.abs()
        coded_between = (overlap.abs() / magnitude_product).item()
        # What a real "phase alignment" measure should give:
        correct_cosine = (overlap.real / magnitude_product).item()
        phase_diff = (torch.angle(z_o) - torch.angle(z_op)).item()
        print(f"  z_o={z_o.item():.3f}, z_o'={z_op.item():.3f}, phase_diff={phase_diff:.3f} rad")
        print(f"    loss_m1.py's formula (overlap.abs()/mag):  {coded_between:.8f}  <- always ~1")
        print(f"    what it should be (Re(overlap)/mag = cos): {correct_cosine:.8f}  <- varies with phase")


if __name__ == "__main__":
    from overlap_bench.dataset_hashes import CAE_DIR
    with np.load(CAE_DIR / "2shapes_val.npz") as data:
        images, labels = data["images"][:3], data["labels"][:3]

    check_baseline_under_surrogate(images, labels)
    print()
    diagnose_the_formula()
    print()
    print("Conclusion: overlap.abs() discards phase information before dividing;")
    print("|a*conj(b)| == |a|*|b| for ANY complex a, b, regardless of their phases.")
    print("The fix is Re(overlap)/magnitude_product, not overlap.abs()/magnitude_product.")
