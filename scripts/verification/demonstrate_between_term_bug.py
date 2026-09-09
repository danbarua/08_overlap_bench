"""How-to: reproduce NOTE_40's finding in about two seconds, no project data
required.

This is the minimal, standalone version of the bug found by
check_surrogate_at_baseline.py: `loss_m1.py`'s between-object term computes
`|z_o . conj(z_o')| / (|z_o| |z_o'|)`, which is IDENTICALLY 1 for any two
nonzero complex numbers -- modulus of a product is the product of moduli,
and modulus of a conjugate equals the modulus of the original, so
`|z * conj(w)| = |z| * |w|` always, regardless of the angle between z and w.
The intended formula (a real cosine similarity between phases) needed the
REAL PART of the product, not its absolute value:
`Re(z_o . conj(z_o')) / (|z_o| |z_o'|) = cos(phase_o - phase_o')`.

Two demonstrations:

1. Direct algebra on random complex numbers: the coded formula reads 1.0 no
   matter the phase difference; the corrected formula varies with it, as a
   cosine similarity should.

2. The actual loss (`phase_coherence_loss` from `loss_m1.py`) evaluated on
   two synthetic "objects" whose phases are moved from perfectly aligned
   (delta_phi=0) through perpendicular (pi/2) to perfectly opposed (pi).
   A working between-within loss should penalise pi/2 and pi far more than
   0. The actual loss barely moves, because its between term never saw the
   phase difference in the first place.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from overlap_bench.loss_m1 import phase_coherence_loss


def algebra_demo() -> None:
    print("--- 1. |z . conj(w)| / (|z||w|) is identically 1, for any z, w ---")
    torch.manual_seed(0)
    for _ in range(4):
        z = torch.polar(torch.tensor(1.0 + torch.rand(1).item(), dtype=torch.float64),
                         (torch.rand(1) * 2 * torch.pi).to(torch.float64))
        w = torch.polar(torch.tensor(1.0 + torch.rand(1).item(), dtype=torch.float64),
                         (torch.rand(1) * 2 * torch.pi).to(torch.float64))
        overlap = z * w.conj()
        mag_product = z.abs() * w.abs()
        coded = (overlap.abs() / mag_product).item()          # loss_m1.py's actual formula
        correct = (overlap.real / mag_product).item()         # what it should be: cos(phase diff)
        phase_diff = torch.angle(overlap).item()
        print(f"  phase_diff={phase_diff:+.3f} rad -> coded (BUGGY)={coded:.8f}   correct=cos(phase_diff)={correct:+.4f}")


def loss_insensitivity_demo() -> None:
    print("\n--- 2. phase_coherence_loss barely moves as two objects' phases diverge ---")
    n_per_object = 50
    labels = torch.cat([torch.ones(n_per_object, dtype=torch.long), 2 * torch.ones(n_per_object, dtype=torch.long)])
    for delta_phi in [0.0, torch.pi / 4, torch.pi / 2, 3 * torch.pi / 4, torch.pi]:
        phase_object_1 = torch.zeros(n_per_object, dtype=torch.float64)
        phase_object_2 = torch.full((n_per_object,), delta_phi, dtype=torch.float64)
        x_T = torch.exp(1j * torch.cat([phase_object_1, phase_object_2]))
        loss = phase_coherence_loss(x_T, labels).item()
        print(f"  delta_phi={delta_phi:.4f} rad ({delta_phi / torch.pi:.2f}*pi): loss = {loss:.6f}"
              f"  <- should fall sharply as delta_phi grows if 'between' worked; it barely moves")


if __name__ == "__main__":
    algebra_demo()
    loss_insensitivity_demo()
    print("\nConclusion: within-object coherence is real and correctly measured (|z_o| terms).")
    print("Between-object coherence has never been measured at all -- loss ~= 1 - within,")
    print("always, for any phase configuration. See NOTE_40 on this project's LabKit record.")
