"""The between-object term NOTE_16 intended, and NOTE_44 named.

`loss_m1.phase_coherence_loss` is left exactly as it was: it is what
`arc1c-d` trained under, it is cited by `ART_12`/`CLM_21`, and editing it
would rewrite what happened. Its between term is
`|z_o conj(z_o')| / (|z_o||z_o'|)`, which is 1 for any complex pair
(`NOTE_41`) — so the loss it computes is `1 - within` and contains no
term against different objects sharing a phase.

This module is the corrected term, as a separate function:

    between = mean over unordered pairs of
        Re(z_o conj(z_o')) / (|z_o| |z_o'|)  =  cos(phi_o - phi_o')

`within` is unchanged. `loss = between - within`, minimised, so the
optimum is -2 (every object internally coherent, every pair antiphase)
rather than the old formula's -1 ceiling.

`assert_known_answers()` is `NOTE_49`'s prespecified check with
`NOTE_50`'s tolerance: four constructed cases whose expected values were
derived by hand and are shown in the note, run against this code before
any training. Case 2 is the one that fails the old formula.
"""

from __future__ import annotations

import math

import torch

# NOTE_50: 1e-12 sits at the scale of the denominator guard below, so case 1
# fails on the boundary. 1e-9 is loose enough to clear the guard and tight
# enough that a sign or formula error cannot pass.
KNOWN_ANSWER_TOLERANCE = 1e-9

_GUARD = 1e-12


def object_order_parameters(x_T: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-object mean unit phase, `z_o`, for positive labels in ascending order.

    Overlap (-1) and background (0) pixels take no part, matching the meter.
    """
    unit_phase = x_T / (x_T.abs() + _GUARD)
    object_ids = [int(o) for o in torch.unique(labels) if int(o) > 0]
    return torch.stack([unit_phase[labels == o].mean() for o in object_ids])


def cos_coherence_loss(x_T: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """loss = between - within, with between the real part rather than the modulus.

    x_T: (N,) complex final state. labels: (N,) int64, -1 overlap, 0 background,
    positive per object.
    """
    z = object_order_parameters(x_T, labels)
    within = z.abs().mean()

    if z.shape[0] < 2:
        # One object: no pair exists, so between is not defined. Zero, and the
        # loss reduces to -within. The Arc-1 datasets always have two.
        between = torch.zeros((), dtype=within.dtype, device=within.device)
    else:
        overlap = z[:, None] * z.conj()[None, :]
        magnitude_product = z.abs()[:, None] * z.abs()[None, :] + _GUARD
        # The correction. `.real`, not `.abs()`: the modulus of a product is the
        # product of the moduli, so `.abs()` here is identically the denominator.
        cosine = overlap.real / magnitude_product
        k = z.shape[0]
        pair_mask = ~torch.eye(k, dtype=torch.bool, device=cosine.device)
        between = cosine[pair_mask].mean()

    return between - within


def _two_object_labels(n_per_object: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.ones(n_per_object, dtype=torch.int64),
            torch.full((n_per_object,), 2, dtype=torch.int64),
        ]
    )


def assert_known_answers(tolerance: float = KNOWN_ANSWER_TOLERANCE) -> dict:
    """NOTE_49's four cases, expected values derived by hand in that note.

    Raises AssertionError on any mismatch. Returns the measured values so a
    caller can record them. Runs in milliseconds; call it before training.
    """
    phi = 0.7
    labels = _two_object_labels(4)
    results: dict[str, dict[str, float]] = {}

    def case(name: str, x_T: torch.Tensor, expected: float) -> None:
        got = float(cos_coherence_loss(x_T, labels))
        results[name] = {"expected": expected, "got": got, "error": abs(got - expected)}
        assert abs(got - expected) < tolerance, (
            f"{name}: expected {expected!r}, got {got!r} "
            f"(tolerance {tolerance!r}). The loss does not compute what "
            f"NOTE_49 says it should; do not train."
        )

    # (1) both objects coherent, in phase. within=1, between=cos 0=1, loss=0.
    a = torch.full((4,), math.cos(phi) + 1j * math.sin(phi), dtype=torch.complex128)
    case("in_phase", torch.cat([a, a]), 0.0)

    # (2) both coherent, antiphase. within=1, between=cos pi=-1, loss=-2.
    # This is the case the old formula fails: it returns 0 here.
    b = torch.full(
        (4,), math.cos(phi + math.pi) + 1j * math.sin(phi + math.pi), dtype=torch.complex128
    )
    case("antiphase", torch.cat([a, b]), -2.0)

    # (3) both coherent, d radians apart. loss = cos(d) - 1.
    d = 0.8627
    c = torch.full((4,), math.cos(phi + d) + 1j * math.sin(phi + d), dtype=torch.complex128)
    case("d_apart", torch.cat([a, c]), math.cos(d) - 1.0)

    # (4) object A half at phi+pi/3 and half at phi-pi/3, so z_A = 0.5 e^{i phi};
    # object B coherent at phi. within=(0.5+1)/2=0.75, between=Re(0.5)/0.5=1,
    # loss = 1 - 0.75 = 0.25.
    hi = torch.full((2,), math.cos(phi + math.pi / 3) + 1j * math.sin(phi + math.pi / 3), dtype=torch.complex128)
    lo = torch.full((2,), math.cos(phi - math.pi / 3) + 1j * math.sin(phi - math.pi / 3), dtype=torch.complex128)
    case("half_coherent", torch.cat([hi, lo, a]), 0.25)

    return results


def assert_old_formula_fails_case_two() -> dict:
    """The control on the control: the modulus form returns 0 for antiphase.

    If this ever stops holding, either `loss_m1` changed (it must not) or the
    construction below no longer builds an antiphase pair, and
    `assert_known_answers` above is no longer testing what it claims.
    """
    from overlap_bench.loss_m1 import phase_coherence_loss

    phi = 0.7
    labels = _two_object_labels(4)
    a = torch.full((4,), math.cos(phi) + 1j * math.sin(phi), dtype=torch.complex128)
    b = torch.full(
        (4,), math.cos(phi + math.pi) + 1j * math.sin(phi + math.pi), dtype=torch.complex128
    )
    old = float(phase_coherence_loss(torch.cat([a, b]), labels))
    new = float(cos_coherence_loss(torch.cat([a, b]), labels))
    assert abs(old - 0.0) < 1e-9, f"old formula gave {old!r} on antiphase, expected 0.0"
    assert abs(new - (-2.0)) < KNOWN_ANSWER_TOLERANCE
    return {"old_formula_antiphase": old, "corrected_antiphase": new}
