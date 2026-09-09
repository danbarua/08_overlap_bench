"""How-to: diagnosing whether a finite-difference gradient check is telling you
something real, or just measuring its own numerical floor.

Context this came from: harness-M1d's H1b-d check reported "PASS" using
absolute finite-difference error (~1e-10) against a criterion that actually
asked for *relative* error. Recomputing the relative error properly showed
6-11% at several K_2 entries -- alarming, until you notice the analytic
gradients themselves are tiny (~1e-9). A fixed-eps, per-entry central
difference cannot resolve a gradient that small to 1e-6 relative precision:
the check's own floating-point-cancellation floor is comparable to the
quantity being measured. See NOTE_27/NOTE_32/NOTE_37 in this project's
LabKit record for the full trail.

Four escalating techniques, each more informative than the last:

1. Per-entry check at a fixed eps (naive, and *ill-conditioned* whenever
   the true gradient is small). Central-difference cancellation error is
   roughly ulp(L)/eps ~ 1e-16*|L|/eps: with |L| ~ 1e-2 and eps=1e-6, that
   floor is ~1e-12 absolute -- the same order as the ~1e-9 K_2 gradients
   themselves, which is exactly why the relative error reads 1-10%.
2. Directional-derivative check: perturb along the unit gradient direction
   d = g/||g||, compare [L(p+eps*d) - L(p-eps*d)]/(2*eps) against g.d =
   ||g||. This tests an O(||g||)-scale quantity instead of a possibly-tiny
   per-entry value, so it is far better conditioned.
3. An eps-sweep: if relative error falls ~100x per decade of eps (O(eps^2)
   truncation error shrinking) and then rises again at very small eps
   (floating-point cancellation), that V-shape is the signature of a
   *correct* gradient tested imprecisely -- not a wrong one. Report the
   FLOOR of the sweep, not the single lowest point: several nearby eps
   values often land on the identical error (the loss difference itself
   is quantised to the same few ulps there), which is the honest
   resolution limit, not one eps happening to do better than its neighbours.
4. Scaled eps, matched to the parameter's own magnitude (here
   eps=1e-3*max(|param|,1)): this sidesteps the conditioning problem
   instead of just diagnosing it. At eps=1e-3, L(p+eps)-L(p-eps) is of
   order 1e-12 or larger -- far above the loss's own ulp(L)~1e-18 floor --
   while truncation error for a single-entry perturbation of this size
   stays small, because the loss is smooth in any one entry even though
   the full 140-step recurrence is highly nonlinear overall.

Run against this project's actual harness-M1d model and checkpoint to
reproduce the numbers discussed above.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from overlap_bench.harness_m1d import _build_m1d_at_init
from overlap_bench.loss_m1 import phase_coherence_loss
from overlap_bench.dataset_hashes import CAE_DIR
from overlap_bench.reference_repo import ensure_on_path

ensure_on_path()
from src.cv_rnn.cv_rnn_segmentation import run_2layer_torch  # noqa: E402


def _build_reference_model():
    """One M1d model at initialisation, and the loss closure to test against."""
    image = torch.from_numpy(np.asarray(np.load(CAE_DIR / "2shapes_train.npz")["images"][0, 0], dtype=np.float64))
    nrow, ncol = image.shape
    generator = torch.Generator().manual_seed(1)
    states_m0, mask = run_2layer_torch(image, generator=generator)
    x0_masked = states_m0[:, 0].masked_fill(mask, 0)
    with np.load(CAE_DIR / "2shapes_train.npz") as data:
        truth = torch.from_numpy(np.asarray(data["labels"][0], dtype=np.int64)).T.reshape(-1)
    model = _build_m1d_at_init(image, mask, nrow, ncol)

    def loss_fn():
        x_T = model.orbit(x0_masked, steps=140)[:, -1]
        return phase_coherence_loss(x_T, truth)

    return model, loss_fn


def per_entry_check(model, loss_fn, eps: float = 1e-6) -> None:
    """Technique 1: naive, fixed-eps, per-entry. Reports RELATIVE error
    explicitly -- absolute error alone (as originally logged) hides how bad
    this can be when the true gradient is tiny."""
    model.zero_grad()
    loss = loss_fn()
    loss.backward()
    analytic = model.K2.grad.clone()
    unmasked_idx = (~model.mask).nonzero(as_tuple=True)[0]

    print(f"--- Technique 1: per-entry check, fixed eps={eps:.0e} ---")
    with torch.no_grad():
        for i, j in [(unmasked_idx[0].item(), unmasked_idx[0].item()),
                     (unmasked_idx[5].item(), unmasked_idx[7].item())]:
            original = model.K2[i, j].item()
            model.K2[i, j] = original + eps
            plus = loss_fn().item()
            model.K2[i, j] = original - eps
            minus = loss_fn().item()
            model.K2[i, j] = original
            fd = (plus - minus) / (2 * eps)
            a = analytic[i, j].item()
            rel_err = abs(fd - a) / max(abs(a), 1e-30)
            print(f"  K[{i},{j}]: analytic={a:.3e} fd={fd:.3e} rel_err={rel_err:.3e}"
                  f"  <- looks alarming, but see technique 3")


def directional_derivative_check(model, loss_fn, eps: float = 1e-4) -> float:
    """Technique 2: perturb along the unit gradient direction. Tests an
    O(||g||)-scale quantity, not a possibly-tiny individual entry."""
    model.zero_grad()
    loss = loss_fn()
    loss.backward()
    g = model.K2.grad.clone()
    d = g / g.norm()
    analytic_dir = g.norm().item()  # g . d == ||g|| by construction

    with torch.no_grad():
        original = model.K2.data.clone()
        model.K2.data = original + eps * d
        plus = loss_fn().item()
        model.K2.data = original - eps * d
        minus = loss_fn().item()
        model.K2.data = original
    fd_dir = (plus - minus) / (2 * eps)
    rel_err = abs(fd_dir - analytic_dir) / abs(analytic_dir)
    print(f"--- Technique 2: directional derivative, eps={eps:.0e} ---")
    print(f"  analytic (=||g||)={analytic_dir:.6e}  fd={fd_dir:.6e}  rel_err={rel_err:.3e}")
    return rel_err


def eps_sweep(model, loss_fn, eps_values=(1e-3, 1e-4, 1e-5, 3e-6, 1.5e-6, 1e-6, 1e-7)) -> None:
    """Technique 3: sweep eps and look at the shape of the error curve.
    Falls ~100x/decade then rises again => truncation-then-cancellation,
    i.e. a correct gradient tested imprecisely. Stays flat/large => a real
    discrepancy worth reporting as such."""
    model.zero_grad()
    loss = loss_fn()
    loss.backward()
    g = model.K2.grad.clone()
    d = g / g.norm()
    analytic_dir = g.norm().item()

    print("--- Technique 3: eps sweep (look for the V-shape) ---")
    original = model.K2.data.clone()
    for eps in eps_values:
        with torch.no_grad():
            model.K2.data = original + eps * d
            plus = loss_fn().item()
            model.K2.data = original - eps * d
            minus = loss_fn().item()
            model.K2.data = original
        fd = (plus - minus) / (2 * eps)
        rel_err = abs(fd - analytic_dir) / abs(analytic_dir)
        print(f"  eps={eps:.1e}: rel_err={rel_err:.3e}")


def scaled_eps_check(model, loss_fn) -> bool:
    """The method actually adopted for this project's CRIT_22 (see
    harness_m1d.py:h1b_d_gradient_check_crit22): scale eps to the parameter's
    own magnitude (here, 1e-3*max(|param|,1)) and use an absolute bound where
    the analytic gradient is itself below the relative-check's floor. This
    sidesteps the conditioning problem instead of just diagnosing it."""
    model.zero_grad()
    loss = loss_fn()
    loss.backward()
    analytic = model.K2.grad.clone()
    unmasked_idx = (~model.mask).nonzero(as_tuple=True)[0]

    print("--- Scaled-eps method (what actually resolved this) ---")
    all_pass = True
    with torch.no_grad():
        for i, j in [(unmasked_idx[0].item(), unmasked_idx[0].item()),
                     (unmasked_idx[5].item(), unmasked_idx[7].item())]:
            original = model.K2[i, j].item()
            eps = 1e-3 * max(abs(original), 1.0)
            model.K2[i, j] = original + eps
            plus = loss_fn().item()
            model.K2[i, j] = original - eps
            minus = loss_fn().item()
            model.K2[i, j] = original
            fd = (plus - minus) / (2 * eps)
            a = analytic[i, j].item()
            abs_err = abs(fd - a)
            if abs(a) > 1e-8:
                rel_err = abs_err / abs(a)
                ok = rel_err < 1e-3
                print(f"  K[{i},{j}]: rel_err={rel_err:.3e} -> {'PASS' if ok else 'FAIL'}")
            else:
                ok = abs_err < 1e-10
                print(f"  K[{i},{j}]: abs_err={abs_err:.3e} -> {'PASS' if ok else 'FAIL'}")
            all_pass = all_pass and ok
    return all_pass


if __name__ == "__main__":
    model, loss_fn = _build_reference_model()
    per_entry_check(model, loss_fn)
    print()
    directional_derivative_check(model, loss_fn)
    print()
    eps_sweep(model, loss_fn)
    print()
    print("all_pass:", scaled_eps_check(model, loss_fn))
