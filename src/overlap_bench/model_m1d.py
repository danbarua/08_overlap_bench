"""M1d: the trainable discrete recurrence. NOTE_20 (superseding NOTE_18) locks
this module's scope: M0's layer 2 only, with K_2 and omega_2 trainable, given
M0's own (fixed, non-differentiable) mask from its unmodified layer 1.

No matrix exponential, no matrix logarithm -- the recurrence x_{n+1} = A_2 x_n
(A_2 = K_2 + i*diag(omega_2), applied by matvec) is M0's own layer-2 formula
directly (cv_rnn_segmentation.py:157-166), made trainable. Masked rows/cols of
K_2 and masked entries of omega_2 stay exactly zero -- enforced at every
forward call via multiplicative buffers, not just at initialisation, so the
optimiser cannot move them off zero.
"""

import torch
import torch.nn as nn


class LinearRecurrence2(nn.Module):
    """M0's layer 2 (``x_{n+1} = A_2 x_n``) with K_2 and omega_2 trainable on
    unmasked pixels only. ``k2_init`` and ``omega2_init`` must already have
    mask zeroing applied (M0's own convention); ``mask`` fixes which rows,
    columns and entries stay non-trainable at exactly zero.
    """

    def __init__(self, k2_init: torch.Tensor, omega2_init: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        n = k2_init.shape[0]
        if k2_init.shape != (n, n):
            raise ValueError("k2_init must be square")
        if omega2_init.shape != (n,):
            raise ValueError("omega2_init must have shape (N,)")
        if mask.shape != (n,) or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean vector of shape (N,)")

        self.K2 = nn.Parameter(k2_init.clone().to(torch.float64))
        self.omega2 = nn.Parameter(omega2_init.clone().to(torch.float64))

        unmasked = (~mask).to(torch.float64)
        mask2d = unmasked.unsqueeze(0) * unmasked.unsqueeze(1)  # zero if either endpoint masked
        self.register_buffer("mask2d", mask2d)
        self.register_buffer("unmasked", unmasked)
        self.register_buffer("mask", mask)

    def generator(self) -> torch.Tensor:
        """A_2 = K_2 + i*diag(omega_2), with masked rows/cols/entries forced to zero."""
        k2_effective = self.K2 * self.mask2d
        omega2_effective = self.omega2 * self.unmasked
        return k2_effective.to(torch.complex128) + 1j * torch.diag(omega2_effective).to(torch.complex128)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """One layer-2 step: x_next = K_2 x + i*omega_2*x (M0's own elementwise form,
        not a dense generator matmul, to match M0's exact floating-point operations)."""
        k2_effective = self.K2 * self.mask2d
        omega2_effective = self.omega2 * self.unmasked
        return k2_effective.to(torch.complex128) @ x + 1j * omega2_effective.to(torch.complex128) * x

    def orbit(self, x0_masked: torch.Tensor, steps: int = 140) -> torch.Tensor:
        """Layer-2 orbit from the restarted, masked initial state. Returns
        ``(N, steps)``, columns corresponding to M0's saved indices 60..199
        for the default ``steps=140``."""
        n = x0_masked.shape[0]
        states = torch.empty((n, steps), dtype=torch.complex128, device=x0_masked.device)
        x = x0_masked.to(torch.complex128)
        for i in range(steps):
            x = self.step(x)
            states[:, i] = x
        return states
