"""M1: the trainable exact linear propagator. NOTE_16 fixes the time
convention this module implements: samples at integer t = 1..200 to match
M0's 200-step trajectory shape.

x(t) = exp(A*t) @ x0, A = K + i*diag(omega), K and omega real and
trainable. Computed as exp(A) once (torch.linalg.matrix_exp), then the
full trajectory via repeated matmul -- exact for integer t (exp(A*t) =
exp(A)^t identically), not an approximation, and the same cost structure
as M0's own per-step loop.
"""

import torch
import torch.nn as nn


class LinearPropagator(nn.Module):
    def __init__(self, K_init: torch.Tensor, omega_init: torch.Tensor, dtype: torch.dtype = torch.float64):
        super().__init__()
        n = K_init.shape[0]
        if K_init.shape != (n, n):
            raise ValueError("K_init must be square")
        if omega_init.shape != (n,):
            raise ValueError("omega_init must have one entry per node")
        self.n = n
        self.K = nn.Parameter(K_init.to(dtype).clone())
        self.omega = nn.Parameter(omega_init.to(dtype).clone())

    def generator(self) -> torch.Tensor:
        """A = K + i*diag(omega), complex."""
        complex_dtype = torch.complex128 if self.K.dtype == torch.float64 else torch.complex64
        return self.K.to(complex_dtype) + 1j * torch.diag(self.omega.to(complex_dtype))

    def one_step(self) -> torch.Tensor:
        """exp(A), the operator that advances one unit of t. Exact for any
        integer number of subsequent applications: exp(A*t) = exp(A)^t."""
        return torch.linalg.matrix_exp(self.generator())

    def trajectory(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        """Returns (n, steps) states at t = 1..steps. x0 is the t=0 state,
        not included in the output (matching run_2layer_torch's save_x[:,0]
        convention would require the caller to prepend it separately)."""
        step_operator = self.one_step()
        states = torch.empty((self.n, steps), dtype=step_operator.dtype, device=step_operator.device)
        x = x0.to(step_operator.dtype)
        for t in range(steps):
            x = step_operator @ x
            states[:, t] = x
        return states
