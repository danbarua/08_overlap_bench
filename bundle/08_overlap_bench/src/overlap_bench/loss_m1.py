"""M1's loss surrogate. Expression fixed in NOTE_16 before any code here ran.

Differentiable in x_T only; depends on ground-truth foreground labels
(train split, by the caller's choice of which image/labels it's given);
no readout, no k-means, no dependence on val.
"""

import torch


def phase_coherence_loss(x_T: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """loss = between-object alignment - within-object alignment.

    x_T: (N,) complex final state.
    labels: (N,) integer ground truth; positive values are foreground
        object ids, <=0 (background 0, overlap -1) excluded entirely.

    within = mean over objects o of |z_o|, z_o = mean_{i in o}(x_T[i]/|x_T[i]|).
    between = mean over unordered pairs o != o' of |z_o . conj(z_o')| / (|z_o||z_o'|).
    Minimizing this loss maximizes within-object coherence while minimizing
    cross-object phase alignment.
    """
    object_ids = sorted(int(v) for v in torch.unique(labels).tolist() if v > 0)
    if len(object_ids) < 1:
        raise ValueError("phase_coherence_loss needs at least one foreground object")
    unit_phase = x_T / (x_T.abs() + 1e-12)
    order_parameters = []
    for object_id in object_ids:
        mask = labels == object_id
        if mask.sum() < 1:
            raise ValueError(f"object {object_id} has no pixels")
        order_parameters.append(unit_phase[mask].mean())
    z = torch.stack(order_parameters)  # (K,) complex

    within = z.abs().mean()

    if len(object_ids) < 2:
        between = torch.zeros((), dtype=within.dtype, device=within.device)
    else:
        overlap = z[:, None] * z.conj()[None, :]  # (K,K) complex, entry (o,o') = z_o * conj(z_o')
        magnitude_product = z.abs()[:, None] * z.abs()[None, :] + 1e-12
        cosine = overlap.abs() / magnitude_product  # (K,K) real in [0,1]
        k = len(object_ids)
        pair_mask = ~torch.eye(k, dtype=torch.bool, device=cosine.device)
        between = cosine[pair_mask].mean()

    return between - within
