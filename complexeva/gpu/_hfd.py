"""Higuchi fractal dimension on the selected PyTorch device."""

import numpy
import torch
from numpy.typing import NDArray

from .._validation import check_positive_int
from ._inputs import Device, Signal, scalar_or_batch, tensor_rows


def _hfd_rows(x: torch.Tensor, k_max: int) -> torch.Tensor:
    """Compute the Higuchi curve-length regression of each prepared row."""
    n = x.shape[-1]
    lengths = []
    for k in range(1, k_max + 1):
        diff = (x[:, k:] - x[:, :-k]).abs()
        # Interleave offsets in the last dimension; pad incomplete groups with zero increments.
        paths = torch.nn.functional.pad(diff, (0, -diff.shape[-1] % k)).reshape(len(x), (n - 1) // k, k).sum(dim=1)
        steps = (n - 1 - torch.arange(k, device=x.device)) // k
        lm = paths * ((n - 1) / (steps.to(x.dtype) * k * k))
        lengths.append(lm.mean(dim=-1).log())
    y = torch.stack(lengths, dim=-1)
    log_inverse_k = -torch.arange(1, k_max + 1, device=x.device, dtype=x.dtype).log()
    centered = log_inverse_k - log_inverse_k.mean()
    return ((y - y.mean(dim=-1, keepdim=True)) * centered).sum(dim=-1) / centered.square().sum()


def hfd(x: Signal, k_max: int | numpy.integer = 10, *, device: Device = None) -> float | NDArray[numpy.float64]:
    """Higuchi fractal dimension along the last axis, using PyTorch.

    As in :func:`complexeva.hfd`, ``2 <= k_max <= x.shape[-1] // 2``. The
    result is a Python float or a NumPy float64 batch. Undefined regressions
    return NaN. MPS computes in float32; CPU and CUDA use float64.
    """
    rows, shape = tensor_rows(x, device)
    n = rows.shape[-1]
    k_max = check_positive_int(k_max, "k_max", minimum=2)
    if k_max > n // 2:
        raise ValueError(f"k_max must be between 2 and x.shape[-1] // 2 = {n // 2}, got {k_max}")
    result = _hfd_rows(rows, k_max).cpu().numpy().astype(numpy.float64, copy=False) if len(rows) else numpy.empty(0)
    return scalar_or_batch(result, shape)
