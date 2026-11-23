"""Compare sample entropy against Fourier phase-randomized surrogates."""

from typing import NamedTuple

import numpy
import torch
from numpy.typing import NDArray

from .._validation import check_positive_int
from ..sample_entropy import sample_entropy
from ._inputs import Device, Signal, _real_tensor, scalar_or_batch, tensor_rows

_FFT_BYTES = 32 * 1024**2


class SpectrumShuffleResult(NamedTuple):
    """Observed entropy, surrogate entropies, and lower-tail Monte Carlo p-values."""

    observed: float | NDArray[numpy.float64]
    shuffled: NDArray[numpy.float64]
    pvalue: float | NDArray[numpy.float64]


def _phase_shuffles(x: torch.Tensor, count: int, rng: numpy.random.Generator) -> torch.Tensor:
    """Randomize the Fourier phases independently, preserving DC and Nyquist."""
    rows, n = x.shape
    bins = n // 2 + 1
    complex_dtype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    spectrum = torch.empty((rows, bins), dtype=complex_dtype, device=x.device)
    torch.fft.rfft(x, out=spectrum)
    # Use a host RNG so a seed specifies the same phases on every device.
    angles = torch.from_numpy(rng.uniform(-numpy.pi, numpy.pi, (rows, count, bins))).to(dtype=x.dtype).to(x.device)
    rotation = torch.complex(angles.cos(), angles.sin())
    rotation[..., 0] = 1
    if n % 2 == 0:
        rotation[..., -1] = 1
    copies = x.new_empty((rows, count, n))
    return torch.fft.irfft(spectrum[:, None] * rotation, n=n, out=copies)


def spectrum_shuffle_test(
    x: Signal,
    order: int | numpy.integer = 2,
    tolerance: float | numpy.floating | numpy.integer | None = None,
    *,
    n_shuffles: int = 39,
    seed: int | None = None,
    device: Device = None,
) -> SpectrumShuffleResult:
    """Test whether sample entropy is lower than in phase-randomized copies.

    Each surrogate retains the original mean and Fourier magnitudes (up to
    roundoff), not the original sample histogram. The test conditions on the
    real DC and (even lengths only) Nyquist bins; it is not a time permutation.
    Use it to compare against a stationary, random-phase null, not to prove a
    specific nonlinear mechanism.

    The selected device generates the surrogates. The original and surrogate
    scores use CPU :func:`complexeva.sample_entropy`, including its ``order``
    and ``tolerance`` options. A fixed ``seed`` gives repeatable phases.
    ``observed`` and ``pvalue`` have the usual batch shape; ``shuffled`` adds a
    final axis of length ``n_shuffles``. The p-value is
    ``(1 + number of surrogates <= observed) / (n_shuffles + 1)``. A row with
    any NaN entropy has a NaN p-value.
    """
    n_shuffles = check_positive_int(n_shuffles, "n_shuffles")
    rng = numpy.random.default_rng(seed)
    original = _real_tensor(x).cpu().to(dtype=torch.float64).numpy() if isinstance(x, torch.Tensor) else x
    observed = numpy.asarray(sample_entropy(original, order, tolerance)).reshape(-1)
    rows, shape = tensor_rows(x, device)
    shuffled = numpy.empty((len(rows), n_shuffles))
    n = rows.shape[-1]
    chunk = max(1, _FFT_BYTES // (n_shuffles * n * 32))
    for start in range(0, len(rows), chunk):
        batch = rows[start : start + chunk]
        copies = _phase_shuffles(batch, n_shuffles, rng).reshape(-1, n).cpu().numpy()
        scores = sample_entropy(copies, order, tolerance)
        shuffled[start : start + chunk] = numpy.asarray(scores).reshape(len(batch), n_shuffles)
    pvalue = (1 + (shuffled <= observed[:, None]).sum(axis=-1)) / (n_shuffles + 1)
    pvalue[numpy.isnan(observed) | numpy.isnan(shuffled).any(axis=-1)] = numpy.nan
    return SpectrumShuffleResult(
        scalar_or_batch(observed, shape), shuffled.reshape((*shape, n_shuffles)), scalar_or_batch(pvalue, shape)
    )
