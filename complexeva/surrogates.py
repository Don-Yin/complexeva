"""Fourier surrogates and empirical tests for real, time-last signals."""

from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy
from numpy.typing import ArrayLike, NDArray

from ._validation import MAX_SAFE_AMPLITUDE, MIN_SAFE_AMPLITUDE, as_signals, check_positive_int, float64_rows

__all__ = ["SurrogateTestResult", "iaaft_surrogates", "phase_randomized_surrogates", "surrogate_test"]

Seed = int | numpy.integer | numpy.random.Generator | None
Method = Literal["phase", "iaaft"]
Alternative = Literal["two-sided", "less", "greater"]
Measure = Callable[[NDArray[numpy.float64]], ArrayLike]
Score = float | NDArray[numpy.float64]


class SurrogateTestResult(NamedTuple):
    """Observed score, mean surrogate score, their difference, and empirical p-value."""

    observed: Score
    surrogate_mean: Score
    delta: Score
    pvalue: Score


def _check_input(x: ArrayLike, n: int | numpy.integer) -> tuple[NDArray, int]:
    """Validate real time-last signals and the number of surrogates per signal."""
    values = as_signals(x)
    if values.shape[-1] < 2:
        raise ValueError(f"x needs at least 2 samples, got {values.shape[-1]}")
    return values, check_positive_int(n, "n")


def _fft_scale(row: NDArray[numpy.float64]) -> float:
    """Scale extreme amplitudes into a safe range for the Fourier transforms."""
    largest = float(numpy.max(numpy.abs(row)))
    return largest if 0 < largest < MIN_SAFE_AMPLITUDE or largest > MAX_SAFE_AMPLITUDE else 1.0


def _phase_copies(row: NDArray[numpy.float64], n: int, rng: numpy.random.Generator, scale: float) -> NDArray[numpy.float64]:
    """Independently rotate interior Fourier bins, leaving real endpoints fixed."""
    spectrum = numpy.fft.rfft(row if scale == 1 else row / scale)
    shuffled = numpy.broadcast_to(spectrum, (n, spectrum.size)).copy()
    stop = 1 + (row.size - 1) // 2
    angles = rng.uniform(-numpy.pi, numpy.pi, (n, stop - 1))
    shuffled[:, 1:stop] *= numpy.exp(1j * angles)
    copies = numpy.fft.irfft(shuffled, n=row.size)
    if scale != 1:
        with numpy.errstate(over="ignore"):
            copies *= scale
        if not numpy.isfinite(copies).all():
            raise ValueError(f"phase surrogate exceeds the float64 range in the units of x (maximum amplitude {scale!r})")
    return copies


def _iaaft_copies(
    row: NDArray[numpy.float64], n: int, rng: numpy.random.Generator, scale: float, max_iter: int
) -> NDArray[numpy.float64]:
    """Alternate a Fourier-magnitude projection and an exact rank remapping."""
    reference = numpy.fft.rfft(row if scale == 1 else row / scale)
    target = numpy.abs(reference)
    amplitudes = numpy.broadcast_to(numpy.sort(row), (n, row.size))
    copies = rng.permuted(amplitudes, axis=-1)
    updated = numpy.empty_like(copies)
    for _ in range(max_iter):
        current = numpy.fft.rfft(copies if scale == 1 else copies / scale)
        magnitude = numpy.abs(current)
        phase = numpy.divide(current, magnitude, out=numpy.ones_like(current), where=magnitude != 0)
        projected = target * phase
        projected[:, 0] = reference[0]
        order = numpy.fft.irfft(projected, n=row.size).argsort(axis=-1, kind="stable")
        numpy.put_along_axis(updated, order, amplitudes, axis=-1)
        if numpy.array_equal(updated, copies):
            break
        copies, updated = updated, copies
    return copies


def _copies(
    row: NDArray[numpy.float64], n: int, rng: numpy.random.Generator, method: Method, max_iter: int
) -> NDArray[numpy.float64]:
    """Generate the whole surrogate batch for one signal."""
    if (method == "phase" and row.size == 2) or numpy.all(row == row[0]):
        return numpy.broadcast_to(row, (n, row.size)).copy()
    scale = _fft_scale(row)
    return _phase_copies(row, n, rng, scale) if method == "phase" else _iaaft_copies(row, n, rng, scale, max_iter)


def _generate(values: NDArray, n: int, seed: Seed, method: Method, max_iter: int) -> NDArray[numpy.float64]:
    """Generate surrogates in input row order without copying the input batch."""
    rng = numpy.random.default_rng(seed)
    rows = float64_rows(values)
    if values.ndim == 1:
        return _copies(next(rows), n, rng, method, max_iter)
    output = numpy.empty((*values.shape[:-1], n, values.shape[-1]))
    for dest, row in zip(output.reshape(-1, n, values.shape[-1]), rows, strict=True):
        dest[:] = _copies(row, n, rng, method, max_iter)
    return output


def phase_randomized_surrogates(x: ArrayLike, n: int | numpy.integer, *, seed: Seed = None) -> NDArray[numpy.float64]:
    """Make random-phase surrogates with the input's mean and Fourier magnitudes.

    Args:
        x: Finite, real samples with at least two samples on the last axis.
        n: Positive number of copies per signal.
        seed: Seed or NumPy random generator. An integer reproduces the same
            output for the same input; a generator advances in place.

    Returns:
        A float64 array of shape ``(*x.shape[:-1], n, x.shape[-1])``. The
        zero-frequency bin and, for even lengths, the Nyquist bin are unchanged.
        The sample amplitude distribution need not match the original.
    """
    values, n = _check_input(x, n)
    return _generate(values, n, seed, "phase", 0)


def iaaft_surrogates(
    x: ArrayLike, n: int | numpy.integer, *, seed: Seed = None, max_iter: int | numpy.integer = 100
) -> NDArray[numpy.float64]:
    """Make iterative amplitude-adjusted Fourier transform (IAAFT) surrogates.

    Args:
        x: Finite, real samples with at least two samples on the last axis.
        n: Positive number of copies per signal.
        seed: Seed or NumPy random generator, as in :func:`phase_randomized_surrogates`.
        max_iter: Positive iteration cap. Iteration stops early if no copy changes.

    Returns:
        A float64 array of shape ``(*x.shape[:-1], n, x.shape[-1])``. Each
        copy starts from an independent permutation. The final rank remapping
        preserves the sorted input samples exactly in float64; it only
        approximates the Fourier magnitudes.
    """
    values, n = _check_input(x, n)
    return _generate(values, n, seed, "iaaft", check_positive_int(max_iter, "max_iter"))


def _score(measure: Measure, row: NDArray[numpy.float64]) -> float:
    """Evaluate a measure and require it to return one real scalar."""
    value = numpy.asarray(measure(row))
    if value.ndim != 0 or value.dtype.kind not in "biuf":
        raise ValueError(f"measure must return one real scalar, got shape {value.shape} and dtype {value.dtype}")
    return float(value.item())


def _pvalue(observed: float, scores: NDArray[numpy.float64], alternative: Alternative) -> float:
    """Count ties inclusively and add the observed score to the reference count."""
    if numpy.isnan(observed) or numpy.isnan(scores).any():
        return float("nan")
    total = len(scores) + 1
    lower = (1 + numpy.count_nonzero(scores <= observed)) / total
    upper = (1 + numpy.count_nonzero(scores >= observed)) / total
    if alternative == "less":
        return lower
    if alternative == "greater":
        return upper
    return min(1.0, 2 * min(lower, upper))


def surrogate_test(
    measure: Measure,
    x: ArrayLike,
    n: int | numpy.integer = 39,
    *,
    method: Method = "phase",
    seed: Seed = None,
    alternative: Alternative = "two-sided",
    max_iter: int | numpy.integer = 100,
) -> SurrogateTestResult:
    """Compare a scalar time-series measure with its score on surrogate signals.

    Args:
        measure: Callable returning one real scalar per 1D float64 signal.
            It must not modify the samples. Use ``functools.partial`` to set options.
        x: Finite, real samples with at least two samples on the last axis.
        n: Positive number of copies per signal.
        method: ``"phase"`` for phase randomization or ``"iaaft"`` for iterative
            amplitude adjustment.
        seed: Seed or NumPy random generator, as in :func:`phase_randomized_surrogates`.
        alternative: ``"less"``, ``"greater"``, or ``"two-sided"`` (doubled smaller tail, capped at one).
        max_iter: Positive iteration cap, used only with ``method="iaaft"``.

    Returns:
        Observed score, surrogate mean, observed minus surrogate mean, and
        empirical p-value. The four fields are Python floats for a single
        signal or float64 arrays of shape ``x.shape[:-1]`` for a batch. Each
        one-sided tail is ``(1 + count) / (n + 1)`` with ties included.
        A NaN score makes the corresponding p-value undefined (NaN).
    """
    values, n = _check_input(x, n)
    if not callable(measure):
        raise TypeError(f"measure must be callable, got {measure!r}")
    if method not in ("phase", "iaaft"):
        raise ValueError(f"method must be 'phase' or 'iaaft', got {method!r}")
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError(f"alternative must be 'less', 'greater', or 'two-sided', got {alternative!r}")
    iterations = check_positive_int(max_iter, "max_iter") if method == "iaaft" else 0
    rng = numpy.random.default_rng(seed)
    results = numpy.empty((*values.shape[:-1], 4))
    for dest, row in zip(results.reshape(-1, 4), float64_rows(values), strict=True):
        observed = _score(measure, row)
        copies = _copies(row, n, rng, method, iterations)
        scores = numpy.fromiter((_score(measure, copy) for copy in copies), numpy.float64, count=n)
        with numpy.errstate(invalid="ignore", over="ignore"):
            mean = float(scores.mean())
        if numpy.isinf(mean) and numpy.isfinite(scores).all():  # A finite sum can overflow before division.
            scale = numpy.max(numpy.abs(scores))
            mean = float((scores / scale).mean() * scale)
        dest[:] = observed, mean, observed - mean, _pvalue(observed, scores, alternative)
    if values.ndim == 1:
        return SurrogateTestResult(*results.tolist())
    observed, mean, delta, pvalue = numpy.moveaxis(results, -1, 0)
    return SurrogateTestResult(observed, mean, delta, pvalue)
