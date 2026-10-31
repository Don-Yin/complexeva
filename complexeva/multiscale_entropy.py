import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_array, as_signals, check_finite_real, check_positive_int, float64_rows, unit_stride_rows
from .sample_entropy import _match_counts


@njit(cache=True)
def _is_constant(row):
    """Recognize exact constants before rounded means can give them a nonzero std."""
    for j in range(1, len(row)):
        if row[j] != row[0]:
            return False
    return True


@njit(cache=True)
def _multiscale_rows(values, direct, scales, order, tolerances):
    """Reuse one coarse-graining buffer and the exact sample-entropy counting kernel."""
    n = values.shape[-1]
    rows = values.size // n
    out = numpy.empty((rows, len(scales)))
    buffer = numpy.empty(n) if direct is None else None
    minimum_coarse_scale = n + 1
    for scale in scales:
        if scale > 1:
            minimum_coarse_scale = min(minimum_coarse_scale, scale)
    coarse = numpy.empty(n // minimum_coarse_scale)
    for i in range(rows):
        if direct is None:
            assert buffer is not None
            for j in range(n):
                buffer[j] = values.flat[i * n + j]
            row = buffer
        else:
            row = direct[i]
        for j, scale in enumerate(scales):
            if tolerances[i] == 0:
                out[i, j] = numpy.nan
                continue
            if scale == 1:
                current = row
            else:
                length = n // scale
                for k in range(length):
                    total = 0.0
                    for offset in range(scale):
                        total += row[k * scale + offset]
                    coarse[k] = total / scale
                current = coarse[:length]
            count_m, count_m1 = _match_counts(current, order, tolerances[i])
            out[i, j] = numpy.nan if count_m == 0 else numpy.inf if count_m1 == 0 else -numpy.log(count_m1 / count_m)
    return out


def _scales_array(scales, n, order):
    """Validate explicit scale factors or expand a positive maximum scale."""
    if isinstance(scales, (int, numpy.integer)):
        maximum = check_positive_int(scales, "scales")
        factors = numpy.arange(1, maximum + 1, dtype=numpy.intp)
    else:
        factors = as_array(scales, "scales")
        if factors.ndim != 1 or factors.size == 0:
            raise ValueError(f"scales must be a nonempty 1D sequence of positive integers, got {scales!r}")
        maximum = max(check_positive_int(factor, "scales") for factor in factors)
        factors = numpy.asarray(factors, dtype=numpy.intp)
    if maximum > n // (order + 2):
        raise ValueError(f"x needs at least {maximum * (order + 2)} samples for scale {maximum}, got {n}")
    return factors


def multiscale_entropy(
    x: ArrayLike,
    scales: int | numpy.integer | ArrayLike = 20,
    order: int | numpy.integer = 2,
    r: float | numpy.floating | numpy.integer | None = None,
) -> NDArray[numpy.float64]:
    """Costa multiscale entropy: sample entropy of nonoverlapping block means, in nats.

    Args:
        x: Finite, real samples with time on the last axis. At every requested scale,
            floor(len(row) / scale) must be at least order + 2.
        scales: Positive maximum scale, meaning 1 through scales inclusive, or a nonempty
            1D sequence of positive integer factors. Explicit factors retain their order
            and repetitions. Defaults to 20, the range used by Costa et al. (2002).
        order: Sample-entropy template length before extension by one sample. Defaults to 2.
        r: Nonnegative, finite absolute matching threshold, in the units of x. None uses
            0.15 * std(row, ddof=0), as in Costa et al. (2002), independently for each
            original signal. This threshold is fixed across all scales, never recalculated
            from the coarse-grained series. This differs from sample_entropy's default
            0.2 * std(row); set r=0.2*std(x) to match common library defaults.

    Returns:
        A float64 array of shape x.shape[:-1] + (number_of_scales,). The final axis holds
        the sample entropy at each requested scale; every batch row equals its 1D call
        bit for bit. A scale returns NaN when there are no matching order-length pairs
        (including a constant original signal with default r), or infinity when those
        pairs exist but none remain after extension. With explicit positive r, a constant
        signal has zero entropy at every scale.

    Raises:
        ValueError: If x is scalar, not finite and real, too short at a requested scale,
            or order, scales, or r is outside its allowed range. Non-finite input errors
            include the full index.

    Note:
        This implements Costa, Goldberger and Peng (2002), Physical Review Letters 89,
        068102, doi:10.1103/PhysRevLett.89.068102. At scale s, consecutive nonoverlapping
        blocks of s samples are averaged; an incomplete final block is discarded.
        Sample entropy reuses sample_entropy's exact Chebyshev counting kernel: self
        matches are excluded, distances must be strictly less than r, and both template
        lengths use the same number of extendible templates. Coarse-grained means use
        float64 arithmetic, without fastmath.

        The curve is returned rather than collapsing scales. For ordered factors s and
        the returned h, numpy.trapezoid(h, x=s, axis=-1) gives the area under the scale
        curve. For consecutive unit-spaced scales, h.sum(axis=-1) gives the commonly
        used complexity index (as in EntropyHub). These summaries are different; report
        which one is used. Neither summary silently drops NaN or infinity.

        Signals are processed serially in compiled code, reusing one coarse-graining
        buffer and, only when needed, one float64 row buffer. This keeps batch working
        memory independent of the number of signals apart from the returned curves
        and one tolerance per signal. The imported sample-entropy kernel has O(N)
        auxiliary memory for long low-order signals and O(1) for its diagonal scan.
    """
    values = as_signals(x)
    order = check_positive_int(order, "order")
    n = values.shape[-1]
    factors = _scales_array(scales, n, order)
    rows = int(numpy.prod(values.shape[:-1], dtype=numpy.intp))
    if r is None:
        tolerances = numpy.fromiter(
            (
                check_finite_real(0.0 if _is_constant(row) else float(0.15 * numpy.std(row, ddof=0)), "r", allow_zero=True)
                for row in float64_rows(values)
            ),
            numpy.float64,
        )
    else:
        tolerances = numpy.full(rows, check_finite_real(r, "r", allow_zero=True))
    if rows == 0:
        return numpy.empty(values.shape[:-1] + (len(factors),))
    result = _multiscale_rows(values, unit_stride_rows(values), factors, order, tolerances)
    return result.reshape(values.shape[:-1] + (len(factors),))
