import numpy
from numba import get_num_threads, njit, prange
from numpy.typing import ArrayLike, NDArray

from ._sampen_tree import _TREE_BATCH_BYTES, _TREE_MAX_ORDER, _TREE_MIN_TEMPLATES, _tree_match_counts, _tree_workspace_bytes
from ._validation import (
    as_array,
    as_signals,
    check_finite_real,
    check_positive_int,
    default_radius,
    float64_rows,
    unit_stride_rows,
)
from .sample_entropy import _match_counts


@njit(cache=True)
def _offset_counts(x, order, delay, tolerance, first, stop):
    """Count delayed templates at the requested start-index separations, without arrays."""
    n = len(x) - order * delay
    count_m = count_m1 = 0
    for offset in range(first, stop):
        for i in range(n - offset):
            matched = True
            for k in range(order):
                if not abs(x[i + k * delay] - x[i + offset + k * delay]) < tolerance:
                    matched = False
                    break
            if matched:
                count_m += 1
                count_m1 += abs(x[i + order * delay] - x[i + offset + order * delay]) < tolerance
    return count_m, count_m1


@njit(cache=True)
def _modified_counts(x, order, scale, tolerance):
    """Use the SampEn tree, removing pairs closer than one scale as in Wu's Appendix."""
    n = len(x) - order * scale
    if n < _TREE_MIN_TEMPLATES or order > _TREE_MAX_ORDER:
        return _offset_counts(x, order, scale, tolerance, scale, n)
    count_m, count_m1 = _tree_match_counts(x, order, n, tolerance, scale)
    local_m, local_m1 = _offset_counts(x, order, scale, tolerance, 1, min(scale, n))
    return count_m - local_m, count_m1 - local_m1


@njit(cache=True)
def _multiscale_chunk(values, direct, scales, order, tolerances, modified, first, stop, out):
    """Reuse one coarse-graining buffer for a serial call or one parallel row chunk."""
    n = values.shape[-1]
    rows = values.size // n
    if out is None:
        out = numpy.empty((rows, len(scales)))
    buffer = numpy.empty(n) if direct is None else None
    minimum_coarse_scale = n + 1
    for scale in scales:
        if scale > 1:
            minimum_coarse_scale = min(minimum_coarse_scale, scale)
    coarse = numpy.empty(max(0, n - minimum_coarse_scale + 1) if modified else n // minimum_coarse_scale)
    for i in range(first, stop):
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
                length = n - scale + 1 if modified else n // scale
                for k in range(length):
                    total = 0.0
                    start = k if modified else k * scale
                    for offset in range(scale):
                        total += row[start + offset]
                    coarse[k] = total / scale
                current = coarse[:length]
            if modified and scale > 1:
                count_m, count_m1 = _modified_counts(current, order, scale, tolerances[i])
            else:
                count_m, count_m1 = _match_counts(current, order, tolerances[i])
            out[i, j] = numpy.nan if count_m == 0 else numpy.inf if count_m1 == 0 else -numpy.log(count_m1 / count_m)
    return out


@njit(cache=True, parallel=True)
def _multiscale_rows(values, direct, scales, order, tolerances, modified, chunks):
    """Each active chunk owns one coarse buffer; the serial kernel allocates it once."""
    rows = len(tolerances)
    out = numpy.empty((rows, len(scales)))
    for chunk in prange(chunks):
        _multiscale_chunk(
            values,
            direct,
            scales,
            order,
            tolerances,
            modified,
            rows * chunk // chunks,
            rows * (chunk + 1) // chunks,
            out,
        )
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
    *,
    method: str = "coarse",
) -> NDArray[numpy.float64]:
    """Conventional (Costa) or modified (Wu) multiscale sample entropy, in nats.

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
        method: "coarse" (default) averages nonoverlapping blocks as in Costa (2002).
            "modified" uses every moving-average window, embedding delay equal to scale,
            and excludes template pairs whose start indices differ by less than scale,
            following Wu et al. (2013), Section 2.3, Eq. (15) and the Appendix.

    Returns:
        A float64 array of shape x.shape[:-1] + (number_of_scales,). The final axis holds
        the sample entropy at each requested scale; every batch row equals its 1D call
        bit for bit. A scale returns NaN when there are no matching order-length pairs
        (including a constant original signal with default r), or infinity when those
        pairs exist but none remain after extension. With explicit positive r, a constant
        signal has zero entropy at every scale.

    Raises:
        ValueError: If x is scalar, not finite and real, too short at a requested scale,
            or order, scales, r, or method is outside its allowed range. Non-finite input errors
            include the full index.

    Note:
        This implements Costa, Goldberger and Peng (2002), Physical Review Letters 89,
        068102, doi:10.1103/PhysRevLett.89.068102. At scale s, consecutive nonoverlapping
        blocks of s samples are averaged; an incomplete final block is discarded.
        Sample entropy reuses sample_entropy's exact Chebyshev counting kernel: self
        matches are excluded, distances must be strictly less than r, and both template
        lengths use the same number of extendible templates. Coarse-grained means use
        float64 arithmetic, without fastmath.

        For method="modified", Wu, Wu, Lee and Lin (2013), Physica A 392:5865-5873,
        doi:10.1016/j.physa.2013.07.075, replace block means with
        z[j] = mean(x[j:j+s]), j=0,...,N-s. The length is N-s+1, and templates
        use delay s with N-(order+1)*s+1 common starting positions. Only pairs with
        j-i >= s are counted. This follows the paper's equations and Appendix loop;
        its prose j>i+s conflicts by one. The strict <r convention follows the
        Appendix and this package's SampEn, rather than the prose's <=r. The
        Appendix's movingaverage code leaves s-1 original tail values; they are
        discarded here as required by Section 2.3's length formula. At scale 1,
        both methods return identical values.

        The curve is returned rather than collapsing scales. For ordered factors s and
        the returned h, numpy.trapezoid(h, x=s, axis=-1) gives the area under the scale
        curve. For consecutive unit-spaced scales, h.sum(axis=-1) gives the commonly
        used complexity index (as in EntropyHub). These summaries are different; report
        which one is used. Neither summary silently drops NaN or infinity.

        Batch rows run in parallel, reusing one coarse-graining buffer and, only when
        needed, one float64 row buffer per active worker. Concurrent tree and row
        workspaces share sample_entropy's 12 MiB array budget (at least one worker).
        Single rows avoid the parallel scheduler. The imported sample-entropy kernel
        has O(N) auxiliary memory for long low-order signals and O(1) for its diagonal scan.
    """
    values = as_signals(x)
    order = check_positive_int(order, "order")
    if not isinstance(method, str) or method not in ("coarse", "modified"):
        raise ValueError(f"method must be 'coarse' or 'modified', got {method!r}")
    n = values.shape[-1]
    factors = _scales_array(scales, n, order)
    rows = int(numpy.prod(values.shape[:-1], dtype=numpy.intp))
    if r is None:
        tolerances = numpy.fromiter(
            (check_finite_real(default_radius(row, 0.15), "r", allow_zero=True) for row in float64_rows(values)), numpy.float64
        )
    else:
        tolerances = numpy.full(rows, check_finite_real(r, "r", allow_zero=True))
    if rows == 0:
        return numpy.empty(values.shape[:-1] + (len(factors),))
    direct = unit_stride_rows(values)
    modified = method == "modified"
    chunks = min(rows, get_num_threads())
    if chunks > 1:
        minimum_scale = int(numpy.min(factors))
        minimum_coarse = min((int(scale) for scale in factors if scale > 1), default=n + 1)
        coarse_n = max(0, n - minimum_coarse + 1) if modified else n // minimum_coarse
        workspace = 8 * (coarse_n + (n if direct is None else 0))
        length = n - minimum_scale + 1 if modified else n // minimum_scale
        templates = length - order * (minimum_scale if modified else 1)
        if templates >= _TREE_MIN_TEMPLATES and order <= _TREE_MAX_ORDER:
            workspace += _tree_workspace_bytes(templates, order + 1)
        if workspace:
            chunks = min(chunks, max(1, _TREE_BATCH_BYTES // workspace))
    if chunks > 1:
        result = _multiscale_rows(values, direct, factors, order, tolerances, modified, chunks)
    else:
        result = _multiscale_chunk(values, direct, factors, order, tolerances, modified, 0, rows, None)
    return result.reshape(values.shape[:-1] + (len(factors),))
