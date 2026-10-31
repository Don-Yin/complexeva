import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_signals, unit_stride_rows

_MOMENT_BLOCK = 64  # Anchor short blocks to bound moment cancellation without per-sample divisions.


@njit(cache=True, inline="always")
def _merge_moments(count, mean, square, block_count, anchor, block_sum, block_square):
    """Merge an anchored block's moments with the accumulated centered sum of squares."""
    fraction = block_count / (count + block_count)
    delta = anchor + block_sum / block_count - mean
    square += block_square - block_sum * block_sum / block_count + delta * delta * count * fraction
    return count + block_count, mean + delta * fraction, square


@njit(cache=True)
def _hjorth_row(values):
    """Accumulate all three population variances in one pass without derivative arrays."""
    n = len(values)
    origin = values[0]
    difference_origin = values[1] - origin
    second_origin = (values[2] - values[1]) - difference_origin
    previous = origin
    previous_difference = 0.0
    count = count_difference = count_second = 0
    mean = mean_difference = mean_second = 0.0
    square = square_difference = square_second = 0.0
    for start in range(0, n, _MOMENT_BLOCK):
        stop = min(start + _MOMENT_BLOCK, n)
        anchor = values[start]
        difference_anchor = difference_origin if start == 0 else values[start] - values[start - 1]
        second_anchor = second_origin if start == 0 else difference_anchor - (values[start - 1] - values[start - 2])
        block_sum = block_difference = block_second = 0.0
        block_square = block_square_difference = block_square_second = 0.0
        for i in range(start, stop):
            value = values[i]
            centered = value - anchor
            block_sum += centered
            block_square += centered * centered
            if i > 0:
                difference = value - previous
                centered_difference = difference - difference_anchor
                block_difference += centered_difference
                block_square_difference += centered_difference * centered_difference
                if i > 1:
                    centered_second = difference - previous_difference - second_anchor
                    block_second += centered_second
                    block_square_second += centered_second * centered_second
                previous_difference = difference
            previous = value
        count, mean, square = _merge_moments(count, mean, square, stop - start, anchor - origin, block_sum, block_square)
        count_difference, mean_difference, square_difference = _merge_moments(
            count_difference,
            mean_difference,
            square_difference,
            stop - max(start, 1),
            difference_anchor - difference_origin,
            block_difference,
            block_square_difference,
        )
        count_second, mean_second, square_second = _merge_moments(
            count_second,
            mean_second,
            square_second,
            stop - max(start, 2),
            second_anchor - second_origin,
            block_second,
            block_square_second,
        )
    if square == 0.0:
        return numpy.nan, numpy.nan
    mobility = numpy.sqrt((square_difference / (n - 1)) / (square / n))
    if square_difference == 0.0:
        return mobility, numpy.nan
    complexity = numpy.sqrt((square_second / (n - 2)) / (square_difference / (n - 1))) / mobility
    return mobility, complexity


@njit(cache=True)
def _hjorth_rows(values, direct):
    """Evaluate rows serially, reusing one float64 conversion buffer only when needed."""
    n = values.shape[-1]
    count = values.size // n
    mobility = numpy.empty(count)
    complexity = numpy.empty(count)
    buffer = numpy.empty(n) if direct is None else None
    for i in range(count):
        if direct is None:
            assert buffer is not None
            for j in range(n):
                buffer[j] = values.flat[i * n + j]
            row = buffer
        else:
            row = direct[i]
        mobility[i], complexity[i] = _hjorth_row(row)
    return mobility, complexity


def hjorth(x: ArrayLike) -> tuple[float, float] | tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Hjorth mobility and complexity of signals along the last axis.

    Args:
        x: Finite, real-valued samples, converted to float64, with time on the last axis.
            Each signal requires at least three samples.

    Returns:
        A tuple (mobility, complexity). Each element is a float for one signal, or a
        float64 array of shape x.shape[:-1] for a batch. Batch entries equal the
        corresponding one-dimensional call bit for bit. Mobility is
        sqrt(var(diff(x)) / var(x)), in inverse samples. Complexity is the mobility
        of diff(x) divided by the mobility of x. Variances use ddof=0. Constant
        signals return (NaN, NaN); a nonconstant signal with constant first
        differences has zero mobility and NaN complexity.

    Raises:
        ValueError: If x is scalar, not finite and real, or has fewer than three
            samples on its last axis. Non-finite samples include their full index
            in the error.

    Note:
        Implements Hjorth (1970), EEG analysis based on time domain properties,
        Electroencephalography and Clinical Neurophysiology 29, 306–310,
        doi:10.1016/0013-4694(70)90143-4. Derivatives are successive unpadded
        first differences, as in antropy.hjorth_params and
        neurokit2.complexity_hjorth. One compiled pass computes all three
        variances with constant workspace, without materializing derivatives.
    """
    values = as_signals(x)
    n = values.shape[-1]
    if n < 3:
        raise ValueError(f"x needs at least 3 samples, got {n}")
    if values.ndim == 1:
        mobility, complexity = _hjorth_row(values)
        return float(mobility), float(complexity)
    mobility, complexity = _hjorth_rows(values, unit_stride_rows(values))
    return mobility.reshape(values.shape[:-1]), complexity.reshape(values.shape[:-1])
