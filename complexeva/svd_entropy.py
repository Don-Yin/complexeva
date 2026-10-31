from math import log2

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_signals, check_positive_int, unit_stride_rows

# Forming X.T @ X squares the condition number. Below this eigenvalue ratio,
# accumulate a small QR factor instead, retaining singular values near zero.
_MIN_GRAM_EIGENVALUE_RATIO = 1e-4
# Products remain normal and finite at every allowed embedding dimension.
_MIN_PRODUCT_SCALE = 1e-100
_MAX_PRODUCT_SCALE = 1e100


@njit(cache=True)
def _singular_values_qr(values, rows, columns, row_stride, column_stride, scale):
    """Reduce delay rows with Givens rotations, retaining only a small square factor."""
    factor = numpy.zeros((columns, columns))
    row = numpy.empty(columns)
    for start in range(rows):
        for j in range(columns):
            row[j] = values[start * row_stride + j * column_stride] / scale
        for j in range(columns):
            length = numpy.hypot(factor[j, j], row[j])
            if length == 0.0:
                continue
            cosine = factor[j, j] / length
            sine = row[j] / length
            factor[j, j] = length
            for k in range(j + 1, columns):
                previous = factor[j, k]
                factor[j, k] = cosine * previous + sine * row[k]
                row[k] = -sine * previous + cosine * row[k]
    return numpy.linalg.svd(factor, full_matrices=False)[1]


@njit(cache=True)
def _svd_entropy_row(values, order, delay, divisor):
    """Compute one row without allocating its delay embedding."""
    count = len(values) - (order - 1) * delay
    maximum = 0.0
    constant = True
    first = values[0]
    # With fewer rows than delay, gaps in x never enter the embedding. Such
    # samples must not determine the scale or cause used values to underflow.
    blocks = order if count < delay else 1
    block_length = count if count < delay else len(values)
    for block in range(blocks):
        for start in range(block_length):
            value = values[block * delay + start]
            maximum = max(maximum, abs(value))
            constant = constant and value == first
    if maximum == 0.0:
        return numpy.nan
    if constant:
        return 0.0
    scale = maximum if maximum < _MIN_PRODUCT_SCALE or maximum > _MAX_PRODUCT_SCALE else 1.0
    columns = min(count, order)
    rows = max(count, order)
    row_stride, column_stride = (1, delay) if count >= order else (delay, 1)
    gram = numpy.empty((columns, columns))
    # Transpose short, wide embeddings so both Gram and QR storage stay small.
    # Equal column separations share an overlapping dot product when their
    # strides align. Endpoint corrections are exact, not a Toeplitz assumption.
    for lag in range(columns):
        total = 0.0
        correction = 0.0
        for start in range(rows):
            offset = start * row_stride
            product = (values[offset] / scale) * (values[offset + lag * column_stride] / scale)
            term = product - correction
            updated = total + term
            correction = (updated - total) - term
            total = updated
        gram[0, lag] = total
        gram[lag, 0] = total
        for left in range(1, columns - lag):
            if column_stride % row_stride == 0:
                for start in range((left - 1) * column_stride, left * column_stride, row_stride):
                    total -= (values[start] / scale) * (values[start + lag * column_stride] / scale)
                    end = start + rows * row_stride
                    total += (values[end] / scale) * (values[end + lag * column_stride] / scale)
            else:
                total = 0.0
                correction = 0.0
                for start in range(rows):
                    offset = start * row_stride + left * column_stride
                    product = (values[offset] / scale) * (values[offset + lag * column_stride] / scale)
                    term = product - correction
                    updated = total + term
                    correction = (updated - total) - term
                    total = updated
            gram[left, left + lag] = total
            gram[left + lag, left] = total
    eigenvalues = numpy.linalg.eigvalsh(gram)
    largest = eigenvalues[-1]
    if largest == 0.0:
        return numpy.nan
    if eigenvalues[0] < _MIN_GRAM_EIGENVALUE_RATIO * largest:
        singular = _singular_values_qr(values, rows, columns, row_stride, column_stride, scale)
    else:
        singular = numpy.sqrt(eigenvalues)
    total = singular.sum()
    if total == 0.0:
        return numpy.nan
    entropy = 0.0
    for value in singular:
        if value > 0.0:
            probability = value / total
            entropy -= probability * numpy.log2(probability)
    return entropy / divisor


@njit(cache=True)
def _svd_entropy_rows(values, direct, order, delay, divisor):
    """Share the scalar kernel and one conversion buffer across all batch rows."""
    n = values.shape[-1]
    count = values.size // n
    out = numpy.empty(count)
    buffer = numpy.empty(n) if direct is None else None
    for i in range(count):
        if direct is None:
            assert buffer is not None
            for j in range(n):
                buffer[j] = values.flat[i * n + j]
            row = buffer
        else:
            row = direct[i]
        out[i] = _svd_entropy_row(row, order, delay, divisor)
    return out


def svd_entropy(
    x: ArrayLike, order: int | numpy.integer = 3, delay: int | numpy.integer = 1, normalize: bool = False
) -> float | NDArray[numpy.float64]:
    """Singular-value entropy of delay-embedded signals along the last axis.

    Args:
        x: Finite, real-valued samples, converted to float64, with time on the last axis.
            Each signal requires at least (order - 1) * delay + 1 samples.
        order: Embedding dimension, an integer at least 2. Defaults to 3.
        delay: Positive integer sample spacing within each embedding vector. Starting
            points advance by one sample, irrespective of delay. Defaults to 1.
        normalize: Divide the entropy in bits by log2(order), yielding a value between
            0 and 1. Defaults to False.

    Returns:
        Shannon entropy -sum(p * log2(p)), where p contains the singular values of
        the uncentered delay embedding divided by their sum. An all-zero embedding
        returns NaN; a nonzero constant signal returns 0. A float for one signal;
        for a batch, a float64 array of shape x.shape[:-1], with each entry equal
        bit for bit to a call on that signal alone.

    Raises:
        ValueError: If x is scalar, not finite and real, too short on the time axis,
            or order/delay is not an integer in its allowed range. Invalid samples
            include their full index in the error.

    Note:
        Implements the temporal embedding-space entropy in Roberts, Penny and Rezek
        (1999), Medical and Biological Engineering and Computing 37:93-98, section 3.3,
        DOI: 10.1007/BF02513272. Samples are not demeaned, and singular values, not
        squared singular values, define the probabilities. The lagged Gram matrix
        avoids copying the embedding; ill-conditioned embeddings use a streaming QR
        factor to preserve small singular values. Working storage is O(k**2), with
        k = min(order, N - (order - 1) * delay), plus a single length-N conversion
        buffer for batch rows that need one.
    """
    values = as_signals(x)
    order = check_positive_int(order, "order", minimum=2)
    delay = check_positive_int(delay, "delay")
    window = (order - 1) * delay + 1
    n = values.shape[-1]
    if n < window:
        raise ValueError(f"x needs at least (order - 1) * delay + 1 = {window} samples, got {n}")
    divisor = log2(order) if normalize else 1.0
    if values.ndim == 1:
        return float(_svd_entropy_row(values, order, delay, divisor))
    out = _svd_entropy_rows(values, unit_stride_rows(values), order, delay, divisor)
    return out.reshape(values.shape[:-1])
