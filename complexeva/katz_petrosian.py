"""Linear-time Katz and Petrosian fractal dimensions without signal-sized work arrays."""

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_signals


@njit(cache=True, error_model="numpy")
def _katz_core(values, start, n):
    """Accumulate planar path length and first-point diameter in one ordered pass."""
    first = numpy.float64(values.flat[start])
    previous = first
    # Four independent sums remove the scalar dependency bottleneck without fastmath or work arrays.
    length0 = length1 = length2 = length3 = 0.0
    diameter0 = diameter1 = diameter2 = diameter3 = 0.0
    stop = 1 + 4 * ((n - 1) // 4)
    for j in range(1, stop, 4):
        x0 = numpy.float64(values.flat[start + j])
        x1 = numpy.float64(values.flat[start + j + 1])
        x2 = numpy.float64(values.flat[start + j + 2])
        x3 = numpy.float64(values.flat[start + j + 3])
        difference = x0 - previous
        length0 += numpy.sqrt(1.0 + difference * difference)
        difference = x1 - x0
        length1 += numpy.sqrt(1.0 + difference * difference)
        difference = x2 - x1
        length2 += numpy.sqrt(1.0 + difference * difference)
        difference = x3 - x2
        length3 += numpy.sqrt(1.0 + difference * difference)
        displacement = x0 - first
        time = numpy.float64(j)
        diameter0 = max(diameter0, time * time + displacement * displacement)
        displacement = x1 - first
        time = numpy.float64(j + 1)
        diameter1 = max(diameter1, time * time + displacement * displacement)
        displacement = x2 - first
        time = numpy.float64(j + 2)
        diameter2 = max(diameter2, time * time + displacement * displacement)
        displacement = x3 - first
        time = numpy.float64(j + 3)
        diameter3 = max(diameter3, time * time + displacement * displacement)
        previous = x3
    length = ((length0 + length1) + length2) + length3
    diameter_squared = max(max(diameter0, diameter1), max(diameter2, diameter3))
    for j in range(stop, n):
        value = numpy.float64(values.flat[start + j])
        difference = value - previous
        length += numpy.sqrt(1.0 + difference * difference)
        displacement = value - first
        time = numpy.float64(j)
        diameter_squared = max(diameter_squared, time * time + displacement * displacement)
        previous = value
    log_steps = numpy.log10(n - 1)
    return log_steps / (log_steps + numpy.log10(numpy.sqrt(diameter_squared) / length))


@njit(cache=True)
def _petrosian_core(values, start, n):
    """Count derivative sign-bit changes without materializing the derivative or symbols."""
    changes = 0
    # Adjacent triples remove a loop-carried sign dependency, enabling an exact integer SIMD reduction.
    for j in range(2, n):
        left = numpy.float64(values.flat[start + j - 2])
        middle = numpy.float64(values.flat[start + j - 1])
        right = numpy.float64(values.flat[start + j])
        changes += numpy.signbit(middle - left) != numpy.signbit(right - middle)
    log_n = numpy.log10(n)
    return log_n / (log_n + numpy.log10(n / (n + 0.4 * changes)))


def _make_fd_rows(core):
    """Specialize a serial batch loop that reads every supported dtype and layout in place."""

    @njit(cache=True)
    def rows(values):
        n = values.shape[-1]
        count = values.size // n
        out = numpy.empty(count)
        for i in range(count):
            out[i] = core(values, i * n, n)
        return out

    return rows


_KATZ_ROWS = _make_fd_rows(_katz_core)
_PETROSIAN_ROWS = _make_fd_rows(_petrosian_core)


def _evaluate_fd(x, core, rows):
    """Validate once and use the same arithmetic for a single signal and each batch row."""
    values = as_signals(x)
    n = values.shape[-1]
    if n < 2:
        raise ValueError(f"x needs at least 2 samples on the time axis, got {n}")
    if values.ndim == 1:
        return float(core(values, 0, n))
    return rows(values).reshape(values.shape[:-1])


def katz_fd(x: ArrayLike) -> float | NDArray[numpy.float64]:
    """Katz fractal dimension of one signal, or of every signal along the last axis.

    Args:
        x: Finite, real-valued samples with time on the last axis and at least 2 samples
            per signal. Arithmetic uses float64; batch rows are read without work arrays.

    Returns:
        log10(N - 1) / (log10(N - 1) + log10(d / L)), where L is the sum of Euclidean
        distances between successive points (i, x[i]), and d is the largest Euclidean
        distance from (0, x[0]). A float for one signal; for a batch, a float64 array of
        shape x.shape[:-1] whose entries equal katz_fd on each signal bit for bit.
        A constant signal is a horizontal line and returns 1. A signal of length 2
        returns NaN because there is only one step and both logarithms vanish.

    Raises:
        ValueError: If x is scalar, not finite and real, or has fewer than 2 samples on
            its time axis. Invalid samples include their full index in the error.

    Note:
        Implements Katz (1988), Computers in Biology and Medicine 18(3), 145-156,
        doi:10.1016/0010-4825(88)90041-8, Methods equations (2)-(5) and the Appendix
        program, on unit-spaced time coordinates. Both time and amplitude contribute
        to the planar Euclidean distance, so amplitude scaling can change the result.
        antropy.katz_fd, neurokit2.fractal_katz, and mne_features.compute_katz_fd instead
        use amplitude-only distances and therefore compute a different quantity.
        The method uses one O(N) pass after input validation and O(1) workspace per
        signal, with no fastmath.
    """
    return _evaluate_fd(x, _katz_core, _KATZ_ROWS)


def petrosian_fd(x: ArrayLike) -> float | NDArray[numpy.float64]:
    """Petrosian fractal dimension of one signal, or of every signal along the last axis.

    Args:
        x: Finite, real-valued samples with time on the last axis and at least 2 samples
            per signal. Arithmetic uses float64; batch rows are read without work arrays.

    Returns:
        log10(N) / (log10(N) + log10(N / (N + 0.4 * N_delta))), where N is the number
        of input samples and N_delta counts sign-bit changes in successive differences.
        A float for one signal; for a batch, a float64 array of shape x.shape[:-1] whose
        entries equal petrosian_fd on each signal bit for bit. A constant or monotonic
        signal has no sign changes and returns 1, not NaN, because the formula is defined.

    Raises:
        ValueError: If x is scalar, not finite and real, or has fewer than 2 samples on
            its time axis. Invalid samples include their full index in the error.

    Note:
        Uses Petrosian's (1995) derivative-sign symbolization (method C), Proceedings
        Eighth IEEE Symposium on Computer-Based Medical Systems, 212-217,
        doi:10.1109/CBMS.1995.465426. As in antropy.petrosian_fd, +0 differences belong
        to the nonnegative class; a negative signed zero retains its sign bit.
        N is the original signal length. NeuroKit2's default method C instead uses
        N - 1, the derivative-symbol length, and therefore differs for finite N.
        The method uses one O(N) pass after input validation and O(1) workspace per
        signal, with no derivative array, symbol array, or fastmath.
    """
    return _evaluate_fd(x, _petrosian_core, _PETROSIAN_ROWS)
