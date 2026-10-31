from math import log2

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.fftpack import rfft

from ._validation import as_signals, check_finite_real, check_positive_int

# Rescale only extreme amplitudes; ordinary samples retain their original precision when centred.
_MIN_SAFE_AMPLITUDE = 1e-100
_MAX_SAFE_AMPLITUDE = 1e100


@njit(cache=True)
def _hann(length):
    """Periodic Hann taper, including SciPy's length-one convention."""
    window = numpy.ones(length)
    if length > 1:
        for j in range(length):
            window[j] = 0.5 - 0.5 * numpy.cos(2.0 * numpy.pi * j / length)
    return window


@njit(cache=True)
def _prepare_spectrum(row, work, window, step):
    """Centre each segment into one reusable packed-real FFT buffer."""
    low = high = numpy.float64(row[0])
    for j in range(1, len(row)):
        value = numpy.float64(row[j])
        low = min(low, value)
        high = max(high, value)
    if low == high:
        return False
    amplitude = max(abs(low), abs(high))
    scale = amplitude if amplitude < _MIN_SAFE_AMPLITUDE or amplitude > _MAX_SAFE_AMPLITUDE else 1.0
    length = work.shape[1]
    for segment in range(work.shape[0]):
        start = segment * step
        # Subtract an anchor before averaging to retain small variations on large offsets.
        anchor = numpy.float64(row[start]) / scale
        total = 0.0
        for j in range(length):
            value = numpy.float64(row[start + j]) / scale - anchor
            work[segment, j] = value
            total += value
        mean = total / length
        for j in range(length):
            value = work[segment, j] - mean
            work[segment, j] = value if window is None else value * window[j]
    return True


@njit(cache=True)
def _spectrum_entropy(packed, divisor):
    """Accumulate one-sided power in the consumed part of the FFT buffer, then Shannon entropy."""
    segments, length = packed.shape
    bins = length // 2 + 1
    total = 0.0
    for k in range(bins):
        power = 0.0
        for segment in range(segments):
            if k == 0:
                real, imag = packed[segment, 0], 0.0
            elif length % 2 == 0 and k == bins - 1:
                real, imag = packed[segment, length - 1], 0.0
            else:
                real, imag = packed[segment, 2 * k - 1], packed[segment, 2 * k]
            power += real * real + imag * imag
        if k > 0 and not (length % 2 == 0 and k == bins - 1):
            power *= 2.0
        # Increasing k writes strictly behind the unread packed coefficients.
        packed[0, k] = power
        total += power
    if total == 0.0:
        return numpy.nan
    entropy = 0.0
    for k in range(bins):
        probability = packed[0, k] / total
        if probability > 0.0:
            entropy -= probability * numpy.log2(probability)
    return entropy / divisor


def _row_entropy(row, work, window, step, divisor):
    """Use the identical preparation, in-place FFT and reduction for scalars and batch rows."""
    if not _prepare_spectrum(row, work, window, step):
        return float("nan")
    spectrum = rfft(work, axis=-1, overwrite_x=True)
    return float(_spectrum_entropy(spectrum, divisor))


def spectral_entropy(
    x: ArrayLike,
    sf: float,
    method: str = "fft",
    nperseg: int | numpy.integer | None = None,
    normalize: bool = False,
) -> float | NDArray[numpy.float64]:
    """Shannon entropy of the normalized one-sided power spectrum, in bits.

    Args:
        x: Finite, real-valued samples, with time on the last axis. Each signal needs
            at least two samples. Computation uses float64 without modifying the input.
        sf: Positive, finite sampling frequency in Hz. Its common PSD scaling cancels
            during probability normalization, so it does not change the entropy.
        method: Spectral estimator: 'fft' for a boxcar periodogram of the whole signal,
            or 'welch' for the mean of periodic-Hann periodograms with 50% overlap.
            Each segment is detrended by subtracting its mean. Defaults to 'fft'.
        nperseg: Positive integer Welch segment length. None uses min(256, len(x)); a
            value above the signal length is clipped to that length. Ignored by 'fft'.
        normalize: Divide the entropy by log2(number of one-sided frequency bins),
            giving a value between 0 and 1. Defaults to False.

    Returns:
        Entropy -sum(p * log2(p)), with p obtained by normalizing the estimated PSD.
        Zero-power bins contribute zero. Constant signals, or Welch segments whose
        detrended power is all zero, return NaN because their probabilities are undefined.
        One signal returns a float; batches return a float64 array of shape x.shape[:-1].
        Every batch entry equals the corresponding 1D call bit for bit.

    Raises:
        ValueError: If x has no time axis, is not real and finite, or has fewer than two
            samples; if sf is not positive and finite; if method is not 'fft' or 'welch';
            or if a Welch nperseg is not a positive integer. Non-finite input errors name
            the full sample index.

    Note:
        Spectral entropy follows Inouye et al. (1991), Electroencephalography and
        Clinical Neurophysiology 79, 204–210, doi:10.1016/0013-4694(91)90138-T.
        The estimator, base-2 logarithm, endpoint handling and optional normalization
        match antropy.spectral_entropy: include DC and the Nyquist bin, and double
        only the non-endpoint one-sided powers. Unlike a full-batch spectrogram, the
        implementation reuses one row's FFT workspace for every batch row.
    """
    values = as_signals(x)
    check_finite_real(sf, "sf", allow_zero=False)
    if method not in ("fft", "welch"):
        raise ValueError(f"method must be 'fft' or 'welch', got {method!r}")
    n = values.shape[-1]
    if n < 2:
        raise ValueError(f"x needs at least 2 samples on the time axis, got {n}")
    if method == "welch":
        length = min(n, 256 if nperseg is None else check_positive_int(nperseg, "nperseg"))
        step = length - length // 2
        segments = 1 + (n - length) // step
        window = _hann(length)
    else:
        length, step, segments, window = n, n, 1, None
    divisor = log2(length // 2 + 1) if normalize else 1.0
    work = numpy.empty((segments, length))
    if values.ndim == 1:
        return _row_entropy(values, work, window, step, divisor)
    result = numpy.empty(values.shape[:-1])
    for index in numpy.ndindex(values.shape[:-1]):
        result[index] = _row_entropy(values[index], work, window, step, divisor)
    return result
