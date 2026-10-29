import numpy
from numpy.typing import ArrayLike, NDArray
from scipy.fft import fft, ifft

from ._validation import as_finite_1d


def hilbert_envelope(x: ArrayLike, name: str = "x") -> NDArray[numpy.float64]:
    """Validate a nonempty signal and return abs(scipy.signal.hilbert(x)), identical in every bit.

    Importing scipy.signal costs about 0.4 s, four times numba's import, so its spectrum arithmetic is kept here.
    """
    values = as_finite_1d(x, name)
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one sample for the Hilbert transform, got len({name})=0")
    # asarray only narrows scipy.fft's uarray return type; it does not copy.
    spectrum = numpy.asarray(fft(values))
    spectrum[1 : (len(values) + 1) // 2] *= 2.0
    spectrum[len(values) // 2 + 1 :] = 0.0
    # The spectrum is private, so the inverse may overwrite it.
    return numpy.abs(numpy.asarray(ifft(spectrum, overwrite_x=True)))
