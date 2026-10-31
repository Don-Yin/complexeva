import zlib

import numpy
from numpy.typing import NDArray


def compression_complexity(x: NDArray[numpy.generic]) -> float:
    """Measure zlib-compressed size relative to the original array bytes.

    Args:
        x: One-dimensional NumPy array whose byte representation is nonempty. Its dtype is
            preserved; values, including NaN and infinity, are serialized with x.tobytes() in
            C order. No floating-point conversion, scaling, or finiteness check is performed.

    Returns:
        Compressed byte count divided by uncompressed byte count, using zlib's default compression
        level. Lower values indicate more compressible bytes. Small inputs can exceed 1 because
        the compressed stream includes overhead; the result depends on dtype as well as values.

    Raises:
        ValueError: If x is not one-dimensional or contains no bytes. The error names its shape
            and, for empty bytes, its dtype.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array, got shape {x.shape}")
    raw = x.tobytes()
    if not raw:
        raise ValueError(f"x must contain at least one byte, got shape {x.shape} and dtype {x.dtype}")
    return len(zlib.compress(raw)) / len(raw)
