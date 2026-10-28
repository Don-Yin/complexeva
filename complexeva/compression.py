import zlib

import numpy as np
from numpy.typing import NDArray


def compression_complexity(data: NDArray[np.generic]) -> float:
    """Measure zlib-compressed size relative to the original array bytes.

    Args:
        data: One-dimensional NumPy array whose byte representation is nonempty. Its dtype is
            preserved; values, including NaN and infinity, are serialized with data.tobytes() in
            C order. No floating-point conversion, scaling, or finiteness check is performed.

    Returns:
        Compressed byte count divided by uncompressed byte count, using zlib's default compression
        level. Lower values indicate more compressible bytes. Small inputs can exceed 1 because
        the compressed stream includes overhead; the result depends on dtype as well as values.

    Raises:
        ValueError: If data is not one-dimensional or contains no bytes. The error names its shape
            and, for empty bytes, its dtype.
    """
    if data.ndim != 1:
        raise ValueError(f"data must be a 1D array, got shape {data.shape}")
    raw = data.tobytes()
    if not raw:
        raise ValueError(f"data must contain at least one byte, got shape {data.shape} and dtype {data.dtype}")
    return len(zlib.compress(raw)) / len(raw)
