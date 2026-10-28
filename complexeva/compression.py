import zlib

import numpy as np


def compression_complexity(data: np.ndarray) -> float:
    """zlib-compressed size divided by raw size of the array bytes."""
    raw = data.tobytes()
    return len(zlib.compress(raw)) / len(raw)
