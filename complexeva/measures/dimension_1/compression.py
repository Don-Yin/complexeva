import zlib
import numpy as np


def compression_complexity(data):
    """
    Calculate compression complexity using zlib compression.

    Args:
        data: numpy array or list of numerical values

    Returns:
        float: compression complexity score
    """
    # Convert numerical data to bytes
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    elif isinstance(data, list):
        data = np.array(data).tobytes()
    else:
        raise ValueError("Input must be a numpy array or list")

    # Calculate compression ratio
    compressed = zlib.compress(data)
    return len(compressed) / len(data)  # Normalized complexity score
