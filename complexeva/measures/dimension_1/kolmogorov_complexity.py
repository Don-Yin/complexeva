import zlib
import numpy as np


def kolmogorov_complexity(data):
    """
    Calculate an approximation of Kolmogorov complexity using zlib compression.

    Args:
        data: numpy array or list of numerical values

    Returns:
        int: approximated Kolmogorov complexity (compressed length)
    """
    # Convert numerical data to bytes
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    elif isinstance(data, list):
        data = np.array(data).tobytes()
    else:
        raise ValueError("Input must be a numpy array or list")

    # Calculate compressed length as approximation of Kolmogorov complexity
    return len(zlib.compress(data))
