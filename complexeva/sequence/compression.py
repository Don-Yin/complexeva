#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

import zlib
import numpy as np


def compression_complexity(data: np.ndarray) -> float:
    """
    calculate compression complexity using zlib compression -> float: compression complexity score
    """
    data = data.tobytes()
    compressed = zlib.compress(data)
    return len(compressed) / len(data)  # Normalized complexity score


if __name__ == "__main__":
    signal = np.random.rand(1000)  # shape: (1000,)
    print(compression_complexity(signal))
