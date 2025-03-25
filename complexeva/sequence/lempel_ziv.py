#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

import numpy


def lempel_ziv(time_series, method="median", **kwargs):
    """
    - compared to other implementations, this implementation is makes the most sense

    Lempel-Ziv complexity as described in Kaspar and Schuster, Phys. Rev. A.
    Counts the number of distinct patterns that need to be copied to reproduce the sequence.
    Returns: int: The number of distinct patterns (complexity measure)
    """
    sequence = time_series_to_binary(time_series, method=method)

    history_pos = 0  # position in history we're comparing against
    current_len = 1  # length of current pattern we're checking
    max_pattern_len = 1  # length of longest matching pattern found
    seq_pos = 1  # current position in sequence
    complexity = 1  # start with 1 as first character is always a new pattern
    seq_len = len(sequence)

    while seq_pos + current_len <= seq_len:
        if sequence[history_pos + current_len - 1] == sequence[seq_pos + current_len - 1]:
            current_len += 1
        else:
            if current_len > max_pattern_len:
                max_pattern_len = current_len
            history_pos += 1
            if history_pos == seq_pos:
                complexity += 1
                seq_pos += max_pattern_len
                history_pos = 0
                current_len = 1
                max_pattern_len = 1
            else:
                current_len = 1
    return complexity


def time_series_to_binary(vector, method="median"):
    """
    - convert a time series to a binary sequence based on a threshold.
    - returns: str: A binary sequence as a string of '0's and '1's.
    - https://doi.org/10.1101/2021.09.23.461002 this paper used median, making median the default method.
    - diff also makes somewhat sense
    """

    vector = numpy.array(vector)

    match method:

        case "mean":
            threshold = numpy.mean(vector)
            binary_seq = "".join(["1" if x > threshold else "0" for x in vector])

        case "median":
            threshold = numpy.median(vector)
            binary_seq = "".join(["1" if x > threshold else "0" for x in vector])

        case "diff":  # this measures whether the value is increasing or decreasing as compared to the previous value
            diffs = numpy.diff(vector)
            binary_seq = "".join(["1" if x > 0 else "0" for x in diffs])

        case _:
            raise ValueError("Method must be 'mean', 'median', or 'diff'.")

    return binary_seq


if __name__ == "__main__":
    sequence = "1001111011000010"
    print(lempel_ziv(sequence))
    signal = numpy.random.rand(1000)
    binary_sequence = time_series_to_binary(signal, method="median")
    print(lempel_ziv(binary_sequence))
