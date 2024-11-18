#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

import numpy


def lempel_ziv(time_series, method="median"):
    """
    - compared to other implementations, this implementation is makes the most sense

    Lempel-Ziv complexity as described in Kaspar and Schuster, Phys. Rev. A.
    Counts the number of distinct patterns that need to be copied to reproduce the sequence.
    Returns: int: The number of distinct patterns (complexity measure)
    """
    sequence = time_series_to_binary(time_series, method=method)

    history_pos = 0  # Position in history we're comparing against
    current_len = 1  # Length of current pattern we're checking
    max_pattern_len = 1  # Length of longest matching pattern found
    seq_pos = 1  # Current position in sequence
    complexity = 1  # Start with 1 as first character is always a new pattern
    seq_len = len(sequence)

    while seq_pos + current_len <= seq_len:
        # Check if current pattern matches what's in history
        if sequence[history_pos + current_len - 1] == sequence[seq_pos + current_len - 1]:
            current_len += 1
        else:
            # Update longest matching pattern length if needed
            if current_len > max_pattern_len:
                max_pattern_len = current_len

            history_pos += 1

            # If we've checked all history, we found a new pattern
            if history_pos == seq_pos:
                complexity += 1
                seq_pos += max_pattern_len

                # Reset for next pattern search
                history_pos = 0
                current_len = 1
                max_pattern_len = 1
            else:
                # Try next position in history
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

    if method == "mean":
        threshold = numpy.mean(vector)
        binary_seq = "".join(["1" if x > threshold else "0" for x in vector])
    elif method == "median":
        threshold = numpy.median(vector)
        binary_seq = "".join(["1" if x > threshold else "0" for x in vector])
    elif method == "diff":
        diffs = numpy.diff(
            vector
        )  # this measures whether the value is increasing or decreasing as compared to the previous value
        binary_seq = "".join(["1" if x > 0 else "0" for x in diffs])
    else:
        raise ValueError("Method must be 'mean', 'median', or 'diff'.")

    return binary_seq


def lempel_ziv_v1(time_series: numpy.ndarray) -> int:
    r"""Manual implementation of the Lempel-Ziv complexity.

    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:

    >>> s = '1001111011000010'
    >>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
    8

    Marking in the different substrings the sequence complexity :math:`\mathrm{Lempel-Ziv}(s) = 8`: :math:`s = 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010`.

    - See the page https://en.wikipedia.org/wiki/Lempel-Ziv_complexity for more details.


    Other examples:

    >>> lempel_ziv_complexity('1010101010101010')  # 1, 0, 10, 101, 01, 010, 1010
    7
    >>> lempel_ziv_complexity('1001111011000010000010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000
    9
    >>> lempel_ziv_complexity('100111101100001000001010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000, 0101
    10

    - Note: it is faster to give the sequence as a string of characters, like `'10001001'`, instead of a list or a numpy array.
    - Note: see this notebook for more details, comparison, benchmarks and experiments: https://Nbviewer.Jupyter.org/github/Naereen/Lempel-Ziv_Complexity/Short_study_of_the_Lempel-Ziv_complexity.ipynb
    - Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.
    """

    sequence = time_series_to_binary(time_series, method="median")

    sub_strings = set()

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)


def lempel_ziv_v2(time_series):
    """
    Compute the Lempel-Ziv complexity of a binary sequence.

    Parameters:
        sequence (str): The binary sequence as a string of '0' and '1'.

    Returns:
        int: The Lempel-Ziv complexity.
    """
    sequence = time_series_to_binary(time_series, method="median")

    sub_strings = set()
    n = len(sequence)
    ind = 0
    count = 0

    while ind < n:
        inc = 1
        # Find the shortest substring not already in sub_strings
        while sequence[ind : ind + inc] in sub_strings and ind + inc <= n:
            inc += 1

        # Add the new substring to the set
        sub_strings.add(sequence[ind : ind + inc])
        count += 1
        ind += inc

    return count


def lempel_ziv_v4(time_series):
    """
    Computes the Lempel-Ziv complexity of the input sequence s.

    Parameters:
    s (str): The input sequence (e.g., a binary string).

    Returns:
    int: The Lempel-Ziv complexity of the sequence.
    """

    sequence = time_series_to_binary(time_series, method="median")
    n = len(sequence)
    i = 0
    c = 1  # Complexity starts at 1 since the first character is always new
    k = 1
    l = 1

    while l + k <= n:
        # Check if the substring s[l:l+k] has occurred in s[0:l]
        if sequence[l : l + k] in sequence[0:l]:
            k += 1
            if l + k - 1 == n:
                c += 1
                break
        else:
            c += 1
            l = l + k
            k = 1

    return c


if __name__ == "__main__":
    sequence = "1001111011000010"
    print(lempel_ziv_v1(sequence))
    print(lempel_ziv_v2(sequence))
    print(lempel_ziv(sequence))
    print(lempel_ziv_v4(sequence))

    signal = numpy.random.rand(1000)
    binary_sequence = time_series_to_binary(signal, method="median")
    print(lempel_ziv_v1(binary_sequence))
    print(lempel_ziv_v2(binary_sequence))
    print(lempel_ziv(binary_sequence))
    print(lempel_ziv_v4(binary_sequence))
