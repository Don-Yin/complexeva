import numpy
from numba import get_num_threads, njit, prange
from numpy.typing import ArrayLike, NDArray

from ._hilbert import hilbert_envelope
from ._validation import as_signals, float64_rows

_DIRECT_COUNT_LIMIT = 512
_WIDE_PREFIX_LIMIT = 8192


@njit(cache=True, inline="always")
def _word_at(words, position):
    """Read 64 little-endian symbols at any bit offset, with zero padding."""
    word, shift = position >> 6, position & 63
    if shift == 0:
        return words[word]
    return (words[word] >> shift) | (words[word + 1] << (64 - shift))


@njit(cache=True, inline="always")
def _match_length(words, left, right, limit):
    """Compare 64 symbols at a time, including self-overlapping occurrences."""
    length = 0
    while length < limit:
        difference = _word_at(words, left + length) ^ _word_at(words, right + length)
        if difference:
            for shift in (32, 16, 8, 4, 2, 1):
                if difference & ((numpy.uint64(1) << shift) - numpy.uint64(1)) == 0:
                    length += shift
                    difference >>= shift
            return min(length, limit)
        length += 64
    return limit


@njit(cache=True)
def _count_packed(words, n):
    """Count exact LZ76 phrases with 2N-byte links and N/8 packed symbols.

    A compact occurrence index replaces the 32N-byte suffix automaton. Comparisons
    skip whole words and reject candidates at the current best match length, but
    occurrence-chain search has a quadratic worst case, unlike an automaton.
    """
    if n < 2:
        return 1
    if n >= 2**30:
        raise ValueError("binary sequence too long for int32 occurrence positions")
    count, start = 1, 1
    if n < _DIRECT_COUNT_LIMIT:
        while start < n:
            longest = 0
            for source in range(start):
                longest = max(longest, _match_length(words, source, start, n - start))
                if longest == n - start:
                    return count + 1
            start += longest + 1
            count += 1
        return count
    # Distances, not state ids, fit uint16. 65535 is an escape: scan from that
    # distance to the next occurrence, preserving arbitrarily long gaps.
    prefix_bits = 12 if n >= _WIDE_PREFIX_LIMIT else 8
    alphabet = 1 << prefix_bits
    mask = numpy.uint64(alphabet - 1)
    following = numpy.zeros(n, dtype=numpy.uint16)
    first = numpy.full(2 * alphabet, n, dtype=numpy.int32)
    for position in range(n - 1, -1, -1):
        key = int(_word_at(words, position) & mask)
        target = first[alphabet + key]
        if target < n:
            following[position] = min(target - position, 65535)
        first[alphabet + key] = position
    # The same table indexes every shorter prefix without scanning earlier symbols.
    for width in range(prefix_bits - 1, 0, -1):
        for key in range(1 << width):
            first[(1 << width) + key] = min(first[(2 << width) + key], first[(3 << width) + key])
    while start < n:
        remaining = n - start
        key = int(_word_at(words, start) & mask)
        longest = 0
        source = first[alphabet + key]
        if remaining >= prefix_bits and source < start:
            while source < start:
                offset = max(0, longest - 63)
                difference = _word_at(words, source + offset) ^ _word_at(words, start + offset)
                if longest < 63:
                    difference &= (numpy.uint64(1) << (longest + 1)) - numpy.uint64(1)
                if difference == 0:
                    longest = max(longest, _match_length(words, source, start, remaining))
                    if longest == remaining:
                        return count + 1
                distance = int(following[source])
                if distance == 0:
                    break
                source += distance
                if distance == 65535:
                    while source < start and int(_word_at(words, source) & mask) != key:
                        source += 1
        else:
            for width in range(min(prefix_bits - 1, remaining), 0, -1):
                if first[(1 << width) + (key & ((1 << width) - 1))] < start:
                    longest = width
                    break
            if longest == remaining:
                return count + 1
        start += longest + 1
        count += 1
    return count


@njit(cache=True)
def _count_components(binary_array: NDArray[numpy.bool_]) -> int:
    """Pack Boolean symbols, then count exact Kaspar-Schuster components, including an unfinished final copy."""
    n = len(binary_array)
    words = numpy.zeros((n + 63) // 64 + 1, dtype=numpy.uint64)
    for position in range(n):
        if binary_array[position]:
            words[position >> 6] |= numpy.uint64(1) << (position & 63)
    return _count_packed(words, n)


def _binarize(vector: NDArray[numpy.float64], method: str) -> NDArray[numpy.bool_]:
    """Convert numeric samples to binary symbols for Lempel-Ziv counting.

    Args:
        vector: One-dimensional, finite float64 samples.
        method: "median" or "mean" marks samples strictly above the threshold; "diff" marks
            strictly increasing adjacent samples. Median thresholding follows doi:10.1101/2021.09.23.461002.

    Returns:
        Boolean array of symbols. Its length is len(x), or max(len(x) - 1, 0)
        for "diff". An empty signal produces an empty array.

    Raises:
        ValueError: If method is unknown.
    """
    if vector.size == 0 and method in ("diff", "mean", "median"):
        return numpy.empty(0, dtype=numpy.bool_)
    if method == "diff":
        bits = numpy.diff(vector) > 0
    elif method == "mean":
        bits = vector > numpy.mean(vector)
    elif method == "median":
        bits = vector > numpy.median(vector)
    else:
        raise ValueError(f"method must be 'mean', 'median', or 'diff', got {method!r}")
    return bits


@njit(cache=True)
def _lz_chunk(values, thresholds, method, begin, end):
    """Read arbitrary layouts directly into packed bits; overwrite thresholds with counts."""
    n = values.shape[-1]
    counts = thresholds.view(numpy.int64)
    size = max(0, n - 1) if method == 2 else n
    for row in range(begin, end):
        words = numpy.zeros((size + 63) // 64 + 1, dtype=numpy.uint64)
        threshold = thresholds[row]
        for j in range(size):
            value = numpy.float64(values.flat[row * n + j])
            bit = (numpy.float64(values.flat[row * n + j + 1]) > value) if method == 2 else value > threshold
            if bit:
                words[j >> 6] |= numpy.uint64(1) << (j & 63)
        counts[row] = _count_packed(words, size)


@njit(cache=True, parallel=True)
def _lz_rows(values, thresholds, method, chunks):
    """Parallelize only when the one-row scratch budget admits several indices."""
    rows = len(thresholds)
    for chunk in prange(chunks):
        _lz_chunk(values, thresholds, method, rows * chunk // chunks, rows * (chunk + 1) // chunks)


@njit(cache=True)
def _row_median(values, offset, n):
    """Select the exact finite-row median in one scratch copy, without a partition copy."""
    row = numpy.empty(n, dtype=numpy.float64)
    for j in range(n):
        row[j] = values.flat[offset + j]
    low, high, middle = 0, n - 1, n // 2
    while low < high:
        pivot = row[(low + high) // 2]
        left, right = low, high
        while left <= right:
            while row[left] < pivot:
                left += 1
            while row[right] > pivot:
                right -= 1
            if left <= right:
                row[left], row[right] = row[right], row[left]
                left += 1
                right -= 1
        if middle <= right:
            high = right
        elif middle >= left:
            low = left
        else:
            break
    upper = row[middle]
    if n & 1:
        return upper
    lower = row[0]
    for j in range(1, middle):
        lower = max(lower, row[j])
    return (lower + upper) / 2


@njit(cache=True)
def _median_thresholds(values, out):
    """Reuse the exact selection workspace sequentially across rows."""
    n = values.shape[-1]
    for row in range(len(out)):
        out[row] = _row_median(values, row * n, n)


def _batch_lz(values: NDArray, method: str, normalize: bool) -> NDArray[numpy.int64] | NDArray[numpy.float64]:
    """Use one median scratch row and bound parallel indices by that same 8N budget."""
    if method not in ("median", "mean", "diff"):
        raise ValueError(f"method must be 'mean', 'median', or 'diff', got {method!r}")
    n = values.shape[-1]
    rows = int(numpy.prod(values.shape[:-1], dtype=numpy.intp))
    thresholds = numpy.empty(rows, dtype=numpy.float64)
    if method == "median" and n:
        _median_thresholds(values, thresholds)
    elif method == "mean" and n:
        # Keep NumPy's pairwise summation order, exactly as in the scalar path.
        for i, row in enumerate(float64_rows(values)):
            thresholds[i] = numpy.mean(row)
    else:
        thresholds.fill(0)
    # Links, packed symbols, prefix table, and small array headers fit within one
    # median's 8N-byte scratch budget; a one-chunk batch avoids the thread team.
    workspace = 2 * n + (n + 7) // 8 + (32768 if n >= _WIDE_PREFIX_LIMIT else 2048) + 256
    chunks = min(rows, get_num_threads(), max(1, 8 * n // workspace))
    code = ("median", "mean", "diff").index(method)
    if chunks > 1:
        _lz_rows(values, thresholds, code, chunks)
    else:
        _lz_chunk(values, thresholds, code, 0, rows)
    counts = thresholds.view(numpy.int64)
    if method == "diff":
        n = max(n - 1, 0)
    if normalize:
        counts = counts * numpy.log2(n) / n if n > 1 else numpy.zeros(rows)
    return counts.reshape(values.shape[:-1])


def lempel_ziv(
    x: ArrayLike | str, method: str = "median", normalize: bool = False
) -> int | float | NDArray[numpy.int64] | NDArray[numpy.float64]:
    """Compute Kaspar-Schuster Lempel-Ziv complexity along the last axis.

    Args:
        x: A binary string containing only "0" and "1", used without binarization, or
            finite, real-valued samples with time on the last axis, converted to float64 per row.
            A numeric batch has shape (..., time); NUMBA_NUM_THREADS limits parallel workers,
            additionally bounded by a one-row scratch budget. Small batches run serially.
            Strings and lempel_ziv_hilbert remain one-dimensional.
        method: Numeric binarization rule: "median" (default) or "mean" marks samples strictly above
            that row's threshold, and "diff" marks strictly increasing adjacent samples. Ignored
            for binary-string input, which is already encoded.
        normalize: False returns the integer count c. True returns c * log2(n) / n, where n is the
            binary sequence length (one fewer numeric sample for "diff"). Defaults to False.

    Returns:
        Raw integer component count, or a normalized float that is approximately 1 for a long
        random binary sequence and need not be bounded by 1. For n <= 1, normalized complexity
        is 0.0; the raw count of an empty or single-symbol sequence is 1. Returns a Python scalar
        for one signal or an int64 (raw) / float64 (normalized) array of shape x.shape[:-1]
        for a numeric batch, bitwise equal to separate 1D calls. A constant row follows the 1D rule.

    Raises:
        ValueError: If a string contains nonbinary characters, numeric input is scalar, not finite
            and real, or the numeric binarization method is unknown. Non-finite errors identify
            the full index within a batch.
    """
    if isinstance(x, str):
        if any(c not in "01" for c in x):
            raise ValueError(f"x must be a binary string containing only '0' and '1', got {x!r}")
        binary_array = numpy.array([c == "1" for c in x], dtype=numpy.bool_)
    else:
        values = as_signals(x, "x")
        if values.ndim > 1:
            return _batch_lz(values, method, normalize)
        binary_array = _binarize(values, method)
    complexity = int(_count_components(binary_array))
    if not normalize:
        return complexity
    n = len(binary_array)
    return float(complexity * numpy.log2(n) / n) if n > 1 else 0.0


def lempel_ziv_hilbert(x: ArrayLike, method: str = "median", normalize: bool = False) -> int | float:
    """Compute Lempel-Ziv complexity on the Hilbert envelope of a signal.

    Args:
        x: Nonempty, one-dimensional, finite, real-valued samples, converted to float64
            before taking the Hilbert transform. Binary strings are not accepted by this variant.
        method: Envelope binarization rule: "median" (default), "mean", or "diff" for rising steps.
            Threshold comparisons are strict, as in lempel_ziv.
        normalize: False returns the raw integer count. True returns c * log2(n) / n using the
            binarized envelope length n, or 0.0 for n <= 1. Defaults to False.

    Returns:
        Integer count or normalized float for abs(hilbert(x)), the instantaneous
        amplitude. Normalized values near 1 indicate random binary structure, not a strict bound.

    Raises:
        ValueError: If x is empty or not a finite, real, one-dimensional signal, or the
            binarization method is unknown.
    """
    result = lempel_ziv(hilbert_envelope(x, "x"), method=method, normalize=normalize)
    return float(result) if normalize else int(result)
