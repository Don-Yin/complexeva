"""Exact unordered Chebyshev pair counting for low-dimensional embeddings."""

import numpy
from numba import njit

_EXTENSION_SORT_MIN_PAIRS = 4096  # A sorted window scan pays off only for large matched blocks.
_LEAF_SIZE = 8  # Points per leaf; _tree_workspace_bytes mirrors the node count this implies.
# Where sample, approximate and multiscale entropy switch from the diagonal scan to the tree.
_TREE_MIN_TEMPLATES = 4096  # An allocation-free scan beats tree setup on short series.
_TREE_MAX_ORDER = 4  # Bound tree storage and avoid ineffective high-dimensional pruning.
_TREE_BATCH_BYTES = 12 * 1024 * 1024  # Bound concurrent row trees, not the process's thread pool.


@njit(cache=True)
def _partition(x, permutation, start, stop, axis, target, delay):
    """Partition an index range around its median without copying vectors."""
    low, high = start, stop - 1
    while low < high:
        pivot = x[permutation[(low + high) // 2] + axis * delay]
        i, j = low, high
        while i <= j:
            while x[permutation[i] + axis * delay] < pivot:
                i += 1
            while x[permutation[j] + axis * delay] > pivot:
                j -= 1
            if i <= j:
                permutation[i], permutation[j] = permutation[j], permutation[i]
                i += 1
                j -= 1
        if target <= j:
            high = j
        elif target >= i:
            low = i
        else:
            break


@njit(cache=True)
def _build(x, dimension, n, index_dtype=numpy.int32, delay=1, extra=False):
    """Split on dimension coordinates, optionally retaining one extension coordinate."""
    leaves = 1
    while leaves * _LEAF_SIZE < n:
        leaves *= 2
    capacity = 2 * leaves - 1
    permutation = numpy.arange(n, dtype=index_dtype)
    coordinates = dimension + extra
    low = numpy.empty((capacity, dimension))
    high = numpy.empty_like(low)
    bounds = numpy.empty(capacity)
    children = numpy.full(capacity, -1, dtype=index_dtype)
    starts, stops = numpy.empty(capacity, dtype=index_dtype), numpy.empty(capacity, dtype=index_dtype)
    starts[0], stops[0] = 0, n
    node, nodes = 0, 1
    while node < nodes:
        start, stop = starts[node], stops[node]
        for k in range(dimension):
            lower = upper = x[permutation[start] + k * delay]
            for i in range(start + 1, stop):
                value = x[permutation[i] + k * delay]
                lower = min(lower, value)
                upper = max(upper, value)
            low[node, k], high[node, k] = lower, upper
        axis = numpy.int64(0)  # not the literal 0: numba would compile _partition a second time for it
        for k in range(1, dimension):
            if high[node, k] - low[node, k] > high[node, axis] - low[node, axis]:
                axis = k
        bounds[node] = high[node, axis] - low[node, axis]
        if stop - start > _LEAF_SIZE and bounds[node] != 0:
            middle = (start + stop) // 2
            _partition(x, permutation, start, stop, axis, middle, delay)
            children[node] = nodes
            starts[nodes], stops[nodes] = start, middle
            starts[nodes + 1], stops[nodes + 1] = middle, stop
            nodes += 2
        node += 1
    points = numpy.empty((n, coordinates))
    for i in range(n):
        for k in range(coordinates):
            points[i, k] = x[permutation[i] + k * delay]
    return points, low, high, bounds, children, starts, stops


@njit(cache=True)
def _sorted_extension_count(points, first_a, stop_a, first_b, stop_b, same, tolerance, dimension):
    """Count a dense one-coordinate block with sorted, strict sliding windows."""
    a = numpy.sort(points[first_a:stop_a, dimension])
    b = a if same else numpy.sort(points[first_b:stop_b, dimension])
    left, right, count = 0, 0, 0
    for i in range(len(a)):
        while left < len(b) and a[i] - b[left] >= tolerance:
            left += 1
        if same:
            count += i - left
        else:
            while right < len(b) and b[right] - a[i] < tolerance:
                right += 1
            count += right - left
    return count


@njit(cache=True)
def _extension_count(points, first_a, stop_a, first_b, stop_b, same, tolerance, dimension):
    """Scan only the extension coordinate after the base box is wholly matched."""
    low_a = high_a = points[first_a, dimension]
    for i in range(first_a + 1, stop_a):
        low_a = min(low_a, points[i, dimension])
        high_a = max(high_a, points[i, dimension])
    low_b, high_b = low_a, high_a
    if not same:
        low_b = high_b = points[first_b, dimension]
        for j in range(first_b + 1, stop_b):
            low_b = min(low_b, points[j, dimension])
            high_b = max(high_b, points[j, dimension])
    if low_a - high_b >= tolerance or low_b - high_a >= tolerance:
        return 0
    if high_a - low_b < tolerance and high_b - low_a < tolerance:
        size_a, size_b = numpy.int64(stop_a - first_a), numpy.int64(stop_b - first_b)
        return size_a * (size_a - 1) // 2 if same else size_a * size_b
    if (stop_a - first_a) * (stop_b - first_b) >= _EXTENSION_SORT_MIN_PAIRS:
        return _sorted_extension_count(points, first_a, stop_a, first_b, stop_b, same, tolerance, dimension)
    count = 0
    for i in range(first_a, stop_a):
        start = i + 1 if same else first_b
        value = points[i, dimension]
        for j in range(start, stop_b):
            count += abs(value - points[j, dimension]) < tolerance
    return count


@njit(cache=True)
def _count_matches(points, low, high, bounds, children, starts, stops, tolerance, dimension):
    """Count base matches and their extension subset in one unordered pair traversal."""
    # A depth-first pair traversal uses O(tree depth), not O(node count), slots.
    depth, nodes = 1, len(starts)
    while nodes > 1:
        depth += 1
        nodes //= 2
    stack = numpy.empty((4 * depth, 2), dtype=numpy.int64)
    stack[0, 0], stack[0, 1] = 0, 0
    pending = 1
    count_m, count_m1 = 0, 0
    while pending:
        pending -= 1
        a, b = stack[pending, 0], stack[pending, 1]
        inside, outside = True, False
        for k in range(dimension):
            if low[a, k] - high[b, k] >= tolerance or low[b, k] - high[a, k] >= tolerance:
                outside = True
                break
            if high[a, k] - low[b, k] >= tolerance or high[b, k] - low[a, k] >= tolerance:
                inside = False
        if outside:
            continue
        size_a, size_b = numpy.int64(stops[a] - starts[a]), numpy.int64(stops[b] - starts[b])
        if inside:
            count_m += size_a * (size_a - 1) // 2 if a == b else size_a * size_b
            count_m1 += _extension_count(points, starts[a], stops[a], starts[b], stops[b], a == b, tolerance, dimension)
            continue
        leaf_a, leaf_b = children[a] == -1, children[b] == -1
        if leaf_a and leaf_b:
            for i in range(starts[a], stops[a]):
                start = i + 1 if a == b else starts[b]
                for j in range(start, stops[b]):
                    # Branch-free: an early exit mispredicts on about half of the leaf pairs.
                    match = True
                    for k in range(dimension):
                        match &= abs(points[i, k] - points[j, k]) < tolerance
                    count_m += match
                    count_m1 += match & (abs(points[i, dimension] - points[j, dimension]) < tolerance)
            continue
        if a == b:
            left, right = children[a], children[a] + 1
            stack[pending, 0], stack[pending, 1] = left, left
            stack[pending + 1, 0], stack[pending + 1, 1] = left, right
            stack[pending + 2, 0], stack[pending + 2, 1] = right, right
            pending += 3
        elif leaf_b or (not leaf_a and bounds[a] >= bounds[b]):
            stack[pending, 0], stack[pending, 1] = children[a], b
            stack[pending + 1, 0], stack[pending + 1, 1] = children[a] + 1, b
            pending += 2
        else:
            stack[pending, 0], stack[pending, 1] = a, children[b]
            stack[pending + 1, 0], stack[pending + 1, 1] = a, children[b] + 1
            pending += 2
    return int(count_m), int(count_m1)


@njit(cache=True)
def _tree_count_typed(x, dimension, n, tolerance, index_dtype, delay=1):
    """Keep each tree's lifetime inside one call, independent of index width."""
    ordered, low, high, bounds, children, starts, stops = _build(x, dimension, n, index_dtype, delay, True)
    return _count_matches(ordered, low, high, bounds, children, starts, stops, tolerance, dimension)


@njit(cache=True)
def _tree_match_counts(x, dimension, n, tolerance, delay=1):
    """Build one tree for the common templates and count both embedding lengths.

    Compact indices narrow neither the pair counts nor the accepted input length.
    """
    if n <= 2147483647:
        return _tree_count_typed(x, dimension, n, tolerance, numpy.int32, delay)
    return _tree_count_typed(x, dimension, n, tolerance, numpy.int64, delay)


def _tree_workspace_bytes(n, dimension):
    """Bound the build's live array bytes, including permutation and reordered points."""
    leaves = 1
    while leaves * _LEAF_SIZE < n:
        leaves *= 2
    index_bytes = 4 if n <= 2147483647 else 8
    return n * (8 * dimension + index_bytes) + (2 * leaves - 1) * (16 * dimension + 8 + 3 * index_bytes)
