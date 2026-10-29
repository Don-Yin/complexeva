"""Exact unordered Chebyshev pair counting for low-dimensional embeddings."""

import numpy
from numba import njit


@njit(cache=True)
def _partition(x, permutation, start, stop, axis, target):
    """Partition an index range around its median without copying vectors."""
    low, high = start, stop - 1
    while low < high:
        pivot = x[permutation[(low + high) // 2] + axis]
        i, j = low, high
        while i <= j:
            while x[permutation[i] + axis] < pivot:
                i += 1
            while x[permutation[j] + axis] > pivot:
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
def _build(x, dimension, n, leafsize):
    """Build a balanced tree directly over x, copying only reordered rows."""
    leaves = 1
    while leaves * leafsize < n:
        leaves *= 2
    capacity = 2 * leaves - 1
    permutation = numpy.arange(n)
    low = numpy.empty((capacity, dimension))
    high = numpy.empty_like(low)
    bounds = numpy.empty(capacity)
    children = numpy.full(capacity, -1)
    starts, stops = numpy.empty(capacity, dtype=numpy.int64), numpy.empty(capacity, dtype=numpy.int64)
    starts[0], stops[0] = 0, n
    node, nodes = 0, 1
    while node < nodes:
        start, stop = starts[node], stops[node]
        for k in range(dimension):
            lower = upper = x[permutation[start] + k]
            for i in range(start + 1, stop):
                value = x[permutation[i] + k]
                lower = min(lower, value)
                upper = max(upper, value)
            low[node, k], high[node, k] = lower, upper
        axis = numpy.int64(0)  # not the literal 0: numba would compile _partition a second time for it
        for k in range(1, dimension):
            if high[node, k] - low[node, k] > high[node, axis] - low[node, axis]:
                axis = k
        bounds[node] = high[node, axis] - low[node, axis]
        if stop - start > leafsize and bounds[node] != 0:
            middle = (start + stop) // 2
            _partition(x, permutation, start, stop, axis, middle)
            children[node] = nodes
            starts[nodes], stops[nodes] = start, middle
            starts[nodes + 1], stops[nodes + 1] = middle, stop
            nodes += 2
        node += 1
    points = numpy.empty((n, dimension))
    for i in range(n):
        for k in range(dimension):
            points[i, k] = x[permutation[i] + k]
    return points, low, high, bounds, children, starts, stops


@njit(cache=True)
def _count(points, low, high, bounds, children, starts, stops, tolerance):
    """Visit disjoint unordered node pairs, pruning with strict bounds."""
    # A depth-first pair traversal uses O(tree depth), not O(node count), slots.
    depth, nodes = 1, len(starts)
    while nodes > 1:
        depth += 1
        nodes //= 2
    stack = numpy.empty((4 * depth, 2), dtype=numpy.int64)
    stack[0, 0], stack[0, 1] = 0, 0
    pending = 1
    count = 0
    while pending:
        pending -= 1
        a, b = stack[pending, 0], stack[pending, 1]
        inside, outside = True, False
        for k in range(points.shape[1]):
            if low[a, k] - high[b, k] >= tolerance or low[b, k] - high[a, k] >= tolerance:
                outside = True
                break
            if high[a, k] - low[b, k] >= tolerance or high[b, k] - low[a, k] >= tolerance:
                inside = False
        if outside:
            continue
        size_a, size_b = stops[a] - starts[a], stops[b] - starts[b]
        if inside:
            count += size_a * (size_a - 1) // 2 if a == b else size_a * size_b
            continue
        leaf_a, leaf_b = children[a] == -1, children[b] == -1
        if leaf_a and leaf_b:
            for i in range(starts[a], stops[a]):
                start = i + 1 if a == b else starts[b]
                for j in range(start, stops[b]):
                    # Branch-free: an early exit mispredicts on about half of the leaf pairs.
                    match = True
                    for k in range(points.shape[1]):
                        match &= abs(points[i, k] - points[j, k]) < tolerance
                    count += match
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
    return count


@njit(cache=True)
def _tree_match_count(x, dimension, n, tolerance):
    """Count strict distinct-row matches in an embedding of exactly n rows."""
    ordered, low, high, bounds, children, starts, stops = _build(x, dimension, n, 8)
    return _count(ordered, low, high, bounds, children, starts, stops, tolerance)
