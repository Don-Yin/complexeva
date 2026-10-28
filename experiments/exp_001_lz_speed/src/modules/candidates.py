"""Two deliberately retained alternatives for the exact-count speed comparison."""

import numpy as np
from numba import njit


@njit
def packed_count(bits):
    """Compare 64-bit prefixes, then check full matches to avoid hash collisions."""
    n = len(bits)
    words = np.empty(n, dtype=np.uint64)
    word = np.uint64(0)
    for i in range(n - 1, -1, -1):
        word = (word >> np.uint64(1)) | (np.uint64(bits[i]) << np.uint64(63))
        words[i] = word
    position, count = 1, 1
    while position < n:
        smallest = np.uint64(0xFFFFFFFFFFFFFFFF)
        for i in range(position):
            difference = words[i] ^ words[position]
            smallest = min(smallest, difference)
        if smallest != 0:
            length = 0
            while (smallest & (np.uint64(1) << np.uint64(63))) == 0:
                length += 1
                smallest <<= np.uint64(1)
        else:
            length = min(64, n - position)
            for i in range(position):
                if words[i] != words[position]:
                    continue
                matched = 64
                while position + matched < n and bits[i + matched] == bits[position + matched]:
                    matched += 1
                length = max(length, matched)
                if position + length >= n:
                    break
        position += length + 1
        count += 1
    return count


@njit
def automaton_count(bits):
    """Parse exhaustive-history LZ using earliest occurrences in a suffix automaton."""
    n = len(bits)
    if n < 2:
        return 1
    # Native indices avoid int32 overflow; 10*n*sizeof(intp) bytes of workspace.
    transitions = np.full((2 * n, 2), -1, dtype=np.intp)
    link = np.full(2 * n, -1, dtype=np.intp)
    length = np.zeros(2 * n, dtype=np.intp)
    first_end = np.empty(2 * n, dtype=np.intp)
    last, size = 0, 1
    for position in range(n):
        symbol = int(bits[position])
        current = size
        size += 1
        length[current] = length[last] + 1
        first_end[current] = position
        previous = last
        while previous >= 0 and transitions[previous, symbol] < 0:
            transitions[previous, symbol] = current
            previous = link[previous]
        if previous < 0:
            link[current] = 0
        else:
            target = transitions[previous, symbol]
            if length[previous] + 1 == length[target]:
                link[current] = target
            else:
                clone = size
                size += 1
                length[clone] = length[previous] + 1
                first_end[clone] = first_end[target]
                link[clone] = link[target]
                transitions[clone, 0] = transitions[target, 0]
                transitions[clone, 1] = transitions[target, 1]
                while previous >= 0 and transitions[previous, symbol] == target:
                    transitions[previous, symbol] = clone
                    previous = link[previous]
                link[target] = clone
                link[current] = clone
        last = current
    position, count = 1, 1
    while position < n:
        state, matched = 0, 0
        while position + matched < n:
            state = transitions[state, int(bits[position + matched])]
            matched += 1
            # A prior start, not a prior end, preserves self-overlapping copies.
            if first_end[state] - matched + 1 >= position:
                break
        position += matched
        count += 1
    return count
