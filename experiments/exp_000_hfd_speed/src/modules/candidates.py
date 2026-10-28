"""Alternatives coexist only because comparing them is this experiment's deliverable."""

from dataclasses import dataclass
from functools import partial

import numpy as np
from numba import njit, prange

from complexeva._validation import as_finite_1d, check_positive_int
from complexeva.higuchi import compute_L_x, hfd


@njit(cache=True, inline="always")
def mean_length(x, k):
    """Keep increment order, but remove per-k allocations and duplicate loads."""
    n = len(x)
    total = 0.0
    for m in range(k):
        n_max = (n - 1 - m) // k
        length = 0.0
        previous = x[m]
        for index in range(m + k, n, k):
            current = x[index]
            length += np.abs(current - previous)
            previous = current
        length *= (n - 1) / (n_max * k)
        total += length / k
    return total / k


@njit(cache=True)
def loop_lengths(x, k_max):
    """Serial allocation-light Higuchi lengths with no reassociation/fastmath."""
    log_x = np.empty(k_max)
    log_l = np.empty(k_max)
    for index in range(k_max):
        k = index + 1
        log_x[index] = np.log(1.0 / k)
        log_l[index] = np.log(mean_length(x, k))
    return log_x, log_l


@njit(cache=True, parallel=True)
def parallel_lengths(x, k_max):
    """Parallelize independent k values; each k retains serial addition order."""
    log_x = np.empty(k_max)
    log_l = np.empty(k_max)
    for index in prange(k_max):
        k = index + 1
        log_x[index] = np.log(1.0 / k)
        log_l[index] = np.log(mean_length(x, k))
    return log_x, log_l


@njit(cache=True)
def slope(log_x, log_l):
    """OLS slope with an intercept, using centered sums rather than an SVD."""
    mean_x = np.mean(log_x)
    mean_l = np.mean(log_l)
    numerator = 0.0
    denominator = 0.0
    for index in range(len(log_x)):
        centered_x = log_x[index] - mean_x
        numerator += centered_x * (log_l[index] - mean_l)
        denominator += centered_x * centered_x
    return numerator / denominator


@njit(cache=True)
def slope_only_core(x, k_max):
    """Leave production lengths unchanged and remove only the Python SVD path."""
    log_x, log_l = compute_L_x(x, k_max)
    return slope(log_x, log_l)


@njit(cache=True)
def loop_core(x, k_max):
    """Fuse the allocation-light lengths and centered OLS in one JIT call."""
    log_x, log_l = loop_lengths(x, k_max)
    return slope(log_x, log_l)


@njit(cache=True)
def parallel_core(x, k_max):
    """Use parallel k lengths, then a deterministic serial OLS reduction."""
    log_x, log_l = parallel_lengths(x, k_max)
    return slope(log_x, log_l)


def numpy_lengths(x, k_max):
    """Vectorize increments and grouped m sums, retaining float64 accumulators."""
    n = len(x)
    log_l = np.empty(k_max)
    for k in range(1, k_max + 1):
        differences = np.abs(x[k:] - x[:-k])
        full = (len(differences) // k) * k
        totals = differences[:full].reshape(-1, k).sum(axis=0, dtype=np.float64)
        tail = differences[full:]
        totals[: len(tail)] += tail
        n_max = (n - 1 - np.arange(k)) // k
        totals *= (n - 1) / (n_max * k)
        log_l[k - 1] = np.log((totals / k).mean())
    return np.log(1.0 / np.arange(1, k_max + 1)), log_l


def check_input(x, k_max):
    """Reuse the current production boundary, including its float64 conversion."""
    x = as_finite_1d(x, "X")
    k_max = check_positive_int(k_max, "k_max", minimum=2)
    if k_max > len(x) // 2:
        raise ValueError(f"k_max must be between 2 and len(X) // 2 = {len(x) // 2}, got {k_max}")
    return x, k_max


def with_lstsq(x, k_max, *, lengths):
    """Hold regression fixed while comparing different length kernels."""
    x, k_max = check_input(x, k_max)
    log_x, log_l = lengths(x, k_max)
    design = np.column_stack((log_x, np.ones_like(log_x)))
    return float(np.linalg.lstsq(design, log_l, rcond=None)[0][0])


def with_core(x, k_max, *, core):
    """Expose a fused native core behind the same current production boundary."""
    x, k_max = check_input(x, k_max)
    return core(x, k_max)


@dataclass(frozen=True)
class Candidate:
    """One implementation and its explicitly controlled Numba thread count."""

    name: str
    function: object
    threads: int = 1


def candidates():
    """Enumerate the requested algorithm families and regression ablations."""
    return [
        Candidate("current", hfd),
        Candidate("loop_lstsq", partial(with_lstsq, lengths=loop_lengths)),
        Candidate("slope_only", partial(with_core, core=slope_only_core)),
        Candidate("loop_fused", partial(with_core, core=loop_core)),
        Candidate("numpy_lstsq", partial(with_lstsq, lengths=numpy_lengths)),
        *[Candidate(f"parallel_{threads}", partial(with_core, core=parallel_core), threads) for threads in (1, 2, 4)],
    ]
