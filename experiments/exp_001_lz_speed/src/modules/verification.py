"""Exact equality checks over exhaustive, structured, and numeric inputs."""

import logging

import numpy as np
from scipy.signal import hilbert

from complexeva import lempel_ziv, lempel_ziv_hilbert
from complexeva.lempel_ziv import _binarize, lempel_ziv_core

from .candidates import automaton_count, packed_count

logger = logging.getLogger(__name__)


def verify_case(bits, label):
    """Reject a single unequal integer rather than using a numerical tolerance."""
    expected = int(lempel_ziv_core(bits))
    for candidate in (packed_count, automaton_count):
        actual = int(candidate(bits))
        if actual != expected:
            raise AssertionError(f"{candidate.__name__} {label}: got {actual}, expected {expected}, n={len(bits)}")
    return expected


def verify():
    """Verify all binary words up to length 14 plus seeded and overlapping cases."""
    cases = 0
    for n in range(1, 15):
        logger.info("[%s/14] exhaustive equality n=%s", n, n)
        shifts = np.arange(n, dtype=np.int32)
        for value in range(1 << n):
            verify_case(((value >> shifts) & 1).astype(np.int32), f"exhaustive:{n}:{value}")
            cases += 1
    rng = np.random.default_rng(761987)
    for index in range(1000):
        n = int(rng.integers(2, 4097))
        bits = (rng.random(n) < (0.01, 0.1, 0.5, 0.9, 0.99)[index % 5]).astype(np.int32)
        verify_case(bits, f"random:{index}")
        cases += 1
    for n in (63, 64, 65, 127, 128, 129, 1000, 10000):
        for period in (1, 2, 3, 7, 63, 64, 65):
            for seed in (np.zeros(period, dtype=np.int32), rng.integers(0, 2, period, dtype=np.int32)):
                verify_case(np.resize(seed, n), f"periodic:{n}:{period}")
                cases += 1
    numeric = 0
    for n in (2, 3, 63, 64, 65, 300, 1000):
        source = rng.standard_normal(n)
        for signal in (source, np.round(source), np.zeros(n), source[::-1], source.astype(np.float32)):
            for use_hilbert in (False, True):
                envelope = np.abs(np.asarray(hilbert(signal))) if use_hilbert else signal
                public = lempel_ziv_hilbert if use_hilbert else lempel_ziv
                for method in ("median", "mean", "diff"):
                    bits = _binarize(envelope, method)
                    raw = verify_case(bits, f"numeric:{n}:{use_hilbert}:{method}")
                    verify_case(bits.astype(np.bool_), "boolean storage")
                    if raw != public(signal, method=method):
                        raise AssertionError(f"public count differs: n={n}, hilbert={use_hilbert}, method={method}")
                    expected = float(raw * np.log2(len(bits)) / len(bits)) if len(bits) > 1 else 0.0
                    if expected != public(signal, method=method, normalize=True):
                        raise AssertionError(f"normalization differs: n={n}, hilbert={use_hilbert}, method={method}")
                    numeric += 1
    return {"binary_cases": cases, "numeric_api_cases": numeric, "integer_mismatches": 0}
