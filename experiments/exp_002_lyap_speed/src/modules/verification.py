"""Differential fixtures covering Theiler boundaries, exact ties and float scales."""

from __future__ import annotations

import numpy as np

from .benchmark import lyap, save_table
from .candidates import SEARCHES, estimate, prepare_embedding
from .kernels import follow_divergence, neighbours_original


def logistic(n):
    """Generate the logistic-map fixture used by the package's known-answer test."""
    x = np.empty(n)
    x[0] = 0.3
    for i in range(1, n):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    return x


def henon(n):
    """Generate the Henon-map fixture used by the package's known-answer test."""
    x, y, out = 0.1, 0.1, np.empty(n)
    for i in range(n):
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        out[i] = x
    return out


def cases():
    """Build deterministic cases spanning the accepted valid-input parameter space."""
    rng = np.random.default_rng(481516)
    result = [
        ("logistic", logistic(1000), {"m": 1, "trajectory_len": 8}),
        ("henon", henon(1000), {"m": 2, "trajectory_len": 8}),
        ("sine", np.sin(np.linspace(0, 40 * np.pi, 500)), {}),
        ("constant", np.zeros(500), {"min_tsep": 5}),
        ("periodic-ties", np.tile([0.0, 1.0, 0.0, -1.0, 2.0, 2.0], 85), {"min_tsep": 5}),
        ("quantized-ties", rng.integers(-3, 4, 500).astype(float), {"min_tsep": 7}),
        ("alternating-ties", np.tile([0.0, 1.0], 250), {"m": 3, "min_tsep": 5}),
        ("wide-window", np.arange(1000.0), {"min_tsep": 350}),
        ("theiler-boundary", np.arange(80.0), {"min_tsep": 29}),
        ("no-exclusion", rng.normal(size=500), {"min_tsep": 0}),
        ("near-ties", rng.integers(0, 4, 500) + 1e-14 * rng.normal(size=500), {"m": 3, "min_tsep": 0}),
        ("noncontiguous", rng.normal(size=1024)[::2], {"tau": 3}),
        ("reversed", rng.normal(size=500)[::-1], {}),
        ("tiny-scale", rng.normal(size=500) * 1e-100, {"min_tsep": 5}),
        ("large-scale", rng.normal(size=500) * 1e100, {"min_tsep": 5}),
        ("squared-overflow", np.random.default_rng(1).normal(size=128) * 1e200, {"min_tsep": 5}),
        ("squared-underflow", np.random.default_rng(1).normal(size=128) * 1e-200, {"min_tsep": 5}),
        ("fit-offset-dt", rng.normal(size=500), {"m": 3, "tau": 2, "dt": 0.125, "fit_offset": 3}),
    ]
    for dimension in (1, 2, 3, 8, 16, 32):
        result.append((f"embedding-{dimension}", rng.normal(size=600), {"m": dimension, "tau": 2, "min_tsep": 11}))
    for seed in range(10):
        result.append((f"random-{seed}", np.random.default_rng(seed).normal(size=500), {"min_tsep": seed * 7}))
    return result


def assert_identical(actual, expected):
    """Require absolute error <= 1e-12, matching signed infinities exactly."""
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    finite = np.isfinite(expected)
    return float(np.max(np.abs(np.asarray(actual)[finite] - np.asarray(expected)[finite]), initial=0))


def verify_case(name, x, parameters):
    """Compare neighbours, full divergence trajectory and final public estimate."""
    y, window, length = prepare_embedding(x, **parameters)
    expected_neighbours = neighbours_original(y, window, length)
    expected_divergence = lyap._divergence(y, window, length)
    expected_value = lyap.lyapunov_exponent(x, **parameters)
    rows = []
    for method, search in SEARCHES.items():
        neighbours = search(y, window, length)
        np.testing.assert_array_equal(neighbours, expected_neighbours, err_msg=f"{name}: {method} neighbour mismatch")
        divergence_error = assert_identical(follow_divergence(y, neighbours, length), expected_divergence)
        value = estimate(x, method, **parameters)
        value_error = assert_identical(value, expected_value)
        rows.append(
            {
                "case": name,
                "method": method,
                "n": len(x),
                "m": y.shape[1],
                "min_tsep": window,
                "neighbours_identical": True,
                "divergence_error": divergence_error,
                "exponent_error": value_error,
            }
        )
    return rows


def verify():
    """Run all fixed cases and retain numerical fidelity evidence."""
    rows = [row for name, x, parameters in cases() for row in verify_case(name, x, parameters)]
    save_table("verification", rows)
    return rows
