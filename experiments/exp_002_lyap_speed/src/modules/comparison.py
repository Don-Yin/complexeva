"""Warm, interleaved end-to-end timings with exact differential checks."""

import json
import logging
import os
from datetime import UTC, datetime
from time import perf_counter

import numpy as np

from ..utils.paths import RESULTS
from .benchmark import lyap, save_table
from .candidates import SEARCHES, estimate, fit_exponent, prepare_embedding
from .kernels import follow_divergence, neighbours_original
from .verification import assert_identical, logistic, verify

logger = logging.getLogger(__name__)


def warm_candidates():
    """Record compilation/startup separately from warmed throughput."""
    x = np.random.default_rng(20260429).normal(size=256)
    timings = {}
    for method in SEARCHES:
        started = perf_counter()
        estimate(x, method)
        timings[method] = perf_counter() - started
    return timings


def workloads(sizes, seed):
    """Cover the requested scale range plus bounded pathological workloads."""
    for n in sizes:
        yield "gaussian", np.random.default_rng(seed + n).normal(size=n), {}
        yield "logistic", logistic(n), {"m": 1, "trajectory_len": 8}
    n = 2000
    yield "constant-ties", np.zeros(n), {"min_tsep": 5}
    yield "quantized-ties", np.random.default_rng(seed).integers(-3, 4, n).astype(float), {"min_tsep": 7}
    yield "wide-window", np.arange(float(n)), {"min_tsep": 700}
    for m in (16, 32):
        yield f"embedding-{m}", np.random.default_rng(seed).normal(size=n), {"m": m}


def compare_one(name, x, parameters, repeats, rng):
    """Check every neighbour before timing randomized method-order repetitions."""
    y, window, length = prepare_embedding(x, **parameters)
    neighbours = neighbours_original(y, window, length)
    expected_divergence = follow_divergence(y, neighbours, length)
    expected = fit_exponent(expected_divergence, parameters.get("dt", 1.0), parameters.get("fit_offset", 0))
    errors = {"original": 0.0}
    for method, search in SEARCHES.items():
        actual = search(y, window, length)
        np.testing.assert_array_equal(actual, neighbours, err_msg=f"{name} n={len(x)} method={method}")
        errors[method] = assert_identical(follow_divergence(y, actual, length), expected_divergence)
    rows = []
    for repeat in range(repeats):
        for method in rng.permutation(["original", *SEARCHES]):
            started = perf_counter()
            value = lyap.lyapunov_exponent(x, **parameters) if method == "original" else estimate(x, method, **parameters)
            elapsed = perf_counter() - started
            error = assert_identical(value, expected)
            rows.append(
                {
                    "scenario": name,
                    "n": len(x),
                    "m": y.shape[1],
                    "min_tsep": window,
                    "method": method,
                    "repeat": repeat + 1,
                    "seconds": elapsed,
                    "exponent": value,
                    "divergence_error": errors[method],
                    "exponent_error": error,
                }
            )
    return rows


def summarize(rows):
    """Summarize medians and spread without hiding background-load variability."""
    output = []
    for scenario, n in dict.fromkeys((row["scenario"], row["n"]) for row in rows):
        group = [row for row in rows if row["scenario"] == scenario and row["n"] == n]
        baseline = np.median([row["seconds"] for row in group if row["method"] == "original"])
        for method in ["original", *SEARCHES]:
            selected = [row for row in group if row["method"] == method]
            times = [row["seconds"] for row in selected]
            median = float(np.median(times))
            output.append(
                {
                    "scenario": scenario,
                    "n": n,
                    "m": selected[0]["m"],
                    "min_tsep": selected[0]["min_tsep"],
                    "method": method,
                    "median_seconds": median,
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "speedup": float(baseline / median),
                    "divergence_error": max(row["divergence_error"] for row in selected),
                    "exponent_error": max(row["exponent_error"] for row in selected),
                }
            )
    return output


def compare(sizes, metadata, repeats):
    """Run reproducible correctness checks and the full speed comparison."""
    logger.info("checking all differential fixtures before timed comparisons")
    fidelity = verify()
    metadata = {
        **metadata,
        "repeats": repeats,
        "fidelity_rows": len(fidelity),
        "timing_order": "interleaved seed=20260429",
        "shared_timing_lock_held": os.environ.get("COMPLEXEVA_BENCH_LOCK_HELD") == "1",
        "started_at": datetime.now(UTC).isoformat(),
    }
    (RESULTS / "manifests").mkdir(parents=True, exist_ok=True)
    (RESULTS / "manifests" / "comparison-environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = list(workloads(sizes, metadata["seed"]))
    rows = []
    rng = np.random.default_rng(metadata["seed"])
    for index, (name, x, parameters) in enumerate(cases, 1):
        logger.info("[%d/%d] compare scenario=%s n=%d", index, len(cases), name, len(x))
        rows.extend(compare_one(name, x, parameters, repeats, rng))
        save_table("timings-raw", rows)
        save_table("timings", summarize(rows))
    logger.info("all %d timing rows passed atol=1e-12 rtol=0", len(rows))
