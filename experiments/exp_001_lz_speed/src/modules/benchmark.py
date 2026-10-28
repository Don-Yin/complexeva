"""Measure production and two exact alternatives with isolated stage timings."""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
from statistics import median
from time import perf_counter
from typing import Any

import numba
import numpy as np
import scipy
from scipy.signal import hilbert

from complexeva import lempel_ziv, lempel_ziv_hilbert
from complexeva._validation import as_finite_1d
from complexeva.lempel_ziv import _binarize, lempel_ziv_core

from ..utils.timing import benchmark_lock
from .candidates import automaton_count, packed_count
from .verification import verify

logger = logging.getLogger(__name__)
CORES = {"production": lempel_ziv_core, "packed64": packed_count, "automaton": automaton_count}


def no_cast_binarize(signal, method):
    """Measure removal of the int32 allocation without changing threshold arithmetic."""
    vector = as_finite_1d(signal, "time_series")
    if vector.size == 0 and method in ("diff", "mean", "median"):
        return np.empty(0, dtype=np.bool_)
    if method == "diff":
        return np.diff(vector) > 0
    if method == "median":
        return vector > np.median(vector)
    if method == "mean":
        return vector > np.mean(vector)
    raise ValueError(f"method must be 'mean', 'median', or 'diff', got {method!r}")


def candidate_pipeline(signal, core, use_hilbert):
    """Keep the exact production transform and binarization, replacing only the core."""
    if use_hilbert:
        signal = as_finite_1d(signal, "time_series")
        if signal.size == 0:
            raise ValueError("time_series must contain at least one sample for the Hilbert transform, got len(time_series)=0")
        signal = np.abs(np.asarray(hilbert(signal)))
    return int(core(_binarize(signal, "median")))


def measure(function, arguments, repeats=1, loops=1):
    """Record steady-state wall times without including a redundant warm-up call."""
    samples = []
    output: Any = None
    with benchmark_lock():
        for _ in range(repeats):
            start = perf_counter()
            for _ in range(loops):
                output = function(*arguments)
            samples.append((perf_counter() - start) / loops)
    return {
        "seconds_median": median(samples),
        "seconds_min": min(samples),
        "repeats": repeats,
        "loops": loops,
        "lock_held": True,
    }, output


def save_rows(path, rows):
    """Persist completed work after every input length."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_calls(output):
    """Measure first calls explicitly, then keep compilation out of steady-state rows."""
    bits = np.random.default_rng(300).integers(0, 2, 300, dtype=np.int32)
    rows = []
    for name, core in CORES.items():
        cold, count = measure(core, (bits,))
        warm, warmed_count = measure(core, (bits,), repeats=3, loops=100)
        if count != warmed_count:
            raise AssertionError(f"warm-up changed {name}: {count} != {warmed_count}")
        rows.append(
            {
                "algorithm": name,
                "first_call_seconds": cold["seconds_median"],
                "warm_call_seconds": warm["seconds_median"],
                "first_minus_warm_seconds": cold["seconds_median"] - warm["seconds_median"],
                "count": int(count),
            }
        )
        logger.info("first call %s seconds=%.6f", name, cold["seconds_median"])
    save_rows(output, rows)


def benchmark_length(n, rows, cores):
    """Time each numeric stage, raw and Hilbert pipelines, and structured core inputs."""
    source = np.random.default_rng(n).standard_normal(n)
    loops = 30 if n <= 1000 else 1
    repeats = 3 if n <= 10000 else 1

    def record(family, stage, algorithm, function, arguments, repetitions=3, inner_loops=1):
        timing, output = measure(function, arguments, repetitions, inner_loops)
        count = int(output) if isinstance(output, (int, np.integer)) else ""
        rows.append({"n": n, "family": family, "stage": stage, "algorithm": algorithm, **timing, "count": count})
        return output

    analytic = record("hilbert", "transform", "scipy", hilbert, (source,), inner_loops=loops)
    envelope = record("hilbert", "envelope_abs", "numpy", np.abs, (analytic,), inner_loops=loops)
    for family, signal in (("raw", source), ("hilbert", envelope)):
        logger.info("n=%s family=%s preprocessing and cores", n, family)
        for method in ("median", "mean", "diff"):
            bits = record(family, f"binarize_{method}", "production", _binarize, (signal, method), inner_loops=loops)
            boolean = record(
                family, f"binarize_{method}", "no_int32_cast", no_cast_binarize, (signal, method), inner_loops=loops
            )
            if not np.array_equal(bits, boolean):
                raise AssertionError(f"binarization mismatch: n={n}, family={family}, method={method}")
        bits = _binarize(signal, "median")
        expected = None
        for name, core in cores.items():
            logger.info("n=%s family=%s core=%s", n, family, name)
            count = record(family, "core", name, core, (bits,), repeats, loops)
            if expected is None:
                expected = count
            if count != expected:
                raise AssertionError(f"core mismatch: n={n}, family={family}, {name}={count}, expected={expected}")
        for name, core in cores.items():
            logger.info("n=%s family=%s pipeline=%s", n, family, name)
            public = lempel_ziv_hilbert if family == "hilbert" else lempel_ziv
            function = public if name == "production" else candidate_pipeline
            arguments = (source,) if name == "production" else (source, core, family == "hilbert")
            count = record(family, "pipeline", name, function, arguments, repeats, loops)
            if count != expected:
                raise AssertionError(f"pipeline mismatch: n={n}, family={family}, {name}={count}, expected={expected}")
    if n in (300, 1000000):
        for family, bits in (("constant", np.zeros(n, dtype=np.int32)), ("alternating", np.arange(n, dtype=np.int32) % 2)):
            expected = None
            for name, core in cores.items():
                count = record(family, "core", name, core, (bits,), repeats, loops)
                if expected is None:
                    expected = count
                if count != expected:
                    raise AssertionError(f"structured mismatch: n={n}, family={family}, {name}={count}, expected={expected}")


def run(args, results):
    """Run cold calls, exact verification, then the requested length sweep."""
    if any(n < 2 for n in args.lengths):
        raise ValueError(f"lengths must all be >= 2, got {args.lengths}")
    table_dir = results / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    prefix = "candidates-" if args.candidates_only else ""
    cores = {name: core for name, core in CORES.items() if name != "production" or not args.candidates_only}
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "numba_cache_dir_set": bool(os.environ.get("NUMBA_CACHE_DIR")),
        "source": str(__import__("complexeva.lempel_ziv", fromlist=["__file__"]).__file__),
        "algorithms": list(cores),
        "validation_matched": True,
        "automaton_index_dtype": np.dtype(np.intp).name,
        "automaton_workspace_bytes_per_sample": 10 * np.dtype(np.intp).itemsize,
        "seed_rule": "numpy.default_rng(n).standard_normal(n)",
        "cold_call_note": "fresh signatures, production first includes Numba runtime initialization",
        "timing_note": "same profiled process; repeats=3 through N=10000, otherwise 1 for cores/pipelines",
        "lock": "/tmp/complexeva-bench.lock, atomic mkdir per timed row, released in finally; waits excluded",
    }
    (table_dir / f"{prefix}environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    first_calls(table_dir / f"{prefix}first-calls.csv")
    if args.verify:
        with benchmark_lock():
            verification = verify()
        (table_dir / f"{prefix}verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    rows = []
    for index, n in enumerate(args.lengths, 1):
        logger.info("[%s/%s] benchmarking n=%s", index, len(args.lengths), n)
        benchmark_length(n, rows, cores)
        save_rows(table_dir / f"{prefix}timings.csv", rows)
