"""Profile the unmodified package before implementing speed candidates."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import os
import platform
import sys
from time import perf_counter

import numba
import numpy as np
import scipy

from ..utils.paths import RESULTS
from .kernels import follow_divergence, neighbours_original

lyap = importlib.import_module("complexeva.lyapunov_exponent")
logger = logging.getLogger(__name__)


def prepare() -> dict:
    """Warm the production JIT and record runtime identity outside timing."""
    x = np.random.default_rng(20260429).normal(size=256)
    started = perf_counter()
    lyap.lyapunov_exponent(x)
    y = np.stack((x[:-1], x[1:]), axis=1)
    follow_divergence(y, neighbours_original(y, 4, 20), 20)
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numba_threads": numba.get_num_threads(),
        "warmup_seconds": perf_counter() - started,
        "parameters": {"tau": 1, "m": 2, "dt": 1.0, "min_tsep": None, "trajectory_len": 20, "fit_offset": 0},
        "seed": 20260429,
    }


def baseline(sizes: list[int], metadata: dict) -> None:
    """Profile the public function on deterministic Gaussian time series."""
    rows = []
    for index, n in enumerate(sizes, 1):
        logger.info("[%d/%d] baseline n=%d", index, len(sizes), n)
        x = np.random.default_rng(metadata["seed"] + n).normal(size=n)
        started = perf_counter()
        result = lyap.lyapunov_exponent(x)
        elapsed = perf_counter() - started
        rows.append({"n": n, "seconds": elapsed, "exponent": result})
        logger.info("baseline n=%d seconds=%.6f exponent=%.17g", n, elapsed, result)
    save_table("baseline", rows)
    (RESULTS / "manifests").mkdir(parents=True, exist_ok=True)
    (RESULTS / "manifests" / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def save_table(name: str, rows: list[dict]) -> None:
    """Store reader-facing numerical evidence under the experiment tables."""
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "tables" / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stages(sizes: list[int], metadata: dict) -> None:
    """Time an exact stage split and compare all divergence values bitwise."""
    rows = []
    for index, n in enumerate(sizes, 1):
        logger.info("[%d/%d] stage profile n=%d", index, len(sizes), n)
        x = np.random.default_rng(metadata["seed"] + n).normal(size=n)
        w = min(lyap._mean_period(x), n // 4)
        y = np.stack((x[:-1], x[1:]), axis=1)
        expected = lyap._divergence(y, w, 20)
        started = perf_counter()
        neighbours = neighbours_original(y, w, 20)
        search_seconds = perf_counter() - started
        started = perf_counter()
        actual = follow_divergence(y, neighbours, 20)
        follow_seconds = perf_counter() - started
        np.testing.assert_array_equal(actual, expected)
        row = {"n": n, "min_tsep": w, "search_seconds": search_seconds, "follow_seconds": follow_seconds}
        row["search_fraction"] = search_seconds / (search_seconds + follow_seconds)
        row["max_abs_error"] = float(np.max(np.abs(actual - expected)))
        rows.append(row)
        logger.info("stage profile n=%d search=%.6f follow=%.6f", n, search_seconds, follow_seconds)
    save_table("stages", rows)
    x = np.random.default_rng(metadata["seed"]).normal(size=500)
    y = np.stack((x[:-1], x[1:]), axis=1)
    logger.info("interpreted diagnostic n=500; not a throughput measurement")
    np.testing.assert_allclose(lyap._divergence.py_func(y, 4, 20), lyap._divergence(y, 4, 20), rtol=0, atol=1e-12)
