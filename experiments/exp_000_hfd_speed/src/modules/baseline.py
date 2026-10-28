"""Profile the production HFD before implementing any alternatives."""

import json
import logging
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
import scipy

from complexeva.higuchi import compute_L_x, hfd
from experiments.exp_000_hfd_speed.src.utils.profiling import profiled

LOGGER = logging.getLogger(__name__)
SIZES = (300, 3000, 30000, 1_000_000)
SEED = 20261028


def signal(n):
    """Return a deterministic, contiguous float64 white-noise signal."""
    return np.random.default_rng(SEED + n).standard_normal(n)


def write_json(path, value):
    """Write structured experiment evidence below its canonical results path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def metadata(results, stage):
    """Record runtime, thread controls and exact source text used in this stage."""
    import complexeva.higuchi

    write_json(
        results / "json" / f"{stage}-environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "scipy": scipy.__version__,
            "cpu_count": os.cpu_count(),
            "numba_num_threads": numba.get_num_threads(),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "NUMBA_CACHE_DIR",
                    "NUMBA_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_WAIT_POLICY",
                    "KMP_BLOCKTIME",
                )
            },
            "seed": SEED,
            "k_max": 10,
            "sizes": SIZES,
            "production_source": "complexeva/higuchi.py",
        },
    )
    source = Path(complexeva.higuchi.__file__).read_text()
    (results / "json" / f"{stage}-higuchi-source.txt").write_text(source)


def median_seconds(fn, x, k_max, batches=7, target=0.03):
    """Calibrate a batch, then return warm median and spread without tracing."""
    start = perf_counter()
    fn(x, k_max)
    estimate = max(perf_counter() - start, 1e-9)
    count = max(1, min(100_000, int(target / estimate)))
    samples = []
    for _ in range(batches):
        start = perf_counter()
        for _ in range(count):
            fn(x, k_max)
        samples.append((perf_counter() - start) / count)
    return {
        "median_s": float(np.median(samples)),
        "min_s": min(samples),
        "max_s": max(samples),
        "samples_s": samples,
        "calls_per_batch": count,
    }


def baseline(results, *, enabled=True):
    """Warm JIT once, profile each requested N, and time full/core costs."""
    metadata(results, "baseline")
    start = perf_counter()
    hfd(signal(300), 10)
    cold_s = perf_counter() - start
    rows = []
    for i, n in enumerate(SIZES, 1):
        LOGGER.info("[%s/%s] profiling production hfd n=%s k_max=10", i, len(SIZES), n)
        x = signal(n)
        timing = median_seconds(hfd, x, 10)
        core = median_seconds(compute_L_x, x, 10)
        count = max(1, int(0.6 / timing["median_s"]))
        output = results / "profile" / f"baseline-n{n}.html" if enabled else None
        with profiled(output):
            for _ in range(count):
                hfd(x, 10)
        row = {"n": n, "k_max": 10, "hfd": float(hfd(x, 10)), "whole": timing, "compute_L_x": core, "profile_calls": count}
        rows.append(row)
        LOGGER.info("n=%s whole=%.3fus core=%.3fus", n, timing["median_s"] * 1e6, core["median_s"] * 1e6)
    write_json(results / "json" / "baseline.json", {"first_call_s": cold_s, "rows": rows})
