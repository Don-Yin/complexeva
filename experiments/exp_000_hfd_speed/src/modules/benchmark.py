"""Randomized, warmed and batched wall-clock comparisons."""

import csv
import logging
from time import perf_counter

import numba
import numpy as np

from experiments.exp_000_hfd_speed.src.modules.baseline import SEED, SIZES, metadata, signal, write_json
from experiments.exp_000_hfd_speed.src.modules.candidates import candidates
from experiments.exp_000_hfd_speed.src.modules.validation import validate

LOGGER = logging.getLogger(__name__)


def batched(method, x, k_max, count):
    """Time full calls after selecting threads outside the measured interval."""
    numba.set_num_threads(method.threads)
    start = perf_counter()
    for _ in range(count):
        method.function(x, k_max)
    return (perf_counter() - start) / count


def measure_case(methods, x, k_max, repeats, target, rng):
    """Shuffle implementation order in every round to reduce drift bias."""
    counts = {}
    samples = {m.name: [] for m in methods}
    for method in methods:
        estimate = batched(method, x, k_max, 3)
        counts[method.name] = max(3, min(100_000, int(target / max(estimate, 1e-9))))
    for _ in range(repeats):
        for index in rng.permutation(len(methods)):
            method = methods[index]
            samples[method.name].append(batched(method, x, k_max, counts[method.name]))
    rows = []
    for method in methods:
        values = samples[method.name]
        row = {
            "n": len(x),
            "k_max": k_max,
            "method": method.name,
            "threads": method.threads,
            "median_s": float(np.median(values)),
            "min_s": min(values),
            "max_s": max(values),
            "paired_speedup": float(np.median(np.asarray(samples["current"]) / values)),
            "calls_per_batch": counts[method.name],
            "samples_s": values,
        }
        rows.append(row)
        LOGGER.info(
            "n=%s k_max=%s method=%s median=%.3fus paired speedup=%.2fx",
            len(x),
            k_max,
            method.name,
            row["median_s"] * 1e6,
            row["paired_speedup"],
        )
    return rows


def benchmark(results, *, profiled_run=False, repeats=9, target=0.025):
    """Validate every implementation, then time default and larger k_max."""
    if repeats < 3 or target <= 0:
        raise ValueError(f"repeats must be >=3 and target >0, got repeats={repeats}, target={target}")
    methods = candidates()
    label = "benchmark-profiled" if profiled_run else "benchmark"
    metadata(results, label)
    validation = validate(results, methods)
    rng = np.random.default_rng(SEED)
    rows = []
    scenarios = [(n, k) for k in (10, 64) for n in SIZES]
    for index, (n, k_max) in enumerate(scenarios, 1):
        LOGGER.info("[%s/%s] benchmarking n=%s k_max=%s", index, len(scenarios), n, k_max)
        rows.extend(measure_case(methods, signal(n), k_max, repeats, target, rng))
    report = {
        "profiled": profiled_run,
        "repeats": repeats,
        "target_batch_s": target,
        "timing": "warm full-call wall seconds; randomized interleaved batches",
        "input": "contiguous float64 white noise",
        "validation": validation,
        "rows": rows,
    }
    write_json(results / "json" / f"{label}.json", report)
    table = results / "tables" / f"{label}.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key in rows[0] if key != "samples_s"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "samples_s"} for row in rows)
