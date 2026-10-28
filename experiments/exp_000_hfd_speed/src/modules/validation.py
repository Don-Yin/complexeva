"""Strict absolute-tolerance checks against the production HFD."""

import logging

import numba
import numpy as np

from complexeva.higuchi import hfd
from experiments.exp_000_hfd_speed.src.modules.baseline import SEED, SIZES, signal, write_json

LOGGER = logging.getLogger(__name__)
TOLERANCE = 1e-12


def cases():
    """Cover sizes, k boundaries, layouts, signal families and input dtypes."""
    for n in SIZES:
        for k_max in (10, 64):
            yield f"benchmark-n{n}-k{k_max}", signal(n), k_max
    rng = np.random.default_rng(SEED)
    for n in (4, 20, 21, 64, 300, 3000):
        noise = rng.standard_normal(n)
        families = {
            "white": noise,
            "brown": np.cumsum(noise),
            "sine": np.sin(np.linspace(0.0, 20 * np.pi, n)),
            "ramp": np.arange(n, dtype=np.float64),
            "plateau": np.repeat(np.arange((n + 2) // 3), 3)[:n].astype(np.float64),
            "strided": rng.standard_normal(2 * n)[::2],
            "reversed": noise[::-1],
            "float32": noise.astype(np.float32),
            "int16": rng.integers(-30_000, 30_000, n, dtype=np.int16),
            "offset": 1e6 + noise * 1e-6,
            "tiny": noise * 1e-120,
            "large": noise * 1e120,
            "constant": np.ones(n),
        }
        k_values = sorted({2, min(10, n // 2), min(64, n // 2), min(n // 2, 150)})
        for name, x in families.items():
            for k_max in k_values:
                yield f"{name}-n{n}-k{k_max}", x, k_max


def validate(results, methods):
    """Write all worst-case errors and fail the run on any fidelity violation."""
    summary = {m.name: {"max_abs_error": 0.0, "worst_case": None, "finite": 0, "nan": 0} for m in methods}
    failures = []
    all_cases = list(cases())
    with np.errstate(divide="ignore", invalid="ignore"):
        for index, (name, x, k_max) in enumerate(all_cases, 1):
            if index == 1 or index % 40 == 0 or index == len(all_cases):
                LOGGER.info("[%s/%s] fidelity case=%s", index, len(all_cases), name)
            expected = hfd(x, k_max)
            for method in methods:
                numba.set_num_threads(method.threads)
                actual = method.function(x, k_max)
                entry = summary[method.name]
                if np.isnan(expected) and np.isnan(actual):
                    entry["nan"] += 1
                    continue
                error = float(abs(actual - expected))
                entry["finite"] += 1
                if error > entry["max_abs_error"]:
                    entry["max_abs_error"] = error
                    entry["worst_case"] = name
                if not np.isfinite(error) or error >= TOLERANCE:
                    failures.append(
                        {
                            "case": name,
                            "method": method.name,
                            "expected": str(expected),
                            "actual": str(actual),
                            "error": str(error),
                        }
                    )
    report = {
        "absolute_tolerance_strict": TOLERANCE,
        "cases": len(all_cases),
        "comparisons": len(all_cases) * len(methods),
        "methods": summary,
        "failures": failures,
    }
    write_json(results / "json" / "validation.json", report)
    if failures:
        raise AssertionError(f"{len(failures)} fidelity failures; first={failures[0]}")
    LOGGER.info(
        "fidelity passed: %s comparisons, max abs error=%s",
        report["comparisons"],
        max(item["max_abs_error"] for item in summary.values()),
    )
    return report
