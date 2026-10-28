"""Focused regressions for count semantics and the experiment entry contract."""

import importlib.util
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from complexeva.lempel_ziv import _binarize, lempel_ziv_core
from experiments.exp_001_lz_speed.src.modules.benchmark import candidate_pipeline, no_cast_binarize
from experiments.exp_001_lz_speed.src.modules.candidates import automaton_count, packed_count
from experiments.exp_001_lz_speed.src.utils import timing
from experiments.exp_001_lz_speed.src.utils.profiling import profiled

EXPERIMENT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("candidate", [packed_count, automaton_count])
@pytest.mark.parametrize("word, expected", [("", 1), ("1", 1), ("0" * 130, 2), ("01" * 65, 3), ("0001101001000101", 6)])
def test_known_counts(candidate, word, expected):
    """Keep the final copied component and self-overlapping copies."""
    assert candidate(np.array([int(bit) for bit in word], dtype=np.int32)) == expected


def test_exhaustive_small():
    """Check every nonempty binary word through length nine."""
    for n in range(1, 10):
        for word in itertools.product((0, 1), repeat=n):
            bits = np.asarray(word, dtype=np.int32)
            expected = lempel_ziv_core(bits)
            assert packed_count(bits) == expected
            assert automaton_count(bits) == expected


@pytest.mark.parametrize("dtype", [np.bool_, np.int32])
def test_noncontiguous_readonly_input(dtype):
    """Read input without mutating it or requiring a contiguous array."""
    bits = np.random.default_rng(6).integers(0, 2, 2048).astype(dtype)[::2]
    bits.flags.writeable = False
    expected = lempel_ziv_core(bits)
    assert packed_count(bits) == expected
    assert automaton_count(bits) == expected


@pytest.mark.parametrize("method", ["median", "mean", "diff"])
def test_no_cast_retains_validation_and_thresholds(method):
    """Only storage changes: preprocessing keeps production validation."""
    for signal in ([], [1.0], [0.0, 0.0, 1.0, 1.0], np.arange(30, dtype=np.float32)[::2]):
        np.testing.assert_array_equal(no_cast_binarize(signal, method), _binarize(signal, method))
    for signal in ([np.nan], [1j], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            no_cast_binarize(signal, method)


def test_candidate_hilbert_retains_input_validation():
    """The replacement pipeline pays for the same real/finite input validation."""
    for signal in ([], [np.nan], [1j], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            candidate_pipeline(signal, automaton_count, True)


def test_import_does_not_profile(monkeypatch):
    """An imported entry must not create or overwrite the default report."""
    report = EXPERIMENT / ".results" / "profile" / "run.html"
    before = report.stat().st_mtime_ns if report.exists() else None
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    spec = importlib.util.spec_from_file_location("lz_speed_import_test", EXPERIMENT / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    after = report.stat().st_mtime_ns if report.exists() else None
    assert before == after


def test_profile_written_on_failure(tmp_path):
    """Save the actual failing workload profile without swallowing its exception."""
    report = tmp_path / "profile" / "run.html"
    with pytest.raises(ValueError, match="intentional profile failure"), profiled(report):
        raise ValueError("intentional profile failure")
    assert report.stat().st_size > 0
    assert report.with_suffix(".txt").stat().st_size > 0


def test_entry_propagates_failure():
    """A failing benchmark exits nonzero rather than becoming a successful launch."""
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT / "run.py"), "--lengths", "1", "--no-profile"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "lengths must all be >= 2, got [1]" in result.stderr


def test_benchmark_lock_released_on_failure(monkeypatch, tmp_path):
    """The directory lock is always released after a Python exception."""
    lock = tmp_path / "benchmark.lock"
    monkeypatch.setattr(timing, "Path", lambda _path: lock)
    with pytest.raises(ValueError, match="intentional lock failure"), timing.benchmark_lock():
        assert lock.is_dir()
        raise ValueError("intentional lock failure")
    assert not lock.exists()
