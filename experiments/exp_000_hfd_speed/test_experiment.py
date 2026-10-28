"""Regression checks for evidence gates and entrypoint side effects."""

import importlib

import numpy as np
import pytest

from complexeva.higuchi import hfd
from experiments.exp_000_hfd_speed.src.modules.candidates import Candidate
from experiments.exp_000_hfd_speed.src.modules.validation import validate
from experiments.exp_000_hfd_speed.src.utils.profiling import profiled


def test_fidelity_failure_is_written_and_raised(tmp_path, monkeypatch):
    """A numerical mismatch must not be hidden or silently benchmarked."""
    x = np.random.default_rng(7).standard_normal(40)
    monkeypatch.setattr("experiments.exp_000_hfd_speed.src.modules.validation.cases", lambda: iter([("mismatch", x, 10)]))
    wrong = Candidate("deliberately_wrong", lambda data, k: hfd(data, k) + 1e-10)
    with pytest.raises(AssertionError, match="fidelity failures"):
        validate(tmp_path, [wrong])
    assert '"deliberately_wrong"' in (tmp_path / "json" / "validation.json").read_text()


def test_profile_preserves_exception_and_writes_evidence(tmp_path):
    """Default-on profiling records failing work without consuming its error."""
    path = tmp_path / "run.html"
    with pytest.raises(RuntimeError, match="expected test failure"), profiled(path):
        sum(range(100_000))
        raise RuntimeError("expected test failure")
    assert path.stat().st_size > 0
    assert path.with_suffix(".txt").stat().st_size > 0


def test_no_profile_creates_no_report(tmp_path):
    """The explicit opt-out runs work without generating profiling products."""
    with profiled(None):
        result = 2 + 2
    assert result == 4
    assert list(tmp_path.iterdir()) == []


def test_import_does_not_start_profiling(monkeypatch):
    """Importing the experiment entry must never run its workload or profiler."""
    from pyinstrument import Profiler

    def forbidden_start(*args, **kwargs):
        raise AssertionError("profiling started during import")

    monkeypatch.setattr(Profiler, "start", forbidden_start)
    importlib.import_module("experiments.exp_000_hfd_speed.run")
