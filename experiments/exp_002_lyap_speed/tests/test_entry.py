"""Verify import safety and nonzero exits without changing package state."""

import subprocess
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
PROFILE = EXPERIMENT / ".results" / "profile" / "run.html"


def test_import_does_not_start_profiling():
    """Importing the entry must leave its profile untouched."""
    before = PROFILE.stat().st_mtime_ns if PROFILE.exists() else None
    subprocess.run(
        [sys.executable, "-c", "import runpy; runpy.run_path('run.py', run_name='inspect_only')"],
        cwd=EXPERIMENT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = PROFILE.stat().st_mtime_ns if PROFILE.exists() else None
    assert before == after


def test_failure_propagates_and_still_saves_profile():
    """A real invalid workload must fail while retaining its default HTML profile."""
    result = subprocess.run(
        [sys.executable, "run.py", "--phase", "baseline", "--sizes", "20"],
        cwd=EXPERIMENT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "series too short" in result.stderr
    assert PROFILE.is_file() and PROFILE.stat().st_size > 0
