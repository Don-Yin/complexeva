"""Lazy profiling of the actual measured workload, including failing runs."""

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def profiled(output: Path | None):
    """Write HTML and readable text without starting a profiler at import time."""
    if output is None:
        yield
        return
    from pyinstrument import Profiler

    profiler = Profiler(interval=0.001)
    profiler.start()
    try:
        yield
    finally:
        profiler.stop()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(profiler.output_html())
        output.with_suffix(".txt").write_text(profiler.output_text(unicode=False, color=False, show_all=False))
