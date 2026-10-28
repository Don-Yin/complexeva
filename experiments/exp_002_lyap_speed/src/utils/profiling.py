"""Lazy profiling with reports retained even when the workload fails."""

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def profiled(output: Path | None):
    """Write HTML, text and JSON profiles unless explicitly disabled."""
    if output is None:
        yield
        return
    from pyinstrument import Profiler
    from pyinstrument.renderers import JSONRenderer

    profiler = Profiler(interval=0.001)
    profiler.start()
    try:
        yield
    finally:
        profiler.stop()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(profiler.output_html())
        output.with_suffix(".txt").write_text(profiler.output_text(unicode=False, color=False, show_all=True))
        output.with_suffix(".json").write_text(profiler.output(JSONRenderer()))
