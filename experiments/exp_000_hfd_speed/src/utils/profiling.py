"""Default-on profiling of the real experiment workload."""

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def profiled(out: Path | None):
    """Save HTML and text reports, including when the workload raises."""
    if out is None:
        yield
        return
    from pyinstrument import Profiler

    # All work is synchronous; allow per-size sessions inside the whole-run one.
    profiler = Profiler(interval=0.001, async_mode="disabled")
    profiler.start()
    try:
        yield
    finally:
        profiler.stop()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(profiler.output_html())
        out.with_suffix(".txt").write_text(profiler.output_text(unicode=False, color=False, show_all=True))
