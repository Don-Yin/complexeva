"""Serialize timed work with the shared checkout's speed workers."""

import logging
import signal
from contextlib import contextmanager
from pathlib import Path
from time import sleep

logger = logging.getLogger(__name__)


def terminate(signum, _frame):
    """Unwind the lock context on SIGTERM, just as an EXIT shell trap would."""
    raise SystemExit(128 + signum)


@contextmanager
def benchmark_lock():
    """Atomically acquire the agreed directory lock and release it on failure."""
    lock = Path("/tmp/complexeva-bench.lock")
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            logger.info("waiting for benchmark lock %s", lock)
            sleep(5)
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
        lock.rmdir()
