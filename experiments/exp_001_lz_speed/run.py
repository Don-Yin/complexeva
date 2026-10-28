"""Profile and benchmark exact Lempel-Ziv component counts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modules.benchmark import run
from src.utils.profiling import profiled

RESULTS = Path(__file__).resolve().parent / ".results"


def parse_args() -> argparse.Namespace:
    """Parse workload size and the explicit profiling opt-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000])
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidates-only", action="store_true", help="skip production timing, save a separate comparison")
    parser.add_argument("--no-profile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = RESULTS / "profile" / ("candidates.html" if args.candidates_only else "run.html")
    with profiled(None if args.no_profile else report):
        run(args, RESULTS)
