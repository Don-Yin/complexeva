"""Profile and benchmark Higuchi implementations without changing complexeva."""

import argparse
import logging
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS = EXPERIMENT_DIR / ".results"
sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))

from experiments.exp_000_hfd_speed.src.modules.baseline import baseline
from experiments.exp_000_hfd_speed.src.modules.benchmark import benchmark
from experiments.exp_000_hfd_speed.src.modules.candidates import candidates
from experiments.exp_000_hfd_speed.src.modules.validation import validate
from experiments.exp_000_hfd_speed.src.utils.profiling import profiled


def parse_args():
    """Select the experiment stage and the explicit profiling opt-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--stage", choices=("baseline", "benchmark", "validate", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--target-seconds", type=float, default=0.025)
    return parser.parse_args()


def main(args):
    """Run selected stages and preserve evidence under this experiment."""
    if args.stage in ("baseline", "all"):
        baseline(RESULTS, enabled=not args.no_profile)
    if args.stage == "validate":
        validate(RESULTS, candidates())
    if args.stage in ("benchmark", "all"):
        benchmark(RESULTS, profiled_run=not args.no_profile, repeats=args.repeats, target=args.target_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    with profiled(None if args.no_profile else RESULTS / "profile" / "run.html"):
        main(args)
