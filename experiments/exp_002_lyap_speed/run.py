"""Profile and benchmark equivalent Lyapunov nearest-neighbour searches."""

from __future__ import annotations

import argparse
import logging

# Bootstrap the checkout, then use one canonical module name in both CLI and tests.
from src.utils.paths import RESULTS

from experiments.exp_002_lyap_speed.src.modules.benchmark import baseline, prepare, stages
from experiments.exp_002_lyap_speed.src.modules.comparison import compare, warm_candidates
from experiments.exp_002_lyap_speed.src.utils.profiling import profiled


def parse_args() -> argparse.Namespace:
    """Parse explicit workload sizes and the sole profiling opt-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["baseline", "stages", "compare"], default="compare")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 1000, 2000, 5000, 10000, 20000])
    parser.add_argument("--no-profile", action="store_true", help="disable pyinstrument reports")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Warm JIT compilation, then profile actual measured package calls."""
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        raise ValueError(f"positive repeats and sizes required: repeats={args.repeats}, sizes={args.sizes}")
    metadata = prepare()
    if args.phase == "compare":
        metadata["candidate_warmup_seconds"] = warm_candidates()
    with profiled(None if args.no_profile else RESULTS / "profile" / "run.html"):
        if args.phase == "compare":
            compare(args.sizes, metadata, args.repeats)
        else:
            {"baseline": baseline, "stages": stages}[args.phase](args.sizes, metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main(parse_args())
