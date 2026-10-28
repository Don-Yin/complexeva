"""Resolve the experiment's sole output root and source checkout."""

import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[2]
ROOT = EXPERIMENT.parents[1]
RESULTS = EXPERIMENT / ".results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
