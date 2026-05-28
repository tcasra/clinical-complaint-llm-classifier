"""Backward-compatible module alias for the refactored evaluation engine."""

import sys

from src import evaluate as _evaluate


if __name__ == "__main__":
    raise SystemExit(_evaluate.main())

sys.modules[__name__] = _evaluate
