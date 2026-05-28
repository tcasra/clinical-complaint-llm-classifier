"""Backward-compatible module alias for the refactored workflow runner."""

import sys

from src import train as _train


if __name__ == "__main__":
    raise SystemExit(_train.main())

sys.modules[__name__] = _train
