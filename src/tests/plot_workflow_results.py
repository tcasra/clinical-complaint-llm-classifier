"""Backward-compatible module alias for the refactored figure generator."""

import sys

from src import visualize as _visualize


if __name__ == "__main__":
    raise SystemExit(_visualize.main())

sys.modules[__name__] = _visualize
