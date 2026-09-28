"""Download and verify the model weights. Thin wrapper around `ampx.weights`.

The implementation lives in the package rather than here because `uv` installs
only `src/ampx`: `ampx.generate` has to call it on a clean clone, and it could
not import this file.

    uv run python scripts/fetch_weights.py
"""

from __future__ import annotations

import sys

from ampx.weights import main

if __name__ == "__main__":
    sys.exit(main())
