#!/usr/bin/env python3
"""Entry point of the oracle harness: `python oracles/harness.py <command>`.

Needs Python 3.12+ and nothing else: the harness has no third-party Python
dependencies and is not installed into the engine's environment.
"""

import sys
from pathlib import Path

if sys.version_info < (3, 12):  # noqa: UP036 - the point is a clear message on an older interpreter
    sys.exit("the oracle harness needs Python 3.12 or newer")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_harness.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
