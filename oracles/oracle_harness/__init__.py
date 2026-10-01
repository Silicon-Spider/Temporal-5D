"""Differential-testing harness over the pinned rules oracles.

This package is test infrastructure. Nothing outside ``oracles/`` may import it,
link against it, or reference the third-party code it drives; ``boundary.py``
enforces that on every commit. The engine reaches the gate through a process
boundary (see ``adapters/engine.py``), never through an import.
"""

__version__ = "1"
