"""The rules engine from Python: a ctypes binding to its C API, and the network's input.

Stage N4 of the untrained build (spec §15.1, Phase 2-3): real engine positions become the
network's ``PositionState`` (``python/t5d/net/state.py``), and the network serves the
search's external-evaluator protocol.

* ``capi.py`` - the C API's declarations; finding and loading the shared library
  (an explicit path, ``T5D_ENGINE_LIB``, or ``build/engine/`` under the repository root).
* ``errors.py`` - one exception per C API status, carrying the engine's message.
* ``game.py`` - ``Game``: a game from a 5dpgn record, 5DFEN boards or a variant name;
  replay, apply, submit; the engine's move list; ``expand``.
* ``expansion.py`` - ``Expansion``: one expansion copied out of the engine (its arrays
  outlive the handle's next call), and ``Move``.
* ``convert.py`` - ``to_position``: an expansion as a validated ``PositionState`` with its
  candidate moves in the engine's order, the lineage's completeness and the W/E scope.
* ``evaluator.py`` - the evaluator side of the search protocol: positions in, per-move
  policy logits and 51-bin value distributions out, batched through ``collate`` and the
  model (dense or cached); the C step API's adapter point.
* ``bench.py`` - conversion and forward-pass timings on real positions.

The engine's rules are its own: nothing here computes a rule, and nothing here reads the
oracle harness. The library is built by ``py engine/build.py`` (engine README).
"""

from .capi import ENV_VAR, Library, default_build_dir, find_library, library_available, load_library
from .convert import EngineScope, Position, PresentMismatch, relations_for, to_position, to_position_state
from .errors import (
    EngineArgumentError,
    EngineError,
    EngineInternalError,
    EngineMemoryError,
    EngineNotFoundError,
    IllegalError,
    NotationError,
    UnsupportedError,
)
from .evaluator import (
    CachedNetworkEvaluator,
    CStepDriver,
    Evaluation,
    Evaluator,
    Leaf,
    NetworkEvaluator,
    SearchDriver,
    ServeStats,
    check_evaluation,
    serve,
)
from .expansion import Expansion, Move
from .game import Game, GameInfo, RecordProgress, StateSnapshot

__all__ = [
    "CStepDriver",
    "CachedNetworkEvaluator",
    "ENV_VAR",
    "Evaluation",
    "Evaluator",
    "Leaf",
    "NetworkEvaluator",
    "SearchDriver",
    "ServeStats",
    "check_evaluation",
    "serve",
    "EngineArgumentError",
    "EngineError",
    "EngineInternalError",
    "EngineMemoryError",
    "EngineNotFoundError",
    "EngineScope",
    "Expansion",
    "Game",
    "GameInfo",
    "IllegalError",
    "Library",
    "Move",
    "NotationError",
    "Position",
    "PresentMismatch",
    "RecordProgress",
    "StateSnapshot",
    "UnsupportedError",
    "default_build_dir",
    "find_library",
    "library_available",
    "load_library",
    "relations_for",
    "to_position",
    "to_position_state",
]
