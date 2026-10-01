"""The network stack: input contract, synthetic states, the dense v1 network and its heads.

Stage N1 of the untrained build (spec section 15.1, Phase 2): the dense v1 network with
random weights. Stage N2: the attention structure of section 6 - the move-graph mask
(B9, on), the routing table, typed edge biases and board-pair smolgen (B10-B12, flags) -
and B14's attention paths, block-sparse (the default) and chunked, which build masks and
biases one query block at a time - ``structure.py``, ``smolgen.py``, ``attention.py``
(B14's FlexAttention ``score_mod`` is not built: no compiler or GPU here). Stage N3: the
cached path. See ``config.py`` for which register rows are built here and which are
flags for later stages.
"""

from .config import F1, F2_LARGER_ARM, PRESETS, NetConfig
from .model import NetOutput, T5DNet, build_model, parameter_breakdown
from .state import (
    PIECE_IDS,
    PIECE_LETTERS,
    RELATION_IDS,
    RELATION_TYPES,
    Batch,
    PositionState,
    TokenLayout,
    collate,
    permute_boards,
    relayout,
    timeline_index,
    translate,
)
from .synthetic import synthetic_state, synthetic_states

__all__ = [
    "F1",
    "F2_LARGER_ARM",
    "PRESETS",
    "NetConfig",
    "NetOutput",
    "T5DNet",
    "build_model",
    "parameter_breakdown",
    "PIECE_IDS",
    "PIECE_LETTERS",
    "RELATION_IDS",
    "RELATION_TYPES",
    "Batch",
    "PositionState",
    "TokenLayout",
    "collate",
    "permute_boards",
    "relayout",
    "timeline_index",
    "translate",
    "synthetic_state",
    "synthetic_states",
]
