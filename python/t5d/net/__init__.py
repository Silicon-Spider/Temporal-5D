"""The network stack: input contract, synthetic states, the dense v1 network and its heads.

Stage N1 of the untrained build (spec section 15.1, Phase 2): the dense v1 network with
random weights. Stage N2: the attention structure of section 6 - the move-graph mask
(B9, on), the routing table, typed edge biases and board-pair smolgen (B10-B12, flags) -
and B14's attention paths, block-sparse (the default) and chunked, which build masks and
biases one query block at a time - ``structure.py``, ``smolgen.py``, ``attention.py``;
B14's FlexAttention score function runs eagerly on CPU (``flex.py``: forward only, small
inputs; compiled kernels need a GPU). CHECKPOINT 6: B15's 4D displacement bias, the v1
default (``displacement.py``), superseding B6's binned biases; B16's score-time status key
biases on frozen boards, with the cached configuration (``status.py``); B17's key-side Shaw
term on top of B15, behind its flag (``shaw.py``). Stage N3: the
block-causal scope (C17, C19 - ``scope.py``, ``NetConfig.attn_scope``), the lineage it rests
on (``PositionState.parent`` / ``creation``), the cached evaluation path with supersession
recompute, prefix sharing, refresh epochs and FP8 storage (C2, C3, C3a rung 1, C15, C20 -
``cache.py``), the refresh ladder's complexity trigger (C24, prototype - ``complexity.py``),
a search-tree fixture (``tree.py``), and section 15.3's cost model and
benchmark (``cost.py``, ``cache_bench.py``). See ``config.py`` for which register rows are
built here and which are flags for later stages. Register A7: the input is read from the acting
player's side (``state.py``; ``mirror`` is a position's colour-swapped mirror).
"""

from .cache import CacheConfig, CachedEvaluator, Epoch, KVStore, reference_output
from .complexity import ComplexitySignals, complexity_score
from .config import F1, F2_LARGER_ARM, PRESETS, NetConfig
from .model import NetOutput, T5DNet, build_model, parameter_breakdown
from .scope import BlockCausalScope, board_scope
from .state import (
    PIECE_IDS,
    PIECE_LETTERS,
    RELATION_IDS,
    RELATION_TYPES,
    Batch,
    PositionState,
    TokenLayout,
    collate,
    mirror,
    permute_boards,
    relayout,
    timeline_index,
    translate,
)
from .synthetic import synthetic_state, synthetic_states
from .tree import Multiverse, SearchTree

__all__ = [
    "BlockCausalScope",
    "CacheConfig",
    "CachedEvaluator",
    "ComplexitySignals",
    "complexity_score",
    "Epoch",
    "KVStore",
    "Multiverse",
    "SearchTree",
    "board_scope",
    "reference_output",
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
    "mirror",
    "permute_boards",
    "relayout",
    "timeline_index",
    "translate",
    "synthetic_state",
    "synthetic_states",
]
