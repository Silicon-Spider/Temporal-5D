"""The ctypes layer over the engine's C API (``engine/include/t5d/capi.h``, ``records.h`` and
``capi_search.h`` - the main search's step API, engine stage E6a).

This module mirrors the C declarations exactly - structures, constants, signatures - and
finds and loads the shared library. Nothing here owns engine memory; ``game.py`` and
``search.py`` wrap the handles, and ``expansion.py`` copies an expansion's arrays out before
the next call.

**Finding the library** (``find_library``), first match wins:

1. an explicit path (``load_library(path)``), which must exist;
2. the environment variable ``T5D_ENGINE_LIB`` (a file, or a directory holding it),
   which must exist when set;
3. ``build/engine/`` under the repository root (where ``py engine/build.py`` puts it):
   ``t5d.dll`` on Windows, ``libt5d.so`` on Linux, ``libt5d.dylib`` on macOS.

A missing library raises ``EngineNotFoundError`` naming every place looked and how to build
it. On Windows the DLL links the MinGW runtime statically and imports only KERNEL32 and the
Universal CRT (engine README, "Building and testing"), so it loads without MSYS2 on PATH.

**Layouts.** Each ``Structure`` below is the C struct field for field, in order, so ctypes
lays it out as the C compiler does (natural alignment, no packing); ``_check_layouts``
asserts the sizes the headers fix (``t5d_move`` is 32 bytes, no padding). ``MOVE_DTYPE`` is
the same record as a numpy structured dtype, for copying a move array in one step. At load,
``Library`` checks every mirrored struct's size and every field's offset against the
library's own - ``t5d/capi.h``'s against ``t5d_layout``, ``t5d/capi_search.h``'s against
``t5d_search_layout``: a binding and a library of different stages refuse each other
(``EngineNotFoundError``) rather than let the library write past a Python buffer. A library
without the search's step API is refused the same way (it lacks functions).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from ctypes import POINTER, c_char, c_char_p, c_double, c_float, c_int8, c_int32, c_size_t, c_uint8, c_uint32, c_uint64, c_void_p
from pathlib import Path

import numpy as np

from .errors import T5D_OK, EngineNotFoundError, error_for

#: The environment variable that names the library (a file or its directory).
ENV_VAR = "T5D_ENGINE_LIB"

# ---- constants (t5d/capi.h, t5d/records.h) ------------------------------------------------

T5D_PARITY_INFER, T5D_PARITY_ODD, T5D_PARITY_EVEN = -1, 0, 1
T5D_TO_MOVE_INFER, T5D_TO_MOVE_WHITE, T5D_TO_MOVE_BLACK = -1, 0, 1
T5D_RULES_STANDARD = 0
#: Q4's Guide-literal reading - no move once the present has shifted - a diagnostic. Bit 3
#: meant the opposite, under the name ``T5D_RULES_MOVES_AFTER_SHIFT``, until engine stage E7
#: made moves after the shift the default (rulebook R8, register D8).
T5D_RULES_NO_MOVES_AFTER_SHIFT = 1 << 3
#: Moves after the shift, until submission: the default since R8, so no bit (kept so that a
#: caller that asked for them still gets them).
T5D_RULES_MOVES_AFTER_SHIFT = 0
T5D_FEATURES_ALL, T5D_FEATURES_SCOPE, T5D_FEATURES_NONE = 0, 1, 2
T5D_PSEUDO_LEGAL, T5D_LEGAL = 0, 1
T5D_NO_WINDOW = 2**31 - 1  # INT32_MAX

# Move flags (t5d_move.flags).
T5D_MOVE_CAPTURE = 1
T5D_MOVE_BRANCHING = 2
T5D_MOVE_CASTLE = 4
T5D_MOVE_EN_PASSANT = 8
T5D_MOVE_DOUBLE_STEP = 16
T5D_MOVE_PROMOTION = 32
T5D_MOVE_TRAVEL = 64

# Relation ids: python/t5d/net/state.py RELATION_TYPES, in its order (records.h says so;
# tests/engine checks the two agree).
T5D_RELATION_MOVE = 0
T5D_RELATION_ATTACK = 1
T5D_RELATION_DEFENCE = 2
T5D_RELATION_PIN = 3  # reserved, not built
T5D_RELATION_XRAY = 4
T5D_RELATION_CHECK = 5  # reserved, not built
T5D_RELATION_TRAVEL_THREAT = 6  # reserved, not built

#: The relations the engine builds, by the network's names (``t5d_expand_options.relations``).
BUILT_RELATIONS: dict[str, int] = {"attack": T5D_RELATION_ATTACK, "defence": T5D_RELATION_DEFENCE, "xray": T5D_RELATION_XRAY}
#: Named by the network's contract but refused by the engine (T5D_ERR_UNSUPPORTED): pin
#: and check would read check detection; travel-threat has no edge defined by the spec.
RESERVED_RELATIONS: dict[str, int] = {"pin": T5D_RELATION_PIN, "check": T5D_RELATION_CHECK, "travel_threat": T5D_RELATION_TRAVEL_THREAT}


# ---- structures -----------------------------------------------------------------------------


class Fingerprint(ctypes.Structure):
    _fields_ = [("fingerprint", c_char * 33), ("placement", c_char * 33), ("boards", c_uint64), ("pieces", c_uint64)]


class StateInfo(ctypes.Structure):
    _fields_ = [
        ("boards", c_uint64),
        ("pieces", c_uint64),
        ("width", c_int32),
        ("height", c_int32),
        ("even", c_int32),
        ("to_move", c_int32),
        ("min_timeline", c_int32),
        ("max_timeline", c_int32),
        ("created_by_white", c_int32),
        ("created_by_black", c_int32),
        ("ruleset", c_uint32),
    ]


class MoveRecord(ctypes.Structure):
    """``t5d_move`` (records.h): 32 bytes, no padding."""

    _fields_ = [
        ("from_timeline", c_int32),
        ("from_half_turn", c_int32),
        ("to_timeline", c_int32),
        ("to_half_turn", c_int32),
        ("from_board", c_int32),
        ("to_board", c_int32),
        ("from_square", c_int8),
        ("to_square", c_int8),
        ("from_cell", c_int8),
        ("to_cell", c_int8),
        ("piece", c_uint8),
        ("captured", c_uint8),
        ("promotion", c_uint8),
        ("flags", c_uint8),
    ]


#: ``t5d_move`` as a numpy record (same field names, same offsets, 32 bytes).
MOVE_DTYPE = np.dtype(
    [
        ("from_timeline", "<i4"),
        ("from_half_turn", "<i4"),
        ("to_timeline", "<i4"),
        ("to_half_turn", "<i4"),
        ("from_board", "<i4"),
        ("to_board", "<i4"),
        ("from_square", "i1"),
        ("to_square", "i1"),
        ("from_cell", "i1"),
        ("to_cell", "i1"),
        ("piece", "u1"),
        ("captured", "u1"),
        ("promotion", "u1"),
        ("flags", "u1"),
    ]
)


class ExpandOptions(ctypes.Structure):
    _fields_ = [
        ("window_k", c_int32),
        ("square_features", c_int32),
        ("relations", c_uint32),
        # appended in engine ABI round 2 (register I15): the search leaves' legal-moves filter for
        # t5d_game_expand (a search refuses it in its own expand options)
        ("legal_moves", c_int32),  # 1: the legal moves, classified under the budget below; 0: pseudo-legal
        ("legality_checks", c_uint64),  # the budget, as t5d_search_params' (T5D_NO_LIMIT both: exact)
        ("legality_nodes", c_uint64),
    ]


class GameSummary(ctypes.Structure):
    _fields_ = [
        ("boards", c_uint64),
        ("width", c_int32),
        ("height", c_int32),
        ("even", c_int32),
        ("to_move", c_int32),
        ("present", c_int32),
        ("timelines", c_int32),
        ("active_timelines", c_int32),
        ("obligations_met", c_int32),
        ("may_move", c_int32),
        ("may_submit", c_int32),
        ("ruleset", c_uint32),
        ("rules", c_uint64),
        # appended in engine ABI round 1 (register A6): the handle's count - no state records it
        ("moves_made", c_int32),  # moves made in the action in progress: 0 at a turn start
    ]


class CExpansion(ctypes.Structure):
    """``t5d_expansion``: counts, and pointers into the game handle's buffers - valid only
    until the handle is next changed (``expansion.py`` copies them out at once)."""

    _fields_ = [
        ("width", c_int32),
        ("height", c_int32),
        ("even", c_int32),
        ("to_move", c_int32),
        ("present", c_int32),
        ("timelines", c_int32),
        ("active_timelines", c_int32),
        ("window_k", c_int32),
        ("may_move", c_int32),
        ("origins_unknown", c_int32),
        ("board_count", c_size_t),
        ("move_count", c_size_t),
        ("edge_count", c_size_t),
        ("relation_count", c_size_t),
        ("window_count", c_size_t),
        ("e_count", c_size_t),
        ("src_extra_count", c_size_t),
        ("feature_board_count", c_size_t),
        ("origin_count", c_size_t),
        # per board
        ("timeline", POINTER(c_int32)),
        ("half_turn", POINTER(c_int32)),
        ("is_head", POINTER(c_uint8)),
        ("active", POINTER(c_uint8)),
        ("side_to_move", POINTER(c_uint8)),
        ("playable", POINTER(c_uint8)),
        ("window_depth", POINTER(c_int32)),
        ("band_depth", POINTER(c_int32)),
        ("is_origin", POINTER(c_uint8)),
        ("parent", POINTER(c_int32)),
        # per square
        ("feature_boards", POINTER(c_int32)),
        ("piece", POINTER(c_uint8)),
        ("colour", POINTER(c_uint8)),
        ("moved", POINTER(c_uint8)),
        # lists
        ("moves", POINTER(MoveRecord)),
        ("edges", POINTER(c_int32)),
        ("relations", POINTER(c_int32)),
        ("window", POINTER(c_int32)),
        ("e_boards", POINTER(c_int32)),
        ("src_extra", POINTER(c_int32)),
        # appended at engine stage E7 (register D8): the action - who acts, and the submit action
        ("acting_player", c_int32),  # 0 white, 1 black: whose action this is (= to_move)
        ("submit_possible", c_int32),  # 1: the action's obligations are met (T5)
        ("submit_legal", c_int32),  # 1: ... and no royal of the acting player's capturable (C4, R1)
        # appended in engine ABI round 1 (register A6, spec 2.3): the turn state, exact counts
        ("mandatory_remaining", c_int32),  # boards the acting player must still play (T5, T6): 0 iff submit_possible
        ("optional_playable", c_int32),  # its other playable boards (L5); 0 when no move may be made (Q4)
        ("moves_made", c_int32),  # moves made in the action in progress (the handle's, a leaf's path's): 0 at a turn start
    ]


# ---- the search's step API (t5d/capi_search.h; engine stage E6a) ------------------------------

T5D_VALUE_BINS = 51
#: D3: the WDL head's outputs per leaf - win, draw, loss.
T5D_WDL = 3
#: The uncertainty signal search consumes (register I10; ablation 6m).
T5D_UNCERTAINTY_CATEGORICAL, T5D_UNCERTAINTY_ERROR_HEAD = 0, 1
T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_WHOLE_TURN, T5D_SPLIT_TREE = 0, 1, 2
#: D2's tree form: its interaction gates (spec 8.2), a mask - ALWAYS is a test seam (every
#: mid-turn node evaluated: spec 15.2's "every interaction gate firing", the ply split exactly).
T5D_TREE_GATE_BETWEEN, T5D_TREE_GATE_CAPTURE, T5D_TREE_GATE_CHECK, T5D_TREE_GATE_PRESENT = 1, 2, 4, 8
T5D_TREE_GATES_INTERACTION, T5D_TREE_GATE_ALWAYS = 15, 16
#: ... and its readings (placeholders): the boards still to move; a piece that could reach them;
#: a change in check; a shift of the present.
T5D_STILL_TO_MOVE_PLAYABLE, T5D_STILL_TO_MOVE_MANDATORY = 0, 1
T5D_CAPTURE_REACH, T5D_CAPTURE_ANY = 0, 1
T5D_CHECK_PASSED, T5D_CHECK_SUBMISSION = 0, 1
T5D_PRESENT_ANY, T5D_PRESENT_REWIND = 0, 1
T5D_ACTION_PRIOR_SUM, T5D_ACTION_PRIOR_MEAN = 0, 1
T5D_NODE_KEY_STATE, T5D_NODE_KEY_STATE_AND_ORIGINS = 0, 1
T5D_FINAL_VISITS, T5D_FINAL_LCB = 0, 1
T5D_VALUE_PROBABILITIES, T5D_VALUE_LOGITS = 0, 1
T5D_EVALUATOR_UNIFORM, T5D_EVALUATOR_RANDOM, T5D_EVALUATOR_MATERIAL = 0, 1, 2
T5D_LEGALITY_LEGAL, T5D_LEGALITY_ILLEGAL, T5D_LEGALITY_UNKNOWN = 0, 1, 2
T5D_CHOICE_PROVEN_WIN, T5D_CHOICE_SECURE_CHILD, T5D_CHOICE_LCB, T5D_CHOICE_VISITS, T5D_CHOICE_ALL_LOST = 0, 1, 2, 3, 4
T5D_STOP_PLAYOUTS, T5D_STOP_EVALUATIONS, T5D_STOP_FINISHED, T5D_STOP_EXHAUSTED = 0, 1, 2, 3
T5D_STOP_NODES, T5D_STOP_MEMORY, T5D_STOP_DEADLINE = 4, 5, 6
#: No limit, for the search's budgets (UINT64_MAX).
T5D_NO_LIMIT = 2**64 - 1


class CSearchParams(ctypes.Structure):
    """``t5d_search_params`` (``SearchParams``, field for field): fill with
    ``t5d_search_params_default``. ``expand`` - a struct of ``t5d/capi.h`` - is last."""

    _fields_ = [
        ("split", c_int32),
        ("action_prior", c_int32),
        ("max_actions", c_uint64),
        ("listing_nodes", c_uint64),
        ("submit_share", c_double),
        ("legal_moves_only", c_int32),
        ("legality_checks", c_uint64),
        ("legality_nodes", c_uint64),
        ("unknown_prior_scale", c_double),
        ("verify_unknown", c_int32),
        ("verify_checks", c_uint64),
        ("verify_nodes", c_uint64),
        ("cpuct", c_double),
        ("widening", c_int32),
        ("widening_alpha", c_double),
        ("widening_scale", c_double),
        ("widening_min", c_double),
        ("fpu_reduction", c_double),
        ("root_noise", c_int32),
        ("dirichlet_alpha", c_double),
        ("dirichlet_epsilon", c_double),
        ("policy_temperature", c_double),
        ("variance_scaled_cpuct", c_int32),
        ("cpuct_variance_scale", c_double),
        ("variance_reference", c_double),
        ("variance_prior_weight", c_double),
        ("cpuct_scale_min", c_double),
        ("cpuct_scale_max", c_double),
        ("uncertainty_weighted_backup", c_int32),
        ("variance_bonus", c_int32),
        ("variance_bonus_scale", c_double),
        ("variance_bonus_cap", c_double),
        ("graph", c_int32),
        ("node_key", c_int32),
        ("residual_epsilon", c_double),
        ("proof_channel", c_int32),  # H41: off by rule until the gate passes and the channel is signed off
        ("solved_visit_cap", c_uint32),
        ("final_rule", c_int32),
        ("lcb_floor", c_double),
        ("lcb_z", c_double),
        ("lcb_categorical_weight", c_double),
        ("secure_child_a", c_double),
        ("batch_size", c_uint32),
        ("virtual_loss", c_double),
        ("synchronous", c_int32),
        ("seed", c_uint64),
        ("mate_nodes", c_uint64),
        ("mate_bytes", c_uint64),
        ("turn_cap_turns", c_int32),
        ("turn_cap_boards", c_int32),
        ("eval_cache_entries", c_uint64),
        ("eval_cache_bytes", c_uint64),
        # engine ABI round 2 (register D3, I10), in front of the embedded struct
        ("uncertainty_source", c_int32),  # T5D_UNCERTAINTY_*: a node's variance for E2, E3 and E5's term (6m)
        # engine ABI round 2b (register E1, I19): placeholders
        ("dirichlet_concentration", c_double),  # the root noise's total, split across the open edges; 0: dirichlet_alpha each
        ("dirichlet_shaping", c_int32),  # KataGo-style shaping: a prototype, off
        ("dirichlet_shaping_share", c_double),
        ("dirichlet_shaping_cap", c_double),
        ("root_policy_temperature", c_double),  # the root's softmax temperature, before the noise (1: off)
        # D2's tree form (split T5D_SPLIT_TREE only; refused with graph - spec issue 301): placeholders
        ("tree_gates", c_uint32),  # T5D_TREE_GATE_* mask; 0: the network only at turn starts and the root
        ("tree_still_to_move", c_int32),  # T5D_STILL_TO_MOVE_*
        ("tree_capture", c_int32),  # T5D_CAPTURE_*
        ("tree_check", c_int32),  # T5D_CHECK_*
        ("tree_inherit_value", c_int32),  # 1: an inherited node's FPU base and variance are its parent's evaluation's
        ("tree_present", c_int32),  # T5D_PRESENT_*: a shift of the present - any move of it, or a rewind only
        ("expand", ExpandOptions),
    ]


class CLeaf(ctypes.Structure):
    """``t5d_leaf``: one leaf of a gathered batch - its pointers (``legality``, the expansion's
    arrays) valid until the next gather, deliver, abandon or destroy on the search."""

    _fields_ = [
        ("key", c_uint64),
        ("epoch", c_uint64),
        ("acting_player", c_int32),  # D8: whose action, until it submits - the value's perspective
        ("submit_possible", c_int32),  # D8: submitting is an action here (board-move split)
        ("submit_legal", c_int32),  # ... and legal (C4, exact)
        ("legality", POINTER(c_uint8)),  # per move: T5D_LEGALITY_LEGAL or _UNKNOWN; NULL: none made
        ("expansion", CExpansion),  # logit i scores expansion.moves[i]; last
    ]


class CSearchSummary(ctypes.Structure):
    """``t5d_search_summary``."""

    _fields_ = [
        ("playouts", c_uint64),
        ("evaluations", c_uint64),
        ("cache_hits", c_uint64),
        ("nodes", c_uint64),
        ("nodes_created", c_uint64),
        ("transposition_hits", c_uint64),
        ("checkmates", c_uint64),
        ("stalemates", c_uint64),
        ("turn_caps", c_uint64),
        ("unknown_verdicts", c_uint64),
        ("unknown_edges", c_uint64),
        ("verified_illegal", c_uint64),
        ("root_turn_start", c_int32),
        ("root_terminal", c_int32),
        ("root_solved", c_int32),
        ("root_lo", c_int32),
        ("root_hi", c_int32),
        ("root_visits", c_uint32),
        ("root_q", c_double),
        ("root_children", c_uint64),
        ("epoch", c_uint64),
        ("memory_bytes", c_uint64),
        ("cache_entries", c_uint64),
        ("cache_bytes", c_uint64),
        ("cache_purged", c_uint64),
        ("last_stop", c_int32),
        # appended in engine ABI round 2 (register D3): the root's backed-up draw probability and
        # its own evaluation - the network's raw outputs at the root (I15); NaN: not given
        ("root_evaluated", c_int32),
        ("root_d", c_double),
        ("root_wdl", c_double * 3),  # win, draw, loss - the root player's view
        ("root_moves_left", c_double),
        ("root_error", c_double),
        # appended in engine ABI round 2c (register I15, I2, D4): the root's node key, its effective
        # cPUCT now (cpuct x E2's root scale), the effort label (the root's visits when its most
        # visited child last changed)
        ("root_key", c_uint64),
        ("root_cpuct", c_double),
        ("root_best_since", c_uint32),
        # appended in the engine ABI round 2 review fixes (register I10, 6m): nodes expanded without
        # the selected uncertainty signal - only while nothing reads it (else the delivery is refused)
        ("uncertainty_missing", c_uint64),
        # appended for D2's tree form: mid-turn nodes expanded on inherited priors (no evaluation),
        # and those sent to the network instead, by the first gate found to fire
        ("inherited_nodes", c_uint64),
        ("gate_between", c_uint64),
        ("gate_capture", c_uint64),
        ("gate_check", c_uint64),
        ("gate_present", c_uint64),
        ("gate_always", c_uint64),
    ]


class CSearchLimits(ctypes.Structure):
    """``t5d_search_limits``."""

    _fields_ = [("playouts", c_uint64), ("evaluations", c_uint64), ("nodes", c_uint64), ("bytes", c_uint64), ("seconds", c_double)]


class CRootChild(ctypes.Structure):
    """``t5d_root_child``: a root child's statistics, from the root player's view."""

    _fields_ = [
        ("visits", c_uint32),
        ("submit", c_int32),
        ("legality", c_int32),
        ("lo", c_int32),
        ("hi", c_int32),
        ("prior", c_double),
        ("q", c_double),  # the backed-up value, W - L (D3)
        ("move_count", c_size_t),
        # appended in engine ABI round 2 (register D3): the search's draw probability, and the
        # child's own evaluation - the network's (I13's and I15's memo), the root player's view
        ("d", c_double),  # backed up beside q: the search WDL is ((1 - d + q) / 2, d, (1 - d - q) / 2)
        ("evaluated", c_int32),
        ("network_wdl", c_double * 3),
        ("network_moves_left", c_double),  # NaN: none given (or not evaluated)
        ("network_error", c_double),
        # appended in engine ABI round 2c (register I15, I2): the root's side table
        ("child_key", c_uint64),  # the child node's key; 0: no child node yet
        ("prior_noise_free", c_double),  # the prior before the root noise
        ("value_variance", c_double),  # of the values backed up through it
        ("boost", c_double),  # E10's asset-map boost: 0 (not wired)
        ("exploration_scale", c_double),  # E3's factor on its exploration term (1: off)
        ("forced_visits", c_uint32),  # forced playouts: 0 (not built - pruning is target-side)
        ("steered_visits", c_uint32),  # visits a steering source decided: 0 (none wired, E10)
    ]


class CEvaluations(ctypes.Structure):
    """``t5d_evaluations``: one batch's evaluations for ``t5d_search_deliver_evaluations``, leaf
    by leaf - a NULL array with count 0 is "none given" where the field is optional."""

    _fields_ = [
        ("logits", POINTER(c_float)),
        ("logit_count", c_size_t),
        ("submit_logits", POINTER(c_float)),
        ("submit_count", c_size_t),
        ("wdl", POINTER(c_float)),  # T5D_WDL per leaf (required: D3's value)
        ("wdl_count", c_size_t),
        ("wdl_kind", c_int32),  # T5D_VALUE_PROBABILITIES or _LOGITS
        ("value_kind", c_int32),
        ("values", POINTER(c_float)),  # T5D_VALUE_BINS per leaf (the categorical), or NULL
        ("value_count", c_size_t),
        ("errors", POINTER(c_float)),  # one per leaf (>= 0; NaN: none), or NULL
        ("error_count", c_size_t),
        ("moves_left", POINTER(c_float)),  # one per leaf (>= 0; NaN: none), or NULL
        ("moves_left_count", c_size_t),
        ("board_logits", POINTER(c_float)),  # reserved for D2's flat form (not built): NULL
        ("board_logit_count", c_size_t),
    ]


def _check_layouts() -> None:
    if ctypes.sizeof(MoveRecord) != 32 or MOVE_DTYPE.itemsize != 32:
        raise ImportError("t5d_move must be 32 bytes (t5d/records.h)")
    for name, _ in MoveRecord._fields_:
        if getattr(MoveRecord, name).offset != MOVE_DTYPE.fields[name][1]:
            raise ImportError(f"t5d_move.{name}: the ctypes and numpy layouts disagree")
    if ctypes.sizeof(ExpandOptions) != 32:
        raise ImportError("t5d_expand_options must be 32 bytes (four 32-bit fields, two 64-bit)")
    for struct, last in ((CSearchParams, "expand"), (CLeaf, "expansion")):
        if struct._fields_[-1][0] != last:
            raise ImportError(f"{struct.__name__}: the embedded struct of t5d/capi.h must be the last member (t5d/capi_search.h)")


_check_layouts()

#: Every struct of ``t5d/capi.h`` this module mirrors, by its C name (``t5d_layout``'s ``type``).
MIRRORED_STRUCTS: dict[str, type[ctypes.Structure]] = {
    "t5d_fingerprint": Fingerprint,
    "t5d_info": StateInfo,
    "t5d_move": MoveRecord,
    "t5d_expand_options": ExpandOptions,
    "t5d_game_summary": GameSummary,
    "t5d_expansion": CExpansion,
}

#: Every struct of ``t5d/capi_search.h`` this module mirrors (``t5d_search_layout``'s ``type``).
MIRRORED_SEARCH_STRUCTS: dict[str, type[ctypes.Structure]] = {
    "t5d_search_params": CSearchParams,
    "t5d_leaf": CLeaf,
    "t5d_search_summary": CSearchSummary,
    "t5d_search_limits": CSearchLimits,
    "t5d_root_child": CRootChild,
    "t5d_evaluations": CEvaluations,
}


def _mismatches(layout, structs: dict[str, type[ctypes.Structure]]) -> list[str]:
    out: list[str] = []
    value = c_size_t()
    for cname, struct in structs.items():
        ctype = cname.encode("ascii")
        if layout(ctype, None, ctypes.byref(value)) != T5D_OK:
            out.append(f"{cname}: unknown to the library")
            continue
        if value.value != ctypes.sizeof(struct):
            out.append(f"{cname}: size {ctypes.sizeof(struct)} here, {value.value} in the library")
        for name, _ in struct._fields_:
            offset = getattr(struct, name).offset
            if layout(ctype, name.encode("ascii"), ctypes.byref(value)) != T5D_OK:
                out.append(f"{cname}.{name}: not in the library")
            elif value.value != offset:
                out.append(f"{cname}.{name}: offset {offset} here, {value.value} in the library")
    return out


def layout_mismatches(lib: Library) -> list[str]:
    """Where the mirrored structs differ from the library's own - ``t5d/capi.h``'s against
    ``t5d_layout``, ``t5d/capi_search.h``'s against ``t5d_search_layout``: each struct's size
    and each field's offset - ``"t5d_expansion: size 288 here, 272 in the library"``. Empty
    when they agree."""
    return _mismatches(lib.fn.t5d_layout, MIRRORED_STRUCTS) + _mismatches(lib.fn.t5d_search_layout, MIRRORED_SEARCH_STRUCTS)


# ---- the library ------------------------------------------------------------------------------

_STATE = c_void_p  # t5d_state*
_GAME = c_void_p  # t5d_game*
_SEARCH = c_void_p  # t5d_search*
_ERR = POINTER(c_void_p)  # char** error (a library string, freed with t5d_free)

#: name -> (restype, argtypes). Every function of capi.h.
_SIGNATURES: dict[str, tuple[object, list[object]]] = {
    "t5d_version": (c_char_p, []),
    "t5d_free": (None, [c_void_p]),
    "t5d_layout": (c_int32, [c_char_p, c_char_p, POINTER(c_size_t)]),
    "t5d_state_new_standard": (c_int32, [POINTER(_STATE)]),
    "t5d_state_new_standard_two_timelines": (c_int32, [POINTER(_STATE)]),
    "t5d_state_new_from_fen": (
        c_int32,
        [POINTER(c_char_p), c_size_t, c_int32, c_int32, c_int32, c_int32, c_uint32, POINTER(_STATE), _ERR],
    ),
    "t5d_state_copy": (c_int32, [_STATE, POINTER(_STATE)]),
    "t5d_state_destroy": (None, [_STATE]),
    "t5d_state_info": (c_int32, [_STATE, POINTER(StateInfo)]),
    "t5d_state_fingerprint": (c_int32, [_STATE, POINTER(Fingerprint)]),
    "t5d_state_canonical": (c_int32, [_STATE, POINTER(c_void_p)]),
    "t5d_state_zobrist": (c_int32, [_STATE, POINTER(c_uint64)]),
    "t5d_state_board_count": (c_int32, [_STATE, POINTER(c_size_t)]),
    "t5d_state_board": (c_int32, [_STATE, c_size_t, POINTER(c_int32), POINTER(c_int32), POINTER(c_void_p)]),
    "t5d_game_new_from_pgn": (c_int32, [c_char_p, c_size_t, c_uint64, POINTER(_GAME), _ERR]),
    "t5d_game_new_from_fen": (
        c_int32,
        [POINTER(c_char_p), c_size_t, c_int32, c_int32, c_int32, c_int32, c_uint64, POINTER(_GAME), _ERR],
    ),
    "t5d_game_new_variant": (c_int32, [c_char_p, c_uint64, POINTER(_GAME), _ERR]),
    "t5d_game_copy": (c_int32, [_GAME, POINTER(_GAME)]),
    "t5d_game_destroy": (None, [_GAME]),
    "t5d_game_info": (c_int32, [_GAME, POINTER(GameSummary)]),
    "t5d_game_record_actions": (c_int32, [_GAME, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_int32)]),
    "t5d_game_replay_next": (c_int32, [_GAME, _ERR]),
    "t5d_game_apply": (c_int32, [_GAME, POINTER(MoveRecord), _ERR]),
    "t5d_game_apply_expanded": (c_int32, [_GAME, c_size_t, _ERR]),
    "t5d_game_submit": (c_int32, [_GAME, _ERR]),
    "t5d_game_set_legality": (c_int32, [_GAME, c_int32, _ERR]),
    "t5d_game_expand": (c_int32, [_GAME, POINTER(ExpandOptions), POINTER(CExpansion), _ERR]),
    "t5d_game_state": (c_int32, [_GAME, POINTER(_STATE)]),
    # t5d/capi_search.h (engine stage E6a): every function
    "t5d_search_params_default": (None, [POINTER(CSearchParams)]),
    "t5d_search_new": (c_int32, [_GAME, c_int32, POINTER(CSearchParams), POINTER(_SEARCH), _ERR]),
    "t5d_search_new_sharing": (c_int32, [_SEARCH, _GAME, c_int32, POINTER(CSearchParams), POINTER(_SEARCH), _ERR]),
    "t5d_search_destroy": (None, [_SEARCH]),
    "t5d_search_gather": (c_int32, [_SEARCH, c_size_t, c_uint64, POINTER(c_size_t), _ERR]),
    "t5d_search_leaf": (c_int32, [_SEARCH, c_size_t, POINTER(CLeaf), _ERR]),
    "t5d_search_leaf_state": (c_int32, [_SEARCH, c_size_t, POINTER(_STATE), _ERR]),
    "t5d_search_deliver_evaluations": (c_int32, [_SEARCH, POINTER(CEvaluations), _ERR]),
    "t5d_search_abandon": (c_int32, [_SEARCH]),
    "t5d_search_run": (c_int32, [_SEARCH, c_int32, c_uint64, c_uint64, _ERR]),
    "t5d_search_run_limits": (c_int32, [_SEARCH, c_int32, c_uint64, POINTER(CSearchLimits), POINTER(c_int32), _ERR]),
    "t5d_search_info": (c_int32, [_SEARCH, POINTER(CSearchSummary)]),
    "t5d_search_child": (c_int32, [_SEARCH, c_size_t, POINTER(CRootChild), POINTER(MoveRecord), c_size_t, _ERR]),
    "t5d_search_choose": (c_int32, [_SEARCH, POINTER(c_size_t), POINTER(c_int32), _ERR]),
    "t5d_search_choose_by_visits": (c_int32, [_SEARCH, c_double, c_double, POINTER(c_size_t), _ERR]),
    "t5d_search_verify": (c_int32, [_SEARCH, c_size_t, POINTER(c_int32), _ERR]),
    "t5d_search_advance": (c_int32, [_SEARCH, c_size_t, _ERR]),
    "t5d_search_advance_action": (c_int32, [_SEARCH, POINTER(MoveRecord), c_size_t, _ERR]),
    "t5d_search_root_game": (c_int32, [_SEARCH, POINTER(_GAME), _ERR]),
    "t5d_search_set_epoch": (c_int32, [_SEARCH, c_uint64, _ERR]),
    "t5d_search_digest": (c_int32, [_SEARCH, POINTER(c_uint64)]),
    "t5d_search_layout": (c_int32, [c_char_p, c_char_p, POINTER(c_size_t)]),
}


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32" or sys.platform == "cygwin":
        return ("t5d.dll", "libt5d.dll")
    if sys.platform == "darwin":
        return ("libt5d.dylib", "libt5d.so")
    return ("libt5d.so",)


def repository_root() -> Path:
    """The repository root: this file is ``python/t5d/engine/capi.py``."""
    return Path(__file__).resolve().parents[3]


def default_build_dir() -> Path:
    """Where ``py engine/build.py`` puts the library: ``build/engine`` under the root."""
    return repository_root() / "build" / "engine"


def _in_directory(directory: Path) -> Path | None:
    for name in _library_names():
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def find_library(path: str | os.PathLike | None = None) -> Path:
    """The engine library's path (see the module docstring); ``EngineNotFoundError`` if none."""
    names = " or ".join(_library_names())
    if path is not None:
        p = Path(path)
        found = _in_directory(p) if p.is_dir() else (p if p.is_file() else None)
        if found is None:
            raise EngineNotFoundError(f"no engine library at {p} (expected a file, or a directory holding {names})")
        return found
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        p = Path(env)
        found = _in_directory(p) if p.is_dir() else (p if p.is_file() else None)
        if found is None:
            raise EngineNotFoundError(f"{ENV_VAR}={env} names no engine library (expected a file, or a directory holding {names})")
        return found
    build = default_build_dir()
    found = _in_directory(build)
    if found is None:
        raise EngineNotFoundError(
            f"the engine library ({names}) is not built: looked in {build} and {ENV_VAR} is unset. "
            "Build it with `py engine/build.py` (python3 elsewhere) from the repository root, "
            f"or set {ENV_VAR} to the library's path."
        )
    return found


class Library:
    """The loaded shared library with every C API function declared (``lib.fn.t5d_...``)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.fn = ctypes.CDLL(str(path))
        except OSError as exc:
            raise EngineNotFoundError(f"cannot load the engine library {path}: {exc}") from exc
        missing = []
        for name, (restype, argtypes) in _SIGNATURES.items():
            try:
                f = getattr(self.fn, name)
            except AttributeError:
                missing.append(name)
                continue
            f.restype = restype
            f.argtypes = argtypes
        if missing:
            raise EngineNotFoundError(f"{path} is not the engine's C API (or is out of date): it lacks {', '.join(missing)}")
        self.version = self.fn.t5d_version().decode("ascii", "replace")
        mismatches = layout_mismatches(self)
        if mismatches:
            raise EngineNotFoundError(
                f"{path} (engine {self.version}) lays out its structs otherwise than this binding - "
                f"rebuild the library or update t5d/engine/capi.py: {'; '.join(mismatches)}"
            )

    def __repr__(self) -> str:
        return f"Library({str(self.path)!r}, version={self.version!r})"

    # -- calling conventions --

    def take_string(self, pointer: c_void_p | int | None) -> str:
        """Copy a library-allocated string and free it."""
        address = pointer.value if isinstance(pointer, c_void_p) else pointer
        if not address:
            return ""
        try:
            return ctypes.string_at(address).decode("utf-8", "replace")
        finally:
            self.fn.t5d_free(address)

    def call(self, name: str, *args, error: bool = False) -> None:
        """Call ``name``; on a non-zero status raise its exception with the engine's message.
        ``error``: the function takes a trailing ``char** error``, supplied here."""
        f = getattr(self.fn, name)
        if error:
            message = c_void_p()
            status = f(*args, ctypes.byref(message))
            text = self.take_string(message)  # freed whatever the status (NULL on success)
        else:
            status = f(*args)
            text = ""
        if status != T5D_OK:
            raise error_for(status, text, name)


_LOCK = threading.Lock()
_LOADED: dict[Path, Library] = {}


def load_library(path: str | os.PathLike | None = None) -> Library:
    """Find (``find_library``) and load the engine library; one ``Library`` per file."""
    found = find_library(path).resolve()
    with _LOCK:
        lib = _LOADED.get(found)
        if lib is None:
            lib = _LOADED[found] = Library(found)
        return lib


def library_available(path: str | os.PathLike | None = None) -> bool:
    """Whether the engine library can be found and loaded (for skipping tests)."""
    try:
        load_library(path)
    except EngineNotFoundError:
        return False
    return True


def library_status(path: str | os.PathLike | None = None, *, fresh: bool = False) -> tuple[str, str]:
    """``("available", "")``; ``("missing", why)`` - no library file where ``find_library``
    looks (none built, no path or ``T5D_ENGINE_LIB`` given); or ``("refused", why)`` - a
    library that was named or found but does not load: not a library, without the C API's
    functions, or laid out otherwise than this binding (``layout_mismatches``), or an explicit
    path or ``T5D_ENGINE_LIB`` that names nothing. A refused library is a fault to report, never
    a reason to skip. ``fresh``: check with a new ``Library`` instead of the loaded one."""
    named = path is not None or bool(os.environ.get(ENV_VAR, "").strip())
    try:
        found = find_library(path).resolve()
    except EngineNotFoundError as exc:
        return ("refused" if named else "missing"), str(exc)
    try:
        if fresh:
            Library(found)
        else:
            load_library(found)
    except EngineNotFoundError as exc:
        return "refused", str(exc)
    return "available", ""


def char_array(strings: list[str]) -> ctypes.Array:
    """A ``const char* const*`` of UTF-8 strings (the array keeps the bytes alive)."""
    encoded = [s.encode("utf-8") for s in strings]
    array = (c_char_p * max(1, len(encoded)))()
    for i, b in enumerate(encoded):
        array[i] = b
    return array


__all__ = [
    "BUILT_RELATIONS",
    "CEvaluations",
    "CExpansion",
    "CLeaf",
    "CRootChild",
    "CSearchLimits",
    "CSearchParams",
    "CSearchSummary",
    "ENV_VAR",
    "MIRRORED_SEARCH_STRUCTS",
    "MIRRORED_STRUCTS",
    "ExpandOptions",
    "Fingerprint",
    "GameSummary",
    "Library",
    "MOVE_DTYPE",
    "MoveRecord",
    "RESERVED_RELATIONS",
    "StateInfo",
    "char_array",
    "default_build_dir",
    "find_library",
    "layout_mismatches",
    "library_available",
    "library_status",
    "load_library",
    "repository_root",
]
