"""The main search from Python: ``Search`` wraps one ``t5d_search`` of the engine's step API
(``t5d/capi_search.h``; engine stage E6a, ``t5d/mcts.hpp``).

The search is the engine's - PUCT or MCGS over game states under any arm of D2 (the tree form
under PUCT only), with its
legality policy, terminal detection, turn cap, evaluation cache and final selection. Python
is its **evaluator**: ``gather`` selects a batch of leaves, ``leaf`` describes each (its
expansion - the network's input - with D8's acting player and submit flags), ``deliver``
hands back one logit per candidate move (logit ``i`` scores ``leaf.expansion.moves[i]``), one
submit logit per leaf (NaN: none) and D3's value heads per leaf, from the leaf's acting
player: the WDL - the value search uses (W - L backed up, its draw probability beside it) -
and, optionally, the 51-bin categorical, the error head's output and moves-left, and - read
by D2's flat form (``T5D_SPLIT_FLAT``: whole legal turns generated best first) - one "leave
it" logit per expansion board (NaN: none for that board, the search's ``leave_share``)
(``t5d_search_deliver_evaluations``). ``evaluator.CStepDriver`` runs that loop with the
network; this module is the thin layer over the C functions, and nothing in it computes a
rule or a search decision.

* **Parameters.** ``search_params(**overrides)`` fills a ``t5d_search_params`` with the
  library's defaults (``t5d_search_params_default``) and the overrides by field name;
  ``expand`` takes an ``ExpandOptions`` or a ``(window_k, square_features, relations)``
  triple. **The proof channel stays off** (register H41): ``proof_channel`` must be 0 - no
  solved-status propagation and no export until the gate passes and the channel is signed
  off; ``search_params`` refuses anything else.
* **The root's turn state (register A6).** A search's root takes the game handle's count of
  moves made in its action (``GameInfo.moves_made``): the network sees the count, and the
  search keeps states with different counts apart (its graph's identity, its evaluation cache,
  its batches). ``turn_start`` defaults to what the handle says; given, the engine checks it
  against the count.
* **Fresh searches per decision (C14).** ``Search(game, share=donor)`` makes a search that
  shares ``donor``'s evaluation cache (``t5d_search_new_sharing``): self-play's fresh search
  per move with its state-keyed evaluations carried. A search's cache must only ever hold
  one evaluator's evaluations - never share between searches fed by different networks - and
  only evaluations of one input: sharing is refused unless the donor's split, expansion
  options and legality classification match (``cache_mismatches``). An evaluator whose
  evaluations depend on the search's root (the cached path's refresh epochs) sets the search's
  epoch, so that the cache keeps them apart (``evaluator.start_epoch``). The cache has no lock:
  the searches sharing it must never be used from two threads at once - the caller's duty, as
  the C calls release the GIL and nothing here checks it.
* **Lifetimes.** A leaf's pointers live until the next ``gather``, ``deliver``, ``abandon``
  or ``close``: ``leaf_expansion`` copies at once (``Expansion.from_c``). ``close`` (or the
  context manager, or garbage collection) releases the handle.

Errors raise the exception of the C API's status (``errors.py``) with the engine's message:
the caller's mistakes (a gather while a batch waits, a deliver with the wrong counts, an
infinite submit logit) are ``EngineArgumentError``; the search's own failures
``EngineInternalError``.
"""

from __future__ import annotations

import contextlib
import ctypes
import math
from collections.abc import Sequence
from ctypes import byref, c_double, c_float, c_int32, c_size_t, c_uint64, c_void_p
from dataclasses import dataclass

import numpy as np

from .capi import (
    T5D_LEGALITY_LEGAL,
    T5D_NO_LIMIT,
    T5D_VALUE_LOGITS,
    T5D_VALUE_PROBABILITIES,
    CEvaluations,
    CLeaf,
    CRootChild,
    CSearchLimits,
    CSearchParams,
    CSearchSummary,
    ExpandOptions,
    Library,
    MoveRecord,
    load_library,
)
from .errors import EngineArgumentError
from .expansion import SQUARE_FEATURES, Expansion, Move
from .game import Game


def search_params(lib: Library | None = None, **overrides) -> CSearchParams:
    """A ``t5d_search_params`` with the library's defaults and ``overrides`` by field name.
    ``expand``: an ``ExpandOptions`` or ``(window_k, square_features, relation_bits)``, the
    features by name ("all", "scope", "none") or constant. ``proof_channel`` must stay 0 (H41)."""
    lib = load_library() if lib is None else lib
    p = CSearchParams()
    lib.fn.t5d_search_params_default(byref(p))
    names = {name for name, _ in CSearchParams._fields_}
    for name, value in overrides.items():
        if name not in names:
            raise ValueError(f"t5d_search_params has no field {name!r}")
        if name == "expand" and not isinstance(value, ExpandOptions):
            k, features, relations = value
            features = SQUARE_FEATURES[features] if isinstance(features, str) else int(features)
            setattr(p, name, ExpandOptions(int(k), features, int(relations)))
        else:
            setattr(p, name, value)
    if p.proof_channel != 0:
        raise ValueError("proof_channel stays 0: register H41 keeps solved-status propagation and export off until the gate passes")
    return p


#: The parameters that decide what a leaf's evaluation is of - the network's input (the
#: expansion's options, the split: D8's submit flags) and the leaf's candidate moves (the legality
#: classification drops proved-illegal moves, so its budget changes the list the logits score).
CACHE_KEYS: tuple[str, ...] = ("split", "legal_moves_only", "legality_checks", "legality_nodes")


def cache_mismatches(a: CSearchParams, b: CSearchParams) -> list[str]:
    """The parameters on which two searches' evaluations would differ for one state (empty: they
    may share an evaluation cache - fed by the same evaluator)."""
    out = [name for name in CACHE_KEYS if getattr(a, name) != getattr(b, name)]
    out += [f"expand.{name}" for name, _ in ExpandOptions._fields_ if getattr(a.expand, name) != getattr(b.expand, name)]
    return out


@dataclass(frozen=True)
class SearchInfo:
    """``t5d_search_summary``."""

    playouts: int
    evaluations: int  # leaves evaluated (cache hits not counted)
    cache_hits: int
    nodes: int
    nodes_created: int
    transposition_hits: int
    checkmates: int
    stalemates: int
    turn_caps: int
    unknown_verdicts: int
    unknown_edges: int
    verified_illegal: int
    root_turn_start: bool
    root_terminal: bool
    root_solved: bool
    root_lo: int
    root_hi: int
    root_visits: int
    root_q: float
    root_children: int
    epoch: int
    memory_bytes: int
    cache_entries: int
    cache_bytes: int
    cache_purged: int
    last_stop: int
    # engine ABI round 2 (register D3): the root's backed-up draw probability and its own
    # evaluation - the network's raw outputs at the root (I15), its player's view; NaN: not given
    root_evaluated: bool
    root_d: float
    root_wdl: tuple[float, float, float]
    root_moves_left: float
    root_error: float
    # engine ABI round 2c (register I15, I2, D4)
    root_key: int  # the root's node key: with the moves made in its action and the epoch, its evaluation's identity
    root_cpuct: float  # the root's effective cPUCT now (cpuct x E2's root scale): I2's pruning replica
    root_best_since: int  # the effort label: the root's visits when its most visited child last changed
    # engine ABI round 2 review fixes (register I10, 6m)
    uncertainty_missing: int  # nodes expanded without the selected uncertainty signal (none read it)
    # D2's tree form: mid-turn nodes expanded on inherited priors (never evaluated), and the mid-turn
    # nodes sent to the network (or the cache) instead, by the first gate found to fire
    inherited_nodes: int
    gate_between: int
    gate_capture: int
    gate_check: int
    gate_present: int
    gate_always: int
    # D2's flat form: turns generated (children made), and generators a budget stopped (their
    # nodes' children incomplete - never proven lost, H37)
    generated_turns: int
    stopped_generators: int
    fallback_listings: int  # ... of which before their first turn: E4's listing's turns instead

    @property
    def root_finished(self) -> bool:
        """The root needs no evaluation: terminal (checkmate, stalemate, the cap) or solved."""
        return self.root_terminal or self.root_solved


@dataclass(frozen=True)
class RootChild:
    """``t5d_root_child`` with its moves (boards and cells of the root state's order)."""

    index: int
    visits: int
    submit: bool  # the submit action (board-move split)
    legality: int  # T5D_LEGALITY_LEGAL or _UNKNOWN
    lo: int  # solver bounds: search state, never to be persisted as a proof (H19, H23)
    hi: int
    prior: float
    q: float  # the backed-up value, W - L (D3)
    moves: tuple[Move, ...]  # 1 (a move), a turn's (flat form), 0 (submit)
    # engine ABI round 2 (register D3): the search's draw probability, and the child's own
    # evaluation - the network's, the root player's view (NaN where not given)
    d: float  # the search WDL: ((1 - d + q) / 2, d, (1 - d - q) / 2)
    evaluated: bool
    network_wdl: tuple[float, float, float]
    network_moves_left: float
    network_error: float
    # engine ABI round 2c (register I15, I2): the root's side table
    child_key: int  # the child node's key; 0: no child node yet
    prior_noise_free: float  # the prior before the root noise (after the temperatures)
    value_variance: float  # of the values backed up through it
    boost: float  # E10's asset-map boost: 0 (not wired)
    exploration_scale: float  # E3's factor (1: off)
    forced_visits: int  # 0: forced playouts not built (pruning is target-side, §12.5)
    steered_visits: int  # 0: no steering source wired (E10)


@dataclass(frozen=True)
class LeafView:
    """One gathered leaf, copied out: the expansion (the network's input) and D8's flags."""

    index: int
    key: int  # the state's node key; the leaf's evaluation identity is (key, expansion.moves_made, epoch) (A6)
    epoch: int
    acting_player: int
    submit_possible: bool
    submit_legal: bool
    legality: np.ndarray | None  # uint8 per move (T5D_LEGALITY_*), None when none was made
    expansion: Expansion


class Search:
    """One search handle (see the module docstring)."""

    def __init__(
        self,
        game: Game,
        *,
        turn_start: bool | None = None,
        params: CSearchParams | None = None,
        share: Search | None = None,
        lib: Library | None = None,
    ) -> None:
        self._lib = game.lib if lib is None else lib
        self.params = search_params(self._lib) if params is None else params
        if self.params.proof_channel != 0:
            raise ValueError("proof_channel stays 0 (register H41)")
        if share is not None:
            differ = cache_mismatches(share.params, self.params)
            if differ:
                raise ValueError(
                    "a search shares an evaluation cache only with a donor whose leaves give the network the same input - "
                    f"these parameters differ: {', '.join(differ)}"
                )
        out = c_void_p()
        game_handle = game._live()
        with game._lock:
            if turn_start is None:  # A6: the handle knows whether a move has been made in its action
                turn_start = game.info().turn_start
            if share is None:
                self._lib.call("t5d_search_new", game_handle, int(bool(turn_start)), byref(self.params), byref(out), error=True)
            else:
                self._lib.call(
                    "t5d_search_new_sharing", share._live(), game_handle, int(bool(turn_start)), byref(self.params), byref(out), error=True
                )
        self._handle = out

    # -- lifetime --

    def _live(self) -> c_void_p:
        if self._handle is None:
            raise EngineArgumentError("the search has been closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            self._lib.fn.t5d_search_destroy(handle)

    def __enter__(self) -> Search:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    @property
    def lib(self) -> Library:
        return self._lib

    @property
    def square_features(self) -> str:
        names = {v: k for k, v in SQUARE_FEATURES.items()}
        return names.get(int(self.params.expand.square_features), str(self.params.expand.square_features))

    # -- the step API --

    def gather(self, max_leaves: int, max_playouts: int = T5D_NO_LIMIT) -> int:
        """Select up to ``max_leaves`` leaves while fewer than ``max_playouts`` simulations of
        this call are completed or waiting; the number of evaluations wanted (0: none needed -
        the root finished, or every visit of the call completed without the evaluator)."""
        count = c_size_t()
        self._lib.call("t5d_search_gather", self._live(), c_size_t(int(max_leaves)), c_uint64(int(max_playouts)), byref(count), error=True)
        return int(count.value)

    def leaf(self, index: int) -> CLeaf:
        """Leaf ``index`` of the waiting batch as the C struct (pointers valid until the next
        gather, deliver, abandon or close)."""
        out = CLeaf()
        self._lib.call("t5d_search_leaf", self._live(), c_size_t(int(index)), byref(out), error=True)
        return out

    def leaf_view(self, index: int) -> LeafView:
        """Leaf ``index``, copied out (``Expansion.from_c``): it outlives the batch."""
        c = self.leaf(index)
        x = Expansion.from_c(c.expansion, self.square_features, int(self.params.expand.relations))
        legality = None
        if c.legality and x.move_count:
            legality = np.ctypeslib.as_array(c.legality, shape=(x.move_count,)).copy()
        return LeafView(
            index=int(index),
            key=int(c.key),
            epoch=int(c.epoch),
            acting_player=int(c.acting_player),
            submit_possible=bool(c.submit_possible),
            submit_legal=bool(c.submit_legal),
            legality=legality,
            expansion=x,
        )

    def deliver(
        self,
        logits: Sequence[float] | np.ndarray,
        wdl: Sequence[float] | np.ndarray,
        *,
        values: Sequence[float] | np.ndarray | None = None,
        errors: Sequence[float] | np.ndarray | None = None,
        moves_left: Sequence[float] | np.ndarray | None = None,
        submit_logits: Sequence[float] | np.ndarray | None = None,
        board_logits: Sequence[float] | np.ndarray | None = None,
        wdl_kind: int = T5D_VALUE_LOGITS,
        value_kind: int = T5D_VALUE_LOGITS,
    ) -> None:
        """The waiting batch's evaluations, leaf by leaf in order (``t5d_search_deliver_evaluations``):
        ``logits`` - each leaf's move logits concatenated; ``wdl`` - D3's value, 3 per leaf (win,
        draw, loss; ``wdl_kind``: logits or probabilities); ``values`` - the categorical, 51 per
        leaf (``value_kind``), or None; ``errors`` and ``moves_left`` - one per leaf (NaN: none
        given), or None; ``submit_logits`` - one per leaf (NaN: none given) or None (the search's
        placeholder share everywhere); ``board_logits`` - each leaf's ``expansion.board_count``
        "leave it" logits concatenated (D2's flat form reads them at the acting player's playable
        boards; NaN: none for that board), or None. On a refusal the engine abandons the batch."""
        for name, kind in (("wdl_kind", wdl_kind), ("value_kind", value_kind)):
            if kind not in (T5D_VALUE_LOGITS, T5D_VALUE_PROBABILITIES):
                raise ValueError(f"{name} must be T5D_VALUE_LOGITS or T5D_VALUE_PROBABILITIES")
        fp = ctypes.POINTER(c_float)
        keep: list[np.ndarray] = []  # the arrays the struct points into, alive through the call

        def array(data) -> tuple[object, int]:
            if data is None:
                return None, 0
            a = np.ascontiguousarray(np.asarray(data, dtype=np.float32).reshape(-1))
            keep.append(a)
            return (a.ctypes.data_as(fp) if a.size else None), int(a.shape[0])

        b = CEvaluations()
        b.logits, b.logit_count = array(logits)
        b.submit_logits, b.submit_count = array(submit_logits)
        b.wdl, b.wdl_count = array(wdl)
        b.values, b.value_count = array(values)
        b.errors, b.error_count = array(errors)
        b.moves_left, b.moves_left_count = array(moves_left)
        b.board_logits, b.board_logit_count = array(board_logits)
        b.wdl_kind, b.value_kind = int(wdl_kind), int(value_kind)
        self._lib.call("t5d_search_deliver_evaluations", self._live(), byref(b), error=True)

    def abandon(self) -> None:
        """Release the waiting batch's virtual loss without backing anything up."""
        self._lib.call("t5d_search_abandon", self._live())

    def run(self, evaluator: int, playouts: int, *, seed: int = 0) -> None:
        """``playouts`` simulations with a built-in evaluator (``T5D_EVALUATOR_*``)."""
        self._lib.call("t5d_search_run", self._live(), int(evaluator), c_uint64(int(seed)), c_uint64(int(playouts)), error=True)

    def run_limits(
        self,
        evaluator: int,
        *,
        seed: int = 0,
        playouts: int = T5D_NO_LIMIT,
        evaluations: int = T5D_NO_LIMIT,
        nodes: int = T5D_NO_LIMIT,
        bytes: int = T5D_NO_LIMIT,  # noqa: A002 - the C field's name
        seconds: float = 0.0,
    ) -> int:
        """The same within budgets; why it stopped (``T5D_STOP_*``)."""
        limits = CSearchLimits(int(playouts), int(evaluations), int(nodes), int(bytes), float(seconds))
        stop = c_int32()
        self._lib.call("t5d_search_run_limits", self._live(), int(evaluator), c_uint64(int(seed)), byref(limits), byref(stop), error=True)
        return int(stop.value)

    # -- results --

    def info(self) -> SearchInfo:
        s = CSearchSummary()
        self._lib.call("t5d_search_info", self._live(), byref(s))
        flags = {"root_turn_start", "root_terminal", "root_solved", "root_evaluated"}

        def value(name: str):
            v = getattr(s, name)
            if name in flags:
                return bool(v)
            return tuple(float(x) for x in v) if isinstance(v, ctypes.Array) else v

        return SearchInfo(**{name: value(name) for name, _ in CSearchSummary._fields_})

    def child(self, index: int) -> RootChild:
        c = CRootChild()
        self._lib.call("t5d_search_child", self._live(), c_size_t(int(index)), byref(c), None, c_size_t(0), error=True)
        moves: tuple[Move, ...] = ()
        if c.move_count:
            buf = (MoveRecord * int(c.move_count))()
            self._lib.call("t5d_search_child", self._live(), c_size_t(int(index)), byref(c), buf, c_size_t(int(c.move_count)), error=True)
            moves = tuple(Move.from_record(r) for r in buf)
        return RootChild(
            index=int(index),
            visits=int(c.visits),
            submit=bool(c.submit),
            legality=int(c.legality),
            lo=int(c.lo),
            hi=int(c.hi),
            prior=float(c.prior),
            q=float(c.q),
            moves=moves,
            d=float(c.d),
            evaluated=bool(c.evaluated),
            network_wdl=(float(c.network_wdl[0]), float(c.network_wdl[1]), float(c.network_wdl[2])),
            network_moves_left=float(c.network_moves_left),
            network_error=float(c.network_error),
            child_key=int(c.child_key),
            prior_noise_free=float(c.prior_noise_free),
            value_variance=float(c.value_variance),
            boost=float(c.boost),
            exploration_scale=float(c.exploration_scale),
            forced_visits=int(c.forced_visits),
            steered_visits=int(c.steered_visits),
        )

    def children(self) -> list[RootChild]:
        return [self.child(i) for i in range(self.info().root_children)]

    def choose(self) -> tuple[int, int]:
        """Final selection (E5, E9): the child to play and why (``T5D_CHOICE_*``)."""
        index, reason = c_size_t(), c_int32()
        self._lib.call("t5d_search_choose", self._live(), byref(index), byref(reason), error=True)
        return int(index.value), int(reason.value)

    def choose_by_visits(self, temperature: float, random: float) -> int:
        """Self-play's choice: sampled from the root's visits at ``temperature`` (<= 0: the
        most visited), ``random`` uniform in [0, 1)."""
        if not (math.isfinite(random) and 0.0 <= random < 1.0):
            raise ValueError("random must be in [0, 1)")
        index = c_size_t()
        self._lib.call("t5d_search_choose_by_visits", self._live(), c_double(temperature), c_double(random), byref(index), error=True)
        return int(index.value)

    def verify(self, index: int) -> int:
        """Classify root child ``index``'s move exactly (``T5D_LEGALITY_*``); an illegal child
        is removed and the children's indices change."""
        out = c_int32()
        self._lib.call("t5d_search_verify", self._live(), c_size_t(int(index)), byref(out), error=True)
        return int(out.value)

    def advance(self, index: int) -> None:
        """Re-root on root child ``index`` (C12): statistics carried, unreachable nodes deleted."""
        self._lib.call("t5d_search_advance", self._live(), c_size_t(int(index)), error=True)

    def advance_action(self, moves: Sequence[Move]) -> None:
        """Re-root on the turn start an action reaches (the opponent's reply); atomic. From a
        state during an action an empty ``moves`` submits it."""
        buf = (MoveRecord * max(1, len(moves)))()
        for i, m in enumerate(moves):
            buf[i] = m.to_c()
        self._lib.call("t5d_search_advance_action", self._live(), buf, c_size_t(len(moves)), error=True)

    def root_game(self) -> Game:
        out = c_void_p()
        self._lib.call("t5d_search_root_game", self._live(), byref(out), error=True)
        return Game(out, self._lib, description="search root")

    def set_epoch(self, epoch: int) -> None:
        self._lib.call("t5d_search_set_epoch", self._live(), c_uint64(int(epoch)), error=True)

    def digest(self) -> int:
        """H15's digest of the whole search graph: equal for bit-identical searches."""
        out = c_uint64()
        self._lib.call("t5d_search_digest", self._live(), byref(out))
        return int(out.value)


def is_legal(child: RootChild) -> bool:
    return child.legality == T5D_LEGALITY_LEGAL


__all__ = ["CACHE_KEYS", "LeafView", "RootChild", "Search", "SearchInfo", "cache_mismatches", "is_legal", "search_params"]
