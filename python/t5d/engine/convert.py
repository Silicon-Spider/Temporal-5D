"""From one engine expansion to the network's input: ``to_position`` and ``to_position_state``.

``Expansion`` (``expansion.py``) is the engine's description of one position for the
player to move; ``PositionState`` (``python/t5d/net/state.py``) is the network's input
contract. The engine writes its arrays in the contract's own conventions (``t5d/records.h``,
engine README "Expansion"), so the conversion is a re-indexing - no rule is computed here:

=============================================  ==============================================
Engine (``t5d_expansion``)                     ``PositionState``
=============================================  ==============================================
``width``, ``height``                          ``width``, ``height``
boards, canonical order (timeline, then h)     storage order (board ``b`` is engine board ``b``)
``timeline`` (p; "-0" -> -1, "+0" -> 0)        ``timeline`` (= ``state.timeline_index``)
``half_turn`` (h = 2T + black)                 ``half_turn``
``is_head``                                    ``is_head`` (head vs superseded)
``active``                                     ``active``
``side_to_move`` (h & 1)                       ``side_to_move``
``present``                                    checked equal to ``h_present`` (Δt is derived)
``acting_player`` (= ``to_move``)              ``acting_player`` (register D8, stage N5)
``submit_possible``, ``submit_legal``          ``submit_possible``, ``submit_legal``
``piece`` (cells, 0 / 1..12 P N B R Q K U D    ``piece`` (``s = y*W + x``, ``PIECE_LETTERS``)
S W C Y)
``colour`` (0 / 1 white / 2 black)             ``colour``
``moved`` (P R K W without '*')                ``moved``
``edges`` (src b, src cell, dst b, dst cell)   ``edges`` rows ``(b*(W*H+R) + cell, ..., 0)``
``relations`` (..., id)                        ``edges`` rows ``(src token, dst token, id)``
``moves`` (from/to board, from/to cell,        ``moves`` rows ``(src token, dst token,
promotion)                                     promotion)`` - row i is engine move i
``parent`` (-1: none known)                    ``parent``
(nothing)                                      ``creation`` = None, ``source`` = all -1,
                                               ``opp_moves`` empty
``window``, ``e_boards``, ``src_extra``,       ``Position.scope`` (``EngineScope``): board
``window_depth``, ``band_depth``,              sets and token subsets, at any k
``is_origin``
=============================================  ==============================================

**Readings**, each the engine's or forced by the contract:

* **Δt-to-present.** ``PositionState`` derives it from ``h - h_present``, ``h_present`` the
  least ``h`` over active heads; the engine's ``present`` (T4) is checked to be that value
  (``PresentMismatch`` otherwise).
* **The action (register D8; stage N5).** Turns have variable length (Q4, ruled R8): once
  the present has shifted to the opponent within an action not yet submitted (T5), the
  boards' present names the opponent while the action is still the mover's, and under the
  default rules the mover may go on moving until they submit - so search meets such states.
  The expansion's ``acting_player`` (whose action this is, = ``to_move``) becomes the
  network's global acting-player input (``PositionState.acting_player``), and its
  ``submit_possible`` / ``submit_legal`` say whether the submit action - its own policy logit,
  read from a global register - is offered, and whether it is masked. Such states convert:
  stage N4 refused them (``PresentMismatch``), the network having no player-to-move feature
  apart from the present. What is still checked (``PresentMismatch``, unless
  ``strict=False``): the acting player is ``to_move``, and submitting is possible exactly
  when the present has shifted (T5: the obligations are met when the present lies on the
  opponent's colour) - so the acting player is the present's colour or, past the shift, its
  opponent. Under Q4's Guide-literal reading (``T5D_RULES_NO_MOVES_AFTER_SHIFT``) a state
  past the shift has no move and offers submit alone.
* **Moves.** The engine's pseudo-legal moves of the player to move (the legal-moves
  filter is not in the C API), one row per move in the engine's order, so policy logit
  ``i`` scores engine move ``i`` (``t5d_game_apply_expanded(i)``). A branching move's
  destination token is the history board it targets, which exists (never the board it
  creates). Promotions: one row per promoted-to piece.
* **Edges.** The move graph - one edge per distinct (source, destination) square pair,
  promotions merged - as relation ``move`` (0); then the typed relations the expansion was
  asked for (attack 1, defence 2, x-ray 4: the ids are ``RELATION_TYPES``'s). The engine
  builds them only for the player to move's pieces on the boards its moves start from.
  **Pin (3), check (5) and travel-threat (6) are absent**: the engine refuses them (pin and
  check would read check detection; the spec defines no travel-threat edge).
* **Opponent moves** (``opp_moves``, the D4 reply head) are not supplied: the expansion
  is the mover's (spec issue 17: E is read from the mover's moves); empty.
* **Lineage - the causal past is ancestry (register C23).** ``parent`` is the engine's: a
  timeline's previous board, or for a timeline's first board its branch origin
  (``Timeline::origin``), -1 for an initial timeline or an origin nobody recorded - every
  branched timeline of a position read from 5DFEN (spec issue 25). Its transitive closure -
  ``NetConfig.causal_past="lineage"``, the default (``net/scope.py``) - is C23's ruled
  reading: a board's own timeline's earlier boards and, recursively, its branch parent up to
  the branch point; never a board merely earlier on the time axis, never one merely created
  earlier. ``Position.origins_unknown`` is the engine's count over active timelines;
  ``Position.origins_unknown_all`` (when the game's creator count is given) counts every
  timeline. ``creation`` (board creation order) and ``source`` (a travel arrival's
  departure board) are not in the C API: None and all -1, so the ``creation`` causal past
  cannot be used and ``two_parent`` reads as ``lineage`` (spec issues 29-30).
* **Has-moved** is the engine's: 1 exactly on a pawn, rook, king or brawn without its
  unmoved marker (the build log's proposed spec edit 23).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..net.config import TYPED_RELATIONS, NetConfig
from ..net.state import RELATION_IDS, PositionState, TokenLayout
from .capi import BUILT_RELATIONS, T5D_MOVE_TRAVEL, T5D_NO_WINDOW
from .expansion import Expansion


class PresentMismatch(ValueError):
    """The engine's present, acting player or submit flags disagree with what the boards say:
    the present is not the least half-turn over active heads, the acting player is not the
    player to move, or submitting is possible where the present has not shifted (or the
    reverse). See the ``convert`` module docstring, "Δt-to-present" and "The action"; a state
    past the shift is not a mismatch (stage N5)."""


def relations_for(cfg: NetConfig) -> tuple[str, ...]:
    """The typed relations a network configuration reads that the engine builds: B11's
    typed biases read every type, B10's relations head ``relations_head_types``; neither
    flag on: none (B9's move graph is always made)."""
    wanted: set[str] = set()
    if cfg.typed_edge_bias:
        wanted |= set(TYPED_RELATIONS)
    if cfg.routing_table:
        wanted |= set(cfg.relations_head_types)
    return tuple(name for name in BUILT_RELATIONS if name in wanted)


@dataclass(frozen=True, eq=False)
class EngineScope:
    """W(k), E(k) and src_extra(k) of one expansion (spec §10.4.1; engine README
    "Expansion" (c)-(d)) as board sets, and as token subsets of the full position. The
    per-board depths and the move-derived board sets answer every k (``at``)."""

    k: int
    bands_only: bool
    window: np.ndarray  # W(k): board indices, ascending
    e_boards: np.ndarray  # E(k): travel destinations outside W(k)
    src_extra: np.ndarray  # src_extra(k): move sources outside W(k)
    window_depth: np.ndarray  # per board: least k with the board in W(k); T5D_NO_WINDOW none
    band_depth: np.ndarray  # the same without the branch origins
    is_origin: np.ndarray  # per board: an active timeline's branch origin
    travel_destinations: np.ndarray  # boards a travel move lands on (every k)
    source_boards: np.ndarray  # boards a move starts from (every k)

    @classmethod
    def from_expansion(cls, x: Expansion) -> EngineScope:
        travel = (x.moves["flags"] & T5D_MOVE_TRAVEL) != 0
        return cls(
            k=x.window_k,
            bands_only=False,
            window=x.window.astype(np.int64),
            e_boards=x.e_boards.astype(np.int64),
            src_extra=x.src_extra.astype(np.int64),
            window_depth=x.window_depth.astype(np.int64),
            band_depth=x.band_depth.astype(np.int64),
            is_origin=x.is_origin.copy(),
            travel_destinations=np.unique(x.moves["to_board"][travel]).astype(np.int64),
            source_boards=np.unique(x.moves["from_board"]).astype(np.int64),
        )

    def at(self, k: int, *, bands_only: bool = False) -> EngineScope:
        """The scope at another ``k`` (half-turns), against W(k) or the bands alone (the
        engine's ``Window::BandsOnly``): W(k) is every board whose depth is at most k (a k
        of T5D_NO_WINDOW or more counts as T5D_NO_WINDOW - 1, as the engine takes it)."""
        if k < 0:
            raise ValueError("k must be 0 or more")
        k = min(int(k), T5D_NO_WINDOW - 1)
        depth = self.band_depth if bands_only else self.window_depth
        window = np.flatnonzero(depth <= k)
        outside = np.ones(depth.shape[0], dtype=bool)
        outside[window] = False
        return EngineScope(
            k=k,
            bands_only=bands_only,
            window=window.astype(np.int64),
            e_boards=self.travel_destinations[outside[self.travel_destinations]],
            src_extra=self.source_boards[outside[self.source_boards]],
            window_depth=self.window_depth,
            band_depth=self.band_depth,
            is_origin=self.is_origin,
            travel_destinations=self.travel_destinations,
            source_boards=self.source_boards,
        )

    def boards(self, *, src_extra: bool = True) -> np.ndarray:
        """W(k) + E(k) (+ src_extra(k): the policy pointer's out-of-window sources), ascending."""
        parts = [self.window, self.e_boards] + ([self.src_extra] if src_extra else [])
        return np.unique(np.concatenate(parts)).astype(np.int64)

    def tokens(self, layout: TokenLayout, *, src_extra: bool = True, registers: bool = True, global_registers: bool = True) -> np.ndarray:
        """The scope's tokens in the full position's layout, ascending: every square (and,
        with ``registers``, every per-board register) of its boards, then the global
        registers. A board set, not a position: it can hold a history board of a timeline
        without the timeline's head (an E board of an inactive timeline)."""
        boards = self.boards(src_extra=src_extra)
        per_board = layout.tokens_per_board if registers else layout.squares_per_board
        body = (boards[:, None] * layout.tokens_per_board + np.arange(per_board)[None, :]).reshape(-1)
        tail = layout.n_body + np.arange(layout.global_regs) if global_registers else np.zeros(0, dtype=np.int64)
        return np.concatenate([body, tail]).astype(np.int64)


@dataclass(frozen=True, eq=False)
class Position:
    """One engine position as the network's input, with what maps the network's outputs back
    to the engine. ``state.moves`` row ``i`` is ``moves[i]``, the engine's move ``i``."""

    state: PositionState
    moves: np.ndarray  # MOVE_DTYPE (M,): the engine's move records, in its order
    to_move: int  # the acting player (0 white, 1 black; the engine's to_move) - the value's perspective
    present: int
    may_move: bool
    origins_unknown: int  # the engine's: active branched timelines with no recorded origin
    origins_unknown_all: int | None  # every timeline (None: the creator count was not given)
    relations: tuple[str, ...]  # the typed relations ``state.edges`` holds besides the move graph
    scope: EngineScope | None
    present_matches: bool = True  # False only under strict=False (see PresentMismatch)

    @property
    def move_count(self) -> int:
        return int(self.moves.shape[0])

    @property
    def acting_player(self) -> int:
        """Whose action this is (register D8): the network's input and the value's perspective."""
        return int(self.state.acting_player)

    @property
    def shifted(self) -> bool:
        """The present has shifted to the opponent inside the acting player's action (T5)."""
        return self.state.shifted

    @property
    def submit_possible(self) -> bool:
        """Submitting is an action here (D8: offered beside the moves under the ply split)."""
        return self.state.submit_possible

    @property
    def submit_legal(self) -> bool:
        """... and legal (C4) - the submit logit is masked otherwise."""
        return self.state.submit_legal

    @property
    def lineage_complete(self) -> bool | None:
        """Whether every branched timeline's origin is known (None: not counted)."""
        return None if self.origins_unknown_all is None else self.origins_unknown_all == 0


def _tokens(board: np.ndarray, cell: np.ndarray, tpb: int) -> np.ndarray:
    return board.astype(np.int64) * tpb + cell.astype(np.int64)


def _check_cells(name: str, board: np.ndarray, cell: np.ndarray, n: int, wh: int) -> None:
    if board.size and (board.min() < 0 or board.max() >= n or cell.min() < 0 or cell.max() >= wh):
        raise ValueError(f"the expansion's {name} name a board or cell out of range (N = {n}, W*H = {wh})")


def to_position_state(x: Expansion, *, regs_per_board: int = 2, global_regs: int = 16) -> PositionState:
    """The ``PositionState`` of an expansion made with every board's features (see the module
    docstring's table) - the acting player and the submit action's flags included (D8);
    validated by ``PositionState`` itself."""
    if x.square_features != "all" or not x.all_features:
        raise ValueError(
            "a PositionState needs every board's square features: expand with square_features='all' "
            f"(this expansion has features for {x.feature_boards.shape[0]} of {x.board_count} boards)"
        )
    n, wh = x.board_count, x.width * x.height
    tpb = wh + int(regs_per_board)
    m = x.moves
    _check_cells("moves", np.concatenate([m["from_board"], m["to_board"]]), np.concatenate([m["from_cell"], m["to_cell"]]), n, wh)
    _check_cells("edges", x.edges[:, [0, 2]].reshape(-1), x.edges[:, [1, 3]].reshape(-1), n, wh)
    _check_cells("relations", x.relations[:, [0, 2]].reshape(-1), x.relations[:, [1, 3]].reshape(-1), n, wh)
    moves = np.stack(
        [_tokens(m["from_board"], m["from_cell"], tpb), _tokens(m["to_board"], m["to_cell"], tpb), m["promotion"].astype(np.int64)], axis=1
    )
    graph = np.stack(
        [
            _tokens(x.edges[:, 0], x.edges[:, 1], tpb),
            _tokens(x.edges[:, 2], x.edges[:, 3], tpb),
            np.full(x.edges.shape[0], RELATION_IDS["move"]),
        ],
        axis=1,
    )
    typed = np.stack(
        [
            _tokens(x.relations[:, 0], x.relations[:, 1], tpb),
            _tokens(x.relations[:, 2], x.relations[:, 3], tpb),
            x.relations[:, 4].astype(np.int64),
        ],
        axis=1,
    )
    return PositionState(
        width=x.width,
        height=x.height,
        timeline=x.timeline.astype(np.int64),
        half_turn=x.half_turn.astype(np.int64),
        is_head=x.is_head.copy(),
        active=x.active.copy(),
        side_to_move=x.side_to_move.astype(np.int64),
        piece=x.piece.astype(np.int64),
        colour=x.colour.astype(np.int64),
        moved=x.moved.copy(),
        edges=np.concatenate([graph, typed]).reshape(-1, 3),
        moves=moves.reshape(-1, 3),
        opp_moves=np.zeros((0, 3), dtype=np.int64),
        regs_per_board=int(regs_per_board),
        global_regs=int(global_regs),
        parent=x.parent.astype(np.int64),
        creation=None,
        source=None,
        acting_player=int(x.acting_player),
        submit_possible=bool(x.submit_possible),
        submit_legal=bool(x.submit_legal),
    )


def _relation_names(bits: int) -> tuple[str, ...]:
    return tuple(name for name, rid in BUILT_RELATIONS.items() if bits & (1 << rid))


def to_position(
    x: Expansion,
    *,
    regs_per_board: int = 2,
    global_regs: int = 16,
    created_timelines: int | None = None,
    scope: bool = True,
    strict: bool = True,
) -> Position:
    """The network's input for an expansion (see the module docstring): the validated
    ``PositionState``, the engine's moves it scores, the lineage's completeness and the W/E
    scope. ``created_timelines``: the game's timelines with a creator (``Game.created_timelines``)
    - given, ``origins_unknown_all`` counts the branched timelines whose first board has no
    parent. A state whose present has shifted within an unsubmitted action converts, with its
    acting player and the submit action's flags (register D8). ``strict``: refuse an expansion
    whose present, acting player or submit flags disagree with its boards
    (``PresentMismatch``); with ``strict=False`` the remaining checks are ``PositionState``'s."""
    live = x.is_head & x.active
    if not live.any():
        raise ValueError("the expansion has no active head: the present is undefined")
    h_present = int(x.half_turn[live].min())
    shifted = int(x.acting_player) != (h_present & 1)
    matches = h_present == x.present and x.acting_player == x.to_move and bool(x.submit_possible) == shifted
    if strict and not matches:
        raise PresentMismatch(
            f"the engine's present is h = {x.present}, its acting player {'wb'[x.acting_player]} (to move: {'wb'[x.to_move]}) and "
            f"submitting {'possible' if x.submit_possible else 'not possible'}, but the boards' active heads put the present at "
            f"h = {h_present} ({'wb'[h_present & 1]} to move on it): the present must be the boards', the acting player the player "
            "to move, and submitting possible exactly when the present has shifted to the acting player's opponent (T5)"
        )
    state = to_position_state(x, regs_per_board=regs_per_board, global_regs=global_regs)
    unknown_all = None
    if created_timelines is not None:
        first = state.parent[_first_boards(state)]
        unknown_all = int(created_timelines) - int(np.count_nonzero(first >= 0))
    return Position(
        state=state,
        moves=x.moves.copy(),
        to_move=x.to_move,
        present=x.present,
        may_move=x.may_move,
        origins_unknown=x.origins_unknown,
        origins_unknown_all=unknown_all,
        relations=_relation_names(x.relation_bits),
        scope=EngineScope.from_expansion(x) if scope else None,
        present_matches=matches,
    )


def _first_boards(state: PositionState) -> np.ndarray:
    """Each timeline's first board (boards are in canonical order: ascending p, then h)."""
    p = state.timeline
    first = np.ones(p.shape[0], dtype=bool)
    first[1:] = p[1:] != p[:-1]
    return np.flatnonzero(first)


__all__ = ["EngineScope", "Position", "PresentMismatch", "relations_for", "to_position", "to_position_state"]
