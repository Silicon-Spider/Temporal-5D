"""One expansion, copied out of the engine: ``Expansion``, and one move: ``Move``.

``t5d_game_expand`` fills a ``t5d_expansion`` whose arrays belong to the game handle and
stay valid only until the handle is next changed (``t5d/capi.h``). ``Expansion.from_c``
copies every array into numpy at once, so an ``Expansion`` is a snapshot that outlives the
handle, its next expansion and its destruction; nothing in it points into the engine.

Field names and meanings are the C API's (engine README, "Expansion"): boards are indices
in the canonical order (ascending timeline, then half-turn); per-square arrays are indexed
by **cells**, ``rank * width + file`` - the network's square index; moves carry both the
engine's squares (``rank * 8 + file``) and cells. ``python/t5d/engine/convert.py`` turns an
expansion into the network's ``PositionState``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

import numpy as np

from ..net.state import PIECE_LETTERS
from .capi import (
    MOVE_DTYPE,
    T5D_FEATURES_ALL,
    T5D_FEATURES_NONE,
    T5D_FEATURES_SCOPE,
    T5D_MOVE_BRANCHING,
    T5D_MOVE_CAPTURE,
    T5D_MOVE_CASTLE,
    T5D_MOVE_DOUBLE_STEP,
    T5D_MOVE_EN_PASSANT,
    T5D_MOVE_PROMOTION,
    T5D_MOVE_TRAVEL,
    CExpansion,
    MoveRecord,
)
from .errors import EngineInternalError

#: ``square_features`` by name (``t5d_expand_options.square_features``).
SQUARE_FEATURES: dict[str, int] = {"all": T5D_FEATURES_ALL, "scope": T5D_FEATURES_SCOPE, "none": T5D_FEATURES_NONE}

_FLAG_NAMES = (
    (T5D_MOVE_CAPTURE, "capture"),
    (T5D_MOVE_BRANCHING, "branching"),
    (T5D_MOVE_CASTLE, "castle"),
    (T5D_MOVE_EN_PASSANT, "en_passant"),
    (T5D_MOVE_DOUBLE_STEP, "double_step"),
    (T5D_MOVE_PROMOTION, "promotion"),
    (T5D_MOVE_TRAVEL, "travel"),
)


def timeline_label(p: int, even: bool) -> str:
    """The notation's label of timeline position ``p`` (the inverse of
    ``state.timeline_index``): odd games ``0``, ``+1``, ``-1``; even games ``-0`` (p = -1),
    ``+0`` (p = 0), ``-n`` (p = -n - 1), ``+n``."""
    if even:
        return f"+{p}" if p >= 0 else f"-{-p - 1}"
    return "0" if p == 0 else (f"+{p}" if p > 0 else str(p))


def board_label(p: int, h: int, even: bool) -> str:
    """``(<timeline>T<turn>)``: the board at timeline ``p``, half-turn ``h`` (T = h // 2)."""
    return f"({timeline_label(p, even)}T{h // 2})"


def square_name(square: int) -> str:
    """The engine's square index (``rank * 8 + file``) as a name: 0 -> a1."""
    return f"{'abcdefgh'[square % 8]}{square // 8 + 1}"


@dataclass(frozen=True)
class Move:
    """One move of an expansion (``t5d_move``, ``t5d/records.h``)."""

    from_timeline: int
    from_half_turn: int
    to_timeline: int  # for a branching move: the history board it targets, not the board it creates
    to_half_turn: int
    from_board: int  # board indices in the expansion's canonical order
    to_board: int
    from_square: int  # rank * 8 + file
    to_square: int
    from_cell: int  # rank * width + file
    to_cell: int
    piece: int  # type id 1..12 (P N B R Q K U D S W C Y)
    captured: int  # 0, or the captured piece's type id
    promotion: int  # 0, or the type id promoted to
    flags: int  # T5D_MOVE_* bits

    @classmethod
    def from_record(cls, record) -> Move:
        """From a numpy ``MOVE_DTYPE`` record or a ctypes ``MoveRecord``."""
        if isinstance(record, MoveRecord):
            return cls(**{name: int(getattr(record, name)) for name, _ in MoveRecord._fields_})
        return cls(**{name: int(record[name]) for name in MOVE_DTYPE.names})

    def to_c(self) -> MoveRecord:
        return MoveRecord(**{name: getattr(self, name) for name, _ in MoveRecord._fields_})

    def has(self, flag: int) -> bool:
        return bool(self.flags & flag)

    @property
    def flag_names(self) -> tuple[str, ...]:
        return tuple(name for bit, name in _FLAG_NAMES if self.flags & bit)

    def text(self, even: bool = False) -> str:
        """The 5dpgn export form with the piece letter, as the engine's messages write it:
        ``(0T1)Pe2(0T1)e4``, ``=Q`` after a promotion (for messages and tests)."""
        src = board_label(self.from_timeline, self.from_half_turn, even)
        dst = board_label(self.to_timeline, self.to_half_turn, even)
        promo = f"={PIECE_LETTERS[self.promotion]}" if self.promotion else ""
        return f"{src}{PIECE_LETTERS[self.piece]}{square_name(self.from_square)}{dst}{square_name(self.to_square)}{promo}"


def _copy(pointer, n: int, dtype, what: str) -> np.ndarray:
    """``n`` items of an engine-owned array, copied (an empty array may be NULL)."""
    if n == 0:
        return np.zeros(0, dtype=dtype)
    if not pointer:
        raise EngineInternalError(f"the expansion's {what} is NULL with {n} entries")
    return np.ctypeslib.as_array(pointer, shape=(n,)).astype(dtype, copy=True)


@dataclass(frozen=True, eq=False)
class Expansion:
    """A copy of one ``t5d_expansion`` (see the module docstring). Every array is owned by
    this object; per-board arrays have length ``board_count``."""

    width: int
    height: int
    even: bool
    to_move: int  # 0 white, 1 black
    present: int  # the present (T4), a half-turn
    timelines: int
    active_timelines: int
    window_k: int
    may_move: bool  # a move may be made now (always by default, R8; under Q4's Guide-literal reading only before the shift)
    # the action (register D8; engine stage E7) - global, not per board
    acting_player: int  # 0 white, 1 black: whose action this is (= to_move); once the present has shifted it differs from present & 1
    submit_possible: bool  # the action's obligations are met (T5): submitting is an action now
    submit_legal: bool  # ... and no royal of the acting player's capturable (C4, R1): the submit action's legality
    origins_unknown: int  # active branched timelines with no recorded origin (5DFEN)
    origin_count: int
    # per board (canonical order)
    timeline: np.ndarray  # int32: timeline position p ("-0" -> -1, "+0" -> 0)
    half_turn: np.ndarray  # int32: h = 2T + (1 if black to move on the board)
    is_head: np.ndarray  # bool
    active: np.ndarray  # bool: its timeline is active (T10)
    side_to_move: np.ndarray  # uint8: 0 white, 1 black
    playable: np.ndarray  # bool: the player to move may move from it now
    window_depth: np.ndarray  # int32: least k with the board in W(k); T5D_NO_WINDOW: none
    band_depth: np.ndarray  # int32: the same without the branch origins
    is_origin: np.ndarray  # bool: an active timeline's branch origin
    parent: np.ndarray  # int32: lineage parent board, -1 none known
    # per square: one block of width * height cells per feature board
    feature_boards: np.ndarray  # int32 (F,)
    piece: np.ndarray  # uint8 (F, W*H): 0 empty, 1..12
    colour: np.ndarray  # uint8 (F, W*H): 0 none, 1 white, 2 black
    moved: np.ndarray  # bool (F, W*H)
    # lists
    moves: np.ndarray  # MOVE_DTYPE (M,)
    edges: np.ndarray  # int32 (E, 4): src board, src cell, dst board, dst cell
    relations: np.ndarray  # int32 (R, 5): src board, src cell, dst board, dst cell, relation id
    window: np.ndarray  # int32: W(k), ascending
    e_boards: np.ndarray  # int32: E(k), ascending
    src_extra: np.ndarray  # int32: src_extra(k), ascending
    #: The options it was made with.
    square_features: str = "all"
    relation_bits: int = 0
    _move_cache: list | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_c(cls, x: CExpansion, square_features: str = "all", relation_bits: int = 0) -> Expansion:
        """Copy every array of a filled ``t5d_expansion`` (before the handle changes)."""
        n = int(x.board_count)
        wh = int(x.width) * int(x.height)
        nf = int(x.feature_board_count)
        nm = int(x.move_count)
        if nm:
            if not x.moves:
                raise EngineInternalError(f"the expansion's moves are NULL with {nm} entries")
            raw = ctypes.string_at(ctypes.cast(x.moves, ctypes.c_void_p), nm * MOVE_DTYPE.itemsize)
            moves = np.frombuffer(raw, dtype=MOVE_DTYPE).copy()
        else:
            moves = np.zeros(0, dtype=MOVE_DTYPE)
        return cls(
            width=int(x.width),
            height=int(x.height),
            even=bool(x.even),
            to_move=int(x.to_move),
            present=int(x.present),
            timelines=int(x.timelines),
            active_timelines=int(x.active_timelines),
            window_k=int(x.window_k),
            may_move=bool(x.may_move),
            acting_player=int(x.acting_player),
            submit_possible=bool(x.submit_possible),
            submit_legal=bool(x.submit_legal),
            origins_unknown=int(x.origins_unknown),
            origin_count=int(x.origin_count),
            timeline=_copy(x.timeline, n, np.int32, "timeline"),
            half_turn=_copy(x.half_turn, n, np.int32, "half_turn"),
            is_head=_copy(x.is_head, n, bool, "is_head"),
            active=_copy(x.active, n, bool, "active"),
            side_to_move=_copy(x.side_to_move, n, np.uint8, "side_to_move"),
            playable=_copy(x.playable, n, bool, "playable"),
            window_depth=_copy(x.window_depth, n, np.int32, "window_depth"),
            band_depth=_copy(x.band_depth, n, np.int32, "band_depth"),
            is_origin=_copy(x.is_origin, n, bool, "is_origin"),
            parent=_copy(x.parent, n, np.int32, "parent"),
            feature_boards=_copy(x.feature_boards, nf, np.int32, "feature_boards"),
            piece=_copy(x.piece, nf * wh, np.uint8, "piece").reshape(nf, wh),
            colour=_copy(x.colour, nf * wh, np.uint8, "colour").reshape(nf, wh),
            moved=_copy(x.moved, nf * wh, bool, "moved").reshape(nf, wh),
            moves=moves,
            edges=_copy(x.edges, 4 * int(x.edge_count), np.int32, "edges").reshape(-1, 4),
            relations=_copy(x.relations, 5 * int(x.relation_count), np.int32, "relations").reshape(-1, 5),
            window=_copy(x.window, int(x.window_count), np.int32, "window"),
            e_boards=_copy(x.e_boards, int(x.e_count), np.int32, "e_boards"),
            src_extra=_copy(x.src_extra, int(x.src_extra_count), np.int32, "src_extra"),
            square_features=square_features,
            relation_bits=int(relation_bits),
        )

    # -- convenience --

    @property
    def board_count(self) -> int:
        return int(self.timeline.shape[0])

    @property
    def move_count(self) -> int:
        return int(self.moves.shape[0])

    @property
    def all_features(self) -> bool:
        """Whether every board has its per-square features (the full-history input)."""
        return self.feature_boards.shape[0] == self.board_count and bool(np.all(self.feature_boards == np.arange(self.board_count)))

    def move(self, index: int) -> Move:
        return Move.from_record(self.moves[index])

    def move_list(self) -> list[Move]:
        """Every move as a ``Move``, in the engine's order (cached)."""
        if self._move_cache is None:
            object.__setattr__(self, "_move_cache", [Move.from_record(r) for r in self.moves])
        return list(self._move_cache)

    def move_texts(self) -> list[str]:
        return [m.text(self.even) for m in self.move_list()]

    def board_index(self, timeline: int, half_turn: int) -> int:
        """The index of the board at ``(timeline, half_turn)``, or -1."""
        hit = np.flatnonzero((self.timeline == timeline) & (self.half_turn == half_turn))
        return int(hit[0]) if hit.size else -1
