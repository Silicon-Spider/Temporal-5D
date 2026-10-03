"""A game handle: ``Game`` wraps one ``t5d_game`` of the engine's C API (``t5d/capi.h``).

A game is a state under the rules - the player to move, the present, the action in
progress - with its expansion buffers. It is made from a 5dpgn record (set up at its first
position, then replayed action by action), from 5DFEN board strings, or from a named
variant; it is moved (``apply``), submitted (``submit``) and expanded (``expand``).

**Moves are pseudo-legal; submissions are legal** (engine README, "Status"): the moves an
expansion offers - ``moves()`` lists them, in the engine's order - are the pseudo-legal
moves of the player to move; a submission that leaves a royal piece of the mover's
capturable is refused (``IllegalError``) unless ``set_legality(False)``. The engine's
legal-moves filter (``ExpandOptions::legal_moves``) is not in the C API yet.

**Lifetimes.** The handle is released by ``close()`` (or the context manager, or garbage
collection). ``expand()`` copies the expansion out at once (``Expansion.from_c``), so what
it returns stays valid after the handle changes. A handle is used from one thread at a
time (the C API's rule); this class serialises its own calls with a lock.

Errors raise the exception of the C API's status (``errors.py``) with the engine's message.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
from collections.abc import Iterable, Iterator, Sequence
from ctypes import byref, c_char_p, c_int32, c_size_t, c_uint64, c_void_p
from dataclasses import dataclass

from .capi import (
    BUILT_RELATIONS,
    RESERVED_RELATIONS,
    T5D_LEGAL,
    T5D_PARITY_EVEN,
    T5D_PARITY_INFER,
    T5D_PARITY_ODD,
    T5D_PSEUDO_LEGAL,
    T5D_TO_MOVE_BLACK,
    T5D_TO_MOVE_INFER,
    T5D_TO_MOVE_WHITE,
    CExpansion,
    ExpandOptions,
    Fingerprint,
    GameSummary,
    Library,
    StateInfo,
    char_array,
    load_library,
)
from .convert import Position, to_position
from .errors import EngineArgumentError
from .expansion import SQUARE_FEATURES, Expansion, Move

_PARITY = {"infer": T5D_PARITY_INFER, "odd": T5D_PARITY_ODD, "even": T5D_PARITY_EVEN}
_TO_MOVE = {"infer": T5D_TO_MOVE_INFER, "white": T5D_TO_MOVE_WHITE, "black": T5D_TO_MOVE_BLACK}


def relation_bits(relations: Iterable[str | int]) -> int:
    """``t5d_expand_options.relations`` from relation names or ids. Names the network's
    contract has but the engine does not build (pin, check, travel_threat) are passed on,
    for the engine to refuse with its own message (``UnsupportedError``)."""
    known = {**BUILT_RELATIONS, **RESERVED_RELATIONS}
    bits = 0
    for r in relations:
        if isinstance(r, str) and r not in known:
            raise ValueError(f"unknown relation {r!r}: expected one of {sorted(known)}")
        bits |= 1 << int(known[r] if isinstance(r, str) else r)
    return bits


def _choice(value, table: dict[str, int], what: str) -> int:
    if isinstance(value, str):
        try:
            return table[value.lower()]
        except KeyError:
            raise ValueError(f"{what} must be one of {sorted(table)} (or the C constant), not {value!r}") from None
    return int(value)


@dataclass(frozen=True)
class GameInfo:
    """``t5d_game_summary``."""

    boards: int
    width: int
    height: int
    even: bool
    to_move: int  # 0 white, 1 black
    present: int  # the present (T4): a half-turn
    timelines: int
    active_timelines: int
    obligations_met: bool  # the present has shifted to the opponent (T5)
    may_move: bool  # a move may be made now (always by default, R8; Q4's Guide-literal reading: only before the shift)
    may_submit: bool  # the present shifted (T5) and every royal is safe (C4)
    ruleset: int
    rules: int


@dataclass(frozen=True)
class RecordProgress:
    """``t5d_game_record_actions``: the record's actions, how many are replayed, and whether
    its move text stopped early (unreadable from there on)."""

    total: int
    played: int
    truncated: bool

    @property
    def remaining(self) -> int:
        return self.total - self.played


@dataclass(frozen=True)
class StateSnapshot:
    """The game's multiverse as the ``t5d_state`` functions read it (``t5d_game_state``)."""

    fingerprint: str  # H30: MD5 of the canonical state string
    placement: str  # the same without '*'
    zobrist: int  # the 5D Zobrist key (H18)
    canonical: str  # the canonical state string
    boards: tuple[tuple[int, int, str], ...]  # per board, canonical order: (timeline p, half-turn h, 5DFEN board string)
    pieces: int
    width: int
    height: int
    even: bool
    to_move: int
    min_timeline: int
    max_timeline: int
    created_by_white: int  # timelines created by each player (5DFEN: by label, L7)
    created_by_black: int
    ruleset: int


class Game:
    """One engine game handle (see the module docstring). Make one with ``from_pgn``,
    ``from_fen`` or ``variant``."""

    def __init__(self, handle: c_void_p, lib: Library, *, description: str = "") -> None:
        if not handle:
            raise EngineArgumentError("a game needs a non-NULL handle")
        self._lib = lib
        self._handle = handle
        self._lock = threading.RLock()
        self.description = description

    # -- construction --

    @classmethod
    def from_pgn(cls, text: str | bytes, rules: int = 0, *, legal: bool = True, lib: Library | None = None) -> Game:
        """A game from a 5dpgn record (its headers name the variant or give 5DFEN boards), at
        its first position; ``replay_next`` plays its actions. ``rules``: the ruleset encoding
        (0: standard; the record's Promotions header and variant family are added).
        ``legal=False`` replays (and submits) pseudo-legally, as the gate's parse mode does."""
        lib = load_library() if lib is None else lib
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        out = c_void_p()
        lib.call("t5d_game_new_from_pgn", data, len(data), c_uint64(int(rules)), byref(out), error=True)
        game = cls(out, lib, description="record")
        if not legal:
            game.set_legality(False)
        return game

    @classmethod
    def from_pgn_file(cls, path: str | os.PathLike, rules: int = 0, *, legal: bool = True, lib: Library | None = None) -> Game:
        with open(path, "rb") as f:
            data = f.read()
        game = cls.from_pgn(data, rules, legal=legal, lib=lib)
        game.description = os.fspath(path)
        return game

    @classmethod
    def from_fen(
        cls,
        boards: Sequence[str] | str,
        *,
        width: int = 0,
        height: int = 0,
        parity: str | int = "infer",
        to_move: str | int = "infer",
        rules: int = 0,
        lib: Library | None = None,
    ) -> Game:
        """A game from 5DFEN board strings (``[pieces:timeline:turn:colour]``; a single string
        may hold several, separated by white space). ``width``/``height`` 0: from the first
        board. ``parity``: "odd", "even" or "infer" (from the labels). ``to_move``: "white",
        "black" or "infer" - the colour to move on the present (Q9). 5DFEN records no branch
        origins and no creation order: a branched timeline's first board has no parent."""
        lib = load_library() if lib is None else lib
        if isinstance(boards, str):
            boards = boards.split()
        strings = list(boards)
        array = char_array(strings)
        out = c_void_p()
        lib.call(
            "t5d_game_new_from_fen",
            ctypes.cast(array, ctypes.POINTER(c_char_p)),
            len(strings),
            int(width),
            int(height),
            _choice(parity, _PARITY, "parity"),
            _choice(to_move, _TO_MOVE, "to_move"),
            c_uint64(int(rules)),
            byref(out),
            error=True,
        )
        return cls(out, lib, description="5DFEN")

    @classmethod
    def variant(cls, name: str = "Standard", rules: int = 0, *, lib: Library | None = None) -> Game:
        """A named variant's start ("Standard", "Standard - Two Timelines", ...)."""
        lib = load_library() if lib is None else lib
        out = c_void_p()
        lib.call("t5d_game_new_variant", name.encode("utf-8"), c_uint64(int(rules)), byref(out), error=True)
        return cls(out, lib, description=name)

    def copy(self) -> Game:
        """An independent copy (sharing the immutable boards; not the expansion)."""
        with self._lock:
            out = c_void_p()
            self._lib.call("t5d_game_copy", self._live(), byref(out))
            return Game(out, self._lib, description=self.description)

    # -- lifetime --

    def _live(self) -> c_void_p:
        if self._handle is None:
            raise EngineArgumentError("the game has been closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            self._lib.fn.t5d_game_destroy(handle)

    def __enter__(self) -> Game:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):  # interpreter shutdown: the library may already be gone
            self.close()

    def __repr__(self) -> str:
        if self.closed:
            return f"Game({self.description!r}, closed)"
        i = self.info()
        return f"Game({self.description!r}, boards={i.boards}, timelines={i.timelines}, to_move={'wb'[i.to_move]}, present={i.present})"

    @property
    def lib(self) -> Library:
        return self._lib

    # -- reading --

    def info(self) -> GameInfo:
        with self._lock:
            s = GameSummary()
            self._lib.call("t5d_game_info", self._live(), byref(s))
        return GameInfo(
            boards=int(s.boards),
            width=int(s.width),
            height=int(s.height),
            even=bool(s.even),
            to_move=int(s.to_move),
            present=int(s.present),
            timelines=int(s.timelines),
            active_timelines=int(s.active_timelines),
            obligations_met=bool(s.obligations_met),
            may_move=bool(s.may_move),
            may_submit=bool(s.may_submit),
            ruleset=int(s.ruleset),
            rules=int(s.rules),
        )

    def record_actions(self) -> RecordProgress:
        with self._lock:
            total, played, truncated = c_size_t(), c_size_t(), c_int32()
            self._lib.call("t5d_game_record_actions", self._live(), byref(total), byref(played), byref(truncated))
        return RecordProgress(int(total.value), int(played.value), bool(truncated.value))

    def snapshot(self) -> StateSnapshot:
        """The multiverse as ``t5d_state`` reads it: fingerprint, canonical string, Zobrist
        key, summary and every board's (timeline, half-turn, 5DFEN board string)."""
        lib = self._lib
        with self._lock:
            state = c_void_p()
            lib.call("t5d_game_state", self._live(), byref(state))
        try:
            fp = Fingerprint()
            lib.call("t5d_state_fingerprint", state, byref(fp))
            info = StateInfo()
            lib.call("t5d_state_info", state, byref(info))
            key = c_uint64()
            lib.call("t5d_state_zobrist", state, byref(key))
            text = c_void_p()
            lib.call("t5d_state_canonical", state, byref(text))
            canonical = lib.take_string(text)
            count = c_size_t()
            lib.call("t5d_state_board_count", state, byref(count))
            boards = []
            for i in range(int(count.value)):
                p, h, fen = c_int32(), c_int32(), c_void_p()
                lib.call("t5d_state_board", state, c_size_t(i), byref(p), byref(h), byref(fen))
                boards.append((int(p.value), int(h.value), lib.take_string(fen)))
        finally:
            lib.fn.t5d_state_destroy(state)
        return StateSnapshot(
            fingerprint=fp.fingerprint.decode("ascii"),
            placement=fp.placement.decode("ascii"),
            zobrist=int(key.value),
            canonical=canonical,
            boards=tuple(boards),
            pieces=int(info.pieces),
            width=int(info.width),
            height=int(info.height),
            even=bool(info.even),
            to_move=int(info.to_move),
            min_timeline=int(info.min_timeline),
            max_timeline=int(info.max_timeline),
            created_by_white=int(info.created_by_white),
            created_by_black=int(info.created_by_black),
            ruleset=int(info.ruleset),
        )

    def created_timelines(self) -> int:
        """Timelines with a creator (``t5d_state_info``): created by a move, or read from
        5DFEN with a non-central label (which the engine reads as created by the colour of
        its sign, L7 - Timeline Battleground's outer initial timelines included)."""
        lib = self._lib
        with self._lock:
            state = c_void_p()
            lib.call("t5d_game_state", self._live(), byref(state))
        try:
            info = StateInfo()
            lib.call("t5d_state_info", state, byref(info))
        finally:
            lib.fn.t5d_state_destroy(state)
        return int(info.created_by_white) + int(info.created_by_black)

    # -- playing --

    def set_legality(self, legal: bool) -> None:
        """What submissions (and a record's replay) are tested for: legal (the default: the
        present's shift and every royal of the mover's safe, C4/R1) or pseudo-legal (the
        present's shift only - for pseudo-legal play such as random or synthetic games)."""
        with self._lock:
            self._lib.call("t5d_game_set_legality", self._live(), T5D_LEGAL if legal else T5D_PSEUDO_LEGAL, error=True)

    def replay_next(self) -> None:
        """Replay the record's next action (its moves, then the submission)."""
        with self._lock:
            self._lib.call("t5d_game_replay_next", self._live(), error=True)

    def replay(self) -> Iterator[int]:
        """Replay the record's remaining actions one at a time, yielding the number played
        after each (the game is the position after that action when the value is yielded)."""
        while True:
            progress = self.record_actions()
            if progress.played >= progress.total:
                return
            self.replay_next()
            yield progress.played + 1

    def replay_all(self) -> int:
        """Replay every remaining action; the number replayed."""
        return sum(1 for _ in self.replay())

    def apply(self, move: Move | int) -> None:
        """Apply one move of the player to move: a ``Move`` (matched by source, destination and
        promotion against the generated moves) or an index into the handle's last expansion,
        which must be of the current state."""
        with self._lock:
            if isinstance(move, Move):
                record = move.to_c()
                self._lib.call("t5d_game_apply", self._live(), byref(record), error=True)
            else:
                self._lib.call("t5d_game_apply_expanded", self._live(), c_size_t(int(move)), error=True)

    def submit(self) -> None:
        """Submit the action (T5)."""
        with self._lock:
            self._lib.call("t5d_game_submit", self._live(), error=True)

    # -- expanding --

    def expand(self, *, window_k: int = 8, square_features: str | int = "all", relations: Iterable[str | int] = ()) -> Expansion:
        """Expand the current state and copy the result out (``Expansion``). ``window_k``: k
        of W(k), E(k) and src_extra(k), in half-turns. ``square_features``: "all" (every
        board - what a ``PositionState`` needs), "scope" (W + E + src_extra) or "none".
        ``relations``: typed relations to build - "attack", "defence", "xray" (provisional,
        B11); "pin", "check" and "travel_threat" are refused by the engine."""
        features = _choice(square_features, SQUARE_FEATURES, "square_features")
        names = {v: k for k, v in SQUARE_FEATURES.items()}
        bits = relation_bits(relations)
        options = ExpandOptions(int(window_k), features, bits)
        out = CExpansion()
        with self._lock:
            self._lib.call("t5d_game_expand", self._live(), byref(options), byref(out), error=True)
            return Expansion.from_c(out, names.get(features, str(features)), bits)

    def moves(self) -> list[Move]:
        """The moves the engine offers the player to move now, in its order - pseudo-legal
        (see the module docstring); empty once no move may be made (Q4)."""
        return self.expand(square_features="none").move_list()

    def position(
        self, *, regs_per_board: int = 2, global_regs: int = 16, relations: Iterable[str | int] = (), window_k: int = 8, strict: bool = True
    ) -> Position:
        """The current state as the network's input (``convert.Position``): expanded with
        every board's features, converted, validated, with the W/E scope at ``window_k``
        and the count of unknown branch origins over every timeline. ``relations``: the typed
        relations to put in the edges (``convert.relations_for(cfg)`` gives a network's);
        ``strict``: see ``convert.to_position``."""
        x = self.expand(window_k=window_k, square_features="all", relations=relations)
        created = self.created_timelines()
        return to_position(x, regs_per_board=regs_per_board, global_regs=global_regs, created_timelines=created, strict=strict)


__all__ = ["Game", "GameInfo", "RecordProgress", "StateSnapshot", "relation_bits"]
