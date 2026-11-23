"""Register I22's seeded starts: the stored positions a share of self-play games begins from (spec
section 12, "Coverage across board counts"; the 2026-11-15 review, decisions 22 and 23).
Provisional, and OFF in the generator (``selfplay.SeededStartConfig.enabled``): the pilot run plays
without seeded starts (section 15.7).

"A share of self-play games begins from a stored position instead of the opening: one of the
engine's own longer games, the human games (which reach 186 boards), the audit lane's
contradictions, the league's finds. The game is played out and scored as usual, so every target is
still the engine's own; what changes is that large positions are reached without paying for the
forty moves before them ... A seeded game keeps its source's turn count: the temperature schedule
and the cap read the position's own turn number, not zero" (section 12).

* **A stored position** (``StoredPosition``) is a 5dpgn record and the number of its actions
  replayed. The engine's own reader sets it up (``open_position``: the record read, that many
  actions replayed legally), so the start is exact - branch origins, creation order and unmoved
  flags included - and the handle is then played on (a game made from a record may leave it). The
  second form, 5DFEN boards with a stated turn count, is accepted but lossy: 5DFEN records no
  branch origins (the network's lineage input then reads them as unknown) and no en-passant right
  on a head board.
* **Its turn count** (``StoredPosition.turn_count``) is in actions - one player's turn each (spec
  issue 305), the unit I19's temperature schedules count: the source's actions replayed, unless
  ``turn`` states it (a source record that itself starts late; a 5DFEN position, which has no
  source game to count).
* **Where stored positions come from.** ``positions_from_pgn`` - any 5dpgn the engine's reader
  reads: one position per turn start of the record that passes a board-count filter.
  ``positions_from_record`` - an earlier self-play game's own I15 record (its ``pgn`` and each
  turn start's action index; no engine call). ``write_positions`` / ``read_positions`` keep a pool
  as JSON Lines. What goes in a pool is the caller's: no default pool, minimum board count or
  source is built, and nothing here reads a data set. **The human games are not a source** (the
  user's ruling of 2026-11-23, BUILD_LOG spec issue 734: they are not to be used as training
  data, and a seeded game's start position becomes a training input - so the passage quoted above
  no longer holds for them; ``data/t0database`` stays measurement data, as its README says). A
  pool is built from the engine's own games. **Re-seeding from a seeded game's own record
  is refused**: its exact start plus its own actions is not one 5dpgn (its ``pgn`` is the
  continuation from the start's 5DFEN boards), and Python cannot cut or re-emit a record's move
  text.

The engine's reader is the only reader used; nothing here parses notation, and nothing outside the
engine binding is imported (AGENTS.md).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..engine.capi import Library
from ..engine.errors import IllegalError, NotationError
from ..engine.game import Game
from .records import GameRecord


@dataclass(frozen=True)
class StoredPosition:
    """One stored position (module docstring): ``pgn`` and ``actions``, or ``fen`` and ``turn``."""

    pgn: str | None = None  # a 5dpgn record ...
    actions: int = 0  # ... and the number of its actions replayed (0: the record's own set-up); the start is that turn start
    fen: tuple[str, ...] | None = None  # the lossy form: 5DFEN board strings (no branch origins)
    #: The source's turn count at the position, in actions (one player's turn each, issue 305).
    #: None: ``actions``. Required for a 5DFEN position.
    turn: int | None = None
    source: str = ""  # where it came from, recorded in the seeded game's header
    rules: int = 0  # the 5dpgn's ruleset encoding (``Game.from_pgn``'s ``rules``; 0: standard)
    n_boards: int = 0  # its board count (0: not counted) - what a pool is filtered by

    def __post_init__(self) -> None:
        if (self.pgn is None) == (self.fen is None):
            raise ValueError("I22: a stored position is a 5dpgn record (pgn) or 5DFEN boards (fen), one of the two")
        if self.fen is not None:
            object.__setattr__(self, "fen", tuple(str(b) for b in self.fen))
            if self.turn is None or self.actions != 0 or self.rules != 0:
                raise ValueError("I22: a 5DFEN stored position states its turn count (turn), with actions 0 and rules 0")
        if self.actions < 0 or (self.turn is not None and self.turn < 0) or self.n_boards < 0:
            raise ValueError("I22: actions, turn and n_boards are non-negative")

    @property
    def kind(self) -> str:
        """``"pgn"`` (exact) or ``"fen"`` (lossy)."""
        return "pgn" if self.fen is None else "fen"

    @property
    def turn_count(self) -> int:
        """The source's turn count at the position, in actions (module docstring)."""
        return int(self.actions if self.turn is None else self.turn)


def open_position(position: StoredPosition, *, lib: Library | None = None) -> Game:
    """A game handle at ``position``, set up by the engine: the 5dpgn read and ``actions`` of it
    replayed legally (``ValueError`` when the record has fewer, the engine's own error when an
    action does not replay), or the 5DFEN boards read. At a turn start; the caller plays it on and
    closes it."""
    if position.fen is not None:
        return Game.from_fen(list(position.fen), lib=lib)
    game = Game.from_pgn(position.pgn, position.rules, lib=lib)
    try:
        total = game.record_actions().total
        if total < position.actions:
            raise ValueError(f"I22: the stored position replays {position.actions} actions of a record that has {total}")
        for _ in range(position.actions):
            game.replay_next()
    except BaseException:
        game.close()
        raise
    return game


def _in_range(n_boards: int, min_boards: int, max_boards: int | None) -> bool:
    return n_boards >= min_boards and (max_boards is None or n_boards <= max_boards)


def positions_from_pgn(
    text: str,
    *,
    source: str = "",
    rules: int = 0,
    min_boards: int = 0,
    max_boards: int | None = None,
    include_last: bool = False,
    lib: Library | None = None,
) -> list[StoredPosition]:
    """The stored positions of one 5dpgn record: the turn start after each of its actions whose
    board count is in ``[min_boards, max_boards]`` - the record replayed legally by the engine,
    action by action. The record's own set-up (no action replayed) is not one (make it directly:
    ``StoredPosition(pgn=text)``), and the position after its last action - where the game may be
    over - only with ``include_last``. Replay stops at the first action the engine refuses
    (``IllegalError``, ``NotationError``); the positions before it are kept, and the record is
    never edited."""
    out: list[StoredPosition] = []
    with Game.from_pgn(text, rules, lib=lib) as game:
        total = game.record_actions().total
        for done in range(1, total + (1 if include_last else 0)):
            try:
                game.replay_next()
            except (IllegalError, NotationError):
                break
            n = game.info().boards
            if _in_range(n, min_boards, max_boards):
                out.append(StoredPosition(pgn=text, actions=done, source=source, rules=rules, n_boards=n))
    return out


def positions_from_record(record: GameRecord, *, min_boards: int = 0, max_boards: int | None = None) -> list[StoredPosition]:
    """The stored positions of an earlier self-play game, from its I15 record alone (no engine
    call): each turn start after the first at which the game made a decision, as the record's
    ``pgn`` and that decision's action index, its board count the decision's. ``ValueError`` for a
    record that was itself seeded (module docstring: re-seeding is refused) and for one with no
    5dpgn (a synthetic record)."""
    h = record.header
    if h.get("seeded_start") is not None:
        raise ValueError(
            "I22: a seeded game's own record supplies no stored positions - its exact start plus its own actions is not one "
            "5dpgn (its pgn is the continuation from the start's 5DFEN boards)"
        )
    if not h.get("pgn"):
        raise ValueError("I22: the record carries no 5dpgn to replay (a synthetic record)")
    source = f"selfplay:{h.get('seed')}:{str(h.get('result', {}).get('final_fingerprint', ''))[:8]}"
    out: list[StoredPosition] = []
    for d in record.decisions:
        if d.moves_made == 0 and d.action >= 1 and _in_range(d.n_boards, min_boards, max_boards):
            out.append(StoredPosition(pgn=h["pgn"], actions=int(d.action), source=source, n_boards=int(d.n_boards)))
    return out


def write_positions(path: str | Path, positions: Iterable[StoredPosition]) -> Path:
    """Write a pool as JSON Lines - one object per position, the dataclass's fields; the path."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in positions:
            f.write(json.dumps(asdict(p), sort_keys=True) + "\n")
    return path


def read_positions(path: str | Path) -> list[StoredPosition]:
    """Read a pool written by ``write_positions`` (each position validated as it is made)."""
    with Path(path).open(encoding="utf-8") as f:
        return [StoredPosition(**json.loads(line)) for line in f if line.strip()]


__all__ = ["StoredPosition", "open_position", "positions_from_pgn", "positions_from_record", "read_positions", "write_positions"]
