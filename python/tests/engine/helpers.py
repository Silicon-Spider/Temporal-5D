"""Shared test helpers for the engine binding (stage N4).

* ``requires_engine``: a test that needs the engine's shared library (see
  ``t5d/engine/capi.py`` for where it is looked for). Through the ``engine_required``
  fixture (``tests/conftest.py``): it runs when the library loads; it **fails** when a library
  was found or named but is refused - not a library, without the C API, or laid out otherwise
  than the binding (a mirror bug must never turn into skipped tests) - or when
  ``T5D_REQUIRE_ENGINE=1`` is set (CI's network job sets it) and no library is built; it is
  skipped only when no library is built and none is required.
* ``RECORDS``: short 5dpgn records written here, from the notation's move forms - a
  branching jump, a non-branching cross-timeline jump, a turn-zero start. They are this
  project's own data.
* ``decode_board``: a 5DFEN board string (``[rows:timeline:turn:colour]``) as the network's
  per-cell arrays - an independent reading of the engine's canonical board strings, to
  check the expansion's square features against.
* ``corpus_paths``: opt-in record files named by the ``T5D_RECORDS`` environment variable
  (directories are searched for ``*.5dpgn``; several entries separated by ``os.pathsep``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from t5d.engine import library_status
from t5d.net.state import COLOUR_BLACK, COLOUR_WHITE, PIECE_IDS

#: The variable that makes a missing engine library a failure (CI sets it).
REQUIRE_VAR = "T5D_REQUIRE_ENGINE"
ENGINE_STATUS, ENGINE_PROBLEM = library_status()
ENGINE = ENGINE_STATUS == "available"
requires_engine = pytest.mark.usefixtures("engine_required")


def engine_verdict(status: str, problem: str, required: bool) -> tuple[str, str]:
    """What a test that needs the engine does: ``("run", "")``, ``("fail", why)`` or
    ``("skip", why)`` - see the module docstring."""
    if status == "available":
        return "run", ""
    if status == "refused":
        return "fail", f"the engine library was found but refused: {problem}"
    if required:
        return "fail", f"{REQUIRE_VAR}=1 and the engine library is not available: {problem}"
    return "skip", "the engine library is not built (py engine/build.py)"


def engine_required() -> None:
    """Run, fail or skip the calling test (``engine_verdict`` of this session's library)."""
    action, why = engine_verdict(ENGINE_STATUS, ENGINE_PROBLEM, os.environ.get(REQUIRE_VAR, "").strip() == "1")
    if action == "fail":
        pytest.fail(why, pytrace=False)
    if action == "skip":
        pytest.skip(why)


#: Kinds that carry the unmoved marker (the notation's "5DFEN Strictness" list).
TRACKED = set("PRKW")

RECORDS: dict[str, str] = {
    # A knight jumps back a turn onto a history board and branches timeline +1 (J3, L7);
    # play continues on both timelines.
    "branching": """[Board "Standard"]
[Mode "5D"]

1. Nf3 / Nf6
2. (0T2)Nf3>>(0T1)f5 / (1T1)e6
3. (1T2)Nxg7 / (0T2)e6 (1T2)Bxg7
""",
    # Two Timelines: a knight jumps from -0 onto +0's head in the same turn - no branch,
    # both boards played (J4).
    "cross_timeline": """[Board "Standard - Two Timelines"]
[Mode "5D"]

1. (-0T1)e4 (+0T1)d4 / (-0T1)e5 (+0T1)d5
2. (-0T2)Ng1>(+0T2)g3 / (-0T2)Nc6 (+0T2)Nf6
""",
    # Turn Zero: a history board from the start.
    "turn_zero": """[Board "Standard - Turn Zero"]
[Mode "5D"]

1. e4 / e5
2. Nf3 / Nc6
""",
}

#: Variants the engine sets up, by board size (engine README, "The variants").
VARIANTS = (
    "Standard",
    "Standard - Two Timelines",
    "Standard - Turn Zero",
    "Standard - Reversed Royalty",
    "Very Small - Open",
    "Small",
    "Focused - Just Kings",
    "Focused - Just Brawns",
    "Misc - Timeline Battleground",
    "Misc - Timeline Strategos",
)

#: Every named variant the engine sets up - its whole table (``engine/src/variants.cpp``;
#: engine README, "The variants"), each once (``Misc - Small`` is ``Small``'s alias).
ENGINE_VARIANTS = (
    "Standard",
    "Standard - Turn Zero",
    "Standard - Two Timelines",
    "Standard - Half Reflected",
    "Standard - Defended Pawn",
    "Standard - Reversed Royalty",
    "Very Small - Open",
    "Very Small",
    "Small",
    "Focused - Just Kings",
    "Focused - Just Pawns",
    "Focused - Just Brawns",
    "Misc - Timeline Battleground",
    "Misc - Timeline Tactician",
    "Misc - Timeline Strategos",
    "Misc - Timeline Formations",
)


def decode_board(text: str, width: int, height: int) -> dict:
    """A 5DFEN board string as per-cell arrays (cell = rank * width + file) and its fields."""
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"not a board string: {text!r}")
    rows_text, timeline, turn, colour = text[1:-1].split(":")
    piece = np.zeros(width * height, dtype=np.int64)
    col = np.zeros(width * height, dtype=np.int64)
    moved = np.zeros(width * height, dtype=bool)
    rows = rows_text.split("/")
    if len(rows) != height:
        raise ValueError(f"{len(rows)} rows on a board {height} high")
    for r, row in enumerate(rows):
        rank = height - 1 - r  # rows are written from the top
        file, i = 0, 0
        while i < len(row):
            ch = row[i]
            if ch.isdigit():
                j = i
                while j < len(row) and row[j].isdigit():
                    j += 1
                file += int(row[i:j])
                i = j
                continue
            if ch == "+":
                i += 1
                continue
            letter = ch.upper()
            unmoved = i + 1 < len(row) and row[i + 1] == "*"
            cell = rank * width + file
            piece[cell] = PIECE_IDS[letter]
            col[cell] = COLOUR_WHITE if ch.isupper() else COLOUR_BLACK
            moved[cell] = letter in TRACKED and not unmoved
            file += 1
            i += 2 if unmoved else 1
        if file != width:
            raise ValueError(f"row {row!r} is {file} squares wide, not {width}")
    return {"piece": piece, "colour": col, "moved": moved, "timeline": timeline, "turn": int(turn), "colour_to_move": colour}


def corpus_paths() -> list[Path]:
    """Record files named by ``T5D_RECORDS`` (empty when it is unset)."""
    raw = os.environ.get("T5D_RECORDS", "").strip()
    out: list[Path] = []
    for entry in filter(None, raw.split(os.pathsep)):
        p = Path(entry)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.5dpgn")))
        elif p.is_file():
            out.append(p)
    return out


def check_position(game, label: str = "", *, relations=("attack", "defence", "pin", "xray", "check")):
    """Convert ``game``'s current position and check the result against the engine itself:
    coordinates and flags, every board's square features against an independent decoding of
    the engine's canonical 5DFEN board string, the candidate moves against the engine's move
    list (its order), the edges against the move graph and the relations, the lineage. Returns
    the ``Position``."""
    from t5d.engine import to_position
    from t5d.engine.expansion import timeline_label

    x = game.expand(relations=relations)
    pos = to_position(x, created_timelines=game.created_timelines())
    s, snap = pos.state, game.snapshot()
    tpb = s.width * s.height + s.regs_per_board
    # boards: coordinates and flags
    assert len(snap.boards) == s.n_boards, label
    assert s.timeline.tolist() == [b[0] for b in snap.boards] and s.half_turn.tolist() == [b[1] for b in snap.boards], label
    assert np.array_equal(s.is_head, x.is_head) and np.array_equal(s.active, x.active), label
    assert np.array_equal(s.side_to_move, s.half_turn & 1), label
    assert s.h_present == x.present == game.info().present and s.acting_player == x.acting_player == x.to_move, label
    assert s.shifted == x.submit_possible == game.info().obligations_met and s.submit_legal == x.submit_legal, label
    assert np.allclose(s.dt_present, s.half_turn - x.present), label
    # squares: the engine's canonical 5DFEN board strings, decoded independently
    for b, (p, h, text) in enumerate(snap.boards):
        d = decode_board(text, s.width, s.height)
        assert d["timeline"] == timeline_label(p, snap.even) and d["turn"] == h // 2 and d["colour_to_move"] == "wb"[h & 1], (label, b)
        assert np.array_equal(s.piece[b], d["piece"]), (label, b, text)
        assert np.array_equal(s.colour[b], d["colour"]), (label, b, text)
        assert np.array_equal(s.moved[b], d["moved"]), (label, b, text)
    # candidate moves: the engine's list, in its order
    m = x.moves
    assert s.moves.shape == (x.move_count, 3), label
    assert np.array_equal(s.moves[:, 0], m["from_board"].astype(np.int64) * tpb + m["from_cell"]), label
    assert np.array_equal(s.moves[:, 1], m["to_board"].astype(np.int64) * tpb + m["to_cell"]), label
    assert np.array_equal(s.moves[:, 2], m["promotion"]), label
    assert np.array_equal(pos.moves, m), label
    for row in m:  # cells and squares name the same square; the piece is on the source
        for side in ("from", "to"):
            cell, square = int(row[f"{side}_cell"]), int(row[f"{side}_square"])
            assert (cell % s.width, cell // s.width) == (square % 8, square // 8), label
        assert s.piece[row["from_board"], row["from_cell"]] == row["piece"], label
        assert s.colour[row["from_board"], row["from_cell"]] == 1 + pos.to_move, label
    # edges: the move graph - one per distinct (source, destination) pair, in move order - then relations
    graph = s.edges[s.edges[:, 2] == 0]
    assert s.edges[: graph.shape[0], 2].tolist() == [0] * graph.shape[0], label
    pairs = list(dict.fromkeys(map(tuple, s.moves[:, :2].tolist())))
    assert [tuple(e) for e in graph[:, :2].tolist()] == pairs, label
    typed = s.edges[graph.shape[0] :]
    assert typed.shape[0] == x.relations.shape[0] and set(typed[:, 2].tolist()) <= {1, 2, 3, 4, 5}, label
    # lineage: the engine's parent; a branch's first board descends from a superseded board one half-turn back
    assert np.array_equal(s.parent, x.parent), label
    return pos
