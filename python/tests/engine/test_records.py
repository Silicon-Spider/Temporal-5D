"""Stage N4: every position of a 5dpgn record as the network's input.

The records of ``helpers.RECORDS`` always run (with the engine). Other record files run when
the environment variable ``T5D_RECORDS`` names them (a directory, searched for ``*.5dpgn``,
or files; ``os.pathsep`` between entries) - e.g. the gate's corpus, read as input data.
Each record is replayed as the gate's parse mode does (pseudo-legal submissions), every
position is converted and checked against the engine (``helpers.check_position``), and the
replay stops at the first action the engine refuses (a record the engine reads differently,
or - under Q4's Guide-literal reading, no longer the default since engine stage E7 - one
that moves after the present has shifted) - that is the gate's business, not this test's.
"""

from __future__ import annotations

import pytest

from t5d.engine import EngineError, Game, NotationError, UnsupportedError
from tests.engine.helpers import RECORDS, check_position, corpus_paths, requires_engine

pytestmark = requires_engine


def _positions_of(game: Game, label: str) -> tuple[int, str | None]:
    """Check every position of a record; (positions checked, why it stopped early or None)."""
    check_position(game, f"{label} position 0")
    count = 1
    while game.record_actions().remaining:
        try:
            game.replay_next()
        except EngineError as exc:
            return count, f"{type(exc).__name__}: {exc.message}"
        count += 1
        check_position(game, f"{label} position {game.record_actions().played}")
    return count, None


@pytest.mark.parametrize("key", sorted(RECORDS))
def test_every_position_of_our_records(key):
    with Game.from_pgn(RECORDS[key]) as game:  # legal replay: these records are legal
        total = game.record_actions().total
        count, stopped = _positions_of(game, key)
    assert stopped is None and count == total + 1


def test_the_branching_record_carries_its_origin():
    with Game.from_pgn(RECORDS["branching"]) as game:
        game.replay_all()
        pos = check_position(game, "branching")
    s = pos.state
    first = [b for b in range(s.n_boards) if s.timeline[b] == 1 and s.half_turn[b] == s.half_turn[s.timeline == 1].min()][0]
    origin = int(s.parent[first])
    assert (int(s.timeline[origin]), int(s.half_turn[origin])) == (0, 2) and not s.is_head[origin]
    assert pos.lineage_complete and pos.scope.is_origin[origin]


_CORPUS = corpus_paths()


@pytest.mark.skipif(not _CORPUS, reason="set T5D_RECORDS to record files or directories to run every position of them")
@pytest.mark.parametrize("path", _CORPUS, ids=[p.name for p in _CORPUS])
def test_every_position_of_a_named_record(path):
    try:
        game = Game.from_pgn_file(path, legal=False)
    except UnsupportedError as exc:
        pytest.skip(f"the engine does not play it: {exc.message}")
    except NotationError as exc:
        pytest.skip(f"the engine cannot read its set-up: {exc.message}")
    with game:
        count, _stopped = _positions_of(game, path.name)
    assert count >= 1
