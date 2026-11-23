"""Register I22's seeded starts in the generator (spec section 12, "Seeded starts"; review
decisions 22 and 23; ``train.seeded``, ``train.selfplay``): a stored position is a 5dpgn record
and the number of its actions replayed, set up by the engine's own reader with its history; one
game in ten takes one, by the game's seed; a seeded game records its origin, rebuilds from its
record, keeps its source's turn count for the temperature schedules, and meets I11's cap at the
position's own turn; re-seeding from a seeded game's own record is refused; and seeding off - the
default: the pilot run plays without it (section 15.7) - leaves a game's record as it was.

The stored positions here are this project's own: the records of ``tests.engine.helpers``, the
engine's test data (``engine/testdata/turn-width``), and games these tests generate. Nothing is
trained."""

from __future__ import annotations

import inspect
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from t5d.engine.evaluator import NetworkEvaluator, UniformEvaluator
from t5d.engine.game import Game
from t5d.engine.netplay import SMALL
from t5d.engine.play import random_position_game
from t5d.net import NetConfig, build_model
from t5d.train.records import RECORD_VERSION, GameRecord, read_game, record_differences, write_game
from t5d.train.seeded import StoredPosition, open_position, positions_from_pgn, positions_from_record, read_positions, write_positions
from t5d.train.selfplay import (
    SALT_START,
    GeneratorConfig,
    SeededStartConfig,
    TemperatureSchedule,
    board_range,
    generate_game,
    move_temperature,
    pick_start,
    replay_positions,
    root_temperature,
)
from t5d.train.trainer import Trainer, TrainerConfig, synthetic_record
from t5d.train.window import WindowConfig
from tests.engine.helpers import RECORDS, requires_engine
from tests.train.test_records import same_input

NET = NetConfig(**SMALL)
ON = SeededStartConfig(enabled=True, fraction=1.0)  # every game seeded: the tests' games are few
BRANCHING = StoredPosition(pgn=RECORDS["branching"], actions=4, source="test:branching")
#: A game seeded from the branching record's fourth turn start (6 boards, two timelines).
SEEDED = GeneratorConfig(playouts=6, batch_size=3, cap_turns=4, cap_boards=20, seed=11, seeded_starts=ON)
#: The engine's own test record that starts at 49 boards (a custom 3x3 king-and-king position).
START_49 = Path(__file__).resolve().parents[3] / "engine" / "testdata" / "turn-width" / "edges" / "start-49.5dpgn"


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(0)
    return NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8)


# --- stored positions and the draw (no engine) -------------------------------------------------------------------


def test_stored_position_validation_and_file_round_trip(tmp_path):
    fen = ("[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]",)
    for bad in (
        dict(),  # neither form
        dict(pgn="x", fen=fen, turn=3),  # both
        dict(pgn="x", actions=-1),
        dict(pgn="x", turn=-1),
        dict(fen=fen),  # a 5DFEN position has no source game to count: it states its turn
        dict(fen=fen, turn=3, actions=2),
        dict(fen=fen, turn=3, rules=1),
    ):
        with pytest.raises(ValueError, match="I22"):
            StoredPosition(**bad)
    exact = StoredPosition(pgn=RECORDS["branching"], actions=4, source="a", rules=0, n_boards=6)
    late = StoredPosition(pgn=RECORDS["branching"], actions=2, turn=40)
    lossy = StoredPosition(fen=list(fen), turn=12, source="b", n_boards=1)
    assert (exact.kind, exact.turn_count) == ("pgn", 4) and late.turn_count == 40  # the actions replayed, unless stated
    assert (lossy.kind, lossy.turn_count, lossy.fen) == ("fen", 12, fen)
    path = write_positions(tmp_path / "pool.jsonl", [exact, late, lossy])
    assert read_positions(path) == [exact, late, lossy]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_one_game_in_ten_is_seeded_by_the_game_s_seed():
    """Section 16's placeholder: one self-play game in ten starts from a stored position - off by
    default, drawn from the game's seed (re-derivable), the position uniform over the pool."""
    assert SeededStartConfig() == SeededStartConfig(enabled=False, fraction=0.1)
    assert GeneratorConfig().seeded_starts == SeededStartConfig() and SALT_START == 5
    pool = [StoredPosition(pgn="x", actions=i) for i in range(4)]
    on = SeededStartConfig(enabled=True)
    picks = [pick_start(GeneratorConfig(seed=s, seeded_starts=on), pool) for s in range(4000)]
    assert picks == [pick_start(GeneratorConfig(seed=s, seeded_starts=on), pool) for s in range(4000)]
    taken = [p for p in picks if p is not None]
    assert len(taken) / 4000 == pytest.approx(0.1, abs=0.02)
    assert all(pool[i] is p for i, p in taken)
    for i in range(4):
        assert sum(k == i for k, _ in taken) / len(taken) == pytest.approx(0.25, abs=0.07)
    assert all(pick_start(GeneratorConfig(seed=s), pool) is None for s in range(200))  # off: never, pool or no pool
    assert all(pick_start(GeneratorConfig(seed=s, seeded_starts=replace(on, fraction=0.0)), pool) is None for s in range(200))
    assert all(pick_start(GeneratorConfig(seed=s, seeded_starts=ON), pool) is not None for s in range(200))
    for empty in ([], None):
        with pytest.raises(ValueError, match="I22"):
            pick_start(GeneratorConfig(seeded_starts=on), empty)
    for fraction in (1.5, -0.1):
        with pytest.raises(ValueError, match="I22"):
            GeneratorConfig(seeded_starts=SeededStartConfig(enabled=True, fraction=fraction)).validate()


def test_the_generator_s_signature_is_kept():
    """Other callers hand ``generate_game`` its earlier arguments: the pool of stored positions is a
    keyword of its own, None by default."""
    params = inspect.signature(generate_game).parameters
    assert list(params)[:2] == ["evaluator", "cfg"]
    assert params["stored"].kind is inspect.Parameter.KEYWORD_ONLY and params["stored"].default is None
    for name in ("net_cfg", "network", "keep_positions", "target_config", "on_decision"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_synthetic_record_supplies_no_stored_position():
    from t5d.net import synthetic_states

    with pytest.raises(ValueError, match="no 5dpgn"):
        positions_from_record(synthetic_record(1, synthetic_states(1, [1, 2])))


# --- with the engine ------------------------------------------------------------------------------------------------


@requires_engine
def test_a_stored_position_opens_with_its_history():
    """Why a stored position is a record replayed and not 5DFEN boards: the engine sets it up with
    its branch origins - the same state read back from its own 5DFEN boards has a timeline of
    unknown origin - and the handle is then played on."""
    with Game.from_pgn(RECORDS["branching"]) as by_hand:
        for _ in range(4):
            by_hand.replay_next()
        want = by_hand.snapshot()
    with open_position(BRANCHING) as game:
        info = game.info()
        assert (info.boards, info.timelines, info.moves_made) == (6, 2, 0)
        assert game.snapshot().fingerprint == want.fingerprint
        assert game.position().origins_unknown_all == 0
        with Game.from_fen([fen for _, _, fen in want.boards]) as lossy:
            assert lossy.snapshot().fingerprint == want.fingerprint and lossy.position().origins_unknown_all == 1
        game.apply(game.moves()[0])  # the game leaves its record and is played on
        assert game.info().moves_made == 1
    with open_position(StoredPosition(pgn=RECORDS["branching"])) as start:
        assert start.info().boards == 1  # no action replayed: the record's own set-up
    with pytest.raises(ValueError, match="I22"):
        open_position(StoredPosition(pgn=RECORDS["branching"], actions=7))  # the record has six


@requires_engine
def test_a_seeded_game_records_its_origin_and_replays(evaluator, tmp_path):
    """A seeded game is played out and recorded as usual; its header names its origin - the source,
    the 5dpgn and the actions replayed, the source's turn count, the start as the engine set it up,
    the draw - and its inputs rebuild from the stored record alone, branch origins known."""
    with torch.no_grad():
        g = generate_game(evaluator, SEEDED, net_cfg=NET, keep_positions=True, stored=[BRANCHING])
    rec = g.record
    h = rec.header
    with open_position(BRANCHING) as start:
        fingerprint = start.snapshot().fingerprint
    assert (h["variant"], h["start_fen"], h["record_version"]) == ("seeded", None, RECORD_VERSION)
    assert h["seeded_start"] == {
        "source": "test:branching",
        "kind": "pgn",
        "pgn": RECORDS["branching"],
        "actions": 4,
        "rules": 0,
        "turn": 4,
        "fingerprint": fingerprint,
        "n_boards": 6,
        "pool": {"size": 1, "index": 0},
    }
    assert rec.seeded_start == h["seeded_start"] and rec.start_turn == 4
    assert h["generator_config"]["seeded_starts"] == {"enabled": True, "fraction": 1.0}
    first = rec.decisions[0]
    assert (first.n_boards, first.action, first.moves_made) == (6, 0, 0)  # its decisions count actions from its own start
    assert rec.n_actions >= 1 and '[Seeded "test:branching; turn 4"]' in h["pgn"] and '[Board "custom"]' in h["pgn"]
    with Game.from_pgn(h["pgn"]) as continuation:  # the continuation from the start's 5DFEN boards reads back here
        assert continuation.replay_all() == rec.n_actions
        assert continuation.snapshot().fingerprint == h["result"]["final_fingerprint"]
    back = read_game(write_game(tmp_path / "seeded.npz", rec))
    assert record_differences(rec, back) == [] and back.start_turn == 4
    rebuilt = replay_positions(back)
    assert len(rebuilt) == len(g.positions) == len(back.decisions) > 1
    for live, again in zip(g.positions, rebuilt, strict=True):
        assert same_input(live.state, again.state)
        assert live.origins_unknown_all == again.origins_unknown_all == 0
    trainer = Trainer(build_model(NET, seed=0), NET, TrainerConfig(batch_size=4, window=WindowConfig(min_rows=1000)))
    with torch.no_grad():
        assert trainer.add_game(back) > 0
        assert all(math.isfinite(v) for v in trainer.losses_on(trainer.window.rows()).values()) and trainer.steps == 0
    for wrong in ({"fingerprint": "0" * 32}, {"actions": 3}):
        bad = GameRecord(header={**h, "seeded_start": {**h["seeded_start"], **wrong}}, decisions=rec.decisions)
        with pytest.raises(ValueError, match="seeded start"):
            replay_positions(bad)


@requires_engine
def test_a_seeded_game_keeps_its_source_s_turn_count():
    """Section 12: "A seeded game keeps its source's turn count: the temperature schedule ... read the position's
    own turn number, not zero" - in actions, one player's turn each (issue 305): both
    schedules run at the source's turn plus the game's own action, so a game seeded at turn 4 under
    a cutoff at turn 5 plays its first action at the initial temperature and the rest at the
    endgame's, where the same game from the start is still at the initial one."""
    schedule = TemperatureSchedule(cutoff_turn=5, root_decay_turns=8)
    cfg = replace(SEEDED, schedule=schedule)
    g = generate_game(UniformEvaluator(), cfg, net_cfg=NET, stored=[BRANCHING])
    ds = g.record.decisions
    assert {d.action for d in ds} >= {0, 1}
    for d in ds:
        assert d.move_temperature == move_temperature(cfg, 4 + d.action) == (0.9 if d.action == 0 else 0.45)
        assert d.root_policy_temperature == root_temperature(cfg, 4 + d.action) != root_temperature(cfg, d.action)
    unseeded = generate_game(UniformEvaluator(), replace(cfg, seeded_starts=SeededStartConfig()), net_cfg=NET, stored=[BRANCHING])
    assert {d.action for d in unseeded.record.decisions} >= {0, 1}
    for d in unseeded.record.decisions:
        assert d.move_temperature == move_temperature(cfg, d.action) == (0.9 if d.action < 5 else 0.45)  # action 1: still 0.9
        assert d.root_policy_temperature == root_temperature(cfg, d.action)
    late = generate_game(UniformEvaluator(), cfg, net_cfg=NET, stored=[replace(BRANCHING, turn=40)])
    assert late.record.start_turn == 40 and late.record.decisions
    assert all(d.move_temperature == 0.45 and d.root_policy_temperature == root_temperature(cfg, 40) for d in late.record.decisions)


@requires_engine
def test_the_cap_reads_the_stored_position_s_own_turn():
    """I11's cap is defined on the state, so nothing is built for it: a stored position whose
    present is already past the turn cap, or at the board cap, ends at once - unscored, no decision
    - and plays on under a cap its own turn is within."""
    with open_position(BRANCHING) as start:
        assert (start.info().present // 2, start.info().boards) == (2, 6)  # its present is turn 2
    for caps in (dict(cap_turns=1, cap_boards=20), dict(cap_turns=0, cap_boards=6)):
        g = generate_game(UniformEvaluator(), replace(SEEDED, **caps), net_cfg=NET, stored=[BRANCHING])
        assert (g.record.end, len(g.record.decisions), g.record.n_actions) == ("turn_cap", 0, 0), caps
        assert g.record.start_turn == 4 and g.record.header["result"]["final_fingerprint"] == g.record.seeded_start["fingerprint"]
        assert replay_positions(g.record) == []
    played = generate_game(UniformEvaluator(), replace(SEEDED, cap_turns=3), net_cfg=NET, stored=[BRANCHING])
    assert played.record.decisions and played.record.n_actions >= 1


@requires_engine
def test_positions_from_a_5dpgn():
    """One stored position per turn start of a record, filtered by board count; the last position
    only when asked for; a record whose third action the engine refuses yields the two before it."""
    ps = positions_from_pgn(RECORDS["branching"], source="helpers:branching")
    assert [(p.actions, p.n_boards) for p in ps] == [(1, 2), (2, 3), (3, 5), (4, 6), (5, 7)]
    assert all(p.pgn == RECORDS["branching"] and p.source == "helpers:branching" and p.turn_count == p.actions for p in ps)
    assert ps[3] == replace(BRANCHING, source="helpers:branching", n_boards=6)
    assert [p.actions for p in positions_from_pgn(RECORDS["branching"], min_boards=5)] == [3, 4, 5]
    assert [p.actions for p in positions_from_pgn(RECORDS["branching"], min_boards=3, max_boards=6)] == [2, 3, 4]
    with_last = positions_from_pgn(RECORDS["branching"], include_last=True)
    assert [(p.actions, p.n_boards) for p in with_last][-1] == (6, 9) and len(with_last) == 6
    for p in with_last:
        with open_position(p) as game:
            assert game.info().boards == p.n_boards
    refused = '[Board "Standard"]\n[Mode "5D"]\n\n1. e4 / e5\n2. Ke3 / e6\n'  # White's king cannot reach e3
    assert [(p.actions, p.n_boards) for p in positions_from_pgn(refused, include_last=True)] == [(1, 2), (2, 3)]


@requires_engine
def test_positions_from_an_earlier_self_play_record(evaluator):
    """An earlier game's own record supplies stored positions - its 5dpgn and each later turn
    start's action - which the engine sets up as the game reached them; a game seeded from one
    records that turn; and a seeded game's own record supplies none (re-seeding is refused)."""
    base = GeneratorConfig(playouts=6, batch_size=3, cap_turns=3, cap_boards=20, seed=12)
    with torch.no_grad():
        source = generate_game(evaluator, base, net_cfg=NET).record
    starts = [d for d in source.decisions if d.moves_made == 0 and d.action >= 1]
    ps = positions_from_record(source)
    assert len(ps) == len(starts) >= 2
    assert [(p.actions, p.n_boards, p.turn_count, p.rules) for p in ps] == [(d.action, d.n_boards, d.action, 0) for d in starts]
    assert all(p.pgn == source.header["pgn"] and p.source.startswith("selfplay:12:") for p in ps)
    for p in ps:
        with Game.from_pgn(source.header["pgn"]) as by_hand:
            for _ in range(p.actions):
                by_hand.replay_next()
            want = by_hand.snapshot().fingerprint
        with open_position(p) as game:
            assert game.snapshot().fingerprint == want and game.info().rules == source.header["rules"]
            assert game.info().boards == p.n_boards
    assert [p.actions for p in positions_from_record(source, min_boards=ps[-1].n_boards)] == [ps[-1].actions]
    seeded = generate_game(UniformEvaluator(), replace(SEEDED, cap_turns=5, seed=13), net_cfg=NET, stored=ps).record
    index = seeded.seeded_start["pool"]["index"]
    assert seeded.seeded_start["pool"]["size"] == len(ps) and seeded.start_turn == ps[index].actions == seeded.seeded_start["actions"]
    assert seeded.seeded_start["source"] == ps[index].source and len(replay_positions(seeded)) == len(seeded.decisions)
    with pytest.raises(ValueError, match="seeded game's own record"):
        positions_from_record(seeded)


@requires_engine
def test_a_5dfen_position_is_accepted_with_its_turn(evaluator):
    """The lossy form: 5DFEN boards with a stated turn count - recorded as a custom start, its turn
    in ``seeded_start``, rebuilt from the record like any custom start."""
    for seed in (1, 2, 3):
        try:
            with random_position_game("Standard", seed, boards=10, travel_prob=1.0) as source:
                fens = tuple(fen for _, _, fen in source.snapshot().boards)
        except ValueError:
            continue
        stored = StoredPosition(fen=fens, turn=12, source="test:random play", n_boards=len(fens))
        cfg = GeneratorConfig(playouts=6, batch_size=3, cap_turns=0, cap_boards=16, max_actions=3, seed=seed, seeded_starts=ON)
        with torch.no_grad():
            g = generate_game(evaluator, cfg, net_cfg=NET, keep_positions=True, stored=[stored])
        h = g.record.header
        assert (h["variant"], h["start_fen"]) == ("custom", list(fens))
        assert (h["seeded_start"]["kind"], h["seeded_start"]["turn"], h["seeded_start"]["pgn"]) == ("fen", 12, None)
        assert h["seeded_start"]["n_boards"] == len(fens) and g.record.start_turn == 12
        assert all(d.root_policy_temperature == root_temperature(cfg, 12 + d.action) for d in g.record.decisions)
        rebuilt = replay_positions(g.record)
        assert len(rebuilt) == len(g.record.decisions)
        assert all(same_input(a.state, b.state) for a, b in zip(g.positions, rebuilt, strict=True))
        with pytest.raises(ValueError, match="seeded game's own record"):
            positions_from_record(g.record)
        return
    pytest.skip("no seeded travel start reached 10 boards")


@requires_engine
def test_a_stored_large_position_reaches_the_next_range():
    """What seeded starts are for: a game begun at 49 boards (the engine's own test record) is
    searched from its first decision in the under-50 range and plays into the 50-99 range within a
    few actions - where a game from the opening pays for every turn before."""
    text = START_49.read_text(encoding="utf-8")
    stored = StoredPosition(pgn=text, turn=48, source="engine/testdata/turn-width/edges/start-49.5dpgn", n_boards=49)
    cfg = GeneratorConfig(playouts=4, batch_size=2, cap_turns=0, cap_boards=56, seed=3, seeded_starts=ON)
    g = generate_game(UniformEvaluator(), cfg, net_cfg=NET, stored=[stored])
    ds = g.record.decisions
    ranges = (50, 100, 200)
    assert ds[0].n_boards == 49 and board_range(ds[0].n_boards, ranges) == "0-49"
    assert any(board_range(d.n_boards, ranges) == "50-99" for d in ds)
    assert g.record.start_turn == 48 and g.record.seeded_start["n_boards"] == 49 and g.record.end == "turn_cap"
    assert len(replay_positions(g.record)) == len(ds)


@requires_engine
def test_seeding_off_changes_nothing(evaluator):
    """Off (the default), a pool given or not, and on with this game's draw not taking a stored
    position: the game is the game from the start - the same record, no ``seeded_start`` key, turn
    count 0 - but for the generator's configuration, which names the setting."""
    cfg = GeneratorConfig(playouts=6, batch_size=3, cap_turns=2, cap_boards=20, seed=14)
    assert not cfg.seeded_starts.enabled
    with torch.no_grad():
        plain = generate_game(evaluator, cfg, net_cfg=NET).record
        given = generate_game(evaluator, cfg, net_cfg=NET, stored=[BRANCHING]).record
        not_taken = generate_game(
            evaluator, replace(cfg, seeded_starts=SeededStartConfig(enabled=True, fraction=0.0)), net_cfg=NET, stored=[BRANCHING]
        ).record
    assert record_differences(plain, given) == []
    assert record_differences(plain, not_taken) == ["header['generator_config']"]
    for rec in (plain, given, not_taken):
        assert "seeded_start" not in rec.header and rec.seeded_start is None and rec.start_turn == 0
        assert (rec.header["variant"], rec.header["start_fen"]) == ("Standard", None) and "Seeded" not in rec.header["pgn"]
    assert plain.header["generator_config"]["seeded_starts"] == {"enabled": False, "fraction": 0.1}
    assert all(d.move_temperature == move_temperature(cfg, d.action) for d in plain.decisions)
