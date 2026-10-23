"""D2's flat head through the engine (spec section 8.2; BUILD_LOG issues 164-166, 241): the network
delivers a "leave it" logit for every board, and a flat search it evaluates reads them - no board
falls back on ``leave_share``. Checked two ways: every delivered evaluation carries a finite leave
logit at each of its leaf's playable boards, and the root's turn priors are the softmax of the turns'
scores computed here from the network's own move and leave logits (issues 164-166's reading, at the
root's temperature) - and not those the fallback would give."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from t5d.engine.capi import T5D_FLAT_JUMPS_BOTH, T5D_FLAT_JUMPS_SOURCE, T5D_SPLIT_FLAT
from t5d.engine.convert import relations_for
from t5d.engine.evaluator import NetworkEvaluator, check_evaluation, run_search
from t5d.engine.game import Game, relation_bits
from t5d.engine.netplay import SMALL
from t5d.engine.play import random_position_game
from t5d.engine.search import Search, search_params
from t5d.net import NetConfig, build_model
from t5d.train.selfplay import playable_boards, root_position, turn_moves
from t5d.train.targets import LEAVE, flat_boards, turn_option
from tests.engine.helpers import requires_engine

NET = NetConfig(**SMALL)


class Recorder:
    """A network evaluator that keeps every evaluation it gives (the leaves' positions with them)."""

    def __init__(self, inner):
        self.inner = inner
        self.model = inner.model
        self.seen: list[tuple] = []

    def __call__(self, positions):
        out = self.inner(positions)
        self.seen += list(zip(positions, out, strict=True))
        return out


def _scores(raw, position, playable, children, temperature: float, *, arrivals: bool, leave_share: float | None):
    """Each root child's score - the sum over the playable boards of the log-probability of the option
    its turn takes there, each board softmaxed over its moves, arrivals and leave logit at
    ``temperature`` (issues 164, 165); ``leave_share`` given: the fallback - leaving at the share,
    the moves sharing the rest."""
    d = SimpleNamespace(
        candidates=position.moves, boards=np.stack([position.state.timeline, position.state.half_turn], 1), playable=playable
    )
    boards = flat_boards(d, arrivals)
    logp = []
    for fb in boards:
        z = raw.policy_logits.astype(np.float64)[list(fb.own) + list(fb.arrivals)] / temperature
        if leave_share is None:
            z = np.append(z, float(raw.board_logits[fb.board]) / temperature)
            lp = z - (z.max() + math.log(np.exp(z - z.max()).sum()))
        else:
            lp = z - (z.max() + math.log(np.exp(z - z.max()).sum())) + math.log1p(-leave_share) if z.size else z
            lp = np.append(lp, math.log(leave_share) if z.size else 0.0)
        logp.append(dict(zip(fb.options, lp.tolist(), strict=True)))
    out = []
    for c in children:
        turn = turn_moves([c])
        s = 0.0
        for fb, table in zip(boards, logp, strict=True):
            k = turn_option(turn, fb, position.moves, arrivals)
            if k == -2:  # an arrival under SourceOnly: log 1
                continue
            s += table[k] if k in table else min(table.values())  # unmatched: the board's lowest option
        out.append(s)
    return np.array(out)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


@requires_engine
@pytest.mark.parametrize("jumps", [T5D_FLAT_JUMPS_BOTH, T5D_FLAT_JUMPS_SOURCE], ids=["both", "source"])
@pytest.mark.parametrize("boards", [0, 8, 14])
def test_a_flat_search_reads_the_networks_leave_logits(jumps, boards):
    """A flat search from Two Timelines' start (two playable boards) and from seeded random play
    (more timelines, travel preferred), with non-branching jumps as arrivals and without."""
    torch.manual_seed(0)
    ev = Recorder(NetworkEvaluator(build_model(NET, seed=1).eval(), max_batch=8))
    relations = relation_bits(relations_for(NET))
    start = (
        Game.variant("Standard - Two Timelines")
        if not boards
        else random_position_game("Standard - Two Timelines", 3, boards=boards, travel_prob=0.7)
    )
    with start as game:
        params = search_params(
            game.lib,
            split=T5D_SPLIT_FLAT,
            seed=5,
            root_noise=0,
            flat_jumps=jumps,
            root_policy_temperature=1.3,
            expand=(8, "all", relations),
        )
        with Search(game, params=params) as search:
            with torch.no_grad():
                run_search(search, ev, playouts=24, batch_size=4, self_play=True)
            children = search.children()
            position = root_position(game, params, NET, 8)
            playable = playable_boards(game, position)
        raw = ev([position])[0]
    assert len(children) >= 2
    # Every evaluation delivered a finite leave logit at each playable board of its leaf: no fallback.
    assert ev.seen
    for i, (p, e) in enumerate(ev.seen):
        check_evaluation(SimpleNamespace(id=i, position=p), e, NET.value_bins)
        assert e.board_logits is not None and np.isfinite(e.board_logits).all()
    assert playable.sum() >= 1 and np.isfinite(raw.board_logits[playable]).all()
    # The root's priors are the network's, leave logits included - not the fallback's.
    t = float(params.root_policy_temperature) * float(params.policy_temperature)
    arrivals = jumps == T5D_FLAT_JUMPS_BOTH
    want = _softmax(_scores(raw, position, playable, children, t, arrivals=arrivals, leave_share=None))
    got = np.array([c.prior_noise_free for c in children])
    assert np.allclose(got, want, rtol=0, atol=2e-6), np.abs(got - want).max()
    assert np.allclose([c.prior for c in children], got, rtol=0, atol=1e-12)  # no root noise
    # Where some turns leave a board and others play it, the leave logit moves the priors: the
    # fallback's would differ. (Where every turn plays every board - all mandatory - each board's
    # normaliser is common to every turn and the priors cannot tell the two apart.)
    d = SimpleNamespace(
        candidates=position.moves, boards=np.stack([position.state.timeline, position.state.half_turn], 1), playable=playable
    )
    leaves = [[turn_option(turn_moves([c]), fb, position.moves, arrivals) == LEAVE for fb in flat_boards(d, arrivals)] for c in children]
    mixed = any(len({row[b] for row in leaves}) == 2 for b in range(len(leaves[0])))
    fallback = _softmax(_scores(raw, position, playable, children, t, arrivals=arrivals, leave_share=float(params.leave_share)))
    if mixed:
        assert np.abs(fallback - got).max() > 1e-4
    MIXED.append(mixed)


#: Whether each case had a board some turns leave and others play (``test_some_case_leaves_a_board``).
MIXED: list[bool] = []


def test_some_case_leaves_a_board():
    """The cases above include positions where the leave logit shows in the priors."""
    if not MIXED:
        pytest.skip("run with the flat search cases")
    assert any(MIXED)


class _Hidden:
    """A wrapper that names no prior of its own (``policy_source`` reads "normal"): its evaluations
    still carry theirs."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, positions):
        return self.inner(positions)


@requires_engine
def test_the_flat_form_refuses_the_optimistic_prior():
    """D9's optimistic head has no leave logit: under the flat form its move logits would be scored
    beside the normal head's leave logits, whose offset against them is arbitrary (issue 242(h)) - so
    ``run_search`` refuses an optimistic evaluator on a flat search, the step driver refuses every
    optimistic evaluation it is handed (whatever wraps the evaluator, the batch released), and
    netplay refuses the combination; the ply split still searches with it."""
    from t5d.engine.evaluator import FLAT_OPTIMISTIC, CStepDriver, serve
    from t5d.engine.netplay import PlayConfig, main

    cfg = NetConfig(**{**SMALL, "optimistic_policy": True})
    ev = NetworkEvaluator(build_model(cfg, seed=0).eval(), max_batch=4, policy="optimistic")
    relations = relation_bits(relations_for(cfg))
    with Game.variant("Standard - Two Timelines") as game:
        params = search_params(game.lib, split=T5D_SPLIT_FLAT, seed=5, root_noise=0, expand=(8, "all", relations))
        with Search(game, params=params) as search, torch.no_grad():
            with pytest.raises(ValueError, match="flat form"):
                run_search(search, ev, playouts=8, batch_size=4)
            driver = CStepDriver(search, playouts=8)
            with pytest.raises(ValueError, match="flat form") as err:
                serve(driver, _Hidden(ev), batch_size=4)
            assert str(err.value) == FLAT_OPTIMISTIC and driver.batch == ()
            stats, _ = run_search(search, NetworkEvaluator(ev.model, max_batch=4), playouts=8, batch_size=4)
            assert stats.leaves > 0
        ply = search_params(game.lib, seed=5, root_noise=0, expand=(8, "all", relations))
        with Search(game, params=ply) as search, torch.no_grad():
            run_search(search, ev, playouts=8, batch_size=4)
            assert sum(c.visits for c in search.children()) > 0
    PlayConfig(split="ply", mode="match", optimistic=True).validate()
    with pytest.raises(ValueError, match="flat form"):
        PlayConfig(split="flat", mode="match", optimistic=True).validate()
    with pytest.raises(SystemExit):
        main(["--optimistic", "--mode", "match", "--split", "flat", "--no-replay"])
