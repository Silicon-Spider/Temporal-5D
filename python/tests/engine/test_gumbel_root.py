"""Register E12 (Gumbel root search; a prototype, off by default) through the binding: the
parameters' defaults and their mirror of the library, the step loop against the engine's own
run, the choice and the refusals - and the numbers the engine pins against the target side's
(``train.completed_q``: v_mix and sigma).

The engine's arm is ``t5d_search_params.gumbel_root`` (``engine/include/t5d/gumbel.hpp``; spec
section 8.4): at a self-play root, a Gumbel sample of m moves and sequential halving over them;
the move to play is ``Search.choose_gumbel()``. The training generator's arm is not built (its
policy target and the root value under halving wait on I21 and a ruling; build-log issue 415).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from t5d.engine import Game
from t5d.engine.capi import T5D_EVALUATOR_UNIFORM, T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_FLAT, T5D_SPLIT_TREE, layout_mismatches
from t5d.engine.errors import EngineArgumentError
from t5d.engine.evaluator import UniformEvaluator, run_search
from t5d.engine.search import Search, search_params
from tests.engine.helpers import requires_engine

GUMBEL = dict(gumbel_root=1, gumbel_simulations=63, asset_map=0, hunt=0, settling=0)


@requires_engine
def test_gumbel_parameters_default_off_and_mirror_the_library():
    """Off by default (the as-built behaviour bit for bit); m 16, c_visit 50, c_scale 1 - the
    paper's placeholders; n 0 (the driver gives it); the struct as the library lays it out."""
    with Game.variant("Standard") as g:
        assert layout_mismatches(g.lib) == []
        p = search_params(g.lib)
    assert (p.gumbel_root, p.gumbel_m, p.gumbel_simulations) == (0, 16, 0)
    assert (p.gumbel_c_visit, p.gumbel_c_scale) == (50.0, 1.0)
    from t5d.train.records import params_dict, params_from_dict

    q = params_from_dict(params_dict(search_params(gumbel_root=1, gumbel_m=4, gumbel_simulations=9, gumbel_c_visit=30.0)))
    assert (q.gumbel_root, q.gumbel_m, q.gumbel_simulations, q.gumbel_c_visit, q.gumbel_c_scale) == (1, 4, 9, 30.0, 1.0)


@requires_engine
@pytest.mark.parametrize("split", [T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_TREE])
def test_the_gumbel_step_loop_is_the_engines_run(split):
    """The step API driven by the binding's loop (the uniform evaluator) builds the tree the engine's
    own run builds - the same digest, the same playouts, the same choice."""
    with Game.variant("Standard") as g:
        params = search_params(g.lib, split=split, seed=3, **GUMBEL)
        with Search(g, params=params) as stepped, Search(g, params=params) as run:
            run_search(stepped, UniformEvaluator(), playouts=64, batch_size=int(params.batch_size))
            run.run(T5D_EVALUATOR_UNIFORM, 64)
            assert stepped.info().playouts == run.info().playouts == 64
            assert stepped.digest() == run.digest()
            assert stepped.choose_gumbel() == run.choose_gumbel()


@requires_engine
def test_gumbel_choice_and_refusals_reach_python():
    """The choice is a root child's index; match play's choice and re-roots are refused under the
    arm, and so are the parameters it replaces or cannot take - each named; without the flag there
    is no Gumbel choice."""
    with Game.variant("Standard") as g:
        with Search(g, params=search_params(g.lib, **GUMBEL)) as s:
            s.run(T5D_EVALUATOR_UNIFORM, 64)
            assert 0 <= s.choose_gumbel() < s.info().root_children
            with pytest.raises(EngineArgumentError):
                s.choose()
            with pytest.raises(EngineArgumentError):
                s.choose_by_visits(0.9, 0.5)
            s.choose_by_visits(0.0, 0.5)  # the most visited child: a read, allowed
            with pytest.raises(EngineArgumentError):
                s.advance(0)
        for override, name in (
            ({"asset_map": 1}, "asset_map"),
            ({"root_noise": 1}, "root_noise"),
            ({"root_policy_temperature": 1.25}, "root_policy_temperature"),
            ({"split": T5D_SPLIT_FLAT}, "flat"),
            ({"gumbel_simulations": 0}, "gumbel_simulations"),
        ):
            with pytest.raises(EngineArgumentError, match=name):
                Search(g, params=search_params(g.lib, **{**GUMBEL, **override}))
        with Search(g) as plain:
            plain.run(T5D_EVALUATOR_UNIFORM, 16)
            with pytest.raises(EngineArgumentError, match="gumbel_root"):
                plain.choose_gumbel()


def test_the_engines_numbers_are_the_target_sides():
    """No engine needed: the target side's v_mix and sigma (``train.completed_q``) on the numbers the
    engine's test pins (test_gumbel.cpp: 0, 31.8, 15.9 and 26.5), and the SplitMix64 finaliser the
    engine's Gumbel stream is built on, on the inputs the engine pins - so a record can derive an
    edge's g again (gumbel.hpp's formula, from search_seed, root_key and the candidate index)."""
    from t5d.train import completed_q
    from t5d.train.selfplay import mix

    prior = np.array([0.5, 0.3, 0.2])
    visits = np.array([3, 1, 0])
    q = np.array([0.2, -0.4, 0.0])
    v_mix = completed_q.mixed_value(prior, q, visits, 0.1)
    assert math.isclose(v_mix, 0.0, abs_tol=1e-12)
    cfg = completed_q.CompletedQConfig()  # c_visit 50, c_scale 1, q mapped onto [0, 1]
    assert (cfg.c_visit, cfg.c_scale, cfg.value_range) == (50.0, 1.0, "unit")
    for value, want in ((0.2, 31.8), (-0.4, 15.9), (v_mix, 26.5)):
        assert math.isclose(float(completed_q.sigma(np.float64(value), 3, cfg)), want, rel_tol=1e-12)
    assert mix(0) == 0 and mix(1) == 0x5692161D100B05E5 and mix(2) == 0xDBD238973A2B148A
    assert mix(2**64 - 1) == 0xB4D055FCF2CBBD7B
