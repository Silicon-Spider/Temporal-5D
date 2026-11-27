"""The forward bench's arms for the quiet-machine block (``t5d.net.bench``): smoke runs at tiny size.

No timing is read here - the machine running the tests is shared, and a test is not a measurement.
What is pinned: each arm is the network its row names (section 15.3's "FFN ratio 2x and 4x" against
the v1's 8x; register A5's piece-centric tokens beside A1's, dense scope), and the bench's command
line runs each - every pass in its own subprocess, as a measurement would - to a row without an
error. Networks at 1 layer, FP32 trunk; positions ``synthetic_state(0, N)`` at 2 to 3 boards.
"""

from __future__ import annotations

import json

import pytest

from t5d.net import bench, opcount
from t5d.net.bench import CONFIGS
from t5d.net.config import NetConfig
from t5d.net.synthetic import synthetic_state


def test_the_ffn_arms_are_the_v1_with_the_trunk_ffn_at_2x_and_4x():
    v1 = NetConfig(**CONFIGS["n2-default"])
    assert v1 == NetConfig() and v1.ffn_ratio == 8  # F1's 8x: the row's baseline
    for name, ratio in (("ffn-2x", 2), ("ffn-4x", 4)):
        arm = NetConfig(**CONFIGS[name])
        arm.validate()
        assert arm == NetConfig(ffn_ratio=ratio)
        assert arm.d_ffn == ratio * arm.d_model
        assert arm.layout_ffn_ratio == v1.layout_ffn_ratio  # the trunk's FFN, not A8's (off in both)


def test_the_a5_arm_is_the_v1_on_piece_centric_tokens_in_the_dense_scope():
    arm = NetConfig(**CONFIGS["a5-piece-centric"])
    arm.validate()
    assert arm == NetConfig(piece_centric=True)
    assert arm.attn_scope == "dense"


@pytest.mark.parametrize("config", ["ffn-2x", "ffn-4x", "a5-piece-centric"])
def test_each_arm_runs_one_forward_in_process(config):
    row = bench.measure(config, 3, "fp32", layers=1)
    assert row["config"] == config and row["boards"] == 3 and row["layers"] == 1 and row["pass"] == "forward"
    assert row["seconds"] >= 0 and row["peak_gb"] > 0


def test_the_a5_arm_reads_fewer_tokens_than_a1_on_the_same_position():
    a1 = bench.measure("n2-default", 3, "fp32", layers=1)
    a5 = bench.measure("a5-piece-centric", 3, "fp32", layers=1)
    assert a1["tokens"] == 66 * 3 + 16  # A1: every cell, the per-board and the global registers
    assert a5["tokens"] < a1["tokens"]
    assert a5["tokens"] == synthetic_state(0, 3, piece_centric=True).n_tokens


def test_the_operator_counter_reaches_the_a5_arm():
    row = opcount.measure("a5-piece-centric", 2, "fp32", layers=1)
    assert row["config"] == "a5-piece-centric" and row["ops"] > 0
    assert row["tokens"] == synthetic_state(0, 2, piece_centric=True).n_tokens


def test_the_command_line_runs_the_arms_beside_the_v1(capsys):
    # As the quiet-machine block runs them (``--boards 44 240``), at a smoke size: each pass in its
    # own subprocess, one JSON line each, none failed.
    configs = ["n2-default", "ffn-2x", "ffn-4x", "a5-piece-centric"]
    assert bench.main(["--config", *configs, "--boards", "2", "--dtype", "fp32", "--layers", "1"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["config"] for r in rows] == configs
    for r in rows:
        assert "error" not in r, r
        assert r["boards"] == 2 and r["dtype"] == "fp32" and r["layers"] == 1
    assert rows[3]["tokens"] < rows[0]["tokens"]
