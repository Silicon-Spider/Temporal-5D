"""The forward bench's arms for the quiet-machine block (``t5d.net.bench``): smoke runs at tiny size.

No timing is read here - the machine running the tests is shared, and a test is not a measurement.
What is pinned: each arm is the network its row names (section 15.3's "FFN ratio 2x and 4x" against
the v1's 8x; register A5's piece-centric tokens beside A1's, dense scope; A8 without its FFN, the
A8 table's third column), and the bench's command line runs each - every pass in its own
subprocess, as a measurement would - to a row without an error. ``--embed`` times
``T5DNet.embed_tokens`` alone beside the forward (the A8 table's embedding column); ``--repeat``
interleaves the configurations, the order reversed every other repetition, and reports each one's
medians and minima (the A8 section's protocol) - its order and arithmetic checked on a stubbed pass.
Networks at 1 layer, FP32 trunk; positions ``synthetic_state(0, N)`` at 2 to 3 boards.
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


# --- the embedding's time and the repetitions (A8's protocol) ---


def test_the_a8_table_s_third_arm_is_a8_without_its_ffn():
    arm = NetConfig(**CONFIGS["a8-no-ffn"])
    arm.validate()
    assert arm == NetConfig(board_layout_embedding=True, layout_ffn_ratio=0)


def test_the_embedding_is_timed_alone_beside_the_forward():
    for config in ("n2-default", "6p-layout-ff8", "6p-layout-ff1", "a8-no-ffn"):
        row = bench.measure(config, 3, "fp32", layers=1, embed=True)
        assert row["embed_ms"] > 0 and row["seconds"] >= 0
        assert list(row)[-1] == "embed_ms"  # appended: every earlier field as it was
    plain = bench.measure("n2-default", 3, "fp32", layers=1)
    assert "embed_ms" not in plain
    with pytest.raises(ValueError, match="not with a training step"):
        bench.measure("n2-default", 2, "fp32", train=True, layers=1, embed=True)


def test_threads_are_set_in_the_pass():
    assert bench.measure("n2-default", 2, "fp32", layers=1, threads=1)["threads"] == 1


def test_repetitions_reverse_the_order_every_other_time_and_report_medians(monkeypatch, capsys):
    # The pass itself stubbed: the order, the arguments each pass gets, and the summary's arithmetic.
    calls = []
    times = {"n2-default": iter([3.0, 1.0, 2.0]), "6p-layout-ff8": iter([5.0, 9.0])}

    def fake_run(config, boards, dtype, train=False, layers=None, embed=False, threads=None):
        calls.append((config, boards, dtype, embed, threads))
        if len(calls) == 3:  # 6p-layout-ff8's second pass fails
            return {"config": config, "boards": boards, "dtype": dtype, "error": "out of memory"}
        seconds = next(times[config])
        row = {"config": config, "boards": boards, "dtype": dtype, "pass": "forward", "layers": layers, "threads": threads}
        return row | {"tokens": 7, "seconds": seconds, "embed_ms": 10 * seconds}

    monkeypatch.setattr(bench, "run", fake_run)
    argv = ["--config", "n2-default", "6p-layout-ff8", "--boards", "44", "--dtype", "bf16", "--layers", "1"]
    assert bench.main([*argv, "--embed", "--repeat", "3", "--threads", "2"]) == 0
    order = [c[0] for c in calls]
    assert order == ["n2-default", "6p-layout-ff8", "6p-layout-ff8", "n2-default", "n2-default", "6p-layout-ff8"]
    assert {c[1:] for c in calls} == {(44, "bf16", True, 2)}
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    passes, summary = lines[:6], lines[6:]
    assert [p["repetition"] for p in passes] == [1, 1, 2, 2, 3, 3]
    assert [s["config"] for s in summary] == ["n2-default", "6p-layout-ff8"]
    a1, a8 = summary
    assert a1["summary"] == "median" and a1["runs"] == 3 and a1["failed"] == 0
    assert (a1["seconds_median"], a1["seconds_min"], a1["embed_ms_median"], a1["embed_ms_min"]) == (2.0, 1.0, 20.0, 10.0)
    assert a1["threads"] == 2 and a1["boards"] == 44 and a1["dtype"] == "bf16"
    # The failed pass (its 2nd) is counted, never timed: the median of the other two.
    assert a8["runs"] == 2 and a8["failed"] == 1
    assert (a8["seconds_median"], a8["seconds_min"], a8["embed_ms_median"]) == (7.0, 5.0, 70.0)


def test_one_repetition_prints_the_passes_alone(monkeypatch, capsys):
    monkeypatch.setattr(bench, "run", lambda config, boards, dtype, **kw: {"config": config, "seconds": 1.0})
    assert bench.main(["--config", "n2-default", "--boards", "2", "--dtype", "fp32"]) == 0
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [{"config": "n2-default", "seconds": 1.0}]


def test_embed_with_a_training_step_is_refused():
    with pytest.raises(SystemExit):
        bench.main(["--embed", "--train", "--config", "n2-default", "--boards", "2"])


def test_the_command_line_times_the_embedding_with_repetitions(capsys):
    # As the block runs A8's columns (``--embed --repeat 5``), at a smoke size: real passes, each in
    # its own subprocess.
    configs = ["n2-default", "6p-layout-ff8"]
    argv = ["--config", *configs, "--boards", "2", "--dtype", "fp32", "--layers", "1", "--embed", "--repeat", "2", "--threads", "2"]
    assert bench.main(argv) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    passes = [x for x in lines if "summary" not in x]
    summary = [x for x in lines if "summary" in x]
    assert [p["config"] for p in passes] == ["n2-default", "6p-layout-ff8", "6p-layout-ff8", "n2-default"]
    for p in passes:
        assert "error" not in p, p
        assert p["embed_ms"] > 0 and p["threads"] == 2
    assert [s["config"] for s in summary] == configs
    for s in summary:
        assert s["runs"] == 2 and s["failed"] == 0 and s["embed_ms_median"] > 0 and s["seconds_median"] >= 0
