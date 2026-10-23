"""Stage E10, H44: the offline audit lane's decision log written by Python's netplay is read by the
engine's own reader - t5d-audit samples it, runs its jobs and reports (the engine's
``tools/audit.hpp``; ``engine/tests/test_audit.cpp`` holds the engine writer's round trip).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from t5d.engine.capi import load_library
from t5d.engine.evaluator import UniformEvaluator
from t5d.engine.netplay import DECISIONS_FORMAT, SMALL, PlayConfig, decision_log_lines, play_game
from t5d.net import NetConfig
from tests.engine.helpers import requires_engine


def _audit_exe() -> Path:
    lib = load_library()
    return Path(lib.path).parent / ("t5d-audit.exe" if os.name == "nt" else "t5d-audit")


@requires_engine
def test_netplays_decision_log_is_read_by_t5d_audit(tmp_path):
    cfg = PlayConfig(
        variant="Very Small - Open",
        split="ply",
        playouts=12,
        cap_turns=4,
        cap_boards=40,
        max_actions=3,
        decision_log=True,
        hunt=False,
        settling=False,
        asset_map=False,
    )
    res = play_game(UniformEvaluator(), cfg, net_cfg=NetConfig(**SMALL))
    assert res.decision_log and res.actions
    lines = decision_log_lines(res)
    rows = [json.loads(line) for line in lines]
    assert rows[0]["kind"] == "game" and rows[0]["format"] == DECISIONS_FORMAT and rows[0]["version"] == 1
    finals = [r for r in rows if r.get("category") == "final"]
    assert len(finals) == len(res.actions)  # one final move per action, its step the action played
    assert [f["step"]["moves"] for f in finals] == rows[0]["actions"]
    exe = _audit_exe()
    if not exe.exists():
        pytest.fail(f"{exe} is not built (py engine/build.py)")
    log = tmp_path / "decisions.jsonl"
    log.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    jobs = tmp_path / "jobs.jsonl"
    out = subprocess.run(
        [str(exe), "sample", "--rates", "1,1,1", "--out", str(jobs), str(log)], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    sampled = [json.loads(line) for line in jobs.read_text(encoding="utf-8").splitlines() if line]
    assert sum(1 for r in sampled if r["kind"] == "job") == len(rows) - 1  # every event sampled at rate 1
    results = tmp_path / "results.jsonl"
    out = subprocess.run(
        [str(exe), "run", "--guard-nodes", "100000", "--mate-nodes", "50000", "--out", str(results), str(jobs)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert len([line for line in results.read_text(encoding="utf-8").splitlines() if line]) == len(rows) - 1
    out = subprocess.run([str(exe), "report", str(results)], capture_output=True, text=True, check=False)
    assert out.returncode == 0 and "finals" in out.stdout, out.stderr
