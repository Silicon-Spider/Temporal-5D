"""The default path, bit for bit: a record of the network's outputs on fixed positions, to compare a
change with the commit before it (decision 7's evidence, BUILD_LOG session log 2026-11-27; review
finding SN-7). Every output is recorded whole and compared with ``torch.equal``.

    python -m t5d.net.default_path record OUT.pt        # from python/: this tree's t5d
    python -m t5d.net.default_path compare BEFORE.pt AFTER.pt

To record a commit that predates this module, run this file against that commit's package:

    git archive <commit> python | tar -x -C <dir>
    PYTHONPATH=<dir>/python python python/t5d/net/default_path.py record before.pt

**What is recorded - 119 outputs** (each the trunk and every head's logits, ``FIELDS``), on three
synthetic positions (``positions``: seeds 3, 5 and 7 - 10 boards and 3 timelines, 16 boards and 5
with deep travel, 14 boards and 8), 3 torch threads:

* the v1 network as built (``NetConfig()``, model seed 0; its parameter count is recorded too): the
  dense forward of each position alone and of the three batched - 4;
* the cached configuration (``attn_scope="block_causal"``, 2 layers, model seed 1) under nine arms -
  the default, ``attn_impl`` chunked, dense and flex, FP8 frozen K/V, B12's board-pair smolgen, C20's
  second reading, current frozen features, and no global registers (4 per-board registers) - each
  position's full pass, its cached evaluation in the causal epoch and with the root refresh, and the
  two-level reference anchored at it: 9 x 3 x 4 = 108;
* the search-tree fixture with the root refresh and node refreshes (V = 1; model seed 2), three
  physical moves down from a synthetic root (seed 11, 12 boards, 4 timelines): 3;
* register A5 in the dense scope (model seed 3), each position's A5 view: 3;
* and the v1 parameter count: 1.
"""

from __future__ import annotations

import sys

import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output
from t5d.net.state import relayout, to_piece_centric
from t5d.net.tree import Multiverse, SearchTree

#: The outputs recorded of each evaluation.
FIELDS = ("trunk", "policy_logits", "wdl_logits", "value_logits", "board_logits", "square_logits", "leave_logits", "submit_raw")

#: The cached configuration's arms (``NetConfig`` overrides).
ARMS = (
    ("bc", {}),
    ("bc_chunked", dict(attn_impl="chunked")),
    ("bc_dense", dict(attn_impl="dense")),
    ("bc_flex", dict(attn_impl="flex")),
    ("bc_fp8", dict(frozen_kv_format="fp8")),
    ("bc_smolgen", dict(board_pair_smolgen=True)),
    ("bc_r1", dict(supersession_scope="past_and_epoch_history")),
    ("bc_current", dict(frozen_features="current")),
    ("bc_noglobal", dict(global_regs=0, regs_per_board=4)),
)


def fields(out) -> dict[str, torch.Tensor]:
    return {name: getattr(out, name).detach().clone() for name in FIELDS if getattr(out, name, None) is not None}


def positions():
    return [
        synthetic_state(3, 10, n_timelines=3),
        synthetic_state(5, 16, n_timelines=5, deep_travel_prob=1.0),
        synthetic_state(7, 14, n_timelines=8),
    ]


def record(path: str) -> int:
    torch.set_num_threads(3)
    res: dict = {}
    with torch.no_grad():
        v1 = build_model(NetConfig(), seed=0).eval()
        res["v1_params"] = torch.tensor(sum(p.numel() for p in v1.parameters()))
        for i, s in enumerate(positions()):
            res[f"v1_dense_{i}"] = fields(v1(collate([s]), return_trunk=True))
        res["v1_batch"] = fields(v1(collate(positions()), return_trunk=True))
        for name, flags in ARMS:
            m = build_model(NetConfig(n_layers=2, attn_scope="block_causal", **flags), seed=1).eval()
            for i, position in enumerate(positions()):
                s = relayout(position, flags["regs_per_board"], 0) if flags.get("global_regs") == 0 else position
                res[f"{name}_full_{i}"] = fields(m(collate([s]), return_trunk=True))
                for cname, cfg in (("causal", CacheConfig()), ("refresh", CacheConfig(root_refresh=True))):
                    ev = CachedEvaluator(m, cfg)
                    res[f"{name}_{cname}_{i}"] = fields(ev.evaluate(s, ev.root_epoch(s)))
                res[f"{name}_ref_{i}"] = fields(reference_output(m, s, anchor=s))
        m = build_model(NetConfig(n_layers=2, attn_scope="block_causal"), seed=2).eval()
        root = Multiverse.from_state(synthetic_state(11, 12, n_timelines=4), seed=1)
        tree = SearchTree(CachedEvaluator(m, CacheConfig(root_refresh=True, node_refresh=True, refresh_visit_threshold=1)), root)
        node = tree.root
        for k, p in enumerate(sorted(root.heads())[:3]):
            node = tree.child(node, ("physical", p))
            tree.visit(node)
            res[f"tree_{k}"] = fields(tree.evaluate(node))
        m5 = build_model(NetConfig(n_layers=2, piece_centric=True), seed=3).eval()
        for i, s in enumerate(positions()):
            res[f"a5_dense_{i}"] = fields(m5(collate([to_piece_centric(s, strict=False)]), return_trunk=True))
    torch.save(res, path)
    print(f"recorded {len(res)} outputs to {path}")
    return 0


def compare(before: str, after: str) -> int:
    x, y = torch.load(before), torch.load(after)
    if x.keys() != y.keys():
        print("the records hold different outputs:", sorted(set(x) ^ set(y)))
        return 1
    bad = []
    for k in x:
        if isinstance(x[k], dict):
            bad += [(k, f, float((x[k][f] - y[k][f]).abs().max())) for f in x[k] if not torch.equal(x[k][f], y[k][f])]
        elif not torch.equal(x[k], y[k]):
            bad.append((k, x[k], y[k]))
    print(f"compared {len(x)} outputs:", "bit for bit" if not bad else f"{len(bad)} differ, e.g. {bad[:5]}")
    print("v1 parameters:", int(x["v1_params"]), int(y["v1_params"]))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "record":
        return record(argv[1])
    if len(argv) == 3 and argv[0] == "compare":
        return compare(argv[1], argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
