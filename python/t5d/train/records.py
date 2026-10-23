"""Register I15's training records: one versioned file per self-play game (spec section 12.8).

"Each record keeps what later choices need: the network's raw outputs at the root (WDL, value
distribution, prior over the legal moves and submit, error head), the root search value, the best
and played children's values, visit counts with section 12.2's forced and steered flags, the
prior-to-target divergence, proofs by solved-store reference, and the conditions - ruleset, board
count, network, search configuration, temperature applied, playout-cap flag ... Data generated
without these fields cannot get them back" (section 12.8). The generator (``train.selfplay``)
writes them; the target builder (``train.targets``) reads them - the same interface whether the
record came from a live game or from disk, so that section 15.2's row "targets re-derived from
stored records = the targets written at generation" compares the two.

**The format, version 1** (``RECORD_VERSION``): one ``.npz`` per game, written with
``allow_pickle=False`` and read the same way (nothing executable in an archive):

* ``header`` - a JSON object as UTF-8 bytes (``uint8``): the game's conditions and result
  (``GameRecord.header``; the keys below).
* ``d_<field>`` - one entry per decision for every ``DecisionRecord`` scalar (``DECISION_FIELDS``),
  in the dtype the table names; ``d_raw_wdl_logits`` ``(D, 3)``, ``d_raw_value_logits``
  ``(D, value_bins)``, ``d_root_wdl`` ``(D, 3)``.
* ``b_boards`` ``(sum N, 2)`` int32 with ``b_offset`` ``(D + 1,)``: each decision's boards as
  ``(timeline, half_turn)`` in the expansion's (the network input's) order - what the per-board
  targets index.
* ``c_moves`` (``MOVE_DTYPE``) and ``c_logit`` (float32), with ``c_offset``: each decision's **candidate moves** - the root leaf's, in the
  engine's order (the network's input rebuilt with ``Game.expand(legal_moves=...)`` and the
  search's legality budget has exactly these, issue 155) - and the network's **raw** policy
  logit for each (temperature 1, no noise). Their legality (legal, or unknown under the budget)
  is recorded for the root children (``e_legality``); ``t5d_game_expand`` reports none per move.
* ``e_<field>`` with ``e_offset``: each decision's **root children** (``CHILD_FIELDS``) - the
  search's statistics per instantiated, unpruned root edge (``t5d_root_child``): the candidate it
  plays (-1: submit), visits, the search's prior (after the noise) and the noise-free prior, the
  backed-up (q, d), the variance of its backed-up values, E3's factor, E10's boost and the
  forced and steered visit counts (I2's flags: 0 until forced playouts and E10's steering exist -
  issue 153), the child's key and its own network evaluation (WDL, moves-left, error; NaN where
  not evaluated), the solver bounds (search state, never a proof - H19, H23).

Header keys: ``record_version``, ``engine_version``, ``generator``, ``variant``, ``start_fen``
(None or the 5DFEN boards), ``ruleset``, ``rules``, ``width``, ``height``, ``even``,
``first_player``, ``first_turn``, ``network`` (``id`` - ``network_id``, a hash of the published
weights; ``config`` - the ``NetConfig``; ``seed``; ``swa_snapshots``), ``value_source``
(``"wdl"``: the search backs up W - L of the network's WDL, D3 since engine ABI round 2a),
``search`` (every ``t5d_search_params`` field of the full and the cheap searches, the expand
options and legality budget included - ``params_dict``), ``expand`` (the input's window and
typed relations, the register counts), ``generator_config``, ``seed``, ``result`` (``end``,
``winner``, ``result``, ``actions``, ``final_boards``, ``final_fingerprint``,
``final_active_timelines``), ``pgn`` (the game's 5dpgn), ``resign`` (``enabled`` and the
would-resign events, I18), ``proof_refs`` (solved-store references: always empty - no proof
enters a store before the H23/H41 sign-off), ``target_config`` (the target builder's
configuration that ``prior_to_target_kl`` was measured under - the builder's defaults when the
generator was given none). A synthetic record (``trainer.synthetic_record``, never the
generator's) also says ``synthetic``: the only rows the untrained build steps on.

Decision ``d_proof_ref`` is -1 throughout for the same reason. Another version is refused on
reading (``RecordVersionError``); a field is never removed or reinterpreted within a version.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from ..engine.capi import MOVE_DTYPE, CSearchParams, ExpandOptions
from ..net.quant import trunk_format

#: The record format's version: bumped on any change to a field's presence or meaning.
RECORD_VERSION = 1

#: Why a search stopped (``DecisionRecord.stop``): its playouts spent, the root needing none
#: (terminal or solved), no simulation possible, or I20's KL-adaptive rule.
STOP_CODES: tuple[str, ...] = ("playouts", "finished", "exhausted", "kl")
#: How the played child was chosen (``DecisionRecord.choice``): sampled from the visits at the
#: move temperature, the most visited (temperature 0), the most visited child known legal after
#: the chosen one stayed unknown under verification, or none played (the mover resigned, I18).
CHOICE_CODES: tuple[str, ...] = ("sampled", "most_visited", "verified_fallback", "resigned")

#: Each decision's scalar fields: name -> dtype (``d_<name>`` in the file).
DECISION_FIELDS: dict[str, str] = {
    "decision": "<i4",  # its index in the game
    "action": "<i4",  # the action (one player's turn - issue 305) it belongs to
    "move_in_action": "<i4",  # decisions of its action before it
    "acting_player": "<i1",  # 0 white, 1 black: every value field's perspective (D8)
    "moves_made": "<i4",  # A6: moves made in the action (0: a turn start)
    "n_boards": "<i4",
    "present": "<i4",
    "submit_possible": "?",
    "submit_legal": "?",
    "full_search": "?",  # I5: a full search (policy targets) or a cheap one
    "playout_cap": "<i8",  # the search's cap
    "playouts": "<i8",  # simulations run
    "stop": "<i1",  # STOP_CODES
    "move_temperature": "<f8",  # I19: the move temperature the choice was made at
    "root_policy_temperature": "<f8",  # I19: the root's softmax temperature (before the noise)
    "root_noise": "?",
    "search_seed": "<u8",
    "root_key": "<u8",  # t5d_search_summary.root_key
    "epoch": "<u8",
    "raw_error": "<f4",  # the network's raw outputs at the root (NaN: none)
    "raw_moves_left": "<f4",
    "raw_submit_logit": "<f4",  # NaN: submitting is not a legal candidate here
    "root_evaluated": "?",
    "root_visits": "<i8",
    "root_q": "<f8",  # the root search value (W - L backed up), the acting player's view
    "root_d": "<f8",
    "root_cpuct": "<f8",  # cpuct x E2's root scale (I2's PUCT replica)
    "root_best_since": "<i8",  # the effort label's change point (issue 154)
    "best_child": "<i4",  # the most visited child (the search's choice at temperature 0)
    "played_child": "<i4",  # the child played (-1: none - resigned)
    "choice": "<i1",  # CHOICE_CODES
    "verified_illegal": "<i4",  # chosen children classified illegal and removed before the choice stood
    "prior_to_target_kl": "<f8",  # KL(policy target || raw prior) under header["target_config"]; NaN: none
    "proof_ref": "<i8",  # -1: no proof (none may be stored before the sign-off)
}
#: Each decision's vector fields: name -> (dtype, width) - width None: the network's value bins.
DECISION_VECTORS: dict[str, tuple[str, int | None]] = {
    "raw_wdl_logits": ("<f4", 3),
    "raw_value_logits": ("<f4", None),
    "root_wdl": ("<f8", 3),  # the engine's normalised WDL of the root's evaluation (t5d_search_summary)
}
#: Each root child's fields: name -> dtype (``e_<name>``); ``network_wdl`` is ``(n, 3)``.
CHILD_FIELDS: dict[str, str] = {
    "candidate": "<i4",  # index into the decision's candidates; -1: the submit action
    "legality": "<i1",
    "visits": "<i8",
    "prior": "<f8",  # the search's (after the root noise)
    "prior_noise_free": "<f8",
    "q": "<f8",
    "d": "<f8",
    "value_variance": "<f8",
    "boost": "<f8",
    "exploration_scale": "<f8",
    "forced_visits": "<i8",
    "steered_visits": "<i8",
    "child_key": "<u8",
    "evaluated": "?",
    "network_wdl": "<f8",
    "network_moves_left": "<f8",
    "network_error": "<f8",
    "lo": "<i1",
    "hi": "<i1",
}


class RecordVersionError(ValueError):
    """A record of another format version."""


@dataclass
class Children:
    """One decision's root children, column by column (``CHILD_FIELDS``)."""

    candidate: np.ndarray
    legality: np.ndarray
    visits: np.ndarray
    prior: np.ndarray
    prior_noise_free: np.ndarray
    q: np.ndarray
    d: np.ndarray
    value_variance: np.ndarray
    boost: np.ndarray
    exploration_scale: np.ndarray
    forced_visits: np.ndarray
    steered_visits: np.ndarray
    child_key: np.ndarray
    evaluated: np.ndarray
    network_wdl: np.ndarray  # (n, 3)
    network_moves_left: np.ndarray
    network_error: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        for name, dtype in CHILD_FIELDS.items():
            a = np.asarray(getattr(self, name), dtype=dtype)
            setattr(self, name, a.reshape(-1, 3) if name == "network_wdl" else a.reshape(-1))

    def __len__(self) -> int:
        return int(self.visits.shape[0])

    @property
    def submit(self) -> np.ndarray:
        return self.candidate < 0


@dataclass
class DecisionRecord:
    """One decision: a search at one state of the game, and its choice (module docstring)."""

    decision: int
    action: int
    move_in_action: int
    acting_player: int
    moves_made: int
    n_boards: int
    present: int
    submit_possible: bool
    submit_legal: bool
    full_search: bool
    playout_cap: int
    playouts: int
    stop: str
    move_temperature: float
    root_policy_temperature: float
    root_noise: bool
    search_seed: int
    root_key: int
    epoch: int
    raw_error: float
    raw_moves_left: float
    raw_submit_logit: float
    root_evaluated: bool
    root_visits: int
    root_q: float
    root_d: float
    root_cpuct: float
    root_best_since: int
    best_child: int
    played_child: int
    choice: str
    verified_illegal: int
    raw_wdl_logits: np.ndarray
    raw_value_logits: np.ndarray
    root_wdl: np.ndarray
    boards: np.ndarray  # (N, 2) int32: (timeline, half_turn) per board, the input's order
    candidates: np.ndarray  # MOVE_DTYPE (M,)
    raw_logits: np.ndarray  # float32 (M,)
    children: Children
    prior_to_target_kl: float = math.nan
    proof_ref: int = -1

    def __post_init__(self) -> None:
        if self.stop not in STOP_CODES:
            raise ValueError(f"stop {self.stop!r} is not one of {STOP_CODES}")
        if self.choice not in CHOICE_CODES:
            raise ValueError(f"choice {self.choice!r} is not one of {CHOICE_CODES}")
        self.raw_wdl_logits = np.asarray(self.raw_wdl_logits, dtype="<f4").reshape(3)
        self.raw_value_logits = np.asarray(self.raw_value_logits, dtype="<f4").reshape(-1)
        self.root_wdl = np.asarray(self.root_wdl, dtype="<f8").reshape(3)
        self.boards = np.asarray(self.boards, dtype="<i4").reshape(-1, 2)
        self.candidates = np.asarray(self.candidates, dtype=MOVE_DTYPE).reshape(-1)
        self.raw_logits = np.asarray(self.raw_logits, dtype="<f4").reshape(-1)
        m = self.candidates.shape[0]
        if self.raw_logits.shape[0] != m:
            raise ValueError("one raw logit per candidate move")
        if self.boards.shape[0] != self.n_boards:
            raise ValueError(f"{self.boards.shape[0]} board keys for {self.n_boards} boards")
        c = self.children.candidate
        if c.size and (c.max() >= m or c.min() < -1):
            raise ValueError("a child's candidate index is outside the candidate list")
        if not -1 <= self.played_child < len(self.children) or not -1 <= self.best_child < len(self.children):
            raise ValueError("the best and played children are root children (or -1)")

    @property
    def move_count(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def turn_start(self) -> bool:
        return self.moves_made == 0

    @property
    def played_submit(self) -> bool:
        return self.played_child >= 0 and bool(self.children.candidate[self.played_child] < 0)

    def played_move(self) -> np.void | None:
        """The candidate move played (None: a submit, or nothing played)."""
        if self.played_child < 0:
            return None
        k = int(self.children.candidate[self.played_child])
        return None if k < 0 else self.candidates[k]


@dataclass
class GameRecord:
    """One game's record: its header (module docstring) and its decisions in order."""

    header: dict
    decisions: list[DecisionRecord] = field(default_factory=list)

    @property
    def end(self) -> str:
        return self.header["result"]["end"]

    @property
    def winner(self) -> int | None:
        return self.header["result"]["winner"]

    @property
    def n_actions(self) -> int:
        return int(self.header["result"]["actions"])


# --- conditions ------------------------------------------------------------------------------------


def network_id(model) -> str:
    """The network's identity: BLAKE2b-128 over its state dict in order - each tensor's name,
    dtype, shape and bytes (I15's "network" condition; an SWA-published network has its own) -
    and, for a network in an 8-bit trunk format (G2, ``NetConfig.trunk_quant``), that format
    (``quant.trunk_format``): the same weights in FP8 compute another function, so they are
    another network. A network in "none" hashes its weights alone, as before the format existed."""
    h = hashlib.blake2b(digest_size=16)
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "trunk_quant", "none") != "none":
        h.update(b"trunk_format")
        h.update(repr(trunk_format(cfg)).encode("ascii"))
    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(repr(tuple(t.shape)).encode("ascii"))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def params_dict(p: CSearchParams) -> dict:
    """Every ``t5d_search_params`` field by name (the expand options as a nested object)."""
    out = {}
    for name, _ in CSearchParams._fields_:
        v = getattr(p, name)
        if isinstance(v, ExpandOptions):
            out[name] = {n: int(getattr(v, n)) for n, _ in ExpandOptions._fields_}
        else:
            out[name] = v.item() if hasattr(v, "item") else v
    return out


def params_from_dict(d: dict) -> CSearchParams:
    """The inverse of ``params_dict``."""
    p = CSearchParams()
    for name, _ in CSearchParams._fields_:
        if name == "expand":
            setattr(p, name, ExpandOptions(**d[name]))
        else:
            setattr(p, name, d[name])
    return p


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON: {type(o).__name__}")


def canonical_header(header: dict) -> dict:
    """The header as it reads back from a file (JSON round trip: tuples become lists)."""
    return json.loads(json.dumps(header, default=_json_default, allow_nan=True))


# --- writing and reading ------------------------------------------------------------------------------


def _concat(parts: Sequence[np.ndarray], dtype, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(parts) + 1, dtype="<i8")
    for i, p in enumerate(parts):
        offsets[i + 1] = offsets[i] + p.shape[0]
    shape = (0,) if width is None else (0, width)
    data = np.concatenate([np.asarray(p, dtype=dtype) for p in parts]) if parts else np.zeros(shape, dtype=dtype)
    return data, offsets


def to_arrays(record: GameRecord) -> dict[str, np.ndarray]:
    """The record as the file's named arrays (module docstring)."""
    ds = record.decisions
    out: dict[str, np.ndarray] = {}
    header = json.dumps(record.header, default=_json_default, allow_nan=True, sort_keys=True)
    out["header"] = np.frombuffer(header.encode("utf-8"), dtype=np.uint8).copy()
    for name, dtype in DECISION_FIELDS.items():
        values = [getattr(d, name) for d in ds]
        if name == "stop":
            values = [STOP_CODES.index(v) for v in values]
        elif name == "choice":
            values = [CHOICE_CODES.index(v) for v in values]
        out[f"d_{name}"] = np.array(values, dtype=dtype).reshape(-1)
    for name, (dtype, width) in DECISION_VECTORS.items():
        rows = [getattr(d, name) for d in ds]
        w = width if width is not None else (rows[0].shape[0] if rows else 0)
        out[f"d_{name}"] = np.stack(rows).astype(dtype) if rows else np.zeros((0, w), dtype=dtype)
    out["b_boards"], out["b_offset"] = _concat([d.boards for d in ds], "<i4", 2)
    out["c_moves"], out["c_offset"] = _concat([d.candidates for d in ds], MOVE_DTYPE)
    out["c_logit"], _ = _concat([d.raw_logits for d in ds], "<f4")
    for name, dtype in CHILD_FIELDS.items():
        width = 3 if name == "network_wdl" else None
        out[f"e_{name}"], offsets = _concat([getattr(d.children, name) for d in ds], dtype, width)
    out["e_offset"] = offsets if ds else np.zeros(1, dtype="<i8")
    return out


def from_arrays(arrays: dict[str, np.ndarray]) -> GameRecord:
    """A record from the file's named arrays; another version is refused."""
    header = json.loads(bytes(arrays["header"].astype(np.uint8)).decode("utf-8"))
    version = header.get("record_version")
    if version != RECORD_VERSION:
        raise RecordVersionError(f"a record of version {version!r}; this reader reads version {RECORD_VERSION}")
    n = int(arrays["d_decision"].shape[0])
    b_off, c_off, e_off = arrays["b_offset"], arrays["c_offset"], arrays["e_offset"]
    decisions = []
    for i in range(n):
        kw = {}
        for name in DECISION_FIELDS:
            v = arrays[f"d_{name}"][i]
            if name == "stop":
                kw[name] = STOP_CODES[int(v)]
            elif name == "choice":
                kw[name] = CHOICE_CODES[int(v)]
            elif v.dtype == np.bool_:
                kw[name] = bool(v)
            elif v.dtype.kind in "iu":
                kw[name] = int(v)
            else:
                kw[name] = float(v)
        for name in DECISION_VECTORS:
            kw[name] = arrays[f"d_{name}"][i].copy()
        b0, b1 = int(b_off[i]), int(b_off[i + 1])
        c0, c1 = int(c_off[i]), int(c_off[i + 1])
        e0, e1 = int(e_off[i]), int(e_off[i + 1])
        kw["boards"] = arrays["b_boards"][b0:b1].copy()
        kw["candidates"] = arrays["c_moves"][c0:c1].copy()
        kw["raw_logits"] = arrays["c_logit"][c0:c1].copy()
        kw["children"] = Children(**{name: arrays[f"e_{name}"][e0:e1].copy() for name in CHILD_FIELDS})
        decisions.append(DecisionRecord(**kw))
    return GameRecord(header=header, decisions=decisions)


def write_game(path: str | Path, record: GameRecord) -> Path:
    """Write one game's record (compressed ``.npz``); the path written."""
    path = Path(path)
    buf = io.BytesIO()
    np.savez_compressed(buf, **to_arrays(record))
    path.write_bytes(buf.getvalue())
    return path


def read_game(path: str | Path) -> GameRecord:
    """Read one game's record (``allow_pickle=False``); ``RecordVersionError`` for another version."""
    with np.load(Path(path), allow_pickle=False) as z:
        return from_arrays({k: z[k] for k in z.files})


# --- comparing ------------------------------------------------------------------------------------------


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype.names:
            return a.tobytes() == b.tobytes()
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def record_differences(a: GameRecord, b: GameRecord) -> list[str]:
    """Where two records differ, field by field and bit for bit (empty: the same record) - the
    header compared as it reads back from a file (``canonical_header``)."""
    out = []
    ha, hb = canonical_header(a.header), canonical_header(b.header)
    for k in sorted(set(ha) | set(hb)):
        if json.dumps(ha.get(k), sort_keys=True) != json.dumps(hb.get(k), sort_keys=True):
            out.append(f"header[{k!r}]")
    if len(a.decisions) != len(b.decisions):
        return [*out, f"{len(a.decisions)} decisions against {len(b.decisions)}"]
    for i, (x, y) in enumerate(zip(a.decisions, b.decisions, strict=True)):
        for f in fields(DecisionRecord):
            if f.name == "children":
                for name in CHILD_FIELDS:
                    if not _same(getattr(x.children, name), getattr(y.children, name)):
                        out.append(f"decision {i}: children.{name}")
            elif not _same(getattr(x, f.name), getattr(y, f.name)):
                out.append(f"decision {i}: {f.name}")
    return out


__all__ = [
    "CHILD_FIELDS",
    "CHOICE_CODES",
    "DECISION_FIELDS",
    "DECISION_VECTORS",
    "RECORD_VERSION",
    "STOP_CODES",
    "Children",
    "DecisionRecord",
    "GameRecord",
    "RecordVersionError",
    "canonical_header",
    "from_arrays",
    "network_id",
    "params_dict",
    "params_from_dict",
    "read_game",
    "record_differences",
    "to_arrays",
    "write_game",
]
