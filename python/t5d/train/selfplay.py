"""The training self-play generator: games through the engine's search, written as I15's records.

Register G9: PyTorch hosts the network and drives the engine's search through the C API, so the
generator is Python. It plays as ``engine.netplay``'s self-play does - a turn start judged first
(E4's exhaustive detector, then I11's cap), a fresh search per decision sharing the previous
one's evaluation cache (C14), root noise on, a chosen child made known legal before it is played
- and writes, per decision, the record ``train.records`` defines (register I15, spec section
12.8). ``netplay.play_game`` is unchanged: it plays and writes a 5dpgn, and writes no record.

* **D2's arms**: the board-move split (``"ply"``), the tree form (``"tree"``) and the flat form
  (``"flat"``). Under the flat form a decision is a whole turn: its root children are turns (record
  version 2: ``DecisionRecord.flat``, each child's moves in ``turn_moves``), the turn played is
  applied move by move and submitted, and its policy target is the per-board visit marginals that
  D2's flat head trains on (``train.targets``; spec issue 242 - the refusal of issue 234(b) lifted
  once that target existed). Every decision records the acting player's playable boards (the
  expansion's ``playable``) and the network's raw per-board leave logits.
* **D9's optimistic policy is refused** (section 8.1: "kept out of self-play"): every search runs
  with ``run_search``'s ``self_play``.
* **The raw root outputs** (I15) are the network's evaluation of the decision's own input - the
  root state expanded as a search leaf is (``Game.position`` with the search's legal-moves filter
  and legality budget: the same candidate moves in the same order, issue 155) - evaluated once
  per decision by the same evaluator. The search's own evaluation of its root is often a cache
  hit served from an earlier search (C14) that Python never sees, so it is not read back; the
  engine's normalised WDL of it is recorded beside (``root_wdl``) and the tests hold the two
  together. A reading (spec issue 234): BUILD_LOG's I15 row planned a memo of every delivered
  evaluation keyed by ``root_key``; the root's own evaluation is exact for the archive (the
  network and the rebuilt input give it again) and needs no memo spanning a game.
* **Proofs**: none. The proof channel is off (H23, H41): records carry ``proof_ref`` -1 and an
  empty ``proof_refs``.
* **The asset map: off** (``GeneratorConfig.asset_map``, a placeholder; the E10 review): I2's
  steered-visit definition is unvalidated (issue 179), so self-play does not steer - the records'
  boosts and steered visits are 0. Hunt and the settling scans run (their proofs withheld).
* **Playout cap randomisation (I5, section 12.5)**: each decision is a full search with probability
  ``PCRConfig.full_fraction`` (KataGo's 0.25), drawn from the game's seed and the decision's index,
  else a cheap one at ``cheap_playouts`` without root noise (placeholders); only full searches emit
  policy-type targets (``train.targets``). Difficulty-biasing is allowed "only as a mixture" (I5) -
  ``difficulty_mix`` > 0 is not built (it needs the effort head's output at the root, which the
  evaluator does not deliver; spec issue 236).
* **KL-adaptive full searches (I20, a prototype behind ``KLAdaptiveConfig.enabled``, off)**: a full
  search runs in chunks; once past ``min_playouts`` it stops when the root's visit distribution
  moved less than ``min_gain`` (KL per playout) over the last chunk - Lc0's T50 rule, bounded by
  the full cap ("composes with the cap randomisation", section 12.5). The stop is recorded (``kl``)
  and is the effort label's convergence.
* **Resignation and the false-resign check (I18, section 12.8)**: at a turn start the mover resigns
  once its search value has been below ``-threshold`` at ``consecutive`` of its turn starts in a row;
  in a ``disabled_fraction`` (10%, AlphaGo Zero's protocol) of games resignation is off and every
  would-resign event is recorded instead, so that ``false_resign_report`` measures false
  resignations per board-count range. The threshold is None (no resignation) until the value head
  is calibrated - a placeholder, as are ``consecutive`` and the ranges.
* **The move-temperature schedule (I19, section 12.8)**: Lc0's T60 read as Lc0's options define it -
  ``temp-cutoff-move``, "Move number, starting from which endgame temperature is used rather than
  initial temperature", and ``temp-endgame``, "Temperature used during endgame (starting from cutoff
  move). Endgame temperature doesn't decay" - so 0.9 before the cutoff turn (60) and a constant
  endgame temperature 0.45 from it (the main loop's ruling on the review, issue 236(d)); a linear
  decay before the cutoff is an option (``decay_to``, off). A turn is one player's action (issue
  305). The root policy temperature KataGo's 1.25 decaying to 1.1 over 60 turns (register I19). All
  placeholders, SPSA-tuned later (G7); each decision records the temperatures it ran at.

Nothing here trains (the untrained build's rule): the generator only plays and records.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..engine.capi import (
    DEFAULT_MATE_NODES,
    MOVE_DTYPE,
    T5D_LEGALITY_ILLEGAL,
    T5D_LEGALITY_LEGAL,
    T5D_SPLIT_BOARD_MOVE,
    T5D_SPLIT_FLAT,
    T5D_SPLIT_TREE,
)
from ..engine.convert import Position, relations_for
from ..engine.evaluator import Evaluation, Evaluator, policy_source, run_search
from ..engine.expansion import Move
from ..engine.game import Game, relation_bits
from ..engine.netplay import turn_start_verdict, write_record
from ..engine.search import RootChild, Search, search_params
from ..net.config import NetConfig
from ..net.quant import TRUNK_QUANT_FIELDS, trunk_format
from .records import CANDIDATE_SUBMIT, CANDIDATE_TURN, RECORD_VERSION, Children, DecisionRecord, GameRecord, network_id, params_dict
from .targets import GameTargets, TargetConfig, build_targets

#: D2's arms the generator plays (module docstring).
SPLITS = {"ply": T5D_SPLIT_BOARD_MOVE, "tree": T5D_SPLIT_TREE, "flat": T5D_SPLIT_FLAT}

_GOLDEN = 0x9E3779B97F4A7C15
_MASK = 0xFFFFFFFFFFFFFFFF


def mix(x: int) -> int:
    """SplitMix64's finaliser: a decision's seeds from the game's (as netplay's)."""
    x &= _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def decision_seed(game_seed: int, decision: int, salt: int = 0) -> int:
    """The seed of decision ``decision`` of a game (``salt`` separates the streams: the search's,
    the choice's, PCR's ...), so that a record re-derives every draw from its seeds."""
    return mix(int(game_seed) + _GOLDEN * (int(decision) + 1) + 0xD1B54A32D192ED03 * int(salt))


def decision_random(game_seed: int, decision: int, salt: int) -> float:
    """A uniform number in [0, 1) for one draw of one decision (``decision_seed``)."""
    return float(np.random.default_rng(decision_seed(game_seed, decision, salt)).random())


#: The draws' salts (``decision_seed``).
SALT_SEARCH, SALT_CHOICE, SALT_PCR, SALT_RESIGN = 0, 1, 2, 3


@dataclass(frozen=True)
class PCRConfig:
    """Register I5's playout cap randomisation (module docstring): PLACEHOLDERS but KataGo's fraction."""

    enabled: bool = True
    full_fraction: float = 0.25  # KataGo's
    cheap_playouts: int | None = None  # None: the full cap / 8 (section 12.5's "cap ~ 100" against 800)
    cheap_root_noise: bool = False  # KataGo's cheap searches add no noise
    difficulty_mix: float = 0.0  # I5: "only as a mixture"; > 0 not built (issue 236)

    def cheap_cap(self, full: int) -> int:
        return max(1, full // 8) if self.cheap_playouts is None else int(self.cheap_playouts)


@dataclass(frozen=True)
class KLAdaptiveConfig:
    """Register I20 (a prototype, off): Lc0's KL-adaptive full searches - PLACEHOLDERS."""

    enabled: bool = False
    interval: int = 64  # playouts per chunk
    min_playouts: int = 128
    min_gain: float = 1e-4  # KL(now || before) per playout below which the search stops


@dataclass(frozen=True)
class ResignConfig:
    """Register I18: resignation with the false-resign check - PLACEHOLDERS but the 10%."""

    threshold: float | None = None  # resign below -threshold (the search value); None: never (uncalibrated)
    consecutive: int = 2  # of the mover's turn starts in a row
    disabled_fraction: float = 0.1  # I18: AlphaGo Zero's protocol
    ranges: tuple[int, ...] = (50, 100, 200)  # board-count ranges the false-resign rate is reported in (6b's)


@dataclass(frozen=True)
class TemperatureSchedule:
    """Register I19's self-play exploration schedules (module docstring) - PLACEHOLDERS."""

    start: float = 0.9  # Lc0's T60: the initial temperature, before the cutoff
    cutoff_turn: int = 60  # Lc0's temp-cutoff-move: from this turn on, the endgame temperature
    endgame: float = 0.45  # Lc0's temp-endgame: constant from the cutoff on (it "doesn't decay")
    decay_to: float | None = None  # an optional linear decay before the cutoff, start -> decay_to (None: none)
    decay_turns: int = 60  # the optional decay's span
    root_start: float = 1.25  # KataGo's root policy temperature, early ...
    root_end: float = 1.1  # ... and late
    root_decay_turns: int = 60


@dataclass(frozen=True)
class GeneratorConfig:
    """One game's settings. Every search constant not named here is the engine's default
    (``t5d_search_params_default``) unless ``search_overrides`` names it; the proof channel is
    off (H41)."""

    variant: str = "Standard"
    start_fen: tuple[str, ...] | None = None  # a custom start (5DFEN boards) instead of ``variant``'s
    split: str = "ply"  # D2: "ply", "tree" or "flat" (module docstring)
    playouts: int = 800  # the full search's cap (a PLACEHOLDER)
    batch_size: int = 8
    cap_turns: int = 6  # I11 (0: none)
    cap_boards: int = 80
    max_actions: int = 200  # a safety stop, not a rule (scored like I11's cap: unscored, "action_limit")
    seed: int = 1
    window_k: int = 8
    root_noise: bool = True
    #: The asset map's boosts (H13) in self-play: OFF - a named placeholder - until I2's
    #: steered-visit definition is validated (the E10 review measured the counterfactual count
    #: biasing the policy target against the boosted moves; build-log issues 179, 189). The
    #: engine's own default and match play (``netplay``) keep it on.
    asset_map: bool = False
    pcr: PCRConfig = PCRConfig()  # I5
    kl: KLAdaptiveConfig = KLAdaptiveConfig()  # I20
    resign: ResignConfig = ResignConfig()  # I18
    schedule: TemperatureSchedule = TemperatureSchedule()  # I19
    #: Further ``t5d_search_params`` fields by name, for every search (e.g. the legality budget).
    search_overrides: tuple[tuple[str, object], ...] = ()

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(
                f"split must be one of {sorted(SPLITS)} (the enumerating whole-turn arm, 'turn', is retired: D2's flat form replaced it)"
            )
        if self.playouts < 1 or self.batch_size < 1 or self.max_actions < 1:
            raise ValueError("playouts, batch_size and max_actions must be positive")
        pcr = self.pcr
        if not 0.0 <= pcr.full_fraction <= 1.0 or not 1 <= pcr.cheap_cap(self.playouts) <= self.playouts:
            raise ValueError("PCR: full_fraction in [0, 1], and the cheap cap between 1 and the full cap (I5)")
        if not 0.0 <= pcr.difficulty_mix < 1.0:
            raise ValueError("difficulty-biasing only as a mixture with a random component (I5): difficulty_mix in [0, 1)")
        if pcr.difficulty_mix > 0.0:
            raise NotImplementedError("PCR's difficulty mixture needs the effort head's output at the root, not delivered (issue 236)")
        if self.kl.enabled and (self.kl.interval < 1 or self.kl.min_playouts < 1 or self.kl.min_gain < 0):
            raise ValueError("I20: interval and min_playouts positive, min_gain >= 0")
        r = self.resign
        if r.threshold is not None and not (0.0 < r.threshold <= 1.0 and r.consecutive >= 1):
            raise ValueError("I18: the resignation threshold in (0, 1], consecutive >= 1")
        if not 0.0 <= r.disabled_fraction <= 1.0:
            raise ValueError("I18: disabled_fraction in [0, 1]")
        t = self.schedule
        moves = (t.start, t.endgame) + (() if t.decay_to is None else (t.decay_to,))
        if min(moves) < 0 or min(t.root_start, t.root_end) <= 0 or min(t.decay_turns, t.root_decay_turns, t.cutoff_turn) < 1:
            raise ValueError("I19: temperatures non-negative (root temperatures positive), the cutoff and decay spans positive")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["search_overrides"] = [[k, v] for k, v in self.search_overrides]
        return d


@dataclass
class GeneratedGame:
    """One game: its I15 record, and what the game needed besides."""

    record: GameRecord
    actions: list[list[Move]]
    #: Each decision's network input, kept when ``generate_game(keep_positions=True)`` (the live
    #: side of section 15.2's round trip; the reply target's positions).
    positions: list[Position] = field(default_factory=list)
    #: The targets written at generation (``generate_game(target_config=...)``, the builder's
    #: defaults when none is given): the live side of section 15.2's I15 row.
    targets: GameTargets | None = None
    seconds: float = 0.0


def search_parameters(game: Game, cfg: GeneratorConfig, relations: int, *, seed: int, noise: bool, root_temperature: float = 1.0):
    """One decision's ``t5d_search_params``."""
    params = dict(
        split=SPLITS[cfg.split],
        seed=seed,
        root_noise=int(noise),
        batch_size=cfg.batch_size,
        turn_cap_turns=cfg.cap_turns,
        turn_cap_boards=cfg.cap_boards,
        expand=(cfg.window_k, "all", relations),
        root_policy_temperature=float(root_temperature),
        asset_map=int(cfg.asset_map),
    )
    params.update(dict(cfg.search_overrides))
    return search_params(game.lib, **params)


def root_position(game: Game, params, net_cfg: NetConfig, window_k: int) -> Position:
    """The decision's network input: the root state expanded as the search's leaves are (the
    legal-moves filter and the legality budget of ``params``; issue 155)."""
    return game.position(
        regs_per_board=net_cfg.regs_per_board,
        global_regs=net_cfg.global_regs,
        relations=relations_for(net_cfg),
        window_k=window_k,
        legal_moves=bool(params.legal_moves_only),
        legality_checks=int(params.legality_checks),
        legality_nodes=int(params.legality_nodes),
    )


def _move_key(m: Move) -> tuple[int, ...]:
    return tuple(int(getattr(m, name)) for name in MOVE_DTYPE.names)


def candidate_index(position: Position) -> dict[tuple[int, ...], int]:
    """Each candidate move's index by its fields (``MOVE_DTYPE``'s)."""
    return {tuple(int(x) for x in row.tolist()): i for i, row in enumerate(position.moves)}


def turn_moves(children: list[RootChild]) -> np.ndarray:
    """D2's flat form: the root children's turns, their moves concatenated in child order
    (``MOVE_DTYPE``; record version 2's ``turn_moves``)."""
    rows = [_move_key(m) for c in children for m in c.moves]
    return np.array(rows, dtype=MOVE_DTYPE) if rows else np.zeros(0, dtype=MOVE_DTYPE)


def children_record(children: list[RootChild], index: dict[tuple[int, ...], int], *, flat: bool = False) -> Children:
    """The root children as I15's columns; each child's candidate index (-1: submit; -2: a whole turn,
    D2's flat form - its moves in ``turn_moves``, ``turn_length`` each)."""
    cand = []
    for c in children:
        if flat:
            cand.append(CANDIDATE_TURN)
            continue
        if c.submit:
            cand.append(CANDIDATE_SUBMIT)
            continue
        if len(c.moves) != 1:
            raise ValueError("a root child of the board-move split plays one move")
        key = _move_key(c.moves[0])
        if key not in index:
            raise RuntimeError(f"root child {c.index}'s move is not one of the decision's candidate moves (the input is not the leaf's)")
        cand.append(index[key])
    return Children(
        candidate=cand,
        legality=[c.legality for c in children],
        visits=[c.visits for c in children],
        prior=[c.prior for c in children],
        prior_noise_free=[c.prior_noise_free for c in children],
        q=[c.q for c in children],
        d=[c.d for c in children],
        value_variance=[c.value_variance for c in children],
        boost=[c.boost for c in children],
        exploration_scale=[c.exploration_scale for c in children],
        forced_visits=[c.forced_visits for c in children],
        steered_visits=[c.steered_visits for c in children],
        child_key=[c.child_key for c in children],
        evaluated=[c.evaluated for c in children],
        network_wdl=np.array([c.network_wdl for c in children], dtype=np.float64).reshape(-1, 3),
        network_moves_left=[c.network_moves_left for c in children],
        network_error=[c.network_error for c in children],
        lo=[c.lo for c in children],
        hi=[c.hi for c in children],
        turn_length=[len(c.moves) if flat else 0 for c in children],
    )


def network_info(evaluator, net_cfg: NetConfig, *, seed: int | None = None, swa_snapshots: int | None = None) -> dict:
    """The record's ``network`` condition for an evaluator with a ``model`` (its weights' hash,
    its trunk format folded in when it has one - ``records.network_id``); ``swa_snapshots`` None:
    the count a published network carries (``Trainer.publish`` sets ``model.swa_snapshots``), else
    0. The trunk format (G2's ``trunk_quant*``) is the evaluated network's own, never the caller's:
    a ``net_cfg`` whose format disagrees with the network's is refused (QAT switches a network's
    format in place - ``train.trainer.apply_qat`` - so a caller's configuration can lag it)."""
    model = getattr(evaluator, "model", None)
    if swa_snapshots is None:
        swa_snapshots = getattr(model, "swa_snapshots", 0) if model is not None else 0
    config = asdict(net_cfg)
    own = getattr(model, "cfg", None)
    if isinstance(own, NetConfig):
        if trunk_format(own) != trunk_format(net_cfg):
            raise ValueError(
                f"the record's network condition: the evaluated network's trunk format is {trunk_format(own)}, the "
                f"configuration given says {trunk_format(net_cfg)} (register G2, G8; I15)"
            )
        config.update({f: getattr(own, f) for f in TRUNK_QUANT_FIELDS})
    return {
        "id": None if model is None else network_id(model),
        "config": config,
        "seed": seed,
        "swa_snapshots": int(swa_snapshots),
    }


def _choose(search: Search, temperature: float, random: float, rerun: Callable[[], None]) -> tuple[int, str, int]:
    """The decision's child: sampled from the visits at ``temperature`` (<= 0: the most visited),
    then made known legal - a chosen unknown child is classified exactly first (illegal: removed
    and the choice made again; still unknown: the most visited child known legal). The pick, how
    it was chosen (``records.CHOICE_CODES``) and the children found illegal on the way."""
    illegal = 0

    def choose_now() -> int:
        if not any(c.visits > 0 for c in search.children()):
            rerun()  # every visited child was removed: search again
        return search.choose_by_visits(temperature, random)

    choice = "sampled" if temperature > 0 else "most_visited"
    pick = choose_now()
    while search.child(pick).legality != T5D_LEGALITY_LEGAL:
        verdict = search.verify(pick)
        if verdict == T5D_LEGALITY_LEGAL:
            break
        if verdict == T5D_LEGALITY_ILLEGAL:
            illegal += 1
            pick = choose_now()
            continue
        legal = [c for c in search.children() if c.legality == T5D_LEGALITY_LEGAL]
        if not legal:
            raise RuntimeError("no root child could be shown legal (the memory ceiling)")
        pick, choice = max(legal, key=lambda c: c.visits).index, "verified_fallback"
        break
    return pick, choice, illegal


def _decay(start: float, end: float, span: int, turn: int) -> float:
    return float(start + (end - start) * min(1.0, turn / span))


def move_temperature(cfg: GeneratorConfig, action: int) -> float:
    """I19's move temperature at turn ``action`` (a turn one player's action, issue 305), Lc0's
    semantics: ``start`` before ``cutoff_turn`` (with ``decay_to``, linear from ``start`` to it over
    ``decay_turns``), the constant ``endgame`` temperature from ``cutoff_turn`` on."""
    t = cfg.schedule
    if action >= t.cutoff_turn:
        return float(t.endgame)
    return float(t.start) if t.decay_to is None else _decay(t.start, t.decay_to, t.decay_turns, action)


def root_temperature(cfg: GeneratorConfig, action: int) -> float:
    """I19's root policy temperature at turn ``action`` (linear from ``root_start`` to ``root_end``)."""
    t = cfg.schedule
    return _decay(t.root_start, t.root_end, t.root_decay_turns, action)


def is_full_search(cfg: GeneratorConfig, decision: int) -> bool:
    """I5: whether decision ``decision`` of the game runs a full search (re-derivable from the seed)."""
    return not cfg.pcr.enabled or decision_random(cfg.seed, decision, SALT_PCR) < cfg.pcr.full_fraction


def resign_enabled(cfg: GeneratorConfig) -> bool:
    """I18: whether this game may resign (off in ``disabled_fraction`` of games, by the seed)."""
    return cfg.resign.threshold is not None and decision_random(cfg.seed, -1, SALT_RESIGN) >= cfg.resign.disabled_fraction


def _visit_shares(search: Search) -> dict[tuple, float]:
    children = search.children()
    total = float(sum(c.visits for c in children))
    if total <= 0:
        return {}

    def key(c: RootChild) -> tuple:
        return ("submit",) if c.submit else tuple(_move_key(m) for m in c.moves)  # a whole turn under the flat form

    return {key(c): c.visits / total for c in children if c.visits > 0}


def _kl_shares(now: dict, before: dict) -> float:
    out = 0.0
    for k, p in now.items():
        q = before.get(k, 0.0)
        if q <= 0:
            return math.inf
        out += p * math.log(p / q)
    return out


def _run(search: Search, evaluator, cfg: GeneratorConfig, *, full: bool, cap: int, regs: dict) -> str:
    """One decision's search: ``cap`` playouts, or I20's KL-adaptive chunks for a full search; why
    it stopped (``records.STOP_CODES``)."""
    if not (full and cfg.kl.enabled):
        _, driver = run_search(search, evaluator, playouts=cap, batch_size=cfg.batch_size, self_play=True, **regs)
        return driver.stop or "playouts"
    before: dict | None = None
    while True:
        done = search.info().playouts
        if done >= cap:
            return "playouts"
        chunk = min(cfg.kl.interval, cap - done)
        _, driver = run_search(search, evaluator, playouts=chunk, batch_size=cfg.batch_size, self_play=True, **regs)
        if driver.stop != "playouts":
            return driver.stop or "playouts"
        now = _visit_shares(search)
        played = search.info().playouts - done
        settled = before is not None and search.info().playouts >= cfg.kl.min_playouts and played > 0
        if settled and _kl_shares(now, before) / played < cfg.kl.min_gain:
            return "kl"
        before = now


def generate_game(
    evaluator: Evaluator,
    cfg: GeneratorConfig,
    *,
    net_cfg: NetConfig,
    network: dict | None = None,
    keep_positions: bool = False,
    target_config: TargetConfig | None = None,
    on_decision: Callable[[Search, DecisionRecord, Position], None] | None = None,
) -> GeneratedGame:
    """Play one self-play game and record it (module docstring). ``network``: the record's
    network condition (``network_info``; None: from the evaluator's model). ``target_config``:
    the target builder's configuration (``train.targets``; None: its defaults at the network's
    short-term horizons) - the game's targets are built at generation, each decision's
    prior-to-target divergence recorded under it and the configuration named in the header; the
    reply prototype's targets too when ``keep_positions``. ``on_decision``: called with the live
    search, the decision's record and its input once each decision is recorded, before its child
    is played (the tests hold the record to the live search)."""
    cfg.validate()
    if target_config is None:
        target_config = TargetConfig(short_term_horizons=tuple(net_cfg.short_term_horizons))
    if policy_source(evaluator) != "normal":
        raise ValueError("an optimistic evaluator searches at match play only, never in self-play (register D9, section 8.1)")
    t0 = time.perf_counter()
    relations = relation_bits(relations_for(net_cfg))
    regs = dict(regs_per_board=net_cfg.regs_per_board, global_regs=net_cfg.global_regs)
    flat = cfg.split == "flat"
    game = Game.variant(cfg.variant) if cfg.start_fen is None else Game.from_fen(list(cfg.start_fen))
    start = game.info()
    start_boards = [fen for _, _, fen in game.snapshot().boards] if cfg.start_fen is not None else []
    decisions: list[DecisionRecord] = []
    positions: list[Position] = []
    actions: list[list[Move]] = []
    shared: Search | None = None
    end, winner = "", None
    # The searches' parameters (each decision's seed and root temperature are its own, recorded per decision).
    searches = {"full": params_dict(search_parameters(game, cfg, relations, seed=0, noise=cfg.root_noise))}
    if cfg.pcr.enabled:
        searches["cheap"] = params_dict(search_parameters(game, cfg, relations, seed=0, noise=cfg.pcr.cheap_root_noise))
    may_resign = resign_enabled(cfg)
    low = [0, 0]  # I18: each player's turn starts in a row below the threshold
    events: list[dict] = []
    resigned = False
    try:
        while True:
            verdict = turn_start_verdict(game, cfg)
            if verdict is not None:
                end = verdict
                if verdict == "checkmate":
                    winner = 1 - game.info().to_move
                break
            if len(actions) >= cfg.max_actions:
                end = "action_limit"
                break
            action: list[Move] = []
            a = len(actions)
            while True:
                d = len(decisions)
                ginfo = game.info()
                full = is_full_search(cfg, d)
                cap = cfg.playouts if full else cfg.pcr.cheap_cap(cfg.playouts)
                noise = cfg.root_noise if full else cfg.pcr.cheap_root_noise
                seed = decision_seed(cfg.seed, d, SALT_SEARCH)
                params = search_parameters(game, cfg, relations, seed=seed, noise=noise, root_temperature=root_temperature(cfg, a))
                search = Search(game, params=params, share=shared)
                if shared is not None:
                    shared.close()  # the cache lives on in the new search
                shared = search
                stop = _run(search, evaluator, cfg, full=full, cap=cap, regs=regs)
                position = root_position(game, params, net_cfg, cfg.window_k)
                raw: Evaluation = evaluator([position])[0]
                t = move_temperature(cfg, a)
                resigning = False
                if ginfo.moves_made == 0 and cfg.resign.threshold is not None:
                    value = search.info().root_q  # the search's value, the mover's view
                    mover = position.acting_player
                    low[mover] = low[mover] + 1 if value < -cfg.resign.threshold else 0
                    if low[mover] >= cfg.resign.consecutive:
                        if may_resign:
                            resigning = True
                        else:
                            events.append({"decision": d, "player": mover, "value": value, "n_boards": int(position.state.n_boards)})

                def rerun(search=search, cap=cap) -> None:
                    run_search(search, evaluator, playouts=cap, batch_size=cfg.batch_size, self_play=True, **regs)

                if resigning:
                    pick, choice, illegal = -1, "resigned", 0
                else:
                    pick, choice, illegal = _choose(search, t, decision_random(cfg.seed, d, SALT_CHOICE), rerun)
                info = search.info()
                children = search.children()
                best = search.choose_by_visits(0.0, 0.5)
                index = candidate_index(position)
                playable = playable_boards(game, position)
                decisions.append(
                    DecisionRecord(
                        decision=d,
                        action=a,
                        move_in_action=len(action),
                        acting_player=position.acting_player,
                        moves_made=ginfo.moves_made,
                        n_boards=int(position.state.n_boards),
                        present=int(position.present),
                        submit_possible=bool(position.submit_possible),
                        submit_legal=bool(position.submit_legal),
                        full_search=full,
                        playout_cap=cap,
                        playouts=int(info.playouts),
                        stop=stop,
                        move_temperature=t,
                        root_policy_temperature=float(params.root_policy_temperature),
                        root_noise=bool(params.root_noise),
                        search_seed=int(params.seed),
                        root_key=info.root_key,
                        epoch=info.epoch,
                        raw_error=raw.error,
                        raw_moves_left=raw.moves_left,
                        raw_submit_logit=raw.submit_logit,
                        root_evaluated=info.root_evaluated,
                        root_visits=info.root_visits,
                        root_q=info.root_q,
                        root_d=info.root_d,
                        root_cpuct=info.root_cpuct,
                        root_best_since=info.root_best_since,
                        best_child=best,
                        played_child=pick,
                        choice=choice,
                        verified_illegal=illegal,
                        raw_wdl_logits=raw.wdl_logits,
                        raw_value_logits=raw.value_logits,
                        root_wdl=info.root_wdl,
                        boards=np.stack([position.state.timeline, position.state.half_turn], axis=1),
                        candidates=position.moves,
                        raw_logits=raw.policy_logits,
                        children=children_record(children, index, flat=flat),
                        flat=flat,
                        playable=playable,
                        raw_leave_logits=raw.board_logits,
                        turn_moves=turn_moves(children) if flat else None,
                    )
                )
                if keep_positions:
                    positions.append(position)
                if on_decision is not None:
                    on_decision(search, decisions[-1], position)
                if resigning:
                    resigned = True
                    end, winner = "resign", 1 - position.acting_player
                    break
                child = children[pick]
                if flat:  # D2's flat form: the whole turn, then its submission
                    for m in child.moves:
                        game.apply(m)
                        action.append(m)
                    game.submit()
                    break
                if child.submit:
                    game.submit()
                    break
                game.apply(child.moves[0])
                action.append(child.moves[0])
            if resigned:
                break
            actions.append(action)
        snap = game.snapshot()
        final = game.expand(square_features="none")
        active = sorted({int(p) for p, live in zip(final.timeline.tolist(), final.active.tolist(), strict=True) if live})
        decided = "1-0" if winner == 0 else "0-1"
        result = {"checkmate": decided, "resign": decided, "stalemate": "1/2-1/2"}.get(end, "*")
        termination = end + (" (self-play's convention, I11: scored as a draw; not a game rule)" if end == "turn_cap" else "")
        pgn = write_record(
            cfg.variant if cfg.start_fen is None else "custom",
            actions,
            even=start.even,
            first_turn=start.present // 2,
            result=result,
            termination=termination,
            extra=[("White", "t5d selfplay"), ("Black", "t5d selfplay"), ("Engine", f"t5d training generator, seed {cfg.seed}")],
            first_player=start.to_move,
            size=(start.width, start.height),
            boards=start_boards,
        )
        header = {
            "record_version": RECORD_VERSION,
            "engine_version": game.lib.version,
            "generator": "t5d.train.selfplay",
            "variant": cfg.variant if cfg.start_fen is None else "custom",
            "start_fen": None if cfg.start_fen is None else list(cfg.start_fen),
            "ruleset": start.ruleset,
            "rules": start.rules,
            "width": start.width,
            "height": start.height,
            "even": start.even,
            "first_player": start.to_move,
            "first_turn": start.present // 2,
            "network": network if network is not None else network_info(evaluator, net_cfg),
            "value_source": "wdl",
            "search": searches,
            "expand": {
                "window_k": cfg.window_k,
                "relations": list(relations_for(net_cfg)),
                "regs_per_board": net_cfg.regs_per_board,
                "global_regs": net_cfg.global_regs,
            },
            "generator_config": cfg.to_dict(),
            "seed": cfg.seed,
            "result": {
                "end": end,
                "winner": winner,
                "result": result,
                "actions": len(actions),
                "final_boards": len(snap.boards),
                "final_fingerprint": snap.fingerprint,
                "final_active_timelines": active,
            },
            "pgn": pgn,
            "resign": {"enabled": may_resign, "events": events},
            "proof_refs": [],
            "target_config": None,
        }
    finally:
        if shared is not None:
            shared.close()
        game.close()
    record = GameRecord(header=header, decisions=decisions)
    targets = build_targets(record, target_config, positions=positions if keep_positions else None)
    for d, kl in zip(decisions, targets.kl.tolist(), strict=True):
        d.prior_to_target_kl = kl
    header["target_config"] = target_config.to_dict()
    return GeneratedGame(record=record, actions=actions, positions=positions, targets=targets, seconds=time.perf_counter() - t0)


def playable_boards(game: Game, position: Position) -> np.ndarray:
    """The acting player's playable boards at the game's state (L5; the expansion's ``playable``), in
    the network input's board order - the boards D2's flat form gives a distribution."""
    x = game.expand(square_features="none")
    keys = np.stack([position.state.timeline, position.state.half_turn], axis=1)
    if not np.array_equal(np.stack([x.timeline, x.half_turn], axis=1), keys):
        raise RuntimeError("the expansion's boards are not the network input's")
    return np.asarray(x.playable, dtype=bool).copy()


def board_range(n_boards: int, ranges: tuple[int, ...]) -> str:
    """The board-count range ``n_boards`` falls in (``ranges`` its lower bounds after the first)."""
    lo = 0
    for hi in ranges:
        if n_boards < hi:
            return f"{lo}-{hi - 1}"
        lo = hi
    return f"{lo}+"


def false_resign_report(records, ranges: tuple[int, ...] = ResignConfig().ranges) -> dict[str, dict]:
    """I18's check over the games played with resignation off: per board-count range, the games in
    which a player would have resigned (its first would-resign event per game), and how many of
    them that player did not go on to lose - false resignations (AlphaGo Zero held them under 5%).
    A game that ended unscored (``unknown``, the safety stop) is left out."""
    out: dict[str, dict] = {}
    for rec in records:
        h = rec.header
        if h["resign"]["enabled"] or h["result"]["end"] in ("unknown", "action_limit"):
            continue
        winner = h["result"]["winner"]
        seen: set[int] = set()
        for e in h["resign"]["events"]:
            player = int(e["player"])
            if player in seen:
                continue
            seen.add(player)
            row = out.setdefault(board_range(int(e["n_boards"]), ranges), {"would_resign": 0, "false": 0})
            row["would_resign"] += 1
            row["false"] += int(winner != 1 - player)
    for row in out.values():
        row["rate"] = row["false"] / row["would_resign"]
    return out


def replay_positions(
    record: GameRecord,
    net_cfg: NetConfig | None = None,
    *,
    window_k: int | None = None,
    square_labels: bool = False,
    mate_nodes: int = DEFAULT_MATE_NODES,
) -> list[Position]:
    """Each decision's network input rebuilt from the record alone (I15: "later architectures
    retrained on the archive"): the start replayed through the engine, every decision's state
    expanded with the recorded window, relations and legal-moves filter under the recorded
    legality budget, its candidate moves checked against the record's, then the recorded child
    played. ``net_cfg``: the register counts and relations to build (None: the record's) - a
    trainer passes its own network's; ``window_k``: the history window (None: the record's).
    ``square_labels``: each position also carries its per-square labels' targets
    (``Position.square_labels``; D4), re-derived from the replayed state by the engine's
    generator (``Game.square_labels``, each mate search under ``mate_nodes``) - never stored in
    the record.
    Raises ``ValueError`` where the record and the engine disagree, the ruleset and the rules
    encoding included (the engine version is not compared: a rebuild that differs shows in the
    candidate moves and the final state, both checked - issue 234(f))."""
    h = record.header
    exp = h["expand"]
    if window_k is None:
        window_k = exp["window_k"]
    regs = dict(regs_per_board=exp["regs_per_board"], global_regs=exp["global_regs"])
    relations = list(exp["relations"])
    if net_cfg is not None:
        regs = dict(regs_per_board=net_cfg.regs_per_board, global_regs=net_cfg.global_regs)
        relations = list(relations_for(net_cfg))
    budget = {}
    for kind in ("full", "cheap"):
        if kind in h["search"]:
            s = h["search"][kind]
            budget[kind] = (bool(s["legal_moves_only"]), int(s["legality_checks"]), int(s["legality_nodes"]))
    game = Game.variant(h["variant"]) if h["start_fen"] is None else Game.from_fen(list(h["start_fen"]))
    out: list[Position] = []
    try:
        info = game.info()
        for key in ("ruleset", "rules"):
            if key in h and int(h[key]) != int(getattr(info, key)):
                raise ValueError(f"the record's {key} {h[key]} is not the replayed game's {getattr(info, key)}")
        for d in record.decisions:
            legal, checks, nodes = budget["full" if d.full_search or "cheap" not in budget else "cheap"]
            pos = game.position(
                **regs, relations=relations, window_k=window_k, legal_moves=legal, legality_checks=checks, legality_nodes=nodes
            )
            if pos.moves.tobytes() != d.candidates.tobytes():
                raise ValueError(f"decision {d.decision}: the rebuilt input's candidate moves are not the record's")
            if square_labels:
                pos = replace(pos, square_labels=game.square_labels(mate_nodes=mate_nodes).targets())
            out.append(pos)
            if d.played_child < 0:
                break
            if d.flat:  # D2's flat form: the turn's moves, then its submission
                for m in d.turn(d.played_child):
                    game.apply(Move.from_record(m))
                game.submit()
                continue
            k = int(d.children.candidate[d.played_child])
            if k < 0:
                game.submit()
            else:
                game.apply(Move.from_record(d.candidates[k]))
        if game.snapshot().fingerprint != h["result"]["final_fingerprint"]:
            raise ValueError("the replayed game does not end in the record's final state")
    finally:
        game.close()
    return out


__all__ = [
    "SPLITS",
    "KLAdaptiveConfig",
    "PCRConfig",
    "ResignConfig",
    "TemperatureSchedule",
    "GeneratedGame",
    "GeneratorConfig",
    "candidate_index",
    "children_record",
    "decision_random",
    "board_range",
    "decision_seed",
    "false_resign_report",
    "generate_game",
    "is_full_search",
    "mix",
    "move_temperature",
    "network_info",
    "playable_boards",
    "replay_positions",
    "resign_enabled",
    "root_temperature",
    "root_position",
    "search_parameters",
    "turn_moves",
]
