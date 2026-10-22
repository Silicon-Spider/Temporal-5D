"""The training self-play generator: games through the engine's search, written as I15's records.

Register G9: PyTorch hosts the network and drives the engine's search through the C API, so the
generator is Python. It plays as ``engine.netplay``'s self-play does - a turn start judged first
(E4's exhaustive detector, then I11's cap), a fresh search per decision sharing the previous
one's evaluation cache (C14), root noise on, a chosen child made known legal before it is played
- and writes, per decision, the record ``train.records`` defines (register I15, spec section
12.8). ``netplay.play_game`` is unchanged: it plays and writes a 5dpgn, and writes no record.

* **D2's arms**: the board-move split (``"ply"``) and the tree form (``"tree"``). The whole-turn
  arm is refused: its root children are actions while the policy scores single moves, so its
  records would carry no defined policy target (the arm is superseded by the flat form at
  CHECKPOINT 6 - spec issue 234).
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

Nothing here trains (the untrained build's rule): the generator only plays and records.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from ..engine.capi import MOVE_DTYPE, T5D_LEGALITY_ILLEGAL, T5D_LEGALITY_LEGAL, T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_TREE
from ..engine.convert import Position, relations_for
from ..engine.evaluator import Evaluation, Evaluator, policy_source, run_search
from ..engine.expansion import Move
from ..engine.game import Game, relation_bits
from ..engine.netplay import turn_start_verdict, write_record
from ..engine.search import RootChild, Search, search_params
from ..net.config import NetConfig
from .records import RECORD_VERSION, Children, DecisionRecord, GameRecord, network_id, params_dict
from .targets import GameTargets, TargetConfig, build_targets

#: D2's arms the generator plays (module docstring).
SPLITS = {"ply": T5D_SPLIT_BOARD_MOVE, "tree": T5D_SPLIT_TREE}

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
class GeneratorConfig:
    """One game's settings. Every search constant not named here is the engine's default
    (``t5d_search_params_default``) unless ``search_overrides`` names it; the proof channel is
    off (H41)."""

    variant: str = "Standard"
    start_fen: tuple[str, ...] | None = None  # a custom start (5DFEN boards) instead of ``variant``'s
    split: str = "ply"  # D2: "ply" or "tree" (module docstring)
    playouts: int = 16  # the full search's cap
    batch_size: int = 8
    cap_turns: int = 6  # I11 (0: none)
    cap_boards: int = 80
    max_actions: int = 200  # a safety stop, not a rule (scored like I11's cap: unscored, "action_limit")
    seed: int = 1
    window_k: int = 8
    root_noise: bool = True
    #: The move temperature for the first ``temperature_turns`` turns (one player's action each,
    #: issue 305), then 0 (the most visited) - netplay's step schedule.
    temperature: float = 1.0
    temperature_turns: int = 16
    #: Further ``t5d_search_params`` fields by name, for every search (e.g. the legality budget).
    search_overrides: tuple[tuple[str, object], ...] = ()

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)} (the whole-turn arm writes no policy target - spec issue 234)")
        if self.playouts < 1 or self.batch_size < 1 or self.max_actions < 1:
            raise ValueError("playouts, batch_size and max_actions must be positive")

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
    #: The targets written at generation (``generate_game(target_config=...)``): the live side of
    #: section 15.2's I15 row; None when none were asked for.
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
        legal_moves=bool(params.legal_moves_only) and params.split != 1,
        legality_checks=int(params.legality_checks),
        legality_nodes=int(params.legality_nodes),
    )


def _move_key(m: Move) -> tuple[int, ...]:
    return tuple(int(getattr(m, name)) for name in MOVE_DTYPE.names)


def candidate_index(position: Position) -> dict[tuple[int, ...], int]:
    """Each candidate move's index by its fields (``MOVE_DTYPE``'s)."""
    return {tuple(int(x) for x in row.tolist()): i for i, row in enumerate(position.moves)}


def children_record(children: list[RootChild], index: dict[tuple[int, ...], int]) -> Children:
    """The root children as I15's columns; each child's candidate index (-1: submit)."""
    cand = []
    for c in children:
        if c.submit:
            cand.append(-1)
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
    )


def network_info(evaluator, net_cfg: NetConfig, *, seed: int | None = None, swa_snapshots: int = 0) -> dict:
    """The record's ``network`` condition for an evaluator with a ``model`` (its weights' hash)."""
    model = getattr(evaluator, "model", None)
    return {
        "id": None if model is None else network_id(model),
        "config": asdict(net_cfg),
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


def move_temperature(cfg: GeneratorConfig, action: int) -> float:
    """The move temperature at turn ``action`` (module docstring of ``GeneratorConfig``)."""
    return float(cfg.temperature) if action < cfg.temperature_turns else 0.0


def generate_game(
    evaluator: Evaluator,
    cfg: GeneratorConfig,
    *,
    net_cfg: NetConfig,
    network: dict | None = None,
    keep_positions: bool = False,
    target_config: TargetConfig | None = None,
) -> GeneratedGame:
    """Play one self-play game and record it (module docstring). ``network``: the record's
    network condition (``network_info``; None: from the evaluator's model). ``target_config``:
    build the game's targets at generation (``train.targets``) - each decision's prior-to-target
    divergence is then recorded, and the configuration in the header; the reply prototype's
    targets too when ``keep_positions``."""
    cfg.validate()
    if policy_source(evaluator) != "normal":
        raise ValueError("an optimistic evaluator searches at match play only, never in self-play (register D9, section 8.1)")
    t0 = time.perf_counter()
    relations = relation_bits(relations_for(net_cfg))
    regs = dict(regs_per_board=net_cfg.regs_per_board, global_regs=net_cfg.global_regs)
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
                cap = cfg.playouts
                params = search_parameters(game, cfg, relations, seed=decision_seed(cfg.seed, d, SALT_SEARCH), noise=cfg.root_noise)
                search = Search(game, params=params, share=shared)
                if shared is not None:
                    shared.close()  # the cache lives on in the new search
                shared = search
                _, driver = run_search(search, evaluator, playouts=cap, batch_size=cfg.batch_size, self_play=True, **regs)
                stop = driver.stop or "playouts"
                position = root_position(game, params, net_cfg, cfg.window_k)
                raw: Evaluation = evaluator([position])[0]
                t = move_temperature(cfg, a)

                def rerun(search=search, cap=cap) -> None:
                    run_search(search, evaluator, playouts=cap, batch_size=cfg.batch_size, self_play=True, **regs)

                pick, choice, illegal = _choose(search, t, decision_random(cfg.seed, d, SALT_CHOICE), rerun)
                info = search.info()
                children = search.children()
                best = search.choose_by_visits(0.0, 0.5)
                index = candidate_index(position)
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
                        full_search=True,
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
                        children=children_record(children, index),
                    )
                )
                if keep_positions:
                    positions.append(position)
                child = children[pick]
                if child.submit:
                    game.submit()
                    break
                game.apply(child.moves[0])
                action.append(child.moves[0])
            actions.append(action)
        snap = game.snapshot()
        final = game.expand(square_features="none")
        active = sorted({int(p) for p, live in zip(final.timeline.tolist(), final.active.tolist(), strict=True) if live})
        result = {"checkmate": "1-0" if winner == 0 else "0-1", "stalemate": "1/2-1/2"}.get(end, "*")
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
            "resign": {"enabled": False, "events": []},
            "proof_refs": [],
            "target_config": None,
        }
    finally:
        if shared is not None:
            shared.close()
        game.close()
    record = GameRecord(header=header, decisions=decisions)
    targets = None
    if target_config is not None:
        targets = build_targets(record, target_config, positions=positions if keep_positions else None)
        for d, kl in zip(decisions, targets.kl.tolist(), strict=True):
            d.prior_to_target_kl = kl
        header["target_config"] = target_config.to_dict()
    return GeneratedGame(record=record, actions=actions, positions=positions, targets=targets, seconds=time.perf_counter() - t0)


def replay_positions(record: GameRecord, net_cfg: NetConfig | None = None) -> list[Position]:
    """Each decision's network input rebuilt from the record alone (I15: "later architectures
    retrained on the archive"): the start replayed through the engine, every decision's state
    expanded with the recorded window, relations and legal-moves filter under the recorded
    legality budget, its candidate moves checked against the record's, then the recorded child
    played. ``net_cfg``: the register counts and relations to build (None: the record's).
    Raises ``ValueError`` where the record and the engine disagree."""
    h = record.header
    exp = h["expand"]
    regs = dict(regs_per_board=exp["regs_per_board"], global_regs=exp["global_regs"])
    relations = list(exp["relations"])
    if net_cfg is not None:
        regs = dict(regs_per_board=net_cfg.regs_per_board, global_regs=net_cfg.global_regs)
        relations = list(relations_for(net_cfg))
    budget = {}
    for kind in ("full", "cheap"):
        if kind in h["search"]:
            s = h["search"][kind]
            budget[kind] = (bool(s["legal_moves_only"]) and s["split"] != 1, int(s["legality_checks"]), int(s["legality_nodes"]))
    game = Game.variant(h["variant"]) if h["start_fen"] is None else Game.from_fen(list(h["start_fen"]))
    out: list[Position] = []
    try:
        for d in record.decisions:
            legal, checks, nodes = budget["full" if d.full_search or "cheap" not in budget else "cheap"]
            pos = game.position(
                **regs, relations=relations, window_k=exp["window_k"], legal_moves=legal, legality_checks=checks, legality_nodes=nodes
            )
            if pos.moves.tobytes() != d.candidates.tobytes():
                raise ValueError(f"decision {d.decision}: the rebuilt input's candidate moves are not the record's")
            out.append(pos)
            if d.played_child < 0:
                break
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
    "GeneratedGame",
    "GeneratorConfig",
    "candidate_index",
    "children_record",
    "decision_random",
    "decision_seed",
    "generate_game",
    "mix",
    "move_temperature",
    "network_info",
    "replay_positions",
    "root_position",
    "search_parameters",
]
