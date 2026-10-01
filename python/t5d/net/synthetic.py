"""Seeded synthetic multiverse states for tests and cost benchmarks.

The states are structurally plausible, not legal: they follow 5D chess's lattice
structure but no rules engine stands behind their contents. Growth mirrors the game's
own: every step adds exactly one board (section 8.6, "every board-move adds a board"),
either by extending a timeline's head or by travelling to an existing board, which
branches a new timeline whose first board sits one half-turn after the destination.

What the generator guarantees, so that tests can rely on it:

* exactly ``n_boards`` boards; ``n_timelines`` timelines (default: grows with N);
* timelines branch from earlier boards, labelled by their creator (white-created
  timelines take the next positive label, black-created the next negative; an
  even-start variant uses the ``-0``/``+0`` pair, see ``state.timeline_index``);
* heads at varying times; active/inactive flags by the usual count rule (a player's
  k-th timeline is active while k <= the opponent's count + 1);
* random but plausible piece placements, carried forward board to board with one
  piece move per new board, so neighbouring boards share most of their content;
  some boards are exact duplicates of their predecessor (``duplicate_prob``);
* deep travel: some branches start from the earliest boards (``deep_travel_prob``);
* a move graph and typed relations from simplified 4D movement geometry (``reach_graph``):
  every piece on every board reaches along its piece's 4D directions - sliders until
  blocked, within ``max_reach`` steps - across boards along the time and timeline axes
  as well as within its board, so the graph has cross-board edges wherever the lattice
  has boards; edges are square-token pairs only;
* candidate moves (for both players) drawn from that move graph: the player's own
  pieces on its playable boards (heads of its colour), in random order, promotions
  included where a pawn reaches its last rank - so every candidate move is a move-graph
  edge, as a legal move is.

Nothing here comes from a rules engine; the contents are for exercising the network's
shapes, invariances and costs only. The geometry is plausible, not the rules: no check,
no present-line obligations, pieces on superseded boards reach as if they could move,
and pawns are simplified.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .state import (
    COLOUR_BLACK,
    COLOUR_NONE,
    COLOUR_WHITE,
    NO_PROMOTION,
    PIECE_IDS,
    RELATION_IDS,
    PositionState,
    TokenLayout,
)

_PAWN = PIECE_IDS["P"]
_KING = PIECE_IDS["K"]
_COMMON = [PIECE_IDS[c] for c in "PPPPNBRQ"]  # weighted towards pawns
_FAIRY = [PIECE_IDS[c] for c in "UDSWCY"]
_PROMOTE_TO = [PIECE_IDS[c] for c in "NBRQ"]


@dataclass
class _Board:
    p: int
    h: int
    piece: np.ndarray
    colour: np.ndarray
    moved: np.ndarray


def default_timelines(n_boards: int) -> int:
    """Timelines for a synthetic state of ``n_boards`` boards: ~1 + 0.9 sqrt(N).

    About 15 timelines at 240 boards, in line with section 1's "5-15 timelines x 20-40
    half-turns" for mid-game states of 100-600 boards.
    """
    return max(1, min(n_boards, int(round(1 + 0.9 * math.sqrt(n_boards)))))


def _initial_board(rng: np.random.Generator, width: int, height: int, fairy_prob: float):
    wh = width * height
    piece = np.zeros(wh, dtype=np.int64)
    colour = np.zeros(wh, dtype=np.int64)
    moved = np.zeros(wh, dtype=bool)
    squares = rng.permutation(wh)
    # One king each, then a random scatter of other pieces.
    for c, sq in ((COLOUR_WHITE, squares[0]), (COLOUR_BLACK, squares[1])):
        piece[sq], colour[sq] = _KING, c
    density = rng.uniform(0.2, 0.45)
    for sq in squares[2:]:
        if rng.random() >= density:
            continue
        kind = int(rng.choice(_FAIRY)) if rng.random() < fairy_prob else int(rng.choice(_COMMON))
        y = sq // width
        if kind == _PAWN and (y == 0 or y == height - 1):
            kind = PIECE_IDS["N"]
        piece[sq] = kind
        colour[sq] = COLOUR_WHITE if rng.random() < 0.5 else COLOUR_BLACK
        moved[sq] = bool(kind == _PAWN and rng.random() < 0.3)
    return piece, colour, moved


def _apply_random_move(rng: np.random.Generator, board: _Board, mover: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    piece, colour, moved = board.piece.copy(), board.colour.copy(), board.moved.copy()
    own = np.flatnonzero(colour == mover)
    targets = np.flatnonzero(colour != mover)
    if own.size == 0 or targets.size == 0:
        return piece, colour, moved
    src = int(rng.choice(own))
    dst = int(rng.choice(targets))
    piece[dst], colour[dst], moved[dst] = piece[src], colour[src], True
    piece[src], colour[src], moved[src] = 0, COLOUR_NONE, False
    return piece, colour, moved


def _arrive(rng: np.random.Generator, board: _Board, mover: int, fairy_prob: float):
    """The destination board's content plus one piece arriving by time travel."""
    piece, colour, moved = board.piece.copy(), board.colour.copy(), board.moved.copy()
    targets = np.flatnonzero(colour != mover)
    if targets.size:
        dst = int(rng.choice(targets))
        kind = int(rng.choice(_FAIRY)) if rng.random() < fairy_prob else int(rng.choice([PIECE_IDS[c] for c in "NBRQK"]))
        piece[dst], colour[dst], moved[dst] = kind, mover, True
    return piece, colour, moved


def synthetic_state(
    seed: int,
    n_boards: int,
    *,
    width: int = 8,
    height: int = 8,
    n_timelines: int | None = None,
    deep_travel_prob: float = 0.25,
    duplicate_prob: float = 0.1,
    fairy_prob: float = 0.05,
    even_start: bool = False,
    max_reach: int = 8,
    relations: bool = True,
    n_moves: int | None = None,
    n_opp_moves: int | None = None,
    regs_per_board: int = 2,
    global_regs: int = 16,
) -> PositionState:
    """A seeded synthetic position with exactly ``n_boards`` boards (see module docstring)."""
    if n_boards < 1:
        raise ValueError("n_boards must be at least 1")
    rng = np.random.default_rng(seed)
    if even_start and n_boards < 2:
        raise ValueError("an even-start state has at least two boards")
    k_target = default_timelines(n_boards) if n_timelines is None else int(n_timelines)
    k_start = 2 if even_start else 1
    k_target = max(k_start, min(k_target, n_boards))

    boards: list[_Board] = []
    timelines: dict[int, list[int]] = {}  # p -> board indices in increasing h

    def add(p: int, h: int, content) -> None:
        piece, colour, moved = content
        boards.append(_Board(p, h, piece, colour, moved))
        timelines.setdefault(p, []).append(len(boards) - 1)

    first = _initial_board(rng, width, height, fairy_prob)
    origins = (-1, 0) if even_start else (0,)
    for p in origins:
        add(p, 0, tuple(a.copy() for a in first))

    # Which growth steps branch: k_target - k_start of the remaining steps, spread at random.
    steps = n_boards - len(boards)
    n_branches = k_target - len(origins)
    branch_steps = set(rng.choice(steps, size=n_branches, replace=False).tolist()) if n_branches > 0 else set()
    next_white = 1  # label magnitude of white's next timeline
    next_black = 1

    for step in range(steps):
        if step in branch_steps:
            # Travel to an existing board: deep (earliest boards) or recent (near a head).
            # A branch starts from a superseded board where one exists (travelling onto a
            # head continues that timeline instead of branching).
            hs = np.array([b.h for b in boards])
            heads = {timelines[p][-1] for p in timelines}
            past = np.array([i not in heads for i in range(len(boards))])
            if not past.any():
                past[:] = True
            if rng.random() < deep_travel_prob:
                pool = np.flatnonzero(past & (hs <= hs[past].min() + 1))
            else:
                pool = np.flatnonzero(past & (hs >= hs[past].max() - 4))
            dest = boards[int(rng.choice(pool))]
            mover = COLOUR_WHITE if dest.h % 2 == 0 else COLOUR_BLACK
            if mover == COLOUR_WHITE:
                p_new = next_white  # "+n" -> n in both conventions
                next_white += 1
            else:
                p_new = -next_black - 1 if even_start else -next_black  # even start: "-n" -> -n-1
                next_black += 1
            add(p_new, dest.h + 1, _arrive(rng, dest, mover, fairy_prob))
        else:
            # Extend a head, preferring timelines that lag behind (heads at varying times).
            ps = list(timelines)
            head_h = np.array([boards[timelines[p][-1]].h for p in ps], dtype=np.float64)
            weights = np.exp(-(head_h - head_h.min()) / 2.0)
            p = ps[int(rng.choice(len(ps), p=weights / weights.sum()))]
            head = boards[timelines[p][-1]]
            mover = COLOUR_WHITE if head.h % 2 == 0 else COLOUR_BLACK
            if rng.random() < duplicate_prob:
                content = (head.piece.copy(), head.colour.copy(), head.moved.copy())
            else:
                content = _apply_random_move(rng, head, mover)
            add(p, head.h + 1, content)

    # Active timelines: a player's k-th timeline is active while k <= opponent's count + 1.
    n_white = next_white - 1
    n_black = next_black - 1

    def is_active(p: int) -> bool:
        if p in origins:
            return True
        if p > 0:
            return p <= n_black + 1
        k = -p - 1 if even_start else -p
        return k <= n_white + 1

    boards.sort(key=lambda b: (b.p, b.h))
    n = len(boards)
    assert n == n_boards
    p_arr = np.array([b.p for b in boards], dtype=np.int64)
    h_arr = np.array([b.h for b in boards], dtype=np.int64)
    head = np.zeros(n, dtype=bool)
    for p in np.unique(p_arr):
        idx = np.flatnonzero(p_arr == p)
        head[idx[np.argmax(h_arr[idx])]] = True
    active = np.array([is_active(int(p)) for p in p_arr], dtype=bool)
    state_kwargs = dict(
        width=width,
        height=height,
        timeline=p_arr,
        half_turn=h_arr,
        is_head=head,
        active=active,
        side_to_move=h_arr & 1,
        piece=np.stack([b.piece for b in boards]),
        colour=np.stack([b.colour for b in boards]),
        moved=np.stack([b.moved for b in boards]),
        regs_per_board=regs_per_board,
        global_regs=global_regs,
    )
    probe = PositionState(**state_kwargs)
    layout = probe.layout
    graph = reach_graph(probe, max_reach=max_reach)
    player = probe.player_to_move
    moves = _candidate_moves(rng, probe, layout, graph, player, n_moves)
    opp = _candidate_moves(rng, probe, layout, graph, 1 - player, n_opp_moves)
    edges = graph if relations else graph[graph[:, 2] == RELATION_IDS["move"]]
    return PositionState(**state_kwargs, edges=edges, moves=moves, opp_moves=opp)


# --- the move graph: simplified 4D movement geometry -------------------------------------------

#: 4D directions (dx file, dy rank, dp timeline, dt time in full turns of two half-turns).
_DIRECTIONS: tuple[tuple[int, int, int, int], ...] = tuple(d for d in itertools.product((-1, 0, 1), repeat=4) if any(d))
#: Knight leaps: two steps along one axis and one along another.
_LEAPS: tuple[tuple[int, int, int, int], ...] = tuple(
    d for d in itertools.product((-2, -1, 0, 1, 2), repeat=4) if sorted(abs(c) for c in d) == [0, 0, 1, 2]
)
#: How many axes a direction may move along, per piece (5D chess's movement classes):
#: sliders move any number of steps, steppers one.
_SLIDERS: dict[int, tuple[int, ...]] = {
    PIECE_IDS["R"]: (1,),
    PIECE_IDS["B"]: (2,),
    PIECE_IDS["U"]: (3,),
    PIECE_IDS["D"]: (4,),
    PIECE_IDS["Q"]: (1, 2, 3, 4),
    PIECE_IDS["S"]: (1, 2),
    PIECE_IDS["Y"]: (1, 2, 3, 4),
}
_STEPPERS: dict[int, tuple[int, ...]] = {PIECE_IDS["K"]: (1, 2, 3, 4), PIECE_IDS["C"]: (1, 2, 3, 4)}
_PAWNS = (PIECE_IDS["P"], PIECE_IDS["W"])
_ROYALS = (PIECE_IDS["K"], PIECE_IDS["Y"])
_REL = RELATION_IDS


class _Lattice:
    """Every occupied square of a state and the boards around it on the 4D lattice."""

    def __init__(self, state: PositionState) -> None:
        self.width, self.height = int(state.width), int(state.height)
        self.tpb = state.layout.tokens_per_board
        self.piece, self.colour = state.piece, state.colour
        self.p0 = int(state.timeline.min())
        n_p = int(state.timeline.max()) - self.p0 + 1
        n_h = int(state.half_turn.max()) + 1
        self.grid = np.full((n_p, n_h), -1, dtype=np.int64)
        self.grid[state.timeline - self.p0, state.half_turn] = np.arange(state.n_boards)
        self.board, self.square = np.nonzero(state.piece)
        self.x, self.y = self.square % self.width, self.square // self.width
        self.p = state.timeline[self.board] - self.p0
        self.h = state.half_turn[self.board]
        self.kind = state.piece[self.board, self.square]
        self.side = state.colour[self.board, self.square]
        self.token = self.board * self.tpb + self.square
        self.edges: list[np.ndarray] = []

    def land(self, sel: np.ndarray, d: tuple[int, int, int, int], k: int = 1):
        """Where pieces ``sel`` land after ``k`` steps along ``d``: (on the lattice, board, square)."""
        nx, ny = self.x[sel] + k * d[0], self.y[sel] + k * d[1]
        np_, nh = self.p[sel] + k * d[2], self.h[sel] + 2 * k * d[3]
        ok = (nx >= 0) & (nx < self.width) & (ny >= 0) & (ny < self.height)
        ok &= (np_ >= 0) & (np_ < self.grid.shape[0]) & (nh >= 0) & (nh < self.grid.shape[1])
        board = np.full(sel.shape[0], -1, dtype=np.int64)
        board[ok] = self.grid[np_[ok], nh[ok]]
        ok &= board >= 0
        return ok, board, np.where(ok, ny * self.width + nx, 0)

    def emit(self, src: np.ndarray, dst: np.ndarray, relation: int) -> None:
        if src.size:
            self.edges.append(np.stack([src, dst, np.full(src.shape[0], relation)], axis=1))

    def arrive(self, sel: np.ndarray, board: np.ndarray, square: np.ndarray, capture_only: bool = False, quiet_only: bool = False):
        """Record what reaching ``(board, square)`` means for pieces ``sel``; returns the
        landing square's (occupied, enemy) flags."""
        occupant = self.piece[board, square]
        occupied = occupant != 0
        enemy = occupied & (self.colour[board, square] != self.side[sel])
        src, dst = self.token[sel], board * self.tpb + square
        if not capture_only:
            self.emit(src[~occupied], dst[~occupied], _REL["move"])
        if not quiet_only:
            self.emit(src[enemy], dst[enemy], _REL["move"])
            self.emit(src[enemy], dst[enemy], _REL["attack"])
            royal = enemy & np.isin(occupant, _ROYALS)
            self.emit(src[royal], dst[royal], _REL["check"])
            travel = enemy & (board != self.board[sel])
            self.emit(src[travel], dst[travel], _REL["travel_threat"])
            own = occupied & ~enemy
            self.emit(src[own], dst[own], _REL["defence"])
        return occupied, enemy

    def slide(self, sel: np.ndarray, d: tuple[int, int, int, int], max_steps: int) -> None:
        """Rays from pieces ``sel`` along ``d``: moves until the first piece, then on to the
        second piece for the x-ray and pin relations."""
        stage = np.zeros(sel.shape[0], dtype=np.int64)  # 0: clear so far; 1: behind one piece
        pinnable = np.zeros(sel.shape[0], dtype=bool)  # the piece passed is an enemy non-royal
        passed = np.zeros(sel.shape[0], dtype=np.int64)  # its token
        alive = np.arange(sel.shape[0])
        for k in range(1, max_steps + 1):
            if alive.size == 0:
                return
            ok, board, square = self.land(sel[alive], d, k)
            alive, board, square = alive[ok], board[ok], square[ok]
            occupant = self.piece[board, square]
            occupied = occupant != 0
            first = stage[alive] == 0
            # Still clear: moves, captures, defence.
            hit, enemy = self.arrive(sel[alive[first]], board[first], square[first])
            # Behind one piece: the next piece is x-rayed, and pinned-through if royal.
            second = ~first & occupied
            src = self.token[sel[alive[second]]]
            dst = board[second] * self.tpb + square[second]
            self.emit(src, dst, _REL["xray"])
            enemy2 = self.colour[board[second], square[second]] != self.side[sel[alive[second]]]
            pin = pinnable[alive[second]] & enemy2 & np.isin(occupant[second], _ROYALS)
            self.emit(src[pin], passed[alive[second]][pin], _REL["pin"])
            # The first piece hit: the ray goes on behind it, once.
            blocked = alive[first][hit]
            stage[blocked] = 1
            pinnable[blocked] = enemy[hit] & ~np.isin(occupant[first][hit], _ROYALS)
            passed[blocked] = board[first][hit] * self.tpb + square[first][hit]
            alive = alive[~second]

    def pawns(self) -> None:
        """Pawns and brawns (simplified): one step forward along the rank axis (two from
        the start rank) or the timeline axis, capturing one step diagonally forward."""
        for side, forward_y, forward_p, start in ((COLOUR_WHITE, 1, -1, 1), (COLOUR_BLACK, -1, 1, self.height - 2)):
            sel = np.flatnonzero(np.isin(self.kind, _PAWNS) & (self.side == side))
            if sel.size == 0:
                continue
            for d in ((0, forward_y, 0, 0), (0, 0, forward_p, 0)):
                ok, board, square = self.land(sel, d)
                occupied, _ = self.arrive(sel[ok], board[ok], square[ok], quiet_only=True)
                if d[1]:
                    two = sel[ok][~occupied & (self.y[sel[ok]] == start)]
                    ok2, board2, square2 = self.land(two, d, 2)
                    self.arrive(two[ok2], board2[ok2], square2[ok2], quiet_only=True)
            for d in ((1, forward_y, 0, 0), (-1, forward_y, 0, 0), (0, 0, forward_p, 1), (0, 0, forward_p, -1)):
                ok, board, square = self.land(sel, d)
                self.arrive(sel[ok], board[ok], square[ok], capture_only=True)


def reach_graph(state: PositionState, max_reach: int = 8) -> np.ndarray:
    """A plausible move graph and typed relations ``(src, dst, relation)`` for a state.

    Every piece on every board reaches along its 4D directions - rook 1 axis, bishop 2,
    unicorn 3, dragon 4, queen and royal queen any, princess 1 or 2, sliding up to
    ``max_reach`` steps until blocked; king and common king one step along any; knight two
    along one axis and one along another; pawns simplified (``_Lattice.pawns``). A step
    along the time axis moves two half-turns (same colour to move), along the timeline
    axis to the neighbouring timeline at the same half-turn; a landing exists only where
    the lattice has a board. Relations: ``move`` onto an empty or enemy square; ``attack``
    onto an enemy, ``check`` onto an enemy royal (K or royal queen), ``travel_threat`` an
    attack onto another board, ``defence`` onto an own piece; for sliders, ``xray`` to the
    second piece along a ray and ``pin`` to the first when the second is an enemy royal
    behind an enemy non-royal. Rows are distinct and sorted.
    """
    lat = _Lattice(state)
    for d in _DIRECTIONS:
        n_axes = sum(1 for c in d if c)
        sliders = np.flatnonzero(np.isin(lat.kind, [t for t, axes in _SLIDERS.items() if n_axes in axes]))
        if sliders.size:
            lat.slide(sliders, d, max_reach)
        steppers = np.flatnonzero(np.isin(lat.kind, [t for t, axes in _STEPPERS.items() if n_axes in axes]))
        if steppers.size:
            ok, board, square = lat.land(steppers, d)
            lat.arrive(steppers[ok], board[ok], square[ok])
    knights = np.flatnonzero(lat.kind == PIECE_IDS["N"])
    if knights.size:
        for d in _LEAPS:
            ok, board, square = lat.land(knights, d)
            lat.arrive(knights[ok], board[ok], square[ok])
    lat.pawns()
    if not lat.edges:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(lat.edges).astype(np.int64), axis=0)


def _candidate_moves(
    rng: np.random.Generator, state: PositionState, layout: TokenLayout, graph: np.ndarray, player: int, count: int | None
) -> np.ndarray:
    """``count`` move-graph edges of ``player``'s own pieces on its playable boards (heads of
    its colour; every head if it has none), in random order, a pawn reaching its last rank
    once per promotion piece."""
    colour = COLOUR_WHITE if player == 0 else COLOUR_BLACK
    playable = np.flatnonzero(state.is_head & ((state.half_turn & 1) == player))
    if playable.size == 0:
        playable = np.flatnonzero(state.is_head)
    if count is None:
        count = int(min(512, 24 + 12 * playable.size))
    moves = graph[graph[:, 2] == _REL["move"], :2]
    board, square = layout.split_square_tokens(moves[:, 0])
    moves = moves[np.isin(board, playable) & (state.colour[board, square] == colour)]
    if moves.shape[0] == 0 or count <= 0:
        return np.zeros((0, 3), dtype=np.int64)
    board, square = layout.split_square_tokens(moves[:, 0])
    _, dst_square = layout.split_square_tokens(moves[:, 1])
    last_rank = layout.height - 1 if player == 0 else 0
    promotes = (state.piece[board, square] == _PAWN) & (dst_square // layout.width == last_rank)
    plain = np.concatenate([moves[~promotes], np.full((int((~promotes).sum()), 1), NO_PROMOTION)], axis=1)
    promoted = [np.concatenate([moves[promotes], np.full((int(promotes.sum()), 1), piece)], axis=1) for piece in _PROMOTE_TO]
    out = np.concatenate([plain, *promoted]).astype(np.int64)
    return out[rng.permutation(out.shape[0])[:count]]


def synthetic_states(seed: int, board_counts: list[int], **kwargs) -> list[PositionState]:
    """Several synthetic states with distinct derived seeds."""
    return [synthetic_state(seed * 1_000_003 + i, n, **kwargs) for i, n in enumerate(board_counts)]
