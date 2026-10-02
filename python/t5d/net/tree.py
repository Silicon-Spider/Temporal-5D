"""A search-tree-like fixture for the cached path (stage N3): positions that grow as a
search's do, and nodes that carry epochs.

The real search is not built yet (spec section 15.1, Phase 2), so the equivalence suite
(section 15.2) and the cost figures (section 15.3) drive the cached evaluator
(``cache.py``) through this stand-in. Like ``synthetic.py`` it is structurally plausible,
not legal: no rules engine stands behind the moves.

* ``Multiverse`` - an immutable multiverse with lineage: every board knows the board it
  was made from (its timeline's previous board, or its branch origin) and when it was
  created. Moves make new multiverses and never change an old one (A4):

  - ``physical(p)``: the side to move on timeline ``p``'s head moves a piece on it - one
    new board, the old head superseded;
  - ``jump(p, (q, t))``: a piece leaves timeline ``p``'s head (a new board there) and
    arrives on board ``(q, t)``: if that board is a head, its timeline gets the next
    board (a non-branching jump); otherwise a new timeline branches from it, labelled by
    the mover (white +n, black -n; ``-0``/``+0`` pairs for even starts), its first board
    one half-turn after its origin.

  The arrival board records both sides of the travel: its parent (arrival-side lineage:
  the board it was made from) and its source (the board the piece departed from - the
  second parent of ``causal_past="two_parent"``).

  A move's content is drawn from a generator seeded by the move itself (the timelines,
  half-turns and destination), never by the path: two move orders that make the same
  moves reach the same boards. Their creation orders differ - ``PositionState.creation``
  is a property of the path (spec issue 5).
* ``SearchTree`` - nodes over multiverses, each carrying its epoch (section 7.9: "each
  node carries its epoch"): the root's from ``CachedEvaluator.root_epoch`` (a root refresh
  or the causal epoch), a child's its parent's. ``visit`` counts visits and applies C3a
  rung 1 when ``CacheConfig.node_refresh`` is on: a node reached by a travel move that
  created a timeline is refreshed once its visits reach ``refresh_visit_threshold``
  (V = 50 by default) - the refresh forks a new epoch for the node and every descendant
  that still has the node's old epoch; siblings and their subtrees keep it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from .cache import CachedEvaluator, Epoch
from .model import NetOutput
from .state import COLOUR_BLACK, COLOUR_NONE, COLOUR_WHITE, PositionState
from .synthetic import _candidate_moves, reach_graph

#: A move: ("physical", p) or ("jump", p, (q, t)).
Move = tuple


@dataclass(frozen=True, eq=False)
class FixtureBoard:
    p: int
    h: int
    piece: np.ndarray
    colour: np.ndarray
    moved: np.ndarray
    parent: tuple[int, int] | None  # (p, h) of the board it was made from
    creation: int
    source: tuple[int, int] | None = None  # (p, h) a travelling piece departed from, for its arrival board


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF] + [int(x) + 2**20 for x in parts])


class Multiverse:
    """An immutable multiverse with lineage (see the module docstring)."""

    def __init__(
        self,
        width: int,
        height: int,
        boards: dict[tuple[int, int], FixtureBoard],
        creators: dict[int, int],
        even_start: bool,
        next_creation: int,
        seed: int = 0,
        regs_per_board: int = 2,
        global_regs: int = 16,
    ) -> None:
        self.width, self.height = int(width), int(height)
        self.boards = dict(boards)
        self.creators = dict(creators)  # timeline -> COLOUR_NONE (initial), COLOUR_WHITE or COLOUR_BLACK
        self.even_start = bool(even_start)
        self.next_creation = int(next_creation)
        self.seed = int(seed)
        self.regs_per_board, self.global_regs = int(regs_per_board), int(global_regs)
        self._state: PositionState | None = None

    # -- construction --

    @classmethod
    def from_state(cls, state: PositionState, seed: int = 0, even_start: bool | None = None) -> Multiverse:
        """A fixture multiverse from a position with lineage (``synthetic_state`` gives one).
        Timelines are attributed to creators by their labels (positive white, negative black;
        ``0`` - and ``-1`` too for an even start - initial)."""
        if state.creation is None:
            raise ValueError("the fixture needs the boards' creation order")
        if even_start is None:
            first_m1 = state.half_turn[state.timeline == -1]
            even_start = bool(first_m1.size) and int(first_m1.min()) == int(state.half_turn.min())
        origins = (-1, 0) if even_start else (0,)
        boards = {}
        for i in range(state.n_boards):
            a = int(state.parent[i])
            parent = None if a < 0 else (int(state.timeline[a]), int(state.half_turn[a]))
            d = int(state.source[i])
            source = None if d < 0 else (int(state.timeline[d]), int(state.half_turn[d]))
            key = (int(state.timeline[i]), int(state.half_turn[i]))
            boards[key] = FixtureBoard(
                *key, _frozen(state.piece[i]), _frozen(state.colour[i]), _frozen(state.moved[i]), parent, int(state.creation[i]), source
            )
        creators = {}
        for p in np.unique(state.timeline).tolist():
            creators[p] = COLOUR_NONE if p in origins else (COLOUR_WHITE if p > 0 else COLOUR_BLACK)
        return cls(
            state.width,
            state.height,
            boards,
            creators,
            even_start,
            int(state.creation.max()) + 1,
            seed,
            state.regs_per_board,
            state.global_regs,
        )

    # -- queries --

    def heads(self) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for p, h in self.boards:
            if p not in out or h > out[p][1]:
                out[p] = (p, h)
        return out

    def is_head(self, key: tuple[int, int]) -> bool:
        return self.heads().get(key[0]) == key

    def digest(self) -> str:
        """The multiverse's content (boards, lineage), creation order excluded: equal for two
        move orders that make the same moves."""
        h = hashlib.blake2b(digest_size=16)
        for key in sorted(self.boards):
            b = self.boards[key]
            h.update(np.array([*key, *(b.parent or (-(2**30), 0)), *(b.source or (-(2**30), 0))], dtype=np.int64).tobytes())
            h.update(b.piece.astype(np.int8).tobytes() + b.colour.astype(np.int8).tobytes() + b.moved.astype(np.int8).tobytes())
        return h.hexdigest()

    def _count(self, colour: int) -> int:
        return sum(1 for c in self.creators.values() if c == colour)

    def _active(self, p: int) -> bool:
        """The count rule: a player's k-th timeline is active while k <= the opponent's count + 1."""
        creator = self.creators[p]
        if creator == COLOUR_NONE:
            return True
        k = p if p > 0 else (-p - 1 if self.even_start else -p)
        other = self._count(COLOUR_BLACK if creator == COLOUR_WHITE else COLOUR_WHITE)
        return k <= other + 1

    # -- moves --

    def _with(self, new_boards: Sequence[FixtureBoard], creators: dict[int, int] | None = None) -> Multiverse:
        boards = dict(self.boards)
        for b in new_boards:
            boards[(b.p, b.h)] = b
        return Multiverse(
            self.width,
            self.height,
            boards,
            self.creators if creators is None else creators,
            self.even_start,
            self.next_creation + len(new_boards),
            self.seed,
            self.regs_per_board,
            self.global_regs,
        )

    def physical(self, p: int) -> Multiverse:
        """The side to move on timeline ``p``'s head moves one of its pieces on that board."""
        key = self.heads()[p]
        head = self.boards[key]
        mover = COLOUR_WHITE if head.h % 2 == 0 else COLOUR_BLACK
        rng = _rng(self.seed, 1, p, head.h + 1)
        piece, colour, moved = head.piece.copy(), head.colour.copy(), head.moved.copy()
        own = np.flatnonzero(colour == mover)
        targets = np.flatnonzero(colour != mover)
        if own.size and targets.size:
            src, dst = int(rng.choice(own)), int(rng.choice(targets))
            piece[dst], colour[dst], moved[dst] = piece[src], colour[src], True
            piece[src], colour[src], moved[src] = 0, COLOUR_NONE, False
        board = FixtureBoard(p, head.h + 1, _frozen(piece), _frozen(colour), _frozen(moved), key, self.next_creation)
        return self._with([board])

    def jump(self, p: int, dest: tuple[int, int]) -> Multiverse:
        """A piece leaves timeline ``p``'s head and arrives on board ``dest`` (see the module
        docstring); ``dest`` must have the mover to move and be no later than the source."""
        src_key = self.heads()[p]
        src = self.boards[src_key]
        dest = (int(dest[0]), int(dest[1]))
        if dest not in self.boards:
            raise ValueError(f"no board at {dest}")
        target = self.boards[dest]
        if target.h % 2 != src.h % 2 or target.h > src.h or dest == src_key:
            raise ValueError("a jump lands on another board of the mover's colour, no later than its source")
        mover = COLOUR_WHITE if src.h % 2 == 0 else COLOUR_BLACK
        rng = _rng(self.seed, 2, p, src.h, *dest)
        piece, colour, moved = src.piece.copy(), src.colour.copy(), src.moved.copy()
        own = np.flatnonzero(colour == mover)
        kind = 2  # a knight, if the source holds no piece of the mover's
        if own.size:
            sq = int(rng.choice(own))
            kind = int(piece[sq])
            piece[sq], colour[sq], moved[sq] = 0, COLOUR_NONE, False
        left = FixtureBoard(p, src.h + 1, _frozen(piece), _frozen(colour), _frozen(moved), src_key, self.next_creation)
        piece, colour, moved = target.piece.copy(), target.colour.copy(), target.moved.copy()
        free = np.flatnonzero(colour != mover)
        if free.size:
            sq = int(rng.choice(free))
            piece[sq], colour[sq], moved[sq] = kind, mover, True
        creators = None
        if self.is_head(dest):
            q = dest[0]
        else:
            n = self._count(mover) + 1
            q = n if mover == COLOUR_WHITE else (-n - 1 if self.even_start else -n)
            creators = dict(self.creators)
            creators[q] = mover
        arrived = FixtureBoard(q, dest[1] + 1, _frozen(piece), _frozen(colour), _frozen(moved), dest, self.next_creation + 1, src_key)
        return self._with([left, arrived], creators)

    def apply(self, move: Move) -> Multiverse:
        if move[0] == "physical":
            return self.physical(int(move[1]))
        if move[0] == "jump":
            return self.jump(int(move[1]), tuple(move[2]))
        raise ValueError(f"unknown move {move!r}")

    def branches(self, move: Move) -> bool:
        """Whether ``move`` creates a timeline (rung 1's trigger: travel drift is a step)."""
        return move[0] == "jump" and not self.is_head(tuple(move[2]))

    # -- the network's input --

    def state(self) -> PositionState:
        """The position (memoised): boards in ``(p, h)`` order, lineage, flags by the count
        rule, a geometric move graph and candidate moves seeded by the content."""
        if self._state is not None:
            return self._state
        keys = sorted(self.boards)
        index = {k: i for i, k in enumerate(keys)}
        heads = set(self.heads().values())
        p = np.array([k[0] for k in keys], dtype=np.int64)
        h = np.array([k[1] for k in keys], dtype=np.int64)
        kwargs = dict(
            width=self.width,
            height=self.height,
            timeline=p,
            half_turn=h,
            is_head=np.array([k in heads for k in keys]),
            active=np.array([self._active(k[0]) for k in keys]),
            side_to_move=h & 1,
            piece=np.stack([self.boards[k].piece for k in keys]),
            colour=np.stack([self.boards[k].colour for k in keys]),
            moved=np.stack([self.boards[k].moved for k in keys]),
            regs_per_board=self.regs_per_board,
            global_regs=self.global_regs,
            parent=np.array([index[self.boards[k].parent] if self.boards[k].parent is not None else -1 for k in keys], dtype=np.int64),
            creation=np.array([self.boards[k].creation for k in keys], dtype=np.int64),
            source=np.array([index[self.boards[k].source] if self.boards[k].source is not None else -1 for k in keys], dtype=np.int64),
        )
        probe = PositionState(**kwargs)
        graph = reach_graph(probe)
        rng = np.random.default_rng(int(self.digest()[:15], 16))
        player = probe.player_to_move
        moves = _candidate_moves(rng, probe, probe.layout, graph, player, None)
        opp = _candidate_moves(rng, probe, probe.layout, graph, 1 - player, None)
        self._state = PositionState(**kwargs, edges=graph, moves=moves, opp_moves=opp)
        return self._state


# --- the search tree -----------------------------------------------------------------------------


@dataclass(eq=False)
class Node:
    universe: Multiverse
    parent: Node | None = None
    move: Move | None = None
    epoch: Epoch | None = None
    children: dict = field(default_factory=dict)
    visits: int = 0
    branched: bool = False  # reached by a move that created a timeline
    refreshed: bool = False

    @property
    def state(self) -> PositionState:
        return self.universe.state()

    def descendants(self) -> Iterator[Node]:
        for child in self.children.values():
            yield child
            yield from child.descendants()


class SearchTree:
    """Nodes over multiverses, with epochs and C3a rung 1 (see the module docstring)."""

    def __init__(self, evaluator: CachedEvaluator, root: Multiverse) -> None:
        self.evaluator = evaluator
        self.root = Node(root)
        self.root.epoch = evaluator.root_epoch(root.state())
        self.refreshes = 0

    def child(self, node: Node, move: Move) -> Node:
        """The child of ``node`` by ``move`` (created once), in its parent's epoch."""
        key = (move[0], *move[1:2], *(tuple(move[2]) if len(move) > 2 else ()))
        if key not in node.children:
            node.children[key] = Node(node.universe.apply(move), node, move, node.epoch, branched=node.universe.branches(move))
        return node.children[key]

    def path(self, moves: Sequence[Move], start: Node | None = None) -> Node:
        node = self.root if start is None else start
        for move in moves:
            node = self.child(node, move)
        return node

    def evaluate(self, node: Node) -> NetOutput:
        return self.evaluator.evaluate(node.state, node.epoch)

    def visit(self, node: Node) -> bool:
        """Count a visit; C3a rung 1: refresh a node reached by a branching travel move once
        its visits reach V. Returns whether it refreshed."""
        node.visits += 1
        cfg = self.evaluator.config
        if cfg.node_refresh and node.branched and not node.refreshed and node.visits >= cfg.refresh_visit_threshold:
            self.refresh(node)
            return True
        return False

    def refresh(self, node: Node) -> Epoch:
        """Fork a new epoch at ``node`` (copy-on-write): the node and every descendant still
        in its old epoch move to the new one; everything else keeps its epoch."""
        old = node.epoch
        new = self.evaluator.refresh(node.state, old)
        node.epoch = new
        for d in node.descendants():
            if d.epoch is old:
                d.epoch = new
        node.refreshed = True
        self.refreshes += 1
        return new

    def nodes(self) -> Iterator[Node]:
        yield self.root
        yield from self.root.descendants()
