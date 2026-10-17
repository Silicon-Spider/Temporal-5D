"""The network's input contract: one position, its token layout, and batching.

This module is the boundary between the rules engine (which will produce positions) and
the network (which consumes them). It depends on neither: a ``PositionState`` is plain
numpy data, validated on construction.

Representation (design spec section 2; register A1-A4):

* **Tokens (A1, section 2.1).** One token per square of every board, plus
  ``regs_per_board`` registers per board (2) and ``global_regs`` global registers (16)
  (section 5.2). ``T = N * (W*H + regs_per_board) + global_regs``; at the standard
  8x8 size that is ``T = 66N + 16``.
* **Board-major order (B14, section 6.1).** For each board in storage order: its
  ``W*H`` squares, then its registers; the global registers come last. The square order
  within a board is **a1, b1, ..., h1, a2, ..., h8**: square index ``s = y*W + x`` with
  ``x`` the file (0 = a) and ``y`` the rank (0 = rank 1). So for board ``b``:
  ``square token = b*(W*H + R) + y*W + x``, ``register r = b*(W*H + R) + W*H + r`` and
  global register ``g`` is ``N*(W*H + R) + g``.
* **Per-square features (A2, section 2.2).** Piece type id (0 = empty, 1..12 for
  P N B R Q K U D S W C Y), colour (0 none, 1 white, 2 black), has-moved flag.
  Coordinates are *not* features: they are carried by the positional encoding (B4).
* **Per-board features (A3, section 2.3).** Timeline position ``p``, half-turn index
  ``h = 2*T + (1 if black to move)``, head/superseded, active/inactive timeline, side to
  move, and the continuous Delta-t-to-present ``h - h_present`` (derived here, so it
  can never disagree with ``h``).
* **Relations and moves.** Typed edges ``(src_token, dst_token, relation)`` between
  square tokens, relation ids in ``RELATION_TYPES``: ``move`` edges are the move graph
  that opens the move-graph heads' mask (B9, section 6: "which squares reach which,
  across all boards"); the other five are the typed relations of B11 (attack, defence,
  pin, x-ray, check - the relations head's mask under B10, and the typed biases).
  Travel-threat, once a sixth, is folded into attack (CHECKPOINT 6, reversal 26: a travel
  threat is a cross-board attack); the C ABI keeps its id, 6, reserved and never emitted,
  and validation refuses it here. Cross-board edges are ordinary edges. What each relation
  means is the rules engine's to define (it builds attack, defence and x-ray; pin and check
  are engine ABI round 1's, so their lists stay empty until then); the network only
  consumes the lists (``structure.py``). Candidate moves
  ``(src_square_token, dst_square_token, promotion)`` feed the policy pointer head (D1,
  section 8.1). Registers are never edge or move endpoints (section 5.3); validation
  rejects any that are.
* **Lineage (stage N3; spec section 7.3, register C17).** The block-causal scope needs a
  board's causal past, which the spec leaves undefined (build log, spec issue 5), and
  every reading of it rests on two per-board facts, carried here:

  - ``parent``: the board this board was made from - its timeline's previous board, or,
    for the first board of a timeline created by travel, the board the travelling piece
    arrived on (its *branch origin*, one half-turn earlier on another timeline; the rules
    engine's ``Timeline::origin``). -1 for the first board of an initial timeline and
    for an origin nobody recorded (a position read from 5DFEN, which holds none).
    Default: the timeline's previous board where the state holds it, -1 otherwise - the
    branch origins unknown.
  - ``creation``: the order in which the boards were created in the game, as distinct
    integers (larger = later), or None when unknown. A board is created after its
    parent. Two move orders that reach one multiverse can create its boards in
    different orders, so ``creation`` is a property of the path, not of the multiverse
    (spec issue 5, reading (iii)).

  - ``source``: for a board a travelling piece arrived on (the first board of a branch,
    or the next board of a non-branching jump's destination timeline), the board the piece
    departed from; -1 otherwise or when unknown (default: all -1). ``parent`` is the
    *arrival-side* lineage - a branch's first board descends by content from the board it
    branched from, not from the board the piece left; ``source`` carries the departure
    side, for the two-parent reading of spec issue 5 (``NetConfig.causal_past``).

  Validation checks them against the timeline structure: a timeline's later board's
  parent is its previous board; a branch origin sits one half-turn before the
  timeline's first board, on another timeline, and is superseded (a travel onto a head
  continues that timeline rather than branching); a departure board is superseded (the
  jump makes its timeline's next board) and created before the arrival.
* **The action (stage N5; register D8, spec section 8.2).** Turns have variable length
  (rulebook R8): the mover keeps moving until it submits, and once the present has shifted
  to the opponent (T5) the boards' present names the opponent while the action is still the
  mover's. Three global facts carry it: ``acting_player`` (whose action this is; default the
  present's colour, ``player_to_move`` - it differs exactly when the present has shifted,
  ``shifted``), ``submit_possible`` (the action's obligations are met: under the ply split
  submitting is an action beside the moves) and ``submit_legal`` (the submission passes C4).
  The first is a network input (``NetConfig.acting_player_input``); the other two say
  whether the submit action is a candidate and whether it is masked, as ``moves`` lists the
  candidate moves. Validation: submitting is legal only where possible, and possible only
  once the present has shifted. Values are from the acting player's perspective.

Board storage order is free: the network has no absolute positions (section 4), so a
reordering of boards permutes the per-board outputs and changes nothing else. Synthetic
states store boards sorted by ``(p, h)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

# --- vocabularies ----------------------------------------------------------------------

#: Piece letters by type id (0 = empty square). P pawn, N knight, B bishop, R rook,
#: Q queen, K king, U unicorn, D dragon, S princess, W brawn, C common king, Y royal queen.
PIECE_LETTERS: tuple[str, ...] = ("", "P", "N", "B", "R", "Q", "K", "U", "D", "S", "W", "C", "Y")
PIECE_IDS: dict[str, int] = {letter: i for i, letter in enumerate(PIECE_LETTERS) if letter}
N_PIECE_TYPES = len(PIECE_LETTERS)  # 13, empty included (section 2.2: "~13 states")

COLOUR_NONE, COLOUR_WHITE, COLOUR_BLACK = 0, 1, 2
N_COLOURS = 3

#: Edge relation ids (section 6). ``move`` is the move graph (B9's mask); the rest are
#: B11's typed relations (``config.TYPED_RELATIONS``). The ids are the C ABI's
#: (``engine/include/t5d/records.h``), which keeps id 6 - travel-threat, folded into attack at
#: CHECKPOINT 6 - reserved: the engine never emits it, and no network relation has it.
RELATION_TYPES: tuple[str, ...] = ("move", "attack", "defence", "pin", "xray", "check")
RELATION_IDS: dict[str, int] = {name: i for i, name in enumerate(RELATION_TYPES)}

#: Token kinds.
TOKEN_SQUARE, TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_PAD = 0, 1, 2, 3

#: Promotion id 0 means "no promotion"; otherwise the promoted-to piece's type id.
NO_PROMOTION = 0


def timeline_index(label: str | int, *, even_start: bool = False) -> int:
    """Map a timeline label to the integer timeline coordinate ``p``.

    Adjacent timelines must differ by exactly 1 so that the relative encoding never sees
    two distinct timelines at distance zero (section 12.6's +-0 note; register part K
    lists the convention as open). Odd-start games (one starting timeline) use the label
    itself: ``"0"``, ``"+1"``/``"1"``, ``"-1"``. Even-start variants have a ``-0``/``+0``
    pair: ``"-0" -> -1`` and ``"+0" -> 0``, so ``"-n" -> -n - 1`` and ``"+n" -> n``.

    The network is invariant to a common shift of every ``p`` (RoPE, B15's displacement
    bias and smolgen's relative cells see only differences), so only the adjacency property
    is load-bearing; which of the pair is 0 is a convention.
    """
    if isinstance(label, (int, np.integer)):
        if even_start:
            raise ValueError("even-start timelines need their signed label (e.g. '-0', '+0')")
        return int(label)
    text = label.strip()
    if not text:
        raise ValueError("empty timeline label")
    sign = 0
    if text[0] in "+-":
        sign = 1 if text[0] == "+" else -1
        text = text[1:]
    if not text.isdigit():
        raise ValueError(f"bad timeline label {label!r}")
    magnitude = int(text)
    if not even_start:
        if magnitude == 0 and sign != 0:
            raise ValueError(f"{label!r} is an even-start label; pass even_start=True")
        return -magnitude if sign < 0 else magnitude
    if sign == 0:
        raise ValueError(f"even-start timeline labels are signed ('-0', '+0', ...), got {label!r}")
    return magnitude if sign > 0 else -magnitude - 1


# --- token layout ------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenLayout:
    """Board-major token layout of one position (A1, B14)."""

    n_boards: int
    width: int
    height: int
    regs_per_board: int = 2
    global_regs: int = 16

    @property
    def squares_per_board(self) -> int:
        return self.width * self.height

    @property
    def tokens_per_board(self) -> int:
        return self.width * self.height + self.regs_per_board

    @property
    def n_body(self) -> int:
        """Tokens that belong to a board (squares and per-board registers)."""
        return self.n_boards * self.tokens_per_board

    @property
    def n_tokens(self) -> int:
        """``T = N * (W*H + R) + G``."""
        return self.n_body + self.global_regs

    def square_token(self, board: int, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"square ({x}, {y}) is off a {self.width}x{self.height} board")
        return self.square_token_from_index(board, y * self.width + x)

    def square_token_from_index(self, board: int, square: int) -> int:
        if not 0 <= board < self.n_boards:
            raise ValueError(f"board {board} out of range (N = {self.n_boards})")
        if not 0 <= square < self.squares_per_board:
            raise ValueError(f"square index {square} out of range")
        return board * self.tokens_per_board + square

    def board_register_token(self, board: int, slot: int) -> int:
        if not 0 <= slot < self.regs_per_board:
            raise ValueError(f"register slot {slot} out of range")
        return board * self.tokens_per_board + self.squares_per_board + slot

    def global_register_token(self, index: int) -> int:
        if not 0 <= index < self.global_regs:
            raise ValueError(f"global register {index} out of range")
        return self.n_body + index

    def kinds(self) -> np.ndarray:
        """Token kind of every token, shape ``(T,)``."""
        per_board = np.full(self.tokens_per_board, TOKEN_BOARD_REGISTER, dtype=np.int64)
        per_board[: self.squares_per_board] = TOKEN_SQUARE
        return np.concatenate([np.tile(per_board, self.n_boards), np.full(self.global_regs, TOKEN_GLOBAL_REGISTER, dtype=np.int64)])

    def is_square_token(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens)
        inside = (tokens >= 0) & (tokens < self.n_body)
        return inside & ((tokens % self.tokens_per_board) < self.squares_per_board)

    def split_square_tokens(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(board, square index)`` of square tokens (no check: use ``is_square_token``)."""
        tokens = np.asarray(tokens)
        return tokens // self.tokens_per_board, tokens % self.tokens_per_board


# --- one position ------------------------------------------------------------------------


def _as_int(a, shape_hint: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == bool:
        arr = arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating) and arr.size and not np.all(arr == np.round(arr)):
            raise ValueError(f"{shape_hint}: expected integers")
        arr = arr.astype(np.int64)
    return arr.astype(np.int64, copy=False)


def _triples(a, name: str) -> np.ndarray:
    arr = _as_int(np.zeros((0, 3), dtype=np.int64) if a is None else a, name)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"{name}: expected shape (n, 3) (or (n, 2) for moves without promotion)")
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.int64)], axis=1)
    return np.ascontiguousarray(arr)


@dataclass(eq=False)
class PositionState:
    """One position: every board of the multiverse, as the network sees it.

    Per-board arrays have length ``N`` (boards in storage order); per-square arrays
    have shape ``(N, W*H)`` in a1..h8 order (see the module docstring). ``edges``,
    ``moves`` and ``opp_moves`` hold token indices in this position's own layout
    (``layout``). ``opp_moves`` are the opponent's candidate moves, scored by the
    opponent-reply auxiliary head (D4, section 12.4).

    ``side_to_move`` is per board (section 2.3): the colour to move *on that board*,
    0 white / 1 black. By the definition of ``h`` it equals ``h & 1``, which validation
    checks. The position's present ``h_present`` is the minimum ``h`` over the heads of
    active timelines (section 2.3, "present"); the player to move is ``h_present & 1``.
    """

    width: int
    height: int
    timeline: np.ndarray
    half_turn: np.ndarray
    is_head: np.ndarray
    active: np.ndarray
    side_to_move: np.ndarray
    piece: np.ndarray
    colour: np.ndarray
    moved: np.ndarray
    edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    moves: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    opp_moves: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    regs_per_board: int = 2
    global_regs: int = 16
    #: Lineage (see the module docstring): each board's parent board index (-1: none or
    #: unknown; None: derived - the timeline's previous board, origins unknown), its
    #: creation order (None: unknown) and a travel arrival's departure board (None: none).
    parent: np.ndarray | None = None
    creation: np.ndarray | None = None
    source: np.ndarray | None = None
    #: The action (register D8, stage N5; see the module docstring): the player whose action
    #: is in progress until it submits (0 white, 1 black; None: derived - the present's
    #: colour, ``player_to_move``, i.e. no shift inside the action); whether submitting is
    #: possible (the action's obligations met, T5 - an action of its own under the ply
    #: split, offered beside the moves) and legal (C4: no royal of the acting player's left
    #: capturable). The value is from the acting player's perspective.
    acting_player: int | None = None
    submit_possible: bool = False
    submit_legal: bool = False

    def __post_init__(self) -> None:
        self.timeline = _as_int(self.timeline, "timeline").reshape(-1)
        self.half_turn = _as_int(self.half_turn, "half_turn").reshape(-1)
        self.is_head = np.asarray(self.is_head, dtype=bool).reshape(-1)
        self.active = np.asarray(self.active, dtype=bool).reshape(-1)
        self.side_to_move = _as_int(self.side_to_move, "side_to_move").reshape(-1)
        n = self.timeline.shape[0]
        wh = int(self.width) * int(self.height)
        self.piece = _as_int(self.piece, "piece").reshape(n, wh)
        self.colour = _as_int(self.colour, "colour").reshape(n, wh)
        self.moved = np.asarray(self.moved, dtype=bool).reshape(n, wh)
        self.edges = _triples(self.edges, "edges")
        self.moves = _triples(self.moves, "moves")
        self.opp_moves = _triples(self.opp_moves, "opp_moves")
        if self.parent is None:
            self.parent = self.previous_boards()
        else:
            self.parent = _as_int(self.parent, "parent").reshape(-1)
        if self.creation is not None:
            self.creation = _as_int(self.creation, "creation").reshape(-1)
        if self.source is None:
            self.source = np.full(n, -1, dtype=np.int64)
        else:
            self.source = _as_int(self.source, "source").reshape(-1)
        self.submit_possible = bool(self.submit_possible)
        self.submit_legal = bool(self.submit_legal)
        self.validate()
        if self.acting_player is None:
            self.acting_player = self.player_to_move
        self.acting_player = int(self.acting_player)

    def previous_boards(self) -> np.ndarray:
        """``(N,)``: each board's timeline predecessor (the board at ``(p, h - 1)``), -1 where
        the state holds none - the default ``parent``, with every branch origin unknown."""
        index = {(int(p), int(h)): i for i, (p, h) in enumerate(zip(self.timeline.tolist(), self.half_turn.tolist(), strict=True))}
        return np.array([index.get((int(p), int(h) - 1), -1) for p, h in zip(self.timeline, self.half_turn, strict=True)], dtype=np.int64)

    # -- derived quantities --

    @property
    def n_boards(self) -> int:
        return int(self.timeline.shape[0])

    @property
    def layout(self) -> TokenLayout:
        return TokenLayout(self.n_boards, int(self.width), int(self.height), int(self.regs_per_board), int(self.global_regs))

    @property
    def n_tokens(self) -> int:
        return self.layout.n_tokens

    @property
    def h_present(self) -> int:
        """Minimum half-turn index over the heads of active timelines (section 2.3)."""
        live = self.is_head & self.active
        return int(self.half_turn[live].min())

    @property
    def player_to_move(self) -> int:
        """0 white, 1 black: the colour of the present boards. Not the acting player once the
        present has shifted inside an unsubmitted action (``shifted``)."""
        return self.h_present & 1

    @property
    def shifted(self) -> bool:
        """Whether the present has shifted to the opponent inside the acting player's action
        (T5's obligations met): the acting player is not the present's colour."""
        return int(self.acting_player) != self.player_to_move

    @property
    def dt_present(self) -> np.ndarray:
        """Delta-t-to-present per board, ``h - h_present`` in half-turns (A3), float."""
        return (self.half_turn - self.h_present).astype(np.float32)

    @property
    def p_ref(self) -> int:
        """A translation-equivariant timeline reference: the smallest ``p`` present."""
        return int(self.timeline.min())

    # -- validation --

    def validate(self) -> None:
        """Check the contract; raise ``ValueError`` on the first violation."""
        n = self.n_boards
        if n < 1:
            raise ValueError("a position has at least one board")
        if int(self.width) < 1 or int(self.height) < 1:
            raise ValueError("board size must be positive")
        if int(self.regs_per_board) < 0 or int(self.global_regs) < 0:
            raise ValueError("register counts must be non-negative")
        for name in ("half_turn", "is_head", "active", "side_to_move"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name}: expected shape ({n},)")
        if np.any(self.half_turn < 0):
            raise ValueError("half_turn must be non-negative")
        if not np.isin(self.side_to_move, (0, 1)).all():
            raise ValueError("side_to_move must be 0 (white) or 1 (black)")
        if np.any(self.side_to_move != (self.half_turn & 1)):
            raise ValueError("side_to_move must equal h & 1 (h = 2T + 1 when black is to move)")
        if np.any((self.piece < 0) | (self.piece >= N_PIECE_TYPES)):
            raise ValueError(f"piece ids must be in 0..{N_PIECE_TYPES - 1}")
        if np.any((self.colour < 0) | (self.colour >= N_COLOURS)):
            raise ValueError("colour must be 0 (none), 1 (white) or 2 (black)")
        if np.any((self.piece == 0) != (self.colour == COLOUR_NONE)):
            raise ValueError("a square has a colour exactly when it holds a piece")
        if np.any(self.moved & (self.piece == 0)):
            raise ValueError("an empty square cannot carry the has-moved flag")
        # Timeline structure: distinct (p, h); one head per timeline, at its latest board;
        # the active flag is a property of the timeline.
        keys = set()
        for p, h in zip(self.timeline.tolist(), self.half_turn.tolist(), strict=True):
            if (p, h) in keys:
                raise ValueError(f"two boards at (p={p}, h={h})")
            keys.add((p, h))
        for p in np.unique(self.timeline):
            on = self.timeline == p
            heads = np.flatnonzero(on & self.is_head)
            if heads.size != 1:
                raise ValueError(f"timeline {p} must have exactly one head board, has {heads.size}")
            if self.half_turn[heads[0]] != self.half_turn[on].max():
                raise ValueError(f"timeline {p}: the head must be its latest board")
            if len(set(self.active[on].tolist())) != 1:
                raise ValueError(f"timeline {p}: boards disagree about whether it is active")
        if not np.any(self.is_head & self.active):
            raise ValueError("no active timeline: the present is undefined")
        self._validate_action()
        self._validate_lineage()
        layout = self.layout
        for name in ("edges", "moves", "opp_moves"):
            arr = getattr(self, name)
            if arr.size == 0:
                continue
            ends = arr[:, :2]
            if not layout.is_square_token(ends).all():
                raise ValueError(f"{name}: every endpoint must be a square token (registers are never endpoints)")
        if self.edges.size and np.any((self.edges[:, 2] < 0) | (self.edges[:, 2] >= len(RELATION_TYPES))):
            raise ValueError(f"edges: relation ids must be in 0..{len(RELATION_TYPES) - 1}")
        for name in ("moves", "opp_moves"):
            arr = getattr(self, name)
            if arr.size and np.any((arr[:, 2] < 0) | (arr[:, 2] >= N_PIECE_TYPES)):
                raise ValueError(f"{name}: promotion ids must be 0 (none) or a piece type id")

    def _validate_action(self) -> None:
        """The action (D8): the acting player is a colour; submitting is legal only where it is
        possible, and possible only once the present lies on the opponent's colour (T5)."""
        acting = self.player_to_move if self.acting_player is None else self.acting_player
        if isinstance(acting, (bool, np.bool_)) or int(acting) != acting or int(acting) not in (0, 1):
            raise ValueError("acting_player must be 0 (white) or 1 (black)")
        if self.submit_legal and not self.submit_possible:
            raise ValueError("submit_legal needs submit_possible: a submission is legal only where it is possible")
        if self.submit_possible and int(acting) == self.player_to_move:
            raise ValueError(
                "submit_possible needs the present shifted to the opponent (T5): the acting player cannot be the present's colour"
            )

    def _validate_lineage(self) -> None:
        """``parent`` and ``creation`` against the timeline structure (module docstring)."""
        n = self.n_boards
        parent = self.parent
        if parent.shape != (n,):
            raise ValueError(f"parent: expected shape ({n},)")
        if np.any((parent < -1) | (parent >= n)) or np.any(parent == np.arange(n)):
            raise ValueError("parent: each entry is -1 or the index of another board")
        previous = self.previous_boards()
        first_h = {int(p): int(self.half_turn[self.timeline == p].min()) for p in np.unique(self.timeline)}
        for b in range(n):
            p, h, a = int(self.timeline[b]), int(self.half_turn[b]), int(parent[b])
            if h > first_h[p]:
                if a != previous[b]:
                    raise ValueError(f"parent: board (p={p}, h={h}) must have its timeline's previous board as parent")
            elif a >= 0:
                if int(self.timeline[a]) == p or int(self.half_turn[a]) != h - 1:
                    raise ValueError(f"parent: the branch origin of timeline {p} must sit one half-turn before it, on another timeline")
                if self.is_head[a]:
                    raise ValueError(f"parent: the branch origin of timeline {p} is a head (travel onto a head does not branch)")
        source = self.source
        if source.shape != (n,):
            raise ValueError(f"source: expected shape ({n},)")
        if np.any((source < -1) | (source >= n)) or np.any(source == np.arange(n)):
            raise ValueError("source: each entry is -1 or the index of another board")
        if np.any(self.is_head[source[source >= 0]]):
            raise ValueError("source: a travel's departure board is superseded by the jump")
        if self.creation is None:
            return
        creation = self.creation
        has_source = source >= 0
        if np.any(creation[source[has_source]] >= creation[has_source]):
            raise ValueError("creation: a travelling piece departs before it arrives")
        if creation.shape != (n,):
            raise ValueError(f"creation: expected shape ({n},)")
        if len(np.unique(creation)) != n:
            raise ValueError("creation: every board needs its own creation order")
        has = parent >= 0
        if np.any(creation[parent[has]] >= creation[has]):
            raise ValueError("creation: a board is created after its parent")
        for p in np.unique(self.timeline):
            on = np.flatnonzero(self.timeline == p)
            by_time = on[np.argsort(self.half_turn[on])]
            if np.any(np.diff(creation[by_time]) <= 0):
                raise ValueError(f"creation: timeline {p}'s boards are created in time order")

    # -- token-level arrays --

    def token_arrays(self) -> dict[str, np.ndarray]:
        """Per-token arrays in this position's own (canonical) layout, length ``T``."""
        lay = self.layout
        n, wh, tpb, nb = self.n_boards, lay.squares_per_board, lay.tokens_per_board, lay.n_body
        t = np.arange(nb)
        board = t // tpb
        local = t % tpb
        is_sq = local < wh
        sq = np.where(is_sq, local, 0)
        kind = np.where(is_sq, TOKEN_SQUARE, TOKEN_BOARD_REGISTER)
        out = {
            "kind": np.concatenate([kind, np.full(lay.global_regs, TOKEN_GLOBAL_REGISTER)]),
            "board": np.concatenate([board, np.full(lay.global_regs, -1)]),
            "reg_index": np.concatenate([np.where(is_sq, 0, local - wh), np.arange(lay.global_regs)]),
            "piece": np.concatenate([np.where(is_sq, self.piece[board, sq], 0), np.zeros(lay.global_regs, np.int64)]),
            "colour": np.concatenate([np.where(is_sq, self.colour[board, sq], 0), np.zeros(lay.global_regs, np.int64)]),
            "moved": np.concatenate([np.where(is_sq, self.moved[board, sq], False), np.zeros(lay.global_regs, bool)]),
            "x": np.concatenate([np.where(is_sq, sq % self.width, 0), np.zeros(lay.global_regs, np.int64)]),
            "y": np.concatenate([np.where(is_sq, sq // self.width, 0), np.zeros(lay.global_regs, np.int64)]),
        }
        dt = self.dt_present
        for name, per_board in (
            ("p", self.timeline),
            ("h", self.half_turn),
            ("is_head", self.is_head.astype(np.int64)),
            ("active", self.active.astype(np.int64)),
            ("stm", self.side_to_move),
        ):
            out[name] = np.concatenate([per_board[board], np.zeros(lay.global_regs, np.int64)])
        out["dt"] = np.concatenate([dt[board], np.zeros(lay.global_regs, np.float32)]).astype(np.float32)
        assert all(v.shape[0] == lay.n_tokens for v in out.values()) and n >= 1
        return out


# --- transforms used by tests and by the data pipeline -------------------------------------


def _remap_tokens(arr: np.ndarray, old: TokenLayout, new: TokenLayout, board_map: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.copy()
    out = arr.copy()
    for col in (0, 1):
        board, sq = old.split_square_tokens(arr[:, col])
        out[:, col] = board_map[board] * new.tokens_per_board + sq
    return out


def permute_boards(state: PositionState, order: Sequence[int]) -> PositionState:
    """The same position with its boards stored in a different order.

    ``order[i]`` is the old index of the board stored at new index ``i``. Coordinates,
    flags and contents travel with their boards; edge and move token indices are
    remapped, and their list order is kept.
    """
    order = np.asarray(order, dtype=np.int64)
    n = state.n_boards
    if sorted(order.tolist()) != list(range(n)):
        raise ValueError("order must be a permutation of the board indices")
    new_index = np.empty(n, dtype=np.int64)
    new_index[order] = np.arange(n)
    lay = state.layout
    old_parent = state.parent[order]
    parent = np.where(old_parent >= 0, new_index[np.maximum(old_parent, 0)], -1)
    old_source = state.source[order]
    source = np.where(old_source >= 0, new_index[np.maximum(old_source, 0)], -1)
    return PositionState(
        width=state.width,
        height=state.height,
        timeline=state.timeline[order],
        half_turn=state.half_turn[order],
        is_head=state.is_head[order],
        active=state.active[order],
        side_to_move=state.side_to_move[order],
        piece=state.piece[order],
        colour=state.colour[order],
        moved=state.moved[order],
        edges=_remap_tokens(state.edges, lay, lay, new_index),
        moves=_remap_tokens(state.moves, lay, lay, new_index),
        opp_moves=_remap_tokens(state.opp_moves, lay, lay, new_index),
        regs_per_board=state.regs_per_board,
        global_regs=state.global_regs,
        parent=parent,
        creation=None if state.creation is None else state.creation[order],
        source=source,
        acting_player=state.acting_player,
        submit_possible=state.submit_possible,
        submit_legal=state.submit_legal,
    )


def translate(state: PositionState, dp: int = 0, dh: int = 0) -> PositionState:
    """Shift every timeline by ``dp`` and every board's half-turn index by ``dh``.

    ``dh`` must be even so that each board keeps its side to move. Under the relative
    encoding (section 4) the network's outputs are invariant to this.
    """
    if dh % 2:
        raise ValueError("dh must be even: an odd shift changes every board's side to move")
    if np.any(state.half_turn + dh < 0):
        raise ValueError("translation would make a half-turn index negative")
    return replace(state, timeline=state.timeline + int(dp), half_turn=state.half_turn + int(dh))


def relayout(state: PositionState, regs_per_board: int, global_regs: int) -> PositionState:
    """The same position for a different register count (ablation 3: registers on/off,
    2 vs 4 per board), with edge and move token indices remapped."""
    old = state.layout
    new = TokenLayout(old.n_boards, old.width, old.height, int(regs_per_board), int(global_regs))
    ident = np.arange(old.n_boards)
    return replace(
        state,
        edges=_remap_tokens(state.edges, old, new, ident),
        moves=_remap_tokens(state.moves, old, new, ident),
        opp_moves=_remap_tokens(state.opp_moves, old, new, ident),
        regs_per_board=int(regs_per_board),
        global_regs=int(global_regs),
    )


# --- batching ---------------------------------------------------------------------------


@dataclass
class Batch:
    """A padded batch of positions, as tensors.

    Batch token layout: position ``b``'s board tokens (squares and per-board registers,
    in its canonical order) occupy ``[0, n_body[b])``, padding fills ``[n_body[b],
    body_len)``, and the global registers occupy ``[body_len, body_len + G)`` for every
    position. For a single position there is no padding and the batch layout *is* the
    canonical layout. Square-token indices coincide in both layouts, so edges and moves
    are passed through unchanged. (Token order carries no meaning to the network, so
    where padding sits changes nothing; padded keys are masked.)
    """

    piece: torch.Tensor  # (B, L) long
    colour: torch.Tensor  # (B, L) long
    moved: torch.Tensor  # (B, L) long
    kind: torch.Tensor  # (B, L) long: TOKEN_* constants
    reg_index: torch.Tensor  # (B, L) long: register slot (per-board) or index (global)
    board: torch.Tensor  # (B, L) long: board index within the position, -1 if none
    is_head: torch.Tensor  # (B, L) long, board flag per token (0 where no board)
    active: torch.Tensor  # (B, L) long
    stm: torch.Tensor  # (B, L) long
    dt: torch.Tensor  # (B, L) float32, Delta-t-to-present of the token's board
    coords: torch.Tensor  # (B, L, 4) float32: file x, rank y, timeline p, time h (raw)
    coord_ref: torch.Tensor  # (B, 2) float32: (p_ref, h_present) per position
    valid: torch.Tensor  # (B, L) bool: not padding
    n_body: torch.Tensor  # (B,) long
    n_boards: torch.Tensor  # (B,) long
    board_valid: torch.Tensor  # (B, Nmax) bool
    moves: torch.Tensor  # (B, M, 3) long
    move_valid: torch.Tensor  # (B, M) bool
    opp_moves: torch.Tensor  # (B, M2, 3) long
    opp_move_valid: torch.Tensor  # (B, M2) bool
    edges: torch.Tensor  # (B, E, 3) long
    edge_valid: torch.Tensor  # (B, E) bool
    body_len: int
    regs_per_board: int
    global_regs: int
    layouts: tuple[TokenLayout, ...]
    # Per board (stage N3, the block-causal scope): (B, Nmax), padding -1 / False.
    board_parent: torch.Tensor | None = None  # long: parent board index (PositionState.parent)
    board_creation: torch.Tensor | None = None  # long: creation order, -1 where unknown
    board_source: torch.Tensor | None = None  # long: a travel arrival's departure board, -1 none
    board_head: torch.Tensor | None = None  # bool: head (live) board
    has_creation: torch.Tensor | None = None  # (B,) bool: the position carries a creation order
    # Per position (stage N5, register D8): (B,).
    acting_player: torch.Tensor | None = None  # long: 0 white, 1 black - whose action this is
    submit_possible: torch.Tensor | None = None  # bool: submitting is an action here (T5)
    submit_legal: torch.Tensor | None = None  # bool: ... and legal (C4); the submit logit is masked otherwise

    def acting(self) -> torch.Tensor:
        """``(B,)`` long: each position's acting player (the present's colour for a batch made
        without one)."""
        if self.acting_player is not None:
            return self.acting_player
        return self.coord_ref[:, 1].long() & 1

    def submit_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(possible, legal)``, each ``(B,)`` bool (all False for a batch made without them)."""
        bsz = self.piece.shape[0]
        none = torch.zeros(bsz, dtype=torch.bool, device=self.piece.device)
        possible = none if self.submit_possible is None else self.submit_possible
        legal = none if self.submit_legal is None else self.submit_legal
        return possible, legal & possible

    @property
    def batch_size(self) -> int:
        return int(self.piece.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.piece.shape[1])

    @property
    def square_mask(self) -> torch.Tensor:
        return self.valid & (self.kind == TOKEN_SQUARE)

    def batch_token(self, position: int, canonical_token: int) -> int:
        """Map a token index of position ``position``'s own layout to the batch layout."""
        nb = int(self.n_body[position])
        if canonical_token < nb:
            return int(canonical_token)
        return self.body_len + int(canonical_token) - nb

    def to(self, device: torch.device | str) -> Batch:
        moved = {name: (value.to(device) if isinstance(value, torch.Tensor) else value) for name, value in self.__dict__.items()}
        return Batch(**moved)


def _pad_triples(arrays: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max([a.shape[0] for a in arrays] + [0])
    out = np.zeros((len(arrays), width, 3), dtype=np.int64)
    valid = np.zeros((len(arrays), width), dtype=bool)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
        valid[i, : a.shape[0]] = True
    return torch.from_numpy(out), torch.from_numpy(valid)


def collate(states: Sequence[PositionState]) -> Batch:
    """Pad and stack positions into a ``Batch`` (see its docstring for the layout).

    Positions may differ in board count and board size; they must share the register
    counts (use ``relayout`` first otherwise).
    """
    if not states:
        raise ValueError("collate needs at least one position")
    regs = {(s.regs_per_board, s.global_regs) for s in states}
    if len(regs) != 1:
        raise ValueError(f"positions disagree about register counts: {sorted(regs)}")
    r, g = regs.pop()
    layouts = tuple(s.layout for s in states)
    body_len = max(lay.n_body for lay in layouts)
    big_l = body_len + g
    bsz = len(states)
    n_max = max(lay.n_boards for lay in layouts)

    long_fields = ("piece", "colour", "moved", "kind", "reg_index", "board", "is_head", "active", "stm")
    arrays = {name: np.zeros((bsz, big_l), dtype=np.int64) for name in long_fields}
    arrays["kind"][:] = TOKEN_PAD
    arrays["board"][:] = -1
    dt = np.zeros((bsz, big_l), dtype=np.float32)
    coords = np.zeros((bsz, big_l, 4), dtype=np.float32)
    coord_ref = np.zeros((bsz, 2), dtype=np.float32)
    valid = np.zeros((bsz, big_l), dtype=bool)
    board_valid = np.zeros((bsz, n_max), dtype=bool)
    board_parent = np.full((bsz, n_max), -1, dtype=np.int64)
    board_creation = np.full((bsz, n_max), -1, dtype=np.int64)
    board_source = np.full((bsz, n_max), -1, dtype=np.int64)
    board_head = np.zeros((bsz, n_max), dtype=bool)
    has_creation = np.zeros(bsz, dtype=bool)

    for i, (state, lay) in enumerate(zip(states, layouts, strict=True)):
        board_parent[i, : lay.n_boards] = state.parent
        board_source[i, : lay.n_boards] = state.source
        board_head[i, : lay.n_boards] = state.is_head
        if state.creation is not None:
            board_creation[i, : lay.n_boards] = state.creation
            has_creation[i] = True
        tok = state.token_arrays()
        nb = lay.n_body
        dest = np.concatenate([np.arange(nb), body_len + np.arange(g)])
        for name in long_fields:
            arrays[name][i, dest] = tok[name].astype(np.int64)
        dt[i, dest] = tok["dt"]
        coords[i, dest, 0] = tok["x"]
        coords[i, dest, 1] = tok["y"]
        coords[i, dest, 2] = tok["p"]
        coords[i, dest, 3] = tok["h"]
        coord_ref[i] = (state.p_ref, state.h_present)
        valid[i, dest] = True
        board_valid[i, : lay.n_boards] = True

    moves, move_valid = _pad_triples([s.moves for s in states])
    opp_moves, opp_valid = _pad_triples([s.opp_moves for s in states])
    edges, edge_valid = _pad_triples([s.edges for s in states])
    t = {name: torch.from_numpy(a) for name, a in arrays.items()}
    return Batch(
        piece=t["piece"],
        colour=t["colour"],
        moved=t["moved"],
        kind=t["kind"],
        reg_index=t["reg_index"],
        board=t["board"],
        is_head=t["is_head"],
        active=t["active"],
        stm=t["stm"],
        dt=torch.from_numpy(dt),
        coords=torch.from_numpy(coords),
        coord_ref=torch.from_numpy(coord_ref),
        valid=torch.from_numpy(valid),
        n_body=torch.tensor([lay.n_body for lay in layouts], dtype=torch.long),
        n_boards=torch.tensor([lay.n_boards for lay in layouts], dtype=torch.long),
        board_valid=torch.from_numpy(board_valid),
        moves=moves,
        move_valid=move_valid,
        opp_moves=opp_moves,
        opp_move_valid=opp_valid,
        edges=edges,
        edge_valid=edge_valid,
        body_len=int(body_len),
        regs_per_board=int(r),
        global_regs=int(g),
        layouts=layouts,
        board_parent=torch.from_numpy(board_parent),
        board_creation=torch.from_numpy(board_creation),
        board_source=torch.from_numpy(board_source),
        board_head=torch.from_numpy(board_head),
        has_creation=torch.from_numpy(has_creation),
        acting_player=torch.tensor([int(s.acting_player) for s in states], dtype=torch.long),
        submit_possible=torch.tensor([bool(s.submit_possible) for s in states], dtype=torch.bool),
        submit_legal=torch.tensor([bool(s.submit_legal) for s in states], dtype=torch.bool),
    )
