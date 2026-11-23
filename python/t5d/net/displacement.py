"""B15: the 4D displacement bias (provisional, the v1 default; spec section 4.1, register B15).

Written from section 4.1 and the rulebook's conventions (``docs/rules/README.md``), never from
an oracle or from Lc0's or Chessformer's code. Section 4.1, pinned at CHECKPOINT 6: "one
learned scalar per head per layer, indexed by the pair's displacement (Delta-file, Delta-rank,
Delta-timeline, Delta-time in half-turns)":

* "A full 15 x 15 file-rank slice for every board-distance class. Classes are exact up to two
  timelines and two turns apart (|Delta-timeline| <= 2, |Delta-time| <= 4 half-turns), then
  binned and clamped, so unseen board counts extrapolate as the binned biases did; the
  same-board class is Chessformer's table."
* "Plus exact line classes for displacements beyond them: movement along one axis only, or
  equal steps along two, three or four axes - the rook, bishop, unicorn and dragon lines."
* "Pairs involving registers get their own entries: per-board registers by their
  board-distance class, global registers one entry per head (section 5.3). It is computed at
  score time from coordinates, so nothing cached goes stale. About 1.6M parameters."
* It replaces B6's binned Delta-time/Delta-timeline biases and sits beside RoPE (B4), which
  stays. Ablation 2 keeps a no-table arm: ``NetConfig.displacement_bias=False``.

**The displacement** of a pair is key minus query, signed: ``(dx, dy, dp, dh)`` - file, rank,
timeline (the timeline coordinate ``p``, adjacent timelines 1 apart: ``state.timeline_index``)
and time in half-turns (``h = 2T + [black to move]``). The rulebook's displacement is ``(dx,
dy, dL, dT)`` with ``dT`` in whole turns; a pair of boards an odd number of half-turns apart
has no whole-turn ``dT`` (its boards have different colours to move).

**Board-distance classes** (of ``(dp, dh)``; ``board_class``):

* *exact* (pinned): ``|dp| <= 2`` and ``|dh| <= 4`` - 5 x 9 = 45 classes, odd ``dh``
  included; ``(0, 0)`` is the same-board class (Chessformer's 15 x 15 table). Knight and king
  relations fall inside them. Equivalently ``D <= 2`` for ``D = max(|dp|, ceil(|dh| / 2))``,
  the board distance in timelines and whole turns;
* *far* - "binned and clamped", whose bins section 4.1 does not give. COMMITTED-CLASS
  PLACEHOLDER (``NetConfig.disp_far_edges``; spec issue 205): the exact box's own distance
  ``D``, log2-binned from 3 - bins ``[3, 4]``, ``[5, 8]``, ``[9, inf)`` by default, the last
  clamped, as B6's buckets were - crossed with the sign of ``dh`` (past, same time, future;
  B6 kept Delta-time signed): 3 x 3 = 9 far classes. 54 classes in all, each with its
  15 x 15 slice. The edges are the setting; the form (one distance, ``dh``'s sign) is fixed
  here. Each far class mixes B6's Delta-time and Delta-timeline buckets, so the far classes
  alone hold a B6 bias only where it depends on nothing but ``dh``'s sign (issue 205) - the
  per-axis tables, next, are what keeps the rest.

**B6's two per-axis tables beyond the exact classes** (register B6 and B15; section 4.1, the
2026-11-15 review, decision 1: "a far pair ... adds B6's two clamped log-bucket scalars, one for
Delta-time and one for Delta-timeline"; ``NetConfig.disp_far_axis``, on - the v1 default;
``FarAxes``, ``far_cell``, ``pair_far_cell``). The far slice alone cannot tell boards apart that
one distance ``D`` merges - a board 20 half-turns back on the query's own timeline and one 2
half-turns back 10 timelines over share a class and, off the lines, every entry. So every pair
of board tokens (squares or per-board registers; a pair on a line included) whose boards lie
outside the exact classes adds two learned scalars per head per layer to its entry: one by the
bucket of its signed ``dh`` - B6's, exact below 4 half-turns, then 4-7, 8-15, ..., 64-127, 128+,
past and future apart: 19 - and one by the bucket of ``|dp|`` - exact below 4, then 4-7, 8-15,
16-31, 32+: 8 (``disp_far_time_*``, ``disp_far_timeline_*``; issue 205 records B6's buckets).
A pair inside the exact classes, a pair with a global register, and padding add nothing. With
the main table constant on file and rank and zero beyond the exact classes, the bias is B6's:
"B6 is exactly its special case that ignores file and rank" (section 4.1). The readings built
where section 4.1 does not say: the terms go on every pair of board tokens on far boards, not
on far-slice cells only (B6 biased by board pair whatever the token), and the tables are per
head per layer (section 4.1's 3,456 = (19 + 8) x 8 x 16).

**The layer-shared arm** (ablation 2; section 4.1 and 10.2, the 2026-11-15 review;
``NetConfig.disp_layer_shared``, off): one main table and one pair of per-axis tables per head,
read by every layer - "a sixteenth of the parameters and a bias that can be assembled once per
forward". The structure then needs no layer (``AttnStructure.layer_dependent``), and at
inference it assembles each selection of the bias once per forward and every layer reads it
(register G11; ``AttnStructure.disp_rows``, within ``structure.DISP_ONCE_BYTES``).

**Line classes** (``line_entry``): beyond the exact classes, a pair of squares whose
displacement ``(dx, dy, dp, dT)`` in whole turns (``dh`` even) has every non-zero component of
one magnitude ``k`` - "equal steps along one, two, three or four axes", the rulebook's 80
directions (its section 4: the queen's 8 + 24 + 32 + 16) - gets an entry of its own per
direction and ``k``, which replaces its far slice's entry (one scalar per pair - the partition
reading). Beyond the exact box such a line moves ``k >= 3`` on a board axis. Of the 72
directions with a board component, 64 also move on the board (``k <= 7`` there: 5 entries
each) and 8 move on the board axes only, where ``k`` is unbounded: entries ``k = 3 ..
disp_line_max``, the last clamped (every longer line shares it). COMMITTED-CLASS PLACEHOLDERS
(spec issue 206): the partition reading, the time unit (whole turns: an odd ``dh`` is never a
line), the clamp ``disp_line_max = 8`` - the clamp a setting, the other two fixed here.

**Register and global entries** (spec issue 207). A pair with a per-board register gets the
entry of its two boards' class and its kind - square reads register, register reads square,
register reads register (``disp_register_pairs = "by_kind"``, the placeholder; ``"shared"``:
one entry per class) - and never a (file, rank): registers are position-free on file and
rank (B8, CHECKPOINT 6). A pair with a global register gets the one global entry of its head
and layer, either side. (Every key of a global query's row then shares it, so it moves only
board queries' attention to the global registers - softmax is shift-invariant.) Padding gets
zero, not a parameter.

**Size** (defaults): 54 x 225 slices + 320 + 48 lines + 54 x 3 register entries + 1 global =
12,681 entries per head per layer; x 8 heads x 16 layers = 1,623,168 parameters. Three entries
per head and layer are never indexed: the (0, 0) cell of the three same-time far slices, whose
displacements are all board-axis lines. The per-axis tables add 19 + 8 = 27 scalars per head per
layer, 3,456: B15 is 1,626,624 parameters in the v1 (2,439,936 in F2's 24 layers). Under the
layer-shared arm it is 8 x (12,681 + 27) = 101,664, a sixteenth.

**Two implementations of one function**, compared by the tests and by section 15.2's B15 row:

* ``pair_entry`` - the entry of a pair computed from its tokens' kinds and coordinates by
  integer arithmetic: B14's score function (``flex.py``) evaluates it inside the attention
  "at score time from coordinates", per pair, and indexes the layer's compact table;
* the CPU paths' lookup (``AttnStructure`` in ``structure.py``): a board pair's *tile*
  (``tile_index``: its class, or for a far pair that can hold lines its board direction and
  step) and each token's *code* (a square ``y * 8 + x``, a per-board register 64) index a
  static ``class_map`` ``(tiles, 65, 65)``, built once from ``pair_entry``; then the layer's
  table. A tile stands for every board pair beyond ``tile_cap`` steps along its direction.

The per-axis terms have the same two: ``pair_far_cell`` by arithmetic inside the score function,
and for the CPU paths a per-batch ``(Nb + 2, Nb + 2)`` matrix of far cells by board slot
(``DisplacementIndex.far_cells``), through which the layer's far table - the two tables summed
per (time bucket, timeline bucket) cell, a zero cell last - is gathered by the two tokens'
slots. Both add ``main + (time + timeline)`` in that order, so they agree bit for bit.

**Initialisation** (placeholder, ``NetConfig.disp_init_std``; spec issue 208): ``N(0, 0.1)``,
drawn from a generator of B15's own seeded from the model's seed (``model.b15_seed``, as D8's
are), never from the global stream - so the no-table arm has every other weight of the v1. The
two per-axis tables are drawn the same way from a second generator (``model.b15_far_seed``),
after the main table: adding them moved no parameter's draw, the main table's included.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass

import torch
from torch import nn

from .config import NetConfig
from .state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_PAD, TOKEN_SQUARE, Batch

#: Section 4.1, pinned at CHECKPOINT 6: classes are exact up to two timelines and two turns
#: (four half-turns) apart.
EXACT_TIMELINES = 2
EXACT_HALF_TURNS = 4
N_EXACT = (2 * EXACT_TIMELINES + 1) * (2 * EXACT_HALF_TURNS + 1)
#: The file-rank slice: 15 x 15, |Delta-file| and |Delta-rank| up to 7 - boards up to 8 x 8.
SPAN = 7
SIDE = 2 * SPAN + 1
SLICE = SIDE * SIDE
MAX_BOARD = SPAN + 1
#: Token codes on the 8 x 8 canvas: a square ``y * 8 + x``; every per-board register 64.
REGISTER_CODE = MAX_BOARD * MAX_BOARD
N_CODES = REGISTER_CODE + 1
#: The board distance just beyond the exact classes: a line beyond them moves at least this
#: many steps on a board axis (both half-widths of the exact box are two steps).
FIRST_FAR = EXACT_TIMELINES + 1
#: Register-pair kinds under ``disp_register_pairs="by_kind"``.
KIND_SQUARE_READS_REGISTER, KIND_REGISTER_READS_SQUARE, KIND_REGISTER_READS_REGISTER = 0, 1, 2


def _directions() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """The line directions that leave the board, as unit steps ``(x, y, p, T)``: those that
    also move on the board (64), and those along the board axes only (8). (The 8 directions
    with no board component stay on one board - the same-board slice holds them.)"""
    spatial, board_only = [], []
    for u in itertools.product((-1, 0, 1), repeat=4):
        if u[2] == 0 and u[3] == 0:
            continue
        (spatial if (u[0] or u[1]) else board_only).append(u)
    return spatial, board_only


SPATIAL_DIRECTIONS, BOARD_DIRECTIONS = _directions()


def direction_code(u) -> int:
    """A direction's base-3 code ``(x+1) 27 + (y+1) 9 + (p+1) 3 + (T+1)`` in ``[0, 81)``."""
    return (u[0] + 1) * 27 + (u[1] + 1) * 9 + (u[2] + 1) * 3 + (u[3] + 1)


def _lookup(directions) -> torch.Tensor:
    out = torch.full((81,), -1, dtype=torch.long)
    for i, u in enumerate(directions):
        out[direction_code(u)] = i
    return out


#: Each direction code's index among the spatial or the board-only line directions (-1: none).
SPATIAL_INDEX = _lookup(SPATIAL_DIRECTIONS)
BOARD_INDEX = _lookup(BOARD_DIRECTIONS)


@dataclass(frozen=True)
class DisplacementGeometry:
    """The class layout of one configuration (the module docstring): which entry of a head's
    table, per layer, each pair indexes."""

    far_edges: tuple[int, ...] = (3, 5, 9)
    line_max: int = 8
    register_kinds: int = 3

    @classmethod
    def from_config(cls, cfg: NetConfig) -> DisplacementGeometry:
        kinds = 3 if cfg.disp_register_pairs == "by_kind" else 1
        return cls(tuple(int(e) for e in cfg.disp_far_edges), int(cfg.disp_line_max), kinds)

    # -- the layout --

    @property
    def n_far_bins(self) -> int:
        return len(self.far_edges)

    @property
    def n_board_classes(self) -> int:
        """Exact classes, then the far ones: ``45 + 3 x bins`` (54 by default)."""
        return N_EXACT + 3 * self.n_far_bins

    @property
    def spatial_steps(self) -> int:
        """Line entries per direction that also moves on the board: ``k = 3 .. 7``."""
        return SPAN - FIRST_FAR + 1

    @property
    def board_steps(self) -> int:
        """Line entries per board-axis direction: ``k = 3 .. line_max``, the last clamped."""
        return self.line_max - FIRST_FAR + 1

    @property
    def lines_offset(self) -> int:
        return self.n_board_classes * SLICE

    @property
    def board_lines_offset(self) -> int:
        return self.lines_offset + len(SPATIAL_DIRECTIONS) * self.spatial_steps

    @property
    def registers_offset(self) -> int:
        return self.board_lines_offset + len(BOARD_DIRECTIONS) * self.board_steps

    @property
    def global_entry(self) -> int:
        return self.registers_offset + self.n_board_classes * self.register_kinds

    @property
    def n_entries(self) -> int:
        """Learned scalars per head per layer (12,681 by default)."""
        return self.global_entry + 1

    @property
    def zero_entry(self) -> int:
        """The index of the zero appended to a layer's table (padding; not a parameter)."""
        return self.n_entries

    # -- the CPU paths' tiles --

    @property
    def tile_cap(self) -> int:
        """The step from which a far board pair that can hold lines no longer changes its tile:
        no square line is longer than 7, board-axis lines clamp at ``line_max``, and the far
        bins at their last edge."""
        return max(MAX_BOARD, self.line_max, self.far_edges[-1])

    @property
    def tile_steps(self) -> int:
        return self.tile_cap - FIRST_FAR + 1

    @property
    def n_tiles(self) -> int:
        """Board-class tiles, then one per board-axis direction and step ``3 .. tile_cap``."""
        return self.n_board_classes + len(BOARD_DIRECTIONS) * self.tile_steps

    @property
    def global_tile(self) -> int:
        return self.n_tiles

    @property
    def zero_tile(self) -> int:
        return self.n_tiles + 1


def _log_edges(n_exact: int, max_distance: int) -> tuple[int, ...]:
    """The lower edges of the log2-spaced buckets above the exact ones: ``n_exact 2^j`` for ``j =
    1, 2, ...`` up to ``max_distance`` (the bucket ``[n_exact, 2 n_exact)`` needs none)."""
    edges, edge = [], 2 * n_exact
    while edge <= max_distance:
        edges.append(edge)
        edge *= 2
    return tuple(edges)


@dataclass(frozen=True)
class FarAxes:
    """The buckets of B6's two per-axis tables, which B15 keeps beyond its exact classes (the
    module docstring; register B6, B15; section 4.1, the 2026-11-15 review). An offset's size has
    its own bucket below ``*_exact``, then log2-spaced buckets up to ``*_max``, the last clamped.
    Delta-time (half-turns) is signed - bucket ``b`` of ``|dh|`` for ``dh >= 0``, ``n - 1 + b``
    for ``dh < 0`` (``n`` sizes; the signs share bucket 0 only): 19 by default; Delta-timeline is
    not: 8. A far board pair's *cell* is ``time bucket x n_timeline + timeline bucket``."""

    time_exact: int = 4
    time_max: int = 128
    timeline_exact: int = 4
    timeline_max: int = 32

    @classmethod
    def from_config(cls, cfg: NetConfig) -> FarAxes:
        time, timeline = (cfg.disp_far_time_exact, cfg.disp_far_time_max), (cfg.disp_far_timeline_exact, cfg.disp_far_timeline_max)
        return cls(int(time[0]), int(time[1]), int(timeline[0]), int(timeline[1]))

    @property
    def time_edges(self) -> tuple[int, ...]:
        """``(8, 16, 32, 64, 128)`` by default."""
        return _log_edges(self.time_exact, self.time_max)

    @property
    def timeline_edges(self) -> tuple[int, ...]:
        """``(8, 16, 32)`` by default."""
        return _log_edges(self.timeline_exact, self.timeline_max)

    @property
    def n_time_sizes(self) -> int:
        """Buckets of ``|dh|``: the exact ones, ``[exact, 2 exact)``, and one per edge (10)."""
        return self.time_exact + 1 + len(self.time_edges)

    @property
    def n_time(self) -> int:
        """Signed Delta-time buckets (19 by default): the time table's entries per head per layer."""
        return 2 * self.n_time_sizes - 1

    @property
    def n_timeline(self) -> int:
        """|Delta-timeline| buckets (8 by default): the timeline table's entries per head per layer."""
        return self.timeline_exact + 1 + len(self.timeline_edges)

    @property
    def n_cells(self) -> int:
        """(time bucket, timeline bucket) cells (152 by default)."""
        return self.n_time * self.n_timeline

    @property
    def zero_cell(self) -> int:
        """The index of the zero appended to a layer's far table: a pair that takes no per-axis term
        (inside the exact classes, a global register, padding; not a parameter)."""
        return self.n_cells


# --- the class of a pair, by arithmetic (elementwise; also inside B14's score function) -----------
#
# Every function takes integer tensors that broadcast together and uses only elementwise torch
# operations and lookups into small tensors, so that it runs per element under vmap - which is
# how torch's eager FlexAttention evaluates a score function (``flex.py``).


def _sign(t: torch.Tensor) -> torch.Tensor:
    return (t > 0).long() - (t < 0).long()


def _max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(a >= b, a, b)


def board_class(dp: torch.Tensor, dh: torch.Tensor, geom: DisplacementGeometry) -> torch.Tensor:
    """The board-distance class of a pair of boards ``dp`` timelines and ``dh`` half-turns apart
    (key minus query): ``(dp + 2) 9 + (dh + 4)`` inside the exact box, else ``45 + (sign(dh)
    + 1) bins + bin(D)``, ``D = max(|dp|, ceil(|dh| / 2))``."""
    ap, ah = dp.abs(), dh.abs()
    exact = (ap <= EXACT_TIMELINES) & (ah <= EXACT_HALF_TURNS)
    exact_id = (dp + EXACT_TIMELINES) * (2 * EXACT_HALF_TURNS + 1) + (dh + EXACT_HALF_TURNS)
    d = _max(ap, (ah + 1) // 2)
    far_bin = torch.zeros_like(d)
    for edge in geom.far_edges[1:]:
        far_bin = far_bin + (d >= edge).long()
    far_id = N_EXACT + (_sign(dh) + 1) * geom.n_far_bins + far_bin
    return torch.where(exact, exact_id, far_id)


def beyond_exact(dp: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    """Whether a board pair lies outside the exact classes."""
    return (dp.abs() > EXACT_TIMELINES) | (dh.abs() > EXACT_HALF_TURNS)


def line_entry(dx, dy, dp, dh, geom: DisplacementGeometry) -> torch.Tensor:
    """The line entry of a pair of squares beyond the exact classes, or -1 where the pair is
    on no line (or inside the exact box, where every displacement is exact already)."""
    even = torch.remainder(dh, 2) == 0
    dt = torch.div(dh, 2, rounding_mode="floor")  # whole turns (exact where even)
    ax, ay, ap, at = dx.abs(), dy.abs(), dp.abs(), dt.abs()
    k = _max(_max(ax, ay), _max(ap, at))
    equal = ((ax == 0) | (ax == k)) & ((ay == 0) | (ay == k)) & ((ap == 0) | (ap == k)) & ((at == 0) | (at == k))
    on = even & equal & beyond_exact(dp, dh)
    code = (_sign(dx) + 1) * 27 + (_sign(dy) + 1) * 9 + (_sign(dp) + 1) * 3 + (_sign(dt) + 1)
    spatial = (ax > 0) | (ay > 0)
    s_index = SPATIAL_INDEX.to(code.device)[code]
    b_index = BOARD_INDEX.to(code.device)[code]
    e_spatial = geom.lines_offset + s_index * geom.spatial_steps + (k - FIRST_FAR)
    e_board = geom.board_lines_offset + b_index * geom.board_steps + (k.clamp(max=geom.line_max) - FIRST_FAR)
    return torch.where(on, torch.where(spatial, e_spatial, e_board), torch.full_like(k, -1))


def entry_from_class(bc, line, kind_q, kind_k, dx, dy, geom: DisplacementGeometry) -> torch.Tensor:
    """A pair's entry given its board class ``bc`` and its line entry ``line`` (-1: none): the
    line's, else its slice cell ``(dx, dy)`` for two squares; the register entry by class and
    kind for a pair with a per-board register; the global entry for a pair with a global
    register; the zero entry for padding."""
    squares = (kind_q == TOKEN_SQUARE) & (kind_k == TOKEN_SQUARE)
    cell = bc * SLICE + (dx + SPAN) * SIDE + (dy + SPAN)
    e = torch.where(squares & (line >= 0), line, cell)
    if geom.register_kinds == 3:
        kind = torch.where(
            kind_q == TOKEN_SQUARE,
            torch.full_like(bc, KIND_SQUARE_READS_REGISTER),
            torch.where(
                kind_k == TOKEN_SQUARE, torch.full_like(bc, KIND_REGISTER_READS_SQUARE), torch.full_like(bc, KIND_REGISTER_READS_REGISTER)
            ),
        )
    else:
        kind = torch.zeros_like(bc)
    e = torch.where(squares, e, geom.registers_offset + bc * geom.register_kinds + kind)
    e = torch.where((kind_q == TOKEN_GLOBAL_REGISTER) | (kind_k == TOKEN_GLOBAL_REGISTER), torch.full_like(e, geom.global_entry), e)
    return torch.where((kind_q == TOKEN_PAD) | (kind_k == TOKEN_PAD), torch.full_like(e, geom.zero_entry), e)


def pair_entry(kind_q, kind_k, dx, dy, dp, dh, geom: DisplacementGeometry) -> torch.Tensor:
    """The entry of a pair from its tokens' kinds and its displacement (key minus query) -
    "computed at score time from coordinates" (section 4.1)."""
    bc = board_class(dp, dh, geom)
    return entry_from_class(bc, line_entry(dx, dy, dp, dh, geom), kind_q, kind_k, dx, dy, geom)


def _size_bucket(size: torch.Tensor, n_exact: int, edges: tuple[int, ...]) -> torch.Tensor:
    """The bucket of a non-negative integer offset size: itself below ``n_exact``, else ``n_exact``
    plus the number of ``edges`` it has reached (the last bucket holds every larger size). Integer
    comparisons only - no float log2 - so that it is exact and runs under vmap."""
    log = torch.full_like(size, n_exact)
    for edge in edges:
        log = log + (size >= edge).long()
    return torch.where(size < n_exact, size, log)


def far_time_bucket(dh: torch.Tensor, far: FarAxes) -> torch.Tensor:
    """B6's signed Delta-time bucket of ``dh`` half-turns (key minus query), in ``[0, n_time)``."""
    b = _size_bucket(dh.abs(), far.time_exact, far.time_edges)
    return torch.where(dh >= 0, b, far.n_time_sizes - 1 + b)


def far_timeline_bucket(dp: torch.Tensor, far: FarAxes) -> torch.Tensor:
    """B6's |Delta-timeline| bucket of ``dp`` timelines, in ``[0, n_timeline)``."""
    return _size_bucket(dp.abs(), far.timeline_exact, far.timeline_edges)


def far_cell(dp: torch.Tensor, dh: torch.Tensor, far: FarAxes) -> torch.Tensor:
    """The per-axis cell of a pair of boards ``dp`` timelines and ``dh`` half-turns apart (key
    minus query): ``time bucket x n_timeline + timeline bucket`` beyond the exact classes, the zero
    cell inside them (section 4.1: B6's two scalars are added "beyond the exact classes")."""
    cell = far_time_bucket(dh, far) * far.n_timeline + far_timeline_bucket(dp, far)
    return torch.where(beyond_exact(dp, dh), cell, torch.full_like(cell, far.zero_cell))


def pair_far_cell(kind_q, kind_k, dp, dh, far: FarAxes) -> torch.Tensor:
    """The per-axis cell of a pair of tokens from their kinds and their boards' displacement: the
    boards' ``far_cell`` where both are board tokens - a square or a per-board register, each on
    its board - and the zero cell for a pair with a global register or padding (neither has a
    board). A function of the board pair alone: file and rank never enter (B6 had neither)."""
    on_q = (kind_q == TOKEN_SQUARE) | (kind_q == TOKEN_BOARD_REGISTER)
    on_k = (kind_k == TOKEN_SQUARE) | (kind_k == TOKEN_BOARD_REGISTER)
    cell = far_cell(dp, dh, far)
    return torch.where(on_q & on_k, cell, torch.full_like(cell, far.zero_cell))


# --- the CPU paths' lookup ----------------------------------------------------------------------------


def tile_index(dp: torch.Tensor, dh: torch.Tensor, geom: DisplacementGeometry) -> torch.Tensor:
    """The tile of a board pair (key minus query): its board class, or - for a far pair that
    can hold lines (``dh`` even; ``dp`` and the whole-turn ``dT`` each 0 or of one size ``k``)
    - its board direction's tile at step ``min(k, tile_cap)``."""
    bc = board_class(dp, dh, geom)
    even = torch.remainder(dh, 2) == 0
    dt = torch.div(dh, 2, rounding_mode="floor")
    ap, at = dp.abs(), dt.abs()
    k = _max(ap, at)
    capable = even & ((ap == 0) | (at == 0) | (ap == at)) & beyond_exact(dp, dh)
    code = direction_code((0, 0, 0, 0)) + (_sign(dp) * 3 + _sign(dt))  # the board direction's code, file and rank 0
    b_index = BOARD_INDEX.to(code.device)[code]
    line_tile = geom.n_board_classes + b_index * geom.tile_steps + (k.clamp(max=geom.tile_cap) - FIRST_FAR)
    return torch.where(capable, line_tile, bc)


def class_map(geom: DisplacementGeometry) -> torch.Tensor:
    """``(n_tiles + 2, 65, 65)`` long: the entry of every (tile, query code, key code), the
    global tile (every entry the global one) and the zero tile (padding) last. Built from
    ``pair_entry`` at each tile's displacement; a plain board-class tile takes no line."""
    codes = torch.arange(N_CODES)
    kind = torch.where(codes == REGISTER_CODE, TOKEN_BOARD_REGISTER, TOKEN_SQUARE)
    x, y = codes % MAX_BOARD, codes // MAX_BOARD
    kq, kk = kind[:, None], kind[None, :]
    dx, dy = x[None, :] - x[:, None], y[None, :] - y[:, None]
    tiles = []
    no_line = torch.full((N_CODES, N_CODES), -1, dtype=torch.long)
    for bc in range(geom.n_board_classes):
        tiles.append(entry_from_class(torch.full_like(dx, bc), no_line, kq, kk, dx, dy, geom))
    for u in BOARD_DIRECTIONS:
        for k in range(FIRST_FAR, geom.tile_cap + 1):
            dp, dh = torch.full_like(dx, k * u[2]), torch.full_like(dx, 2 * k * u[3])
            tiles.append(pair_entry(kq, kk, dx, dy, dp, dh, geom))
    tiles.append(torch.full((N_CODES, N_CODES), geom.global_entry, dtype=torch.long))
    tiles.append(torch.full((N_CODES, N_CODES), geom.zero_entry, dtype=torch.long))
    out = torch.stack(tiles)
    assert out.shape[0] == geom.n_tiles + 2
    return out


@functools.lru_cache(maxsize=8)
def _shared_class_map(geom: DisplacementGeometry) -> torch.Tensor:
    """``class_map(geom)``, built once per geometry and shared by every module that reads it
    (a read-only buffer)."""
    return class_map(geom)


@dataclass
class DisplacementIndex:
    """What B15 needs of one batch, per token and per board pair (``displacement_index``)."""

    slot: torch.Tensor  # (B, L) long: the token's board, Nb for a global register, Nb + 1 for padding
    code: torch.Tensor  # (B, L) long: a square y * 8 + x, a per-board register 64, else 0
    tiles: torch.Tensor  # (B, Nb + 2, Nb + 2) long: the tile of each (query slot, key slot)
    kind: torch.Tensor  # (B, L) long: token kinds (padding where the batch is padded)
    coords: torch.Tensor  # (B, L, 4) long: raw (file, rank, timeline, half-turn)
    #: (B, Nb + 2, Nb + 2) long: the per-axis far cell of each (query slot, key slot) - a pair of
    #: boards beyond the exact classes; the zero cell for every other pair of slots (None: the
    #: per-axis tables are off, ``NetConfig.disp_far_axis``).
    far_cells: torch.Tensor | None = None


@dataclass
class PairSides:
    """What B15's lookups read of a set of query rows and a set of keys - each token's board slot
    and code, laid out to index a pair: integers that depend on the position and the two sets,
    never on the layer (register G11, section 10.2: "structure is computed once per forward"), so
    a caller that asks for the same rows and keys at every layer builds them once
    (``AttnStructure``'s memo) and each layer gathers its own table through them."""

    slot_q: torch.Tensor  # (B, n) long: the queries' slots
    slot_k: torch.Tensor  # (B, Lk) long: the keys' slots
    code_q: torch.Tensor  # (B, n, 1) long: the queries' codes, times ``N_CODES``
    code_k: torch.Tensor  # (B, 1, Lk) long: the keys' codes
    b: torch.Tensor  # (B, 1, 1) long: the position, to index a (B, slots, slots) table by pair

    def __post_init__(self) -> None:
        # The slots as views that index a table by pair - (B, slots, slots), or per head (B, H,
        # slots, slots) - laid out with the sides so that no lookup lays them out again.
        self.q3, self.k3 = self.slot_q[:, :, None], self.slot_k[:, None, :]  # (B, n, 1), (B, 1, Lk)
        self.b4 = self.b[..., None]  # (B, 1, 1, 1)
        self.q4, self.k4 = self.slot_q[:, None, :, None], self.slot_k[:, None, None, :]  # (B, 1, n, 1), (B, 1, 1, Lk)


def pair_sides(
    index: DisplacementIndex, rows: torch.Tensor | None, cols: torch.Tensor | None = None, span: tuple[int, int] | None = None
) -> PairSides:
    """The sides of the queries ``rows`` ``(B, n)`` - or of the row span ``[start, stop)``, read
    by slicing instead of gathering (the same integers) - and the keys ``cols`` ``(B, K)`` (every
    key when None)."""
    if span is not None:
        slot_q, code_q = index.slot[:, span[0] : span[1]], index.code[:, span[0] : span[1]]
    else:
        slot_q, code_q = index.slot.gather(1, rows), index.code.gather(1, rows)
    slot_k = index.slot if cols is None else index.slot.gather(1, cols)
    code_k = index.code if cols is None else index.code.gather(1, cols)
    b = torch.arange(slot_q.shape[0], device=slot_q.device)[:, None, None]
    return PairSides(slot_q, slot_k, (code_q * N_CODES)[:, :, None], code_k[:, None, :], b)


def lookup_sides(index: DisplacementIndex, table_map: torch.Tensor, sides: PairSides) -> torch.Tensor:
    """``lookup`` for sides already built: ``table_map`` at (the pair's board tile, the query's
    code, the key's code), ``(B, n, Lk)``."""
    tile = index.tiles[sides.b, sides.q3, sides.k3]
    flat = tile * (N_CODES * N_CODES) + sides.code_q + sides.code_k
    return table_map.view(-1)[flat]


def lookup(index: DisplacementIndex, table_map: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor | None = None) -> torch.Tensor:
    """``(B, n, Lk)``: ``table_map`` - a ``(tiles, 65, 65)`` map like ``class_map`` (B15's entries,
    or B17's classes) - at (the pair's board tile, the query's code, the key's code) for the queries
    ``rows`` ``(B, n)`` and the keys ``cols`` ``(B, K)`` (every key when None)."""
    return lookup_sides(index, table_map, pair_sides(index, rows, cols))


def check_board_sizes(batch: Batch) -> None:
    """B15's slice spans boards up to 8 x 8 (section 4.1: "15 x 15")."""
    for lay in batch.layouts:
        if lay.width > MAX_BOARD or lay.height > MAX_BOARD:
            raise ValueError(
                f"B15's 15 x 15 file-rank slice spans boards up to {MAX_BOARD} x {MAX_BOARD}; this batch has a "
                f"{lay.width} x {lay.height} board (no variant has one - the slice would need widening)"
            )


def displacement_index(batch: Batch, geom: DisplacementGeometry, far: FarAxes | None = None) -> DisplacementIndex:
    """Each token's slot and code, and each board pair's tile, from the batch's raw coordinates
    (only differences enter, so neither recentring nor a common shift changes anything). With
    ``far`` (the per-axis tables' buckets), each pair of slots' far cell too."""
    check_board_sizes(batch)
    bsz, length = batch.kind.shape
    n_max = batch.board_valid.shape[1]
    device = batch.kind.device
    coords = batch.coords.round().long()
    kind = torch.where(batch.valid, batch.kind, torch.full_like(batch.kind, TOKEN_PAD))
    on_board = batch.valid & (batch.board >= 0)
    is_global = kind == TOKEN_GLOBAL_REGISTER
    slot = torch.where(
        on_board, batch.board, torch.where(is_global, torch.full_like(batch.board, n_max), torch.full_like(batch.board, n_max + 1))
    )
    code = torch.where(kind == TOKEN_SQUARE, coords[..., 1] * MAX_BOARD + coords[..., 0], torch.zeros_like(kind))
    code = torch.where(kind == TOKEN_BOARD_REGISTER, torch.full_like(code, REGISTER_CODE), code)
    # Per-board (p, h): every token of a board carries its board's.
    ph = torch.zeros(bsz, n_max + 2, 2, dtype=torch.long, device=device)
    ph.scatter_(1, slot[..., None].expand(bsz, length, 2), coords[..., 2:4])
    p, h = ph[:, :n_max, 0], ph[:, :n_max, 1]
    dp, dh = p[:, None, :] - p[:, :, None], h[:, None, :] - h[:, :, None]  # key minus query, (B, Nb, Nb)
    board_tiles = tile_index(dp, dh, geom)
    tiles = torch.full((bsz, n_max + 2, n_max + 2), geom.zero_tile, dtype=torch.long, device=device)
    valid_boards = batch.board_valid.to(device)
    both = valid_boards[:, :, None] & valid_boards[:, None, :]
    tiles[:, :n_max, :n_max] = torch.where(both, board_tiles, torch.full_like(board_tiles, geom.zero_tile))
    with_global = torch.cat([valid_boards, torch.ones(bsz, 1, dtype=torch.bool, device=device)], dim=1)  # boards and the global slot
    glob = torch.full_like(with_global, geom.global_tile, dtype=torch.long)
    tiles[:, : n_max + 1, n_max] = torch.where(with_global, glob, tiles[:, : n_max + 1, n_max])
    tiles[:, n_max, : n_max + 1] = torch.where(with_global, glob, tiles[:, n_max, : n_max + 1])
    far_cells = None
    if far is not None:
        # The per-axis terms are a function of the board pair alone: two valid boards beyond the
        # exact classes; the global-register and padding slots take the zero cell.
        board_cells = far_cell(dp, dh, far)
        far_cells = torch.full((bsz, n_max + 2, n_max + 2), far.zero_cell, dtype=torch.long, device=device)
        far_cells[:, :n_max, :n_max] = torch.where(both, board_cells, torch.full_like(board_cells, far.zero_cell))
    return DisplacementIndex(slot, code, tiles, kind, coords, far_cells)


class DisplacementBias(nn.Module):
    """B15's table, ``(layers, n_heads, n_entries)``, its static class map, and - with
    ``NetConfig.disp_far_axis`` - B6's two per-axis tables for the pairs beyond the exact classes,
    ``far_time`` ``(layers, n_heads, n_time)`` and ``far_timeline`` ``(layers, n_heads,
    n_timeline)``. ``layers`` is ``n_layers``, or 1 under ablation 2's layer-shared arm
    (``NetConfig.disp_layer_shared``: every layer reads the one table - ``per_layer`` False).

    Constructed without drawing from any RNG; ``model._init_weights`` draws the table from
    B15's own generator (``model.b15_seed``) and the per-axis tables from a second one
    (``model.b15_far_seed``)."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.geom = DisplacementGeometry.from_config(cfg)
        self.per_layer = not cfg.disp_layer_shared
        layers = cfg.n_layers if self.per_layer else 1
        self.table = nn.Parameter(torch.empty(layers, cfg.n_heads, self.geom.n_entries))
        self.far = FarAxes.from_config(cfg) if cfg.disp_far_axis else None
        if self.far is not None:
            self.far_time = nn.Parameter(torch.empty(layers, cfg.n_heads, self.far.n_time))
            self.far_timeline = nn.Parameter(torch.empty(layers, cfg.n_heads, self.far.n_timeline))
        else:  # B15 as built before the 2026-11-15 review: no per-axis table, no parameter
            self.register_parameter("far_time", None)
            self.register_parameter("far_timeline", None)
        self.register_buffer("class_map", _shared_class_map(self.geom), persistent=False)

    def layer_table(self, layer: int) -> torch.Tensor:
        """``(n_heads, n_entries + 1)``: layer ``layer``'s scalars (the shared table's, whatever the
        layer, under the layer-shared arm), FP32, with the zero entry (padding) appended."""
        t = self.table[layer if self.per_layer else 0].float()
        return torch.cat([t, t.new_zeros(t.shape[0], 1)], dim=1)

    def layer_far_table(self, layer: int) -> torch.Tensor | None:
        """``(n_heads, n_cells + 1)``: layer ``layer``'s per-axis term of every far cell - the time
        table's scalar plus the timeline table's, ``far_time[t] + far_timeline[l]`` at cell ``t x
        n_timeline + l`` - FP32, with the zero cell appended; None without the per-axis tables."""
        if self.far is None:
            return None
        i = layer if self.per_layer else 0
        cells = self.far_time[i].float()[:, :, None] + self.far_timeline[i].float()[:, None, :]
        cells = cells.reshape(cells.shape[0], self.far.n_cells)
        return torch.cat([cells, cells.new_zeros(cells.shape[0], 1)], dim=1)


__all__ = [
    "BOARD_DIRECTIONS",
    "DisplacementBias",
    "DisplacementGeometry",
    "DisplacementIndex",
    "FarAxes",
    "PairSides",
    "SPATIAL_DIRECTIONS",
    "board_class",
    "class_map",
    "displacement_index",
    "entry_from_class",
    "far_cell",
    "far_time_bucket",
    "far_timeline_bucket",
    "line_entry",
    "lookup",
    "lookup_sides",
    "pair_entry",
    "pair_far_cell",
    "pair_sides",
    "tile_index",
]
