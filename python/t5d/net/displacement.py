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
  here. Under it B6 is not B15's special case, as section 4.1 says it is: each far class
  mixes B6's Delta-time and Delta-timeline buckets, so beyond the exact box the far classes
  hold a B6 bias only where it depends on nothing but ``dh``'s sign - its clamped
  extrapolation survives, its per-axis buckets do not (issue 205).

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
displacements are all board-axis lines.

**Two implementations of one function**, compared by the tests and by section 15.2's B15 row:

* ``pair_entry`` - the entry of a pair computed from its tokens' kinds and coordinates by
  integer arithmetic: B14's score function (``flex.py``) evaluates it inside the attention
  "at score time from coordinates", per pair, and indexes the layer's compact table;
* the CPU paths' lookup (``AttnStructure`` in ``structure.py``): a board pair's *tile*
  (``tile_index``: its class, or for a far pair that can hold lines its board direction and
  step) and each token's *code* (a square ``y * 8 + x``, a per-board register 64) index a
  static ``class_map`` ``(tiles, 65, 65)``, built once from ``pair_entry``; then the layer's
  table. A tile stands for every board pair beyond ``tile_cap`` steps along its direction.

**Initialisation** (placeholder, ``NetConfig.disp_init_std``; spec issue 208): ``N(0, 0.1)``,
drawn from a generator of B15's own seeded from the model's seed (``model.b15_seed``, as D8's
are), never from the global stream - so the no-table arm has every other weight of the v1.
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


def check_board_sizes(batch: Batch) -> None:
    """B15's slice spans boards up to 8 x 8 (section 4.1: "15 x 15")."""
    for lay in batch.layouts:
        if lay.width > MAX_BOARD or lay.height > MAX_BOARD:
            raise ValueError(
                f"B15's 15 x 15 file-rank slice spans boards up to {MAX_BOARD} x {MAX_BOARD}; this batch has a "
                f"{lay.width} x {lay.height} board (no variant has one - the slice would need widening)"
            )


def displacement_index(batch: Batch, geom: DisplacementGeometry) -> DisplacementIndex:
    """Each token's slot and code, and each board pair's tile, from the batch's raw coordinates
    (only differences enter, so neither recentring nor a common shift changes anything)."""
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
    board_tiles = tile_index(p[:, None, :] - p[:, :, None], h[:, None, :] - h[:, :, None], geom)
    tiles = torch.full((bsz, n_max + 2, n_max + 2), geom.zero_tile, dtype=torch.long, device=device)
    valid_boards = batch.board_valid.to(device)
    both = valid_boards[:, :, None] & valid_boards[:, None, :]
    tiles[:, :n_max, :n_max] = torch.where(both, board_tiles, torch.full_like(board_tiles, geom.zero_tile))
    with_global = torch.cat([valid_boards, torch.ones(bsz, 1, dtype=torch.bool, device=device)], dim=1)  # boards and the global slot
    glob = torch.full_like(with_global, geom.global_tile, dtype=torch.long)
    tiles[:, : n_max + 1, n_max] = torch.where(with_global, glob, tiles[:, : n_max + 1, n_max])
    tiles[:, n_max, : n_max + 1] = torch.where(with_global, glob, tiles[:, n_max, : n_max + 1])
    return DisplacementIndex(slot, code, tiles, kind, coords)


class DisplacementBias(nn.Module):
    """B15's table, ``(n_layers, n_heads, n_entries)``, and its static class map.

    Constructed without drawing from any RNG; ``model._init_weights`` draws the table from
    B15's own generator (``model.b15_seed``)."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.geom = DisplacementGeometry.from_config(cfg)
        self.table = nn.Parameter(torch.empty(cfg.n_layers, cfg.n_heads, self.geom.n_entries))
        self.register_buffer("class_map", _shared_class_map(self.geom), persistent=False)

    def layer_table(self, layer: int) -> torch.Tensor:
        """``(n_heads, n_entries + 1)``: layer ``layer``'s scalars, FP32, with the zero entry
        (padding) appended."""
        t = self.table[layer].float()
        return torch.cat([t, t.new_zeros(t.shape[0], 1)], dim=1)


__all__ = [
    "BOARD_DIRECTIONS",
    "DisplacementBias",
    "DisplacementGeometry",
    "DisplacementIndex",
    "SPATIAL_DIRECTIONS",
    "board_class",
    "class_map",
    "displacement_index",
    "entry_from_class",
    "line_entry",
    "pair_entry",
    "tile_index",
]
