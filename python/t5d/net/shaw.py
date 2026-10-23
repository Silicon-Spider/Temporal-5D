"""B17: the key-side Shaw relative-position term (prototype, behind its flag; spec section 4.1;
register B17; ablation 2).

Register B17: "Key-side Shaw relative-position term on top of B15: each query dotted with learned
vectors for a compact class set (~1,400: the exact near classes plus the line classes) - ~0.5x
QK^T at 44 boards, ~9% at 240; value-side vectors held open (custom kernel)". Section 4.1: Lc0's
relative-position experiment used Shaw-style representations, "learned vectors added to queries,
keys and values for each square pair, so the positional effect depends on content"; "in Shaw et
al.'s own ablation the key-side term alone matched the full version. So the scalar table is v1;
a key-side, compact-class Shaw term on top of it is a prototype arm of ablation 2 (B17), built
behind a flag and costed in the untrained build; value-side vectors are held open."

**The term.** Shaw et al. (2018, "Self-Attention with Relative Position Representations") add a
learned vector ``a_ij`` to the key: ``e_ij = q_i . (k_j + a_ij) / sqrt(d)``. The key-side term is
the second part, ``q_i . R[c(i, j)] / sqrt(d_head)``, with ``c(i, j)`` the pair's class in a compact
set and ``R`` a learned vector per class; it is added to the pair's score beside B15's scalar
(``NetConfig.shaw_key_term``, off by default; it needs ``displacement_bias``: "on top of B15").
Value-side vectors are not built (held open, section 14.4).

**The compact class set** (``compact_classes``) - "the exact near classes plus the line classes",
which section 4.1 does not define. COMMITTED-CLASS PLACEHOLDER (spec issue 217): B15's own
classes, restricted - inside B15's 45 exact board-distance classes (|Delta-timeline| <= 2,
|Delta-time| <= 4 half-turns) the file-rank offsets with |Delta-file|, |Delta-rank| <=
``shaw_near`` (2: 25 per class, 1,125 in all), plus every line class of B15's (368 at its
defaults: one per direction and step beyond the exact classes): 1,493 classes - section 4.1's
~1,400, and a reading that gives both of its cost figures (``cost.shaw_flops``: 0.51 x QK^T at
44 boards, 9.4% at 240). Every other pair - the far slices' cells, near-class offsets beyond the
window, and every pair with a register or a global register (no (file, rank), B8) - has no class
and gets no term (placeholder too). It is not the only such reading: the nearest board-distance
classes with their full 15 x 15 slices - say the same board and the four one timeline or one turn
away, 5 x 225 + 368 = 1,493 - give the same count and the same costs, and cover other pairs (every
relation on one board, which this window cuts at two squares; none two boards away). Issue 217
puts the two to the user; the other would be another ``compact_classes``, not a setting.

**The query** is the unrotated one - the projection's output, before RoPE: the class is a function
of the pair's displacement, so ``q_i . R[c]`` is relative; a rotated query would make it depend
on the query's absolute angle - an absolute positional encoding (section 14.1) - and on the
frame a computation centres its coordinates on (the cached path centres each frozen board on
itself). The scale is Shaw et al.'s ``1 / sqrt(d_head)``, the scores' own. Both are spec issue 219's;
the first is forced, the second a placeholder.

**Per head and per layer** (``shaw_shared_heads=False``; placeholder, spec issue 218): a vector per
class per head per layer, ``1,493 x 32 x 8 x 16 = 6,115,328`` parameters at the v1's dimensions -
26.6% of the v1's 22,973,907, 21.0% of the network with it; Shaw et al. share theirs across heads
(``shaw_shared_heads=True``: 764,416). No document states the count. **Initialisation** (placeholder, spec issue 220): ``N(0,
shaw_init_std = 0.1)`` from a generator of B17's own (``model.b17_seed``): at initialisation a
query's entries have unit spread, so the term starts with the spread of B15's scalars, ~0.1; the
flag changes no other weight.

**Two computations of one function**, section 15.2's B17 row ("Key-side Shaw term via precomputed
query-class products = direct per-pair computation, on small positions | The gather implementation
is exact"):

* ``gather_term`` - the precomputed query-class products: ``P = q R^T`` (every query against every
  class, plus a zero column for "no class"), then each pair's entry of ``P`` gathered by its class.
  ``T_q x C x d_head`` multiply-adds per head per layer - growing with T, not T^2. The CPU paths use
  it block by block (``attention.py``), the cached path for a frozen board's rows (``cache.py``),
  and B14's score function reads ``P`` inside the attention, the class computed there from the two
  tokens' coordinates (``flex.py``);
* ``direct_term`` - per pair, the query dotted with its pair's class vector. The dense reference
  (``attn_impl="dense"``, ``AttnContext.densified``) materialises the term this way.

The classes come from B15's per-batch index and a static map like B15's (``class_map``):
``B17 class = compact[B15 entry]``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace

import torch
from torch import nn

from .config import NetConfig
from .displacement import N_EXACT, SIDE, SLICE, SPAN, DisplacementGeometry, DisplacementIndex, _shared_class_map, lookup


@functools.lru_cache(maxsize=8)
def compact_classes(geom: DisplacementGeometry, near: int) -> tuple[torch.Tensor, int]:
    """``(compact, C)``: ``compact`` ``(n_entries + 1,)`` long maps each of B15's entries (and its
    padding entry) to its B17 class - the exact classes' cells with |Delta-file|, |Delta-rank| <=
    ``near`` (class ``bc (2 near + 1)^2 + (dx + near)(2 near + 1) + (dy + near)``), then the line
    entries in B15's order - or to ``C``, "no class"; ``C`` the class count."""
    width = 2 * near + 1
    compact = torch.full((geom.n_entries + 1,), -1, dtype=torch.long)
    offsets = torch.arange(-near, near + 1)
    dx, dy = offsets[:, None].expand(width, width).reshape(-1), offsets[None, :].expand(width, width).reshape(-1)
    for bc in range(N_EXACT):
        cells = bc * SLICE + (dx + SPAN) * SIDE + (dy + SPAN)
        compact[cells] = bc * width * width + (dx + near) * width + (dy + near)
    n_near = N_EXACT * width * width
    lines = torch.arange(geom.lines_offset, geom.registers_offset)
    compact[lines] = n_near + (lines - geom.lines_offset)
    n_classes = n_near + lines.numel()
    compact[compact < 0] = n_classes
    return compact, n_classes


def n_shaw_classes(cfg: NetConfig) -> int:
    """B17's class count under ``cfg`` (1,493 at the defaults)."""
    return compact_classes(DisplacementGeometry.from_config(cfg), int(cfg.shaw_near))[1]


class ShawKeyTerm(nn.Module):
    """B17's vectors ``(n_layers, heads, C, d_head)`` - ``heads`` 1 when shared by the heads - and
    its static maps. Constructed without drawing from any RNG; ``model._init_weights`` draws the
    vectors from B17's own generator (``model.b17_seed``)."""

    def __init__(self, cfg: NetConfig, geom: DisplacementGeometry) -> None:
        super().__init__()
        compact, n_classes = compact_classes(geom, int(cfg.shaw_near))
        self.n_classes = n_classes
        self.scale = cfg.d_head**-0.5
        heads = 1 if cfg.shaw_shared_heads else cfg.n_heads
        self.vectors = nn.Parameter(torch.empty(cfg.n_layers, heads, n_classes, cfg.d_head))
        #: B15's entry -> B17's class (``n_classes``: none).
        self.register_buffer("compact", compact.clone(), persistent=False)
        #: B17's class of every (tile, query code, key code): B15's class map through ``compact``.
        self.register_buffer("class_map", compact[_shared_class_map(geom)], persistent=False)

    def layer_vectors(self, layer: int) -> torch.Tensor:
        """``(heads, C + 1, d_head)``: layer ``layer``'s vectors, FP32, with a zero vector - no
        class - appended."""
        v = self.vectors[layer].float()
        return torch.cat([v, v.new_zeros(v.shape[0], 1, v.shape[2])], dim=1)


def _no_autocast(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


def gather_term(q: torch.Tensor, vectors: torch.Tensor, classes: torch.Tensor, scale: float) -> torch.Tensor:
    """B17's term ``(B, Hg, n, Lk)`` through the precomputed query-class products: ``P = q R^T``
    ``(B, Hg, n, C + 1)``, then each pair's entry gathered by its class. ``q`` ``(B, Hg, n, d)``
    unrotated queries; ``vectors`` ``(Hg or 1, C + 1, d)`` with the zero vector last; ``classes``
    ``(B, n, Lk)``. At least FP32, outside autocast."""
    dtype = torch.promote_types(q.dtype, torch.float32)
    with _no_autocast(q.device):
        p = torch.matmul(q.to(dtype), vectors.to(dtype).transpose(-1, -2)) * scale
    bsz, hg, n, _ = p.shape
    return p.gather(3, classes[:, None].expand(bsz, hg, n, classes.shape[-1]))


#: Per-pair elements ``direct_term`` holds at a time (a head and a chunk of rows: its memory bound).
DIRECT_CHUNK = 1 << 22


def direct_term(q: torch.Tensor, vectors: torch.Tensor, classes: torch.Tensor, scale: float) -> torch.Tensor:
    """B17's term ``(B, Hg, n, Lk)`` pair by pair: each query dotted with its pair's class vector
    (``gather_term``'s arguments) - the per-pair computation of section 15.2's B17 row. It holds a
    vector per pair, so it runs head by head over chunks of rows (small inputs: the reference)."""
    dtype = torch.promote_types(q.dtype, torch.float32)
    bsz, n_heads, n, d = q.shape
    step = max(1, DIRECT_CHUNK // max(1, bsz * classes.shape[-1] * d))
    with _no_autocast(q.device):
        v = vectors.to(dtype)
        heads = []
        for h in range(n_heads):
            vh = v[0 if v.shape[0] == 1 else h]  # (C + 1, d)
            qh = q[:, h].to(dtype)  # (B, n, d)
            parts = [(qh[:, s : s + step, None, :] * vh[classes[:, s : s + step]]).sum(dim=-1) for s in range(0, n, step)]
            heads.append(torch.cat(parts, dim=1))
        return torch.stack(heads, dim=1) * scale


@dataclass
class ShawRows:
    """B17 for one batch: its pairs' classes (from B15's per-batch index) and one layer's vectors
    (``for_layer``). ``direct``: the per-pair computation (the dense reference's), else the
    gather."""

    term: ShawKeyTerm
    index: DisplacementIndex
    vectors: torch.Tensor | None = None  # (heads, C + 1, d_head) FP32: this layer's
    direct: bool = False

    def for_layer(self, layer: int) -> ShawRows:
        return replace(self, vectors=self.term.layer_vectors(layer))

    def classes(self, rows: torch.Tensor, cols: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, n, Lk)``: the class of every pair of the queries ``rows`` ``(B, n)`` and the keys
        ``cols`` ``(B, K)`` (every key when None); ``C`` where the pair has none."""
        return lookup(self.index, self.term.class_map, rows, cols)

    def heads(self, heads: torch.Tensor | None) -> torch.Tensor:
        """This layer's vectors on ``heads`` (every head when None; shared vectors serve all)."""
        if self.vectors is None:
            raise RuntimeError("B17's vectors are per layer: use AttnContext.for_layer to choose the layer")
        return self.vectors if heads is None or self.vectors.shape[0] == 1 else self.vectors[heads]

    def rows(
        self, q: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor | None = None, heads: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The term ``(B, Hg, n, Lk)`` of the queries ``rows`` ``(B, n)`` - ``q`` ``(B, Hg, n, d)``
        their unrotated queries on ``heads`` - over the keys ``cols`` (every key when None)."""
        compute = direct_term if self.direct else gather_term
        return compute(q, self.heads(heads), self.classes(rows, cols), self.term.scale)

    def products(self, q: torch.Tensor) -> torch.Tensor:
        """``P`` ``(B, H, L, C + 1)``: every query of ``q`` ``(B, H, L, d)`` (unrotated) against
        every class of this layer, the last column zero - what B14's score function reads."""
        vectors = self.heads(None)
        dtype = torch.promote_types(q.dtype, torch.float32)
        with _no_autocast(q.device):
            return torch.matmul(q.to(dtype), vectors.to(dtype).transpose(-1, -2)) * self.term.scale


__all__ = ["ShawKeyTerm", "ShawRows", "compact_classes", "direct_term", "gather_term", "n_shaw_classes"]
