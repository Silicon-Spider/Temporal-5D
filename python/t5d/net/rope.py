"""4D rotary position encoding (B4) and per-head specialisation (B5).

**B4 (committed), spec section 4.1.** Each head's ``d_head`` dims form ``d_head/2``
rotary pairs. Pair ``k`` rotates dims ``(k, k + d_head/2)`` (the "rotate-half" layout)
by ``coordinate[axis_k] * freq_k``. The default allocation over 16 pairs is file 3,
rank 3, timeline 4, time 6 - asymmetric, sized by coordinate range ("files span 0-7
while time spans 0-40+"). Coordinates: file ``x``, rank ``y``, timeline ``p``, time
``h`` (half-turns). Rotation is applied to Q and K before the attention kernel, in FP32
(G1), so the kernel itself is stock.

**Frequencies (implementation choice; the spec is silent).** Standard RoPE's ladder
``freq_k = base^(-k/n)`` (k = 0..n-1) per axis, with the base sized to the axis's range:
the highest frequency is ``rope_max_freq`` (1 rad per unit, as in standard RoPE) and the
lowest is ``pi / range``, so that every offset within the range maps to a distinct
angle without aliasing. That gives ``base = (max_freq * range / pi)^(n / (n - 1))``:

======== ===== ======= ===============================================
axis     pairs range   frequencies (rad / unit)
======== ===== ======= ===============================================
file     3     8       1.000, 0.627, 0.393
rank     3     8       1.000, 0.627, 0.393
timeline 4     64      1.000, 0.366, 0.134, 0.049
time     6     256     1.000, 0.415, 0.172, 0.071, 0.030, 0.012
======== ===== ======= ===============================================

**Tokens without a coordinate (section 5.3; open committed-class points, see
``NetConfig``).** Under a relative encoding a fixed angle on one side of a pair is itself
a position: an unrotated token sits at coordinate 0 of whatever frame the other side is
rotated in. So a token that must have *no* coordinate on an axis gets that axis's pairs
unrotated on *both* sides of every pair it is in (exact NoPE for that part of the pair;
``attention.py`` applies the rule). Per-board registers carry their board's ``(p, h)``
on the timeline/time pairs; on the spatial pairs ``board_register_spatial`` selects a
learned off-lattice coordinate (``learned_angle``), no coordinate (``position_free``) or
angle 0 (``zero_angle`` - square a1's position). Global registers hold no board:
``global_register_pairs`` makes every pair with one position-free (``position_free``),
or places them at recentred coordinate 0 (``anchored``: the anchor ``(a1, p_ref,
h_present)`` moves with the position, so this stays translation-invariant; with raw
coordinates it would be the absolute origin - an absolute positional encoding, which
section 14.1 rejects, hence ``anchored`` requires recentring).

**B5 (provisional, flag ``rope_per_head``).** Allocation per head role, following
section 6's routing table: the same-board head spends all its pairs on file/rank (8 + 8),
the same-square-across-boards head all on timeline/time (``rope_same_square_split``),
the move-graph head runs near-NoPE (every pair rotates by at most
``rope_near_nope_max_angle`` over its axis's whole range), dense heads keep the B4
split. The relations head is a parameter (``rope_relations_head``).

How B5 meets the masks of stage N2 (``NetConfig.rope_roles``): one role table,
``rope_head_roles``, serves both B5 (rotation) and B10 (masks), so under the routing
table every head's rotation and mask come from the same entry. With the routing table
off, B9 puts the move-graph mask on heads 0..k-1 and ``rope_roles_source`` (an open
committed-class point) decides the rotation: "table" keeps N1's roles by head index,
"masks" gives each head the role of the mask it carries (near-NoPE on the move-graph
heads, the B4 split elsewhere).

**RoPE's positional term factorises per axis** (register B4): no rotation can favour the
displacements (1, 1) and (2, 2) while disfavouring (1, 2) and (2, 1), so diagonals and knight
moves are inexpressible as position. That joint-axis geometry is B15's - the 4D displacement
bias beside RoPE (``displacement.py``), which superseded B6's binned Delta-time/Delta-timeline
biases (CHECKPOINT 6; B6 and its "RoPE for near-time" arm are removed).
"""

from __future__ import annotations

import math

import torch

from .config import NetConfig

AXIS_FILE, AXIS_RANK, AXIS_TIMELINE, AXIS_TIME = range(4)


def axis_frequencies(n_pairs: int, axis_range: float, max_freq: float = 1.0) -> list[float]:
    """Standard RoPE ladder ``max_freq * base^(-k/n)`` with its lowest rung at ``pi/range``."""
    if n_pairs <= 0:
        return []
    min_freq = math.pi / axis_range
    if n_pairs == 1:
        return [min_freq]
    if min_freq >= max_freq:
        return [max_freq] * n_pairs
    base = (max_freq / min_freq) ** (n_pairs / (n_pairs - 1))
    return [max_freq * base ** (-k / n_pairs) for k in range(n_pairs)]


def _split_allocation(pairs: tuple[int, ...], axes: tuple[int, ...], cfg: NetConfig) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for n, axis in zip(pairs, axes, strict=True):
        out += [(axis, f) for f in axis_frequencies(n, cfg.rope_ranges[axis], cfg.rope_max_freq)]
    return out


def head_allocation(role: str, cfg: NetConfig) -> list[tuple[int, float]]:
    """``(axis, frequency)`` of each rotary pair of a head with the given role."""
    dense = _split_allocation(cfg.rope_pairs, (0, 1, 2, 3), cfg)
    if role == "dense" or (role == "relations" and cfg.rope_relations_head == "dense"):
        alloc = dense
    elif role == "same_board":
        half = cfg.n_pairs // 2
        alloc = _split_allocation((half, cfg.n_pairs - half), (AXIS_FILE, AXIS_RANK), cfg)
    elif role == "same_square":
        alloc = _split_allocation(tuple(cfg.rope_same_square_split), (AXIS_TIMELINE, AXIS_TIME), cfg)
    elif role in ("move_graph", "relations"):  # near-NoPE
        alloc = [(axis, cfg.rope_near_nope_max_angle / cfg.rope_ranges[axis]) for axis, _ in dense]
    else:
        raise ValueError(f"unknown head role {role!r}")
    assert len(alloc) == cfg.n_pairs
    return alloc


def rope_tables(cfg: NetConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head pair tables: ``axis`` (Hr, P) long and ``freq`` (Hr, P) float32.

    ``Hr`` is ``n_heads`` under B5 and 1 otherwise (one allocation shared by all heads).
    Under B5 each head's role is ``cfg.rope_roles`` (see the module docstring).
    """
    roles = cfg.rope_roles
    allocs = [head_allocation(role, cfg) for role in roles]
    axis = torch.tensor([[a for a, _ in alloc] for alloc in allocs], dtype=torch.long)
    freq = torch.tensor([[f for _, f in alloc] for alloc in allocs], dtype=torch.float32)
    return axis, freq


def spatial_dims(axis: torch.Tensor) -> torch.Tensor:
    """``(Hr, d_head)`` bool: the dims of each head's file and rank rotary pairs.

    Pair ``k`` occupies dims ``k`` and ``k + d_head/2`` (rotate-half layout).
    """
    spatial = (axis == AXIS_FILE) | (axis == AXIS_RANK)
    return torch.cat([spatial, spatial], dim=-1)


def rope_angles(coords: torch.Tensor, axis: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    """Angles ``(B, Hr, L, P)`` in FP32 from coordinates ``(B, L, 4)``."""
    with torch.autocast(device_type=coords.device.type, enabled=False):
        coords = coords.float()
        bsz, length, _ = coords.shape
        hr, p = axis.shape
        picked = coords[:, None, :, :].expand(bsz, hr, length, 4).gather(-1, axis[None, :, None, :].expand(bsz, hr, length, p))
        return picked * freq[None, :, None, :].to(coords.device)


def rope_cos_sin(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin, each ``(B, Hr, L, d_head)`` (duplicated for the rotate-half layout)."""
    with torch.autocast(device_type=angles.device.type, enabled=False):
        cos, sin = torch.cos(angles), torch.sin(angles)
        return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``(B, H, L, d_head)`` in FP32; ``cos``/``sin`` broadcast over heads."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin
