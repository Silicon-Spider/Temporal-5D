"""B16: score-time status key biases on frozen boards (committed with the cached configuration;
spec sections 2.3, 4.1, 7.3; register B16, A3, M20).

Section 4.1: "A3's Delta-t-to-present and the active/inactive flag describe a board's current
status, which changes after the board freezes (section 7.3). Under the cached configuration
frozen boards therefore carry only intrinsic features, and current status enters as score-time
key biases - per head, by the key board's Delta-t-to-present bin and timeline status -
recomputed every evaluation for the cost of a lookup. Live tokens keep A3's features as
inputs." M20 ruled spec issue 31 so: "frozen features -> intrinsic, with score-time status
biases (B16)" (``NetConfig.frozen_features``).

**Where it applies.** Under the block-causal scope (``attn_scope="block_causal"``, the cached
configuration) with intrinsic frozen features, on the score of every pair whose query is live -
a head board's token or a global register (``BlockCausalScope.live_tokens``) - and whose key is
a token of a frozen (superseded) board, its squares and its per-board registers alike: the
head's entry for the key board's status. Nowhere else:

* never on a frozen query row: a frozen board's outputs, and so its K/V, would then depend on
  the present, which C17 allows only at refreshes - so frozen rows are computed without it
  and stay valid between refreshes (a forced reading of "key biases"; spec issue 216);
* a live key carries its status as inputs (A3); global registers and padding have no board;
* not under the dense scope: a dense pass gives every board its A3 features as inputs. The
  refresh, a dense pass, stays as built - M20 ruled issue 31's option, "a refresh gives them
  the anchor's values" - so a refreshed epoch's history boards keep the anchor's features in
  their K/V, and B16 biases them at every node below the anchor with that node's own status.
  At the anchor itself the evaluation is the refresh pass (section 7.9: "at the root the
  two-level mask reduces to full dense attention"), so B16 adds nothing there
  (``T5DNet.make_context(status_bias=False)``, which ``cache.py`` passes for the anchor's own
  state in a refreshed epoch; spec issue 216);
* not in a network of the dense scope, which has no table - and a block-causal forward of such
  a network, which would strip its frozen boards' status from their inputs with nothing in its
  place, is refused while ``status_key_bias`` is on (``T5DNet.make_context``).

**A key board's status**: its Delta-t-to-present ``h - h_present`` in half-turns, binned, and
its timeline's active/inactive flag (A3's) - ``bin * 2 + active``. Head/superseded is no status
here: every keyed board is superseded. COMMITTED-CLASS PLACEHOLDERS (section 4.1 gives none of
it; spec issues 213-215):

* the bins (``status_dt_exact``, ``status_dt_max``): signed log2 buckets, exact below 4
  half-turns, one per doubling up to 128, then clamped, past and future apart - 19 bins, the
  time buckets B6 used and smolgen keeps (``buckets.signed_log_bucket``);
* one table shared by all layers (``status_bias_per_layer=False``), as B11's scalars are: 8
  heads x 19 bins x 2 = 304 parameters (4,864 with one table per layer, 16 x 304);
* ``N(0, status_bias_init_std = 0.1)``, the scale of B11's and B15's tables, drawn from a
  generator of B16's own (``model.b16_seed``): switching B16 changes no other weight.

**One function on every path.** A pair's term is a lookup through a per-token index
(``status_index``): ``AttnStructure`` adds it to the bias rows of any query block, key
selection or row set (the CPU paths; ``structure.py``), B14's score function computes it inside
the attention (``flex.py``), and the dense reference materialises it. The cached path adds it in
its live pass, from the node's own present (``cache.py``); a frozen board's stored rows never
carry it. Section 15.2's row 1 (cached = dense) compares the two with it on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .buckets import n_signed_log_buckets, signed_log_bucket
from .config import NetConfig
from .state import Batch


class StatusKeyBias(nn.Module):
    """B16's table ``(layers, n_heads, n_status)`` - ``layers`` 1 when shared by every layer.

    Constructed without drawing from any RNG; ``model._init_weights`` draws the table from B16's
    own generator (``model.b16_seed``)."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.dt_exact, self.dt_max = int(cfg.status_dt_exact), int(cfg.status_dt_max)
        self.n_bins = n_signed_log_buckets(self.dt_exact, self.dt_max)
        self.n_status = 2 * self.n_bins
        self.per_layer = bool(cfg.status_bias_per_layer)
        self.table = nn.Parameter(torch.empty(cfg.n_layers if self.per_layer else 1, cfg.n_heads, self.n_status))

    def layer_table(self, layer: int) -> torch.Tensor:
        """``(n_heads, n_status + 1)``: layer ``layer``'s entries (the shared table's, whatever the
        layer, when it is shared), FP32, with the zero entry - no status - appended."""
        t = self.table[layer if self.per_layer else 0].float()
        return torch.cat([t, t.new_zeros(t.shape[0], 1)], dim=1)

    def status(self, dt: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """A board's status entry from its Delta-t-to-present (half-turns) and active flag:
        ``bin * 2 + active``."""
        bins = signed_log_bucket(dt.round().long(), self.dt_exact, self.dt_max)
        return bins * 2 + active.long().clamp(0, 1)


@dataclass
class StatusIndex:
    """What B16 needs of one batch, per token (``status_index``)."""

    key: torch.Tensor  # (B, L) long: a frozen board's token's status entry; n_status (none) elsewhere
    live: torch.Tensor  # (B, L) bool: the live query rows - head boards' tokens and global registers
    any_frozen: bool  # some key is a frozen board's token: else B16 adds nothing at all


def status_index(batch: Batch, status: StatusKeyBias, live: torch.Tensor) -> StatusIndex:
    """B16's per-token index of a batch under the block-causal scope: each frozen board's
    tokens' status from the batch's own Delta-t-to-present and active flags (the present of the
    position evaluated - recomputed every evaluation), and the live rows ``live`` ``(B, L)``."""
    frozen = batch.valid & (batch.board >= 0) & (batch.is_head == 0)
    key = torch.where(frozen, status.status(batch.dt, batch.active), torch.full_like(batch.kind, status.n_status))
    return StatusIndex(key=key, live=live & batch.valid, any_frozen=bool(frozen.any()))


__all__ = ["StatusIndex", "StatusKeyBias", "status_index"]
