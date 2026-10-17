"""The block-causal attention scope (stage N3; spec section 7.3; register C17, C19, C20).

Under ``NetConfig.attn_scope="block_causal"`` who may read whom is decided per board,
for one position (``board_scope``), and then per token (``BlockCausalScope``):

* **Within a board: bidirectional.** All of a board's tokens - its squares and its
  per-board registers - read each other ("all of a board's tokens are created together,
  so full within-board attention is compatible with freezing"; per-board registers
  "belong to their board's block: frozen with it, and a cheap summary key for live
  tokens").
* **A frozen (superseded) board** reads the frozen boards of its causal past
  (``causal_past``), never a live head ("superseded boards attend only to their own past,
  never to live heads") and never a global register ("being read by frozen tokens =
  present-to-frozen flow, allowed only at refreshes", C17). Under
  ``supersession_scope="past_and_epoch_history"`` (C20's second reading; see
  ``NetConfig.supersession_scope``) the epoch's history - the boards frozen at its anchor -
  is a block fixed for the epoch: a history board reads the history boards of its past
  only, and a board frozen after the anchor reads its frozen past and every history board.
  (Were a history board to read its whole frozen past instead, the creation reading would
  close a cycle: a root head superseded in the search, created before a history board,
  would read that history board and be read by it.)
* **The live set** - the head boards - reads every board and the global registers; the
  heads read each other in both directions (C19: "every head attends to every head").
* **Global registers** read everything ("reading is free") and are read by live tokens.

Under ``causal_past="lineage"`` a frozen board's keys are boards that were already
superseded when it was created, so its key set never changes. Under ``"creation"`` an
earlier-created board that is still a head joins a frozen board's past only once it is
superseded (live heads are never read by frozen tokens): that changes the frozen board's
past, and the cache recomputes it (``cache.py``).

**The causal past** (spec issue 5 - the spec did not define it; ``NetConfig.causal_past``;
since ruled by register C23: the board's ancestry, which is ``"lineage"``):

* ``"lineage"`` - arrival-side lineage: the boards a board was made from by content - the
  transitive closure of ``PositionState.parent``: its timeline's earlier boards and,
  through its branch origin, the origin's own lineage. The board a travelling piece
  departed from is not in it. Every such board is superseded and stays superseded, so a
  frozen board's past never changes.
* ``"two_parent"`` - the same closure over two parents: a travel arrival also descends
  from the board its piece departed from (``PositionState.source``) and that board's
  lineage. Also fixed at creation and all superseded (the jump supersedes its departure
  board). A candidate reading for the user's ruling on spec issue 5 - not the one C23 made.
* ``"creation"``: every board created before it (``PositionState.creation``).

**The section 6 structure under this scope - a forced reading, documented here.** The
move graph (B9, the routing table's relations head, B10) and the typed relations (B11)
are generated for each position by the rules engine: they are present information, and
so are board-pair smolgen's biases (B12), generated from the global registers. A frozen
token whose mask or bias depended on them would read the present through the structure
- its K/V would change from node to node. So under the block-causal scope these open and
bias **live query rows only** (``live_tokens``): a frozen square's row on a move-graph or
relations head holds itself alone, and carries no typed or smolgen bias. Rows that are
pure geometry stay as they are for every token: the same-board and same-square heads
(B10) and B15's displacement bias, a function of the two tokens' coordinates alone
(section 4.1: "computed at score time from coordinates, so nothing cached goes stale").
Section 6's own remark - "under the cached configuration
the bias rows for frozen boards freeze with them" - is consistent with this: in the rules
engine's export (stage E3) every move and relation starts on a playable board, so a
frozen row's edges lead to live keys, which the scope already closes.

**B16's status key biases** (``status.py``) follow the same rule from the key's side: a
frozen board's current status (Delta-t-to-present bin, active flag) biases the live query
rows that read its tokens, recomputed every evaluation, and never a frozen row - whose input
carries intrinsic features only (``frozen_features="intrinsic"``, M20).

Padding queries read every valid key (their outputs are discarded), so no row of the
scope is ever empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import NetConfig
from .state import TOKEN_GLOBAL_REGISTER, Batch, PositionState


def lineage_matrix(parent: np.ndarray, source: np.ndarray | None = None) -> np.ndarray:
    """``(N, N)`` bool: ``[i, j]`` - board ``j`` is a proper ancestor of board ``i`` through
    ``parent`` and, if given, ``source`` (a travel arrival's departure board; -1: none).
    Raises on a cycle."""
    parent = np.asarray(parent, dtype=np.int64)
    n = parent.shape[0]
    source = np.full(n, -1, dtype=np.int64) if source is None else np.asarray(source, dtype=np.int64)
    parents = [[int(a) for a in (parent[i], source[i]) if a >= 0] for i in range(n)]
    anc = np.zeros((n, n), dtype=bool)
    state = np.zeros(n, dtype=np.int8)  # 0 new, 1 on the stack, 2 done
    for start in range(n):
        if state[start]:
            continue
        stack = [(start, 0)]
        state[start] = 1
        while stack:
            i, k = stack[-1]
            if k < len(parents[i]):
                stack[-1] = (i, k + 1)
                a = parents[i][k]
                if state[a] == 1:
                    raise ValueError("parent/source: the lineage has a cycle")
                if state[a] == 0:
                    state[a] = 1
                    stack.append((a, 0))
                continue
            for a in parents[i]:
                anc[i] |= anc[a]
                anc[i, a] = True
            state[i] = 2
            stack.pop()
    return anc


def creation_matrix(creation: np.ndarray) -> np.ndarray:
    """``(N, N)`` bool: ``[i, j]`` - board ``j`` was created before board ``i``."""
    creation = np.asarray(creation, dtype=np.int64)
    return creation[None, :] < creation[:, None]


def causal_past(parent: np.ndarray, creation: np.ndarray | None, reading: str, source: np.ndarray | None = None) -> np.ndarray:
    """``(N, N)`` bool: ``[i, j]`` - ``j`` is in board ``i``'s causal past (never ``i`` itself)."""
    if reading == "lineage":
        return lineage_matrix(parent)
    if reading == "two_parent":
        return lineage_matrix(parent, source)
    if reading == "creation":
        if creation is None:
            raise ValueError("causal_past='creation' needs the boards' creation order (PositionState.creation)")
        return creation_matrix(creation)
    raise ValueError(f"unknown causal-past reading {reading!r}")


def board_scope(
    parent: np.ndarray,
    creation: np.ndarray | None,
    is_head: np.ndarray,
    reading: str,
    supersession: str = "causal_past",
    history: np.ndarray | None = None,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """``(N, N)`` bool: ``[i, j]`` - board ``i``'s tokens may read board ``j``'s tokens.

    A frozen board reads itself and the frozen boards of its causal past. Under
    ``supersession="past_and_epoch_history"`` with a ``history`` (its epoch's history: the
    boards frozen at the epoch's anchor), a history board reads the history boards of its
    past only, and a frozen board outside the history reads its frozen past and every
    history board. A head (live) board reads every board. ``history`` None: the position is
    its own anchor - every frozen board is history, and both readings of C20 coincide."""
    is_head = np.asarray(is_head, dtype=bool)
    frozen = ~is_head
    past = causal_past(parent, creation, reading, source)
    allowed = past & frozen[None, :]
    if supersession == "past_and_epoch_history" and history is not None:
        history = np.asarray(history, dtype=bool)
        if np.any(history & is_head):
            raise ValueError("an epoch's history holds frozen boards only")
        allowed[history] = past[history] & history[None, :]
        allowed |= (frozen & ~history)[:, None] & history[None, :]
    elif supersession not in ("causal_past", "past_and_epoch_history"):
        raise ValueError(f"unknown supersession scope {supersession!r}")
    allowed[is_head] = True
    np.fill_diagonal(allowed, True)
    return allowed


def state_scope(state: PositionState, cfg: NetConfig, history: np.ndarray | None = None) -> np.ndarray:
    """``board_scope`` of one position under ``cfg``'s readings."""
    return board_scope(state.parent, state.creation, state.is_head, cfg.causal_past, cfg.supersession_scope, history, state.source)


@dataclass
class BlockCausalScope:
    """The block-causal scope of a batch, rows built on demand (like ``AttnStructure``).

    Every token has a *slot*: its board's index, ``n_boards`` for a global register, or
    ``n_boards + 1`` for padding; ``allowed`` ``(B, S, S)`` says which slot reads which,
    so a block of query rows costs ``O(rows x L)`` and no ``(L, L)`` tensor exists."""

    slot: torch.Tensor  # (B, L) long
    allowed: torch.Tensor  # (B, S, S) bool: query slot -> key slot
    live_tokens: torch.Tensor  # (B, L) bool: tokens of head boards, and global registers
    live_boards: torch.Tensor  # (B, Nb) bool: head boards

    @classmethod
    def build(cls, batch: Batch, cfg: NetConfig, history: torch.Tensor | None = None) -> BlockCausalScope:
        """The scope of every position of ``batch`` (``history``: ``(B, Nmax)`` bool, each
        position's epoch history - see ``board_scope``; None: each its own anchor)."""
        bsz, length = batch.kind.shape
        nb = batch.board_valid.shape[1]
        slots = nb + 2
        allowed = torch.zeros(bsz, slots, slots, dtype=torch.bool)
        for i in range(bsz):
            n = int(batch.n_boards[i])
            creation = batch.board_creation[i, :n].numpy() if bool(batch.has_creation[i]) else None
            hist = None if history is None else history[i, :n].cpu().numpy()
            head = batch.board_head[i, :n].numpy()
            source = batch.board_source[i, :n].numpy()
            scope = board_scope(batch.board_parent[i, :n].numpy(), creation, head, cfg.causal_past, cfg.supersession_scope, hist, source)
            allowed[i, :n, :n] = torch.from_numpy(scope)
            allowed[i, :n, nb] = torch.from_numpy(head)  # the live set reads the global registers
            allowed[i, nb, :n] = True  # global registers read everything
            allowed[i, nb, nb] = True
            allowed[i, nb + 1, :n] = True  # padding: every valid key (discarded)
            allowed[i, nb + 1, nb] = True
        on_board = batch.valid & (batch.board >= 0)
        is_global = batch.valid & (batch.kind == TOKEN_GLOBAL_REGISTER)
        slot = torch.where(
            on_board, batch.board, torch.where(is_global, torch.full_like(batch.board, nb), torch.full_like(batch.board, nb + 1))
        )
        live_boards = batch.board_valid & batch.board_head
        head_slot = torch.cat([live_boards, torch.ones(bsz, 1, dtype=torch.bool), torch.zeros(bsz, 1, dtype=torch.bool)], dim=1)
        live_tokens = head_slot.gather(1, slot) & batch.valid
        device = batch.kind.device
        return cls(slot.to(device), allowed.to(device), live_tokens.to(device), live_boards.to(device))

    def _rows_for(self, slot_q: torch.Tensor) -> torch.Tensor:
        bsz, n = slot_q.shape
        slots = self.allowed.shape[-1]
        length = self.slot.shape[1]
        rows = self.allowed.gather(1, slot_q[..., None].expand(bsz, n, slots))
        return rows.gather(2, self.slot[:, None, :].expand(bsz, n, length))[:, None]

    def rows(self, start: int, stop: int) -> torch.Tensor:
        """Allowed keys ``(B, 1, stop - start, L)`` of the queries ``[start, stop)``."""
        return self._rows_for(self.slot[:, start:stop])

    def rows_at(self, idx: torch.Tensor) -> torch.Tensor:
        """Allowed keys ``(B, 1, n, L)`` of the queries ``idx`` ``(B, n)``."""
        return self._rows_for(self.slot.gather(1, idx))

    def dense(self) -> torch.Tensor:
        """Every row ``(B, 1, L, L)`` (small inputs; tests)."""
        return self.rows(0, self.slot.shape[1])
