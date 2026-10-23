"""Input embeddings (A2, A3, B8; spec sections 2.2, 2.3, 5.3).

Square tokens: learned embeddings of piece type (13 incl. empty), owner (none, ours,
theirs - relative to the acting player, register A7) and the has-moved flag, summed (A2).
Coordinates are not embedded here: they reach attention through the 4D RoPE (B4), in the
acting player's frame (A7: ranks mirrored and timelines negated when Black acts -
``state.py``).

Every token of a board - its squares and its per-board registers - additionally gets
the board's flag embeddings (head/superseded, active/inactive timeline, side to move - us
or them, relative to the acting player, A7; section 2.3: "added to every token on the board") and its Delta-t-to-present embedding
(A3). Delta-t is continuous: a small learned MLP (GELU) over the features
``(dt / dt_scale, sign(dt) * log1p|dt|, [dt == 0])``, so the binary "present" flag is
kept as Delta-t's zero (section 2.3) while the distance to the present stays readable
at any range (the spec says only "embedded per board": the features and the width are
implementation choices). F6's Mish is the activation of the trunk FFN and of every head's
hidden layers (CHECKPOINT 8); this MLP is neither, and F6 does not name it, so it keeps stage
N1's GELU - a placeholder fixed in code: the activation is part of the input encoding,
committed-class and open (spec issue 221).

Register tokens get a register-type embedding (per-board slot 0..R-1, or global index
0..G-1) plus one learned "not a square" embedding (section 5.3) instead of piece/colour/
has-moved. Global registers hold no board, so they get no board flags and no Delta-t.
Padding tokens embed to zero (they are masked out of attention as keys).

Under the block-causal scope with ``frozen_features="intrinsic"`` (stage N3; ruled by M20 -
``NetConfig.frozen_features``), a superseded board's tokens get neither of the two flags that
change after the board is frozen - Delta-t-to-present and the active/inactive timeline flag -
so that a frozen board's input depends on the board alone - and, under A7, on whose view it is
encoded in (the acting player: below); head boards keep both ("Live tokens
keep A3's features as inputs", section 4.1). The frozen boards' current values reach the live
tokens that read them as B16's score-time key biases instead (``status.py``). The model passes
``frozen_intrinsic`` for that scope only.

**The acting player (stage N5, register D8: "a global acting-player input"; A6, A7: "acting
colour kept as a global input").** One learned embedding per colour - whose action this is,
which differs from the present's colour once the present has shifted inside an unsubmitted
action (``PositionState.acting_player``) - is added to the global register tokens, and to no
other (M20: "acting player -> global registers"; section 2.3's global turn-state inputs). Under
A7 it is the acting *colour*: every board token's features are already the acting player's
view (owner, side to move, canonical coordinates), so "acting player" as such is always "us";
this embedding is what is left of the colour, "for the few colour-asymmetric facts (White
moves first)" (section 2.2). Every board token's embedding depends on who acts through that
view - frozen boards' included, so the cached path keys frozen K/V by the acting player
(``cache.py``; spec issue 303) - and on nothing else of it: the colour reaches the boards
through attention to the global registers.

**With no global registers** (``global_regs = 0``, an ablation-3 arm: "registers on/off")
there is nothing to carry it, and section 2.3 says nothing of that case; dropping it would
drop a committed input with the registers (A7 keeps the acting colour as an input). So it is
added to every board token instead - squares and per-board registers - and under the
block-causal scope (``block_causal``) to the live (head) boards' tokens only, as with global
registers, where no frozen board reads it. A PLACEHOLDER of a committed-class silence, fixed in
code (spec issue 225).

**The turn state (register A6, section 2.3: "embedded into the global registers").** The three
exact counts of ``PositionState.turn_state`` - mandatory boards still to move, optional boards
still playable, moves already made in the action (``Batch.turn_counts``) - enter beside the
acting colour, on the same tokens: every global register, and with none (issue 225's
placeholder, followed here) the board tokens, the live boards' alone under the block-causal
scope, so no frozen board's input - and no frozen K/V - depends on them. Each count ``n`` is
fed as ``(n / turn_state_scale, log1p n, [n == 0])`` - Delta-t's features, a count being
non-negative - and the nine features go through a small MLP (``turn_in``, GELU as Delta-t's,
``turn_out``) to one ``d``-vector per position: exact counts at any size, the zero (nothing owed,
nothing optional, a turn start) kept as a flag. The spec says only "embedded into the global
registers": the features, the scale, the width and the activation are a PLACEHOLDER of a
committed-class silence, fixed in code (spec issue 230; ``NetConfig.turn_state_hidden``,
``turn_state_scale``). Its parameters draw nothing while they are made and are drawn from a
generator of their own (``model._init_weights``, ``a6_seed``), so no other parameter's draw
changed when it was added.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import NetConfig
from .state import N_OWNERS, N_PIECE_TYPES, TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_SQUARE, TURN_STATE_FIELDS, Batch

#: Features per turn-state count: ``(n / turn_state_scale, log1p n, [n == 0])``.
TURN_FEATURES_PER_COUNT = 3


class InputEmbedding(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.piece = nn.Embedding(N_PIECE_TYPES, d)
        self.owner = nn.Embedding(N_OWNERS, d)  # A7: none, ours, theirs (A2's colour, relative)
        self.moved = nn.Embedding(2, d)
        self.head_flag = nn.Embedding(2, d)  # 0 superseded, 1 head
        self.active_flag = nn.Embedding(2, d)  # 0 inactive, 1 active timeline
        self.stm_flag = nn.Embedding(2, d)  # 0 us (the acting player) to move on the board, 1 them (A7)
        self.dt_in = nn.Linear(3, cfg.dt_hidden)
        self.dt_out = nn.Linear(cfg.dt_hidden, d)
        self.board_register = nn.Embedding(max(cfg.regs_per_board, 1), d)
        self.global_register = nn.Embedding(max(cfg.global_regs, 1), d)
        self.not_square = nn.Parameter(torch.empty(d))
        #: Register D8 / A7: the acting colour (0 white, 1 black), added to the global registers
        #: (M20) - with none, to the board tokens (the module docstring; spec issue 225). Made
        #: without drawing from the RNG, and drawn from D8's own generator
        #: (``model._init_weights``), never from the global stream.
        self.acting = nn.utils.skip_init(nn.Embedding, 2, d)
        #: Register A6: the turn-state counts' MLP (the module docstring; spec issue 230), on the
        #: same tokens as the acting colour. Made without drawing from the RNG, and drawn from
        #: A6's own generator (``model._init_weights``).
        self.turn_in = nn.utils.skip_init(nn.Linear, TURN_FEATURES_PER_COUNT * len(TURN_STATE_FIELDS), cfg.turn_state_hidden)
        self.turn_out = nn.utils.skip_init(nn.Linear, cfg.turn_state_hidden, d)

    def dt_features(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.float()
        return torch.stack([dt / self.cfg.dt_scale, torch.sign(dt) * torch.log1p(dt.abs()), (dt == 0).float()], dim=-1)

    def turn_features(self, counts: torch.Tensor) -> torch.Tensor:
        """``(B, 9)``: each of the ``(B, 3)`` turn-state counts as ``(n / turn_state_scale,
        log1p n, [n == 0])`` (register A6; spec issue 230)."""
        n = counts.float()
        return torch.stack([n / self.cfg.turn_state_scale, torch.log1p(n), (n == 0).float()], dim=-1).flatten(1)

    def turn_state(self, batch: Batch) -> torch.Tensor:
        """``(B, d)``: register A6's turn-state embedding of each position."""
        return self.turn_out(torch.nn.functional.gelu(self.turn_in(self.turn_features(batch.turn_counts()))))

    def forward(self, batch: Batch, frozen_intrinsic: bool = False, block_causal: bool = False) -> torch.Tensor:
        """Token embeddings ``(B, L, d)``. ``frozen_intrinsic``: superseded boards' tokens
        get no Delta-t and no active flag; ``block_causal``: the forward's scope is the
        block-causal one, so with no global registers only the head boards' tokens get the
        acting player and the turn state (see the module docstring)."""
        kind = batch.kind
        is_square = (kind == TOKEN_SQUARE).unsqueeze(-1)
        is_breg = (kind == TOKEN_BOARD_REGISTER).unsqueeze(-1)
        is_greg = (kind == TOKEN_GLOBAL_REGISTER).unsqueeze(-1)
        on_board = is_square | is_breg

        square = self.piece(batch.piece) + self.owner(batch.owner) + self.moved(batch.moved)
        board = (
            self.head_flag(batch.is_head)
            + self.active_flag(batch.active)
            + self.stm_flag(batch.stm)
            + self.dt_out(torch.nn.functional.gelu(self.dt_in(self.dt_features(batch.dt))))
        )
        if frozen_intrinsic:
            # Superseded boards: the flags that cannot change once a board is frozen only.
            frozen = self.head_flag(batch.is_head) + self.stm_flag(batch.stm)
            board = torch.where((batch.is_head == 1).unsqueeze(-1), board, frozen)
        slots = batch.reg_index.clamp(min=0)
        breg = self.board_register(slots.clamp(max=self.board_register.num_embeddings - 1)) + self.not_square
        greg = self.global_register(slots.clamp(max=self.global_register.num_embeddings - 1)) + self.not_square

        zero = torch.zeros((), dtype=square.dtype, device=square.device)
        # The global inputs: register D8's acting colour (A7) and A6's turn state.
        glob = (self.acting(batch.acting()) + self.turn_state(batch))[:, None, :]
        x = torch.where(is_square, square, zero)
        x = x + torch.where(is_breg, breg, zero)
        x = x + torch.where(on_board, board, zero)
        # On the global registers (M20; section 2.3).
        x = x + torch.where(is_greg, greg + glob, zero)
        if batch.global_regs == 0:
            # None to carry them (an ablation-3 arm): on the board tokens - the live boards' only
            # under the block-causal scope (a placeholder, spec issue 225).
            carry = on_board & (batch.is_head == 1).unsqueeze(-1) if block_causal else on_board
            x = x + torch.where(carry, glob, zero)
        return x
