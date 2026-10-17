"""Input embeddings (A2, A3, B8; spec sections 2.2, 2.3, 5.3).

Square tokens: learned embeddings of piece type (13 incl. empty), colour (none, white,
black) and the has-moved flag, summed (A2). Coordinates are not embedded here: they
reach attention through the 4D RoPE (B4).

Every token of a board - its squares and its per-board registers - additionally gets
the board's flag embeddings (head/superseded, active/inactive timeline, side to move;
section 2.3: "added to every token on the board") and its Delta-t-to-present embedding
(A3). Delta-t is continuous: a small learned MLP (GELU) over the features
``(dt / dt_scale, sign(dt) * log1p|dt|, [dt == 0])``, so the binary "present" flag is
kept as Delta-t's zero (section 2.3) while the distance to the present stays readable
at any range (implementation choice: the spec says only "embedded per board"; F6's Mish
is the trunk FFN's activation, and this MLP is neither the FFN nor a head - spec issue
204).

Register tokens get a register-type embedding (per-board slot 0..R-1, or global index
0..G-1) plus one learned "not a square" embedding (section 5.3) instead of piece/colour/
has-moved. Global registers hold no board, so they get no board flags and no Delta-t.
Padding tokens embed to zero (they are masked out of attention as keys).

Under the block-causal scope with ``frozen_features="intrinsic"`` (stage N3; ruled by M20 -
``NetConfig.frozen_features``), a superseded board's tokens get neither of the two flags that
change after the board is frozen - Delta-t-to-present and the active/inactive timeline flag -
so that a frozen board's input depends on the board alone; head boards keep both ("Live tokens
keep A3's features as inputs", section 4.1). The frozen boards' current values reach the live
tokens that read them as B16's score-time key biases instead (``status.py``). The model passes
``frozen_intrinsic`` for that scope only.

**The acting player (stage N5, register D8: "a global acting-player input").** One learned
embedding per colour - whose action this is, which differs from the present's colour once
the present has shifted inside an unsubmitted action (``PositionState.acting_player``) - is
added to the global register tokens (``acting_player_input="global_registers"``, the
placeholder default), or to every token (``"all_tokens"``; a superseded board's tokens
excepted under ``frozen_intrinsic``, as for Delta-t). Under the default a board token's
embedding is unchanged: the input reaches the boards through attention to the global
registers, which the live set reads and frozen boards never do under the block-causal scope.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import NetConfig
from .state import N_COLOURS, N_PIECE_TYPES, TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_SQUARE, Batch


class InputEmbedding(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.piece = nn.Embedding(N_PIECE_TYPES, d)
        self.colour = nn.Embedding(N_COLOURS, d)
        self.moved = nn.Embedding(2, d)
        self.head_flag = nn.Embedding(2, d)  # 0 superseded, 1 head
        self.active_flag = nn.Embedding(2, d)  # 0 inactive, 1 active timeline
        self.stm_flag = nn.Embedding(2, d)  # 0 white to move on the board, 1 black
        self.dt_in = nn.Linear(3, cfg.dt_hidden)
        self.dt_out = nn.Linear(cfg.dt_hidden, d)
        self.board_register = nn.Embedding(max(cfg.regs_per_board, 1), d)
        self.global_register = nn.Embedding(max(cfg.global_regs, 1), d)
        self.not_square = nn.Parameter(torch.empty(d))
        #: Register D8: the acting player (0 white, 1 black) - where it is added is
        #: ``cfg.acting_player_input``. Made without drawing from the RNG, and drawn from D8's
        #: own generator (``model._init_weights``), never from the global stream.
        self.acting = nn.utils.skip_init(nn.Embedding, 2, d)

    def dt_features(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.float()
        return torch.stack([dt / self.cfg.dt_scale, torch.sign(dt) * torch.log1p(dt.abs()), (dt == 0).float()], dim=-1)

    def forward(self, batch: Batch, frozen_intrinsic: bool = False) -> torch.Tensor:
        """Token embeddings ``(B, L, d)``. ``frozen_intrinsic``: superseded boards' tokens
        get no Delta-t and no active flag (see the module docstring)."""
        kind = batch.kind
        is_square = (kind == TOKEN_SQUARE).unsqueeze(-1)
        is_breg = (kind == TOKEN_BOARD_REGISTER).unsqueeze(-1)
        is_greg = (kind == TOKEN_GLOBAL_REGISTER).unsqueeze(-1)
        on_board = is_square | is_breg

        square = self.piece(batch.piece) + self.colour(batch.colour) + self.moved(batch.moved)
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
        x = torch.where(is_square, square, zero)
        x = x + torch.where(is_breg, breg, zero)
        x = x + torch.where(on_board, board, zero)
        x = x + torch.where(is_greg, greg, zero)
        return x + torch.where(self.acting_mask(batch, frozen_intrinsic).unsqueeze(-1), self.acting(batch.acting())[:, None, :], zero)

    def acting_mask(self, batch: Batch, frozen_intrinsic: bool = False) -> torch.Tensor:
        """``(B, L)`` bool: the tokens that get the acting-player embedding (register D8;
        ``NetConfig.acting_player_input``): the global registers, or every token of a board
        too - but not a superseded board's under ``frozen_intrinsic`` (the acting player
        changes after the board is frozen). Never padding."""
        kind = batch.kind
        mask = kind == TOKEN_GLOBAL_REGISTER
        if self.cfg.acting_player_input == "all_tokens":
            on_board = (kind == TOKEN_SQUARE) | (kind == TOKEN_BOARD_REGISTER)
            if frozen_intrinsic:
                on_board = on_board & (batch.is_head == 1)
            mask = mask | on_board
        return mask
