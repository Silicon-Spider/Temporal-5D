/* t5d/records.h - the plain-C records and ids of an expansion (t5d/expand.hpp),
 * shared by the C++ core and the C API (t5d/capi.h): the C API hands out the
 * expansion's own arrays, so the two must agree on these types exactly.
 *
 * Two square indices are in use, and the names say which:
 *
 *  - a SQUARE is the engine's index, rank * 8 + file on every board size
 *    (a1 = 0; t5d/types.hpp);
 *  - a CELL is the network's index (python/t5d/net/state.py: "square index
 *    s = y*W + x"), rank * width + file.
 *
 * They agree on 8x8 boards and differ on smaller ones. Every array an
 * expansion makes for the network - per-square features, move edges, typed
 * relations - is indexed by cells; a move record carries both, the square for
 * the rules and the cell for the network.
 *
 * Piece type ids (python/t5d/net/state.py PIECE_LETTERS): 0 = no piece, then
 * 1..12 = P N B R Q K U D S W C Y - one more than t5d::PieceKind. Colour ids:
 * 0 = no piece, 1 = white, 2 = black.
 */
#ifndef T5D_RECORDS_H
#define T5D_RECORDS_H

#include <stdint.h>

/* ---- move flags (t5d_move.flags) ------------------------------------------
 * The first six are t5d::Move's own flags, bit for bit. */

#define T5D_MOVE_CAPTURE 1u     /* the move takes a piece (en passant included) */
#define T5D_MOVE_BRANCHING 2u   /* it lands on a history board and creates a timeline (J3) */
#define T5D_MOVE_CASTLE 4u      /* castling (S1): the king's two-square move */
#define T5D_MOVE_EN_PASSANT 8u  /* en passant (S7) */
#define T5D_MOVE_DOUBLE_STEP 16u /* a pawn's or brawn's two-step advance (4.6) */
#define T5D_MOVE_PROMOTION 32u  /* the pawn or brawn becomes `promotion` (S8) */
#define T5D_MOVE_TRAVEL 64u     /* cross-board: the destination board is not the source board
                                   (E(k) is the travel moves' destinations outside W(k)) */

/* ---- relation ids -----------------------------------------------------------
 * python/t5d/net/state.py RELATION_TYPES, in its order: "move" is the move
 * graph (register B9), the rest are the typed relations of design spec §6
 * (register B11, provisional). */

#define T5D_RELATION_MOVE 0
#define T5D_RELATION_ATTACK 1
#define T5D_RELATION_DEFENCE 2
#define T5D_RELATION_PIN 3   /* reserved, not built: needs check detection (stage E4; R1) */
#define T5D_RELATION_XRAY 4
#define T5D_RELATION_CHECK 5 /* reserved, not built: needs check detection (stage E4; R1) */
#define T5D_RELATION_TRAVEL_THREAT 6 /* reserved, not built: the spec defines no edge for it */

/* ---- one pseudo-legal move ---------------------------------------------------
 * 32 bytes, no padding. Boards are indices into the expansion's boards (its
 * canonical order: ascending timeline, then half-turn). */

typedef struct t5d_move {
  int32_t from_timeline;  /* timeline position p of the source board */
  int32_t from_half_turn; /* h = 2T + (1 if black to move) */
  int32_t to_timeline;    /* the board the move lands on - for a branching move */
  int32_t to_half_turn;   /*   the history board it targets, not the board it creates */
  int32_t from_board;     /* the source board's index in the expansion */
  int32_t to_board;       /* the destination board's index in the expansion */
  int8_t from_square;     /* rank * 8 + file */
  int8_t to_square;
  int8_t from_cell;       /* rank * width + file */
  int8_t to_cell;
  uint8_t piece;          /* the moving piece's type id, 1..12 */
  uint8_t captured;       /* 0, or the captured piece's type id (en passant: the pawn taken) */
  uint8_t promotion;      /* 0, or the type id promoted to */
  uint8_t flags;          /* T5D_MOVE_* */
} t5d_move;

#endif /* T5D_RECORDS_H */
