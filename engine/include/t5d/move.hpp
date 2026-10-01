// t5d/move.hpp - one move: a piece from a square of the mover's playable
// board to a square of a board of the mover's colour.
//
// Rules reference: a move starts from a piece of the mover's colour on a
// playable board of the mover's colour (L5) and ends on a square of an
// existing board of the mover's colour (J2, L4). It is
//
//  * PHYSICAL when it stays on its board (one new board, T2);
//  * a NON-BRANCHING JUMP when it lands on another timeline's playable board
//    (J4: both timelines get a new board);
//  * a BRANCHING JUMP when it lands on a history board (J3, J5: the source
//    timeline gets a new board, and a new timeline is created whose first
//    board is the destination's half-step successor).
//
// A Move records the source and destination boards by (timeline position,
// half-turn) and squares, the moving piece and the captured one, and the
// special cases. The new timeline a branching move creates is not recorded:
// its index follows from the state the move is applied to (L7; J9 - several
// branches in one action take their indices in the order they are made).
//
// A Move is a plain value. Generated moves (t5d/movegen.hpp) are exactly the
// pseudo-legal moves of a state; GameState::apply checks a move's structure
// against the state, so a malformed move cannot corrupt it.

#ifndef T5D_MOVE_HPP
#define T5D_MOVE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "t5d/types.hpp"

namespace t5d {

struct Move {
  enum Flag : std::uint8_t {
    kCapture = 1,     // the destination held an enemy piece (or en passant)
    kBranching = 2,   // the destination is a history board (J3)
    kCastle = 4,      // castling (S1): the king moves two squares; aux_square is the rook's
    kEnPassant = 8,   // en passant (S7); aux_square is the captured pawn's square
    kDoubleStep = 16, // a pawn's or brawn's two-step advance (4.6), rank or timeline
    kPromotion = 32,  // the pawn or brawn becomes `promotion` (S8)
  };

  std::int32_t from_timeline = 0;
  std::int32_t from_half_turn = 0;
  std::int32_t to_timeline = 0;
  std::int32_t to_half_turn = 0;
  std::int8_t from_square = kNoSquare;
  std::int8_t to_square = kNoSquare;
  /// Castling: the rook's square. En passant: the captured pawn's square.
  /// Otherwise kNoSquare.
  std::int8_t aux_square = kNoSquare;
  /// The moving piece, as it stood on the source square.
  Piece piece;
  /// The captured piece; empty if none.
  Piece captured;
  /// The promotion piece's kind; meaningful only with kPromotion (Queen
  /// otherwise, so that equal moves compare equal).
  PieceKind promotion = PieceKind::Queen;
  std::uint8_t flags = 0;

  /// Within one board.
  constexpr bool physical() const noexcept {
    return from_timeline == to_timeline && from_half_turn == to_half_turn;
  }
  constexpr bool jump() const noexcept { return !physical(); }
  constexpr bool has(Flag f) const noexcept { return (flags & f) != 0; }
  constexpr bool capture() const noexcept { return has(kCapture); }
  constexpr bool branching() const noexcept { return has(kBranching); }
  constexpr bool castle() const noexcept { return has(kCastle); }
  constexpr bool en_passant() const noexcept { return has(kEnPassant); }
  constexpr bool double_step() const noexcept { return has(kDoubleStep); }
  constexpr bool promotes() const noexcept { return has(kPromotion); }

  friend constexpr bool operator==(const Move&, const Move&) noexcept = default;
};

using MoveList = std::vector<Move>;

/// The move in the 5dpgn "export" form with the piece letter (notation
/// specification, "Export moves"): "(0T1)Pe2(0T1)e4", "(-1T4)Nc3(0T2)c3",
/// castling as the king's two-square move "(0T6)Ke1(0T6)g1"; "=Q" after a
/// promotion. Letters are upper case for both colours, as in 5dpgn moves.
/// `even` selects the labels of a game with an even number of starting
/// timelines ("-0"/"+0"). For messages, tests and logs.
std::string to_string(const Move& move, bool even);

}  // namespace t5d

#endif  // T5D_MOVE_HPP
