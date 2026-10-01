// t5d/board.hpp - one chess board: an immutable value, shared by reference.
//
// Decision A4 (spec §2.4): superseded boards are permanently immutable - a
// travel move creates a new timeline rather than editing a board - and frozen
// K/V, frozen NNUE accumulators and the asset map all rest on it. So every
// Board is immutable from birth: a BoardBuilder assembles one, publish()
// freezes a copy into a Board behind a std::shared_ptr<const Board>, and from
// then on it is only read, by any number of timelines, multiverse copies and
// threads. Head boards are immutable too; "changing" a head means publishing a
// new board in its place (Multiverse::replace_head).
//
// A board holds its placement twice - a mailbox (one Piece per square) and
// bitboards per colour and per kind, plus the bitboard of unmoved pieces -
// and an en-passant square, a field reserved for the move rules (stage E2;
// 5DFEN has no field for it, so parsed boards have none). Its content hash is
// coordinate-free (see t5d/zobrist.hpp): the board does not know where it is.

#ifndef T5D_BOARD_HPP
#define T5D_BOARD_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "t5d/types.hpp"

namespace t5d {

class Board;
class BoardBuilder;

/// How boards are held: shared, immutable.
using BoardRef = std::shared_ptr<const Board>;

namespace detail {

struct BoardData {
  std::array<Piece, kSquareCount> squares{};
  std::array<std::uint64_t, 2> by_colour{};
  std::array<std::uint64_t, kPieceKindCount> by_kind{};
  std::uint64_t unmoved = 0;
  std::uint64_t content_hash = 0;
  std::int8_t en_passant = kNoSquare;
  std::uint8_t width = 8;
  std::uint8_t height = 8;
};

}  // namespace detail

class Board {
 public:
  /// Only BoardBuilder can make a Key, so only BoardBuilder::publish() makes
  /// boards - with bitboards and hash consistent with the mailbox.
  class Key {
    Key() = default;
    friend class BoardBuilder;
  };

  Board(Key, const detail::BoardData& data) noexcept : d_(data) {}

  BoardSize size() const noexcept { return BoardSize{d_.width, d_.height}; }
  int width() const noexcept { return d_.width; }
  int height() const noexcept { return d_.height; }

  /// The piece on a square; the empty square for any square off the board -
  /// outside 0..63, or on the 8x8 grid but outside a smaller board.
  Piece at(int square) const noexcept {
    return static_cast<unsigned>(square) < static_cast<unsigned>(kSquareCount)
               ? d_.squares[static_cast<std::size_t>(square)]
               : Piece{};
  }
  /// The piece at (file, rank); the empty square for any coordinate off the
  /// board (file outside 0..width-1 or rank outside 0..height-1), so a slider
  /// or leaper may step past an edge and read "empty". (Never folded into a
  /// square index first: make_square(-1, 1) is h1's index, not "off the board".)
  /// The square overload's own range check is redundant for a valid board, and
  /// lets the compiler see the index is in range.
  Piece at(int file, int rank) const noexcept {
    return size().contains(file, rank) ? at(make_square(file, rank)) : Piece{};
  }

  std::uint64_t pieces(Colour c) const noexcept {
    return d_.by_colour[static_cast<std::size_t>(index_of(c))];
  }
  std::uint64_t pieces(PieceKind k) const noexcept {
    return d_.by_kind[static_cast<std::size_t>(index_of(k))];
  }
  std::uint64_t pieces(Colour c, PieceKind k) const noexcept { return pieces(c) & pieces(k); }
  std::uint64_t occupied() const noexcept { return d_.by_colour[0] | d_.by_colour[1]; }
  /// Squares whose piece has not moved (only pawns, rooks, kings, brawns).
  std::uint64_t unmoved() const noexcept { return d_.unmoved; }
  /// The royal pieces of one colour: kings and royal queens.
  std::uint64_t royals(Colour c) const noexcept {
    return pieces(c) & (pieces(PieceKind::King) | pieces(PieceKind::RoyalQueen));
  }

  /// The en-passant square, or kNoSquare. Reserved for stage E2.
  int en_passant() const noexcept { return d_.en_passant; }

  /// The mailbox as bytes: Piece::code() of every square, index = square
  /// (64 bytes, whatever the board's size; squares off the board are 0). For
  /// readers that decode many squares at once (t5d/expand.hpp's features).
  const std::uint8_t* codes() const noexcept {
    static_assert(sizeof(Piece) == 1, "a Piece is its one-byte code");
    return reinterpret_cast<const std::uint8_t*>(d_.squares.data());
  }

  int piece_count() const noexcept { return std::popcount(occupied()); }

  /// The coordinate-free content hash (t5d/zobrist.hpp), computed as the board
  /// was built.
  std::uint64_t content_hash() const noexcept { return d_.content_hash; }

  /// Same size, same piece (kind, colour, unmoved) on every square, same
  /// en-passant square.
  friend bool operator==(const Board& a, const Board& b) noexcept {
    return a.d_.content_hash == b.d_.content_hash && a.d_.width == b.d_.width &&
           a.d_.height == b.d_.height && a.d_.en_passant == b.d_.en_passant &&
           a.d_.squares == b.d_.squares;
  }

 private:
  friend class BoardBuilder;
  detail::BoardData d_;
};

/// Assembles a board. Not shared, not thread-safe; publish() freezes a copy.
class BoardBuilder {
 public:
  /// An empty board. Throws std::invalid_argument for a size outside 1..8.
  explicit BoardBuilder(BoardSize size);
  /// A board to be derived from `base` (what a move does).
  explicit BoardBuilder(const Board& base) noexcept;

  BoardSize size() const noexcept { return BoardSize{d_.width, d_.height}; }

  /// As Board::at: the empty square for anything off the board.
  Piece at(int square) const noexcept {
    return static_cast<unsigned>(square) < static_cast<unsigned>(kSquareCount)
               ? d_.squares[static_cast<std::size_t>(square)]
               : Piece{};
  }
  Piece at(int file, int rank) const noexcept {
    return size().contains(file, rank) ? at(make_square(file, rank)) : Piece{};
  }

  /// Puts `piece` on `square`, replacing whatever was there; an empty piece
  /// empties the square. The content hash is updated incrementally. Throws
  /// std::out_of_range if the square is not on the board.
  void put(int square, Piece piece);
  /// The same at (file, rank). Throws std::out_of_range for a coordinate off
  /// the board - checked as a coordinate, so (-1, 1) is refused rather than
  /// read as h1's index.
  void put(int file, int rank, Piece piece);
  void remove(int square) { put(square, Piece{}); }

  /// Sets the en-passant square (kNoSquare clears it). Reserved for stage E2.
  /// Throws std::out_of_range for a square that is not on the board.
  void set_en_passant(int square);

  std::uint64_t content_hash() const noexcept { return d_.content_hash; }
  int piece_count() const noexcept {
    return std::popcount(d_.by_colour[0] | d_.by_colour[1]);
  }

  /// A frozen copy. The builder stays usable; later edits do not reach the copy.
  BoardRef publish() const;

 private:
  detail::BoardData d_;
};

/// The content hash recomputed from the mailbox alone. Tests hold the
/// incremental hash to it.
std::uint64_t content_hash_from_scratch(const Board& board) noexcept;

}  // namespace t5d

#endif  // T5D_BOARD_HPP
