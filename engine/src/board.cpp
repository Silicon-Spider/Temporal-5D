// BoardBuilder: assembling boards, with the content hash kept incrementally.

#include "t5d/board.hpp"

#include <stdexcept>
#include <string>

#include "t5d/zobrist.hpp"

namespace t5d {

namespace {

std::string not_on_board(int square, BoardSize size) {
  return "square " + std::to_string(square) + " is off the " + to_string(size) + " board";
}

}  // namespace

BoardBuilder::BoardBuilder(BoardSize size) {
  if (!size.valid()) {
    throw std::invalid_argument("board size " + to_string(size) + " is outside 1x1..8x8");
  }
  d_.width = static_cast<std::uint8_t>(size.width);
  d_.height = static_cast<std::uint8_t>(size.height);
}

BoardBuilder::BoardBuilder(const Board& base) noexcept : d_(base.d_) {}

void BoardBuilder::put(int square, Piece piece) {
  if (!size().contains_square(square)) throw std::out_of_range(not_on_board(square, size()));
  Piece& slot = d_.squares[static_cast<std::size_t>(square)];
  const Piece old = slot;
  if (old == piece) return;
  const std::uint64_t bit = square_bit(square);
  if (!old.empty()) {
    d_.by_colour[static_cast<std::size_t>(index_of(old.colour()))] &= ~bit;
    d_.by_kind[static_cast<std::size_t>(index_of(old.kind()))] &= ~bit;
    d_.unmoved &= ~bit;
    d_.content_hash ^= zobrist::piece_key(old, square);
  }
  if (!piece.empty()) {
    d_.by_colour[static_cast<std::size_t>(index_of(piece.colour()))] |= bit;
    d_.by_kind[static_cast<std::size_t>(index_of(piece.kind()))] |= bit;
    if (piece.unmoved()) d_.unmoved |= bit;
    d_.content_hash ^= zobrist::piece_key(piece, square);
  }
  slot = piece;
}

void BoardBuilder::put(int file, int rank, Piece piece) {
  if (!size().contains(file, rank)) {
    throw std::out_of_range("(file " + std::to_string(file) + ", rank " + std::to_string(rank) +
                            ") is off the " + to_string(size()) + " board");
  }
  put(make_square(file, rank), piece);
}

void BoardBuilder::set_en_passant(int square) {
  if (square != kNoSquare && !size().contains_square(square)) {
    throw std::out_of_range(not_on_board(square, size()));
  }
  if (d_.en_passant != kNoSquare) d_.content_hash ^= zobrist::en_passant_key(d_.en_passant);
  d_.en_passant = static_cast<std::int8_t>(square);
  if (square != kNoSquare) d_.content_hash ^= zobrist::en_passant_key(square);
}

BoardRef BoardBuilder::publish() const { return std::make_shared<const Board>(Board::Key{}, d_); }

std::uint64_t content_hash_from_scratch(const Board& board) noexcept {
  std::uint64_t hash = 0;
  for (int square = 0; square < kSquareCount; ++square) {
    const Piece piece = board.at(square);
    if (!piece.empty()) hash ^= zobrist::piece_key(piece, square);
  }
  if (board.en_passant() != kNoSquare) hash ^= zobrist::en_passant_key(board.en_passant());
  return hash;
}

}  // namespace t5d
