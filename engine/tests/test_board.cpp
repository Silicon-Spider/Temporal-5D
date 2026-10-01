// The immutable board and its builder: placement, bitboards, content hash,
// equality, immutability after publication.

#include <bit>
#include <cstdint>
#include <stdexcept>

#include "t5d/board.hpp"
#include "t5d/zobrist.hpp"
#include "test_framework.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

// Recomputes every derived field from the mailbox and compares.
void check_consistent(const Board& board) {
  std::uint64_t by_colour[2] = {0, 0};
  std::uint64_t by_kind[kPieceKindCount] = {};
  std::uint64_t unmoved = 0;
  const std::uint64_t on_board = board.size().mask();
  for (int sq = 0; sq < kSquareCount; ++sq) {
    const Piece p = board.at(sq);
    if (p.empty()) continue;
    CHECK((on_board & square_bit(sq)) != 0);
    by_colour[index_of(p.colour())] |= square_bit(sq);
    by_kind[index_of(p.kind())] |= square_bit(sq);
    if (p.unmoved()) unmoved |= square_bit(sq);
  }
  CHECK_EQ(board.pieces(Colour::White), by_colour[0]);
  CHECK_EQ(board.pieces(Colour::Black), by_colour[1]);
  CHECK_EQ(board.occupied(), by_colour[0] | by_colour[1]);
  for (int k = 0; k < kPieceKindCount; ++k) {
    CHECK_EQ(board.pieces(static_cast<PieceKind>(k)), by_kind[k]);
  }
  CHECK_EQ(board.unmoved(), unmoved);
  CHECK_EQ(board.piece_count(), std::popcount(by_colour[0] | by_colour[1]));
  CHECK_EQ(board.content_hash(), content_hash_from_scratch(board));
}

}  // namespace

TEST(empty_board) {
  const BoardRef board = BoardBuilder(BoardSize{6, 5}).publish();
  CHECK_EQ(board->width(), 6);
  CHECK_EQ(board->height(), 5);
  CHECK_EQ(board->occupied(), std::uint64_t{0});
  CHECK_EQ(board->piece_count(), 0);
  CHECK_EQ(board->content_hash(), std::uint64_t{0});
  CHECK_EQ(board->en_passant(), kNoSquare);
  for (int sq = 0; sq < kSquareCount; ++sq) CHECK(board->at(sq).empty());
}

TEST(builder_rejects_invalid_sizes_and_squares) {
  CHECK_THROWS_AS(BoardBuilder(BoardSize{0, 8}), std::invalid_argument);
  CHECK_THROWS_AS(BoardBuilder(BoardSize{8, 9}), std::invalid_argument);
  BoardBuilder b(BoardSize{5, 5});
  const Piece rook(PieceKind::Rook, Colour::White);
  CHECK_THROWS_AS(b.put(make_square(5, 0), rook), std::out_of_range);
  CHECK_THROWS_AS(b.put(make_square(0, 5), rook), std::out_of_range);
  CHECK_THROWS_AS(b.put(-1, rook), std::out_of_range);
  CHECK_THROWS_AS(b.put(64, rook), std::out_of_range);
  CHECK_THROWS_AS(b.set_en_passant(make_square(6, 2)), std::out_of_range);
  CHECK_NOTHROW(b.put(make_square(4, 4), rook));
  CHECK_NOTHROW(b.set_en_passant(kNoSquare));
}

TEST(coordinates_off_the_board_read_empty_and_cannot_be_written) {
  // The move rules step sliders and leapers past the edges: off the board is
  // empty - never another square's piece (make_square(-1, 1) is h1's index,
  // make_square(8, 0) is a2's) and never a read past the mailbox.
  BoardBuilder full(BoardSize{8, 8});
  for (int file = 0; file < 8; ++file) {
    for (int rank = 0; rank < 8; ++rank) full.put(file, rank, Piece(PieceKind::Rook, Colour::White));
  }
  const BoardRef board = full.publish();
  CHECK(board->at(8, 0).empty());
  CHECK(board->at(-1, 1).empty());
  CHECK(board->at(0, 8).empty());
  CHECK(board->at(0, -1).empty());
  CHECK(board->at(-1, -1).empty());
  CHECK(board->at(100, 100).empty());
  CHECK(board->at(-1).empty());
  CHECK(board->at(64).empty());
  CHECK(board->at(7, 0) == Piece(PieceKind::Rook, Colour::White));
  CHECK(full.at(8, 0).empty());
  CHECK(full.at(-1, 1).empty());
  CHECK(full.at(0, 8).empty());
  CHECK(full.at(64).empty());

  // On a smaller board, the 8x8 grid's squares beyond its edges are off it too.
  BoardBuilder small(BoardSize{5, 4});
  small.put(4, 3, Piece(PieceKind::Queen, Colour::Black));
  CHECK(small.at(4, 3) == Piece(PieceKind::Queen, Colour::Black));
  CHECK(small.at(5, 3).empty());
  CHECK(small.at(4, 4).empty());
  CHECK(small.publish()->at(5, 0).empty());

  // Writing: refused as a coordinate, not folded into a square index first.
  const Piece queen(PieceKind::Queen, Colour::Black);
  CHECK_THROWS_AS(full.put(-1, 1, queen), std::out_of_range);
  CHECK_THROWS_AS(full.put(8, 0, queen), std::out_of_range);
  CHECK_THROWS_AS(full.put(0, 8, queen), std::out_of_range);
  CHECK_THROWS_AS(small.put(5, 0, queen), std::out_of_range);
  CHECK_THROWS_AS(small.put(0, 4, queen), std::out_of_range);
  CHECK(full.at(7, 0) == Piece(PieceKind::Rook, Colour::White));  // h1 untouched
  CHECK(full.at(0, 1) == Piece(PieceKind::Rook, Colour::White));  // a2 untouched
  CHECK(*full.publish() == *board);
}

TEST(builder_keeps_bitboards_and_hash_consistent) {
  t5d_test::Rng rng(42);
  for (int round = 0; round < 60; ++round) {
    const BoardSize size = t5d_test::random_size(rng);
    BoardBuilder b(size);
    for (int op = 0; op < 120; ++op) {
      const int sq = make_square(rng.below(size.width), rng.below(size.height));
      const int action = rng.below(10);
      if (action < 6) {
        b.put(sq, t5d_test::random_piece(rng));
      } else if (action < 9) {
        b.remove(sq);
      } else {
        b.set_en_passant(rng.chance(50) ? kNoSquare : sq);
      }
      if (op % 17 == 0) {
        const BoardRef snapshot = b.publish();
        check_consistent(*snapshot);
        CHECK_EQ(b.content_hash(), snapshot->content_hash());
        CHECK_EQ(b.piece_count(), snapshot->piece_count());
      }
    }
    check_consistent(*b.publish());
  }
}

TEST(published_boards_never_change) {
  BoardBuilder b(BoardSize{8, 8});
  b.put(make_square(4, 0), Piece(PieceKind::King, Colour::White, true));
  const BoardRef first = b.publish();
  const std::uint64_t hash = first->content_hash();
  b.put(make_square(4, 0), Piece{});
  b.put(make_square(4, 7), Piece(PieceKind::King, Colour::Black, true));
  b.set_en_passant(make_square(3, 5));
  const BoardRef second = b.publish();
  CHECK(first->at(4, 0) == Piece(PieceKind::King, Colour::White, true));
  CHECK(first->at(4, 7).empty());
  CHECK_EQ(first->en_passant(), kNoSquare);
  CHECK_EQ(first->content_hash(), hash);
  CHECK(!(*first == *second));
  // A builder made from a board starts as an exact copy.
  BoardBuilder copy(*second);
  CHECK(*copy.publish() == *second);
  CHECK_EQ(copy.content_hash(), second->content_hash());
}

TEST(equal_contents_equal_boards_and_hashes) {
  t5d_test::Rng rng(7);
  for (int round = 0; round < 40; ++round) {
    const BoardSize size = t5d_test::random_size(rng);
    const BoardRef a = t5d_test::random_board(rng, size, 50);
    // The same placement, built in reverse order with detours.
    BoardBuilder b(size);
    for (int sq = kSquareCount - 1; sq >= 0; --sq) {
      if (!size.contains_square(sq)) continue;
      b.put(sq, Piece(PieceKind::Dragon, Colour::Black));
      b.put(sq, a->at(sq));
    }
    const BoardRef copy = b.publish();
    CHECK(*a == *copy);
    CHECK_EQ(a->content_hash(), copy->content_hash());
  }
}

TEST(boards_differing_anywhere_are_unequal) {
  BoardBuilder b(BoardSize{8, 8});
  b.put(make_square(0, 0), Piece(PieceKind::Rook, Colour::White, true));
  const BoardRef base = b.publish();

  BoardBuilder moved(*base);
  moved.put(make_square(0, 0), Piece(PieceKind::Rook, Colour::White, false));
  const BoardRef unmoved_differs = moved.publish();
  CHECK(!(*base == *unmoved_differs));
  CHECK_NE(base->content_hash(), unmoved_differs->content_hash());

  BoardBuilder colour(*base);
  colour.put(make_square(0, 0), Piece(PieceKind::Rook, Colour::Black, true));
  CHECK(!(*base == *colour.publish()));

  BoardBuilder ep(*base);
  ep.set_en_passant(make_square(4, 2));
  const BoardRef ep_differs = ep.publish();
  CHECK(!(*base == *ep_differs));
  CHECK_NE(base->content_hash(), ep_differs->content_hash());

  // Same pieces, different board size.
  BoardBuilder small(BoardSize{5, 5});
  small.put(make_square(0, 0), Piece(PieceKind::Rook, Colour::White, true));
  const BoardRef other_size = small.publish();
  CHECK(!(*base == *other_size));
  // The content hash is placement only; the multiverse key adds the size
  // (test_zobrist.cpp, zobrist_board_size_is_in_the_key).
  CHECK_EQ(base->content_hash(), other_size->content_hash());
}

TEST(unmoved_flag_dropped_from_kinds_that_do_not_track_it) {
  BoardBuilder b(BoardSize{8, 8});
  b.put(make_square(3, 0), Piece(PieceKind::Queen, Colour::White, true));
  b.put(make_square(4, 0), Piece(PieceKind::King, Colour::White, true));
  b.put(make_square(1, 1), Piece(PieceKind::Brawn, Colour::White, true));
  b.put(make_square(2, 2), Piece(PieceKind::CommonKing, Colour::Black, true));
  const BoardRef board = b.publish();
  CHECK_EQ(board->unmoved(), square_bit(make_square(4, 0)) | square_bit(make_square(1, 1)));
  CHECK(!board->at(3, 0).unmoved());
  CHECK(!board->at(2, 2).unmoved());
}

TEST(royals_are_kings_and_royal_queens) {
  BoardBuilder b(BoardSize{8, 8});
  b.put(make_square(4, 0), Piece(PieceKind::King, Colour::White));
  b.put(make_square(3, 0), Piece(PieceKind::RoyalQueen, Colour::White));
  b.put(make_square(2, 0), Piece(PieceKind::CommonKing, Colour::White));
  b.put(make_square(4, 7), Piece(PieceKind::King, Colour::Black));
  b.put(make_square(3, 7), Piece(PieceKind::Queen, Colour::Black));
  const BoardRef board = b.publish();
  CHECK_EQ(board->royals(Colour::White),
           square_bit(make_square(4, 0)) | square_bit(make_square(3, 0)));
  CHECK_EQ(board->royals(Colour::Black), square_bit(make_square(4, 7)));
  CHECK_EQ(board->pieces(Colour::White, PieceKind::King), square_bit(make_square(4, 0)));
}

TEST(content_hash_is_the_xor_of_piece_keys) {
  BoardBuilder b(BoardSize{8, 8});
  const Piece knight(PieceKind::Knight, Colour::Black);
  const Piece pawn(PieceKind::Pawn, Colour::White, true);
  b.put(make_square(1, 7), knight);
  b.put(make_square(4, 1), pawn);
  const std::uint64_t expected =
      zobrist::piece_key(knight, make_square(1, 7)) ^ zobrist::piece_key(pawn, make_square(4, 1));
  CHECK_EQ(b.content_hash(), expected);
  b.set_en_passant(make_square(4, 2));
  CHECK_EQ(b.content_hash(), expected ^ zobrist::en_passant_key(make_square(4, 2)));
  b.set_en_passant(kNoSquare);
  b.remove(make_square(1, 7));
  b.remove(make_square(4, 1));
  CHECK_EQ(b.content_hash(), std::uint64_t{0});
}
