// Colours, piece kinds, pieces, squares, board sizes, half-turns.

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>

#include "t5d/types.hpp"
#include "test_framework.hpp"

using namespace t5d;

namespace {

constexpr PieceKind kAllKinds[] = {
    PieceKind::Pawn,    PieceKind::Knight,  PieceKind::Bishop,   PieceKind::Rook,
    PieceKind::Queen,   PieceKind::King,    PieceKind::Unicorn,  PieceKind::Dragon,
    PieceKind::Princess, PieceKind::Brawn,  PieceKind::CommonKing, PieceKind::RoyalQueen};

}  // namespace

TEST(piece_packs_kind_colour_and_unmoved) {
  for (const PieceKind kind : kAllKinds) {
    for (const Colour colour : {Colour::White, Colour::Black}) {
      for (const bool unmoved : {false, true}) {
        const Piece p(kind, colour, unmoved);
        CHECK(!p.empty());
        CHECK(p.kind() == kind);
        CHECK(p.colour() == colour);
        CHECK_EQ(p.unmoved(), unmoved && tracks_unmoved(kind));
        CHECK(p.code() >= 1 && p.code() <= Piece::kMaxCode);
        CHECK(Piece::from_code(p.code()) == p);
        CHECK(p.as_moved() == Piece(kind, colour, false));
        CHECK_EQ(p.royal(), is_royal(kind));
      }
    }
  }
}

TEST(distinct_pieces_have_distinct_codes) {
  std::set<int> codes;
  for (const PieceKind kind : kAllKinds) {
    for (const Colour colour : {Colour::White, Colour::Black}) {
      for (const bool unmoved : {false, true}) codes.insert(Piece(kind, colour, unmoved).code());
    }
  }
  // 12 kinds x 2 colours, plus an unmoved form for the 4 kinds that track it.
  CHECK_EQ(codes.size(), static_cast<std::size_t>(12 * 2 + 4 * 2));
  CHECK(codes.count(0) == 0);
}

TEST(empty_piece) {
  const Piece none;
  CHECK(none.empty());
  CHECK_EQ(none.code(), 0);
  CHECK(!none.unmoved());
  CHECK(!none.royal());
  CHECK_EQ(none.letter(), '.');
  CHECK(Piece::from_code(0).empty());
  CHECK(Piece::from_code(49).empty());
  CHECK(Piece::from_code(-3).empty());
  CHECK(none.as_moved().empty());
}

TEST(pieces_from_codes_are_normalised) {
  for (int code = 1; code <= Piece::kMaxCode; ++code) {
    const Piece p = Piece::from_code(code);
    REQUIRE(!p.empty());
    const int v = code - 1;
    CHECK(p.kind() == static_cast<PieceKind>(v >> 2));
    CHECK(p.colour() == static_cast<Colour>((v >> 1) & 1));
    CHECK_EQ(p.unmoved(), (v & 1) != 0 && tracks_unmoved(p.kind()));
    CHECK(Piece::from_code(p.code()) == p);
  }
  // The code of an "unmoved knight" gives a knight.
  const int unmoved_knight = 1 + ((index_of(PieceKind::Knight) << 2) | 1);
  CHECK(Piece::from_code(unmoved_knight) == Piece(PieceKind::Knight, Colour::White));
}

TEST(unmoved_flag_only_on_pawns_rooks_kings_brawns) {
  for (const PieceKind kind : kAllKinds) {
    const bool expected = kind == PieceKind::Pawn || kind == PieceKind::Rook ||
                          kind == PieceKind::King || kind == PieceKind::Brawn;
    CHECK_EQ(tracks_unmoved(kind), expected);
  }
  CHECK(!Piece(PieceKind::Queen, Colour::White, true).unmoved());
  CHECK(Piece(PieceKind::Queen, Colour::White, true) == Piece(PieceKind::Queen, Colour::White));
  CHECK(Piece(PieceKind::Brawn, Colour::Black, true).unmoved());
}

TEST(royal_pieces_are_king_and_royal_queen) {
  for (const PieceKind kind : kAllKinds) {
    CHECK_EQ(is_royal(kind), kind == PieceKind::King || kind == PieceKind::RoyalQueen);
  }
  CHECK(!is_royal(PieceKind::CommonKing));
}

TEST(letters_follow_the_notation) {
  const std::string letters = "PNBRQKUDSWCY";
  for (std::size_t i = 0; i < letters.size(); ++i) {
    const auto kind = static_cast<PieceKind>(i);
    CHECK_EQ(letter_of(kind), letters[i]);
    CHECK(kind_from_letter(letters[i]) == kind);
    CHECK(kind_from_letter(static_cast<char>(letters[i] - 'A' + 'a')) == kind);
    CHECK_EQ(Piece(kind, Colour::White).letter(), letters[i]);
    CHECK_EQ(Piece(kind, Colour::Black).letter(), static_cast<char>(letters[i] - 'A' + 'a'));
  }
  for (const char c : std::string("AEFGHIJLMOTVXZaefghijlmotvxz0*+/ ")) {
    CHECK(!kind_from_letter(c).has_value());
  }
}

TEST(squares_index_rank_times_eight_plus_file) {
  CHECK_EQ(make_square(0, 0), 0);  // a1
  CHECK_EQ(make_square(7, 0), 7);  // h1
  CHECK_EQ(make_square(0, 1), 8);  // a2
  CHECK_EQ(make_square(7, 7), 63);  // h8
  CHECK_EQ(make_square(4, 3), 28);  // e4
  for (int sq = 0; sq < 64; ++sq) {
    CHECK_EQ(make_square(file_of(sq), rank_of(sq)), sq);
    CHECK_EQ(square_bit(sq), std::uint64_t{1} << sq);
  }
  CHECK_EQ(square_name(0), std::string("a1"));
  CHECK_EQ(square_name(28), std::string("e4"));
  CHECK_EQ(square_name(63), std::string("h8"));
}

TEST(board_sizes) {
  CHECK((BoardSize{8, 8}.valid()));
  CHECK((BoardSize{1, 1}.valid()));
  CHECK((BoardSize{5, 7}.valid()));
  CHECK(!(BoardSize{0, 8}.valid()));
  CHECK(!(BoardSize{8, 9}.valid()));
  CHECK(!(BoardSize{-1, 4}.valid()));
  CHECK_EQ((BoardSize{8, 8}.mask()), ~std::uint64_t{0});
  CHECK_EQ((BoardSize{5, 5}.mask()), std::uint64_t{0x1F1F1F1F1F});
  CHECK_EQ((BoardSize{1, 2}.mask()), std::uint64_t{0x101});
  CHECK_EQ((BoardSize{0, 2}.mask()), std::uint64_t{0});
  const BoardSize five{5, 5};
  CHECK(five.contains(4, 4));
  CHECK(!five.contains(5, 0));
  CHECK(!five.contains(0, 5));
  CHECK(!five.contains(-1, 0));
  CHECK(five.contains_square(make_square(4, 4)));
  CHECK(!five.contains_square(make_square(5, 4)));
  CHECK(!five.contains_square(-1));
  CHECK(!five.contains_square(64));
  CHECK_EQ(five.square_count(), 25);
  for (int sq = 0; sq < 64; ++sq) {
    CHECK_EQ(five.contains_square(sq), (five.mask() & square_bit(sq)) != 0);
  }
  CHECK_EQ(to_string(BoardSize{7, 6}), std::string("7x6"));
}

TEST(board_size_header_values) {
  CHECK((parse_board_size("8x8") == BoardSize{8, 8}));
  CHECK((parse_board_size("5x5") == BoardSize{5, 5}));
  CHECK((parse_board_size("7x6") == BoardSize{7, 6}));
  for (const char* bad : {"", "8", "8x", "x8", "9x9", "0x8", "8X8", "8x8 ", "88", "10x10", "axb"}) {
    CHECK(!parse_board_size(bad).has_value());
  }
}

TEST(half_turns) {
  CHECK_EQ(half_turn(1, Colour::White), 2);
  CHECK_EQ(half_turn(1, Colour::Black), 3);
  CHECK_EQ(half_turn(0, Colour::White), 0);
  CHECK_EQ(half_turn(0, Colour::Black), 1);
  for (int h = 0; h < 200; ++h) {
    CHECK_EQ(half_turn(turn_of(h), to_move_at(h)), h);
  }
  CHECK(to_move_at(2) == Colour::White);
  CHECK(to_move_at(7) == Colour::Black);
  CHECK_EQ(turn_of(7), 3);
  CHECK(opposite(Colour::White) == Colour::Black);
  CHECK(opposite(Colour::Black) == Colour::White);
  CHECK_EQ(colour_letter(Colour::White), 'w');
  CHECK_EQ(colour_letter(Colour::Black), 'b');
}

TEST(board_coordinates_order_by_timeline_then_half_turn) {
  CHECK((BoardCoord{0, 5} < BoardCoord{1, 2}));
  CHECK((BoardCoord{-1, 9} < BoardCoord{0, 0}));
  CHECK((BoardCoord{0, 2} < BoardCoord{0, 3}));
  CHECK((BoardCoord{3, 4} == BoardCoord{3, 4}));
  CHECK(!(BoardCoord{3, 4} < BoardCoord{3, 4}));
}
