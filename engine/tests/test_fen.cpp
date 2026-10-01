// 5DFEN board strings (notation specification, "5DFEN and custom variants"):
// reading, canonical writing, malformed input, and multiverses built from
// board strings.

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "t5d/error.hpp"
#include "t5d/fen.hpp"
#include "test_framework.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

const std::string kStandardPieces =
    "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*";
const std::string kStandard = "[" + kStandardPieces + ":0:1:w]";
constexpr BoardSize k8x8{8, 8};
constexpr BoardSize k5x5{5, 5};

}  // namespace

// ---- reading one board -------------------------------------------------------------

TEST(fen_reads_the_standard_board) {
  const FenBoard fb = parse_fen_board(kStandard, k8x8);
  CHECK(*fb.board == *standard_board());
  CHECK((fb.timeline == TimelineLabel{TimelineLabel::Sign::None, 0}));
  CHECK_EQ(fb.turn, 1);
  CHECK(fb.to_move == Colour::White);
  CHECK_EQ(fb.half_turn(), 2);
  CHECK_EQ(fb.board->piece_count(), 32);
}

TEST(fen_white_is_upper_case) {
  // The specification's "Rook Tactics I": white king a1 and rook c1, black
  // king e5. (Its grammar's piece_white/piece_black productions have the cases
  // swapped; the prose and every example make white upper case.)
  const FenBoard fb = parse_fen_board("[4k/5/5/5/K1R2:0:1:w]", k5x5);
  CHECK(fb.board->at(0, 0) == Piece(PieceKind::King, Colour::White));
  CHECK(fb.board->at(2, 0) == Piece(PieceKind::Rook, Colour::White));
  CHECK(fb.board->at(4, 4) == Piece(PieceKind::King, Colour::Black));
  CHECK_EQ(fb.board->piece_count(), 3);
}

TEST(fen_rows_run_from_the_top_rank_down) {
  const FenBoard fb = parse_fen_board("[Q2/3/2n:0:1:b]", BoardSize{3, 3});
  CHECK(fb.board->at(0, 2) == Piece(PieceKind::Queen, Colour::White));  // a3: top row
  CHECK(fb.board->at(2, 0) == Piece(PieceKind::Knight, Colour::Black));  // c1: bottom row
  CHECK_EQ(fb.half_turn(), 3);
}

TEST(fen_reads_every_piece_letter) {
  const FenBoard fb = parse_fen_board("[PNBRQKUD/SWCY4/pnbrqkud/swcy4:0:1:w]", BoardSize{8, 4});
  const std::string letters = "PNBRQKUDSWCY";
  for (std::size_t i = 0; i < 12; ++i) {
    const int file = static_cast<int>(i % 8);
    const int white_rank = i < 8 ? 3 : 2;
    const int black_rank = i < 8 ? 1 : 0;
    const auto kind = static_cast<PieceKind>(i);
    CHECK(fb.board->at(file, white_rank) == Piece(kind, Colour::White));
    CHECK(fb.board->at(file, black_rank) == Piece(kind, Colour::Black));
  }
}

TEST(fen_turn_zero_and_black_to_move) {
  const FenBoard fb = parse_fen_board("[" + kStandardPieces + ":0:0:b]", k8x8);
  CHECK_EQ(fb.turn, 0);
  CHECK(fb.to_move == Colour::Black);
  CHECK_EQ(fb.half_turn(), 1);
}

TEST(fen_accepts_the_specification_variants) {
  // '+' before a letter: read as the same piece. A project ruling, not the
  // specification's reading (its '+' extends the piece alphabet, and it
  // defines no '+' piece) - t5d/fen.hpp. Change this test with the ruling.
  const FenBoard plus = parse_fen_board("[+K4/5/5/5/4+k:0:1:w]", k5x5);
  CHECK(plus.board->at(0, 4) == Piece(PieceKind::King, Colour::White));
  CHECK(plus.board->at(4, 0) == Piece(PieceKind::King, Colour::Black));
  // '*' after a kind that does not track it: ignored.
  const FenBoard star = parse_fen_board("[Q*N*B*S*/4/4/C*Y*U*D*:0:1:w]", BoardSize{4, 4});
  CHECK_EQ(star.board->unmoved(), std::uint64_t{0});
  CHECK_EQ(fen_pieces(*star.board), std::string("QNBS/4/4/CYUD"));
  // An unsigned timeline label.
  const FenBoard unsigned_label = parse_fen_board("[5/5/5/5/K4:3:1:w]", k5x5);
  CHECK((unsigned_label.timeline == TimelineLabel{TimelineLabel::Sign::None, 3}));
}

TEST(fen_multi_digit_blanks_are_one_number) {
  // "10" is ten squares - one blank - and no board is ten wide.
  CHECK_THROWS_WITH(parse_fen_board("[10/8/8/8/8/8/8/8:0:1:w]", k8x8), NotationError,
                    "blank \"10\"");
  CHECK_THROWS_WITH(parse_fen_board("[1R10/8/8/8/8/8/8/8:0:1:w]", k8x8), NotationError,
                    "blank \"10\"");
}

TEST(fen_rejects_malformed_strings) {
  const std::string ok_rows = "4k/5/5/5/K1R2";
  const std::vector<std::string> bad = {
      "",                                  // empty
      "4k/5/5/5/K1R2:0:1:w]",              // no '['
      "[4k/5/5/5/K1R2:0:1:w",              // no ']'
      "[",                                 // nothing
      "[]",                                // no fields
      "[4k/5/5/5/K1R2:0:1]",               // three fields
      "[4k/5/5/5/K1R2:0:1:w:x]",           // five fields
      "[4k/5/5/5/K1R2::1:w]",              // empty timeline
      "[4k/5/5/5/K1R2:0::w]",              // empty turn
      "[4k/5/5/5/K1R2:0:1:]",              // empty colour
      "[:0:1:w]",                          // empty piece field
      "[4k/5/5/K1R2:0:1:w]",               // four rows
      "[4k/5/5/5/5/K1R2:0:1:w]",           // six rows
      "[4k/5/5/5/K1R1:0:1:w]",             // short row
      "[4k/5/5/5/K1R3:0:1:w]",             // long row
      "[4k/5/5//K1R2:0:1:w]",              // empty row
      "[4x/5/5/5/K1R2:0:1:w]",             // unknown letter
      "[4A/5/5/5/K1R2:0:1:w]",             // unknown letter
      "[*4k/5/5/5/K1R2:0:1:w]",            // '*' without a piece
      "[4*k/5/5/5/K1R2:0:1:w]",            // '*' after a blank
      "[4k**/5/5/5/K1R2:0:1:w]",           // two '*'
      "[4+/5/5/5/K1R2:0:1:w]",             // '+' at the end of a row
      "[+4k/5/5/5/K1R2:0:1:w]",            // '+' before a blank
      "[++k4/5/5/5/K1R2:0:1:w]",           // '+' twice
      "[04k/5/5/5/K1R2:0:1:w]",            // leading zero blank
      "[0k4/5/5/5/K1R2:0:1:w]",            // a blank of 0
      "[4k/5/5/5/K1R2 :0:1:w]",            // a space
      "[4k/5/5/5/K1R2:0:1:w ]",            // a space
      " [4k/5/5/5/K1R2:0:1:w]",            // a leading space
      "[4k/5/5/5/K1R2:0:1:w]\n",           // a trailing newline
      "[4k/5/5/5/K1R2:+:1:w]",             // sign without digits
      "[4k/5/5/5/K1R2:01:1:w]",            // leading zero label
      "[4k/5/5/5/K1R2:L0:1:w]",            // not a label
      "[4k/5/5/5/K1R2:0:-1:w]",            // negative turn
      "[4k/5/5/5/K1R2:0:01:w]",            // leading zero turn
      "[4k/5/5/5/K1R2:0:1.5:w]",           // not an integer
      "[4k/5/5/5/K1R2:0:T1:w]",            // not a number
      "[4k/5/5/5/K1R2:0:1:W]",             // colour upper case
      "[4k/5/5/5/K1R2:0:1:white]",         // colour spelled out
      "[4k/5/5/5/K1R2:0:9999999:w]",       // turn out of range
      "[4k/5/5/5/K1R2/5/5/5/5:0:1:w]",     // nine rows
      "[4k/5/5/5/K1R2:0:1:w][5/5/5/5/5:0:1:b]",  // two board strings
  };
  for (const std::string& text : bad) {
    bool threw = false;
    try {
      (void)parse_fen_board(text, k5x5);
    } catch (const NotationError&) {
      threw = true;
    }
    if (!threw) t5d_test::record_failure(__FILE__, __LINE__, "accepted: \"" + text + "\"");
  }
  CHECK_NOTHROW(parse_fen_board("[" + ok_rows + ":0:1:w]", k5x5));
}

TEST(fen_error_messages_say_what_and_where) {
  CHECK_THROWS_WITH(parse_fen_board("[4x/5/5/5/K1R2:0:1:w]", k5x5), NotationError,
                    "unknown piece letter 'x'");
  CHECK_THROWS_WITH(parse_fen_board("[4x/5/5/5/K1R2:0:1:w]", k5x5), NotationError,
                    "at offset 2");
  CHECK_THROWS_WITH(parse_fen_board("[4k/5/5/5/K1R2:0:1]", k5x5), NotationError,
                    "expected 4 fields");
  CHECK_THROWS_WITH(parse_fen_board("[4k/5/5/K1R2:0:1:w]", k5x5), NotationError,
                    "has 4 rows; a 5x5 board has 5");
  CHECK_THROWS_WITH(parse_fen_board("[4k/5/5/5/K1R1:0:1:w]", k5x5), NotationError,
                    "row 5 (from the top) describes 4 squares");
  CHECK_THROWS_WITH(parse_fen_board("[4k/5/5/5/K1R2:0:1:W]", k5x5), NotationError,
                    "the colour is 'w' or 'b'");
  CHECK_THROWS_WITH(parse_fen_board("[4k/5/5/5/K1R2:0:1:w]", BoardSize{0, 5}),
                    std::invalid_argument, "outside");
}

TEST(fen_board_size_is_the_given_size) {
  // Valid for 5x5, not for 8x8 (and the other way round).
  CHECK_THROWS_AS(parse_fen_board("[4k/5/5/5/K1R2:0:1:w]", k8x8), NotationError);
  CHECK_THROWS_AS(parse_fen_board(kStandard, k5x5), NotationError);
  CHECK((infer_fen_board_size(kStandard) == k8x8));
  CHECK((infer_fen_board_size("[4k/5/5/5/K1R2:0:1:w]") == k5x5));
  CHECK((infer_fen_board_size("[3/3:0:1:w]") == BoardSize{3, 2}));
  CHECK_THROWS_AS(infer_fen_board_size("[4k/5/4:0:1:w]"), NotationError);
  CHECK_THROWS_AS(infer_fen_board_size("[/:0:1:w]"), NotationError);
}

// ---- writing ----------------------------------------------------------------------

TEST(fen_writes_canonical_strings) {
  const std::vector<std::pair<std::string, BoardSize>> canonical = {
      {kStandard, k8x8},
      {"[4k/5/5/5/K1R2:0:1:w]", k5x5},
      {"[" + kStandardPieces + ":-0:1:w]", k8x8},
      {"[" + kStandardPieces + ":+0:1:w]", k8x8},
      {"[" + kStandardPieces + ":+3:12:b]", k8x8},
      {"[" + kStandardPieces + ":-7:0:b]", k8x8},
      {"[w*W*1C/3Y/4/S2y:0:4:b]", BoardSize{4, 4}},
      {"[1:0:1:w]", BoardSize{1, 1}},
      {"[K:0:1:w]", BoardSize{1, 1}},
  };
  for (const auto& [text, size] : canonical) {
    const FenBoard fb = parse_fen_board(text, size);
    const bool even = implied_even(fb.timeline).value_or(false);
    const int p = timeline_position(fb.timeline, even);
    CHECK_EQ(fen_board_string(*fb.board, p, fb.half_turn(), even), text);
  }
}

TEST(fen_writing_normalises_the_variants) {
  const FenBoard fb = parse_fen_board("[+R*+n*2/4/4/1q*1K:2:7:b]", BoardSize{4, 4});
  // '+' dropped, '*' kept on the rook and king only, labels signed.
  CHECK_EQ(fen_board_string(*fb.board, 2, fb.half_turn(), false), std::string("[R*n2/4/4/1q1K:+2:7:b]"));
  CHECK_EQ(fen_board_string(*fb.board, -3, fb.half_turn(), true), std::string("[R*n2/4/4/1q1K:-2:7:b]"));
}

TEST(fen_round_trips_random_boards) {
  t5d_test::Rng rng(5);
  for (int round = 0; round < 300; ++round) {
    const BoardSize size = t5d_test::random_size(rng);
    const BoardRef board = t5d_test::random_board(rng, size, rng.between(0, 100));
    const bool even = rng.chance(50);
    const int p = rng.between(-9, 9);
    const int h = rng.between(0, 99);
    const std::string text = fen_board_string(*board, p, h, even);
    const FenBoard fb = parse_fen_board(text, size);
    CHECK(*fb.board == *board);
    CHECK_EQ(timeline_position(fb.timeline, even), p);
    CHECK_EQ(fb.half_turn(), h);
    CHECK_EQ(fen_board_string(*fb.board, p, h, even), text);
  }
}

TEST(fen_parser_survives_random_mutations) {
  // Mutated board strings are either rejected with a NotationError (any other
  // exception fails the test) or read into a board whose canonical string
  // reads back to the same board.
  t5d_test::Rng rng(55);
  const std::vector<std::string> seeds = {
      kStandard, "[4k/5/5/5/K1R2:0:1:w]", "[" + kStandardPieces + ":-0:1:w]",
      "[w*W*1C/3Y/4/S2y:+3:4:b]", "[+R*+n*2/4/4/1q*1K:2:7:b]"};
  const std::string alphabet = "[]:/*+-0123456789PNBRQKUDSWCYpnbrqkudswcyxA \n";
  const BoardSize sizes[] = {k8x8, k5x5, BoardSize{4, 4}};
  int accepted = 0;
  for (int round = 0; round < 6000; ++round) {
    std::string text = seeds[static_cast<std::size_t>(rng.below(static_cast<int>(seeds.size())))];
    for (int edits = rng.between(1, 3); edits > 0; --edits) {
      const auto at = static_cast<std::size_t>(rng.below(static_cast<int>(text.size()) + 1));
      const char c = alphabet[static_cast<std::size_t>(rng.below(static_cast<int>(alphabet.size())))];
      switch (rng.below(3)) {
        case 0:
          if (at < text.size()) text[at] = c;
          break;
        case 1:
          text.insert(at, 1, c);
          break;
        default:
          if (at < text.size()) text.erase(at, 1);
          break;
      }
    }
    for (const BoardSize size : sizes) {
      try {
        const FenBoard fb = parse_fen_board(text, size);
        ++accepted;
        const bool even = implied_even(fb.timeline).value_or(false);
        const int p = timeline_position(fb.timeline, even);
        const std::string canonical = fen_board_string(*fb.board, p, fb.half_turn(), even);
        const FenBoard again = parse_fen_board(canonical, size);
        CHECK(*again.board == *fb.board);
        CHECK_EQ(again.half_turn(), fb.half_turn());
        CHECK_EQ(fen_board_string(*again.board, p, again.half_turn(), even), canonical);
      } catch (const NotationError&) {
        // rejected cleanly
      }
    }
  }
  CHECK(accepted > 200);  // the mutations do not merely break every string
}

TEST(fen_standard_board) {
  const BoardRef board = standard_board();
  CHECK_EQ(fen_pieces(*board), kStandardPieces);
  CHECK_EQ(board->piece_count(), 32);
  CHECK_EQ(std::popcount(board->unmoved()), 22);  // 16 pawns, 4 rooks, 2 kings
  CHECK(board.get() == standard_board().get());  // one shared instance
}

// ---- multiverses from board strings ------------------------------------------------

TEST(fen_multiverse_standard) {
  const Multiverse mv = multiverse_from_fen(std::vector<std::string>{kStandard}, k8x8, Parity::Infer);
  CHECK(mv == standard_start());
  CHECK(!mv.even());
  CHECK_EQ(mv.timeline_count(), 1);
  CHECK(mv.to_move() == Colour::White);
  CHECK(!mv.timeline(0)->creator().has_value());
  CHECK_EQ(mv.zobrist(), standard_start().zobrist());
}

TEST(fen_multiverse_two_timelines) {
  const std::vector<std::string> boards = {"[" + kStandardPieces + ":-0:1:w]",
                                           "[" + kStandardPieces + ":+0:1:w]"};
  const Multiverse mv = multiverse_from_fen(boards, k8x8, Parity::Infer);
  CHECK(mv == standard_two_timelines_start());
  CHECK(mv.even());
  CHECK_EQ(mv.min_timeline(), -1);
  CHECK_EQ(mv.max_timeline(), 0);
  CHECK(!mv.timeline(-1)->creator().has_value());
  CHECK(!mv.timeline(0)->creator().has_value());
  CHECK(multiverse_from_fen(boards, k8x8, Parity::Even) == mv);
  CHECK_THROWS_AS(multiverse_from_fen(boards, k8x8, Parity::Odd), NotationError);
}

TEST(fen_multiverse_parity_conflicts) {
  const std::string zero = "[4k/5/5/5/K1R2:0:1:w]";
  const std::string plus_zero = "[4k/5/5/5/K1R2:+0:1:w]";
  const std::string plus_one = "[4k/5/5/5/K1R2:+1:1:w]";
  CHECK_THROWS_AS(multiverse_from_fen(std::vector<std::string>{zero}, k5x5, Parity::Even),
                  NotationError);
  CHECK_THROWS_AS(multiverse_from_fen(std::vector<std::string>{plus_zero}, k5x5, Parity::Odd),
                  NotationError);
  CHECK_THROWS_WITH(
      multiverse_from_fen(std::vector<std::string>{zero, plus_zero}, k5x5, Parity::Infer),
      NotationError, "odd number");
  CHECK_THROWS_WITH(multiverse_from_fen(std::vector<std::string>{plus_one}, k5x5, Parity::Infer),
                    NotationError, "cannot tell");
  // Declared, a lone "+1" is fine either way and means the same position.
  const Multiverse odd = multiverse_from_fen(std::vector<std::string>{plus_one}, k5x5, Parity::Odd);
  const Multiverse even = multiverse_from_fen(std::vector<std::string>{plus_one}, k5x5, Parity::Even);
  CHECK_EQ(odd.min_timeline(), 1);
  CHECK_EQ(even.min_timeline(), 1);
}

TEST(fen_multiverse_rejects_impossible_board_sets) {
  const auto at = [](const std::string& label, int turn, char colour) {
    return "[4k/5/5/5/K1R2:" + label + ":" + std::to_string(turn) + ":" + colour + "]";
  };
  const auto build = [](const std::vector<std::string>& boards) {
    return multiverse_from_fen(boards, k5x5, Parity::Infer);
  };
  CHECK_THROWS_WITH(build({at("0", 1, 'w'), at("0", 1, 'w')}), NotationError, "both for");
  CHECK_THROWS_WITH(build({at("0", 1, 'w'), at("0", 2, 'w')}), NotationError, "without gaps");
  CHECK_THROWS_WITH(build({at("0", 1, 'w'), at("+2", 1, 'w')}), NotationError, "contiguous");
  CHECK_THROWS_WITH(build({at("-2", 1, 'w'), at("0", 1, 'w')}), NotationError, "contiguous");
  CHECK_THROWS_WITH(build({at("0", 1, 'w'), "[5/5/5:0:1:w]"}), NotationError, "board string #2");
  CHECK_THROWS_AS(build({}), NotationError);
  CHECK_THROWS_AS(multiverse_from_fen(std::vector<std::string>{at("0", 1, 'w')}, BoardSize{9, 9},
                                      Parity::Odd),
                  std::invalid_argument);
}

TEST(fen_multiverse_several_boards_per_timeline_in_any_order) {
  const auto at = [](const std::string& pieces, const std::string& label, int turn, char colour) {
    return "[" + pieces + ":" + label + ":" + std::to_string(turn) + ":" + colour + "]";
  };
  std::vector<std::string> boards = {
      at("4k/5/5/5/K1R2", "0", 1, 'w'), at("4k/5/5/1K3/2R2", "0", 1, 'b'),
      at("5/4k/5/1K3/2R2", "0", 2, 'w'), at("5/4k/5/1K3/2R2", "+1", 1, 'b'),
      at("5/4k/5/5/1KR2", "+1", 2, 'w'), at("5/3k1/5/5/1KR2", "-1", 2, 'w'),
  };
  const Multiverse first = multiverse_from_fen(boards, k5x5, Parity::Odd);
  std::reverse(boards.begin(), boards.end());
  std::rotate(boards.begin(), boards.begin() + 2, boards.end());
  const Multiverse second = multiverse_from_fen(boards, k5x5, Parity::Odd);
  CHECK(first == second);
  CHECK_EQ(first.board_count(), static_cast<std::size_t>(6));
  CHECK_EQ(first.timeline(0)->first_half_turn(), 2);
  CHECK_EQ(first.timeline(0)->length(), 3);
  CHECK_EQ(first.timeline(1)->first_half_turn(), 3);
  CHECK_EQ(first.timeline(-1)->first_half_turn(), 4);
  // The earliest head is timeline 0's at T2 white (h = 4) and timeline -1's
  // at h = 4: white to move.
  CHECK(first.to_move() == Colour::White);
}

TEST(fen_multiverse_turn_zero) {
  const std::vector<std::string> boards = {"[" + kStandardPieces + ":0:0:b]",
                                           "[" + kStandardPieces + ":0:1:w]"};
  const Multiverse mv = multiverse_from_fen(boards, k8x8, Parity::Infer);
  CHECK_EQ(mv.timeline(0)->first_half_turn(), 1);
  CHECK_EQ(mv.timeline(0)->length(), 2);
  CHECK(mv.to_move() == Colour::White);
}

TEST(fen_multiverse_infers_creators_from_labels) {
  const auto at = [](const std::string& label) { return "[4k/5/5/5/K1R2:" + label + ":3:w]"; };
  const Multiverse odd = multiverse_from_fen(
      std::vector<std::string>{at("-2"), at("-1"), at("0"), at("+1"), at("+2"), at("+3")}, k5x5,
      Parity::Infer);
  CHECK(!odd.timeline(0)->creator().has_value());
  CHECK_EQ(odd.timeline(0)->creation_order(), 0);
  CHECK(odd.timeline(1)->creator() == Colour::White);
  CHECK_EQ(odd.timeline(3)->creation_order(), 3);
  CHECK(odd.timeline(-2)->creator() == Colour::Black);
  CHECK_EQ(odd.timeline(-2)->creation_order(), 2);
  CHECK_EQ(odd.created_by(Colour::White), 3);
  CHECK_EQ(odd.created_by(Colour::Black), 2);

  const Multiverse even = multiverse_from_fen(
      std::vector<std::string>{at("-1"), at("-0"), at("+0"), at("+1")}, k5x5, Parity::Infer);
  CHECK(!even.timeline(-1)->creator().has_value());
  CHECK(!even.timeline(0)->creator().has_value());
  CHECK(even.timeline(1)->creator() == Colour::White);
  CHECK_EQ(even.timeline(1)->creation_order(), 1);
  CHECK(even.timeline(-2)->creator() == Colour::Black);
  CHECK_EQ(even.timeline(-2)->creation_order(), 1);
  CHECK_EQ(even.created_by(Colour::White), 1);
  CHECK_EQ(even.created_by(Colour::Black), 1);
  CHECK(!even.timeline(1)->origin().has_value());
}

TEST(fen_multiverse_infers_to_move_only_where_the_boards_settle_it) {
  const auto at = [](const std::string& label, int turn, char colour) {
    return "[4k/5/5/5/K1R2:" + label + ":" + std::to_string(turn) + ":" + colour + "]";
  };
  const auto parse = [](const std::vector<std::string>& boards) {
    return multiverse_from_fen(boards, k5x5, Parity::Odd);
  };
  // The earliest head is on the initial timeline: it is the present.
  CHECK(parse({at("0", 3, 'b'), at("+1", 4, 'w')}).to_move() == Colour::Black);
  CHECK(parse({at("0", 3, 'b'), at("0", 4, 'w'), at("-1", 4, 'w')}).to_move() == Colour::White);
  // An earlier head on a created timeline, of the same colour as the initial
  // timeline's: the present is one of the two, black to move either way (white
  // has just branched into the past).
  CHECK(parse({at("0", 5, 'b'), at("+1", 3, 'b')}).to_move() == Colour::Black);
  // No initial timeline, one head: it is the present.
  CHECK(parse({at("+1", 1, 'w')}).to_move() == Colour::White);

  // The reviewers' position: white has created +1 and +2, black none, so +2
  // is inactive under the activity rule (stage E2) and the present is T5 on 0
  // and +1 - white to move. The earliest head of all, +2's T3 black board, is
  // not the present. Until E2 implements activity, the parser refuses to
  // guess (it used to say black), and the caller's answer is taken.
  const std::vector<std::string> inactive_earliest = {
      at("0", 4, 'w'), at("0", 4, 'b'), at("0", 5, 'w'),
      at("+1", 4, 'b'), at("+1", 5, 'w'),
      at("+2", 3, 'b')};
  CHECK_THROWS_WITH(parse(inactive_earliest), std::invalid_argument,
                    "cannot tell whose turn it is");
  CHECK_THROWS_WITH(parse(inactive_earliest), std::invalid_argument,
                    "timeline +2 at turn 3, black to move");
  const Multiverse given =
      multiverse_from_fen(inactive_earliest, k5x5, Parity::Odd, 0, Colour::White);
  CHECK(given.to_move() == Colour::White);
  CHECK_EQ(given.created_by(Colour::White), 2);
  CHECK_EQ(given.created_by(Colour::Black), 0);
  // Both colours, no initial timeline.
  CHECK_THROWS_AS(parse({at("+1", 1, 'w'), at("+2", 1, 'b')}), std::invalid_argument);

  // A given player is taken as given, even where the boards would settle it;
  // the fingerprint does not depend on it, the key does.
  const Multiverse black_start = multiverse_from_fen(std::vector<std::string>{kStandard}, k8x8,
                                                     Parity::Odd, 0, Colour::Black);
  CHECK(black_start.to_move() == Colour::Black);
  CHECK_NE(black_start.zobrist(), standard_start().zobrist());
  CHECK(multiverse_from_fen(std::vector<std::string>{kStandard}, k8x8, Parity::Odd, 0,
                            Colour::White) == standard_start());
}

TEST(fen_multiverse_keeps_the_ruleset) {
  const Multiverse mv =
      multiverse_from_fen(std::vector<std::string>{kStandard}, k8x8, Parity::Odd, 7);
  CHECK_EQ(mv.ruleset(), 7u);
  CHECK_NE(mv.zobrist(), standard_start().zobrist());
}
