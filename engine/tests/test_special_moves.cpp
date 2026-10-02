// Castling (S1-S6), en passant (S7), promotion (S8) and the Turn Zero board
// (Q8), with their parameters (Q15-Q17, Q21).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/state.hpp"
#include "t5d/zobrist.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using t5d_test::at;
using t5d_test::contains;
using t5d_test::game;
using t5d_test::piece_texts;
using t5d_test::play;

namespace {

constexpr BoardSize k8x8{8, 8};
constexpr BoardSize k7x7{7, 7};
constexpr BoardSize k5x5{5, 5};
constexpr BoardSize k4x4{4, 4};

// The state read back from its canonical 5DFEN string (which records no
// en-passant square), with the same rules and player to move.
GameState read_back(const GameState& s) {
  const std::string canonical = canonical_state_string(s.multiverse());
  std::vector<std::string> boards;
  for (std::size_t i = 0; i < canonical.size();) {
    const std::size_t end = canonical.find(']', i);
    boards.push_back(canonical.substr(i, end - i + 1));
    i = end + 1;
  }
  return game_from_fen(boards, s.multiverse().board_size(),
                       s.multiverse().even() ? Parity::Even : Parity::Odd, s.rules(), s.to_move());
}

GameState board8(const std::string& pieces, char colour = 'w', const RuleOptions& r = {}) {
  return game({"[" + pieces + ":0:1:" + std::string(1, colour) + "]"}, k8x8, r);
}

std::vector<std::string> king_moves(const GameState& s, const char* square = "e1") {
  return piece_texts(s, 0, square);
}

bool castles(const GameState& s, const char* text) {
  const std::optional<Move> m = t5d_test::find_move(s, text);
  return m.has_value() && m->castle();
}

const std::string kBothWays = "r*3k*2r*/8/8/8/8/8/8/R*3K*2R*";

}  // namespace

// ---- castling -------------------------------------------------------------------------------------

TEST(castling_both_ways_on_one_board) {
  // S1: the king moves two squares towards an unmoved rook of its colour on
  // its board; the rook goes to the square the king crossed. One new board.
  const GameState s = board8(kBothWays);
  CHECK(castles(s, "(0T1)Ke1(0T1)g1"));
  CHECK(castles(s, "(0T1)Ke1(0T1)c1"));
  CHECK_EQ(king_moves(s).size(), std::size_t{5 + 2});
  GameState o_o = s;
  const Move m = play(o_o, "(0T1)Ke1(0T1)g1");
  CHECK_EQ(m.aux_square, make_square(7, 0));
  const Board* after = o_o.multiverse().head(0);
  // Both pieces lose their unmoved flags; the other rook keeps its own.
  CHECK_EQ(fen_pieces(*after), std::string("r*3k*2r*/8/8/8/8/8/8/R*4RK1"));
  CHECK_EQ(o_o.multiverse().board_count(), std::size_t{2});
  GameState o_o_o = s;
  play(o_o_o, "(0T1)Ke1(0T1)c1");
  CHECK_EQ(fen_pieces(*o_o_o.multiverse().head(0)), std::string("r*3k*2r*/8/8/8/8/8/8/2KR3R*"));
  // Black castles the same way on its board.
  const GameState b = board8(kBothWays, 'b');
  CHECK(castles(b, "(0T1)Ke8(0T1)g8"));
  CHECK(castles(b, "(0T1)Ke8(0T1)c8"));
}

TEST(castling_needs_both_pieces_unmoved_and_the_way_clear) {
  // S2: neither the king nor that rook may have moved; the squares between
  // them must be empty (b1 too, which the king does not cross).
  CHECK(!castles(board8("8/8/8/8/8/8/8/R3K2R"), "(0T1)Ke1(0T1)g1"));
  CHECK(!castles(board8("8/8/8/8/8/8/8/R*3K2R*"), "(0T1)Ke1(0T1)g1"));
  CHECK(!castles(board8("8/8/8/8/8/8/8/R*3K*2R"), "(0T1)Ke1(0T1)g1"));
  CHECK(castles(board8("8/8/8/8/8/8/8/R*3K*2R"), "(0T1)Ke1(0T1)c1"));
  const GameState knight = board8("8/8/8/8/8/8/8/R*N2K*2R*");
  CHECK(!castles(knight, "(0T1)Ke1(0T1)c1"));
  CHECK(castles(knight, "(0T1)Ke1(0T1)g1"));
  // Q21's alternative: only the two destination squares must be free.
  RuleOptions loose;
  loose.castling_empty_between = false;
  CHECK(castles(board8("8/8/8/8/8/8/8/R*N2K*2R*", 'w', loose), "(0T1)Ke1(0T1)c1"));
  CHECK(!castles(board8("8/8/8/8/8/8/8/R*1N1K*2R*", 'w', loose), "(0T1)Ke1(0T1)c1"));
}

TEST(castling_is_barred_by_attacks_on_the_castling_board) {
  // S2 (the developer's ruling): barred if the king's square or the square
  // it crosses is attacked by an enemy piece on the castling board.
  const auto rook_on = [](const char* rank5) {
    return board8(std::string("8/8/8/") + rank5 + "/8/8/8/R*3K*2R*");
  };
  const GameState e = rook_on("4r3");  // attacks e1: no castling at all
  CHECK(!castles(e, "(0T1)Ke1(0T1)g1"));
  CHECK(!castles(e, "(0T1)Ke1(0T1)c1"));
  const GameState f = rook_on("5r2");  // attacks f1, crossed on the king side
  CHECK(!castles(f, "(0T1)Ke1(0T1)g1"));
  CHECK(castles(f, "(0T1)Ke1(0T1)c1"));
  const GameState d = rook_on("3r4");  // attacks d1, crossed on the queen side
  CHECK(castles(d, "(0T1)Ke1(0T1)g1"));
  CHECK(!castles(d, "(0T1)Ke1(0T1)c1"));
  // S3: no castling-specific test of the king's destination (the turn-level
  // royal safety test, C4, covers it) - unless the ruleset asks for one.
  const GameState g = rook_on("6r1");
  CHECK(castles(g, "(0T1)Ke1(0T1)g1"));
  RuleOptions dest;
  dest.castling_destination_test = true;
  CHECK(!castles(board8("8/8/8/6r1/8/8/8/R*3K*2R*", 'w', dest), "(0T1)Ke1(0T1)g1"));
  // Q21: an attacked square the rook crosses (b1) does not bar castling by
  // default; it does under the rook-path option.
  const GameState b = rook_on("1r6");
  CHECK(castles(b, "(0T1)Ke1(0T1)c1"));
  RuleOptions path;
  path.castling_rook_path_test = true;
  CHECK(!castles(board8("8/8/8/1r6/8/8/8/R*3K*2R*", 'w', path), "(0T1)Ke1(0T1)c1"));
  CHECK(castles(board8("8/8/8/1r6/8/8/8/R*3K*2R*", 'w', path), "(0T1)Ke1(0T1)g1"));
}

TEST(castling_attack_test_covers_every_piece_within_the_board) {
  const auto barred = [](const char* pieces) {
    return !castles(board8(pieces), "(0T1)Ke1(0T1)g1");
  };
  CHECK(barred("8/8/8/8/8/8/3p4/R*3K*2R*"));   // pawn d2 attacks e1
  CHECK(barred("8/8/8/8/8/8/6p1/R*3K*2R*"));   // pawn g2 attacks f1
  CHECK(!barred("8/8/8/8/8/8/7p/R*3K*2R*"));   // pawn h2 attacks g1 only (S3)
  CHECK(barred("8/8/8/8/8/8/6w1/R*3K*2R*"));   // brawn g2: the pawn's in-board captures
  CHECK(barred("8/8/8/8/8/6n1/8/R*3K*2R*"));   // knight g3 attacks f1
  CHECK(barred("8/8/8/8/8/8/5k2/R*3K*2R*"));   // king f2
  CHECK(barred("8/8/8/8/8/8/5c2/R*3K*2R*"));   // common king f2
  CHECK(barred("8/8/8/8/8/2b5/8/R*3K*2R*"));   // bishop c3 attacks e1
  CHECK(barred("8/8/8/8/8/2y5/8/R*3K*2R*"));   // royal queen c3
  CHECK(barred("8/8/8/8/8/2s5/8/R*3K*2R*"));   // princess c3
  CHECK(barred("8/8/8/8/4q3/8/8/R*3K*2R*"));   // queen e4, down the file
  CHECK(!barred("8/8/8/8/4q3/8/4P3/R*3K*2R*"));  // ... unless something stands between
  // Unicorns and dragons cannot move within one board: no attack here.
  CHECK(!barred("8/8/8/8/8/8/5u2/R*3K*2R*"));
  CHECK(!barred("8/8/8/8/8/8/5d2/R*3K*2R*"));
}

TEST(castling_attacks_from_other_boards_do_not_bar_it) {
  // S2: "attacks through time or from other timelines do not bar castling".
  // A black rook on +1's playable board at f1 could reach f1 of 0's next
  // board; the developer's ruling ignores it (R6, register H40: the per-board
  // test is the rule).
  const GameState s = game({"[8/8/8/8/8/8/8/R*3K*2R*:0:1:w]", "[8/8/8/8/8/8/8/5r2:+1:1:b]",
                            "[8/8/8/8/8/8/8/8:+1:1:w]"},
                           k8x8);
  CHECK(castles(s, "(0T1)Ke1(0T1)g1"));
  CHECK(castles(s, "(0T1)Ke1(0T1)c1"));
}

TEST(castling_geometry_with_the_king_off_the_e_file) {
  // S5: Simple - No Queens (rules reference V3; kings on d1 and d7), the
  // other pieces removed. O-O: Kd1-f1, Rg1-e1; O-O-O: Kd1-b1, Ra1-c1.
  const GameState s = game({"[r*2k*2r*/7/7/7/7/7/R*2K*2R*:0:1:w]"}, k7x7);
  GameState right = s;
  play(right, "(0T1)Kd1(0T1)f1");
  CHECK_EQ(fen_pieces(*right.multiverse().head(0)), std::string("r*2k*2r*/7/7/7/7/7/R*3RK1"));
  GameState left = s;
  play(left, "(0T1)Kd1(0T1)b1");
  CHECK_EQ(fen_pieces(*left.multiverse().head(0)), std::string("r*2k*2r*/7/7/7/7/7/1KR3R*"));
}

TEST(castling_with_a_rook_two_or_one_squares_away) {
  // S5: a rook at least two squares away; at exactly two the king lands on
  // the rook's square and the rook on the square the king crossed (inferred).
  const GameState two = board8("8/8/8/8/8/8/8/4K*1R*1");
  REQUIRE(castles(two, "(0T1)Ke1(0T1)g1"));
  GameState after = two;
  play(after, "(0T1)Ke1(0T1)g1");
  CHECK_EQ(fen_pieces(*after.multiverse().head(0)), std::string("8/8/8/8/8/8/8/5RK1"));
  RuleOptions three;
  three.castling_rook_two_away = false;
  CHECK(!castles(board8("8/8/8/8/8/8/8/4K*1R*1", 'w', three), "(0T1)Ke1(0T1)g1"));
  // V4: a rook next to its king never castles (Very Small's K*R*).
  const GameState adjacent = game({"[4/4/4/K*R*BN:0:1:w]"}, BoardSize{4, 4});
  for (const Move& m : t5d_test::moves_of(adjacent)) CHECK(!m.castle());
}

TEST(castling_only_by_a_royal_king) {
  // S4: a common king never castles (5DFEN cannot even mark it unmoved); a
  // royal queen moves like a queen and never castles (Reversed Royalty).
  for (const char* pieces : {"8/8/8/8/8/8/8/R*3C2R*", "8/8/8/8/8/8/8/R*3Y2R*"}) {
    for (const Move& m : t5d_test::moves_of(board8(pieces))) CHECK(!m.castle());
  }
}

TEST(castling_rank_and_optional_board_options) {
  // S5's note: "the pieces need not be on the player's first rank".
  const GameState second_rank = board8("8/8/8/8/8/8/R*3K*2R*/8");
  CHECK(castles(second_rank, "(0T1)Ke2(0T1)g2"));
  RuleOptions first;
  first.castling_any_rank = false;
  CHECK(!castles(board8("8/8/8/8/8/8/R*3K*2R*/8", 'w', first), "(0T1)Ke2(0T1)g2"));
  CHECK(castles(board8(kBothWays, 'w', first), "(0T1)Ke1(0T1)g1"));
  // S1: on a playable board - including an optional one (0 is ahead of the
  // present, which -1 sets at T1) - unless the ruleset says otherwise.
  const std::vector<std::string> boards = {"[8/8/8/8/8/8/8/8:0:1:w]", "[8/8/8/8/8/8/8/8:0:1:b]",
                                           "[8/8/8/8/8/8/8/R*3K*2R*:0:2:w]",
                                           "[8/8/8/8/8/8/8/8:-1:1:w]"};
  const GameState optional = game(boards, k8x8);
  CHECK(!optional.mandatory(0));
  CHECK(castles(optional, "(0T2)Ke1(0T2)g1"));
  RuleOptions mandatory_only;
  mandatory_only.castling_on_optional_boards = false;
  CHECK(!castles(game(boards, k8x8, mandatory_only), "(0T2)Ke1(0T2)g1"));
}

// ---- en passant (S7; Q16) -------------------------------------------------------------------------

TEST(en_passant_within_a_board) {
  // S7: Black's d7-d5 passes d6 next to White's e5 pawn: on the board it
  // creates, White may take en passant, removing the pawn on d5.
  GameState s = board8("8/3p*4/8/4P3/8/8/8/8", 'b');
  const Move two = play(s, "(0T1)Pd7(0T1)d5");
  CHECK(two.double_step());
  CHECK_EQ(s.multiverse().head(0)->en_passant(), make_square(3, 5));
  s.submit();
  const std::optional<Move> ep = t5d_test::find_move(s, "(0T2)Pe5(0T2)d6");
  REQUIRE(ep.has_value());
  CHECK(ep->en_passant());
  CHECK(ep->capture());
  CHECK(ep->captured == Piece(PieceKind::Pawn, Colour::Black));
  play(s, "(0T2)Pe5(0T2)d6");
  CHECK_EQ(fen_pieces(*s.multiverse().head(0)), std::string("8/8/3P4/8/8/8/8/8"));
  CHECK_EQ(s.multiverse().head(0)->en_passant(), kNoSquare);
}

TEST(en_passant_only_right_after_and_only_when_takeable) {
  // No enemy pawn beside the arrival square: no en-passant square is recorded
  // (equal positions keep equal keys).
  GameState lone = board8("8/3p*4/8/7P/8/8/8/8", 'b');
  play(lone, "(0T1)Pd7(0T1)d5");
  CHECK_EQ(lone.multiverse().head(0)->en_passant(), kNoSquare);
  // Not taken at once, it is gone: the next board has none.
  GameState late = board8("8/3p*4/8/4P3/8/8/8/8", 'b');
  play(late, "(0T1)Pd7(0T1)d5");
  late.submit();
  play(late, "(0T2)Pe5(0T2)e6");
  late.submit();
  CHECK_EQ(late.multiverse().head(0)->en_passant(), kNoSquare);
}

TEST(en_passant_not_recorded_after_a_promoting_double_step) {
  // On a 4x4 board Black's flagged pawn on b3 - its second rank - double-steps
  // to b1, its last rank, and promotes (S8); White's pawn stands on a1 beside
  // the arrival square. What arrived is a queen, which no pawn takes en
  // passant (S7), so no en-passant square is recorded: equal positions keep
  // equal keys.
  GameState s = game({"[4/1p*2/4/P3:0:1:b]"}, k4x4);
  REQUIRE(s.to_move() == Colour::Black);
  const Move m = play(s, "(0T1)Pb3(0T1)b1=Q");
  CHECK(m.double_step());
  CHECK(m.promotes());
  CHECK_EQ(s.multiverse().head(0)->en_passant(), kNoSquare);
  s.submit();
  for (const Move& w : t5d_test::moves_of(s)) CHECK(!w.en_passant());
  // A brawn that does not promote (Q17's brawns_promote = false) arrives as a
  // brawn: the square is recorded, and White's pawn may take en passant.
  RuleOptions no;
  no.brawns_promote = false;
  GameState b = game({"[4/1w*2/4/P3:0:1:b]"}, k4x4, no);
  play(b, "(0T1)Wb3(0T1)b1");
  CHECK_EQ(b.multiverse().head(0)->en_passant(), make_square(1, 1));
  b.submit();
  const std::optional<Move> ep = t5d_test::find_move(b, "(0T2)Pa1(0T2)b2");
  REQUIRE(ep.has_value());
  CHECK(ep->en_passant());
  CHECK(ep->captured == Piece(PieceKind::Brawn, Colour::Black));
}

TEST(en_passant_right_leaves_the_key_once_its_board_is_history) {
  // Black's d7-d5 creates (0T2, white) with en passant on d6 (White's e5 pawn
  // stands beside d5). 5DFEN cannot record the right, so the state reached in
  // play and the same boards read back from its canonical string differ in
  // key while the right can still be used. Once White has played that board
  // the right has lapsed - it is taken only from the board the double step
  // created (S7), and a history board is never moved from (L5) - and the keys
  // agree again: the key no longer counts it.
  GameState s = board8("4k3/3p*4/8/4P3/8/8/8/4K3", 'b');
  play(s, "(0T1)Pd7(0T1)d5");
  s.submit();
  REQUIRE_EQ(s.multiverse().head(0)->en_passant(), make_square(3, 5));
  CHECK_EQ(s.multiverse().zobrist(), zobrist::key_from_scratch(s.multiverse()));
  CHECK(fingerprint(read_back(s).multiverse()) == fingerprint(s.multiverse()));
  CHECK_NE(read_back(s).zobrist(), s.zobrist());
  // White declines; the board with the right is history now.
  play(s, "(0T2)Ke1(0T2)e2");
  CHECK_EQ(s.multiverse().board_at(0, half_turn(2, Colour::White))->en_passant(),
           make_square(3, 5));
  CHECK_EQ(s.multiverse().zobrist(), zobrist::key_from_scratch(s.multiverse()));
  s.submit();
  CHECK_EQ(read_back(s).zobrist(), s.zobrist());
  t5d_test::act(s, {"(0T2)Ke8(0T2)e7"});
  CHECK_EQ(read_back(s).zobrist(), s.zobrist());
  CHECK_EQ(s.multiverse().zobrist(), zobrist::key_from_scratch(s.multiverse()));
  // Every move from here leads to equal keys too.
  for (const Move& m : t5d_test::moves_of(s)) {
    GameState child = s;
    child.apply(m);
    GameState other = read_back(s);
    other.apply(m);
    CHECK_EQ(child.zobrist(), other.zobrist());
  }
}

TEST(en_passant_never_across_boards) {
  // S7: no en passant through time or across timelines. A timeline double
  // step records none; a pawn on another timeline beside the passed square
  // cannot take.
  GameState s = game({"[5/5/5/5/5:-1:1:w]", "[5/5/5/5/5:0:1:w]", "[5/5/5/2P*2/5:+1:1:w]"}, k5x5);
  play(s, "(+1T1)Pc2(-1T1)c2");
  for (const TimelineRef& t : s.multiverse().timelines()) {
    CHECK_EQ(t->head()->en_passant(), kNoSquare);
  }
  // Black double-steps on 0 at T1; White's pawn stands on e5 of +1 at T2,
  // beside the passed square's file but on another timeline: no en passant.
  GameState t = game({"[8/3p*4/8/8/8/8/8/8:0:1:b]", "[8/8/8/4P3/8/8/8/8:+1:2:w]"}, k8x8);
  REQUIRE(t.to_move() == Colour::Black);
  t5d_test::act(t, {"(0T1)Pd7(0T1)d5"});
  CHECK_EQ(t.multiverse().head(0)->en_passant(), kNoSquare);
  REQUIRE(t.to_move() == Colour::White);
  for (const Move& m : t5d_test::moves_of(t)) CHECK(!m.en_passant());
}

TEST(en_passant_with_brawns_q16) {
  // Q16: by default brawns take part, capturing and being captured (wiki.gg,
  // the only source); the alternative leaves en passant to pawns.
  GameState s = board8("8/3w*4/8/4P3/8/8/8/8", 'b');
  play(s, "(0T1)Wd7(0T1)d5");
  CHECK_EQ(s.multiverse().head(0)->en_passant(), make_square(3, 5));
  s.submit();
  CHECK(t5d_test::find_move(s, "(0T2)Pe5(0T2)d6").has_value());
  RuleOptions pawns_only;
  pawns_only.en_passant_brawns = false;
  GameState p = board8("8/3w*4/8/4P3/8/8/8/8", 'b', pawns_only);
  play(p, "(0T1)Wd7(0T1)d5");
  CHECK_EQ(p.multiverse().head(0)->en_passant(), kNoSquare);
  // A brawn capturing a pawn en passant.
  GameState w = board8("8/3p*4/8/4W3/8/8/8/8", 'b');
  play(w, "(0T1)Pd7(0T1)d5");
  w.submit();
  const std::optional<Move> ep = t5d_test::find_move(w, "(0T2)We5(0T2)d6");
  REQUIRE(ep.has_value());
  CHECK(ep->en_passant());
}

// ---- promotion (S8; Q17) ----------------------------------------------------------------------------

TEST(promotion_to_a_queen_by_default) {
  // S8: on reaching its last rank a pawn becomes a queen - the only choice.
  const GameState s = board8("r7/1P6/8/8/8/8/8/8");
  const std::vector<std::string> r = piece_texts(s, 0, "b7");
  CHECK_EQ(r.size(), std::size_t{2});
  CHECK(contains(r, "(0T1)Pb7(0T1)b8=Q"));
  CHECK(contains(r, "(0T1)Pb7(0T1)a8=Q"));
  GameState after = s;
  play(after, "(0T1)Pb7(0T1)a8=Q");
  CHECK(at(after.multiverse().head(0), "a8") == Piece(PieceKind::Queen, Colour::White));
  // Black promotes on rank 1.
  CHECK(contains(piece_texts(board8("8/8/8/8/8/8/1p6/8", 'b'), 0, "b2"), "(0T1)Pb2(0T1)b1=Q"));
  // A game's Promotions list (a client extension): one move per piece.
  RuleOptions four;
  four.promotions = PromotionSet::of(
      {PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop, PieceKind::Knight});
  const std::vector<std::string> q = piece_texts(board8("8/1P6/8/8/8/8/8/8", 'w', four), 0, "b7");
  CHECK_EQ(q.size(), std::size_t{4});
  CHECK(contains(q, "(0T1)Pb7(0T1)b8=N"));
}

TEST(promotion_of_a_brawn_through_time_q17) {
  // A white brawn on c4 captures one rank forward and one turn back onto c5 -
  // its last rank - of T1's board: a branching capture that promotes (S8,
  // 4.7). Q17's alternative: brawns do not promote.
  const std::vector<std::string> on_c4 = {"[2r2/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]",
                                          "[5/2W2/5/5/5:0:2:w]"};
  const GameState s = game(on_c4, k5x5);
  const std::vector<std::string> moves = piece_texts(s, 0, "c4");
  CHECK(contains(moves, "(0T2)Wc4(0T1)c5=Q"));
  CHECK(contains(moves, "(0T2)Wc4(0T2)c5=Q"));
  const std::optional<Move> m = t5d_test::find_move(s, "(0T2)Wc4(0T1)c5=Q");
  REQUIRE(m.has_value());
  CHECK(m->branching());
  CHECK(m->promotes());
  RuleOptions no;
  no.brawns_promote = false;
  const std::vector<std::string> plain = piece_texts(game(on_c4, k5x5, no), 0, "c4");
  CHECK(contains(plain, "(0T2)Wc4(0T1)c5"));
  CHECK(!contains(plain, "(0T2)Wc4(0T1)c5=Q"));
  // Q17's other question: a brawn arriving on its last rank of ANOTHER board.
  // Default: it promotes like any arrival there (S8). The alternative: only an
  // arrival by a move within its own board promotes.
  RuleOptions in_board;
  in_board.brawn_cross_board_promotion = false;
  const GameState only_in_board = game(on_c4, k5x5, in_board);
  const std::vector<std::string> mixed = piece_texts(only_in_board, 0, "c4");
  CHECK(contains(mixed, "(0T2)Wc4(0T1)c5"));       // across boards: stays a brawn
  CHECK(!contains(mixed, "(0T2)Wc4(0T1)c5=Q"));
  CHECK(contains(mixed, "(0T2)Wc4(0T2)c5=Q"));     // within the board: promotes
  CHECK(!contains(mixed, "(0T2)Wc4(0T2)c5"));
  GameState after = only_in_board;
  play(after, "(0T2)Wc4(0T1)c5");
  REQUIRE(after.multiverse().has_timeline(1));
  CHECK(at(after.multiverse().head(1), "c5") == Piece(PieceKind::Brawn, Colour::White));
  // apply holds a move to the ruleset's promotion rule: the promoting version
  // is refused under the alternative, the plain one under the default.
  Move promoted = *m;
  GameState refuse = only_in_board;
  CHECK_THROWS_AS(refuse.apply(promoted), std::invalid_argument);
  CHECK(refuse == only_in_board);
  Move unpromoted = promoted;
  unpromoted.flags = static_cast<std::uint8_t>(unpromoted.flags & ~Move::kPromotion);
  GameState standard = s;
  CHECK_THROWS_AS(standard.apply(unpromoted), std::invalid_argument);
  CHECK(standard == s);
}

// ---- the Turn Zero board (Q8) -------------------------------------------------------------------------

TEST(turn_zero_board_is_reachable_by_black_branching) {
  // Standard - Turn Zero (V3): the standard placement at T0 with Black to move
  // and at T1 with White to move; White moves first. By L5 the T0 board is a
  // history board, so only Black can reach it, by branching.
  const std::string standard = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*";
  const std::vector<std::string> boards = {"[" + standard + ":0:0:b]", "[" + standard + ":0:1:w]"};
  GameState s = game(boards, k8x8);
  CHECK(s.to_move() == Colour::White);
  CHECK_EQ(s.present(), half_turn(1, Colour::White));
  for (const Move& m : t5d_test::moves_of(s)) CHECK(m.to_half_turn >= half_turn(1, Colour::White));
  t5d_test::act(s, {"(0T1)Pe2(0T1)e4"});
  const std::optional<Move> back = t5d_test::find_move(s, "(0T1)Nb8(0T0)b6");
  REQUIRE(back.has_value());
  CHECK(back->branching());
  play(s, "(0T1)Nb8(0T0)b6");
  CHECK(s.multiverse().has_timeline(-1));
  CHECK_EQ(s.multiverse().timeline(-1)->first_half_turn(), half_turn(1, Colour::White));
  // Q8's alternative: a T0 board is no destination.
  RuleOptions r;
  r.turn_zero_reachable = false;
  GameState q = game(boards, k8x8, r);
  t5d_test::act(q, {"(0T1)Pe2(0T1)e4"});
  for (const Move& m : t5d_test::moves_of(q)) CHECK_NE(turn_of(m.to_half_turn), 0);
}
