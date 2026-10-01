// Pseudo-legal movement of every piece (rules reference sections 3 and 4):
// within a board, through time and across timelines; blocking by pieces on
// intermediate boards (J7), the colour rule (L4), void (J8), destinations
// (J2), branching versus non-branching jumps (J3, J4).

#include <cstddef>
#include <string>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using t5d_test::contains;
using t5d_test::game;
using t5d_test::piece_texts;

namespace {

constexpr BoardSize k8x8{8, 8};
constexpr BoardSize k5x5{5, 5};

// One 8x8 board at (0T1), white to move: nothing else exists.
GameState lone(const std::string& pieces) { return game({"[" + pieces + ":0:1:w]"}, k8x8); }

// Timeline 0 with five empty 5x5 boards T1w..T3w, the last holding `head`
// (the playable board, white to move); `t2w` and `t1w` replace those boards'
// placements, `t2b` and `t1b` the black boards'.
GameState history(const std::string& head, const std::string& t2w = "5/5/5/5/5",
                  const std::string& t1w = "5/5/5/5/5", const std::string& t2b = "5/5/5/5/5",
                  const std::string& t1b = "5/5/5/5/5", const RuleOptions& rules = {}) {
  return game({"[" + t1w + ":0:1:w]", "[" + t1b + ":0:1:b]", "[" + t2w + ":0:2:w]",
               "[" + t2b + ":0:2:b]", "[" + head + ":0:3:w]"},
              k5x5, rules);
}

std::size_t count_of(const GameState& s, int timeline, const char* square) {
  return piece_texts(s, timeline, square).size();
}

}  // namespace

// ---- within one board ---------------------------------------------------------------------

TEST(movegen_lone_board_counts_match_ordinary_chess) {
  // On a lone board with no other board anywhere, every piece moves as in
  // ordinary chess (4.1-4.5, 4.10-4.12); the unicorn and dragon cannot move
  // at all (4.8, 4.9: three and four axes).
  CHECK_EQ(count_of(lone("8/8/8/8/3R4/8/8/8"), 0, "d4"), std::size_t{14});
  CHECK_EQ(count_of(lone("8/8/8/8/3B4/8/8/8"), 0, "d4"), std::size_t{13});
  CHECK_EQ(count_of(lone("8/8/8/8/3Q4/8/8/8"), 0, "d4"), std::size_t{27});
  CHECK_EQ(count_of(lone("8/8/8/8/3Y4/8/8/8"), 0, "d4"), std::size_t{27});
  CHECK_EQ(count_of(lone("8/8/8/8/3S4/8/8/8"), 0, "d4"), std::size_t{27});
  CHECK_EQ(count_of(lone("8/8/8/8/3N4/8/8/8"), 0, "d4"), std::size_t{8});
  CHECK_EQ(count_of(lone("8/8/8/8/8/8/8/N7"), 0, "a1"), std::size_t{2});
  CHECK_EQ(count_of(lone("8/8/8/8/3K4/8/8/8"), 0, "d4"), std::size_t{8});
  CHECK_EQ(count_of(lone("8/8/8/8/3C4/8/8/8"), 0, "d4"), std::size_t{8});
  CHECK_EQ(count_of(lone("8/8/8/8/3U4/8/8/8"), 0, "d4"), std::size_t{0});
  CHECK_EQ(count_of(lone("8/8/8/8/3D4/8/8/8"), 0, "d4"), std::size_t{0});
}

TEST(movegen_within_a_board_pieces_block_and_captures_stop_a_slide) {
  // Rook d4 with a white pawn d6 and a black knight f4: up to d5, and f4 taken.
  const GameState s = lone("8/8/3P4/8/3R1n2/8/8/8");
  const std::vector<std::string> r = piece_texts(s, 0, "d4");
  CHECK(contains(r, "(0T1)Rd4(0T1)d5"));
  CHECK(!contains(r, "(0T1)Rd4(0T1)d6"));
  CHECK(!contains(r, "(0T1)Rd4(0T1)d7"));
  CHECK(contains(r, "(0T1)Rd4(0T1)e4"));
  CHECK(contains(r, "(0T1)Rd4(0T1)f4"));
  CHECK(!contains(r, "(0T1)Rd4(0T1)g4"));
  CHECK_EQ(r.size(), std::size_t{1 + 2 + 3 + 3});  // up 1, right 2, down 3, left 3
  const std::optional<Move> capture = t5d_test::find_move(s, "(0T1)Rd4(0T1)f4");
  REQUIRE(capture.has_value());
  CHECK(capture->capture());
  CHECK(capture->captured == Piece(PieceKind::Knight, Colour::Black));
  CHECK(capture->physical());
  CHECK(!capture->branching());
}

TEST(movegen_after_e4_e5_ordinary_moves_and_two_knight_jumps) {
  // Worked out by hand from the rules: after 1. e4 e5 White has ordinary
  // chess's 29 moves, and only two moves through time - the knights' jumps
  // two ranks up and one turn back, to g3 and b3 of the T1 board (branching,
  // J3). Every other piece's way back in time is blocked by its own earlier
  // self or another piece on the T1 board (J7).
  GameState s(standard_start());
  t5d_test::act(s, {"(0T1)Pe2(0T1)e4"});
  CHECK_EQ(count_moves(s), std::size_t{20});  // Black at T1: nothing earlier to reach
  t5d_test::act(s, {"(0T1)Pe7(0T1)e5"});
  const MoveList moves = t5d_test::moves_of(s);
  CHECK_EQ(moves.size(), std::size_t{31});
  std::vector<std::string> jumps;
  for (const Move& m : moves) {
    if (m.jump()) {
      CHECK(m.branching());
      jumps.push_back(to_string(m, false));
    }
  }
  CHECK((jumps == std::vector<std::string>{"(0T2)Nb1(0T1)b3", "(0T2)Ng1(0T1)g3"}));
}

TEST(movegen_only_the_movers_pieces_move) {
  // Black's pieces on White's board never move (L5): only White's.
  const GameState s = lone("8/8/8/3r4/3R4/8/8/8");
  for (const Move& m : t5d_test::moves_of(s)) CHECK(m.piece.colour() == Colour::White);
  CHECK_EQ(t5d_test::moves_of(s).size(), std::size_t{1 + 3 + 3 + 4});
}

// ---- through time (J1, J7, L4) -----------------------------------------------------------

TEST(movegen_rook_slides_back_in_time_on_its_own_timeline) {
  // A rook that arrived on c3 with its previous move: c3 is empty on every
  // earlier white board, so it may slide back one or two turns (J7) - each a
  // branching move onto a history board (J3, 4.1). Forward in time there is
  // no board (J2).
  const GameState s = history("5/5/2R2/5/5");
  const std::vector<std::string> r = piece_texts(s, 0, "c3");
  CHECK_EQ(r.size(), std::size_t{8 + 2});
  CHECK(contains(r, "(0T3)Rc3(0T2)c3"));
  CHECK(contains(r, "(0T3)Rc3(0T1)c3"));
  const std::optional<Move> back = t5d_test::find_move(s, "(0T3)Rc3(0T1)c3");
  REQUIRE(back.has_value());
  CHECK(back->branching());
  CHECK(back->jump());
  CHECK_EQ(back->to_half_turn, 2);
}

TEST(movegen_pieces_on_intermediate_boards_block) {
  // J7: every intermediate square must be empty - a white piece on c3 at T2
  // blocks the slide back; a black one is captured there and stops it.
  CHECK_EQ(count_of(history("5/5/2R2/5/5", "5/5/2N2/5/5"), 0, "c3"), std::size_t{8});
  const GameState s = history("5/5/2R2/5/5", "5/5/2n2/5/5");
  const std::vector<std::string> r = piece_texts(s, 0, "c3");
  CHECK_EQ(r.size(), std::size_t{9});
  CHECK(contains(r, "(0T3)Rc3(0T2)c3"));
  CHECK(!contains(r, "(0T3)Rc3(0T1)c3"));
}

TEST(movegen_opponent_coloured_boards_are_ignored) {
  // L4: movement between boards ignores boards on which the opponent is to
  // move - they are neither steps nor blockers. Pieces on c3 of the black
  // boards change nothing.
  const GameState s = history("5/5/2R2/5/5", "5/5/5/5/5", "5/5/5/5/5", "5/5/2N2/5/5",
                              "5/5/2n2/5/5");
  CHECK_EQ(count_of(s, 0, "c3"), std::size_t{10});
  for (const Move& m : t5d_test::moves_of(s)) {
    CHECK(to_move_at(m.to_half_turn) == Colour::White);
  }
}

TEST(movegen_the_past_self_blocks_by_default) {
  // J7 / Q19: "Any piece blocks, including the moving piece's own earlier
  // self" - a rook that has stood on c3 since T1 cannot slide back at all.
  const GameState blocked = history("5/5/2R2/5/5", "5/5/2R2/5/5", "5/5/2R2/5/5");
  CHECK_EQ(count_of(blocked, 0, "c3"), std::size_t{8});
  // Q19's alternative: the past self does not block - but it is still a
  // piece of the mover's, so it is not a destination: still no time move here.
  RuleOptions r;
  r.past_self_blocks = false;
  const GameState passes = history("5/5/2R2/5/5", "5/5/2R2/5/5", "5/5/5/5/5",
                                   "5/5/5/5/5", "5/5/5/5/5", r);
  const std::vector<std::string> moves = piece_texts(passes, 0, "c3");
  CHECK(contains(moves, "(0T3)Rc3(0T1)c3"));   // passes its past self at T2
  CHECK(!contains(moves, "(0T3)Rc3(0T2)c3"));  // but cannot land on it
  // Another piece of the mover's still blocks.
  const GameState other = history("5/5/2R2/5/5", "5/5/2N2/5/5", "5/5/5/5/5", "5/5/5/5/5",
                                  "5/5/5/5/5", r);
  CHECK(!contains(piece_texts(other, 0, "c3"), "(0T3)Rc3(0T1)c3"));
}

TEST(movegen_bishop_through_time) {
  // 4.2: two axes, n steps on each. From c3 at T3: the 8 diagonals within the
  // board, and along (file, turn) and (rank, turn) one or two turns back -
  // b3/d3/c2/c4 at T2, a3/e3/c1/c5 at T1.
  const GameState s = history("5/5/2B2/5/5");
  const std::vector<std::string> r = piece_texts(s, 0, "c3");
  CHECK_EQ(r.size(), std::size_t{8 + 8});
  for (const char* t : {"(0T3)Bc3(0T2)b3", "(0T3)Bc3(0T2)d3", "(0T3)Bc3(0T2)c2",
                        "(0T3)Bc3(0T2)c4", "(0T3)Bc3(0T1)a3", "(0T3)Bc3(0T1)e3",
                        "(0T3)Bc3(0T1)c1", "(0T3)Bc3(0T1)c5"}) {
    CHECK(contains(r, t));
  }
  CHECK(!contains(r, "(0T3)Bc3(0T2)c3"));  // one axis: a rook's move
  CHECK(!contains(r, "(0T3)Bc3(0T2)b2"));  // three axes: a unicorn's
}

TEST(movegen_unicorn_reaches_its_own_past_with_one_timeline) {
  // 4.8: along x, y and T the unicorn can go into its own timeline's past
  // (branching) while only one timeline exists. From c3 at T3: b2/d2/b4/d4 at
  // T2 and a1/e1/a5/e5 at T1 - nothing within the board.
  const GameState s = history("5/5/2U2/5/5");
  const std::vector<std::string> r = piece_texts(s, 0, "c3");
  CHECK_EQ(r.size(), std::size_t{8});
  for (const char* t : {"(0T3)Uc3(0T2)b2", "(0T3)Uc3(0T2)d2", "(0T3)Uc3(0T2)b4",
                        "(0T3)Uc3(0T2)d4", "(0T3)Uc3(0T1)a1", "(0T3)Uc3(0T1)e1",
                        "(0T3)Uc3(0T1)a5", "(0T3)Uc3(0T1)e5"}) {
    CHECK(contains(r, t));
  }
  // 4.9: the dragon needs all four axes - no move with one timeline.
  CHECK_EQ(count_of(history("5/5/2D2/5/5"), 0, "c3"), std::size_t{0});
}
