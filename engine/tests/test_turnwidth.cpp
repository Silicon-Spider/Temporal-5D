// The turn width of an action (register M22; tools/turn_width.hpp): its
// counts on constructed actions with known answers - Standard's first action,
// Two Timelines' (T13), a jump between two present boards (J4, A6's "two
// fewer"), a rewinding branch that releases a mandatory board (T8), moves
// after the shift (R8), moves on boards ahead of the present and on an
// inactive timeline (T6), the jumps across the line between mandatory and
// optional boards - from an optional board onto a mandatory one (T6), from a
// mandatory board onto an optional one, and between two optional boards (T3,
// J4) - and a reactivation that makes a board mandatory mid-action, with a
// rewind (T12) and without one (T11); the summary's ranges and classes at
// their edges, and its statistics on constructed widths; and t5d-measure
// --turns / --turn-summary run end to end on the hand-written records of
// engine/testdata/turn-width - the counts, how records are classified (a
// repeat, a prefix, a truncated record, two records that differ only in
// their last action) and the board-count ranges at their edges (the build
// passes the executable's path, T5D_MEASURE_EXE, the engine directory,
// T5D_ENGINE_DIR, and a scratch directory, T5D_TEST_SCRATCH).
//
// Every position and record here is written for the test from the rules
// reference (docs/rules/README.md) and the notation specification's text -
// never taken from t0database, a game corpus or another implementation.

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "t5d/fen.hpp"
#include "t5d/move.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "turn_width.hpp"

using namespace t5d;
using t5d::tools::classify_action;
using t5d::tools::TurnCounter;
using t5d::tools::TurnWidth;
using t5d_test::act;
using t5d_test::game;

namespace {

constexpr BoardSize k5x5{5, 5};

Move move_of(const GameState& state, const char* text) {
  const std::optional<Move> m = t5d_test::find_move(state, text);
  if (!m) throw std::runtime_error(std::string("no generated move ") + text);
  return *m;
}

// Counts an action given by its moves' export texts, each found among the
// moves generated for the state the earlier ones reach.
TurnWidth count(const GameState& start, std::initializer_list<const char*> texts) {
  TurnCounter counter(start);
  std::vector<Move> moves;
  for (const char* text : texts) {
    moves.push_back(move_of(counter.state(), text));
    counter.apply(moves.back());
  }
  // The convenience function counts the same.
  CHECK(classify_action(start, moves) == counter.width());
  return counter.width();
}

// 0's playable board at T2 (the present: mandatory); +1, White's first
// timeline (active), playable at T3 - ahead of the present; +2, White's
// second (inactive: Black has created none), playable at T1 - behind the
// present. Optional both (T6).
GameState optional_boards() {
  return game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/R4:0:2:w]",
               "[5/5/5/5/R4:+1:3:w]", "[5/5/5/5/R4:+2:1:w]"},
              k5x5);
}

// Two Timelines after 1. e3 e3 / e6 e6: White to move on both boards of T2.
GameState two_timelines_at_t2() {
  GameState s(standard_two_timelines_start());
  act(s, {"(-0T1)Pe2(-0T1)e3", "(+0T1)Pe2(+0T1)e3"});
  act(s, {"(-0T1)Pe7(-0T1)e6", "(+0T1)Pe7(+0T1)e6"});
  return s;
}

}  // namespace

TEST(turn_width_standard_first_action) {
  const GameState s(standard_start());
  const TurnWidth w = count(s, {"(0T1)Pe2(0T1)e4"});
  CHECK_EQ(w.boards, std::size_t{1});
  CHECK_EQ(w.timelines, 1);
  CHECK_EQ(w.active, 1);
  CHECK_EQ(w.mandatory, 1);
  CHECK_EQ(w.playable, 1);
  CHECK_EQ(w.optional, 0);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 0);
  CHECK_EQ(w.optional_when_made, 0);
  CHECK_EQ(w.boards_played, 1);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.jumps, 0);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.rewinds, 0);
  CHECK_EQ(w.mandatory_released, 0);
  CHECK_EQ(w.mandatory_gained, 0);
}

TEST(turn_width_two_timelines_first_action_t13) {
  // T13: both initial boards are in the present and mandatory. The second
  // move is made before the shift - one obligation is still open.
  const GameState s(standard_two_timelines_start());
  const TurnWidth w = count(s, {"(-0T1)Pe2(-0T1)e3", "(+0T1)Pe2(+0T1)e3"});
  CHECK_EQ(w.boards, std::size_t{2});
  CHECK_EQ(w.timelines, 2);
  CHECK_EQ(w.active, 2);
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.playable, 2);
  CHECK_EQ(w.optional, 0);
  CHECK_EQ(w.moves, 2);
  CHECK_EQ(w.moves_from_mandatory, 2);
  CHECK_EQ(w.optional_moves, 0);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 2);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.jumps_between_mandatory, 0);
}

TEST(turn_width_a_jump_between_two_present_boards_plays_both_j4) {
  // J4: a knight from +0 onto -0's playable board in the same column makes
  // a move on both boards - one move, two mandatory boards played (A6's
  // "two fewer"); the obligations are met by it alone.
  const GameState s(standard_two_timelines_start());
  TurnCounter c(s);
  c.apply(move_of(s, "(+0T1)Nb1(-0T1)b3"));
  CHECK(c.state().obligations_met());
  const TurnWidth w = c.width();
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 0);
  CHECK_EQ(w.optional_by_source, 0);
  CHECK_EQ(w.optional_when_made, 0);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 2);
  CHECK_EQ(w.optional_played, 0);
  CHECK_EQ(w.jumps, 1);
  CHECK_EQ(w.branches, 0);
  CHECK_EQ(w.jumps_between_mandatory, 1);
  CHECK_EQ(w.optional_onto_mandatory, 0);
  CHECK_EQ(w.mandatory_onto_optional, 0);
  CHECK_EQ(w.mandatory_released, 0);
}

TEST(turn_width_a_rewinding_branch_releases_a_mandatory_board_t8) {
  // T8: the knight's branch from +0 back to T1 creates White's first
  // timeline, active, behind the present: the present moves back to it,
  // Black's colour, and -0's board - mandatory at the start - is released.
  const GameState s = two_timelines_at_t2();
  const TurnWidth w = count(s, {"(+0T2)Ng1(+0T1)g3"});
  CHECK_EQ(w.boards, std::size_t{6});
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.jumps, 1);
  CHECK_EQ(w.branches, 1);
  CHECK_EQ(w.rewinds, 1);
  CHECK_EQ(w.boards_played, 1);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.mandatory_released, 1);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.optional_moves, 0);
  // R8: the released board may still be moved on, after the shift. By its
  // status at the start the move is not optional; by its status when made it
  // is - counted apart, and the board is no longer released.
  const TurnWidth v = count(s, {"(+0T2)Ng1(+0T1)g3", "(-0T2)Pd2(-0T2)d3"});
  CHECK_EQ(v.moves, 2);
  CHECK_EQ(v.moves_from_mandatory, 2);
  CHECK_EQ(v.optional_moves, 0);
  CHECK_EQ(v.optional_by_source, 0);
  CHECK_EQ(v.optional_when_made, 1);
  CHECK_EQ(v.moves_on_released, 1);
  CHECK_EQ(v.moves_after_shift, 1);
  CHECK_EQ(v.rewinds, 1);
  CHECK_EQ(v.mandatory_released, 0);
  CHECK_EQ(v.boards_played, 2);
}

TEST(turn_width_moves_after_the_shift_r8) {
  // R8: once 0's board is played the obligations are met; the move on +1,
  // ahead of the present, comes after the shift and is optional.
  const TurnWidth w = count(optional_boards(), {"(0T2)Ra1(0T2)a2", "(+1T3)Ra1(+1T3)a2"});
  CHECK_EQ(w.boards, std::size_t{5});
  CHECK_EQ(w.timelines, 3);
  CHECK_EQ(w.active, 2);
  CHECK_EQ(w.mandatory, 1);
  CHECK_EQ(w.playable, 3);
  CHECK_EQ(w.optional, 2);
  CHECK_EQ(w.moves, 2);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_when_made, 1);
  CHECK_EQ(w.moves_after_shift, 1);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.optional_played, 1);
}

TEST(turn_width_optional_boards_ahead_and_inactive_t6) {
  // T6: +1 (ahead of the present) and +2 (an inactive timeline, behind it)
  // may be played before the mandatory board - optional moves, not after
  // the shift.
  const TurnWidth w =
      count(optional_boards(), {"(+1T3)Ra1(+1T3)a2", "(+2T1)Ra1(+2T1)a2", "(0T2)Ra1(0T2)a2"});
  CHECK_EQ(w.optional, 2);
  CHECK_EQ(w.moves, 3);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 2);
  CHECK_EQ(w.optional_by_source, 2);
  CHECK_EQ(w.optional_when_made, 2);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.boards_played, 3);
  CHECK_EQ(w.optional_played, 2);
  CHECK_EQ(w.mandatory_released, 0);
}

TEST(turn_width_a_jump_from_an_optional_board_onto_a_mandatory_one_t6) {
  // T6: a mandatory board is also satisfied when a piece arrives on it. A
  // bishop on +1 (ahead of the present) jumps back one timeline and one turn
  // onto 0's playable board (J3: a playable board, so no branch): by its
  // source the move is optional, and it plays the mandatory board.
  const GameState s = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/5:0:2:w]",
                            "[5/5/5/5/B4:+1:3:w]"},
                           k5x5);
  REQUIRE(s.mandatory_timelines() == std::vector<int>({0}));
  TurnCounter c(s);
  c.apply(move_of(s, "(+1T3)Ba1(0T2)a1"));
  CHECK(c.state().obligations_met());
  const TurnWidth w = c.width();
  CHECK_EQ(w.mandatory, 1);
  CHECK_EQ(w.optional, 1);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.moves_from_mandatory, 0);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_by_source, 1);
  CHECK_EQ(w.optional_when_made, 1);
  CHECK_EQ(w.jumps, 1);
  CHECK_EQ(w.branches, 0);
  CHECK_EQ(w.optional_onto_mandatory, 1);
  CHECK_EQ(w.mandatory_onto_optional, 0);
  CHECK_EQ(w.jumps_between_mandatory, 0);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.optional_played, 1);
  CHECK_EQ(w.mandatory_released, 0);
}

TEST(turn_width_a_jump_from_a_mandatory_board_onto_an_optional_one_t3_j4) {
  // The mirror case: a bishop on 0's board in the present (mandatory) jumps
  // forward one timeline and one turn onto +1's playable board, which lies
  // ahead of the present (optional; J3: a playable board, so no branch). The
  // move is made on both boards (T3, J4): by its source it is not optional,
  // by its destination it is - an optional move under M22's reading, and a
  // move on an optional board played.
  const GameState s = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/B4:0:2:w]",
                            "[5/5/5/5/5:+1:3:w]"},
                           k5x5);
  REQUIRE(s.mandatory_timelines() == std::vector<int>({0}));
  REQUIRE(s.playable_timelines() == std::vector<int>({0, 1}));
  TurnCounter c(s);
  const Move jump = move_of(s, "(0T2)Ba1(+1T3)a1");
  CHECK(!jump.branching());
  c.apply(jump);
  CHECK(c.state().obligations_met());
  const TurnWidth w = c.width();
  CHECK_EQ(w.mandatory, 1);
  CHECK_EQ(w.optional, 1);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_by_source, 0);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_when_made, 1);
  CHECK_EQ(w.jumps, 1);
  CHECK_EQ(w.branches, 0);
  CHECK_EQ(w.mandatory_onto_optional, 1);
  CHECK_EQ(w.optional_onto_mandatory, 0);
  CHECK_EQ(w.jumps_between_mandatory, 0);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.optional_played, 1);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.mandatory_released, 0);
}

TEST(turn_width_a_jump_between_two_optional_boards_plays_both_t3_j4) {
  // A rook on +1 (ahead of the present) moves one timeline up onto +2's
  // playable board (an inactive timeline: optional too) - one optional move,
  // two optional boards played; then the mandatory board.
  const GameState s = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/K4:0:2:w]",
                            "[5/5/5/5/R4:+1:3:w]", "[5/5/5/5/5:+2:3:w]"},
                           k5x5);
  REQUIRE(s.mandatory_timelines() == std::vector<int>({0}));
  REQUIRE(!s.active(2));
  const TurnWidth w = count(s, {"(+1T3)Ra1(+2T3)a1", "(0T2)Ka1(0T2)a2"});
  CHECK_EQ(w.mandatory, 1);
  CHECK_EQ(w.optional, 2);
  CHECK_EQ(w.moves, 2);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_by_source, 1);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_when_made, 1);
  CHECK_EQ(w.jumps, 1);
  CHECK_EQ(w.jumps_between_mandatory, 0);
  CHECK_EQ(w.optional_onto_mandatory, 0);
  CHECK_EQ(w.mandatory_onto_optional, 0);
  CHECK_EQ(w.boards_played, 3);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.optional_played, 2);
  CHECK_EQ(w.moves_after_shift, 0);
}

TEST(turn_width_a_reactivation_gains_a_mandatory_board_t12) {
  // T10-T12: White's second branch is inactive; Black's first activates it,
  // and its playable board - Black's, behind the present - becomes the
  // present: mandatory mid-action (gained), while +1's board, mandatory at
  // the start, now lies ahead and is released.
  GameState s = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[n4/5/5/5/5:0:2:w]",
                      "[5/5/5/5/K4:0:2:b]", "[4r/5/2R2/5/5:0:3:w]", "[5/5/5/5/R4:+1:3:w]"},
                     k5x5);
  // White's action: the inactive branch, then +1 (both from mandatory boards).
  const TurnWidth white = count(s, {"(0T3)Rc3(0T2)c3", "(+1T3)Ra1(+1T3)a2"});
  CHECK_EQ(white.mandatory, 2);
  CHECK_EQ(white.moves_from_mandatory, 2);
  CHECK_EQ(white.branches, 1);
  CHECK_EQ(white.rewinds, 0);  // T8: an inactive branch never moves the present
  act(s, {"(0T3)Rc3(0T2)c3", "(+1T3)Ra1(+1T3)a2"});
  REQUIRE(s.mandatory_timelines() == std::vector<int>({0, 1}));
  REQUIRE(!s.active(2));
  const TurnWidth w = count(s, {"(0T3)Re5(0T2)e5", "(+2T2)Na5(+2T2)b3"});
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.playable, 3);
  CHECK_EQ(w.optional, 1);  // +2: inactive at the start
  CHECK_EQ(w.moves, 2);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_by_source, 1);
  CHECK_EQ(w.optional_when_made, 0);  // +2 was mandatory when its move was made
  CHECK_EQ(w.moves_on_gained, 1);
  CHECK_EQ(w.mandatory_gained, 1);
  CHECK_EQ(w.rewinds, 1);
  CHECK_EQ(w.mandatory_released, 1);  // +1
  CHECK_EQ(w.branches, 1);
  CHECK_EQ(w.moves_after_shift, 0);
}

TEST(turn_width_a_reactivation_in_the_present_gains_a_board_without_a_rewind_t11) {
  // Black to move on 0 and +1 at T3 (the present); +2, White's second timeline,
  // inactive (Black has created none), also has Black's board at T3. Black
  // plays 0, then jumps from +1 onto 0's T3 board - history now, so the jump
  // branches (J6) - creating Black's first timeline, active, whose first board
  // (T4, White's) lies after the present: no rewind. The creation activates
  // +2 (T11), whose board is in the present: mandatory mid-action, without the
  // present moving.
  const GameState s = game({"[4r/5/5/5/5:0:3:b]", "[r4/5/5/5/5:+1:3:b]", "[1r3/5/5/5/5:+2:3:b]"},
                           k5x5);
  REQUIRE(s.to_move() == Colour::Black);
  REQUIRE(!s.active(2));
  REQUIRE(s.mandatory_timelines() == std::vector<int>({0, 1}));
  TurnCounter c(s);
  c.apply(move_of(c.state(), "(0T3)Re5(0T3)e4"));
  const Move branch = move_of(c.state(), "(+1T3)Ra5(0T3)a5");
  CHECK(branch.branching());
  c.apply(branch);
  CHECK(c.state().active(2));
  CHECK(c.state().mandatory_timelines() == std::vector<int>({2}));
  CHECK(!c.state().obligations_met());
  c.apply(move_of(c.state(), "(+2T3)Rb5(+2T3)b4"));
  CHECK(c.state().obligations_met());
  const TurnWidth w = c.width();
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.playable, 3);
  CHECK_EQ(w.optional, 1);
  CHECK_EQ(w.moves, 3);
  CHECK_EQ(w.moves_from_mandatory, 2);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_when_made, 0);
  CHECK_EQ(w.moves_on_gained, 1);
  CHECK_EQ(w.mandatory_gained, 1);
  CHECK_EQ(w.rewinds, 0);
  CHECK_EQ(w.mandatory_released, 0);
  CHECK_EQ(w.branches, 1);
  CHECK_EQ(w.boards_played, 3);
}

TEST(turn_width_a_move_that_does_not_fit_changes_nothing) {
  GameState s(standard_start());
  const Move second = move_of(s, "(0T1)Pd2(0T1)d4");
  TurnCounter c(s);
  c.apply(move_of(s, "(0T1)Pe2(0T1)e4"));
  const TurnWidth before = c.width();
  const GameState state_before = c.state();
  // T3: the board has been played - its move no longer fits.
  CHECK_THROWS_AS(c.apply(second), std::invalid_argument);
  CHECK(c.width() == before);
  CHECK(c.state() == state_before);
}

// ---- the summary: ranges, classes and statistics -------------------------------------------

namespace {

// The label of the one range of `ranges` holding `boards` ("" if none; a
// failed check if several do).
template <std::size_t N>
std::string range_of(const t5d::tools::BoardRange (&ranges)[N], std::size_t boards) {
  std::string found;
  for (const t5d::tools::BoardRange& r : ranges) {
    if (!r.contains(boards)) continue;
    CHECK(found.empty());
    found = r.label;
  }
  return found;
}

}  // namespace

TEST(turn_width_summary_ranges_and_classes_at_their_edges) {
  using t5d::tools::kAllBoards;
  using t5d::tools::kTurnBuckets;
  using t5d::tools::kTurnRanges;
  using t5d::tools::mandatory_class;
  // 6b's ranges, by the board count at the action's start: [lo, hi).
  CHECK_EQ(range_of(kTurnRanges, 1), std::string("<50"));
  CHECK_EQ(range_of(kTurnRanges, 49), std::string("<50"));
  CHECK_EQ(range_of(kTurnRanges, 50), std::string("50-99"));
  CHECK_EQ(range_of(kTurnRanges, 99), std::string("50-99"));
  CHECK_EQ(range_of(kTurnRanges, 100), std::string("100-199"));
  CHECK_EQ(range_of(kTurnRanges, 199), std::string("100-199"));
  CHECK_EQ(range_of(kTurnRanges, 200), std::string("200+"));
  CHECK_EQ(range_of(kTurnRanges, 5000), std::string("200+"));
  // Ten-board buckets under 100, each inside one range; none from 100.
  CHECK_EQ(range_of(kTurnBuckets, 1), std::string("1-9"));
  CHECK_EQ(range_of(kTurnBuckets, 9), std::string("1-9"));
  CHECK_EQ(range_of(kTurnBuckets, 10), std::string("10-19"));
  CHECK_EQ(range_of(kTurnBuckets, 19), std::string("10-19"));
  CHECK_EQ(range_of(kTurnBuckets, 20), std::string("20-29"));
  CHECK_EQ(range_of(kTurnBuckets, 49), std::string("40-49"));
  CHECK_EQ(range_of(kTurnBuckets, 50), std::string("50-59"));
  CHECK_EQ(range_of(kTurnBuckets, 90), std::string("90-99"));
  CHECK_EQ(range_of(kTurnBuckets, 99), std::string("90-99"));
  CHECK_EQ(range_of(kTurnBuckets, 100), std::string());
  for (std::size_t n = 1; n < 1000; ++n) {
    CHECK(!range_of(kTurnRanges, n).empty());  // exactly one range
    CHECK(kAllBoards.contains(n));
    const std::string bucket = range_of(kTurnBuckets, n);
    CHECK_EQ(bucket.empty(), n >= 100);
    for (const t5d::tools::BoardRange& b : kTurnBuckets) {
      if (!b.contains(n)) continue;
      for (const t5d::tools::BoardRange& r : kTurnRanges) {
        if (r.contains(n)) CHECK(r.lo <= b.lo && b.hi <= r.hi);
      }
    }
  }
  // Mandatory classes: 1, 2, 3, 4+.
  CHECK_EQ(mandatory_class(1), 0);
  CHECK_EQ(mandatory_class(2), 1);
  CHECK_EQ(mandatory_class(3), 2);
  CHECK_EQ(mandatory_class(4), 3);
  CHECK_EQ(mandatory_class(5), 3);
  CHECK_EQ(mandatory_class(40), 3);
  CHECK_EQ(std::string(t5d::tools::kMandatoryClasses[3]), std::string("4+"));
  CHECK_THROWS_AS(mandatory_class(0), std::invalid_argument);
}

TEST(turn_width_summary_quantiles_interpolate_between_order_statistics) {
  using t5d::tools::quantile;
  CHECK_EQ(quantile({}, 0.5), 0.0);
  CHECK_EQ(quantile({3, 1, 2}, 0.5), 2.0);
  CHECK_EQ(quantile({1, 2}, 0.5), 1.5);
  std::vector<double> eleven;
  for (int i = 10; i >= 0; --i) eleven.push_back(i);  // 0..10, unsorted
  CHECK_EQ(quantile(eleven, 0.9), 9.0);
  CHECK_EQ(quantile(eleven, 0.95), 9.5);
  CHECK_EQ(quantile(eleven, 1.0), 10.0);
}

TEST(turn_width_summary_statistics_of_constructed_widths) {
  // Eleven actions with known counts (eleven, so that the median and p90 are
  // order statistics): each statistic has a known value, and the three
  // readings of "optional" differ in both the actions with one and the totals.
  const auto width = [](int mandatory, int optional, int moves) {
    TurnWidth w;
    w.boards = 10;
    w.mandatory = mandatory;
    w.optional = optional;
    w.playable = mandatory + optional;
    w.moves = moves;
    w.boards_played = moves;
    return w;
  };
  std::vector<TurnWidth> a;
  a.push_back(width(1, 0, 1));  // 0
  a.push_back(width(1, 0, 1));  // 1
  a.push_back(width(1, 1, 1));  // 2: an optional board left
  a.push_back(width(1, 1, 2));  // 3: an optional move after the shift
  a.back().optional_moves = a.back().optional_by_source = a.back().optional_when_made = 1;
  a.back().optional_played = a.back().moves_after_shift = 1;
  a.push_back(width(2, 0, 1));  // 4: one jump plays both mandatory boards
  a.back().jumps = a.back().jumps_between_mandatory = 1;
  a.back().boards_played = 2;
  a.push_back(width(1, 1, 1));  // 5: a jump from the mandatory board onto the optional one
  a.back().jumps = a.back().mandatory_onto_optional = 1;
  a.back().optional_moves = a.back().optional_when_made = a.back().optional_played = 1;
  a.back().boards_played = 2;
  a.push_back(width(3, 0, 2));  // 6: a rewinding branch, then a move on a released board
  a.back().jumps = a.back().branches = a.back().rewinds = 1;
  a.back().mandatory_released = a.back().optional_when_made = a.back().moves_on_released = 1;
  a.back().moves_after_shift = 1;
  a.push_back(width(4, 3, 7));  // 7: every board played
  a.back().optional_moves = a.back().optional_by_source = a.back().optional_when_made = 3;
  a.back().optional_played = a.back().moves_after_shift = 3;
  a.push_back(width(5, 1, 2));  // 8: as 6, three boards released
  a.back().jumps = a.back().branches = a.back().rewinds = 1;
  a.back().mandatory_released = 3;
  a.back().optional_when_made = a.back().moves_on_released = a.back().moves_after_shift = 1;
  a.push_back(width(2, 1, 2));  // 9: a reactivation (T12): the optional board gained
  a.back().jumps = a.back().branches = a.back().rewinds = 1;
  a.back().mandatory_released = a.back().mandatory_gained = a.back().moves_on_gained = 1;
  a.back().optional_moves = a.back().optional_by_source = a.back().optional_played = 1;
  a.push_back(width(1, 1, 1));  // 10: a jump from the optional board onto the mandatory one
  a.back().jumps = a.back().optional_onto_mandatory = 1;
  a.back().optional_moves = a.back().optional_by_source = a.back().optional_when_made = 1;
  a.back().optional_played = 1;
  a.back().boards_played = 2;
  std::vector<const TurnWidth*> in;
  for (const TurnWidth& w : a) in.push_back(&w);

  const t5d::tools::TurnStats s = t5d::tools::turn_stats(in);
  using Counts = std::array<std::size_t, 4>;
  CHECK_EQ(s.actions, std::size_t{11});
  CHECK(s.mandatory_classes == Counts({6, 2, 1, 2}));  // 4 and 5 are both "4+"
  CHECK_EQ(s.mandatory_over_1, std::size_t{5});
  CHECK_EQ(s.mandatory_over_2, std::size_t{3});
  CHECK_EQ(s.playable_over_1, std::size_t{9});
  CHECK_EQ(s.with_optional_boards, std::size_t{7});
  // Mandatory boards 1,1,1,1,1,1,2,2,3,4,5; optional 0,0,0,0,1,1,1,1,1,1,3;
  // moves 1,1,1,1,1,1,2,2,2,2,7 (sorted).
  CHECK(s.mandatory == (t5d::tools::Spread{2.0, 1.0, 4.0, 5.0}));
  CHECK(s.optional_boards == (t5d::tools::Spread{9.0 / 11.0, 1.0, 1.0, 3.0}));
  CHECK(s.moves == (t5d::tools::Spread{21.0 / 11.0, 1.0, 2.0, 7.0}));
  CHECK(s.moves_classes == Counts({6, 4, 0, 1}));
  CHECK(s.optional_moves_classes == Counts({6, 4, 0, 1}));
  CHECK_EQ(s.one_move_over_1_mandatory, std::size_t{1});
  CHECK_EQ(s.with_optional_moves, std::size_t{5});
  CHECK_EQ(s.with_optional_by_source, std::size_t{4});
  CHECK_EQ(s.with_optional_when_made, std::size_t{6});
  CHECK_EQ(s.optional_moves, 7);
  CHECK_EQ(s.optional_by_source, 6);
  CHECK_EQ(s.optional_when_made, 8);
  CHECK_EQ(s.optional_played, 7);
  CHECK_EQ(s.boards_played, 24);
  CHECK_EQ(s.with_travel, std::size_t{6});
  CHECK_EQ(s.with_branch, std::size_t{3});
  CHECK_EQ(s.rewound, std::size_t{3});
  CHECK_EQ(s.with_moves_after_shift, std::size_t{4});
  CHECK_EQ(s.moves_after_shift, 6);
  CHECK_EQ(s.with_mandatory_released, std::size_t{3});
  CHECK_EQ(s.mandatory_released, 5);
  CHECK_EQ(s.mandatory_gained, 1);
  CHECK_EQ(s.moves_on_released, 2);
  CHECK_EQ(s.moves_on_gained, 1);
  CHECK_EQ(s.jumps_between_mandatory, 1);
  CHECK_EQ(s.optional_onto_mandatory, 1);
  CHECK_EQ(s.mandatory_onto_optional, 1);

  // No action: nothing counted. An action without a mandatory board or a
  // move is not one (T5).
  CHECK_EQ(t5d::tools::turn_stats({}).actions, std::size_t{0});
  TurnWidth none = width(0, 1, 1);
  const TurnWidth* bad[] = {&none};
  CHECK_THROWS_AS(t5d::tools::turn_stats(bad), std::invalid_argument);
  none = width(1, 0, 0);
  CHECK_THROWS_AS(t5d::tools::turn_stats(bad), std::invalid_argument);
}

// ---- t5d-measure --turns / --turn-summary, end to end -----------------------------------------

namespace {

std::string read_all(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> v;
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t nl = text.find('\n', at);
    if (nl == std::string::npos) nl = text.size();
    if (nl > at) v.push_back(text.substr(at, nl - at));
    at = nl + 1;
  }
  return v;
}

// Runs t5d-measure with `args`; standard output's lines, and the exit status.
std::vector<std::string> run_measure(const std::vector<std::string>& args, int& status) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "measure-exe";
  fs::create_directories(dir);
  const fs::path out = dir / "stdout.txt";
  const fs::path err = dir / "stderr.txt";
  fs::remove(out);
  fs::remove(err);
  std::string command = quoted(fs::path(T5D_MEASURE_EXE).make_preferred().string());
  for (const std::string& a : args) command += " " + quoted(a);
  command += " > " + quoted(out.string()) + " 2> " + quoted(err.string());
#if defined(_WIN32)
  command = "\"" + command + "\"";  // cmd /c strips one pair of quotes around the whole
#endif
  std::fflush(nullptr);
  status = std::system(command.c_str());
#if !defined(_WIN32)
  status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
  return lines_of(read_all(out));
}

// The integer value of "name": in a JSON line (-1 if absent or not one).
long long int_field(const std::string& line, const std::string& name) {
  for (const char* before : {"{", ","}) {
    const std::string key = before + ("\"" + name + "\":");
    const std::size_t at = line.find(key);
    if (at == std::string::npos) continue;
    std::size_t p = at + key.size();
    std::size_t end = p;
    while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) != 0 ||
                                 (end == p && line[end] == '-'))) {
      ++end;
    }
    if (end == p) return -1;
    return std::stoll(line.substr(p, end - p));
  }
  return -1;
}

bool has(const std::string& line, const std::string& text) {
  return line.find(text) != std::string::npos;
}

std::vector<std::string> with(const std::vector<std::string>& lines, const std::string& text) {
  std::vector<std::string> out;
  for (const std::string& l : lines) {
    if (has(l, text)) out.push_back(l);
  }
  return out;
}

// The one turn_summary line of `reading` and `range` (failing if there is not
// exactly one).
std::string summary_line(const std::vector<std::string>& lines, const std::string& reading,
                         const std::string& range) {
  const std::vector<std::string> found =
      with(lines, "{\"event\":\"turn_summary\",\"reading\":\"" + reading + "\",\"range\":\"" +
                      range + "\",");
  CHECK_EQ(found.size(), std::size_t{1});
  return found.empty() ? std::string() : found[0];
}

std::string data_path(const char* name) {
  return (std::filesystem::path(T5D_ENGINE_DIR) / "testdata" / "turn-width" / name).string();
}

}  // namespace

TEST(measure_exe_turns_on_hand_written_records) {
  // two-timelines.5dpgn (Standard - Two Timelines), four actions:
  //   1. (-0T1)e3 (+0T1)e3       two mandatory boards (T13), two moves
  //    / (+0T1)Nb8>(-0T1)b6       a jump between the two present boards (J4)
  //   2. (+0T2)Ng1>>(+0T1)g3      a rewinding branch (T8), then a move on the
  //      (-0T2)d3                 board it released, after the shift (R8)
  //    / (+1T1)e6 (-0T2)d6        the new timeline's board is the present's;
  //                               -0's is ahead: an optional move, after the shift
  // prefix.5dpgn: its first action pair - a record that repeats the first.
  // refused.5dpgn (Standard): 1. e4 / e5 2. Ke3 - refused at its third action.
  namespace fs = std::filesystem;
  const fs::path data = fs::path(T5D_ENGINE_DIR) / "testdata" / "turn-width";
  const std::string two = (data / "two-timelines.5dpgn").string();
  const std::string prefix = (data / "prefix.5dpgn").string();
  const std::string refused = (data / "refused.5dpgn").string();
  int status = -1;
  const std::vector<std::string> lines =
      run_measure({"--turns", "--turn-summary", two, prefix, refused}, status);
  CHECK_EQ(status, 0);

  // Every action is played twice and checked; a mismatch would end the
  // record in "error" (exit status 1).
  const std::vector<std::string> ends = with(lines, "\"event\":\"end\"");
  REQUIRE_EQ(ends.size(), std::size_t{3});
  CHECK(has(ends[0], "\"status\":\"ok\""));
  CHECK(has(ends[1], "\"status\":\"ok\""));
  CHECK(has(ends[2], "\"status\":\"rejected\""));
  CHECK(has(ends[2], "\"reason\":\"illegal_move\""));
  CHECK_EQ(int_field(ends[2], "actions_accepted"), 2);

  std::vector<std::string> turns;
  for (const std::string& l : with(lines, "\"event\":\"turn\"")) {
    if (has(l, "two-timelines.5dpgn")) turns.push_back(l);
  }
  REQUIRE_EQ(turns.size(), std::size_t{4});
  struct Expected {
    const char* side;
    long long boards, timelines, mandatory, playable, optional, moves, from_mandatory,
        optional_moves, when_made, played, jumps, branches, between, after_shift, rewinds,
        released, on_released, by_source, optional_played;
  };
  const Expected expected[4] = {
      {"white", 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {"black", 4, 2, 2, 2, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0},
      {"white", 6, 2, 2, 2, 0, 2, 2, 0, 1, 2, 1, 1, 0, 1, 1, 0, 1, 0, 0},
      {"black", 9, 3, 1, 3, 2, 2, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 1},
  };
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string& t = turns[i];
    const Expected& e = expected[i];
    CHECK_EQ(int_field(t, "index"), static_cast<long long>(i));
    CHECK(has(t, std::string("\"side\":\"") + e.side + "\""));
    CHECK_EQ(int_field(t, "boards"), e.boards);
    CHECK_EQ(int_field(t, "timelines"), e.timelines);
    CHECK_EQ(int_field(t, "mandatory"), e.mandatory);
    CHECK_EQ(int_field(t, "playable"), e.playable);
    CHECK_EQ(int_field(t, "optional"), e.optional);
    CHECK_EQ(int_field(t, "moves"), e.moves);
    CHECK_EQ(int_field(t, "moves_from_mandatory"), e.from_mandatory);
    CHECK_EQ(int_field(t, "optional_moves"), e.optional_moves);
    CHECK_EQ(int_field(t, "optional_when_made"), e.when_made);
    CHECK_EQ(int_field(t, "boards_played"), e.played);
    CHECK_EQ(int_field(t, "jumps"), e.jumps);
    CHECK_EQ(int_field(t, "branches"), e.branches);
    CHECK_EQ(int_field(t, "jumps_between_mandatory"), e.between);
    CHECK_EQ(int_field(t, "moves_after_shift"), e.after_shift);
    CHECK_EQ(int_field(t, "rewinds"), e.rewinds);
    CHECK_EQ(int_field(t, "mandatory_released"), e.released);
    CHECK_EQ(int_field(t, "moves_on_released"), e.on_released);
    CHECK_EQ(int_field(t, "optional_by_source"), e.by_source);
    CHECK_EQ(int_field(t, "optional_played"), e.optional_played);
    CHECK_EQ(int_field(t, "mandatory_onto_optional"), 0);
    CHECK_EQ(int_field(t, "optional_onto_mandatory"), 0);
  }
  // Each action's position after it is the next action's start.
  for (std::size_t i = 0; i + 1 < 4; ++i) {
    const std::size_t a = turns[i].find("\"fingerprint_after\":\"");
    const std::size_t b = turns[i + 1].find("\"fingerprint\":\"");
    REQUIRE(a != std::string::npos && b != std::string::npos);
    CHECK_EQ(turns[i].substr(a + 21, 32), turns[i + 1].substr(b + 15, 32));
  }

  // The summary: prefix.5dpgn repeats two-timelines.5dpgn, so the strict
  // reading counts one game of four actions; the refused record's two
  // actions are apart.
  const std::vector<std::string> head = with(lines, "\"event\":\"turn_summary_records\"");
  REQUIRE_EQ(head.size(), std::size_t{1});
  CHECK_EQ(int_field(head[0], "records"), 3);
  CHECK_EQ(int_field(head[0], "replayed"), 2);
  CHECK_EQ(int_field(head[0], "refused"), 1);
  CHECK_EQ(int_field(head[0], "errors"), 0);
  CHECK_EQ(int_field(head[0], "actions_checked"), 4 + 2 + 2);
  CHECK_EQ(int_field(head[0], "strict_games"), 1);
  CHECK_EQ(int_field(head[0], "grouped_games"), 1);
  CHECK(has(head[0], "prefix.5dpgn\",\"repeats\":"));
  // The grouped game of two records is listed.
  const std::vector<std::string> groups = with(lines, "\"event\":\"turn_summary_group\"");
  REQUIRE_EQ(groups.size(), std::size_t{1});
  CHECK(has(groups[0], "two-timelines.5dpgn\",\"also\":[\""));
  CHECK(has(groups[0], "prefix.5dpgn\"]}"));
  const std::string strict = summary_line(lines, "strict", "<50");
  const std::string grouped = summary_line(lines, "grouped", "<50");
  const std::string apart = summary_line(lines, "refused", "<50");
  std::string by_two;
  for (const std::string& l : lines) {
    if (has(l, "\"reading\":\"strict\",\"range\":\"<50\",\"mandatory_class\":\"2\"")) by_two = l;
  }
  REQUIRE(!strict.empty() && !grouped.empty() && !apart.empty() && !by_two.empty());
  CHECK_EQ(int_field(strict, "actions"), 4);
  CHECK_EQ(int_field(strict, "games"), 1);
  CHECK(has(strict, "\"mandatory\":{\"1\":1,\"2\":3,\"3\":0,\"4+\":0}"));
  CHECK(has(strict, "\"mandatory_over_1\":0.75,"));
  // Playable 2, 2, 2, 3; optional boards 0, 0, 0, 2 (p90 1.4: 0.7 of the way
  // from the third to the fourth).
  CHECK(has(strict, "\"playable_over_1\":1,\"with_optional_boards\":0.25,"));
  CHECK(has(strict, "\"optional_boards_mean\":0.5,\"optional_boards_median\":0,"
                    "\"optional_boards_p90\":1.4,\"optional_boards_max\":2,"));
  CHECK(has(strict, "\"optional_moves\":{\"0\":3,\"1\":1,\"2\":0,\"3+\":0}"));
  CHECK(has(strict, "\"optional_moves_mean\":0.25,\"with_optional_moves\":0.25,"));
  CHECK(has(strict, "\"optional_by_source_mean\":0.25,\"with_optional_by_source\":0.25,"));
  // The move on the board the rewind released is optional when made.
  CHECK(has(strict, "\"optional_when_made_mean\":0.5,\"with_optional_when_made\":0.5,"));
  CHECK(has(strict, "\"optional_played_mean\":0.25,"));
  CHECK(has(strict, "\"moves\":{\"1\":1,\"2\":3,\"3\":0,\"4+\":0}"));
  CHECK(has(strict, "\"moves_mean\":1.75,\"moves_median\":2,\"moves_p90\":2,\"moves_max\":2,"));
  CHECK_EQ(int_field(strict, "one_move_over_1_mandatory"), 1);  // the jump between them
  CHECK(has(strict, "\"with_travel\":0.5,\"with_branch\":0.25,"));
  CHECK(has(strict, "\"with_moves_after_shift\":0.5,"));
  CHECK_EQ(int_field(strict, "jumps_between_mandatory"), 1);
  CHECK_EQ(int_field(strict, "optional_onto_mandatory"), 0);
  CHECK_EQ(int_field(strict, "mandatory_onto_optional"), 0);
  CHECK_EQ(int_field(strict, "moves_on_released"), 1);
  CHECK_EQ(int_field(grouped, "actions"), 4);
  CHECK_EQ(int_field(grouped, "games"), 1);
  CHECK_EQ(int_field(apart, "actions"), 2);
  CHECK_EQ(int_field(apart, "records"), 1);
  CHECK_EQ(int_field(by_two, "actions"), 3);
  CHECK_EQ(int_field(by_two, "games"), 1);
  // Every action under 10 boards; nothing at 50 or more.
  CHECK_EQ(int_field(summary_line(lines, "strict", "1-9"), "actions"), 4);
  CHECK_EQ(int_field(summary_line(lines, "strict", "1-9"), "games"), 1);
  CHECK_EQ(int_field(summary_line(lines, "strict", "10-19"), "actions"), 0);
  CHECK_EQ(int_field(summary_line(lines, "strict", "all"), "actions"), 4);
  for (const std::string& l : with(lines, "\"reading\":\"strict\",\"range\":\"50-99\"")) {
    CHECK_EQ(int_field(l, "actions"), 0);
    CHECK_EQ(int_field(l, "games"), 0);
  }

  // Without --turns and --turn-summary the tool prints no turn line.
  const std::vector<std::string> plain = run_measure({two}, status);
  CHECK_EQ(status, 0);
  CHECK(with(plain, "\"event\":\"turn").empty());
  CHECK_EQ(with(plain, "\"event\":\"position\"").size(), std::size_t{5});
}

TEST(measure_exe_turn_summary_classifies_records) {
  // last-action.5dpgn is two-timelines.5dpgn but for its last action ("/
  // (+1T1)e6" alone): neither is a prefix of the other, so the strict reading
  // counts both - four actions each - and the grouped reading too (they share
  // four positions of five: not a long enough opening to be one game).
  // prefix.5dpgn is a prefix of both: it repeats the first. truncated.5dpgn's
  // move text stops after its second action (1. d4 / d5 2. c): every action
  // read is played, yet the record is refused - its two actions apart, as at
  // M7 - and it is not among the games.
  const std::string two = data_path("two-timelines.5dpgn");
  const std::string prefix = data_path("prefix.5dpgn");
  const std::string last = data_path("last-action.5dpgn");
  const std::string truncated = data_path("truncated.5dpgn");
  int status = -1;
  const std::vector<std::string> lines =
      run_measure({"--turn-summary", two, prefix, last, truncated}, status);
  CHECK_EQ(status, 0);
  const std::vector<std::string> ends = with(lines, "\"event\":\"end\"");
  REQUIRE_EQ(ends.size(), std::size_t{4});
  CHECK(has(ends[2], "\"status\":\"ok\""));
  CHECK(has(ends[3], "\"status\":\"rejected\""));
  CHECK(has(ends[3], "\"truncated\":true"));
  CHECK(has(ends[3], "\"reason\":\"unreadable\""));
  CHECK_EQ(int_field(ends[3], "actions_accepted"), 2);
  CHECK_EQ(int_field(ends[3], "actions_in_record"), 2);

  const std::vector<std::string> head = with(lines, "\"event\":\"turn_summary_records\"");
  REQUIRE_EQ(head.size(), std::size_t{1});
  CHECK_EQ(int_field(head[0], "records"), 4);
  CHECK_EQ(int_field(head[0], "replayed"), 3);
  CHECK_EQ(int_field(head[0], "refused"), 1);
  CHECK_EQ(int_field(head[0], "errors"), 0);
  CHECK_EQ(int_field(head[0], "actions_checked"), 4 + 2 + 4 + 2);
  CHECK_EQ(int_field(head[0], "strict_games"), 2);
  CHECK_EQ(int_field(head[0], "grouped_games"), 2);
  CHECK(has(head[0], "\"strict_repeats\":[{\"record\":"));
  CHECK(has(head[0], "prefix.5dpgn\",\"repeats\":"));
  CHECK(!has(head[0], "last-action.5dpgn\",\"repeats\":"));
  CHECK(!has(head[0], "truncated.5dpgn"));

  const std::string strict = summary_line(lines, "strict", "<50");
  CHECK_EQ(int_field(strict, "actions"), 8);
  CHECK_EQ(int_field(strict, "games"), 2);
  CHECK(has(strict, "\"moves\":{\"1\":3,\"2\":5,\"3\":0,\"4+\":0}"));
  CHECK(has(strict, "\"optional_moves\":{\"0\":7,\"1\":1,\"2\":0,\"3+\":0}"));
  const std::string grouped = summary_line(lines, "grouped", "<50");
  CHECK_EQ(int_field(grouped, "actions"), 8);
  CHECK_EQ(int_field(grouped, "games"), 2);
  const std::string apart = summary_line(lines, "refused", "<50");
  CHECK_EQ(int_field(apart, "actions"), 2);
  CHECK_EQ(int_field(apart, "records"), 1);
  CHECK(has(apart, "\"mandatory\":{\"1\":2,\"2\":0,\"3\":0,\"4+\":0}"));
}

TEST(measure_exe_turn_summary_ranges_at_their_edges) {
  // Custom positions (one timeline of 3x3 boards, the kings alone) whose
  // first action starts at 9, 49, 99 and 199 boards; each action adds one
  // board, so the second starts at 10, 50, 100 and 200 - every edge of the
  // ranges, and the first of the ten-board buckets.
  int status = -1;
  const std::vector<std::string> lines = run_measure(
      {"--turn-summary", data_path("edges/start-9.5dpgn"), data_path("edges/start-49.5dpgn"),
       data_path("edges/start-99.5dpgn"), data_path("edges/start-199.5dpgn")},
      status);
  CHECK_EQ(status, 0);
  const std::vector<std::string> head = with(lines, "\"event\":\"turn_summary_records\"");
  REQUIRE_EQ(head.size(), std::size_t{1});
  CHECK_EQ(int_field(head[0], "replayed"), 4);
  CHECK_EQ(int_field(head[0], "strict_games"), 4);
  struct Expected {
    const char* range;
    long long actions, games;
  };
  const Expected expected[] = {
      {"<50", 3, 2},     {"50-99", 2, 2}, {"100-199", 2, 2}, {"200+", 1, 1},
      {"1-9", 1, 1},     {"10-19", 1, 1}, {"20-29", 0, 0},   {"30-39", 0, 0},
      {"40-49", 1, 1},   {"50-59", 1, 1}, {"60-69", 0, 0},   {"70-79", 0, 0},
      {"80-89", 0, 0},   {"90-99", 1, 1}, {"all", 8, 4},
  };
  for (const Expected& e : expected) {
    const std::string line = summary_line(lines, "strict", e.range);
    CHECK_EQ(int_field(line, "actions"), e.actions);
    CHECK_EQ(int_field(line, "games"), e.games);
  }
  // Per mandatory class: every action has one mandatory board.
  for (const std::string& l : with(lines, "\"event\":\"turn_summary_by_mandatory\"")) {
    const bool one = has(l, "\"mandatory_class\":\"1\"");
    if (has(l, "\"range\":\"<50\"")) CHECK_EQ(int_field(l, "actions"), one ? 3 : 0);
    if (has(l, "\"range\":\"200+\"")) CHECK_EQ(int_field(l, "actions"), one ? 1 : 0);
  }
}
