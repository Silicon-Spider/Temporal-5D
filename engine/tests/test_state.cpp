// The game state machine (rules reference T1-T13, J6; Q1-Q6, Q9): the
// present, mandatory and optional boards, submission, activation and
// reactivation, rewinding by branching and by reactivation, and the hooks
// left for the check ruling.

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using t5d_test::act;
using t5d_test::contains;
using t5d_test::game;
using t5d_test::play;
using t5d_test::texts;

namespace {

constexpr BoardSize k5x5{5, 5};
constexpr BoardSize k4x4{4, 4};

std::vector<int> ints(std::initializer_list<int> v) { return std::vector<int>(v); }

}  // namespace

// ---- T1, T4, T5, T9: turns and the present -------------------------------------------------

TEST(state_standard_start) {
  const GameState s(standard_start());
  CHECK(s.to_move() == Colour::White);
  CHECK_EQ(s.present(), half_turn(1, Colour::White));
  CHECK(!s.present_shifted());
  CHECK(s.mandatory_timelines() == ints({0}));
  CHECK(s.playable_timelines() == ints({0}));
  CHECK(!s.obligations_met());
  CHECK(s.may_move());
  CHECK(s.inactive_timelines().empty());
  CHECK(s.submission_status() == SubmitStatus::ObligationsUnmet);
  // Ordinary chess's twenty first moves: nothing else exists to move to.
  CHECK_EQ(count_moves(s), std::size_t{20});
}

TEST(state_the_present_advances_one_half_step_per_turn) {
  // T9: with no timeline created, the present moves (T,w) -> (T,b) after
  // White's turn and (T,b) -> (T+1,w) after Black's. T5: each turn needs a
  // move, and submission only once the present has shifted.
  GameState s(standard_start());
  CHECK_THROWS_AS(s.submit(), std::logic_error);
  CHECK(s == GameState(standard_start()));  // a refused submission changes nothing
  play(s, "(0T1)Pe2(0T1)e4");
  CHECK(s.present_shifted());
  CHECK(s.obligations_met());
  CHECK(!s.may_move());  // Q4's default: no move once the present has shifted
  CHECK(t5d_test::moves_of(s).empty());
  CHECK(s.submission_status() == SubmitStatus::Safe);  // C4 (R1): no royal is left capturable
  s.submit();
  CHECK(s.to_move() == Colour::Black);
  CHECK_EQ(s.present(), half_turn(1, Colour::Black));
  CHECK(!s.present_shifted());
  act(s, {"(0T1)Pe7(0T1)e5"});
  CHECK(s.to_move() == Colour::White);
  CHECK_EQ(s.present(), half_turn(2, Colour::White));
  CHECK_EQ(s.multiverse().board_count(), std::size_t{3});
}

TEST(state_moves_are_refused_once_the_obligations_are_met) {
  // T5 with Q4's default: once the present has shifted, the action must be
  // submitted; apply refuses another move, leaving the state unchanged.
  GameState s(standard_start());
  const Move second = *t5d_test::find_move(s, "(0T1)Pd2(0T1)d4");
  play(s, "(0T1)Pe2(0T1)e4");
  const GameState before = s;
  CHECK_THROWS_AS(s.apply(second), std::logic_error);
  CHECK(s == before);
}

// ---- T6: mandatory and optional boards ---------------------------------------------------------

namespace {

// 0's playable board at T2 (the present); +1, White's first timeline
// (active), playable at T3 - ahead of the present; +2, White's second
// (inactive: Black has created none), playable at T1 - behind the present.
GameState optional_boards(const RuleOptions& rules = {}) {
  return game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/R4:0:2:w]",
               "[5/5/5/5/R4:+1:3:w]", "[5/5/5/5/R4:+2:1:w]"},
              k5x5, rules);
}

}  // namespace

TEST(state_only_active_boards_in_the_present_are_mandatory) {
  // T4: the inactive +2, though earlier, does not set the present. T6: only
  // 0's board is mandatory; +1 (ahead) and +2 (inactive) are optional, and
  // playable: their moves are generated.
  GameState s = optional_boards();
  CHECK(s.active(1));
  CHECK(!s.active(2));
  CHECK_EQ(s.present(), half_turn(2, Colour::White));
  CHECK(s.mandatory_timelines() == ints({0}));
  CHECK(s.playable_timelines() == ints({0, 1, 2}));
  const std::vector<std::string> all = texts(s);
  CHECK(contains(all, "(+1T3)Ra1(+1T3)a2"));
  CHECK(contains(all, "(+2T1)Ra1(+2T1)a2"));
  // Optional moves first, then the mandatory one (Q4's default orders them).
  play(s, "(+1T3)Ra1(+1T3)a2");
  play(s, "(+2T1)Ra1(+2T1)a2");
  CHECK(!s.obligations_met());
  play(s, "(0T2)Ra1(0T2)a2");
  CHECK(s.obligations_met());
  CHECK(s.submission_status() == SubmitStatus::Safe);  // no royal piece at all (C13)
}

TEST(state_optional_boards_after_the_shift_q4) {
  // Q4: by default nothing may be played after the present has shifted; the
  // alternative lets the mover go on to optional boards before submitting.
  GameState strict = optional_boards();
  play(strict, "(0T2)Ra1(0T2)a2");
  CHECK(t5d_test::moves_of(strict).empty());
  RuleOptions r;
  r.moves_after_shift = true;
  GameState loose = optional_boards(r);
  play(loose, "(0T2)Ra1(0T2)a2");
  CHECK(loose.obligations_met());
  CHECK(loose.may_move());
  CHECK(t5d_test::has_move(loose, "(+1T3)Ra1(+1T3)a2"));
  CHECK(t5d_test::has_move(loose, "(+2T1)Ra1(+2T1)a2"));
  play(loose, "(+2T1)Ra1(+2T1)a2");
  CHECK(loose.submission_status() == SubmitStatus::Safe);  // no royal piece at all (C13)
}

TEST(state_every_active_board_is_mandatory_under_q5) {
  // Q5's alternative: every active timeline's playable board of the mover's
  // must be played - +1 ahead of the present too (not the inactive +2).
  RuleOptions r;
  r.mandatory = MandatoryBoards::AllActive;
  GameState s = optional_boards(r);
  CHECK(s.mandatory_timelines() == ints({0, 1}));
  play(s, "(+1T3)Ra1(+1T3)a2");
  CHECK(!s.obligations_met());
  play(s, "(0T2)Ra1(0T2)a2");
  CHECK(s.obligations_met());
  // Played in the other order, the present shifts first, but the obligations
  // are met only once +1 is played too: its move stays available.
  GameState t = optional_boards(r);
  play(t, "(0T2)Ra1(0T2)a2");
  CHECK(t.present_shifted());
  CHECK(!t.obligations_met());
  CHECK(t.mandatory_timelines() == ints({1}));
  CHECK(t.may_move());
  CHECK(t5d_test::has_move(t, "(+1T3)Ra1(+1T3)a2"));
}

// ---- T7, T8: a new timeline, rewinding by branching ------------------------------------------

TEST(state_rewinding_by_branching) {
  // T8, in Standard - Two Timelines: after 1. e3 on both boards / e6 on both,
  // White's knight jumps from +0 back to T1 - a branch onto a history board.
  // White's first timeline is active (T10), so the present moves back to its
  // first board, (+1, T1, black); the board of -0 left in T2 is ahead of the
  // present and need not be played: White may submit.
  GameState s(standard_two_timelines_start());
  CHECK(s.mandatory_timelines() == ints({-1, 0}));
  act(s, {"(-0T1)Pe2(-0T1)e3", "(+0T1)Pe2(+0T1)e3"});
  act(s, {"(-0T1)Pe7(-0T1)e6", "(+0T1)Pe7(+0T1)e6"});
  CHECK(s.mandatory_timelines() == ints({-1, 0}));
  const Move branch = play(s, "(+0T2)Ng1(+0T1)g3");
  CHECK(branch.branching());
  CHECK(s.multiverse().has_timeline(1));                   // "+1": White's first (L7)
  CHECK_EQ(timeline_label_string(1, true), std::string("+1"));
  CHECK(s.active(1));
  CHECK_EQ(s.multiverse().timeline(1)->first_half_turn(), half_turn(1, Colour::Black));
  CHECK_EQ(s.present(), half_turn(1, Colour::Black));       // rewound
  CHECK(s.obligations_met());
  CHECK(s.mandatory_timelines().empty());
  CHECK(s.playable_by_mover(-1));                            // -0 is still White's, unplayed
  // T7: the new timeline's first board is Black's: White can neither move on it
  // nor reach it in this action.
  CHECK(!s.playable_by_mover(1));
  s.submit();
  CHECK(s.to_move() == Colour::Black);
  CHECK(s.mandatory_timelines() == ints({1}));               // Black must answer on +1
}

TEST(state_a_new_timeline_is_never_a_destination_for_its_creator) {
  // T7: in the action that created it, nothing of the creator's lands on the
  // new timeline (its first board is the opponent's colour). With Q4's
  // alternative the action can go on after the rewind; no move targets +1.
  RuleOptions r;
  r.moves_after_shift = true;
  GameState s(standard_two_timelines_start(), r);
  act(s, {"(-0T1)Pe2(-0T1)e3", "(+0T1)Pe2(+0T1)e3"});
  act(s, {"(-0T1)Pe7(-0T1)e6", "(+0T1)Pe7(+0T1)e6"});
  play(s, "(+0T2)Ng1(+0T1)g3");
  REQUIRE(s.may_move());
  const MoveList moves = t5d_test::moves_of(s);
  CHECK(!moves.empty());  // -0's board is still playable
  for (const Move& m : moves) {
    CHECK_NE(m.to_timeline, 1);
    CHECK_NE(m.from_timeline, 1);
  }
}

// ---- T10-T12: activation, reactivation, rewinding by reactivation ---------------------------

namespace {

// White to move. 0: T1w..T3w, a white rook c3 and a black rook e5 on the
// playable T3 board, a black knight a5 on the T2 white board and a white king
// a1 on the T2 black board (pieces for the copies to play); +1 (White's first
// timeline, active): one board at T3 with a white rook a1.
GameState two_branch_scenario(const RuleOptions& rules = {}) {
  return game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[n4/5/5/5/5:0:2:w]",
               "[5/5/5/5/K4:0:2:b]", "[4r/5/2R2/5/5:0:3:w]", "[5/5/5/5/R4:+1:3:w]"},
              k5x5, rules);
}

}  // namespace

TEST(state_count_rule_and_reactivation) {
  GameState s = two_branch_scenario();
  CHECK_EQ(s.present(), half_turn(3, Colour::White));
  CHECK(s.mandatory_timelines() == ints({0, 1}));
  // White's second timeline while Black has created none: inactive (T10; G:
  // "If a player branches twice before their opponent branches once, the
  // extra branches will be inactive").
  const Move branch = play(s, "(0T3)Rc3(0T2)c3");
  CHECK(branch.branching());
  REQUIRE(s.multiverse().has_timeline(2));
  CHECK(!s.active(2));
  CHECK(s.inactive_timelines() == ints({2}));
  CHECK_EQ(s.multiverse().timeline(2)->creation_order(), 2);
  // T8: an inactive branch never moves the present - +1 must still be played.
  CHECK_EQ(s.present(), half_turn(3, Colour::White));
  CHECK(s.mandatory_timelines() == ints({1}));
  act(s, {"(+1T3)Ra1(+1T3)a2"});
  // Black's turn: the present is (T3, black); the inactive +2 is playable
  // (T6) but optional.
  CHECK_EQ(s.present(), half_turn(3, Colour::Black));
  CHECK(s.mandatory_timelines() == ints({0, 1}));
  CHECK(s.playable_by_mover(2));
  // Black's first timeline: active (White has created 2 >= 0). T11: it
  // activates White's earliest-created inactive timeline, +2 - whose playable
  // board, (+2, T2, black), is behind the present and Black's: the present
  // jumps back to it (T12) and Black must play it (Q6's default).
  play(s, "(0T3)Re5(0T2)e5");
  REQUIRE(s.multiverse().has_timeline(-1));
  CHECK(s.active(-1));
  CHECK(s.active(2));
  CHECK(s.inactive_timelines().empty());
  CHECK_EQ(s.present(), half_turn(2, Colour::Black));
  CHECK(s.mandatory_timelines() == ints({2}));
  CHECK(!s.obligations_met());
  // The copy on +2 holds the black knight; playing it shifts the present to
  // White's colour, (-1, T3, white).
  play(s, "(+2T2)Na5(+2T2)b3");
  CHECK_EQ(s.present(), half_turn(3, Colour::White));
  CHECK(s.obligations_met());
  s.submit();
  CHECK(s.to_move() == Colour::White);
}

TEST(state_reactivation_deferred_under_q6) {
  // Q6's alternative: the reactivated board of Black's own colour does not
  // oblige Black in this action; it does from Black's next turn on.
  RuleOptions r;
  r.reactivation = Reactivation::Deferred;
  GameState s = two_branch_scenario(r);
  play(s, "(0T3)Rc3(0T2)c3");
  act(s, {"(+1T3)Ra1(+1T3)a2"});
  play(s, "(0T3)Re5(0T2)e5");
  CHECK(s.active(2));
  REQUIRE_EQ(s.deferred_timelines().size(), std::size_t{1});
  CHECK_EQ(s.deferred_timelines()[0].timeline, 2);
  CHECK(s.deferred_timelines()[0].owner == Colour::Black);
  // The present is the earliest of the others: (-1, T3, white) - shifted.
  CHECK_EQ(s.present(), half_turn(3, Colour::White));
  CHECK(s.obligations_met());
  s.submit();
  // White's turn: still deferred.
  CHECK_EQ(s.present(), half_turn(3, Colour::White));
  CHECK_EQ(s.deferred_timelines().size(), std::size_t{1});
  // White plays -1 (the king copied there by Black's branch); then Black's
  // turn begins and +2 obliges Black.
  CHECK(s.mandatory_timelines() == ints({-1}));
  while (!s.obligations_met()) {
    const MoveList moves = t5d_test::moves_of(s);
    REQUIRE(!moves.empty());
    // Prefer a move within a board: keeps the scenario simple.
    const Move* chosen = &moves.front();
    for (const Move& m : moves) {
      if (m.physical() && s.mandatory(m.from_timeline)) {
        chosen = &m;
        break;
      }
    }
    s.apply(*chosen);
  }
  s.submit();
  CHECK(s.to_move() == Colour::Black);
  CHECK(s.deferred_timelines().empty());
  CHECK_EQ(s.present(), half_turn(2, Colour::Black));
  CHECK(s.mandatory_timelines() == ints({2}));
  // The key carries the deferral under this ruleset.
  CHECK(r.history_dependent_activity());
}

TEST(state_rewinding_by_reactivation_onto_the_opponents_colour) {
  // T12: Black's first timeline reactivates White's second, whose playable
  // board (T2, white) is behind the present: the present jumps back to it -
  // White's colour - so Black's action is complete although the boards in the
  // T3 column were not all played.
  const GameState start = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/5:0:2:w]",
                                "[5/5/5/5/5:0:2:b]", "[5/5/5/5/5:0:3:w]", "[4r/5/5/5/5:0:3:b]",
                                "[5/5/5/5/5:+1:3:b]", "[5/5/5/5/5:+2:2:w]"},
                               k5x5);
  CHECK(start.to_move() == Colour::Black);  // Q9: the present, (T3, black)
  CHECK(!start.active(2));
  CHECK(start.mandatory_timelines() == ints({0, 1}));
  GameState s = start;
  play(s, "(0T3)Re5(0T2)e5");
  CHECK(s.active(-1));
  CHECK(s.active(2));
  CHECK_EQ(s.present(), half_turn(2, Colour::White));
  CHECK(s.obligations_met());
  CHECK(s.playable_by_mover(1));  // +1's T3 board: Black's, unplayed, no longer required
  s.submit();
  CHECK(s.mandatory_timelines() == ints({2}));
}

TEST(state_consecutive_creations_q1) {
  // Q1: Black has created one timeline, White none. White creates two in a
  // row (Black's action between them creates nothing). The count rule (T10,
  // default): both active. The "consecutive" reading: the second inactive.
  const std::vector<std::string> boards = {"[k4/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]",
                                           "[5/5/2R2/5/5:0:2:w]", "[5/5/4R/5/5:-1:2:w]"};
  for (const bool consecutive : {false, true}) {
    RuleOptions r;
    if (consecutive) r.activation = ActivationRule::Consecutive;
    GameState s = game(boards, k5x5, r);
    CHECK(s.active(-1));
    act(s, {"(0T2)Rc3(0T1)c3"});  // White's first: the present rewinds to (+1, T1, black)
    CHECK(s.active(1));
    CHECK(s.last_creator() == Colour::White);
    act(s, {"(+1T1)Ka5(+1T1)a4"});
    play(s, "(-1T2)Re3(0T2)e3");  // White's second, onto 0's history board
    REQUIRE(s.multiverse().has_timeline(2));
    CHECK_EQ(s.active(2), !consecutive);
  }
}

TEST(state_the_last_creator_counts_only_under_q1_consecutive) {
  // Who made the last creation is read by Q1's Consecutive reading alone. So
  // under the count rule a state reached in play equals - in value and in
  // key - the same boards read afresh (which record no last creator); under
  // Consecutive it does not, since the next creation's activity depends on it.
  const std::vector<std::string> boards = {"[k4/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]",
                                           "[5/5/2R2/5/5:0:2:w]", "[5/5/4R/5/5:-1:2:w]"};
  for (const bool consecutive : {false, true}) {
    RuleOptions r;
    if (consecutive) r.activation = ActivationRule::Consecutive;
    GameState s = game(boards, k5x5, r);
    act(s, {"(0T2)Rc3(0T1)c3"});
    REQUIRE(s.last_creator() == Colour::White);
    const GameState fresh(s.multiverse(), r);
    CHECK(!fresh.last_creator().has_value());
    CHECK_EQ(fresh == s, !consecutive);
    CHECK_EQ(fresh.zobrist() == s.zobrist(), !consecutive);
  }
}

TEST(state_initial_timelines_and_q2) {
  // T10, T13: initial timelines are always active. Q2's alternative counts
  // "-0" as Black's first creation. Positions read from 5DFEN: Black has
  // created "-1"; White "+1" and "+2".
  const std::vector<std::string> boards = {"[4k/5/5/5/5:-1:1:w]", "[5/5/5/5/5:-0:1:w]",
                                           "[5/5/5/5/5:+0:1:w]", "[5/5/5/5/5:+1:1:w]",
                                           "[5/5/5/5/5:+2:1:w]"};
  const GameState neutral = game(boards, k5x5);
  CHECK(neutral.active(-1) && neutral.active(0));  // the -0/+0 pair (positions -1, 0)
  CHECK(neutral.active(-2));                       // Black's first: White has created 2
  CHECK(neutral.active(1));
  CHECK(neutral.active(2));                        // White's second: Black has created 1
  RuleOptions r;
  r.initial_credit = InitialTimelineCredit::MinusZeroToBlack;
  const GameState credited = game(boards, k5x5, r);
  CHECK(credited.active(-2));
  CHECK(credited.active(2));
  // Where it matters: White alone has created two - the second is inactive
  // by default, active with Black credited one.
  const std::vector<std::string> white_only = {"[5/5/5/5/5:-0:1:w]", "[5/5/5/5/5:+0:1:w]",
                                               "[5/5/5/5/5:+1:1:w]", "[5/5/5/5/5:+2:1:w]"};
  CHECK(!game(white_only, k5x5).active(2));
  CHECK(game(white_only, k5x5, r).active(2));
  // And Black's first timeline with White at none: active by default, not
  // when Black is credited the -0.
  const std::vector<std::string> black_only = {"[5/5/5/5/5:-1:1:w]", "[5/5/5/5/5:-0:1:w]",
                                               "[5/5/5/5/5:+0:1:w]"};
  CHECK(game(black_only, k5x5).active(-2));
  CHECK(!game(black_only, k5x5, r).active(-2));
  // The credit is for even starts only.
  const std::vector<std::string> odd = {"[5/5/5/5/5:-1:1:w]", "[5/5/5/5/5:0:1:w]"};
  CHECK(game(odd, k5x5, r).active(-1));
}

TEST(state_play_activation_q3) {
  // Q3: by default playing on an inactive timeline changes nothing; under
  // the "yes" reading, a move by the player who did not create it activates it.
  const std::vector<std::string> boards = {"[k4/5/5/5/5:0:1:b]", "[k4/5/5/5/5:+1:1:b]",
                                           "[r4/5/5/5/5:+2:1:b]"};
  GameState s = game(boards, k5x5);
  CHECK(s.to_move() == Colour::Black);
  CHECK(!s.active(2));
  play(s, "(+2T1)Ra5(+2T1)a4");
  CHECK(!s.active(2));
  RuleOptions r;
  r.play_activates = true;
  GameState t = game(boards, k5x5, r);
  play(t, "(+2T1)Ra5(+2T1)a4");
  CHECK(t.active(2));
  CHECK(t.inactive_timelines().empty());
}

TEST(state_timeline_tactician_needs_a_jump_t13) {
  // T13, in Misc - Timeline Tactician (rules reference V3): both initial
  // timelines are active and in the present, and -0 holds no White piece, so
  // White's first action must include a jump into it.
  const GameState s = game({"[k*bnr*/p*p*p*p*/4/4:-0:1:w]", "[4/4/P*P*P*P*/K*BNR*:+0:1:w]"},
                           k4x4);
  CHECK(s.mandatory_timelines() == ints({-1, 0}));
  // A move within +0 leaves -0 mandatory with nothing to play there.
  GameState stuck = s;
  play(stuck, "(+0T1)Pa2(+0T1)a3");
  CHECK(!stuck.obligations_met());
  CHECK(t5d_test::moves_of(stuck).empty());
  CHECK(stuck.submission_status() == SubmitStatus::ObligationsUnmet);
  // A jump from +0 onto -0's playable board plays both (J4, T3).
  GameState done = s;
  const Move jump = play(done, "(+0T1)Pa2(-0T1)a2");
  CHECK(!jump.branching());
  CHECK(done.obligations_met());
  GameState capture = s;
  const Move take = play(capture, "(+0T1)Nc1(-0T1)c3");
  CHECK(take.capture());
  CHECK(capture.obligations_met());
}

// ---- J6: order matters within an action -----------------------------------------------------

TEST(state_order_matters_within_an_action_j6) {
  // J6 (after Tricky Checkmates III's solution, in a position of our own): a
  // queen on -0 and a pawn on +0's b1.
  const std::vector<std::string> boards = {"[4k/5/5/5/2Q2:-0:1:w]", "[5/5/5/5/1P3:+0:1:w]"};
  // The pawn's jump first lands on -0's playable board: no branch, and -0's
  // move is used up (the queen can no longer move).
  GameState first = game(boards, k5x5);
  const Move jump = play(first, "(+0T1)Pb1(-0T1)b1");
  CHECK(!jump.branching());
  CHECK(!first.playable_by_mover(-1));
  CHECK_EQ(first.multiverse().timeline_count(), 2);
  // The queen first: -0's board at T1 becomes history, and the same pawn move
  // now branches - a third timeline.
  GameState second = game(boards, k5x5);
  play(second, "(-0T1)Qc1(-0T1)c2");
  const Move branch = play(second, "(+0T1)Pb1(-0T1)b1");
  CHECK(branch.branching());
  CHECK_EQ(second.multiverse().timeline_count(), 3);
  CHECK(second.multiverse().has_timeline(1));
}

// ---- Q9 and construction -------------------------------------------------------------------

TEST(state_the_player_to_move_is_inferred_from_the_present_q9) {
  // The owner of the earliest active playable board moves. Here the earliest
  // head of all - +2's, at (T1, black) - is inactive, so White moves at T2:
  // the case the 5DFEN reader could not settle without activity.
  const std::vector<std::string> boards = {"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]",
                                           "[5/5/5/5/5:0:2:w]", "[5/5/5/5/5:+1:2:w]",
                                           "[5/5/5/5/5:+2:1:b]"};
  std::vector<std::string_view> views(boards.begin(), boards.end());
  CHECK_THROWS_AS(multiverse_from_fen(std::span<const std::string_view>(views), k5x5,
                                      Parity::Infer),
                  std::invalid_argument);
  const GameState s = game(boards, k5x5);
  CHECK(s.to_move() == Colour::White);
  CHECK_EQ(s.present(), half_turn(2, Colour::White));
  CHECK_EQ(initial_present(s.multiverse()), half_turn(2, Colour::White));
  // Given explicitly and inconsistent with the present: refused (T5).
  CHECK_THROWS_WITH(game(boards, k5x5, {}, Colour::Black), std::invalid_argument, "present");
}

TEST(state_construction_needs_the_central_timelines) {
  // Every game starts with its central timeline(s) (L7, L8).
  const std::vector<std::string> no_centre = {"[5/5/5/5/5:+1:1:w]"};
  CHECK_THROWS_WITH(game_from_fen(no_centre, k5x5, Parity::Odd), std::invalid_argument,
                    "central");
  Multiverse empty(k5x5, false);
  CHECK_THROWS_AS(GameState(empty), std::invalid_argument);
}

// ---- the check ruling's hooks ---------------------------------------------------------------

namespace {

Verdict say_yes(const GameState&) { return Verdict::Yes; }
Verdict say_no(const GameState&) { return Verdict::No; }
Verdict say_unknown(const GameState&) { return Verdict::Unknown; }

}  // namespace

TEST(state_submission_consults_the_royal_safety_hook) {
  // C4 is reached through a hook: by default the engine's (t5d/check.hpp);
  // without one (CheckRuling::none()), a submittable action is Unchecked;
  // with any other, its verdict decides.
  GameState s(standard_start());
  play(s, "(0T1)Pe2(0T1)e4");
  CHECK(s.submission_status() == SubmitStatus::Safe);
  CHECK(s.submission_status(CheckRuling::none()) == SubmitStatus::Unchecked);
  CheckRuling unsafe;
  unsafe.royals_safe = &say_no;
  CHECK(s.submission_status(unsafe) == SubmitStatus::Unsafe);
  const GameState before = s;
  CHECK_THROWS_WITH(s.submit(unsafe), std::logic_error, "C4");
  CHECK(s == before);
  CheckRuling unknown;
  unknown.royals_safe = &say_unknown;
  CHECK(s.submission_status(unknown) == SubmitStatus::Unchecked);
  CheckRuling safe;
  safe.royals_safe = &say_yes;
  CHECK(s.submission_status(safe) == SubmitStatus::Safe);
  s.submit(safe);
  CHECK(s.to_move() == Colour::Black);
  // The hook is never consulted while the obligations are unmet.
  CHECK(s.submission_status(safe) == SubmitStatus::ObligationsUnmet);
}

TEST(state_moves_of_either_colour_for_the_check_ruling) {
  // append_moves_of: the hook for check detection - a colour's moves from a
  // playable board on which that colour is to move, whatever the turn.
  GameState s(standard_start());
  const RuleOptions rules;
  MoveList black;
  append_moves_of(s.multiverse(), rules, Colour::Black, 0, black);
  CHECK(black.empty());  // (0T1) has White to move
  play(s, "(0T1)Pe2(0T1)e4");
  // Mid-action, White still to move: Black's moves from the new board exist.
  REQUIRE(s.to_move() == Colour::White);
  append_moves_of(s.multiverse(), rules, Colour::Black, 0, black);
  CHECK_EQ(black.size(), std::size_t{20});
  for (const Move& m : black) CHECK(m.piece.colour() == Colour::Black);
  // A hypothetical multiverse - here the T2 board passed, advanced unchanged -
  // is accepted as it is.
  Multiverse phantom = s.multiverse();
  phantom.append_board(0, phantom.board_ref(0, 3));
  MoveList white;
  append_moves_of(phantom, rules, Colour::White, 0, white);
  CHECK(!white.empty());
  bool back_in_time = false;
  for (const Move& m : white) back_in_time = back_in_time || m.branching();
  CHECK(back_in_time);  // the knights can jump to T1, as after 1. e4 e5
}

TEST(state_outcome_is_unknown_without_the_check_ruling) {
  const GameState s(standard_start());
  CHECK(outcome(s, {}) == Outcome::Unknown);
  CheckRuling r;
  r.has_legal_action = &say_yes;
  CHECK(outcome(s, r) == Outcome::Ongoing);
  r.has_legal_action = &say_no;
  CHECK(outcome(s, r) == Outcome::Unknown);  // in_check missing
  r.in_check = &say_yes;
  CHECK(outcome(s, r) == Outcome::Checkmate);
  r.in_check = &say_no;
  CHECK(outcome(s, r) == Outcome::Stalemate);
  r.in_check = &say_unknown;
  CHECK(outcome(s, r) == Outcome::Unknown);
  r.has_legal_action = &say_unknown;
  CHECK(outcome(s, r) == Outcome::Unknown);
}
