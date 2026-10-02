// Legal actions, their count, and mate detection (rulebook C5, C8, C9, C13,
// ruling R1; register H37). Every position is written here from the rulebook -
// never taken from a game corpus or another implementation's test data - and
// its legal actions are derived by hand in the comments. Each test runs every
// detector in kDetectors; they must agree.
//
// 5x5 boards unless said otherwise; 5DFEN rows run from rank 5 down to rank 1.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/expand.hpp"
#include "t5d/fen.hpp"
#include "t5d/legal.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"

using namespace t5d;
using t5d_test::game;
using t5d_test::play;

namespace {

constexpr BoardSize k5x5{5, 5};

const std::vector<Detector> kDetectors = {Detector::Naive, Detector::Hypercuboid};

Outcome verdict_of(const GameState& s, Detector d) { return detect_mate(s, d).verdict; }

std::size_t count_of(const GameState& s, Detector d, std::size_t cap = 100000) {
  const ActionCount c = count_legal_actions(s, cap, d);
  CHECK(c.known());
  return c.count;
}

// The position an action reaches, replayed through the state machine.
GameState replay(const GameState& start, const std::vector<Move>& moves) {
  GameState s = start;
  for (const Move& m : moves) {
    CHECK(is_pseudo_legal(s, m));
    s.apply(m);
  }
  CHECK(s.submission_status() == SubmitStatus::Safe);
  s.submit();
  return s;
}

// Two kings a side on two timelines, -0 and +0, White to move: kings a1,
// black kings a5. White must play both boards (T13). Each king has three
// moves on its board (a2, b1, b2) and three jumps onto the other board's
// a2, b1, b2 - a playable board of White's, so a jump that plays both boards
// (J4). And once one board is played, its former head is history: the other
// king's jump onto it BRANCHES (J6), creating +1. So:
//   3 x 3 pairs of moves within the boards (either order: one action),
//   3 + 3 non-branching jumps, each the whole action,
//   2 x 3 x 3 one king's move, then the other's branching jump onto it,
// 33 legal actions, all safe (the black kings stand three ranks away, on
// every board - the copies included), and as many distinct successors: no
// action creates two timelines, and no two reach the same position.
GameState two_kings() { return game({"[k4/5/5/5/K4:-0:1:w]", "[k4/5/5/5/K4:+0:1:w]"}, k5x5); }

// C8, stale-checkmate: Black must play 0 and +1 (both at T1, active). On 0 its
// king a5 is checked by a rook a1; Kb4 or Kb5 would answer that alone - but +1
// (white rooks a1 and b1 on Black's board) holds no black piece, and the only
// way to play it, the king's jump onto it, lands on a4, a5, b4 or b5 under
// those rooks. No board of Black's lies in the past: no rewind (C9). In check:
// checkmate.
GameState stale_checkmate() {
  return game({"[k4/5/5/5/R4:0:1:b]", "[5/5/5/5/RR3:+1:1:b]"}, k5x5);
}

// C9: the same trap one turn later, with a board of Black's in the past:
// (0T1), where the king stood on c5. The king can jump back to (0T1)a4, a5,
// b4 or b5 - a branch; Black's first timeline is active (White has created
// one, T10), its first board (T2, white) is behind the present (T2, black),
// so the present rewinds (T8) and +1 need not be played. Safe: nothing of
// White's stands on the copy of (0T1). Four legal actions.
GameState rewind_escape() {
  return game({"[2k2/5/5/5/5:0:1:b]", "[k4/5/5/5/5:0:2:w]", "[k4/5/5/5/R4:0:2:b]",
               "[5/5/5/5/RR3:+1:2:b]"},
              k5x5);
}

// The same, but Black has already created two timelines (-1, -2; their White
// boards ahead, empty) to White's one: Black's third would be inactive (T10),
// and an inactive branch does not rewind (T8, C9). Checkmate.
GameState rewind_blocked() {
  return game({"[5/5/5/5/5:-2:3:w]", "[5/5/5/5/5:-1:3:w]", "[2k2/5/5/5/5:0:1:b]",
               "[k4/5/5/5/5:0:2:w]", "[k4/5/5/5/R4:0:2:b]", "[5/5/5/5/RR3:+1:2:b]"},
              k5x5);
}

}  // namespace

// ---- counting -----------------------------------------------------------------------

TEST(search_counts_the_standard_start) {
  // Ordinary chess's twenty first moves, one board, nothing to travel to: 20
  // legal actions - and after 1. e3, Black's 20 (the gate contract's example).
  for (const Detector d : kDetectors) {
    GameState s(standard_start());
    CHECK_EQ(count_of(s, d), std::size_t{20});
    play(s, "(0T1)Pe2(0T1)e3");
    s.submit();
    CHECK_EQ(count_of(s, d), std::size_t{20});
  }
}

TEST(search_counts_the_legal_actions) {
  for (const Detector d : kDetectors) {
    const GameState s = two_kings();
    CHECK_EQ(count_of(s, d), std::size_t{33});
    const ActionCount successors = count_legal_successors(s, 100000, d);
    CHECK(successors.stats.exhaustive);
    CHECK_EQ(successors.count, std::size_t{33});
    // The cap: count == cap means "at least".
    const ActionCount capped = count_legal_actions(s, 7, d);
    CHECK_EQ(capped.count, std::size_t{7});
    CHECK(capped.reached_cap);
    CHECK(!capped.stats.exhaustive);
    CHECK(capped.known());
    const ActionCount zero = count_legal_actions(s, 0, d);
    CHECK_EQ(zero.count, std::size_t{0});
    CHECK(zero.known());
  }
}

TEST(search_counts_on_one_board_drop_moves_into_check) {
  // Focused - Just Kings (3x3): the white king a1 may go to a2 or b1, not b2
  // beside the black king c3 (C4).
  for (const Detector d : kDetectors) {
    CHECK_EQ(count_of(game({"[2k/3/K2:0:1:w]"}, BoardSize{3, 3}), d), std::size_t{2});
  }
}

TEST(search_lists_the_legal_actions_canonically) {
  // One entry per legal action, sorted by its canonical text; each replays
  // (in its own order) to the state listed.
  for (const Detector d : kDetectors) {
    const GameState s = two_kings();
    const ActionList list = list_legal_actions(s, 1000, d);
    REQUIRE_EQ(list.actions.size(), std::size_t{33});
    CHECK(list.stats.exhaustive);
    std::vector<std::string> texts;
    int branching = 0;
    for (const LegalAction& a : list.actions) {
      texts.push_back(action_text(a.moves, true));
      CHECK(same_position(replay(s, a.moves), a.result));
      if (a.moves.back().branching()) ++branching;
    }
    CHECK_EQ(branching, 18);
    CHECK(std::is_sorted(texts.begin(), texts.end()));
    CHECK_EQ(std::set<std::string>(texts.begin(), texts.end()).size(), texts.size());
    const auto listed = [&](const char* text) {
      return std::find(texts.begin(), texts.end(), text) != texts.end();
    };
    CHECK(listed("(-0T1)Ka1(-0T1)a2 (+0T1)Ka1(+0T1)b2"));  // both within their boards
    CHECK(listed("(+0T1)Ka1(-0T1)b1"));                    // a jump that plays both
    CHECK(listed("(-0T1)Ka1(-0T1)a2 (+0T1)Ka1(-0T1)b1"));  // J6: then a branch onto -0
  }
}

TEST(search_two_creation_orders_are_two_actions_j6) {
  // The harness's definition (t5d/search.hpp): an action is the set of its
  // moves with its timeline-creating moves in the order they create their
  // timelines - two orders of two branching moves are two actions, even where
  // they reach the same state. 1x1 boards: a piece moves only through time
  // and across timelines. White has created +1, Black nothing, so White's
  // next creations are inactive (T10) and rewind nothing (T8). On -0 and +0,
  // at T2 (White's, the present: both mandatory, T6), a rook - each with one
  // move: back to its own timeline's T1, empty, a branch (L5). (Across the
  // timelines it meets its own rooks: on the other board, and on +1's T2.)
  // Both must be made, in either order: the new timelines +2 and +3 each
  // begin with a copy of an empty board with a rook arrived - identical - so
  // the two orders reach one position (their origins aside): two legal
  // actions, one successor.
  const GameState s = game({"[1:-0:1:w]", "[1:-0:1:b]", "[R:-0:2:w]", "[1:+0:1:w]", "[1:+0:1:b]",
                            "[R:+0:2:w]", "[1:+1:1:b]", "[R:+1:2:w]", "[1:+1:2:b]"},
                           BoardSize{1, 1});
  for (const Detector d : kDetectors) {
    CHECK_EQ(count_of(s, d), std::size_t{2});
    const ActionCount successors = count_legal_successors(s, 100, d);
    CHECK(successors.stats.exhaustive);
    CHECK_EQ(successors.count, std::size_t{1});
    const ActionList list = list_legal_actions(s, 100, d);
    REQUIRE_EQ(list.actions.size(), std::size_t{2});
    CHECK_EQ(action_text(list.actions[0].moves, true),
             std::string("(+0T2)Ra1(+0T1)a1 (-0T2)Ra1(-0T1)a1"));
    CHECK_EQ(action_text(list.actions[1].moves, true),
             std::string("(-0T2)Ra1(-0T1)a1 (+0T2)Ra1(+0T1)a1"));
    CHECK(same_position(list.actions[0].result, list.actions[1].result));
    CHECK(same_position(replay(s, list.actions[0].moves), replay(s, list.actions[1].moves)));
  }
}

TEST(search_positions_differing_only_in_origin_are_one) {
  // Timeline::origin (the board a timeline branched from) is read by no rule
  // and kept by neither key nor fingerprint: the count treats positions that
  // differ only in it as one (same_position); GameState's == tells them apart.
  const BoardRef empty = BoardBuilder(k5x5).publish();
  BoardBuilder kb(k5x5);
  kb.put(0, Piece(PieceKind::King, Colour::White));
  const BoardRef king = kb.publish();
  const auto build = [&](int origin_timeline) {
    Multiverse mv(k5x5, false);
    for (const int p : {-1, 0}) {
      NewTimeline nt;
      nt.position = p;
      nt.first_half_turn = 2;
      nt.boards = {king, empty};
      if (p == -1) {
        nt.creator = Colour::Black;
        nt.creation_order = 1;
      }
      mv.add_timeline(std::move(nt));
    }
    NewTimeline branch;
    branch.position = 1;
    branch.first_half_turn = 3;
    branch.boards = {empty};
    branch.creator = Colour::White;
    branch.creation_order = 1;
    branch.origin = BoardCoord{origin_timeline, 2};
    mv.add_timeline(std::move(branch));
    mv.set_to_move(Colour::Black);
    return GameState(std::move(mv));
  };
  const GameState a = build(-1);
  const GameState b = build(0);
  CHECK(!(a == b));
  CHECK(same_position(a, b));
  CHECK_EQ(a.zobrist(), b.zobrist());
}

TEST(search_positions_differing_only_in_history_dependent_activity_are_two) {
  // Where activity is not a function of the boards (Q1's consecutive reading,
  // Q3's activation by play, Q6's deferral), it is part of the position:
  // same_position (and the key) tell apart two states with the same boards
  // that differ in it. 2x1 boards (a1, b1); White to move at T2 on 0 (queen
  // a1) and Black's -1 (rook b1), both mandatory (T6); Black's -2, its second
  // creation, is inactive (White has created none, T10), its board at T2
  // White's and empty. Each played state is compared with the state read
  // fresh from the same boards (the count rule's activity, no last creator,
  // no deferral).
  const std::vector<std::string> boards = {"[2:0:1:w]",   "[2:0:1:b]",  "[Q1:0:2:w]",
                                           "[2:-1:1:w]",  "[2:-1:1:b]", "[1R:-1:2:w]",
                                           "[2:-2:1:w]",  "[2:-2:1:b]", "[2:-2:2:w]"};
  // Action A: the queen branches back to (0T1)a1 - White's first creation,
  // active, before the present: it rewinds it (T8), and the action ends; it
  // reactivates -2 (T11), whose board (T2, White's) is then behind the new
  // present (T1, Black's). Action B: the rook a1, then the queen across to
  // -2's playable board (a jump that plays both, J4) - no creation.
  const auto after = [&](const RuleOptions& rules, bool branch) {
    GameState s = game(boards, BoardSize{2, 1}, rules);
    if (branch) {
      play(s, "(0T2)Qa1(0T1)a1");
    } else {
      play(s, "(-1T2)Rb1(-1T2)a1");
      play(s, "(0T2)Qa1(-2T2)a1");
    }
    s.submit();
    return s;
  };
  const auto fresh = [](const GameState& s) { return GameState(s.multiverse(), s.rules()); };
  const auto same = [&](const GameState& s) {
    const GameState f = fresh(s);
    CHECK_EQ(same_position(s, f), s.zobrist() == f.zobrist());
    return same_position(s, f);
  };
  // The default rules: activity follows from the boards.
  CHECK(same(after({}, true)));
  CHECK(same(after({}, false)));
  // Q1, consecutive: who created last is part of the position.
  RuleOptions consecutive;
  consecutive.activation = ActivationRule::Consecutive;
  const GameState c = after(consecutive, true);
  CHECK(c.last_creator() == Colour::White);
  CHECK(!same(c));
  CHECK(same(after(consecutive, false)));  // no creation: no last creator either way
  // Q6, deferred: the reactivated -2 (White's board) is left out of the present
  // until White's next turn - a deferral the fresh state does not have.
  RuleOptions deferred;
  deferred.reactivation = Reactivation::Deferred;
  const GameState d = after(deferred, true);
  REQUIRE_EQ(d.deferred_timelines().size(), std::size_t{1});
  CHECK_EQ(d.deferred_timelines()[0].timeline, -2);
  CHECK(!same(d));
  // At Black's turn start the deferred board is not Black's: the check status
  // reads it as no board of the mover's (check.cpp's passed_timelines).
  CHECK(!in_check(d));
  // Q3, activation by play: the queen's arrival on -2 activated it; read
  // fresh, -2 is inactive (White has still created none).
  RuleOptions by_play;
  by_play.play_activates = true;
  const GameState p = after(by_play, false);
  CHECK(p.inactive_timelines().empty());
  const GameState read = fresh(p);
  CHECK_EQ(read.inactive_timelines(), (std::vector<int>{-2}));
  CHECK(!same(p));
  CHECK(same(after(by_play, true)));
}

// ---- mate detection -------------------------------------------------------------------

TEST(search_checkmate_and_stalemate_on_one_board) {
  // (0T1) is the only board: nothing to travel to, so ordinary chess. Black
  // king a5, white queen b4 guarded by the king b3: checkmate. With the queen
  // on c4 instead (a4, b4, b5 covered, a5 not): stalemate.
  for (const Detector d : kDetectors) {
    const MateResult mate = detect_mate(game({"[k4/1Q3/1K3/5/5:0:1:b]"}, k5x5), d);
    CHECK(mate.verdict == Outcome::Checkmate);
    CHECK(mate.in_check);
    CHECK(mate.stats.exhaustive);
    const MateResult stale = detect_mate(game({"[k4/2Q2/5/5/4K:0:1:b]"}, k5x5), d);
    CHECK(stale.verdict == Outcome::Stalemate);
    CHECK(!stale.in_check);
    CHECK(stale.stats.exhaustive);
  }
}

TEST(search_stalemate_by_a_threat_across_timelines_c6) {
  // Black's king a5 on (0T1), its only board and only piece. White's rooks
  // stand on +1's board (T2, white: White's, ahead of the present) at a4, b4
  // and b5: each covers the same square of timeline 0 one step down the
  // timelines. Every king move (a4, b4, b5) lands under one of them; passing
  // (0T1) leaves a5 untouched (no rook stands on +1's a5). No legal action,
  // not in check: stalemate - though only a cross-board threat exists.
  const GameState s = game({"[k4/5/5/5/5:0:1:b]", "[1R3/RR3/5/5/5:+1:2:w]"}, k5x5);
  CHECK(!in_check(s));
  for (const Detector d : kDetectors) {
    CHECK(verdict_of(s, d) == Outcome::Stalemate);
    CHECK_EQ(count_of(s, d), std::size_t{0});
  }
}

TEST(search_stale_checkmate_c8) {
  const GameState s = stale_checkmate();
  CHECK(in_check(s));
  for (const Detector d : kDetectors) {
    CHECK(verdict_of(s, d) == Outcome::Checkmate);
  }
  // Each half of the trap alone is answerable: without +1, Kb4 escapes.
  for (const Detector d : kDetectors) {
    CHECK(verdict_of(game({"[k4/5/5/5/R4:0:1:b]"}, k5x5), d) == Outcome::Ongoing);
  }
}

TEST(search_rewinding_the_present_escapes_c9) {
  const GameState s = rewind_escape();
  CHECK(in_check(s));
  const GameState t = rewind_blocked();
  CHECK(in_check(t));
  for (const Detector d : kDetectors) {
    const MateResult r = detect_mate(s, d);
    REQUIRE(r.verdict == Outcome::Ongoing);
    REQUIRE_EQ(r.witness.size(), std::size_t{1});
    CHECK(r.witness.front().branching());
    CHECK_EQ(count_of(s, d), std::size_t{4});
    CHECK(verdict_of(t, d) == Outcome::Checkmate);
  }
}

TEST(search_a_king_lost_on_the_opponents_board_ahead_is_mated_r1) {
  // R1: White's board ahead of the present, (+1T2) with White to move, holds a
  // rook that takes the black king there. Black can touch neither: every
  // action leaves it capturable. In check, no legal action: checkmate.
  const GameState s = game({"[k4/5/5/5/5:0:1:b]", "[4k/5/5/5/4R:+1:2:w]"}, k5x5);
  for (const Detector d : kDetectors) {
    CHECK(verdict_of(s, d) == Outcome::Checkmate);
  }
}

TEST(search_no_royal_means_stalemate_c13) {
  // Black has a pawn on a5 blocked by a white rook on a4, nothing else - and
  // no royal piece: never in check (C13), no legal action - stalemate.
  const GameState s = game({"[p4/R4/5/5/K4:0:1:b]"}, k5x5);
  for (const Detector d : kDetectors) {
    const MateResult r = detect_mate(s, d);
    CHECK(r.verdict == Outcome::Stalemate);
    CHECK(!r.in_check);
  }
}

TEST(search_reversed_royalty_royal_queens) {
  // Reversed Royalty's pieces (4.11, 4.12), on one board: Black's royal queen
  // (Y) a5, its own pawns on b5 and b4 shutting the fifth rank and the
  // diagonal. A white rook a1 checks it down the a-file, guarded by a rook b1.
  // The royal queen moves like a queen: a4, a3, a2 stay on the rook's file,
  // and Yxa1 is retaken by the rook b1; the pawns' moves (b4-b3) leave the
  // check. Checkmate. With a common king (C) there instead, Black has no royal
  // piece (4.12, C13): no check, and moves enough.
  RuleOptions rr;
  rr.family = RuleFamily::ReversedRoyalty;
  for (const Detector d : kDetectors) {
    const GameState s = game({"[yp3/1p3/5/5/RR3:0:1:b]"}, k5x5, rr);
    CHECK(in_check(s));
    CHECK(verdict_of(s, d) == Outcome::Checkmate);
    const GameState c = game({"[cp3/1p3/5/5/RR3:0:1:b]"}, k5x5, rr);
    CHECK(!in_check(c));
    CHECK(verdict_of(c, d) == Outcome::Ongoing);
  }
}

TEST(search_outcome_through_the_engines_check_ruling) {
  const CheckRuling r = standard_check_ruling();
  CHECK(outcome(GameState(standard_start()), r) == Outcome::Ongoing);
  CHECK(outcome(stale_checkmate(), r) == Outcome::Checkmate);
  CHECK(outcome(game({"[k4/2Q2/5/5/4K:0:1:b]"}, k5x5), r) == Outcome::Stalemate);
}

// ---- the hypercuboid's cost: regression bounds ------------------------------------------

namespace {

// 1x1 boards on timelines -n..+n (each side has created n: all active, T10),
// each with empty boards from T1 to T3 and a White queen at T4 (White's turn;
// all mandatory, T6). A queen on 1x1 meets its own queens across the
// timelines at T4; its moves are the branches into the past: back along its
// own timeline to T3, T2 or T1, and diagonally (one timeline per turn back) up
// to three turns, as far as the timelines go (L5) - 3 per queen back along its
// timeline, and diagonally 3 + 3 per queen less 1 + 2 + 3 at each edge. The
// first branch is active (White's (n+1)-th creation, Black has n) and lands
// before the present, which it rewinds (T8) onto Black's colour: the
// obligations are met, and the other boards are unplayed and optional now.
GameState every_branch_rewinds(int n, const RuleOptions& rules = {}) {
  std::vector<std::string> boards;
  for (int p = -n; p <= n; ++p) {
    const std::string label = p == 0 ? "0" : (p > 0 ? "+" : "") + std::to_string(p);
    for (int t = 1; t <= 3; ++t) {
      boards.push_back("[1:" + label + ":" + std::to_string(t) + ":w]");
      boards.push_back("[1:" + label + ":" + std::to_string(t) + ":b]");
    }
    boards.push_back("[Q:" + label + ":4:w]");
  }
  return game(boards, BoardSize{1, 1}, rules);
}

}  // namespace

TEST(search_the_hypercuboid_stays_small_when_every_branch_rewinds) {
  // every_branch_rewinds under Q4's Guide-literal reading (the hypercuboid's
  // Ordered regime): the action ends with its first branch. So the legal
  // actions are the single branches: 3(2n+1) + 2(3(2n+1) - 6) = 18n - 3 for
  // n >= 2. A successor is fixed by the queen that moved and how far back
  // (every destination is an empty board): 3(2n+1). The product space has
  // (19)^(2n+1) points; the hypercuboid once spent 34 million nodes on n = 3.
  // (Under the default, R8, the other queens may go on branching:
  // search_moves_after_the_shift_multiply_the_actions_r8, below.)
  RuleOptions guide;
  guide.moves_after_shift = false;
  for (int n = 2; n <= 6; ++n) {
    const GameState s = every_branch_rewinds(n, guide);
    for (const Detector d : kDetectors) {
      const ActionCount actions = count_legal_actions(s, 100000, d);
      CHECK(actions.stats.exhaustive);
      CHECK_EQ(actions.count, static_cast<std::size_t>(18 * n - 3));
      // A move and a test per action - and, for the hypercuboid, a node per box
      // it takes up and per step of its abstract check: 16 per action at
      // n = 6, where the search once took 34 million nodes in all.
      CHECK(actions.stats.nodes <= 20 * actions.count);
      const ActionCount successors = count_legal_successors(s, 100000, d);
      CHECK_EQ(successors.count, static_cast<std::size_t>(3 * (2 * n + 1)));
    }
  }
}

TEST(search_moves_after_the_shift_multiply_the_actions_r8) {
  // every_branch_rewinds under the default rules (R8: moves until
  // submission). Once the first branch has shifted the present, every other
  // queen may still branch - and each later creation is inactive (White's
  // (n+2)-th and on, Black has n: T10), so the present stays shifted and the
  // action may be submitted after any of them (no royal: C13). An action is
  // then a non-empty set S of queens, each with one of its m_p moves, in a
  // creation order: sum over S of |S|! * prod m_p. n = 1: m = 5, 5, 5 (3 back
  // along the timeline, 2 diagonally) - 3 * 5 + 3 * 2! * 25 + 3! * 125 =
  // 15 + 150 + 750 = 915 (the Guide-literal reading: 15). A successor is fixed
  // by the queens that moved and the depths of the new timelines in creation
  // order (every new board is an empty board's copy with the queen, J5): 3^|S|
  // per set, 4^(2n+1) - 1 in all - 63 at n = 1, 1023 at n = 2.
  const GameState s1 = every_branch_rewinds(1);
  for (const Detector d : kDetectors) {
    const ActionCount actions = count_legal_actions(s1, 100000, d);
    CHECK(actions.stats.exhaustive);
    CHECK_EQ(actions.count, std::size_t{915});
    const ActionCount successors = count_legal_successors(s1, 100000, d);
    CHECK(successors.stats.exhaustive);
    CHECK_EQ(successors.count, std::size_t{63});
  }
  const GameState s2 = every_branch_rewinds(2);
  for (const Detector d : kDetectors) {
    const ActionCount successors = count_legal_successors(s2, 100000, d);
    CHECK(successors.stats.exhaustive);
    CHECK_EQ(successors.count, std::size_t{1023});
  }
  // Along one action: after the first branch and after each further one the
  // action may end (obligations met, every royal - none - safe) or go on,
  // until every board has been played.
  GameState t = s1;
  CHECK(!t.obligations_met());
  for (int i = 0; i < 3; ++i) {
    t.apply(t5d_test::moves_of(t).front());
    CHECK(t.obligations_met());
    CHECK(t.may_move());
    CHECK(t.submission_status() == SubmitStatus::Safe);
  }
  CHECK(t5d_test::moves_of(t).empty());  // every board played: only submitting is left
}

TEST(search_the_hypercuboid_finds_actions_past_reactivations) {
  // The shape of a position that once stalled the hypercuboid: the opponent's
  // inactive timelines with boards of the mover's behind the present, which
  // the mover's creations reactivate (T11) and the present then rewinds to
  // (T12) - so most points of the product space, those with branches, have
  // no order that meets the obligations. 2x1 boards (a1, b1), White queens
  // on a1. Timelines 0, White's +1 and Black's -1 to -3 head at T4 (White's),
  // -4 at T3 and -5 at T2. Black has created five, White one: Black's -3, -4
  // and -5 are inactive (T10), and White's first three creations reactivate
  // them in that order - -4 and -5 behind the present. The legal
  // actions are many (each mandatory queen may step to b1 or jump across):
  // both detectors reach a cap of 1000 - the hypercuboid within a bound, by
  // its abstract check of which skeletons can be realised.
  std::vector<std::string> boards;
  const auto add = [&](const std::string& label, int first, int last) {
    for (int j = first; j <= last; ++j) {
      boards.push_back("[" + std::string(j == last ? "Q1" : "2") + ":" + label + ":" +
                       std::to_string(j / 2 + 1) + ":" + (j % 2 == 0 ? "w" : "b") + "]");
    }
  };
  add("0", 0, 6);
  add("+1", 1, 6);
  add("-1", 1, 6);
  add("-2", 1, 6);
  add("-3", 1, 6);
  add("-4", 1, 4);
  add("-5", 1, 2);
  const GameState s = game(boards, BoardSize{2, 1});
  REQUIRE_EQ(s.inactive_timelines(), (std::vector<int>{-5, -4, -3}));
  for (const Detector d : kDetectors) {
    const ActionCount c = count_legal_actions(s, 1000, d);
    CHECK(c.reached_cap);
    CHECK_EQ(c.count, std::size_t{1000});
    if (d == Detector::Hypercuboid) CHECK(c.stats.nodes < 12000);  // 8,671: boxes and abstract steps counted
  }
}

// ---- H37: a budget can only ever yield "unknown" ----------------------------------------

TEST(search_a_budget_limited_search_answers_unknown_never_mate_h37) {
  // For positions whose exhaustive search needs K steps, every node budget
  // below K answers Unknown - never checkmate or stalemate - and K answers the
  // proof. Mates, a stalemate, and positions with a legal action (where a
  // small budget may still find one: "none" with its witness, never a mate).
  const std::vector<GameState> positions = {
      stale_checkmate(), rewind_blocked(), game({"[k4/1Q3/1K3/5/5:0:1:b]"}, k5x5),
      game({"[k4/5/5/5/5:0:1:b]", "[1R3/RR3/5/5/5:+1:2:w]"}, k5x5), rewind_escape(),
      two_kings()};
  for (const Detector d : kDetectors) {
    for (const GameState& s : positions) {
      const MateResult full = detect_mate(s, d);
      REQUIRE(full.verdict != Outcome::Unknown);
      const std::uint64_t needed = full.stats.nodes;
      for (std::uint64_t budget = 0; budget < needed; ++budget) {
        const MateResult r = detect_mate(s, d, SearchBudget::nodes(budget));
        CHECK(r.verdict == Outcome::Unknown || r.verdict == Outcome::Ongoing);
        if (full.verdict != Outcome::Ongoing) {
          CHECK(r.verdict == Outcome::Unknown);
          CHECK(r.stats.stopped_by_budget);
          CHECK(!r.stats.exhaustive);
          CHECK(r.stats.nodes <= budget);
        }
      }
      const MateResult enough = detect_mate(s, d, SearchBudget::nodes(needed));
      CHECK(enough.verdict == full.verdict);
      // Time and memory budgets stop it the same way.
      if (full.verdict == Outcome::Checkmate || full.verdict == Outcome::Stalemate) {
        SearchBudget no_time;
        no_time.max_time = std::chrono::nanoseconds(-1);
        CHECK(detect_mate(s, d, no_time).verdict == Outcome::Unknown);
        SearchBudget no_memory;
        no_memory.max_stored = 0;
        CHECK(detect_mate(s, d, no_memory).verdict == Outcome::Unknown);
      }
    }
  }
}

TEST(search_a_search_at_its_memory_ceiling_answers_unknown_h37) {
  // Every search has a memory ceiling (SearchBudget::max_bytes, by default
  // kDefaultSearchBytes, unlimited() included). Reaching it stops the search
  // like any budget: Unknown - never checkmate or stalemate - and
  // stopped_by_memory says why. A ceiling below one held position stops it
  // at once.
  CHECK_EQ(SearchBudget::unlimited().max_bytes, kDefaultSearchBytes);
  for (const Detector d : kDetectors) {
    for (const GameState& s : {stale_checkmate(), two_kings()}) {
      SearchBudget tiny;
      tiny.max_bytes = 1;
      const MateResult r = detect_mate(s, d, tiny);
      CHECK(r.verdict == Outcome::Unknown);
      CHECK(r.stats.stopped_by_budget);
      CHECK(r.stats.stopped_by_memory);
      CHECK(!r.stats.exhaustive);
      const ActionCount c = count_legal_actions(s, 1000, d, tiny);
      CHECK(!c.known());
      CHECK(c.stats.stopped_by_memory);
    }
    // The default ceiling is far above what these need.
    const MateResult mate = detect_mate(stale_checkmate(), d);
    CHECK(mate.verdict == Outcome::Checkmate);
    CHECK(!mate.stats.stopped_by_memory);
  }
}

TEST(search_legal_moves_are_those_that_begin_a_legal_action_r1) {
  // R1 judges legality per action ("moving into check during a turn is
  // allowed"), so a single move is legal if a legal action begins with it
  // (build-log spec issue 28; t5d/search.hpp's legal_moves). In rewind_escape
  // (C9), Black's legal actions are the king's four jumps back to (0T1), each
  // a whole action: they are its legal moves. Its other pseudo-legal moves -
  // the king's steps on board 0, the jumps onto +1's board - leave no way to
  // finish: +1 (or 0) is then left to play with nothing on it, and nothing
  // rewinds. In rewind_blocked, checkmate, nothing is legal.
  for (const Detector d : kDetectors) {
    MoveList all;
    generate_moves(rewind_escape(), all);
    CHECK(all.size() > 4);
    MoveList legal;
    legal_moves(rewind_escape(), legal, d);
    REQUIRE_EQ(legal.size(), std::size_t{4});
    for (const Move& m : legal) {
      CHECK(m.branching());
      CHECK_EQ(m.to_timeline, 0);
    }
    legal_moves(rewind_blocked(), legal, d);
    CHECK(legal.empty());
    // During an action: in two_kings after (-0T1)Ka1(-0T1)a2, every move the
    // +0 king has - three on its board, three jumps branching onto -0's T1
    // board, now history (J6) - finishes a legal action.
    GameState mid = two_kings();
    play(mid, "(-0T1)Ka1(-0T1)a2");
    generate_moves(mid, all);
    legal_moves(mid, legal, d);
    CHECK_EQ(all.size(), std::size_t{6});
    CHECK(legal == all);
  }
  // The expansion's switch (ExpandOptions::legal_moves) offers the same.
  ExpandOptions o;
  Expansion x;
  expand(rewind_escape(), o, x);
  const std::size_t pseudo = x.move_records().size();
  o.legal_moves = true;
  expand(rewind_escape(), o, x);
  CHECK_EQ(x.move_records().size(), std::size_t{4});
  CHECK(pseudo > 4);
  REQUIRE_EQ(x.move_legality().size(), std::size_t{4});
  for (const MoveLegality l : x.move_legality()) CHECK(l == MoveLegality::Legal);
  // Under a budget (t5d/legal.hpp): the moves proved illegal dropped, the
  // undecided kept and marked Unknown - never a legal move dropped, never an
  // illegal one called legal.
  MoveList legal;
  legal_moves(rewind_escape(), legal);
  for (const LegalityOptions& budget : {LegalityOptions::budgeted(0, 0), LegalityOptions::budgeted(1, 10)}) {
    o.legality = budget;
    expand(rewind_escape(), o, x);
    REQUIRE_EQ(x.move_legality().size(), x.moves().size());
    CHECK(x.moves().size() >= 4);
    CHECK(x.moves().size() <= pseudo);
    for (const Move& m : legal) CHECK(std::find(x.moves().begin(), x.moves().end(), m) != x.moves().end());
    for (std::size_t i = 0; i < x.moves().size(); ++i) {
      const bool is_legal = std::find(legal.begin(), legal.end(), x.moves()[i]) != legal.end();
      if (x.move_legality()[i] == MoveLegality::Legal) CHECK(is_legal);
    }
  }
  // Pseudo-legal: no statuses.
  o.legal_moves = false;
  expand(rewind_escape(), o, x);
  CHECK(x.move_legality().empty());
}

TEST(search_every_proven_verdict_is_exhaustive_h37) {
  // A property, over random play from small starts and random node budgets:
  // checkmate and stalemate come only with an exhaustive search; Unknown only
  // with a stopped one; "none" always with a witness that is a legal action.
  const std::vector<std::vector<std::string>> starts = {
      {"[2k/3/K2:0:1:w]"},
      {"[nbr*k*/p*p*p*p*/P*P*P*P*/K*R*BN:0:1:w]"},
      {"[k4/5/5/5/K4:-0:1:w]", "[k4/5/5/5/K4:+0:1:w]"},
  };
  t5d_test::Rng rng(97);
  int verdicts = 0;
  int proofs = 0;
  std::vector<std::vector<std::string>> rounds = starts;
  rounds.insert(rounds.end(), starts.begin(), starts.end());  // each start twice
  for (const auto& boards : rounds) {
    const BoardSize size = boards.front().find("2k/3/K2") != std::string::npos ? BoardSize{3, 3}
                           : boards.front().find("nbr") != std::string::npos ? BoardSize{4, 4}
                                                                             : k5x5;
    GameState s = game(boards, size);
    for (int turn = 0; turn < 16; ++turn) {
      for (const Detector d : kDetectors) {
        const std::uint64_t nodes =
            rng.chance(50) ? 20000 : static_cast<std::uint64_t>(rng.below(60));
        const MateResult r = detect_mate(s, d, SearchBudget::nodes(nodes));
        ++verdicts;
        if (r.verdict == Outcome::Checkmate || r.verdict == Outcome::Stalemate) {
          CHECK(r.stats.exhaustive);
          CHECK(!r.stats.stopped_by_budget);
          ++proofs;
        }
        CHECK_EQ(r.verdict == Outcome::Unknown, r.stats.stopped_by_budget);
        if (r.verdict == Outcome::Ongoing) {
          REQUIRE(!r.witness.empty());
          replay(s, r.witness);
        }
      }
      // On to a random legal successor (of the first few found).
      const ActionList legal =
          list_legal_actions(s, 24, kDetectors.front(), SearchBudget::nodes(20000));
      if (legal.actions.empty()) break;
      s = legal.actions[static_cast<std::size_t>(
                            rng.below(static_cast<int>(legal.actions.size())))]
              .result;
    }
  }
  CHECK(verdicts > 40);
  CHECK(proofs > 0);  // the games end in mates: proofs were made, and checked
}
