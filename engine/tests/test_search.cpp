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
#include "t5d/fen.hpp"
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
//   3 x 3 pairs of moves within the boards (either order: one state each),
//   3 + 3 non-branching jumps, each the whole action,
//   2 x 3 x 3 one king's move, then the other's branching jump onto it,
// 33 distinct successors, all safe (the black kings stand three ranks away,
// on every board - the copies included).
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

TEST(search_counts_distinct_successor_states) {
  for (const Detector d : kDetectors) {
    const GameState s = two_kings();
    CHECK_EQ(count_of(s, d), std::size_t{33});
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
  // One action per distinct successor, sorted by its canonical text; each
  // replays (in its own order) to the state listed.
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
