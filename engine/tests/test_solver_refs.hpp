// The references of the solver's cross-check against df-pn (register H41;
// spec §11.1, §15.4; tools/solver_proofs.hpp holds the derivation and the
// comparison, shared with t5d-solver-check and t5d-proof-records): the
// constructed positions, each under a ruleset, and the search's D2 arms the
// check runs under.
//
// Every position is written here from the rulebook (the derivations are in
// test_scanner.cpp and test_mcts.cpp, where the positions were first
// written), or found by the engine's own search and held by df-pn (below) -
// never taken from a game corpus or another implementation's test data.

#ifndef T5D_TEST_SOLVER_REFS_HPP
#define T5D_TEST_SOLVER_REFS_HPP

#include <string>
#include <vector>

#include "solver_proofs.hpp"
#include "t5d/mcts.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "test_positions.hpp"

namespace t5d_test::solver {

using t5d::BoardSize;
using t5d::GameState;
using t5d::RuleOptions;

inline constexpr BoardSize k5x5{5, 5};
inline constexpr BoardSize k4x4{4, 4};

// The prover's constructed mates (test_prover.cpp, from test_scanner.cpp).
inline GameState mate_in_one(const RuleOptions& r) { return game({"[k4/4Q/1K3/5/5:0:1:w]"}, k5x5, r); }
inline GameState cross_board_mate(const RuleOptions& r) {
  return game({"[5/Q4/2p2/5/4K:-0:1:w]", "[k4/5/1K3/5/5:+0:1:w]"}, k5x5, r);
}
inline GameState stale_checkmate_in_one(const RuleOptions& r) {
  return game({"[k3K/5/5/5/4R:0:1:w]", "[5/5/5/4P/RR3:+1:1:w]"}, k5x5, r);
}
inline GameState mate_in_two(const RuleOptions& r) { return game({"[5/k4/4R/3R1/4K:0:1:w]"}, k5x5, r); }
inline GameState history_royal_mate(const RuleOptions& r) {
  return game({"[kp1p1/pp2N/5/5/4K:0:1:w]", "[kp1p1/pp2N/5/5/3K1:0:1:b]", "[kp3/pp1pN/5/5/3K1:0:2:w]",
               "[kp3/pp1pN/5/5/4K:0:2:b]", "[kp3/pp2N/3p1/5/4K:0:3:w]"},
              k5x5, r);
}
// The search's solver positions (test_mcts.cpp): every action stalemates;
// every action allows a mate in one.
inline GameState forced_draw(const RuleOptions& r) { return game({"[3k/4/B1qK/4:0:1:w]"}, k4x4, r); }
inline GameState forced_loss(const RuleOptions& r) { return game({"[4/K2b/3n/1k2:0:1:w]"}, k4x4, r); }

// MT-01's review (2026-11-01, M4): positions whose proofs exercise the
// derivation - losses whose defences lose at different speeds (the root's
// Loss(k) the longest defence's, its children at more than one distance) and
// draws other than the forced draw. Found by a seeded random search over 4x4
// set-ups with the search's solver (2026-11-01, issue 409), each claim the
// same under every D2 arm and both Q4 readings, and each held by df-pn in the
// literal class with its proofs certified by E4 - the test re-checks all of it.
inline GameState slow_loss_in_4(const RuleOptions& r) { return game({"[qk2/2K1/p3/4:0:1:w]"}, k4x4, r); }
inline GameState minor_loss_in_4(const RuleOptions& r) { return game({"[4/2N1/2k1/B2K:0:1:b]"}, k4x4, r); }
inline GameState bishop_loss_in_6(const RuleOptions& r) { return game({"[B3/4/k1K1/4:0:1:b]"}, k4x4, r); }
inline GameState knight_loss_in_6(const RuleOptions& r) { return game({"[4/N3/4/1k1K:0:1:b]"}, k4x4, r); }
inline GameState queen_rook_draw(const RuleOptions& r) { return game({"[1q2/2kK/r1B1/4:0:1:b]"}, k4x4, r); }
inline GameState rook_draw(const RuleOptions& r) { return game({"[2k1/1KR1/1r2/4:0:1:b]"}, k4x4, r); }
inline GameState bishop_rook_draw(const RuleOptions& r) { return game({"[4/1B2/KR2/qk2:0:1:b]"}, k4x4, r); }

struct Named {
  const char* name;
  GameState state;
  /// The root's claim the solver must derive: "win 1", "loss 4", "draw" -
  /// or empty: not asserted.
  const char* claim = "";
};

/// The fourteen constructed positions under `rules`.
inline std::vector<Named> constructed(const RuleOptions& rules) {
  return {{"mate_in_one", mate_in_one(rules), "win 1"},
          {"cross_board_mate", cross_board_mate(rules), "win 1"},
          {"stale_checkmate_in_one", stale_checkmate_in_one(rules), "win 1"},
          {"mate_in_two", mate_in_two(rules), "win 3"},
          {"history_royal_mate", history_royal_mate(rules), "win 1"},
          {"forced_draw", forced_draw(rules), "draw"},
          {"forced_loss", forced_loss(rules), "loss 2"},
          {"slow_loss_in_4", slow_loss_in_4(rules), "loss 4"},
          {"minor_loss_in_4", minor_loss_in_4(rules), "loss 4"},
          {"bishop_loss_in_6", bishop_loss_in_6(rules), "loss 6"},
          {"knight_loss_in_6", knight_loss_in_6(rules), "loss 6"},
          {"queen_rook_draw", queen_rook_draw(rules), "draw"},
          {"rook_draw", rook_draw(rules), "draw"},
          {"bishop_rook_draw", bishop_rook_draw(rules), "draw"}};
}

/// The search's D2 arms the solver is checked under (SearchParams::split):
/// the ply split (as a tree and as MCGS's graph), the tree form (a tree only:
/// refused with graph search) and the flat form.
struct Arm {
  const char* name;
  t5d::mcts::TurnSplit split;
  bool graph;
};
inline std::vector<Arm> arms(bool with_graphs) {
  using t5d::mcts::TurnSplit;
  std::vector<Arm> out{{"ply split", TurnSplit::BoardMove, false},
                       {"tree form", TurnSplit::Tree, false},
                       {"flat form", TurnSplit::Flat, false}};
  if (with_graphs) {
    out.push_back({"ply split, graph", TurnSplit::BoardMove, true});
    out.push_back({"flat form, graph", TurnSplit::Flat, true});
  }
  return out;
}

}  // namespace t5d_test::solver

#endif  // T5D_TEST_SOLVER_REFS_HPP
