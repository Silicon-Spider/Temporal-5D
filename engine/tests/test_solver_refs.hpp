// The references of the solver's cross-check against df-pn (register H41;
// spec §11.1, §15.4; tools/solver_proofs.hpp holds the derivation and the
// comparison, shared with t5d-solver-check and t5d-proof-records): the
// constructed positions, each under a ruleset, and the search's D2 arms the
// check runs under.
//
// Every position is written here from the rulebook (the derivations are in
// test_scanner.cpp and test_mcts.cpp, where the positions were first
// written) - never taken from a game corpus or another implementation's test
// data.

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

struct Named {
  const char* name;
  GameState state;
};

/// The seven constructed positions under `rules`.
inline std::vector<Named> constructed(const RuleOptions& rules) {
  return {{"mate_in_one", mate_in_one(rules)},
          {"cross_board_mate", cross_board_mate(rules)},
          {"stale_checkmate_in_one", stale_checkmate_in_one(rules)},
          {"mate_in_two", mate_in_two(rules)},
          {"history_royal_mate", history_royal_mate(rules)},
          {"forced_draw", forced_draw(rules)},
          {"forced_loss", forced_loss(rules)}};
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
