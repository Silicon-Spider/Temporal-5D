// The helpers of stage E9's tests (Guard, Hunt, the prover's additions,
// certificates): positions written from the rulebook, the literal class,
// actions replayed, mates by E4, every successor by the brute-force reference
// or E4's listing, and random cases - an action of the side to move at turn
// starts of random legal play.

#ifndef T5D_TEST_V3_SUPPORT_HPP
#define T5D_TEST_V3_SUPPORT_HPP

#include <cstdint>
#include <optional>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/prover.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "test_mate_refs.hpp"
#include "test_play.hpp"
#include "test_positions.hpp"
#include "test_reference.hpp"
#include "test_support.hpp"

namespace t5d_test::v3 {

using namespace t5d;
using t5d_test::game;
using t5d_test::playable;

inline constexpr BoardSize k5x5{5, 5};

inline GameState mate_in_one(const RuleOptions& r = {}) { return game({"[k4/4Q/1K3/5/5:0:1:w]"}, k5x5, r); }
inline GameState mate_in_two(const RuleOptions& r = {}) { return game({"[5/k4/4R/3R1/4K:0:1:w]"}, k5x5, r); }
inline GameState cross_board_mate(const RuleOptions& r = {}) {
  return game({"[5/Q4/2p2/5/4K:-0:1:w]", "[k4/5/1K3/5/5:+0:1:w]"}, k5x5, r);
}

inline RuleOptions guide_literal_q4() {
  RuleOptions r;
  r.moves_after_shift = false;
  return r;
}

inline ProverOptions literal() {
  ProverOptions o;
  o.seeds = SeedClass::All;
  o.completion = CompletionClass::Every;
  return o;
}

inline GameState after_action(const GameState& s, const std::vector<Move>& action) {
  GameState t = s;
  for (const Move& m : action) t.apply(m);
  t.submit(CheckRuling::none());
  return t;
}

// A line replayed from `s`: each action's moves, then its submission. False
// if a move is not pseudo-legal or an action cannot be submitted.
inline bool replay(GameState& s, const std::vector<std::vector<Move>>& line) {
  for (const std::vector<Move>& action : line) {
    for (const Move& m : action) {
      if (!s.may_move() || !is_pseudo_legal(s, m)) return false;
      s.apply(m);
    }
    if (!s.obligations_met() || !royals_safe(s)) return false;
    s.submit(CheckRuling::none());
  }
  return true;
}

inline bool mated(const GameState& s) {
  return detect_mate(s, Detector::Hypercuboid, SearchBudget::nodes(2'000'000)).verdict == Outcome::Checkmate;
}

// Every successor of `s` (its every legal action): the brute-force
// reference, else E4's exhaustive listing. False: neither could list them.
inline bool successors(const GameState& s, std::vector<GameState>& out) {
  out.clear();
  const t5d_test::Reference r = t5d_test::reference_actions(s, 5000);
  if (!r.overflow) {
    out = r.successors;
    return true;
  }
  const ActionList l = list_legal_actions(s, 200000, Detector::Hypercuboid, SearchBudget::nodes(3'000'000));
  if (!l.stats.exhaustive || l.reached_cap) return false;
  for (const LegalAction& a : l.actions) out.push_back(a.result);
  return true;
}

// Whether the side to move at `after` mates in one: 1 yes, 0 no, -1 the
// references could not tell. `actions`: how many legal actions it has.
inline int mates_in_one(const GameState& after, std::size_t& actions) {
  std::vector<GameState> next;
  actions = 0;
  if (!successors(after, next)) return -1;
  actions = next.size();
  for (const GameState& s : next) {
    const MateResult m = detect_mate(s, Detector::Hypercuboid, SearchBudget::nodes(2'000'000));
    if (m.verdict == Outcome::Unknown) return -1;
    if (m.verdict == Outcome::Checkmate) return 1;
  }
  return 0;
}

// Positions and an action of the side to move at each: random legal play.
struct Case {
  GameState s;
  std::vector<Move> action;
};
inline std::vector<Case> cases(const RuleOptions& rules, int games, int turns, std::uint64_t seed) {
  std::vector<Case> out;
  t5d_test::Rng rng(seed * 7919 + 1);
  for (const GameState& s : playable(t5d_test::sparse_positions(rules, games, turns, seed))) {
    const std::optional<std::vector<Move>> a = t5d_test::random_legal_action(s, rng);
    if (a) out.push_back(Case{s, *a});
  }
  return out;
}

}  // namespace t5d_test::v3

#endif  // T5D_TEST_V3_SUPPORT_HPP
