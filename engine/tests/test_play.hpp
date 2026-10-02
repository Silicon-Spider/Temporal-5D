// Random legal play for the tests of the verification layer (stage E8): the
// sparse starts of the scanner's tests (few pieces, several timelines, so that
// random play runs into checks, mates and stalemates within a few turns), a
// random legal action, and positions by random play under a ruleset.
//
// Every start is written here from the rulebook - never taken from a game
// corpus or another implementation's test data.

#ifndef T5D_TEST_PLAY_HPP
#define T5D_TEST_PLAY_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"

namespace t5d_test {

struct SparseStart {
  std::vector<std::string> boards;
  t5d::BoardSize size;
};

inline std::vector<SparseStart> sparse_starts() {
  const t5d::BoardSize k5{5, 5};
  const t5d::BoardSize k4{4, 4};
  return {
      {{"[k4/5/5/5/K3R:-0:1:w]", "[k4/5/5/5/K3R:+0:1:w]"}, k5},
      {{"[2k2/5/5/5/QK3:0:1:w]"}, k5},
      {{"[k3/4/4/K1Q1:0:1:w]"}, k4},
      {{"[2k/3/K2:0:1:w]"}, t5d::BoardSize{3, 3}},
      {{"[3k/4/4/KR2:-1:1:w]", "[4/1p2/2P1/4:0:1:w]", "[k3/4/4/2RK:+1:1:w]"}, k4},
      {{"[k4/p4/5/4P/K4:-0:1:w]", "[4k/5/2N2/5/4K:+0:1:w]"}, k5},
      {{"[1k2/4/4/1KN1:-0:1:w]", "[2k1/4/4/B1K1:+0:1:w]"}, k4},
      {{"[k4/5/2q2/5/2K1R:0:1:w]"}, k5},
      {{"[k3/4/1R2/K3:0:1:w]"}, k4},
      {{"[1k3/5/1K3/5/R3R:0:1:w]"}, k5},
  };
}

/// A random legal action: random pseudo-legal sequences until one is safe,
/// else E4's witness; nullopt if the turn has none.
inline std::optional<std::vector<t5d::Move>> random_legal_action(const t5d::GameState& s, Rng& rng) {
  t5d::MoveList moves;
  for (int attempt = 0; attempt < 24; ++attempt) {
    t5d::GameState t = s;
    std::vector<t5d::Move> seq;
    for (int step = 0; step < 64; ++step) {
      if (t.obligations_met() && (!t.rules().moves_after_shift || rng.chance(60))) break;
      t5d::generate_moves(t, moves);
      if (moves.empty()) break;
      const t5d::Move m = moves[static_cast<std::size_t>(rng.below(static_cast<int>(moves.size())))];
      t.apply(m);
      seq.push_back(m);
    }
    if (t.obligations_met() && t5d::royals_safe(t)) return seq;
  }
  const t5d::MateResult r =
      t5d::detect_mate(s, t5d::Detector::Hypercuboid, t5d::SearchBudget::nodes(100000));
  if (r.verdict == t5d::Outcome::Ongoing) return r.witness;
  return std::nullopt;
}

/// Turn starts of random legal play from the sparse starts under `rules`: up
/// to `turns` per game, `games` games per start.
inline std::vector<t5d::GameState> sparse_positions(const t5d::RuleOptions& rules, int games,
                                                    int turns, std::uint64_t seed) {
  std::vector<t5d::GameState> out;
  Rng rng(seed);
  for (const SparseStart& start : sparse_starts()) {
    for (int g = 0; g < games; ++g) {
      t5d::GameState s = game(start.boards, start.size, rules);
      for (int turn = 0; turn < turns; ++turn) {
        out.push_back(s);
        const std::optional<std::vector<t5d::Move>> a = random_legal_action(s, rng);
        if (!a) break;
        for (const t5d::Move& m : *a) s.apply(m);
        s.submit();
      }
    }
  }
  return out;
}

/// Turn starts of random legal play from named variants under `rules`.
inline std::vector<t5d::GameState> variant_positions(const t5d::RuleOptions& rules,
                                                     const std::vector<const char*>& names,
                                                     int turns, std::uint64_t seed) {
  std::vector<t5d::GameState> out;
  Rng rng(seed);
  for (const char* name : names) {
    const t5d::Variant* v = t5d::find_variant(name);
    if (v == nullptr) continue;
    t5d::GameState s = t5d::variant_start(*v, rules);
    for (int turn = 0; turn < turns; ++turn) {
      out.push_back(s);
      const std::optional<std::vector<t5d::Move>> a = random_legal_action(s, rng);
      if (!a) break;
      for (const t5d::Move& m : *a) s.apply(m);
      s.submit();
    }
  }
  return out;
}

}  // namespace t5d_test

#endif  // T5D_TEST_PLAY_HPP
