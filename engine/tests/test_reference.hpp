// A brute-force reference for the legal-action searches (t5d/search.hpp):
// every sequence of moves the state machine allows, one at a time, with no
// transposition table, no product space and no pruning - slow and simple, so
// that the two detectors can be checked against something that shares none of
// their cleverness. It counts the legal actions as t5d/search.hpp defines
// them (a set of moves, its branching moves in order) and the distinct
// successor states, by its own keys and comparisons.

#ifndef T5D_TEST_REFERENCE_HPP
#define T5D_TEST_REFERENCE_HPP

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/move.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"

namespace t5d_test {

struct Reference {
  std::set<std::string> actions;              // one key per action (see key)
  std::vector<t5d::GameState> successors;     // distinct (same_position)
  std::uint64_t sequences = 0;                // move sequences tried
  bool overflow = false;                      // more than the limit: incomplete

  /// An action's key: its non-branching moves sorted, then its branching
  /// moves in the order made.
  static std::string key(const std::vector<t5d::Move>& seq, bool even) {
    std::vector<std::string> plain;
    std::string branches;
    for (const t5d::Move& m : seq) {
      if (m.branching()) {
        branches += t5d::to_string(m, even) + ";";
      } else {
        plain.push_back(t5d::to_string(m, even));
      }
    }
    std::sort(plain.begin(), plain.end());
    std::string out;
    for (const std::string& p : plain) out += p + ",";
    return out + "|" + branches;
  }
};

namespace detail {

inline void enumerate(const t5d::GameState& s, std::vector<t5d::Move>& seq, Reference& ref,
                      std::uint64_t limit) {
  if (++ref.sequences > limit) {
    ref.overflow = true;
    return;
  }
  // T5: submittable once the obligations are met; C4, R1: legal if every
  // royal of the mover's is then safe.
  if (s.obligations_met() && t5d::royals_safe(s)) {
    ref.actions.insert(Reference::key(seq, s.multiverse().even()));
    t5d::GameState t = s;
    t.submit(t5d::CheckRuling::none());
    if (std::none_of(ref.successors.begin(), ref.successors.end(),
                     [&](const t5d::GameState& u) { return t5d::same_position(u, t); })) {
      ref.successors.push_back(t);
    }
  }
  if (!s.may_move()) return;  // Q4
  t5d::MoveList moves;
  t5d::generate_moves(s, moves);
  for (const t5d::Move& m : moves) {
    t5d::GameState t = s;
    t.apply(m);
    seq.push_back(m);
    enumerate(t, seq, ref, limit);
    seq.pop_back();
    if (ref.overflow) return;
  }
}

}  // namespace detail

/// Every legal action of `turn_start` by enumeration, trying at most `limit`
/// move sequences (overflow set beyond it).
inline Reference reference_actions(const t5d::GameState& turn_start, std::uint64_t limit) {
  Reference ref;
  std::vector<t5d::Move> seq;
  detail::enumerate(turn_start, seq, ref, limit);
  return ref;
}

}  // namespace t5d_test

#endif  // T5D_TEST_REFERENCE_HPP
