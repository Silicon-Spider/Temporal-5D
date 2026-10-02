// A brute-force game-tree search for tiny positions: the exact game value
// within a horizon, for checking the scanner's proofs (t5d/scanner.hpp;
// design spec §15.4: "scanner proofs on constructed mate positions
// cross-checked against the gate's references").
//
// It shares none of the scanner's machinery: a node's legal successors come
// from the brute-force reference's enumeration of every move sequence
// (test_reference.hpp - no transposition table, no product space, no
// pruning); at the horizon, whether a legal action exists at all is the same
// enumeration stopped at the first legal one; the verdict at a node without
// one is R1's check status (C5); and the value is plain minimax over EVERY
// successor, memoised by position.

#ifndef T5D_TEST_GAME_TREE_HPP
#define T5D_TEST_GAME_TREE_HPP

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "test_reference.hpp"

namespace t5d_test {

/// The value of a position for its side to move, within a horizon.
struct GameValue {
  enum class Kind : std::uint8_t {
    Unknown = 0,  // the horizon (or a budget) was reached first
    Win,          // mates within `distance` half-turns, whatever the defence
    Loss,         // is mated within `distance`, whatever it does
    Draw,         // stalemate now (distance 0)
    Other,        // neither side forces mate within the horizon (all known)
  };
  Kind kind = Kind::Unknown;
  int distance = 0;
};

inline const char* name_of(GameValue::Kind k) {
  switch (k) {
    case GameValue::Kind::Unknown: return "unknown";
    case GameValue::Kind::Win: return "win";
    case GameValue::Kind::Loss: return "loss";
    case GameValue::Kind::Draw: return "draw";
    case GameValue::Kind::Other: return "other";
  }
  return "?";
}

class GameTree {
 public:
  /// `limit`: the most move sequences the enumeration may try per position;
  /// `max_successors`: the most successor states it stores in all - beyond
  /// them every answer not yet known is Unknown (overflowed()), so that a
  /// test's memory stays bounded on a shared machine.
  /// `max_work`: the most move sequences tried in all (the time it may take).
  explicit GameTree(std::uint64_t limit = 20000, std::uint64_t max_successors = 60000,
                    std::uint64_t max_work = 300000)
      : limit_(limit), max_successors_(max_successors), max_work_(max_work) {}

  /// Exact minimax within `depth` half-turns: Win(d) with the least d, Loss(d)
  /// with the greatest (the loser delays), Draw for a stalemate now; Other
  /// when every line is known to the horizon and none is forced; Unknown when
  /// some line ran into the horizon or a budget undecided.
  GameValue solve(const t5d::GameState& s, int depth) {
    if (depth == 0) return compute(s, 0);  // cheap, and the most numerous: not kept
    const std::uint64_t key = s.zobrist() * 31 + static_cast<std::uint64_t>(depth);
    for (const Memo& m : memo_[key]) {
      if (t5d::same_position(m.state, s)) return m.value;
    }
    const GameValue v = compute(s, depth);
    memo_[key].push_back(Memo{s, v});
    return v;
  }

  bool overflowed() const noexcept { return overflow_; }

 private:
  struct Memo {
    t5d::GameState state;
    GameValue value;
  };

  // Whether some move sequence of the turn ends in a legal submission (T5,
  // C4): the reference's enumeration, stopped at the first. -1: the limit
  // was reached first.
  int any_legal(const t5d::GameState& s, std::uint64_t& tried) {
    if (++tried > limit_) return -1;
    if (s.obligations_met() && t5d::royals_safe(s)) return 1;
    if (!s.may_move()) return 0;
    t5d::MoveList moves;
    t5d::generate_moves(s, moves);
    for (const t5d::Move& m : moves) {
      t5d::GameState t = s;
      t.apply(m);
      const int r = any_legal(t, tried);
      if (r != 0) return r;
    }
    return 0;
  }

  GameValue terminal(const t5d::GameState& s) const {
    // C5: no legal action - checkmate if in check (R1), else stalemate.
    return t5d::in_check(s) ? GameValue{GameValue::Kind::Loss, 0}
                            : GameValue{GameValue::Kind::Draw, 0};
  }

  GameValue compute(const t5d::GameState& s, int depth) {
    if (work_ >= max_work_) {
      overflow_ = true;
      return GameValue{};
    }
    if (depth == 0) {
      std::uint64_t tried = 0;
      const int r = any_legal(s, tried);
      work_ += tried;
      if (r < 0) overflow_ = true;
      return r == 0 ? terminal(s) : GameValue{};
    }
    if (stored_ >= max_successors_) {
      overflow_ = true;
      return GameValue{};
    }
    const Reference ref = reference_actions(s, std::min(limit_, max_work_ - work_));
    work_ += ref.sequences;
    if (ref.overflow) {
      overflow_ = true;
      return GameValue{};
    }
    stored_ += ref.successors.size();
    if (ref.successors.empty()) return terminal(s);
    int win = -1;
    int longest_loss = 0;
    bool all_lost = true;
    bool unknown = false;
    for (const t5d::GameState& c : ref.successors) {
      const GameValue v = solve(c, depth - 1);
      if (v.kind == GameValue::Kind::Loss) {
        if (win < 0 || v.distance + 1 < win) win = v.distance + 1;
      }
      if (v.kind == GameValue::Kind::Win) {
        longest_loss = std::max(longest_loss, v.distance + 1);
      } else {
        all_lost = false;
      }
      if (v.kind == GameValue::Kind::Unknown) unknown = true;
    }
    if (win >= 0) return GameValue{GameValue::Kind::Win, win};
    if (all_lost) return GameValue{GameValue::Kind::Loss, longest_loss};
    if (unknown) return GameValue{};
    return GameValue{GameValue::Kind::Other, 0};
  }

  std::uint64_t limit_;
  std::uint64_t max_successors_;
  std::uint64_t max_work_;
  std::uint64_t work_ = 0;
  std::uint64_t stored_ = 0;
  bool overflow_ = false;
  std::unordered_map<std::uint64_t, std::vector<Memo>> memo_;
};

}  // namespace t5d_test

#endif  // T5D_TEST_GAME_TREE_HPP
