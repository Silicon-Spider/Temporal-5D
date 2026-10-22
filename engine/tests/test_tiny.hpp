// Random tiny multiverses for the legal-action searches' tests (moved from
// test_detectors.cpp, whose brute-force comparison describes them, so that the
// flat form's turn generator - test_turngen.cpp - is checked on the same
// positions): 1x1 or 2x1 boards, up to four timelines of each creator beside
// the central ones at staggered times, under random rulesets (Q1-Q6, Q8, Q20,
// J3's other reading). The same draws as before the move, so the detectors'
// tests see the same positions.

#ifndef T5D_TEST_TINY_HPP
#define T5D_TEST_TINY_HPP

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/state.hpp"
#include "test_support.hpp"

namespace t5d_test {

using namespace t5d;

inline std::optional<t5d::GameState> tiny_position(Rng& rng, int max_axes, bool kings, int width,
                                       std::vector<std::string>& boards) {
  const BoardSize size{width, 1};
  const bool even = rng.chance(50);
  const int white_side = rng.below(5);
  const int black_side = rng.below(5);
  const auto label = [&](int p) {
    if (even) return p >= 0 ? "+" + std::to_string(p) : "-" + std::to_string(-p - 1);
    return p == 0 ? std::string("0") : (p > 0 ? "+" : "") + std::to_string(p);
  };
  std::vector<std::pair<int, int>> places;  // (timeline, half-turns from T1 White)
  for (int p = even ? -1 - black_side : -black_side; p <= white_side; ++p) {
    const bool central = even ? (p == -1 || p == 0) : p == 0;
    const int start = central ? 0 : rng.below(3);
    const int head = start + 1 + rng.below(4);
    for (int j = start; j <= head; ++j) places.emplace_back(p, j);
  }
  const auto make = [&](const std::vector<std::string>& contents) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < places.size(); ++i) {
      const int j = places[i].second;
      out.push_back("[" + contents[i] + ":" + label(places[i].first) + ":" + std::to_string(j / 2 + 1) +
                    ":" + (j % 2 == 0 ? "w" : "b") + "]");
    }
    return out;
  };
  RuleOptions r;
  r.activation = rng.chance(30) ? ActivationRule::Consecutive : ActivationRule::Count;
  r.initial_credit = rng.chance(25) ? InitialTimelineCredit::MinusZeroToBlack : InitialTimelineCredit::None;
  r.play_activates = rng.chance(20);
  r.moves_after_shift = !rng.chance(25);  // Q4: R8 (the default), or the Guide-literal reading
  r.mandatory = rng.chance(25) ? MandatoryBoards::AllActive : MandatoryBoards::InPresent;
  r.reactivation = rng.chance(20) ? Reactivation::Deferred : Reactivation::Rewinds;
  r.earlier_playable_jump_branches = rng.chance(20);
  r.inactive_history_is_destination = !rng.chance(25);
  r.turn_zero_reachable = !rng.chance(20);
  const Parity parity = even ? Parity::Even : Parity::Odd;
  Colour mover = Colour::White;
  try {
    mover = game_from_fen(make(std::vector<std::string>(places.size(), std::to_string(width))), size,
                          parity, r)
                .to_move();
  } catch (const std::exception&) {
    return std::nullopt;  // not a position (no present on the colour to move, ...)
  }
  const auto piece = [&](bool mine) {
    // Every third position king-heavy: royal captures across created timelines.
    const char* kinds = kings ? "QKRKBK" : "QQRBNK";
    const char c = kinds[rng.below(6)];
    const bool white = mine == (mover == Colour::White);
    return std::string(1, white ? c : static_cast<char>(c - 'A' + 'a'));
  };
  // One square's content ("" for empty): on the mover's playable boards
  // mostly the mover's pieces (the first square), on the others mostly empty.
  const auto square = [&](bool head, bool movers, bool first) {
    const int roll = rng.below(100);
    if (head && movers) {
      if (first) return roll < 15 ? std::string() : roll < 85 ? piece(true) : piece(false);
      return roll < 60 ? std::string() : roll < 80 ? piece(false) : piece(true);
    }
    if (movers) return roll < 60 ? std::string() : roll < 90 ? piece(false) : piece(true);
    return roll < 50 ? std::string() : roll < 80 ? piece(false) : piece(true);
  };
  std::vector<std::string> contents;
  for (std::size_t i = 0; i < places.size(); ++i) {
    const bool head = i + 1 == places.size() || places[i + 1].first != places[i].first;
    const bool movers = (places[i].second % 2 == 0) == (mover == Colour::White);
    std::string row;
    int empty = 0;
    for (int f = 0; f < width; ++f) {
      const std::string q = square(head, movers, f == 0);
      if (q.empty()) {
        ++empty;
        continue;
      }
      if (empty > 0) row += std::to_string(empty);
      empty = 0;
      row += q;
    }
    if (empty > 0) row += std::to_string(empty);
    contents.push_back(row);
  }
  boards = make(contents);
  try {
    GameState s = game_from_fen(boards, size, parity, r);
    if (static_cast<int>(s.playable_timelines().size()) > max_axes) return std::nullopt;
    return s;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace t5d_test

#endif  // T5D_TEST_TINY_HPP
