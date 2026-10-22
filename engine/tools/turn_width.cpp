// The width of one action, and the statistics of many (see turn_width.hpp for
// the definitions).

#include "turn_width.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace t5d::tools {

namespace {

bool contains(const std::vector<int>& sorted, int p) {
  return std::binary_search(sorted.begin(), sorted.end(), p);
}

// Inserts p into a sorted vector; false if it was there already.
bool insert(std::vector<int>& sorted, int p) {
  const auto at = std::lower_bound(sorted.begin(), sorted.end(), p);
  if (at != sorted.end() && *at == p) return false;
  sorted.insert(at, p);
  return true;
}

}  // namespace

TurnCounter::TurnCounter(const GameState& turn_start)
    : state_(turn_start),
      start_mandatory_(turn_start.mandatory_timelines()),
      start_playable_(turn_start.playable_timelines()) {
  const Multiverse& mv = turn_start.multiverse();
  counts_.boards = mv.board_count();
  counts_.timelines = mv.timeline_count();
  for (const TimelineRef& t : mv.timelines()) {
    if (turn_start.active(t->position())) ++counts_.active;
  }
  counts_.mandatory = turn_start.mandatory_count();
  counts_.playable = static_cast<int>(start_playable_.size());
  counts_.optional = counts_.playable - counts_.mandatory;
  if (static_cast<std::size_t>(counts_.mandatory) != start_mandatory_.size()) {
    throw std::logic_error("TurnCounter: the mandatory count and the mandatory boards disagree");
  }
}

void TurnCounter::apply(const Move& move) {
  // The boards the move is made on (T3): its source and, for a jump that does
  // not branch, its destination (J4) - a branching jump's destination is a
  // history board, which no move changes (J5). Their status at the action's
  // start, and now, before the move.
  const int source = move.from_timeline;
  const bool two_boards = move.jump() && !move.branching();
  const int destination = move.to_timeline;
  const bool source_at_start = contains(start_mandatory_, source);
  const bool source_now = state_.mandatory(source);
  const bool destination_at_start = two_boards && contains(start_mandatory_, destination);
  const bool destination_now = two_boards && state_.mandatory(destination);
  const bool shifted = state_.obligations_met();
  const int present_before = state_.present();

  state_.apply(move);  // throws for a move that does not fit, nothing changed

  // T3: a board takes one move; T7: nothing lands on a board the action made.
  // So each board played is one of the start's playable boards, played once.
  const auto play = [&](int p) {
    if (!contains(start_playable_, p)) {
      throw std::logic_error("TurnCounter: a move was made on timeline " + std::to_string(p) +
                             ", whose board was not playable at the action's start (T3, T7)");
    }
    if (!insert(played_, p)) {
      throw std::logic_error("TurnCounter: timeline " + std::to_string(p) +
                             "'s board was moved on twice in one action (T3)");
    }
  };
  play(source);
  if (two_boards) play(destination);
  ++counts_.moves;
  // By every board the move is made on: optional if one of them was not
  // mandatory - at the action's start, or when the move was made. (Without a
  // destination board both of its flags are false.)
  if (!source_at_start || (two_boards && !destination_at_start)) ++counts_.optional_moves;
  if (!source_now || (two_boards && !destination_now)) ++counts_.optional_when_made;
  if ((source_at_start && !source_now) || (destination_at_start && !destination_now)) {
    ++counts_.moves_on_released;
  }
  if ((!source_at_start && source_now) || (!destination_at_start && destination_now)) {
    ++counts_.moves_on_gained;
  }
  // By the source board alone.
  if (source_at_start) {
    ++counts_.moves_from_mandatory;
  } else {
    ++counts_.optional_by_source;
  }
  if (shifted) ++counts_.moves_after_shift;
  if (move.jump()) {
    ++counts_.jumps;
    if (move.branching()) {
      ++counts_.branches;
    } else if (source_at_start && destination_at_start) {
      ++counts_.jumps_between_mandatory;
    } else if (destination_at_start) {
      ++counts_.optional_onto_mandatory;
    } else if (source_at_start) {
      ++counts_.mandatory_onto_optional;
    }
  }
  if (state_.present() < present_before) ++counts_.rewinds;  // T8, T12
  for (const int p : state_.mandatory_timelines()) {
    if (!contains(start_mandatory_, p)) insert(gained_, p);  // T11, T12
  }
}

TurnWidth TurnCounter::width() const {
  TurnWidth w = counts_;
  w.boards_played = static_cast<int>(played_.size());
  for (const int p : played_) {
    if (contains(start_mandatory_, p)) ++w.mandatory_played;
  }
  w.optional_played = w.boards_played - w.mandatory_played;
  w.mandatory_released = w.mandatory - w.mandatory_played;
  w.mandatory_gained = static_cast<int>(gained_.size());
  return w;
}

TurnWidth classify_action(const GameState& turn_start, std::span<const Move> moves) {
  TurnCounter counter(turn_start);
  for (const Move& m : moves) counter.apply(m);
  return counter.width();
}

// ---- the summary ----------------------------------------------------------------------------

int mandatory_class(int mandatory) {
  if (mandatory < 1) {
    throw std::invalid_argument("mandatory_class: an action starts with a mandatory board (T5)");
  }
  return std::min(mandatory, 4) - 1;
}

double dense_eval_flops(std::size_t boards) {
  const double t = kCostTokensPerBoard * static_cast<double>(boards) + kCostGlobalTokens;
  return 4 * t * t * kCostDModel * kCostLayers +
         kCostLinearCoefficient * t * kCostDModel * kCostDModel * kCostLayers;
}

double share_of(double part, double whole) { return whole == 0 ? 0 : part / whole; }

double quantile(std::vector<double> values, double q) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const double at = q * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(at));
  const auto hi = static_cast<std::size_t>(std::ceil(at));
  return values[lo] + (values[hi] - values[lo]) * (at - static_cast<double>(lo));
}

namespace {

Spread spread_of(const std::vector<double>& v) {
  Spread s;
  if (v.empty()) return s;
  double sum = 0;
  for (const double x : v) sum += x;
  s.mean = sum / static_cast<double>(v.size());
  s.median = quantile(v, 0.5);
  s.p90 = quantile(v, 0.9);
  s.max = *std::max_element(v.begin(), v.end());
  return s;
}

}  // namespace

TurnStats turn_stats(std::span<const TurnWidth* const> actions) {
  TurnStats s;
  std::vector<double> mandatory, optional_boards, moves;
  for (const TurnWidth* a : actions) {
    const TurnWidth& w = *a;
    if (w.moves < 1) {
      throw std::invalid_argument("turn_stats: an action makes a move (T5)");
    }
    ++s.actions;
    ++s.mandatory_classes[static_cast<std::size_t>(mandatory_class(w.mandatory))];
    ++s.moves_classes[static_cast<std::size_t>(std::min(w.moves, 4) - 1)];
    ++s.optional_moves_classes[static_cast<std::size_t>(std::clamp(w.optional_moves, 0, 3))];
    mandatory.push_back(w.mandatory);
    optional_boards.push_back(w.optional);
    moves.push_back(w.moves);
    s.mandatory_over_1 += w.mandatory > 1 ? 1 : 0;
    s.mandatory_over_2 += w.mandatory > 2 ? 1 : 0;
    s.playable_over_1 += w.playable > 1 ? 1 : 0;
    s.with_optional_boards += w.optional > 0 ? 1 : 0;
    s.one_move_over_1_mandatory += w.moves == 1 && w.mandatory > 1 ? 1 : 0;
    s.with_optional_moves += w.optional_moves > 0 ? 1 : 0;
    s.with_optional_by_source += w.optional_by_source > 0 ? 1 : 0;
    s.with_optional_when_made += w.optional_when_made > 0 ? 1 : 0;
    s.with_travel += w.jumps > 0 ? 1 : 0;
    s.with_branch += w.branches > 0 ? 1 : 0;
    s.with_moves_after_shift += w.moves_after_shift > 0 ? 1 : 0;
    s.rewound += w.rewinds > 0 ? 1 : 0;
    s.with_mandatory_released += w.mandatory_released > 0 ? 1 : 0;
    s.optional_moves += w.optional_moves;
    s.optional_by_source += w.optional_by_source;
    s.optional_when_made += w.optional_when_made;
    s.optional_played += w.optional_played;
    s.boards_played += w.boards_played;
    s.moves_after_shift += w.moves_after_shift;
    s.jumps_between_mandatory += w.jumps_between_mandatory;
    s.optional_onto_mandatory += w.optional_onto_mandatory;
    s.mandatory_onto_optional += w.mandatory_onto_optional;
    s.mandatory_released += w.mandatory_released;
    s.mandatory_gained += w.mandatory_gained;
    s.moves_on_released += w.moves_on_released;
    s.moves_on_gained += w.moves_on_gained;
    // The compute: moves x the cost of a move, constant and dense.
    const double flops = static_cast<double>(w.moves) * dense_eval_flops(w.boards);
    s.moves_total += w.moves;
    s.dense_flops += flops;
    if (w.mandatory > 1) {
      s.moves_mandatory_over_1 += w.moves;
      s.dense_flops_mandatory_over_1 += flops;
    }
    if (w.mandatory > 2) {
      s.moves_mandatory_over_2 += w.moves;
      s.dense_flops_mandatory_over_2 += flops;
    }
    if (w.playable > 1) {
      s.moves_playable_over_1 += w.moves;
      s.dense_flops_playable_over_1 += flops;
    }
  }
  s.mandatory = spread_of(mandatory);
  s.optional_boards = spread_of(optional_boards);
  s.moves = spread_of(moves);
  return s;
}

}  // namespace t5d::tools
