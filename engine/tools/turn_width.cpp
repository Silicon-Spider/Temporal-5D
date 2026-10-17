// The width of one action (see turn_width.hpp for the definitions).

#include "turn_width.hpp"

#include <algorithm>
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
  const int source = move.from_timeline;
  const bool source_mandatory_at_start = contains(start_mandatory_, source);
  const bool source_mandatory_now = state_.mandatory(source);
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
  ++counts_.moves;
  if (source_mandatory_at_start) {
    ++counts_.moves_from_mandatory;
  } else {
    ++counts_.optional_moves;
  }
  if (!source_mandatory_now) ++counts_.optional_when_made;
  if (source_mandatory_at_start && !source_mandatory_now) ++counts_.moves_on_released;
  if (!source_mandatory_at_start && source_mandatory_now) ++counts_.moves_on_gained;
  if (shifted) ++counts_.moves_after_shift;
  if (move.jump()) {
    ++counts_.jumps;
    if (move.branching()) {
      ++counts_.branches;
    } else {
      // J4: a jump that does not branch is made on its destination board too.
      const int destination = move.to_timeline;
      play(destination);
      const bool destination_mandatory_at_start = contains(start_mandatory_, destination);
      if (source_mandatory_at_start && destination_mandatory_at_start) {
        ++counts_.jumps_between_mandatory;
      }
      if (!source_mandatory_at_start && destination_mandatory_at_start) {
        ++counts_.optional_onto_mandatory;
      }
    }
  }
  if (state_.present() < present_before) ++counts_.rewinds;  // T8, T12
  for (const int p : state_.mandatory_timelines()) {
    if (!contains(start_mandatory_, p)) insert(gained_, p);  // T12
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

}  // namespace t5d::tools
