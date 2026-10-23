// The two views of a decision state (see t5d/views.hpp). Rule IDs refer to the
// rules reference, docs/rules/README.md. The opponent's view is check status's
// own pass (t5d/check.hpp), only called.

#include "t5d/views.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "t5d/check.hpp"

namespace t5d {

int PassView::state_half_turn(int timeline, int half_turn) const noexcept {
  if (half_turn == present + 1 && std::binary_search(passed.begin(), passed.end(), timeline)) {
    return present;
  }
  return half_turn;
}

namespace {

// Q6's alternative: a deferred timeline of the mover's, active and in the
// present - the board check status would pass. passed_timelines refuses such a
// state (t5d/check.hpp: the pass is check status's at a turn's start, where no
// deferral of the mover's is left); this is that precondition, tested first, so
// that nothing else passed_timelines might throw is taken for it.
bool pass_undefined(const GameState& state) {
  for (const GameState::Deferral& d : state.deferred_timelines()) {
    const Timeline* t = state.multiverse().timeline(d.timeline);
    if (t != nullptr && state.active(d.timeline) && t->last_half_turn() == state.present() &&
        state.playable_by_mover(d.timeline)) {
      return true;
    }
  }
  return false;
}

}  // namespace

PassView pass_view(const GameState& state) {
  if (pass_undefined(state)) return PassView{state.multiverse(), {}, state.present(), false};
  std::vector<int> passed = passed_timelines(state);
  return PassView{passed_multiverse(state), std::move(passed), state.present(), true};
}

}  // namespace t5d
