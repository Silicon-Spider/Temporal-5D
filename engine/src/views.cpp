// The two views of a decision state (see t5d/views.hpp). Rule IDs refer to the
// rules reference, docs/rules/README.md. The opponent's view is check status's
// own pass (t5d/check.hpp), only called.

#include "t5d/views.hpp"

#include <algorithm>
#include <stdexcept>
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

PassView pass_view(const GameState& state) {
  std::vector<int> passed;
  try {
    passed = passed_timelines(state);
  } catch (const std::logic_error&) {
    // Q6's alternative: a deferred timeline of the mover's in the present -
    // the pass is check status's at a turn's start only (t5d/check.hpp).
    return PassView{state.multiverse(), {}, state.present(), false};
  }
  return PassView{passed_multiverse(state), std::move(passed), state.present(), true};
}

}  // namespace t5d
