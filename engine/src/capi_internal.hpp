// What the C API's translation units share (src/capi.cpp, src/capi_search.cpp):
// a game handle's state and its count of moves in the action, making a game
// handle, and the description of an expansion in the C layout. Internal: not
// part of the library's interface.

#ifndef T5D_SRC_CAPI_INTERNAL_HPP
#define T5D_SRC_CAPI_INTERNAL_HPP

#include <stdexcept>

#include "t5d/capi.h"
#include "t5d/expand.hpp"
#include "t5d/state.hpp"

namespace t5d::capi_internal {

/// A feature the engine does not implement (T5D_ERR_UNSUPPORTED).
class Unsupported : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// The state a game handle holds.
const GameState& game_state(const t5d_game& game) noexcept;
/// The moves made in the handle's action in progress (register A6; 0 at a turn start).
int game_moves_in_action(const t5d_game& game) noexcept;
/// A new game handle on `state` with `moves_in_action` moves made in its action
/// (legal submissions, no record). Throws std::bad_alloc, or std::invalid_argument
/// for a negative count.
t5d_game* new_game(const GameState& state, int moves_in_action);
/// An expansion's arrays in t5d_expansion's layout (t5d_game_expand's).
void describe_expansion(const Expansion& expansion, t5d_expansion& out);
/// An expansion's options from the C layout (t5d_game_expand's reading; NULL:
/// the defaults). Throws std::invalid_argument, or Unsupported for a relation
/// that is not built.
ExpandOptions expand_options(const t5d_expand_options* options);

}  // namespace t5d::capi_internal

#endif  // T5D_SRC_CAPI_INTERNAL_HPP
