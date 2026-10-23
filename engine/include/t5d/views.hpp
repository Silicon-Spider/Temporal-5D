// t5d/views.hpp - the two views a decision state is read from (design spec
// §6; register B11; build-log spec issue 172): what the expansion's pin and
// check relations (t5d/expand.hpp) read of each side.
//
// THE TWO VIEWS (spec issue 172). Pieces move only on boards of their own
// colour (L4, L5), so at a decision state only the acting player's pieces can
// move, and the opponent's captures exist only once the acting player's
// boards have advanced. Two multiverses are read:
//
//  * the ACTING player's view: the state itself - its pieces on its playable
//    boards, as the move generator offers them (t5d/movegen.hpp);
//  * the OPPONENT's view: the multiverse check status reads (R1, C6;
//    t5d/check.hpp passed_multiverse) - the acting player passes on every
//    ACTIVE board of its IN THE PRESENT that it has not played, and the
//    opponent's pieces on all its playable boards (the passed copies, and every
//    other board with the opponent to move) are read. At a turn start this is
//    R1's check status exactly; mid-action it is the same pass of the boards
//    still unplayed (none once the present has shifted: then the state itself,
//    as C4's submission test reads it). Where the pass is undefined - Q6's
//    alternative, a deferred timeline of the mover's in the present mid-action
//    (passed_timelines throws) - the state itself is read.
//
// A place on a passed copy - (p, present + 1) - is reported at the board it
// copies, (p, present): the copy's squares are that board's, one half-turn on.
// Every other place of the opponent's view is a board of the state.

#ifndef T5D_VIEWS_HPP
#define T5D_VIEWS_HPP

#include <vector>

#include "t5d/multiverse.hpp"
#include "t5d/state.hpp"

namespace t5d {

/// The opponent's view of a decision state (see the header comment).
struct PassView {
  /// The multiverse the opponent's pieces are read on.
  Multiverse multiverse;
  /// The timelines passed, ascending (empty: the state's own multiverse).
  std::vector<int> passed;
  /// The state's present: a passed copy stands at present + 1.
  int present = 0;
  /// False where the pass is undefined (Q6's alternative; the header comment).
  bool pass_defined = true;

  /// The half-turn of the state's board a place of `multiverse` is reported
  /// at: present for a passed copy's place (p, present + 1), else `half_turn`.
  int state_half_turn(int timeline, int half_turn) const noexcept;
};

/// The opponent's view of `state` - check status's pass (t5d/check.hpp
/// passed_timelines, passed_multiverse), called, never changed.
PassView pass_view(const GameState& state);

}  // namespace t5d

#endif  // T5D_VIEWS_HPP
