// tools/turn_width.hpp - the width of one action (register M22; design spec
// §8.2 "Measure first", §15.1 Phase 1): the boards the mover must move on at
// the action's start, the moves the action makes, which of them are
// optional, and how the action's obligations change while it is made. Counted
// by the rules engine alone - GameState and its moves, nothing else (no
// network, no search) - from the turn rules of docs/rules/README.md (T3-T8,
// T12, J3-J5, R8, R9). t5d-measure's --turns and --turn-summary report it for
// every action of 5dpgn records.
//
// THE DEFINITIONS. An action is one player's moves from the start of their
// turn to its submission (T1). Every count below is of one action.
//
//  * At the action's START - the turn-start state, before any move:
//    - boards: every board of the multiverse, history boards included (a
//      Turn Zero game's T0 board too, as M7 counted);
//    - timelines, and active: the active timelines (T10-T13);
//    - MANDATORY: the boards the mover must move on - the active boards in
//      the present, on the mover's colour (T6; ruling R9) -
//      GameState::mandatory_count() (under the default MandatoryBoards::
//      InPresent; whatever the ruleset says under another);
//    - playable: the mover's playable boards, active or not, in the present
//      or not (L5, T6);
//    - OPTIONAL: playable minus mandatory - the boards ahead of the present
//      and those on inactive timelines, wherever they lie (T6).
//  * A move is made ON its source board and, when it is a jump that does not
//    branch, on its destination board too (T3, J4: "the move counts on both
//    boards"); a branching jump's destination is a history board, which no
//    move changes (J5). Each board takes one move (T3) and a new timeline's
//    first board is the opponent's (T7), so every board an action moves on is
//    one of the start's playable boards. PLAYED means moved on, from or onto.
//  * OPTIONAL MOVES - register M22's "optional moves played", as measured:
//    the moves whose source board was not mandatory at the action's start
//    (by its source: a jump that does not branch from an optional board onto
//    a mandatory one counts as optional, and satisfies the mandatory board,
//    T6 - counted apart too, optional_onto_mandatory). moves_from_mandatory
//    is the rest.
//  * A board's status can change during an action. Under the default rules
//    that happens only through a creation. When the present moves back (a
//    REWIND) - a branch that creates an active timeline behind it (T8), or a
//    creation that reactivates a timeline behind it (T12) - every mandatory
//    board left unplayed is RELEASED: it lies ahead of the new present. And a
//    reactivated board (T11) of the mover's colour that lies in the present -
//    the new one, or the present's own column - is GAINED: the mover must now
//    move on it (T12). The counts above go by the status at the action's
//    start; the moves whose source board's status had changed by the time
//    they were made are counted apart (moves_on_released, moves_on_gained),
//    and optional_when_made is the other reading of "optional": the moves
//    whose source board was not mandatory when the move was made.
//  * moves_after_shift: the moves made once the obligations were met - the
//    present on the opponent's colour (T5), by advancing it or by a rewind;
//    the moves ruling R8 allows until submission.
//
// Two consequences of T3, J4 and T7 are checked as each move is counted (a
// std::logic_error if one breaks - a fault of the engine, not of a record):
// every board played was playable at the action's start, and no board is
// played twice - so boards played = moves + jumps that do not branch.

#ifndef T5D_TOOLS_TURN_WIDTH_HPP
#define T5D_TOOLS_TURN_WIDTH_HPP

#include <cstddef>
#include <span>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/state.hpp"

namespace t5d::tools {

/// One action's counts (see the header comment for each definition).
struct TurnWidth {
  // -- at the action's start
  std::size_t boards = 0;
  int timelines = 0;
  int active = 0;
  int mandatory = 0;
  int playable = 0;
  int optional = 0;  // playable - mandatory

  // -- the action's moves
  int moves = 0;
  int moves_from_mandatory = 0;     // source board mandatory at the start
  int optional_moves = 0;           // source board not mandatory at the start
  int optional_when_made = 0;       // source board not mandatory when the move was made
  int boards_played = 0;            // the start's playable boards moved on, from or onto
  int mandatory_played = 0;         // ... of them mandatory at the start
  int optional_played = 0;          // ... of them optional at the start
  int jumps = 0;                    // moves between boards, branching or not ("travel")
  int branches = 0;                 // branching jumps (J3, J5)
  int jumps_between_mandatory = 0;  // non-branching jumps between two boards mandatory at start
  int optional_onto_mandatory = 0;  // non-branching jumps from an optional board onto a mandatory
  int moves_after_shift = 0;        // made with the obligations met (R8)

  // -- changes of status during the action
  int rewinds = 0;                  // moves after which the present lay earlier (T8, T12)
  int mandatory_released = 0;       // mandatory at the start, never played
  int mandatory_gained = 0;         // not mandatory at the start, mandatory after a move (T11, T12)
  int moves_on_released = 0;        // source mandatory at the start, not when the move was made
  int moves_on_gained = 0;          // source not mandatory at the start, mandatory when made

  friend bool operator==(const TurnWidth&, const TurnWidth&) noexcept = default;
};

/// Counts an action move by move. Construct it on the state at the action's
/// start - a turn start: no move of the action made yet, which the state
/// cannot show, so it is the caller's to ensure - then apply the action's
/// moves in order; width() gives the counts so far.
class TurnCounter {
 public:
  explicit TurnCounter(const GameState& turn_start);

  /// The state the moves so far lead to (not submitted).
  const GameState& state() const noexcept { return state_; }

  /// Applies `move` - a move generated for state() - and counts it. Throws
  /// what GameState::apply throws for a move that does not fit, leaving the
  /// state and the counts unchanged; std::logic_error if an invariant of the
  /// header comment breaks (then the counter is spent).
  void apply(const Move& move);

  /// The counts of the moves applied so far: the start's, the moves', and
  /// the boards played, released and gained.
  TurnWidth width() const;

 private:
  GameState state_;
  TurnWidth counts_;                // everything but the per-board totals
  std::vector<int> start_mandatory_;  // timelines, ascending
  std::vector<int> start_playable_;   // timelines, ascending
  std::vector<int> played_;           // timelines, ascending
  std::vector<int> gained_;           // timelines, ascending
};

/// The counts of a whole action: `moves` applied in order to `turn_start`
/// with a TurnCounter. The action is not submitted.
TurnWidth classify_action(const GameState& turn_start, std::span<const Move> moves);

}  // namespace t5d::tools

#endif  // T5D_TOOLS_TURN_WIDTH_HPP
