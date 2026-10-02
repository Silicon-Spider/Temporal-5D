// t5d/state.hpp - the game state machine: active timelines, the present, the
// action in progress, move application and submission (design spec §11.0
// item 5; rules reference T1-T13, J4-J6, L5, L7).
//
// A GameState is a Multiverse (t5d/multiverse.hpp) plus the rules it is played
// under (t5d/rules.hpp) plus the little that the boards do not record:
//
//  * ACTIVITY (T10-T13). Initial timelines are always active (T10, T13). Under
//    the default rules the rest follows from the timelines' creators: the n-th
//    timeline a player created is active iff the opponent has created at least
//    n-1 (the count rule, kept true by T11's reactivation: each creation
//    activates the opponent's earliest-created inactive timeline, exactly
//    one). The state keeps the set of inactive timelines (usually empty) and
//    updates it at every creation; under the alternative readings (Q1
//    Consecutive, Q3 play activation, Q6 Deferred) activity depends on the
//    history of play, and the set - with the deferred timelines and, under
//    Q1 Consecutive, the last creator - is part of the state and of its key.
//  * THE PRESENT (T4): the half-turn column of the earliest playable board
//    among the active timelines (inactive ones never affect it). It is
//    brought up to date after every move: a move can only advance a timeline,
//    but a branch can create an active timeline behind it (T8, rewinding by
//    branching) and a creation can reactivate one behind it (T12, rewinding by
//    reactivation). Kept incrementally - O(1) per move, and one pass over the
//    timelines only when the last board in the present's column is played.
//  * THE ACTION IN PROGRESS (T1-T7). The player to move (the multiverse's
//    to_move) makes moves one at a time. A board is playable iff it is the
//    latest of its timeline (L5); the mover may move on every playable board
//    of their colour, active or not, in the present or not (T6), and each
//    board takes one move, because it then has a successor on the opponent's
//    colour (T3) - so "which boards the mover has played" is read off the
//    boards themselves. The MANDATORY boards are the active boards in the
//    present on the mover's colour (T6; Q5 AllActive: every active board of
//    the mover's). The action's OBLIGATIONS are met when none is left - then
//    the present lies on the opponent's colour (T5). The mover may still
//    move on their remaining playable boards after that, until they submit
//    (Q4, ruled R8; register D8): a state whose obligations are met offers
//    submission alongside any further move. Under Q4's Guide-literal
//    alternative (RuleOptions::moves_after_shift = false) no move may be made
//    once the obligations are met; the action is then submitted.
//  * SUBMISSION (T5, C4): allowed once the obligations are met, and - C4 - only
//    if no royal piece of the mover's is left capturable. That test is the
//    check ruling's (R1, H36; t5d/check.hpp), reached through a hook
//    (CheckRuling::royals_safe): submission_status() and submit() apply the
//    engine's ruling by default (standard_check_ruling) and say Safe or
//    Unsafe. CheckRuling::none() leaves royal safety untested (Unchecked):
//    pseudo-legal play, for the movement rules' own tests and parse mode.
//
// Like a Multiverse, a GameState is a value: copies share every board and
// every timeline they have not edited, and apply() and submit() change only
// the state they are called on.

#ifndef T5D_STATE_HPP
#define T5D_STATE_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/move.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/rules.hpp"
#include "t5d/types.hpp"

namespace t5d {

class GameState;

// ---------------------------------------------------------------------------
// The check ruling's insertion points
// ---------------------------------------------------------------------------

/// A three-valued answer: the hooks below may not be installed.
enum class Verdict : std::uint8_t { Unknown = 0, Yes, No };

/// The check ruling's hooks (spec §11.0 items 1-3; rulebook R1, C1-C9, C13;
/// register H36, H37). The engine's own ruling is standard_check_ruling()
/// (below; t5d/check.hpp and t5d/search.hpp implement it); a default-
/// constructed CheckRuling - CheckRuling::none() - has no hook. A null hook
/// means "not tested": every function that consults one answers
/// Unknown/Unchecked rather than guess.
struct CheckRuling {
  /// C4: in `after_action` - the state reached by the mover's moves, with
  /// the obligations met and before submission - can no enemy attacker
  /// capture any of the mover's royal pieces? Yes = all safe.
  Verdict (*royals_safe)(const GameState& after_action) = nullptr;
  /// C6: is the player to move at the start of their turn in check - the
  /// status that separates checkmate from stalemate?
  Verdict (*in_check)(const GameState& turn_start) = nullptr;
  /// C5: does the player to move have a legal action (a sequence of moves
  /// that shifts the present and leaves every royal piece safe)?
  Verdict (*has_legal_action)(const GameState& turn_start) = nullptr;

  /// No hook: pseudo-legal play - submission tests the present's shift (T5)
  /// alone, and outcome() answers Unknown.
  static constexpr CheckRuling none() noexcept { return CheckRuling{}; }
};

/// The engine's check ruling (H36, H37): royals_safe is C4's submission test
/// and in_check R1's check status (t5d/check.hpp); has_legal_action is C5's,
/// an exhaustive search for a legal action (t5d/search.hpp: it never answers
/// Unknown, and can take as long as the position needs). GameState's
/// submission uses it unless told otherwise.
CheckRuling standard_check_ruling() noexcept;

/// Whether the action in progress may be submitted.
enum class SubmitStatus : std::uint8_t {
  /// T5: the present has not shifted to the opponent (or, Q5 AllActive, an
  /// active board of the mover's is unplayed). Not submittable.
  ObligationsUnmet = 0,
  /// The obligations are met; royal safety (C4) was NOT tested - the ruling
  /// given has no royals_safe hook (CheckRuling::none()), or it answered
  /// Unknown. Submittable, pseudo-legally.
  Unchecked,
  /// The obligations are met and the installed test found every royal safe.
  Safe,
  /// The obligations are met but the installed test found a royal piece
  /// capturable (C4). Not submittable.
  Unsafe,
};

/// The game's result at the start of a turn (C5), from the check ruling's
/// hooks; Unknown whenever a hook it needs is missing or undecided.
enum class Outcome : std::uint8_t { Unknown = 0, Ongoing, Checkmate, Stalemate };

// ---------------------------------------------------------------------------
// The state
// ---------------------------------------------------------------------------

class GameState {
 public:
  /// A state at the start of `multiverse.to_move()`'s turn, under `rules`.
  /// Validates the rules (RuleOptions::validate) and the multiverse: it must
  /// hold the central timeline(s) - "0", or "-0" and "+0" - and the present
  /// must lie on the colour of the player to move (T5: every turn begins so).
  /// Activity is the count rule's (with Q2's credit if chosen): 5DFEN records
  /// none, and under the default rules none is needed. The multiverse's
  /// ruleset ID is set to rules.ruleset_id(). Throws std::invalid_argument.
  explicit GameState(Multiverse multiverse, const RuleOptions& rules = {});

  const Multiverse& multiverse() const noexcept { return mv_; }
  const RuleOptions& rules() const noexcept { return rules_; }
  /// The player whose turn it is - whose action is in progress.
  Colour to_move() const noexcept { return mv_.to_move(); }

  // -- activity (T10-T13) ---------------------------------------------------

  /// Whether timeline p is active. False for a position with no timeline.
  bool active(int p) const noexcept;
  /// The inactive timelines, ascending. Usually empty.
  const std::vector<int>& inactive_timelines() const noexcept { return inactive_; }
  /// Q6 Deferred only: the timelines left out of the present until the given
  /// colour's next turn (see Reactivation::Deferred). Always empty otherwise.
  struct Deferral {
    int timeline = 0;
    Colour owner = Colour::White;
    friend bool operator==(const Deferral&, const Deferral&) noexcept = default;
  };
  const std::vector<Deferral>& deferred_timelines() const noexcept { return deferred_; }
  /// Who made the most recent creation in play, if known (a position read
  /// from 5DFEN records none). Only Q1's Consecutive reading reads it, so only
  /// under that rule is it part of the state - of equality and of the key.
  std::optional<Colour> last_creator() const noexcept { return last_creator_; }

  // -- the present (T4) -----------------------------------------------------

  /// The present: the half-turn column of the earliest playable board among
  /// the active timelines (less any deferred ones, Q6).
  int present() const noexcept { return present_; }
  /// The colour to move on the present's column.
  Colour present_colour() const noexcept { return to_move_at(present_); }
  /// T5: the present lies on the opponent's colour.
  bool present_shifted() const noexcept { return present_colour() != to_move(); }

  // -- the action in progress (T1-T7) ------------------------------------------

  /// L5: timeline p's playable board is the mover's (its latest board has the
  /// mover to move). The mover may move from exactly these boards.
  bool playable_by_mover(int p) const noexcept;
  /// T6 (Q5): the mover must still play timeline p's playable board.
  bool mandatory(int p) const noexcept;
  /// The mandatory boards left, by timeline, ascending.
  std::vector<int> mandatory_timelines() const;
  int mandatory_count() const noexcept { return mandatory_count_; }
  /// The playable boards of the mover, by timeline, ascending.
  std::vector<int> playable_timelines() const;
  /// T5: the action's obligations are met - no mandatory board is left (so
  /// the present lies on the opponent's colour).
  bool obligations_met() const noexcept { return mandatory_count_ == 0; }
  /// Q4: whether the mover may make another move now - always, by default
  /// (R8: until submission; a move still needs a playable board of the
  /// mover's); under the Guide-literal alternative (moves_after_shift =
  /// false) only while the obligations are unmet.
  bool may_move() const noexcept { return rules_.moves_after_shift || !obligations_met(); }

  // -- moves -----------------------------------------------------------------

  /// Applies one move of the mover's - a move generated for this state
  /// (t5d/movegen.hpp). It checks the move's structure against the state -
  /// the source on a playable board of the mover's, the piece there, the
  /// destination board existing, of the mover's colour and a destination
  /// under the ruleset (Q8, Q20), the branching flag agreeing with it (J3),
  /// the captured piece, the promotion (S8, Q17), and for the special moves
  /// everything their effects rest on: castling's king, rook and squares
  /// (S1), en passant's square and victim (S7), and a double step's right,
  /// direction and empty squares (4.6, Q15) - so a malformed move cannot
  /// corrupt the state, and throws std::invalid_argument (std::logic_error if
  /// no move may be made now - Q4's Guide-literal alternative, after the
  /// shift) leaving the state unchanged. It does not
  /// re-derive the geometry of ordinary moves: for an arbitrary move use
  /// is_pseudo_legal (t5d/movegen.hpp) first.
  ///
  /// Effects (T2, T3, J4, J5): the source timeline gets the source board
  /// without the piece; a physical move puts the piece on that same new
  /// board; a non-branching jump appends the destination board with the piece
  /// to the destination timeline; a branching jump creates a new timeline
  /// (White's at the next position above every timeline, Black's below,
  /// L7) whose only board is the destination's half-step successor - a copy
  /// with every piece of the destination board, duplicates included (J5).
  /// The moving piece loses its unmoved flag (so do both castling pieces) -
  /// unless rules().jumps_keep_unmoved and it is a pawn or brawn that left its
  /// board (Q15). Every new board has no en-passant square, except the one a
  /// pawn's (or brawn's) double step along its rank creates, when the piece
  /// that arrives is still a pawn (or brawn) and an enemy pawn (or brawn)
  /// stands beside its arrival square to take it (S7). Then activity (T10,
  /// T11, Q1-Q3, Q6) and the present (T4) are brought up to date. (Should
  /// memory run out midway through a move, the state is left unspecified:
  /// discard it.)
  void apply(const Move& move);

  // -- submission (T5, C4) ------------------------------------------------------

  /// Whether the action may be submitted; consults ruling.royals_safe once the
  /// obligations are met - by default the engine's (C4, R1): Safe or Unsafe.
  SubmitStatus submission_status(const CheckRuling& ruling = standard_check_ruling()) const;
  /// Ends the action and begins the opponent's turn. Throws std::logic_error,
  /// leaving the state unchanged, if the status is ObligationsUnmet or Unsafe.
  /// Unchecked is accepted (pass CheckRuling::none() to submit pseudo-legal
  /// play): see SubmitStatus.
  void submit(const CheckRuling& ruling = standard_check_ruling());

  // -- key -----------------------------------------------------------------------

  /// The state's 5D Zobrist key: the multiverse's (which carries the ruleset
  /// ID) - and, under rules whose activity depends on the history of play,
  /// the inactive set, the deferred timelines and (Q1 Consecutive) the last
  /// creator too.
  std::uint64_t zobrist() const noexcept;

  /// Same rules, same multiverse, same activity state (the last creator
  /// compared only under Q1 Consecutive, the one rule that reads it).
  friend bool operator==(const GameState& a, const GameState& b) noexcept;

 private:
  bool counts_for_present(int p) const noexcept;
  bool deferred(int p) const noexcept;
  void set_inactive(int p, bool inactive);
  int credited_count(Colour c) const noexcept;
  int credited_order(Colour creator, int order) const noexcept;
  void on_created(int p, Colour creator);
  void on_played(int p, Colour mover);

  // The present's bookkeeping (T4, T6), kept incrementally. A timeline
  // "counts" when it is active and not deferred (Q6); counting never stops
  // once it has begun (nothing deactivates a timeline). present_ is the
  // earliest head among the counting timelines, at_present_ how many of
  // them have their head there - 0 only while a move has just played the
  // last of them, until settle() rescans - and heads_ how many counting
  // heads have each colour to move.
  void count_in(int p) noexcept;        // timeline p starts to count
  void head_added(int h) noexcept;      // a counting head appears at half-turn h
  void head_removed(int h) noexcept;    // a counting head at h is superseded
  void rescan() noexcept;               // present_, at_present_, heads_ from scratch
  void settle() noexcept;               // rescan if needed; the mandatory count

  Multiverse mv_;
  RuleOptions rules_;
  std::vector<int> inactive_;            // sorted
  std::vector<Deferral> deferred_;       // sorted by timeline
  std::optional<Colour> last_creator_;   // Q1 Consecutive
  int present_ = 0;
  int at_present_ = 0;
  std::array<int, 2> heads_{};
  int mandatory_count_ = 0;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// The present a multiverse would have under `rules` at the start of a turn,
/// with the count rule's activity (no deferrals) - what GameState computes on
/// construction. Requires a non-empty multiverse.
int initial_present(const Multiverse& multiverse, const RuleOptions& rules = {});

/// A state from 5DFEN board strings (t5d/fen.hpp, multiverse_from_fen). The
/// player to move is `to_move` when given; otherwise the owner of the earliest
/// active playable board - the colour to move on the present (T4, Q9: no
/// source says who moves first in a custom position; this is the inferred
/// answer). Throws NotationError for malformed strings and
/// std::invalid_argument as GameState's constructor does.
GameState game_from_fen(std::span<const std::string_view> boards, BoardSize size, Parity parity,
                        const RuleOptions& rules = {},
                        std::optional<Colour> to_move = std::nullopt);
GameState game_from_fen(const std::vector<std::string>& boards, BoardSize size, Parity parity,
                        const RuleOptions& rules = {},
                        std::optional<Colour> to_move = std::nullopt);

/// C5 from the hooks: has_legal_action Yes -> Ongoing; No -> in_check Yes ->
/// Checkmate, No -> Stalemate; Unknown wherever a hook is missing or answers
/// Unknown. Meaningful at the start of a turn. Implements no rule itself
/// (the engine's verdict with its search statistics: t5d/search.hpp).
Outcome outcome(const GameState& turn_start, const CheckRuling& ruling);

}  // namespace t5d

#endif  // T5D_STATE_HPP
