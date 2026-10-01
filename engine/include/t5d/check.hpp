// t5d/check.hpp - check status and submission legality (design spec §11.0
// items 1-2; decision H36; rulebook docs/rules/README.md, ruling R1 and C1-C7,
// C13).
//
// Both rest on one question, asked of a multiverse: can a piece of the
// attacker's, standing on one of the attacker's PLAYABLE boards - the latest
// board of its timeline, with the attacker to move on it (L5) - capture a
// ROYAL piece of the other colour (a king or a royal queen; never a common
// king: 4.11-4.13, C1), by a pseudo-legal move (sections 3-5)? A royal on any
// board counts, history boards included (C2): a capture that would branch
// counts, even into a timeline that would be inactive (R1). Pins of the
// attacker's pieces do not matter: capturing a royal is all that is asked
// (C3, derived from R1's "could capture"). The answer is exactly the move
// generator's (t5d/movegen.hpp, append_moves_of): the same paths through
// boards (J7, J8; Q18, Q19), the same destinations (J2, L4; Q8, Q20), the
// same pawn and brawn captures (4.6, 4.7; Q14). Castling and en passant never
// capture a royal. The scan does not build move lists - it walks each piece's
// lines and stops at the first royal capture - and the tests hold it equal to
// the generator's royal captures on many random multiverses.
//
//  * CHECK STATUS (R1, H36; C6): the side to move S, at the start of its
//    turn, is in check iff S passes - a null move: the board is advanced
//    unchanged to a copy with the opponent to move - on every ACTIVE board IN
//    THE PRESENT (its mandatory boards under the default T6; optional boards
//    ahead of the present and boards on inactive timelines are NOT passed),
//    and then the opponent could capture one of S's royals from ANY of the
//    opponent's playable boards: the passed copies, and every other board on
//    which the opponent is to move, on an active or an inactive timeline. The
//    passed boards are really there - a slide may need to cross them (J7, J8).
//    A side with no royal piece is never in check (C13).
//  * SUBMISSION LEGALITY (R1, H36; C4): a separate test, on the ACTUAL turn -
//    in the position after the mover's moves, no piece on any of the
//    opponent's playable boards (active or inactive timelines; a board on
//    which the mover is still to move is not the opponent's) can capture any
//    of the mover's royals. No pass. Moving into check during the turn is
//    allowed: only the submitted position is tested. GameState::submit and
//    submission_status apply it by default (standard_check_ruling).

#ifndef T5D_CHECK_HPP
#define T5D_CHECK_HPP

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "t5d/geometry.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "t5d/types.hpp"

namespace t5d {

/// One capture of a royal piece: the move the generator would produce, and
/// the line it follows (what the capture's existence rests on).
struct RoyalCapture {
  /// The capturing piece, on a playable board of its owner's.
  Piece attacker;
  int from_timeline = 0;
  int from_half_turn = 0;
  int from_square = kNoSquare;
  /// The royal piece captured, on a board of the attacker's colour (L4).
  Piece royal;
  int to_timeline = 0;
  int to_half_turn = 0;
  int to_square = kNoSquare;
  /// The displacement of one step (dT in whole turns, t5d/geometry.hpp) and
  /// the number of steps: a slide crosses steps - 1 places before the
  /// royal's; a leap, a king's step and a pawn's or brawn's capture are one
  /// step and cross none.
  Vec4 step{};
  int steps = 1;
  /// Whether the royal stands on a history board, so the capture would branch
  /// (J3; under earlier_playable_jump_branches also a capture at an earlier
  /// turn onto a playable board - the generator's branching flag).
  bool branching = false;

  friend bool operator==(const RoyalCapture&, const RoyalCapture&) noexcept = default;
};

/// One place a capture rests on: a square of the board at (timeline,
/// half_turn) - or, under sliders_cross_void (Q18), a place with no board,
/// which a slide crosses.
struct CapturePlace {
  int timeline = 0;
  int half_turn = 0;
  int square = kNoSquare;

  friend bool operator==(const CapturePlace&, const CapturePlace&) noexcept = default;
};

/// The places `capture` rests on, in order along its line: the attacker's
/// square, every square crossed (empty - or, under past_self_blocks = false,
/// holding the attacker's own earlier self, Q19 - or void under
/// sliders_cross_void), and the royal's square. The capture exists in any
/// multiverse in which those places hold the same pieces (and the same
/// places are void), the attacker's board is still one of its owner's
/// playable boards, and the royal's board is still a history board or still
/// a playable one as before (that decides `branching`, and under the
/// non-default Q8 and Q20 options whether it may be landed on).
std::vector<CapturePlace> capture_places(const RoyalCapture& capture);

/// Whether a piece of `attacker` on one of its playable boards can capture a
/// royal piece of the other colour, by the move generator's rules. Activity is
/// not part of a Multiverse: `inactive_timelines` (ascending; a GameState's
/// inactive_timelines()) is read only under inactive_history_is_destination
/// = false (Q20). Stops at the first capture found.
bool royal_capture_exists(const Multiverse& multiverse, const RuleOptions& rules, Colour attacker,
                          std::span<const int> inactive_timelines = {});

/// Calls `visit` with every such capture, in the generator's order (playable
/// boards by ascending timeline, pieces by ascending square, directions in
/// table order); stops as soon as `visit` returns true. Returns whether it
/// stopped early.
bool for_each_royal_capture(const Multiverse& multiverse, const RuleOptions& rules,
                            Colour attacker, std::span<const int> inactive_timelines,
                            const std::function<bool(const RoyalCapture&)>& visit);

/// The first such capture, or nullopt.
std::optional<RoyalCapture> find_royal_capture(const Multiverse& multiverse,
                                               const RuleOptions& rules, Colour attacker,
                                               std::span<const int> inactive_timelines = {});

// ---------------------------------------------------------------------------
// Check status (R1, C6)
// ---------------------------------------------------------------------------

/// The boards check status passes (R1): the timelines whose playable board is
/// the mover's, ACTIVE, and IN THE PRESENT (its half-turn is the present's;
/// a timeline deferred under Q6's alternative is not in the present).
/// Ascending. Meaningful at the start of a turn, when these are the mover's
/// mandatory boards under the default T6 - whatever Q5's option says.
std::vector<int> passed_timelines(const GameState& turn_start);

/// The hypothetical multiverse of check status: every passed timeline
/// advanced by a copy of its playable board with the opponent to move - the
/// same pieces, unmoved flags kept (nothing moved), no en-passant square (no
/// double step was made).
Multiverse passed_multiverse(const GameState& turn_start);

/// R1 (H36): whether the player to move is in check at the start of its
/// turn. False for a side with no royal piece (C13).
bool in_check(const GameState& turn_start);

/// A royal capture that puts the player to move in check (in the passed
/// multiverse), if any - for messages and tests.
std::optional<RoyalCapture> check_witness(const GameState& turn_start);

// ---------------------------------------------------------------------------
// Submission legality (R1, C4)
// ---------------------------------------------------------------------------

/// R1 (H36), C4: in `after_action` - the position the mover's moves reached,
/// before submission - whether no piece on any of the opponent's playable
/// boards can capture any of the mover's royal pieces. (The present-shift
/// half of submission, T5, is GameState::obligations_met.)
bool royals_safe(const GameState& after_action);

/// A royal capture that makes the submission illegal, if any.
std::optional<RoyalCapture> submission_witness(const GameState& after_action);

/// "the rook on (0T3)e5 can capture the king on (0T2)e5 (branching)" - for
/// messages.
std::string describe(const RoyalCapture& capture, bool even);

}  // namespace t5d

#endif  // T5D_CHECK_HPP
