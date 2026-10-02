// t5d/movegen.hpp - pseudo-legal move generation (design spec §11.0, the
// movement part of the rules; rules reference sections 3-5).
//
// A move is PSEUDO-LEGAL when it obeys every movement rule - the piece's
// geometry (4.1-4.13), paths through boards (J7, J8), destinations (J2, L4,
// L6), castling's own conditions (S1-S5), en passant (S7), promotion (S8) -
// and the action's bookkeeping: it starts on a playable board of the mover's
// that has not been played yet this action (L5, T3) - after the action's
// obligations are met too, until submission (Q4, ruled R8), unless the ruleset
// is Q4's Guide-literal reading, which offers no move once they are met.
// Whether the action it belongs to leaves a royal piece capturable (C4) is NOT
// tested here: that is the check ruling's, at submission (R1; t5d/check.hpp).
//
// Movement in brief (Vec4 conventions in t5d/geometry.hpp):
//
//  * Sliders - rook, bishop, queen, princess, royal queen, unicorn, dragon -
//    advance one unit per moved axis per step, over boards of the mover's
//    colour (a T step is two half-turns, so opponent-coloured boards are never
//    visited, never counted and never block - L4). Every intermediate square
//    must be empty: any piece blocks, the moving piece's own earlier self
//    included (J7; Q19 as a parameter). A place where no board exists - void -
//    blocks too (J8; Q18 as a parameter). The slide ends on the first piece:
//    an enemy one is captured.
//  * The knight leaps (two steps along one axis, one along another) over
//    pieces and void alike; only the destination must exist (4.5, J8). The
//    king and the common king step once along any set of axes (4.4, 4.12).
//  * Pawns and brawns (4.6, 4.7): forward is +rank / -timeline for White,
//    -rank / +timeline for Black. A move is one step forward along exactly one
//    of those axes to an empty square, or - with the double-step right (Q15) -
//    two, through an empty intermediate square (on the in-between timeline's
//    board, for a timeline double step). Pawn captures: forward rank and +-1
//    file, or forward timeline and +-1 turn. Brawn captures: one forward axis
//    plus one sideways axis (8), and - Q14, the Guide's literal reading - the
//    two forward axes together. En passant only within a board (S7).
//    Promotion on reaching the last rank (S8; the ruleset's promotion set).
//  * Destinations exist and are of the mover's colour (J2, L4); a friendly
//    piece never is one. A move that lands on a history board branches (J3);
//    on another timeline's playable board it does not (J4) - unless, under the
//    Quick Rules' literal reading (earlier_playable_jump_branches), it lands
//    at an earlier turn. A turn-zero board is a history board (L5; Q8 as a
//    parameter); a history board of an inactive timeline is a destination
//    (Q20 as a parameter). A brawn's arrival on its last rank from another
//    board promotes (Q17 as a parameter).
//  * Castling (S1-S6): on one board, by an unmoved royal king with an
//    unmoved rook on its rank at least two squares away (S5), the squares
//    between empty; barred if the king's square or the square it crosses is
//    attacked by an enemy piece ON THE CASTLING BOARD (S2, the developer's
//    ruling; R6, register H40: attacked_on_board below).
//
// Order: playable boards by ascending timeline, pieces by ascending square,
// directions in table order, distances ascending, promotions in
// PromotionSet::kOrder - deterministic on every platform.

#ifndef T5D_MOVEGEN_HPP
#define T5D_MOVEGEN_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/move.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "t5d/types.hpp"

namespace t5d {

/// Every pseudo-legal move of the player to move, from every playable board
/// of the mover's colour. Clears `out` first; reusing one list across calls
/// avoids reallocating. Empty when no move may be made now (Q4's Guide-literal
/// reading, after the shift) or no playable board of the mover's is left.
void generate_moves(const GameState& state, MoveList& out);

/// A geometric relation of one of the mover's pieces (design spec §6, "Typed
/// edge biases"; register B11 - PROVISIONAL; t5d/expand.hpp builds the typed
/// relations on these). Found by the move generator's own geometry, so a
/// relation and a move never disagree about what a piece reaches:
///
///  * Defence: the target square holds a piece of the mover's that the
///    source piece's movement geometry reaches with only that piece in the
///    way - the capture the source would make if that piece were an enemy
///    (the same paths, J7, J8, and the same landing rules, Q8, Q20).
///  * XRay (sliders only): along a ray, the first piece's square - of either
///    colour - is passed as if empty, the ray continues under the same rules
///    and stops at the next piece, of either colour, which is the target -
///    recorded when a capture or defence of it would be (its board a
///    destination under the ruleset).
///
/// Neither reads what counts as an attack on a royal piece (check and pins:
/// check detection, stage E4's under the ruling, R1).
struct GeometricRelation {
  enum class Kind : std::uint8_t { Defence = 0, XRay = 1 };

  std::int32_t from_timeline = 0;
  std::int32_t from_half_turn = 0;
  std::int32_t to_timeline = 0;
  std::int32_t to_half_turn = 0;
  std::int8_t from_square = kNoSquare;
  std::int8_t to_square = kNoSquare;
  Kind kind = Kind::Defence;

  friend constexpr bool operator==(const GeometricRelation&,
                                   const GeometricRelation&) noexcept = default;
};

using RelationList = std::vector<GeometricRelation>;

/// Which geometric relations to collect.
struct RelationKinds {
  bool defence = false;
  bool xray = false;
};

/// generate_moves - the same moves, in the same order - and, in the same pass,
/// the geometric relations `kinds` asks for, of the same pieces (the mover's
/// pieces on the boards the moves start from; none when no move may be made
/// now). Clears both lists first. Order: the moves' (playable boards by
/// ascending timeline, pieces by ascending square, directions in table order).
void generate_moves(const GameState& state, MoveList& out, RelationList& relations,
                    RelationKinds kinds);

/// The pseudo-legal moves from timeline p's playable board, appended to `out`
/// (nothing if that board is not the mover's or no move may be made now).
void append_board_moves(const GameState& state, int timeline, MoveList& out);

/// The pseudo-legal moves of the piece on `square` of timeline p's playable
/// board, appended to `out` (nothing if it is not the mover's piece there).
void append_piece_moves(const GameState& state, int timeline, int square, MoveList& out);

/// A hook for the check ruling (§11.0 item 1: check is "implemented directly
/// by generating every move by the attacking colour that captures an enemy
/// royal"); this stage does not use it. The pseudo-legal moves of `colour`'s
/// pieces on timeline p's latest board, appended to `out` - provided that
/// board has `colour` to move - whatever the turn, the present or the state
/// of an action: a plain Multiverse, so the check ruling may also pass a
/// hypothetical one (check status's passed boards, R1: advanced without a
/// move). The optional-board castling option (castling_on_optional_boards) is
/// not applied, there being no action; castling never captures. Activity is not
/// part of a Multiverse: `inactive_timelines` (ascending; a GameState's
/// inactive_timelines()) is what inactive_history_is_destination = false
/// (Q20) reads. Setup is O(1), so calling it once per timeline is O(T) in all.
void append_moves_of(const Multiverse& multiverse, const RuleOptions& rules, Colour colour,
                     int timeline, MoveList& out,
                     std::span<const int> inactive_timelines = {});

/// Whether `move` is a pseudo-legal move of the state: one that
/// generate_moves would produce, field for field. Generates the moving
/// piece's moves only; the overload with `scratch` reuses a caller's list (no
/// allocation once it has grown).
bool is_pseudo_legal(const GameState& state, const Move& move);
bool is_pseudo_legal(const GameState& state, const Move& move, MoveList& scratch);

/// The number of pseudo-legal moves (generates them; into `scratch`, if
/// given, which then holds them).
std::size_t count_moves(const GameState& state);
std::size_t count_moves(const GameState& state, MoveList& scratch);

/// S2: whether `square` on `board` is attacked by a piece of colour `by`
/// standing ON THAT BOARD - by its movement within the board: pawn and brawn
/// captures, knight leaps, king and common king steps, rook / bishop / queen /
/// princess / royal queen lines (unicorns and dragons cannot move within one
/// board). Attacks through time or from other timelines are not considered:
/// this is the castling test of the developer's ruling, not check detection.
bool attacked_on_board(const Board& board, int square, Colour by);

}  // namespace t5d

#endif  // T5D_MOVEGEN_HPP
