// t5d/rules.hpp - the ruleset: every point the rules leave open, as a named
// parameter with a documented default.
//
// The rules engine implements the project's rules reference
// (docs/rules/README.md; rule IDs L*, T*, J*, 4.*, S*, C*, Q*). Where that
// reference marks a point as an open question (Q1, Q2, ...) or a conflict, the
// engine implements the high-confidence reading - or the in-game Guide's (G) -
// as the DEFAULT, and the alternative reading as a parameter here. Where the
// design spec (docs/design/5d-chess-transformer-design.md) and the rules
// reference disagree, the reference is the default and the spec's reading a
// parameter where feasible. The engine README lists every parameter.
//
// The parameters ARE the rules, so they are folded into every exact-store key
// (spec §11.3, decision I9): RuleOptions::ruleset_id() is 0 for the defaults -
// the standard rules, which position and structure variants share (§12.6) -
// and a hash of the canonical encoding otherwise, and GameState stamps it on
// its multiverse (t5d/state.hpp), so the Zobrist key carries it.
//
// No parameter here concerns check: the check ruling (rulebook R1, register
// H36; t5d/check.hpp) closed Q10, and its definitions are the rules. Castling's
// attack test is per-board (S2; R6, register H40), so the design spec's former
// cross-board reading is no option either.

#ifndef T5D_RULES_HPP
#define T5D_RULES_HPP

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "t5d/types.hpp"

namespace t5d {

/// T10 / Q1: how a newly created timeline's activity is decided.
enum class ActivationRule : std::uint8_t {
  /// Default (T10, high). The count rule: the n-th timeline a player creates
  /// (counting inactive ones) is active iff the opponent has, at that moment,
  /// created at least n-1. With T11's reactivation this keeps "the n-th
  /// timeline of a player is active iff the opponent has created at least n-1"
  /// true at every moment.
  Count = 0,
  /// Q1's "consecutive" reading: a creation is inactive when the same player
  /// made the previous creation ("one player may not time travel twice before
  /// their opponent time travels once"). T11's reactivation still applies.
  /// Activity then depends on the order of past creations, which neither the
  /// boards nor 5DFEN record: positions read from 5DFEN start from the count
  /// rule's answer, with no previous creation.
  Consecutive = 1,
};

/// Q2: how initial timelines count in the activation rule.
enum class InitialTimelineCredit : std::uint8_t {
  /// Default: initial timelines count for neither side - the count rule
  /// counts created timelines. (Any credit given to both sides alike changes
  /// nothing, so only an asymmetric reading is an alternative.)
  None = 0,
  /// Q2's alternative, from the 5dpgn author's unverified remark "Internally
  /// in the game, -0 is -1": in a game that starts with an even number of
  /// timelines, Black is counted as having created one timeline already (the
  /// "-0"), which gives White the branching priority.
  MinusZeroToBlack = 1,
};

/// T6 / Q5: which boards the mover must play before submitting.
enum class MandatoryBoards : std::uint8_t {
  /// Default (T6, high; G): only the active boards in the present.
  InPresent = 0,
  /// Q5's alternative (Wikipedia's former paragraph): every active timeline
  /// whose playable board is the mover's must be played each turn.
  AllActive = 1,
};

/// T12 / Q6: a creation that reactivates a timeline whose playable board is
/// the mover's.
enum class Reactivation : std::uint8_t {
  /// Default (T4, T5, high; T12, low): the present is always the earliest
  /// playable board of the active timelines, so a reactivated board behind the
  /// present rewinds it, and if that board is the mover's, the mover must play
  /// it (or rewind further) before submitting.
  Rewinds = 0,
  /// Q6's alternative ("must the mover then play the reactivated board?" -
  /// no). The rules reference gives no mechanism; this is the engine's
  /// formalisation: a timeline that the mover's own creation reactivates while
  /// its playable board is the mover's is left out of the present until the
  /// mover's next turn begins (or until the mover plays it), and only then
  /// obliges the mover.
  Deferred = 1,
};

/// 4.6 / Q15: what entitles a pawn or brawn to its double step.
enum class DoubleStepRight : std::uint8_t {
  /// Default: the piece's own "unmoved" flag (5DFEN's '*'; 4.6 "a pawn that
  /// has not moved"). The flag is cleared on any piece that moves.
  UnmovedFlag = 0,
  /// Q15's alternative: the rank, not the piece - a pawn or brawn on its
  /// colour's second rank (white rank 2, black rank H-1) may double-step.
  HomeRank = 1,
};

/// Design spec §12.6 / §11.3 (I9): the rule family a ruleset belongs to. A
/// RULE variant (§12.6's third tier) gets a ruleset ID of its own, so that
/// none of its exact-store facts is ever served for another variant's
/// positions ("a fact proved under standard rules is simply false under
/// Reversed Royalty", §11.3).
enum class RuleFamily : std::uint8_t {
  /// Default: the standard rules, which every position and structure variant
  /// shares (§12.6) - and the rules of every custom (5DFEN) position.
  Standard = 0,
  /// Standard - Reversed Royalty (rules reference V3; 4.11, 4.12, S4). Its
  /// rules are the standard rules applied to its pieces: royalty is carried by
  /// the pieces themselves - the royal queen (Y) is royal, the common king (C)
  /// is not, and neither castles (S4) - so no movement rule reads this field.
  /// It exists for the ruleset ID alone (§12.6 lists Reversed Royalty as a
  /// rule variant).
  ReversedRoyalty = 1,
};

/// The pieces a pawn (or brawn) may promote to: a set of kinds.
class PromotionSet {
 public:
  /// The queen alone (S8: the official game promotes to a queen
  /// automatically and offers no other piece).
  constexpr PromotionSet() noexcept = default;

  static constexpr PromotionSet of(std::initializer_list<PieceKind> kinds) noexcept {
    PromotionSet set;
    set.mask_ = 0;
    for (const PieceKind k : kinds) set.mask_ |= bit(k);
    return set;
  }
  static constexpr PromotionSet from_mask(std::uint16_t mask) noexcept {
    PromotionSet set;
    set.mask_ = mask;
    return set;
  }

  constexpr bool contains(PieceKind k) const noexcept { return (mask_ & bit(k)) != 0; }
  constexpr bool empty() const noexcept { return mask_ == 0; }
  constexpr std::uint16_t mask() const noexcept { return mask_; }
  constexpr void insert(PieceKind k) noexcept { mask_ |= bit(k); }

  /// The order in which promotions are generated (a fixed order; the 5dpgn
  /// header's own order, "from most important to least important", is a
  /// matter of presentation, not of legality).
  static constexpr std::array<PieceKind, 10> kOrder = {
      PieceKind::Queen,    PieceKind::Rook,    PieceKind::Bishop, PieceKind::Knight,
      PieceKind::Princess, PieceKind::Unicorn, PieceKind::Dragon, PieceKind::RoyalQueen,
      PieceKind::CommonKing, PieceKind::King};

  friend constexpr bool operator==(PromotionSet, PromotionSet) noexcept = default;

 private:
  static constexpr std::uint16_t bit(PieceKind k) noexcept {
    return static_cast<std::uint16_t>(1u << index_of(k));
  }
  std::uint16_t mask_ = static_cast<std::uint16_t>(1u << index_of(PieceKind::Queen));
};

/// Every rules parameter. Trivially copyable and small: a GameState holds one.
struct RuleOptions {
  // ---- §2 turns, the present, active timelines -------------------------------

  /// T10 / Q1.
  ActivationRule activation = ActivationRule::Count;
  /// T10, T13 / Q2.
  InitialTimelineCredit initial_credit = InitialTimelineCredit::None;
  /// T11 / Q3: does a move by the player who did NOT create an inactive
  /// timeline, on or onto its playable board, activate it? Default false (the
  /// count rule implies neither play activation nor deactivation; T10, T11).
  /// True is Q3's "yes" reading ("if the other player makes a play on them, it
  /// will also become active"). Deactivation has no described mechanism and is
  /// not offered.
  bool play_activates = false;
  /// T5 / Q4: may the mover keep moving once the action's obligations are met
  /// (the present has shifted to the opponent) and before submitting? Default
  /// false (G: "A player makes moves while it is their turn in The Present").
  bool moves_after_shift = false;
  /// T6 / Q5.
  MandatoryBoards mandatory = MandatoryBoards::InPresent;
  /// T12 / Q6.
  Reactivation reactivation = Reactivation::Rewinds;
  /// L5, J3 / Q8: may a piece land on a turn-zero board that is not playable
  /// (Standard - Turn Zero's T0 board)? Default true: by L5 (high) every board
  /// but the latest of its timeline is a history board, so a move onto it
  /// branches - only Black's pieces can get there. False is the reading that a
  /// T0 board, on which no move was ever made, is no history board
  /// ("a board where a move has already occured", J3) and so no destination.
  bool turn_zero_reachable = true;

  // ---- §3-4 movement -------------------------------------------------------------

  /// J8 / Q18: may a sliding piece pass places where no board exists (void)?
  /// Default false (J8: void blocks every piece but the knight, "just like it
  /// would be blocked by another piece"). True is Dark MAvL's dissent; a slide
  /// still never ends on void (J2).
  bool sliders_cross_void = false;
  /// J7 / Q19: does a sliding piece's own earlier self block it? Default true
  /// (J7: any piece blocks). False is the 2022 guide's reading, formalised
  /// narrowly because pieces have no identity here: on a slide straight back
  /// in time along the piece's own timeline and square, a square holding a
  /// piece of the same kind and colour does not block.
  bool past_self_blocks = true;
  /// J3 (a conflict) / Q20: does a move that lands at an EARLIER turn than it
  /// started branch even when its destination is another timeline's playable
  /// board? Default false (J3, high: a move between boards branches iff its
  /// destination is a history board, "a board where a move has already
  /// occured"; G-old: "If a piece is moved to a board where a move has not yet
  /// been played ... no branch occurs"). True is the Quick Rules' literal
  /// "Moving a piece to the past causes a new timeline to be created": such a
  /// move always branches, by J5's mechanism - the destination board D is left
  /// unchanged (so it stays playable, and is not played by the move), and a new
  /// timeline starts with D's half-step successor, a copy of D with the piece.
  bool earlier_playable_jump_branches = false;
  /// Q20: may a piece land on a history board of an INACTIVE timeline (so
  /// branching from it, J3)? Default true: J3's criterion makes no exception
  /// for inactive timelines ("presumed, never stated", Q20). False: such a
  /// board is no destination; slides still pass over it, or are blocked on
  /// it, as on any board.
  bool inactive_history_is_destination = true;

  // ---- 4.6-4.7 pawns and brawns, S7-S8 ------------------------------------------

  /// 4.7 / Q14: does a brawn also capture one step forward on rank and on
  /// timeline at once (white (0,+1,-1,0))? Default true: the Guide's literal
  /// "any diagonal that includes a forward direction and does not include any
  /// backward direction". False is Perseus's reading (8 captures).
  bool brawn_rank_timeline_capture = true;
  /// 4.6 / Q15.
  DoubleStepRight double_step_right = DoubleStepRight::UnmovedFlag;
  /// 4.6 / Q15: may a pawn double-step along the timeline axis (two
  /// timelines)? Default true (G: "one or two squares"); false is the 2022
  /// guide's "can also only move one board at a time".
  bool timeline_double_step = true;
  /// 4.6, J8 / Q15: may a timeline double step pass a timeline that has no
  /// board at that turn? Default false (void blocks, J8).
  bool double_step_over_void = false;
  /// J5 / Q15, Q21: do the pieces of a board copied into a new timeline keep
  /// their unmoved flags? Default true (J5: the copy keeps every piece of the
  /// board; the 5dpgn example games, a low lead, agree). False clears every
  /// flag on the copy.
  bool branch_copies_keep_unmoved = true;
  /// 4.6 / Q15: does a pawn or brawn that moves BETWEEN boards keep its
  /// unmoved flag (and so its double step)? Default false: every move clears
  /// the moving piece's flag ("a pawn that has not moved"). True is Q15's
  /// "a pawn that has moved only across timelines keeps the right".
  bool jumps_keep_unmoved = false;
  /// S7 / Q16: do brawns take part in en passant, capturing and being
  /// captured? Default true: the only source (wiki.gg) says so, and the brawn
  /// is "an alternate interpretation of the pawn" (G). False: pawns only.
  bool en_passant_brawns = true;
  /// S8, 4.7 / Q17: does a brawn promote? Default true (medium: it has the
  /// pawn's promotion ability), to the same set as a pawn.
  bool brawns_promote = true;
  /// Q17: does a brawn that arrives on its last rank of ANOTHER board - by a
  /// capture forward in rank and back or forward in time, or (Q14) forward in
  /// rank and timeline at once - promote? Default true (S8's general
  /// statement: a pawn that reaches its last rank promotes, wherever the move
  /// began). False: a brawn promotes only on reaching its last rank by a move
  /// within its own board. (Pawns are unaffected: no move of a pawn between
  /// boards changes its rank.)
  bool brawn_cross_board_promotion = true;
  /// S8: the promotion pieces. Default the queen alone (the base game). A
  /// game's 5dpgn "Promotions" header - a client extension, not the base game -
  /// may give others. Never a pawn or a brawn.
  PromotionSet promotions{};

  // ---- S1-S6 castling --------------------------------------------------------------
  //
  // Only a royal king castles (S4) - never a royal queen, and never a common
  // king, whose unmoved state 5DFEN cannot even record ("5DFEN Strictness"
  // tracks pawns, rooks, kings and brawns), so S4's alternative is not offered.
  // Castling is barred only by attacks from enemy pieces ON THE CASTLING BOARD
  // (S2, the developer's ruling; R6, register H40): no option.

  /// S2 / Q21: must every square between the king and the rook be empty?
  /// Default true (as in ordinary chess, medium). False: only the two
  /// destination squares must be free.
  bool castling_empty_between = true;
  /// S3 / Q21: must the king's destination square also be free of attack (by
  /// an enemy piece on the castling board)? Default false (S3: the ruling names
  /// only the king's square and the square it crosses; the turn-level royal
  /// safety test, C4, covers the destination).
  bool castling_destination_test = false;
  /// Q21: must the squares the rook crosses be free of attack (on the board)?
  /// Default false (S2 names only the king's squares).
  bool castling_rook_path_test = false;
  /// S5: may the king and rook castle on any rank? Default true (S1 does not
  /// restrict the rank; S5's note: "the pieces need not be on the player's
  /// first rank"). False: only on the castling player's back rank.
  bool castling_any_rank = true;
  /// S5 / Q21: may the king castle with a rook exactly two squares away, the
  /// king landing on the rook's square and the rook on the square the king
  /// crossed? Default true (S5: "a rook ... that is at least two squares
  /// away"; the landing square is inferred, low). False: the rook must be at
  /// least three squares away.
  bool castling_rook_two_away = true;
  /// S1 / Q21: may the mover castle on an optional board (one that is not
  /// mandatory, T6)? Default true (S1: "On a playable board").
  bool castling_on_optional_boards = true;

  // ---- §12.6 the rule family -----------------------------------------------------

  /// §12.6, §11.3 (I9): the rule family (see RuleFamily). Read by no rule;
  /// part of the ruleset ID.
  RuleFamily family = RuleFamily::Standard;

  /// Throws std::invalid_argument for a malformed option (an empty promotion
  /// set, a pawn or brawn in it).
  void validate() const;

  /// The canonical 64-bit encoding: 0 for the defaults, one bit per
  /// non-default choice, the rule family's number in bits 28-31, the
  /// promotion set (XOR the default) in bits 32-47.
  std::uint64_t encode() const noexcept;

  /// The ruleset ID folded into the Zobrist key (§11.3, I9): 0 for the
  /// defaults, otherwise a non-zero 32-bit hash of encode().
  std::uint32_t ruleset_id() const noexcept;

  /// Whether activity depends on the history of play rather than on the
  /// timelines' creators alone (Q1 Consecutive, Q3 play activation, Q6
  /// Deferred): GameState then carries extra state, folded into its key.
  constexpr bool history_dependent_activity() const noexcept {
    return activation != ActivationRule::Count || play_activates ||
           reactivation != Reactivation::Rewinds;
  }

  friend bool operator==(const RuleOptions&, const RuleOptions&) noexcept = default;
};

/// Every non-default choice by name, with its rule ID - "activation=Consecutive
/// (Q1), past_self_blocks=false (Q19)" - or "standard" when there is none. For
/// messages and tests.
std::string describe(const RuleOptions& rules);

}  // namespace t5d

#endif  // T5D_RULES_HPP
