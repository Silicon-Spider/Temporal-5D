// t5d/variants.hpp - the named variants: starting positions by name.
//
// A 5dpgn record names its variant in its Board header (notation
// specification, "Headers"; "Variant" is read as a synonym when there is no
// Board header - spec issue 2g of the build log). The engine sets up the
// variants listed here, each from its row of the project's rules reference
// (docs/rules/README.md, section 8: V1 names the variants, V2 says how they
// start, V3 gives the placements) - never from another implementation's table
// of layouts (decision H28; build log, spec issue 4). Every other name is
// unsupported: the engine refuses the record rather than guess a layout.
//
// Placements. V3 writes placements without 5DFEN's '*' (except Standard). The
// engine marks every pawn, rook, king and brawn of a set-up as unmoved: none of
// them has moved (the notation's "5DFEN Strictness" kinds; the gate's state
// contract asks for exactly this). Which of them may then double-step or
// castle is the ruleset's business (Q15, Q21) - a back-rank pawn of Focused -
// Just Pawns has the double step under the default UnmovedFlag right and not
// under HomeRank.
//
// Every variant starts at T1 with White to move (V2), Standard - Turn Zero on
// two boards (its T0 board a history board from the start). The
// multi-timeline variants' boards are listed top first as seen from White's
// side and taken to be -0 (or -1) first - V3's inferred mapping. Their
// non-central initial timelines (Timeline Battleground's -1 and +1) are read as
// 5DFEN reads any non-central timeline: created by the colour of their sign,
// creation order 1 each. The count rule (T10) treats the two sides alike there,
// so that matches "initial timelines count for neither side" (Q2's default);
// Q2's alternative is the ruleset option initial_credit.
//
// Reversed Royalty is a rule variant (design spec §12.6): its set-up carries
// RuleFamily::ReversedRoyalty, which gives it a ruleset ID of its own (§11.3,
// I9). Q17 asks whether its pawns promote to a queen or to a royal queen; the
// default is S8's queen (high: "promoted to a queen automatically"), and the
// alternative is the promotion set {royal queen} (RuleOptions::promotions).

#ifndef T5D_VARIANTS_HPP
#define T5D_VARIANTS_HPP

#include <cstdint>
#include <span>
#include <string_view>

#include "t5d/fen.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "t5d/types.hpp"

namespace t5d {

/// The rules reference's confidence in a variant's set-up (its V3 rating).
enum class Confidence : std::uint8_t { Low = 0, Medium, High };

/// One named variant.
struct Variant {
  /// The name, as the game's variant menu writes it (V1) - what a Board header
  /// holds.
  std::string_view name;
  /// Another name accepted for it (the 2020 name, V3), or empty.
  std::string_view alias;
  BoardSize size;
  /// Labels "-0"/"+0" (even) or "0" (odd).
  Parity parity = Parity::Odd;
  /// The initial boards as 5DFEN board strings, unmoved markers included.
  std::span<const std::string_view> boards;
  /// The rule family (RuleFamily::Standard except Reversed Royalty).
  RuleFamily family = RuleFamily::Standard;
  /// Its row in the rules reference, with the reference's rating and notes.
  std::string_view source;
  Confidence confidence = Confidence::Medium;
};

/// Every named variant the engine sets up, in the rules reference's order.
std::span<const Variant> variants() noexcept;

/// The variant a Board (or Variant) header names, or nullptr. Names compare
/// without regard to case or to runs of whitespace ("standard  - turn zero"
/// names "Standard - Turn Zero"); nothing else is forgiven.
const Variant* find_variant(std::string_view name) noexcept;

/// The variant's starting state under `rules`, with rules.family replaced by
/// the variant's own family. White to move (V2).
GameState variant_start(const Variant& variant, const RuleOptions& rules = {});

}  // namespace t5d

#endif  // T5D_VARIANTS_HPP
