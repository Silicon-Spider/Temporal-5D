// The ruleset: validation, the canonical encoding and the ruleset ID.

#include "t5d/rules.hpp"

#include <stdexcept>
#include <string>

#include "t5d/zobrist.hpp"

namespace t5d {

namespace {

constexpr std::uint64_t flag(bool non_default, int bit) noexcept {
  return non_default ? (std::uint64_t{1} << bit) : 0;
}

// The salt that turns an encoding into a ruleset ID: the ASCII bytes of
// "t5d-rul1". Changing it changes every non-standard ruleset ID.
constexpr std::uint64_t kRulesetSalt = 0x7435642D72756C31ull;

}  // namespace

void RuleOptions::validate() const {
  if (promotions.empty()) {
    throw std::invalid_argument("ruleset option promotions: a pawn must be able to promote "
                                "to at least one kind of piece");
  }
  if (promotions.contains(PieceKind::Pawn) || promotions.contains(PieceKind::Brawn)) {
    throw std::invalid_argument(
        "ruleset option promotions: a pawn or brawn cannot promote to a pawn or a brawn");
  }
  if ((promotions.mask() >> kPieceKindCount) != 0) {
    throw std::invalid_argument("ruleset option promotions: unknown piece kinds in the mask");
  }
}

std::uint64_t RuleOptions::encode() const noexcept {
  const RuleOptions d{};
  std::uint64_t e = 0;
  e |= flag(activation != d.activation, 0);
  e |= flag(initial_credit != d.initial_credit, 1);
  e |= flag(play_activates != d.play_activates, 2);
  e |= flag(moves_after_shift != d.moves_after_shift, 3);
  e |= flag(mandatory != d.mandatory, 4);
  e |= flag(reactivation != d.reactivation, 5);
  e |= flag(turn_zero_reachable != d.turn_zero_reachable, 6);
  e |= flag(sliders_cross_void != d.sliders_cross_void, 7);
  e |= flag(past_self_blocks != d.past_self_blocks, 8);
  e |= flag(brawn_rank_timeline_capture != d.brawn_rank_timeline_capture, 9);
  e |= flag(double_step_right != d.double_step_right, 10);
  e |= flag(timeline_double_step != d.timeline_double_step, 11);
  e |= flag(double_step_over_void != d.double_step_over_void, 12);
  e |= flag(branch_copies_keep_unmoved != d.branch_copies_keep_unmoved, 13);
  e |= flag(jumps_keep_unmoved != d.jumps_keep_unmoved, 14);
  e |= flag(en_passant_brawns != d.en_passant_brawns, 15);
  e |= flag(brawns_promote != d.brawns_promote, 16);
  // bit 17: unused (castling_attacks, removed when H40 closed it)
  e |= flag(castling_empty_between != d.castling_empty_between, 18);
  e |= flag(castling_destination_test != d.castling_destination_test, 19);
  e |= flag(castling_rook_path_test != d.castling_rook_path_test, 20);
  // bit 21: unused (no common-king castling option - see the engine README)
  e |= flag(castling_any_rank != d.castling_any_rank, 22);
  e |= flag(castling_rook_two_away != d.castling_rook_two_away, 23);
  e |= flag(castling_on_optional_boards != d.castling_on_optional_boards, 24);
  e |= flag(earlier_playable_jump_branches != d.earlier_playable_jump_branches, 25);
  e |= flag(inactive_history_is_destination != d.inactive_history_is_destination, 26);
  e |= flag(brawn_cross_board_promotion != d.brawn_cross_board_promotion, 27);
  // Every enum above has exactly two values, so one bit each is enough; an
  // enum that gains a third value needs more bits here.
  // Bits 28-31: the rule family's number (0 = Standard), room for 16.
  e |= std::uint64_t{static_cast<std::uint8_t>(family) & 0xFu} << 28;
  e |= std::uint64_t{static_cast<std::uint16_t>(promotions.mask() ^ d.promotions.mask())} << 32;
  return e;
}

std::uint32_t RuleOptions::ruleset_id() const noexcept {
  const std::uint64_t e = encode();
  if (e == 0) return 0;
  const auto id = static_cast<std::uint32_t>(zobrist::avalanche(e ^ kRulesetSalt));
  return id != 0 ? id : 1;
}

std::string describe(const RuleOptions& r) {
  const RuleOptions d{};
  std::string out;
  const auto add = [&](bool non_default, const char* name) {
    if (!non_default) return;
    if (!out.empty()) out += ", ";
    out += name;
  };
  add(r.activation == ActivationRule::Consecutive, "activation=Consecutive (Q1)");
  add(r.initial_credit == InitialTimelineCredit::MinusZeroToBlack,
      "initial_credit=MinusZeroToBlack (Q2)");
  add(r.play_activates, "play_activates (Q3)");
  add(r.moves_after_shift, "moves_after_shift (Q4)");
  add(r.mandatory == MandatoryBoards::AllActive, "mandatory=AllActive (Q5)");
  add(r.reactivation == Reactivation::Deferred, "reactivation=Deferred (Q6)");
  add(!r.turn_zero_reachable, "turn_zero_reachable=false (Q8)");
  add(r.sliders_cross_void, "sliders_cross_void (Q18)");
  add(!r.past_self_blocks, "past_self_blocks=false (Q19)");
  add(r.earlier_playable_jump_branches, "earlier_playable_jump_branches (J3)");
  add(!r.inactive_history_is_destination, "inactive_history_is_destination=false (Q20)");
  add(!r.brawn_rank_timeline_capture, "brawn_rank_timeline_capture=false (Q14)");
  add(r.double_step_right == DoubleStepRight::HomeRank, "double_step_right=HomeRank (Q15)");
  add(!r.timeline_double_step, "timeline_double_step=false (Q15)");
  add(r.double_step_over_void, "double_step_over_void (Q15)");
  add(!r.branch_copies_keep_unmoved, "branch_copies_keep_unmoved=false (Q15)");
  add(r.jumps_keep_unmoved, "jumps_keep_unmoved (Q15)");
  add(!r.en_passant_brawns, "en_passant_brawns=false (Q16)");
  add(!r.brawns_promote, "brawns_promote=false (Q17)");
  add(!r.brawn_cross_board_promotion, "brawn_cross_board_promotion=false (Q17)");
  add(r.promotions != d.promotions, "promotions (S8)");
  add(!r.castling_empty_between, "castling_empty_between=false (Q21)");
  add(r.castling_destination_test, "castling_destination_test (S3)");
  add(r.castling_rook_path_test, "castling_rook_path_test (Q21)");
  add(!r.castling_any_rank, "castling_any_rank=false (S5)");
  add(!r.castling_rook_two_away, "castling_rook_two_away=false (S5)");
  add(!r.castling_on_optional_boards, "castling_on_optional_boards=false (Q21)");
  add(r.family == RuleFamily::ReversedRoyalty, "family=ReversedRoyalty (spec §12.6)");
  return out.empty() ? std::string("standard") : out;
}

}  // namespace t5d
