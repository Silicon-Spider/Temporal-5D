// The direction tables (rules reference 4.0-4.13) and the ruleset: defaults,
// validation, the ruleset ID folded into the key (spec §11.3, I9).

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/geometry.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;

namespace {

std::tuple<int, int, int, int> key_of(Vec4 v) { return {v.x, v.y, v.l, v.t}; }

}  // namespace

TEST(geometry_unit_directions_are_grouped_by_axis_count) {
  // 4.1 rook 8, 4.2 bishop 24, 4.8 unicorn 32, 4.9 dragon 16; the queen all 80
  // (4.3: 8+24+32+16); the princess rook+bishop, 32 (4.10).
  CHECK_EQ(geometry::rook_directions().size(), std::size_t{8});
  CHECK_EQ(geometry::bishop_directions().size(), std::size_t{24});
  CHECK_EQ(geometry::unicorn_directions().size(), std::size_t{32});
  CHECK_EQ(geometry::dragon_directions().size(), std::size_t{16});
  CHECK_EQ(geometry::princess_directions().size(), std::size_t{32});
  CHECK_EQ(geometry::all_directions().size(), std::size_t{80});
  for (const Vec4 v : geometry::rook_directions()) CHECK_EQ(v.axes(), 1);
  for (const Vec4 v : geometry::bishop_directions()) CHECK_EQ(v.axes(), 2);
  for (const Vec4 v : geometry::unicorn_directions()) CHECK_EQ(v.axes(), 3);
  for (const Vec4 v : geometry::dragon_directions()) CHECK_EQ(v.axes(), 4);
  std::set<std::tuple<int, int, int, int>> seen;
  for (const Vec4 v : geometry::all_directions()) {
    CHECK(v.x >= -1 && v.x <= 1 && v.y >= -1 && v.y <= 1 && v.l >= -1 && v.l <= 1 &&
          v.t >= -1 && v.t <= 1);
    seen.insert(key_of(v));
  }
  CHECK_EQ(seen.size(), std::size_t{80});
  CHECK(seen.count({0, 0, 0, 0}) == 0);
  CHECK((geometry::slider_directions(PieceKind::Queen).size() == 80));
  CHECK((geometry::slider_directions(PieceKind::RoyalQueen).size() == 80));
  CHECK(geometry::slider_directions(PieceKind::Knight).empty());
  CHECK(geometry::slider_directions(PieceKind::King).empty());
  CHECK(geometry::slider_directions(PieceKind::Pawn).empty());
}

TEST(geometry_knight_leaps_are_two_and_one_on_distinct_axes) {
  // 4.5: 48 vectors - 12 ordered axis pairs x 4 sign combinations.
  std::set<std::tuple<int, int, int, int>> seen;
  for (const Vec4 v : geometry::knight_leaps()) {
    CHECK_EQ(v.axes(), 2);
    const int sum = (v.x < 0 ? -v.x : v.x) + (v.y < 0 ? -v.y : v.y) + (v.l < 0 ? -v.l : v.l) +
                    (v.t < 0 ? -v.t : v.t);
    CHECK_EQ(sum, 3);
    seen.insert(key_of(v));
  }
  CHECK_EQ(seen.size(), std::size_t{48});
}

TEST(geometry_pawn_forward_directions) {
  // 4.6: white +rank and -timeline, black -rank and +timeline.
  CHECK_EQ(geometry::forward_rank(Colour::White), 1);
  CHECK_EQ(geometry::forward_rank(Colour::Black), -1);
  CHECK_EQ(geometry::forward_timeline(Colour::White), -1);
  CHECK_EQ(geometry::forward_timeline(Colour::Black), 1);
}

TEST(rules_defaults_are_the_standard_ruleset) {
  const RuleOptions r;
  CHECK_EQ(r.encode(), std::uint64_t{0});
  CHECK_EQ(r.ruleset_id(), 0u);
  CHECK_EQ(describe(r), std::string("standard"));
  CHECK_NOTHROW(r.validate());
  CHECK(!r.history_dependent_activity());
  CHECK(r.promotions.contains(PieceKind::Queen));
  CHECK(!r.promotions.contains(PieceKind::Rook));
}

TEST(rules_every_option_changes_the_ruleset_id) {
  std::vector<RuleOptions> variants;
  const auto with = [&](auto&& edit) {
    RuleOptions r;
    edit(r);
    variants.push_back(r);
  };
  with([](RuleOptions& r) { r.activation = ActivationRule::Consecutive; });
  with([](RuleOptions& r) { r.initial_credit = InitialTimelineCredit::MinusZeroToBlack; });
  with([](RuleOptions& r) { r.play_activates = true; });
  with([](RuleOptions& r) { r.moves_after_shift = true; });
  with([](RuleOptions& r) { r.mandatory = MandatoryBoards::AllActive; });
  with([](RuleOptions& r) { r.reactivation = Reactivation::Deferred; });
  with([](RuleOptions& r) { r.turn_zero_reachable = false; });
  with([](RuleOptions& r) { r.sliders_cross_void = true; });
  with([](RuleOptions& r) { r.past_self_blocks = false; });
  with([](RuleOptions& r) { r.brawn_rank_timeline_capture = false; });
  with([](RuleOptions& r) { r.double_step_right = DoubleStepRight::HomeRank; });
  with([](RuleOptions& r) { r.timeline_double_step = false; });
  with([](RuleOptions& r) { r.double_step_over_void = true; });
  with([](RuleOptions& r) { r.branch_copies_keep_unmoved = false; });
  with([](RuleOptions& r) { r.jumps_keep_unmoved = true; });
  with([](RuleOptions& r) { r.en_passant_brawns = false; });
  with([](RuleOptions& r) { r.brawns_promote = false; });
  with([](RuleOptions& r) { r.promotions.insert(PieceKind::Knight); });
  with([](RuleOptions& r) { r.castling_empty_between = false; });
  with([](RuleOptions& r) { r.castling_destination_test = true; });
  with([](RuleOptions& r) { r.castling_rook_path_test = true; });
  with([](RuleOptions& r) { r.castling_any_rank = false; });
  with([](RuleOptions& r) { r.castling_rook_two_away = false; });
  with([](RuleOptions& r) { r.castling_on_optional_boards = false; });
  with([](RuleOptions& r) { r.earlier_playable_jump_branches = true; });
  with([](RuleOptions& r) { r.inactive_history_is_destination = false; });
  with([](RuleOptions& r) { r.brawn_cross_board_promotion = false; });
  with([](RuleOptions& r) { r.family = RuleFamily::ReversedRoyalty; });  // spec §12.6, §11.3
  std::set<std::uint64_t> encodings;
  std::set<std::uint32_t> ids;
  for (const RuleOptions& r : variants) {
    CHECK_NE(r.encode(), std::uint64_t{0});
    CHECK_NE(r.ruleset_id(), 0u);
    CHECK_NE(describe(r), std::string("standard"));
    encodings.insert(r.encode());
    ids.insert(r.ruleset_id());
  }
  CHECK_EQ(encodings.size(), variants.size());
  CHECK_EQ(ids.size(), variants.size());
}

TEST(rules_validation_refuses_malformed_promotion_sets) {
  // (Castling's cross-board attack option, which validate() used to refuse,
  // is gone: the per-board test is the rule - R6, register H40.)
  RuleOptions none;
  none.promotions = PromotionSet::from_mask(0);
  CHECK_THROWS_AS(GameState(standard_start(), none), std::invalid_argument);
  none.promotions = PromotionSet::from_mask(0);
  CHECK_THROWS_AS(none.validate(), std::invalid_argument);
  RuleOptions pawn;
  pawn.promotions = PromotionSet::of({PieceKind::Queen, PieceKind::Pawn});
  CHECK_THROWS_AS(pawn.validate(), std::invalid_argument);
  RuleOptions brawn;
  brawn.promotions = PromotionSet::of({PieceKind::Brawn});
  CHECK_THROWS_AS(brawn.validate(), std::invalid_argument);
  RuleOptions wide;
  wide.promotions = PromotionSet::of({PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop,
                                      PieceKind::Knight, PieceKind::RoyalQueen});
  CHECK_NOTHROW(wide.validate());
}

TEST(rules_the_state_carries_the_ruleset_id_in_its_key) {
  // I9: exact stores are per ruleset - so the key of one position differs
  // between rulesets, and the multiverse is stamped with the options' ID.
  const GameState standard(standard_start());
  CHECK_EQ(standard.multiverse().ruleset(), 0u);
  CHECK_EQ(standard.zobrist(), standard_start().zobrist());
  RuleOptions r;
  r.brawn_rank_timeline_capture = false;
  const GameState other(standard_start(), r);
  CHECK_EQ(other.multiverse().ruleset(), r.ruleset_id());
  CHECK_NE(other.zobrist(), standard.zobrist());
  CHECK(!(other == standard));
  // A state built from 5DFEN under those options is stamped the same way.
  const GameState parsed = t5d_test::game(
      {"[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]"},
      BoardSize{8, 8}, r);
  CHECK(parsed == other);
  CHECK_EQ(parsed.zobrist(), other.zobrist());
}
