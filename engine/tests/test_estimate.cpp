// The estimate channel (t5d/estimate.hpp; spec §11.1's redesign, part 1;
// register M14, H2, H6). It is inexact by design - its outputs are soft
// priors - so what is held here is what it must still be: the v1 material
// eval exactly; the plain tactics a capture search sees (a free piece, a
// defended one, a royal left capturable, a threat that carries); bounded by
// its node cap; deterministic; and robust on random play under every ruleset.
//
// 5x5 boards unless said otherwise; positions written from the rulebook.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "t5d/estimate.hpp"
#include "t5d/scanner.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_play.hpp"
#include "test_positions.hpp"
#include "test_rulesets.hpp"

using namespace t5d;
using t5d_test::game;

namespace {
constexpr BoardSize k5x5{5, 5};
}  // namespace

TEST(estimate_static_eval_is_the_material_count_h6) {
  Scanner scanner;
  int checked = 0;
  for (const GameState& s : t5d_test::sparse_positions(RuleOptions{}, 1, 6, 31)) {
    CHECK_EQ(estimate(s).static_eval, scanner.evaluate(s));
    ++checked;
  }
  for (const GameState& s : t5d_test::variant_positions(RuleOptions{}, {"Standard", "Standard - Two Timelines"}, 6, 32)) {
    CHECK_EQ(estimate(s).static_eval, scanner.evaluate(s));
    ++checked;
  }
  CHECK(checked > 50);
}

TEST(estimate_takes_a_free_piece) {
  // White to move: Rd1 can take the black queen d4, which nothing defends.
  const GameState s = game({"[k4/3q1/5/5/3RK:0:1:w]"}, k5x5);
  const EstimateResult r = estimate(s);
  CHECK_EQ(r.static_eval, 500 - 900);
  CHECK(r.score >= r.static_eval + 900 - 1);
  CHECK(r.live);
  REQUIRE(r.best.has_value());
  CHECK(r.best->capture());
  CHECK(r.best->captured.kind() == PieceKind::Queen);
}

TEST(estimate_sees_a_defended_piece_cost_the_capturer) {
  // White's queen can take a pawn b4 that a black pawn a5 defends: queen for
  // pawn is a loss, so standing pat is better - the score is the static eval
  // (captures only: with threats, Qa1+ forks the king e5 and the pawn a5,
  // which the next check finds).
  const GameState s = game({"[p3k/1p3/5/5/1Q2K:0:1:w]"}, k5x5);
  EstimateOptions o;
  o.threat_plies = 0;
  const EstimateResult r = estimate(s, o);
  CHECK_EQ(r.score, r.static_eval);
  CHECK(!r.live);
  // The fork: Qa1+ and Qxa5 win the pawn.
  o.threat_plies = 1;
  const EstimateResult f = estimate(s, o);
  CHECK_EQ(f.score, f.static_eval + 100);
  REQUIRE(f.best.has_value());
  CHECK(to_string(*f.best, false) == "(0T1)Qb1(0T1)a1");
}

TEST(estimate_a_royal_left_capturable_is_captured) {
  // White to move can take the black king: the previous action broke C4.
  const GameState s = game({"[k4/5/5/5/R3K:0:1:w]"}, k5x5);
  EstimateOptions o;
  const EstimateResult r = estimate(s, o);
  CHECK_EQ(r.score, o.royal_value);
}

TEST(estimate_a_threat_carries_through_a_pass) {
  // White to move: Rd1-d5 checks the king a5 along the fifth rank on the
  // board the move creates; if Black does not answer there, the board is
  // passed and the rook takes the king. Black can answer (Ka5-a4, b4, b5).
  // With threats searched the score is at least the static eval; the search
  // looked at the check.
  const GameState s = game({"[k4/5/5/5/3RK:0:1:w]"}, k5x5);
  EstimateOptions o;
  o.threat_plies = 1;
  const EstimateResult with = estimate(s, o);
  o.threat_plies = 0;
  const EstimateResult without = estimate(s, o);
  CHECK(with.nodes > without.nodes);
  CHECK(with.score >= with.static_eval);
}

TEST(estimate_is_bounded_and_deterministic_under_every_ruleset) {
  int positions = 0;
  int live = 0;
  std::uint64_t seed = 40;
  EstimateOptions o;
  o.max_nodes = 2000;
  for (const RuleOptions& rules : t5d_test::every_ruleset()) {
    std::vector<GameState> all = t5d_test::sparse_positions(rules, 1, 5, seed++);
    for (GameState& s : t5d_test::variant_positions(rules, {"Standard", "Misc - Timeline Battleground"}, 5, seed++)) {
      all.push_back(std::move(s));
    }
    for (const GameState& s : all) {
      const EstimateResult a = estimate(s, o);
      const EstimateResult b = estimate(s, o);
      CHECK(a.positions <= o.max_nodes);
      CHECK(a.depth <= o.max_plies);
      CHECK_EQ(a.score, b.score);
      CHECK_EQ(a.nodes, b.nodes);
      if (a.live) ++live;
      ++positions;
    }
  }
  std::fprintf(stderr, "  %d positions, %d tactically live\n", positions, live);
  CHECK(positions > 300);
  CHECK(live > 0);
}
