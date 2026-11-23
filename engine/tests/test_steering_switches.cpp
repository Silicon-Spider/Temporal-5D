// H13 and H43's steering switches, for 6e's on/off match test (spec §11.2:
// "it has an on/off test of its own at match play"; §13.1's 6e: "each
// steering term on and off at match play - the asset map, Hunt's suggested
// lines"). The engine's switches are SearchParams::asset_map (on by default,
// off in self-play) and SearchParams::hunt_suggest (off). These tests pin what
// a match between two settings relies on: each switch off is the search
// without the feature (H15's digest); Hunt's lines, on, raise only the root
// edges Hunt suggested; a mate Hunt proves raises no prior and decides nothing
// while the proof channel is off (decision 12, build-log spec issue 180: a
// proof is an estimate before the sign-off; no proof-to-prior hook is built -
// spec issue 186(a)); and the as-built limit at a carried root (build-log spec
// issue 477): a turn-start root the carried tree had already expanded takes no
// suggestions.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/ladder.hpp"
#include "t5d/mcts.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_play.hpp"
#include "test_positions.hpp"
#include "test_v3_support.hpp"

using namespace t5d;
using namespace t5d::mcts;
using t5d_test::game;
using t5d_test::v3::k5x5;

namespace {

GameState variant(const char* name) {
  const Variant* v = find_variant(name);
  if (v == nullptr) throw std::logic_error(std::string("no variant ") + name);
  return variant_start(*v, RuleOptions{});
}

// The tests' parameters (test_ladder.cpp's sync_params, copied): no square
// features, the synchronous mode - every ladder job answered where it is
// asked, by nodes, one thread (H15).
SearchParams sync_params() {
  SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.synchronous = true;
  return p;
}

// test_ladder.cpp's asset position, copied: three 5 x 5 boards, White's queen
// on the present board's a1 - the asset map boosts there.
GameState asset_position() {
  return game({"[k4/5/5/5/Q3K:0:1:w]", "[k4/5/5/5/4K:0:1:b]", "[k4/5/5/5/3K1:0:2:w]"}, k5x5);
}

// Hunt's lines: Hunt shallow and cheap, so that it stops at its budget with
// open root children (and suggests), the settling scans and the map off.
SearchParams hunt_lines(TurnSplit split, bool suggest) {
  SearchParams p = sync_params();
  p.split = split;
  p.settling = false;
  p.asset_map = false;
  p.hunt_depth = 3;
  p.sync_job_nodes = 3000;
  p.hunt_suggest = suggest;
  return p;
}

const LadderVerdict* root_hunt(const Search& s) {
  for (const LadderVerdict& v : s.verdicts()) {
    if (v.job == LadderJob::Hunt && !v.carried && v.path.empty()) return &v;
  }
  return nullptr;
}

// A root child's step is a suggested action's: its first move (the ply split,
// the tree form) or the whole action (the flat form).
bool suggested(const RootChild& c, const std::vector<HuntSuggestion>& suggestions, TurnSplit split) {
  for (const HuntSuggestion& h : suggestions) {
    if (h.action.empty()) continue;
    if (split == TurnSplit::Flat) {
      if (c.moves == h.action) return true;
    } else if (c.moves.size() == 1 && c.moves[0] == h.action.front()) {
      return true;
    }
  }
  return false;
}

const char* split_name(TurnSplit s) { return s == TurnSplit::Flat ? "flat" : s == TurnSplit::Tree ? "tree" : "ply"; }

}  // namespace

TEST(steering_switches_default_to_the_built_behaviour) {
  const SearchParams d;
  CHECK(d.asset_map);
  CHECK(!d.hunt_suggest);
  CHECK_EQ(d.hunt_suggest_boost, 0.25);
  CHECK(!d.proof_channel);
  const SearchParams self = self_play_search_defaults();
  CHECK(!self.asset_map);  // self-play does not steer (I2, issue 179)
  CHECK(!self.hunt_suggest);
  SearchParams only = d;
  only.hunt = only.settling = false;
  only.asset_map = true;
  CHECK(ladder_wanted(only));
  only.asset_map = false;
  only.hunt_suggest = true;
  CHECK(ladder_wanted(only));
  only.hunt_suggest = false;
  CHECK(!ladder_wanted(only));
}

TEST(steering_asset_map_off_is_the_search_without_the_map) {
  // H13: off, the search is the search with no ladder at all - and with the
  // provers on, the same (their proofs withheld, H41) - while on, the map
  // steers this position (the premise).
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 4);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat, TurnSplit::Tree}) {
    for (const bool assets_position : {true, false}) {
      const GameState s = assets_position ? asset_position() : variant("Standard - Two Timelines");
      SearchParams none = sync_params();
      none.split = split;
      none.hunt = none.settling = none.asset_map = false;
      SearchParams map = none;
      map.asset_map = true;
      SearchParams provers = none;
      provers.hunt = provers.settling = true;
      Search a(s, none);
      Search on(s, map);
      Search c(s, provers);
      CHECK(a.shared_ladder() == nullptr);
      a.run(*random, Limits{200});
      on.run(*random, Limits{200});
      c.run(*random, Limits{200});
      c.await_ladder();
      if (assets_position) {
        REQUIRE(on.stats().asset_boosted_edges > 0);
        CHECK(on.digest() != a.digest());
      }
      CHECK_EQ(c.digest(), a.digest());
      CHECK_EQ(c.stats().asset_boosted_edges, std::uint64_t{0});
      CHECK_EQ(c.stats().steered_visits, std::uint64_t{0});
      CHECK_EQ(c.stats().proofs_injected, std::uint64_t{0});
      for (const RootChild& ch : c.root_children()) {
        CHECK_EQ(ch.boost, 0.0);
        CHECK_EQ(ch.steered_visits, 0u);
      }
      CHECK_NOTHROW(c.check_invariants());
    }
  }
}

TEST(steering_hunt_suggest_off_is_the_search_without_hunts_lines) {
  // H43's flag off: Hunt still answers the root's question, suggests nothing,
  // and the tree is the tree without Hunt.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat, TurnSplit::Tree}) {
    SearchParams none_p = hunt_lines(split, false);
    none_p.hunt = false;
    Search off(asset_position(), hunt_lines(split, false));
    Search none(asset_position(), none_p);
    off.run(*uniform, Limits{200});
    off.await_ladder();
    none.run(*uniform, Limits{200});
    CHECK_EQ(off.digest(), none.digest());
    CHECK_EQ(off.stats().hunt_runs, std::uint64_t{1});
    CHECK_EQ(off.stats().suggestion_edges, std::uint64_t{0});
    const LadderVerdict* v = root_hunt(off);
    REQUIRE(v != nullptr);
    CHECK(v->suggestions.empty());
    for (const RootChild& c : off.root_children()) CHECK_EQ(c.boost, 0.0);
  }
}

TEST(steering_hunt_suggest_on_raises_only_the_suggested_lines) {
  // H43's flag on (and, with Hunt off, the flag alone asks Hunt): the root
  // edges raised are the suggested actions' (their first moves; the flat
  // form's whole turns), counted as steering; no proof is involved.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat, TurnSplit::Tree}) {
    for (const bool hunt : {true, false}) {
      SearchParams p = hunt_lines(split, true);
      p.hunt = hunt;
      Search on(asset_position(), p);
      Search off(asset_position(), hunt_lines(split, false));
      on.run(*uniform, Limits{200});
      on.await_ladder();
      off.run(*uniform, Limits{200});
      off.await_ladder();
      CHECK_EQ(on.stats().hunt_runs, std::uint64_t{1});
      const LadderVerdict* v = root_hunt(on);
      REQUIRE(v != nullptr);
      REQUIRE(v->status != ProofStatus::Proven);
      REQUIRE(!v->suggestions.empty());
      const std::uint64_t raised = on.stats().suggestion_edges;
      CHECK(raised >= 1);
      CHECK(raised <= v->suggestions.size());
      std::uint64_t boosted = 0;
      std::uint64_t steered = 0;
      for (const RootChild& c : on.root_children()) {
        steered += c.steered_visits;
        if (c.boost <= 0) continue;
        ++boosted;
        if (!suggested(c, v->suggestions, split)) {
          std::printf("  %s: a boosted root child not suggested\n", split_name(split));
          CHECK(false);
        }
      }
      CHECK(boosted >= 1);
      CHECK_EQ(steered, on.stats().steered_visits);
      CHECK_EQ(on.stats().proofs_withheld, std::uint64_t{0});
      CHECK(on.digest() != off.digest());
      CHECK_NOTHROW(on.check_invariants());
    }
  }
}

TEST(steering_a_proven_mate_raises_no_prior_and_decides_nothing_before_the_sign_off) {
  // Decision 12 (issue 180): before the sign-off a proof may raise a prior,
  // but no hook for it is built - with hunt_suggest on, a Hunt that proves the
  // mate suggests nothing (build-log spec issue 186(a), settled by the
  // ruling), and with the proof channel off the proof is withheld: no bound,
  // no stop, no selection. The search is the search without Hunt.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const GameState& s : {t5d_test::v3::mate_in_one(), t5d_test::v3::mate_in_two()}) {
    SearchParams p = sync_params();
    p.asset_map = false;
    p.settling = false;
    p.hunt_suggest = true;
    SearchParams none_p = p;
    none_p.hunt = none_p.hunt_suggest = false;
    Search on(s, p);
    Search none(s, none_p);
    const StopReason r = on.run(*uniform, Limits{200});
    on.await_ladder();
    none.run(*uniform, Limits{200});
    CHECK_EQ(on.stats().hunt_mates, std::uint64_t{1});
    const LadderVerdict* v = root_hunt(on);
    REQUIRE(v != nullptr);
    CHECK(v->status == ProofStatus::Proven);
    CHECK(v->use == ProofUse::Withheld);
    CHECK(v->suggestions.empty());
    CHECK_EQ(on.stats().suggestion_edges, std::uint64_t{0});
    CHECK(on.stats().proofs_withheld >= 1);
    CHECK_EQ(on.stats().proofs_injected, std::uint64_t{0});
    CHECK(r == StopReason::Playouts);
    CHECK(!on.root_solved());
    for (const RootChild& c : on.root_children()) {
      CHECK(c.lo == -1 && c.hi == 1);
      CHECK_EQ(c.boost, 0.0);
    }
    const Choice choice = on.choose();
    CHECK(choice.reason == ChoiceReason::Lcb || choice.reason == ChoiceReason::Visits);
    CHECK_EQ(on.digest(), none.digest());
  }
}

TEST(steering_a_carried_root_already_expanded_takes_no_suggestions) {
  // The as-built limit (build-log spec issue 477; the plan's open question 1):
  // steering is applied when a node's priors are set, so a turn-start root the
  // carried tree had already expanded takes no suggestions - Hunt answers it
  // and suggests, nothing is raised - while a fresh search of the same state
  // is steered. Flips if a re-steer is ruled and built.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  int found = 0;
  for (const GameState& start : t5d_test::sparse_positions(RuleOptions{}, 2, 4, 7)) {
    if (found > 0) break;
    SearchParams p = hunt_lines(TurnSplit::BoardMove, true);
    Search carried(start, p);
    carried.run(*uniform, Limits{300});
    carried.await_ladder();
    // Along the most visited children to the next turn start, which must
    // already have a visit (an expanded node).
    bool reached = false;
    for (int step = 0; step < 16 && !reached; ++step) {
      const std::vector<RootChild> children = carried.root_children();
      if (children.empty()) break;
      std::size_t best = 0;
      for (std::size_t i = 1; i < children.size(); ++i) {
        if (children[i].visits > children[best].visits) best = i;
      }
      if (children[best].visits < 2) break;
      const bool submit = children[best].submit;
      carried.advance(best);
      reached = submit;
    }
    if (!reached || carried.node(carried.root()).visits == 0) continue;
    if (carried.root_terminal()) continue;
    const std::uint64_t edges = carried.stats().suggestion_edges;
    const std::uint64_t runs = carried.stats().hunt_runs;
    carried.run(*uniform, Limits{100});
    carried.await_ladder();
    if (carried.stats().hunt_runs == runs) continue;  // not in Hunt's range
    const LadderVerdict* v = root_hunt(carried);
    if (v == nullptr || v->suggestions.empty()) continue;
    Search fresh(carried.root_state(), p);
    fresh.run(*uniform, Limits{100});
    fresh.await_ladder();
    if (fresh.stats().suggestion_edges == 0) continue;
    ++found;
    CHECK_EQ(carried.stats().suggestion_edges, edges);
    for (const RootChild& c : carried.root_children()) CHECK_EQ(c.boost, 0.0);
  }
  REQUIRE(found > 0);
}
