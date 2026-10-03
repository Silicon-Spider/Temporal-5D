// Guard (stage E9; spec 11.4; register H42, H37): depth 1 is exactly "the
// opponent has no mate in one against the action" (every opponent action -
// the brute-force reference, else E4's listing - and E4's verdict after it),
// completing within the budget wherever the opponent's actions are few enough
// to enumerate; deeper depths and every refutation against the restricted
// game and the game tree; no budget stop claims more than the full search
// (H37); one thread deterministic; an illegal action refused; the hook called
// exactly on a refutation.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/guard.hpp"
#include "t5d/movegen.hpp"
#include "t5d/prover.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_game_tree.hpp"
#include "test_mate_refs.hpp"
#include "test_play.hpp"
#include "test_positions.hpp"
#include "test_reference.hpp"
#include "test_rulesets.hpp"
#include "test_support.hpp"
#include "test_v3_support.hpp"

using namespace t5d;
using namespace t5d_test::v3;
using t5d_test::describe_position;
using t5d_test::game;
using t5d_test::GameTree;
using t5d_test::GameValue;
using t5d_test::playable;
using t5d_test::RestrictedGame;

// ---- Guard (H42) ----------------------------------------------------------------------------

TEST(guard_depth_one_is_the_opponents_mate_in_one) {
  // Depth 1 against brute force: every opponent action and E4's verdict
  // after it. "Safe to depth 1" only where no opponent action mates, a
  // refutation only where one does - its line that mate; and where the
  // opponent has at most a few thousand actions (Guard enumerates every one
  // of them: its cost is their number, a product over their boards) depth 1
  // completes or a mate is proven, within the budget.
  int safe = 0;
  int refuted = 0;
  int open = 0;
  std::uint64_t seed = 2400;
  for (const RuleOptions& rules : {RuleOptions{}, guide_literal_q4()}) {
    for (const Case& c : cases(rules, 2, 5, seed++)) {
      const GameState after = after_action(c.s, c.action);
      std::size_t actions = 0;
      const int want = mates_in_one(after, actions);
      if (want < 0) continue;
      GuardOptions g;
      g.max_depth = 1;
      const GuardResult r = guard(c.s, c.action, ProveBudget::nodes(3'000'000), g);
      if (r.refuted) CHECK_EQ(want, 1);
      if (r.completed_depth == 1) CHECK_EQ(want, 0);
      if (!(r.completed_depth == 1 || r.refuted)) {
        ++open;
        if (actions <= 3000) {
          std::fprintf(stderr, "  depth 1 not decided (want %d, %zu actions): %llu nodes, %s; after %s\n", want,
                       actions, static_cast<unsigned long long>(r.nodes),
                       r.depths.empty() ? "-" : proof_status_name(r.depths.front().status),
                       describe_position(after).c_str());
          CHECK(false);
        }
        continue;
      }
      if (r.refuted) {
        ++refuted;
        CHECK_EQ(r.refutation_depth, 1);
        GameState t = after;
        CHECK(r.line_complete);
        CHECK(replay(t, r.line));
        CHECK(mated(t));
      } else {
        ++safe;
      }
    }
  }
  std::fprintf(stderr, "  depth 1: %d safe, %d refuted, %d undecided (too many opponent actions)\n", safe,
               refuted, open);
  CHECK(safe > 50);
  CHECK(refuted > 3);
}

TEST(guard_depths_and_refutations_agree_with_the_references) {
  // Up to depth 2: every completed depth is the restricted game's "no mate
  // within 2d - 1" for the opponent (its every action, ours every legal one),
  // every refutation its mate - and the game tree's.
  int depths = 0;
  int refutations = 0;
  int beyond = 0;
  std::uint64_t seed = 2500;
  for (const Case& c : cases(RuleOptions{}, 1, 5, seed)) {
    const GameState after = after_action(c.s, c.action);
    GuardOptions g;
    g.max_depth = 2;
    const GuardResult r = guard(c.s, c.action, ProveBudget::nodes(1'000'000), g);
    for (int d = 1; d <= r.completed_depth; ++d) {
      RestrictedGame ref(after.to_move(), true, 200'000);
      const int m = ref.mates(after, 2 * d - 1);
      if (m < 0) {
        ++beyond;
        continue;
      }
      ++depths;
      if (m != 0) std::fprintf(stderr, "  safe to %d but mated at %s\n", d, describe_position(after).c_str());
      CHECK_EQ(m, 0);
    }
    if (r.refuted) {
      GameTree tree(20000, 60000, 2'000'000);
      const GameValue v = tree.solve(after, 2 * r.refutation_depth - 1);
      if (v.kind == GameValue::Kind::Unknown) {
        RestrictedGame ref(after.to_move(), true, 300'000);
        const int m = ref.mates(after, 2 * r.refutation_depth - 1);
        if (m < 0) {
          ++beyond;
          continue;
        }
        CHECK_EQ(m, 1);
      } else {
        CHECK(v.kind == GameValue::Kind::Win && v.distance <= 2 * r.refutation_depth - 1);
      }
      ++refutations;
    }
  }
  std::fprintf(stderr, "  %d depths and %d refutations confirmed, %d beyond the references\n", depths, refutations,
               beyond);
  CHECK(depths > 30);
  CHECK(refutations > 2);
}

TEST(guard_no_budget_stop_claims_more_than_the_full_search_h37) {
  // One thread, node budgets stepped from 0 to the full search's need: never a
  // refutation the full search does not make, never a depth it does not
  // complete.
  int checked = 0;
  std::uint64_t seed = 2600;
  for (const Case& c : cases(RuleOptions{}, 1, 4, seed)) {
    GuardOptions g;
    g.max_depth = 2;
    const GuardResult full = guard(c.s, c.action, ProveBudget::nodes(1'000'000), g);
    const std::uint64_t step = std::max<std::uint64_t>(1, full.nodes / 25);
    for (std::uint64_t b = 0; b < full.nodes; b += step) {
      const GuardResult r = guard(c.s, c.action, ProveBudget::nodes(b), g);
      ++checked;
      CHECK(r.completed_depth <= full.completed_depth);
      if (r.refuted) {
        CHECK(full.refuted);
        CHECK_EQ(r.refutation_depth, full.refutation_depth);
      }
    }
  }
  std::fprintf(stderr, "  %d budgets\n", checked);
  CHECK(checked > 100);
}

TEST(guard_one_thread_is_deterministic) {
  for (const Case& c : cases(RuleOptions{}, 1, 3, 2700)) {
    GuardOptions g;
    g.max_depth = 2;
    const GuardResult a = guard(c.s, c.action, ProveBudget::nodes(20'000), g);
    const GuardResult b = guard(c.s, c.action, ProveBudget::nodes(20'000), g);
    CHECK_EQ(a.completed_depth, b.completed_depth);
    CHECK_EQ(a.refuted, b.refuted);
    CHECK_EQ(a.nodes, b.nodes);
  }
}

TEST(guard_refuses_an_illegal_action_and_calls_its_hook) {
  const GameState s = mate_in_one();
  // Nothing played: the obligations are not met.
  CHECK_THROWS_AS(guard(s, {}, ProveBudget::nodes(1000)), std::invalid_argument);
  // A blunder of Black's after White's quiet move would be for the search;
  // here: the hook is called exactly on a refutation.
  int calls = 0;
  GuardHook hook;
  hook.on_refutation = [&](const GuardResult&) { ++calls; };
  int refuted = 0;
  for (const Case& c : cases(RuleOptions{}, 1, 5, 2800)) {
    GuardOptions g;
    g.max_depth = 1;
    const GuardResult r = guard(c.s, c.action, ProveBudget::nodes(1'000'000), g, hook);
    if (r.refuted) ++refuted;
  }
  CHECK_EQ(calls, refuted);
}
