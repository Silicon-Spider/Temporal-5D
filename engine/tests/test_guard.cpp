// Guard (stage E9; spec 11.4; register H42, H37): depth 1 is exactly "the
// opponent has no mate in one against the action" (every opponent action -
// the brute-force reference, else E4's listing - and E4's verdict after it),
// completing within the budget wherever the opponent's actions are few enough
// to enumerate; deeper depths and every refutation against the restricted
// game and the game tree; no budget stop claims more than the full search
// (H37); one thread deterministic; an illegal action refused; the hook called
// exactly on a refutation.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
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

namespace {

// Depth 1 against brute force: every opponent action and E4's verdict
// after it. "Safe to depth 1" only where no opponent action mates, a
// refutation only where one does - its line that mate, complete at any
// number of threads (depth 1's stores start empty, so a thread that searched
// the root proved it: t5d/prover.hpp, ProveResult::line); and where the
// opponent has at most a few thousand actions (Guard enumerates every one
// of them: its cost is their number, a product over their boards) depth 1
// completes or a mate is proven, within the budget.
void depth_one_property(const std::vector<RuleOptions>& rulesets, int games, int turns, int threads, int& safe,
                        int& refuted, int& open) {
  std::uint64_t seed = 2400;
  for (const RuleOptions& rules : rulesets) {
    for (const Case& c : cases(rules, games, turns, seed++)) {
      const GameState after = after_action(c.s, c.action);
      std::size_t actions = 0;
      const int want = mates_in_one(after, actions);
      if (want < 0) continue;
      GuardOptions g;
      g.max_depth = 1;
      g.threads = threads;
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
        CHECK_EQ(r.line.size(), std::size_t{1});
        CHECK(replay(t, r.line));
        CHECK(mated(t));
      } else {
        ++safe;
      }
    }
  }
}

}  // namespace

TEST(guard_depth_one_is_the_opponents_mate_in_one) {
  int safe = 0;
  int refuted = 0;
  int open = 0;
  depth_one_property({RuleOptions{}}, 1, 3, 1, safe, refuted, open);
  std::fprintf(stderr, "  depth 1: %d safe, %d refuted, %d undecided (too many opponent actions)\n", safe,
               refuted, open);
  CHECK(safe > 15);
  CHECK(refuted > 1);
}

TEST(guard_at_five_threads_depth_one_is_the_opponents_mate_in_one) {
  // M13's 5 threads: the same answers.
  int safe = 0;
  int refuted = 0;
  int open = 0;
  depth_one_property({RuleOptions{}}, 1, 3, 5, safe, refuted, open);
  std::fprintf(stderr, "  depth 1 at 5 threads: %d safe, %d refuted, %d undecided\n", safe, refuted, open);
  CHECK(safe > 15);
  CHECK(refuted > 1);
}

TEST(long_guard_depth_one_is_the_opponents_mate_in_one) {
  int safe = 0;
  int refuted = 0;
  int open = 0;
  depth_one_property({RuleOptions{}, guide_literal_q4()}, 2, 5, 1, safe, refuted, open);
  std::fprintf(stderr, "  depth 1: %d safe, %d refuted, %d undecided (too many opponent actions)\n", safe,
               refuted, open);
  CHECK(safe > 50);
  CHECK(refuted > 3);
}

namespace {

// Up to depth 2: every completed depth is the restricted game's "no mate
// within 2d - 1" for the opponent (its every action, ours every legal one),
// every refutation its mate - and the game tree's. And every refutation's
// line keeps its contract (t5d/guard.hpp): a prefix that replays from the
// position after our action; complete at depth 1; complete, the opponent's
// mate at the refutation's depth, by E4.
void depths_property(int turns, int threads, std::uint64_t guard_nodes, std::uint64_t reference_work, int& depths,
                     int& refutations, int& beyond, int& complete_lines) {
  std::uint64_t seed = 2500;
  for (const Case& c : cases(RuleOptions{}, 1, turns, seed)) {
    const GameState after = after_action(c.s, c.action);
    GuardOptions g;
    g.max_depth = 2;
    g.threads = threads;
    const GuardResult r = guard(c.s, c.action, ProveBudget::nodes(guard_nodes), g);
    for (int d = 1; d <= r.completed_depth; ++d) {
      RestrictedGame ref(after.to_move(), true, reference_work);
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
      const auto half_turns = static_cast<std::size_t>(2 * r.refutation_depth - 1);
      GameState t = after;
      CHECK(replay(t, r.line));
      CHECK(r.line.size() <= half_turns);
      if (r.refutation_depth == 1) CHECK(r.line_complete);
      if (r.line_complete) {
        ++complete_lines;
        CHECK_EQ(r.line.size(), half_turns);
        CHECK(mated(t));
      }
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
}

}  // namespace

TEST(guard_depths_and_refutations_agree_with_the_references) {
  int depths = 0;
  int refutations = 0;
  int beyond = 0;
  int lines = 0;
  depths_property(2, 1, 300'000, 30'000, depths, refutations, beyond, lines);
  std::fprintf(stderr, "  %d depths and %d refutations confirmed (%d complete lines), %d beyond the references\n",
               depths, refutations, lines, beyond);
  CHECK(depths > 15);
  CHECK(refutations > 0);
}

TEST(guard_at_five_threads_depths_and_refutations_agree_with_the_references) {
  int depths = 0;
  int refutations = 0;
  int beyond = 0;
  int lines = 0;
  depths_property(2, 5, 300'000, 30'000, depths, refutations, beyond, lines);
  std::fprintf(stderr,
               "  at 5 threads: %d depths and %d refutations confirmed (%d complete lines), %d beyond the references\n",
               depths, refutations, lines, beyond);
  CHECK(depths > 15);
  CHECK(refutations > 0);
}

TEST(long_guard_depths_and_refutations_agree_with_the_references) {
  int depths = 0;
  int refutations = 0;
  int beyond = 0;
  int lines = 0;
  depths_property(5, 1, 1'000'000, 200'000, depths, refutations, beyond, lines);
  std::fprintf(stderr, "  %d depths and %d refutations confirmed (%d complete lines), %d beyond the references\n",
               depths, refutations, lines, beyond);
  CHECK(depths > 30);
  CHECK(refutations > 2);
}

namespace {

// Every result reports only what it searched: each depth up to the
// completed one disproven, a stop leaving the next depth Unknown, a
// refutation the last depth's proof (the review's M8: a depth its budget
// stopped never claimed).
void consistent(const GuardResult& r) {
  CHECK(static_cast<int>(r.depths.size()) >= r.completed_depth);
  for (int d = 0; d < r.completed_depth; ++d) {
    CHECK(r.depths[static_cast<std::size_t>(d)].status == ProofStatus::Disproven);
  }
  if (r.stopped && static_cast<int>(r.depths.size()) > r.completed_depth) {
    CHECK(r.depths.back().status == ProofStatus::Unknown);
  }
  if (r.refuted) {
    CHECK(!r.depths.empty() && r.depths.back().status == ProofStatus::Proven);
    CHECK(r.refutation_depth > r.completed_depth);
  }
  CHECK(static_cast<int>(r.depths.size()) <= r.completed_depth + 1);
}

// One thread, node budgets stepped from 0 to the full search's need: never a
// refutation the full search does not make, never a depth it does not
// complete; every result consistent.
int budget_property(int turns, std::uint64_t steps, std::uint64_t full_nodes) {
  int checked = 0;
  std::uint64_t seed = 2600;
  for (const Case& c : cases(RuleOptions{}, 1, turns, seed)) {
    GuardOptions g;
    g.max_depth = 2;
    const GuardResult full = guard(c.s, c.action, ProveBudget::nodes(full_nodes), g);
    consistent(full);
    const std::uint64_t step = std::max<std::uint64_t>(1, full.nodes / steps);
    for (std::uint64_t b = 0; b < full.nodes; b += step) {
      const GuardResult r = guard(c.s, c.action, ProveBudget::nodes(b), g);
      ++checked;
      consistent(r);
      CHECK(r.completed_depth <= full.completed_depth);
      if (r.refuted) {
        CHECK(full.refuted);
        CHECK_EQ(r.refutation_depth, full.refutation_depth);
      }
    }
  }
  return checked;
}

}  // namespace

TEST(guard_no_budget_stop_claims_more_than_the_full_search_h37) {
  const int checked = budget_property(2, 5, 100'000);
  std::fprintf(stderr, "  %d budgets\n", checked);
  CHECK(checked > 40);
}

TEST(long_guard_no_budget_stop_claims_more_than_the_full_search_h37) {
  const int checked = budget_property(4, 25, 1'000'000);
  std::fprintf(stderr, "  %d budgets\n", checked);
  CHECK(checked > 100);
}

TEST(guard_reports_only_the_depths_it_searched) {
  // Small budgets at depth 4 across random play: every result consistent -
  // a depth the budget stopped is never claimed (the review's M8).
  int stopped = 0;
  int results = 0;
  for (const Case& c : cases(RuleOptions{}, 1, 3, 2650)) {
    for (const std::uint64_t b : {std::uint64_t{50}, std::uint64_t{400}, std::uint64_t{3000}}) {
      GuardOptions g;
      g.max_depth = 4;
      const GuardResult r = guard(c.s, c.action, ProveBudget::nodes(b), g);
      consistent(r);
      ++results;
      if (r.stopped) ++stopped;
    }
  }
  std::fprintf(stderr, "  %d results, %d stopped by their budgets\n", results, stopped);
  CHECK(stopped > 10);
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

// ---- the review's findings (stage E9's adversarial review) -------------------------------------

namespace {

// Black (us) to move; Black's action - a pawn push on +0 and the knight's
// jump from -0 onto +1's head - lets White mate in one, but only by a J6 twin
// (the soundness review's position; test_certificate.cpp's twin_position).
GameState twin_guard_position() {
  return game({"[4/4/2P1/3n:-0:1:w]", "[4/4/2P1/3n:-0:1:b]", "[1p2/4/1P2/4:+0:1:w]", "[1p2/4/1P2/4:+0:1:b]",
               "[4/4/1k2/4:+1:1:b]", "[4/4/1k2/4:+1:2:w]", "[4/4/1k2/4:+1:2:b]", "[4/4/2P1/4:-1:2:w]",
               "[4/4/4/4:-1:2:b]"},
              BoardSize{4, 4}, RuleOptions{}, Colour::Black);
}

std::vector<Move> twin_guard_action(const GameState& s) {
  MoveList moves;
  generate_moves(s, moves);
  std::vector<Move> action;
  for (const char* text : {"(+0T1)Pb4(+0T1)b3", "(-0T1)Nd1(+1T2)d1"}) {
    for (const Move& m : moves) {
      if (to_string(m, s.multiverse().even()) == text) action.push_back(m);
    }
  }
  return action;
}

}  // namespace

TEST(guard_the_check_filter_does_not_hide_a_twins_mate) {
  const GameState s = twin_guard_position();
  const std::vector<Move> action = twin_guard_action(s);
  REQUIRE_EQ(action.size(), std::size_t{2});
  for (const bool filter : {false, true}) {
    for (const int threads : {1, 5}) {
      for (const int depth : {1, 2}) {
        GuardOptions g;
        g.max_depth = depth;
        g.threads = threads;
        g.prover.check_filter = filter;
        const GuardResult r = guard(s, action, ProveBudget::nodes(1'000'000), g);
        CHECK(r.refuted);
        CHECK_EQ(r.refutation_depth, 1);
        CHECK_EQ(r.completed_depth, 0);
      }
    }
  }
  // A disproof store shared by the filter's two settings: no class mixes.
  const GameState after = after_action(s, action);
  DisproofStore shared;
  for (const bool filter : {true, false}) {
    ProverOptions o = literal();
    o.depth = 1;
    o.check_filter = filter;
    Prover p(o, nullptr, &shared);
    CHECK(p.prove(after, ProveBudget::nodes(1'000'000)).status == ProofStatus::Proven);
  }
}

TEST(guard_asks_at_most_its_maximum_depth) {
  // White mates in one: after its mating action Black has no legal action,
  // so every depth is safe at once - up to Guard's limit, 50, never beyond.
  const GameState s = mate_in_one();
  Prover p;
  const ProveResult r = p.prove(s, ProveBudget::nodes(200'000));
  REQUIRE(r.status == ProofStatus::Proven);
  GuardOptions g;
  g.max_depth = 60;
  const GuardResult gr = guard(s, r.first_action, ProveBudget::nodes(2'000'000), g);
  CHECK_EQ(gr.completed_depth, kGuardMaxDepth);
  CHECK_EQ(static_cast<int>(gr.depths.size()), kGuardMaxDepth);
  CHECK(!gr.refuted);
}

namespace {

// A solved store that holds back the calling thread's first lookup until the
// root's proof has been exported: the prover's thread 0 runs on the calling
// thread and builds its root after starting the helpers, so this makes certain
// what a loaded machine makes likely (CI run 37146667584, g++-13 on two
// vCPUs) - a helper proves the root first, and thread 0 reads that proof from
// the store, with no tree under its root.
class LateRootStore final : public SolvedStore {
 public:
  explicit LateRootStore(GameState root) : root_(std::move(root)), caller_(std::this_thread::get_id()) {}
  std::optional<Proof> find(const GameState& position) const override {
    if (std::this_thread::get_id() == caller_ && !held_) {
      held_ = true;  // the calling thread's only: no other thread reads it
      std::unique_lock<std::mutex> lock(mutex_);
      waited_ = cv_.wait_for(lock, std::chrono::seconds(60), [&] { return root_put_; });
    }
    return inner_.find(position);
  }
  void put(const GameState& position, const Proof& proof) override {
    inner_.put(position, proof);
    if (position.zobrist() == root_.zobrist() && same_position(position, root_)) {
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        root_put_ = true;
      }
      cv_.notify_all();
    }
  }
  bool concurrent() const noexcept override { return true; }
  /// Whether the calling thread's first lookup came after the root's export.
  bool held_back() const { return held_ && waited_; }

 private:
  GameState root_;
  std::thread::id caller_;
  ConcurrentSolvedStore inner_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  bool root_put_ = false;
  mutable bool held_ = false;
  mutable bool waited_ = false;
};

}  // namespace

TEST(guard_at_five_threads_a_thread_late_to_its_root_reports_the_line_of_one_that_searched) {
  // The intermittent failure above, made certain (LateRootStore): thread 0
  // decides the root by reading a helper's export, its tree empty beneath it.
  // The prover reported the lowest-numbered thread at the least distance - an
  // empty line, line_complete false, and no first action. Now: the line of a
  // thread whose tree holds it - at depth 1, the thread that proved the root
  // by search - complete, its mate E4's.
  struct Case {
    const char* name;
    GameState after;  // the attacker to move
    int classes;      // Guard's literal class; then the default class and Hunt's prove-only
  };
  std::vector<Case> positions{{"mate_in_one", mate_in_one(), 3}, {"cross_board_mate", cross_board_mate(), 3}};
  {
    // The twin's mate is a J6 twin's: Guard's literal class finds it.
    const GameState s = twin_guard_position();
    const std::vector<Move> action = twin_guard_action(s);
    REQUIRE_EQ(action.size(), std::size_t{2});
    positions.push_back(Case{"twin (Guard's question)", after_action(s, action), 1});
  }
  for (const Case& c : positions) {
    for (int klass = 0; klass < c.classes; ++klass) {
      std::fprintf(stderr, "  %s, class %d\n", c.name, klass);
      GuardOptions g;
      g.max_depth = 1;
      g.threads = 5;
      ProverOptions o = guard_prover_options(g);
      if (klass == 1) {
        o = ProverOptions{};
        o.threads = 5;
        o.depth = 1;
      }
      if (klass == 2) {
        o = ProverOptions{};
        o.threads = 5;
        o.depth = 1;
        o.prove_only = true;
      }
      LateRootStore store(c.after);
      DisproofStore disproofs;
      Prover p(o, &store, &disproofs);
      const ProveResult r = p.prove(c.after, ProveBudget::nodes(2'000'000), 1);
      CHECK(store.held_back());  // the race happened: thread 0 read the root's proof from the store
      REQUIRE(r.status == ProofStatus::Proven);
      CHECK(r.line_complete);
      CHECK_EQ(r.line.size(), std::size_t{1});
      GameState t = c.after;
      CHECK(replay(t, r.line));
      CHECK(mated(t));
      REQUIRE(!r.first_action.empty());
      if (!r.line.empty()) CHECK(r.first_action == r.line.front());
    }
  }
}
