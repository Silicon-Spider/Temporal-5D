// The main search and memory (t5d/mcts.hpp; review findings MEDIUM 7, 8 and
// 10): an allocation failure injected at every point of gather and of deliver
// (test_alloc.hpp's FailAfter) leaves the search consistent and searching on;
// the evaluation cache's byte count and the graph's memory estimate against
// the heap actually held (test_alloc.hpp's Counting); the cache purged of the
// states behind the root, so that it stays useful along a game; the node and
// memory budgets that stop a run.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "t5d/mcts.hpp"
#include "t5d/search.hpp"
#include "t5d/variants.hpp"
#include "test_alloc.hpp"
#include "test_framework.hpp"

using namespace t5d;
using namespace t5d::mcts;

namespace {

GameState start(const char* name) { return variant_start(*find_variant(name)); }

// A position `actions` legal actions into a variant (each the first listed).
GameState after(const char* name, int actions) {
  GameState s = start(name);
  for (int a = 0; a < actions; ++a) {
    const ActionList list = list_legal_actions(s, 1, Detector::Hypercuboid);
    if (list.actions.empty()) break;
    s = list.actions.front().result;
  }
  return s;
}

SearchParams quick() {
  SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.batch_size = 8;
  return p;
}

}  // namespace

TEST(mcts_an_allocation_failure_in_gather_or_deliver_leaves_the_search_consistent) {
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  std::size_t gather_faults = 0;
  std::size_t deliver_faults = 0;
  for (const bool graph : {false, true}) {
    for (long long k = 0; k < 2400; k += 13) {
      SearchParams p = quick();
      p.graph = graph;
      Search s(start("Standard - Two Timelines"), p);
      s.run(*uniform, Limits{40});
      bool threw = false;
      std::size_t n = 0;
      {
        const t5d_test::alloc::FailAfter fault(k);
        try {
          n = s.gather(8);
        } catch (const std::bad_alloc&) {
          threw = true;
        }
      }
      if (!threw && n > 0) {
        std::vector<Evaluation> ev(n);
        uniform->evaluate(s.requests(), ev);
        const t5d_test::alloc::FailAfter fault(k / 4);
        try {
          s.deliver(ev);
        } catch (const std::bad_alloc&) {
          ++deliver_faults;
        }
      } else if (threw) {
        ++gather_faults;
      }
      CHECK_NOTHROW(s.check_invariants());  // no batch waiting, no virtual loss, counts consistent
      // ... and it searches on.
      s.run(*uniform, Limits{20});
      CHECK_NOTHROW(s.check_invariants());
    }
  }
  CHECK(gather_faults > 10);
  CHECK(deliver_faults > 10);
}

TEST(mcts_the_evaluation_cache_counts_the_heap_it_holds) {
  // A cache filled by a search, the search gone: the bytes the cache counts
  // against the heap its entries free when cleared (review B measured the old
  // estimate 1.4 to 2.2 times too low).
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 7);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const int actions : {2, 12}) {
      SearchParams p = quick();
      p.split = split;
      const GameState root = after("Standard - Two Timelines", actions);
      const auto cache = std::make_shared<EvalCache>(std::size_t{1} << 30, std::size_t{1} << 40);
      std::size_t counted = 0;
      std::size_t entries = 0;
      long long held = 0;
      {
        const t5d_test::alloc::Counting on;
        {
          Search s(root, p, 0, cache);
          s.run(*random, Limits{split == TurnSplit::BoardMove ? 1500u : 300u});
        }
        counted = cache->bytes();
        entries = cache->size();
        const long long before = t5d_test::alloc::live_bytes;
        cache->clear();
        held = before - t5d_test::alloc::live_bytes;
      }
      REQUIRE(held > 0);
      const double ratio = static_cast<double>(counted) / static_cast<double>(held);
      std::printf("  cache, %s split, %zu boards: %zu entries, counted %zu bytes, held %lld: x%.2f\n",
                  split == TurnSplit::BoardMove ? "ply" : "turn", root.multiverse().board_count(), entries,
                  counted, held, ratio);
      CHECK(ratio > 0.75);
      CHECK(ratio < 1.33);
    }
  }
}

TEST(mcts_the_memory_estimate_follows_the_heap) {
  // The graph's estimate (memory_bytes: nodes, edges, actions, states' own
  // heap, the MCGS table) against the heap the search holds, its evaluation
  // cache off.
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 7);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const bool graph : {false, true}) {
      SearchParams p = quick();
      p.split = split;
      p.graph = graph;
      p.eval_cache_entries = 0;
      const GameState root = after("Standard - Two Timelines", 6);
      const t5d_test::alloc::Counting on;
      Search s(root, p);
      s.run(*random, Limits{split == TurnSplit::BoardMove ? 2000u : 400u});
      const long long live = t5d_test::alloc::live_bytes;
      const double ratio = static_cast<double>(s.memory_bytes()) / static_cast<double>(live);
      std::printf("  graph memory, %s split, %s: %zu nodes, estimate %zu bytes, heap %lld: x%.2f\n",
                  split == TurnSplit::BoardMove ? "ply" : "turn", graph ? "graph" : "tree", s.node_count(),
                  s.memory_bytes(), live, ratio);
      CHECK(ratio > 0.75);
      CHECK(ratio < 1.33);
    }
  }
}

TEST(mcts_the_cache_is_purged_of_states_behind_the_root) {
  // Along a game, searches sharing one small cache (2,000 entries): every
  // search purges the states with fewer boards than its root - they cannot
  // come back - so the cache keeps room for the states ahead, and the last
  // searches still find transpositions in it (the ply split's two orders of
  // Two Timelines' two boards). Without the purge it fills with dead states
  // and stops taking new ones.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  const auto cache = std::make_shared<EvalCache>(2000, std::size_t{1} << 30);
  GameState s = start("Standard - Two Timelines");
  int moves_made = 0;  // A6: the moves made in the root's action
  std::uint64_t purged = 0;
  std::uint64_t late_hits = 0;
  for (int decision = 0; decision < 40; ++decision) {
    SearchParams p = quick();
    Search search(s, p, moves_made, cache);
    search.run(*uniform, Limits{300});
    purged += search.stats().cache_purged;
    if (decision >= 30) late_hits += search.stats().cache_hits;
    // Every entry left is at or ahead of the root.
    CHECK(cache->size() <= cache->capacity());
    const RootChild c = search.root_children().at(search.choose().index);
    if (c.submit) {
      s.submit();
      moves_made = 0;
    } else {
      for (const Move& m : c.moves) s.apply(m);
      moves_made += static_cast<int>(c.moves.size());
      if (p.split == TurnSplit::WholeTurn) {
        s.submit();
        moves_made = 0;
      }
    }
    if (moves_made == 0 && detect_mate(s, Detector::Hypercuboid).verdict != Outcome::Ongoing) break;
  }
  CHECK(purged > 0);
  CHECK(late_hits > 0);
  // A purge leaves nothing with fewer boards than asked.
  const std::size_t boards = s.multiverse().board_count();
  cache->purge_fewer_boards(boards + 1000);
  CHECK_EQ(cache->size(), std::size_t{0});
  CHECK_EQ(cache->bytes(), std::size_t{0});
}

TEST(mcts_the_node_and_memory_budgets_stop_a_run) {
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    SearchParams p = quick();
    p.split = split;
    {
      Search s(start("Standard - Two Timelines"), p);
      Limits l;
      l.playouts = 20000;  // finite: a budget not honoured fails (stops on playouts), never hangs
      l.nodes = 400;
      CHECK(s.run(*uniform, l) == StopReason::Nodes);
      CHECK(s.stats().last_stop == StopReason::Nodes);
      CHECK(s.node_count() >= 400);
      CHECK(s.node_count() < 400 + 8 * (1 + s.node(s.root()).edge_count));  // one batch past it at most
      CHECK_NOTHROW(s.check_invariants());
    }
    {
      Search s(start("Standard - Two Timelines"), p);
      Limits l;
      l.playouts = 20000;  // finite: a budget not honoured fails (stops on playouts), never hangs
      l.bytes = 2u << 20;
      CHECK(s.run(*uniform, l) == StopReason::Memory);
      CHECK(s.memory_bytes() >= l.bytes);
      CHECK(s.memory_bytes() < l.bytes + (1u << 20));
      CHECK_NOTHROW(s.check_invariants());
      // A run within the budget again stops at once.
      const std::uint64_t playouts = s.stats().playouts;
      CHECK(s.run(*uniform, l) == StopReason::Memory);
      CHECK_EQ(s.stats().playouts, playouts);
    }
    {
      Search s(start("Standard - Two Timelines"), p);
      CHECK(s.run(*uniform, Limits{50}) == StopReason::Playouts);
      Limits l;
      l.evaluations = 5;
      CHECK(s.run(*uniform, l) == StopReason::Evaluations);
    }
  }
  CHECK(std::string(stop_reason_name(StopReason::Memory)) == "memory");
}
