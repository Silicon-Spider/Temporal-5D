// The main search and memory (t5d/mcts.hpp; review findings MEDIUM 7, 8 and
// 10): an allocation failure injected at every point of gather and of deliver
// (test_alloc.hpp's FailAfter) leaves the search consistent and searching on -
// under every split, D2's flat form included, with no node marked complete
// while a turn is missing and no node the root does not reach (build-log issue
// 535) - and so does one injected into a re-root or into the root's expansion,
// whose noise survives it (issue 600); the evaluation cache's byte count and
// the graph's memory estimate against the heap actually held
// (test_alloc.hpp's Counting); the cache purged of the states behind the
// root, so that it stays useful along a game; the node and memory budgets that
// stop a run.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

#include "t5d/mcts.hpp"
#include "t5d/search.hpp"
#include "t5d/turngen.hpp"
#include "t5d/variants.hpp"
#include "test_alloc.hpp"
#include "test_framework.hpp"

using namespace t5d;
using namespace t5d::mcts;

namespace {

GameState start(const char* name) { return variant_start(*find_variant(name)); }

// The allocations `f` makes on this thread: test_alloc's countdown armed out
// of reach, and read back.
template <class F>
long long allocations_of(F&& f) {
  constexpr long long kFar = std::numeric_limits<long long>::max();
  const t5d_test::alloc::FailAfter far(kFar);
  f();
  return kFar - t5d_test::alloc::fail_countdown;
}

// Scores every option alike: a generator over it lists every legal turn.
class Even final : public TurnScorer {
 public:
  double leave(int) const override { return 0.0; }
  double move(const Move&, int) const override { return 0.0; }
};

// Issue 535's false exhaustion: a node marked children_complete holds every
// legal successor of its state. Each such node's turns (D2's flat form) - its
// edges' actions, played and submitted - against a full listing of its
// state's turns by a generator of its own, unbudgeted: as many, and each one
// among them. Returns how many nodes were checked.
std::size_t check_complete_turns(const Search& s) {
  std::size_t checked = 0;
  for (NodeId id = 0; id < s.node_count(); ++id) {
    const NodeView v = s.node(id);
    if (!v.expanded || !v.children_complete) continue;
    const Even even;
    TurnGenerator listing(*v.state, even, s.params().flat_unit);
    std::vector<GameState> all;
    while (std::optional<GeneratedTurn> t = listing.next()) all.push_back(std::move(t->result));
    REQUIRE(listing.exhausted());
    CHECK_EQ(v.edge_count, all.size());
    for (std::size_t i = 0; i < v.edge_count; ++i) {
      GameState next = *v.state;
      for (const Move& m : s.edge(id, i).action) next.apply(m);
      next.submit();
      CHECK(std::any_of(all.begin(), all.end(), [&next](const GameState& a) { return same_position(a, next); }));
    }
    ++checked;
  }
  return checked;
}

// D2's flat form's nodes after a fault: no node stuck - every turn start here
// has legal turns, so a stuck one would be an expansion a fault cut short (its
// first pull threw and the node was left expanded with no turn) - and each
// node's priors the softmax of its turns' scores (FlatPrior::Generated, the
// default; no turn steered in these positions: each turn's logit is its
// score, as a float), summing to 1, also after a pull that threw had its turns
// taken back (flat_priors sets nothing unless it succeeds; review finding M2:
// a sum alone let new turns keep a prior of 0).
void check_flat_nodes(const Search& s) {
  for (NodeId id = 0; id < s.node_count(); ++id) {
    const NodeView v = s.node(id);
    CHECK(!v.stuck);
    if (!v.expanded || v.edge_count == 0) continue;
    double sum = 0;
    double top = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < v.edge_count; ++i) top = std::max(top, s.edge(id, i).logit);
    double z = 0;
    for (std::size_t i = 0; i < v.edge_count; ++i) z += std::exp(s.edge(id, i).logit - top);
    for (std::size_t i = 0; i < v.edge_count; ++i) {
      const EdgeView e = s.edge(id, i);
      sum += e.prior;
      const double want = std::exp(e.logit - top) / z;
      CHECK(e.prior > 0 && std::abs(e.prior - want) <= 1e-5 * want + 1e-12);
    }
    CHECK(std::abs(sum - 1.0) < 1e-9);
  }
}

// `f` with its allocation `k` failing: whether it threw std::bad_alloc - and a
// fault that fired must come out of the call (review finding L8: a failure
// swallowed would never reach the caller as T5D_ERR_MEMORY).
template <class F>
bool faulted(long long k, F&& f) {
  bool threw = false;
  bool fired = false;
  {
    const t5d_test::alloc::FailAfter fault(k);
    try {
      f();
    } catch (const std::bad_alloc&) {
      threw = true;
    }
    fired = t5d_test::alloc::fail_countdown < 0;
  }
  CHECK(threw || !fired);
  return threw;
}

// The counts describe the nodes made (a search never re-rooted): one each, and
// one mate check per turn start (classify) - none for a node not made, or
// taken back.
bool counts_match_nodes(const Search& s) {
  std::uint64_t starts = 0;
  for (NodeId id = 0; id < s.node_count(); ++id) starts += s.node(id).turn_start ? 1u : 0u;
  return s.stats().nodes_created == s.node_count() && s.stats().mate_checks == starts;
}

// The verification ladder off: its jobs answer on other threads, and what
// arrives when changes what gather allocates - the tests that count
// allocations, or compare two searches bit for bit, need it the same each run.
SearchParams without_ladder(SearchParams p) {
  p.hunt = false;
  p.settling = false;
  p.asset_map = false;
  return p;
}

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
  // The ply split in a tree and a graph, and D2's tree form (a tree only), whose
  // inherited expansions happen inside gather.
  for (const int arm : {0, 1, 2}) {
    const bool graph = arm == 1;
    for (long long k = 0; k < 2400; k += 13) {
      SearchParams p = quick();
      p.graph = graph;
      if (arm == 2) p.split = TurnSplit::Tree;
      Search s(start("Standard - Two Timelines"), p);
      s.run(*uniform, Limits{40});
      std::size_t n = 0;
      const bool threw = faulted(k, [&] { n = s.gather(8); });
      if (!threw && n > 0) {
        std::vector<Evaluation> ev(n);
        uniform->evaluate(s.requests(), ev);
        if (faulted(k / 4, [&] { s.deliver(ev); })) ++deliver_faults;
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

TEST(mcts_an_allocation_failure_in_the_flat_form_leaves_the_search_consistent) {
  // The test above for D2's flat form, in a tree and a graph, whose turn
  // generators run inside gather and deliver (the done-means audit, build-log
  // issue 535: section 15.4's "virtual loss always released" was tested under
  // allocation failure for the ply split and the tree form only, and the flat
  // form crashed - a generator that had thrown was asked again, and read past
  // the end of its tables). After every fault no node is marked complete with
  // a turn missing, and some faults land inside the generators (each counted
  // as a dropped generator).
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  std::size_t gather_faults = 0;
  std::size_t deliver_faults = 0;
  std::uint64_t dropped = 0;
  for (const bool graph : {false, true}) {
    // A flat gather of 8 makes some 6,000-8,000 allocations here, its deliver
    // more: past 2,400 the step widens so that deliver's faults are reached.
    for (long long k = 0; k < 40000; k += k < 2400 ? 13 : 97) {
      SearchParams p = quick();
      p.split = TurnSplit::Flat;
      p.graph = graph;
      Search s(start("Standard - Two Timelines"), p);
      s.run(*uniform, Limits{40});
      std::size_t n = 0;
      const bool threw = faulted(k, [&] { n = s.gather(8); });
      if (!threw && n > 0) {
        std::vector<Evaluation> ev(n);
        uniform->evaluate(s.requests(), ev);
        if (faulted(k / 4, [&] { s.deliver(ev); })) ++deliver_faults;
      } else if (threw) {
        ++gather_faults;
      }
      CHECK_NOTHROW(s.check_invariants());  // no batch waiting, no virtual loss, counts consistent
      check_complete_turns(s);
      check_flat_nodes(s);
      s.run(*uniform, Limits{20});
      CHECK_NOTHROW(s.check_invariants());
      check_complete_turns(s);
      check_flat_nodes(s);
      dropped += s.stats().dropped_generators;
    }
  }
  std::printf("  flat form: %zu gather faults, %zu deliver faults, %llu generators dropped\n", gather_faults,
              deliver_faults, static_cast<unsigned long long>(dropped));
  CHECK(gather_faults > 10);
  CHECK(deliver_faults > 10);
  CHECK(dropped > 0);
}

TEST(mcts_an_allocation_failure_never_marks_a_flat_node_complete_with_a_turn_missing) {
  // Issue 535's false exhaustion: a generator that threw had lost the box it
  // was judging, ran on, and reported exhausted() - and its node was marked
  // complete with successors missing (24 of 484 in the audit's sweep), a bound
  // no proof backs (H37). From Standard, whose nodes have about 20 turns: with
  // widening off every expansion generates them all, and with it on a node's
  // second visit generates its last ones - so complete nodes are many, their
  // generators exhausted at the first pull and at a later one. The faults land
  // in both kinds of pull (each counted as a dropped generator), and every
  // node marked complete is checked against a full listing.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  std::size_t faults = 0;
  std::size_t checked = 0;
  std::uint64_t dropped = 0;
  for (const bool widening : {false, true}) {
    for (const bool graph : {false, true}) {
      for (long long k = 0; k < 9000; k += 47) {
        SearchParams p = without_ladder(quick());
        p.split = TurnSplit::Flat;
        p.graph = graph;
        p.widening = widening;
        Search s(start("Standard"), p);
        s.run(*uniform, Limits{16});
        std::size_t n = 0;
        bool threw = faulted(k, [&] { n = s.gather(8); });
        if (!threw && n > 0) {
          std::vector<Evaluation> ev(n);
          uniform->evaluate(s.requests(), ev);
          threw = faulted(k / 2, [&] { s.deliver(ev); });
        }
        faults += threw ? 1u : 0u;
        CHECK_NOTHROW(s.check_invariants());
        checked += check_complete_turns(s);
        check_flat_nodes(s);
        CHECK(counts_match_nodes(s));
        s.run(*uniform, Limits{16});
        CHECK_NOTHROW(s.check_invariants());
        checked += check_complete_turns(s);
        check_flat_nodes(s);
        dropped += s.stats().dropped_generators;
      }
    }
  }
  std::printf("  %zu faults, %llu generators dropped, %zu complete nodes checked\n", faults,
              static_cast<unsigned long long>(dropped), checked);
  CHECK(faults > 100);
  CHECK(dropped > 10);
  CHECK(checked > 1000);
}

TEST(mcts_an_allocation_failure_never_leaves_a_node_the_root_does_not_reach) {
  // Issue 535: ensure_child made a node, then entered it in the graph's table,
  // then linked it - so a fault in the table's insertion left a node no edge
  // reaches ("a node the root does not reach"), in every split's graph. Every
  // allocation of one gather is tried, each on a fresh search (their number
  // counted first, on an identical one): the ply split in a graph and a tree
  // and D2's tree form (a tree only), a gather of 8; the flat form in a graph
  // and a tree, from Standard (its turn starts cheap to judge) and a gather of
  // 2 - each new turn start is judged by the mate detector before it is
  // entered in the table, so a gather of 8 makes some 2,300 allocations.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  struct Arm {
    TurnSplit split;
    bool graph;
    const char* variant;
    std::size_t leaves;
  };
  for (const Arm& arm : {Arm{TurnSplit::BoardMove, true, "Standard - Two Timelines", 8},
                         Arm{TurnSplit::BoardMove, false, "Standard - Two Timelines", 8},
                         Arm{TurnSplit::Tree, false, "Standard - Two Timelines", 8},
                         Arm{TurnSplit::Flat, true, "Standard", 2}, Arm{TurnSplit::Flat, false, "Standard", 2}}) {
    SearchParams p = without_ladder(quick());
    p.split = arm.split;
    p.graph = arm.graph;
    const auto prepared = [&] {
      Search s(start(arm.variant), p);
      s.run(*uniform, Limits{40});
      return s;
    };
    long long total = 0;
    {
      Search s = prepared();
      total = allocations_of([&] { (void)s.gather(arm.leaves); });
    }
    std::size_t faults = 0;
    for (long long k = 0; k < total; ++k) {
      Search s = prepared();
      const bool threw = faulted(k, [&] { (void)s.gather(arm.leaves); });
      if (threw) {
        ++faults;
      } else {
        s.abandon();  // the gather's own work is under test
      }
      CHECK_NOTHROW(s.check_invariants());
      CHECK(counts_match_nodes(s));
      s.run(*uniform, Limits{8});
      CHECK_NOTHROW(s.check_invariants());
    }
    std::printf("  %s split, %s: a gather of %zu, %lld allocations, %zu faults\n",
                arm.split == TurnSplit::BoardMove ? "ply"
                : arm.split == TurnSplit::Flat    ? "flat"
                                                  : "tree-form",
                arm.graph ? "graph" : "tree", arm.leaves, total, faults);
    CHECK(faults > 0);
  }
}

TEST(mcts_an_allocation_failure_in_a_reroot_leaves_the_search_consistent) {
  // A re-root under an allocation failure (build-log issue 600, found beside
  // issue 535): it happens whole or not at all - the tree, the graph's table,
  // the root's side tables and effort label - and a node advance_to made for
  // it is taken back (before, a fault part-way through re-registering the
  // graph's nodes left the table part-filled and the effort label pointing
  // at the old root's edges, and one before the swap left advance_to's node
  // unreachable). Checked against two searches run alike: one that never
  // re-rooted and one whose re-root went through - a failed re-root leaves
  // the first's graph (H15's digest, after both search on), a successful one
  // the second's; the sweep ends on the point past the last allocation, a
  // re-root that goes through, and check_invariants reads the graph's table
  // (review finding M3: the sweep never saw a successful re-root before).
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  const GameState root = start("Standard - Two Timelines");
  const GameState far = after("Standard - Two Timelines", 3);  // no search here reaches it: advance_to makes its node
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
    for (const bool graph : {false, true}) {
      SearchParams p = without_ladder(quick());
      p.split = split;
      p.graph = graph;
      for (const bool to_far : {false, true}) {
        const auto prepared = [&] {
          Search s(root, p);
          s.run(*uniform, Limits{40});
          return s;
        };
        const auto reroot = [&](Search& s) {
          if (to_far) {
            s.advance_to(far, 0);
          } else {
            s.advance(s.choose().index);  // a visited child: it exists, so the re-root alone allocates
          }
        };
        Search stayed = prepared();
        stayed.run(*uniform, Limits{10});
        Search went = prepared();
        reroot(went);
        CHECK_NOTHROW(went.check_invariants());
        went.run(*uniform, Limits{10});
        CHECK_NOTHROW(went.check_invariants());
        long long total = 0;
        {
          Search s = prepared();
          total = allocations_of([&] { reroot(s); });
        }
        std::size_t faults = 0;
        std::size_t through = 0;
        // advance_to tries the root's untaken edges and judges its node
        // (detect_mate) before the re-root: its last allocations are the re-root's.
        for (long long k = 0; k <= total; k += !to_far || k + 64 >= total ? 1 : 53) {
          Search s = prepared();
          const bool threw = faulted(k, [&] { reroot(s); });
          faults += threw ? 1u : 0u;
          through += threw ? 0u : 1u;
          CHECK_NOTHROW(s.check_invariants());
          s.run(*uniform, Limits{10});
          CHECK_NOTHROW(s.check_invariants());
          const Search& like = threw ? stayed : went;
          CHECK_EQ(s.stats().nodes_created - s.stats().nodes_collected, std::uint64_t{s.node_count()});
          CHECK_EQ(s.node_count(), like.node_count());
          CHECK_EQ(s.digest(), like.digest());
          CHECK_EQ(s.root_best_since(), like.root_best_since());
        }
        CHECK(faults > 0);
        CHECK(through > 0);  // the successful re-root, compared with `went`
      }
    }
  }
}

TEST(mcts_root_noise_survives_an_allocation_failure_in_the_roots_expansion) {
  // A fault inside the root's expansion after its noise was drawn: the
  // expansion is undone (the flat form: build-log issue 535; the ply split's
  // was never marked done), and the noise must be drawn again when the root is
  // expanded afresh. Before, the flag saying it was drawn was set with the
  // draw, so the root was expanded again without noise for the rest of the
  // search (issue 600). Every allocation of the root's first expansion (the
  // first batch's deliver; the flat form's generator and first pull, past
  // their first 400, at a stride).
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
    SearchParams p = without_ladder(quick());
    p.split = split;
    p.root_noise = true;
    const GameState root = start("Standard");
    const auto first_batch = [&](Search& s) {
      const std::size_t n = s.gather(8);  // the root alone: the batch waits for its evaluation
      REQUIRE_EQ(n, std::size_t{1});
      std::vector<Evaluation> ev(n);
      uniform->evaluate(s.requests(), ev);
      return ev;
    };
    long long total = 0;
    {
      Search s(root, p);
      const std::vector<Evaluation> ev = first_batch(s);
      total = allocations_of([&] { s.deliver(ev); });
    }
    std::size_t faults = 0;
    for (long long k = 0; k < total; k += k < 400 ? 1 : 17) {
      Search s(root, p);
      const std::vector<Evaluation> ev = first_batch(s);
      faults += faulted(k, [&] { s.deliver(ev); }) ? 1u : 0u;
      CHECK_NOTHROW(s.check_invariants());
      s.run(*uniform, Limits{8});
      CHECK_NOTHROW(s.check_invariants());
      // The noise is in the root's priors: a child's prior is not its noise-free one.
      const std::vector<RootChild> children = s.root_children();
      REQUIRE(!children.empty());
      CHECK(std::any_of(children.begin(), children.end(),
                        [](const RootChild& c) { return c.prior != c.prior_noise_free; }));
    }
    std::printf("  %s split: the root's first expansion makes %lld allocations, %zu faults\n",
                split == TurnSplit::Flat ? "flat" : "ply", total, faults);
    CHECK(faults > 10);
  }
}

TEST(mcts_the_evaluation_cache_counts_the_heap_it_holds) {
  // A cache filled by a search, the search gone: the bytes the cache counts
  // against the heap its entries free when cleared (review B measured the old
  // estimate 1.4 to 2.2 times too low).
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 7);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
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
                  split == TurnSplit::BoardMove ? "ply" : "flat", root.multiverse().board_count(), entries,
                  counted, held, ratio);
      CHECK(ratio > 0.75);
      CHECK(ratio < 1.33);
    }
  }
}

TEST(mcts_the_memory_estimate_follows_the_heap) {
  // The graph's estimate (memory_bytes: nodes, edges, turns, states' own
  // heap, the flat form's generators, the MCGS table) against the heap the
  // search holds, its evaluation cache off.
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 7);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat, TurnSplit::Tree}) {
    for (const bool graph : {false, true}) {
      if (split == TurnSplit::Tree && graph) continue;  // refused (build-log spec issue 301)
      SearchParams p = quick();
      p.split = split;
      p.graph = graph;
      p.eval_cache_entries = 0;
      const GameState root = after("Standard - Two Timelines", 6);
      const t5d_test::alloc::Counting on;
      Search s(root, p);
      s.run(*random, Limits{split != TurnSplit::Flat ? 2000u : 400u});
      const long long live = t5d_test::alloc::live_bytes;
      const double ratio = static_cast<double>(s.memory_bytes()) / static_cast<double>(live);
      std::printf("  graph memory, %s split, %s: %zu nodes, estimate %zu bytes, heap %lld: x%.2f\n",
                  split == TurnSplit::BoardMove ? "ply"
                  : split == TurnSplit::Flat    ? "flat"
                                                : "tree-form",
                  graph ? "graph" : "tree", s.node_count(),
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
      if (p.split == TurnSplit::Flat) {
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
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
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
