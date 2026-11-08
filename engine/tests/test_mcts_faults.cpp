// The main search under allocation failure, the second round (t5d/mcts.hpp;
// build-log issue 600's review round): the tests of the review of lane
// fix535's c61ed3a - most of them its own probes, adopted (its mutation pass,
// P1-P13) - each section naming the finding it answers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "t5d/assets.hpp"
#include "t5d/ladder.hpp"
#include "t5d/mcts.hpp"
#include "t5d/search.hpp"
#include "t5d/turngen.hpp"
#include "t5d/variants.hpp"
#include "test_alloc.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

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

// A split, tree or graph, the verification ladder off (its jobs answer on
// other threads, and what arrives when changes what a call allocates).
SearchParams base(TurnSplit split, bool graph) {
  SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.batch_size = 8;
  p.split = split;
  p.graph = graph;
  p.hunt = false;
  p.settling = false;
  p.asset_map = false;
  return p;
}

const char* split_name(TurnSplit s) {
  return s == TurnSplit::Flat ? "flat" : s == TurnSplit::Tree ? "tree-form" : "ply";
}

// The allocations `f` makes on this thread: test_alloc's countdown armed out
// of reach, and read back.
template <class F>
long long allocations_of(F&& f) {
  constexpr long long kFar = std::numeric_limits<long long>::max();
  const t5d_test::alloc::FailAfter far(kFar);
  f();
  return kFar - t5d_test::alloc::fail_countdown;
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

std::uint64_t sum_edges(const Search& s) {
  std::uint64_t e = 0;
  for (NodeId id = 0; id < s.node_count(); ++id) e += s.node(id).edge_count;
  return e;
}

// D2's flat form, after a fault: what is wrong with the nodes - one count per
// stuck node (every turn start here has turns), per node whose priors are not
// the softmax of its turn scores (FlatPrior::Generated, no steering: each
// turn's logit is its score, as a float), per node marked complete while a
// full listing has more turns.
std::size_t flat_wrongs(const Search& s) {
  struct Even final : public TurnScorer {
    double leave(int) const override { return 0.0; }
    double move(const Move&, int) const override { return 0.0; }
  };
  std::size_t wrong = 0;
  for (NodeId id = 0; id < s.node_count(); ++id) {
    const NodeView v = s.node(id);
    if (v.stuck) ++wrong;
    if (v.expanded && v.edge_count > 0) {
      double sum = 0;
      double top = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < v.edge_count; ++i) top = std::max(top, s.edge(id, i).logit);
      double z = 0;
      for (std::size_t i = 0; i < v.edge_count; ++i) z += std::exp(s.edge(id, i).logit - top);
      bool off = false;
      for (std::size_t i = 0; i < v.edge_count; ++i) {
        const EdgeView e = s.edge(id, i);
        sum += e.prior;
        const double want = std::exp(e.logit - top) / z;
        if (!(e.prior > 0) || std::abs(e.prior - want) > 1e-5 * want + 1e-12) off = true;
      }
      if (!(std::abs(sum - 1.0) < 1e-9) || off) ++wrong;
    }
    if (!v.expanded || !v.children_complete) continue;
    const Even even;
    TurnGenerator listing(*v.state, even, s.params().flat_unit);
    std::size_t all = 0;
    while (listing.next()) ++all;
    if (all != v.edge_count) ++wrong;
  }
  return wrong;
}

// The ladder tests' asset position (test_ladder.cpp): the White queen's board
// in turn 1 flagged, the king's travel move onto it boosted.
GameState asset_position() {
  return t5d_test::game({"[k4/5/5/5/Q3K:0:1:w]", "[k4/5/5/5/4K:0:1:b]", "[k4/5/5/5/3K1:0:2:w]"}, BoardSize{5, 5});
}

// The synchronous ladder with its asset map and no job (test_ladder.cpp's
// sync_params with Hunt and the settling scans off), under the flat form.
SearchParams flat_steered() {
  SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.synchronous = true;
  p.hunt = false;
  p.settling = false;
  p.split = TurnSplit::Flat;
  return p;
}

}  // namespace

// ---- L8: every fault that fires comes out of the call -------------------------------------------

TEST(mcts_fault_every_injected_fault_surfaces) {
  // Nothing in the search catches an allocation failure for good: every catch
  // puts back what it changed and rethrows (the C API turns it into
  // T5D_ERR_MEMORY). A call that returns after the injected fault fired
  // swallowed it (review finding L8; the review's P7). Every allocation of a
  // gather, a deliver, advance and advance_to (strided past 1,500 points),
  // each on a fresh search, under every split - the ladder off, and on with
  // its asset map and its asynchronous jobs (their own threads' allocations
  // take no part in this thread's countdown; the asset map, on this thread,
  // rethrows since review finding L12).
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  const GameState far = after("Standard - Two Timelines", 3);
  struct Arm {
    TurnSplit split;
    bool graph;
    bool ladder;
  };
  for (const Arm& arm : {Arm{TurnSplit::BoardMove, true, false}, Arm{TurnSplit::BoardMove, false, false},
                         Arm{TurnSplit::Tree, false, false}, Arm{TurnSplit::Flat, true, false},
                         Arm{TurnSplit::Flat, false, false}, Arm{TurnSplit::BoardMove, false, true},
                         Arm{TurnSplit::Flat, false, true}}) {
    SearchParams p = base(arm.split, arm.graph);
    if (arm.ladder) {
      p.hunt = true;
      p.settling = true;
      p.asset_map = true;
      p.verification_threads = 1;
    }
    const GameState root = start("Standard - Two Timelines");
    const auto prepared = [&] {
      Search s(root, p);
      s.run(*uniform, Limits{24});
      return s;
    };
    std::size_t faults = 0;
    for (int op = 0; op < 4; ++op) {
      const std::size_t leaves = arm.split == TurnSplit::Flat ? 2 : 8;
      const auto setup = [&](Search& s, std::vector<Evaluation>& ev) {
        if (op != 1) return;
        (void)s.gather(leaves);
        ev.resize(s.requests().size());
        uniform->evaluate(s.requests(), ev);
      };
      const auto act = [&](Search& s, std::vector<Evaluation>& ev) {
        switch (op) {
          case 0: (void)s.gather(leaves); break;
          case 1: s.deliver(ev); break;
          case 2: s.advance(s.choose().index); break;
          default: s.advance_to(far, 0); break;
        }
      };
      long long total = 0;
      {
        Search s = prepared();
        std::vector<Evaluation> ev;
        setup(s, ev);
        if (op == 1 && ev.empty()) continue;
        total = allocations_of([&] { act(s, ev); });
      }
      const long long stride = total > 3000 ? 1 + total / 1500 : 1;
      for (long long k = 0; k < total; k += stride) {
        Search s = prepared();
        std::vector<Evaluation> ev;
        setup(s, ev);
        const bool threw = faulted(k, [&] { act(s, ev); });  // a fault that fired, thrown
        if (op == 0 && !threw) s.abandon();
        faults += threw ? 1u : 0u;
        CHECK_NOTHROW(s.check_invariants());
      }
    }
    std::printf("  %s %s, ladder %s: %zu faults, each thrown\n", split_name(arm.split), arm.graph ? "graph" : "tree",
                arm.ladder ? "on" : "off", faults);
    CHECK(faults > 0);
  }
}

// ---- L12: the asset map's allocation failures, through the search -------------------------------

TEST(mcts_fault_the_asset_maps_failures_reach_the_caller) {
  // assess_board caught every std::exception - std::bad_alloc included - so a
  // failure inside the asset map's assessment never reached the search's
  // caller, and the map kept the material-only fallback for the ladder's life
  // (review finding L12; the map's own test is in test_assets.cpp). Through
  // the search - the asset map on, the flat form steering every expansion -
  // every allocation of a delivery after 0 to 8 playouts, each on a fresh
  // search: each fault that fires is thrown (faulted's check), and the search
  // keeps every invariant.
  const GameState s0 = asset_position();
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  const SearchParams p = flat_steered();
  std::size_t faults = 0;
  for (const std::uint64_t n : {0u, 1u, 2u, 3u, 5u, 8u}) {
    const auto prepared = [&] {
      Search s(s0, p);
      if (n > 0) s.run(*uniform, Limits{n});
      (void)s.gather(1);
      return s;
    };
    long long total = 0;
    {
      Search s = prepared();
      std::vector<Evaluation> ev(s.requests().size());
      if (ev.empty()) continue;
      uniform->evaluate(s.requests(), ev);
      total = allocations_of([&] { s.deliver(ev); });
    }
    for (long long k = 0; k < total; ++k) {
      Search s = prepared();
      std::vector<Evaluation> ev(s.requests().size());
      uniform->evaluate(s.requests(), ev);
      faults += faulted(k, [&] { s.deliver(ev); }) ? 1u : 0u;  // each fault thrown
      CHECK_NOTHROW(s.check_invariants());
    }
  }
  std::printf("  the asset map in the search: %zu delivery faults, each thrown\n", faults);
  CHECK(faults > 0);
}

// ---- M1: a batch whose request list or buffers fail ---------------------------------------------

TEST(mcts_fault_a_batch_whose_requests_or_buffers_fail_is_abandoned) {
  // gather filled its request list after its try, and run sized its evaluation
  // buffers before its own: a failure there left the batch waiting - the
  // leaves pending, virtual loss on their paths, the requests cut short - and
  // every later gather and run refused ("a batch is waiting") until abandon
  // (review finding M1; the list grows whenever a gather asks more than any
  // before it). Every allocation of a fresh search's first gather, of its
  // second, and of run(16), each on a fresh search: the search keeps every
  // invariant and the next gather or run goes through, with no abandon.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  struct Arm {
    TurnSplit split;
    bool graph;
    const char* variant;
  };
  for (const Arm& arm : {Arm{TurnSplit::BoardMove, false, "Standard - Two Timelines"},
                         Arm{TurnSplit::BoardMove, true, "Standard - Two Timelines"},
                         Arm{TurnSplit::Tree, false, "Standard - Two Timelines"},
                         Arm{TurnSplit::Flat, false, "Standard"}, Arm{TurnSplit::Flat, true, "Standard"}}) {
    const SearchParams p = base(arm.split, arm.graph);
    const GameState root = start(arm.variant);
    const auto deliver_all = [&](Search& s) {
      std::vector<Evaluation> ev(s.requests().size());
      if (ev.empty()) return;
      uniform->evaluate(s.requests(), ev);
      s.deliver(ev);
    };
    std::size_t faults[3] = {0, 0, 0};
    long long totals[3] = {0, 0, 0};
    for (int phase = 0; phase < 3; ++phase) {
      // phase 0: the first gather; 1: the second (after the first's batch); 2: run(16).
      const auto prepared = [&] {
        Search s(root, p);
        if (phase == 1) {
          (void)s.gather(8);
          deliver_all(s);
        }
        return s;
      };
      const auto act = [&](Search& s) {
        if (phase == 2) {
          s.run(*uniform, Limits{16});
        } else {
          (void)s.gather(8);
        }
      };
      {
        Search s = prepared();
        totals[phase] = allocations_of([&] { act(s); });
      }
      for (long long k = 0; k < totals[phase]; ++k) {
        Search s = prepared();
        const bool threw = faulted(k, [&] { act(s); });
        if (!threw) continue;
        ++faults[phase];
        CHECK_NOTHROW(s.check_invariants());  // no batch waiting
        if (phase == 2) {
          CHECK_NOTHROW(s.run(*uniform, Limits{16}));  // no abandon first
        } else {
          CHECK_NOTHROW((void)s.gather(8));
          deliver_all(s);
        }
        CHECK_NOTHROW(s.check_invariants());
      }
    }
    std::printf("  %s %s: first gather %lld allocations (%zu faults), second %lld (%zu), run(16) %lld (%zu)\n",
                split_name(arm.split), arm.graph ? "graph" : "tree", totals[0], faults[0], totals[1], faults[1],
                totals[2], faults[2]);
    CHECK(faults[0] > 0);
    CHECK(faults[1] > 0);
    CHECK(faults[2] > 0);
  }
}

// ---- M2: a failed pull leaves its node as it found it -------------------------------------------

TEST(mcts_fault_a_failed_pull_leaves_its_node_as_it_found_it) {
  // pull_turns puts back the node's turns and priors as it found them (the
  // review's P3). Every allocation of one gather of a single leaf, each on a
  // fresh search, against an untouched twin: every node both hold with as many
  // turns in both has the twin's priors and logits bit for bit (a pull that
  // threw, or none), and the root's side tables too; and densely, no node
  // stuck, every node's priors the softmax of its turn scores, no node
  // complete with a turn missing.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const char* variant : {"Standard", "Standard - Two Timelines"}) {
    for (const bool graph : {false, true}) {
      SearchParams p = base(TurnSplit::Flat, graph);
      p.root_noise = true;  // the root's noise-free scores too
      const GameState root = start(variant);
      const std::uint64_t playouts = std::string(variant) == "Standard" ? 60u : 24u;
      const auto prepared = [&] {
        Search s(root, p);
        s.run(*uniform, Limits{playouts});
        return s;
      };
      const Search twin = prepared();
      long long total = 0;
      {
        Search s = prepared();
        total = allocations_of([&] { (void)s.gather(1); });
      }
      std::size_t faults = 0;
      std::size_t changed = 0;
      std::size_t wrong = 0;
      std::size_t compared = 0;
      for (long long k = 0; k < total; ++k) {
        Search s = prepared();
        const bool threw = faulted(k, [&] { (void)s.gather(1); });
        if (!threw) s.abandon();
        faults += threw ? 1u : 0u;
        CHECK_NOTHROW(s.check_invariants());
        wrong += flat_wrongs(s);
        for (NodeId id = 0; id < twin.node_count(); ++id) {
          const NodeView a = s.node(id);
          const NodeView b = twin.node(id);
          if (!a.expanded || !b.expanded || a.edge_count != b.edge_count) continue;
          ++compared;
          for (std::size_t i = 0; i < a.edge_count; ++i) {
            const EdgeView x = s.edge(id, i);
            const EdgeView y = twin.edge(id, i);
            if (std::memcmp(&x.prior, &y.prior, sizeof(double)) != 0 ||
                std::memcmp(&x.logit, &y.logit, sizeof(double)) != 0) {
              ++changed;
              break;
            }
          }
        }
        const std::vector<RootChild> rc = s.root_children();
        const std::vector<RootChild> tc = twin.root_children();
        if (rc.size() == tc.size()) {
          for (std::size_t i = 0; i < rc.size(); ++i) {
            if (rc[i].prior_noise_free != tc[i].prior_noise_free || rc[i].prior_unsteered != tc[i].prior_unsteered) {
              ++changed;
              break;
            }
          }
        }
      }
      std::printf("  flat %s %s: %lld allocations, %zu faults, %zu node comparisons, %zu changed, %zu wrong\n",
                  variant, graph ? "graph" : "tree", total, faults, compared, changed, wrong);
      CHECK_EQ(changed, std::size_t{0});
      CHECK_EQ(wrong, std::size_t{0});
      CHECK(faults > 0);
    }
  }
}

TEST(mcts_fault_a_failed_root_pull_leaves_the_root_as_it_was) {
  // The root's later pulls run flat_priors' root block - I15's and E10's side
  // tables, the per-edge counts grown (review finding M2; the review's P3b).
  // Searches of N playouts, N where the next gather of one leaf widens the root
  // (fault-free); every allocation of that gather, each on a fresh search,
  // against the untouched twin: where the root's turns did not grow, its
  // priors, logits and side tables - noise-free, unsteered, boost, steered
  // visits, PV flags - are the twin's bit for bit; and the checks of the test
  // above.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const bool graph : {false, true}) {
    SearchParams p = base(TurnSplit::Flat, graph);
    p.root_noise = true;
    const GameState root = start("Standard - Two Timelines");
    std::size_t sizes = 0;
    std::size_t faults = 0;
    std::size_t compared = 0;
    std::size_t changed = 0;
    std::size_t wrong = 0;
    for (std::uint64_t n = 8; n < 160 && sizes < 3; ++n) {
      const auto prepared = [&] {
        Search s(root, p);
        s.run(*uniform, Limits{n});
        return s;
      };
      const Search twin = prepared();
      {
        Search s = prepared();
        (void)s.gather(1);
        s.abandon();
        if (s.node(s.root()).edge_count == twin.node(twin.root()).edge_count) continue;
      }
      ++sizes;
      long long total = 0;
      {
        Search s = prepared();
        total = allocations_of([&] { (void)s.gather(1); });
      }
      for (long long k = 0; k < total; ++k) {
        Search s = prepared();
        const bool threw = faulted(k, [&] { (void)s.gather(1); });
        if (!threw) s.abandon();
        faults += threw ? 1u : 0u;
        CHECK_NOTHROW(s.check_invariants());
        wrong += flat_wrongs(s);
        if (s.node(s.root()).edge_count != twin.node(twin.root()).edge_count) continue;
        ++compared;
        const std::vector<RootChild> a = s.root_children();
        const std::vector<RootChild> b = twin.root_children();
        bool same = a.size() == b.size();
        for (std::size_t i = 0; same && i < a.size(); ++i) {
          same = std::memcmp(&a[i].prior, &b[i].prior, sizeof(double)) == 0 &&
                 std::memcmp(&a[i].prior_noise_free, &b[i].prior_noise_free, sizeof(double)) == 0 &&
                 std::memcmp(&a[i].prior_unsteered, &b[i].prior_unsteered, sizeof(double)) == 0 &&
                 std::memcmp(&a[i].boost, &b[i].boost, sizeof(double)) == 0 &&
                 a[i].steered_visits == b[i].steered_visits && a[i].was_pv == b[i].was_pv;
        }
        for (std::size_t i = 0; same && i < s.node(s.root()).edge_count; ++i) {
          const EdgeView x = s.edge(s.root(), i);
          const EdgeView y = twin.edge(twin.root(), i);
          same = std::memcmp(&x.prior, &y.prior, sizeof(double)) == 0 &&
                 std::memcmp(&x.logit, &y.logit, sizeof(double)) == 0;
        }
        if (!same) ++changed;
      }
    }
    std::printf("  flat %s, root noise: %zu search sizes widening the root, %zu faults, %zu roots compared, %zu "
                "changed, %zu wrong\n",
                graph ? "graph" : "tree", sizes, faults, compared, changed, wrong);
    CHECK(sizes > 0);
    CHECK_EQ(changed, std::size_t{0});
    CHECK_EQ(wrong, std::size_t{0});
  }
}

TEST(mcts_fault_a_failed_exhausting_pull_leaves_the_node_incomplete) {
  // The one pull whose rollback must clear children_complete - H37's only guard
  // on this route (review finding M2; the review's P3c): a later pull that
  // generates a node's last turns, so that the generator is exhausted and the
  // node marked complete, and then fails in flat_priors. From Standard (about
  // 20 turns a node: with widening, a node's third visit generates its last),
  // at the default widening: every search of 3 to 9 playouts whose next gather
  // of one leaf completes a node (fault-free), every allocation of that gather
  // on a fresh search - no node complete with a turn missing (a full listing),
  // no node stuck, priors the softmax of the turn scores.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const bool graph : {false, true}) {
    const SearchParams p = base(TurnSplit::Flat, graph);
    const GameState root = start("Standard");
    std::size_t sizes = 0;
    std::size_t faults = 0;
    std::size_t wrong = 0;
    for (std::uint64_t n = 3; n <= 9; ++n) {
      const auto prepared = [&] {
        Search s(root, p);
        s.run(*uniform, Limits{n});
        return s;
      };
      const Search twin = prepared();
      bool completes = false;
      {
        Search s = prepared();
        (void)s.gather(1);
        s.abandon();
        for (NodeId id = 0; id < twin.node_count(); ++id) {
          const NodeView a = twin.node(id);
          if (a.expanded && !a.children_complete && s.node(id).children_complete) completes = true;
        }
      }
      if (!completes) continue;
      ++sizes;
      long long total = 0;
      {
        Search s = prepared();
        total = allocations_of([&] { (void)s.gather(1); });
      }
      for (long long k = 0; k < total; ++k) {
        Search s = prepared();
        const bool threw = faulted(k, [&] { (void)s.gather(1); });
        if (!threw) s.abandon();
        faults += threw ? 1u : 0u;
        CHECK_NOTHROW(s.check_invariants());
        wrong += flat_wrongs(s);
      }
    }
    std::printf("  flat %s: %zu search sizes (3-9 playouts) whose next leaf completes a node, %zu faults, %zu wrong\n",
                graph ? "graph" : "tree", sizes, faults, wrong);
    CHECK(sizes > 0);
    CHECK(faults > 0);
    CHECK_EQ(wrong, std::size_t{0});
  }
}

TEST(mcts_fault_an_undone_expansion_leaves_a_fresh_leaf) {
  // expand_flat's undo leaves "the node a leaf again, as it came" (the review's
  // P6): every allocation of a deliver, each on a fresh search - every node
  // not expanded reads as a node never expanded: no turn, k0 and instantiated
  // at their defaults, its children "complete" (the default), not stuck.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const bool graph : {false, true}) {
    const SearchParams p = base(TurnSplit::Flat, graph);
    const GameState root = start("Standard");
    const auto prepared = [&] {
      Search s(root, p);
      s.run(*uniform, Limits{16});
      (void)s.gather(4);
      return s;
    };
    long long total = 0;
    {
      Search s = prepared();
      std::vector<Evaluation> ev(s.requests().size());
      uniform->evaluate(s.requests(), ev);
      total = allocations_of([&] { s.deliver(ev); });
    }
    std::size_t faults = 0;
    std::size_t unlike = 0;
    for (long long k = 0; k < total; ++k) {
      Search s = prepared();
      std::vector<Evaluation> ev(s.requests().size());
      uniform->evaluate(s.requests(), ev);
      faults += faulted(k, [&] { s.deliver(ev); }) ? 1u : 0u;
      CHECK_NOTHROW(s.check_invariants());
      for (NodeId id = 0; id < s.node_count(); ++id) {
        const NodeView v = s.node(id);
        if (v.expanded) continue;
        if (v.edge_count != 0 || v.k0 != 1.0 || v.instantiated != 0 || !v.children_complete || v.stuck) ++unlike;
      }
    }
    std::printf("  flat %s: a deliver of %lld allocations, %zu faults, %zu unexpanded nodes unlike a fresh leaf\n",
                graph ? "graph" : "tree", total, faults, unlike);
    CHECK_EQ(unlike, std::size_t{0});
    CHECK(faults > 0);
  }
}

// ---- L6: the counts the rollbacks restore -------------------------------------------------------

TEST(mcts_fault_counts_describe_the_nodes_made) {
  // make_node and unmake_last_node restore the counts classify() makes (review
  // finding L6; the review's P1). Every count is recounted from the nodes held:
  // the verdicts by asking the detector again under the search's own budget (a
  // node budget: deterministic), the turn caps and game ends from the nodes'
  // terminals - with a detector given no budget (every verdict Unknown) and a
  // turn cap that caps every turn start after Black's first. Every allocation
  // of one gather, each on a fresh search.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  struct Arm {
    TurnSplit split;
    bool graph;
    std::size_t leaves;
  };
  for (const Arm& arm : {Arm{TurnSplit::BoardMove, true, 8}, Arm{TurnSplit::BoardMove, false, 8},
                         Arm{TurnSplit::Flat, true, 2}, Arm{TurnSplit::Flat, false, 2}}) {
    SearchParams p = base(arm.split, arm.graph);
    p.mate_nodes = 0;
    p.turn_cap.turns = 1;
    const GameState root = start("Standard");
    const auto prepared = [&] {
      Search s(root, p);
      s.run(*uniform, Limits{40});
      return s;
    };
    const auto recount_ok = [&](const Search& s) {
      std::uint64_t unknown = 0;
      std::uint64_t checks = 0;
      std::uint64_t caps = 0;
      std::uint64_t mates = 0;
      std::uint64_t stales = 0;
      for (NodeId id = 0; id < s.node_count(); ++id) {
        const NodeView v = s.node(id);
        if (!v.turn_start) continue;
        ++checks;
        SearchBudget b;
        b.max_nodes = p.mate_nodes;
        b.max_bytes = p.mate_bytes;
        if (detect_mate(*v.state, p.detector, b).verdict == Outcome::Unknown) ++unknown;
        if (v.terminal == Terminal::TurnCap) ++caps;
        if (v.terminal == Terminal::Checkmate) ++mates;
        if (v.terminal == Terminal::Stalemate) ++stales;
      }
      const mcts::SearchStats& st = s.stats();
      return st.nodes_created == s.node_count() && st.mate_checks == checks && st.unknown_verdicts == unknown &&
             st.turn_caps == caps && st.checkmates == mates && st.stalemates == stales;
    };
    {
      const Search s = prepared();
      REQUIRE(recount_ok(s));
      REQUIRE(s.stats().unknown_verdicts > 0);
    }
    long long total = 0;
    {
      Search s = prepared();
      total = allocations_of([&] { (void)s.gather(arm.leaves); });
    }
    std::size_t faults = 0;
    std::size_t bad = 0;
    for (long long k = 0; k < total; ++k) {
      Search s = prepared();
      const bool threw = faulted(k, [&] { (void)s.gather(arm.leaves); });
      if (!threw) s.abandon();
      faults += threw ? 1u : 0u;
      CHECK_NOTHROW(s.check_invariants());
      if (!recount_ok(s)) ++bad;
    }
    std::printf("  %s %s: %lld allocations, %zu faults, %zu with counts off\n", split_name(arm.split),
                arm.graph ? "graph" : "tree", total, faults, bad);
    CHECK_EQ(bad, std::size_t{0});
    CHECK(faults > 0);
  }
}

TEST(mcts_fault_flat_turns_counted_and_drops_reported) {
  // generated_turns counts the turns taken in - in a search never re-rooted,
  // the sum of the nodes' turns, whatever failed - and a dropped generator is
  // always a pull that threw, the throw going on (review finding L6; the
  // review's P2). Every allocation of a gather and of the deliver after it,
  // each on a fresh search, widening on and off.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const bool graph : {false, true}) {
    for (const bool widening : {true, false}) {
      SearchParams p = base(TurnSplit::Flat, graph);
      p.widening = widening;
      const GameState root = start("Standard");
      const auto prepared = [&] {
        Search s(root, p);
        s.run(*uniform, Limits{16});
        return s;
      };
      {
        const Search s = prepared();
        REQUIRE_EQ(s.stats().generated_turns, sum_edges(s));
      }
      long long total_g = 0;
      long long total_d = 0;
      {
        Search s = prepared();
        total_g = allocations_of([&] { (void)s.gather(4); });
        std::vector<Evaluation> ev(s.requests().size());
        uniform->evaluate(s.requests(), ev);
        total_d = allocations_of([&] { s.deliver(ev); });
      }
      std::size_t faults = 0;
      std::size_t unreported = 0;
      std::size_t miscounted = 0;
      for (int phase = 0; phase < 2; ++phase) {
        const long long total = phase == 0 ? total_g : total_d;
        for (long long k = 0; k < total; ++k) {
          Search s = prepared();
          std::vector<Evaluation> ev;
          if (phase == 1) {
            (void)s.gather(4);
            ev.resize(s.requests().size());
            uniform->evaluate(s.requests(), ev);
          }
          const std::uint64_t dropped0 = s.stats().dropped_generators;
          const bool threw = faulted(k, [&] {
            if (phase == 0) {
              (void)s.gather(4);
            } else {
              s.deliver(ev);
            }
          });
          if (!threw && phase == 0) s.abandon();
          faults += threw ? 1u : 0u;
          if (s.stats().dropped_generators != dropped0 && !threw) ++unreported;
          CHECK_NOTHROW(s.check_invariants());
          if (s.stats().generated_turns != sum_edges(s)) ++miscounted;
          s.run(*uniform, Limits{4});
          CHECK_NOTHROW(s.check_invariants());
          if (s.stats().generated_turns != sum_edges(s)) ++miscounted;
        }
      }
      std::printf("  flat %s widening %s: %lld + %lld allocations, %zu faults, %zu drops unreported, %zu miscounted\n",
                  graph ? "graph" : "tree", widening ? "on" : "off", total_g, total_d, faults, unreported, miscounted);
      CHECK_EQ(unreported, std::size_t{0});
      CHECK_EQ(miscounted, std::size_t{0});
      CHECK(faults > 0);
    }
  }
}

TEST(mcts_fault_key_collisions_count_the_nodes_made) {
  // ensure_child's collision branch makes an unshared node and counts it once
  // the node exists (review finding L6; the review's P9). Keys cut to 3 bits
  // (the key_bits seam): collisions everywhere; in a graph never re-rooted the
  // count is the nodes whose identity (key, moves made) an earlier node holds.
  // Every allocation of a gather, each on a fresh search.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
    SearchParams p = base(split, true);
    p.key_bits = 3;
    const GameState root = after("Standard - Two Timelines", 1);
    const std::size_t leaves = split == TurnSplit::Flat ? 2 : 8;
    const auto prepared = [&] {
      Search s(root, p);
      s.run(*uniform, Limits{40});
      return s;
    };
    const auto recount = [](const Search& s) {
      std::vector<std::pair<std::uint64_t, int>> ids;
      for (NodeId id = 0; id < s.node_count(); ++id) ids.emplace_back(s.node(id).key, s.node(id).moves_made);
      std::sort(ids.begin(), ids.end());
      std::uint64_t c = 0;
      for (std::size_t i = 1; i < ids.size(); ++i) c += ids[i] == ids[i - 1] ? 1u : 0u;
      return c;
    };
    {
      const Search s = prepared();
      REQUIRE(s.stats().key_collisions > 0);
      REQUIRE_EQ(s.stats().key_collisions, recount(s));
    }
    long long total = 0;
    {
      Search s = prepared();
      total = allocations_of([&] { (void)s.gather(leaves); });
    }
    std::size_t faults = 0;
    std::size_t off = 0;
    for (long long k = 0; k < total; ++k) {
      Search s = prepared();
      const bool threw = faulted(k, [&] { (void)s.gather(leaves); });
      if (!threw) s.abandon();
      faults += threw ? 1u : 0u;
      CHECK_NOTHROW(s.check_invariants());
      if (s.stats().key_collisions != recount(s)) ++off;
    }
    std::printf("  %s graph, 3-bit keys: %lld allocations, %zu faults, %zu with the collision count off\n",
                split_name(split), total, faults, off);
    CHECK_EQ(off, std::size_t{0});
    CHECK(faults > 0);
  }
}

TEST(mcts_fault_boosted_turns_and_suggestions_count_once) {
  // pull_turns' rollback puts back asset_boosted_edges and suggestion_edges
  // (turn_boost counts each boost a turn it takes in gets): a pull that threw
  // raises them by no more than the turns it kept (review finding L6; the
  // review's P13 through the public interface, and Hunt's suggestions with
  // hunt_suggest on - a turn matches at most one suggestion). The asset
  // position under the flat form, the synchronous ladder; every third
  // allocation (every one of the first and last 64) of a gather of one leaf
  // and of its deliver, each on a fresh search after 15 playouts - and with
  // Hunt's suggestions, the deliver that expands the root as well, though not
  // a fresh search's first gather: Hunt runs inside it (begin_decision), and a
  // fault in its prover's tree is build-log issue 603's (this toolchain's
  // std::deque, not the search's rollback).
  const GameState s0 = asset_position();
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const bool suggest : {false, true}) {
    for (const bool graph : {false, true}) {
      SearchParams p = flat_steered();
      p.graph = graph;
      if (suggest) {
        p.asset_map = false;
        p.hunt_suggest = true;  // Hunt (prove-only) answers before the root's expansion, its suggestions priors
        p.hunt_depth = 3;
        p.sync_job_nodes = 3000;
      }
      const auto counted = [&](const Search& s) {
        return suggest ? s.stats().suggestion_edges : s.stats().asset_boosted_edges;
      };
      std::size_t faults = 0;
      std::size_t over = 0;
      std::uint64_t most = 0;
      for (const std::uint64_t n : {std::uint64_t{0}, std::uint64_t{15}}) {
        const auto prepared = [&] {
          Search s(s0, p);
          if (n > 0) s.run(*uniform, Limits{n});
          return s;
        };
        {
          const Search s = prepared();
          most = std::max(most, counted(s));
        }
        for (int phase = 0; phase < 2; ++phase) {
          // The asset map: after 15 playouts (the review's P13); Hunt's
          // suggestions: the root's own expansion too, not the fresh search's
          // first gather - Hunt's own allocations (issue 603).
          if (n == 0 && (!suggest || phase == 0)) continue;
          long long total = 0;
          {
            Search s = prepared();
            if (phase == 0) {
              total = allocations_of([&] { (void)s.gather(1); });
            } else {
              (void)s.gather(1);
              std::vector<Evaluation> ev(s.requests().size());
              if (ev.empty()) continue;
              uniform->evaluate(s.requests(), ev);
              total = allocations_of([&] { s.deliver(ev); });
            }
          }
          for (long long k = 0; k < total; k += k < 64 || k + 64 >= total ? 1 : 3) {
            Search s = prepared();
            std::vector<Evaluation> ev;
            if (phase == 1) {
              (void)s.gather(1);
              ev.resize(s.requests().size());
              uniform->evaluate(s.requests(), ev);
            }
            const std::uint64_t count0 = counted(s);
            const std::uint64_t turns0 = sum_edges(s);
            const bool threw = faulted(k, [&] {
              if (phase == 0) {
                (void)s.gather(1);
              } else {
                s.deliver(ev);
              }
            });
            if (phase == 0 && !threw) s.abandon();
            faults += threw ? 1u : 0u;
            CHECK_NOTHROW(s.check_invariants());
            if (counted(s) - count0 > sum_edges(s) - turns0) ++over;
            s.run(*uniform, Limits{4});
            CHECK_NOTHROW(s.check_invariants());
            most = std::max(most, counted(s));
          }
        }
      }
      std::printf("  flat %s, %s: %zu faults, up to %llu counted, %zu counted beyond the turns kept\n",
                  graph ? "graph" : "tree", suggest ? "Hunt's suggestions" : "asset map", faults,
                  static_cast<unsigned long long>(most), over);
      CHECK(faults > 0);
      CHECK(most > 0);  // not vacuous: boosts (or suggestions) were counted
      CHECK_EQ(over, std::size_t{0});
    }
  }
}

// ---- L9, L10: a root hook that throws, a failed re-root with the ladder on ----------------------

TEST(mcts_fault_a_root_hook_that_throws_after_the_reroot) {
  // reroot counts stats_.reroots before the root hook, and advance_to takes its
  // node back only if that count did not move: a hook that throws after the
  // re-root leaves the new root in place, its node kept (review finding L9;
  // the review's P4).
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  const GameState root = start("Standard - Two Timelines");
  const GameState far = after("Standard - Two Timelines", 3);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
    for (const bool graph : {false, true}) {
      for (const bool to_far : {false, true}) {
        const SearchParams p = base(split, graph);
        Search s(root, p);
        s.run(*uniform, Limits{40});
        int calls = 0;
        s.set_root_hook([&calls](const GameState&, std::uint64_t) {
          if (calls++ == 0) throw std::runtime_error("the root hook");
        });
        bool threw = false;
        try {
          if (to_far) {
            s.advance_to(far, 0);
          } else {
            s.advance(s.choose().index);
          }
        } catch (const std::runtime_error&) {
          threw = true;
        }
        CHECK(threw);
        CHECK_EQ(s.stats().reroots, std::uint64_t{1});
        CHECK_NOTHROW(s.check_invariants());
        CHECK_EQ(s.stats().nodes_created - s.stats().nodes_collected, std::uint64_t{s.node_count()});
        if (to_far) {
          CHECK_EQ(s.node_count(), std::size_t{1});
          CHECK(s.root_state() == far);
        }
        s.run(*uniform, Limits{10});
        CHECK_NOTHROW(s.check_invariants());
      }
    }
  }
}

TEST(mcts_fault_a_failed_reroot_with_the_ladder_on) {
  // reroot ends the ladder's decision after building its table: a fault before
  // that leaves the decision open, as a search that never tried to re-root has
  // it - the verdicts it takes and the jobs it runs after both search on
  // alike (review finding L10; the review's P5). The synchronous mode (H15)
  // keeps both deterministic; every allocation of advance.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
    SearchParams p;
    p.expand.square_features = SquareFeatures::None;
    p.batch_size = 8;
    p.split = split;
    p.synchronous = true;
    p.sync_job_nodes = 2000;
    const GameState root = start("Standard - Two Timelines");
    const auto prepared = [&] {
      Search s(root, p);
      s.run(*uniform, Limits{40});
      return s;
    };
    const auto line = [](const Search& s) {
      const mcts::SearchStats& t = s.stats();
      char b[256];
      std::snprintf(b, sizeof b, "hunt %llu/%llu settle %llu/%llu/%llu cancelled %llu pv %llu verdicts %zu digest %llx",
                    static_cast<unsigned long long>(t.hunt_runs), static_cast<unsigned long long>(t.hunt_mates),
                    static_cast<unsigned long long>(t.settling_runs), static_cast<unsigned long long>(t.settling_proven),
                    static_cast<unsigned long long>(t.settling_disproven),
                    static_cast<unsigned long long>(t.ladder_cancelled), static_cast<unsigned long long>(t.pv_changes),
                    s.verdicts().size(), static_cast<unsigned long long>(s.digest()));
      return std::string(b);
    };
    Search stayed = prepared();
    const std::size_t index = stayed.choose().index;
    stayed.run(*uniform, Limits{16});
    const std::string want = line(stayed);
    long long total = 0;
    {
      Search s = prepared();
      total = allocations_of([&] { s.advance(index); });
    }
    std::size_t faults = 0;
    std::size_t differ = 0;
    for (long long k = 0; k < total; ++k) {
      Search s = prepared();
      if (!faulted(k, [&] { s.advance(index); })) continue;
      ++faults;
      CHECK_NOTHROW(s.check_invariants());
      s.run(*uniform, Limits{16});
      CHECK_NOTHROW(s.check_invariants());
      if (line(s) != want) ++differ;
    }
    std::printf("  %s, ladder on: a re-root of %lld allocations, %zu faults, %zu unlike the search that never "
                "re-rooted\n",
                split_name(split), total, faults, differ);
    CHECK(faults > 0);
    CHECK_EQ(differ, std::size_t{0});
  }
}
