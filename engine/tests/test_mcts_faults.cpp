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
