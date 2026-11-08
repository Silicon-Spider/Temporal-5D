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

}  // namespace

// ---- L8: every fault that fires comes out of the call -------------------------------------------

TEST(mcts_fault_every_injected_fault_surfaces) {
  // Nothing in the search catches an allocation failure for good: every catch
  // puts back what it changed and rethrows (the C API turns it into
  // T5D_ERR_MEMORY). A call that returns after the injected fault fired
  // swallowed it (review finding L8; the review's P7). Every allocation of a
  // gather, a deliver, advance and advance_to (strided past 1,500 points),
  // each on a fresh search, under every split, the ladder off.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  const GameState far = after("Standard - Two Timelines", 3);
  struct Arm {
    TurnSplit split;
    bool graph;
    bool ladder;
  };
  for (const Arm& arm : {Arm{TurnSplit::BoardMove, true, false}, Arm{TurnSplit::BoardMove, false, false},
                         Arm{TurnSplit::Tree, false, false}, Arm{TurnSplit::Flat, true, false},
                         Arm{TurnSplit::Flat, false, false}}) {
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
