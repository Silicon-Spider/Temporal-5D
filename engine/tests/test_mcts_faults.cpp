// The main search when an allocation fails (t5d/mcts.hpp; review finding
// MEDIUM 10): a bad_alloc injected at every point of gather and of deliver
// (test_alloc.hpp's FailAfter) leaves the search consistent - the batch
// abandoned, no virtual loss left, what was delivered kept - and the search
// goes on.

#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include "t5d/mcts.hpp"
#include "t5d/variants.hpp"
#include "test_alloc.hpp"
#include "test_framework.hpp"

using namespace t5d;
using namespace t5d::mcts;

namespace {

GameState start(const char* name) { return variant_start(*find_variant(name)); }

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
          threw = true;
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
