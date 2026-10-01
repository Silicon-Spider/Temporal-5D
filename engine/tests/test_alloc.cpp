// expand() is allocation-free in steady state (t5d/expand.hpp): once an
// Expansion's buffers have grown, expanding positions no larger than those it
// has seen allocates nothing. Counted by replacing the global operator new of
// the test runner (this file only: replacement functions are program-wide, and
// counting changes nothing else).

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#include "synthetic.hpp"
#include "t5d/expand.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"

namespace {
std::atomic<long long> g_allocations{0};
}  // namespace

void* operator new(std::size_t size) {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace t5d;

TEST(expand_allocates_nothing_in_steady_state) {
  synthetic::Spec spec;
  spec.seed = 9;
  spec.target_boards = 160;
  spec.target_active = 6;
  spec.snapshots = {60};
  const synthetic::Game g = synthetic::play_until_reached(spec);
  REQUIRE_EQ(g.snapshots.size(), std::size_t{1});
  const GameState& big = g.state;
  const GameState& small = g.snapshots[0].state;
  for (const SquareFeatures features :
       {SquareFeatures::All, SquareFeatures::Scope, SquareFeatures::None}) {
    ExpandOptions o;
    o.attack = o.defence = o.xray = true;
    o.square_features = features;
    Expansion x;
    expand(big, o, x);  // the buffers grow to the larger position
    (void)x.window_depth();  // the per-board arrays grow too
    expand(small, o, x);
    const long long before = g_allocations.load();
    long long boards = 0;
    for (int i = 0; i < 3; ++i) {
      expand(small, o, x);
      boards += static_cast<long long>(x.board_parent().size());  // computed on access
      expand(big, o, x);
      boards += static_cast<long long>(x.board_playable().size());
    }
    CHECK_EQ(g_allocations.load() - before, 0LL);
    CHECK(boards > 0);
    // Without relations too.
    o.attack = o.defence = o.xray = false;
    expand(big, o, x);
    const long long again = g_allocations.load();
    expand(small, o, x);
    expand(big, o, x);
    CHECK_EQ(g_allocations.load() - again, 0LL);
  }
  // The counter counts: a first expansion allocates.
  const long long before = g_allocations.load();
  Expansion fresh;
  expand(big, ExpandOptions{}, fresh);
  CHECK(g_allocations.load() - before > 0);
}
