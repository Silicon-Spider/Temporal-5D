// expand() is allocation-free in steady state (t5d/expand.hpp): once an
// Expansion's buffers have grown, expanding positions no larger than those it
// has seen allocates nothing. Counted by replacing the global operator new of
// the test runner (this file only: replacement functions are program-wide, and
// counting changes nothing else). The same replacement serves the other
// tests' allocation hooks (test_alloc.hpp): a fault injected at the n-th
// allocation, and the live bytes a structure holds - each block carries a
// 16-byte header with its size and whether it was counted.

#include "test_alloc.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>

#include "synthetic.hpp"
#include "t5d/expand.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"

namespace t5d_test::alloc {
thread_local long long fail_countdown = -1;
thread_local bool counting = false;
thread_local long long live_bytes = 0;
}  // namespace t5d_test::alloc

namespace {
std::atomic<long long> g_allocations{0};

struct Header {
  std::size_t size;
  std::size_t counted;
};
static_assert(sizeof(Header) == 16);
}  // namespace

void* operator new(std::size_t size) {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  namespace a = t5d_test::alloc;
  if (a::fail_countdown >= 0 && a::fail_countdown-- == 0) throw std::bad_alloc();
  void* raw = std::malloc(sizeof(Header) + (size == 0 ? 1 : size));
  if (raw == nullptr) throw std::bad_alloc();
  const Header h{size, a::counting ? std::size_t{1} : std::size_t{0}};
  std::memcpy(raw, &h, sizeof h);
  if (a::counting) a::live_bytes += static_cast<long long>(size);
  return static_cast<char*>(raw) + sizeof(Header);
}
void operator delete(void* p) noexcept {
  if (p == nullptr) return;
  char* raw = static_cast<char*>(p) - sizeof(Header);
  Header h;
  std::memcpy(&h, raw, sizeof h);
  namespace a = t5d_test::alloc;
  if (h.counted != 0 && a::counting) a::live_bytes -= static_cast<long long>(h.size);
  std::free(raw);
}
void operator delete(void* p, std::size_t) noexcept { operator delete(p); }
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete[](void* p) noexcept { operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { operator delete(p); }

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
