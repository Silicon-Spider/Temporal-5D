// The stores a search's threads share (register H17, H19; spec §11.3):
// ConcurrentSolvedStore (t5d/scanner.hpp) and DisproofStore (t5d/prover.hpp),
// each 64 shards behind a ShardLock - taken by many threads at once from the
// lock's first use, as the scanner's and the prover's threads take a store
// built for their search.
//
// The defect these pin (found 2026-10-17, H17's row in docs/BUILD_LOG.md): with
// MinGW-w64's winpthreads, a std::shared_mutex is a statically initialised
// pthread_rwlock_t that the first call initialises, and a thread that loses
// that race gets EINVAL - which libstdc++'s shared_mutex ignores, so the thread
// ran its critical section without the lock (and then unlocked a lock it did
// not hold). scanner_lazy_smp_store_refuses_contradictions segfaulted in 11 of
// 600 runs. The tests below release many threads at once on fresh locks, round
// after round, so that they meet at each lock's first use: built with the old
// lock, each fails in nearly every run (see the build log); built with
// ShardLock, never.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "t5d/prover.hpp"
#include "t5d/rules.hpp"
#include "t5d/scanner.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_play.hpp"

using namespace t5d;

namespace {

constexpr int kThreads = 8;

// Starts `threads` threads running `body(t)` and releases them together once
// every one is running; joins them.
template <class Body>
void run_together(int threads, Body body) {
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      body(t);
    });
  }
  while (ready.load() < threads) std::this_thread::yield();
  go.store(true, std::memory_order_release);
  for (std::thread& th : pool) th.join();
}

}  // namespace

TEST(shared_store_lock_excludes_at_first_concurrent_use) {
  // The lock itself: 64 fresh locks a round, 8 threads released at once, each
  // taking every lock in the same order - so they meet on each lock's first
  // use - and holding it a few microseconds. Inside, a thread must find no
  // other thread, and no increment made under the lock may be lost.
  constexpr int kLocks = 64;
  constexpr int kRounds = 400;
  constexpr int kHold = 4096;
  struct Slot {
    ShardLock lock;
    std::atomic<int> inside{0};
    std::atomic<int> taken{0};  // read and written as a plain counter, under the lock
  };
  long overlaps = 0;
  long lost = 0;
  for (int round = 0; round < kRounds; ++round) {
    const auto slots = std::make_unique<std::array<Slot, kLocks>>();
    std::atomic<long> overlap{0};
    run_together(kThreads, [&](int) {
      for (Slot& s : *slots) {
        const std::lock_guard<ShardLock> guard(s.lock);
        if (s.inside.fetch_add(1) != 0) overlap.fetch_add(1);
        volatile int sink = 0;
        for (int k = 0; k < kHold; ++k) sink = sink + k;
        s.taken.store(s.taken.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        s.inside.fetch_sub(1);
      }
    });
    overlaps += overlap.load();
    for (const Slot& s : *slots) lost += kThreads - s.taken.load();
  }
  CHECK_EQ(overlaps, 0L);
  CHECK_EQ(lost, 0L);
}

TEST(shared_stores_hold_at_first_concurrent_use) {
  // The stores themselves, fresh each round: 8 threads released at once, each
  // probing and writing every position in the same order (positions spread
  // over the shards), so that they meet on each shard's first use. Every
  // write is kept: the shortest proof of each position, the deepest disproof
  // of each class, one entry per position.
  std::vector<GameState> positions;
  for (const GameState& p : t5d_test::sparse_positions(RuleOptions{}, 3, 8, 4040)) {
    bool fresh = true;
    for (const GameState& q : positions) {
      if (q.zobrist() == p.zobrist() && same_position(q, p)) {
        fresh = false;
        break;
      }
    }
    if (fresh) positions.push_back(p);
  }
  REQUIRE(positions.size() >= 100);
  const std::size_t n = positions.size();
  const std::array<DisproofClass, 2> classes{DisproofClass{true, SeedClass::Checks, CompletionClass::Canonical, 7},
                                             DisproofClass{false, SeedClass::All, CompletionClass::Every, 7}};
  // Thread t's writes for position i.
  const auto proof_of = [](std::size_t i, int t) {
    const int k = static_cast<int>(i % 64) + t;
    return i % 2 == 0 ? Proof::win(1 + k % 5) : Proof::loss(k % 3);
  };
  const auto class_of = [](std::size_t i, int t) { return static_cast<std::size_t>((i + static_cast<std::size_t>(t)) % 2); };
  const auto depth_of = [](std::size_t i, int t) { return 1 + (static_cast<int>(i % 64) + t) % 7; };
  // What the stores must hold once every thread is done.
  std::vector<Proof> best(n);
  std::vector<std::array<int, 2>> deepest(n, std::array<int, 2>{-1, -1});
  for (std::size_t i = 0; i < n; ++i) {
    for (int t = 0; t < kThreads; ++t) {
      const Proof p = proof_of(i, t);
      if (!best[i].proven() || p.distance < best[i].distance) best[i] = p;
      int& d = deepest[i][class_of(i, t)];
      d = std::max(d, depth_of(i, t));
    }
  }
  constexpr int kRounds = 150;
  long wrong = 0;
  long errors = 0;
  for (int round = 0; round < kRounds; ++round) {
    ConcurrentSolvedStore proofs;
    DisproofStore disproofs;
    std::atomic<long> thrown{0};
    run_together(kThreads, [&](int t) {
      try {
        for (std::size_t i = 0; i < n; ++i) {
          const GameState& s = positions[i];
          (void)proofs.find(s);
          proofs.put(s, proof_of(i, t));
          (void)disproofs.find(s, classes[class_of(i, t)]);
          disproofs.put(s, classes[class_of(i, t)], depth_of(i, t));
        }
      } catch (const std::exception&) {
        thrown.fetch_add(1);  // a contradiction no thread wrote: a torn store
      }
    });
    errors += thrown.load();
    if (proofs.size() != n) ++wrong;
    if (disproofs.size() != n) ++wrong;
    for (std::size_t i = 0; i < n; ++i) {
      if (proofs.find(positions[i]) != std::optional<Proof>(best[i])) ++wrong;
      for (std::size_t c = 0; c < 2; ++c) {
        if (disproofs.find(positions[i], classes[c]) != deepest[i][c]) ++wrong;
      }
    }
  }
  CHECK_EQ(errors, 0L);
  CHECK_EQ(wrong, 0L);
}
