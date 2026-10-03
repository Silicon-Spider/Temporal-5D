// The test runner's allocation hooks (tests/test_alloc.cpp replaces the global
// operator new and delete for the whole runner - t5d-tests is linked
// statically, so every allocation passes through them). Per thread, and off
// unless a test turns them on:
//
//  * FailAfter(n): the n-th allocation from now (0: the next) throws
//    std::bad_alloc - fault injection;
//  * Counting: the bytes allocated and not yet freed while it is on - what a
//    structure actually holds on the heap (the size asked for, not malloc's
//    own overhead).

#ifndef T5D_TEST_ALLOC_HPP
#define T5D_TEST_ALLOC_HPP

namespace t5d_test::alloc {

extern thread_local long long fail_countdown;  // -1: off
extern thread_local bool counting;
extern thread_local long long live_bytes;

class FailAfter {
 public:
  explicit FailAfter(long long n) noexcept { fail_countdown = n; }
  ~FailAfter() { fail_countdown = -1; }
  FailAfter(const FailAfter&) = delete;
  FailAfter& operator=(const FailAfter&) = delete;
};

class Counting {
 public:
  Counting() noexcept {
    live_bytes = 0;
    counting = true;
  }
  ~Counting() { counting = false; }
  Counting(const Counting&) = delete;
  Counting& operator=(const Counting&) = delete;
};

}  // namespace t5d_test::alloc

#endif  // T5D_TEST_ALLOC_HPP
