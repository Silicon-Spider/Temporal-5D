# With `-fassume-sane-operators-new-delete` (the default), a store to a `std::deque`'s end is moved past the `operator new` call that throws: the deque is left pointing into its freed map

- **Project:** GCC, https://gcc.gnu.org/bugzilla/ (suggested component: `tree-optimization`; the library code involved is libstdc++'s, unchanged)
- **Tested at:** g++ 16.2.0 (MSYS2 ucrt64 package, `Rev3, Built by MSYS2 project`), target `x86_64-w64-mingw32`, on Windows 11, at `-O2` and `-O3`; run 2026-11-28. Not tried on another target or another GCC release.

## Summary

`std::deque<T>::push_back` (and `emplace_back`) at the end of a full chunk calls, in `_M_push_back_aux` (`bits/deque.tcc`):

1. `_M_reserve_map_at_back()`, which - when the map of chunk pointers is full - calls `_M_reallocate_map`: a new map is allocated, the chunk pointers are copied, **the old map is freed**, and both iterators are pointed into the new map (`_M_impl._M_start._M_set_node(...)`, `_M_impl._M_finish._M_set_node(...)`);
2. `_M_allocate_node()`, a call of the replaceable `operator new` for the new chunk, **which may throw `std::bad_alloc`**;
3. after constructing the element, `_M_impl._M_finish._M_set_node(_M_impl._M_finish._M_node + 1)`.

When the allocation in step 2 throws, the deque must be as step 1 left it: consistent, with one element fewer than the caller wanted (the strong guarantee `push_back` gives at the end of a deque).

Built at `-O2` with g++ 16.2.0 and its defaults, that is not what happens. The stores of step 1 into `_M_impl._M_finish` are not made before the call in step 2; they are merged with step 3's and made once, after the allocation returns. When the allocation throws, `_M_impl._M_finish` still points into the old map, which step 1 has already freed, while `_M_impl._M_start` points into the new one. `size()` then returns garbage, and the destructor walks freed memory.

With `-fno-assume-sane-operators-new-delete` the same source gives the consistent deque. We read that as the optimiser treating the `operator new` call as unable to observe the object - true of the call itself - and then moving stores across it as if the call could not leave the function either. It can: by throwing. On that path the caller's frames, and every destructor run by the unwinding, read the object.

## Reproduction

Two translation units, so that the replacement `operator new` is not visible where the deque is used (no LTO):

```cpp
// alloc.cpp - a replacement operator new that throws at a chosen call.
#include <cstdlib>
#include <new>

long calls = 0;     // allocations since the count was last reset
long fail_at = -1;  // the allocation that throws std::bad_alloc (-1: none)

void* operator new(std::size_t size) {
  if (calls++ == fail_at) throw std::bad_alloc();
  if (void* p = std::malloc(size != 0 ? size : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
```

```cpp
// main.cpp - std::deque::emplace_back when the chunk allocation that follows
// a reallocation of the deque's map throws.
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <new>

extern long calls;
extern long fail_at;

namespace {

// A type local to this translation unit: std::deque<Node>'s members are then
// local too, and each is inlined where it is called once.
struct Node {
  long value = 0;
};

// The elements are made here; the exception leaves this function.
__attribute__((noinline)) void make(std::deque<Node>& d, std::size_t& made) {
  for (int i = 0; i < 20000; ++i) {
    d.emplace_back();
    ++made;
  }
}

}  // namespace

int main() {
  for (long k = 0; k < 40; ++k) {
    std::deque<Node> d;
    std::size_t made = 0;
    calls = 0;
    fail_at = k;  // the k-th allocation made by the emplace_backs throws
    try {
      make(d, made);
    } catch (const std::bad_alloc&) {
    }
    fail_at = -1;
    if (d.size() != made) {
      std::printf("allocation %ld throws: size() is %zu after %zu emplace_backs\n", k, d.size(), made);
      std::fflush(stdout);
      std::_Exit(1);  // the deque's destructor would walk freed memory
    }
  }
  std::puts("every deque is consistent after the failed emplace_back");
  return 0;
}
```

Steps:

```sh
g++ -std=c++20 -O2 alloc.cpp main.cpp -o repro && ./repro
g++ -std=c++20 -O2 -fno-assume-sane-operators-new-delete alloc.cpp main.cpp -o repro-off && ./repro-off
```

**Expected** (both builds): `every deque is consistent after the failed emplace_back`. A `std::deque` of this `Node` starts with a map of 8 chunk pointers and 128 elements to a chunk, the first chunk in the middle of the map; allocations 0 to 3 are the next four chunks, allocation 4 the first new map (at the 640th element) and allocation 5 the chunk that follows it - the first allocation whose failure meets the fault. The deque then holds 639 elements.

**Actual, at the defaults** (run three times on 2026-11-28; the value differs from run to run - it is a difference of pointers into two different maps):

```
allocation 5 throws: size() is 18446744073709278975 after 639 emplace_backs
allocation 5 throws: size() is 308735 after 639 emplace_backs
allocation 5 throws: size() is 229631 after 639 emplace_backs
```

(exit status 1 each time). The same at `-O3`; at `-O1` the deque is consistent.

**Actual, with `-fno-assume-sane-operators-new-delete`:**

```
every deque is consistent after the failed emplace_back
```

**What the program needs to show it.** Two things, both true of the code in which we met it, and neither exotic:

- *The element type is local to the translation unit* (here, in an unnamed namespace). `std::deque<Node>`'s members are then local too, and `_M_push_back_aux` and `_M_reallocate_map` are inlined into their one caller, so the stores of steps 1 and 3 and the allocation between them end up in one function. With `std::deque<int>` `_M_reallocate_map` stays out of line and the deque comes out consistent.
- *The function that makes the elements has no handler of its own* (`make` above; the exception leaves it). With the loop and the `catch` in one function - the same program with `make`'s body written inside `main`'s `try` - the deque comes out consistent: the handler's read of the deque is in sight.

## Where we met it

In a program whose tests inject an allocation failure at every allocation of an operation in turn, by a replacement `operator new` with a countdown, and then check the program's invariants. A `std::deque` of tree nodes (`tree.emplace_back()`) whose chunk allocation failed right after a map reallocation was left as above; the program then failed in the destructor of the object holding the deque, with a heap corruption (`0xC0000374`) or an access violation on freed blocks. The same tests pass in builds by g++ 13 and 14 on Linux, which do not have the option.

## Analysis

`_M_push_back_aux` and `_M_reallocate_map`, as shipped with g++ 16.2.0 (`include/c++/16.2.0/bits/deque.tcc`):

```cpp
	_M_reserve_map_at_back();
	*(this->_M_impl._M_finish._M_node + 1) = this->_M_allocate_node();   // may throw
	__try
	  {
	    _Alloc_traits::construct(this->_M_impl, this->_M_impl._M_finish._M_cur, std::forward<_Args>(__args)...);
	    this->_M_impl._M_finish._M_set_node(this->_M_impl._M_finish._M_node + 1);
	    this->_M_impl._M_finish._M_cur = this->_M_impl._M_finish._M_first;
	  }
```

```cpp
	  _M_deallocate_map(this->_M_impl._M_map, this->_M_impl._M_map_size);

	  this->_M_impl._M_map = __new_map;
	  this->_M_impl._M_map_size = __new_map_size;
	}

      this->_M_impl._M_start._M_set_node(__new_nstart);
      this->_M_impl._M_finish._M_set_node(__new_nstart + __old_num_nodes - 1);
```

The library's order is right: everything `_M_reallocate_map` stores is stored before `_M_allocate_node` is called, so an exception from it leaves a consistent deque. The stores to `_M_impl._M_finish` (its `_M_node`, `_M_first`, `_M_last`) made by the last line of `_M_reallocate_map` are overwritten on the non-throwing path by `_M_set_node(_M_finish._M_node + 1)`. With the assumption that `operator new` neither reads nor writes memory the caller can see, the earlier stores look dead - nothing between them and the later ones reads `_M_impl._M_finish` - and they are removed, or sunk below the call. That is only valid if the call returns. `operator new` is not `noexcept`; its documented failure mode is to throw; and on the exceptional edge the object is read by whoever handles the exception and by every destructor on the way.

The documentation of `-fassume-sane-operators-new-delete` speaks of the replaceable operators not reading or writing global memory other than their own. It does not say that the compiler may assume they do not throw, and a store that is dead only when the callee returns normally is not a store the callee's "sanity" makes dead.

## Suggested direction

We have not read GCC's source for this and do not propose a patch. What the report asks for is that stores to memory reachable from the caller are kept before a call to a (non-`noexcept`) replaceable `operator new` unless they are also dead on the call's exceptional edge - the same treatment any other call that may throw gets.

## Workaround

`-fno-assume-sane-operators-new-delete`. We measured no cost outside run-to-run noise in our program (a search-heavy C++20 code base) with it.
