// Helpers shared by the tests: a deterministic random generator and random
// boards. The generator is the tests' own (xorshift64*), independent of the
// engine's key schedule, and the same on every platform and compiler (the
// standard library's distributions are not).

#ifndef T5D_TEST_SUPPORT_HPP
#define T5D_TEST_SUPPORT_HPP

#include <cstdint>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/types.hpp"

namespace t5d_test {

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed == 0 ? 0x9E3779B97F4A7C15ull : seed) {}

  std::uint64_t next() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return state_ * 0x2545F4914F6CDD1Dull;
  }
  /// Uniform enough in [0, n), n >= 1.
  int below(int n) { return static_cast<int>((next() >> 11) % static_cast<std::uint64_t>(n)); }
  /// In [lo, hi].
  int between(int lo, int hi) { return lo + below(hi - lo + 1); }
  bool chance(int percent) { return below(100) < percent; }

 private:
  std::uint64_t state_;
};

inline t5d::Piece random_piece(Rng& rng) {
  const auto kind = static_cast<t5d::PieceKind>(rng.below(t5d::kPieceKindCount));
  const auto colour = rng.below(2) == 0 ? t5d::Colour::White : t5d::Colour::Black;
  return t5d::Piece(kind, colour, rng.below(2) == 0);
}

/// A random board: each square holds a random piece with probability
/// `density` percent.
inline t5d::BoardRef random_board(Rng& rng, t5d::BoardSize size, int density = 35) {
  t5d::BoardBuilder builder(size);
  for (int rank = 0; rank < size.height; ++rank) {
    for (int file = 0; file < size.width; ++file) {
      if (rng.chance(density)) builder.put(file, rank, random_piece(rng));
    }
  }
  return builder.publish();
}

inline t5d::BoardSize random_size(Rng& rng) {
  return t5d::BoardSize{rng.between(1, 8), rng.between(1, 8)};
}

}  // namespace t5d_test

#endif  // T5D_TEST_SUPPORT_HPP
