// t5d/geometry.hpp - 4D displacements and the pieces' direction tables.
//
// The rules reference's conventions (docs/rules/README.md, 4.0 and L3): a
// displacement is (dx, dy, dL, dT) - file, rank, timeline and turn - with dT
// in WHOLE turns. One turn is two half-turns along a timeline, the distance
// from a player's board to that player's next board, so a step in T changes
// the half-turn index by 2 and never lands on a board of the other colour:
// movement between boards ignores opponent-coloured boards (L4), which are
// therefore never destinations, never counted as steps and never blockers.
//
// A "diagonal" is equal steps along exactly two axes, a "triagonal" along
// three, a "quadragonal" along four (4.0). The tables are computed at compile
// time, in a fixed order (generation order is deterministic):
//
//   kUnitDirections, the 80 vectors of {-1,0,1}^4 other than 0, grouped by
//   the number of axes they move along -
//     [0, 8)    one axis    rook       (4.1:  8 directions)
//     [8, 32)   two axes    bishop     (4.2: 24 directions)
//     [32, 64)  three axes  unicorn    (4.8: 32 directions)
//     [64, 80)  four axes   dragon     (4.9: 16 directions)
//   so the queen and the royal queen slide along all 80 (4.3, 4.11), the
//   princess along the first 32 (rook and bishop, 4.10), and the king and the
//   common king step once along any of the 80 (4.4, 4.12). Within a group the
//   order is lexicographic in (dx, dy, dL, dT), -1 before 0 before +1.
//
//   kKnightLeaps, the 48 knight vectors (4.5): two steps along one axis, then
//   one along a different axis - 12 ordered axis pairs x 4 sign combinations.

#ifndef T5D_GEOMETRY_HPP
#define T5D_GEOMETRY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>

#include "t5d/types.hpp"

namespace t5d {

/// A displacement: file, rank, timeline, turn (T in whole turns).
struct Vec4 {
  std::int8_t x = 0;
  std::int8_t y = 0;
  std::int8_t l = 0;
  std::int8_t t = 0;

  /// The number of axes the displacement moves along.
  constexpr int axes() const noexcept {
    return (x != 0 ? 1 : 0) + (y != 0 ? 1 : 0) + (l != 0 ? 1 : 0) + (t != 0 ? 1 : 0);
  }
  /// Within one board: no timeline or turn component.
  constexpr bool physical() const noexcept { return l == 0 && t == 0; }

  friend constexpr bool operator==(Vec4, Vec4) noexcept = default;
};

namespace geometry {

inline constexpr std::size_t kRookCount = 8;
inline constexpr std::size_t kBishopCount = 24;
inline constexpr std::size_t kUnicornCount = 32;
inline constexpr std::size_t kDragonCount = 16;
inline constexpr std::size_t kUnitCount = 80;
inline constexpr std::size_t kKnightCount = 48;

namespace detail {

constexpr std::array<Vec4, kUnitCount> make_unit_directions() noexcept {
  std::array<Vec4, kUnitCount> out{};
  std::size_t n = 0;
  for (int axes = 1; axes <= 4; ++axes) {
    for (int x = -1; x <= 1; ++x) {
      for (int y = -1; y <= 1; ++y) {
        for (int l = -1; l <= 1; ++l) {
          for (int t = -1; t <= 1; ++t) {
            const Vec4 v{static_cast<std::int8_t>(x), static_cast<std::int8_t>(y),
                         static_cast<std::int8_t>(l), static_cast<std::int8_t>(t)};
            if (v.axes() == axes) out[n++] = v;
          }
        }
      }
    }
  }
  return out;
}

constexpr std::array<Vec4, kKnightCount> make_knight_leaps() noexcept {
  std::array<Vec4, kKnightCount> out{};
  std::size_t n = 0;
  for (int a = 0; a < 4; ++a) {      // the axis of the two steps
    for (int b = 0; b < 4; ++b) {    // the axis of the one step
      if (a == b) continue;
      for (const int sa : {-1, 1}) {
        for (const int sb : {-1, 1}) {
          std::array<std::int8_t, 4> c{};
          c[static_cast<std::size_t>(a)] = static_cast<std::int8_t>(2 * sa);
          c[static_cast<std::size_t>(b)] = static_cast<std::int8_t>(sb);
          out[n++] = Vec4{c[0], c[1], c[2], c[3]};
        }
      }
    }
  }
  return out;
}

}  // namespace detail

inline constexpr std::array<Vec4, kUnitCount> kUnitDirections = detail::make_unit_directions();
inline constexpr std::array<Vec4, kKnightCount> kKnightLeaps = detail::make_knight_leaps();

/// One axis: the rook (4.1).
constexpr std::span<const Vec4> rook_directions() noexcept {
  return std::span<const Vec4>(kUnitDirections.data(), kRookCount);
}
/// Two axes: the bishop (4.2).
constexpr std::span<const Vec4> bishop_directions() noexcept {
  return std::span<const Vec4>(kUnitDirections.data() + kRookCount, kBishopCount);
}
/// Three axes: the unicorn (4.8).
constexpr std::span<const Vec4> unicorn_directions() noexcept {
  return std::span<const Vec4>(kUnitDirections.data() + kRookCount + kBishopCount, kUnicornCount);
}
/// Four axes: the dragon (4.9).
constexpr std::span<const Vec4> dragon_directions() noexcept {
  return std::span<const Vec4>(
      kUnitDirections.data() + kRookCount + kBishopCount + kUnicornCount, kDragonCount);
}
/// Rook and bishop: the princess (4.10).
constexpr std::span<const Vec4> princess_directions() noexcept {
  return std::span<const Vec4>(kUnitDirections.data(), kRookCount + kBishopCount);
}
/// All 80: the queen and the royal queen slide along them (4.3, 4.11); the king
/// and the common king step once along them (4.4, 4.12).
constexpr std::span<const Vec4> all_directions() noexcept {
  return std::span<const Vec4>(kUnitDirections.data(), kUnitCount);
}
/// The 48 knight leaps (4.5).
constexpr std::span<const Vec4> knight_leaps() noexcept {
  return std::span<const Vec4>(kKnightLeaps.data(), kKnightCount);
}

/// The sliding directions of a kind; empty for the kinds that do not slide
/// (pawn, knight, king, brawn, common king).
constexpr std::span<const Vec4> slider_directions(PieceKind kind) noexcept {
  switch (kind) {
    case PieceKind::Rook: return rook_directions();
    case PieceKind::Bishop: return bishop_directions();
    case PieceKind::Unicorn: return unicorn_directions();
    case PieceKind::Dragon: return dragon_directions();
    case PieceKind::Princess: return princess_directions();
    case PieceKind::Queen:
    case PieceKind::RoyalQueen: return all_directions();
    default: return {};
  }
}

/// Pawn and brawn directions (4.6, 4.7). "Forward" on the rank axis is towards
/// the opponent's back rank (white +y, black -y); on the timeline axis it is
/// towards the opponent's timelines (white towards decreasing L, black towards
/// increasing L), whoever created the timeline the piece stands on. File and
/// turn are sideways axes, with no forward.
constexpr int forward_rank(Colour c) noexcept { return c == Colour::White ? 1 : -1; }
constexpr int forward_timeline(Colour c) noexcept { return c == Colour::White ? -1 : 1; }

}  // namespace geometry
}  // namespace t5d

#endif  // T5D_GEOMETRY_HPP
