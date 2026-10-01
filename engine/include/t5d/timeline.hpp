// t5d/timeline.hpp - timeline positions and the notation's timeline labels.
//
// Inside the engine a timeline is an integer position p, and adjacent timelines
// differ by 1. The notation labels them differently in games that start with an
// even number of timelines (notation specification, "Even-numbered starting
// boards"): the central pair is "-0" and "+0", which an integer cannot tell
// apart. The mapping used everywhere:
//
//   odd games:   label n  <->  p = n
//                ... "-2" -2, "-1" -1, "0" 0, "+1" 1, "+2" 2 ...
//   even games:  "+0" <-> 0, "-0" <-> -1, "+n" <-> n, "-n" <-> -(n+1)
//                ... "-1" -2, "-0" -1, "+0" 0, "+1" 1, "+2" 2 ...
//
// So positive labels mean the same p in both kinds of game, and in an even game
// every non-positive label shifts down by one. The spec asks for exactly such
// an index mapping (§12.6, "Relative RoPE": the ±0 pair must not look like two
// timelines at distance zero).
//
// Canonical labels (what the engine writes): "0" for the odd game's central
// timeline, "-0"/"+0" for the even game's central pair, and a sign on every
// other label ("+1", "-1"), as the 5DFEN grammar's board_layer production
// requires.

#ifndef T5D_TIMELINE_HPP
#define T5D_TIMELINE_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace t5d {

/// The largest label magnitude accepted from notation. Far above any real game.
inline constexpr int kMaxTimelineMagnitude = 1 << 20;

/// The positions whose labels stay within kMaxTimelineMagnitude, so that every
/// timeline the engine holds can be written and read back: -M..M in an odd
/// game, -M-1..M in an even one (where "-M" is p = -M-1), M the magnitude cap.
constexpr int min_timeline_position(bool even) noexcept {
  return even ? -kMaxTimelineMagnitude - 1 : -kMaxTimelineMagnitude;
}
inline constexpr int kMaxTimelinePosition = kMaxTimelineMagnitude;

/// A timeline label as written: an optional sign and a magnitude.
/// "0" is {None, 0}; "-0" is {Minus, 0}; "+0" is {Plus, 0}; "+3" is {Plus, 3};
/// "3" is {None, 3}.
struct TimelineLabel {
  enum class Sign : std::uint8_t { None, Plus, Minus };

  Sign sign = Sign::None;
  int magnitude = 0;

  friend constexpr bool operator==(const TimelineLabel&, const TimelineLabel&) noexcept = default;
};

/// Parses a label: "0", "-0", "+0", "+n", "-n", or an unsigned "n" (n >= 1).
/// No leading zeros, nothing else. Throws NotationError if malformed.
///
/// The unsigned form is accepted although the 5DFEN grammar's board_layer
/// production requires a sign: the notation's own note on even-numbered
/// starting boards says "the + sign can be omitted for other integers". It is
/// never written.
TimelineLabel parse_timeline_label(std::string_view text);

/// The label as written ("0", "+0", "-3", "3").
std::string to_string(const TimelineLabel& label);

/// Whether a label fixes the kind of game: "-0" and "+0" occur only in even
/// games (true), the unsigned "0" only in odd games (false). Others: nullopt.
std::optional<bool> implied_even(const TimelineLabel& label) noexcept;

/// The position of a label in an odd (even = false) or even game. Throws
/// NotationError for a label that does not exist in that kind of game ("-0" or
/// "+0" in an odd game, the unsigned "0" in an even one).
int timeline_position(const TimelineLabel& label, bool even);

/// The canonical label of position p. Precondition: p > INT_MIN (the magnitude
/// must fit an int); every position a Multiverse holds is far inside that.
TimelineLabel timeline_label(int position, bool even) noexcept;

/// The canonical label of position p, as text.
std::string timeline_label_string(int position, bool even);

/// Whether p is a central timeline: 0 in an odd game, -1 ("-0") or 0 ("+0")
/// in an even one.
constexpr bool is_central_timeline(int position, bool even) noexcept {
  return even ? (position == -1 || position == 0) : position == 0;
}

}  // namespace t5d

#endif  // T5D_TIMELINE_HPP
