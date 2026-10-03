// t5d/attacks.hpp - incremental royal-attack maps (design spec §11.1, "The
// redesign after 6b (CHECKPOINT 3)", part 3; register M14, A4). Rule IDs are
// the rulebook's (docs/rules/README.md).
//
// A RoyalAttackMap answers check.hpp's question - can a piece of `attacker`'s,
// standing on one of its PLAYABLE boards (the latest board of a timeline with
// the attacker to move, L5), capture a royal piece of the other colour by the
// move generator's rules? (royal_capture_exists, for_each_royal_capture) - and
// keeps how it found the answer, per SOURCE (one playable board): the royal
// captures from it, and every line from it that stopped at VOID (a place with
// no board: J8 - void blocks every piece but the knight's leap, which still
// cannot land there, J2), with the place it stopped at.
//
// Boards never change (A4) and a new board appears only where there was void.
// So a multiverse that EXTENDS the map's - every board of the map's still in
// its place, more boards added (a move, a pass, a branch) - changes the answer
// only through
//  * sources that are no longer playable (their board superseded): dropped;
//  * new playable boards: scanned as new sources;
//  * the lines that stopped at void exactly where a new board now stands:
//    resumed from it. Every other line is unchanged - a line that stopped at a
//    piece or at a board's edge cannot be reopened, and one that reached a
//    capture crossed no void, so no new board can block it.
// advanced() does exactly that, and a source's record is computed once and
// shared, unchanged, by every map derived from the one that made it: a board
// left unplayed keeps its record from turn to turn, and a superseded board is
// never scanned again - only crossed by lines from others.
//
// Under the rules where that argument fails, advanced() recomputes from
// scratch: Q18's alternative (sliders_cross_void - a new board on void could
// block a slide), Q8's and Q20's (turn_zero_reachable = false,
// inactive_history_is_destination = false - whether a capture may land
// depends on a board being history, or on activity). A multiverse that does
// not extend the map's is also recomputed from scratch. The answer is always
// the from-scratch one (the tests hold it so under every ruleset); the
// incrementality is only what it costs.
//
// A capture's `branching` flag depends on whether the royal's board is still
// a playable board, so it is read from the map's multiverse when the captures
// are reported, never stored.

#ifndef T5D_ATTACKS_HPP
#define T5D_ATTACKS_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/rules.hpp"
#include "t5d/types.hpp"

namespace t5d {

/// What deriving a map cost (advanced, any_capture_in).
struct AttackMapStats {
  /// Sources scanned whole (new playable boards; every source on a recompute).
  std::size_t sources_scanned = 0;
  /// Sources kept as they were (shared).
  std::size_t sources_kept = 0;
  /// Lines resumed from a new board where they had stopped at void.
  std::size_t lines_resumed = 0;
  /// Recomputed from scratch (the rules, or not an extension).
  bool recomputed = false;
};

/// What a map's captures take: royal pieces (check.hpp's question), or every
/// enemy piece (the estimate channel's captures, t5d/estimate.hpp).
enum class CaptureTargets : std::uint8_t { Royals = 0, All };

class RoyalAttackMap {
 public:
  /// No multiverse: no source, no capture.
  RoyalAttackMap() = default;
  /// The map of `multiverse` for `attacker`'s pieces. `inactive_timelines`
  /// (ascending; a GameState's inactive_timelines()) is read only under
  /// inactive_history_is_destination = false (Q20), as check.hpp reads it.
  /// `targets` All: every capture of an enemy piece, the RoyalCapture's
  /// `royal` field holding the piece taken - by the same lines and landing
  /// rules (a pawn's or brawn's captures, J2, J7, J8, L4, Q8, Q14, Q18-Q20;
  /// en passant, which takes a piece not on the square landed on, is not
  /// among them).
  RoyalAttackMap(const Multiverse& multiverse, const RuleOptions& rules, Colour attacker,
                 std::span<const int> inactive_timelines = {},
                 CaptureTargets targets = CaptureTargets::Royals);

  /// The map of `next` (same attacker and rules): incremental when `next`
  /// extends this map's multiverse (see the header comment), else from
  /// scratch. `stats`, if given, says what it cost.
  RoyalAttackMap advanced(const Multiverse& next, std::span<const int> inactive_timelines = {},
                          AttackMapStats* stats = nullptr) const;

  /// advanced(next).any(), without building the map: stops at the first
  /// capture found.
  bool any_capture_in(const Multiverse& next, std::span<const int> inactive_timelines = {},
                      AttackMapStats* stats = nullptr) const;

  /// Whether some royal capture exists (royal_capture_exists).
  bool any() const noexcept { return capture_count_ > 0; }
  std::size_t capture_count() const noexcept { return capture_count_; }
  /// Every capture, in for_each_royal_capture's order (playable boards by
  /// ascending timeline, pieces by ascending square, directions in table
  /// order), its branching flag read from the map's multiverse.
  std::vector<RoyalCapture> captures() const;

  Colour attacker() const noexcept { return attacker_; }
  CaptureTargets targets() const noexcept { return targets_; }
  /// The number of sources (the attacker's playable boards).
  std::size_t source_count() const noexcept { return sources_.size(); }
  /// The lines that stopped at void, over every source.
  std::size_t stop_count() const noexcept;
  bool empty() const noexcept { return !mv_; }
  /// The map's multiverse (nullptr for an empty map).
  const Multiverse* multiverse() const noexcept { return mv_.get(); }

  struct Source;  // one playable board's record (src/attacks.cpp)

 private:
  RoyalAttackMap(std::shared_ptr<const Multiverse> mv, const RuleOptions& rules, Colour attacker,
                 std::vector<int> inactive, CaptureTargets targets);
  void scan_all();
  // The work advanced() and any_capture_in() share; `out` null: stop at the
  // first capture, building nothing (returns whether one was found).
  bool derive(const Multiverse& next, std::span<const int> inactive, RoyalAttackMap* out,
              AttackMapStats* stats) const;

  std::shared_ptr<const Multiverse> mv_;
  RuleOptions rules_{};
  Colour attacker_ = Colour::White;
  CaptureTargets targets_ = CaptureTargets::Royals;
  std::vector<int> inactive_;
  std::vector<std::shared_ptr<const Source>> sources_;  // ascending timeline
  std::size_t capture_count_ = 0;
};

/// Whether the incremental path applies under `rules` (see the header
/// comment): not under Q8's, Q18's or Q20's alternative.
bool attack_maps_incremental(const RuleOptions& rules) noexcept;

}  // namespace t5d

#endif  // T5D_ATTACKS_HPP
