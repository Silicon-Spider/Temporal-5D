// t5d/assets.hpp - THE SUPERSEDED-BOARD ASSET MAP (stage E9; design spec
// 11.2 rung 4, "The superseded-board asset map", kept by verification v3,
// 11.4; register H10, H6).
//
// Pieces on a superseded board never move: it is a travel DESTINATION, not a
// source. So its assessment is not tactical but a one-off static judgement
// of its value as a BRANCH TARGET - "the opponent is catastrophically weak
// here; travelling in wins" - for the TRAVELLER, the colour to move on it
// (a jump lands on a board of the mover's colour, and the branch it makes
// copies that board).
//
// THE ASSESSMENT (assess_board), per board: its material balance for the
// traveller (H6's v1 piece values, the scanner's - placeholders), then a
// SMALL FIXED QUIESCENCE on the board alone - a one-board position, the
// traveller to move: stand-pat, captures most-valuable-victim first, every
// move of a side whose royal is attacked there, a royal capture worth
// royal_value, at most qsearch_plies deep and qsearch_nodes nodes - on the
// engine's own move generator (no piece of the retired estimate channel).
// A board is FLAGGED when that value reaches flag_margin; its BOOST rises
// with the value to max_boost at boost_scale.
//
// ONCE PER BOARD, EVER. Boards are immutable (A4), so an assessment is
// cached by the board's fingerprint - its content (the coordinate-free
// content hash, then a full comparison) and the colour to move on it - for
// the map's ruleset; a board met again, in this position or any later one of
// the game, is never assessed again. Deterministic: no clock, no threads.
//
// THE OUTPUT, for the search (wired with the search core, the next stage): a
// standing prior boost on travel moves into flagged boards (boost()). It is
// an estimate - a prior, never a bound (H2) - and its steering falls under
// spec 12.2's exclusions like any scanner steering.

#ifndef T5D_ASSETS_HPP
#define T5D_ASSETS_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/move.hpp"
#include "t5d/rules.hpp"
#include "t5d/scanner.hpp"
#include "t5d/state.hpp"

namespace t5d {

struct AssetOptions {
  /// H6 v1: the scanner's piece values (placeholders, build-log issue 56).
  PieceValues values;
  int royal_value = 20000;
  /// The quiescence's small fixed budget (PLACEHOLDERS).
  int qsearch_plies = 4;
  std::uint64_t qsearch_nodes = 128;
  /// Flagged at this value for the traveller (PLACEHOLDER: a minor piece).
  int flag_margin = 300;
  /// The boost: max_boost * min(1, value / boost_scale) on a flagged board
  /// (PLACEHOLDERS: saturating at a queen's worth).
  int boost_scale = 900;
  double max_boost = 0.25;
};

struct BoardAsset {
  /// Whose travel moves land here: the colour to move on the board.
  Colour traveller = Colour::White;
  /// The traveller's material there minus the other side's.
  int material = 0;
  /// The quiescence's value for the traveller (traveller to move).
  int value = 0;
  /// The traveller could capture a royal there at once.
  bool royal_capture = false;
  bool flagged = false;
  double boost = 0;
  std::uint64_t nodes = 0;  // the quiescence's
};

/// One board's assessment (`half_turn` gives the colour to move on it), no
/// cache.
BoardAsset assess_board(const BoardRef& board, int half_turn, BoardSize size, bool even, std::uint32_t ruleset,
                        const RuleOptions& rules, const AssetOptions& options = {});

class AssetMap {
 public:
  explicit AssetMap(const AssetOptions& options = {}, const RuleOptions& rules = {});

  /// Assesses every superseded board of `state` not assessed before (once per
  /// board, ever). Returns how many it assessed now.
  std::size_t refresh(const GameState& state);
  /// The asset of the superseded board at (timeline, half_turn) of `state`,
  /// assessed now if it was not; null if there is no superseded board there.
  const BoardAsset* asset(const GameState& state, int timeline, int half_turn);
  /// The standing prior boost of `move` at `state`: the boost of the
  /// superseded board a travel move lands on, if it is flagged for the
  /// mover; 0 for every other move.
  double boost(const GameState& state, const Move& move);

  /// Boards assessed (the cache's size), and assessments ever run (each board
  /// once: equal).
  std::size_t boards() const noexcept { return entries_.size(); }
  std::uint64_t assessments() const noexcept { return assessments_; }
  const AssetOptions& options() const noexcept { return options_; }

 private:
  struct Entry {
    BoardRef board;
    Colour traveller;
    BoardAsset asset;
  };
  const BoardAsset& lookup(const GameState& state, const BoardRef& board, int half_turn);

  AssetOptions options_;
  RuleOptions rules_;
  std::deque<Entry> entries_;  // stable: asset() hands out pointers into it
  std::unordered_multimap<std::uint64_t, std::size_t> index_;
  std::uint64_t assessments_ = 0;
};

}  // namespace t5d

#endif  // T5D_ASSETS_HPP
