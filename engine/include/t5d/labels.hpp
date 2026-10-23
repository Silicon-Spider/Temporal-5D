// t5d/labels.hpp - the per-square labels of a decision state (design spec
// §6, §12.4; register D4, M20, A7; build-log spec issues 172, 175, 176). Each
// side is read from its own view (t5d/views.hpp; spec issue 172).
//
// THE PER-SQUARE LABELS (register D4: "per-square attack facts (label set a
// placeholder)"; M20: the set "stand[s] as v1"; python/t5d/net/heads.py
// SQUARE_LABELS, in its order). Every square of every board, in the canonical
// board order (ascending timeline, then half-turn; t5d/expand.hpp), relative to
// the ACTING player (A7: "ours" is the acting player's, "theirs" the
// opponent's). Each is 0 or 1; arrival_gives_mate may be kLabelUnknown (its
// mate search stopped by the budget). The readings - each a PLACEHOLDER of the
// committed class, named in the engine README ("Per-square labels") and the
// build log (spec issues 175, 176):
//
//  0. exchange_loses_occupant - the static exchange on an occupied square,
//     made by the occupant's enemy, wins material: the swap algorithm over the
//     enemy's captures of the occupant (its view) and the occupant's side's
//     defences of the square (its view; t5d/movegen.hpp Defence), each side's
//     least valuable piece first, values the scanner's H6 v1 table
//     (t5d/scanner.hpp PieceValues) and royal_value for a royal piece; no
//     x-ray recapturer joins. 1 when it ends above 0.
//  1. reachable_in_two_ours  - some piece of the acting player reaches the
//  2. reachable_in_two_theirs  square in two of its own moves and not in one,
//     the square not its own: from each of its pseudo-legal moves (its view),
//     its moves from where that move put it (THE STATIC READING: the board it
//     landed on as it stands, with the piece on it - moved off its square if
//     the move stayed on its board, the captured piece and a castling rook
//     handled, a promotion made - the rest of the multiverse unchanged;
//     t5d/movegen.hpp append_moves_from).
//  3. arrival_gives_check_ours  - some pseudo-legal move of the side's (its
//  4. arrival_gives_check_theirs  view) onto the square leaves the moved piece
//     able, under the static reading, to capture an enemy royal piece (4.13,
//     C1, C2: a king or a royal queen, on any board). Only the moved piece
//     ("gives check from it"): a discovered attack is not counted.
//  5. royal_zone_ours  - the square is a royal piece's of the side, or one king
//  6. royal_zone_theirs  step from one (t5d/geometry.hpp all_directions: every
//     combination of -1, 0, +1 along file, rank, timeline and turn), on a board
//     that exists; whatever stands there.
//  7. arrival_gives_mate - some pseudo-legal move of the acting player onto the
//     square after which submitting is legal (T5 and C4), and the opponent is
//     then checkmated (C5: in check, R1, with no legal action - E4's
//     hypercuboid detector under SquareLabelOptions::mate_budget).
//
// A move's square is its destination's - for a branching move the history
// board it targets (as t5d/records.h) - and a capture's square the captured
// piece's (en passant: the pawn taken).

#ifndef T5D_LABELS_HPP
#define T5D_LABELS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "t5d/scanner.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/types.hpp"
#include "t5d/views.hpp"

namespace t5d {

// ---------------------------------------------------------------------------
// The per-square labels
// ---------------------------------------------------------------------------

/// The labels, in python/t5d/net/heads.py SQUARE_LABELS' order.
enum SquareLabel : int {
  kExchangeLosesOccupant = 0,
  kReachableInTwoOurs = 1,
  kReachableInTwoTheirs = 2,
  kArrivalGivesCheckOurs = 3,
  kArrivalGivesCheckTheirs = 4,
  kRoyalZoneOurs = 5,
  kRoyalZoneTheirs = 6,
  kArrivalGivesMate = 7,
};
inline constexpr int kSquareLabelCount = 8;
/// A label not decided: arrival_gives_mate where a mate search stopped on its
/// budget and no other move decided the square.
inline constexpr std::uint8_t kLabelUnknown = 255;

struct SquareLabelOptions {
  /// The budget of each mate search (arrival_gives_mate). PLACEHOLDER: 200,000
  /// nodes; SearchBudget::unlimited() is exact.
  SearchBudget mate_budget = SearchBudget::nodes(200'000);
  /// The exchange's piece values: the scanner's H6 v1 table (placeholders,
  /// build-log issue 56) - and a royal piece's, which that table leaves 0.
  PieceValues values{};
  int royal_value = 20'000;
};

struct SquareLabelStats {
  /// First moves read per view (0: the acting player's, 1: the opponent's).
  std::uint64_t first_moves[2] = {0, 0};
  /// Second moves generated per view (the static reading).
  std::uint64_t second_moves[2] = {0, 0};
  /// arrival_gives_mate: the moves after which submitting was legal and the
  /// opponent in check (the mate searches run), and those left Unknown.
  std::uint64_t mate_candidates = 0;
  std::uint64_t mate_searches = 0;
  std::uint64_t mate_unknown = 0;
};

struct SquareLabels {
  BoardSize size{};
  std::size_t board_count = 0;
  Colour acting_player = Colour::White;
  /// board_count x cells x kSquareLabelCount; a cell is rank * width + file
  /// (t5d/records.h).
  std::vector<std::uint8_t> labels;
  SquareLabelStats stats;

  int cells() const noexcept { return size.width * size.height; }
  std::uint8_t at(int board, int cell, int label) const noexcept {
    return labels[(static_cast<std::size_t>(board) * static_cast<std::size_t>(cells()) +
                   static_cast<std::size_t>(cell)) *
                      kSquareLabelCount +
                  static_cast<std::size_t>(label)];
  }
};

/// The per-square labels of `state` (the header comment), into `out`.
void square_labels(const GameState& state, const SquareLabelOptions& options, SquareLabels& out);

/// The labels' names, in their order (python/t5d/net/heads.py SQUARE_LABELS).
const char* square_label_name(int label) noexcept;

}  // namespace t5d

#endif  // T5D_LABELS_HPP
