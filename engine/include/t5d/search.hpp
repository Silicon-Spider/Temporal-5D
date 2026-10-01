// t5d/search.hpp - legal actions and mate detection (design spec §11.0 item 3;
// register H37; rulebook C5, C8, C9, C13 and ruling R1).
//
// A LEGAL ACTION (C5) is a sequence of moves by the side to move - at most
// one from each board it starts on (T3), as the state machine allows them
// (t5d/state.hpp: the present, its shift, Q4's default that no move is made
// once the present has shifted) - that ends in a submittable state (T5) in
// which every royal piece of the mover's is safe (C4, R1; t5d/check.hpp).
//
// COUNTING. Two actions are the same when they reach the same position: the
// count is the number of DISTINCT LEGAL SUCCESSOR STATES - the positions
// after submission, compared in full (the same boards in the same places,
// timelines and their creators, the player to move, activity), through the
// Zobrist key with an exact comparison on a key match. The board a timeline
// branched from (Timeline::origin) is not compared: no rule reads it, and
// neither the key nor the fingerprint holds it. Two orders of the same moves
// therefore count once, and two orders of the same branching moves twice when
// the timelines they create differ (J9). The gate's contract says only
// "distinct legal actions (whole turns)"; "distinct" is this engine's reading.
// A count stops at a cap: count == cap means "at least cap".
//
// MATE DETECTION (C5): at the start of a turn, the verdict is
//   none (Outcome::Ongoing)  - a legal action exists (one is returned);
//   checkmate                - none exists and the side to move is in check
//                              (R1, t5d/check.hpp);
//   stalemate                - none exists and it is not in check (C13: a
//                              side with no royal piece is never in check);
//   unknown                  - the search stopped on its budget.
// H37: a checkmate or stalemate verdict is a PROOF - it needs a search that
// examined every action and terminated. A search stopped by a budget (nodes,
// time, memory) answers Unknown, never checkmate or stalemate.
//
// Two detectors, two different algorithms that must agree on every proven
// verdict (spec §11.0 item 3):
//  * NAIVE - depth-first over move sequences: the moves the state machine
//    generates, applied one at a time, with a transposition table of the
//    states reached (two orders of the same moves reach one state, searched
//    once). Fast where legal actions abound; exponential in the number of
//    boards where they are scarce.
//  * HYPERCUBOID - a turn as one SEMIMOVE per playable board ("play nothing
//    here" one of them), searched as a product space with conflict-driven
//    elimination of whole sub-boxes ("hypercuboids") of it: the engine
//    README's "The hypercuboid detector" sets out the design.

#ifndef T5D_SEARCH_HPP
#define T5D_SEARCH_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/state.hpp"

namespace t5d {

/// What a search may spend. A search that reaches any limit stops, and its
/// verdict is Unknown (H37).
struct SearchBudget {
  /// Search steps: one per move applied (to a state of the action being
  /// built) and one per finished action tested for royal safety (C4).
  std::uint64_t max_nodes = std::numeric_limits<std::uint64_t>::max();
  /// Wall-clock time from the search's start (checked every 64 nodes).
  std::optional<std::chrono::nanoseconds> max_time;
  /// Memory: the most positions the search may hold at once (its
  /// transposition table and the successors it has collected).
  std::size_t max_stored = std::numeric_limits<std::size_t>::max();

  static SearchBudget unlimited() noexcept { return SearchBudget{}; }
  static SearchBudget nodes(std::uint64_t n) noexcept {
    SearchBudget b;
    b.max_nodes = n;
    return b;
  }
};

/// What a search did.
struct SearchStats {
  /// Steps spent (see SearchBudget::max_nodes).
  std::uint64_t nodes = 0;
  /// Finished actions tested for royal safety (C4).
  std::uint64_t actions_tested = 0;
  /// The most positions held at once.
  std::size_t stored = 0;
  /// Hypercuboid only: points of the product space evaluated, and regions
  /// (sub-boxes) of it eliminated.
  std::uint64_t points = 0;
  std::uint64_t regions = 0;
  /// The search examined every action - it terminated without a budget
  /// stopping it, and without stopping early at a legal action or a cap.
  bool exhaustive = false;
  /// A budget stopped it: the answer is Unknown.
  bool stopped_by_budget = false;
  double seconds = 0;
};

enum class Detector : std::uint8_t { Naive, Hypercuboid };

/// "naive" or "hypercuboid".
const char* detector_name(Detector detector) noexcept;

/// A mate search's answer (see the header comment).
struct MateResult {
  /// Ongoing ("none"), Checkmate, Stalemate or Unknown. Checkmate and
  /// Stalemate only with stats.exhaustive; Unknown only with
  /// stats.stopped_by_budget.
  Outcome verdict = Outcome::Unknown;
  /// R1's check status at the turn's start (computed whatever the verdict).
  bool in_check = false;
  /// For Ongoing: one legal action, its moves in an order in which they can
  /// be played.
  std::vector<Move> witness;
  SearchStats stats;
};

/// Whether the side to move at `turn_start` has a legal action, by the given
/// detector, within `budget`. turn_start must be at the start of a turn (no
/// move made yet).
MateResult detect_mate(const GameState& turn_start, Detector detector,
                       const SearchBudget& budget = SearchBudget::unlimited());

/// One legal action and the position it reaches.
struct LegalAction {
  /// The moves, in an order in which they can be played.
  std::vector<Move> moves;
  /// The successor state: after submission, the opponent to move.
  GameState result;
};

/// The distinct legal successor states of a turn (see "COUNTING" above).
struct ActionCount {
  /// How many were found: the exact count when stats.exhaustive, at least
  /// `cap` when reached_cap, and (stopped by the budget) a lower bound.
  std::size_t count = 0;
  bool reached_cap = false;
  SearchStats stats;

  /// Whether `count` answers the question "min(count, cap)": the search was
  /// exhaustive or reached the cap.
  bool known() const noexcept { return stats.exhaustive || reached_cap; }
};

/// Counts the distinct legal successor states, stopping at `cap` (cap 0 counts
/// nothing and answers 0 at once).
ActionCount count_legal_actions(const GameState& turn_start, std::size_t cap, Detector detector,
                                const SearchBudget& budget = SearchBudget::unlimited());

/// The legal actions themselves, one per distinct successor state (up to
/// `cap`), sorted by action_text. For each state the action listed is the one
/// with the least canonical text among those the search met (the hypercuboid
/// search meets every one).
struct ActionList {
  std::vector<LegalAction> actions;
  bool reached_cap = false;
  SearchStats stats;
};
ActionList list_legal_actions(const GameState& turn_start, std::size_t cap,
                              Detector detector = Detector::Hypercuboid,
                              const SearchBudget& budget = SearchBudget::unlimited());

/// An action's canonical text: its moves in the 5dpgn export form
/// (t5d/move.hpp), separated by spaces - first the moves that create no
/// timeline, by ascending source timeline, then the branching moves in the
/// order they were made (that order numbers the timelines they create, J9).
/// It names the action; it is not always an order in which the moves could be
/// played (Q4's default puts the move that shifts the present last).
std::string action_text(const std::vector<Move>& moves, bool even);

/// Whether two states are the same position for counting: equal in all but
/// the boards timelines branched from (Timeline::origin). GameState's
/// operator== compares those too.
bool same_position(const GameState& a, const GameState& b) noexcept;

}  // namespace t5d

#endif  // T5D_SEARCH_HPP
