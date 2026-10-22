// t5d/turngen.hpp - D2's FLAT FORM's turn generator (design spec §8.2, "Flat
// form"; §11.0 item 3; register D2): the legal turns of a turn start, one at a
// time, BEST FIRST - in non-increasing order of their summed scores.
//
// §8.2: "A generator assembles complete legal turns best-first by summed
// logits, through the rules engine: a move between two present boards resolves
// both; a rewinding branch makes the rest optional; optional boards carry a
// 'leave it' option with its own prior; illegal submissions are skipped as
// they appear. The hypercuboid search already walks this product space for
// mate detection (§11.0 item 3); ordered by prior, it becomes the turn
// generator."
//
// THE SPACE is the hypercuboid detector's (t5d/search.hpp; the engine
// README's "The hypercuboid detector: design"): one AXIS per playable board of
// the mover's at the turn start, one VALUE per option of that board - leave it
// (the detector's Pass), a move within it, one half of a jump between two
// axes (the departure on the source's axis, the arrival on the destination's:
// "a move between two present boards resolves both"), a branching move (J6's
// branching twin of every jump between axes included). A turn is a point - a
// value per axis - that some order of its moves realises; its SCORE is the
// sum over the axes, in axis order (ascending timeline), of its values'
// scores (TurnScorer). Every turn takes exactly one option on every playable
// board, so a turn's score is a sum of one term per board.
//
// THE ORDER. The generator is the detector's search with its stack of boxes
// replaced by a priority queue keyed by each box's BOUND - the sum over its
// axes of the best score left on the axis, an upper bound on every point in
// the box. It takes the box with the highest bound, narrows it as the
// detector does (propagation; outside the general regime the abstract check,
// which removes a box no skeleton of which is realisable), and judges the
// box's best point - each axis's best value, so the point's score is the
// box's bound (re-queued first if narrowing lowered it below another box's).
// Judging is the detector's own (realisation through the state machine, C4's
// test, and the regions that remove whole sub-boxes holding no legal action:
// a jump's halves apart, J6, a royal capture, an unrealisable skeleton); a
// legal point's actions are recorded, and the point alone removed. Since every
// unjudged point lies in a queued box whose bound is at most the judged
// point's score, the points are judged - and the turns emitted - in
// non-increasing score (floating-point sums in a fixed order are monotone).
// The search is complete for the same reason as the detector's: the boxes
// partition what is left, and a region is removed only when it holds no
// legal action not yet recorded.
//
// THE UNIT (TurnUnit). Actions, as count_legal_actions counts them (a set of
// moves, its branching moves in creation order: two creation orders of one
// point are two actions, with equal scores); or distinct successor states, as
// count_legal_successors counts them - the first action reaching each state
// (so its best score) stands for it. Ties are broken deterministically: by
// the order the boxes were queued, and within a point by its realisations'
// order.
//
// BUDGET. The generator spends one SearchBudget over its whole life (its node
// count, memory ceiling and position count; a time limit is measured from its
// construction). A generator the budget stopped is stopped() and never
// exhausted(): turns may remain that it did not emit - never a proof that
// none does (H37).
//
// Written from the spec, the rulebook and the engine's own design note - no
// oracle's code (AGENTS.md; §11.0, "Third-party code": ftxi's fine tree and
// cwmtt are prior art to read about, never to port).

#ifndef T5D_TURNGEN_HPP
#define T5D_TURNGEN_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"

namespace t5d {

/// What a turn generator emits (see the header comment).
enum class TurnUnit : std::uint8_t {
  Actions = 0,     ///< legal actions (count_legal_actions' unit)
  Successors = 1,  ///< distinct legal successor states (count_legal_successors')
};

/// The scores of a turn's options, board by board. Each must be finite; a
/// higher score is a better option. Called during the generator's
/// construction only.
class TurnScorer {
 public:
  virtual ~TurnScorer() = default;
  /// Leaving the playable board of `timeline` unplayed - its "leave it"
  /// option (on a mandatory board, legal only in a turn that rewinds the
  /// present, T8, or reactivates behind it, T12).
  virtual double leave(int timeline) const = 0;
  /// Move `m` as the option of the board of `timeline`: its source board
  /// (timeline == m.from_timeline) - a move within the board, a jump's
  /// departure, a branching move; J6's branching twin of a jump between two
  /// playable boards is passed with its branching flag set and its
  /// coordinates the jump's - or, for a non-branching jump (J4) onto another
  /// playable board, its destination (timeline == m.to_timeline): the
  /// arrival.
  virtual double move(const Move& m, int timeline) const = 0;
};

/// One generated turn.
struct GeneratedTurn {
  /// The moves, in an order in which they can be played.
  std::vector<Move> moves;
  /// The successor state: after submission, the opponent to move.
  GameState result;
  /// The sum of its options' scores, axis by axis in ascending timeline.
  double score = 0;
};

class TurnGenerator {
 public:
  /// A generator of the legal turns of `turn_start` (a turn's start, no move
  /// made: T5's obligations unmet), scored by `scorer`, under `budget` - spent
  /// over the generator's life. The state is copied. Throws
  /// std::invalid_argument for a state whose obligations are met (no turn
  /// start) or a non-finite score.
  TurnGenerator(const GameState& turn_start, const TurnScorer& scorer, TurnUnit unit = TurnUnit::Successors,
                const SearchBudget& budget = SearchBudget::unlimited());
  ~TurnGenerator();
  TurnGenerator(const TurnGenerator&) = delete;
  TurnGenerator& operator=(const TurnGenerator&) = delete;

  /// The next turn: its score is at most every earlier one's and at least
  /// every later one's. nullopt once none is left (exhausted()) or the budget
  /// stopped the search (stopped()).
  std::optional<GeneratedTurn> next();
  /// Every legal turn has been emitted (the search terminated without a
  /// budget stopping it).
  bool exhausted() const noexcept;
  /// The budget stopped the search: turns may remain unemitted.
  bool stopped() const noexcept;
  /// The turns emitted so far.
  std::size_t generated() const noexcept;
  /// An upper bound on the score of every turn not yet emitted (-infinity
  /// once exhausted): the best bound among the queued boxes and the turns
  /// found but not yet emitted.
  double bound() const noexcept;
  /// What the search has done (SearchStats: its nodes, points judged, regions
  /// removed, the most positions held; exhaustive and stopped_by_budget as
  /// exhausted() and stopped()).
  const SearchStats& stats() const noexcept;
  /// Its memory, estimated: the queued boxes, the successor states and keys
  /// it holds, the state copy and its tables.
  std::size_t memory_bytes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace t5d

#endif  // T5D_TURNGEN_HPP
