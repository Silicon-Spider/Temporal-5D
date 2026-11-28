// tools/turngen_time.hpp - what t5d-turngen-time times and how it sums up
// (design spec 15.3: "Turn generation (flat form): time to the first k legal
// turns, and per widening step, by board-count range and mandatory count"),
// apart from the tool so that tests hold the arithmetic.
//
// WHAT IS TIMED. One run of D2's flat form's turn generator (t5d/turngen.hpp)
// at a turn start, as the search's flat form makes one per node:
//  * THE TIME TO THE FIRST k TURNS, k = 1, 8 and 32: from just before the
//    generator is constructed (the state's copy, its axes, the scorer's
//    scores - what a node pays before its first turn) to the return of the
//    next() that gives turn k;
//  * A WIDENING STEP: one next() after the first k0 turns (the search's
//    progressive widening "pulls in the next-best legal turn as visits grow";
//    k0 is at most 16 turns in the search, the default here), each timed from
//    its call to its return - `widen` of them at most.
// A position with fewer than k legal turns, or whose generator its node
// budget stopped first, has no time to k: it is counted, not timed there.
// Whether turn k is reached depends on the position, the scores and the node
// budget alone - never on the clock.
//
// HOW IT IS SUMMED UP. A position is timed once a round, round after round
// over every position (the tool's interleaving); a position's figure is the
// MEDIAN over its rounds, and a cell's - a board-count range and a
// mandatory-board count (turn_cells.hpp) - the median and 95th percentile
// over its positions (the times to k) or over its positions' every widening
// step (the step time).
//
// Information for a cost row, never a verdict (not one of the truth's
// sources: CMakeLists.txt, T5D_TRUTH_SOURCES).

#ifndef T5D_TOOLS_TURNGEN_TIME_HPP
#define T5D_TOOLS_TURNGEN_TIME_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/state.hpp"
#include "t5d/turngen.hpp"
#include "turn_cells.hpp"

namespace t5d::tools::turngen_time {

/// The k of "the first k turns": 1, 8, 32.
inline constexpr int kFirsts = 3;
inline constexpr std::size_t kFirst[kFirsts] = {1, 8, 32};

struct Options {
  /// What the generator emits: the search's flat_unit (its default,
  /// successors).
  TurnUnit unit = TurnUnit::Successors;
  /// The generator's node budget over its life: the search's generator_nodes
  /// (200,000, a placeholder there); 0: none.
  std::uint64_t generator_nodes = 200'000;
  /// The turns before the widening steps: the search generates at most this
  /// many when it expands a node (its k0 cap, 16).
  std::size_t k0 = 16;
  /// The widening steps timed after them.
  std::size_t widen = 16;
};

/// The turns one run asks for at most: the greatest k, or k0 + widen.
std::size_t turns_asked(const Options& options) noexcept;

/// The tool's scores: a logit for every option, a function of the seed and
/// the option's coordinates alone - a move's source, destination and
/// promotion and the board it is scored on (its branching flag left out: J6's
/// twin scores as its jump, as a policy that reads coordinates would score
/// it), leaving a board by its timeline - uniform in [-scale, scale]. With
/// scale 0 every option scores 0 and every turn ties.
class HashScorer final : public TurnScorer {
 public:
  HashScorer(std::uint64_t seed, double scale) noexcept : seed_(seed), scale_(scale) {}
  double leave(int timeline) const override;
  double move(const Move& m, int timeline) const override;

 private:
  double value(std::uint64_t key) const noexcept;
  std::uint64_t seed_;
  double scale_;
};

/// One run of a generator at one turn start.
struct Sample {
  std::size_t turns = 0;   ///< turns emitted (at most turns_asked)
  bool exhausted = false;  ///< the generator ran out of legal turns
  bool stopped = false;    ///< its budget stopped it
  /// Seconds to turn kFirst[i]; negative: fewer turns than that were emitted.
  double first[kFirsts] = {-1, -1, -1};
  /// Seconds of each widening step taken, in order.
  std::vector<double> steps;
  std::uint64_t nodes = 0;  ///< the generator's search steps at the end
};

/// Times one generator at `turn_start` (see the header comment). Throws what
/// TurnGenerator throws (std::invalid_argument for a state that is no turn
/// start).
Sample time_generator(const GameState& turn_start, const TurnScorer& scorer, const Options& options);

/// A position's figures over its rounds.
struct PositionFigures {
  int boards = 0;
  int mandatory = 0;
  /// The first round's: turns, exhausted and stopped are the same every round.
  std::size_t turns = 0;
  bool exhausted = false;
  bool stopped = false;
  /// The median over the rounds of the seconds to turn kFirst[i]; negative:
  /// not reached.
  double first[kFirsts] = {-1, -1, -1};
  /// Per widening step, in order: the median over the rounds of its seconds.
  std::vector<double> steps;
};

/// `rounds`: the position's samples, one a round (the warm-up round left out
/// by the caller); none: no figure.
PositionFigures position_figures(int boards, int mandatory, const std::vector<Sample>& rounds);

/// The figures of a cell - or of a range over every mandatory count.
struct CellFigures {
  std::size_t positions = 0;
  /// Positions that reached turn kFirst[i].
  std::size_t reached[kFirsts] = {0, 0, 0};
  /// Positions whose generator ran out of turns, and whose budget stopped
  /// it, before the turns asked.
  std::size_t exhausted = 0;
  std::size_t stopped = 0;
  /// Seconds to turn kFirst[i], over the positions that reached it.
  cells::Summary first[kFirsts];
  /// Seconds of a widening step, over every step of every position (n: the
  /// steps); and the positions with a step at least.
  cells::Summary step;
  std::size_t widening_positions = 0;
};

/// The figures of the positions in board-count range `range` with mandatory
/// class `cls` (turn_cells.hpp; -1: every class).
CellFigures cell_figures(const std::vector<PositionFigures>& positions, int range, int cls);

}  // namespace t5d::tools::turngen_time

#endif  // T5D_TOOLS_TURNGEN_TIME_HPP
