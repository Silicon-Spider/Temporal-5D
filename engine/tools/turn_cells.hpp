// tools/turn_cells.hpp - what the two tools of design spec 15.3's turn rows
// share (t5d-tree-count --by-range: "Tree form: evaluations per complete turn,
// and inherited expansions per evaluation"; t5d-turngen-time: "Turn generation
// (flat form): time to the first k legal turns, and per widening step" - both
// "by board-count range and mandatory count"), apart from the tools so that
// tests hold the arithmetic:
//
//  * the CELL a turn start is filed in: its board-count range - 6b''s, never
//    pooled (M12): < 50, 50-99, 100-199, 200+ - and its mandatory-board count
//    at the turn's start: 1, 2, 3, 4 or more;
//  * a cell's SUMMARY of a figure: how many values, their median and 95th
//    percentile (linear interpolation between the order statistics, as
//    t5d-bench's quantiles), least, greatest and mean;
//  * the TURN STARTS of records a tool counts or times: which are eligible,
//    and which of them are taken, cell by cell - a function of the records'
//    order and four numbers;
//  * the tree form's two RATIOS from one search's counts.
//
// Information for a cost row, never a verdict; decides no rule (not one of
// the truth's sources: CMakeLists.txt, T5D_TRUTH_SOURCES). The records come
// on the tools' command lines: no code here names a corpus.

#ifndef T5D_TOOLS_TURN_CELLS_HPP
#define T5D_TOOLS_TURN_CELLS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sixb_roots.hpp"
#include "t5d/state.hpp"

namespace t5d::tools::cells {

/// The board-count ranges: < 50, 50-99, 100-199, 200+ (sixbpp.hpp's).
inline constexpr int kRanges = 4;
/// The mandatory-board classes: 1, 2, 3, 4 or more.
inline constexpr int kMandatory = 4;
/// The cells: a range and a mandatory class.
inline constexpr int kCells = kRanges * kMandatory;

/// The range of a position of `boards` boards (0 to kRanges - 1).
int range_of(int boards) noexcept;
/// "< 50 boards", "50-99", "100-199", "200+".
const char* range_name(int range) noexcept;
/// The class of a turn start with `mandatory` mandatory boards: 0 for one, 1
/// for two, 2 for three, 3 for four or more; -1 for none (no turn start of a
/// game that goes on has none: the mover owes the present a move, T5).
int mandatory_class(int mandatory) noexcept;
/// "1", "2", "3", "4+".
const char* mandatory_name(int cls) noexcept;
/// The cell of a turn start: range_of(boards) * kMandatory + its mandatory
/// class; -1 where it has no mandatory board.
int cell_of(int boards, int mandatory) noexcept;

/// The q-quantile (0 <= q <= 1) of `sorted` (ascending, not empty): linear
/// interpolation between the order statistics at q (n - 1). 0 for none.
double quantile(const std::vector<double>& sorted, double q) noexcept;

/// A figure over a cell's values.
struct Summary {
  std::size_t n = 0;
  double median = 0;
  double p95 = 0;
  double min = 0;
  double max = 0;
  double mean = 0;
};
/// The summary of `values` (any order; none: every field 0).
Summary summarise(std::vector<double> values);

/// Which turn starts of the records are taken.
struct SelectOptions {
  /// Eligible: a turn start whose action index (the actions played before it)
  /// is a multiple of `every` ...
  int every = 4;
  /// ... and `min_action` or more.
  int min_action = 0;
  /// Taken: at most this many in a cell ...
  int max_per_cell = 25;
  /// ... and at most this many of one record in a cell (a cell's turn starts
  /// then come from different games).
  int per_record = 1;
};

/// A turn start taken.
struct TurnStart {
  std::string record;      ///< the record's id (as listed)
  std::size_t action = 0;  ///< the actions played before it
  int boards = 0;
  int mandatory = 0;
  std::size_t playable = 0;  ///< the mover's playable boards (mandatory and optional)
  int cell = 0;              ///< cell_of(boards, mandatory)
  GameState state;
};

struct StartSelection {
  std::size_t records = 0;
  /// Records the engine refuses, or cannot read or play to their end: the turn
  /// starts they gave before they stopped are kept.
  std::size_t stopped_early = 0;
  /// Turn starts their records play a further action from.
  std::size_t turn_starts = 0;
  /// Eligible turn starts with no mandatory board: filed in no cell.
  std::size_t no_mandatory = 0;
  /// Eligible turn starts, by cell.
  std::array<std::size_t, kCells> eligible{};
  /// The turn starts taken, in the records' order.
  std::vector<TurnStart> starts;
};

/// The turn starts of `records`, read in order with the reader's defaults and
/// replayed under the standard rules (royal safety tested: a record that
/// plays an illegal action stops there). A turn start is ELIGIBLE when its
/// record plays a further action from it - so it is no game end, and no
/// detector is asked - and its action index passes `options`; it is TAKEN
/// while its cell and its record's share of the cell have room. The record's
/// last position is never taken. Throws std::invalid_argument for an option
/// below its least (every, max_per_cell, per_record: 1; min_action: 0).
StartSelection select_turn_starts(const std::vector<RecordRef>& records, const SelectOptions& options);

/// What one search from a turn start counted (t5d-tree-count): the root is a
/// turn start, and the network evaluated it.
struct TreeCounts {
  /// The network's evaluations (SearchStats::evaluations), the root's among them.
  std::uint64_t evaluations = 0;
  /// Evaluated turn starts in the tree, the root among them: every other one
  /// is a COMPLETE TURN - a submission reached, and the position after it
  /// evaluated.
  std::uint64_t turn_starts_evaluated = 0;
  /// Evaluated nodes during a turn: the ply split's every mid-turn node, the
  /// tree form's where an interaction gate fired.
  std::uint64_t mid_turn_evaluated = 0;
  /// Mid-turn nodes expanded on inherited priors, without the network
  /// (SearchStats::inherited_nodes; the tree form only).
  std::uint64_t inherited = 0;
};

/// EVALUATIONS PER COMPLETE TURN: the evaluations below the root over the
/// complete turns among them - (turn_starts_evaluated - 1 + mid_turn_evaluated)
/// / (turn_starts_evaluated - 1), that is 1 plus the mid-turn evaluations per
/// complete turn. The root's own evaluation completes no turn and is left out
/// on both sides. 1 exactly where no mid-turn node is evaluated (the tree form
/// with no gate; the flat form); the ply split's is its moves per turn plus
/// one, over the paths searched. 0 where no complete turn was evaluated (a
/// figure is then not defined, and the tools leave the search out of it).
double evaluations_per_complete_turn(const TreeCounts& c) noexcept;
/// Whether evaluations_per_complete_turn is defined: a complete turn evaluated.
bool has_complete_turn(const TreeCounts& c) noexcept;

/// INHERITED EXPANSIONS PER EVALUATION: inherited / evaluations - the
/// expansions (each with its legality classification) the tree form makes
/// without the network, for each evaluation it does make. 0 for no evaluation.
double inherited_per_evaluation(const TreeCounts& c) noexcept;

}  // namespace t5d::tools::cells

#endif  // T5D_TOOLS_TURN_CELLS_HPP
