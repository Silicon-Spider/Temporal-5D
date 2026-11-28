// The cells, summaries and ratios of design spec 15.3's turn rows (see
// turn_cells.hpp).

#include "turn_cells.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>

#include "sixbpp.hpp"
#include "t5d/pgn.hpp"
#include "tool_support.hpp"

namespace t5d::tools::cells {

static_assert(kRanges == sixbpp::kRanges, "the ranges are 6b'''s");

int range_of(int boards) noexcept { return sixbpp::range_of(boards); }

const char* range_name(int range) noexcept {
  static const char* const names[kRanges] = {"< 50 boards", "50-99", "100-199", "200+"};
  return range >= 0 && range < kRanges ? names[range] : "?";
}

int mandatory_class(int mandatory) noexcept {
  if (mandatory < 1) return -1;
  return mandatory >= kMandatory ? kMandatory - 1 : mandatory - 1;
}

const char* mandatory_name(int cls) noexcept {
  static const char* const names[kMandatory] = {"1", "2", "3", "4+"};
  return cls >= 0 && cls < kMandatory ? names[cls] : "?";
}

int cell_of(int boards, int mandatory) noexcept {
  const int m = mandatory_class(mandatory);
  return m < 0 ? -1 : range_of(boards) * kMandatory + m;
}

double quantile(const std::vector<double>& sorted, double q) noexcept {
  if (sorted.empty()) return 0;
  const double at = std::clamp(q, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(at));
  const auto hi = static_cast<std::size_t>(std::ceil(at));
  const double f = at - static_cast<double>(lo);
  return sorted[lo] * (1 - f) + sorted[hi] * f;
}

Summary summarise(std::vector<double> values) {
  Summary s;
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  s.n = values.size();
  s.median = quantile(values, 0.5);
  s.p95 = quantile(values, 0.95);
  s.min = values.front();
  s.max = values.back();
  double sum = 0;
  for (const double v : values) sum += v;
  s.mean = sum / static_cast<double>(values.size());
  return s;
}

StartSelection select_turn_starts(const std::vector<RecordRef>& records, const SelectOptions& options) {
  if (options.every < 1 || options.min_action < 0 || options.max_per_cell < 1 || options.per_record < 1) {
    throw std::invalid_argument("select_turn_starts: every, max_per_cell and per_record are 1 or more, min_action 0 or more");
  }
  StartSelection sel;
  std::array<int, kCells> filled{};
  for (const RecordRef& ref : records) {
    ++sel.records;
    std::array<int, kCells> mine{};
    try {
      PgnReplay replay(read_pgn(read_file(ref.path)));
      while (!replay.finished()) {
        const GameState before = replay.state();
        const std::size_t action = replay.actions_played();
        replay.play_next();  // the record plays on from `before`: it is no game end
        ++sel.turn_starts;
        if (action < static_cast<std::size_t>(options.min_action) ||
            action % static_cast<std::size_t>(options.every) != 0) {
          continue;
        }
        const int boards = static_cast<int>(before.multiverse().board_count());
        const int cell = cell_of(boards, before.mandatory_count());
        if (cell < 0) {
          ++sel.no_mandatory;
          continue;
        }
        const auto c = static_cast<std::size_t>(cell);
        ++sel.eligible[c];
        if (filled[c] >= options.max_per_cell || mine[c] >= options.per_record) continue;
        ++filled[c];
        ++mine[c];
        sel.starts.push_back(TurnStart{ref.id, action, boards, before.mandatory_count(),
                                       before.playable_timelines().size(), cell, before});
      }
    } catch (const std::exception&) {
      ++sel.stopped_early;  // what the record gave before it stopped is kept
    }
  }
  return sel;
}

bool has_complete_turn(const TreeCounts& c) noexcept { return c.turn_starts_evaluated > 1; }

double evaluations_per_complete_turn(const TreeCounts& c) noexcept {
  if (!has_complete_turn(c)) return 0;
  const double turns = static_cast<double>(c.turn_starts_evaluated - 1);
  return (turns + static_cast<double>(c.mid_turn_evaluated)) / turns;
}

double inherited_per_evaluation(const TreeCounts& c) noexcept {
  return c.evaluations == 0 ? 0.0 : static_cast<double>(c.inherited) / static_cast<double>(c.evaluations);
}

}  // namespace t5d::tools::cells
