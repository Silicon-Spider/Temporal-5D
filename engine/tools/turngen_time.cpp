// What t5d-turngen-time times and how it sums up (see turngen_time.hpp).

#include "turngen_time.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <utility>

#include "t5d/search.hpp"

namespace t5d::tools::turngen_time {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point a, Clock::time_point b) noexcept {
  return std::chrono::duration<double>(b - a).count();
}

// splitmix64's finaliser: a well-mixed 64-bit value of another.
std::uint64_t mix(std::uint64_t z) noexcept {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::uint64_t fold(std::uint64_t h, std::int64_t v) noexcept { return mix(h ^ static_cast<std::uint64_t>(v)); }

double median_of(std::vector<double> v) { return cells::summarise(std::move(v)).median; }

}  // namespace

std::size_t turns_asked(const Options& options) noexcept {
  return std::max(kFirst[kFirsts - 1], options.k0 + options.widen);
}

double HashScorer::value(std::uint64_t key) const noexcept {
  if (scale_ == 0) return 0;
  const std::uint64_t h = mix(seed_ ^ mix(key));
  // 53 random bits: uniform in [0, 1).
  const double unit = static_cast<double>(h >> 11) * 0x1.0p-53;
  return (2 * unit - 1) * scale_;
}

double HashScorer::leave(int timeline) const { return value(fold(0x6C65617665ull, timeline)); }

double HashScorer::move(const Move& m, int timeline) const {
  std::uint64_t h = 0x6D6F7665ull;
  h = fold(h, m.from_timeline);
  h = fold(h, m.from_half_turn);
  h = fold(h, m.from_square);
  h = fold(h, m.to_timeline);
  h = fold(h, m.to_half_turn);
  h = fold(h, m.to_square);
  h = fold(h, m.promotes() ? static_cast<std::int64_t>(index_of(m.promotion)) + 1 : 0);
  h = fold(h, timeline);
  return value(h);
}

Sample time_generator(const GameState& turn_start, const TurnScorer& scorer, const Options& options) {
  Sample s;
  const std::size_t most = turns_asked(options);
  s.steps.reserve(options.widen);  // before the clock starts
  const SearchBudget budget =
      options.generator_nodes == 0 ? SearchBudget::unlimited() : SearchBudget::nodes(options.generator_nodes);
  const Clock::time_point t0 = Clock::now();
  TurnGenerator generator(turn_start, scorer, options.unit, budget);
  while (s.turns < most) {
    const Clock::time_point before = Clock::now();
    const std::optional<GeneratedTurn> turn = generator.next();
    const Clock::time_point after = Clock::now();
    if (!turn) break;
    ++s.turns;
    for (int i = 0; i < kFirsts; ++i) {
      if (s.turns == kFirst[i]) s.first[i] = seconds_between(t0, after);
    }
    if (s.turns > options.k0 && s.steps.size() < options.widen) s.steps.push_back(seconds_between(before, after));
  }
  s.exhausted = generator.exhausted();
  s.stopped = generator.stopped();
  s.nodes = generator.stats().nodes;
  return s;
}

PositionFigures position_figures(int boards, int mandatory, const std::vector<Sample>& rounds) {
  PositionFigures f;
  f.boards = boards;
  f.mandatory = mandatory;
  if (rounds.empty()) return f;
  f.turns = rounds.front().turns;
  f.exhausted = rounds.front().exhausted;
  f.stopped = rounds.front().stopped;
  for (int i = 0; i < kFirsts; ++i) {
    std::vector<double> v;
    for (const Sample& s : rounds) {
      if (s.first[i] >= 0) v.push_back(s.first[i]);
    }
    if (!v.empty()) f.first[i] = median_of(std::move(v));
  }
  std::size_t steps = 0;
  for (const Sample& s : rounds) steps = std::max(steps, s.steps.size());
  for (std::size_t k = 0; k < steps; ++k) {
    std::vector<double> v;
    for (const Sample& s : rounds) {
      if (k < s.steps.size()) v.push_back(s.steps[k]);
    }
    f.steps.push_back(median_of(std::move(v)));
  }
  return f;
}

CellFigures cell_figures(const std::vector<PositionFigures>& positions, int range, int cls) {
  CellFigures c;
  std::vector<double> first[kFirsts];
  std::vector<double> steps;
  for (const PositionFigures& p : positions) {
    const int cell = cells::cell_of(p.boards, p.mandatory);
    if (cell < 0 || cell / cells::kMandatory != range) continue;
    if (cls >= 0 && cell % cells::kMandatory != cls) continue;
    ++c.positions;
    if (p.exhausted) ++c.exhausted;
    if (p.stopped) ++c.stopped;
    for (int i = 0; i < kFirsts; ++i) {
      if (p.first[i] < 0) continue;
      ++c.reached[i];
      first[i].push_back(p.first[i]);
    }
    if (!p.steps.empty()) ++c.widening_positions;
    steps.insert(steps.end(), p.steps.begin(), p.steps.end());
  }
  for (int i = 0; i < kFirsts; ++i) c.first[i] = cells::summarise(std::move(first[i]));
  c.step = cells::summarise(std::move(steps));
  return c;
}

}  // namespace t5d::tools::turngen_time
