// The arithmetic of design spec 15.3's two turn rows (tools/turn_cells.hpp,
// tools/turngen_time.hpp; t5d-tree-count --by-range, t5d-turngen-time): the
// cells to the boundary (an off-by-one fails here), the median and the 95th
// percentile, which turn starts of a record are taken, the tree form's two
// ratios, a position's figures over its rounds and a cell's over its
// positions - and both tools run on records written here (the engine's tests
// read no corpus): their figures are the arithmetic of the counts and times
// they list. Arithmetic, never a speed: the machine running the tests is
// shared, and a test is not a measurement.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "sixb_roots.hpp"
#include "t5d/search.hpp"
#include "t5d/turngen.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "turn_cells.hpp"
#include "turngen_time.hpp"

#ifndef T5D_TEST_SCRATCH
#error "T5D_TEST_SCRATCH: a directory the tests may write"
#endif
#if !defined(T5D_TREE_COUNT_EXE) || !defined(T5D_TURNGEN_TIME_EXE)
#error "T5D_TREE_COUNT_EXE, T5D_TURNGEN_TIME_EXE: the paths of the two tools"
#endif

namespace cells = t5d::tools::cells;
namespace tt = t5d::tools::turngen_time;

namespace {

// Three Standard games, written here. A: six actions; B: four; C plays an
// illegal action third (the king to e3), so it stops after two.
const char* const kRecordA = "1. e3 / e6\n2. Nf3 / Nf6\n3. Nc3 / Nc6\n";
const char* const kRecordB = "1. d3 / d6\n2. Nd2 / Nd7\n";
const char* const kRecordC = "1. e3 / e6\n2. Ke3 / Nf6\n";

void write(const std::string& path, const std::string& bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("cannot write " + path);
  std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
}

std::string read_all(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

// Runs `exe` with `args`, standard output and error to files; the exit status.
int run_tool(const std::string& exe, const std::vector<std::string>& args, const std::filesystem::path& out,
             const std::filesystem::path& err) {
  std::string command = quoted(std::filesystem::path(exe).make_preferred().string());
  for (const std::string& a : args) command += " " + quoted(a);
  command += " > " + quoted(out.string()) + " 2> " + quoted(err.string());
#if defined(_WIN32)
  command = "\"" + command + "\"";
#endif
  std::fflush(nullptr);
  const int status = std::system(command.c_str());
#if !defined(_WIN32)
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#else
  return status;
#endif
}

// The number after `key` in `json` at or after `from` and before `end`; NaN
// if there is none.
double number_after(const std::string& json, const std::string& key, std::size_t from, std::size_t end) {
  const std::size_t at = json.find(key, from);
  if (at == std::string::npos || at >= end) return std::nan("");
  return std::strtod(json.c_str() + at + key.size(), nullptr);
}

// The records above in a scratch directory of the test's own: their refs, in
// the order A, B, C.
std::vector<t5d::tools::RecordRef> written_records(const char* directory) {
  const std::string dir = (std::filesystem::path(T5D_TEST_SCRATCH) / directory).string();
  std::filesystem::create_directories(dir);
  write(dir + "/a.5dpgn", kRecordA);
  write(dir + "/b.5dpgn", kRecordB);
  write(dir + "/c.5dpgn", kRecordC);
  return {{"a", dir + "/a.5dpgn"}, {"b", dir + "/b.5dpgn"}, {"c", dir + "/c.5dpgn"}};
}

tt::Sample sample(double t1, double t8, double t32, std::vector<double> steps) {
  tt::Sample s;
  s.first[0] = t1;
  s.first[1] = t8;
  s.first[2] = t32;
  s.steps = std::move(steps);
  s.turns = t32 >= 0 ? std::size_t{32} : t8 >= 0 ? std::size_t{8} : t1 >= 0 ? std::size_t{1} : std::size_t{0};
  return s;
}

bool near(double a, double b, double tolerance = 1e-12) { return std::abs(a - b) <= tolerance; }

}  // namespace

TEST(turn_cells_ranges_classes_and_cells_to_the_boundary) {
  // The ranges are 6b'''s.
  CHECK_EQ(cells::range_of(1), 0);
  CHECK_EQ(cells::range_of(49), 0);
  CHECK_EQ(cells::range_of(50), 1);
  CHECK_EQ(cells::range_of(99), 1);
  CHECK_EQ(cells::range_of(100), 2);
  CHECK_EQ(cells::range_of(199), 2);
  CHECK_EQ(cells::range_of(200), 3);
  CHECK_EQ(cells::range_of(5000), 3);
  CHECK_EQ(std::string(cells::range_name(0)), std::string("< 50 boards"));
  CHECK_EQ(std::string(cells::range_name(3)), std::string("200+"));
  // The mandatory classes: 1, 2, 3, 4 or more; none is no class.
  CHECK_EQ(cells::mandatory_class(0), -1);
  CHECK_EQ(cells::mandatory_class(-3), -1);
  CHECK_EQ(cells::mandatory_class(1), 0);
  CHECK_EQ(cells::mandatory_class(2), 1);
  CHECK_EQ(cells::mandatory_class(3), 2);
  CHECK_EQ(cells::mandatory_class(4), 3);
  CHECK_EQ(cells::mandatory_class(17), 3);
  CHECK_EQ(std::string(cells::mandatory_name(0)), std::string("1"));
  CHECK_EQ(std::string(cells::mandatory_name(3)), std::string("4+"));
  // A cell is a range and a class.
  CHECK_EQ(cells::kCells, 16);
  CHECK_EQ(cells::cell_of(49, 1), 0);
  CHECK_EQ(cells::cell_of(49, 4), 3);
  CHECK_EQ(cells::cell_of(50, 1), 4);
  CHECK_EQ(cells::cell_of(50, 2), 5);
  CHECK_EQ(cells::cell_of(199, 3), 10);
  CHECK_EQ(cells::cell_of(200, 9), 15);
  CHECK_EQ(cells::cell_of(10, 0), -1);
}

TEST(turn_cells_summary_is_the_median_and_the_95th_percentile) {
  // Linear interpolation between the order statistics at q (n - 1).
  const std::vector<double> five{1, 2, 3, 4, 5};
  CHECK(near(cells::quantile(five, 0.5), 3.0));
  CHECK(near(cells::quantile(five, 0.95), 4.8));
  CHECK(near(cells::quantile(five, 0.0), 1.0));
  CHECK(near(cells::quantile(five, 1.0), 5.0));
  CHECK_EQ(cells::quantile({}, 0.5), 0.0);
  std::vector<double> hundred_and_one;
  for (int i = 101; i >= 1; --i) hundred_and_one.push_back(i);  // given unsorted
  const cells::Summary s = cells::summarise(hundred_and_one);
  CHECK_EQ(s.n, std::size_t{101});
  CHECK(near(s.median, 51.0));
  CHECK(near(s.p95, 96.0));
  CHECK(near(s.min, 1.0));
  CHECK(near(s.max, 101.0));
  CHECK(near(s.mean, 51.0));
  const cells::Summary two = cells::summarise({3.0, 1.0});
  CHECK(near(two.median, 2.0));
  CHECK(near(two.p95, 2.9));
  CHECK(near(cells::summarise({1.0, 2.0, 9.0}).mean, 4.0));  // the mean, not the median (2)
  const cells::Summary one = cells::summarise({7.0});
  CHECK(near(one.median, 7.0) && near(one.p95, 7.0) && near(one.min, 7.0) && near(one.max, 7.0));
  const cells::Summary none = cells::summarise({});
  CHECK_EQ(none.n, std::size_t{0});
  CHECK_EQ(none.median, 0.0);
  CHECK_EQ(none.p95, 0.0);
  CHECK_EQ(none.mean, 0.0);
}

TEST(turn_cells_the_tree_forms_two_ratios) {
  // 400 evaluations: the root, 141 complete turns and 258 mid-turn leaves -
  // each complete turn cost (141 + 258) / 141 evaluations; 120 nodes
  // expanded without the network.
  cells::TreeCounts c;
  c.evaluations = 400;
  c.turn_starts_evaluated = 142;
  c.mid_turn_evaluated = 258;
  c.inherited = 120;
  CHECK(cells::has_complete_turn(c));
  CHECK(near(cells::evaluations_per_complete_turn(c), 399.0 / 141.0));
  CHECK(near(cells::inherited_per_evaluation(c), 0.3));
  // No gate: every evaluation is a turn start's - exactly one per complete turn.
  cells::TreeCounts none;
  none.evaluations = 400;
  none.turn_starts_evaluated = 400;
  none.inherited = 700;
  CHECK_EQ(cells::evaluations_per_complete_turn(none), 1.0);
  CHECK(near(cells::inherited_per_evaluation(none), 1.75));
  // The ply split with two moves a turn: the move's node, the submission's
  // turn start, and the turn start after it - 3 evaluations a complete turn.
  cells::TreeCounts ply;
  ply.evaluations = 31;
  ply.turn_starts_evaluated = 11;
  ply.mid_turn_evaluated = 20;
  CHECK(near(cells::evaluations_per_complete_turn(ply), 3.0));
  CHECK_EQ(cells::inherited_per_evaluation(ply), 0.0);
  // The root alone: no complete turn, no figure; no evaluation, no ratio.
  cells::TreeCounts root;
  root.evaluations = 1;
  root.turn_starts_evaluated = 1;
  CHECK(!cells::has_complete_turn(root));
  CHECK_EQ(cells::evaluations_per_complete_turn(root), 0.0);
  CHECK_EQ(cells::inherited_per_evaluation(cells::TreeCounts{}), 0.0);
}

TEST(turn_cells_takes_the_turn_starts_its_options_say) {
  const std::vector<t5d::tools::RecordRef> refs = written_records("turn-cells-select");
  // Every second turn start, no cap that binds: A's 0, 2, 4; B's 0, 2; C's 0
  // (it stops at its third action: the two turn starts before it stand).
  cells::SelectOptions o;
  o.every = 2;
  o.max_per_cell = 10;
  o.per_record = 10;
  const cells::StartSelection every_second = cells::select_turn_starts(refs, o);
  CHECK_EQ(every_second.records, std::size_t{3});
  CHECK_EQ(every_second.stopped_early, std::size_t{1});
  CHECK_EQ(every_second.turn_starts, std::size_t{6 + 4 + 2});  // each record's last position is not one
  CHECK_EQ(every_second.no_mandatory, std::size_t{0});
  CHECK_EQ(every_second.eligible[0], std::size_t{6});
  REQUIRE_EQ(every_second.starts.size(), std::size_t{6});
  const char* const records[6] = {"a", "a", "a", "b", "b", "c"};
  const std::size_t actions[6] = {0, 2, 4, 0, 2, 0};
  for (std::size_t i = 0; i < 6; ++i) {
    const cells::TurnStart& s = every_second.starts[i];
    CHECK_EQ(s.record, std::string(records[i]));
    CHECK_EQ(s.action, actions[i]);
    CHECK_EQ(s.boards, static_cast<int>(actions[i]) + 1);  // a board a half-turn on one timeline
    CHECK_EQ(s.mandatory, 1);
    CHECK_EQ(s.playable, std::size_t{1});
    CHECK_EQ(s.cell, 0);
    CHECK_EQ(static_cast<int>(s.state.multiverse().board_count()), s.boards);
  }
  // One of a record in a cell: each record's first.
  o.per_record = 1;
  const cells::StartSelection one_each = cells::select_turn_starts(refs, o);
  REQUIRE_EQ(one_each.starts.size(), std::size_t{3});
  CHECK_EQ(one_each.starts[0].record + one_each.starts[1].record + one_each.starts[2].record, std::string("abc"));
  CHECK_EQ(one_each.eligible[0], std::size_t{6});  // eligible is not what was taken
  // Two in a cell at most, every turn start: A's first two fill it.
  o = cells::SelectOptions{};
  o.every = 1;
  o.max_per_cell = 2;
  o.per_record = 10;
  const cells::StartSelection capped = cells::select_turn_starts(refs, o);
  REQUIRE_EQ(capped.starts.size(), std::size_t{2});
  CHECK_EQ(capped.starts[1].record, std::string("a"));
  CHECK_EQ(capped.starts[1].action, std::size_t{1});
  CHECK_EQ(capped.eligible[0], std::size_t{12});
  // From action 3: A's 3, 4, 5 and B's 3.
  o.max_per_cell = 10;
  o.min_action = 3;
  const cells::StartSelection late = cells::select_turn_starts(refs, o);
  REQUIRE_EQ(late.starts.size(), std::size_t{4});
  CHECK_EQ(late.starts[0].action, std::size_t{3});
  CHECK_EQ(late.starts[3].record, std::string("b"));
  // A record that cannot be read is counted and gives nothing.
  const cells::StartSelection missing = cells::select_turn_starts({{"x", "no-such-record.5dpgn"}}, o);
  CHECK_EQ(missing.records, std::size_t{1});
  CHECK_EQ(missing.stopped_early, std::size_t{1});
  CHECK(missing.starts.empty());
  // The options' least values.
  for (int which = 0; which < 4; ++which) {
    cells::SelectOptions bad;
    (which == 0 ? bad.every : which == 1 ? bad.max_per_cell : which == 2 ? bad.per_record : bad.min_action) =
        which == 3 ? -1 : 0;
    CHECK_THROWS_AS(cells::select_turn_starts(refs, bad), std::invalid_argument);
  }
}

TEST(turngen_time_a_positions_figure_is_the_median_over_its_rounds) {
  CHECK_EQ(tt::turns_asked(tt::Options{}), std::size_t{32});  // k0 16 + 16 steps: the greatest k
  tt::Options wide;
  wide.k0 = 30;
  wide.widen = 10;
  CHECK_EQ(tt::turns_asked(wide), std::size_t{40});
  wide.k0 = wide.widen = 0;
  CHECK_EQ(tt::turns_asked(wide), std::size_t{32});

  const std::vector<tt::Sample> rounds{sample(0.001, 0.010, 0.050, {1e-4, 2e-4}),
                                       sample(0.003, 0.012, 0.040, {3e-4, 1e-4}),
                                       sample(0.002, 0.011, 0.045, {2e-4, 6e-4})};
  const tt::PositionFigures f = tt::position_figures(30, 2, rounds);
  CHECK_EQ(f.boards, 30);
  CHECK_EQ(f.mandatory, 2);
  CHECK_EQ(f.turns, std::size_t{32});
  CHECK(near(f.first[0], 0.002));
  CHECK(near(f.first[1], 0.011));
  CHECK(near(f.first[2], 0.045));
  REQUIRE_EQ(f.steps.size(), std::size_t{2});
  CHECK(near(f.steps[0], 2e-4));  // each step's own median - not the rounds' means
  CHECK(near(f.steps[1], 2e-4));
  // Fewer than 32 turns: no time to 32, in any round; the times it has stand.
  std::vector<tt::Sample> few{sample(0.004, 0.02, -1, {}), sample(0.006, 0.03, -1, {})};
  few[0].exhausted = few[1].exhausted = true;
  const tt::PositionFigures g = tt::position_figures(40, 1, few);
  CHECK(near(g.first[0], 0.005));
  CHECK(near(g.first[1], 0.025));
  CHECK(g.first[2] < 0);
  CHECK(g.exhausted && !g.stopped);
  CHECK(g.steps.empty());
  CHECK_EQ(g.turns, std::size_t{8});
  // No round: no figure.
  const tt::PositionFigures empty = tt::position_figures(10, 1, {});
  CHECK(empty.first[0] < 0 && empty.steps.empty());
}

TEST(turngen_time_a_cells_figures_are_over_its_positions_and_their_steps) {
  const auto position = [](int boards, int mandatory, double t1, double t8, double t32, std::vector<double> steps) {
    tt::PositionFigures p;
    p.boards = boards;
    p.mandatory = mandatory;
    p.first[0] = t1;
    p.first[1] = t8;
    p.first[2] = t32;
    p.steps = std::move(steps);
    return p;
  };
  std::vector<tt::PositionFigures> all{
      position(30, 1, 0.002, 0.011, 0.045, {2e-4, 4e-4}),  // < 50, one mandatory board: every turn
      position(40, 1, 0.004, 0.020, -1, {6e-4}),           // ... ran out before turn 32
      position(45, 2, 0.010, -1, -1, {}),                  // < 50, two: stopped before turn 8
      position(120, 1, 0.5, 0.6, 0.7, {1e-3}),             // 100-199
      position(10, 0, 0.1, 0.1, 0.1, {}),                  // no mandatory board: in no cell
  };
  all[1].exhausted = true;
  all[2].stopped = true;

  const tt::CellFigures one = tt::cell_figures(all, 0, 0);
  CHECK_EQ(one.positions, std::size_t{2});
  CHECK_EQ(one.reached[0], std::size_t{2});
  CHECK_EQ(one.reached[1], std::size_t{2});
  CHECK_EQ(one.reached[2], std::size_t{1});
  CHECK_EQ(one.exhausted, std::size_t{1});
  CHECK_EQ(one.stopped, std::size_t{0});
  CHECK(near(one.first[0].median, 0.003));
  CHECK(near(one.first[0].p95, 0.002 + 0.95 * 0.002));
  CHECK(near(one.first[1].median, 0.0155));
  CHECK_EQ(one.first[2].n, std::size_t{1});  // only the positions that reached turn 32
  CHECK(near(one.first[2].median, 0.045));
  // The step time is over every step of every position: 2e-4, 4e-4, 6e-4.
  CHECK_EQ(one.step.n, std::size_t{3});
  CHECK(near(one.step.median, 4e-4));
  CHECK(near(one.step.p95, 4e-4 + 0.9 * 2e-4));
  CHECK(near(one.step.mean, 4e-4));
  CHECK_EQ(one.widening_positions, std::size_t{2});

  const tt::CellFigures two = tt::cell_figures(all, 0, 1);
  CHECK_EQ(two.positions, std::size_t{1});
  CHECK_EQ(two.reached[0], std::size_t{1});
  CHECK_EQ(two.reached[1], std::size_t{0});
  CHECK_EQ(two.first[1].n, std::size_t{0});
  CHECK_EQ(two.stopped, std::size_t{1});
  CHECK_EQ(two.step.n, std::size_t{0});
  CHECK_EQ(two.widening_positions, std::size_t{0});

  // A range over every mandatory count: its three positions, never another range's.
  const tt::CellFigures range = tt::cell_figures(all, 0, -1);
  CHECK_EQ(range.positions, std::size_t{3});
  CHECK_EQ(range.reached[0], std::size_t{3});
  CHECK(near(range.first[0].median, 0.004));
  CHECK_EQ(range.reached[2], std::size_t{1});
  CHECK_EQ(tt::cell_figures(all, 1, -1).positions, std::size_t{0});
  CHECK_EQ(tt::cell_figures(all, 2, -1).positions, std::size_t{1});
  CHECK(near(tt::cell_figures(all, 2, 0).first[2].median, 0.7));
}

TEST(turngen_time_times_a_generator_as_it_runs) {
  // On the Very Small start (one board: every turn one move), with scores of
  // the tool's own: the run emits what a plain generator emits, in the turns
  // asked; its times to turns 1 and 8 are in order; its widening steps are
  // the next()s after the first k0. A node budget of 1 stops it before a turn.
  const t5d::GameState s = t5d::variant_start(*t5d::find_variant("Very Small"));
  const tt::HashScorer scorer(7, 3.0);
  std::size_t legal = 0;
  {
    t5d::TurnGenerator plain(s, scorer, t5d::TurnUnit::Successors);
    while (plain.next()) ++legal;
  }
  REQUIRE(legal >= 4);
  tt::Options o;
  o.k0 = 2;
  o.widen = 3;
  const tt::Sample run = tt::time_generator(s, scorer, o);
  const std::size_t most = tt::turns_asked(o);
  CHECK_EQ(run.turns, std::min(legal, most));
  CHECK_EQ(run.exhausted, legal < most);
  CHECK(!run.stopped);
  CHECK(run.nodes > 0);
  CHECK(run.first[0] > 0);
  for (int i = 1; i < tt::kFirsts; ++i) {
    if (legal >= tt::kFirst[i]) {
      CHECK(run.first[i] >= run.first[i - 1]);
    } else {
      CHECK(run.first[i] < 0);
    }
  }
  const std::size_t steps_wanted = run.turns - 2 < 3 ? run.turns - 2 : std::size_t{3};
  CHECK_EQ(run.steps.size(), steps_wanted);
  double steps = 0;
  for (const double x : run.steps) {
    CHECK(x >= 0);
    steps += x;
  }
  // The steps lie inside the run: turns 3 to 5 are among the first 8, so
  // together they take no longer than turn 8 took from the construction.
  if (run.first[1] >= 0) {
    CHECK(steps <= run.first[1]);
  }
  tt::Options starved = o;
  starved.generator_nodes = 1;
  const tt::Sample none = tt::time_generator(s, scorer, starved);
  CHECK(none.stopped && !none.exhausted);
  CHECK(none.turns < run.turns);
  CHECK(none.first[tt::kFirsts - 1] < 0);
  if (none.turns == 0) {
    CHECK(none.first[0] < 0 && none.steps.empty());
  }

  // The scores: a function of the seed and the option alone, within the scale;
  // scale 0 scores every option 0.
  t5d::Move a;
  a.from_timeline = 0;
  a.from_half_turn = 1;
  a.to_timeline = 0;
  a.to_half_turn = 1;
  a.from_square = 4;
  a.to_square = 12;
  t5d::Move b = a;
  b.to_square = 20;
  CHECK_EQ(scorer.move(a, 0), tt::HashScorer(7, 3.0).move(a, 0));
  CHECK(scorer.move(a, 0) != scorer.move(b, 0));
  CHECK(scorer.move(a, 0) != scorer.move(a, 1));  // the board it is scored on
  CHECK(scorer.move(a, 0) != tt::HashScorer(8, 3.0).move(a, 0));
  CHECK(scorer.leave(0) != scorer.leave(1));
  for (int t = -8; t <= 8; ++t) {
    CHECK(std::abs(scorer.leave(t)) <= 3.0);
    CHECK(std::abs(scorer.move(a, t)) <= 3.0);
  }
  const tt::HashScorer flat(7, 0.0);
  CHECK_EQ(flat.move(a, 0), 0.0);
  CHECK_EQ(flat.leave(3), 0.0);
}

TEST(tree_count_exe_by_range_counts_what_it_lists) {
  // t5d-tree-count --by-range on records A and B: two turn starts of each
  // (actions 0 and 1; < 50 boards, one mandatory board), the ply split and the
  // tree form with no gate, 12 evaluations a search. The JSON's cell figures
  // are the arithmetic of the counts it lists; with no gate no mid-turn node
  // is evaluated, so every complete turn costs exactly one evaluation; and
  // the standard output is the same bytes on one thread and on three.
  namespace fs = std::filesystem;
  const std::vector<t5d::tools::RecordRef> refs = written_records("tree-count-by-range");
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "tree-count-by-range";
  const fs::path json_path = dir / "by-range.json";
  const std::vector<std::string> args{"--by-range", "--every",       "1",   "--max-per-cell", "4",
                                      "--per-record", "2",           "--evaluations", "12", "--arms",
                                      "ply,tree-none", "--json",     json_path.string(), refs[0].path, refs[1].path};
  REQUIRE_EQ(run_tool(T5D_TREE_COUNT_EXE, args, dir / "stdout-1.txt", dir / "stderr-1.txt"), 0);
  const std::string out = read_all(dir / "stdout-1.txt");
  const std::string json = read_all(json_path);
  CHECK(contains(out, "t5d-tree-count --by-range"));
  CHECK(contains(out, "counted 4: at most 4 in a cell, 2 of one record in a cell"));
  // A figure carries the definition it was counted under (build-log issue 418).
  CHECK(contains(out, "a COMPLETE TURN is an evaluated turn start below the root"));
  CHECK(contains(out, "the root's own evaluation is left out on both sides"));
  CHECK(contains(json, "\"definition_complete_turn\":\"an evaluated turn start below the root\""));
  CHECK(contains(json, "\"counted\":4"));

  // The positions' counts, arm by arm (each position lists the arms in order).
  const std::size_t positions = json.find("\"positions\":[");
  REQUIRE(positions != std::string::npos);
  std::vector<double> per_turn[2];
  std::vector<double> inherited[2];
  std::size_t listed = 0;
  for (std::size_t at = json.find("\"arm\":", positions); at != std::string::npos; at = json.find("\"arm\":", at + 1)) {
    const std::size_t end = json.find('}', at);
    const std::size_t arm = contains(json.substr(at, end - at), "tree-none") ? 1 : 0;
    cells::TreeCounts c;
    c.evaluations = static_cast<std::uint64_t>(number_after(json, "\"evaluations\":", at, end));
    c.turn_starts_evaluated = static_cast<std::uint64_t>(number_after(json, "\"turn_starts_evaluated\":", at, end));
    c.mid_turn_evaluated = static_cast<std::uint64_t>(number_after(json, "\"mid_turn_evaluated\":", at, end));
    c.inherited = static_cast<std::uint64_t>(number_after(json, "\"inherited\":", at, end));
    CHECK(c.evaluations >= 1 && c.evaluations <= 12);
    CHECK_EQ(c.turn_starts_evaluated + c.mid_turn_evaluated, c.evaluations);  // every evaluation is one or the other
    if (arm == 1) CHECK_EQ(c.mid_turn_evaluated, std::uint64_t{0});           // no gate: turn starts only
    if (arm == 0) CHECK_EQ(c.inherited, std::uint64_t{0});                    // the ply split inherits nothing
    if (cells::has_complete_turn(c)) per_turn[arm].push_back(cells::evaluations_per_complete_turn(c));
    inherited[arm].push_back(cells::inherited_per_evaluation(c));
    ++listed;
  }
  CHECK_EQ(listed, std::size_t{8});  // 4 turn starts x 2 arms

  // The cells' figures: "< 50 boards", one mandatory board - and the range's
  // "all", the same four turn starts.
  const char* const arm_keys[2] = {"\"arm\":\"ply\"", "\"arm\":\"tree-none\""};
  for (std::size_t arm = 0; arm < 2; ++arm) {
    const std::size_t from = json.find(arm_keys[arm]);
    REQUIRE(from != std::string::npos && from < positions);
    const cells::Summary want_turn = cells::summarise(per_turn[arm]);
    const cells::Summary want_inherited = cells::summarise(inherited[arm]);
    for (const char* mandatory : {"\"mandatory\":\"all\"", "\"mandatory\":\"1\""}) {
      const std::size_t cell = json.find(mandatory, from);
      REQUIRE(cell != std::string::npos && cell < positions);
      const std::size_t turn = json.find("\"evaluations_per_complete_turn\":", cell);
      const std::size_t turn_end = json.find('}', turn);
      const std::size_t inh = json.find("\"inherited_per_evaluation\":", cell);
      const std::size_t inh_end = json.find('}', inh);
      CHECK_EQ(number_after(json, "\"turn_starts\":", cell, turn), 4.0);
      CHECK_EQ(number_after(json, "\"n\":", turn, turn_end), static_cast<double>(want_turn.n));
      CHECK(near(number_after(json, "\"median\":", turn, turn_end), want_turn.median, 1e-6));
      CHECK(near(number_after(json, "\"p95\":", turn, turn_end), want_turn.p95, 1e-6));
      CHECK_EQ(number_after(json, "\"n\":", inh, inh_end), 4.0);
      CHECK(near(number_after(json, "\"median\":", inh, inh_end), want_inherited.median, 1e-6));
      CHECK(near(number_after(json, "\"p95\":", inh, inh_end), want_inherited.p95, 1e-6));
    }
  }
  REQUIRE(!per_turn[1].empty());
  for (const double x : per_turn[1]) CHECK_EQ(x, 1.0);  // no gate: one evaluation a complete turn

  // The counts do not depend on the threads.
  std::vector<std::string> threaded{"--threads", "3"};
  threaded.insert(threaded.end(), args.begin(), args.end());
  REQUIRE_EQ(run_tool(T5D_TREE_COUNT_EXE, threaded, dir / "stdout-3.txt", dir / "stderr-3.txt"), 0);
  CHECK_EQ(read_all(dir / "stdout-3.txt"), out);

  // An unknown arm is refused.
  CHECK_EQ(run_tool(T5D_TREE_COUNT_EXE, {"--by-range", "--arms", "ply,whole", refs[0].path}, dir / "stdout-bad.txt",
                    dir / "stderr-bad.txt"),
           2);
  CHECK(contains(read_all(dir / "stderr-bad.txt"), "no arm \"whole\""));
}

TEST(turngen_time_exe_figures_are_the_arithmetic_of_its_positions) {
  // t5d-turngen-time on record A's first two turn starts (the Standard start
  // and the position after 1. e3: 20 legal turns each, no board to travel to -
  // turn 8 reached, turn 32 never), 2 rounds after the warm-up, 3 widening
  // steps after the first 2 turns: the cell's figures are the medians and
  // 95th percentiles of the positions' figures it lists, and a turn no
  // position reached has a count and no time. The arithmetic, not a speed.
  namespace fs = std::filesystem;
  const std::vector<t5d::tools::RecordRef> refs = written_records("turngen-time");
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "turngen-time";
  const fs::path json_path = dir / "turngen-time.json";
  REQUIRE_EQ(run_tool(T5D_TURNGEN_TIME_EXE,
                      {"--every", "1", "--max-per-cell", "2", "--per-record", "2", "--rounds", "2", "--k0", "2",
                       "--widen", "3", "--per-position", "--json", json_path.string(), refs[0].path},
                      dir / "stdout.txt", dir / "stderr.txt"),
             0);
  const std::string out = read_all(dir / "stdout.txt");
  const std::string json = read_all(json_path);
  CHECK(contains(out, "t5d-turngen-time"));
  CHECK(contains(out, "2 interleaved rounds over every position after a warm-up round"));
  CHECK(contains(out, "over the counted rounds together"));  // the load, whole, where a round is under a clock tick
  CHECK(contains(out, "load:"));
  CHECK(contains(out, "build:"));
  CHECK(contains(out, "RECORDS: 2 turn starts"));
  CHECK(contains(json, "\"rounds\":2,\"warm_up_rounds\":1"));
  // A figure carries what was timed and what the scores were (build-log issue 419).
  CHECK(contains(out, "runs from just before the generator's construction"));
  CHECK(contains(out, "A WIDENING STEP is one next() after the first 2 turns"));
  CHECK(contains(out, "scores:  NOT A NETWORK'S - a seeded hash of each option's coordinates"));
  CHECK(contains(json, "\"definition_scores\":\"a seeded hash of each option's coordinates, not a network's\""));

  // The positions' figures.
  const std::size_t positions = json.find("\"positions\":[");
  REQUIRE(positions != std::string::npos);
  std::vector<double> first[tt::kFirsts];
  std::vector<double> steps;
  std::size_t listed = 0;
  for (std::size_t at = json.find("\"first_seconds\":[", positions); at != std::string::npos;
       at = json.find("\"first_seconds\":[", at + 1)) {
    const char* p = json.c_str() + at + std::string("\"first_seconds\":[").size();
    for (int i = 0; i < tt::kFirsts; ++i) {
      char* end = nullptr;
      const double v = std::strtod(p, &end);
      REQUIRE(end != p);
      if (v >= 0) first[i].push_back(v);
      p = end + 1;  // past the comma, or the bracket
    }
    const std::size_t s = json.find("\"step_seconds\":[", at);
    REQUIRE(s != std::string::npos);
    const std::size_t s_end = json.find(']', s);
    p = json.c_str() + s + std::string("\"step_seconds\":[").size();
    while (p < json.c_str() + s_end) {
      char* end = nullptr;
      const double v = std::strtod(p, &end);
      REQUIRE(end != p);
      steps.push_back(v);
      p = end + 1;
    }
    ++listed;
  }
  CHECK_EQ(listed, std::size_t{2});
  CHECK_EQ(first[0].size(), std::size_t{2});
  CHECK_EQ(first[1].size(), std::size_t{2});
  CHECK_EQ(first[2].size(), std::size_t{0});  // 20 legal turns: turn 32 is never reached
  CHECK_EQ(steps.size(), std::size_t{6});     // 2 positions x 3 widening steps

  // The cell "< 50 boards", one mandatory board - and the range's "all".
  for (const char* mandatory : {"\"mandatory\":\"all\"", "\"mandatory\":\"1\""}) {
    const std::size_t cell = json.find(mandatory);
    REQUIRE(cell != std::string::npos && cell < positions);
    CHECK_EQ(number_after(json, "\"positions\":", cell, positions), 2.0);
    std::size_t cursor = cell;
    for (int i = 0; i < tt::kFirsts; ++i) {
      const std::size_t k = json.find("{\"k\":", cursor);
      REQUIRE(k != std::string::npos && k < positions);
      const std::size_t end = json.find("}}", k);
      const cells::Summary want = cells::summarise(first[i]);
      CHECK_EQ(number_after(json, "{\"k\":", k, end), static_cast<double>(tt::kFirst[i]));
      CHECK_EQ(number_after(json, "\"reached\":", k, end), static_cast<double>(want.n));
      CHECK_EQ(number_after(json, "\"n\":", k, end), static_cast<double>(want.n));
      // Printed to 9 decimals, positions and cells alike.
      CHECK(near(number_after(json, "\"median\":", k, end), want.median, 2e-9));
      CHECK(near(number_after(json, "\"p95\":", k, end), want.p95, 2e-9));
      cursor = end;
    }
    const std::size_t w = json.find("\"widening_step\":", cursor);
    REQUIRE(w != std::string::npos && w < positions);
    const std::size_t w_end = json.find("}}", w);
    const cells::Summary want = cells::summarise(steps);
    CHECK_EQ(number_after(json, "\"positions\":", w, w_end), 2.0);
    CHECK_EQ(number_after(json, "\"n\":", w, w_end), 6.0);
    CHECK(near(number_after(json, "\"median\":", w, w_end), want.median, 2e-9));
    CHECK(near(number_after(json, "\"p95\":", w, w_end), want.p95, 2e-9));
    CHECK(near(number_after(json, "\"mean\":", w, w_end), want.mean, 2e-9));
  }
  for (const double x : first[0]) CHECK(x > 0);

  // No position: refused.
  CHECK_EQ(run_tool(T5D_TURNGEN_TIME_EXE, {"--rounds", "1"}, dir / "stdout-none.txt", dir / "stderr-none.txt"), 2);
}
