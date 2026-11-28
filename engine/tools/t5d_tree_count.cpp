// t5d-tree-count: D2's tree and flat forms against the ply split on turn starts of
// recorded games - evaluation COUNTS, never timings (design spec §8.2,
// "Evaluating at turn boundaries"; register D2; docs/MEASUREMENTS.md). What
// the tree form buys is turns per evaluation: the ply split spends evaluations
// on the states between a turn's moves, the tree form only where an
// interaction gate fires. The counts are a function of the search's
// parameters and the positions alone - deterministic, the same on any machine
// and under any load.
//
//   t5d-tree-count RECORD ...                       seven turn starts, every arm
//   t5d-tree-count --by-range [options] [RECORD ...]  spec 15.3's row, by cell
//   (--help: the options)
//
// ARMS, each a fresh search from the turn start with the built-in random
// evaluator (seed 1), synchronous (one leaf per evaluation), no evaluation
// cache, square features off (the built-in evaluators read none), every other
// parameter the search's default:
//
//   ply split                    TurnSplit::BoardMove                              "ply"
//   tree form, natural gates     TurnSplit::Tree, §8.2's four gates (the defaults) "tree"
//   tree form, no gates          TurnSplit::Tree, tree_gates 0                     "tree-none"
//   tree, natural, rewind only   TurnSplit::Tree, tree_present Rewind              "tree-rewind"
//   flat form                    TurnSplit::Flat (whole turns generated best       "flat"
//                                first; its every edge a turn - the depth counts
//                                them; and its turns generated, generators a
//                                budget stopped, and fallbacks to E4's listing)
//
// THE FIRST MODE (no option). The records are read in the order given (a
// shell's sorted glob: build/t0db/*.5dpgn) and replayed under the standard
// rules; from each, at most one turn start is taken - the first, at action 6
// or later and not a game end, that fills a class still wanting one: two
// mandatory boards (3 wanted; the k-th of them with at least 20k boards),
// three or more (3), one with 30 boards or more (1). Records the engine
// refuses are skipped and counted. Taking stops once every class is filled.
// Each arm runs twice: to 400 EVALUATIONS (playouts unbounded) - the turn
// starts and mid-turn nodes the network evaluated, the nodes inherited, and
// the deepest evaluated turn start (submit edges from the root) - and to 400
// PLAYOUTS - the evaluations made, the nodes inherited and the mid-turn leaves
// by the gate that fired first (between, capture, check, present).
//
// BY RANGE (--by-range; design spec 15.3: "Tree form: evaluations per complete
// turn, and inherited expansions per evaluation, by board-count range and
// mandatory count"). The records - given directly and by --record-list files
// (one path a line, '#' lines skipped; relative paths from --record-root) -
// are replayed in order under the standard rules, and a turn start is
// ELIGIBLE when its record plays a further action from it (so it is no game
// end: no detector is asked) and its action index is --min-action or more and
// a multiple of --every. Each is filed in its CELL - board-count range (< 50,
// 50-99, 100-199, 200+) and mandatory boards (1, 2, 3, 4+; tools/turn_cells.hpp)
// - and COUNTED while the cell wants one: at most --max-per-cell in a cell,
// and at most --per-record of one record in a cell, so that a cell's turn
// starts come from different games. The choice is a function of the records'
// order and these four numbers alone. At each counted turn start every arm
// asked for (--arms) searches once, to --evaluations evaluations (playouts
// unbounded), and gives (turn_cells.hpp):
//   evaluations per complete turn  (evaluated turn starts below the root +
//                                  evaluated mid-turn nodes) / evaluated turn
//                                  starts below the root - 1 where no mid-turn
//                                  node is evaluated;
//   inherited expansions per evaluation
//                                  mid-turn nodes expanded without the network
//                                  / evaluations.
// Reported per arm and cell, and per range over every mandatory count: the
// turn starts counted, and each ratio's median and 95th percentile over them.
// --threads T runs the turn starts' searches on T threads; a search is its own
// (one thread, its ladder synchronous), so the counts do not depend on T. The
// standard output is the same bytes for the same records and options; the wall
// time goes to standard error.
//
// Not one of the truth's sources (CMakeLists.txt, T5D_TRUTH_SOURCES): it
// decides no verdict.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "sixb_roots.hpp"
#include "t5d/gate.hpp"
#include "t5d/mcts.hpp"
#include "t5d/pgn.hpp"
#include "t5d/search.hpp"
#include "tool_support.hpp"
#include "turn_cells.hpp"

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif

using namespace t5d;
using namespace t5d::mcts;

namespace {

namespace cells = t5d::tools::cells;
using t5d::gate::json_string;

struct Pick {
  std::string name;
  GameState state;
};

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Row {
  std::uint64_t evaluations = 0;
  std::uint64_t playouts = 0;
  std::uint64_t turn_starts_evaluated = 0;
  std::uint64_t mid_turn_evaluated = 0;
  std::uint64_t inherited = 0;
  std::uint64_t deepest = 0;
  std::array<std::uint64_t, 5> gates{};
  std::uint64_t generated = 0;
  std::uint64_t stopped = 0;
  std::uint64_t fallbacks = 0;
};

constexpr int kArms = 5;
const char* const kArmNames[kArms] = {"ply split", "tree form, natural gates", "tree form, no gates",
                                      "tree, natural, rewind only", "flat form"};
// The arms' names on the command line (--arms), by arm.
const char* const kArmKeys[kArms] = {"ply", "tree", "tree-none", "tree-rewind", "flat"};

Row count(const GameState& s, int arm, const Limits& limits) {
  SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.eval_cache_entries = 0;
  p.synchronous = true;
  if (arm > 0) p.split = arm == 4 ? TurnSplit::Flat : TurnSplit::Tree;
  if (arm == 2) p.tree_gates = 0;
  if (arm == 3) p.tree_present = PresentShift::Rewind;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 1);
  Search search(s, p);
  search.run(*ev, limits);
  Row r;
  r.evaluations = search.stats().evaluations;
  r.playouts = search.stats().playouts;
  r.inherited = search.stats().inherited_nodes;
  r.gates = search.stats().gate_fired;
  r.generated = search.stats().generated_turns;
  r.stopped = search.stats().stopped_generators;
  r.fallbacks = search.stats().fallback_listings;
  // Breadth-first over the tree: the evaluated nodes, and how many turns deep
  // (submit edges from the root) the deepest evaluated turn start lies.
  std::vector<std::pair<NodeId, std::uint64_t>> queue{{search.root(), 0}};
  for (std::size_t i = 0; i < queue.size(); ++i) {
    const auto [id, depth] = queue[i];
    const NodeView n = search.node(id);
    if (n.evaluated) (n.turn_start ? r.turn_starts_evaluated : r.mid_turn_evaluated) += 1;
    if (n.turn_start && n.evaluated && depth > r.deepest) r.deepest = depth;
    for (std::size_t e = 0; e < n.edge_count; ++e) {
      const EdgeView edge = search.edge(id, e);
      if (edge.child != kNoNode) queue.emplace_back(edge.child, depth + (edge.submit || arm == 4 ? 1 : 0));
    }
  }
  return r;
}

unsigned long long u(std::uint64_t v) { return static_cast<unsigned long long>(v); }

// ---- the first mode: seven turn starts, every arm ------------------------------------------------

int seven_turn_starts(int argc, char** argv) {
  std::vector<Pick> picks;
  int want2 = 3;
  int want3 = 3;
  int want1 = 1;
  std::size_t refused = 0;
  for (int i = 1; i < argc && (want2 > 0 || want3 > 0 || want1 > 0); ++i) {
    try {
      PgnReplay replay(read_pgn(slurp(argv[i])));
      std::size_t action = 0;
      while (!replay.finished()) {
        const GameState& s = replay.state();
        const int mandatory = s.mandatory_count();
        const std::size_t boards = s.multiverse().board_count();
        if (action >= 6 && outcome(s, standard_check_ruling()) == Outcome::Ongoing) {
          const std::string name = std::string(argv[i]) + " action " + std::to_string(action) + ", " +
                                   std::to_string(boards) + " boards, " + std::to_string(mandatory) + " mandatory, " +
                                   std::to_string(s.playable_timelines().size()) + " playable";
          if (mandatory == 2 && want2 > 0 && boards >= 20 * static_cast<std::size_t>(4 - want2)) {
            picks.push_back(Pick{name, s});
            --want2;
            break;
          }
          if (mandatory >= 3 && want3 > 0) {
            picks.push_back(Pick{name, s});
            --want3;
            break;
          }
          if (mandatory == 1 && want1 > 0 && boards >= 30) {
            picks.push_back(Pick{name, s});
            --want1;
            break;
          }
        }
        replay.play_next();
        ++action;
      }
    } catch (const std::exception&) {
      ++refused;  // a record the engine refuses (or cannot read on): skipped
    }
  }
  std::printf("t5d-tree-count: %zu turn starts; %zu records skipped; random evaluator seed 1, synchronous, no cache\n",
              picks.size(), refused);
  for (const Pick& p : picks) {
    std::printf("%s\n", p.name.c_str());
    for (int arm = 0; arm < kArms; ++arm) {
      Limits by_evaluations;
      by_evaluations.playouts = 1000000;
      by_evaluations.evaluations = 400;
      const Row a = count(p.state, arm, by_evaluations);
      const Row b = count(p.state, arm, Limits{400});
      std::printf(
          "  %-26s 400 evals: %4llu playouts, turn starts evaluated %4llu, mid-turn evaluated %4llu, inherited %4llu, "
          "deepest evaluated turn start %llu turns | 400 playouts: %4llu evals, %4llu inherited, gates b/c/k/p "
          "%llu/%llu/%llu/%llu",
          kArmNames[arm], u(a.playouts), u(a.turn_starts_evaluated), u(a.mid_turn_evaluated), u(a.inherited),
          u(a.deepest), u(b.evaluations), u(b.inherited), u(b.gates[0]), u(b.gates[1]), u(b.gates[2]),
          u(b.gates[3]));
      if (arm == 4) {
        std::printf(" | turns generated %llu, generators stopped %llu, fallbacks %llu (400 evals)", u(a.generated),
                    u(a.stopped), u(a.fallbacks));
      }
      std::printf("\n");
    }
  }
  return 0;
}

// ---- by range: spec 15.3's row -------------------------------------------------------------------

struct RangeOptions {
  std::vector<std::string> records;
  std::vector<std::string> lists;
  std::string record_root;
  cells::SelectOptions select;  // which turn starts are counted (turn_cells.hpp)
  std::uint64_t evaluations = 400;
  std::vector<int> arms{0, 1, 3, 2};  // ply, tree, tree-rewind, tree-none
  int threads = 1;
  std::string json;
};

using Start = cells::TurnStart;

cells::TreeCounts tree_counts(const Row& r) {
  cells::TreeCounts c;
  c.evaluations = r.evaluations;
  c.turn_starts_evaluated = r.turn_starts_evaluated;
  c.mid_turn_evaluated = r.mid_turn_evaluated;
  c.inherited = r.inherited;
  return c;
}

std::string fixed(double value, int digits) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", digits, value);
  return buf;
}

std::string summary_json(const cells::Summary& s) {
  return "{\"n\":" + std::to_string(s.n) + ",\"median\":" + fixed(s.median, 6) + ",\"p95\":" + fixed(s.p95, 6) +
         ",\"min\":" + fixed(s.min, 6) + ",\"max\":" + fixed(s.max, 6) + ",\"mean\":" + fixed(s.mean, 6) + "}";
}

// One arm's figures over the turn starts of a cell (or of a range: every
// mandatory class).
struct CellFigures {
  std::size_t turn_starts = 0;
  std::size_t short_searches = 0;    // fewer evaluations than asked (the search ended first)
  std::size_t no_complete_turn = 0;  // left out of the first ratio
  cells::Summary per_turn;
  cells::Summary inherited;
};

// `cls`: the mandatory class, or -1 for every one.
CellFigures figures(const std::vector<Start>& starts, const std::vector<std::vector<Row>>& rows, std::size_t arm,
                    int range, int cls, std::uint64_t evaluations) {
  CellFigures f;
  std::vector<double> per_turn;
  std::vector<double> inherited;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    if (starts[i].cell / cells::kMandatory != range) continue;
    if (cls >= 0 && starts[i].cell % cells::kMandatory != cls) continue;
    const cells::TreeCounts c = tree_counts(rows[i][arm]);
    ++f.turn_starts;
    if (c.evaluations < evaluations) ++f.short_searches;
    if (cells::has_complete_turn(c)) {
      per_turn.push_back(cells::evaluations_per_complete_turn(c));
    } else {
      ++f.no_complete_turn;
    }
    inherited.push_back(cells::inherited_per_evaluation(c));
  }
  f.per_turn = cells::summarise(std::move(per_turn));
  f.inherited = cells::summarise(std::move(inherited));
  return f;
}

int by_range(const RangeOptions& o) {
  const std::vector<t5d::tools::RecordRef> refs = t5d::tools::record_refs(o.records, o.lists, o.record_root);
  if (refs.empty()) {
    std::fputs("t5d-tree-count --by-range: no record given (RECORD ..., --record-list FILE)\n", stderr);
    return 2;
  }
  const cells::StartSelection sel = cells::select_turn_starts(refs, o.select);
  const std::vector<Start>& starts = sel.starts;
  std::size_t eligible = 0;
  for (const std::size_t n : sel.eligible) eligible += n;

  // Every counted turn start's searches: one per arm, to the evaluation budget.
  Limits limits;
  limits.playouts = 1000000;
  limits.evaluations = o.evaluations;
  std::vector<std::vector<Row>> rows(starts.size(), std::vector<Row>(o.arms.size()));
  std::vector<std::string> errors(starts.size());
  std::atomic<std::size_t> next{0};
  const auto work = [&] {
    for (;;) {
      const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= starts.size()) return;
      try {
        for (std::size_t a = 0; a < o.arms.size(); ++a) rows[i][a] = count(starts[i].state, o.arms[a], limits);
      } catch (const std::exception& e) {
        errors[i] = e.what();
      }
    }
  };
  const auto t0 = std::chrono::steady_clock::now();
  {
    std::vector<std::thread> pool;
    for (int t = 1; t < o.threads; ++t) pool.emplace_back(work);
    work();
    for (std::thread& t : pool) t.join();
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  for (std::size_t i = 0; i < starts.size(); ++i) {
    if (errors[i].empty()) continue;
    std::fprintf(stderr, "t5d-tree-count --by-range: %s action %zu: %s\n", starts[i].record.c_str(), starts[i].action,
                 errors[i].c_str());
    return 3;
  }

  std::string arm_keys;
  for (const int a : o.arms) arm_keys += (arm_keys.empty() ? "" : ",") + std::string(kArmKeys[a]);
  std::printf("t5d-tree-count --by-range (" T5D_VERSION_STRING "): design spec 15.3, \"Tree form: evaluations per "
              "complete turn, and inherited expansions per evaluation, by board-count range and mandatory count\"\n");
  std::printf("records:     %zu read in the order given under the standard rules (%zu refused or not read to their "
              "end: what they gave before is kept)\n",
              sel.records, sel.stopped_early);
  std::printf("turn starts: %zu their records play on from (none a game end); eligible - the action index %d or more "
              "and a multiple of %d - %zu; counted %zu: at most %d in a cell, %d of one record in a cell\n",
              sel.turn_starts, o.select.min_action, o.select.every, eligible, starts.size(), o.select.max_per_cell,
              o.select.per_record);
  if (sel.no_mandatory > 0) {
    std::printf("             %zu eligible turn starts with no mandatory board are filed in no cell\n", sel.no_mandatory);
  }
  std::printf("conditions:  each arm a fresh search from the turn start to %llu evaluations (playouts unbounded); the "
              "built-in random evaluator (seed 1), synchronous (one leaf per evaluation), no evaluation cache, square "
              "features off, every other parameter the search's default. Counts, not timings: deterministic, the same "
              "on any machine and under any load\n",
              u(o.evaluations));
  std::printf("definitions: a COMPLETE TURN is an evaluated turn start below the root - a submission reached and the "
              "position after it evaluated (a turn that ends the game is no evaluation and is not counted). "
              "EVALUATIONS PER COMPLETE TURN = (evaluated turn starts below the root + evaluated mid-turn nodes) / "
              "evaluated turn starts below the root: the root's own evaluation is left out on both sides, and the "
              "figure is 1 exactly where no mid-turn node is evaluated. INHERITED EXPANSIONS PER EVALUATION = "
              "mid-turn nodes expanded without the network / evaluations. The build's readings (build-log issue "
              "418), not the spec's words\n");
  std::printf("figures:     the median and 95th percentile over a cell's turn starts; \"short\": searches that ended "
              "before the evaluation budget; \"no turn\": searches with no complete turn evaluated, left out of the "
              "first figure\n");
  std::printf("arms:        %s\n", arm_keys.c_str());
  for (std::size_t a = 0; a < o.arms.size(); ++a) {
    std::printf("\n%s\n", kArmNames[o.arms[a]]);
    std::printf("  %-12s %-9s %11s %6s %8s  %-22s  %-22s\n", "range", "mandatory", "turn starts", "short", "no turn",
                "evals/turn median p95", "inherited/eval median p95");
    for (int range = 0; range < cells::kRanges; ++range) {
      for (int m = -1; m < cells::kMandatory; ++m) {
        // The range's every mandatory count ("all") after its classes.
        const int cls = m + 1 < cells::kMandatory ? m + 1 : -1;
        const CellFigures f = figures(starts, rows, a, range, cls, o.evaluations);
        if (f.turn_starts == 0 && cls >= 0) continue;
        if (f.turn_starts == 0) {
          std::printf("  %-12s %-9s %11d %6s %8s  no turn start\n", cells::range_name(range), "all", 0, "-", "-");
          continue;
        }
        std::printf("  %-12s %-9s %11zu %6zu %8zu  %10s %10s  %10s %10s\n", cells::range_name(range),
                    cls < 0 ? "all" : cells::mandatory_name(cls), f.turn_starts, f.short_searches, f.no_complete_turn,
                    f.per_turn.n == 0 ? "-" : fixed(f.per_turn.median, 3).c_str(),
                    f.per_turn.n == 0 ? "-" : fixed(f.per_turn.p95, 3).c_str(), fixed(f.inherited.median, 3).c_str(),
                    fixed(f.inherited.p95, 3).c_str());
      }
    }
  }
  std::fprintf(stderr, "t5d-tree-count --by-range: %zu turn starts x %zu arms counted in %.1f s on %d thread%s\n",
               starts.size(), o.arms.size(), seconds, o.threads, o.threads == 1 ? "" : "s");

  if (!o.json.empty()) {
    std::string j = "{\"tool\":\"t5d-tree-count\",\"mode\":\"by-range\",\"version\":\"" T5D_VERSION_STRING "\"";
    j += ",\"conditions\":{\"evaluations\":" + std::to_string(o.evaluations) + ",\"every\":" +
         std::to_string(o.select.every) + ",\"min_action\":" + std::to_string(o.select.min_action) +
         ",\"max_per_cell\":" + std::to_string(o.select.max_per_cell) + ",\"per_record\":" +
         std::to_string(o.select.per_record) +
         ",\"evaluator\":\"random, seed 1\",\"synchronous\":true,\"eval_cache_entries\":0,\"square_features\":\"none\""
         ",\"definition_complete_turn\":\"an evaluated turn start below the root\""
         ",\"definition_evaluations_per_complete_turn\":\"(evaluated turn starts below the root + evaluated "
         "mid-turn nodes) / evaluated turn starts below the root\""
         ",\"definition_inherited_per_evaluation\":\"mid-turn nodes expanded without the network / evaluations\"}";
    j += ",\"selection\":{\"records\":" + std::to_string(sel.records) + ",\"stopped_early\":" +
         std::to_string(sel.stopped_early) + ",\"turn_starts\":" + std::to_string(sel.turn_starts) +
         ",\"eligible\":" + std::to_string(eligible) + ",\"no_mandatory\":" + std::to_string(sel.no_mandatory) +
         ",\"counted\":" + std::to_string(starts.size()) + "}";
    j += ",\"arms\":[";
    for (std::size_t a = 0; a < o.arms.size(); ++a) {
      j += std::string(a == 0 ? "" : ",") + "{\"arm\":" + json_string(kArmKeys[o.arms[a]]) + ",\"name\":" +
           json_string(kArmNames[o.arms[a]]) + ",\"cells\":[";
      bool first = true;
      for (int range = 0; range < cells::kRanges; ++range) {
        for (int cls = -1; cls < cells::kMandatory; ++cls) {
          const CellFigures f = figures(starts, rows, a, range, cls, o.evaluations);
          j += std::string(first ? "" : ",") + "{\"range\":" + json_string(cells::range_name(range)) +
               ",\"mandatory\":" + json_string(cls < 0 ? "all" : cells::mandatory_name(cls)) + ",\"turn_starts\":" +
               std::to_string(f.turn_starts) + ",\"short\":" + std::to_string(f.short_searches) +
               ",\"no_complete_turn\":" + std::to_string(f.no_complete_turn) + ",\"evaluations_per_complete_turn\":" +
               summary_json(f.per_turn) + ",\"inherited_per_evaluation\":" + summary_json(f.inherited) + "}";
          first = false;
        }
      }
      j += "]}";
    }
    j += "],\"positions\":[";
    for (std::size_t i = 0; i < starts.size(); ++i) {
      const Start& s = starts[i];
      j += std::string(i == 0 ? "" : ",") + "{\"record\":" + json_string(s.record) + ",\"action\":" +
           std::to_string(s.action) + ",\"boards\":" + std::to_string(s.boards) + ",\"mandatory\":" +
           std::to_string(s.mandatory) + ",\"playable\":" + std::to_string(s.playable) + ",\"range\":" +
           json_string(cells::range_name(s.cell / cells::kMandatory)) + ",\"arms\":[";
      for (std::size_t a = 0; a < o.arms.size(); ++a) {
        const Row& r = rows[i][a];
        j += std::string(a == 0 ? "" : ",") + "{\"arm\":" + json_string(kArmKeys[o.arms[a]]) + ",\"evaluations\":" +
             std::to_string(r.evaluations) + ",\"playouts\":" + std::to_string(r.playouts) +
             ",\"turn_starts_evaluated\":" + std::to_string(r.turn_starts_evaluated) + ",\"mid_turn_evaluated\":" +
             std::to_string(r.mid_turn_evaluated) + ",\"inherited\":" + std::to_string(r.inherited) + "}";
      }
      j += "]}";
    }
    j += "]}\n";
    std::ofstream out(o.json, std::ios::binary);
    out << j;
    if (!out) {
      std::fprintf(stderr, "t5d-tree-count --by-range: cannot write %s\n", o.json.c_str());
      return 3;
    }
  }
  return 0;
}

void usage(std::FILE* to) {
  std::fputs(
      "usage: t5d-tree-count RECORD ...\n"
      "         seven turn starts of the records, every arm: evaluation counts (D2's tree and flat forms against\n"
      "         the ply split)\n"
      "       t5d-tree-count --by-range [options] [RECORD ...]\n"
      "         design spec 15.3's row - the tree form's evaluations per complete turn and inherited expansions\n"
      "         per evaluation, by board-count range and mandatory count\n"
      "  --record-list FILE   records listed one per line ('#' lines skipped); repeatable\n"
      "  --record-root DIR    where a list's relative paths are taken from\n"
      "  --every K            eligible: a turn start whose action index is a multiple of K (4)\n"
      "  --min-action A       ... and A or more (0)\n"
      "  --max-per-cell N     turn starts counted in a cell - a range and a mandatory count - at most (25)\n"
      "  --per-record N       ... and of one record in a cell at most (1)\n"
      "  --evaluations E      each search's evaluation budget (400)\n"
      "  --arms LIST          of ply, tree, tree-rewind, tree-none, flat (ply,tree,tree-rewind,tree-none)\n"
      "  --threads T          threads the searches are shared out to (1); the counts do not depend on it\n"
      "  --json FILE          the figures, and every turn start's counts\n",
      to);
}

int parse_int(const std::string& option, const char* text, int least) {
  char* end = nullptr;
  const long v = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || v < least || v > 1000000000L) {
    throw std::invalid_argument(option + ": a whole number, " + std::to_string(least) + " or more");
  }
  return static_cast<int>(v);
}

std::vector<int> parse_arms(const std::string& text) {
  std::vector<int> out;
  std::size_t start = 0;
  while (start <= text.size()) {
    std::size_t end = text.find(',', start);
    if (end == std::string::npos) end = text.size();
    const std::string key = text.substr(start, end - start);
    int arm = -1;
    for (int a = 0; a < kArms; ++a) {
      if (key == kArmKeys[a]) arm = a;
    }
    if (arm < 0) throw std::invalid_argument("--arms: no arm \"" + key + "\" (ply, tree, tree-rewind, tree-none, flat)");
    for (const int had : out) {
      if (had == arm) throw std::invalid_argument("--arms: \"" + key + "\" twice");
    }
    out.push_back(arm);
    start = end + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
    usage(argc < 2 ? stderr : stdout);
    return argc < 2 ? 2 : 0;
  }
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  bool ranged = false;
  for (int i = 1; i < argc; ++i) ranged = ranged || std::strcmp(argv[i], "--by-range") == 0;
  if (!ranged) return seven_turn_starts(argc, argv);
  try {
    RangeOptions o;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const auto value = [&]() -> const char* {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--by-range") {
      } else if (a == "--record-list") {
        o.lists.emplace_back(value());
      } else if (a == "--record-root") {
        o.record_root = value();
      } else if (a == "--every") {
        o.select.every = parse_int(a, value(), 1);
      } else if (a == "--min-action") {
        o.select.min_action = parse_int(a, value(), 0);
      } else if (a == "--max-per-cell") {
        o.select.max_per_cell = parse_int(a, value(), 1);
      } else if (a == "--per-record") {
        o.select.per_record = parse_int(a, value(), 1);
      } else if (a == "--evaluations") {
        o.evaluations = static_cast<std::uint64_t>(parse_int(a, value(), 1));
      } else if (a == "--arms") {
        o.arms = parse_arms(value());
      } else if (a == "--threads") {
        o.threads = parse_int(a, value(), 1);
      } else if (a == "--json") {
        o.json = value();
      } else if (a == "--help" || a == "-h") {
        usage(stdout);
        return 0;
      } else if (a.rfind("--", 0) == 0) {
        throw std::invalid_argument("unknown option " + a);
      } else {
        o.records.push_back(a);
      }
    }
    return by_range(o);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "t5d-tree-count: %s\n", e.what());
    usage(stderr);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-tree-count: %s\n", e.what());
    return 3;
  }
}
