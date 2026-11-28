// t5d-turngen-time: what D2's flat form's turn generator costs a node (design
// spec 15.3: "Turn generation (flat form): time to the first k legal turns, and
// per widening step, by board-count range and mandatory count"; register D2;
// docs/MEASUREMENTS.md). A TIMING: it needs a quiet machine, and says how
// busy the machine was.
//
//   t5d-turngen-time [options] [RECORD ...]      (--help lists the options)
//
// POSITIONS, turn starts all:
//  * of RECORDS - given directly and by --record-list files (one path a line,
//    '#' lines skipped; relative paths from --record-root) - replayed in
//    order under the standard rules: a turn start is eligible when its record
//    plays a further action from it (no game end) and its action index is
//    --min-action or more and a multiple of --every; it is taken while its
//    cell wants one - at most --max-per-cell in a cell, --per-record of one
//    record in a cell (tools/turn_cells.hpp: the choice is a function of the
//    records' order and these four numbers);
//  * of t5d-bench's SYNTHETIC classes (--synthetic; tools/synthetic.hpp:
//    seeded random pseudo-legal play, for cost only - never chess anyone
//    plays): 100, 160, 240, 400 and 600 boards with 6, 7, 8, 10 and 12 active
//    timelines, --games games each, a game's seed the bench's function of
//    (--seed, N, T, game) - the bench's own class positions.
// Each is filed in its cell: board-count range (< 50, 50-99, 100-199, 200+)
// and mandatory boards at the turn's start (1, 2, 3, 4+). Records' turn
// starts and synthetic ones are reported apart, never in one figure.
//
// WHAT IS TIMED (tools/turngen_time.hpp): one generator a position a round,
// made and asked for its turns as the search's flat form does at a node -
// --unit successors, a node budget of --generator-nodes (the search's
// placeholders) - with scores that are a seeded function of each option's
// coordinates, uniform in [-S, S] (--score-scale S, 3; --flat-scores: every
// option 0, every turn a tie):
//   the time to the first k turns, k = 1, 8, 32 - from just before the
//     generator's construction to the return of the next() that gives turn k;
//   a widening step - one next() after the first --k0 turns (16, the search's
//     cap), --widen of them (16) at most.
// A position with fewer than k legal turns, or whose generator its budget
// stopped first, is counted and not timed at k.
//
// METHOD (t5d-bench's). One thread. Every position is timed once a round,
// round after round over all of them - INTERLEAVED, so that a drift in the
// machine's load spreads over every position - after a WARM-UP round that is
// not counted. A position's figure is the median over its --rounds rounds; a
// cell's the median and 95th percentile over its positions (the times to k)
// or over its positions' every widening step. THE MACHINE'S LOAD - the busy
// fraction of all its processors - is read around every round and printed;
// the output names the compiler, the flags and the CPU. --pin CPU pins the
// thread to a logical processor.
//
// Not one of the truth's sources (CMakeLists.txt, T5D_TRUTH_SOURCES): it
// decides no verdict.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "sixb_roots.hpp"
#include "synthetic.hpp"
#include "t5d/gate.hpp"
#include "t5d/state.hpp"
#include "t5d/turngen.hpp"
#include "tool_support.hpp"
#include "turn_cells.hpp"
#include "turngen_time.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sched.h>
#endif

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif
#ifndef T5D_BUILD_TYPE
#define T5D_BUILD_TYPE "unknown"
#endif
#ifndef T5D_CXX_FLAGS
#define T5D_CXX_FLAGS "unknown"
#endif

namespace {

namespace tt = t5d::tools::turngen_time;
namespace cells = t5d::tools::cells;
using Clock = std::chrono::steady_clock;
using t5d::gate::json_string;

// ---- the machine and the build -------------------------------------------------------------------

std::string trim(std::string s) {
  const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  while (!s.empty() && space(s.back())) s.pop_back();
  std::size_t i = 0;
  while (i < s.size() && space(s[i])) ++i;
  return s.substr(i);
}

std::string compiler_text() {
#if defined(__clang__)
  return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  return std::string("g++ ") + __VERSION__;
#else
  return "unknown";
#endif
}

std::string cpu_model() {
#if defined(_WIN32)
  char name[256] = {};
  DWORD size = sizeof name;
  if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString",
                   RRF_RT_REG_SZ, nullptr, name, &size) == ERROR_SUCCESS) {
    return trim(name);
  }
  return "unknown";
#else
  std::FILE* f = std::fopen("/proc/cpuinfo", "r");
  if (f == nullptr) return "unknown";
  char line[512];
  std::string model = "unknown";
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (std::strncmp(line, "model name", 10) == 0) {
      const char* colon = std::strchr(line, ':');
      if (colon != nullptr) model = trim(colon + 1);
      break;
    }
  }
  std::fclose(f);
  return model;
#endif
}

// The machine's processor time so far: idle and total (any unit).
struct CpuTimes {
  std::uint64_t idle = 0;
  std::uint64_t total = 0;
};

CpuTimes cpu_times() {
  CpuTimes t;
#if defined(_WIN32)
  FILETIME idle, kernel, user;
  if (GetSystemTimes(&idle, &kernel, &user)) {
    const auto u64 = [](FILETIME f) { return (static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    t.idle = u64(idle);
    t.total = u64(kernel) + u64(user);  // kernel time includes idle time
  }
#else
  std::FILE* f = std::fopen("/proc/stat", "r");
  if (f != nullptr) {
    unsigned long long v[8] = {};
    if (std::fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6],
                    &v[7]) == 8) {
      t.idle = v[3] + v[4];  // idle and iowait
      for (const unsigned long long x : v) t.total += x;
    }
    std::fclose(f);
  }
#endif
  return t;
}

// The busy fraction of every processor between two readings; -1 if unknown.
double busy_between(const CpuTimes& a, const CpuTimes& b) {
  if (b.total <= a.total || b.idle < a.idle) return -1;
  return 1.0 - static_cast<double>(b.idle - a.idle) / static_cast<double>(b.total - a.total);
}

// Pins the calling thread to one logical processor; false if it cannot.
bool pin_current_thread(int cpu) {
#if defined(_WIN32)
  if (cpu < 0 || cpu >= 64) return false;
  return SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu) != 0;
#elif defined(__linux__)
  if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<std::size_t>(cpu), &set);
  return sched_setaffinity(0, sizeof set, &set) == 0;
#else
  (void)cpu;
  return false;
#endif
}

std::string fixed(double value, int digits) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", digits, value);
  return buf;
}

std::string percent(double fraction) { return fraction < 0 ? "?" : fixed(100 * fraction, 1) + "%"; }

// ---- options -------------------------------------------------------------------------------------

struct Options {
  std::vector<std::string> records;
  std::vector<std::string> lists;
  std::string record_root;
  cells::SelectOptions select{4, 0, 20, 1};
  bool synthetic = false;
  std::vector<std::pair<int, int>> classes = {{100, 6}, {160, 7}, {240, 8}, {400, 10}, {600, 12}};  // t5d-bench's
  int games = 2;
  std::uint64_t seed = 1;
  int rounds = 7;
  tt::Options timing;
  bool flat_scores = false;
  double score_scale = 3.0;
  std::uint64_t score_seed = 1;
  int pin = -1;
  bool per_position = false;
  std::string json;
};

void usage(std::FILE* to) {
  std::fputs(
      "usage: t5d-turngen-time [options] [RECORD ...]\n"
      "  the flat form's turn generator: the time to its first 1, 8 and 32 turns and per widening step, by\n"
      "  board-count range and mandatory count (design spec 15.3). A timing: run it on a quiet machine.\n"
      "positions:\n"
      "  --record-list FILE    records listed one per line ('#' lines skipped); repeatable\n"
      "  --record-root DIR     where a list's relative paths are taken from\n"
      "  --every K             eligible: a turn start whose action index is a multiple of K (4)\n"
      "  --min-action A        ... and A or more (0)\n"
      "  --max-per-cell N      turn starts taken in a cell - a range and a mandatory count - at most (20)\n"
      "  --per-record N        ... and of one record in a cell at most (1)\n"
      "  --synthetic           t5d-bench's synthetic classes too: 100-600 boards, random play (for cost only)\n"
      "  --games G             synthetic games a class (2)\n"
      "  --seed S              the synthetic games' seed (1)\n"
      "what is timed:\n"
      "  --unit successors|actions   what the generator emits (successors, the search's)\n"
      "  --generator-nodes N   its node budget; 0: none (200000, the search's)\n"
      "  --k0 N                the turns before the widening steps (16, the search's cap)\n"
      "  --widen N             the widening steps timed (16)\n"
      "  --score-scale S       the options' scores are uniform in [-S, S] (3)\n"
      "  --score-seed N        ... a function of this seed and their coordinates (1)\n"
      "  --flat-scores         every option scores 0: every turn ties\n"
      "method:\n"
      "  --rounds R            rounds over every position, after one warm-up round (7)\n"
      "  --pin CPU             pin the thread to a logical processor (not pinned)\n"
      "output:\n"
      "  --per-position        every position's figures too (the JSON's \"positions\")\n"
      "  --json FILE           the conditions and the figures\n",
      to);
}

// A count on the command line: `least` to 1,000,000,000.
int parse_count(const std::string& option, const char* text, int least) {
  char* end = nullptr;
  const long long v = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || v < least || v > 1000000000LL) {
    throw std::invalid_argument(option + ": a whole number, " + std::to_string(least) + " to 1000000000");
  }
  return static_cast<int>(v);
}

// A seed or a node budget: any unsigned 64-bit number.
std::uint64_t parse_u64(const std::string& option, const char* text) {
  char* end = nullptr;
  const unsigned long long v = std::strtoull(text, &end, 10);
  if (text[0] == '-' || end == text || *end != '\0') {
    throw std::invalid_argument(option + ": a whole number, 0 or more");
  }
  return v;
}

// The bench's seed of a synthetic class's game (t5d_bench.cpp, class_seed): the
// same games, so the same class positions.
std::uint64_t class_seed(std::uint64_t seed, int boards, int active, int game) {
  return seed + std::uint64_t{1000003} * static_cast<std::uint64_t>(boards) +
         std::uint64_t{10007} * static_cast<std::uint64_t>(active) + std::uint64_t{37} * static_cast<std::uint64_t>(game);
}

// ---- the positions -------------------------------------------------------------------------------

struct Position {
  explicit Position(t5d::GameState s) : state(std::move(s)) {}
  std::string label;
  bool synthetic = false;
  int boards = 0;
  int mandatory = 0;
  std::size_t playable = 0;
  t5d::GameState state;
  std::vector<tt::Sample> rounds;
};

std::string summary_json(const cells::Summary& s) {
  return "{\"n\":" + std::to_string(s.n) + ",\"median\":" + fixed(s.median, 9) + ",\"p95\":" + fixed(s.p95, 9) +
         ",\"min\":" + fixed(s.min, 9) + ",\"max\":" + fixed(s.max, 9) + ",\"mean\":" + fixed(s.mean, 9) + "}";
}

std::string ms(const cells::Summary& s, bool p95) {
  return s.n == 0 ? "-" : fixed(1e3 * (p95 ? s.p95 : s.median), 3);
}

std::string us(const cells::Summary& s, bool p95) {
  return s.n == 0 ? "-" : fixed(1e6 * (p95 ? s.p95 : s.median), 1);
}

std::string mean_us(const cells::Summary& s) { return s.n == 0 ? "-" : fixed(1e6 * s.mean, 1); }

// One source's table: its cells, and each range over every mandatory count.
void print_group(const char* title, const std::vector<tt::PositionFigures>& figures) {
  std::printf("\n%s: %zu turn starts\n", title, figures.size());
  if (figures.empty()) return;
  std::printf("  %-12s %-9s %9s | %-27s | %-27s | %-27s | %-39s | %9s %7s\n", "range", "mandatory", "positions",
              "turn 1: n  median ms    p95", "turn 8: n  median ms    p95", "turn 32: n median ms    p95",
              "widening: steps med us  p95 us  mean us", "exhausted", "stopped");
  for (int range = 0; range < cells::kRanges; ++range) {
    for (int m = 0; m <= cells::kMandatory; ++m) {
      const int cls = m < cells::kMandatory ? m : -1;  // the range's classes, then every one ("all")
      const tt::CellFigures c = tt::cell_figures(figures, range, cls);
      if (c.positions == 0) continue;
      std::printf("  %-12s %-9s %9zu | %9zu %8s %8s | %9zu %8s %8s | %9zu %8s %8s | %12zu %8s %8s %8s | %9zu %7zu\n",
                  cells::range_name(range), cls < 0 ? "all" : cells::mandatory_name(cls), c.positions, c.reached[0],
                  ms(c.first[0], false).c_str(), ms(c.first[0], true).c_str(), c.reached[1],
                  ms(c.first[1], false).c_str(), ms(c.first[1], true).c_str(), c.reached[2],
                  ms(c.first[2], false).c_str(), ms(c.first[2], true).c_str(), c.step.n, us(c.step, false).c_str(),
                  us(c.step, true).c_str(), mean_us(c.step).c_str(), c.exhausted, c.stopped);
    }
  }
}

std::string group_json(const char* source, const std::vector<tt::PositionFigures>& figures) {
  std::string j = std::string("{\"source\":") + json_string(source) + ",\"positions\":" + std::to_string(figures.size()) +
                  ",\"cells\":[";
  bool first = true;
  for (int range = 0; range < cells::kRanges; ++range) {
    for (int cls = -1; cls < cells::kMandatory; ++cls) {
      const tt::CellFigures c = tt::cell_figures(figures, range, cls);
      if (c.positions == 0) continue;
      j += std::string(first ? "" : ",") + "{\"range\":" + json_string(cells::range_name(range)) + ",\"mandatory\":" +
           json_string(cls < 0 ? "all" : cells::mandatory_name(cls)) + ",\"positions\":" + std::to_string(c.positions) +
           ",\"exhausted\":" + std::to_string(c.exhausted) + ",\"stopped\":" + std::to_string(c.stopped) + ",\"first\":[";
      for (int i = 0; i < tt::kFirsts; ++i) {
        j += std::string(i == 0 ? "" : ",") + "{\"k\":" + std::to_string(tt::kFirst[i]) + ",\"reached\":" +
             std::to_string(c.reached[i]) + ",\"seconds\":" + summary_json(c.first[i]) + "}";
      }
      j += "],\"widening_step\":{\"positions\":" + std::to_string(c.widening_positions) + ",\"seconds\":" +
           summary_json(c.step) + "}}";
      first = false;
    }
  }
  return j + "]}";
}

int run(const Options& o) {
  std::vector<Position> positions;

  // -- the records' turn starts
  const std::vector<t5d::tools::RecordRef> refs = t5d::tools::record_refs(o.records, o.lists, o.record_root);
  cells::StartSelection sel;
  if (!refs.empty()) sel = cells::select_turn_starts(refs, o.select);
  std::size_t eligible = 0;
  for (const std::size_t n : sel.eligible) eligible += n;
  for (const cells::TurnStart& s : sel.starts) {
    Position p(s.state);
    p.label = s.record + " action " + std::to_string(s.action);
    p.boards = s.boards;
    p.mandatory = s.mandatory;
    p.playable = s.playable;
    positions.push_back(std::move(p));
  }
  const std::size_t from_records = positions.size();

  // -- the bench's synthetic classes
  std::size_t not_reached = 0;
  std::size_t no_mandatory = 0;
  if (o.synthetic) {
    for (const auto& [n, t] : o.classes) {
      for (int g = 0; g < o.games; ++g) {
        t5d::synthetic::Spec spec;
        spec.seed = class_seed(o.seed, n, t, g);
        spec.target_active = t;
        spec.target_boards = n;
        const t5d::synthetic::Game game = t5d::synthetic::play_until_reached(spec);
        if (game.stop != t5d::synthetic::Stop::Target) {
          ++not_reached;
          continue;
        }
        Position p(game.state);
        p.label = "synthetic N" + std::to_string(n) + " T" + std::to_string(t) + " seed " + std::to_string(game.seed);
        p.synthetic = true;
        p.boards = static_cast<int>(game.state.multiverse().board_count());
        p.mandatory = game.state.mandatory_count();
        p.playable = game.state.playable_timelines().size();
        if (cells::cell_of(p.boards, p.mandatory) < 0) {
          ++no_mandatory;
          continue;
        }
        positions.push_back(std::move(p));
      }
    }
  }
  if (positions.empty()) {
    std::fputs("t5d-turngen-time: no position (RECORD ..., --record-list FILE, --synthetic)\n", stderr);
    return 2;
  }

  const bool pinned = o.pin >= 0 && pin_current_thread(o.pin);
  std::vector<tt::HashScorer> scorers;
  scorers.reserve(positions.size());
  for (std::size_t i = 0; i < positions.size(); ++i) {
    scorers.emplace_back(o.score_seed + i, o.flat_scores ? 0.0 : o.score_scale);
  }

  std::printf("t5d-turngen-time %s - design spec 15.3, \"Turn generation (flat form): time to the first k legal "
              "turns, and per widening step, by board-count range and mandatory count\"\n",
              T5D_VERSION_STRING);
  std::printf("machine: %s, %u logical processors; one thread, %s\n", cpu_model().c_str(),
              std::thread::hardware_concurrency(),
              pinned ? ("pinned to logical processor " + std::to_string(o.pin)).c_str() : "not pinned");
  std::printf("build:   %s, %s, flags \"%s\"\n", compiler_text().c_str(), T5D_BUILD_TYPE, trim(T5D_CXX_FLAGS).c_str());
  std::printf("records: %zu read in the order given under the standard rules (%zu refused or not read to their end); "
              "%zu turn starts their records play on from; eligible - the action index %d or more and a multiple of "
              "%d - %zu; taken %zu: at most %d in a cell, %d of one record in a cell\n",
              sel.records, sel.stopped_early, sel.turn_starts, o.select.min_action, o.select.every, eligible,
              from_records, o.select.max_per_cell, o.select.per_record);
  if (o.synthetic) {
    std::printf("synthetic: t5d-bench's classes, %d games a class, seed %llu - %zu turn starts (%zu games did not "
                "reach their board count, %zu positions with no mandatory board left out). SYNTHETIC positions are "
                "random play: for cost only\n",
                o.games, static_cast<unsigned long long>(o.seed), positions.size() - from_records, not_reached,
                no_mandatory);
  }
  std::printf("timed:   a generator a position a round, as the search's flat form makes one at a node - unit %s, "
              "node budget %s\n",
              o.timing.unit == t5d::TurnUnit::Successors ? "successors" : "actions",
              o.timing.generator_nodes == 0 ? "none" : std::to_string(o.timing.generator_nodes).c_str());
  std::printf("definitions: THE TIME TO THE FIRST k TURNS (k = 1, 8, 32) runs from just before the generator's "
              "construction - the state's copy, its axes, the scores: what a node pays before its first turn - to "
              "the return of the next() that gives turn k; a position with fewer than k legal turns, or stopped by "
              "its budget first, is counted and not timed at k. A WIDENING STEP is one next() after the first %zu "
              "turns (the turns an expansion generates at most), timed from its call to its return; %zu of them. "
              "The build's readings (build-log issue 419), not the spec's words\n",
              o.timing.k0, o.timing.widen);
  std::printf("scores:  NOT A NETWORK'S - %s. What best-first generation costs depends on the scores: a trained "
              "policy's figure is not this one\n",
              o.flat_scores ? "every option scores 0, so every turn ties"
                            : ("a seeded hash of each option's coordinates, uniform in [-" + fixed(o.score_scale, 2) +
                               ", " + fixed(o.score_scale, 2) + "], seed " + std::to_string(o.score_seed) +
                               " (a J6 twin scores as its jump)")
                                  .c_str());
  std::printf("method:  %d interleaved rounds over every position after a warm-up round; a position's figure the "
              "median over its rounds, a cell's the median and 95th percentile over its positions (or their every "
              "widening step - with the steps' mean: a generator records up to 32 realisations of a point at once, "
              "so many a step returns a turn already found and the cost sits in the tail)\n",
              o.rounds);
  std::fflush(stdout);

  // -- the rounds: every position once a round, the warm-up round first
  std::vector<double> loads;
  std::uint64_t sink = 0;
  CpuTimes counted_from;  // the machine's times when the first counted round began
  const Clock::time_point t0 = Clock::now();
  for (int round = 0; round <= o.rounds; ++round) {
    const CpuTimes before = cpu_times();
    if (round == 1) counted_from = before;
    for (std::size_t i = 0; i < positions.size(); ++i) {
      tt::Sample s = tt::time_generator(positions[i].state, scorers[i], o.timing);
      sink += s.turns + s.nodes;
      if (round > 0) positions[i].rounds.push_back(std::move(s));
    }
    if (round > 0) loads.push_back(busy_between(before, cpu_times()));
  }
  // Over the counted rounds together: a round shorter than the system's clock
  // tick has no reading of its own ("?").
  const double load_overall = busy_between(counted_from, cpu_times());
  const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();

  std::string load_text;
  std::vector<double> known;
  for (const double b : loads) {
    load_text += (load_text.empty() ? "" : " ") + percent(b);
    if (b >= 0) known.push_back(b);
  }
  const cells::Summary load = cells::summarise(known);
  std::printf("load:    the machine's busy fraction around each round: %s (median %s, greatest %s); over the counted "
              "rounds together %s; this thread is one of %u logical processors' worth\n",
              load_text.c_str(), known.empty() ? "?" : percent(load.median).c_str(),
              known.empty() ? "?" : percent(load.max).c_str(), percent(load_overall).c_str(),
              std::thread::hardware_concurrency());
  std::printf("run:     %zu positions x %d rounds in %.1f s (sink %llu)\n", positions.size(), o.rounds + 1, seconds,
              static_cast<unsigned long long>(sink));

  std::vector<tt::PositionFigures> of_records;
  std::vector<tt::PositionFigures> of_synthetic;
  std::vector<tt::PositionFigures> every;
  for (const Position& p : positions) {
    every.push_back(tt::position_figures(p.boards, p.mandatory, p.rounds));
    (p.synthetic ? of_synthetic : of_records).push_back(every.back());
  }
  print_group("RECORDS", of_records);
  if (o.synthetic) print_group("SYNTHETIC (random play: for cost only)", of_synthetic);
  if (o.per_position) {
    std::printf("\nPOSITIONS (seconds: to turns 1, 8, 32; the widening steps' median)\n");
    for (std::size_t i = 0; i < positions.size(); ++i) {
      const tt::PositionFigures& f = every[i];
      std::printf("  %s: %d boards, %d mandatory, %zu playable; %zu turns%s%s; %s %s %s; %s\n",
                  positions[i].label.c_str(), f.boards, f.mandatory, positions[i].playable, f.turns,
                  f.exhausted ? " (exhausted)" : "", f.stopped ? " (stopped by the budget)" : "",
                  f.first[0] < 0 ? "-" : fixed(f.first[0], 6).c_str(), f.first[1] < 0 ? "-" : fixed(f.first[1], 6).c_str(),
                  f.first[2] < 0 ? "-" : fixed(f.first[2], 6).c_str(),
                  f.steps.empty() ? "-" : fixed(cells::summarise(f.steps).median, 9).c_str());
    }
  }

  if (!o.json.empty()) {
    std::string j = "{\"tool\":\"t5d-turngen-time\",\"version\":\"" T5D_VERSION_STRING "\"";
    j += ",\"build\":{\"compiler\":" + json_string(compiler_text()) + ",\"build_type\":" + json_string(T5D_BUILD_TYPE) +
         ",\"flags\":" + json_string(trim(T5D_CXX_FLAGS)) + "}";
    j += ",\"machine\":{\"cpu\":" + json_string(cpu_model()) + ",\"logical_processors\":" +
         std::to_string(std::thread::hardware_concurrency()) + ",\"pinned\":" + (pinned ? std::to_string(o.pin) : "null") +
         "}";
    j += ",\"conditions\":{\"unit\":" +
         json_string(o.timing.unit == t5d::TurnUnit::Successors ? "successors" : "actions") + ",\"generator_nodes\":" +
         std::to_string(o.timing.generator_nodes) + ",\"k0\":" + std::to_string(o.timing.k0) + ",\"widen\":" +
         std::to_string(o.timing.widen) + ",\"scores\":" + json_string(o.flat_scores ? "flat" : "hash") +
         ",\"score_scale\":" + fixed(o.flat_scores ? 0.0 : o.score_scale, 3) + ",\"score_seed\":" +
         std::to_string(o.score_seed) + ",\"rounds\":" + std::to_string(o.rounds) + ",\"warm_up_rounds\":1" +
         ",\"every\":" + std::to_string(o.select.every) + ",\"min_action\":" + std::to_string(o.select.min_action) +
         ",\"max_per_cell\":" + std::to_string(o.select.max_per_cell) + ",\"per_record\":" +
         std::to_string(o.select.per_record) + ",\"synthetic_games\":" + std::to_string(o.synthetic ? o.games : 0) +
         ",\"synthetic_seed\":" + std::to_string(o.seed) +
         ",\"definition_first_k\":\"seconds from just before the generator's construction to the return of the "
         "next() that gives turn k\",\"definition_widening_step\":\"one next() after the first k0 turns, from its "
         "call to its return\",\"definition_scores\":\"a seeded hash of each option's coordinates, not a network's\"}";
    j += ",\"load\":{\"rounds\":[";
    for (std::size_t i = 0; i < loads.size(); ++i) j += std::string(i == 0 ? "" : ",") + fixed(loads[i], 4);
    j += "],\"median\":" + fixed(known.empty() ? -1.0 : load.median, 4) + ",\"max\":" +
         fixed(known.empty() ? -1.0 : load.max, 4) + ",\"overall\":" + fixed(load_overall, 4) + "}";
    j += ",\"seconds\":" + fixed(seconds, 3);
    j += ",\"groups\":[" + group_json("records", of_records);
    if (o.synthetic) j += "," + group_json("synthetic", of_synthetic);
    j += "]";
    if (o.per_position) {
      j += ",\"positions\":[";
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const tt::PositionFigures& f = every[i];
        j += std::string(i == 0 ? "" : ",") + "{\"source\":" +
             json_string(positions[i].synthetic ? "synthetic" : "records") + ",\"label\":" +
             json_string(positions[i].label) + ",\"boards\":" + std::to_string(f.boards) + ",\"mandatory\":" +
             std::to_string(f.mandatory) + ",\"playable\":" + std::to_string(positions[i].playable) + ",\"turns\":" +
             std::to_string(f.turns) + ",\"exhausted\":" + (f.exhausted ? "true" : "false") + ",\"stopped\":" +
             (f.stopped ? "true" : "false") + ",\"first_seconds\":[";
        for (int k = 0; k < tt::kFirsts; ++k) j += std::string(k == 0 ? "" : ",") + fixed(f.first[k], 9);
        j += "],\"step_seconds\":[";
        for (std::size_t k = 0; k < f.steps.size(); ++k) j += std::string(k == 0 ? "" : ",") + fixed(f.steps[k], 9);
        j += "]}";
      }
      j += "]";
    }
    j += "}\n";
    std::ofstream out(o.json, std::ios::binary);
    out << j;
    if (!out) {
      std::fprintf(stderr, "t5d-turngen-time: cannot write %s\n", o.json.c_str());
      return 3;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options o;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const auto value = [&]() -> const char* {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--help" || a == "-h") {
        usage(stdout);
        return 0;
      } else if (a == "--record-list") {
        o.lists.emplace_back(value());
      } else if (a == "--record-root") {
        o.record_root = value();
      } else if (a == "--every") {
        o.select.every = parse_count(a, value(), 1);
      } else if (a == "--min-action") {
        o.select.min_action = parse_count(a, value(), 0);
      } else if (a == "--max-per-cell") {
        o.select.max_per_cell = parse_count(a, value(), 1);
      } else if (a == "--per-record") {
        o.select.per_record = parse_count(a, value(), 1);
      } else if (a == "--synthetic") {
        o.synthetic = true;
      } else if (a == "--games") {
        o.games = parse_count(a, value(), 1);
      } else if (a == "--seed") {
        o.seed = parse_u64(a, value());
      } else if (a == "--unit") {
        const std::string unit = value();
        if (unit != "successors" && unit != "actions") throw std::invalid_argument("--unit: successors or actions");
        o.timing.unit = unit == "successors" ? t5d::TurnUnit::Successors : t5d::TurnUnit::Actions;
      } else if (a == "--generator-nodes") {
        o.timing.generator_nodes = parse_u64(a, value());
      } else if (a == "--k0") {
        o.timing.k0 = static_cast<std::size_t>(parse_count(a, value(), 0));
      } else if (a == "--widen") {
        o.timing.widen = static_cast<std::size_t>(parse_count(a, value(), 0));
      } else if (a == "--score-scale") {
        char* end = nullptr;
        const char* text = value();
        o.score_scale = std::strtod(text, &end);
        if (end == text || *end != '\0' || !(o.score_scale > 0) || o.score_scale > 1e6) {
          throw std::invalid_argument("--score-scale: a positive number (--flat-scores for none)");
        }
      } else if (a == "--score-seed") {
        o.score_seed = parse_u64(a, value());
      } else if (a == "--flat-scores") {
        o.flat_scores = true;
      } else if (a == "--rounds") {
        o.rounds = parse_count(a, value(), 1);
      } else if (a == "--pin") {
        o.pin = parse_count(a, value(), 0);
      } else if (a == "--per-position") {
        o.per_position = true;
      } else if (a == "--json") {
        o.json = value();
      } else if (a.rfind("--", 0) == 0) {
        throw std::invalid_argument("unknown option " + a);
      } else {
        o.records.push_back(a);
      }
    }
    return run(o);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "t5d-turngen-time: %s\n", e.what());
    usage(stderr);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-turngen-time: %s\n", e.what());
    return 3;
  }
}
