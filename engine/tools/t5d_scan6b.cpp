// t5d-scan6b: experiment 6b - the tactical scanner's horizon per node budget,
// with the stand-pat check (design spec §11.1 "defining quiet", §15.6 "The
// scanner (6b) - anchored"; register M8, H1-H6).
//
//   t5d-scan6b [options] [RECORD ...]
//
// POSITIONS. ROOTS: the turn starts of the 5dpgn records given (replayed as
// the gate adapter's parse mode does, under the ruleset given), every K-th
// (--every), distinct by position; and SYNTHETIC positions (--synthetic):
// random legal play from named variants (--synthetic-variants) and seeded
// random pseudo-legal games of t5d-bench's classes (--synthetic-classes N:T),
// each labelled SYNTHETIC and never part of a verdict. LEAVES: from each
// root, --leaves positions --leaf-depth half-turns below it by random LEGAL
// play (a random pseudo-legal seed move, completed by the scanner's rule;
// E4's witness when no seed completes). A position whose side to move could
// capture an enemy royal (the record's previous action broke C4) is not
// scanned - its leaves are, the first legal action below it restoring C4 -
// nor one with no legal action (terminal: no decision to make); both are
// counted.
//
// THE SCAN of every position: iterative deepening in forcing half-turns, the
// deterministic mode (a node budget, no clock: H15), up to --max-nodes. One
// run gives the whole curve: a run with a smaller budget follows the same
// path and stops inside it, so the depth a budget B resolves is the deepest
// iteration that completed - and was not capped (an in-check node with more
// legal actions than the evasion cap is not fully resolved) - within B
// nodes; a root proof or an exhausted tree within B resolves every depth.
// The scanner's nodes/s is measured over these runs (thread CPU time), or
// fixed by --nps for a reproducible verdict.
//
// THE MEASURES (M8):
//  (i)   the horizon each budget of --budgets buys: its distribution over
//        roots and over leaves; and the scanner's nodes/s;
//  (ii)  rung 1: leaf budget = threads x nodes/s / evaluation rate, for each
//        --threads (default 1 and the machine's physical cores minus one)
//        and each --eval-rates (default §15.6's design rates, ~9,600/s per GPU
//        at 240 boards and ~4,600/s at 600, cached + window: no GPU here);
//        pass: >= 3 forcing half-turns resolved in >= 50% of leaves. The
//        size-matched rate takes 9,600/s below 420 boards and 4,600/s above;
//  (iii) the root final scan: budget = 50 ms x nodes/s (--root-ms); pass:
//        >= 5 forcing half-turns in >= 50% of roots. --root-wall also runs
//        every root for 50 ms of wall clock;
//  (iv)  quiet-leaf reach at the rung-1 budget: of the leaves of the deepest
//        iteration resolved within it (pooled over the leaf positions), the
//        share that reached a quiet position (no forcing move) or a proven
//        one; a scan that resolved nothing has the position itself as its one
//        leaf. Pass: >= 50%;
//  (v)   stand-pat validity at every quiet position (roots and leaves), as
//        t5d/scanner.hpp's stand_pat_validity defines it - operationalised
//        before measuring: invalid when every legal action examined (all of
//        them when at most --spv-actions, else a sample) loses material to
//        the opponent's best forcing reply within --spv-depth forcing
//        half-turns (--spv-nodes nodes). Bands: <= 5% fine; 5-20% disable
//        stand-pat where obligations bind; > 20% redesign.
// Every verdict is printed with the brief's STOP rules: a failed root test
// STOPS the build for redesign; a figure within +-10% (relative) of its
// threshold STOPS it for the user.
//
// Output: a human summary on standard output (or --summary FILE) and, with
// --json FILE, every condition and figure (--per-position: every position's
// curve too). Deterministic given --seed, except the measured times (and so
// the derived budgets, unless --nps fixes them).
//
// The tool never names a corpus or anything under the oracle harness: the
// records are input on the command line.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "synthetic.hpp"
#include "t5d/check.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/gate.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/rules.hpp"
#include "t5d/scanner.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "tool_support.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <time.h>
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

using t5d::gate::json_string;
using Clock = std::chrono::steady_clock;

// ---- the machine ------------------------------------------------------------------------------

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
  if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                   "ProcessorNameString", RRF_RT_REG_SZ, nullptr, name, &size) == ERROR_SUCCESS) {
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

// Physical cores (0 if the OS does not say).
int physical_cores() {
#if defined(_WIN32)
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
  if (length == 0) return 0;
  std::vector<unsigned char> buffer(length);
  auto* first = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &length)) return 0;
  int cores = 0;
  for (DWORD offset = 0; offset < length;) {
    const auto* info =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
    if (info->Size == 0) break;
    if (info->Relationship == RelationProcessorCore) ++cores;
    offset += info->Size;
  }
  return cores;
#else
  std::FILE* f = std::fopen("/proc/cpuinfo", "r");
  if (f == nullptr) return 0;
  char line[512];
  std::vector<std::pair<int, int>> seen;
  int package = 0;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    const char* colon = std::strchr(line, ':');
    if (colon == nullptr) continue;
    if (std::strncmp(line, "physical id", 11) == 0) package = std::atoi(colon + 1);
    if (std::strncmp(line, "core id", 7) == 0) {
      const std::pair<int, int> id{package, std::atoi(colon + 1)};
      if (std::find(seen.begin(), seen.end(), id) == seen.end()) seen.push_back(id);
    }
  }
  std::fclose(f);
  return static_cast<int>(seen.size());
#endif
}

// This thread's CPU time (Windows: the scheduler's accounting, timer-tick
// resolution - read over long spans only).
double thread_cpu_seconds() {
#if defined(_WIN32)
  FILETIME created, exited, kernel, user;
  GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user);
  const auto u64 = [](FILETIME f) {
    return (static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
  };
  return static_cast<double>(u64(kernel) + u64(user)) * 1e-7;
#else
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
}

// The machine's processor time: idle and total, for its busy fraction.
struct CpuTimes {
  std::uint64_t idle = 0;
  std::uint64_t total = 0;
};
CpuTimes cpu_times() {
  CpuTimes t;
#if defined(_WIN32)
  FILETIME idle, kernel, user;
  if (GetSystemTimes(&idle, &kernel, &user)) {
    const auto u64 = [](FILETIME f) {
      return (static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
    };
    t.idle = u64(idle);
    t.total = u64(kernel) + u64(user);  // kernel time includes idle time
  }
#else
  std::FILE* f = std::fopen("/proc/stat", "r");
  if (f != nullptr) {
    unsigned long long v[8] = {};
    if (std::fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3],
                    &v[4], &v[5], &v[6], &v[7]) == 8) {
      t.idle = v[3] + v[4];
      for (const unsigned long long x : v) t.total += x;
    }
    std::fclose(f);
  }
#endif
  return t;
}
double busy_between(const CpuTimes& a, const CpuTimes& b) {
  if (b.total <= a.total) return 0;
  const double total = static_cast<double>(b.total - a.total);
  return 1.0 - static_cast<double>(b.idle - a.idle) / total;
}

// ---- options ----------------------------------------------------------------------------------

struct Options {
  t5d::RuleOptions rules;
  t5d::ScannerOptions scanner;
  std::vector<std::string> records;
  int every = 1;
  std::size_t max_roots = 0;  // 0: all
  int leaves = 1;
  int leaf_depth = 2;
  std::vector<std::string> synthetic_variants;
  int synthetic_actions_min = 8;
  int synthetic_actions_max = 30;
  int synthetic_games = 0;
  std::vector<std::pair<int, int>> synthetic_classes;
  std::uint64_t seed = 1;
  std::uint64_t max_nodes = 200'000;
  std::vector<std::uint64_t> budgets{100, 300, 1'000, 3'000, 10'000, 30'000, 100'000};
  std::vector<int> threads;  // empty: 1 and physical cores - 1
  std::vector<double> eval_rates{9600, 4600};
  double nps = 0;  // 0: measured
  double root_ms = 50;
  bool root_wall = false;
  bool spv = true;
  t5d::StandPatCheck spv_check;
  int jobs = 1;
  std::string json;
  std::string summary;
  bool per_position = false;
};

[[noreturn]] void usage(int status) {
  std::FILE* out = status == 0 ? stdout : stderr;
  std::fputs(
      "t5d-scan6b " T5D_VERSION_STRING
      " - experiment 6b: the scanner's horizon per node budget, rung 1, the root scan,\n"
      "quiet-leaf reach and stand-pat validity (design spec 15.6, register M8).\n"
      "\n"
      "usage: t5d-scan6b [options] [RECORD ...]\n"
      " positions\n"
      "  --every K                  every K-th turn start of each record (1)\n"
      "  --max-roots N              at most N roots, in input order (all)\n"
      "  --leaves N                 leaves per root (1)\n"
      "  --leaf-depth D             half-turns of random legal play below a root (2)\n"
      "  --synthetic-variants V,..  SYNTHETIC: random legal play from these variants\n"
      "  --synthetic-games G        ... G games each (0)\n"
      "  --synthetic-actions A:B    ... a position after A to B random legal actions (8:30)\n"
      "  --synthetic-classes N:T,.. SYNTHETIC: t5d-bench's random pseudo-legal games\n"
      "  --seed S                   every random choice (1)\n"
      "  --moves-after-shift, --rules ENCODING   the ruleset (rules Q4; RuleOptions::encode)\n"
      " the measurement\n"
      "  --max-nodes N              each scan's node cap: its curve (200000)\n"
      "  --budgets B,...            (i)'s budgets (100,300,1000,3000,10000,30000,100000)\n"
      "  --threads T,...            (ii)'s scanner threads (1 and physical cores - 1)\n"
      "  --eval-rates R,...         (ii)'s evaluation rates per second (9600,4600)\n"
      "  --nps N                    the scanner's nodes/s (default: measured here)\n"
      "  --root-ms MS               (iii)'s root scan time (50)\n"
      "  --root-wall                (iii) also by the wall clock, one scan per root\n"
      "  --no-spv                   skip (v)\n"
      "  --spv-actions N, --spv-depth D, --spv-nodes N   (v)'s check (32, 2, 4000)\n"
      "  --jobs N                   worker threads (1; figures per thread CPU time)\n"
      " the scanner (t5d/scanner.hpp)\n"
      "  --completion quiet|greedy|detector     --branch-threat none|arrival|any\n"
      "  --branch-threat-value V    --no-checks    --promotions-forcing\n"
      "  --stand-pat on|off|bind    --stand-pat-bind B    --evasion-cap N\n"
      "  --evasion-probe N          in check: seeds made into actions before E4 (64; 0: E4 first)\n"
      "  --fallback-max-nodes N     a completion fallback's node cap (2000)\n"
      "  --max-depth D              --tt-entries N         --no-stalemate-test\n"
      "  --values P,N,B,R,Q,K,U,D,S,W,C,Y      piece values (centipawns)\n"
      " output\n"
      "  --json FILE  --summary FILE  --per-position  --help  --version\n",
      out);
  std::exit(status);
}

std::vector<std::string> split(const std::string& text, char sep) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find(sep, start);
    out.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return out;
}

std::uint64_t number(const char* text) {
  const std::optional<std::uint64_t> n = t5d::tools::parse_number(text);
  if (!n) throw std::invalid_argument(std::string("not a number: ") + text);
  return *n;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    if (const int used = t5d::tools::ruleset_option(argc, argv, i, o.rules)) {
      i += used - 1;
    } else if (a == "--help" || a == "-h") {
      usage(0);
    } else if (a == "--version") {
      std::puts("t5d-scan6b " T5D_VERSION_STRING);
      std::exit(0);
    } else if (a == "--every") {
      o.every = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--max-roots") {
      o.max_roots = static_cast<std::size_t>(number(value()));
    } else if (a == "--leaves") {
      o.leaves = static_cast<int>(number(value()));
    } else if (a == "--leaf-depth") {
      o.leaf_depth = static_cast<int>(number(value()));
    } else if (a == "--synthetic-variants") {
      o.synthetic_variants = split(value(), ',');
    } else if (a == "--synthetic-games") {
      o.synthetic_games = static_cast<int>(number(value()));
    } else if (a == "--synthetic-actions") {
      const std::vector<std::string> ab = split(value(), ':');
      if (ab.size() != 2) throw std::invalid_argument("--synthetic-actions A:B");
      o.synthetic_actions_min = static_cast<int>(number(ab[0].c_str()));
      o.synthetic_actions_max = static_cast<int>(number(ab[1].c_str()));
    } else if (a == "--synthetic-classes") {
      for (const std::string& c : split(value(), ',')) {
        const std::vector<std::string> nt = split(c, ':');
        if (nt.size() != 2) throw std::invalid_argument("--synthetic-classes N:T,...");
        o.synthetic_classes.emplace_back(static_cast<int>(number(nt[0].c_str())),
                                         static_cast<int>(number(nt[1].c_str())));
      }
    } else if (a == "--seed") {
      o.seed = number(value());
    } else if (a == "--max-nodes") {
      o.max_nodes = number(value());
    } else if (a == "--budgets") {
      o.budgets.clear();
      for (const std::string& b : split(value(), ',')) o.budgets.push_back(number(b.c_str()));
    } else if (a == "--threads") {
      o.threads.clear();
      for (const std::string& t : split(value(), ',')) {
        o.threads.push_back(static_cast<int>(number(t.c_str())));
      }
    } else if (a == "--eval-rates") {
      o.eval_rates.clear();
      for (const std::string& r : split(value(), ',')) o.eval_rates.push_back(std::atof(r.c_str()));
    } else if (a == "--nps") {
      o.nps = std::atof(value());
    } else if (a == "--root-ms") {
      o.root_ms = std::atof(value());
    } else if (a == "--root-wall") {
      o.root_wall = true;
    } else if (a == "--no-spv") {
      o.spv = false;
    } else if (a == "--spv-actions") {
      o.spv_check.max_actions = static_cast<int>(number(value()));
    } else if (a == "--spv-depth") {
      o.spv_check.reply_depth = static_cast<int>(number(value()));
    } else if (a == "--spv-nodes") {
      o.spv_check.reply_nodes = number(value());
    } else if (a == "--jobs") {
      o.jobs = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--completion") {
      const std::string v = value();
      if (v == "quiet") {
        o.scanner.completion = t5d::Completion::Quiet;
      } else if (v == "greedy") {
        o.scanner.completion = t5d::Completion::Greedy;
      } else if (v == "detector") {
        o.scanner.completion = t5d::Completion::Detector;
      } else {
        throw std::invalid_argument("--completion quiet|greedy|detector");
      }
    } else if (a == "--branch-threat") {
      const std::string v = value();
      if (v == "none") {
        o.scanner.branch_threat = t5d::BranchThreat::None;
      } else if (v == "arrival") {
        o.scanner.branch_threat = t5d::BranchThreat::ArrivalCapture;
      } else if (v == "any") {
        o.scanner.branch_threat = t5d::BranchThreat::AnyCapture;
      } else {
        throw std::invalid_argument("--branch-threat none|arrival|any");
      }
    } else if (a == "--branch-threat-value") {
      o.scanner.branch_threat_value = static_cast<int>(number(value()));
    } else if (a == "--no-checks") {
      o.scanner.checks = false;
    } else if (a == "--promotions-forcing") {
      o.scanner.promotions_forcing = true;
    } else if (a == "--stand-pat") {
      const std::string v = value();
      if (v == "on") {
        o.scanner.stand_pat = t5d::StandPat::On;
      } else if (v == "off") {
        o.scanner.stand_pat = t5d::StandPat::Off;
      } else if (v == "bind") {
        o.scanner.stand_pat = t5d::StandPat::OffWhereObligationsBind;
      } else {
        throw std::invalid_argument("--stand-pat on|off|bind");
      }
    } else if (a == "--stand-pat-bind") {
      o.scanner.stand_pat_bind_boards = static_cast<int>(number(value()));
    } else if (a == "--fallback-max-nodes") {
      o.scanner.fallback_max_nodes = number(value());
    } else if (a == "--evasion-cap") {
      o.scanner.evasion_cap = static_cast<std::size_t>(number(value()));
    } else if (a == "--evasion-probe") {
      o.scanner.evasion_probe = static_cast<std::size_t>(number(value()));
    } else if (a == "--max-depth") {
      o.scanner.max_depth = static_cast<int>(number(value()));
    } else if (a == "--tt-entries") {
      o.scanner.tt_entries = static_cast<std::size_t>(number(value()));
    } else if (a == "--no-stalemate-test") {
      o.scanner.quiet_stalemate_test = false;
    } else if (a == "--values") {
      const std::vector<std::string> v = split(value(), ',');
      if (v.size() != 12) throw std::invalid_argument("--values needs 12 numbers (P N B R Q K U D S W C Y)");
      for (std::size_t k = 0; k < 12; ++k) {
        o.scanner.values.value[k] = static_cast<int>(number(v[k].c_str()));
      }
    } else if (a == "--json") {
      o.json = value();
    } else if (a == "--summary") {
      o.summary = value();
    } else if (a == "--per-position") {
      o.per_position = true;
    } else if (!a.empty() && a[0] == '-' && a != "-") {
      throw std::invalid_argument("unknown option " + a);
    } else {
      o.records.push_back(a);
    }
  }
  o.scanner.classify_leaves = true;  // (iv); uncharged: the search is the same
  return o;
}

// ---- positions --------------------------------------------------------------------------------

enum class Role : std::uint8_t { Root = 0, Leaf };

struct Position {
  std::string source;  // the record's path, or "synthetic ..."
  bool synthetic = false;
  int index = 0;       // the turn start's number in its record, or the game's
  Role role = Role::Root;
  int parent = -1;     // a leaf's root (an index into the positions)
  std::uint64_t seed = 0;
  std::optional<t5d::GameState> state;
};

// A random legal action: a random pseudo-legal seed (never a royal capture,
// 4.13), completed by the scanner's rule; E4's witness if no seed completes.
std::optional<std::vector<t5d::Move>> random_legal_action(const t5d::GameState& s,
                                                          t5d::synthetic::Rng& rng,
                                                          t5d::Scanner& scanner) {
  t5d::MoveList moves;
  t5d::generate_moves(s, moves);
  moves.erase(std::remove_if(moves.begin(), moves.end(),
                             [](const t5d::Move& m) { return m.captured.royal(); }),
              moves.end());
  for (int attempt = 0; attempt < 24 && !moves.empty(); ++attempt) {
    const std::size_t i = rng.below(moves.size());
    if (const std::optional<t5d::BuiltAction> a = scanner.complete(s, moves[i])) return a->moves;
    moves.erase(moves.begin() + static_cast<std::ptrdiff_t>(i));
  }
  t5d::SearchBudget b = t5d::SearchBudget::nodes(200'000);
  b.max_bytes = std::size_t{256} << 20;
  const t5d::MateResult r = t5d::detect_mate(s, t5d::Detector::Hypercuboid, b);
  if (r.verdict == t5d::Outcome::Ongoing) return r.witness;
  return std::nullopt;
}

// Whether the side to move could capture an enemy royal: the position's last
// action broke C4 (a pseudo-legal replay, or random pseudo-legal play).
bool broken(const t5d::GameState& s) {
  return t5d::royal_capture_exists(s.multiverse(), s.rules(), s.to_move(), s.inactive_timelines());
}

// ---- results ----------------------------------------------------------------------------------

struct Result {
  bool scanned = false;
  std::string skipped;  // why not
  int boards = 0;
  int timelines = 0;
  int active = 0;
  int mandatory = 0;
  bool in_check = false;
  bool quiet = false;
  t5d::ScanResult scan;
  double cpu_seconds = 0;
  std::optional<t5d::StandPatVerdict> spv;
  double spv_seconds = 0;
  std::optional<t5d::ScanResult> wall;  // --root-wall
};

constexpr int kAll = 1000;  // "every depth": a proof or an exhausted tree

// The deepest depth a budget resolves: kAll for a proof or an exhausted
// tree reached within it.
int horizon(const t5d::ScanResult& r, std::uint64_t budget) {
  int depth = 0;
  for (const t5d::IterationStats& it : r.iterations) {
    if (!it.completed || it.nodes > budget) break;
    if (it.proof.proven() || it.exhausted) return kAll;
    if (!it.capped) depth = it.depth;
  }
  return depth;
}

// The leaves of the deepest iteration a budget resolves (nullptr: none).
const t5d::IterationStats* resolved_iteration(const t5d::ScanResult& r, std::uint64_t budget) {
  const t5d::IterationStats* best = nullptr;
  for (const t5d::IterationStats& it : r.iterations) {
    if (!it.completed || it.nodes > budget) break;
    best = &it;
  }
  return best;
}

// Whether the budget's whole curve was measured (a scan stopped by
// --max-nodes before `budget` says nothing past its cap).
bool covered(const t5d::ScanResult& r, std::uint64_t budget, std::uint64_t cap) {
  return budget <= cap || !r.stats.stopped_by_nodes;
}

std::string fmt(double v, int decimals = 1) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
  return buf;
}

std::string pct(std::size_t a, std::size_t b) {
  return b == 0 ? "-" : fmt(100.0 * static_cast<double>(a) / static_cast<double>(b)) + "%";
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

// The verdict of a share against a threshold of 50%, with the brief's +-10%
// STOP band (relative: 45-55%).
// The brief's band on a depth threshold, read at the median (+-10%: 2.7-3.3
// and 4.5-5.5 forcing half-turns).
std::string depth_band(double median_depth, double threshold) {
  if (median_depth >= kAll) return "";
  if (median_depth >= 0.9 * threshold && median_depth <= 1.1 * threshold) {
    return " [median within +-10% of the depth threshold: STOP for the user]";
  }
  return "";
}

std::string share_verdict(double share, bool stop_on_fail, const char* on_fail) {
  std::string v = share >= 0.5 ? "PASS" : "FAIL";
  if (share >= 0.45 && share <= 0.55) v += " - within +-10% of the threshold: STOP for the user";
  else if (share < 0.5) v += std::string(" - ") + (stop_on_fail ? "STOP: " : "") + on_fail;
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = parse(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-scan6b: %s\n", e.what());
    usage(2);
  }
  const int cores = physical_cores();
  if (o.threads.empty()) {
    o.threads.push_back(1);
    if (cores > 2) o.threads.push_back(cores - 1);
  }
  const std::string cpu = cpu_model();
  const unsigned logical = std::thread::hardware_concurrency();

  // -- the roots ------------------------------------------------------------------------------
  std::vector<Position> positions;
  std::vector<std::string> record_notes;
  std::unordered_set<std::uint64_t> seen;
  std::size_t duplicates = 0;
  const auto add_root = [&](Position p) {
    const std::uint64_t key = p.state->zobrist();
    if (!seen.insert(key).second) {
      ++duplicates;
      return;
    }
    if (o.max_roots != 0 && positions.size() >= o.max_roots) return;
    positions.push_back(std::move(p));
  };
  for (const std::string& path : o.records) {
    try {
      const std::string text = t5d::tools::read_file(path);
      t5d::PgnRecord record = t5d::read_pgn(text);
      const bool synthetic = t5d::tools::is_synthetic(record);
      t5d::PgnReplay replay(std::move(record), t5d::PgnOptions{}, o.rules);
      t5d::tools::parse_mode(replay);
      std::string end = "ok";
      for (;;) {
        const int k = static_cast<int>(replay.actions_played());
        if (k % o.every == 0) {
          Position p;
          p.source = path;
          p.synthetic = synthetic;
          p.index = k;
          p.state = replay.state();
          add_root(std::move(p));
        }
        if (replay.finished()) break;
        try {
          replay.play_next();
        } catch (const t5d::RecordError& e) {
          end = std::string(t5d::fault_name(e.fault())) + " at action " +
                std::to_string(replay.actions_played() + 1);
          break;
        }
      }
      record_notes.push_back(path + ": " + std::to_string(replay.actions_played()) +
                             " actions read, " + end);
    } catch (const std::exception& e) {
      record_notes.push_back(path + ": not read (" + e.what() + ")");
    }
  }
  // SYNTHETIC roots.
  {
    t5d::Scanner helper(o.scanner);
    int game = 0;
    for (const std::string& name : o.synthetic_variants) {
      const t5d::Variant* v = t5d::find_variant(name);
      if (v == nullptr) {
        record_notes.push_back("synthetic " + name + ": no such variant");
        continue;
      }
      for (int g = 0; g < o.synthetic_games; ++g, ++game) {
        t5d::synthetic::Rng rng(o.seed * std::uint64_t{1000003} + static_cast<std::uint64_t>(game) * std::uint64_t{7919} + 17);
        t5d::GameState s = t5d::variant_start(*v, o.rules);
        const int span = std::max(0, o.synthetic_actions_max - o.synthetic_actions_min);
        const int actions = o.synthetic_actions_min +
                            static_cast<int>(rng.below(static_cast<std::size_t>(span) + 1));
        bool ok = true;
        for (int a = 0; a < actions && ok; ++a) {
          const std::optional<std::vector<t5d::Move>> act = random_legal_action(s, rng, helper);
          if (!act) {
            ok = false;
            break;
          }
          for (const t5d::Move& m : *act) s.apply(m);
          s.submit();
        }
        Position p;
        p.source = "synthetic " + name;
        p.synthetic = true;
        p.index = g;
        p.state = s;
        add_root(std::move(p));
      }
    }
    for (const auto& [n, t] : o.synthetic_classes) {
      for (int g = 0; g < std::max(1, o.synthetic_games); ++g) {
        t5d::synthetic::Spec spec;
        spec.rules = o.rules;
        spec.seed = o.seed * std::uint64_t{7777} + static_cast<std::uint64_t>(n) * std::uint64_t{131} +
                    static_cast<std::uint64_t>(t) * std::uint64_t{17} + static_cast<std::uint64_t>(g);
        spec.target_active = t;
        spec.target_boards = n;
        const t5d::synthetic::Game sg = t5d::synthetic::play_until_reached(spec, 8);
        Position p;
        p.source = "synthetic N" + std::to_string(n) + " T" + std::to_string(t);
        p.synthetic = true;
        p.index = g;
        p.state = sg.state;
        add_root(std::move(p));
      }
    }
  }
  const std::size_t roots = positions.size();
  // Leaves: random legal play below each root.
  {
    t5d::Scanner helper(o.scanner);
    for (std::size_t r = 0; r < roots; ++r) {
      for (int l = 0; l < o.leaves; ++l) {
        t5d::synthetic::Rng rng(o.seed * std::uint64_t{104729} + static_cast<std::uint64_t>(r) * std::uint64_t{1009} +
                                static_cast<std::uint64_t>(l) + 3);
        // From a root whose last action broke C4 too: the first legal action
        // (never a royal capture) restores it - its royals safe, C4.
        t5d::GameState s = *positions[r].state;
        bool ok = true;
        for (int d = 0; d < o.leaf_depth && ok; ++d) {
          const std::optional<std::vector<t5d::Move>> act = random_legal_action(s, rng, helper);
          if (!act) {
            ok = false;
            break;
          }
          for (const t5d::Move& m : *act) s.apply(m);
          s.submit();
        }
        if (!ok) continue;
        Position p;
        p.source = positions[r].source;
        p.synthetic = positions[r].synthetic;
        p.index = positions[r].index;
        p.role = Role::Leaf;
        p.parent = static_cast<int>(r);
        p.state = s;
        positions.push_back(std::move(p));
      }
    }
  }

  // -- the scans -------------------------------------------------------------------------------
  std::vector<Result> results(positions.size());
  std::atomic<std::size_t> next{0};
  std::mutex progress;
  std::size_t done = 0;
  const CpuTimes load0 = cpu_times();
  const Clock::time_point wall0 = Clock::now();
  std::vector<double> scan_cpu(static_cast<std::size_t>(o.jobs), 0.0);
  const auto work = [&](int worker) {
    t5d::Scanner scanner(o.scanner);
    for (;;) {
      const std::size_t i = next.fetch_add(1);
      if (i >= positions.size()) break;
      const t5d::GameState& s = *positions[i].state;
      Result& r = results[i];
      r.boards = static_cast<int>(s.multiverse().board_count());
      r.timelines = s.multiverse().timeline_count();
      int active = 0;
      for (const t5d::TimelineRef& t : s.multiverse().timelines()) {
        if (s.active(t->position())) ++active;
      }
      r.active = active;
      r.mandatory = s.mandatory_count();
      if (broken(s)) {
        r.skipped = "a royal capturable by the side to move (the last action broke C4)";
      } else {
        t5d::SearchBudget b = t5d::SearchBudget::nodes(2'000'000);
        b.max_bytes = std::size_t{256} << 20;
        const t5d::MateResult m = t5d::detect_mate(s, t5d::Detector::Hypercuboid, b);
        if (m.verdict == t5d::Outcome::Checkmate || m.verdict == t5d::Outcome::Stalemate) {
          r.skipped = std::string("terminal (") +
                      (m.verdict == t5d::Outcome::Checkmate ? "checkmate" : "stalemate") + ")";
        } else {
          r.in_check = m.in_check;
          r.quiet = !m.in_check && !scanner.has_forcing_move(s);
          scanner.clear();
          const double c0 = thread_cpu_seconds();
          r.scan = scanner.scan(s, t5d::ScanBudget::nodes(o.max_nodes));
          r.cpu_seconds = thread_cpu_seconds() - c0;
          scan_cpu[static_cast<std::size_t>(worker)] += r.cpu_seconds;
          r.scanned = true;
          if (o.root_wall && positions[i].role == Role::Root) {
            t5d::ScanBudget wb;
            wb.max_time = std::chrono::nanoseconds(static_cast<long long>(o.root_ms * 1e6));
            t5d::ScannerOptions plain = o.scanner;
            plain.classify_leaves = false;  // the search alone, by the clock
            t5d::Scanner fresh(plain);
            r.wall = fresh.scan(s, wb);
          }
          if (o.spv && r.quiet) {
            const double s0 = thread_cpu_seconds();
            r.spv = t5d::stand_pat_validity(s, o.scanner, o.spv_check);
            r.spv_seconds = thread_cpu_seconds() - s0;
          }
        }
      }
      std::lock_guard<std::mutex> lock(progress);
      ++done;
      if (done % 50 == 0 || done == positions.size()) {
        std::fprintf(stderr, "t5d-scan6b: %zu of %zu positions\n", done, positions.size());
      }
    }
  };
  {
    std::vector<std::thread> pool;
    for (int w = 1; w < o.jobs; ++w) pool.emplace_back(work, w);
    work(0);
    for (std::thread& t : pool) t.join();
  }
  const double wall_seconds = std::chrono::duration<double>(Clock::now() - wall0).count();
  const double busy = busy_between(load0, cpu_times());

  // -- the figures ------------------------------------------------------------------------------
  // The scans' thread CPU time, less the leaf classification's (measurement
  // only, timed by the scanner itself): the search's own speed.
  std::uint64_t total_nodes = 0;
  double total_cpu = 0;
  double measurement = 0;
  for (const Result& r : results) {
    if (!r.scanned) continue;
    total_nodes += r.scan.stats.nodes;
    total_cpu += r.cpu_seconds;
    measurement += r.scan.stats.measurement_seconds;
  }
  const double search_cpu = std::max(1e-9, total_cpu - measurement);
  const double measured_nps = total_cpu > 0 ? static_cast<double>(total_nodes) / search_cpu : 0;
  const double nps = o.nps > 0 ? o.nps : measured_nps;

  // Which positions count for the verdicts: scanned, not synthetic.
  const auto counted = [&](std::size_t i, Role role) {
    return results[i].scanned && positions[i].role == role && !positions[i].synthetic;
  };
  const auto synthetic_counted = [&](std::size_t i, Role role) {
    return results[i].scanned && positions[i].role == role && positions[i].synthetic;
  };

  std::string out;
  const auto line = [&](const std::string& s) { out += s + "\n"; };
  line("t5d-scan6b " T5D_VERSION_STRING " - experiment 6b (design spec 15.6; register M8)");
  line("");
  line("CONDITIONS");
  line("  machine: " + cpu + ", " + std::to_string(cores) + " physical cores, " +
       std::to_string(logical) + " logical processors; busy " + fmt(100 * busy) +
       "% over the run (all processes)");
  line("  build: " + compiler_text() + ", " + T5D_BUILD_TYPE + ", flags \"" + T5D_CXX_FLAGS + "\"");
  line("  rules: " + t5d::describe(o.rules));
  line("  scanner: completion " +
       std::string(o.scanner.completion == t5d::Completion::Quiet    ? "quiet"
                   : o.scanner.completion == t5d::Completion::Greedy ? "greedy"
                                                                     : "detector") +
       ", checks " + (o.scanner.checks ? "on" : "off") + ", branch threat " +
       (o.scanner.branch_threat == t5d::BranchThreat::None             ? "none"
        : o.scanner.branch_threat == t5d::BranchThreat::ArrivalCapture ? "arrival capture"
                                                                       : "any capture") +
       " >= " + std::to_string(o.scanner.branch_threat_value) + ", promotions " +
       (o.scanner.promotions_forcing ? "forcing" : "not forcing") + ", stand-pat " +
       (o.scanner.stand_pat == t5d::StandPat::On    ? "on"
        : o.scanner.stand_pat == t5d::StandPat::Off ? "off"
                                                     : "off where obligations bind") +
       ", evasion cap " + std::to_string(o.scanner.evasion_cap) + ", evasion probe " +
       std::to_string(o.scanner.evasion_probe) + " seeds, completion fallback cap " +
       std::to_string(o.scanner.fallback_max_nodes) + " nodes, max depth " +
       std::to_string(o.scanner.max_depth) + ", TT " + std::to_string(o.scanner.tt_entries) +
       " entries");
  {
    std::string values = "  piece values (P N B R Q K U D S W C Y):";
    for (const int v : o.scanner.values.value) values += " " + std::to_string(v);
    line(values);
  }
  line("  each scan: deterministic (node budget, no clock), cap " + std::to_string(o.max_nodes) +
       " nodes, TT and histories cleared per position; " + std::to_string(o.jobs) + " job(s); seed " +
       std::to_string(o.seed));
  line("  positions: every " + std::to_string(o.every) + " turn start(s) of " +
       std::to_string(o.records.size()) + " record(s); leaves " + std::to_string(o.leaves) +
       " per root at " + std::to_string(o.leaf_depth) + " half-turns of random legal play; " +
       std::to_string(duplicates) + " duplicate root(s) dropped");
  for (const std::string& n : record_notes) line("    " + n);

  // Position census.
  std::map<std::string, std::size_t> skipped;
  std::size_t real_roots = 0;
  std::size_t real_leaves = 0;
  std::size_t syn_roots = 0;
  std::size_t syn_leaves = 0;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (!results[i].scanned) {
      skipped[(positions[i].role == Role::Root ? "root " : "leaf ") + results[i].skipped]++;
      continue;
    }
    if (counted(i, Role::Root)) ++real_roots;
    if (counted(i, Role::Leaf)) ++real_leaves;
    if (synthetic_counted(i, Role::Root)) ++syn_roots;
    if (synthetic_counted(i, Role::Leaf)) ++syn_leaves;
  }
  line("  scanned: " + std::to_string(real_roots) + " roots and " + std::to_string(real_leaves) +
       " leaves from records; SYNTHETIC " + std::to_string(syn_roots) + " roots, " +
       std::to_string(syn_leaves) + " leaves (never in a verdict)");
  for (const auto& [why, n] : skipped) line("    not scanned: " + std::to_string(n) + " " + why);
  {
    std::vector<double> boards;
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (counted(i, Role::Root) || counted(i, Role::Leaf)) boards.push_back(results[i].boards);
    }
    std::sort(boards.begin(), boards.end());
    if (!boards.empty()) {
      line("    boards per position (records): median " + fmt(median(boards), 0) + ", max " +
           fmt(boards.back(), 0));
    }
  }
  line("  scanner speed: " + fmt(measured_nps, 0) + " nodes/s measured (" +
       std::to_string(total_nodes) + " nodes in " + fmt(search_cpu, 2) +
       " s of thread CPU time - the scans' " + fmt(total_cpu, 2) + " s less " + fmt(measurement, 2) +
       " s of leaf classification, a measurement); wall " + fmt(wall_seconds, 1) + " s" +
       (o.nps > 0 ? "; budgets derived from --nps " + fmt(o.nps, 0) : ""));
  {
    t5d::ScanStats sum;
    for (const Result& r : results) {
      if (!r.scanned) continue;
      const t5d::ScanStats& x = r.scan.stats;
      sum.positions += x.positions;
      sum.applies += x.applies;
      sum.scans += x.scans;
      sum.detector_nodes += x.detector_nodes;
      sum.evasion_nodes += x.evasion_nodes;
      sum.fallback_nodes += x.fallback_nodes;
      sum.horizon_mate_nodes += x.horizon_mate_nodes;
      sum.stalemate_test_nodes += x.stalemate_test_nodes;
      sum.capped_nodes += x.capped_nodes;
      sum.completion_fallbacks += x.completion_fallbacks;
    }
    const auto share = [&](std::uint64_t part) {
      return total_nodes == 0 ? std::string("-")
                              : fmt(100.0 * static_cast<double>(part) /
                                        static_cast<double>(total_nodes)) + "%";
    };
    line("  where the nodes went: E4 searches " + share(sum.detector_nodes) + " (legal actions in check " +
         share(sum.evasion_nodes) + ", completion fallbacks " + share(sum.fallback_nodes) +
         ", horizon mate tests " + share(sum.horizon_mate_nodes) + ", stalemate tests " +
         share(sum.stalemate_test_nodes) + "), moves applied " + share(sum.applies) +
         ", royal-capture scans " + share(sum.scans) + "; " + std::to_string(sum.positions) +
         " tree positions (" +
         fmt(sum.positions == 0 ? 0.0
                                : static_cast<double>(total_nodes) / static_cast<double>(sum.positions),
             1) +
         " nodes each), " + std::to_string(sum.capped_nodes) + " capped in-check nodes, " +
         std::to_string(sum.completion_fallbacks) + " completion fallbacks");
  }
  // The profile of the exhaustive paths, by board count (M12's optimisation
  // pass reads it).
  {
    const char* names[4] = {"< 50 boards", "50-99", "100-199", "200+"};
    struct Sum {
      std::uint64_t nodes = 0, evasion = 0, fallback = 0, horizon = 0, visits = 0, fail_highs = 0,
                    lists = 0, listed = 0, tests = 0, mates = 0, capped = 0, on_mates = 0,
                    probe_seeds = 0, probe_actions = 0, probe_cutoffs = 0, probe_hits = 0,
                    probe_nodes = 0;
      double seconds = 0, evasion_s = 0, horizon_s = 0, fallback_s = 0, probe_s = 0;
    };
    Sum by[4];
    for (const Result& r : results) {
      if (!r.scanned) continue;
      const int b = r.boards < 50 ? 0 : r.boards < 100 ? 1 : r.boards < 200 ? 2 : 3;
      const t5d::ScanStats& x = r.scan.stats;
      Sum& s = by[b];
      s.nodes += x.nodes;
      s.evasion += x.evasion_nodes;
      s.fallback += x.fallback_nodes;
      s.horizon += x.horizon_mate_nodes;
      s.visits += x.evasion_visits;
      s.fail_highs += x.evasion_fail_highs;
      s.lists += x.evasion_lists;
      s.listed += x.evasion_listed;
      s.tests += x.horizon_mate_tests;
      s.mates += x.horizon_mates;
      s.on_mates += x.horizon_mate_nodes_on_mates;
      s.probe_seeds += x.probe_seeds;
      s.probe_actions += x.probe_actions;
      s.probe_cutoffs += x.probe_cutoffs;
      s.probe_hits += x.horizon_probe_hits;
      s.probe_nodes += x.probe_nodes;
      s.probe_s += x.probe_seconds;
      s.capped += x.capped_nodes;
      s.seconds += x.seconds - x.measurement_seconds;
      s.evasion_s += x.evasion_seconds;
      s.horizon_s += x.horizon_mate_seconds;
      s.fallback_s += x.fallback_seconds;
    }
    line("  profile by board count (nodes; wall time inside the scans):");
    for (int b = 0; b < 4; ++b) {
      const Sum& s = by[b];
      if (s.nodes == 0) continue;
      const auto share = [&](std::uint64_t part) {
        return fmt(100.0 * static_cast<double>(part) / static_cast<double>(s.nodes)) + "%";
      };
      const auto tshare = [&](double part) {
        return s.seconds <= 0 ? std::string("-") : fmt(100.0 * part / s.seconds) + "%";
      };
      line(std::string("    ") + names[b] + ": " + std::to_string(s.nodes) + " nodes, " +
           fmt(s.seconds, 1) + " s; in-check listing " + share(s.evasion) + " of nodes, " +
           tshare(s.evasion_s) + " of time (" + std::to_string(s.lists) + " listings, " +
           fmt(s.lists == 0 ? 0.0 : static_cast<double>(s.listed) / static_cast<double>(s.lists), 1) +
           " actions each; " + std::to_string(s.visits) + " in-check nodes, " +
           std::to_string(s.fail_highs) + " failed high, " + std::to_string(s.capped) +
           " capped); horizon mate tests " + share(s.horizon) + ", " + tshare(s.horizon_s) + " (" +
           std::to_string(s.tests) + " tests, " + std::to_string(s.probe_hits) +
           " answered by the probe, " + std::to_string(s.mates) + " mates taking " +
           share(s.on_mates) + "); fallbacks " + share(s.fallback) + ", " + tshare(s.fallback_s) +
           "; probe " + share(s.probe_nodes) + ", " + tshare(s.probe_s) + " (" +
           std::to_string(s.probe_seeds) + " seeds, " + std::to_string(s.probe_actions) +
           " actions searched, " + std::to_string(s.probe_cutoffs) + " cutoffs)");
    }
  }
  line("");

  // (i) horizon per budget.
  const auto horizon_table = [&](Role role, bool synthetic, const char* title) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (synthetic ? synthetic_counted(i, role) : counted(i, role)) ++n;
    }
    if (n == 0) return;
    line(std::string("(i) HORIZON PER NODE BUDGET - ") + title + " (" + std::to_string(n) +
         "): share resolving at least d forcing half-turns");
    std::string head = "  budget     |";
    for (int d = 1; d <= 6; ++d) head += "  >=" + std::to_string(d) + "  ";
    head += "| median | proven | uncovered";
    line(head);
    for (const std::uint64_t b : o.budgets) {
      std::vector<std::size_t> at_least(7, 0);
      std::vector<double> hs;
      std::size_t proven = 0;
      std::size_t uncovered = 0;
      for (std::size_t i = 0; i < positions.size(); ++i) {
        if (!(synthetic ? synthetic_counted(i, role) : counted(i, role))) continue;
        const t5d::ScanResult& r = results[i].scan;
        if (!covered(r, b, o.max_nodes)) ++uncovered;
        const int h = horizon(r, b);
        hs.push_back(h);
        for (int d = 1; d <= 6; ++d) {
          if (h >= d) ++at_least[static_cast<std::size_t>(d)];
        }
        const t5d::IterationStats* it = resolved_iteration(r, b);
        if (it != nullptr && it->proof.proven()) ++proven;
      }
      char buf[32];
      std::snprintf(buf, sizeof buf, "  %-10llu |", static_cast<unsigned long long>(b));
      std::string row = buf;
      for (int d = 1; d <= 6; ++d) {
        std::snprintf(buf, sizeof buf, " %5s ", pct(at_least[static_cast<std::size_t>(d)], n).c_str());
        row += buf;
      }
      const double med = median(hs);
      row += "| " + (med >= kAll ? std::string("all") : fmt(med, 1)) + "    | " +
             std::to_string(proven) + "      | " + std::to_string(uncovered);
      line(row);
    }
  };
  horizon_table(Role::Root, false, "roots from records");
  horizon_table(Role::Leaf, false, "leaves from records");
  horizon_table(Role::Root, true, "SYNTHETIC roots");
  horizon_table(Role::Leaf, true, "SYNTHETIC leaves");
  line("");

  // (ii) rung 1 and (iv) quiet reach.
  struct Rung {
    int threads = 0;
    double rate = 0;  // 0: size-matched
    std::uint64_t budget = 0;
    std::size_t leaves = 0;
    std::size_t pass = 0;
    std::uint64_t leaf_total = 0;
    std::uint64_t leaf_quiet = 0;
    std::vector<double> scan_quiet;  // per scan
    std::vector<double> horizons;    // per leaf
  };
  std::vector<Rung> rungs;
  const auto rate_for = [&](double rate, int boards) {
    return rate > 0 ? rate : (boards < 420 ? 9600.0 : 4600.0);
  };
  std::vector<double> rates = o.eval_rates;
  rates.push_back(0);
  for (const int t : o.threads) {
    for (const double rate : rates) {
      Rung g;
      g.threads = t;
      g.rate = rate;
      for (std::size_t i = 0; i < positions.size(); ++i) {
        if (!counted(i, Role::Leaf)) continue;
        const Result& r = results[i];
        const auto budget = static_cast<std::uint64_t>(
            static_cast<double>(t) * nps / rate_for(rate, r.boards));
        g.budget = budget;  // for a fixed rate, the same for every leaf
        ++g.leaves;
        const int h = horizon(r.scan, budget);
        g.horizons.push_back(h);
        if (h >= 3) ++g.pass;
        const t5d::IterationStats* it = resolved_iteration(r.scan, budget);
        if (it != nullptr) {
          g.leaf_total += it->leaves.total();
          g.leaf_quiet += it->leaves.reached_quiet();
          g.scan_quiet.push_back(it->leaves.total() == 0
                                     ? 1.0
                                     : static_cast<double>(it->leaves.reached_quiet()) /
                                           static_cast<double>(it->leaves.total()));
        } else {
          g.leaf_total += 1;
          g.leaf_quiet += r.quiet ? 1 : 0;
          g.scan_quiet.push_back(r.quiet ? 1.0 : 0.0);
        }
      }
      rungs.push_back(g);
    }
  }
  line("(ii) RUNG 1 - leaf budget = threads x " + fmt(nps, 0) +
       " nodes/s / evaluation rate; pass: >= 3 forcing half-turns resolved in >= 50% of leaves");
  line("     (iv) QUIET-LEAF REACH at the same budget: pooled scan leaves that reached a quiet or");
  line("     proven position; pass >= 50% (per-scan median beside it)");
  for (const Rung& g : rungs) {
    const std::string rate = g.rate > 0 ? fmt(g.rate, 0) + "/s" : std::string("size-matched");
    const double share = g.leaves == 0 ? 0 : static_cast<double>(g.pass) / static_cast<double>(g.leaves);
    const double reach = g.leaf_total == 0 ? 0
                                           : static_cast<double>(g.leaf_quiet) /
                                                 static_cast<double>(g.leaf_total);
    line("  " + std::to_string(g.threads) + " thread(s), " + rate +
         (g.rate > 0 ? ", " + std::to_string(g.budget) + " nodes" : std::string()) + ": (ii) " +
         pct(g.pass, g.leaves) + " of " + std::to_string(g.leaves) + " leaves (median horizon " +
         (median(g.horizons) >= kAll ? std::string("all") : fmt(median(g.horizons), 1)) + ")" +
         depth_band(median(g.horizons), 3) + " -> " +
         share_verdict(share, false, "drop rung 1 (the higher rungs keep their budgets)") +
         "; (iv) " + pct(g.leaf_quiet, g.leaf_total) + " of " + std::to_string(g.leaf_total) +
         " leaves (median per scan " + fmt(100 * median(g.scan_quiet)) + "%) -> " +
         share_verdict(reach, false, "NNUE v2 and quiet-only training deferred"));
  }
  line("");

  // What a pass rests on, and the shares by board count.
  const auto breakdown = [&](Role role, bool synthetic, const auto& budget_of, int depth) {
    std::size_t n = 0, pass = 0, deep = 0, exhausted = 0, proven = 0;
    const char* names[4] = {"< 50 boards", "50-99", "100-199", "200+"};
    std::size_t bucket_n[4] = {0, 0, 0, 0};
    std::size_t bucket_pass[4] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (!(synthetic ? synthetic_counted(i, role) : counted(i, role))) continue;
      const Result& r = results[i];
      const std::uint64_t budget = budget_of(r);
      const int b = r.boards < 50 ? 0 : r.boards < 100 ? 1 : r.boards < 200 ? 2 : 3;
      ++n;
      ++bucket_n[b];
      if (horizon(r.scan, budget) < depth) continue;
      ++pass;
      ++bucket_pass[b];
      bool settled = false;
      for (const t5d::IterationStats& it : r.scan.iterations) {
        if (!it.completed || it.nodes > budget) break;
        if (it.proof.proven()) {
          ++proven;
          settled = true;
          break;
        }
        if (it.exhausted) {
          ++exhausted;
          settled = true;
          break;
        }
      }
      if (!settled) ++deep;
    }
    std::string s = "    " + pct(pass, n) + " of " + std::to_string(n) + " pass, resting on " +
                    std::to_string(deep) + " iterations to depth, " + std::to_string(exhausted) +
                    " exhausted trees, " + std::to_string(proven) + " proofs; by size:";
    for (int b = 0; b < 4; ++b) {
      if (bucket_n[b] == 0) continue;
      s += std::string(" ") + names[b] + " " + pct(bucket_pass[b], bucket_n[b]) + " of " +
           std::to_string(bucket_n[b]) + ";";
    }
    return s;
  };
  for (const Rung& g : rungs) {
    if (g.threads != o.threads.back() || g.rate != 0) continue;  // the headline: size-matched
    line("  (ii) at " + std::to_string(g.threads) + " thread(s), size-matched rate:");
    line(breakdown(Role::Leaf, false,
                   [&](const Result& r) {
                     return static_cast<std::uint64_t>(static_cast<double>(g.threads) * nps /
                                                       rate_for(0, r.boards));
                   },
                   3));
    if (syn_leaves > 0) {
      line("    SYNTHETIC leaves (no verdict):");
      line(breakdown(Role::Leaf, true,
                     [&](const Result& r) {
                       return static_cast<std::uint64_t>(static_cast<double>(g.threads) * nps /
                                                         rate_for(0, r.boards));
                     },
                     3));
    }
  }
  line("");

  // (iii) the root scan.
  {
    const auto budget = static_cast<std::uint64_t>(o.root_ms / 1000.0 * nps);
    std::size_t n = 0;
    std::size_t pass = 0;
    std::size_t uncovered = 0;
    std::size_t wall_pass = 0;
    std::size_t wall_n = 0;
    std::size_t wall_bucket_n[4] = {0, 0, 0, 0};
    std::size_t wall_bucket_pass[4] = {0, 0, 0, 0};
    std::vector<double> hs;
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (!counted(i, Role::Root)) continue;
      ++n;
      const int h = horizon(results[i].scan, budget);
      hs.push_back(h);
      if (h >= 5) ++pass;
      if (!covered(results[i].scan, budget, o.max_nodes)) ++uncovered;
      if (results[i].wall) {
        ++wall_n;
        const int b = results[i].boards < 50 ? 0 : results[i].boards < 100 ? 1 : results[i].boards < 200 ? 2 : 3;
        ++wall_bucket_n[b];
        const t5d::ScanResult& w = *results[i].wall;
        if (w.proof.proven() || w.exhausted || w.resolved_depth >= 5) {
          ++wall_pass;
          ++wall_bucket_pass[b];
        }
      }
    }
    const double share = n == 0 ? 0 : static_cast<double>(pass) / static_cast<double>(n);
    line("(iii) ROOT FINAL SCAN - " + fmt(o.root_ms, 0) + " ms = " + std::to_string(budget) +
         " nodes at " + fmt(nps, 0) + " nodes/s; pass: >= 5 forcing half-turns in >= 50% of roots");
    line("  " + pct(pass, n) + " of " + std::to_string(n) + " roots (median horizon " +
         (median(hs) >= kAll ? std::string("all") : fmt(median(hs), 1)) + ")" +
         depth_band(median(hs), 5) +
         (uncovered > 0 ? "; " + std::to_string(uncovered) +
                              " scan(s) capped by --max-nodes below this budget (a lower bound)"
                        : std::string()) +
         " -> " + share_verdict(share, true, "the verification layer needs a different shape (11.1)"));
    if (wall_n > 0) {
      const char* names[4] = {"< 50 boards", "50-99", "100-199", "200+"};
      std::string by;
      for (int b = 0; b < 4; ++b) {
        if (wall_bucket_n[b] == 0) continue;
        by += std::string(" ") + names[b] + " " + pct(wall_bucket_pass[b], wall_bucket_n[b]) +
              " of " + std::to_string(wall_bucket_n[b]) + ";";
      }
      line("  by the wall clock (one " + fmt(o.root_ms, 0) + " ms scan per root, machine as it was): " +
           pct(wall_pass, wall_n) + " of " + std::to_string(wall_n) + " roots; by size:" + by);
    }
    const auto at = [&](const Result&) { return budget; };
    line(breakdown(Role::Root, false, at, 5));
    if (syn_roots > 0) {
      line("  SYNTHETIC roots (no verdict):");
      line(breakdown(Role::Root, true, at, 5));
    }
  }
  line("");

  // (v) stand-pat validity.
  {
    struct Band {
      std::size_t quiet = 0, valid = 0, invalid = 0, invalid_exhaustive = 0, undetermined = 0,
                  no_action = 0;
    };
    std::map<std::string, Band> bands;
    std::size_t considered = 0;
    double spv_cpu = 0;
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (!(counted(i, Role::Root) || counted(i, Role::Leaf))) continue;
      ++considered;
      const Result& r = results[i];
      if (!r.spv) continue;
      spv_cpu += r.spv_seconds;
      const t5d::StandPatVerdict& v = *r.spv;
      if (v.kind == t5d::StandPatVerdict::Kind::NotQuiet) continue;
      const std::string key = v.mandatory_boards <= 1   ? "1 mandatory board"
                              : v.mandatory_boards <= 3 ? "2-3 mandatory boards"
                                                        : "4+ mandatory boards";
      for (Band* b : {&bands["all"], &bands[key]}) {
        ++b->quiet;
        switch (v.kind) {
          case t5d::StandPatVerdict::Kind::Valid: ++b->valid; break;
          case t5d::StandPatVerdict::Kind::Invalid:
            ++b->invalid;
            if (v.exhaustive) ++b->invalid_exhaustive;
            break;
          case t5d::StandPatVerdict::Kind::Undetermined: ++b->undetermined; break;
          case t5d::StandPatVerdict::Kind::NoAction: ++b->no_action; break;
          default: break;
        }
      }
    }
    line("(v) STAND-PAT VALIDITY - at quiet positions (" + std::to_string(bands["all"].quiet) +
         " of " + std::to_string(considered) + " scanned roots and leaves are quiet); " +
         std::to_string(o.spv_check.max_actions) + " actions, reply depth " +
         std::to_string(o.spv_check.reply_depth) + ", " + std::to_string(o.spv_check.reply_nodes) +
         " nodes per reply; " + fmt(spv_cpu, 1) + " s");
    line("  bands: <= 5% fine; 5-20% disable stand-pat where obligations bind; > 20% redesign");
    for (const auto& [key, b] : bands) {
      const std::size_t decided = b.valid + b.invalid;
      const double share = decided == 0 ? 0 : static_cast<double>(b.invalid) / static_cast<double>(decided);
      std::string verdict = decided == 0             ? "no quiet position decided"
                            : share <= 0.05          ? "fine (<= 5%)"
                            : share <= 0.20          ? "5-20%: disable stand-pat where obligations bind"
                                                     : "> 20%: redesign (an obligation-aware bound)";
      if (decided > 0) {
        for (const double t : {0.05, 0.20}) {
          if (share >= 0.9 * t && share <= 1.1 * t) verdict += " - within +-10% of a band edge: STOP for the user";
        }
      }
      line("  " + key + ": " + std::to_string(b.quiet) + " quiet; invalid " + pct(b.invalid, decided) +
           " of " + std::to_string(decided) + " decided (" + std::to_string(b.invalid_exhaustive) +
           " exhaustively, the rest on a sample: an upper bound); " + std::to_string(b.undetermined) +
           " undetermined -> " + verdict);
    }
  }

  if (!o.summary.empty()) {
    std::FILE* f = std::fopen(o.summary.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "t5d-scan6b: cannot write %s\n", o.summary.c_str());
      return 1;
    }
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
  }
  std::fwrite(out.data(), 1, out.size(), stdout);

  // -- JSON --------------------------------------------------------------------------------------
  if (!o.json.empty()) {
    std::string j = "{\"tool\":\"t5d-scan6b\",\"version\":\"" T5D_VERSION_STRING "\"";
    j += ",\"machine\":{\"cpu\":" + json_string(cpu) + ",\"physical_cores\":" + std::to_string(cores) +
         ",\"logical_processors\":" + std::to_string(logical) + ",\"busy_fraction\":" + fmt(busy, 4) + "}";
    j += ",\"build\":{\"compiler\":" + json_string(compiler_text()) + ",\"build_type\":" +
         json_string(T5D_BUILD_TYPE) + ",\"flags\":" + json_string(T5D_CXX_FLAGS) + "}";
    j += ",\"rules\":" + json_string(t5d::describe(o.rules));
    char enc[32];
    std::snprintf(enc, sizeof enc, "\"0x%llx\"", static_cast<unsigned long long>(o.rules.encode()));
    j += std::string(",\"rules_encoding\":") + enc;
    j += ",\"seed\":" + std::to_string(o.seed) + ",\"jobs\":" + std::to_string(o.jobs);
    j += ",\"max_nodes\":" + std::to_string(o.max_nodes);
    j += ",\"nps_measured\":" + fmt(measured_nps, 1) + ",\"nps_used\":" + fmt(nps, 1);
    j += ",\"scan_cpu_seconds\":" + fmt(total_cpu, 3) + ",\"classification_seconds\":" +
         fmt(measurement, 3) + ",\"wall_seconds\":" + fmt(wall_seconds, 3);
    j += ",\"scanner\":{\"completion\":" + std::to_string(static_cast<int>(o.scanner.completion)) +
         ",\"checks\":" + (o.scanner.checks ? "true" : "false") +
         ",\"branch_threat\":" + std::to_string(static_cast<int>(o.scanner.branch_threat)) +
         ",\"branch_threat_value\":" + std::to_string(o.scanner.branch_threat_value) +
         ",\"promotions_forcing\":" + (o.scanner.promotions_forcing ? "true" : "false") +
         ",\"stand_pat\":" + std::to_string(static_cast<int>(o.scanner.stand_pat)) +
         ",\"evasion_cap\":" + std::to_string(o.scanner.evasion_cap) +
         ",\"evasion_probe\":" + std::to_string(o.scanner.evasion_probe) +
         ",\"fallback_max_nodes\":" + std::to_string(o.scanner.fallback_max_nodes) +
         ",\"max_depth\":" + std::to_string(o.scanner.max_depth) +
         ",\"tt_entries\":" + std::to_string(o.scanner.tt_entries) + ",\"values\":[";
    for (std::size_t k = 0; k < 12; ++k) {
      j += (k ? "," : "") + std::to_string(o.scanner.values.value[k]);
    }
    j += "]}";
    j += ",\"records\":[";
    for (std::size_t k = 0; k < record_notes.size(); ++k) {
      j += (k ? "," : "") + json_string(record_notes[k]);
    }
    j += "],\"rung1\":[";
    for (std::size_t k = 0; k < rungs.size(); ++k) {
      const Rung& g = rungs[k];
      j += std::string(k ? "," : "") + "{\"threads\":" + std::to_string(g.threads) +
           ",\"eval_rate\":" + fmt(g.rate, 0) + ",\"budget\":" + std::to_string(g.budget) +
           ",\"leaves\":" + std::to_string(g.leaves) + ",\"pass\":" + std::to_string(g.pass) +
           ",\"scan_leaves\":" + std::to_string(g.leaf_total) + ",\"scan_leaves_quiet\":" +
           std::to_string(g.leaf_quiet) + "}";
    }
    j += "]";
    if (o.per_position) {
      j += ",\"positions\":[";
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const Position& p = positions[i];
        const Result& r = results[i];
        j += std::string(i ? "," : "") + "{\"source\":" + json_string(p.source) +
             ",\"synthetic\":" + (p.synthetic ? "true" : "false") + ",\"index\":" +
             std::to_string(p.index) + ",\"role\":\"" + (p.role == Role::Root ? "root" : "leaf") +
             "\",\"parent\":" + std::to_string(p.parent) + ",\"fingerprint\":\"" +
             t5d::fingerprint(p.state->multiverse()).fingerprint + "\",\"boards\":" +
             std::to_string(r.boards) + ",\"timelines\":" + std::to_string(r.timelines) +
             ",\"active\":" + std::to_string(r.active) + ",\"mandatory\":" +
             std::to_string(r.mandatory) + ",\"scanned\":" + (r.scanned ? "true" : "false");
        if (!r.scanned) {
          j += ",\"skipped\":" + json_string(r.skipped) + "}";
          continue;
        }
        j += std::string(",\"in_check\":") + (r.in_check ? "true" : "false") + ",\"quiet\":" +
             (r.quiet ? "true" : "false") + ",\"cpu_seconds\":" + fmt(r.cpu_seconds, 4) +
             ",\"proof\":" + json_string(t5d::to_string(r.scan.proof)) + ",\"resolved_depth\":" +
             std::to_string(r.scan.resolved_depth) + ",\"exhausted\":" +
             (r.scan.exhausted ? "true" : "false") + ",\"nodes\":" +
             std::to_string(r.scan.stats.nodes) + ",\"iterations\":[";
        for (std::size_t k = 0; k < r.scan.iterations.size(); ++k) {
          const t5d::IterationStats& it = r.scan.iterations[k];
          j += std::string(k ? "," : "") + "{\"depth\":" + std::to_string(it.depth) +
               ",\"completed\":" + (it.completed ? "true" : "false") + ",\"capped\":" +
               (it.capped ? "true" : "false") + ",\"exhausted\":" + (it.exhausted ? "true" : "false") +
               ",\"nodes\":" + std::to_string(it.nodes) + ",\"positions\":" +
               std::to_string(it.positions) + ",\"score\":" + std::to_string(it.score) +
               ",\"proof\":" + json_string(t5d::to_string(it.proof)) + ",\"max_board_moves\":" +
               std::to_string(it.max_board_moves) + ",\"pv_half_turns\":" +
               std::to_string(it.pv_half_turns) + ",\"pv_board_moves\":" +
               std::to_string(it.pv_board_moves) + ",\"leaves\":" +
               std::to_string(it.leaves.total()) + ",\"leaves_quiet\":" +
               std::to_string(it.leaves.reached_quiet()) + "}";
        }
        j += "]";
        if (r.spv) {
          const t5d::StandPatVerdict& v = *r.spv;
          j += ",\"stand_pat\":{\"kind\":" + std::to_string(static_cast<int>(v.kind)) +
               ",\"exhaustive\":" + (v.exhaustive ? "true" : "false") + ",\"eval\":" +
               std::to_string(v.eval) + ",\"actions\":" + std::to_string(v.actions) +
               ",\"holding\":" + std::to_string(v.holding) + ",\"losing\":" +
               std::to_string(v.losing) + ",\"unknown\":" + std::to_string(v.unknown) + "}";
        }
        if (r.wall) {
          j += ",\"wall_resolved_depth\":" + std::to_string(r.wall->resolved_depth) +
               ",\"wall_proof\":" + json_string(t5d::to_string(r.wall->proof));
        }
        j += "}";
      }
      j += "]";
    }
    j += ",\"summary\":" + json_string(out) + "}\n";
    std::FILE* f = std::fopen(o.json.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "t5d-scan6b: cannot write %s\n", o.json.c_str());
      return 1;
    }
    std::fwrite(j.data(), 1, j.size(), f);
    std::fclose(f);
  }
  return 0;
}
