// t5d-scan6bp: experiment 6b-prime (design spec §15.6, the CHECKPOINT 3 block;
// register M15, M14, M13; §11.1 "The redesign after 6b").
//
//   t5d-scan6bp [options] [RECORD ...]
//
// POSITIONS - the same as 6b's (t5d-scan6b): ROOTS, the turn starts of the
// 5dpgn records given (replayed as the gate adapter's parse mode does, under
// the ruleset given), every K-th (--every), distinct by position; LEAVES,
// --leaves per root --leaf-depth half-turns below it by random legal play,
// made exactly as t5d-scan6b makes them (a random pseudo-legal seed completed
// by the scanner's rule, E4's witness when none completes; the same seeds). A
// position whose side to move could capture a royal (the record's last action
// broke C4), or that has no legal action, is counted and not measured.
//
// THE TESTS (M15), read per BOARD-COUNT RANGE - < 50, 50-99, 100-199, 200+ -
// never pooled (M12), at M13's 5 scanner threads:
//  (A) THE PROOF CHANNEL (t5d/prover.hpp) at every root: depth-limited df-pn
//      for 50 ms of wall clock (--root-ms) by 5 threads (--threads), a fresh
//      prover and stores each, one root at a time. DECIDED: mate within 5
//      half-turns (--depth) proven, or disproven (no such mate within the
//      class). Pass: >= 50% of roots decided.
//  (B) THE ESTIMATE CHANNEL (t5d/estimate.hpp) at every leaf, one thread.
//      FITS, operationalised before measuring: the rung-1 leaf budget is what
//      the 5 scanner threads (--threads) can spend while one leaf waits for
//      its batched evaluation - 5 x the channel's measured nodes per second of
//      thread CPU time (over all the leaves; --nps fixes it) / §15.6's design
//      evaluation rate, size-matched (9,600/s below 420 boards, 4,600/s from
//      420; no GPU here) - and the channel fits a range when the 90th
//      percentile, over its leaves, of a leaf's cost in nodes divided by its
//      budget is at most 1. The per-leaf wall time against 5 / rate is
//      printed beside it, as information.
//  (C) BRUTE FORCE AGAINST DF-PN BELOW 50 BOARDS (M15: keep whichever decides
//      more roots there): the E5c scanner's 6b root scan at every root - 50
//      ms, 5 threads, Lazy SMP, a fresh scanner and store each; DECIDED: 5
//      forcing half-turns resolved, a proof, or an exhausted tree (6b's
//      reading). Every range is printed; the verdict is below 50 boards.
// The summary applies M15's outcome rules to (A) mechanically: every range
// from 50 boards up passes - adopt (below 50, keep whichever of brute force
// and df-pn decides more); 50-199 pass and 200+ fails - phase-dependent
// (above 200 boards proofs only at the root final scan and PV transitions);
// 50-199 fail - STOP for a design review. (B) is judged range by range: rung
// 1 returns, estimate channel only, where it fits. A share within 45-55% of a
// threshold is flagged, as 6b's tool does.
//
// Output: a summary on standard output (or --summary FILE) and, with --json
// FILE, every condition and figure (--per-position: every position's too).
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
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "synthetic.hpp"
#include "t5d/check.hpp"
#include "t5d/estimate.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/gate.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/prover.hpp"
#include "t5d/rules.hpp"
#include "t5d/scanner.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
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

// ---- the machine (as t5d-scan6b reads it) ---------------------------------------------------

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
    t.total = u64(kernel) + u64(user);
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
  return 1.0 - static_cast<double>(b.idle - a.idle) / static_cast<double>(b.total - a.total);
}

// ---- options -------------------------------------------------------------------------------

struct Options {
  t5d::RuleOptions rules;
  t5d::ProverOptions prover;
  t5d::EstimateOptions estimate;
  t5d::ScannerOptions scanner;  // brute force, and the leaves' random play
  std::vector<std::string> records;
  int every = 1;
  std::size_t max_roots = 0;
  int leaves = 1;
  int leaf_depth = 2;
  std::uint64_t seed = 1;
  int threads = 5;  // M13
  double root_ms = 50;
  double brute_ms = 50;
  bool brute = true;
  double nps = 0;  // the estimate channel's; 0: measured
  int min_boards = 0;            // information runs: measure only positions in
  int max_boards = 1 << 30;      // this board-count window
  std::string json;
  std::string summary;
  bool per_position = false;
};

[[noreturn]] void usage(int status) {
  std::FILE* out = status == 0 ? stdout : stderr;
  std::fputs(
      "t5d-scan6bp " T5D_VERSION_STRING
      " - experiment 6b-prime: the proof channel's roots decided at 50 ms, the estimate\n"
      "channel's leaf cost against the rung-1 budget, brute force against df-pn (design spec 15.6,\n"
      "register M15).\n"
      "\n"
      "usage: t5d-scan6bp [options] [RECORD ...]\n"
      " positions (as t5d-scan6b)\n"
      "  --every K  --max-roots N  --leaves N (1)  --leaf-depth D (2)  --seed S (1)\n"
      "  --record-list FILE         records listed one per line ('#' lines skipped)\n"
      "  --record-root DIR          ... their relative paths taken from DIR\n"
      "  --moves-after-shift, --rules ENCODING   the ruleset (rules Q4; RuleOptions::encode)\n"
      " the measurement\n"
      "  --threads T                the scanner thread budget (5, M13): (A), (B)'s budget, (C)\n"
      "  --root-ms MS               (A)'s time per root (50)\n"
      "  --brute-ms MS              (C)'s time per root (50)    --no-brute-force   skip (C)\n"
      "  --nps N                    (B): the estimate channel's nodes/s per thread (measured)\n"
      "  --min-boards N  --max-boards N   information runs: only positions in this window\n"
      " the proof channel (t5d/prover.hpp)\n"
      "  --depth D (5)  --seeds checks|forcing|all  --completion canonical|every\n"
      "  --completion-max-nodes N (2000)  --escape-probe-nodes N (256)  --legality-tiers\n"
      "  --tt-entries N  --listing-cap N  --root-disproofs-only\n"
      " the estimate channel (t5d/estimate.hpp)\n"
      "  --est-plies P (6)  --est-threat-plies P (1)  --est-delta D (200)  --est-max-nodes N\n"
      "  --est-live M (100)\n"
      " output\n"
      "  --json FILE  --summary FILE  --per-position  --help  --version\n",
      out);
  std::exit(status);
}

std::uint64_t number(const char* text) {
  const std::optional<std::uint64_t> n = t5d::tools::parse_number(text);
  if (!n) throw std::invalid_argument(std::string("not a number: ") + text);
  return *n;
}

Options parse(int argc, char** argv) {
  Options o;
  std::vector<std::string> list_files;
  std::string root;
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
      std::puts("t5d-scan6bp " T5D_VERSION_STRING);
      std::exit(0);
    } else if (a == "--every") {
      o.every = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--max-roots") {
      o.max_roots = static_cast<std::size_t>(number(value()));
    } else if (a == "--leaves") {
      o.leaves = static_cast<int>(number(value()));
    } else if (a == "--leaf-depth") {
      o.leaf_depth = static_cast<int>(number(value()));
    } else if (a == "--seed") {
      o.seed = number(value());
    } else if (a == "--threads") {
      o.threads = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--root-ms") {
      o.root_ms = std::atof(value());
    } else if (a == "--brute-ms") {
      o.brute_ms = std::atof(value());
    } else if (a == "--no-brute-force") {
      o.brute = false;
    } else if (a == "--nps") {
      o.nps = std::atof(value());
    } else if (a == "--depth") {
      o.prover.depth = static_cast<int>(number(value()));
    } else if (a == "--seeds") {
      const std::string v = value();
      if (v == "checks") {
        o.prover.seeds = t5d::SeedClass::Checks;
      } else if (v == "forcing") {
        o.prover.seeds = t5d::SeedClass::Forcing;
      } else if (v == "all") {
        o.prover.seeds = t5d::SeedClass::All;
      } else {
        throw std::invalid_argument("--seeds checks|forcing|all");
      }
    } else if (a == "--completion") {
      const std::string v = value();
      if (v == "canonical") {
        o.prover.completion = t5d::CompletionClass::Canonical;
      } else if (v == "every") {
        o.prover.completion = t5d::CompletionClass::Every;
      } else {
        throw std::invalid_argument("--completion canonical|every");
      }
    } else if (a == "--completion-max-nodes") {
      o.prover.completion_max_nodes = number(value());
    } else if (a == "--escape-probe-nodes") {
      o.prover.escape_probe_nodes = number(value());
    } else if (a == "--legality-tiers") {
      o.prover.legality_tiers = true;
    } else if (a == "--tt-entries") {
      o.prover.tt_entries = static_cast<std::size_t>(number(value()));
    } else if (a == "--listing-cap") {
      o.prover.listing_cap = static_cast<std::size_t>(number(value()));
    } else if (a == "--root-disproofs-only") {
      o.prover.record_all_disproofs = false;
    } else if (a == "--est-plies") {
      o.estimate.max_plies = static_cast<int>(number(value()));
    } else if (a == "--est-threat-plies") {
      o.estimate.threat_plies = static_cast<int>(number(value()));
    } else if (a == "--est-delta") {
      o.estimate.delta_margin = static_cast<int>(number(value()));
    } else if (a == "--est-max-nodes") {
      o.estimate.max_nodes = number(value());
    } else if (a == "--est-live") {
      o.estimate.live_margin = static_cast<int>(number(value()));
    } else if (a == "--min-boards") {
      o.min_boards = static_cast<int>(number(value()));
    } else if (a == "--max-boards") {
      o.max_boards = static_cast<int>(number(value()));
    } else if (a == "--record-list") {
      list_files.push_back(value());
    } else if (a == "--record-root") {
      root = value();
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
  // Record lists: one path per line (blank lines and '#' lines skipped),
  // relative paths taken from --record-root when given.
  for (const std::string& file : list_files) {
    const std::string text = t5d::tools::read_file(file);
    std::size_t start = 0;
    while (start < text.size()) {
      std::size_t end = text.find('\n', start);
      if (end == std::string::npos) end = text.size();
      std::string path = trim(text.substr(start, end - start));
      start = end + 1;
      if (path.empty() || path[0] == '#') continue;
      const bool absolute = path[0] == '/' || path[0] == '\\' || (path.size() > 1 && path[1] == ':');
      if (!root.empty() && !absolute) path = root + "/" + path;
      o.records.push_back(path);
    }
  }
  return o;
}

// ---- positions (t5d-scan6b's construction, unchanged) -----------------------------------------

enum class Role : std::uint8_t { Root = 0, Leaf };

struct Position {
  std::string source;
  bool synthetic = false;
  int index = 0;
  Role role = Role::Root;
  int parent = -1;
  std::optional<t5d::GameState> state;
};

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

bool broken(const t5d::GameState& s) {
  return t5d::royal_capture_exists(s.multiverse(), s.rules(), s.to_move(), s.inactive_timelines());
}

// ---- results --------------------------------------------------------------------------------

struct Result {
  bool measured = false;
  std::string skipped;
  int boards = 0;
  int timelines = 0;
  int mandatory = 0;
  bool in_check = false;
  // (A)
  std::optional<t5d::ProveResult> prove;
  // (C)
  std::optional<t5d::ScanResult> brute;
  // (B)
  std::optional<t5d::EstimateResult> estimate;
  double estimate_seconds = 0;  // wall
};

constexpr int kRanges = 4;
const char* const kRangeNames[kRanges] = {"< 50 boards", "50-99", "100-199", "200+"};
int range_of(int boards) { return boards < 50 ? 0 : boards < 100 ? 1 : boards < 200 ? 2 : 3; }

std::string fmt(double v, int decimals = 1) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
  return buf;
}
std::string pct(std::size_t a, std::size_t b) {
  return b == 0 ? "-" : fmt(100.0 * static_cast<double>(a) / static_cast<double>(b)) + "%";
}
double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  // The nearest-rank quantile: the smallest value with at least q of the
  // values at or below it.
  const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(v.size())));
  return v[std::min(v.size() - 1, rank == 0 ? 0 : rank - 1)];
}
std::string band(double share) {
  return share >= 0.45 && share <= 0.55 ? " [within 45-55% of the 50% line: STOP for the user]" : "";
}

bool brute_decided(const t5d::ScanResult& r) {
  return r.proof.proven() || r.exhausted || r.resolved_depth >= 5;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = parse(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-scan6bp: %s\n", e.what());
    usage(2);
  }
  o.prover.threads = o.threads;
  const std::string cpu = cpu_model();
  const unsigned logical = std::thread::hardware_concurrency();

  // -- positions: t5d-scan6b's ----------------------------------------------------------------
  std::vector<Position> positions;
  std::vector<std::string> record_notes;
  std::unordered_set<std::uint64_t> seen;
  std::size_t duplicates = 0;
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
          if (!seen.insert(p.state->zobrist()).second) {
            ++duplicates;
          } else if (o.max_roots == 0 || positions.size() < o.max_roots) {
            positions.push_back(std::move(p));
          }
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
      record_notes.push_back(path + ": " + std::to_string(replay.actions_played()) + " actions read, " + end);
    } catch (const std::exception& e) {
      record_notes.push_back(path + ": not read (" + e.what() + ")");
    }
  }
  const std::size_t roots = positions.size();
  {
    t5d::Scanner helper(o.scanner);
    for (std::size_t r = 0; r < roots; ++r) {
      for (int l = 0; l < o.leaves; ++l) {
        t5d::synthetic::Rng rng(o.seed * std::uint64_t{104729} + static_cast<std::uint64_t>(r) * std::uint64_t{1009} +
                                static_cast<std::uint64_t>(l) + 3);
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

  // -- which positions are measured ------------------------------------------------------------
  std::vector<Result> results(positions.size());
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const t5d::GameState& s = *positions[i].state;
    Result& r = results[i];
    r.boards = static_cast<int>(s.multiverse().board_count());
    r.timelines = s.multiverse().timeline_count();
    r.mandatory = s.mandatory_count();
    if (r.boards < o.min_boards || r.boards > o.max_boards) {
      r.skipped = "outside --min-boards/--max-boards";
      continue;
    }
    if (broken(s)) {
      r.skipped = "a royal capturable by the side to move (the last action broke C4)";
      continue;
    }
    t5d::SearchBudget b = t5d::SearchBudget::nodes(2'000'000);
    b.max_bytes = std::size_t{256} << 20;
    const t5d::MateResult m = t5d::detect_mate(s, t5d::Detector::Hypercuboid, b);
    if (m.verdict == t5d::Outcome::Checkmate || m.verdict == t5d::Outcome::Stalemate) {
      r.skipped = std::string("terminal (") + (m.verdict == t5d::Outcome::Checkmate ? "checkmate" : "stalemate") + ")";
      continue;
    }
    r.in_check = m.in_check;
    r.measured = true;
  }
  const auto counted = [&](std::size_t i, Role role) {
    return results[i].measured && positions[i].role == role && !positions[i].synthetic;
  };

  // -- (A) the proof channel, one root at a time ---------------------------------------------------
  const CpuTimes a0 = cpu_times();
  const Clock::time_point a_wall0 = Clock::now();
  std::size_t done = 0;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (!results[i].measured || positions[i].role != Role::Root) continue;
    t5d::Prover prover(o.prover);
    results[i].prove = prover.prove(*positions[i].state,
                                    t5d::ProveBudget::time(std::chrono::nanoseconds(static_cast<long long>(o.root_ms * 1e6))));
    if (++done % 100 == 0) std::fprintf(stderr, "t5d-scan6bp: (A) %zu roots\n", done);
  }
  const double a_wall = std::chrono::duration<double>(Clock::now() - a_wall0).count();
  const double a_busy = busy_between(a0, cpu_times());

  // -- (C) brute force, one root at a time -----------------------------------------------------------
  const CpuTimes c0 = cpu_times();
  const Clock::time_point c_wall0 = Clock::now();
  if (o.brute) {
    t5d::ScannerOptions timed = o.scanner;
    timed.classify_leaves = false;
    timed.threads = o.threads;
    done = 0;
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (!results[i].measured || positions[i].role != Role::Root) continue;
      t5d::ScanBudget wb;
      wb.max_time = std::chrono::nanoseconds(static_cast<long long>(o.brute_ms * 1e6));
      t5d::Scanner fresh(timed);
      results[i].brute = fresh.scan(*positions[i].state, wb);
      if (++done % 100 == 0) std::fprintf(stderr, "t5d-scan6bp: (C) %zu roots\n", done);
    }
  }
  const double c_wall = std::chrono::duration<double>(Clock::now() - c_wall0).count();
  const double c_busy = busy_between(c0, cpu_times());

  // -- (B) the estimate channel, one leaf at a time, one thread ----------------------------------
  const CpuTimes b0 = cpu_times();
  const double cpu0 = thread_cpu_seconds();
  std::uint64_t est_nodes = 0;
  std::size_t est_capped = 0;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (!results[i].measured || positions[i].role != Role::Leaf) continue;
    const Clock::time_point t0 = Clock::now();
    results[i].estimate = t5d::estimate(*positions[i].state, o.estimate);
    results[i].estimate_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    est_nodes += results[i].estimate->nodes;
    if (results[i].estimate->capped) ++est_capped;
  }
  const double est_cpu = std::max(1e-9, thread_cpu_seconds() - cpu0);
  const double b_busy = busy_between(b0, cpu_times());
  const double measured_nps = static_cast<double>(est_nodes) / est_cpu;
  const double nps = o.nps > 0 ? o.nps : measured_nps;

  // -- the figures ----------------------------------------------------------------------------------
  std::string out;
  const auto line = [&](const std::string& s) { out += s + "\n"; };
  const std::string pooled = "pooled (not a verdict, M12)";
  line("t5d-scan6bp " T5D_VERSION_STRING " - experiment 6b-prime (design spec 15.6, CHECKPOINT 3; register M15)");
  line("");
  line("CONDITIONS");
  line("  machine: " + cpu + ", " + std::to_string(logical) + " logical processors; busy " + fmt(100 * a_busy) +
       "% during (A), " + fmt(100 * c_busy) + "% during (C), " + fmt(100 * b_busy) +
       "% during (B) (all processes; the tool's own threads included)");
  line("  build: " + compiler_text() + ", " + T5D_BUILD_TYPE + ", flags \"" + T5D_CXX_FLAGS + "\"");
  line("  rules: " + t5d::describe(o.rules) + (t5d::decomposes(o.rules) ? " - turns decompose into semimoves"
                                                                          : " - turns do NOT decompose: E4's listings"));
  line("  (A) proof channel: df-pn, depth " + std::to_string(o.prover.depth) + " half-turns, seeds " +
       t5d::seed_class_name(o.prover.seeds) + ", completion " + t5d::completion_class_name(o.prover.completion) +
       " (cap " + std::to_string(o.prover.completion_max_nodes) + " steps), escape probe " +
       std::to_string(o.prover.escape_probe_nodes) + " steps, legality tiers " +
       (o.prover.legality_tiers ? "on" : "off") + ", TT " + std::to_string(o.prover.tt_entries) + ", " +
       std::to_string(o.threads) + " threads, " + fmt(o.root_ms, 0) +
       " ms of wall clock per root, a fresh prover and stores each, one root at a time; wall " + fmt(a_wall, 1) + " s");
  line("  (B) estimate channel: " + std::to_string(o.estimate.max_plies) + " plies, threats in " +
       std::to_string(o.estimate.threat_plies) + ", delta margin " + std::to_string(o.estimate.delta_margin) +
       ", node cap " + std::to_string(o.estimate.max_nodes) + ", one thread; " + std::to_string(est_nodes) +
       " nodes in " + fmt(est_cpu, 3) + " s of thread CPU time = " + fmt(measured_nps, 0) + " nodes/s" +
       (o.nps > 0 ? " (budgets from --nps " + fmt(o.nps, 0) + ")" : "") + "; " + std::to_string(est_capped) +
       " leaves hit the node cap");
  line("  (C) brute force: the E5c scanner (t5d/scanner.hpp defaults), " + std::to_string(o.threads) +
       " threads, Lazy SMP, " + fmt(o.brute_ms, 0) + " ms per root, a fresh scanner and store each" +
       (o.brute ? "; wall " + fmt(c_wall, 1) + " s" : std::string(": not run (--no-brute-force)")));
  line("  positions: every " + std::to_string(o.every) + " turn start(s) of " + std::to_string(o.records.size()) +
       " record(s); leaves " + std::to_string(o.leaves) + " per root at " + std::to_string(o.leaf_depth) +
       " half-turns of random legal play (t5d-scan6b's, seed " + std::to_string(o.seed) + "); " +
       std::to_string(duplicates) + " duplicate root(s) dropped");
  for (const std::string& n : record_notes) line("    " + n);
  {
    std::map<std::string, std::size_t> skipped;
    std::size_t nr = 0;
    std::size_t nl = 0;
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (!results[i].measured) {
        skipped[(positions[i].role == Role::Root ? "root " : "leaf ") + results[i].skipped]++;
        continue;
      }
      if (counted(i, Role::Root)) ++nr;
      if (counted(i, Role::Leaf)) ++nl;
    }
    line("  measured: " + std::to_string(nr) + " roots and " + std::to_string(nl) + " leaves");
    for (const auto& [why, n] : skipped) line("    not measured: " + std::to_string(n) + " " + why);
  }
  line("");

  // (A)
  struct Share {
    std::size_t n = 0, decided = 0, proven = 0, disproven = 0, no_action = 0;
    std::vector<double> nodes;
    std::uint64_t expansions = 0, mate_tests = 0, completions = 0, completion_failures = 0, detector = 0,
                  queries = 0, dead = 0, all_nodes = 0, listings = 0;
    double share() const { return n == 0 ? 0.0 : static_cast<double>(decided) / static_cast<double>(n); }
  };
  Share proof[kRanges + 1];
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (!counted(i, Role::Root) || !results[i].prove) continue;
    const t5d::ProveResult& r = *results[i].prove;
    for (const int k : {range_of(results[i].boards), kRanges}) {
      Share& s = proof[k];
      ++s.n;
      if (r.status != t5d::ProofStatus::Unknown) ++s.decided;
      if (r.status == t5d::ProofStatus::Proven) ++s.proven;
      if (r.status == t5d::ProofStatus::Disproven) {
        ++s.disproven;
        if (r.root_actions == 0) ++s.no_action;
      }
      s.nodes.push_back(static_cast<double>(r.stats.nodes));
      s.expansions += r.stats.expansions;
      s.mate_tests += r.stats.mate_tests;
      s.completions += r.stats.completions;
      s.completion_failures += r.stats.completion_failures;
      s.detector += r.stats.detector_nodes;
      s.queries += r.stats.attack_queries;
      s.dead += r.stats.dead;
      s.all_nodes += r.stats.nodes;
      s.listings += r.stats.listings;
    }
  }
  line("(A) PROOF CHANNEL - df-pn, " + fmt(o.root_ms, 0) + " ms at " + std::to_string(o.threads) +
       " threads per root; decided = mate within " + std::to_string(o.prover.depth) +
       " half-turns proven, or disproven within the class; pass: >= 50% of roots decided");
  for (int k = 0; k <= kRanges; ++k) {
    const Share& s = proof[k];
    if (s.n == 0) continue;
    const bool is_pooled = k == kRanges;
    line(std::string("  ") + (is_pooled ? pooled : kRangeNames[k]) + ": " + pct(s.decided, s.n) + " of " +
         std::to_string(s.n) + " roots decided (" + std::to_string(s.proven) + " proven, " +
         std::to_string(s.disproven) + " disproven - " + std::to_string(s.no_action) +
         " of them with no action of the class at the root - " + std::to_string(s.n - s.decided) +
         " unknown); median " + fmt(quantile(s.nodes, 0.5), 0) + " nodes" +
         (is_pooled ? std::string() : " -> " + std::string(s.share() >= 0.5 ? "PASS" : "FAIL") + band(s.share())));
  }
  line("  where the work went (nodes; every thread's): ");
  for (int k = 0; k < kRanges; ++k) {
    const Share& s = proof[k];
    if (s.n == 0 || s.all_nodes == 0) continue;
    const auto share = [&](std::uint64_t part) {
      return fmt(100.0 * static_cast<double>(part) / static_cast<double>(s.all_nodes)) + "%";
    };
    line(std::string("    ") + kRangeNames[k] + ": " + std::to_string(s.all_nodes) + " nodes; E4 searches " +
         share(s.detector) + ", attack-map queries " + share(s.queries) + "; " + std::to_string(s.expansions) +
         " expansions, " + std::to_string(s.mate_tests) + " mate tests, " + std::to_string(s.completions) +
         " completions (" + std::to_string(s.completion_failures) + " seeds with none), " +
         std::to_string(s.dead) + " partial actions cut by a standing capture, " + std::to_string(s.listings) +
         " listings");
  }
  line("");

  // (B)
  struct Fit {
    std::vector<double> ratio;
    std::vector<double> nodes;
    std::vector<double> budget;
    std::vector<double> wall_us;
    std::vector<double> wall_budget_us;
    std::size_t live = 0;
  };
  Fit fit[kRanges + 1];
  const auto rate_for = [](int boards) { return boards < 420 ? 9600.0 : 4600.0; };
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (!counted(i, Role::Leaf) || !results[i].estimate) continue;
    const t5d::EstimateResult& e = *results[i].estimate;
    const double rate = rate_for(results[i].boards);
    const double budget = static_cast<double>(o.threads) * nps / rate;
    for (const int k : {range_of(results[i].boards), kRanges}) {
      Fit& f = fit[k];
      f.ratio.push_back(static_cast<double>(e.nodes) / std::max(budget, 1e-9));
      f.nodes.push_back(static_cast<double>(e.nodes));
      f.budget.push_back(budget);
      f.wall_us.push_back(results[i].estimate_seconds * 1e6);
      f.wall_budget_us.push_back(static_cast<double>(o.threads) / rate * 1e6);
      if (e.live) ++f.live;
    }
  }
  line("(B) ESTIMATE CHANNEL - per leaf, one thread; leaf budget = " + std::to_string(o.threads) + " x " +
       fmt(nps, 0) + " nodes/s / evaluation rate (9,600/s below 420 boards, 4,600/s from 420); fits: p90 of "
       "cost / budget <= 1");
  for (int k = 0; k <= kRanges; ++k) {
    const Fit& f = fit[k];
    if (f.ratio.empty()) continue;
    const bool is_pooled = k == kRanges;
    const double p90 = quantile(f.ratio, 0.9);
    line(std::string("  ") + (is_pooled ? pooled : kRangeNames[k]) + ": " + std::to_string(f.ratio.size()) +
         " leaves; p90 cost " + fmt(quantile(f.nodes, 0.9), 0) + " nodes (median " + fmt(quantile(f.nodes, 0.5), 0) +
         ") against a budget of " + fmt(quantile(f.budget, 0.5), 0) + " nodes; p90 cost/budget " + fmt(p90, 2) +
         "; wall p90 " + fmt(quantile(f.wall_us, 0.9), 0) + " us against " + fmt(quantile(f.wall_budget_us, 0.5), 0) +
         " us (information); " + pct(f.live, f.ratio.size()) + " tactically live" +
         (is_pooled ? std::string() : std::string(" -> ") + (p90 <= 1.0 ? "FITS" : "DOES NOT FIT")));
  }
  line("");

  // (C)
  Share brute[kRanges + 1];
  Share dfpn_side[kRanges + 1];
  if (o.brute) {
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (!counted(i, Role::Root) || !results[i].brute || !results[i].prove) continue;
      for (const int k : {range_of(results[i].boards), kRanges}) {
        ++brute[k].n;
        if (brute_decided(*results[i].brute)) ++brute[k].decided;
        if (results[i].brute->proof.proven()) ++brute[k].proven;
        ++dfpn_side[k].n;
        if (results[i].prove->status != t5d::ProofStatus::Unknown) ++dfpn_side[k].decided;
      }
    }
    line("(C) BRUTE FORCE AGAINST DF-PN - roots decided at " + fmt(o.brute_ms, 0) + " ms, " +
         std::to_string(o.threads) + " threads (brute force: 5 forcing half-turns resolved, a proof or an "
         "exhausted tree; df-pn: proven or disproven); M15 keeps whichever decides more below 50 boards");
    for (int k = 0; k <= kRanges; ++k) {
      if (brute[k].n == 0) continue;
      const bool is_pooled = k == kRanges;
      line(std::string("  ") + (is_pooled ? pooled : kRangeNames[k]) + ": brute force " + pct(brute[k].decided, brute[k].n) +
           " (" + std::to_string(brute[k].decided) + " of " + std::to_string(brute[k].n) + ", " +
           std::to_string(brute[k].proven) + " by a proof), df-pn " + pct(dfpn_side[k].decided, dfpn_side[k].n) + " (" +
           std::to_string(dfpn_side[k].decided) + ")" + (k == 0 ? "  <- the verdict range" : ""));
    }
    line("");
  }

  // The summary: M15's outcome rules, mechanically.
  enum class V : std::uint8_t { Pass, Fail, Empty };
  const auto v_of = [](const Share& s) { return s.n == 0 ? V::Empty : s.share() >= 0.5 ? V::Pass : V::Fail; };
  const auto word = [](V v) { return v == V::Pass ? "PASS" : v == V::Fail ? "FAIL" : "no roots"; };
  line("SUMMARY - read per board-count range (M12, M15); pooled figures are information, not a verdict");
  {
    std::string row = "  (A) proof channel:";
    std::vector<std::string> in_band;
    for (int k = 0; k < kRanges; ++k) {
      row += std::string(" ") + kRangeNames[k] + " " + word(v_of(proof[k])) +
             (proof[k].n == 0 ? std::string() : " (" + pct(proof[k].decided, proof[k].n) + ")") + ";";
      if (proof[k].n > 0 && proof[k].share() >= 0.45 && proof[k].share() <= 0.55) in_band.push_back(kRangeNames[k]);
    }
    line(row);
    const V v50 = v_of(proof[1]);
    const V v100 = v_of(proof[2]);
    const V v200 = v_of(proof[3]);
    std::string outcome;
    if (v50 == V::Fail || v100 == V::Fail) {
      outcome = "STOP - 50-199 boards fail: exact verification covers only small positions; a design review (M15)";
    } else if (v50 == V::Empty || v100 == V::Empty) {
      outcome = "UNDECIDED - no roots in 50-99 or 100-199 boards";
    } else if (v200 == V::Pass) {
      outcome = "ADOPT - every range from 50 boards up passes (M15)";
    } else if (v200 == V::Fail) {
      outcome =
          "PHASE-DEPENDENT - 50-199 pass, 200+ fails: above 200 boards proofs only at the root final scan "
          "and PV transitions; the coverage gap recorded in 13 (M15)";
    } else {
      outcome = "UNDECIDED - 50-199 pass; no roots of 200+ boards";
    }
    line("  -> M15 OUTCOME: " + outcome);
    if (!in_band.empty()) {
      std::string ranges;
      for (const std::string& r : in_band) ranges += (ranges.empty() ? "" : ", ") + r;
      line("  -> (A) within 45-55% in " + ranges + ": STOP for the user (the brief's band)");
    }
  }
  {
    std::string fits;
    std::string misses;
    for (int k = 0; k < kRanges; ++k) {
      if (fit[k].ratio.empty()) continue;
      std::string& to = quantile(fit[k].ratio, 0.9) <= 1.0 ? fits : misses;
      to += (to.empty() ? "" : ", ") + std::string(kRangeNames[k]);
    }
    line("  (B) estimate channel fits the rung-1 leaf budget at p90 in " + (fits.empty() ? std::string("no range") : fits) +
         "; not in " + (misses.empty() ? std::string("any range") : misses) +
         " -> rung 1 returns, estimate channel only, where it fits; stays dropped elsewhere (H10, M15)");
  }
  if (o.brute && brute[0].n > 0) {
    const bool df = dfpn_side[0].decided >= brute[0].decided;
    line("  (C) below 50 boards: brute force decides " + std::to_string(brute[0].decided) + ", df-pn " +
         std::to_string(dfpn_side[0].decided) + " of " + std::to_string(brute[0].n) + " -> keep " +
         (dfpn_side[0].decided == brute[0].decided ? std::string("either (a tie; df-pn by the rule's >=)")
                                                   : df ? std::string("df-pn") : std::string("brute force")) +
         " below 50 boards (M15)");
  }

  if (!o.summary.empty()) {
    std::FILE* f = std::fopen(o.summary.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "t5d-scan6bp: cannot write %s\n", o.summary.c_str());
      return 1;
    }
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
  }
  std::fwrite(out.data(), 1, out.size(), stdout);

  if (!o.json.empty()) {
    std::string j = "{\"tool\":\"t5d-scan6bp\",\"version\":\"" T5D_VERSION_STRING "\"";
    j += ",\"machine\":{\"cpu\":" + json_string(cpu) + ",\"logical_processors\":" + std::to_string(logical) +
         ",\"busy_a\":" + fmt(a_busy, 4) + ",\"busy_b\":" + fmt(b_busy, 4) + ",\"busy_c\":" + fmt(c_busy, 4) + "}";
    j += ",\"build\":{\"compiler\":" + json_string(compiler_text()) + ",\"build_type\":" + json_string(T5D_BUILD_TYPE) +
         ",\"flags\":" + json_string(T5D_CXX_FLAGS) + "}";
    j += ",\"rules\":" + json_string(t5d::describe(o.rules));
    j += ",\"threads\":" + std::to_string(o.threads) + ",\"root_ms\":" + fmt(o.root_ms, 1) +
         ",\"brute_ms\":" + fmt(o.brute_ms, 1);
    j += ",\"prover\":{\"depth\":" + std::to_string(o.prover.depth) + ",\"seeds\":" +
         json_string(t5d::seed_class_name(o.prover.seeds)) + ",\"completion\":" +
         json_string(t5d::completion_class_name(o.prover.completion)) + ",\"completion_max_nodes\":" +
         std::to_string(o.prover.completion_max_nodes) + ",\"escape_probe_nodes\":" +
         std::to_string(o.prover.escape_probe_nodes) + ",\"legality_tiers\":" +
         (o.prover.legality_tiers ? "true" : "false") + "}";
    j += ",\"estimate\":{\"max_plies\":" + std::to_string(o.estimate.max_plies) + ",\"threat_plies\":" +
         std::to_string(o.estimate.threat_plies) + ",\"delta_margin\":" + std::to_string(o.estimate.delta_margin) +
         ",\"max_nodes\":" + std::to_string(o.estimate.max_nodes) + ",\"nps_measured\":" + fmt(measured_nps, 1) +
         ",\"nps_used\":" + fmt(nps, 1) + "}";
    j += ",\"ranges\":[\"< 50\",\"50-99\",\"100-199\",\"200+\",\"pooled\"],\"proof\":[";
    for (int k = 0; k <= kRanges; ++k) {
      j += std::string(k ? "," : "") + "{\"n\":" + std::to_string(proof[k].n) + ",\"decided\":" +
           std::to_string(proof[k].decided) + ",\"proven\":" + std::to_string(proof[k].proven) + ",\"disproven\":" +
           std::to_string(proof[k].disproven) + "}";
    }
    j += "],\"estimate_p90_ratio\":[";
    for (int k = 0; k <= kRanges; ++k) {
      j += std::string(k ? "," : "") + (fit[k].ratio.empty() ? std::string("null") : fmt(quantile(fit[k].ratio, 0.9), 4));
    }
    j += "],\"brute\":[";
    for (int k = 0; k <= kRanges; ++k) {
      j += std::string(k ? "," : "") + "{\"n\":" + std::to_string(brute[k].n) + ",\"decided\":" +
           std::to_string(brute[k].decided) + "}";
    }
    j += "]";
    if (o.per_position) {
      j += ",\"positions\":[";
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const Position& p = positions[i];
        const Result& r = results[i];
        j += std::string(i ? "," : "") + "{\"source\":" + json_string(p.source) + ",\"index\":" +
             std::to_string(p.index) + ",\"role\":\"" + (p.role == Role::Root ? "root" : "leaf") +
             "\",\"fingerprint\":\"" + t5d::fingerprint(p.state->multiverse()).fingerprint + "\",\"boards\":" +
             std::to_string(r.boards) + ",\"timelines\":" + std::to_string(r.timelines) + ",\"mandatory\":" +
             std::to_string(r.mandatory) + ",\"measured\":" + (r.measured ? "true" : "false");
        if (!r.measured) {
          j += ",\"skipped\":" + json_string(r.skipped) + "}";
          continue;
        }
        j += std::string(",\"in_check\":") + (r.in_check ? "true" : "false");
        if (r.prove) {
          j += ",\"prove\":{\"status\":" + json_string(t5d::proof_status_name(r.prove->status)) + ",\"proof\":" +
               json_string(t5d::to_string(r.prove->proof)) + ",\"root_actions\":" +
               std::to_string(r.prove->root_actions) + ",\"nodes\":" + std::to_string(r.prove->stats.nodes) +
               ",\"seconds\":" + fmt(r.prove->stats.seconds, 4) + "}";
        }
        if (r.brute) {
          j += ",\"brute\":{\"decided\":" + std::string(brute_decided(*r.brute) ? "true" : "false") +
               ",\"resolved_depth\":" + std::to_string(r.brute->resolved_depth) + ",\"proof\":" +
               json_string(t5d::to_string(r.brute->proof)) + ",\"exhausted\":" + (r.brute->exhausted ? "true" : "false") +
               "}";
        }
        if (r.estimate) {
          j += ",\"estimate\":{\"nodes\":" + std::to_string(r.estimate->nodes) + ",\"wall_us\":" +
               fmt(r.estimate_seconds * 1e6, 1) + ",\"score\":" + std::to_string(r.estimate->score) +
               ",\"swing\":" + std::to_string(r.estimate->swing()) + ",\"live\":" + (r.estimate->live ? "true" : "false") +
               ",\"capped\":" + (r.estimate->capped ? "true" : "false") + "}";
        }
        j += "}";
      }
      j += "]";
    }
    j += ",\"summary\":" + json_string(out) + "}\n";
    std::FILE* f = std::fopen(o.json.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "t5d-scan6bp: cannot write %s\n", o.json.c_str());
      return 1;
    }
    std::fwrite(j.data(), 1, j.size(), f);
    std::fclose(f);
  }
  return 0;
}
