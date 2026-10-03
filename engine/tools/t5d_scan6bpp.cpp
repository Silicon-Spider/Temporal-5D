// t5d-scan6bpp: experiment 6b'' - Guard and Hunt (design spec 15.6, the
// CHECKPOINT 4 block "6b'' - Guard and Hunt (pre-registered)"; register M17,
// H42, H43, M13; spec 11.4).
//
//   t5d-scan6bpp --truth DIR [options] [RECORD ...]
//
// POSITIONS - 6b's roots (sixb_roots.hpp): the turn starts of the 5dpgn
// records given (parse mode, the ruleset given), every one, distinct by
// position. A root whose side to move could capture a royal (its record broke
// C4), or that is terminal, is counted and not measured - as 6b, 6b' do.
//
// THE TESTS, read per BOARD-COUNT RANGE - < 50, 50-99, 100-199, 200+ - never
// pooled, at M13's 5 threads and 50 ms each:
//  GUARD (t5d/guard.hpp) at each root, for the CHOSEN ACTION: the action the
//      record plays next from it (a reading: the record's move is the move a
//      search would be about to play; a root whose record plays nothing
//      after it, or whose next action is not legal - C4 broken - is reported
//      and not counted). Iterative deepening from depth 1 under 50 ms of wall
//      clock, 5 threads. PASS in a range where DEPTH 1 COMPLETES - answered:
//      "safe to depth 1", or a refutation at depth 1 - in >= 95% of its
//      roots; the depth reached is reported per root and its distribution
//      per range.
//  HUNT (t5d/hunt.hpp) at each root: df-pn prove-only, 5 half-turns, its
//      default class, 50 ms, 5 threads. PASS in a range where it finds >= 50%
//      of the forced mates within 5 half-turns that the ground truth
//      establishes there (--truth: t5d-hunt-truth's file, refused unless its
//      bytes match its MANIFEST and its roots are these roots - record,
//      index and fingerprint). A range with no mate in the truth has no
//      verdict (nothing to find); Hunt's mates the truth lacks are flagged
//      (an incomplete truth).
// THE VERDICT per range (sixbpp.hpp, where tests hold the counting): adopt
// Guard / adopt Hunt / not - and where each is not adopted, the note for spec
// 13: "the chosen move goes unguarded in that range" (Guard), "proofs there
// come only from the search's solver" (Hunt). ONLY CONFIRMED PROOFS COUNT: a
// refutation or mate counts when its certificate is confirmed by E4 alone;
// one left incomplete or undecided is reported apart; one E4 refutes
// withholds every verdict. A Hunt verdict resting on fewer than 10 truth
// mates is flagged "too few to read". No outcome stops anything (M17): the
// summary says so.
//
// THE CONFIGURATION: Guard and Hunt as they ship (the prover's check filter
// off, the attacker probe on), every option printed in CONDITIONS and the
// JSON. --check-filter, --no-attacker-probe, node budgets, other threads or
// budgets, and subsets are information runs: the summary says no verdict is
// read from them.
//
// THE TRUTH is refused unless it and its provenance match their MANIFEST,
// the records given are its records in its order (their content ids), its
// rules are these (every option's value), and its roots are these roots
// (record, index, fingerprint); the summary says whether this build's code
// signature is the one that recorded it.
//
// Output: a summary on standard output (or --summary FILE); --json FILE with
// every condition and figure (--per-position: every root's too).
//
// The tool never names a corpus or anything under the oracle harness: the
// records come on the command line.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hunt_truth.hpp"
#include "sixb_roots.hpp"
#include "sixbpp.hpp"
#include "t5d/certificate.hpp"
#include "t5d/check.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/gate.hpp"
#include "t5d/guard.hpp"
#include "t5d/hunt.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d_code_signature.hpp"
#include "tool_support.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
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

// ---- the machine (as t5d-scan6b reads it) ----------------------------------------------------

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
    if (std::fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
                    &v[6], &v[7]) == 8) {
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
  std::vector<std::string> direct;
  std::vector<std::string> lists;
  std::string record_root;
  std::string truth;
  bool no_truth = false;
  int threads = 5;  // M13
  double guard_ms = 50;
  double hunt_ms = 50;
  std::uint64_t guard_nodes = 0;  // information: a node budget instead of the wall clock
  std::uint64_t hunt_nodes = 0;
  int guard_depth = 8;
  int hunt_depth = 5;
  std::size_t max_roots = 0;
  int min_boards = 0;
  int max_boards = 1 << 30;
  std::string json;
  std::string summary;
  bool per_position = false;
  bool check_filter = false;    // the shipped configuration (off); --check-filter: information
  bool attacker_probe = true;   // the shipped configuration (on); --no-attacker-probe: information
  bool crosscheck = true;       // every refutation and mate against E4 alone
  std::size_t check_positions = 20000;
};

[[noreturn]] void usage(int status) {
  std::FILE* out = status == 0 ? stdout : stderr;
  std::fputs(
      "t5d-scan6bpp " T5D_VERSION_STRING
      " - experiment 6b'': Guard and Hunt at 6b's roots, per board-count range (design spec 15.6,\n"
      "register M17).\n"
      "\n"
      "usage: t5d-scan6bpp --truth DIR [options] [RECORD ...]\n"
      " positions (6b's roots)\n"
      "  --record-list FILE         records listed one per line ('#' lines skipped); ids as listed\n"
      "  --record-root DIR          ... their relative paths taken from DIR\n"
      "  --rules ENCODING           the ruleset (RuleOptions::encode; default 0, the engine's)\n"
      "  --no-moves-after-shift     a diagnostic: Q4's Guide-literal reading\n"
      "  --max-roots N  --min-boards N  --max-boards N   information runs only\n"
      " the measurement (M17's: 5 threads, 50 ms; Guard and Hunt as they ship)\n"
      "  --threads T (5)  --guard-ms MS (50)  --hunt-ms MS (50)\n"
      "  --guard-depth D            Guard's deepest depth, the opponent's actions (8)\n"
      "  --hunt-depth D             Hunt's half-turns (5)\n"
      "  --truth DIR                Hunt's ground truth (t5d-hunt-truth: hunt-truth.tsv, provenance.tsv,\n"
      "                             MANIFEST); the records given must be its records, in its order\n"
      "  --no-truth                 information runs: Hunt's mates counted, no verdict\n"
      "  --no-crosscheck            skip checking every refutation and mate against E4 alone (then no\n"
      "                             verdict: only confirmed proofs count)\n"
      "  --check-positions N        the cross-check's defender positions each (20000)\n"
      " information runs (not the shipped configuration: no verdict is read from them)\n"
      "  --guard-nodes N  --hunt-nodes N   node budgets instead of the wall clock (deterministic at\n"
      "                             --threads 1)\n"
      "  --check-filter             the prover's last-turn check filter on (it ships off)\n"
      "  --no-attacker-probe        the attacker's turns not asked E4's question first (it ships on)\n"
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
      std::puts("t5d-scan6bpp " T5D_VERSION_STRING ", code " T5D_CODE_SIGNATURE);
      std::exit(0);
    } else if (a == "--record-list") {
      o.lists.push_back(value());
    } else if (a == "--record-root") {
      o.record_root = value();
    } else if (a == "--truth") {
      o.truth = value();
    } else if (a == "--check-filter") {
      o.check_filter = true;
    } else if (a == "--no-attacker-probe") {
      o.attacker_probe = false;
    } else if (a == "--no-crosscheck") {
      o.crosscheck = false;
    } else if (a == "--check-positions") {
      o.check_positions = static_cast<std::size_t>(number(value()));
    } else if (a == "--no-truth") {
      o.no_truth = true;
    } else if (a == "--threads") {
      o.threads = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--guard-ms") {
      o.guard_ms = std::atof(value());
    } else if (a == "--hunt-ms") {
      o.hunt_ms = std::atof(value());
    } else if (a == "--guard-nodes") {
      o.guard_nodes = number(value());
    } else if (a == "--hunt-nodes") {
      o.hunt_nodes = number(value());
    } else if (a == "--guard-depth") {
      o.guard_depth = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--hunt-depth") {
      o.hunt_depth = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--max-roots") {
      o.max_roots = static_cast<std::size_t>(number(value()));
    } else if (a == "--min-boards") {
      o.min_boards = static_cast<int>(number(value()));
    } else if (a == "--max-boards") {
      o.max_boards = static_cast<int>(number(value()));
    } else if (a == "--json") {
      o.json = value();
    } else if (a == "--summary") {
      o.summary = value();
    } else if (a == "--per-position") {
      o.per_position = true;
    } else if (!a.empty() && a[0] == '-' && a != "-") {
      throw std::invalid_argument("unknown option " + a);
    } else {
      o.direct.push_back(a);
    }
  }
  if (o.truth.empty() && !o.no_truth) throw std::invalid_argument("--truth DIR is needed (or --no-truth)");
  return o;
}

// Whether the run is M17's shipped configuration: a verdict is read only
// from such a run.
std::vector<std::string> departures(const Options& o) {
  std::vector<std::string> d;
  if (o.threads != 5) d.push_back("--threads " + std::to_string(o.threads));
  if (o.guard_nodes != 0 || o.hunt_nodes != 0) d.push_back("node budgets");
  if (o.guard_ms != 50 || o.hunt_ms != 50) d.push_back("budgets other than 50 ms");
  if (o.hunt_depth != 5) d.push_back("--hunt-depth");
  if (o.check_filter) d.push_back("--check-filter");
  if (!o.attacker_probe) d.push_back("--no-attacker-probe");
  if (o.max_roots != 0 || o.min_boards != 0 || o.max_boards != (1 << 30)) d.push_back("a subset of the roots");
  return d;
}

t5d::ProveBudget budget(double ms, std::uint64_t nodes) {
  if (nodes != 0) return t5d::ProveBudget::nodes(nodes);
  return t5d::ProveBudget::time(std::chrono::nanoseconds(static_cast<long long>(ms * 1e6)));
}

std::string option_text(const std::vector<std::pair<std::string, std::string>>& options) {
  std::string out;
  for (const auto& [k, v] : options) out += (out.empty() ? "" : ", ") + k + "=" + v;
  return out;
}

std::string option_json(const std::vector<std::pair<std::string, std::string>>& options) {
  std::string out = "{";
  for (std::size_t i = 0; i < options.size(); ++i) {
    out += (i ? "," : "") + json_string(options[i].first) + ":" + json_string(options[i].second);
  }
  return out + "}";
}

// ---- results --------------------------------------------------------------------------------

struct Result {
  bool measured = false;
  std::string skipped;
  int boards = 0;
  std::string fingerprint;
  // Guard
  std::string guard_skip;  // why the chosen action is not counted
  std::optional<t5d::GuardResult> guard;
  // Hunt
  std::optional<t5d::HuntResult> hunt;
  // The cross-checks (t5d/certificate.hpp), outside the timed runs.
  std::optional<t5d::CertificateCheck> guard_check;
  std::optional<t5d::CertificateCheck> hunt_check;
  int truth_row = -1;
};

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
  const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(v.size())));
  return v[std::min(v.size() - 1, rank == 0 ? 0 : rank - 1)];
}

// Whether `action` is legal at `s`: each move pseudo-legal where made, the
// obligations met, no royal left capturable.
bool legal_action(const t5d::GameState& s, const std::vector<t5d::Move>& action) {
  t5d::GameState t = s;
  t5d::MoveList scratch;
  for (const t5d::Move& m : action) {
    if (!t.may_move() || !t5d::is_pseudo_legal(t, m, scratch)) return false;
    t.apply(m);
  }
  return t.obligations_met() && t5d::royals_safe(t);
}

}  // namespace

int main(int argc, char** argv) {
  namespace pp = t5d::tools::sixbpp;
  Options o;
  try {
    o = parse(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-scan6bpp: %s\n", e.what());
    usage(2);
  }
  const std::string cpu = cpu_model();
  const unsigned logical = std::thread::hardware_concurrency();

  // -- positions and the truth -------------------------------------------------------------------
  std::vector<t5d::tools::RecordRef> refs;
  try {
    refs = t5d::tools::record_refs(o.direct, o.lists, o.record_root);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-scan6bpp: %s\n", e.what());
    return 1;
  }
  const t5d::tools::SixbRoots roots = t5d::tools::sixb_roots(refs, o.rules, 1, o.max_roots);
  const std::size_t n = roots.roots.size();
  std::vector<Result> results(n);
  for (std::size_t i = 0; i < n; ++i) {
    results[i].boards = static_cast<int>(roots.roots[i].state.multiverse().board_count());
    results[i].fingerprint = t5d::fingerprint(roots.roots[i].state.multiverse()).fingerprint;
  }
  std::optional<t5d::tools::HuntTruth> truth;
  std::string code_line;
  if (!o.no_truth) {
    try {
      truth = t5d::tools::load_hunt_truth(o.truth);
      t5d::tools::check_records(truth->provenance, t5d::tools::record_contents(refs));
      if (truth->provenance.rules != t5d::tools::absolute_rules_text(o.rules)) {
        throw std::runtime_error("its rules (" + truth->provenance.rules + ") are not these (" +
                                 t5d::tools::absolute_rules_text(o.rules) + ")");
      }
      if (std::stoi(truth->value("depth")) != o.hunt_depth) throw std::runtime_error("its depth is not --hunt-depth");
      if (o.max_roots == 0 && truth->roots.size() != n) {
        throw std::runtime_error("it holds " + std::to_string(truth->roots.size()) + " roots, the records give " +
                                 std::to_string(n));
      }
      for (std::size_t i = 0; i < n; ++i) {
        const t5d::tools::SixbRoot& r = roots.roots[i];
        const t5d::tools::HuntTruthRoot* row = i < truth->roots.size() ? &truth->roots[i] : nullptr;
        if (row == nullptr || row->record != r.content || row->index != r.index ||
            row->fingerprint != results[i].fingerprint) {
          throw std::runtime_error("root " + std::to_string(i) + " (" + r.record + " #" + std::to_string(r.index) +
                                   ") is not the truth's");
        }
        results[i].truth_row = static_cast<int>(i);
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "t5d-scan6bpp: the ground truth is refused: %s\n", e.what());
      return 1;
    }
    const std::string& code = truth->provenance.code;
    code_line = code == T5D_CODE_SIGNATURE
                    ? "the same code signature as this build's"
                    : code.rfind("unrecorded", 0) == 0
                          ? "its code signature " + code + "; this build's " T5D_CODE_SIGNATURE
                          : "recorded by OTHER CODE (" + code + "; this build's " T5D_CODE_SIGNATURE
                            ") - the rules engine, the prover or the recorder changed since: its verdicts stand "
                            "as recorded; re-record it if a change could alter one";
  }

  // -- which roots are measured ------------------------------------------------------------------
  for (std::size_t i = 0; i < n; ++i) {
    const t5d::GameState& s = roots.roots[i].state;
    Result& r = results[i];
    if (r.boards < o.min_boards || r.boards > o.max_boards) {
      r.skipped = "outside --min-boards/--max-boards";
      continue;
    }
    if (roots.roots[i].synthetic) {
      r.skipped = "a synthetic record";
      continue;
    }
    if (t5d::royal_capture_exists(s.multiverse(), s.rules(), s.to_move(), s.inactive_timelines())) {
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
    r.measured = true;
    const std::optional<std::vector<t5d::Move>>& next = roots.roots[i].next_action;
    if (!next) {
      r.guard_skip = roots.roots[i].no_next;
    } else if (!legal_action(s, *next)) {
      r.guard_skip = "its record's next action is not legal (C4)";
    }
  }

  // -- Guard, one root at a time -------------------------------------------------------------------
  const CpuTimes g0 = cpu_times();
  const Clock::time_point g_wall0 = Clock::now();
  t5d::GuardOptions go;
  go.threads = o.threads;
  go.max_depth = o.guard_depth;
  go.prover.check_filter = o.check_filter;
  go.prover.attacker_probe = o.attacker_probe;
  go.prover.record_certificate = o.crosscheck;
  std::size_t done = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!results[i].measured || !results[i].guard_skip.empty()) continue;
    results[i].guard = t5d::guard(roots.roots[i].state, *roots.roots[i].next_action, budget(o.guard_ms, o.guard_nodes), go);
    if (++done % 100 == 0) std::fprintf(stderr, "t5d-scan6bpp: Guard %zu roots\n", done);
  }
  const double g_wall = std::chrono::duration<double>(Clock::now() - g_wall0).count();
  const double g_busy = busy_between(g0, cpu_times());

  // -- Hunt, one root at a time ------------------------------------------------------------------
  const CpuTimes h0 = cpu_times();
  const Clock::time_point h_wall0 = Clock::now();
  t5d::HuntOptions ho;
  ho.threads = o.threads;
  ho.prover.depth = o.hunt_depth;
  ho.prover.check_filter = o.check_filter;
  ho.prover.attacker_probe = o.attacker_probe;
  ho.prover.record_certificate = o.crosscheck;
  done = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!results[i].measured) continue;
    t5d::Hunt hunt(ho);
    results[i].hunt = hunt.hunt(roots.roots[i].state, budget(o.hunt_ms, o.hunt_nodes));
    if (++done % 100 == 0) std::fprintf(stderr, "t5d-scan6bpp: Hunt %zu roots\n", done);
  }
  const double h_wall = std::chrono::duration<double>(Clock::now() - h_wall0).count();
  const double h_busy = busy_between(h0, cpu_times());

  // -- the cross-checks: every refutation and every mate against E4 alone ----------------------------
  t5d::CertificateCheckOptions co;
  co.max_positions = o.check_positions;
  if (o.crosscheck) {
    for (std::size_t i = 0; i < n; ++i) {
      Result& r = results[i];
      if (r.guard && r.guard->refuted && r.guard->certificate) {
        t5d::GameState after = roots.roots[i].state;
        for (const t5d::Move& m : *roots.roots[i].next_action) after.apply(m);
        after.submit(t5d::CheckRuling::none());
        r.guard_check = t5d::check_certificate(*r.guard->certificate, after, 2 * r.guard->refutation_depth - 1, co);
        r.guard->certificate.reset();
      }
      if (r.hunt && r.hunt->mate && r.hunt->certificate) {
        r.hunt_check = t5d::check_certificate(*r.hunt->certificate, roots.roots[i].state, r.hunt->proof.distance, co);
        r.hunt->certificate.reset();
      }
    }
  }

  // -- the counting (sixbpp.hpp): only confirmed proofs count -------------------------------------------
  std::vector<pp::RootOutcome> outcomes;
  for (std::size_t i = 0; i < n; ++i) {
    const Result& r = results[i];
    if (!r.measured) continue;
    pp::RootOutcome x;
    x.boards = r.boards;
    if (r.guard) {
      x.guard = true;
      x.completed_depth = r.guard->completed_depth;
      x.refuted = r.guard->refuted;
      x.refutation_depth = r.guard->refutation_depth;
      if (r.guard->refuted) x.guard_check = r.guard_check ? pp::check_of(*r.guard_check) : pp::Check::Unconfirmed;
    }
    if (r.hunt) {
      x.hunt = true;
      x.hunt_mate = r.hunt->mate;
      if (r.hunt->mate) x.hunt_check = r.hunt_check ? pp::check_of(*r.hunt_check) : pp::Check::Unconfirmed;
    }
    x.truth_mate = truth && r.truth_row >= 0 && truth->roots[static_cast<std::size_t>(r.truth_row)].mate();
    outcomes.push_back(x);
  }
  const pp::Tallies tallies = pp::tally(outcomes);
  const std::vector<std::string> away = departures(o);
  const bool shipped = away.empty();

  // -- the figures ----------------------------------------------------------------------------------
  std::string out;
  const auto line = [&](const std::string& s) { out += s + "\n"; };
  const auto guard_budget = o.guard_nodes != 0 ? std::to_string(o.guard_nodes) + " nodes" : fmt(o.guard_ms, 0) + " ms";
  const auto hunt_budget = o.hunt_nodes != 0 ? std::to_string(o.hunt_nodes) + " nodes" : fmt(o.hunt_ms, 0) + " ms";
  line("t5d-scan6bpp " T5D_VERSION_STRING " - experiment 6b'' (design spec 15.6, CHECKPOINT 4; register M17)");
  line("");
  line("CONDITIONS");
  line("  machine: " + cpu + ", " + std::to_string(logical) + " logical processors; busy " + fmt(100 * g_busy) +
       "% during Guard, " + fmt(100 * h_busy) + "% during Hunt (all processes; the tool's own threads included)");
  line("  build: " + compiler_text() + ", " + T5D_BUILD_TYPE + ", flags \"" + T5D_CXX_FLAGS + "\"; code " +
       T5D_CODE_SIGNATURE);
  line("  rules: " + t5d::tools::ruleset_text(o.rules) +
       (t5d::decomposes(o.rules) ? " - turns decompose into semimoves" : " - turns do NOT decompose: E4's listings"));
  line("  configuration: " + std::string(shipped ? "M17's - Guard and Hunt as they ship, 5 threads, 50 ms"
                                                 : "AN INFORMATION RUN, not M17's: no verdict is read from it (" +
                                                       [&] {
                                                         std::string t;
                                                         for (const std::string& d : away) t += (t.empty() ? "" : ", ") + d;
                                                         return t;
                                                       }() + ")"));
  line("  Guard: the record's next action at each root; iterative deepening from depth 1 to " +
       std::to_string(o.guard_depth) + " (the opponent's actions), every opponent action (the literal class), " +
       guard_budget + " per root; wall " + fmt(g_wall, 1) + " s");
  line("    options: " + option_text(pp::guard_options(go)));
  line("  Hunt: df-pn prove-only, " + std::to_string(o.hunt_depth) + " half-turns, " + hunt_budget +
       " per root; wall " + fmt(h_wall, 1) + " s");
  line("    options: " + option_text(pp::hunt_options(ho)));
  if (truth) {
    line("  ground truth: " + o.truth + "/" + t5d::tools::kHuntTruthFile + ", sha256 " + truth->sha256 +
         " (it and its provenance match their MANIFEST; the records given are its records, in its order); engine " +
         truth->value("engine") + "; " + code_line + "; " + truth->value("search"));
  } else {
    line("  ground truth: none (--no-truth): Hunt's mates counted, no verdict");
  }
  line("  positions: every turn start of " + std::to_string(refs.size()) + " record(s); " +
       std::to_string(roots.duplicates) + " duplicate root(s) dropped");
  for (const std::string& nt : roots.notes) line("    " + nt);
  {
    std::map<std::string, std::size_t> skipped;
    std::map<std::string, std::size_t> guard_skipped;
    std::size_t nr = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!results[i].measured) {
        skipped[results[i].skipped]++;
        continue;
      }
      ++nr;
      if (!results[i].guard_skip.empty()) guard_skipped[results[i].guard_skip]++;
    }
    line("  measured: " + std::to_string(nr) + " roots");
    for (const auto& [why, k] : skipped) line("    not measured: " + std::to_string(k) + " " + why);
    for (const auto& [why, k] : guard_skipped) line("    Guard not counted: " + std::to_string(k) + " - " + why);
  }
  line("");

  // Guard
  struct GuardWork {
    std::vector<double> nodes;
    std::uint64_t check_cut = 0, mate_tests = 0, not_in_check = 0, attacker_tests = 0, listings = 0, memory = 0;
  };
  GuardWork gw[pp::kRanges + 1];
  for (std::size_t i = 0; i < n; ++i) {
    if (!results[i].guard) continue;
    const t5d::GuardResult& g = *results[i].guard;
    for (const int k : {pp::range_of(results[i].boards), pp::kRanges}) {
      GuardWork& w = gw[k];
      w.nodes.push_back(static_cast<double>(g.nodes));
      w.check_cut += g.stats.check_cut;
      w.mate_tests += g.stats.mate_tests;
      w.not_in_check += g.stats.not_in_check;
      w.attacker_tests += g.stats.attacker_tests;
      w.listings += g.stats.listings;
      if (g.stats.stopped_by_memory) ++w.memory;
    }
  }
  const std::string pooled = "pooled (not a verdict, M12)";
  line("GUARD - depth 1 completes (safe to depth 1, or refuted at depth 1 with E4's confirmation) within " +
       guard_budget + " at " + std::to_string(o.threads) + " threads; pass: >= 95% of a range's roots");
  for (int k = 0; k <= pp::kRanges; ++k) {
    const pp::GuardTally& s = tallies.guard[k];
    if (s.roots == 0) continue;
    const bool is_pooled = k == pp::kRanges;
    std::string dist;
    for (const auto& [d, c] : s.completed) dist += (dist.empty() ? "" : ", ") + std::to_string(d) + ": " + std::to_string(c);
    line(std::string("  ") + (is_pooled ? pooled : pp::range_name(k)) + ": depth 1 completed at " +
         pct(s.depth_one, s.roots) + " of " + std::to_string(s.roots) + " roots" +
         (s.unconfirmed > 0 ? " (" + std::to_string(s.unconfirmed) + " more refuted at depth 1 but NOT CONFIRMED by "
                              "E4 - not counted)"
                            : std::string()) +
         "; completed depth (roots) " + dist + "; " + std::to_string(s.refuted) + " refuted (" +
         std::to_string(s.refuted_one) + " at depth 1); median " + fmt(quantile(gw[k].nodes, 0.5), 0) + " nodes" +
         (is_pooled ? std::string()
                    : std::string(" -> ") + (pp::guard_passes(s.depth_one, s.roots) ? "PASS" : "FAIL")));
  }
  line("  the work: (every thread's) partial actions cut by the check filter / our turns decided not in check / "
       "mate tests / attacker turns probed / listings / searches stopped by the tree's memory, per range:");
  for (int k = 0; k < pp::kRanges; ++k) {
    if (tallies.guard[k].roots == 0) continue;
    const GuardWork& w = gw[k];
    line(std::string("    ") + pp::range_name(k) + ": " + std::to_string(w.check_cut) + " / " +
         std::to_string(w.not_in_check) + " / " + std::to_string(w.mate_tests) + " / " +
         std::to_string(w.attacker_tests) + " / " + std::to_string(w.listings) + " / " + std::to_string(w.memory));
  }
  line("");

  // Hunt
  std::vector<double> hunt_nodes[pp::kRanges + 1];
  for (std::size_t i = 0; i < n; ++i) {
    if (!results[i].hunt) continue;
    for (const int k : {pp::range_of(results[i].boards), pp::kRanges}) {
      hunt_nodes[k].push_back(static_cast<double>(results[i].hunt->stats.nodes));
    }
  }
  line("HUNT - forced mates within " + std::to_string(o.hunt_depth) + " half-turns found within " + hunt_budget +
       " at " + std::to_string(o.threads) + " threads, of the ground truth's, counted when E4 confirms them; pass: "
       ">= 50% of a range's truth mates");
  for (int k = 0; k <= pp::kRanges; ++k) {
    const pp::HuntTally& s = tallies.hunt[k];
    if (s.roots == 0) continue;
    const bool is_pooled = k == pp::kRanges;
    std::string verdict;
    if (!is_pooled && truth) {
      verdict = s.truth == 0 ? " -> NO VERDICT (no mate in the truth)"
                             : pp::hunt_passes(s.found, s.truth) ? " -> PASS" : " -> FAIL";
      if (pp::too_few_to_read(tallies, k)) {
        verdict += " (TOO FEW TO READ: " + std::to_string(s.truth) + " truth mate(s), under " +
                   std::to_string(pp::kFewTruthMates) + ")";
      }
    }
    line(std::string("  ") + (is_pooled ? pooled : pp::range_name(k)) + ": " + std::to_string(s.roots) + " roots; " +
         (truth ? "the truth's mates " + std::to_string(s.truth) + ", Hunt found " + std::to_string(s.found) + " (" +
                      pct(s.found, s.truth) + ")" +
                      (s.found_unconfirmed > 0
                           ? " and " + std::to_string(s.found_unconfirmed) + " more NOT CONFIRMED by E4 (not counted)"
                           : std::string()) +
                      "; "
                : std::string()) +
         "Hunt's mates " + std::to_string(s.mates) +
         (s.unconfirmed > 0 ? " (" + std::to_string(s.unconfirmed) + " not confirmed)" : std::string()) +
         (truth && s.extra > 0 ? " - " + std::to_string(s.extra) + " NOT IN THE TRUTH (an incomplete truth)" : "") +
         "; median " + fmt(quantile(hunt_nodes[k], 0.5), 0) + " nodes" + verdict);
  }
  line("");

  if (o.crosscheck) {
    std::map<std::string, std::size_t> gc;
    std::map<std::string, std::size_t> hc;
    std::vector<std::string> notes;
    for (std::size_t i = 0; i < n; ++i) {
      const Result& r = results[i];
      const std::string where = roots.roots[i].record + " #" + std::to_string(roots.roots[i].index);
      if (r.guard_check) {
        gc[t5d::certificate_verdict_name(r.guard_check->verdict)]++;
        if (r.guard_check->verdict != t5d::CertificateVerdict::Confirmed) {
          notes.push_back(std::string(r.guard_check->verdict == t5d::CertificateVerdict::Refuted ? "REFUTED" : "not confirmed") +
                          ": Guard " + where + ": " + t5d::certificate_verdict_name(r.guard_check->verdict) + " - " +
                          r.guard_check->why);
        }
      }
      if (r.hunt_check) {
        hc[t5d::certificate_verdict_name(r.hunt_check->verdict)]++;
        if (r.hunt_check->verdict != t5d::CertificateVerdict::Confirmed) {
          notes.push_back(std::string(r.hunt_check->verdict == t5d::CertificateVerdict::Refuted ? "REFUTED" : "not confirmed") +
                          ": Hunt " + where + ": " + t5d::certificate_verdict_name(r.hunt_check->verdict) + " - " +
                          r.hunt_check->why);
        }
      }
    }
    const auto row = [](const std::map<std::string, std::size_t>& m) {
      std::string text;
      for (const auto& [k, v] : m) text += (text.empty() ? "" : ", ") + std::to_string(v) + " " + k;
      return text.empty() ? std::string("none") : text;
    };
    line("CROSS-CHECKS - every refutation and mate's certificate against E4 alone (t5d/certificate.hpp; at most " +
         std::to_string(o.check_positions) + " defender positions each): Guard's refutations " + row(gc) +
         "; Hunt's mates " + row(hc));
    for (const std::string& r : notes) line("  " + r);
    line("");
  }

  // The verdicts.
  line("SUMMARY - per board-count range (M17); pooled figures are information, not a verdict");
  if (!shipped) line("  AN INFORMATION RUN (not M17's configuration): the rows below are not the verdict.");
  if (!o.crosscheck) line("  NO VERDICT: without the cross-check no proof is confirmed, and only confirmed proofs count.");
  if (tallies.refuted_proofs > 0) {
    line("  VERDICT WITHHELD: " + std::to_string(tallies.refuted_proofs) +
         " proof(s) REFUTED by E4 alone - the prover is unsound somewhere; no figure here can be trusted until it "
         "is fixed.");
  }
  for (int k = 0; k < pp::kRanges; ++k) {
    std::string row = std::string("  ") + pp::range_name(k) + ": ";
    switch (pp::guard_verdict(tallies, k, o.crosscheck)) {
      case pp::Verdict::Adopt: row += "ADOPT GUARD"; break;
      case pp::Verdict::NotAdopted: row += "Guard not adopted - spec 13: the chosen move goes unguarded in that range"; break;
      case pp::Verdict::NoRoots: row += "Guard - no roots"; break;
      default: row += std::string("Guard - ") + pp::verdict_name(pp::guard_verdict(tallies, k, o.crosscheck)); break;
    }
    row += "; ";
    switch (pp::hunt_verdict(tallies, k, truth.has_value(), o.crosscheck)) {
      case pp::Verdict::Adopt: row += "ADOPT HUNT"; break;
      case pp::Verdict::NotAdopted: row += "Hunt not adopted - spec 13: proofs there come only from the search's solver"; break;
      default:
        row += std::string("Hunt - ") + pp::verdict_name(pp::hunt_verdict(tallies, k, truth.has_value(), o.crosscheck));
        break;
    }
    if (pp::too_few_to_read(tallies, k)) row += " (Hunt's verdict: too few truth mates to read)";
    line(row);
  }
  line("  No outcome stops anything (M17): the layer's floor - exact rules, the search's solver, the audit lane, the "
       "league - stands either way.");

  if (!o.summary.empty()) {
    std::FILE* f = std::fopen(o.summary.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "t5d-scan6bpp: cannot write %s\n", o.summary.c_str());
      return 1;
    }
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
  }
  std::fwrite(out.data(), 1, out.size(), stdout);

  if (!o.json.empty()) {
    std::string j = "{\"tool\":\"t5d-scan6bpp\",\"version\":\"" T5D_VERSION_STRING "\",\"code\":\"" T5D_CODE_SIGNATURE "\"";
    j += ",\"machine\":{\"cpu\":" + json_string(cpu) + ",\"logical_processors\":" + std::to_string(logical) +
         ",\"busy_guard\":" + fmt(g_busy, 4) + ",\"busy_hunt\":" + fmt(h_busy, 4) + "}";
    j += ",\"build\":{\"compiler\":" + json_string(compiler_text()) + ",\"build_type\":" + json_string(T5D_BUILD_TYPE) +
         ",\"flags\":" + json_string(T5D_CXX_FLAGS) + "}";
    j += ",\"rules\":" + json_string(t5d::tools::ruleset_text(o.rules)) +
         ",\"rules_absolute\":" + json_string(t5d::tools::absolute_rules_text(o.rules));
    j += ",\"shipped_configuration\":" + std::string(shipped ? "true" : "false");
    j += ",\"threads\":" + std::to_string(o.threads) + ",\"guard_ms\":" + fmt(o.guard_ms, 1) +
         ",\"hunt_ms\":" + fmt(o.hunt_ms, 1) + ",\"guard_nodes\":" + std::to_string(o.guard_nodes) +
         ",\"hunt_nodes\":" + std::to_string(o.hunt_nodes) + ",\"guard_depth\":" + std::to_string(o.guard_depth) +
         ",\"hunt_depth\":" + std::to_string(o.hunt_depth) + ",\"crosscheck\":" + (o.crosscheck ? "true" : "false") +
         ",\"check_positions\":" + std::to_string(o.check_positions);
    j += ",\"guard_options\":" + option_json(pp::guard_options(go));
    j += ",\"hunt_options\":" + option_json(pp::hunt_options(ho));
    j += ",\"truth_sha256\":" + json_string(truth ? truth->sha256 : std::string()) +
         ",\"truth_code\":" + json_string(truth ? truth->provenance.code : std::string());
    j += ",\"refuted_proofs\":" + std::to_string(tallies.refuted_proofs);
    j += ",\"ranges\":[\"< 50\",\"50-99\",\"100-199\",\"200+\",\"pooled\"],\"guard\":[";
    for (int k = 0; k <= pp::kRanges; ++k) {
      const pp::GuardTally& g = tallies.guard[k];
      j += std::string(k ? "," : "") + "{\"n\":" + std::to_string(g.roots) + ",\"depth_one\":" +
           std::to_string(g.depth_one) + ",\"unconfirmed\":" + std::to_string(g.unconfirmed) + ",\"refuted\":" +
           std::to_string(g.refuted) + ",\"refuted_one\":" + std::to_string(g.refuted_one) + ",\"verdict\":" +
           json_string(k < pp::kRanges ? pp::verdict_name(pp::guard_verdict(tallies, k, o.crosscheck)) : "pooled") + "}";
    }
    j += "],\"hunt\":[";
    for (int k = 0; k <= pp::kRanges; ++k) {
      const pp::HuntTally& h = tallies.hunt[k];
      j += std::string(k ? "," : "") + "{\"n\":" + std::to_string(h.roots) + ",\"truth\":" + std::to_string(h.truth) +
           ",\"found\":" + std::to_string(h.found) + ",\"found_unconfirmed\":" + std::to_string(h.found_unconfirmed) +
           ",\"mates\":" + std::to_string(h.mates) + ",\"unconfirmed\":" + std::to_string(h.unconfirmed) +
           ",\"extra\":" + std::to_string(h.extra) + ",\"too_few_to_read\":" +
           (k < pp::kRanges && pp::too_few_to_read(tallies, k) ? "true" : "false") + ",\"verdict\":" +
           json_string(k < pp::kRanges ? pp::verdict_name(pp::hunt_verdict(tallies, k, truth.has_value(), o.crosscheck))
                                       : "pooled") +
           "}";
    }
    j += "]";
    if (o.per_position) {
      j += ",\"positions\":[";
      for (std::size_t i = 0; i < n; ++i) {
        const Result& r = results[i];
        j += std::string(i ? "," : "") + "{\"record\":" + json_string(roots.roots[i].record) + ",\"index\":" +
             std::to_string(roots.roots[i].index) + ",\"fingerprint\":\"" + r.fingerprint + "\",\"boards\":" +
             std::to_string(r.boards) + ",\"measured\":" + (r.measured ? "true" : "false");
        if (!r.measured) {
          j += ",\"skipped\":" + json_string(r.skipped) + "}";
          continue;
        }
        if (!r.guard_skip.empty()) j += ",\"guard_skipped\":" + json_string(r.guard_skip);
        if (r.guard) {
          j += ",\"guard\":{\"completed_depth\":" + std::to_string(r.guard->completed_depth) + ",\"refuted\":" +
               (r.guard->refuted ? "true" : "false") + ",\"refutation_depth\":" +
               std::to_string(r.guard->refutation_depth) + ",\"nodes\":" + std::to_string(r.guard->nodes) +
               ",\"seconds\":" + fmt(r.guard->seconds, 4) + ",\"check\":" +
               json_string(r.guard_check ? t5d::certificate_verdict_name(r.guard_check->verdict) : "-") + "}";
        }
        if (r.hunt) {
          j += ",\"hunt\":{\"mate\":" + std::string(r.hunt->mate ? "true" : "false") + ",\"proof\":" +
               json_string(r.hunt->mate ? t5d::to_string(r.hunt->proof) : std::string()) + ",\"nodes\":" +
               std::to_string(r.hunt->stats.nodes) + ",\"check\":" +
               json_string(r.hunt_check ? t5d::certificate_verdict_name(r.hunt_check->verdict) : "-") + "}";
        }
        if (truth && r.truth_row >= 0) {
          j += ",\"truth\":" + json_string(truth->roots[static_cast<std::size_t>(r.truth_row)].verdict);
        }
        j += "}";
      }
      j += "]";
    }
    j += ",\"summary\":" + json_string(out) + "}\n";
    std::FILE* f = std::fopen(o.json.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "t5d-scan6bpp: cannot write %s\n", o.json.c_str());
      return 1;
    }
    std::fwrite(j.data(), 1, j.size(), f);
    std::fclose(f);
  }
  return 0;
}
