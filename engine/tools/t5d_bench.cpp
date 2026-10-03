// t5d-bench: expansion throughput (design spec §15.6, "Movegen throughput";
// register M9). How many positions per second expand() (t5d/expand.hpp) gets
// through, on one thread and many, in several configurations - and how its
// cost changes with the length of the history (§15.6's flatness check).
//
//   t5d-bench [options] [RECORD ...]          (--help lists the options)
//
// POSITIONS. SYNTHETIC classes (tools/synthetic.hpp: seeded random play, for
// throughput only, never the |E| verdict) at §15.6's assumed N and T - 100,
// 160, 240, 400, 600 boards with 6, 7, 8, 10, 12 active timelines - `games`
// games each, a game's seed a function of (seed, N, T, game) alone; and every
// action start of any RECORD given, by N bucket (records with a "Synthetic"
// header in buckets of their own, marked synthetic). The ruleset options apply
// to all of them: the synthetic games are played under them too.
//
// WHAT IS TIMED. Cycling a handful of class positions lets the branch
// predictor learn them, which flatters the rate; that "loop" is reported apart,
// as an upper bound. The rate proper is over a search-like STREAM: the
// one-move children of the class positions (the positions a search expands
// next), `children` per position per worker, each worker's its own - disjoint
// from the other workers' wherever a position has the moves - rebuilt on
// boards of its own, and started at a different game, so that workers running
// at once (SMT siblings included) expand different games.
//
// METHOD. A pool of worker threads, pinned one per physical core first, then
// on the cores' second logical processors (--no-pin: unpinned), each with its
// own Expansion. A measurement - (class, configuration, threads, stream or
// loop) - is taken in samples of a fixed length, and the samples are
// INTERLEAVED: round after round over every measurement, after a warm-up
// round, so that drifts in the machine's load spread over all of them and the
// 1-thread base of every scaling figure is measured in the same rounds as the
// rest. Each worker counts its expansions to a shared steady_clock deadline
// and reads its own thread CPU time; the machine's load (the busy fraction of
// all its processors) is read around every sample. Reported: the wall rate
// (expansions/s, total and per thread), the thread CPU time per expansion -
// the figure the machine's other load disturbs least, as time spent
// descheduled is not counted - the scaling against the interleaved 1-thread
// rate, and the load. Every result feeds a printed sink, so nothing is
// optimised away. The output names the compiler, the flags, the CPU, its
// cores and logical processors, and the thread counts. Moves are PSEUDO-LEGAL
// in every configuration but the "+legal" ones (--legal: the legal moves only,
// ExpandOptions::legal_moves under the --legal-budget - provisional, R1;
// build-log spec issues 28 and 46).
//
// LEGAL PROFILE (--legal-profile, instead of the throughput): every class
// position and its one-move children expanded one at a time, single-threaded
// - pseudo-legal, then legal exact, under the budget, and fast - each timed;
// per class the median, p95 and worst per expansion, and the share of the
// moves each tier of t5d/legal.hpp settled.
//
// FLATNESS (§15.6: "Scaling at fixed |E| - flat in N - <= 10% growth from 100
// to 600 boards"). Games with their active timelines held at T, taken at every
// action start near each level's N; the positions are the one-move children
// along the action that follows. Playing longer does not hold |E| fixed (nor
// the moves, nor the material: the pseudo-legal travel into a longer past
// grows by the rules), so three series are timed, on one thread, interleaved,
// every growth taken within a round and reported over the rounds: RAW (every
// position; its growth beside what changed, and per generated move), MATCHED
// (anchors at the first level, each paired at every level with its nearest
// position in moves, |E(8)| and the mover's pieces - "at fixed |E|"), and
// NO_MOVES (the action's moves played, so that no move may be made - under
// Q4's Guide-literal reading, the one under which a played action leaves no
// move to make (R8's default allows moves until submission): the
// history-length term itself, everything but move generation, per board).

#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "synthetic.hpp"
#include "t5d/expand.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/gate.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "tool_support.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#define T5D_BENCH_TSC 1
#endif
#else
#include <sched.h>
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

using Clock = std::chrono::steady_clock;
using t5d::gate::json_string;

// ---- the machine and the build -----------------------------------------------------------

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
#elif defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_VER);
#else
  return "unknown";
#endif
}

std::string os_text() {
#if defined(_WIN32)
  return "Windows";
#elif defined(__linux__)
  return "Linux";
#elif defined(__APPLE__)
  return "macOS";
#else
  return "unknown";
#endif
}

// The CPU's model, as the operating system names it.
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

// The physical cores, each as its logical processors (ascending), ordered by
// their first logical processor; empty if the OS does not say (then no thread
// is pinned). Windows: processor group 0 only.
std::vector<std::vector<int>> physical_cores() {
  std::vector<std::vector<int>> cores;
#if defined(_WIN32)
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
  if (length == 0) return cores;
  std::vector<unsigned char> buffer(length);
  auto* first = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &length)) return cores;
  for (DWORD offset = 0; offset < length;) {
    const auto* info =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
    if (info->Size == 0) break;
    if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount >= 1 &&
        info->Processor.GroupMask[0].Group == 0) {
      std::vector<int> cpus;
      const KAFFINITY mask = info->Processor.GroupMask[0].Mask;
      for (int i = 0; i < 64; ++i) {
        if (((mask >> i) & 1u) != 0) cpus.push_back(i);
      }
      if (!cpus.empty()) cores.push_back(cpus);
    }
    offset += info->Size;
  }
#elif defined(__linux__)
  const unsigned n = std::thread::hardware_concurrency();
  std::map<std::pair<int, int>, std::vector<int>> by_core;
  for (unsigned cpu = 0; cpu < n; ++cpu) {
    int ids[2] = {-1, -1};
    const char* names[2] = {"physical_package_id", "core_id"};
    for (int i = 0; i < 2; ++i) {
      char path[160];
      std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%u/topology/%s", cpu, names[i]);
      std::FILE* f = std::fopen(path, "r");
      if (f == nullptr) return {};
      const int got = std::fscanf(f, "%d", &ids[i]);
      std::fclose(f);
      if (got != 1) return {};
    }
    by_core[{ids[0], ids[1]}].push_back(static_cast<int>(cpu));
  }
  for (auto& [key, cpus] : by_core) cores.push_back(cpus);
#endif
  std::sort(cores.begin(), cores.end());
  return cores;
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

// ---- this thread's CPU time, and the machine's load -------------------------------------

#if defined(T5D_BENCH_TSC)
double g_tsc_hz = 0;  // the time-stamp counter's rate, measured at start-up
#endif

// Measures what the thread clock needs; returns how the clock works.
std::string calibrate_thread_clock(double& check) {
  check = 0;
#if defined(T5D_BENCH_TSC)
  // QueryThreadCycleTime counts at the time-stamp counter's rate on current
  // processors; the TSC's rate against steady_clock converts it to seconds.
  const Clock::time_point t0 = Clock::now();
  const unsigned long long r0 = __rdtsc();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const unsigned long long r1 = __rdtsc();
  const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  g_tsc_hz = static_cast<double>(r1 - r0) / seconds;
  // The assumption, checked: thread cycles against TSC ticks while the thread
  // spins (descheduling only lowers the ratio, so the best of five).
  for (int trial = 0; trial < 5; ++trial) {
    ULONG64 q0 = 0;
    ULONG64 q1 = 0;
    QueryThreadCycleTime(GetCurrentThread(), &q0);
    const unsigned long long s0 = __rdtsc();
    const Clock::time_point until = Clock::now() + std::chrono::milliseconds(20);
    while (Clock::now() < until) {
    }
    QueryThreadCycleTime(GetCurrentThread(), &q1);
    const unsigned long long s1 = __rdtsc();
    check = std::max(check, static_cast<double>(q1 - q0) / static_cast<double>(s1 - s0));
  }
  return "QueryThreadCycleTime at the TSC rate (" +
         std::to_string(static_cast<long long>(g_tsc_hz / 1e6)) + " MHz, measured)";
#elif defined(_WIN32)
  check = 1;
  return "GetThreadTimes (timer-tick resolution)";
#else
  check = 1;
  return "clock_gettime(CLOCK_THREAD_CPUTIME_ID)";
#endif
}

double thread_cpu_seconds() noexcept {
#if defined(T5D_BENCH_TSC)
  ULONG64 cycles = 0;
  QueryThreadCycleTime(GetCurrentThread(), &cycles);
  return static_cast<double>(cycles) / g_tsc_hz;
#elif defined(_WIN32)
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

// The machine's processor time so far: idle and total (any unit).
struct CpuTimes {
  std::uint64_t idle = 0;
  std::uint64_t total = 0;
};

CpuTimes cpu_times() noexcept {
  CpuTimes t;
#if defined(_WIN32)
  FILETIME idle, kernel, user;
  if (!GetSystemTimes(&idle, &kernel, &user)) return t;
  const auto u64 = [](FILETIME f) {
    return (static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
  };
  t.idle = u64(idle);
  t.total = u64(kernel) + u64(user);  // kernel time includes idle time
#elif defined(__linux__)
  std::FILE* f = std::fopen("/proc/stat", "r");
  if (f == nullptr) return t;
  unsigned long long v[8] = {};
  const int n = std::fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2],
                            &v[3], &v[4], &v[5], &v[6], &v[7]);
  std::fclose(f);
  if (n < 4) return t;
  for (const unsigned long long x : v) t.total += x;
  t.idle = v[3] + v[4];  // idle and iowait
#endif
  return t;
}

// The busy fraction of every processor between two readings; -1 if unknown.
double busy_between(const CpuTimes& a, const CpuTimes& b) noexcept {
  if (b.total <= a.total || b.idle < a.idle) return -1;
  return 1.0 - static_cast<double>(b.idle - a.idle) / static_cast<double>(b.total - a.total);
}

// ---- small helpers ---------------------------------------------------------------------------

std::uint64_t fnv1a(std::string_view text) noexcept {
  std::uint64_t h = 1469598103934665603ull;
  for (const char c : text) {
    h ^= static_cast<unsigned char>(c);
    h *= 1099511628211ull;
  }
  return h;
}

std::string fixed(double value, int digits) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", digits, value);
  return buf;
}

// 1234567 -> "1,234,567".
std::string grouped(double value) {
  const long long v = std::llround(value);
  std::string digits = std::to_string(v < 0 ? -v : v);
  std::string out;
  for (std::size_t i = 0; i < digits.size(); ++i) {
    if (i != 0 && (digits.size() - i) % 3 == 0) out += ',';
    out += digits[i];
  }
  return v < 0 ? "-" + out : out;
}

std::string percent(double fraction) {
  if (fraction < 0) return "?";
  return fixed(100 * fraction, 0) + "%";
}

std::string signed_percent(double fraction) {
  return (fraction >= 0 ? "+" : "") + fixed(100 * fraction, 0) + "%";
}

// ---- statistics ------------------------------------------------------------------------------

struct Stats {
  double median = 0;
  double min = 0;
  double max = 0;
  double p25 = 0;
  double p75 = 0;
  std::size_t n = 0;
};

double quantile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0;
  const double at = q * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(at));
  const auto hi = static_cast<std::size_t>(std::ceil(at));
  const double f = at - static_cast<double>(lo);
  return sorted[lo] * (1 - f) + sorted[hi] * f;
}

Stats stats_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  Stats s;
  s.n = v.size();
  if (v.empty()) return s;
  s.median = quantile(v, 0.5);
  s.min = v.front();
  s.max = v.back();
  s.p25 = quantile(v, 0.25);
  s.p75 = quantile(v, 0.75);
  return s;
}

std::string json_stats(const Stats& s, int digits = 1) {
  return "{\"median\":" + fixed(s.median, digits) + ",\"min\":" + fixed(s.min, digits) +
         ",\"max\":" + fixed(s.max, digits) + ",\"p25\":" + fixed(s.p25, digits) +
         ",\"p75\":" + fixed(s.p75, digits) + ",\"n\":" + std::to_string(s.n) + "}";
}

// ---- positions -------------------------------------------------------------------------------

// What one expansion of a position holds, at k = 8 with every relation.
struct Shape {
  double boards = 0;
  double timelines = 0;
  double active = 0;
  double moves = 0;
  double edges = 0;
  double w8 = 0;
  double w8_bands = 0;
  double e8 = 0;
  double e8_bands = 0;
  double src8 = 0;
  double scope8 = 0;  // W(8) + E(8) + src_extra(8): the Scope feature boards
  double relations = 0;
  double mover_pieces = 0;  // the mover's pieces on the boards its moves start from
  double pieces = 0;

  void add(const Shape& o) {
    boards += o.boards;
    timelines += o.timelines;
    active += o.active;
    moves += o.moves;
    edges += o.edges;
    w8 += o.w8;
    w8_bands += o.w8_bands;
    e8 += o.e8;
    e8_bands += o.e8_bands;
    src8 += o.src8;
    scope8 += o.scope8;
    relations += o.relations;
    mover_pieces += o.mover_pieces;
    pieces += o.pieces;
  }
  void divide(double n) {
    if (n <= 0) return;
    boards /= n;
    timelines /= n;
    active /= n;
    moves /= n;
    edges /= n;
    w8 /= n;
    w8_bands /= n;
    e8 /= n;
    e8_bands /= n;
    src8 /= n;
    scope8 /= n;
    relations /= n;
    mover_pieces /= n;
    pieces /= n;
  }
};

Shape shape_of(const t5d::GameState& s) {
  t5d::ExpandOptions o;
  o.attack = o.defence = o.xray = true;
  o.square_features = t5d::SquareFeatures::Scope;
  t5d::Expansion x;
  t5d::expand(s, o, x);
  Shape sh;
  sh.boards = static_cast<double>(x.board_count());
  sh.timelines = x.timeline_count();
  sh.active = x.active_timeline_count();
  sh.moves = static_cast<double>(x.move_records().size());
  sh.edges = static_cast<double>(x.move_edge_count());
  sh.w8 = static_cast<double>(x.window_size(8));
  sh.w8_bands = static_cast<double>(x.window_size(8, t5d::Window::BandsOnly));
  sh.e8 = static_cast<double>(x.e_count(8));
  sh.e8_bands = static_cast<double>(x.e_count(8, t5d::Window::BandsOnly));
  sh.src8 = static_cast<double>(x.src_extra_count(8));
  sh.scope8 = static_cast<double>(x.feature_boards().size());
  sh.relations = static_cast<double>(x.relation_count());
  const t5d::Multiverse& mv = s.multiverse();
  sh.pieces = static_cast<double>(mv.piece_count());
  for (std::size_t i = 0; i < x.board_count(); ++i) {
    if (x.board_playable()[i] == 0) continue;
    const t5d::Board* b = mv.board_at(x.board_timelines()[i], x.board_half_turns()[i]);
    sh.mover_pieces += std::popcount(b->pieces(s.to_move()));
  }
  return sh;
}

std::string json_shape(const Shape& s) {
  return "{\"boards\":" + fixed(s.boards, 1) + ",\"timelines\":" + fixed(s.timelines, 2) +
         ",\"active\":" + fixed(s.active, 2) + ",\"moves\":" + fixed(s.moves, 1) +
         ",\"move_edges\":" + fixed(s.edges, 1) + ",\"w_k8\":" + fixed(s.w8, 2) +
         ",\"w_bands_k8\":" + fixed(s.w8_bands, 2) + ",\"e_k8\":" + fixed(s.e8, 2) +
         ",\"e_bands_k8\":" + fixed(s.e8_bands, 2) + ",\"src_extra_k8\":" + fixed(s.src8, 2) +
         ",\"scope_k8\":" + fixed(s.scope8, 2) + ",\"relations\":" + fixed(s.relations, 1) +
         ",\"mover_pieces\":" + fixed(s.mover_pieces, 1) + ",\"pieces\":" + fixed(s.pieces, 1) +
         "}";
}

// A position the bench starts from: an action start of a synthetic game or a
// record, and how to rebuild it on boards of one's own.
struct Base {
  Base(std::string name, const t5d::GameState& s) : label(std::move(name)), state(s) {}

  std::string label;
  bool synthetic = false;
  std::shared_ptr<const t5d::synthetic::Game> game;  // synthetic: the game
  std::size_t actions = 0;                           // actions played to reach it
  t5d::GameState state;                              // the main thread's copy
  Shape shape;
  std::string fingerprint;
  // The moves to each worker's one-move children of this position.
  std::vector<std::vector<t5d::Move>> child_moves;
  bool disjoint = true;  // no two workers share a child
};

t5d::GameState rebuild(const Base& b) {
  if (b.game) {
    const t5d::Variant* v = t5d::find_variant(b.game->spec.variant);
    return t5d::synthetic::replay(t5d::variant_start(*v, b.game->spec.rules), b.game->actions,
                                  b.actions);
  }
  return b.state;  // records are timed on one thread only: shared boards will do
}

// The moves to `count` one-move children of `base` for each of `workers`
// workers: worker w takes, in a shuffle of the moves fixed by `key`, the moves
// at w, w + workers, w + 2 workers, ... - disjoint across workers - keeping a
// move only if the child can move (a child whose action is over expands to
// nothing); then, if that leaves it short, it goes round again (no longer
// disjoint). Fewer if the position has too few such moves.
std::vector<std::vector<t5d::Move>> choose_children(const t5d::GameState& base, std::uint64_t key,
                                                    int workers, std::size_t count,
                                                    bool& disjoint) {
  std::vector<std::vector<t5d::Move>> out(static_cast<std::size_t>(workers));
  t5d::MoveList moves;
  t5d::generate_moves(base, moves);
  if (moves.empty() || count == 0) return out;
  std::vector<std::size_t> order(moves.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  t5d::synthetic::Rng rng(key);
  for (std::size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.below(i)]);
  // Whether each move's child can move: decided once.
  std::vector<std::int8_t> usable(moves.size(), -1);
  t5d::MoveList scratch;
  const auto can_move = [&](std::size_t i) {
    if (usable[i] < 0) {
      t5d::GameState child = base;
      child.apply(moves[i]);
      usable[i] = static_cast<std::int8_t>(t5d::count_moves(child, scratch) > 0 ? 1 : 0);
    }
    return usable[i] == 1;
  };
  const auto w_count = static_cast<std::size_t>(workers);
  for (std::size_t w = 0; w < w_count; ++w) {
    std::vector<t5d::Move>& mine = out[w];
    for (std::size_t j = w; j < order.size() && mine.size() < count; j += w_count) {
      if (can_move(order[j])) mine.push_back(moves[order[j]]);
    }
    for (std::size_t j = 0; j < order.size() && mine.size() < count; ++j) {
      if (j % w_count == w || !can_move(order[j])) continue;
      disjoint = false;
      mine.push_back(moves[order[j]]);
    }
  }
  return out;
}

struct PositionClass {
  std::string name;
  std::string kind;  // "synthetic" or "records"
  bool synthetic = false;
  int boards = 0;  // the class's N (synthetic), or its bucket's floor (records)
  int active = 0;  // the class's T (synthetic)
  bool threaded = false;
  std::vector<Base*> bases;
  std::size_t children = 0;  // per base per worker
  Shape base_mean;
  Shape stream_mean;  // worker 0's stream
  std::size_t stream_size = 0;
  bool disjoint = true;
};

// A position the flatness series times, as how to build it on boards of one's
// own: its base (an action start of a flatness game), then `moves` applied -
// one move (a one-move child), none (the base itself, when it has no child
// that can move), or the next action's moves (a position in which no move may
// be made: everything an expansion does but move generation).
struct FlatItem {
  Base* base = nullptr;
  std::vector<t5d::Move> moves;
  // A NO_MOVES position under rules that allow moves after the shift (R8, the
  // default): the moves are played under the same rules but for Q4's
  // Guide-literal reading, the one under which a played action leaves no move
  // to make - so that the series still times everything but move generation.
  bool guide_literal = false;
  Shape shape;
};

// The state a flatness item times: its base (as `base`, rebuilt or not), then
// its moves - under Q4's Guide-literal reading if the item says so. The
// flatness games' actions are synthetic play, which stops at the shift (T5),
// so they are played alike under either reading of Q4.
t5d::GameState flat_state(const FlatItem& item, const t5d::GameState& base) {
  t5d::GameState s = base;
  if (item.guide_literal) {
    t5d::RuleOptions rules = base.rules();
    rules.moves_after_shift = false;
    s = t5d::GameState(base.multiverse(), rules);
  }
  for (const t5d::Move& m : item.moves) s.apply(m);
  return s;
}

// The series' streams: per level (an N of --flatness-boards), RAW - the
// one-move children of every action start near N - MATCHED - those of them
// whose move count and |E(8)| fall in bands every level can fill, so that the
// levels differ in N and not in the work the live set asks for - and NO_MOVES.
enum FlatKind : int { kFlatRaw = 0, kFlatMatched = 1, kFlatNoMove = 2 };
constexpr const char* kFlatKindName[] = {"raw", "matched", "no_moves"};

struct FlatStream {
  int level = 0;
  FlatKind kind = kFlatRaw;
  std::vector<FlatItem> items;
  Shape mean;
};

// The matched selection's bands, and how they were found.
struct Band {
  double lo = 0;
  double hi = 0;
  bool found = false;
  std::string how;  // "interquartile overlap", "inter-decile overlap", "none"
};

// ---- configurations --------------------------------------------------------------------------

struct Config {
  std::string name;
  std::string what;
  bool movegen_only = false;
  t5d::ExpandOptions options{};
  bool threaded = false;
};

std::vector<Config> configs(bool legal, const t5d::LegalityOptions& legality) {
  std::vector<Config> out;
  Config movegen;
  movegen.name = "movegen";
  movegen.what = "generate_moves only (the reference)";
  movegen.movegen_only = true;
  out.push_back(movegen);

  Config full;
  full.name = "expand";
  full.what = "moves + move edges + W(8) + E(8) + board features + square features of every "
              "board (the full-history input)";
  full.threaded = true;
  out.push_back(full);

  Config full_rel = full;
  full_rel.name = "expand+relations";
  full_rel.what = "the same + typed relations (attack, defence, x-ray)";
  full_rel.options.attack = full_rel.options.defence = full_rel.options.xray = true;
  out.push_back(full_rel);

  Config scope = full;
  scope.name = "expand-scope";
  scope.what = "moves + move edges + W(8) + E(8) + board features + square features of W(8) + "
               "E(8) + src_extra(8) only (the windowed input)";
  scope.options.square_features = t5d::SquareFeatures::Scope;
  scope.threaded = false;
  out.push_back(scope);

  Config m9 = scope;
  m9.name = "scope+relations";
  m9.what = "expand-scope + typed relations: §15.6's expansion - 'legal moves, attention masks, "
            "typed edges and E' - with the windowed configuration's input features";
  m9.options.attack = m9.options.defence = m9.options.xray = true;
  m9.threaded = true;
  out.push_back(m9);
  if (legal) {
    // --legal: the same with the legal moves only (ExpandOptions::legal_moves).
    Config scope_legal = scope;
    scope_legal.name = "expand-scope+legal";
    scope_legal.what = "expand-scope with the legal moves only (t5d/legal.hpp's tiers, under the "
                       "--legal-budget: the moves proved illegal dropped, the undecided kept)";
    scope_legal.options.legal_moves = true;
    scope_legal.options.legality = legality;
    out.push_back(scope_legal);
    Config m9_legal = m9;
    m9_legal.name = "scope+relations+legal";
    m9_legal.what = "scope+relations with the legal moves only";
    m9_legal.options.legal_moves = true;
    m9_legal.options.legality = legality;
    out.push_back(m9_legal);
  }
  return out;
}

// The flatness series' configurations (indices into configs()).
constexpr int kFlatConfigs[] = {0, 3, 4, 1};  // movegen, expand-scope, scope+relations, expand
constexpr int kFlatNoMoveConfigs[] = {3, 1};  // expand-scope, expand
constexpr int kScopeRelations = 4;            // the configuration M9's expansion is

std::vector<int> flat_configs(FlatKind kind) {
  return kind == kFlatNoMove
             ? std::vector<int>(std::begin(kFlatNoMoveConfigs), std::end(kFlatNoMoveConfigs))
             : std::vector<int>(std::begin(kFlatConfigs), std::end(kFlatConfigs));
}

// ---- timing ----------------------------------------------------------------------------------

std::atomic<std::uint64_t> g_sink{0};

// One expansion (or one move generation); a value derived from its result.
inline std::uint64_t run_one(const Config& c, const t5d::GameState& s, t5d::Expansion& x,
                             t5d::MoveList& moves) {
  if (c.movegen_only) {
    t5d::generate_moves(s, moves);
    return moves.size();
  }
  t5d::expand(s, c.options, x);
  std::uint64_t v = x.move_records().size() + x.e_boards().size() + x.window().size() +
                    x.relation_count() + x.move_edge_count();
  if (!x.square_pieces().empty()) v += x.square_pieces()[x.square_pieces().size() / 2];
  // The per-board features (f) of the boards the input holds: every board's
  // arrays for the full-history input, the scope boards' one by one for the
  // windowed input.
  switch (c.options.square_features) {
    case t5d::SquareFeatures::All:
      v += static_cast<std::uint64_t>(x.board_half_turns().back()) + x.board_playable().back();
      break;
    case t5d::SquareFeatures::Scope:
      for (const std::int32_t b : x.feature_boards()) {
        const t5d::BoardInfo info = x.board_info(b);
        v += static_cast<std::uint64_t>(info.half_turn) + (info.playable ? 1u : 0u);
      }
      break;
    case t5d::SquareFeatures::None:
      break;
  }
  return v;
}

enum class JobKind : std::uint8_t { Stream, Loop, Flat, Stop };

struct Job {
  JobKind kind = JobKind::Stop;
  int cls = 0;
  int cfg = 0;
  int threads = 1;
  int stream = 0;  // Flat: the flatness stream
  Clock::time_point deadline{};
};

struct Sample {
  double wall = 0;
  std::vector<std::uint64_t> counts;  // per worker taking part
  std::vector<double> cpu;            // per worker: thread CPU seconds
  double load = -1;
};

// Everything the workers read: built by the main thread before they start.
struct Plan {
  std::vector<PositionClass> classes;
  std::vector<FlatStream> flat;
  std::vector<Config> configs;
  int workers = 1;
  std::vector<int> cpu_of_worker;  // -1: unpinned
};

class Pool {
 public:
  explicit Pool(const Plan& plan)
      : plan_(plan),
        sync_(static_cast<std::ptrdiff_t>(plan.workers) + 1),
        results_(static_cast<std::size_t>(plan.workers)) {
    for (int w = 0; w < plan_.workers; ++w) threads_.emplace_back([this, w] { work(w); });
    sync_.arrive_and_wait();  // every worker has built its positions
  }
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  ~Pool() {
    job_ = Job{};
    job_.kind = JobKind::Stop;
    sync_.arrive_and_wait();
    for (std::thread& t : threads_) t.join();
  }

  bool pinned_all() const noexcept { return pinned_ == plan_.workers; }

  Sample run(Job job, double seconds) {
    const CpuTimes before = cpu_times();
    const Clock::time_point t0 = Clock::now();
    job.deadline = t0 + std::chrono::duration_cast<Clock::duration>(
                            std::chrono::duration<double>(seconds));
    job_ = job;
    sync_.arrive_and_wait();  // go
    sync_.arrive_and_wait();  // done
    Sample s;
    s.wall = std::chrono::duration<double>(Clock::now() - t0).count();
    s.load = busy_between(before, cpu_times());
    const int n = job.kind == JobKind::Stream || job.kind == JobKind::Loop ? job.threads : 1;
    for (int w = 0; w < n; ++w) {
      s.counts.push_back(results_[static_cast<std::size_t>(w)].count);
      s.cpu.push_back(results_[static_cast<std::size_t>(w)].cpu);
    }
    return s;
  }

 private:
  struct Result {
    std::uint64_t count = 0;
    double cpu = 0;
  };

  void work(int w) {
    const auto wi = static_cast<std::size_t>(w);
    if (plan_.cpu_of_worker[wi] >= 0 && pin_current_thread(plan_.cpu_of_worker[wi])) ++pinned_;
    // This worker's positions, on boards of its own.
    std::vector<std::vector<t5d::GameState>> stream(plan_.classes.size());
    std::vector<std::vector<t5d::GameState>> loop(plan_.classes.size());
    std::vector<std::size_t> cursor(plan_.classes.size(), 0);
    for (std::size_t c = 0; c < plan_.classes.size(); ++c) {
      const PositionClass& pc = plan_.classes[c];
      if (w != 0 && !pc.threaded) continue;
      // Worker w starts at the class's w-th game, so that workers running at
      // once - SMT siblings included - expand different games.
      const std::size_t nb = pc.bases.size();
      for (std::size_t i = 0; i < nb; ++i) {
        const Base* b = pc.bases[(i + wi) % nb];
        const t5d::GameState own = rebuild(*b);
        if (w == 0) loop[c].push_back(own);
        const std::vector<t5d::Move>& mine = b->child_moves[wi];
        for (const t5d::Move& m : mine) {
          t5d::GameState child = own;
          child.apply(m);
          stream[c].push_back(std::move(child));
        }
        if (mine.empty()) stream[c].push_back(own);  // no move: the position itself
      }
    }
    // The flatness streams: worker 0 only, in the plan's order (a fixed
    // shuffle, so that consecutive positions come from different games).
    std::vector<std::vector<t5d::GameState>> flat(plan_.flat.size());
    std::vector<std::size_t> flat_cursor(plan_.flat.size(), 0);
    if (w == 0) {
      std::map<const Base*, t5d::GameState> rebuilt;
      for (std::size_t i = 0; i < plan_.flat.size(); ++i) {
        for (const FlatItem& item : plan_.flat[i].items) {
          auto it = rebuilt.find(item.base);
          if (it == rebuilt.end()) it = rebuilt.emplace(item.base, rebuild(*item.base)).first;
          flat[i].push_back(flat_state(item, it->second));
        }
      }
    }
    t5d::Expansion x;
    t5d::MoveList moves;
    std::uint64_t sink = 0;
    sync_.arrive_and_wait();  // ready
    for (;;) {
      sync_.arrive_and_wait();  // go
      const Job job = job_;
      if (job.kind == JobKind::Stop) break;
      Result& r = results_[wi];
      const bool mine = job.kind == JobKind::Stream || job.kind == JobKind::Loop
                            ? w < job.threads
                            : w == 0;
      if (mine) {
        const Config& cfg = plan_.configs[static_cast<std::size_t>(job.cfg)];
        const std::vector<t5d::GameState>* positions = nullptr;
        std::size_t* at = nullptr;
        std::size_t one = 0;
        switch (job.kind) {
          case JobKind::Stream:
            positions = &stream[static_cast<std::size_t>(job.cls)];
            at = &cursor[static_cast<std::size_t>(job.cls)];
            break;
          case JobKind::Loop:
            positions = &loop[static_cast<std::size_t>(job.cls)];
            at = &one;
            break;
          case JobKind::Flat:
            positions = &flat[static_cast<std::size_t>(job.stream)];
            at = &flat_cursor[static_cast<std::size_t>(job.stream)];
            break;
          case JobKind::Stop:
            break;
        }
        const double c0 = thread_cpu_seconds();
        std::uint64_t done = 0;
        std::size_t j = *at;
        const std::size_t n = positions->size();
        do {
          sink += run_one(cfg, (*positions)[j], x, moves);
          ++done;
          if (++j == n) j = 0;
        } while (Clock::now() < job.deadline);
        *at = j;
        r.cpu = thread_cpu_seconds() - c0;
        r.count = done;
      }
      sync_.arrive_and_wait();  // done
    }
    g_sink += sink;
  }

  const Plan& plan_;
  std::barrier<> sync_;
  Job job_{};
  std::vector<Result> results_;
  std::atomic<int> pinned_{0};
  std::vector<std::thread> threads_;
};

// ---- options -------------------------------------------------------------------------------

struct Options {
  bool synthetic = true;
  bool flatness = true;
  std::vector<std::pair<int, int>> classes = {{100, 6}, {160, 7}, {240, 8}, {400, 10}, {600, 12}};
  int flatness_active = 8;
  std::vector<int> flatness_boards = {100, 160, 240, 400, 600};
  int games = 16;
  std::uint64_t seed = 1;
  int branch_percent = 25;
  std::vector<int> threads = {1, 2, 4, 8, 12, 24};
  int samples = 7;
  double sample_s = 0.2;
  double flat_s = 0.05;
  std::size_t children = 6;
  int flat_games = 24;
  int flat_window = 10;           // percent above each level's N
  std::size_t flat_children = 48;  // per action start
  std::size_t flat_max = 64;       // positions per flatness stream
  bool loop = true;
  bool pin = true;
  bool legal = false;
  bool legal_profile = false;
  std::uint64_t legal_checks = 64;
  std::uint64_t legal_nodes = 2000;
  std::string emit_dir;
  std::string json_path;
  t5d::RuleOptions rules{};
  std::vector<std::string> records;
};

void print_usage(std::FILE* stream) {
  std::fputs(
      "t5d-bench " T5D_VERSION_STRING " - expansion throughput (design spec §15.6, register M9).\n"
      "\n"
      "usage: t5d-bench [options] [RECORD ...]\n"
      "  --classes N:T,...     synthetic classes (default 100:6,160:7,240:8,400:10,600:12)\n"
      "  --no-synthetic        no synthetic classes and no flatness series (records only)\n"
      "  --flatness T          the flatness series' active timelines (default 8)\n"
      "  --flatness-boards L   its levels, N (default 100,160,240,400,600)\n"
      "  --flat-games G        its games (default 24)\n"
      "  --flat-window P       a level takes the action starts with N .. N + P% boards\n"
      "                        (default 10)\n"
      "  --flat-children C     one-move children per action start (default 48)\n"
      "  --flat-max M          positions per flatness stream (default 64)\n"
      "  --no-flatness         no flatness series\n"
      "  --games G             synthetic games per class (default 16)\n"
      "  --seed S              base seed (default 1); a class game's seed is\n"
      "                        S + 1000003 N + 10007 T + 37 g, a flatness game's\n"
      "                        S + 500000 + 37 g\n"
      "  --branch-percent P    the generator's branching knob (default 25)\n"
      "  --children C          one-move children per position per worker (default 6)\n"
      "  --threads LIST        thread counts (default 1,2,4,8,12,24; 1 is always added)\n"
      "  --samples S           timed rounds (default 7), after a warm-up round\n"
      "  --sample-ms MS        length of a sample (default 200)\n"
      "  --flat-ms MS          length of a flatness sample (default 50)\n"
      "  --no-loop             skip the loop over the class positions (the upper bound)\n"
      "  --no-pin              do not pin the worker threads\n"
      "  --legal               also time expand-scope and scope+relations with the legal\n"
      "                        moves only (ExpandOptions::legal_moves, under --legal-budget)\n"
      "  --legal-budget C,N    the classification's budget there, per expansion: C checks,\n"
      "                        N search nodes (LegalityOptions; default 64,2000)\n"
      "  --legal-profile       instead of the throughput: every class position and its\n"
      "                        one-move children expanded one by one - pseudo-legal, legal\n"
      "                        (exact), legal (--legal-budget), fast (0,0) - each timed; the\n"
      "                        distribution per expansion and how the moves were settled\n"
      "  --quick               a smoke test: --samples 3 --sample-ms 50 --flat-ms 10\n"
      "                        --games 4 --children 4 --flat-games 4 --flat-max 24\n"
      "  --emit DIR            write every synthetic position's 5dpgn record into DIR\n"
      "  --json FILE           write the JSON summary to FILE (default: standard output)\n"
      "  --no-moves-after-shift  a diagnostic: rules Q4's Guide-literal reading, no move\n"
      "                        once the present has shifted (everything is played under it;\n"
      "                        the default, R8: moves until submission)\n"
      "  --moves-after-shift   accepted and ignored: R8 made it the default\n"
      "  --rules ENCODING      any ruleset (RuleOptions::encode; decimal or 0x hex)\n"
      "  RECORD ...            5dpgn records: every action start is a position\n"
      "\n"
      "Synthetic positions are random play: throughput and stress only, never the |E|\n"
      "verdict. Moves are pseudo-legal but in the --legal configurations (the legal\n"
      "moves only, ExpandOptions::legal_moves: provisional).\n",
      stream);
}

std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> out;
  std::size_t i = 0;
  while (i <= text.size()) {
    const std::size_t comma = text.find(',', i);
    const std::string item =
        text.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
    const std::optional<std::uint64_t> v = t5d::tools::parse_number(item);
    if (!v || *v == 0 || *v > 100000) throw std::invalid_argument("not a positive count: " + item);
    out.push_back(static_cast<int>(*v));
    if (comma == std::string::npos) break;
    i = comma + 1;
  }
  return out;
}

Options parse_options(int argc, char** argv) {
  Options o;
  const auto value = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::invalid_argument(std::string(argv[i]) + " needs a value");
    return argv[++i];
  };
  const auto count = [&](int& i) -> int { return parse_int_list(value(i)).front(); };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (const int used = t5d::tools::ruleset_option(argc, argv, i, o.rules); used > 0) {
      i += used - 1;
    } else if (a == "--classes") {
      o.classes.clear();
      const std::string list = value(i);
      std::size_t at = 0;
      while (at < list.size()) {
        const std::size_t comma = list.find(',', at);
        const std::string item =
            list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        const std::size_t colon = item.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("--classes: N:T, not " + item);
        o.classes.emplace_back(parse_int_list(item.substr(0, colon)).front(),
                               parse_int_list(item.substr(colon + 1)).front());
        if (comma == std::string::npos) break;
        at = comma + 1;
      }
    } else if (a == "--no-synthetic") {
      o.synthetic = false;
    } else if (a == "--flatness") {
      o.flatness_active = count(i);
    } else if (a == "--flatness-boards") {
      o.flatness_boards = parse_int_list(value(i));
      std::sort(o.flatness_boards.begin(), o.flatness_boards.end());
      o.flatness_boards.erase(std::unique(o.flatness_boards.begin(), o.flatness_boards.end()),
                              o.flatness_boards.end());
      if (o.flatness_boards.size() < 2) {
        throw std::invalid_argument("--flatness-boards: at least two levels");
      }
    } else if (a == "--flat-games") {
      o.flat_games = count(i);
    } else if (a == "--flat-window") {
      o.flat_window = std::min(100, count(i));
    } else if (a == "--flat-children") {
      o.flat_children = static_cast<std::size_t>(count(i));
    } else if (a == "--flat-max") {
      o.flat_max = static_cast<std::size_t>(count(i));
    } else if (a == "--no-flatness") {
      o.flatness = false;
    } else if (a == "--games") {
      o.games = count(i);
    } else if (a == "--seed") {
      const std::optional<std::uint64_t> v = t5d::tools::parse_number(value(i));
      if (!v) throw std::invalid_argument("--seed: not a number");
      o.seed = *v;
    } else if (a == "--branch-percent") {
      o.branch_percent = std::min(100, count(i));
    } else if (a == "--children") {
      o.children = static_cast<std::size_t>(count(i));
    } else if (a == "--threads") {
      o.threads = parse_int_list(value(i));
    } else if (a == "--samples") {
      o.samples = count(i);
    } else if (a == "--sample-ms") {
      o.sample_s = count(i) / 1000.0;
    } else if (a == "--flat-ms") {
      o.flat_s = count(i) / 1000.0;
    } else if (a == "--no-loop") {
      o.loop = false;
    } else if (a == "--legal") {
      o.legal = true;
    } else if (a == "--legal-profile") {
      o.legal_profile = true;
    } else if (a == "--legal-budget") {
      const std::string v = value(i);
      const std::size_t comma = v.find(',');
      const std::optional<std::uint64_t> c = t5d::tools::parse_number(v.substr(0, comma));
      const std::optional<std::uint64_t> n =
          comma == std::string::npos ? std::nullopt : t5d::tools::parse_number(v.substr(comma + 1));
      if (!c || !n) throw std::invalid_argument("--legal-budget: C,N (two counts), not " + v);
      o.legal_checks = *c;
      o.legal_nodes = *n;
    } else if (a == "--no-pin") {
      o.pin = false;
    } else if (a == "--quick") {
      o.samples = 3;
      o.sample_s = 0.05;
      o.flat_s = 0.01;
      o.games = 4;
      o.children = 4;
      o.flat_games = 4;
      o.flat_max = 24;
    } else if (a == "--emit") {
      o.emit_dir = value(i);
    } else if (a == "--json") {
      o.json_path = value(i);
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("unknown argument " + a);
    } else {
      o.records.push_back(a);
    }
  }
  o.threads.push_back(1);
  std::sort(o.threads.begin(), o.threads.end());
  o.threads.erase(std::unique(o.threads.begin(), o.threads.end()), o.threads.end());
  return o;
}

// ---- building the positions ---------------------------------------------------------------

std::vector<std::unique_ptr<Base>> g_bases;

Base& add_base(std::string label, const t5d::GameState& state) {
  g_bases.push_back(std::make_unique<Base>(std::move(label), state));
  return *g_bases.back();
}

std::string bucket_name(std::size_t boards, bool synthetic, int& floor) {
  static const int kEdges[] = {1, 50, 100, 160, 240, 400, 600};
  int lo = 1;
  int hi = 0;
  for (std::size_t i = 0; i < std::size(kEdges); ++i) {
    if (static_cast<int>(boards) >= kEdges[i]) {
      lo = kEdges[i];
      hi = i + 1 < std::size(kEdges) ? kEdges[i + 1] - 1 : 0;
    }
  }
  floor = lo;
  const std::string kind = synthetic ? "synthetic records N " : "records N ";
  return hi > 0 ? kind + std::to_string(lo) + "-" + std::to_string(hi)
                : kind + std::to_string(lo) + "+";
}

void write_text(const std::string& path, const std::string& text) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("cannot write " + path);
  std::fwrite(text.data(), 1, text.size(), f);
  std::fclose(f);
}

std::uint64_t class_seed(std::uint64_t seed, int boards, int active, int game) {
  // std::uint64_t throughout: with `ull` literals, g++ 13/14 on Linux (where
  // uint64_t is unsigned long) warn under -Wsign-conversion.
  return seed + std::uint64_t{1000003} * static_cast<std::uint64_t>(boards) +
         std::uint64_t{10007} * static_cast<std::uint64_t>(active) +
         std::uint64_t{37} * static_cast<std::uint64_t>(game);
}

// The band of a quantity that every level can fill: the overlap of the
// levels' interquartile ranges, else of their inter-decile ranges.
Band common_band(const std::vector<std::vector<double>>& per_level) {
  struct Try {
    double lo_q, hi_q;
    const char* how;
  };
  for (const Try t : {Try{0.25, 0.75, "interquartile overlap"}, Try{0.10, 0.90, "inter-decile overlap"}}) {
    double lo = -1e300;
    double hi = 1e300;
    bool ok = !per_level.empty();
    for (std::vector<double> v : per_level) {
      if (v.empty()) {
        ok = false;
        break;
      }
      std::sort(v.begin(), v.end());
      lo = std::max(lo, quantile(v, t.lo_q));
      hi = std::min(hi, quantile(v, t.hi_q));
    }
    if (ok && lo <= hi) return Band{lo, hi, true, t.how};
  }
  return Band{0, 0, false, "none"};
}

std::string json_band(const Band& b) {
  return "{\"found\":" + std::string(b.found ? "true" : "false") + ",\"lo\":" + fixed(b.lo, 2) +
         ",\"hi\":" + fixed(b.hi, 2) + ",\"how\":" + json_string(b.how) + "}";
}

// Least-squares slope of y on x.
double slope_of(const std::vector<double>& x, const std::vector<double>& y) {
  const auto n = static_cast<double>(x.size());
  if (x.size() < 2) return 0;
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  const double d = n * sxx - sx * sx;
  return d != 0 ? (n * sxy - sx * sy) / d : 0;
}

// What a growth's spread over the rounds says about a threshold - never a
// verdict: the rounds either agree on a side, or they do not.
std::string against(const Stats& growth, double threshold) {
  if (growth.n == 0) return "no data";
  if (growth.max < threshold) return "below " + signed_percent(threshold) + " in every round";
  if (growth.min > threshold) return "above " + signed_percent(threshold) + " in every round";
  return "rounds straddle " + signed_percent(threshold) + ": inconclusive";
}

// §15.6's M9 throughput targets ("expansion >= 20,000/s per GPU at 240 boards
// and >= 10,000/s at 600"), by a synthetic class's N; 0 for none.
double m9_target(int boards) { return boards == 240 ? 20000 : boards == 600 ? 10000 : 0; }

// ---- results ----------------------------------------------------------------------------------

struct Measured {
  Stats total;       // expansions/s, every thread together (wall)
  Stats per_thread;  // expansions/s of one thread (wall)
  Stats cpu_us;      // thread CPU microseconds per expansion
  Stats load;        // the machine's busy fraction
};

Measured measure_of(const std::vector<Sample>& samples) {
  std::vector<double> total, per_thread, cpu_us, load;
  for (const Sample& s : samples) {
    std::uint64_t all = 0;
    double cpu = 0;
    for (std::size_t w = 0; w < s.counts.size(); ++w) {
      all += s.counts[w];
      cpu += s.cpu[w];
      per_thread.push_back(static_cast<double>(s.counts[w]) / s.wall);
    }
    total.push_back(static_cast<double>(all) / s.wall);
    if (all > 0) cpu_us.push_back(1e6 * cpu / static_cast<double>(all));
    if (s.load >= 0) load.push_back(s.load);
  }
  return Measured{stats_of(total), stats_of(per_thread), stats_of(cpu_us), stats_of(load)};
}

}  // namespace

// --legal-profile: each class position and its worker-0 children expanded
// one at a time - the windowed expansion with its relations, pseudo-legal,
// then with the legal moves (exact, budgeted, fast) - each timed on its own;
// per class the median, p95 and worst microseconds per expansion, and how the
// classification settled the moves (t5d/legal.hpp's tiers).
struct ProfileSeries {
  std::vector<double> us;
  t5d::LegalityStats tiers;
  std::size_t kept = 0;
  std::size_t unknown = 0;
};

double profile_quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const std::size_t i = std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())));
  return v[i];
}

int legal_profile(const Plan& plan, const Options& opt) {
  const char* names[] = {"pseudo-legal", "legal, exact", "legal, budgeted", "legal, fast"};
  t5d::ExpandOptions base;
  base.square_features = t5d::SquareFeatures::Scope;
  base.attack = base.defence = base.xray = true;
  std::vector<t5d::ExpandOptions> options(4, base);
  options[1].legal_moves = true;
  options[2].legal_moves = true;
  options[2].legality = t5d::LegalityOptions::budgeted(opt.legal_checks, opt.legal_nodes);
  options[3].legal_moves = true;
  options[3].legality = t5d::LegalityOptions::budgeted(0, 0);
  std::printf("t5d-bench --legal-profile: scope+relations expansions one by one, one thread; budgeted = "
              "%llu checks, %llu search nodes per expansion; fast = 0, 0\n",
              static_cast<unsigned long long>(opt.legal_checks), static_cast<unsigned long long>(opt.legal_nodes));
  t5d::Expansion x;
  for (const PositionClass& pc : plan.classes) {
    std::vector<t5d::GameState> states;
    for (const Base* b : pc.bases) {
      states.push_back(b->state);
      for (const t5d::Move& m : b->child_moves[0]) {
        t5d::GameState child = b->state;
        child.apply(m);
        states.push_back(child);
      }
    }
    std::vector<ProfileSeries> series(4);
    for (const t5d::GameState& s : states) {
      if (!s.may_move()) continue;
      for (std::size_t k = 0; k < 4; ++k) {
        if (k == 0) t5d::expand(s, options[k], x);  // warm the buffers
        const auto a = std::chrono::steady_clock::now();
        t5d::expand(s, options[k], x);
        const auto b = std::chrono::steady_clock::now();
        series[k].us.push_back(std::chrono::duration<double, std::micro>(b - a).count());
        const t5d::LegalityStats& st = x.legality_stats();
        t5d::LegalityStats& sum = series[k].tiers;
        sum.moves += st.moves;
        sum.standing_at_start += st.standing_at_start;
        sum.bulk_quiet += st.bulk_quiet;
        sum.bulk_branches += st.bulk_branches;
        sum.refuted_in_bulk += st.refuted_in_bulk;
        sum.modified += st.modified;
        sum.finished += st.finished;
        sum.standing_after += st.standing_after;
        sum.learned += st.learned;
        sum.no_finish_at_all += st.no_finish_at_all;
        sum.searched_legal += st.searched_legal;
        sum.searched_illegal += st.searched_illegal;
        sum.unknown += st.unknown;
        series[k].kept += x.moves().size();
        for (const t5d::MoveLegality l : x.move_legality()) {
          if (l == t5d::MoveLegality::Unknown) ++series[k].unknown;
        }
      }
    }
    std::printf("%s (%zu expansions):\n", pc.name.c_str(), series[0].us.size());
    for (std::size_t k = 0; k < 4; ++k) {
      const ProfileSeries& p = series[k];
      std::printf("  %-16s median %9.1f us  p95 %10.1f us  worst %11.1f us", names[k], profile_quantile(p.us, 0.5),
                  profile_quantile(p.us, 0.95), p.us.empty() ? 0.0 : *std::max_element(p.us.begin(), p.us.end()));
      if (k > 0) {
        const t5d::LegalityStats& t = p.tiers;
        const double m = static_cast<double>(std::max<std::size_t>(t.moves, 1));
        const auto pct = [&](std::size_t v) { return 100.0 * static_cast<double>(v) / m; };
        std::printf("  | moves kept %zu, unknown %zu | settled: standing at the start %.1f%%, stripped "
                    "quiet %.1f%%, stripped branches %.1f%%, refuted by a passed board %.1f%%, own boards "
                    "%.1f%%, finished %.1f%%, standing after %.1f%%, learned %.1f%%, no finish %.1f%%, "
                    "searched %.1f%%, unknown %.1f%%",
                    p.kept, p.unknown, pct(t.standing_at_start), pct(t.bulk_quiet), pct(t.bulk_branches),
                    pct(t.refuted_in_bulk), pct(t.modified), pct(t.finished - t.modified), pct(t.standing_after),
                    pct(t.learned), pct(t.no_finish_at_all), pct(t.searched_legal + t.searched_illegal),
                    pct(t.unknown));
      }
      std::printf("\n");
    }
    std::fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  Options opt;
  try {
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
        print_usage(stdout);
        return 0;
      }
      if (std::strcmp(argv[i], "--version") == 0) {
        std::puts(T5D_VERSION_STRING);
        return 0;
      }
    }
    opt = parse_options(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-bench: %s\n\n", e.what());
    print_usage(stderr);
    return 2;
  }

  try {
    Plan plan;
    plan.configs = configs(opt.legal, t5d::LegalityOptions::budgeted(opt.legal_checks, opt.legal_nodes));
    plan.workers = opt.threads.back();
    std::vector<std::shared_ptr<const t5d::synthetic::Game>> games;

    // -- synthetic classes: N boards, T active timelines (§15.6's table)
    if (opt.synthetic) {
      for (const auto& [n, t] : opt.classes) {
        PositionClass pc;
        pc.name = "synthetic N" + std::to_string(n) + " T" + std::to_string(t);
        pc.kind = "synthetic";
        pc.synthetic = true;
        pc.boards = n;
        pc.active = t;
        pc.threaded = true;
        pc.children = opt.children;
        for (int g = 0; g < opt.games; ++g) {
          t5d::synthetic::Spec spec;
          spec.rules = opt.rules;
          spec.seed = class_seed(opt.seed, n, t, g);
          spec.target_active = t;
          spec.target_boards = n;
          spec.branch_percent = opt.branch_percent;
          auto game = std::make_shared<const t5d::synthetic::Game>(
              t5d::synthetic::play_until_reached(spec));
          if (game->stop != t5d::synthetic::Stop::Target) {
            std::fprintf(stderr, "t5d-bench: %s: no game reached %d boards (seed %llu: %s)\n",
                         pc.name.c_str(), n, static_cast<unsigned long long>(game->seed),
                         t5d::synthetic::stop_name(game->stop));
            continue;
          }
          games.push_back(game);
          Base& b = add_base(pc.name + " seed " + std::to_string(game->seed), game->state);
          b.synthetic = true;
          b.game = game;
          b.actions = game->actions.size();
          pc.bases.push_back(&b);
        }
        plan.classes.push_back(pc);
      }
    }

    // -- records: every action start, by N bucket
    if (!opt.records.empty()) {
      std::vector<PositionClass> buckets;
      for (const std::string& path : opt.records) {
        const std::string text = t5d::tools::read_file(path);
        try {
          t5d::PgnReplay replay(t5d::read_pgn(text), t5d::PgnOptions{}, opt.rules);
          t5d::tools::parse_mode(replay);
          const bool synthetic = t5d::tools::is_synthetic(replay.record());
          for (;;) {
            int floor = 0;
            const std::string bucket =
                bucket_name(replay.state().multiverse().board_count(), synthetic, floor);
            Base& b = add_base(path + "@" + std::to_string(replay.actions_played()),
                               replay.state());
            b.synthetic = synthetic;
            b.actions = replay.actions_played();
            auto it = std::find_if(buckets.begin(), buckets.end(),
                                   [&](const PositionClass& c) { return c.name == bucket; });
            if (it == buckets.end()) {
              PositionClass pc;
              pc.name = bucket;
              pc.kind = "records";
              pc.synthetic = synthetic;
              pc.boards = floor;
              buckets.push_back(pc);
              it = buckets.end() - 1;
            }
            it->bases.push_back(&b);
            if (replay.finished()) break;
            replay.play_next();
          }
        } catch (const t5d::RecordError& e) {
          std::fprintf(stderr, "t5d-bench: %s: stopped: %s\n", path.c_str(), e.what());
        }
      }
      std::stable_sort(buckets.begin(), buckets.end(),
                       [](const PositionClass& a, const PositionClass& b) {
                         return std::tie(a.synthetic, a.boards) < std::tie(b.synthetic, b.boards);
                       });
      for (PositionClass& pc : buckets) {
        // About games x children positions in the stream, whatever the bucket's size.
        const std::size_t target = static_cast<std::size_t>(opt.games) * opt.children;
        pc.children = std::max<std::size_t>(1, (target + pc.bases.size() - 1) / pc.bases.size());
        plan.classes.push_back(pc);
      }
    }

    // -- the flatness series: games with T held, taken at every action start
    //    whose N lies in a level's window [N, N + flat_window%); the candidates
    //    are the one-move children of the positions along the action that
    //    follows (so that every level has positions of every depth into an
    //    action, and a spread of move counts to match on)
    const std::size_t levels = opt.flatness_boards.size();
    std::vector<std::vector<FlatItem>> flat_candidates(levels);
    std::vector<std::vector<FlatItem>> flat_no_move(levels);
    std::vector<std::uint64_t> flat_seeds;
    const auto level_end = [&](int n) { return n + std::max(1, n * opt.flat_window / 100); };
    if (opt.synthetic && opt.flatness) {
      std::vector<int> targets;
      for (const int n : opt.flatness_boards) {
        for (int t = n; t < level_end(n); t += 2) targets.push_back(t);
      }
      for (int g = 0; g < opt.flat_games; ++g) {
        t5d::synthetic::Spec spec;
        spec.rules = opt.rules;
        spec.seed = opt.seed + 500000 + 37 * static_cast<std::uint64_t>(g);
        spec.target_active = opt.flatness_active;
        spec.target_boards = level_end(opt.flatness_boards.back()) + 40;
        spec.snapshots = targets;
        spec.branch_percent = opt.branch_percent;
        auto game =
            std::make_shared<const t5d::synthetic::Game>(t5d::synthetic::play_until_reached(spec));
        if (game->stop != t5d::synthetic::Stop::Target) {
          std::fprintf(stderr, "t5d-bench: flatness game %d: no game reached %d boards\n", g,
                       spec.target_boards);
          continue;
        }
        games.push_back(game);
        flat_seeds.push_back(game->seed);
        std::size_t last = static_cast<std::size_t>(-1);
        for (const t5d::synthetic::Snapshot& snap : game->snapshots) {
          if (snap.actions == last || snap.actions >= game->actions.size()) continue;
          last = snap.actions;
          const auto n = static_cast<int>(snap.state.multiverse().board_count());
          int level = -1;
          for (std::size_t l = 0; l < levels; ++l) {
            if (n >= opt.flatness_boards[l] && n < level_end(opt.flatness_boards[l])) {
              level = static_cast<int>(l);
            }
          }
          if (level < 0) continue;
          const auto li = static_cast<std::size_t>(level);
          Base& b = add_base("flatness N" + std::to_string(opt.flatness_boards[li]) + " T" +
                                 std::to_string(opt.flatness_active) + " seed " +
                                 std::to_string(game->seed) + " action " +
                                 std::to_string(snap.actions),
                             snap.state);
          b.synthetic = true;
          b.game = game;
          b.actions = snap.actions;
          const std::vector<t5d::Move>& next = game->actions[snap.actions];
          const std::size_t per_prefix =
              std::max<std::size_t>(2, (opt.flat_children + next.size() - 1) / next.size());
          t5d::GameState along = snap.state;
          std::vector<t5d::Move> prefix;
          for (std::size_t j = 0; j < next.size(); ++j) {
            bool disjoint = true;
            const std::vector<t5d::Move> kids =
                choose_children(along, fnv1a(b.label) + j, 1, per_prefix, disjoint)[0];
            for (const t5d::Move& m : kids) {
              FlatItem item;
              item.base = &b;
              item.moves = prefix;
              item.moves.push_back(m);
              flat_candidates[li].push_back(std::move(item));
            }
            along.apply(next[j]);
            prefix.push_back(next[j]);
          }
          // The action played, no move may be made: under Q4's Guide-literal
          // reading only - R8's default lets the mover go on after the shift,
          // so then the item is played under that reading (FlatItem). A
          // position rebuilt from its boards loses activity that depends on
          // the history of play (Q1, Q3, Q6): no item then.
          FlatItem item;
          item.base = &b;
          item.moves = prefix;
          item.guide_literal = along.may_move();
          if (!item.guide_literal || !snap.state.rules().history_dependent_activity()) {
            if (!flat_state(item, snap.state).may_move()) flat_no_move[li].push_back(std::move(item));
          }
        }
      }
    }

    // -- every base: its shape, its fingerprint, its children
    for (const std::unique_ptr<Base>& b : g_bases) {
      b->shape = shape_of(b->state);
      b->fingerprint = t5d::fingerprint(b->state.multiverse()).fingerprint;
    }
    for (PositionClass& pc : plan.classes) {
      const int workers = pc.threaded ? plan.workers : 1;
      for (Base* b : pc.bases) {
        bool disjoint = true;
        b->child_moves = choose_children(b->state, fnv1a(b->label), workers, pc.children, disjoint);
        b->child_moves.resize(static_cast<std::size_t>(plan.workers));
        b->disjoint = disjoint;
        pc.disjoint = pc.disjoint && disjoint;
        pc.base_mean.add(b->shape);
        // Worker 0's stream, described.
        for (const t5d::Move& m : b->child_moves[0]) {
          t5d::GameState child = b->state;
          child.apply(m);
          pc.stream_mean.add(shape_of(child));
          ++pc.stream_size;
        }
        if (b->child_moves[0].empty()) {
          pc.stream_mean.add(b->shape);
          ++pc.stream_size;
        }
      }
      pc.base_mean.divide(static_cast<double>(pc.bases.size()));
      pc.stream_mean.divide(static_cast<double>(pc.stream_size));
    }
    if (opt.legal_profile) return legal_profile(plan, opt);
    // The flatness candidates' shapes, and the streams: raw, matched, no-move.
    // The matched selection: anchors at the first level whose move count and
    // |E(8)| lie in the bands every level covers, each paired at every other
    // level with its nearest position in (moves, |E(8)|, the mover's pieces)
    // WITHIN A CALIPER - moves and the mover's pieces within 10%, |E(8)| within
    // max(1, 20%) - or dropped, so that the levels' sets are alike.
    Band flat_moves_band;
    Band flat_e_band;
    std::vector<Shape> flat_mismatch;  // per level: mean |relative difference| to the anchors
    std::size_t flat_anchors_tried = 0;
    bool flat_complete = !flat_seeds.empty();
    for (std::size_t l = 0; l < levels && flat_complete; ++l) {
      flat_complete = !flat_candidates[l].empty();
    }
    if (flat_complete) {
      const auto shape_items = [](std::vector<FlatItem>& items) {
        for (FlatItem& item : items) {
          item.shape = shape_of(flat_state(item, item.base->state));
        }
      };
      std::vector<std::vector<double>> moves_by_level(levels);
      for (std::size_t l = 0; l < levels; ++l) {
        shape_items(flat_candidates[l]);
        shape_items(flat_no_move[l]);
        for (const FlatItem& item : flat_candidates[l]) moves_by_level[l].push_back(item.shape.moves);
      }
      std::vector<std::vector<double>> moves_by(levels);
      std::vector<std::vector<double>> e_by(levels);
      for (std::size_t l = 0; l < levels; ++l) {
        for (const FlatItem& item : flat_candidates[l]) {
          moves_by[l].push_back(item.shape.moves);
          e_by[l].push_back(item.shape.e8);
        }
      }
      flat_moves_band = common_band(moves_by);
      flat_e_band = common_band(e_by);
      const auto in = [](const Band& band, double v) {
        return !band.found || (v >= band.lo && v <= band.hi);
      };
      std::vector<std::vector<FlatItem>> matched(levels);
      for (const FlatItem& item : flat_candidates[0]) {
        if (in(flat_moves_band, item.shape.moves) && in(flat_e_band, item.shape.e8)) {
          matched[0].push_back(item);
        }
      }
      std::vector<FlatItem> anchors;
      anchors.swap(matched[0]);
      {
        t5d::synthetic::Rng rng(0xA11C0DEull);
        for (std::size_t i = anchors.size(); i > 1; --i) {
          std::swap(anchors[i - 1], anchors[rng.below(i)]);
        }
      }
      // The distance: relative differences, summed (|E| shifted by one: it is
      // often 0).
      const auto distance = [](const Shape& a, const Shape& b) {
        return std::abs(std::log(b.moves / a.moves)) +
               std::abs(std::log((b.e8 + 1) / (a.e8 + 1))) +
               std::abs(std::log(b.mover_pieces / a.mover_pieces));
      };
      const auto within = [](const Shape& a, const Shape& b) {
        return std::abs(b.moves / a.moves - 1) <= 0.10 &&
               std::abs(b.mover_pieces / a.mover_pieces - 1) <= 0.10 &&
               std::abs(b.e8 - a.e8) <= std::max(1.0, 0.2 * a.e8);
      };
      flat_mismatch.assign(levels, Shape{});
      std::vector<std::vector<bool>> used(levels);
      for (std::size_t l = 0; l < levels; ++l) used[l].assign(flat_candidates[l].size(), false);
      for (const FlatItem& a : anchors) {
        if (matched[0].size() >= opt.flat_max) break;
        ++flat_anchors_tried;
        std::vector<std::size_t> picks(levels, 0);
        bool paired = true;
        for (std::size_t l = 1; l < levels && paired; ++l) {
          const std::vector<FlatItem>& cands = flat_candidates[l];
          std::size_t best = cands.size();
          double best_d = 0;
          for (std::size_t c = 0; c < cands.size(); ++c) {
            if (used[l][c] || !within(a.shape, cands[c].shape)) continue;
            const double d = distance(a.shape, cands[c].shape);
            if (best == cands.size() || d < best_d) {
              best = c;
              best_d = d;
            }
          }
          paired = best < cands.size();
          picks[l] = best;
        }
        if (!paired) continue;  // no partner within the caliper at some level
        matched[0].push_back(a);
        for (std::size_t l = 1; l < levels; ++l) {
          used[l][picks[l]] = true;
          const Shape& b = flat_candidates[l][picks[l]].shape;
          Shape diff;
          diff.moves = std::abs(b.moves / a.shape.moves - 1);
          diff.e8 = std::abs(b.e8 - a.shape.e8);
          diff.mover_pieces = std::abs(b.mover_pieces / a.shape.mover_pieces - 1);
          flat_mismatch[l].add(diff);
          matched[l].push_back(flat_candidates[l][picks[l]]);
        }
      }
      for (std::size_t l = 1; l < levels; ++l) {
        flat_mismatch[l].divide(static_cast<double>(matched[l].size()));
      }
      const auto add_stream = [&](std::size_t l, FlatKind kind, std::vector<FlatItem> items) {
        // A fixed shuffle (consecutive positions from different games), at
        // most flat_max.
        t5d::synthetic::Rng rng(std::uint64_t{0x5EED} + std::uint64_t{7919} * l +
                                std::uint64_t{104729} * static_cast<unsigned>(kind));
        for (std::size_t i = items.size(); i > 1; --i) std::swap(items[i - 1], items[rng.below(i)]);
        if (items.size() > opt.flat_max) items.resize(opt.flat_max);
        if (items.empty()) return;
        FlatStream fs;
        fs.level = static_cast<int>(l);
        fs.kind = kind;
        for (const FlatItem& item : items) fs.mean.add(item.shape);
        fs.mean.divide(static_cast<double>(items.size()));
        fs.items = std::move(items);
        plan.flat.push_back(std::move(fs));
      };
      bool matched_everywhere = true;
      for (std::size_t l = 0; l < levels; ++l) matched_everywhere = matched_everywhere && !matched[l].empty();
      bool no_move_everywhere = true;
      for (std::size_t l = 0; l < levels; ++l) no_move_everywhere = no_move_everywhere && !flat_no_move[l].empty();
      for (std::size_t l = 0; l < levels; ++l) {
        add_stream(l, kFlatRaw, flat_candidates[l]);
        if (matched_everywhere) add_stream(l, kFlatMatched, matched[l]);
        if (no_move_everywhere) add_stream(l, kFlatNoMove, flat_no_move[l]);
      }
    }
    // Drop empty classes.
    plan.classes.erase(std::remove_if(plan.classes.begin(), plan.classes.end(),
                                      [](const PositionClass& c) { return c.bases.empty(); }),
                       plan.classes.end());

    if (!opt.emit_dir.empty()) {
      for (const std::unique_ptr<Base>& b : g_bases) {
        if (!b->game) continue;
        std::string file = b->label;
        for (char& ch : file) {
          if (ch == ' ') ch = '-';
        }
        write_text(opt.emit_dir + "/" + file + ".5dpgn",
                   t5d::synthetic::record(*b->game, b->actions));
      }
    }

    // -- the machine, the threads, the clock
    const std::string cpu = cpu_model();
    const unsigned hw = std::thread::hardware_concurrency();
    const std::vector<std::vector<int>> cores = physical_cores();
    plan.cpu_of_worker.assign(static_cast<std::size_t>(plan.workers), -1);
    if (opt.pin && !cores.empty()) {
      for (int w = 0; w < plan.workers; ++w) {
        const auto nc = cores.size();
        const auto wi = static_cast<std::size_t>(w);
        const std::vector<int>& core = cores[wi % nc];
        plan.cpu_of_worker[wi] = core[(wi / nc) % core.size()];
      }
    }
    double clock_check = 0;
    const std::string clock_method = calibrate_thread_clock(clock_check);

    std::printf("t5d-bench %s - expansion throughput (spec §15.6, M9); PSEUDO-LEGAL moves "
                "but in the +legal configurations (ExpandOptions::legal_moves): provisional\n",
                T5D_VERSION_STRING);
    std::printf("machine: %s, %llu physical cores, %u logical processors, %s\n", cpu.c_str(),
                static_cast<unsigned long long>(cores.size()), hw, os_text().c_str());
    std::printf("build:   %s, %s, flags \"%s\"\n", compiler_text().c_str(), T5D_BUILD_TYPE,
                trim(T5D_CXX_FLAGS).c_str());
    std::printf("rules:   %s\n", t5d::tools::ruleset_text(opt.rules).c_str());

    Pool pool(plan);
    std::printf("method:  %d interleaved rounds (after a warm-up round) of %.0f ms samples; "
                "worker threads %s; thread CPU clock: %s (check %.3f)\n",
                opt.samples, opt.sample_s * 1000,
                pool.pinned_all() ? "pinned, one per physical core first" : "not pinned",
                clock_method.c_str(), clock_check);
    std::printf("SYNTHETIC classes are random play (throughput only, never the |E| verdict).\n");
    std::fflush(stdout);

    // -- the schedule: rounds over every measurement
    using Key = std::tuple<int, int, int, bool>;  // class, config, threads, loop
    std::map<Key, std::vector<Sample>> results;
    // The flatness streams by (kind, level), and their samples by (stream,
    // config): thread CPU microseconds per expansion, one per round.
    std::map<std::pair<int, int>, std::size_t> flat_index;
    for (std::size_t i = 0; i < plan.flat.size(); ++i) {
      flat_index[{static_cast<int>(plan.flat[i].kind), plan.flat[i].level}] = i;
    }
    std::map<std::pair<std::size_t, int>, std::vector<double>> flat_cpu_us;
    int round = 0;
    const auto run_round = [&](bool record, double scale) {
      for (std::size_t c = 0; c < plan.classes.size(); ++c) {
        const PositionClass& pc = plan.classes[c];
        for (const int t : opt.threads) {
          if (t > 1 && !pc.threaded) continue;
          for (std::size_t k = 0; k < plan.configs.size(); ++k) {
            const Config& cfg = plan.configs[k];
            if (t > 1 && !cfg.threaded) continue;
            for (const bool loop : {false, true}) {
              if (loop && (t != 1 || !opt.loop || !cfg.threaded)) continue;
              Job job;
              job.kind = loop ? JobKind::Loop : JobKind::Stream;
              job.cls = static_cast<int>(c);
              job.cfg = static_cast<int>(k);
              job.threads = t;
              const Sample s = pool.run(job, opt.sample_s * scale);
              if (record) results[Key{job.cls, job.cfg, t, loop}].push_back(s);
            }
          }
        }
      }
      // The flatness series: for each kind and configuration, its levels back
      // to back - ascending in even rounds, descending in odd ones, so that a
      // drift of the machine within a round does not favour either end.
      for (const FlatKind kind : {kFlatRaw, kFlatMatched, kFlatNoMove}) {
        for (const int k : flat_configs(kind)) {
          for (std::size_t step = 0; step < levels; ++step) {
            const std::size_t l = round % 2 == 0 ? step : levels - 1 - step;
            const auto it = flat_index.find({static_cast<int>(kind), static_cast<int>(l)});
            if (it == flat_index.end()) continue;
            Job job;
            job.kind = JobKind::Flat;
            job.cfg = k;
            job.stream = static_cast<int>(it->second);
            const Sample s = pool.run(job, opt.flat_s * scale);
            if (record && s.counts[0] > 0) {
              flat_cpu_us[{it->second, k}].push_back(1e6 * s.cpu[0] /
                                                      static_cast<double>(s.counts[0]));
            }
          }
        }
      }
      ++round;
    };
    run_round(false, 0.25);  // warm-up
    for (int r = 0; r < opt.samples; ++r) {
      run_round(true, 1.0);
      std::fprintf(stderr, "t5d-bench: round %d of %d done\n", r + 1, opt.samples);
    }

    // -- the report
    std::vector<double> all_loads, single_loads;
    for (const auto& [key, samples] : results) {
      for (const Sample& s : samples) {
        if (s.load < 0) continue;
        all_loads.push_back(s.load);
        if (std::get<2>(key) == 1) single_loads.push_back(s.load);
      }
    }
    const Stats load_all = stats_of(all_loads);
    const Stats load_single = stats_of(single_loads);
    std::printf("load:    the machine's busy fraction during the samples: median %s, max %s; "
                "during 1-thread samples median %s (the bench itself: %s of the processors)\n\n",
                percent(load_all.median).c_str(), percent(load_all.max).c_str(),
                percent(load_single.median).c_str(),
                hw > 0 ? percent(1.0 / static_cast<double>(hw)).c_str() : "?");

    std::string json =
        "{\"event\":\"summary\",\"tool\":\"t5d-bench\",\"version\":\"" T5D_VERSION_STRING "\"";
    json += ",\"provisional\":\"pseudo-legal moves but in the +legal configurations "
            "(ExpandOptions::legal_moves; R1, build-log spec issue 28)\"";
    json += ",\"machine\":{\"cpu\":" + json_string(cpu) + ",\"physical_cores\":" +
            std::to_string(cores.size()) + ",\"logical_processors\":" + std::to_string(hw) +
            ",\"os\":" + json_string(os_text()) + "}";
    json += ",\"build\":{\"compiler\":" + json_string(compiler_text()) + ",\"build_type\":" +
            json_string(T5D_BUILD_TYPE) + ",\"flags\":" + json_string(trim(T5D_CXX_FLAGS)) + "}";
    json += ",\"method\":{\"rounds\":" + std::to_string(opt.samples) + ",\"sample_ms\":" +
            std::to_string(std::lround(opt.sample_s * 1000)) + ",\"flat_sample_ms\":" +
            std::to_string(std::lround(opt.flat_s * 1000)) +
            ",\"interleaved\":true,\"warm_up_round\":true,\"workers\":" +
            std::to_string(plan.workers) + ",\"pinned\":" + (pool.pinned_all() ? "true" : "false") +
            ",\"children_per_position_per_worker\":" + std::to_string(opt.children) +
            ",\"thread_cpu_clock\":" + json_string(clock_method) +
            ",\"thread_clock_check\":" + fixed(clock_check, 3) + "}";
    json += ",\"load\":{\"all\":" + json_stats(load_all, 3) + ",\"single_thread\":" +
            json_stats(load_single, 3) + "}";
    char encoding[24];
    std::snprintf(encoding, sizeof encoding, "0x%llx",
                  static_cast<unsigned long long>(opt.rules.encode()));
    json += ",\"rules\":" + json_string(t5d::tools::ruleset_text(opt.rules)) +
            ",\"rules_encoding\":\"" + encoding + "\"";
    json += ",\"configs\":[";
    for (std::size_t i = 0; i < plan.configs.size(); ++i) {
      if (i != 0) json += ',';
      json += "{\"name\":" + json_string(plan.configs[i].name) +
              ",\"what\":" + json_string(plan.configs[i].what) + "}";
    }
    json += "],\"classes\":[";

    // The 1-thread stream figures of every class, for the closing summary.
    std::vector<std::string> summary;
    for (std::size_t c = 0; c < plan.classes.size(); ++c) {
      const PositionClass& pc = plan.classes[c];
      const Shape& m = pc.base_mean;
      const Shape& st = pc.stream_mean;
      std::printf("== %s%s: %llu positions; mean N %.0f, active T %.1f, moves %.0f, |W(8)| %.1f "
                  "(bands %.1f), |E(8)| %.1f (vs bands %.1f), src_extra(8) %.1f, scope %.1f, "
                  "relations %.0f\n",
                  pc.name.c_str(), pc.synthetic && pc.kind == "records" ? " (SYNTHETIC)" : "",
                  static_cast<unsigned long long>(pc.bases.size()), m.boards, m.active, m.moves,
                  m.w8, m.w8_bands, m.e8, m.e8_bands, m.src8, m.scope8, m.relations);
      std::printf("   timed stream: %llu one-move children per position per worker (%llu per "
                  "worker%s): mean N %.0f, moves %.0f, |E(8)| %.1f, scope %.1f, relations %.0f\n",
                  static_cast<unsigned long long>(pc.children),
                  static_cast<unsigned long long>(pc.stream_size),
                  pc.threaded ? (pc.disjoint ? ", disjoint across workers"
                                             : ", NOT disjoint across workers")
                              : "",
                  st.boards, st.moves, st.e8, st.scope8, st.relations);
      std::printf("   %-17s %3s %22s %12s %10s %13s %15s %5s\n", "configuration", "thr",
                  "wall /s [min, max]", "per thread", "CPU us/exp", "CPU-time /s",
                  "scaling/round", "load");

      if (c != 0) json += ',';
      json += "{\"name\":" + json_string(pc.name) + ",\"kind\":" + json_string(pc.kind) +
              ",\"synthetic\":" + (pc.synthetic ? "true" : "false") +
              ",\"boards\":" + std::to_string(pc.boards) +
              ",\"active\":" + std::to_string(pc.active) + ",\"positions\":[";
      for (std::size_t i = 0; i < pc.bases.size(); ++i) {
        const Base* b = pc.bases[i];
        if (i != 0) json += ',';
        json += "{\"label\":" + json_string(b->label) +
                ",\"synthetic\":" + (b->synthetic ? "true" : "false") +
                ",\"actions\":" + std::to_string(b->actions);
        if (b->game) json += ",\"seed\":" + std::to_string(b->game->seed);
        json += ",\"fingerprint\":" + json_string(b->fingerprint) +
                ",\"shape\":" + json_shape(b->shape) + "}";
      }
      json += "],\"mean\":" + json_shape(m) + ",\"stream\":{\"children_per_position\":" +
              std::to_string(pc.children) + ",\"positions_per_worker\":" +
              std::to_string(pc.stream_size) + ",\"disjoint\":" + (pc.disjoint ? "true" : "false") +
              ",\"mean\":" + json_shape(st) + "},\"results\":[";
      bool first = true;
      for (std::size_t k = 0; k < plan.configs.size(); ++k) {
        const Config& cfg = plan.configs[k];
        const auto base_it = results.find(Key{static_cast<int>(c), static_cast<int>(k), 1, false});
        for (const int t : opt.threads) {
          for (const bool loop : {false, true}) {
            const auto it = results.find(Key{static_cast<int>(c), static_cast<int>(k), t, loop});
            if (it == results.end()) continue;
            const Measured r = measure_of(it->second);
            // Scaling: the t-thread total over the 1-thread total OF THE SAME
            // ROUND (both measured within seconds of each other), over rounds.
            std::vector<double> ratios;
            if (!loop && base_it != results.end()) {
              const std::vector<Sample>& one = base_it->second;
              for (std::size_t i = 0; i < it->second.size() && i < one.size(); ++i) {
                const auto total = [](const Sample& x) {
                  std::uint64_t all = 0;
                  for (const std::uint64_t n : x.counts) all += n;
                  return static_cast<double>(all) / x.wall;
                };
                if (total(one[i]) > 0) ratios.push_back(total(it->second[i]) / total(one[i]));
              }
            }
            const Stats scaling = stats_of(ratios);
            const double cpu_rate = r.cpu_us.median > 0 ? 1e6 / r.cpu_us.median : 0;
            std::printf("   %-17s %3d %10s [%s, %s] %12s %10s %13s %15s %5s\n",
                        (cfg.name + (loop ? " (loop)" : "")).c_str(), t,
                        grouped(r.total.median).c_str(), grouped(r.total.min).c_str(),
                        grouped(r.total.max).c_str(), grouped(r.per_thread.median).c_str(),
                        fixed(r.cpu_us.median, 1).c_str(), grouped(cpu_rate).c_str(),
                        loop || t == 1 ? "-"
                                       : (fixed(scaling.median, 2) + "x [" + fixed(scaling.min, 1) +
                                          "," + fixed(scaling.max, 1) + "]")
                                             .c_str(),
                        percent(r.load.median).c_str());
            if (!first) json += ',';
            first = false;
            json += "{\"config\":" + json_string(cfg.name) + ",\"mode\":\"" +
                    (loop ? "loop" : "stream") + "\",\"threads\":" + std::to_string(t) +
                    ",\"total_per_s\":" + json_stats(r.total) +
                    ",\"per_thread_per_s\":" + json_stats(r.per_thread) +
                    ",\"cpu_us_per_expansion\":" + json_stats(r.cpu_us, 3) +
                    ",\"cpu_time_rate_per_thread\":" + fixed(cpu_rate, 1) +
                    ",\"ns_per_move\":" +
                    fixed(st.moves > 0 && !loop ? 1e3 * r.cpu_us.median / st.moves : 0, 2) +
                    ",\"scaling_per_round\":" + json_stats(scaling, 3) +
                    ",\"load\":" + json_stats(r.load, 3) + "}";
            if (t == 1 && !loop) {
              std::string line = "   " + pc.name + ": " + cfg.name + " " + grouped(cpu_rate) +
                                 "/s CPU-time, " + grouped(r.total.median) + "/s wall [" +
                                 grouped(r.total.min) + ", " + grouped(r.total.max) + "]";
              const double target = pc.synthetic && pc.kind == "synthetic" ? m9_target(pc.boards) : 0;
              if (target > 0) {
                line += "; M9 target " + grouped(target) + "/s: wall x" +
                        fixed(r.total.min / target, 2) + " .. x" + fixed(r.total.max / target, 2) +
                        " over rounds, CPU-time x" + fixed(cpu_rate / target, 2);
              }
              summary.push_back(line);
            }
          }
        }
      }
      json += "]}";
      std::printf("\n");
      std::fflush(stdout);
    }
    json += "]";

    // -- the flatness series
    if (!plan.flat.empty()) {
      std::printf("== flatness (SYNTHETIC): T %d, %llu games; at each level the action starts "
                  "with N .. N + %d%% boards; positions: one-move children along the action "
                  "that follows; single thread, interleaved; growth = N%d against N%d, per "
                  "round (median [min, max] over %d rounds)\n",
                  opt.flatness_active, static_cast<unsigned long long>(flat_seeds.size()),
                  opt.flat_window, opt.flatness_boards.back(), opt.flatness_boards.front(),
                  opt.samples);
      std::printf("   matched: anchors at N%d with moves in [%.0f, %.0f] and |E(8)| in [%.1f, %.1f] "
                  "(the bands every level covers: %s, %s), each paired at every level with its "
                  "nearest position in (moves, |E(8)|, mover's pieces) within a caliper - moves "
                  "and pieces within 10%%, |E(8)| within max(1, 20%%) - or dropped: %llu anchors "
                  "tried\n",
                  opt.flatness_boards.front(), flat_moves_band.lo, flat_moves_band.hi,
                  flat_e_band.lo, flat_e_band.hi, flat_moves_band.how.c_str(),
                  flat_e_band.how.c_str(), static_cast<unsigned long long>(flat_anchors_tried));
      json += ",\"flatness\":{\"active\":" + std::to_string(opt.flatness_active) +
              ",\"games\":" + std::to_string(flat_seeds.size()) +
              ",\"window_percent\":" + std::to_string(opt.flat_window) + ",\"seeds\":[";
      for (std::size_t i = 0; i < flat_seeds.size(); ++i) {
        if (i != 0) json += ',';
        json += std::to_string(flat_seeds[i]);
      }
      json += "],\"anchors_tried\":" + std::to_string(flat_anchors_tried) +
              ",\"caliper\":{\"moves\":0.10,\"mover_pieces\":0.10,\"e_k8\":\"max(1, 20%)\"}" +
              ",\"anchor_moves_band\":" + json_band(flat_moves_band) +
              ",\"anchor_e_k8_band\":" + json_band(flat_e_band) + ",\"series\":[";
      bool first_kind = true;
      for (const FlatKind kind : {kFlatRaw, kFlatMatched, kFlatNoMove}) {
        std::vector<std::size_t> idx(levels, plan.flat.size());
        bool complete = true;
        for (std::size_t l = 0; l < levels; ++l) {
          const auto it = flat_index.find({static_cast<int>(kind), static_cast<int>(l)});
          if (it == flat_index.end()) {
            complete = false;
            break;
          }
          idx[l] = it->second;
        }
        if (!complete) {
          std::printf("   %s: not measured (a level has no position of this kind)\n",
                      kFlatKindName[kind]);
          continue;
        }
        std::printf("   -- %s%s\n", kFlatKindName[kind],
                    kind == kFlatRaw       ? ": every candidate (moves and |E| grow with N)"
                    : kind == kFlatMatched ? ": paired with the anchors on moves, |E(8)|, mover's pieces"
                                           : ": the action's moves played, no move left (Q4's "
                                             "Guide-literal reading): everything but move generation");
        std::printf("   %-26s", "level");
        for (std::size_t l = 0; l < levels; ++l) {
          std::printf(" %9s", ("N" + std::to_string(opt.flatness_boards[l])).c_str());
        }
        std::printf("\n");
        struct Row {
          const char* name;
          double Shape::*field;
        };
        const Row rows[] = {{"boards", &Shape::boards},
                            {"active timelines", &Shape::active},
                            {"moves", &Shape::moves},
                            {"|E(8)|", &Shape::e8},
                            {"scope boards (k=8)", &Shape::scope8},
                            {"relations", &Shape::relations},
                            {"mover's pieces (playable)", &Shape::mover_pieces}};
        for (const Row& row : rows) {
          std::printf("   %-26s", row.name);
          for (std::size_t l = 0; l < levels; ++l) {
            std::printf(" %9s", fixed(plan.flat[idx[l]].mean.*row.field, 1).c_str());
          }
          std::printf("\n");
        }
        std::printf("   %-26s", "positions");
        for (std::size_t l = 0; l < levels; ++l) {
          std::printf(" %9llu", static_cast<unsigned long long>(plan.flat[idx[l]].items.size()));
        }
        std::printf("\n");
        if (kind == kFlatMatched) {
          // How far the pairs are from their anchors: |moves| and |pieces| in
          // percent, |E(8)| in boards.
          std::printf("   %-26s", "pair mismatch: moves %");
          for (std::size_t l = 0; l < levels; ++l) {
            std::printf(" %9s", fixed(100 * flat_mismatch[l].moves, 1).c_str());
          }
          std::printf("\n   %-26s", "pair mismatch: |E(8)|");
          for (std::size_t l = 0; l < levels; ++l) {
            std::printf(" %9s", fixed(flat_mismatch[l].e8, 2).c_str());
          }
          std::printf("\n   %-26s", "pair mismatch: pieces %");
          for (std::size_t l = 0; l < levels; ++l) {
            std::printf(" %9s", fixed(100 * flat_mismatch[l].mover_pieces, 1).c_str());
          }
          std::printf("\n");
        }
        if (!first_kind) json += ',';
        first_kind = false;
        json += "{\"kind\":" + json_string(kFlatKindName[kind]) + ",\"levels\":[";
        for (std::size_t l = 0; l < levels; ++l) {
          if (l != 0) json += ',';
          json += "{\"N\":" + std::to_string(opt.flatness_boards[l]) +
                  ",\"positions\":" + std::to_string(plan.flat[idx[l]].items.size()) +
                  ",\"mean\":" + json_shape(plan.flat[idx[l]].mean) + "}";
        }
        json += "],\"results\":[";
        std::printf("   CPU us per expansion, median over rounds\n");
        bool first = true;
        for (const int k : flat_configs(kind)) {
          const Config& cfg = plan.configs[static_cast<std::size_t>(k)];
          std::vector<std::vector<double>> cost(levels);
          bool have = true;
          for (std::size_t l = 0; l < levels; ++l) {
            const auto it = flat_cpu_us.find({idx[l], k});
            if (it == flat_cpu_us.end()) {
              have = false;
              break;
            }
            cost[l] = it->second;
          }
          if (!have) continue;
          std::size_t rounds = cost[0].size();
          for (const std::vector<double>& v : cost) rounds = std::min(rounds, v.size());
          // Per round: the growth from the first level to the last, the same
          // per generated move, and (no moves) the slope per board.
          std::vector<double> growth, per_move, per_board_ns, share;
          const Shape& lo = plan.flat[idx[0]].mean;
          const Shape& hi = plan.flat[idx[levels - 1]].mean;
          for (std::size_t r = 0; r < rounds; ++r) {
            const double c0 = cost[0][r];
            const double c1 = cost[levels - 1][r];
            if (c0 > 0) growth.push_back(c1 / c0 - 1);
            if (kind != kFlatNoMove && c0 > 0 && lo.moves > 0 && hi.moves > 0) {
              per_move.push_back((c1 / hi.moves) / (c0 / lo.moves) - 1);
            }
            if (kind == kFlatNoMove) {
              std::vector<double> xs, ys;
              for (std::size_t l = 0; l < levels; ++l) {
                xs.push_back(plan.flat[idx[l]].mean.boards);
                ys.push_back(cost[l][r]);
              }
              const double slope = slope_of(xs, ys);  // us per board
              per_board_ns.push_back(1e3 * slope);
              // Against the matched scope+relations expansion at the top level,
              // the same round.
              const auto top = flat_index.find(
                  {static_cast<int>(kFlatMatched), static_cast<int>(levels - 1)});
              if (top != flat_index.end()) {
                const auto t = flat_cpu_us.find({top->second, kScopeRelations});
                if (t != flat_cpu_us.end() && r < t->second.size() && t->second[r] > 0) {
                  share.push_back(slope * (hi.boards - lo.boards) / t->second[r]);
                }
              }
            }
          }
          const Stats gs = stats_of(growth);
          std::printf("   %-26s", cfg.name.c_str());
          for (std::size_t l = 0; l < levels; ++l) {
            std::printf(" %9s", fixed(stats_of(cost[l]).median, 2).c_str());
          }
          std::printf("   growth %s [%s, %s]", signed_percent(gs.median).c_str(),
                      signed_percent(gs.min).c_str(), signed_percent(gs.max).c_str());
          if (kind == kFlatMatched) std::printf(" - %s", against(gs, 0.10).c_str());
          if (!per_move.empty()) {
            const Stats pm = stats_of(per_move);
            std::printf("; per move %s [%s, %s]", signed_percent(pm.median).c_str(),
                        signed_percent(pm.min).c_str(), signed_percent(pm.max).c_str());
          }
          if (!per_board_ns.empty()) {
            const Stats pb = stats_of(per_board_ns);
            std::printf("; %s ns per board [%s, %s]", fixed(pb.median, 2).c_str(),
                        fixed(pb.min, 2).c_str(), fixed(pb.max, 2).c_str());
            if (!share.empty()) {
              const Stats sh = stats_of(share);
              std::printf(" = %s [%s, %s] of a matched scope+relations expansion at N%d, over "
                          "N%d..N%d",
                          percent(sh.median).c_str(), percent(sh.min).c_str(),
                          percent(sh.max).c_str(), opt.flatness_boards.back(),
                          opt.flatness_boards.front(), opt.flatness_boards.back());
            }
          }
          std::printf("\n");
          if (!first) json += ',';
          first = false;
          json += "{\"config\":" + json_string(cfg.name) + ",\"cpu_us\":[";
          for (std::size_t l = 0; l < levels; ++l) {
            if (l != 0) json += ',';
            json += json_stats(stats_of(cost[l]), 3);
          }
          json += "],\"growth\":" + json_stats(gs, 4);
          if (!per_move.empty()) json += ",\"growth_per_move\":" + json_stats(stats_of(per_move), 4);
          if (!per_board_ns.empty()) {
            json += ",\"ns_per_board\":" + json_stats(stats_of(per_board_ns), 3);
            if (!share.empty()) {
              json += ",\"share_of_matched_scope_relations_at_top\":" +
                      json_stats(stats_of(share), 4);
            }
          }
          json += "}";
        }
        json += "]}";
      }
      json += "]}";
      std::printf("\n");
    }

    if (!summary.empty()) {
      std::printf("== 1 thread, stream (CPU-time rate = 1 / thread CPU time per expansion):\n");
      for (const std::string& line : summary) std::printf("%s\n", line.c_str());
      std::printf("\n");
    }
    json += ",\"sink\":" + std::to_string(g_sink.load()) + "}";

    if (opt.json_path.empty()) {
      std::printf("%s\n", json.c_str());
    } else {
      write_text(opt.json_path, json + "\n");
      std::printf("JSON summary: %s\n", opt.json_path.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-bench: %s\n", e.what());
    return 1;
  }
}
