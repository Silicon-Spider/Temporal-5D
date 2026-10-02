// t5d-measure: the window measurement's raw data (design spec §10.4.2, §15.6
// "The window"; register M7) - for every position of 5dpgn records, the board
// count, the timelines, the moves, and |W(k)|, |E(k)| and src_extra(k) at
// k = 0, 4, 8, 16 (t5d/expand.hpp), against W(k) and against its bands alone
// (without the branch origins), and the position's fingerprint (H30) so that
// a record that repeats another's game can be told apart. One JSON object per
// line, deterministic.
//
//   t5d-measure [options] [RECORD ...]      no RECORD, or "-": standard input
//
//   --moves-after-shift   rules Q4's other reading (moves after the present
//                         has shifted, until submission)
//   --rules ENCODING      any ruleset: RuleOptions' canonical encoding
//   --name NAME           the name printed for a record read from standard input
//   --summary             after the records, the |E| curve: one line per N bucket
//   --help, --version
//
// Each record is replayed as the gate adapter's parse mode does
// (t5d/pgn.hpp; engine README, "The gate adapter"), under the ruleset given.
// Lines:
//
//   {"event":"record","record":NAME,"game":...,"synthetic":B,"rules":...}
//   {"event":"position","record":NAME,"index":K,...}       at every action start
//   {"event":"end","record":NAME,"status":...,...}          how far, and why it stopped
//   {"event":"summary","bucket":...,...}                    with --summary, per N bucket
//
// THE SUMMARY (--summary) is the curve M7 reads, made so that a game counts
// once: positions are distinct by fingerprint (H30) and side to move - a
// position reached in several records (a standard start, a common opening, a
// game recorded twice) is one position - and records are grouped into
// games: a record whose sequence of position fingerprints is a prefix of
// another's, or that shares with another its first 10 positions or more and
// at least half of the shorter of the two (two endings of one game), is that
// game. Per N bucket (1-49, 50-99, 100-159, 160-239, 240-399, 400-599,
// 600+): the distinct positions and games, median active timelines and
// |W(8)| (with and without the branch origins), |E(k)| at k = 0, 4, 8, 16 as
// median, p95 and max over the distinct positions (quantiles interpolated
// linearly between order statistics), the spread of the games' own medians of
// |E(8)| and |E(4)|, and how many positions have a source outside W(8) and
// E(8) (src_only). Synthetic records are left out of it, and counted.
//
// Moves are PSEUDO-LEGAL (the expansion's default, legal_moves off):
// every |E| is an upper bound. Records marked synthetic ("Synthetic" header)
// are random play, never evidence for the |E| verdict.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "t5d/expand.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/gate.hpp"
#include "t5d/pgn.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "tool_support.hpp"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif

namespace {

using t5d::gate::json_string;

constexpr int kWindowKs[] = {0, 4, 8, 16};

void print_usage(std::FILE* stream) {
  std::fputs(
      "t5d-measure " T5D_VERSION_STRING
      " - |W(k)|, |E(k)| and src_extra(k) at every position of 5dpgn records.\n"
      "\n"
      "usage: t5d-measure [options] [RECORD ...]   (no RECORD, or -, reads standard input)\n"
      "  --moves-after-shift   rules Q4's other reading: moves until submission\n"
      "  --rules ENCODING      any ruleset (RuleOptions::encode; decimal or 0x hex)\n"
      "  --name NAME           the name of a record read from standard input\n"
      "  --summary             after the records, the |E| curve per N bucket (distinct\n"
      "                        positions and games; synthetic records left out)\n"
      "  --help, --version\n"
      "\n"
      "One JSON object per line (see the engine README, \"t5d-measure\"). Moves are\n"
      "pseudo-legal (the expansion's default): every |E| is an upper bound.\n",
      stream);
}

void emit(const std::string& line) {
  std::fwrite(line.data(), 1, line.size(), stdout);
  std::fputc('\n', stdout);
}

std::string field(const char* name, long long value) {
  return std::string(",\"") + name + "\":" + std::to_string(value);
}

std::string position_line(const std::string& name, bool synthetic, std::size_t index,
                          const t5d::GameState& state, const t5d::Expansion& x) {
  std::string line = "{\"event\":\"position\",\"record\":" + json_string(name);
  line += std::string(",\"synthetic\":") + (synthetic ? "true" : "false");
  line += field("index", static_cast<long long>(index));
  line += std::string(",\"side\":\"") +
          (state.to_move() == t5d::Colour::White ? "white" : "black") + "\"";
  line += field("boards", static_cast<long long>(x.board_count()));
  line += field("timelines", x.timeline_count());
  line += field("active", x.active_timeline_count());
  line += field("present", x.present());
  line += field("present_turn", t5d::turn_of(x.present()));
  line += field("moves", static_cast<long long>(x.move_records().size()));
  line += field("move_edges", static_cast<long long>(x.move_edge_count()));
  line += field("travel_destinations", static_cast<long long>(x.travel_destinations().size()));
  line += field("sources", static_cast<long long>(x.source_boards().size()));
  line += field("origins", static_cast<long long>(x.origin_count()));
  line += field("origins_unknown", x.origins_unknown());
  for (const int k : kWindowKs) {
    const std::string suffix = "_k" + std::to_string(k);
    line += field(("w" + suffix).c_str(), static_cast<long long>(x.window_size(k)));
    line += field(("e" + suffix).c_str(), static_cast<long long>(x.e_count(k)));
    line += field(("src_extra" + suffix).c_str(), static_cast<long long>(x.src_extra_count(k)));
    line += field(("src_only" + suffix).c_str(), static_cast<long long>(x.src_only_count(k)));
    // The same against the bands alone (W(k) without the branch origins).
    const t5d::Window bands = t5d::Window::BandsOnly;
    line += field(("w_bands" + suffix).c_str(), static_cast<long long>(x.window_size(k, bands)));
    line += field(("e_bands" + suffix).c_str(), static_cast<long long>(x.e_count(k, bands)));
    line += field(("src_only_bands" + suffix).c_str(),
                  static_cast<long long>(x.src_only_count(k, bands)));
  }
  line += ",\"fingerprint\":\"" + t5d::fingerprint(state.multiverse()).fingerprint + "\"";
  line += '}';
  return line;
}

// What the summary keeps of one position.
struct Kept {
  std::size_t boards = 0;
  int active = 0;
  std::size_t w8 = 0;
  std::size_t w_bands8 = 0;
  std::size_t e[4] = {};  // k = 0, 4, 8, 16
  std::size_t src_only8 = 0;
  std::string key;  // fingerprint and side to move
};

struct RecordKept {
  std::string name;
  bool synthetic = false;
  std::vector<Kept> positions;
};

std::vector<RecordKept> g_kept;
bool g_summary = false;

std::string end_line(const std::string& name, const char* status, std::size_t positions,
                     std::size_t accepted, std::size_t in_record, bool truncated,
                     const char* reason, const std::string& detail) {
  std::string line = "{\"event\":\"end\",\"record\":" + json_string(name);
  line += std::string(",\"status\":\"") + status + "\"";
  line += field("positions", static_cast<long long>(positions));
  line += field("actions_accepted", static_cast<long long>(accepted));
  line += field("actions_in_record", static_cast<long long>(in_record));
  line += std::string(",\"truncated\":") + (truncated ? "true" : "false");
  if (reason != nullptr) line += std::string(",\"reason\":\"") + reason + "\"";
  if (!detail.empty()) line += ",\"detail\":" + json_string(detail);
  line += '}';
  return line;
}

// One record: returns false on an internal error.
bool measure(const std::string& name, const std::string& text, const t5d::RuleOptions& rules) {
  t5d::PgnRecord record = t5d::read_pgn(text);
  const bool synthetic = t5d::tools::is_synthetic(record);
  const std::size_t in_record = record.actions.size();
  const bool truncated = record.unreadable.has_value();
  std::size_t positions = 0;
  std::size_t accepted = 0;
  try {
    t5d::PgnReplay replay(std::move(record), t5d::PgnOptions{}, rules);
    t5d::tools::parse_mode(replay);
    std::string head = "{\"event\":\"record\",\"record\":" + json_string(name);
    head += ",\"game\":" + json_string(replay.game());
    head += std::string(",\"synthetic\":") + (synthetic ? "true" : "false");
    head += ",\"rules\":" + json_string(t5d::tools::ruleset_text(replay.state().rules()));
    char encoding[24];
    std::snprintf(encoding, sizeof encoding, "0x%llx",
                  static_cast<unsigned long long>(replay.state().rules().encode()));
    head += std::string(",\"rules_encoding\":\"") + encoding + "\"";
    head += field("actions_in_record", static_cast<long long>(in_record));
    head += "}";
    emit(head);
    t5d::Expansion x;
    t5d::ExpandOptions options;
    options.square_features = t5d::SquareFeatures::None;  // counts only
    RecordKept* kept = nullptr;
    if (g_summary) {
      g_kept.push_back(RecordKept{name, synthetic, {}});
      kept = &g_kept.back();
    }
    for (;;) {
      t5d::expand(replay.state(), options, x);
      emit(position_line(name, synthetic, replay.actions_played(), replay.state(), x));
      if (kept != nullptr) {
        Kept k;
        k.boards = x.board_count();
        k.active = x.active_timeline_count();
        k.w8 = x.window_size(8);
        k.w_bands8 = x.window_size(8, t5d::Window::BandsOnly);
        for (std::size_t i = 0; i < 4; ++i) k.e[i] = x.e_count(kWindowKs[i]);
        k.src_only8 = x.src_only_count(8);
        k.key = t5d::fingerprint(replay.state().multiverse()).fingerprint +
                (replay.state().to_move() == t5d::Colour::White ? "w" : "b");
        kept->positions.push_back(std::move(k));
      }
      ++positions;
      if (replay.finished()) break;
      replay.play_next();
      accepted = replay.actions_played();
    }
    if (truncated) {
      emit(end_line(name, "rejected", positions, accepted, in_record, true,
                    t5d::fault_name(t5d::RecordFault::Unreadable), *replay.record().unreadable));
    } else {
      emit(end_line(name, "ok", positions, accepted, in_record, false, nullptr, ""));
    }
    return true;
  } catch (const t5d::RecordError& e) {
    const bool unsupported = e.fault() == t5d::RecordFault::Unsupported && positions == 0;
    emit(end_line(name, unsupported ? "unsupported" : "rejected", positions, accepted, in_record,
                  truncated, t5d::fault_name(e.fault()), e.what()));
    return true;
  } catch (const std::exception& e) {
    emit(end_line(name, "error", positions, accepted, in_record, truncated, "error", e.what()));
    return false;
  }
}

// Linear interpolation between order statistics (numpy's default).
double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const double at = q * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(at));
  const auto hi = static_cast<std::size_t>(std::ceil(at));
  return v[lo] + (v[hi] - v[lo]) * (at - static_cast<double>(lo));
}

std::string number(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4g", v);
  return buf;
}

std::string spread(const std::vector<double>& v) {
  if (v.empty()) return "null";
  return "{\"median\":" + number(quantile(v, 0.5)) + ",\"p95\":" + number(quantile(v, 0.95)) +
         ",\"max\":" + number(*std::max_element(v.begin(), v.end())) + ",\"n\":" +
         std::to_string(v.size()) + "}";
}

std::string range(const std::vector<double>& v) {
  if (v.empty()) return "null";
  return "{\"median\":" + number(quantile(v, 0.5)) + ",\"min\":" +
         number(*std::min_element(v.begin(), v.end())) + ",\"max\":" +
         number(*std::max_element(v.begin(), v.end())) + ",\"n\":" + std::to_string(v.size()) +
         "}";
}

void print_summary() {
  // Games: records grouped by shared prefixes of their fingerprint sequences.
  std::vector<std::size_t> real;
  std::size_t synthetic_positions = 0;
  for (std::size_t r = 0; r < g_kept.size(); ++r) {
    if (g_kept[r].synthetic) {
      synthetic_positions += g_kept[r].positions.size();
    } else if (!g_kept[r].positions.empty()) {
      real.push_back(r);
    }
  }
  // Longest first, each record joins the first game it is a prefix of, or
  // shares a long prefix with (never through another record: two games are
  // not one because a short record is a prefix of both).
  std::stable_sort(real.begin(), real.end(), [&](std::size_t a, std::size_t b) {
    return g_kept[a].positions.size() > g_kept[b].positions.size();
  });
  const auto same_game = [&](const RecordKept& a, const RecordKept& b) {
    const std::size_t shorter = std::min(a.positions.size(), b.positions.size());
    std::size_t shared = 0;
    while (shared < shorter && a.positions[shared].key == b.positions[shared].key) ++shared;
    return shared == shorter || (shared >= 10 && 2 * shared >= shorter);
  };
  std::vector<std::vector<std::size_t>> games;
  for (const std::size_t r : real) {
    bool joined = false;
    for (std::vector<std::size_t>& game : games) {
      if (same_game(g_kept[r], g_kept[game[0]])) {
        game.push_back(r);
        joined = true;
        break;
      }
    }
    if (!joined) games.push_back({r});
  }
  std::sort(games.begin(), games.end());
  for (const std::vector<std::size_t>& game : games) {
    if (game.size() < 2) continue;
    std::string line = "{\"event\":\"summary_game\",\"record\":" + json_string(g_kept[game[0]].name) +
                       ",\"also\":[";
    for (std::size_t i = 1; i < game.size(); ++i) {
      if (i != 1) line += ',';
      line += json_string(g_kept[game[i]].name);
    }
    emit(line + "]}");
  }

  static const std::size_t kEdges[] = {1, 50, 100, 160, 240, 400, 600};
  const auto bucket_of = [](std::size_t boards) {
    std::size_t b = 0;
    for (std::size_t i = 0; i < std::size(kEdges); ++i) {
      if (boards >= kEdges[i]) b = i;
    }
    return b;
  };
  for (std::size_t bucket = 0; bucket < std::size(kEdges); ++bucket) {
    std::set<std::string> seen;
    std::vector<double> active, w8, wb8, e[4], src_only;
    std::vector<double> game_e8, game_e4;
    std::vector<std::string> names;
    std::size_t positions = 0;
    for (const std::vector<std::size_t>& game : games) {
      // The game's own distinct positions in the bucket (its records' union).
      std::vector<double> mine8, mine4;
      std::set<std::string> own;
      for (const std::size_t r : game) {
        for (const Kept& k : g_kept[r].positions) {
          if (bucket_of(k.boards) != bucket || !own.insert(k.key).second) continue;
          mine8.push_back(static_cast<double>(k.e[2]));
          mine4.push_back(static_cast<double>(k.e[1]));
          if (!seen.insert(k.key).second) continue;  // a position counts once
          ++positions;
          active.push_back(k.active);
          w8.push_back(static_cast<double>(k.w8));
          wb8.push_back(static_cast<double>(k.w_bands8));
          for (std::size_t i = 0; i < 4; ++i) e[i].push_back(static_cast<double>(k.e[i]));
          src_only.push_back(k.src_only8 > 0 ? 1 : 0);
        }
      }
      if (!mine8.empty()) {
        names.push_back(g_kept[game[0]].name);
        game_e8.push_back(quantile(mine8, 0.5));
        game_e4.push_back(quantile(mine4, 0.5));
      }
    }
    std::string line = "{\"event\":\"summary\",\"bucket\":\"" + std::to_string(kEdges[bucket]) +
                       (bucket + 1 < std::size(kEdges) ? "-" + std::to_string(kEdges[bucket + 1] - 1)
                                                       : std::string("+")) +
                       "\",\"positions\":" + std::to_string(positions) +
                       ",\"games\":" + std::to_string(names.size()) + ",\"games_named_by\":[";
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (i != 0) line += ',';
      line += json_string(names[i]);
    }
    line += "]";
    if (positions > 0) {
      line += ",\"active_median\":" + number(quantile(active, 0.5)) +
              ",\"w_k8_median\":" + number(quantile(w8, 0.5)) +
              ",\"w_bands_k8_median\":" + number(quantile(wb8, 0.5));
      for (std::size_t i = 0; i < 4; ++i) {
        line += ",\"e_k" + std::to_string(kWindowKs[i]) + "\":" + spread(e[i]);
      }
      line += ",\"e_k8_game_medians\":" + range(game_e8) +
              ",\"e_k4_game_medians\":" + range(game_e4);
      double with_src_only = 0;
      for (const double v : src_only) with_src_only += v;
      line += ",\"positions_with_src_only_k8\":" + number(with_src_only);
    }
    line += ",\"synthetic_positions_left_out\":" + std::to_string(synthetic_positions) + "}";
    emit(line);
  }
}

}  // namespace

int main(int argc, char** argv) {
  t5d::RuleOptions rules;
  std::vector<std::string> paths;
  std::string stdin_name = "stdin";
  try {
    for (int i = 1; i < argc; ++i) {
      const char* arg = argv[i];
      if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
        print_usage(stdout);
        return 0;
      }
      if (std::strcmp(arg, "--version") == 0) {
        std::puts(T5D_VERSION_STRING);
        return 0;
      }
      if (const int used = t5d::tools::ruleset_option(argc, argv, i, rules); used > 0) {
        i += used - 1;
        continue;
      }
      if (std::strcmp(arg, "--name") == 0 && i + 1 < argc) {
        stdin_name = argv[++i];
        continue;
      }
      if (std::strcmp(arg, "--summary") == 0) {
        g_summary = true;
        continue;
      }
      if (arg[0] == '-' && arg[1] != '\0') {
        std::fprintf(stderr, "t5d-measure: unknown argument \"%s\"\n\n", arg);
        print_usage(stderr);
        return 2;
      }
      paths.emplace_back(arg);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-measure: %s\n", e.what());
    return 2;
  }
  if (paths.empty()) paths.emplace_back("-");
#if defined(_WIN32)
  _setmode(_fileno(stdout), _O_BINARY);  // the same bytes on every platform
#endif

  bool ok = true;
  for (const std::string& path : paths) {
    std::string text;
    try {
      text = path == "-" ? t5d::tools::read_stdin() : t5d::tools::read_file(path);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "t5d-measure: %s\n", e.what());
      return 2;
    }
    ok = measure(path == "-" ? stdin_name : path, text, rules) && ok;
    std::fflush(stdout);
  }
  if (g_summary) print_summary();
  return ok ? 0 : 1;
}
