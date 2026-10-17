// t5d-measure: the window measurement's raw data (design spec §10.4.2, §15.6
// "The window"; register M7) - for every position of 5dpgn records, the board
// count, the timelines, the moves, and |W(k)|, |E(k)| and src_extra(k) at
// k = 0, 4, 8, 16 (t5d/expand.hpp), against W(k) and against its bands alone
// (without the branch origins), and the position's fingerprint (H30) so that
// a record that repeats another's game can be told apart - and, with --turns
// and --turn-summary, the turn width of every action (register M22; design
// spec §8.2 "Measure first"; tools/turn_width.hpp). One JSON object per
// line, deterministic.
//
//   t5d-measure [options] [RECORD ...]      no RECORD, or "-": standard input
//
//   --no-moves-after-shift   a diagnostic: rules Q4's Guide-literal reading (no
//                         move once the present has shifted; the default, R8,
//                         allows moves until submission)
//   --moves-after-shift   accepted and ignored: R8 made it the default
//   --rules ENCODING      any ruleset: RuleOptions' canonical encoding
//   --name NAME           the name printed for a record read from standard input
//   --summary             after the records, the |E| curve: one line per N bucket
//   --legal               the legal moves only (M7's tighter |E| bound)
//   --turns               a line per action: its turn width (M22)
//   --turn-summary        after the records, the turn width per board-count range
//   --help, --version
//
// Each record is replayed as the gate adapter's parse mode does
// (t5d/pgn.hpp; engine README, "The gate adapter"), under the ruleset given.
// Lines:
//
//   {"event":"record","record":NAME,"game":...,"synthetic":B,"rules":...}
//   {"event":"position","record":NAME,"index":K,...}       at every action start
//   {"event":"turn","record":NAME,"index":K,...}            with --turns, per action
//   {"event":"end","record":NAME,"status":...,...}          how far, and why it stopped
//   {"event":"summary","bucket":...,...}                    with --summary, per N bucket
//   {"event":"turn_summary",...}                            with --turn-summary
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
// THE TURNS (--turns, --turn-summary; M22). Every action is played twice:
// by the replay itself (PgnReplay::play_next, which decides whether the
// record reads on), then move by move from the action's start - each written
// move resolved against the state the earlier ones reached
// (PgnReplay::resolve) and counted (tools/turn_width.hpp, which defines every
// count) - and the result, submitted without the royal-safety test as parse
// mode submits, must equal the replay's state, or the tool stops with an
// internal error. The turn summary's unit is the ACTION, read three ways:
//
//  * "strict" (the figures): every action of every record that replays to
//    the end, a record counted once - a record whose sequence of positions
//    (fingerprint and side to move, every action start and the last) equals
//    or is a prefix of another's is that record (a game recorded twice, or
//    cut short), and is left out;
//  * "grouped" (sensitivity): the records grouped into games as --summary
//    groups them (two endings of one game - a shared opening of 10 positions
//    or more and half the shorter record - are one game), each game's
//    distinct actions (by the positions before and after) counted once;
//  * "refused" (apart): the actions played before a record was refused.
//
// Per board-count range at the action's start (<50, 50-99, 100-199, 200+ -
// 6b's ranges; then ten-board buckets under 100; then every action pooled,
// context only): the actions and the games they come from; the distribution
// of mandatory boards at the start (1, 2, 3, 4+), the share with more than
// one and with more than two, their mean, median, p90 and max; the optional
// boards at the start; optional moves per action (0, 1, 2, 3+), their mean
// and the share with any - and the same by the other reading (not mandatory
// when made); moves per action (1, 2, 3, 4+), mean, median, p90, max; boards
// played; the shares with a jump ("travel"), a branch, a move after the
// shift, a rewind and a mandatory board released; and the counts of jumps
// between two mandatory boards, jumps from an optional board onto a mandatory
// one, boards released and gained, and moves on boards released or gained.
// The strict reading is also given per mandatory class (1, 2, 3, 4+) in each
// range. Synthetic records are left out, and counted.
//
// Moves are PSEUDO-LEGAL (the expansion's default, legal_moves off):
// every |E| is an upper bound. Records marked synthetic ("Synthetic" header)
// are random play, never evidence for the |E| verdict. The turn counts are
// exact either way: they count the moves the record made.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <set>
#include <stdexcept>
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
#include "turn_width.hpp"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif

namespace {

using t5d::gate::json_string;
using t5d::tools::TurnWidth;

constexpr int kWindowKs[] = {0, 4, 8, 16};

void print_usage(std::FILE* stream) {
  std::fputs(
      "t5d-measure " T5D_VERSION_STRING
      " - |W(k)|, |E(k)| and src_extra(k) at every position of 5dpgn records,\n"
      "and the turn width of every action.\n"
      "\n"
      "usage: t5d-measure [options] [RECORD ...]   (no RECORD, or -, reads standard input)\n"
      "  --no-moves-after-shift  a diagnostic: rules Q4's Guide-literal reading, no move\n"
      "                        once the present has shifted (the default, R8: moves until\n"
      "                        submission)\n"
      "  --moves-after-shift   accepted and ignored: R8 made it the default\n"
      "  --rules ENCODING      any ruleset (RuleOptions::encode; decimal or 0x hex)\n"
      "  --name NAME           the name of a record read from standard input\n"
      "  --summary             after the records, the |E| curve per N bucket (distinct\n"
      "                        positions and games; synthetic records left out)\n"
      "  --legal               the legal moves only (ExpandOptions::legal_moves, exact): E\n"
      "                        from the legal moves' destinations - M7's tighter bound\n"
      "  --turns               a line per action: its turn width (register M22) - the\n"
      "                        mandatory boards at its start, its moves, the optional ones\n"
      "  --turn-summary        after the records, the turn width per board-count range\n"
      "                        (<50, 50-99, 100-199, 200+, ten-board buckets under 100)\n"
      "  --help, --version\n"
      "\n"
      "One JSON object per line (see the engine README, \"t5d-measure\"). Moves are\n"
      "pseudo-legal unless --legal (then the record line says \"moves\":\"legal\").\n",
      stream);
}

void emit(const std::string& line) {
  std::fwrite(line.data(), 1, line.size(), stdout);
  std::fputc('\n', stdout);
}

std::string field(const char* name, long long value) {
  return std::string(",\"") + name + "\":" + std::to_string(value);
}

const char* side_name(t5d::Colour c) { return c == t5d::Colour::White ? "white" : "black"; }

std::string position_line(const std::string& name, bool synthetic, std::size_t index,
                          const t5d::GameState& state, const t5d::Expansion& x,
                          const std::string& fingerprint) {
  std::string line = "{\"event\":\"position\",\"record\":" + json_string(name);
  line += std::string(",\"synthetic\":") + (synthetic ? "true" : "false");
  line += field("index", static_cast<long long>(index));
  line += std::string(",\"side\":\"") + side_name(state.to_move()) + "\"";
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
  line += ",\"fingerprint\":\"" + fingerprint + "\"";
  line += '}';
  return line;
}

std::string turn_line(const std::string& name, bool synthetic, std::size_t index,
                      t5d::Colour side, const TurnWidth& w, const std::string& fingerprint,
                      const std::string& fingerprint_after) {
  std::string line = "{\"event\":\"turn\",\"record\":" + json_string(name);
  line += std::string(",\"synthetic\":") + (synthetic ? "true" : "false");
  line += field("index", static_cast<long long>(index));
  line += std::string(",\"side\":\"") + side_name(side) + "\"";
  line += field("boards", static_cast<long long>(w.boards));
  line += field("timelines", w.timelines);
  line += field("active", w.active);
  line += field("mandatory", w.mandatory);
  line += field("playable", w.playable);
  line += field("optional", w.optional);
  line += field("moves", w.moves);
  line += field("moves_from_mandatory", w.moves_from_mandatory);
  line += field("optional_moves", w.optional_moves);
  line += field("optional_when_made", w.optional_when_made);
  line += field("boards_played", w.boards_played);
  line += field("mandatory_played", w.mandatory_played);
  line += field("optional_played", w.optional_played);
  line += field("jumps", w.jumps);
  line += field("branches", w.branches);
  line += field("jumps_between_mandatory", w.jumps_between_mandatory);
  line += field("optional_onto_mandatory", w.optional_onto_mandatory);
  line += field("moves_after_shift", w.moves_after_shift);
  line += field("rewinds", w.rewinds);
  line += field("mandatory_released", w.mandatory_released);
  line += field("mandatory_gained", w.mandatory_gained);
  line += field("moves_on_released", w.moves_on_released);
  line += field("moves_on_gained", w.moves_on_gained);
  line += ",\"fingerprint\":\"" + fingerprint + "\"";
  line += ",\"fingerprint_after\":\"" + fingerprint_after + "\"";
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
bool g_legal = false;  // --legal: the legal moves only

// What the turn summary keeps of one action, and of one record.
struct TurnKept {
  TurnWidth width;
  std::string key;        // the action's start: fingerprint and side to move
  std::string key_after;  // the position after its submission
};

struct TurnRecord {
  std::string name;
  bool synthetic = false;
  bool replayed = false;          // read and played to the end ("ok")
  std::vector<std::string> keys;  // every position: each action's start, then the last
  std::vector<TurnKept> turns;    // every action played
};

std::vector<TurnRecord> g_turn_records;
bool g_turns = false;         // --turns
bool g_turn_summary = false;  // --turn-summary
std::size_t g_turns_checked = 0;

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

std::string position_key(const std::string& fingerprint, const t5d::GameState& state) {
  return fingerprint + (state.to_move() == t5d::Colour::White ? "w" : "b");
}

// The turn width of the action replay.play_next() has just played: its moves
// resolved and counted one by one from `start`, the state before it. The
// result, submitted as parse mode submits, must be the replay's own state.
// Any failure here is the tool's (the replay has accepted the action), so it
// is a std::logic_error, never a RecordError.
TurnWidth count_action(const t5d::PgnReplay& replay, const t5d::GameState& start,
                       std::size_t index) {
  const t5d::PgnAction& action = replay.record().actions[index];
  const std::string which = "action " + std::to_string(index + 1);
  t5d::tools::TurnCounter counter(start);
  try {
    for (const t5d::PgnMove& written : action.moves) {
      counter.apply(replay.resolve(counter.state(), written));
    }
  } catch (const t5d::RecordError& e) {
    throw std::logic_error(which + ": the replay played it, but resolving its moves one by one "
                                   "failed: " + std::string(e.what()));
  }
  t5d::GameState submitted = counter.state();
  submitted.submit(t5d::CheckRuling::none());  // parse mode: royal safety untested
  if (!(submitted == replay.state())) {
    throw std::logic_error(which + ": its moves played one by one do not reach the state the "
                                   "replay reached");
  }
  const TurnWidth w = counter.width();
  // T5: every turn begins with the present on the mover's colour (a board is
  // mandatory) and has a move.
  if (w.mandatory < 1 || w.moves < 1) {
    throw std::logic_error(which + ": no mandatory board at its start, or no move");
  }
  return w;
}

// One record: returns false on an internal error.
bool measure(const std::string& name, const std::string& text, const t5d::RuleOptions& rules) {
  t5d::PgnRecord record = t5d::read_pgn(text);
  const bool synthetic = t5d::tools::is_synthetic(record);
  const std::size_t in_record = record.actions.size();
  const bool truncated = record.unreadable.has_value();
  const bool turns = g_turns || g_turn_summary;
  std::size_t positions = 0;
  std::size_t accepted = 0;
  // Every record, so that one refused before its first position counts too.
  TurnRecord* turn_record = nullptr;
  if (g_turn_summary) {
    g_turn_records.push_back(TurnRecord{name, synthetic, false, {}, {}});
    turn_record = &g_turn_records.back();
  }
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
    head += std::string(",\"moves\":\"") + (g_legal ? "legal" : "pseudo-legal") + "\"";
    head += field("actions_in_record", static_cast<long long>(in_record));
    head += "}";
    emit(head);
    t5d::Expansion x;
    t5d::ExpandOptions options;
    options.square_features = t5d::SquareFeatures::None;  // counts only
    options.legal_moves = g_legal;                        // exact: none left unknown
    RecordKept* kept = nullptr;
    if (g_summary) {
      g_kept.push_back(RecordKept{name, synthetic, {}});
      kept = &g_kept.back();
    }
    std::string fingerprint = t5d::fingerprint(replay.state().multiverse()).fingerprint;
    for (;;) {
      const std::string key = position_key(fingerprint, replay.state());
      t5d::expand(replay.state(), options, x);
      emit(position_line(name, synthetic, replay.actions_played(), replay.state(), x, fingerprint));
      if (kept != nullptr) {
        Kept k;
        k.boards = x.board_count();
        k.active = x.active_timeline_count();
        k.w8 = x.window_size(8);
        k.w_bands8 = x.window_size(8, t5d::Window::BandsOnly);
        for (std::size_t i = 0; i < 4; ++i) k.e[i] = x.e_count(kWindowKs[i]);
        k.src_only8 = x.src_only_count(8);
        k.key = key;
        kept->positions.push_back(std::move(k));
      }
      if (turn_record != nullptr) turn_record->keys.push_back(key);
      ++positions;
      if (replay.finished()) break;
      if (!turns) {
        replay.play_next();
        accepted = replay.actions_played();
        fingerprint = t5d::fingerprint(replay.state().multiverse()).fingerprint;
        continue;
      }
      const std::size_t index = replay.actions_played();
      const t5d::GameState start = replay.state();
      replay.play_next();
      accepted = replay.actions_played();
      const TurnWidth w = count_action(replay, start, index);
      ++g_turns_checked;
      const std::string after = t5d::fingerprint(replay.state().multiverse()).fingerprint;
      if (g_turns) emit(turn_line(name, synthetic, index, start.to_move(), w, fingerprint, after));
      if (turn_record != nullptr) {
        turn_record->turns.push_back(TurnKept{w, key, position_key(after, replay.state())});
      }
      fingerprint = after;
    }
    if (truncated) {
      emit(end_line(name, "rejected", positions, accepted, in_record, true,
                    t5d::fault_name(t5d::RecordFault::Unreadable), *replay.record().unreadable));
    } else {
      if (turn_record != nullptr) turn_record->replayed = true;
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

// ---- the turn summary (--turn-summary; register M22) --------------------------------------

// A board-count range at the action's start: [lo, hi).
struct BoardRange {
  const char* label;
  std::size_t lo;
  std::size_t hi;
};

// 6b's and 6b''s ranges (M22 names none: build-log spec issue 141).
constexpr BoardRange kMainRanges[] = {
    {"<50", 0, 50}, {"50-99", 50, 100}, {"100-199", 100, 200}, {"200+", 200, SIZE_MAX}};
constexpr BoardRange kFineRanges[] = {{"1-9", 0, 10},   {"10-19", 10, 20}, {"20-29", 20, 30},
                                      {"30-39", 30, 40}, {"40-49", 40, 50}, {"50-59", 50, 60},
                                      {"60-69", 60, 70}, {"70-79", 70, 80}, {"80-89", 80, 90},
                                      {"90-99", 90, 100}};
constexpr BoardRange kPooled = {"all", 0, SIZE_MAX};

// One action as a reading counts it, and the game (or record) it is from.
struct Counted {
  const TurnWidth* width;
  std::size_t game;
};

std::string share(std::size_t part, std::size_t whole) {
  return number(whole == 0 ? 0.0 : static_cast<double>(part) / static_cast<double>(whole));
}

// ,"NAME_mean":..,"NAME_median":..,"NAME_p90":..,"NAME_max":.. of a non-empty set.
std::string distribution(const std::vector<double>& v, const char* name) {
  double sum = 0;
  for (const double x : v) sum += x;
  const std::string n = std::string(",\"") + name;
  return n + "_mean\":" + number(sum / static_cast<double>(v.size())) + n +
         "_median\":" + number(quantile(v, 0.5)) + n + "_p90\":" + number(quantile(v, 0.9)) + n +
         "_max\":" + number(*std::max_element(v.begin(), v.end()));
}

// The statistics of a set of actions: the fields after "actions" and its
// games (see the file comment).
std::string turn_stats(const std::vector<Counted>& actions) {
  const std::size_t n = actions.size();
  std::size_t mandatory_class[4] = {}, optional_class[4] = {}, moves_class[4] = {};
  std::vector<double> mandatory, optional_boards, moves;
  std::size_t over_1 = 0, over_2 = 0, with_optional = 0, with_optional_made = 0;
  std::size_t with_travel = 0, with_branch = 0, with_after_shift = 0, rewound = 0;
  std::size_t with_released = 0;
  long long sum_optional = 0, sum_optional_made = 0, sum_played = 0, sum_after_shift = 0;
  long long between = 0, onto = 0, released = 0, gained = 0, on_released = 0, on_gained = 0;
  for (const Counted& c : actions) {
    const TurnWidth& w = *c.width;
    ++mandatory_class[std::min(w.mandatory, 4) - 1];  // w.mandatory >= 1 (count_action)
    ++optional_class[std::min(w.optional_moves, 3)];
    ++moves_class[std::min(w.moves, 4) - 1];  // w.moves >= 1
    mandatory.push_back(w.mandatory);
    optional_boards.push_back(w.optional);
    moves.push_back(w.moves);
    over_1 += w.mandatory > 1 ? 1 : 0;
    over_2 += w.mandatory > 2 ? 1 : 0;
    with_optional += w.optional_moves > 0 ? 1 : 0;
    with_optional_made += w.optional_when_made > 0 ? 1 : 0;
    with_travel += w.jumps > 0 ? 1 : 0;
    with_branch += w.branches > 0 ? 1 : 0;
    with_after_shift += w.moves_after_shift > 0 ? 1 : 0;
    rewound += w.rewinds > 0 ? 1 : 0;
    with_released += w.mandatory_released > 0 ? 1 : 0;
    sum_optional += w.optional_moves;
    sum_optional_made += w.optional_when_made;
    sum_played += w.boards_played;
    sum_after_shift += w.moves_after_shift;
    between += w.jumps_between_mandatory;
    onto += w.optional_onto_mandatory;
    released += w.mandatory_released;
    gained += w.mandatory_gained;
    on_released += w.moves_on_released;
    on_gained += w.moves_on_gained;
  }
  if (n == 0) return "";
  const double dn = static_cast<double>(n);
  const auto mean = [dn](long long sum) { return number(static_cast<double>(sum) / dn); };
  std::string s;
  s += ",\"mandatory\":{\"1\":" + std::to_string(mandatory_class[0]) +
       ",\"2\":" + std::to_string(mandatory_class[1]) +
       ",\"3\":" + std::to_string(mandatory_class[2]) +
       ",\"4+\":" + std::to_string(mandatory_class[3]) + "}";
  s += ",\"mandatory_over_1\":" + share(over_1, n) + ",\"mandatory_over_2\":" + share(over_2, n);
  s += distribution(mandatory, "mandatory");
  s += distribution(optional_boards, "optional_boards");
  s += ",\"optional_moves\":{\"0\":" + std::to_string(optional_class[0]) +
       ",\"1\":" + std::to_string(optional_class[1]) +
       ",\"2\":" + std::to_string(optional_class[2]) +
       ",\"3+\":" + std::to_string(optional_class[3]) + "}";
  s += ",\"optional_moves_mean\":" + mean(sum_optional) +
       ",\"with_optional_moves\":" + share(with_optional, n);
  s += ",\"optional_when_made_mean\":" + mean(sum_optional_made) +
       ",\"with_optional_when_made\":" + share(with_optional_made, n);
  s += ",\"moves\":{\"1\":" + std::to_string(moves_class[0]) +
       ",\"2\":" + std::to_string(moves_class[1]) + ",\"3\":" + std::to_string(moves_class[2]) +
       ",\"4+\":" + std::to_string(moves_class[3]) + "}";
  s += distribution(moves, "moves");
  s += ",\"boards_played_mean\":" + mean(sum_played);
  s += ",\"with_travel\":" + share(with_travel, n) + ",\"with_branch\":" + share(with_branch, n);
  s += ",\"with_moves_after_shift\":" + share(with_after_shift, n) +
       ",\"moves_after_shift_mean\":" + mean(sum_after_shift);
  s += ",\"rewound\":" + share(rewound, n);
  s += ",\"jumps_between_mandatory\":" + std::to_string(between) +
       ",\"optional_onto_mandatory\":" + std::to_string(onto);
  s += ",\"with_mandatory_released\":" + share(with_released, n) +
       ",\"mandatory_released\":" + std::to_string(released) +
       ",\"mandatory_gained\":" + std::to_string(gained) +
       ",\"moves_on_released\":" + std::to_string(on_released) +
       ",\"moves_on_gained\":" + std::to_string(on_gained);
  return s;
}

// One summary line: the actions of `all` whose board count lies in `r`.
void emit_turn_summary(const char* event, const char* reading, const BoardRange& r,
                       const char* mandatory_class, const char* unit,
                       const std::vector<Counted>& all) {
  std::vector<Counted> in;
  std::set<std::size_t> games;
  for (const Counted& c : all) {
    if (c.width->boards < r.lo || c.width->boards >= r.hi) continue;
    if (mandatory_class != nullptr) {
      const int m = c.width->mandatory;
      const bool four_plus = std::strcmp(mandatory_class, "4+") == 0;
      if (four_plus ? m < 4 : std::to_string(m) != mandatory_class) continue;
    }
    in.push_back(c);
    games.insert(c.game);
  }
  std::string line = std::string("{\"event\":\"") + event + "\",\"reading\":\"" + reading +
                     "\",\"range\":\"" + r.label + "\"";
  if (mandatory_class != nullptr) {
    line += std::string(",\"mandatory_class\":\"") + mandatory_class + "\"";
  }
  line += field("actions", static_cast<long long>(in.size()));
  line += field(unit, static_cast<long long>(games.size()));
  line += turn_stats(in);
  emit(line + "}");
}

bool is_prefix(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

void print_turn_summary() {
  std::vector<std::size_t> replayed, refused;
  std::size_t synthetic_actions = 0;
  for (std::size_t r = 0; r < g_turn_records.size(); ++r) {
    const TurnRecord& t = g_turn_records[r];
    if (t.synthetic) {
      synthetic_actions += t.turns.size();
    } else {
      (t.replayed ? replayed : refused).push_back(r);
    }
  }
  const auto longest_first = [](std::vector<std::size_t> v) {
    std::stable_sort(v.begin(), v.end(), [](std::size_t a, std::size_t b) {
      return g_turn_records[a].keys.size() > g_turn_records[b].keys.size();
    });
    return v;
  };

  // Strict: a record that equals, or is a prefix of, another is that record.
  std::vector<std::size_t> strict;
  std::vector<std::pair<std::size_t, std::size_t>> duplicates;  // (record, the one it repeats)
  for (const std::size_t r : longest_first(replayed)) {
    bool repeat = false;
    for (const std::size_t g : strict) {
      if (is_prefix(g_turn_records[r].keys, g_turn_records[g].keys)) {
        duplicates.emplace_back(r, g);
        repeat = true;
        break;
      }
    }
    if (!repeat) strict.push_back(r);
  }
  std::sort(strict.begin(), strict.end());
  std::sort(duplicates.begin(), duplicates.end());

  // Grouped: --summary's games (a prefix, or a shared opening of 10 positions
  // or more and half the shorter record), joined to the longest record.
  const auto same_game = [](const TurnRecord& a, const TurnRecord& b) {
    const std::size_t shorter = std::min(a.keys.size(), b.keys.size());
    std::size_t shared = 0;
    while (shared < shorter && a.keys[shared] == b.keys[shared]) ++shared;
    return shared == shorter || (shared >= 10 && 2 * shared >= shorter);
  };
  std::vector<std::vector<std::size_t>> groups;
  for (const std::size_t r : longest_first(replayed)) {
    bool joined = false;
    for (std::vector<std::size_t>& group : groups) {
      if (same_game(g_turn_records[r], g_turn_records[group[0]])) {
        group.push_back(r);
        joined = true;
        break;
      }
    }
    if (!joined) groups.push_back({r});
  }
  std::sort(groups.begin(), groups.end());

  std::string head = "{\"event\":\"turn_summary_records\"";
  head += field("records", static_cast<long long>(g_turn_records.size()));
  head += field("replayed", static_cast<long long>(replayed.size()));
  head += field("refused", static_cast<long long>(refused.size()));
  head += field("synthetic_actions_left_out", static_cast<long long>(synthetic_actions));
  head += field("actions_checked", static_cast<long long>(g_turns_checked));
  head += field("strict_games", static_cast<long long>(strict.size()));
  head += field("grouped_games", static_cast<long long>(groups.size()));
  head += ",\"strict_repeats\":[";
  for (std::size_t i = 0; i < duplicates.size(); ++i) {
    if (i != 0) head += ',';
    head += "{\"record\":" + json_string(g_turn_records[duplicates[i].first].name) +
            ",\"repeats\":" + json_string(g_turn_records[duplicates[i].second].name) + "}";
  }
  emit(head + "]}");
  for (const std::vector<std::size_t>& group : groups) {
    if (group.size() < 2) continue;
    std::string line = "{\"event\":\"turn_summary_group\",\"record\":" +
                       json_string(g_turn_records[group[0]].name) + ",\"also\":[";
    for (std::size_t i = 1; i < group.size(); ++i) {
      if (i != 1) line += ',';
      line += json_string(g_turn_records[group[i]].name);
    }
    emit(line + "]}");
  }

  std::vector<Counted> strict_actions, grouped_actions, refused_actions;
  for (std::size_t g = 0; g < strict.size(); ++g) {
    for (const TurnKept& t : g_turn_records[strict[g]].turns) {
      strict_actions.push_back({&t.width, g});
    }
  }
  for (std::size_t g = 0; g < groups.size(); ++g) {
    std::set<std::pair<std::string, std::string>> seen;  // an action: the positions around it
    for (const std::size_t r : groups[g]) {
      for (const TurnKept& t : g_turn_records[r].turns) {
        if (seen.insert({t.key, t.key_after}).second) grouped_actions.push_back({&t.width, g});
      }
    }
  }
  for (std::size_t i = 0; i < refused.size(); ++i) {
    for (const TurnKept& t : g_turn_records[refused[i]].turns) {
      refused_actions.push_back({&t.width, i});
    }
  }

  for (const BoardRange& r : kMainRanges) {
    emit_turn_summary("turn_summary", "strict", r, nullptr, "games", strict_actions);
  }
  for (const BoardRange& r : kFineRanges) {
    emit_turn_summary("turn_summary", "strict", r, nullptr, "games", strict_actions);
  }
  emit_turn_summary("turn_summary", "strict", kPooled, nullptr, "games", strict_actions);
  for (const BoardRange& r : kMainRanges) {
    for (const char* m : {"1", "2", "3", "4+"}) {
      emit_turn_summary("turn_summary_by_mandatory", "strict", r, m, "games", strict_actions);
    }
  }
  for (const char* m : {"1", "2", "3", "4+"}) {
    emit_turn_summary("turn_summary_by_mandatory", "strict", kPooled, m, "games", strict_actions);
  }
  for (const BoardRange& r : kMainRanges) {
    emit_turn_summary("turn_summary", "grouped", r, nullptr, "games", grouped_actions);
  }
  emit_turn_summary("turn_summary", "grouped", kPooled, nullptr, "games", grouped_actions);
  for (const BoardRange& r : kMainRanges) {
    emit_turn_summary("turn_summary", "refused", r, nullptr, "records", refused_actions);
  }
  emit_turn_summary("turn_summary", "refused", kPooled, nullptr, "records", refused_actions);
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
      if (std::strcmp(arg, "--legal") == 0) {
        g_legal = true;
        continue;
      }
      if (std::strcmp(arg, "--turns") == 0) {
        g_turns = true;
        continue;
      }
      if (std::strcmp(arg, "--turn-summary") == 0) {
        g_turn_summary = true;
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
  if (g_turn_summary) print_turn_summary();
  return ok ? 0 : 1;
}
