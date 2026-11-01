// t5d-legality-cost: what the search's exact move legality costs, position by
// position (build-log issues 406-408; register E5a's tiers, H37; spec §8.1
// "search verifies an unknown move the first time it selects it", §13's
// legality paragraph, §15.3's legality row).
//
//   t5d-legality-cost [options] RECORD ...
//
// THE POSITIONS: each record's last position (a custom 5DFEN set-up of no
// action, or a game), then - if the record has a LegalityPath header - the
// moves it lists, in the export notation t5d::to_string writes, "|" for a
// submission: a position inside an action, or a few turns on, as a search's
// tree reaches it. A LegalityMoves header names the moves to classify there
// (the export notation; default: every pseudo-legal move) - the one move a
// lazy verification asks about, say. --every K takes every K-th turn start of
// the record instead (no path). The rules come from the command line (the
// reader takes none from a record); fixtures name theirs in a Rules header.
//
// AT EACH POSITION, single-threaded, in order:
//  - the legality classification (classify_moves, t5d/legal.hpp) under each
//    --budget NODES (ascending; "exact" for none) with --checks CHECKS: the
//    seconds, the search nodes spent, and how the moves were settled - by the
//    tiers before the search (0-2), proved legal or illegal by tier 3's
//    searches, or left unknown. The classification stops at the first budget
//    that leaves nothing unknown ("settled at").
//  - with --mate: E4's detector (detect_mate, hypercuboid) under the same
//    budgets: the verdict, the seconds, nodes, points, regions, unrealised
//    points, unrealisable boxes and abstract checks given up - stopping at
//    the first budget that decides.
//  - with --search PLAYOUTS: one search (the defaults - the board-move split
//    unless --split, the uniform evaluator, E10's ladder off: Hunt, the
//    settling scans and the asset map - under the search's own legality
//    budget, or --search-legality NODES|exact for the leaves and the lazy
//    verifications alike, and --search-seconds S as its deadline) and where
//    its time went (SearchStats: terminal checks, expansions, turn generation,
//    verifications).
// --time-limit S gives each classification and each detector run a deadline
// (LegalityOptions::deadline; SearchBudget::max_time): what it stops is
// Unknown, never a verdict - how an exact run too long to wait for is cut.
//
// Indicative only: wall-clock seconds on whatever else the machine runs.
// Exit status: 0; 1 a record could not be read or a path not followed; 2 a bad
// command line.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "t5d/legal.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/search.hpp"
#include "tool_support.hpp"

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif

namespace {

using t5d::mcts::TurnSplit;
constexpr std::uint64_t kExact = std::numeric_limits<std::uint64_t>::max();

struct Options {
  std::vector<std::string> records;
  t5d::RuleOptions rules;
  std::vector<std::uint64_t> budgets;
  std::uint64_t checks = kExact;
  double time_limit = 0;  // seconds; 0: none
  bool mate = false;
  std::uint64_t search_playouts = 0;
  std::optional<std::uint64_t> search_legality;  // kExact: exact
  double search_seconds = 0;
  TurnSplit split = TurnSplit::BoardMove;
  int every = 0;
};

[[noreturn]] void usage(int status) {
  std::FILE* out = status == 0 ? stdout : stderr;
  std::fputs(
      "t5d-legality-cost: the legality classification's cost at each position (build-log issue 406)\n"
      "\n"
      "usage: t5d-legality-cost [options] RECORD ...\n"
      "  --rules ENCODING           the ruleset (RuleOptions::encode; default 0, the engine's)\n"
      "  --no-moves-after-shift     a diagnostic: Q4's Guide-literal reading\n"
      "  --budget NODES|exact       a budget in search nodes (repeatable, ascending; default 2000,\n"
      "                             200000, 2000000)\n"
      "  --checks N                 tier 2's checks (default: no limit; the search's default is 64)\n"
      "  --time-limit S             each classification's and detector run's deadline (default none)\n"
      "  --mate                     E4's detector under the same budgets\n"
      "  --search PLAYOUTS          one search from the position, and where its time went\n"
      "  --split ply|tree|flat      ... its D2 arm (default ply)\n"
      "  --search-legality NODES|exact  ... its legality budget, its lazy verifications' too\n"
      "                             (default: the search's own)\n"
      "  --search-seconds S         ... its deadline (Limits::seconds; default none)\n"
      "  --every K                  every K-th turn start of each record instead of its last position\n"
      "  --help  --version\n",
      out);
  std::exit(status);
}

std::uint64_t number(const std::string& text) {
  const std::optional<std::uint64_t> n = t5d::tools::parse_number(text);
  if (!n) throw std::invalid_argument("not a number: " + text);
  return *n;
}

double seconds_value(const std::string& option, const std::string& text) {
  char* end = nullptr;
  const double s = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0' || !(s >= 0) || !(s < 1e9)) {
    throw std::invalid_argument(option + ": a number of seconds, 0 or more, not " + text);
  }
  return s;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    if (const int used = t5d::tools::ruleset_option(argc, argv, i, o.rules)) {
      i += used - 1;
    } else if (a == "--help" || a == "-h") {
      usage(0);
    } else if (a == "--version") {
      std::puts("t5d-legality-cost " T5D_VERSION_STRING);
      std::exit(0);
    } else if (a == "--budget") {
      const std::string v = value();
      o.budgets.push_back(v == "exact" ? kExact : std::max<std::uint64_t>(1, number(v)));
    } else if (a == "--checks") {
      o.checks = number(value());
    } else if (a == "--time-limit") {
      o.time_limit = seconds_value(a, value());
    } else if (a == "--mate") {
      o.mate = true;
    } else if (a == "--search") {
      o.search_playouts = std::max<std::uint64_t>(1, number(value()));
    } else if (a == "--search-legality") {
      const std::string v = value();
      o.search_legality = v == "exact" ? kExact : std::max<std::uint64_t>(1, number(v));
    } else if (a == "--search-seconds") {
      o.search_seconds = seconds_value(a, value());
    } else if (a == "--split") {
      const std::string v = value();
      if (v == "ply") {
        o.split = TurnSplit::BoardMove;
      } else if (v == "tree") {
        o.split = TurnSplit::Tree;
      } else if (v == "flat") {
        o.split = TurnSplit::Flat;
      } else {
        throw std::invalid_argument("--split takes ply, tree or flat, not " + v);
      }
    } else if (a == "--every") {
      o.every = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("unknown option " + a);
    } else {
      o.records.push_back(a);
    }
  }
  if (o.records.empty()) throw std::invalid_argument("no records given");
  if (o.budgets.empty()) o.budgets = {2000, 200'000, 2'000'000};
  if (!std::is_sorted(o.budgets.begin(), o.budgets.end())) throw std::invalid_argument("--budget: ascending");
  return o;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string budget_text(std::uint64_t b) { return b == kExact ? std::string("exact") : std::to_string(b); }

const char* verdict_name(t5d::Outcome o) {
  switch (o) {
    case t5d::Outcome::Ongoing: return "ongoing";
    case t5d::Outcome::Checkmate: return "checkmate";
    case t5d::Outcome::Stalemate: return "stalemate";
    case t5d::Outcome::Unknown: return "unknown";
  }
  return "?";
}

std::optional<std::chrono::steady_clock::time_point> deadline_of(double seconds) {
  if (!(seconds > 0)) return std::nullopt;
  return std::chrono::steady_clock::now() +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
}

// The move written `text` among the pseudo-legal moves at `s`.
t5d::Move find_move(const t5d::GameState& s, const std::string& text) {
  t5d::MoveList moves;
  t5d::generate_moves(s, moves);
  const bool even = s.multiverse().even();
  for (const t5d::Move& m : moves) {
    if (t5d::to_string(m, even) == text) return m;
  }
  throw std::runtime_error("no pseudo-legal move " + text + " here");
}

struct Position {
  std::string name;
  t5d::GameState state;
  std::optional<std::vector<t5d::Move>> moves;  // LegalityMoves, or every move
  int moves_made = 0;                           // in the action under way (A6)
};

std::vector<Position> positions_of(const std::string& path, const Options& o) {
  const t5d::PgnRecord record = t5d::read_pgn(t5d::tools::read_file(path));
  t5d::PgnReplay replay(record, {}, o.rules);
  std::vector<Position> out;
  int turn = 0;
  const auto take = [&] {
    if (o.every > 0 && turn % o.every == 0) {
      out.push_back(Position{path + " #" + std::to_string(turn), replay.state(), std::nullopt, 0});
    }
  };
  take();
  while (!replay.finished()) {
    replay.play_next();
    ++turn;
    take();
  }
  if (record.unreadable) throw std::runtime_error("the move text stops early: " + *record.unreadable);
  if (o.every > 0) return out;
  Position p{path, replay.state(), std::nullopt, 0};
  if (const std::string* line = record.header("LegalityPath")) {
    std::istringstream in(*line);
    std::string token;
    while (in >> token) {
      if (token == "|") {
        if (p.state.submission_status() != t5d::SubmitStatus::Safe) {
          throw std::runtime_error("LegalityPath: a submission that is not legal");
        }
        p.state.submit();
        p.moves_made = 0;
      } else {
        if (!p.state.may_move()) throw std::runtime_error("LegalityPath: no move may be made before " + token);
        p.state.apply(find_move(p.state, token));
        ++p.moves_made;
      }
    }
  }
  if (const std::string* line = record.header("LegalityMoves")) {
    std::istringstream in(*line);
    std::string token;
    std::vector<t5d::Move> ms;
    while (in >> token) ms.push_back(find_move(p.state, token));
    p.moves = std::move(ms);
  }
  out.push_back(std::move(p));
  return out;
}

void classify_at(const Position& p, const Options& o) {
  std::vector<t5d::Move> moves;
  if (p.moves) {
    moves = *p.moves;
  } else {
    t5d::MoveList all;
    t5d::generate_moves(p.state, all);
    moves.assign(all.begin(), all.end());
  }
  std::optional<std::uint64_t> settled;
  for (const std::uint64_t b : o.budgets) {
    t5d::LegalityOptions lo = t5d::LegalityOptions::budgeted(o.checks, b);
    lo.deadline = deadline_of(o.time_limit);
    std::vector<t5d::MoveLegality> out;
    t5d::LegalityStats st;
    const auto t0 = std::chrono::steady_clock::now();
    t5d::classify_moves(p.state, moves, out, lo, &st);
    const double secs = seconds_since(t0);
    const auto count = [&](t5d::MoveLegality v) {
      return static_cast<std::size_t>(std::count(out.begin(), out.end(), v));
    };
    const std::size_t tier3 = st.searched_legal + st.searched_illegal + st.unknown + st.no_finish_at_all;
    std::printf("  classify budget=%s: %.3f s, %llu nodes; %zu legal, %zu illegal, %zu unknown; tiers 0-2 %zu, "
                "tier 3 %zu (%zu legal, %zu illegal, %zu with no finish at all), %zu learned\n",
                budget_text(b).c_str(), secs, static_cast<unsigned long long>(st.search_nodes),
                count(t5d::MoveLegality::Legal), count(t5d::MoveLegality::Illegal), count(t5d::MoveLegality::Unknown),
                moves.size() - tier3 - st.learned, tier3, st.searched_legal, st.searched_illegal, st.no_finish_at_all,
                st.learned);
    std::fflush(stdout);
    if (st.unknown == 0) {
      settled = b;
      break;
    }
  }
  std::printf("  classify settled at: %s\n", settled ? budget_text(*settled).c_str() : "none of the budgets");
}

void mate_at(const Position& p, const Options& o) {
  for (const std::uint64_t b : o.budgets) {
    t5d::SearchBudget sb = b == kExact ? t5d::SearchBudget::unlimited() : t5d::SearchBudget::nodes(b);
    if (o.time_limit > 0) {
      sb.max_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(o.time_limit));
    }
    const auto t0 = std::chrono::steady_clock::now();
    const t5d::MateResult r = t5d::detect_mate(p.state, t5d::Detector::Hypercuboid, sb);
    std::printf("  detect_mate budget=%s: %s, %.3f s, %llu nodes, %llu points, %llu regions, %llu unrealised, "
                "%llu unrealisable boxes, %llu abstract checks given up\n",
                budget_text(b).c_str(), verdict_name(r.verdict), seconds_since(t0),
                static_cast<unsigned long long>(r.stats.nodes), static_cast<unsigned long long>(r.stats.points),
                static_cast<unsigned long long>(r.stats.regions), static_cast<unsigned long long>(r.stats.unrealised),
                static_cast<unsigned long long>(r.stats.unrealisable_boxes),
                static_cast<unsigned long long>(r.stats.abstract_gave_up));
    std::fflush(stdout);
    if (r.verdict != t5d::Outcome::Unknown) break;
  }
}

void search_at(const Position& p, const Options& o) {
  t5d::mcts::SearchParams sp;
  sp.split = o.split;
  sp.hunt = sp.settling = false;  // the legality path alone: E10's ladder off
  sp.asset_map = false;
  sp.expand.square_features = t5d::SquareFeatures::None;  // the uniform evaluator reads none
  if (o.search_legality) {
    sp.legality = *o.search_legality == kExact ? t5d::LegalityOptions::exact()
                                               : t5d::LegalityOptions::nodes(*o.search_legality);
    sp.verify_legality = sp.legality;
  }
  const std::unique_ptr<t5d::mcts::Evaluator> ev = t5d::mcts::make_evaluator(t5d::mcts::BuiltinEvaluator::Uniform);
  const auto t0 = std::chrono::steady_clock::now();
  t5d::mcts::Search search(p.state, sp, p.moves_made);
  t5d::mcts::Limits limits{o.search_playouts};
  limits.seconds = o.search_seconds;
  search.run(*ev, limits);
  const t5d::mcts::SearchStats& st = search.stats();
  const std::string legality = !o.search_legality               ? std::string("the search's default")
                               : *o.search_legality == kExact ? std::string("exact")
                                                              : "nodes " + std::to_string(*o.search_legality);
  std::printf("  search %llu playouts (%s split, legality %s, deadline %.3g s): %.3f s, %llu playouts, stopped by "
              "%s; terminal checks %.3f s, expansions %.3f s, turn generation %.3f s, verifications %.3f s; %zu nodes\n",
              static_cast<unsigned long long>(o.search_playouts),
              o.split == TurnSplit::Flat   ? "flat"
              : o.split == TurnSplit::Tree ? "tree"
                                           : "ply",
              legality.c_str(), o.search_seconds, seconds_since(t0), static_cast<unsigned long long>(st.playouts),
              t5d::mcts::stop_reason_name(st.last_stop), st.seconds_mate, st.seconds_expand, st.seconds_generate,
              st.seconds_verify, search.node_count());
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = parse(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-legality-cost: %s\n", e.what());
    usage(2);
  }
  std::printf("CONDITIONS t5d-legality-cost %s rules=%s checks=%s budgets=", T5D_VERSION_STRING,
              t5d::tools::ruleset_text(o.rules).c_str(), budget_text(o.checks).c_str());
  for (std::size_t i = 0; i < o.budgets.size(); ++i) {
    std::printf("%s%s", i ? "," : "", budget_text(o.budgets[i]).c_str());
  }
  std::printf(" time_limit=%.3g detector=hypercuboid one thread\n", o.time_limit);
  for (const std::string& r : o.records) {
    std::vector<Position> ps;
    try {
      ps = positions_of(r, o);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "t5d-legality-cost: %s: %s\n", r.c_str(), e.what());
      return 1;
    }
    for (const Position& p : ps) {
      t5d::MoveList all;
      t5d::generate_moves(p.state, all);
      const std::string named = p.moves ? " (" + std::to_string(p.moves->size()) + " named)" : std::string();
      std::printf("POSITION %s: %zu boards, %zu playable timelines, %s to move, %s, %zu pseudo-legal moves%s\n",
                  p.name.c_str(), p.state.multiverse().board_count(), p.state.playable_timelines().size(),
                  p.state.to_move() == t5d::Colour::White ? "white" : "black",
                  p.state.obligations_met() ? "obligations met" : "obligations unmet", all.size(), named.c_str());
      std::fflush(stdout);
      classify_at(p, o);
      if (o.mate) mate_at(p, o);
      if (o.search_playouts > 0) search_at(p, o);
    }
  }
  return 0;
}
