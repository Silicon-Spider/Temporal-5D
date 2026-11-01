// Build-log issue 406: the cost of exact move legality, and the deadlines that
// must reach it. The positions are engine/testdata/legality-cost's fixtures
// (their LegalityCost headers say where each comes from): a deadline - a
// search's Limits::seconds, a game's PlayParams::max_seconds - reaches the
// classification's searches (LegalityOptions::deadline) and the detector, and
// what it stops is Unknown, never a verdict; and t5d-legality-cost, the tool
// that characterises the cost, run on them.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "t5d/legal.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"

namespace {

namespace fs = std::filesystem;
using namespace t5d;

std::string read_all(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

fs::path fixture_path(const char* name) { return fs::path(T5D_ENGINE_DIR) / "testdata" / "legality-cost" / name; }

Move find_move(const GameState& s, const std::string& text) {
  MoveList moves;
  generate_moves(s, moves);
  for (const Move& m : moves) {
    if (to_string(m, s.multiverse().even()) == text) return m;
  }
  throw std::runtime_error("no pseudo-legal move " + text);
}

// A fixture's position: the record replayed, then its LegalityPath ("|": a
// submission); `named` gets its LegalityMoves.
GameState fixture(const char* name, const RuleOptions& rules, std::vector<Move>* named = nullptr,
                  int* moves_made = nullptr) {
  const PgnRecord record = read_pgn(read_all(fixture_path(name)));
  PgnReplay replay(record, {}, rules);
  while (!replay.finished()) replay.play_next();
  GameState s = replay.state();
  int made = 0;
  if (const std::string* line = record.header("LegalityPath")) {
    std::istringstream in(*line);
    std::string t;
    while (in >> t) {
      if (t == "|") {
        s.submit();
        made = 0;
      } else {
        s.apply(find_move(s, t));
        ++made;
      }
    }
  }
  if (named != nullptr) {
    named->clear();
    if (const std::string* line = record.header("LegalityMoves")) {
      std::istringstream in(*line);
      std::string t;
      while (in >> t) named->push_back(find_move(s, t));
    }
  }
  if (moves_made != nullptr) *moves_made = made;
  return s;
}

RuleOptions q3() {
  RuleOptions r;
  r.play_activates = true;
  return r;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

int run(const std::string& exe, const std::vector<std::string>& args, const fs::path& out, const fs::path& err) {
  std::string command = quoted(fs::path(exe).make_preferred().string());
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

}  // namespace

TEST(legality_deadline_leaves_moves_unknown_and_never_misclassifies) {
  // The costliest node of issue 406's Q3 search: exact, about 12 million
  // nodes (some 20 s); a deadline stops the searches, and every move it
  // settled agrees with a budgeted classification wherever both settled one.
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  MoveList all;
  generate_moves(s, all);
  const std::vector<Move> moves(all.begin(), all.end());
  REQUIRE_EQ(moves.size(), std::size_t{141});
  std::vector<MoveLegality> ref;
  LegalityStats ref_stats;
  classify_moves(s, moves, ref, LegalityOptions::nodes(400'000), &ref_stats);
  for (const double wait : {0.0, 0.05, 0.3}) {
    LegalityOptions o = LegalityOptions::exact();
    o.deadline = std::chrono::steady_clock::now() +
                 std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(wait));
    std::vector<MoveLegality> out;
    LegalityStats st;
    const auto t0 = std::chrono::steady_clock::now();
    classify_moves(s, moves, out, o, &st);
    const double took = seconds_since(t0);
    std::fprintf(stderr, "  deadline %.2f s: %.3f s, %llu nodes, %zu unknown\n", wait, took,
                 static_cast<unsigned long long>(st.search_nodes), st.unknown);
    CHECK(took < wait + 5.0);  // a loaded machine; exact without it takes some 20 s
    CHECK(st.unknown > 0);
    for (std::size_t i = 0; i < moves.size(); ++i) {
      if (out[i] != MoveLegality::Unknown && ref[i] != MoveLegality::Unknown) CHECK_EQ(out[i], ref[i]);
    }
  }
}

TEST(search_deadline_reaches_the_legality_classification) {
  // Issue 406's root under the flat form with exact legality: past 17
  // minutes for 800 playouts, nearly all in the leaves' classification. Its
  // deadline now reaches the classification inside a batch.
  const GameState s = fixture("q3-root.5dpgn", q3());
  mcts::SearchParams p;
  p.split = mcts::TurnSplit::Flat;
  p.legality = LegalityOptions::exact();
  p.verify_legality = LegalityOptions::exact();
  p.hunt = p.settling = false;
  p.expand.square_features = SquareFeatures::None;
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search search(s, p);
  mcts::Limits limits{800};
  limits.seconds = 0.5;
  const auto t0 = std::chrono::steady_clock::now();
  search.run(*ev, limits);
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.2f s, %llu playouts, stopped by %s\n", took,
               static_cast<unsigned long long>(search.stats().playouts),
               mcts::stop_reason_name(search.stats().last_stop));
  CHECK(took < 15.0);
  CHECK_NOTHROW(search.check_invariants());
  // The deadline is run()'s alone: a later run without one is exact again.
  CHECK(search.stats().last_stop == mcts::StopReason::Deadline ||
        search.stats().last_stop == mcts::StopReason::Playouts);
}

TEST(play_deadline_reaches_the_verdicts_and_the_chosen_moves_verification) {
  // The same root played with exact legality under the flat form and a
  // 1-second safety stop: the game ends on it (TimeLimit), its moves known
  // legal, within a few seconds - not after the many minutes one exact
  // search takes.
  const GameState s = fixture("q3-root.5dpgn", q3());
  mcts::PlayParams pp;
  pp.search.split = mcts::TurnSplit::Flat;
  pp.search.legality = LegalityOptions::exact();
  pp.search.verify_legality = LegalityOptions::exact();
  pp.search.hunt = pp.search.settling = false;
  pp.search.expand.square_features = SquareFeatures::None;
  pp.playouts = 800;
  pp.max_seconds = 1.0;
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  const auto t0 = std::chrono::steady_clock::now();
  const mcts::GameRecord g = mcts::play_game(s, *ev, pp);
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.2f s, %zu actions, %s\n", took, g.actions.size(), mcts::game_end_name(g.end));
  CHECK(took < 20.0);
  CHECK(g.end == mcts::GameEnd::TimeLimit || g.end == mcts::GameEnd::Checkmate || g.end == mcts::GameEnd::Stalemate);
  // Every action played is legal: replayed and submitted under C4.
  GameState t = s;
  for (const std::vector<Move>& a : g.actions) {
    for (const Move& m : a) t.apply(m);
    CHECK(t.submission_status() == SubmitStatus::Safe);
    t.submit();
  }
}

TEST(play_deadline_reaches_a_turn_start_verdict) {
  // The Very Small - Open game's turn start (191 boards, 49 playable
  // timelines) at which the game's own verdict - no node limit, H37 - ran past
  // 20 minutes: a 2-second safety stop now ends the game there, unjudged
  // (TimeLimit, not Unknown), within seconds.
  const GameState s = fixture("vso-seed4-verdict.5dpgn", RuleOptions{});
  REQUIRE_EQ(s.multiverse().board_count(), std::size_t{191});
  mcts::PlayParams pp;
  pp.search.expand.square_features = SquareFeatures::None;
  pp.search.hunt = pp.search.settling = false;
  pp.playouts = 32;
  pp.max_seconds = 2.0;
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Material);
  const auto t0 = std::chrono::steady_clock::now();
  const mcts::GameRecord g = mcts::play_game(s, *ev, pp);
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.2f s (verdicts %.2f s), %zu actions, %s\n", took, g.seconds_verdicts, g.actions.size(),
               mcts::game_end_name(g.end));
  CHECK(took < 30.0);
  CHECK(g.end == mcts::GameEnd::TimeLimit);
  CHECK(g.actions.empty());
  CHECK(g.seconds_verdicts >= 1.5);  // the verdict ran until the deadline
}

TEST(legality_cost_exe_reports_each_fixture) {
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "legality-cost";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path out = dir / "stdout.txt";
  const fs::path err = dir / "stderr.txt";
  // The Q3 fixtures under Q3, small budgets (the costly ones are the long run's).
  REQUIRE_EQ(run(T5D_LEGALITY_COST_EXE,
                 {"--rules", "4", "--budget", "2000", "--budget", "20000", "--mate", fixture_path("q3-root.5dpgn").string(),
                  fixture_path("q3-flat-node.5dpgn").string()},
                 out, err),
             0);
  std::string text = read_all(out);
  std::fprintf(stderr, "%s", text.c_str());
  CHECK(text.find("CONDITIONS t5d-legality-cost") != std::string::npos);
  CHECK(text.find("rules=play_activates") != std::string::npos);
  CHECK(text.find("5 boards, ") != std::string::npos);
  CHECK(text.find("18 boards, ") != std::string::npos);
  CHECK(text.find("141 pseudo-legal moves") != std::string::npos);
  CHECK(text.find("classify settled at: 2000") != std::string::npos);        // the root
  CHECK(text.find("classify settled at: none of the budgets") != std::string::npos);  // the node
  CHECK(text.find("detect_mate budget=2000: ongoing") != std::string::npos);
  // An exact classification cut by --time-limit: Unknown, quickly.
  const auto t0 = std::chrono::steady_clock::now();
  REQUIRE_EQ(run(T5D_LEGALITY_COST_EXE,
                 {"--rules", "4", "--budget", "exact", "--time-limit", "0.2", fixture_path("q3-flat-node.5dpgn").string()},
                 out, err),
             0);
  text = read_all(out);
  std::fprintf(stderr, "%s", text.c_str());
  CHECK(seconds_since(t0) < 15.0);  // exact takes some 17 s
  CHECK(text.find("classify budget=exact: ") != std::string::npos);
  CHECK(text.find("classify settled at: none of the budgets") != std::string::npos);
  // A game's position inside an action, its one named move, and a search.
  REQUIRE_EQ(run(T5D_LEGALITY_COST_EXE,
                 {"--budget", "2000", "--search", "4", fixture_path("vso-seed4-verify.5dpgn").string()}, out, err),
             0);
  text = read_all(out);
  std::fprintf(stderr, "%s", text.c_str());
  CHECK(text.find("48 playable timelines, white to move") != std::string::npos);
  CHECK(text.find("940 pseudo-legal moves (1 named)") != std::string::npos);
  CHECK(text.find("0 legal, 0 illegal, 1 unknown") != std::string::npos);
  CHECK(text.find("search 4 playouts") != std::string::npos);
  // Every K-th turn start of a record; a bad command line.
  REQUIRE_EQ(run(T5D_LEGALITY_COST_EXE, {"--every", "10", "--budget", "2000", fixture_path("vso-seed4-verify.5dpgn").string()},
                 out, err),
             0);
  text = read_all(out);
  CHECK(text.find("#0: 1 boards") != std::string::npos);
  CHECK(text.find("#20: ") != std::string::npos);
  CHECK_EQ(run(T5D_LEGALITY_COST_EXE, {"--split", "sideways", "x.5dpgn"}, out, err), 2);
  CHECK_EQ(run(T5D_LEGALITY_COST_EXE, {"--budget", "20000", "--budget", "2000", "x.5dpgn"}, out, err), 2);
  CHECK_EQ(run(T5D_LEGALITY_COST_EXE, {(dir / "missing.5dpgn").string()}, out, err), 1);
}
