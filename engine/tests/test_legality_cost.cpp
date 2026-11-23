// Build-log issue 406: the cost of exact move legality, and the deadlines that
// must reach it. The positions are engine/testdata/legality-cost's fixtures
// (their LegalityCost headers say where each comes from): a deadline - a
// search's Limits::seconds, a game's PlayParams::max_seconds - reaches the
// classification's searches (LegalityOptions::deadline) and the detector, and
// what it stops is Unknown, never a verdict; and t5d-legality-cost, the tool
// that characterises the cost, run on them.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "t5d/expand.hpp"
#include "t5d/legal.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_play.hpp"

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
  CHECK(search.stats().last_stop == mcts::StopReason::Deadline ||
        search.stats().last_stop == mcts::StopReason::Playouts);
  // The deadline is run()'s alone (DeadlineScope; search_deadline_does_not_outlive_its_run holds
  // it): a later run without one classifies its new leaves without it, while the nodes this run
  // expanded stay as they were (a node is expanded once) - and the evaluation cache keeps their
  // cut candidate lists apart from uncut ones (the review of 2026-11-09, H1;
  // search_cache_keeps_a_cut_candidate_list_apart). Not run here: exact legality from this root
  // reaches issue 406's 20-second node within a few playouts.
}

TEST(play_flat_with_exact_legality_ends_on_its_stop) {
  // The same root played with exact legality under the flat form and a
  // 1-second safety stop: the game ends on it (TimeLimit - from a turn
  // start's verdict: the flat form's root children are never unknown, so no
  // chosen move is verified here), its actions legal, within seconds - not
  // after the many minutes one exact search takes. (Named, until the review of
  // 2026-11-09's M4, play_deadline_reaches_the_verdicts_and_the_chosen_moves_
  // verification; the verification is play_no_move_is_played_after_the_
  // deadline's.) Measured: 1.0 s unmutated; with the leaves' classification
  // untimed (the review's M07) 19.8 s.
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
  CHECK(took < 10.0);
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
  // The tool's output in text mode, read here in binary: the line ends without their '\r'.
  const auto lf_only = [](std::string t) {
    t.erase(std::remove(t.begin(), t.end(), '\r'), t.end());
    return t;
  };
  std::string text = lf_only(read_all(out));
  std::fprintf(stderr, "%s", text.c_str());
  CHECK(text.find("CONDITIONS t5d-legality-cost") != std::string::npos);
  CHECK(text.find("rules=play_activates") != std::string::npos);
  CHECK(text.find("5 boards, ") != std::string::npos);
  CHECK(text.find("18 boards, ") != std::string::npos);
  CHECK(text.find("141 pseudo-legal moves") != std::string::npos);
  // The root settles at the first budget and stops there (not "...: 20000"; the review of
  // 2026-11-09's T03), its tiers 0-2 counting what they settled, not what tier 3 learned (T04:
  // MEASUREMENTS' rows rest on both).
  CHECK(text.find("classify settled at: 2000\n") != std::string::npos);      // the root
  CHECK(text.find("tiers 0-2 15,") != std::string::npos);
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
  // The 191-board turn start (the fourth fixture): E4's exact verdict cut by --time-limit -
  // Unknown, within seconds (the review of 2026-11-09's T02: --time-limit never reached --mate;
  // measured 0.2 s for the verdict here; the verdict uncut runs past 20 minutes).
  const auto t1 = std::chrono::steady_clock::now();
  REQUIRE_EQ(run(T5D_LEGALITY_COST_EXE,
                 {"--mate", "--budget", "exact", "--time-limit", "0.2", fixture_path("vso-seed4-verdict.5dpgn").string()},
                 out, err),
             0);
  text = read_all(out);
  std::fprintf(stderr, "%s", text.c_str());
  CHECK(seconds_since(t1) < 15.0);
  CHECK(text.find("191 boards, ") != std::string::npos);
  CHECK(text.find("detect_mate budget=exact: unknown") != std::string::npos);
  // A search given its legality and its deadline (T07, T08): the line says what the search
  // applied - the leaves' classification and the lazy verifications exact, a 0.5 s deadline -
  // and the deadline stops it (issue 406's Q3 node under the default rules, the flat form).
  const auto t2 = std::chrono::steady_clock::now();
  REQUIRE_EQ(run(T5D_LEGALITY_COST_EXE,
                 {"--search", "800", "--split", "flat", "--search-legality", "exact", "--search-seconds", "0.5",
                  fixture_path("q3-flat-node.5dpgn").string()},
                 out, err),
             0);
  text = read_all(out);
  std::fprintf(stderr, "%s", text.c_str());
  CHECK(seconds_since(t2) < 15.0);
  CHECK(text.find("legality exact, verifications exact, deadline 0.5 s") != std::string::npos);
  CHECK(text.find("stopped by deadline") != std::string::npos);
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

// ---- The review of 2026-11-09 (lane/engine 0e7d2ed..d86f363): the deadline's path ----------
// H1: a cut classification is another candidate list, kept apart by the cache and the batch;
// M1: a fallback listing the deadline cut leaves its node re-listable, and the play loop ends
// TimeLimit; M2: 1e9 seconds or more, +infinity included, set no deadline; L5: a timed-out
// decision's work counts in the game's totals.

namespace {

mcts::SearchParams quiet(mcts::SearchParams p = {}) {
  p.hunt = p.settling = false;
  p.asset_map = false;
  p.expand.square_features = SquareFeatures::None;
  return p;
}

// Delegates to another evaluator after sleeping (`once`: in its first batch only): the deadline
// passes inside a batch, between gather (the leaves' classification) and deliver (the flat form's
// generator and fallback).
class Sleepy final : public mcts::Evaluator {
 public:
  Sleepy(mcts::Evaluator& inner, double seconds, bool once = false) : inner_(inner), seconds_(seconds), once_(once) {}
  void evaluate(std::span<const mcts::EvalRequest> batch, std::span<mcts::Evaluation> out) override {
    if (!once_ || !slept_) std::this_thread::sleep_for(std::chrono::duration<double>(seconds_));
    slept_ = true;
    inner_.evaluate(batch, out);
  }
  void on_new_root(const GameState& root, std::uint64_t epoch) override { inner_.on_new_root(root, epoch); }

 private:
  mcts::Evaluator& inner_;
  double seconds_;
  bool once_;
  bool slept_ = false;
};

// The uniform evaluator, but for one move: logit 8 (or `logit`) where a leaf's candidates hold it
// - so that the search's visits, and the play loop's choice, go to it.
class Steering final : public mcts::Evaluator {
 public:
  explicit Steering(const Move& favoured, float logit = 8.0f) : favoured_(favoured), logit_(logit) {}
  void evaluate(std::span<const mcts::EvalRequest> batch, std::span<mcts::Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const MoveList& moves = batch[i].expansion->moves();
      out[i].logits.assign(moves.size(), 0.0f);
      for (std::size_t m = 0; m < moves.size(); ++m) {
        if (moves[m] == favoured_) out[i].logits[m] = logit_;
      }
      out[i].wdl = mcts::wdl_of(0.0);
      out[i].value = mcts::two_hot(0.0);
    }
  }

 private:
  Move favoured_;
  float logit_;
};

// Two moves steered: logit 4 for the first, 3.5 for the second, 0 for the rest - the search's
// visits go to the first, then the second.
class TwoSteering final : public mcts::Evaluator {
 public:
  TwoSteering(const Move& first, const Move& second) : first_(first), second_(second) {}
  void evaluate(std::span<const mcts::EvalRequest> batch, std::span<mcts::Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const MoveList& moves = batch[i].expansion->moves();
      out[i].logits.assign(moves.size(), 0.0f);
      for (std::size_t m = 0; m < moves.size(); ++m) {
        if (moves[m] == first_) out[i].logits[m] = 4.0f;
        if (moves[m] == second_) out[i].logits[m] = 3.5f;
      }
      out[i].wdl = mcts::wdl_of(0.0);
      out[i].value = mcts::two_hot(0.0);
    }
  }

 private:
  Move first_;
  Move second_;
};

// The candidate list of a leaf at `s` under `p`, classified within p's legality budget or with
// the deadline already passed - what the search's leaf_expansion hands the network either side
// of run()'s deadline.
Expansion candidates_at(const GameState& s, const mcts::SearchParams& p, bool deadline_passed) {
  ExpandOptions o = p.expand;
  o.legal_moves = p.legal_moves_only;
  o.legality = p.legality;
  if (deadline_passed) o.legality.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  Expansion x;
  expand(s, o, x, 0);
  return x;
}

}  // namespace

TEST(search_cache_keeps_a_cut_candidate_list_apart) {
  // H1. Issue 406's 18-board Q3 node: a classification within 400,000 nodes proves moves illegal
  // that one cut by run()'s deadline keeps (unknown) - two candidate lists for one state, two
  // network inputs. Two searches share an evaluation cache (as play_game's fresh search per
  // decision shares the game's): one untimed, one whose 0.05 s deadline cuts the root's
  // classification (0.4-0.8 s untimed), in either order. Each gets an evaluation of its own list;
  // before the fix the second found the first's under the state alone and threw ("an evaluation
  // has 57 logits for 66 candidate moves", or the reverse).
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  mcts::SearchParams p = quiet();
  p.legality = LegalityOptions::nodes(400'000);
  // The premise, deterministic: the two lists differ.
  const Expansion full = candidates_at(s, p, false);
  const Expansion cut = candidates_at(s, p, true);
  std::fprintf(stderr, "  candidates: %zu within 400,000 nodes, %zu with the deadline passed\n", full.moves().size(),
               cut.moves().size());
  REQUIRE(full.moves().size() < cut.moves().size());
  CHECK(mcts::candidate_digest(full) != mcts::candidate_digest(cut));
  CHECK_EQ(mcts::candidate_digest(full), mcts::candidate_digest(candidates_at(s, p, false)));  // a function of the list
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  const mcts::Limits untimed{1};  // the root's evaluation, into the cache
  mcts::Limits timed{1};
  timed.seconds = 0.05;
  // The root's candidates: its edges (root_children lists the widened ones alone).
  const auto edges = [](const mcts::Search& x) { return x.node(x.root()).edge_count; };
  const std::size_t full_edges = full.moves().size() + (full.submit_legal() ? 1u : 0u);
  {  // untimed, then timed
    auto cache = std::make_shared<mcts::EvalCache>();
    mcts::Search a(s, p, 0, cache);
    a.run(*ev, untimed);
    CHECK_EQ(edges(a), full_edges);
    mcts::Search b(s, p, 0, cache);
    CHECK_NOTHROW(b.run(*ev, timed));
    CHECK_NOTHROW(b.check_invariants());
    std::fprintf(stderr, "  untimed then timed: %zu root edges, then %zu (%s); cache %zu entries\n", edges(a), edges(b),
                 mcts::stop_reason_name(b.stats().last_stop), cache->size());
    CHECK(edges(b) >= full_edges);
    CHECK_EQ(cache->size(), std::size_t{2});  // one entry per list
  }
  {  // timed, then untimed: the later untimed run's list is the first untimed run's
    auto cache = std::make_shared<mcts::EvalCache>();
    mcts::Search b(s, p, 0, cache);
    CHECK_NOTHROW(b.run(*ev, timed));
    mcts::Search a(s, p, 0, cache);
    CHECK_NOTHROW(a.run(*ev, untimed));
    CHECK_NOTHROW(a.check_invariants());
    std::fprintf(stderr, "  timed then untimed: %zu root edges, then %zu; cache %zu entries\n", edges(b), edges(a),
                 cache->size());
    CHECK_EQ(edges(a), full_edges);
  }
}

TEST(cache_shared_across_a_deadline) {
  // H1, adopted from the review (review_cache_shared_across_a_deadline): the first search
  // classifies the root within 2 million nodes and caches its evaluation - logits for the moves
  // that expansion kept; the second, whose deadline cuts the same root's classification (more
  // moves kept, unknown), must not be served it. Before the fix: "an evaluation has 58 logits for
  // 66 candidate moves".
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  mcts::SearchParams p = quiet();
  p.split = mcts::TurnSplit::BoardMove;
  p.legality = LegalityOptions::nodes(2'000'000);
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  auto cache = std::make_shared<mcts::EvalCache>();
  std::size_t first = 0;
  {
    mcts::Search a(s, p, 0, cache);
    a.run(*ev, mcts::Limits{4});
    first = a.node(a.root()).edge_count;
    std::fprintf(stderr, "  first search: root edges %zu, cache %zu entries\n", first, cache->size());
  }
  mcts::Search b(s, p, 0, cache);
  mcts::Limits l{4};
  l.seconds = 0.05;
  CHECK_NOTHROW(b.run(*ev, l));
  CHECK_NOTHROW(b.check_invariants());
  std::fprintf(stderr, "  second search: root edges %zu, stopped by %s\n", b.node(b.root()).edge_count,
               mcts::stop_reason_name(b.stats().last_stop));
  CHECK(b.node(b.root()).edge_count >= first);
}

TEST(search_batch_keeps_a_cut_candidate_list_apart) {
  // H1's other route: one batch holding one state twice (a transposition the tree does not
  // merge), one expansion classified before the deadline and the other after - one request each.
  // The cache off, so that only the batch can merge them; a batch of 512, so that the deadline
  // falls inside one. Before the fix the two shared one evaluation and run() threw - in 51 of the
  // review's 429 such searches; now every run ends.
  std::vector<GameState> positions;
  for (const GameState& s : t5d_test::sparse_positions(RuleOptions{}, 3, 5, 777)) {
    if (s.may_move() && positions.size() < 48) positions.push_back(s);
  }
  REQUIRE_EQ(positions.size(), std::size_t{48});
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  std::size_t runs = 0, deadline_stops = 0;
  for (const GameState& s : positions) {
    for (const double secs : {0.003, 0.01, 0.03}) {
      mcts::SearchParams p = quiet();
      p.eval_cache_entries = 0;
      p.batch_size = 512;
      mcts::Search search(s, p);
      mcts::Limits l{std::uint64_t{1} << 30};
      l.seconds = secs;
      CHECK_NOTHROW(search.run(*ev, l));
      CHECK_NOTHROW(search.check_invariants());
      ++runs;
      if (search.stats().last_stop == mcts::StopReason::Deadline) ++deadline_stops;
    }
  }
  std::fprintf(stderr, "  %zu runs, %zu stopped by the deadline\n", runs, deadline_stops);
}

TEST(play_cache_across_decisions_with_a_stop) {
  // H1 through play_game itself, adopted from the review (review_play_cache_across_decisions_
  // with_a_stop, which at 16 playouts and 2-8 s stops never finished a decision): C14's fresh
  // search per decision shares the game's cache, so a decision whose deadline cuts its root's
  // classification meets the entry the previous decision stored for that state as its leaf. The
  // ply split from issue 406's Q3 node, legality within 200,000 nodes (some 0.4 s a
  // classification there), two playouts a decision: a root and one leaf.
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  for (const double secs : {0.9, 1.2, 1.5, 1.8}) {
    mcts::PlayParams pp;
    pp.search = quiet(pp.search);
    pp.search.split = mcts::TurnSplit::BoardMove;
    pp.search.legality = LegalityOptions::nodes(200'000);
    pp.playouts = 2;
    pp.max_seconds = secs;
    const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
    mcts::GameRecord g(s);
    CHECK_NOTHROW(g = mcts::play_game(s, *ev, pp));
    std::fprintf(stderr, "  stop %.1f s: %.2f s, %zu actions, %s, %llu decisions\n", secs, g.seconds, g.actions.size(),
                 mcts::game_end_name(g.end), static_cast<unsigned long long>(g.decisions));
    CHECK(g.end == mcts::GameEnd::TimeLimit || g.end == mcts::GameEnd::Checkmate ||
          g.end == mcts::GameEnd::Stalemate);
  }
}

TEST(play_with_a_stop_from_sparse_starts_and_small_variants_never_throws) {
  // H1 at t5d-play's own settings (64 playouts, the board-move split, self-play's fresh search
  // per decision sharing the game's cache, root noise, the turn cap 20 turns / 300 boards): games
  // the safety stop reaches end TimeLimit - before the fix 15 of the review's 520 such games threw
  // the logit-count mismatch. Stops from 0.05 to 0.44 s.
  struct Start {
    std::string name;
    GameState state;
  };
  std::vector<Start> starts;
  for (const char* name : {"Very Small - Open", "Very Small", "Focused - Just Kings"}) {
    const Variant* v = find_variant(name);
    REQUIRE(v != nullptr);
    starts.push_back({name, variant_start(*v, RuleOptions{})});
  }
  int k = 0;
  for (const t5d_test::SparseStart& st : t5d_test::sparse_starts()) {
    starts.push_back({"sparse " + std::to_string(k++), t5d_test::game(st.boards, st.size, RuleOptions{})});
  }
  std::size_t ended = 0, time_limits = 0;
  for (int g = 0; g < 26; ++g) {
    const Start& st = starts[static_cast<std::size_t>(g) % starts.size()];
    mcts::PlayParams pp;
    pp.search = quiet(pp.search);
    pp.turn_cap.turns = 20;
    pp.turn_cap.boards = 300;
    pp.seed = 1000 + static_cast<std::uint64_t>(g);
    pp.search.seed = pp.seed;
    pp.max_seconds = 0.05 + 0.017 * static_cast<double>((g * 7) % 23);
    const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform, pp.seed);
    mcts::GameRecord r(st.state);
    CHECK_NOTHROW(r = mcts::play_game(st.state, *ev, pp));
    ++ended;
    if (r.end == mcts::GameEnd::TimeLimit) ++time_limits;
  }
  std::fprintf(stderr, "  %zu games, %zu ended TimeLimit\n", ended, time_limits);
}

TEST(flat_play_fallback_cut_by_the_deadline) {
  // M1, adopted from the review (review_flat_play_fallback_cut_by_the_deadline). The flat form
  // where the generator stops before its first turn (generator_nodes 1 stands in for a ~190-board
  // turn start, where 200,000 nodes find none): the fallback listing supplies the turns - unless
  // the deadline, passing inside the batch (the evaluator sleeps 0.4 s against a 0.3 s stop),
  // cuts it first. The root is then not stuck, run() stops Deadline, and the play loop ends
  // TimeLimit - before the fix the root was stuck, run() stopped Finished and the loop threw
  // "the root has no visited child to choose".
  const GameState s = fixture("q3-root.5dpgn", q3());
  mcts::PlayParams pp;
  pp.search = quiet(pp.search);
  pp.search.split = mcts::TurnSplit::Flat;
  pp.search.generator_nodes = 1;
  pp.playouts = 64;
  pp.max_seconds = 0.3;
  const std::unique_ptr<mcts::Evaluator> inner = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  Sleepy ev(*inner, 0.4);
  mcts::GameRecord g(s);
  CHECK_NOTHROW(g = mcts::play_game(s, ev, pp));
  std::fprintf(stderr, "  %zu actions, %s, %llu decisions\n", g.actions.size(), mcts::game_end_name(g.end),
               static_cast<unsigned long long>(g.decisions));
  CHECK(g.end == mcts::GameEnd::TimeLimit);
  CHECK_EQ(g.decisions, std::uint64_t{0});
  // The same game without the stop plays (the fallback listing exact within its node budget).
  pp.max_seconds = 0;
  pp.max_actions = 1;
  Sleepy quick(*inner, 0.0);
  const mcts::GameRecord g2 = mcts::play_game(s, quick, pp);
  CHECK_EQ(g2.actions.size(), std::size_t{1});
}

TEST(flat_play_at_issue_406s_node_ends_time_limit_when_the_fallback_is_cut) {
  // M1 at its natural setting: issue 406's Q3 node under the flat form, the defaults (its
  // generator stops before its first turn within generator_nodes; the fallback lists 16 turns
  // untimed), the uniform evaluator - in self-play and in match play. Before the fix a stop of
  // 0.087-0.5 s cut the root's fallback listing, left the root stuck and play_game threw.
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  for (const bool match : {false, true}) {
    for (const double secs : {0.1, 0.25, 0.5}) {
      mcts::PlayParams pp;
      pp.search = quiet(pp.search);
      pp.search.split = mcts::TurnSplit::Flat;
      pp.match_play = match;
      pp.max_seconds = secs;
      const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
      mcts::GameRecord g(s);
      CHECK_NOTHROW(g = mcts::play_game(s, *ev, pp));
      std::fprintf(stderr, "  %s, stop %.2f s: %.2f s, %zu actions, %s\n", match ? "match" : "self-play", secs, g.seconds,
                   g.actions.size(), mcts::game_end_name(g.end));
      CHECK(g.end == mcts::GameEnd::TimeLimit);
    }
  }
}

TEST(flat_root_emptied_by_a_cut_listing_is_listed_by_a_later_run) {
  // M1 through the Search API: a flat root whose fallback listing the deadline cut (the
  // generator stopped before its first turn; the evaluator sleeps past the deadline) is not
  // stuck - run() stops Deadline, not Finished - and a later run without a deadline lists its
  // turns, after which a child can be chosen. Before the fix the root stayed stuck for good:
  // every later run stopped Finished at once and choose() threw.
  const GameState s = fixture("q3-root.5dpgn", q3());
  mcts::SearchParams p = quiet();
  p.split = mcts::TurnSplit::Flat;
  p.generator_nodes = 1;
  const std::unique_ptr<mcts::Evaluator> inner = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  Sleepy slow(*inner, 0.2);
  mcts::Search search(s, p);
  mcts::Limits timed{64};
  timed.seconds = 0.05;
  CHECK(search.run(slow, timed) == mcts::StopReason::Deadline);
  CHECK(search.root_children().empty());
  CHECK(!search.node(search.root()).stuck);
  CHECK_EQ(search.stats().stuck, std::uint64_t{0});
  CHECK_NOTHROW(search.check_invariants());
  // One playout: the root's turns listed, a child visited (every flat node falls back here, so
  // more playouts mean more listings, some 0.3 s each).
  const mcts::StopReason r = search.run(*inner, mcts::Limits{1});
  std::fprintf(stderr, "  later run: %s, %zu root children, %llu fallback listings\n", mcts::stop_reason_name(r),
               search.root_children().size(), static_cast<unsigned long long>(search.stats().fallback_listings));
  CHECK(r == mcts::StopReason::Playouts);
  CHECK(!search.root_children().empty());
  CHECK(search.stats().fallback_listings >= 2);  // the root's again (every flat node falls back here)
  CHECK_NOTHROW(search.choose());
  CHECK_NOTHROW(search.check_invariants());
}

TEST(deadline_from_1e9_seconds_or_infinite_is_none) {
  // M2: THE DEADLINE's rule (t5d/mcts.hpp). Before the fix a double of about 9.22e9 s or more,
  // +infinity included, overflowed the conversion to the clock's ticks (undefined; here a
  // deadline some 292 years past), so a "no limit" left every classification Unknown.
  const auto t0 = std::chrono::steady_clock::now();
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(!mcts::deadline_after(t0, 0, "s").has_value());
  CHECK(!mcts::deadline_after(t0, mcts::kNoDeadlineSeconds, "s").has_value());
  CHECK(!mcts::deadline_after(t0, 1e10, "s").has_value());
  CHECK(!mcts::deadline_after(t0, 1e300, "s").has_value());
  CHECK(!mcts::deadline_after(t0, inf, "s").has_value());
  const std::optional<std::chrono::steady_clock::time_point> near = mcts::deadline_after(t0, 999'999'999.0, "s");
  REQUIRE(near.has_value());
  CHECK(*near > t0 + std::chrono::hours(24 * 365 * 31));
  CHECK_THROWS_AS(mcts::deadline_after(t0, std::nan(""), "s"), std::invalid_argument);
  CHECK_THROWS_AS(mcts::deadline_after(t0, -1, "s"), std::invalid_argument);
  CHECK_THROWS_AS(mcts::deadline_after(t0, -inf, "s"), std::invalid_argument);
  CHECK(mcts::time_left(t0 - std::chrono::seconds(1)) == std::chrono::nanoseconds{0});
  CHECK(mcts::time_left(*near) > std::chrono::hours(24 * 365 * 31));
  // Through Search::run: issue 406's Q3 root expanded with exact legality - no deadline leaves
  // none of its moves unknown; with the overflow every classification was cut (61 unknown edges
  // at a mate in two, in the review).
  const GameState s = fixture("q3-root.5dpgn", q3());
  mcts::SearchParams p = quiet();
  p.legality = LegalityOptions::exact();
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  std::size_t children = 0;
  for (const double secs : {0.0, 1e9, 1e10, 1e300, inf}) {
    mcts::Search search(s, p);
    mcts::Limits l{1};  // the root's expansion
    l.seconds = secs;
    CHECK(search.run(*ev, l) == mcts::StopReason::Playouts);
    std::size_t unknown = 0;
    for (const mcts::RootChild& c : search.root_children()) unknown += c.legality == MoveLegality::Unknown ? 1 : 0;
    std::fprintf(stderr, "  seconds %g: %zu root children, %zu unknown, %llu unknown edges\n", secs,
                 search.root_children().size(), unknown, static_cast<unsigned long long>(search.stats().unknown_edges));
    CHECK_EQ(unknown, std::size_t{0});
    CHECK_EQ(search.stats().unknown_edges, std::uint64_t{0});
    if (secs == 0.0) children = search.root_children().size();
    CHECK_EQ(search.root_children().size(), children);
  }
  mcts::Search refused(s, p);
  mcts::Limits nan_limit{4};
  nan_limit.seconds = std::nan("");
  CHECK_THROWS_AS(refused.run(*ev, nan_limit), std::invalid_argument);
  mcts::Limits negative{4};
  negative.seconds = -0.5;
  CHECK_THROWS_AS(refused.run(*ev, negative), std::invalid_argument);
}

TEST(play_from_1e9_seconds_or_infinite_plays_as_with_no_limit) {
  // M2 through play_game: a max_seconds of 1e9 or more, +infinity included, is no stop at all -
  // the game is the same as with none (deterministic: no deadline, one thread). Before the fix the
  // overflowed deadline left every exact verification Unknown and the game played the most
  // visited child known legal instead (Standard: 4 actions, in the review) or ended TimeLimit.
  const GameState s = variant_start(*find_variant("Standard"), RuleOptions{});
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<std::vector<Move>> reference;
  std::vector<std::vector<std::uint32_t>> reference_visits;  // each decision's root children's visits
  for (const double secs : {0.0, 1e10, inf}) {
    mcts::PlayParams pp;
    pp.search = quiet(pp.search);
    pp.playouts = 16;
    pp.max_actions = 3;
    pp.seed = 7;
    pp.decision_log = true;
    pp.max_seconds = secs;
    const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
    const mcts::GameRecord g = mcts::play_game(s, *ev, pp);
    std::vector<std::vector<std::uint32_t>> visits;
    for (const mcts::DecisionLog& d : g.decision_log) {
      visits.emplace_back();
      for (const mcts::DecisionLog::Child& c : d.children) visits.back().push_back(c.visits);
    }
    std::fprintf(stderr, "  max_seconds %g: %zu actions, %zu decisions, %s\n", secs, g.actions.size(), visits.size(),
                 mcts::game_end_name(g.end));
    CHECK(g.end != mcts::GameEnd::TimeLimit);
    if (secs == 0.0) {
      reference = g.actions;
      reference_visits = visits;
    }
    CHECK(g.actions == reference);
    CHECK(visits == reference_visits);  // every search the same: no classification cut
  }
  for (const double bad : {std::nan(""), -1.0}) {
    mcts::PlayParams pp;
    pp.search = quiet(pp.search);
    pp.max_seconds = bad;
    const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
    CHECK_THROWS_AS(mcts::play_game(s, *ev, pp), std::invalid_argument);
  }
}

TEST(play_totals_count_a_timed_out_decision) {
  // L5: the decision the safety stop ends still counts its search's work in the game's totals -
  // before the fix the loop broke before adding them, so a game whose first decision timed out
  // reported no playouts. Issue 406's Q3 node with no legality budget at all (every move unknown;
  // nothing verified in the search), two ways:
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  mcts::PlayParams pp;
  pp.search = quiet(pp.search);
  pp.search.split = mcts::TurnSplit::BoardMove;
  pp.search.legality = LegalityOptions::budgeted(0, 0);
  pp.search.verify_unknown = false;
  const auto report = [](const char* what, const mcts::GameRecord& g) {
    std::fprintf(stderr, "  %s: %.2f s, %s, %llu decisions, %llu playouts, expansions %.3f s, verification %.3f s (%llu)\n",
                 what, g.seconds, mcts::game_end_name(g.end), static_cast<unsigned long long>(g.decisions),
                 static_cast<unsigned long long>(g.playouts), g.seconds_search_expand, g.seconds_search_verify,
                 static_cast<unsigned long long>(g.verified_choices));
  };
  {  // (a) its search runs to the deadline - no choice is made after it (M1): deterministic
    pp.playouts = std::uint64_t{1} << 30;
    pp.max_seconds = 0.3;
    const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
    mcts::GameRecord g(s);
    CHECK_NOTHROW(g = mcts::play_game(s, *ev, pp));
    report("search cut", g);
    CHECK(g.end == mcts::GameEnd::TimeLimit);
    CHECK_EQ(g.decisions, std::uint64_t{0});
    CHECK(g.playouts > 0);
    CHECK(g.seconds_search_expand > 0);
  }
  {  // (b) its chosen move's exact verification, begun before the deadline, is cut: the evaluator
     // steers the choice onto a move whose exact proof takes some 0.8 s (531,432 nodes) and sleeps
     // through the search's first batch, leaving at most 0.15 s - less the turn start's verdict.
     // Where the verdict and the sleep outlast the stop (a slow machine), the stop falls in the
     // search and (b) ends as (a).
    pp.playouts = 2;
    pp.max_seconds = 1.0;
    Steering steer(find_move(s, "(+2T2)Kb4(+1T1)a4"));
    Sleepy ev(steer, 0.85, true);
    mcts::GameRecord g(s);
    CHECK_NOTHROW(g = mcts::play_game(s, ev, pp));
    report("verification cut", g);
    CHECK(g.end == mcts::GameEnd::TimeLimit);
    CHECK_EQ(g.decisions, std::uint64_t{0});
    CHECK(g.playouts >= 1);
    if (g.verified_choices > 0) CHECK(g.seconds_search_verify > 0);  // the cut verification's time, counted
  }
}

// ---- The review of 2026-11-09: the deadline's tests (M3, M4) --------------------------------
// Wall-clock bounds: each at least 5x the time measured here unmutated, and well below the
// mutant's (both numbers in each comment) - CI's engine job runs on 2-4-core runners.

namespace {

// The turn start after the first `actions` actions of a fixture's record.
GameState after_actions(const char* name, const RuleOptions& rules, int actions) {
  const PgnRecord record = read_pgn(read_all(fixture_path(name)));
  PgnReplay replay(record, {}, rules);
  for (int i = 0; i < actions && !replay.finished(); ++i) replay.play_next();
  return replay.state();
}

}  // namespace

TEST(search_deadline_cuts_the_roots_classification) {
  // M3, adopted (review_search_deadline_cuts_the_roots_classification): issue 406's Q3 node as a
  // flat root with exact legality - its own expansion classifies 141 moves exactly, some 12
  // million nodes. A 0.5 s deadline must cut it inside the first batch. Measured: 0.8 s
  // unmutated; with the leaves' classification untimed (the review's M07, M01) 16 s.
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  mcts::SearchParams p = quiet();
  p.split = mcts::TurnSplit::Flat;
  p.legality = LegalityOptions::exact();
  p.verify_legality = LegalityOptions::exact();
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search search(s, p);
  mcts::Limits limits{800};
  limits.seconds = 0.5;
  const auto t0 = std::chrono::steady_clock::now();
  search.run(*ev, limits);
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.2f s, %llu playouts, stopped by %s, expansions %.2f s\n", took,
               static_cast<unsigned long long>(search.stats().playouts), mcts::stop_reason_name(search.stats().last_stop),
               search.stats().seconds_expand);
  CHECK(took < 8.0);
}

TEST(legality_deadline_cuts_one_long_search) {
  // M3, adopted (review_legality_deadline_cuts_one_long_search): the 192-board position's named
  // move, whose exact classification ran past 600 s (MEASUREMENTS). A 0.5 s deadline must cut its
  // one tier-3 search part-way: Unknown. Measured: 0.5 s unmutated (the search alone); with the
  // time left never reaching the search (the review's L03) over 180 s.
  std::vector<Move> named;
  const GameState s = fixture("vso-seed4-verify.5dpgn", RuleOptions{}, &named);
  REQUIRE_EQ(named.size(), std::size_t{1});
  LegalityOptions o = LegalityOptions::exact();
  o.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  std::vector<MoveLegality> out;
  LegalityStats st;
  const auto t0 = std::chrono::steady_clock::now();
  classify_moves(s, named, out, o, &st);
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.3f s, %llu nodes, %zu unknown\n", took, static_cast<unsigned long long>(st.search_nodes),
               st.unknown);
  CHECK(took < 10.0);
  CHECK(out.at(0) == MoveLegality::Unknown);
}

TEST(legality_deadline_with_time_to_spare_settles_everything) {
  // M3, adopted (review_legality_deadline_with_time_to_spare_settles_everything): issue 406's Q3
  // root settles every move in 240 nodes. A deadline far off must change nothing - no move
  // unknown, each as the classification without a deadline says (the review's L04 and L05, the
  // time left's sign flipped and the deadline read inverted, cut every search at once).
  const GameState s = fixture("q3-root.5dpgn", q3());
  MoveList all;
  generate_moves(s, all);
  const std::vector<Move> moves(all.begin(), all.end());
  std::vector<MoveLegality> ref;
  classify_moves(s, moves, ref, LegalityOptions::exact());
  LegalityOptions o = LegalityOptions::exact();
  o.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  std::vector<MoveLegality> out;
  LegalityStats st;
  classify_moves(s, moves, out, o, &st);
  std::fprintf(stderr, "  %zu moves, %zu unknown, %llu nodes\n", moves.size(), st.unknown,
               static_cast<unsigned long long>(st.search_nodes));
  CHECK_EQ(st.unknown, std::size_t{0});
  CHECK(out == ref);
}

TEST(search_turn_start_checks_take_the_deadline) {
  // M3, adopted (review_search_turn_start_checks_take_the_deadline): the default-rules game's turn
  // start after 25 actions (155 boards, Black to move) as a flat root, mate_nodes raised to 2e8:
  // its children are White turn starts of ~190 boards, where E4's check can run for minutes. The
  // search's 2 s deadline must reach those checks (Unknown). Measured: 3.8 s unmutated (the flat
  // generator, never under the deadline, overshoots it); with the checks or the fallback listing
  // untimed (the review's M02, M04, M05) over 180 s.
  const GameState s = after_actions("vso-seed4-verdict.5dpgn", RuleOptions{}, 25);
  REQUIRE(s.to_move() == Colour::Black);
  mcts::SearchParams p = quiet();
  p.split = mcts::TurnSplit::Flat;
  p.mate_nodes = 200'000'000;
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Material);
  mcts::Search search(s, p);
  mcts::Limits limits{64};
  limits.seconds = 2.0;
  const auto t0 = std::chrono::steady_clock::now();
  search.run(*ev, limits);
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.2f s, %llu playouts, stopped by %s, terminal checks %.2f s (%llu checks, %llu unknown)\n",
               took, static_cast<unsigned long long>(search.stats().playouts),
               mcts::stop_reason_name(search.stats().last_stop), search.stats().seconds_mate,
               static_cast<unsigned long long>(search.stats().mate_checks),
               static_cast<unsigned long long>(search.stats().unknown_verdicts));
  CHECK(took < 30.0);
}

TEST(ply_lazy_verification_deadline) {
  // M3, adopted (review_ply_lazy_verification_deadline): the 192-board position under the ply
  // split with exact lazy verifications (c) - its named move's exact classification runs past
  // 600 s; the search's deadline must cut it. Measured: 2.0 s unmutated; with the lazy
  // verifications untimed (the review's M08) past 600 s.
  int made = 0;
  const GameState s = fixture("vso-seed4-verify.5dpgn", RuleOptions{}, nullptr, &made);
  mcts::SearchParams p = quiet();
  p.split = mcts::TurnSplit::BoardMove;
  p.verify_legality = LegalityOptions::exact();
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search search(s, p, made);
  mcts::Limits limits{16};
  limits.seconds = 2.0;
  const auto t0 = std::chrono::steady_clock::now();
  search.run(*ev, limits);
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.2f s, %llu playouts, stopped by %s, verifications %.2f s (%llu unknown)\n", took,
               static_cast<unsigned long long>(search.stats().playouts), mcts::stop_reason_name(search.stats().last_stop),
               search.stats().seconds_verify, static_cast<unsigned long long>(search.stats().verified_unknown));
  CHECK(took < 30.0);
}

TEST(search_far_deadline_changes_nothing) {
  // M3, adopted (review_search_far_deadline_changes_nothing): a deadline that never passes must
  // change nothing - the search is deterministic. The ply split from issue 406's Q3 node, lazy
  // verifications within 100,000 nodes, 16 playouts (the review's 400,000 and 40 took 31 s): with
  // no deadline and with one 120 s away, the same tree (digest) and the same verification counts
  // (the review's M11, the deadline in milliseconds - 0.12 s - and L04 cut the classifications).
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  mcts::SearchParams p = quiet();
  p.split = mcts::TurnSplit::BoardMove;
  p.verify_legality = LegalityOptions::nodes(100'000);
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search a(s, p);
  a.run(*ev, mcts::Limits{16});
  mcts::Search b(s, p);
  mcts::Limits far{16};
  far.seconds = 120.0;
  b.run(*ev, far);
  std::fprintf(stderr, "  digests %llx / %llx; verified legal %llu/%llu, unknown %llu/%llu\n",
               static_cast<unsigned long long>(a.digest()), static_cast<unsigned long long>(b.digest()),
               static_cast<unsigned long long>(a.stats().verified_legal),
               static_cast<unsigned long long>(b.stats().verified_legal),
               static_cast<unsigned long long>(a.stats().verified_unknown),
               static_cast<unsigned long long>(b.stats().verified_unknown));
  CHECK_EQ(a.digest(), b.digest());
  CHECK_EQ(a.stats().verified_unknown, b.stats().verified_unknown);
  CHECK_EQ(a.stats().verified_legal, b.stats().verified_legal);
}

TEST(tree_form_inherited_expansions_take_the_deadline) {
  // M3, adopted (review_tree_form_inherited_no_gates): the tree form with no interaction gate at
  // issue 406's Q3 node - every mid-turn node below a turn start expanded on inherited priors
  // inside selection, up to six per simulation, each classification up to 400,000 nodes (some
  // 0.8 s). The root's own expansion ends first; the 1.2 s deadline then falls among the second
  // batch's inherited expansions and must reach them. Measured: 1.25 s unmutated; with the
  // inherited expansions untimed (the review's M06) 9.5 s.
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  mcts::SearchParams p = quiet();
  p.split = mcts::TurnSplit::Tree;
  p.tree_gates = 0;
  p.legality = LegalityOptions::nodes(400'000);
  p.verify_legality = LegalityOptions::nodes(400'000);
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search search(s, p);
  mcts::Limits limits{800};
  limits.seconds = 1.2;
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_NOTHROW(search.run(*ev, limits));
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.2f s, %llu playouts, stopped by %s, expansions %.2f s, verifications %.2f s\n", took,
               static_cast<unsigned long long>(search.stats().playouts), mcts::stop_reason_name(search.stats().last_stop),
               search.stats().seconds_expand, search.stats().seconds_verify);
  CHECK(took < 7.0);
}

TEST(play_no_child_known_legal_at_the_deadline_ends_time_limit) {
  // M4, adopted (review_play_no_child_known_legal_at_the_deadline_ends_time_limit), given a
  // verification the deadline cuts: no choice is made after the deadline (M1), so the choice must
  // come before it. Issue 406's Q3 node with no legality budget (every move unknown, nothing
  // verified in the search); the evaluator steers the choice onto a move whose exact proof takes
  // some 0.8 s and sleeps through the search's first batch, so the verification begins with at
  // most 0.15 s left and is cut: no child known legal, TimeLimit at the turn start - no throw (the
  // review's S05), no action. Where the stop falls in the search (a slow machine) the game ends
  // TimeLimit there.
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  mcts::PlayParams pp;
  pp.search = quiet(pp.search);
  pp.search.split = mcts::TurnSplit::BoardMove;
  pp.search.legality = LegalityOptions::budgeted(0, 0);
  pp.search.verify_unknown = false;
  pp.match_play = true;  // the most visited child: the steered one
  pp.playouts = 2;
  pp.max_seconds = 1.0;
  Steering steer(find_move(s, "(+2T2)Kb4(+1T1)a4"));
  Sleepy ev(steer, 0.85, true);
  const auto t0 = std::chrono::steady_clock::now();
  mcts::GameRecord g(s);
  CHECK_NOTHROW(g = mcts::play_game(s, ev, pp));
  const double took = seconds_since(t0);
  std::fprintf(stderr, "  %.2f s, %zu actions, %s, %llu verified choices, verification %.3f s, %llu decisions\n", took,
               g.actions.size(), mcts::game_end_name(g.end), static_cast<unsigned long long>(g.verified_choices),
               g.seconds_search_verify, static_cast<unsigned long long>(g.decisions));
  CHECK(g.end == mcts::GameEnd::TimeLimit);
  CHECK(g.actions.empty());
  CHECK_EQ(g.decisions, std::uint64_t{0});
}

TEST(play_no_move_is_played_after_the_deadline) {
  // M4, adopted (review_play_no_move_is_played_after_the_deadline): as the test above, asserting
  // what the deadline must stop - no move shown legal by a cut verification is played, so no
  // decision is made (the review's S06: make_legal's result ignored, the move played), and the
  // cut verification stops at the deadline (S04: the verification untimed ran its 0.8 s past
  // it). Measured: 1.0 s unmutated; S04 1.47 s.
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  mcts::PlayParams pp;
  pp.search = quiet(pp.search);
  pp.search.split = mcts::TurnSplit::BoardMove;
  pp.search.legality = LegalityOptions::budgeted(0, 0);
  pp.search.verify_unknown = false;
  pp.playouts = 2;
  pp.max_seconds = 1.0;
  pp.decision_log = true;
  Steering steer(find_move(s, "(+2T2)Kb4(+1T1)a4"));
  Sleepy ev(steer, 0.85, true);
  const mcts::GameRecord g = mcts::play_game(s, ev, pp);
  std::fprintf(stderr, "  %.2f s, %zu actions, %s, %llu verified choices, verification %.3f s, %llu decisions\n",
               g.seconds, g.actions.size(), mcts::game_end_name(g.end),
               static_cast<unsigned long long>(g.verified_choices), g.seconds_search_verify,
               static_cast<unsigned long long>(g.decisions));
  CHECK(g.end == mcts::GameEnd::TimeLimit);
  CHECK_EQ(g.decisions, std::uint64_t{0});
  CHECK(g.decision_log.empty());
  CHECK(g.seconds < pp.max_seconds + 0.3);  // a cut verification ends at the deadline
}

TEST(play_fallback_after_a_cut_verification_plays_a_child_known_legal) {
  // M4: the chosen move's exact verification cut, a child the search showed legal (lazy
  // verifications on) is played instead - the most visited one known legal. Issue 406's Q3 node,
  // no legality budget in the expansion, lazy verifications within 60,000 nodes: the evaluator
  // steers the search to a move whose exact proof needs 531,432 nodes (it stays unknown) and then
  // to a legal one whose proof needs 43,424 (shown legal). Where the stop falls inside the first
  // move's exact proof (some 0.8 s, from a choice made at about 0.1 s here: every stop), the
  // fallback plays the second; with more time the proof ends first (the first move illegal; the
  // second chosen next), and with less the stop falls in the search. Three stops, so that some
  // machine speed meets each path; whatever the path, the first move played is legal (exact
  // classification here), every action recorded is, and the game ends TimeLimit.
  const GameState s = fixture("q3-flat-node.5dpgn", q3());
  const Move steered = find_move(s, "(+2T2)Kb4(+1T1)a4");
  const Move legal = find_move(s, "(+1T2)Ka3(+2T2)a3");
  int fallbacks = 0;
  for (const double stop : {0.4, 0.6, 0.8}) {
    mcts::PlayParams pp;
    pp.search = quiet(pp.search);
    pp.search.split = mcts::TurnSplit::BoardMove;
    pp.search.legality = LegalityOptions::budgeted(0, 0);
    pp.search.verify_unknown = true;
    pp.search.verify_legality = LegalityOptions::nodes(60'000);
    pp.match_play = true;
    pp.playouts = 4;  // the root, the first move twice (lazily verified: unknown), the second (legal)
    pp.max_seconds = stop;
    pp.decision_log = true;
    TwoSteering steer(steered, legal);
    mcts::GameRecord g(s);
    CHECK_NOTHROW(g = mcts::play_game(s, steer, pp));
    CHECK(g.end == mcts::GameEnd::TimeLimit);
    GameState t = s;  // every action recorded legal: replayed and submitted under C4
    for (const std::vector<Move>& a : g.actions) {
      for (const Move& m : a) t.apply(m);
      CHECK(t.submission_status() == SubmitStatus::Safe);
      t.submit();
    }
    bool fallback = false;
    if (!g.decision_log.empty()) {
      const mcts::DecisionLog& d = g.decision_log.front();
      const std::vector<Move>& played = d.children.at(d.chosen).step.moves;
      REQUIRE_EQ(played.size(), std::size_t{1});
      std::vector<MoveLegality> out;
      classify_moves(s, played, out, LegalityOptions::exact());
      CHECK(out.at(0) == MoveLegality::Legal);
      fallback = g.verified_choices >= 1 && !(played.front() == steered);
    }
    fallbacks += fallback ? 1 : 0;
    std::fprintf(stderr,
                 "  stop %.1f s: %.2f s, %s, %llu verified choices, verification %.3f s, %llu decisions%s\n", stop,
                 g.seconds, mcts::game_end_name(g.end), static_cast<unsigned long long>(g.verified_choices),
                 g.seconds_search_verify, static_cast<unsigned long long>(g.decisions),
                 fallback ? " - a child known legal played in the steered move's place" : "");
  }
  std::fprintf(stderr, "  %d of 3 stops played the fallback\n", fallbacks);
}

TEST(play_late_verdict_gets_the_time_left) {
  // M4, adopted (review_play_late_verdict_gets_no_time), given an action played before the
  // deadline (no choice is made after it, M1): the default-rules game's turn start after 25
  // actions (155 boards, Black to move) under the flat form, one playout (the root's turns, a
  // child) with the search's own terminal checks cheap (mate_nodes 20,000) and the evaluator
  // sleeping 1 s in its first batch: Black's action is played at about 1.3 s, and the next turn
  // start's verdict (~190 boards, minutes untimed) gets the time left and is cut at the 6 s
  // deadline. Measured: 6.01 s unmutated; with each verdict given max_seconds (the review's S08)
  // the game overshoots by the time before that verdict - at least the 1 s sleep: 7.31 s.
  const GameState s = after_actions("vso-seed4-verdict.5dpgn", RuleOptions{}, 25);
  mcts::PlayParams pp;
  pp.search = quiet(pp.search);
  pp.search.split = mcts::TurnSplit::Flat;
  pp.search.mate_nodes = 20'000;
  pp.playouts = 1;
  pp.max_seconds = 6.0;
  const std::unique_ptr<mcts::Evaluator> material = mcts::make_evaluator(mcts::BuiltinEvaluator::Material);
  Sleepy ev(*material, 1.0, true);
  const mcts::GameRecord g = mcts::play_game(s, ev, pp);
  std::fprintf(stderr, "  %.2f s (verdicts %.2f s), %zu actions, %s\n", g.seconds, g.seconds_verdicts, g.actions.size(),
               mcts::game_end_name(g.end));
  CHECK(g.end == mcts::GameEnd::TimeLimit);
  CHECK(g.seconds < pp.max_seconds + 0.2);  // the late verdict cut at the deadline
}
