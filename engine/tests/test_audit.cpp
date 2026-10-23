// Stage E10, its second part: the offline audit lane's tooling (tools/audit.hpp;
// design spec 11.4; register H44, H42): the decision log round-trips and
// rebuilds every position; the sampler is seeded and keeps its strata, rates
// and caps; the runner refutes an action that allows a mate in one (its
// certificate confirmed by E4), finds a missed mate and reports a safe
// action's completed depth, each contradiction to the curriculum's sink; the
// report counts as 6b'' counts, at the range edges; nothing is exported; and
// end to end through t5d-play --decision-log and t5d-audit.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "audit.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using namespace t5d::audit;
using t5d_test::game;

namespace {

GameState variant(const char* name) {
  const Variant* v = find_variant(name);
  if (v == nullptr) throw std::logic_error(std::string("no variant ") + name);
  return variant_start(*v, RuleOptions{});
}

mcts::PlayParams logged(mcts::TurnSplit split, std::uint32_t actions) {
  mcts::PlayParams p;
  p.search.expand.square_features = SquareFeatures::None;
  p.search.synchronous = true;
  p.search.split = split;
  p.search.asset_map = false;
  p.playouts = 48;
  p.max_actions = actions;
  p.turn_cap.boards = 300;
  p.decision_log = true;
  return p;
}

// A one-action custom game's log with one event.
Log custom_log(const std::vector<std::string>& boards, int size, const std::vector<std::string>& action,
               Category category = Category::Final) {
  Log log;
  GameLog g;
  g.record = "custom-1";
  g.variant = "custom";
  g.start = boards;
  g.width = g.height = size;
  g.actions.push_back(action);
  log.games.push_back(g);
  Job j;
  j.id = 1;
  j.event.record = g.record;
  j.event.category = category;
  j.event.step.moves = action;
  j.event.boards = 1;
  j.range = 0;
  log.jobs.push_back(j);
  return log;
}

class Spy final : public CurriculumSink {
 public:
  void emit(const Contradiction& c) override { kinds.push_back(c.kind); }
  std::vector<std::string> kinds;
  std::size_t count(const std::string& k) const {
    return static_cast<std::size_t>(std::count(kinds.begin(), kinds.end(), k));
  }
};

RunOptions small_budgets() {
  RunOptions o;
  o.guard_max_depth = 2;
  o.guard_nodes = 400'000;
  o.mate_nodes = 200'000;
  return o;
}

}  // namespace

TEST(audit_decision_log_round_trips_and_rebuilds_every_position) {
  const std::unique_ptr<mcts::Evaluator> random = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, 4);
  for (const mcts::TurnSplit split : {mcts::TurnSplit::BoardMove, mcts::TurnSplit::Flat}) {
    const mcts::GameRecord g = play_game(variant("Standard - Two Timelines"), *random, logged(split, 4));
    REQUIRE(!g.decision_log.empty());
    EventOptions eo;
    eo.first_pv = true;
    const Log log = log_of_game(g, "Standard - Two Timelines", "game-a", eo);
    REQUIRE_EQ(log.games.size(), std::size_t{1});
    // The text round-trips exactly.
    const std::string text = log_text(log);
    const Log back = read_log(text);
    CHECK_EQ(log_text(back), text);
    CHECK_EQ(back.events.size(), log.events.size());
    // One final move per action, its step the action played.
    std::size_t finals = 0;
    for (const Event& e : back.events) {
      if (e.category != Category::Final) continue;
      ++finals;
      CHECK(e.step.moves == back.games[0].actions.at(e.action));
    }
    CHECK_EQ(finals, g.actions.size());
    // Every turn start rebuilt by replaying the game's actions.
    GameState s = g.start;
    for (std::uint32_t a = 0; a < g.actions.size(); ++a) {
      CHECK(turn_start_of(back.games[0], a) == s);
      for (const Move& m : g.actions[a]) s.apply(m);
      s.submit();
    }
    CHECK(turn_start_of(back.games[0], static_cast<std::uint32_t>(g.actions.size())) == g.final_state);
    // Every event's decision position rebuilt; a demotion never the chosen child.
    std::size_t pv = 0;
    for (const Event& e : back.events) {
      GameState d = turn_start_of(back.games[0], e.action);
      for (const std::string& m : e.prefix) d.apply(move_of_text(d, m));
      CHECK_EQ(static_cast<int>(d.multiverse().board_count()), e.boards);
      pv += e.category == Category::PvChange ? 1u : 0u;
    }
    CHECK(pv > 0);
  }
  // Another writer's layout (key order, spacing) reads the same.
  const std::string line =
      "{ \"kind\": \"game\", \"version\": 1, \"format\": \"t5d-decisions\", \"record\": \"r\", \"variant\": "
      "\"Standard\", \"start\": [], \"width\": 0, \"height\": 0, \"rules\": 0, \"split\": \"ply\", \"actions\": "
      "[[\"(0T1)Pe2(0T1)e3\"]] }\n";
  const Log other = read_log(line);
  REQUIRE_EQ(other.games.size(), std::size_t{1});
  CHECK_EQ(other.games[0].actions.at(0).at(0), std::string("(0T1)Pe2(0T1)e3"));
  CHECK_THROWS_AS(read_log("{\"format\":\"other\",\"version\":1,\"kind\":\"game\"}"), std::invalid_argument);
  CHECK_THROWS_AS(read_log("{\"format\":\"t5d-decisions\",\"version\":2,\"kind\":\"game\"}"), std::invalid_argument);
}

TEST(audit_sampler_is_seeded_and_keeps_its_strata) {
  Log log;
  GameLog g;
  g.record = "g";
  g.variant = "Standard";
  log.games.push_back(g);
  for (int i = 0; i < 400; ++i) {
    Event e;
    e.record = "g";
    e.action = static_cast<std::uint32_t>(i / 3);
    e.decision = static_cast<std::uint32_t>(i);
    e.category = static_cast<Category>(i % 3);
    e.boards = std::array<int, 4>{10, 60, 150, 250}[static_cast<std::size_t>((i / 3) % 4)];
    e.step.moves = {"(0T1)Pe2(0T1)e" + std::to_string(i)};
    log.events.push_back(e);
  }
  SampleOptions o;
  o.seed = 7;
  o.rate = {1.0, 0.5, 0.0};
  o.cap = 1000;
  const Log a = sample(log, o);
  const Log b = sample(log, o);
  CHECK_EQ(log_text(a), log_text(b));  // seeded: the same twice
  o.seed = 8;
  CHECK(log_text(sample(log, o)) != log_text(a));
  std::size_t finals = 0;
  std::size_t pvs = 0;
  std::size_t demotions = 0;
  std::uint32_t last = 0;
  for (const Job& j : a.jobs) {
    finals += j.event.category == Category::Final ? 1u : 0u;
    pvs += j.event.category == Category::PvChange ? 1u : 0u;
    demotions += j.event.category == Category::Demotion ? 1u : 0u;
    CHECK_EQ(j.range, tools::sixbpp::range_of(j.event.boards));
    CHECK(j.event.decision >= last);  // the events' order
    last = j.event.decision;
  }
  CHECK_EQ(finals, std::size_t{134});  // rate 1: every final (decisions 0, 3, ..., 399)
  CHECK(pvs > 40 && pvs < 95);         // rate 0.5 of 133
  CHECK_EQ(demotions, std::size_t{0}); // rate 0
  CHECK_EQ(a.games.size(), std::size_t{1});
  // The cap: at most so many per (category, range).
  o.seed = 7;
  o.rate = {1.0, 1.0, 1.0};
  o.cap = 3;
  const Log capped = sample(log, o);
  std::map<std::pair<int, int>, std::size_t> per;
  for (const Job& j : capped.jobs) ++per[{static_cast<int>(j.event.category), j.range}];
  CHECK_EQ(per.size(), std::size_t{12});
  for (const auto& [k, n] : per) CHECK_EQ(n, std::size_t{3});
  // The jobs read back.
  const Log read = read_log(log_text(capped));
  CHECK_EQ(read.jobs.size(), capped.jobs.size());
  CHECK_EQ(read.jobs.front().id, capped.jobs.front().id);
}

TEST(audit_runner_refutes_an_allowed_mate_with_a_confirmed_certificate) {
  // Rc5-a5 lets Black's Ne3-c2 mate (test_ladder.cpp's position): Guard's
  // question offline refutes it at depth 1, E4 confirming the certificate -
  // one allowed_mate to the sink.
  const Log log = custom_log({"[2R2/4k/4n/PP3/KN3:0:1:w]"}, 5, {"(0T1)Rc5(0T1)a5"});
  Spy spy;
  const std::vector<JobResult> r = run(log, small_budgets(), &spy);
  REQUIRE_EQ(r.size(), std::size_t{1});
  CHECK(r[0].guard.asked);
  CHECK(r[0].guard.refuted);
  CHECK_EQ(r[0].guard.refutation_depth, 1);
  CHECK(r[0].guard.check == tools::sixbpp::Check::Confirmed);
  CHECK_EQ(spy.count("allowed_mate"), std::size_t{1});
}

TEST(audit_runner_finds_a_missed_mate_and_a_safe_actions_depth) {
  // Mate in one (Qe4-... from test_v3_support.hpp's position): a quiet king
  // move instead - one missed_mate, Guard safe to its deepest depth (Black
  // has a bare king).
  const GameState s = game({"[k4/4Q/1K3/5/5:0:1:w]"}, BoardSize{5, 5});
  std::string quiet;
  MoveList moves;
  generate_moves(s, moves);
  for (const Move& m : moves) {
    if (m.piece.kind() != PieceKind::King) continue;
    GameState t = s;
    t.apply(m);
    if (t.submission_status() != SubmitStatus::Safe) continue;
    t.submit();
    if (detect_mate(t, Detector::Hypercuboid).verdict == Outcome::Ongoing) {
      quiet = to_string(m, s.multiverse().even());
      break;
    }
  }
  REQUIRE(!quiet.empty());
  {
    Spy spy;
    const std::vector<JobResult> r = run(custom_log({"[k4/4Q/1K3/5/5:0:1:w]"}, 5, {quiet}), small_budgets(), &spy);
    REQUIRE_EQ(r.size(), std::size_t{1});
    CHECK(r[0].mate.status == ProofStatus::Proven);
    CHECK_EQ(r[0].mate.distance, 1);
    CHECK(r[0].mate.check == tools::sixbpp::Check::Confirmed);
    CHECK(!r[0].guard.refuted);
    CHECK_EQ(r[0].guard.completed_depth, 2);
    CHECK_EQ(spy.kinds.size(), std::size_t{1});
    CHECK_EQ(spy.count("missed_mate"), std::size_t{1});
  }
  // Two bare kings: safe to depth 2, no mate, no contradiction.
  {
    const GameState k = game({"[k4/5/5/5/4K:0:1:w]"}, BoardSize{5, 5});
    MoveList km;
    generate_moves(k, km);
    REQUIRE(!km.empty());
    Spy spy;
    const std::vector<JobResult> r =
        run(custom_log({"[k4/5/5/5/4K:0:1:w]"}, 5, {to_string(km[0], false)}), small_budgets(), &spy);
    REQUIRE_EQ(r.size(), std::size_t{1});
    CHECK(r[0].guard.asked && !r[0].guard.refuted);
    CHECK_EQ(r[0].guard.completed_depth, 2);
    CHECK(r[0].mate.status != ProofStatus::Proven);
    CHECK(spy.kinds.empty());
  }
}

TEST(audit_report_counts_as_6bpp_counts_at_the_range_edges) {
  std::vector<JobResult> results;
  const auto add = [&](int boards, bool refuted, tools::sixbpp::Check check, int depth, int completed,
                       Category category = Category::Final) {
    JobResult r;
    r.job.event.category = category;
    r.job.event.boards = boards;
    r.job.range = tools::sixbpp::range_of(boards);
    r.guard.asked = true;
    r.guard.refuted = refuted;
    r.guard.check = check;
    r.guard.refutation_depth = depth;
    r.guard.completed_depth = completed;
    results.push_back(r);
  };
  using C = tools::sixbpp::Check;
  add(49, true, C::Confirmed, 1, 0);   // range 0: a shallow mate
  add(50, false, C::None, 0, 2);       // range 1: safe
  add(99, true, C::Refuted, 1, 0);     // range 1: withheld
  add(100, true, C::Unconfirmed, 1, 0);  // range 2: unconfirmed
  add(199, false, C::None, 0, 0);      // range 2: undecided
  add(200, true, C::Confirmed, 2, 1);  // range 3: a mate, but not shallow at depth 1 - decided by depth 1
  add(250, true, C::Confirmed, 1, 0, Category::PvChange);  // not a final: no final counted
  const Report rep = make_report(results, 1);
  CHECK_EQ(rep.ranges[0].finals, std::size_t{1});
  CHECK_EQ(rep.ranges[0].shallow, std::size_t{1});
  CHECK_EQ(rep.ranges[0].decided, std::size_t{1});
  CHECK_EQ(rep.ranges[1].finals, std::size_t{2});
  CHECK_EQ(rep.ranges[1].decided, std::size_t{1});
  CHECK_EQ(rep.ranges[1].withheld, std::size_t{1});
  CHECK_EQ(rep.ranges[2].unconfirmed, std::size_t{1});
  CHECK_EQ(rep.ranges[2].undecided, std::size_t{1});
  CHECK_EQ(rep.ranges[3].finals, std::size_t{1});
  CHECK_EQ(rep.ranges[3].decided, std::size_t{1});
  CHECK_EQ(rep.ranges[3].shallow, std::size_t{0});
  CHECK_EQ(rep.ranges[0].shallow_rate(), 1.0);
  for (const RangeReport& rr : rep.ranges) CHECK(rr.too_few);  // under kFewTruthMates finals each
  const std::string text = report_text(rep);
  CHECK(text.find("too few to read") != std::string::npos);
  CHECK(text.find("weak at 200+") != std::string::npos);
  CHECK(report_json(rep).find("\"shallow_rate\"") != std::string::npos);
  // At depth 2 the range-3 mate is shallow.
  const Report deeper = make_report(results, 2);
  CHECK_EQ(deeper.ranges[3].shallow, std::size_t{1});
  // Results round-trip.
  std::string lines;
  for (const JobResult& r : results) lines += result_json(r) + "\n";
  const std::vector<JobResult> back = read_results(lines);
  REQUIRE_EQ(back.size(), results.size());
  for (std::size_t i = 0; i < back.size(); ++i) CHECK_EQ(result_json(back[i]), result_json(results[i]));
}

TEST(audit_exports_nothing_and_repeats_itself) {
  // No solved store is reachable from the runner (its provers' stores are
  // their own, discarded with them - H19, H23): the same jobs give the same
  // results twice, nothing carried from one run to the next.
  const Log log = custom_log({"[2R2/4k/4n/PP3/KN3:0:1:w]"}, 5, {"(0T1)Rc5(0T1)a5"});
  const std::vector<JobResult> a = run(log, small_budgets(), nullptr);
  const std::vector<JobResult> b = run(log, small_budgets(), nullptr);
  REQUIRE_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK_EQ(a[i].guard.refuted, b[i].guard.refuted);
    CHECK_EQ(a[i].guard.nodes, b[i].guard.nodes);
    CHECK_EQ(a[i].mate.nodes, b[i].mate.nodes);
  }
}

namespace {

std::string read_all(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int run_exe(const std::string& exe, const std::vector<std::string>& args, const std::filesystem::path& out,
            const std::filesystem::path& err) {
  namespace fs = std::filesystem;
  const auto quoted = [](const std::string& s) { return "\"" + s + "\""; };
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

TEST(audit_end_to_end_through_t5d_play_and_t5d_audit) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "audit";
  fs::create_directories(dir);
  const fs::path log = dir / "decisions.jsonl";
  const fs::path jobs = dir / "jobs.jsonl";
  const fs::path results = dir / "results.jsonl";
  const fs::path contradictions = dir / "contradictions.jsonl";
  REQUIRE_EQ(run_exe(T5D_PLAY_EXE,
                     {"--variant", "Very Small - Open", "--playouts", "24", "--max-actions", "4", "--synchronous",
                      "--no-assets", "--evaluator", "random", "--decision-log", log.string()},
                     dir / "record.5dpgn", dir / "play.err"),
             0);
  CHECK(read_all(dir / "play.err").find("decision log") != std::string::npos);
  const Log logged = read_log(read_all(log));
  REQUIRE(!logged.events.empty());
  REQUIRE_EQ(run_exe(T5D_AUDIT_EXE, {"sample", "--seed", "3", "--out", jobs.string(), log.string()}, dir / "s.out",
                     dir / "s.err"),
             0);
  const Log sampled = read_log(read_all(jobs));
  CHECK(!sampled.jobs.empty());
  REQUIRE_EQ(run_exe(T5D_AUDIT_EXE,
                     {"run", "--guard-nodes", "200000", "--mate-nodes", "100000", "--contradictions",
                      contradictions.string(), "--out", results.string(), jobs.string()},
                     dir / "r.out", dir / "r.err"),
             0);
  const std::vector<JobResult> got = read_results(read_all(results));
  CHECK_EQ(got.size(), sampled.jobs.size());
  REQUIRE_EQ(run_exe(T5D_AUDIT_EXE, {"report", "--json", (dir / "report.json").string(), results.string()},
                     dir / "report.txt", dir / "report.err"),
             0);
  CHECK(read_all(dir / "report.txt").find("finals") != std::string::npos);
  CHECK(read_all(dir / "report.json").find("\"ranges\"") != std::string::npos);
  // A bad command line: exit 2.
  CHECK_EQ(run_exe(T5D_AUDIT_EXE, {"sideways"}, dir / "x.out", dir / "x.err"), 2);
}

// ---- The E10 review's tests (2026-10-23) ------------------------------------------------------

TEST(audit_runner_takes_no_refutation_e4_rejects) {
  // The seam (RunOptions::certificate_hook_for_tests) substitutes a certificate
  // of another position for Guard's refutation of Rc5-a5: E4 does not confirm
  // it, so the check is not Confirmed and no allowed_mate goes to the sink
  // (kills reviewer B's A04: the check skipped).
  const Log log = custom_log({"[2R2/4k/4n/PP3/KN3:0:1:w]"}, 5, {"(0T1)Rc5(0T1)a5"});
  ProverOptions po;
  po.record_certificate = true;
  Prover prover(po);
  const GameState one = game({"[k4/4Q/1K3/5/5:0:1:w]"}, BoardSize{5, 5});
  const ProveResult other = prover.prove(one, ProveBudget::nodes(200'000), 3);
  REQUIRE(other.status == ProofStatus::Proven && other.certificate != nullptr);
  RunOptions o = small_budgets();
  std::size_t substituted = 0;
  o.certificate_hook_for_tests = [&](std::shared_ptr<const ProofCertificate>& c) {
    c = other.certificate;
    ++substituted;
  };
  Spy spy;
  const std::vector<JobResult> r = run(log, o, &spy);
  REQUIRE_EQ(r.size(), std::size_t{1});
  CHECK(substituted >= 1);
  CHECK(r[0].guard.refuted);
  CHECK(r[0].guard.check != tools::sixbpp::Check::Confirmed);
  CHECK_EQ(spy.count("allowed_mate"), std::size_t{0});
  CHECK(spy.kinds.empty());
}

TEST(audit_demotions_are_never_the_chosen_child_and_a_pv_child_is_logged) {
  // Reviewer B (kills A03, the chosen child logged as a demotion, and S04 and
  // M11, was_pv lost in the log).
  const std::unique_ptr<mcts::Evaluator> random = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, 4);
  const mcts::GameRecord g =
      play_game(variant("Standard - Two Timelines"), *random, logged(mcts::TurnSplit::BoardMove, 4));
  REQUIRE(!g.decision_log.empty());
  for (const mcts::DecisionLog& d : g.decision_log) {
    if (d.root_visits > 1) {
      CHECK(std::any_of(d.children.begin(), d.children.end(),
                        [](const mcts::DecisionLog::Child& c) { return c.was_pv; }));
    }
  }
  EventOptions eo;
  eo.demotion_share = 0.0;  // every child a candidate: only the chosen one must be left out
  const Log log = log_of_game(g, "Standard - Two Timelines", "game-d", eo);
  for (const mcts::DecisionLog& d : g.decision_log) {
    if (d.action >= g.actions.size()) continue;
    std::size_t demotions = 0;
    for (const Event& e : log.events) {
      demotions += e.category == Category::Demotion && e.action == d.action && e.decision == d.decision ? 1u : 0u;
    }
    CHECK_EQ(demotions, d.children.size() - 1);
  }
}

TEST(audit_runner_reports_an_abandoned_mate) {
  // Reviewer B (kills A06): a demoted child that begins the confirmed mate is
  // an abandoned mate.
  const GameState s = game({"[k4/4Q/1K3/5/5:0:1:w]"}, BoardSize{5, 5});
  std::string mating;
  MoveList moves;
  generate_moves(s, moves);
  for (const Move& m : moves) {
    GameState t = s;
    t.apply(m);
    if (t.submission_status() != SubmitStatus::Safe) continue;
    t.submit();
    if (detect_mate(t, Detector::Hypercuboid).verdict == Outcome::Checkmate) {
      mating = to_string(m, s.multiverse().even());
      break;
    }
  }
  REQUIRE(!mating.empty());
  Spy spy;
  const std::vector<JobResult> r =
      run(custom_log({"[k4/4Q/1K3/5/5:0:1:w]"}, 5, {mating}, Category::Demotion), small_budgets(), &spy);
  REQUIRE_EQ(r.size(), std::size_t{1});
  CHECK_EQ(spy.count("abandoned_mate"), std::size_t{1});
}
