// Stage E11: time management (t5d/timeman.hpp; design spec §8.6, §13.1's 6o,
// §15.4; register E7, H15, H41, I4).
//
// What is tested: the allocator's arithmetic by hand - §8.6's table reproduced
// from the dense cost model, the moves-left floor and default, increments, the
// per-decision split, 6o's three arms; the hard bound tightening with N; the
// stop/extend table case by case on constructed root statistics (smart pruning,
// the KLD, a solved root only with the proof channel, overlap with and without
// churn, volatility, disagreement through the proof hook, a falling and a
// rising eval, instability); the spiral guard at high N; the signals off; the
// hard bound held on a virtual clock against a slower evaluator than the model
// predicts, on the wall clock under load, and inside a batch on a 400-board
// position (the rules engine's calls answer Unknown, never a verdict); the
// model clock's runs bit-reproducible; the final choice among the eligible
// children after every stop; match play's clock end to end, its flag, and its
// refusal in self-play; the proof hook refused without the proof channel.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "play_record.hpp"
#include "synthetic.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/timeman.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using namespace t5d::mcts;
using t5d_test::game;

namespace {

GameState variant(const char* name) {
  const Variant* v = find_variant(name);
  if (v == nullptr) throw std::logic_error(std::string("no variant ") + name);
  return variant_start(*v, RuleOptions{});
}

bool near(double a, double b, double tolerance) { return std::abs(a - b) <= tolerance; }

// The tests' search: no square features (the built-in evaluators read none),
// no verification ladder (its jobs are test_ladder.cpp's).
SearchParams quiet_params() {
  SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.hunt = false;
  p.settling = false;
  p.asset_map = false;
  return p;
}

// §8.6's cost model written out independently: T (40 d + 4 T), T = 66 N + 16, d = 256.
double dense(double n) {
  const double t = 66 * n + 16;
  return t * (40 * 256 + 4 * t);
}

// A clock the test moves.
class ManualClock final : public Clock {
 public:
  double t = 0;
  double now() override { return t; }
};

// An evaluator that charges a clock for each leaf: `per_eval` seconds at 100
// boards, scaled to the leaf's board count as the cost model scales; it keeps
// its largest batch's charge.
class ChargingEvaluator final : public Evaluator {
 public:
  ChargingEvaluator(ManualClock& clock, double per_eval) : clock_(clock), per_eval_(per_eval) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    inner_->evaluate(batch, out);
    double charge = 0;
    for (const EvalRequest& r : batch) {
      charge += per_eval_ * cost_relative(TimeParams{}, static_cast<double>(r.state->multiverse().board_count()));
    }
    clock_.t += charge;
    largest_batch = std::max(largest_batch, charge);
  }
  double largest_batch = 0;

 private:
  ManualClock& clock_;
  double per_eval_;
  std::unique_ptr<Evaluator> inner_ = make_evaluator(BuiltinEvaluator::Uniform);
};

// A uniform evaluator that sleeps per batch (the wall clock's tests).
class SleepyEvaluator final : public Evaluator {
 public:
  explicit SleepyEvaluator(std::chrono::microseconds per_batch) : per_batch_(per_batch) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    inner_->evaluate(batch, out);
    std::this_thread::sleep_for(per_batch_);
  }

 private:
  std::chrono::microseconds per_batch_;
  std::unique_ptr<Evaluator> inner_ = make_evaluator(BuiltinEvaluator::Uniform);
};

// A plan of `soft` evaluations at 100 boards, a generous hard bound, the
// default parameters' scale.
TimePlan plan_of(double soft, double hard_seconds = 1e9) {
  TimePlan p;
  p.boards = 100;
  p.seconds_per_evaluation = 0.01;
  p.soft_evaluations = soft;
  p.soft_seconds = soft * 0.01;
  p.hard_seconds = hard_seconds;
  return p;
}

// A snapshot after `evals` evaluations (playouts as many) with `visits` at the root.
RootSnapshot snap(std::uint64_t evals, std::vector<std::uint32_t> visits) {
  RootSnapshot r;
  r.evaluations = evals;
  r.playouts = evals;
  r.visits = std::move(visits);
  for (const std::uint32_t v : r.visits) r.root_visits += v;
  r.batch_size = 8;
  return r;
}

std::array<float, kValueBins> peaked(int bin) {
  std::array<float, kValueBins> a{};
  a[static_cast<std::size_t>(bin)] = 1.0f;
  return a;
}

}  // namespace

// ---- the allocator by hand ------------------------------------------------------------

TEST(timeman_reproduces_section_8_6s_table) {
  // §8.6: at N = 100 with ~100 board-moves remaining, every board-move adding
  // a board - so 1 board per action, every action counted (step 1).
  TimeParams p;
  p.boards_per_action = {1, 1, 1};
  p.cost_fixed_fraction = 0;  // the dense model alone, as the table
  CHECK(near(cost_shape(p, 100), dense(100), 1e-6 * dense(100)));
  // "Per-eval cost, now -> final move: 1.0 -> 3.4x" (the final move at N = 199;
  // 3.43 at N = 200).
  CHECK(near(cost_shape(p, 199) / cost_shape(p, 100), 3.403, 0.001));
  CHECK(near(cost_shape(p, 200) / cost_shape(p, 100), 3.434, 0.001));
  // "Equal-sims (integral) share 0.48%", against the naive 1.00%.
  const double sum = integral_cost_sum(p, 100, 100, 1);
  double by_hand = 0;
  for (int k = 0; k < 100; ++k) by_hand += dense(100 + k) / dense(100);
  CHECK(near(sum, by_hand, 1e-9 * by_hand));
  CHECK(near(100.0 / sum, 0.4795, 0.0005));  // 0.48%
  // "Flat wall-clock leaves the last 20 moves ~3.1x under-simmed": equal time
  // per move gives each move evaluations in inverse proportion to its cost, so
  // the last 20 get ~3.1x fewer than this one.
  double last20 = 0;
  for (int k = 80; k < 100; ++k) last20 += dense(100 + k) / dense(100);
  CHECK(near(last20 / 20, 3.11, 0.02));
  // The TimeManager's own integral, on the plan: this move's share.
  TimeContext ctx;
  ctx.remaining_seconds = 1000;
  ctx.moves_left_turns = 200;  // 100 of ours
  p.overhead_seconds = 0;
  const TimePlan plan = plan_decision(p, 0.01, 100, 0, true, ctx, std::nan(""));
  CHECK_EQ(plan.our_actions, 100.0);
  CHECK(near(plan.share, 1.0 / integral_cost_sum(p, 100, 100, 2), 1e-12));
}

TEST(timeman_plans_by_hand) {
  TimeParams p;  // the defaults: Integral, overhead 0.05 s, floor 20 turns, default 40 ...
  // ... and a fixed share of 5% of the cost at 100 boards:
  CHECK(near(cost_relative(p, 30), 0.05 + 0.95 * dense(30) / dense(100), 1e-12));
  CHECK_EQ(cost_relative(p, 100), 1.0);
  p.cost_fixed_fraction = 0;  // the arithmetic below: the dense model alone
  const double s100 = 0.01;
  TimeContext ctx;
  ctx.remaining_seconds = 100;
  ctx.increment_seconds = 1;
  ctx.moves_left_turns = 40;  // 20 of our actions
  // The ply split at a turn start with one mandatory board: 1 + submit + 0.25.
  const TimePlan a = plan_decision(p, s100, 100, 1, false, ctx, std::nan(""));
  CHECK(a.moves_left_given);
  CHECK_EQ(a.our_actions, 20.0);
  CHECK(near(a.usable_seconds, 100 + 1 * 19 - 0.05 * 20, 1e-9));
  double sum = 0;
  double n = 100;
  for (int k = 0; k < 20; ++k) {
    sum += dense(n) / dense(100);
    n += 2 * 1.77;  // both players' actions, M22's 1.77 at 100+ boards
  }
  CHECK(near(a.share, 1 / sum, 1e-12));
  CHECK(near(a.seconds_per_evaluation, s100, 1e-15));
  CHECK(near(a.action_evaluations, 118 / sum / s100, 1e-6));
  CHECK_EQ(a.decisions_left, 2.25);
  CHECK(near(a.soft_evaluations, 118 / sum / s100 / 2.25, 1e-6));
  CHECK(near(a.soft_seconds, a.soft_evaluations * s100, 1e-12));
  const double multiple = 5 + (100 - 50) / 350.0 * (2 - 5);
  CHECK(near(hard_multiple(p, 100), multiple, 1e-12));
  CHECK(near(a.hard_seconds, multiple * a.soft_seconds, 1e-9));
  // The flat form decides the whole action.
  const TimePlan flat = plan_decision(p, s100, 100, 1, true, ctx, std::nan(""));
  CHECK_EQ(flat.decisions_left, 1.0);
  CHECK(near(flat.soft_evaluations, a.soft_evaluations * 2.25, 1e-6));
  // The conservative floor: 4 turns predicted plan as 20.
  ctx.moves_left_turns = 4;
  const TimePlan floored = plan_decision(p, s100, 100, 0, true, ctx, std::nan(""));
  CHECK_EQ(floored.moves_left_turns, 20.0);
  CHECK_EQ(floored.our_actions, 10.0);
  // No prediction - the context's NaN and none at the root: the default.
  ctx.moves_left_turns = std::nan("");
  const TimePlan none = plan_decision(p, s100, 100, 0, true, ctx, std::nan(""));
  CHECK(!none.moves_left_given);
  CHECK_EQ(none.moves_left_turns, 40.0);
  // ... the root's own prediction where the context gives none.
  const TimePlan root = plan_decision(p, s100, 100, 0, true, ctx, 60.0);
  CHECK(root.moves_left_given);
  CHECK_EQ(root.our_actions, 30.0);
  // Increments raise the plan.
  ctx.increment_seconds = 0;
  const TimePlan bare = plan_decision(p, s100, 100, 0, true, ctx, std::nan(""));
  ctx.increment_seconds = 2;
  const TimePlan inc = plan_decision(p, s100, 100, 0, true, ctx, std::nan(""));
  CHECK(inc.soft_evaluations > bare.soft_evaluations);
  CHECK(near(inc.usable_seconds - bare.usable_seconds, 2 * 19, 1e-9));
  // The floor on evaluations, and the hard bound's cap on the clock: with 0.5
  // s left and 20 actions' overhead (1 s) nothing is usable - the plan is the
  // floor's 8 evaluations (0.08 s), the hard bound 0.25 x (0.5 - 0.05).
  ctx.remaining_seconds = 0.5;
  ctx.increment_seconds = 0;
  ctx.moves_left_turns = 40;
  const TimePlan low = plan_decision(p, s100, 100, 0, true, ctx, std::nan(""));
  CHECK_EQ(low.usable_seconds, 0.0);
  CHECK(near(low.soft_evaluations, 8, 1e-12));
  CHECK(near(low.hard_seconds, 0.25 * 0.45, 1e-12));
  // ... and with less still, the plan is cut to the hard bound.
  ctx.remaining_seconds = 0.2;
  const TimePlan lower = plan_decision(p, s100, 100, 0, true, ctx, std::nan(""));
  CHECK(near(lower.hard_seconds, 0.25 * 0.15, 1e-12));
  CHECK(near(lower.soft_seconds, lower.hard_seconds, 1e-12));
}

TEST(timeman_6os_three_arms) {
  TimeParams integral;
  TimeParams flat_evals;
  flat_evals.allocator = Allocator::FlatEvals;
  TimeParams flat_clock;
  flat_clock.allocator = Allocator::FlatWallClock;
  TimeContext ctx;
  ctx.remaining_seconds = 300;
  ctx.moves_left_turns = 60;  // 30 of ours
  const double s = 0.01;
  const TimePlan i = plan_decision(integral, s, 100, 0, true, ctx, std::nan(""));
  const TimePlan e = plan_decision(flat_evals, s, 100, 0, true, ctx, std::nan(""));
  const TimePlan w = plan_decision(flat_clock, s, 100, 0, true, ctx, std::nan(""));
  // Flat: 1/A of the usable time - the same at the plan, executed by
  // evaluations (FlatEvals) or by the clock (FlatWallClock).
  CHECK(near(e.share, 1.0 / 30, 1e-12));
  CHECK(near(e.soft_seconds, e.usable_seconds / 30, 1e-9));
  CHECK(near(w.soft_seconds, e.soft_seconds, 1e-12));
  // The integral funds this move less: its later moves cost more.
  CHECK(i.share < e.share);
  CHECK(i.soft_evaluations < e.soft_evaluations);
  // The flat arms read no trajectory; the integral does.
  TimeParams slow_growth = integral;
  slow_growth.boards_per_action = {0.5, 0.5, 0.5};
  TimeParams flat_slow = flat_evals;
  flat_slow.boards_per_action = {0.5, 0.5, 0.5};
  CHECK(plan_decision(slow_growth, s, 100, 0, true, ctx, std::nan("")).share > i.share);
  CHECK_EQ(plan_decision(flat_slow, s, 100, 0, true, ctx, std::nan("")).soft_evaluations, e.soft_evaluations);
  // The arms execute differently: a FlatWallClock decision stops on time, a
  // FlatEvals one on work - here the clock runs at twice the plan's price.
  for (const Allocator arm : {Allocator::FlatEvals, Allocator::FlatWallClock}) {
    TimeParams p;
    p.allocator = arm;
    p.signals = false;
    TimeManager tm(p);
    tm.begin(plan_of(100));
    const TimeVerdict v = tm.check(snap(60, {60}), 1.2);  // 60 evaluations, 1.2 s: past 1 s of plan
    CHECK(v.stop == (arm == Allocator::FlatWallClock ? TimeStop::SoftBudget : TimeStop::None));
  }
}

TEST(timeman_the_hard_bound_tightens_with_n) {
  const TimeParams p;
  CHECK_EQ(hard_multiple(p, 10), 5.0);
  CHECK_EQ(hard_multiple(p, 50), 5.0);
  CHECK(near(hard_multiple(p, 225), 3.5, 1e-12));
  CHECK_EQ(hard_multiple(p, 400), 2.0);
  CHECK_EQ(hard_multiple(p, 600), 2.0);
  TimeContext ctx;
  ctx.remaining_seconds = 1e6;  // the clock's cap far away
  double last = 1e9;
  for (const int n : {20, 100, 250, 400, 600}) {
    const TimePlan plan = plan_decision(p, 0.01, n, 0, true, ctx, std::nan(""));
    const double ratio = plan.hard_seconds / plan.soft_seconds;
    CHECK(ratio <= last + 1e-12);
    last = ratio;
  }
  CHECK(near(last, 2.0, 1e-9));
}

// ---- the stop/extend table, case by case ----------------------------------------------

TEST(timeman_stops_smart_pruning_and_the_kld) {
  TimeParams p;
  {
    TimeManager tm(p);
    tm.begin(plan_of(100));
    // 50 of 100 evaluations spent; the lead (60 - 5 = 55) exceeds the 50 left.
    CHECK(tm.check(snap(50, {60, 5, 3}), std::nan("")).stop == TimeStop::SmartPruning);
  }
  {
    TimeManager tm(p);
    tm.begin(plan_of(100));
    CHECK(tm.check(snap(50, {40, 20, 3}), std::nan("")).stop == TimeStop::None);  // 20 < 50: bridgeable
    // Before stop_min_evaluations (32) no stop signal is read.
    TimeManager early(p);
    early.begin(plan_of(100));
    CHECK(early.check(snap(20, {20, 0}), std::nan("")).stop == TimeStop::None);
    // Playouts beyond the evaluations count toward what can still change.
    TimeManager cached(p);
    cached.begin(plan_of(100));
    RootSnapshot r = snap(50, {60, 5, 3});
    r.playouts = 100;  // two playouts an evaluation: 100 more could come
    CHECK(cached.check(r, std::nan("")).stop == TimeStop::None);
  }
  {
    // The KLD per visit over kld_interval (100) root visits: no change - stop.
    TimeManager tm(p);
    tm.begin(plan_of(1000), {50, 50}, 100);
    const TimeVerdict v = tm.check(snap(100, {100, 100}), std::nan(""));
    CHECK(v.stop == TimeStop::KldGain);
    CHECK_EQ(v.kld_gain, 0.0);
  }
  {
    // ... still moving: KL((.5, .5) || (.75, .25)) / 100 visits.
    TimeManager tm(p);
    tm.begin(plan_of(1000), {50, 50}, 100);
    const TimeVerdict v = tm.check(snap(100, {150, 50}), std::nan(""));
    CHECK(v.stop == TimeStop::None);
    CHECK(near(v.kld_gain, (0.5 * std::log(0.5 / 0.75) + 0.5 * std::log(0.5 / 0.25)) / 100, 1e-12));
  }
  {
    // A fresh root (no visits at the start) is not compared with nothing.
    TimeManager tm(p);
    tm.begin(plan_of(1000));
    const TimeVerdict v = tm.check(snap(100, {50, 50}), std::nan(""));
    CHECK(std::isnan(v.kld_gain));
    CHECK(v.stop == TimeStop::None);
  }
}

TEST(timeman_a_solved_root_plays_at_once_only_with_the_proof_channel) {
  TimeParams p;
  for (const bool channel : {false, true}) {
    TimeManager tm(p);
    tm.begin(plan_of(100));
    RootSnapshot r = snap(10, {5, 5});
    r.solved = true;
    r.proof_channel = channel;
    CHECK(tm.check(r, std::nan("")).stop == (channel ? TimeStop::Solved : TimeStop::None));
  }
  // A terminal root: the search stops itself (and any root needs a visit first).
  TimeManager tm(p);
  Search mated(game({"[8/8/8/8/8/5k2/6q1/7K:0:1:w]"}, BoardSize{8, 8}), quiet_params());
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  TimeContext ctx;
  ctx.remaining_seconds = 10;
  const TimedRun run = run_timed(mated, *uniform, tm, ctx);
  CHECK(run.verdict.stop == TimeStop::Search);
  CHECK(run.search_stop == StopReason::Finished);
  CHECK(mated.root_terminal());
}

TEST(timeman_extends_on_the_table_and_only_there) {
  TimeParams p;
  const auto at_target = [&](TimeManager& tm, RootSnapshot r) { return tm.check(r, std::nan("")); };
  // Overlap WITHOUT churn: the top two identical, the KLD unmeasured - the
  // soft bound stops.
  {
    TimeManager tm(p);
    tm.begin(plan_of(100));
    RootSnapshot r = snap(100, {60, 40});
    r.top_value[0] = peaked(25);
    r.top_value[1] = peaked(25);
    const TimeVerdict v = at_target(tm, r);
    CHECK(v.stop == TimeStop::SoftBudget);
    CHECK_EQ(v.extensions, 0u);
  }
  // Overlap WITH churn: extends by extend_overlap (0.5) of the plan.
  {
    TimeManager tm(p);
    tm.begin(plan_of(100), {50, 50, 0}, 100);
    RootSnapshot r = snap(100, {100, 90, 10});  // the KLD per visit 5.3e-4 >= 2e-4; a gap of 10
    r.top_value[0] = peaked(25);
    r.top_value[1] = peaked(25);
    const TimeVerdict v = at_target(tm, r);
    CHECK(v.stop == TimeStop::None);
    CHECK_EQ(v.extensions, kExtendOverlap);
    CHECK_EQ(v.target_evaluations, 150.0);
    // ... while distinct categoricals do not overlap.
    TimeManager apart(p);
    apart.begin(plan_of(100), {50, 50, 0}, 100);
    r.top_value[1] = peaked(40);
    CHECK(at_target(apart, r).stop == TimeStop::SoftBudget);
  }
  // Volatility: the root's uncertainty signal at the threshold (0.2).
  {
    TimeManager tm(p);
    tm.begin(plan_of(100));
    RootSnapshot r = snap(100, {60, 40});
    r.root_evaluated = true;
    r.root_variance = 0.25;
    const TimeVerdict v = at_target(tm, r);
    CHECK_EQ(v.extensions, kExtendVolatility);
    CHECK_EQ(v.target_evaluations, 150.0);
  }
  // Disagreement - a calm net, an alarmed prover (the hook, with the proof
  // channel: test-only) - extends more than either source alone.
  {
    TimeParams hook = p;
    hook.proof_volatility = true;
    const auto run = [&](double variance, double gap) {
      TimeManager tm(hook);
      tm.begin(plan_of(100));
      RootSnapshot r = snap(100, {60, 40});
      r.root_evaluated = true;
      r.root_variance = variance;
      r.proof_channel = true;
      r.proof_gap = gap;
      return at_target(tm, r);
    };
    const TimeVerdict disagreement = run(0.01, 1.5);
    const TimeVerdict net_alone = run(0.3, 0.0);
    const TimeVerdict both = run(0.3, 1.5);
    CHECK_EQ(disagreement.extensions, kExtendDisagreement);
    CHECK_EQ(disagreement.target_evaluations, 200.0);
    CHECK_EQ(net_alone.target_evaluations, 150.0);
    CHECK_EQ(both.extensions, kExtendVolatility);
    CHECK(disagreement.target_evaluations > net_alone.target_evaluations);
    CHECK(disagreement.target_evaluations > both.target_evaluations);
    // Off (the default), the prover is not read.
    TimeManager off(p);
    off.begin(plan_of(100));
    RootSnapshot r = snap(100, {60, 40});
    r.proof_channel = true;
    r.proof_gap = 2.0;
    CHECK(at_target(off, r).stop == TimeStop::SoftBudget);
  }
  // A falling eval extends; within the margin, or rising, it does not.
  for (const double now : {0.3, 0.45, 0.9}) {
    TimeManager tm(p);
    tm.end(0.5);  // the previous decision's root value
    tm.begin(plan_of(100));
    RootSnapshot r = snap(100, {60, 40});
    r.root_q = now;
    const TimeVerdict v = at_target(tm, r);
    CHECK_EQ(v.extensions, now == 0.3 ? kExtendFall : 0u);
  }
  // Instability: the best child changed within the last quarter of the visits.
  for (const std::uint32_t since : {90u, 50u}) {
    TimeManager tm(p);
    tm.begin(plan_of(100));
    RootSnapshot r = snap(100, {60, 40});
    r.best_since = since;
    CHECK_EQ(at_target(tm, r).extensions, since == 90u ? kExtendInstability : 0u);
  }
  // Signals off: nothing extends, nothing stops early.
  {
    TimeParams off = p;
    off.signals = false;
    TimeManager tm(off);
    tm.begin(plan_of(100));
    RootSnapshot early = snap(50, {60, 5});  // smart pruning's case
    CHECK(tm.check(early, std::nan("")).stop == TimeStop::None);
    RootSnapshot r = snap(100, {60, 40});
    r.root_evaluated = true;
    r.root_variance = 0.9;
    r.best_since = 99;
    const TimeVerdict v = tm.check(r, std::nan(""));
    CHECK(v.stop == TimeStop::SoftBudget);
    CHECK_EQ(v.extensions, 0u);
  }
}

TEST(timeman_the_spiral_guard_holds_at_high_n) {
  // 6o: "verify the extension-spiral guard at high N". At 500 boards, every
  // extension signal active at every check, the work advancing a batch at a
  // time on the model clock: all the extensions together stay within
  // max_extension of the plan, and the hard bound is never crossed. (A second
  // take of one kind never comes up here - the cap or the hard bound ends the
  // decision first; once-per-kind is timeman_each_extension_kind_is_taken_once_
  // a_decision's.)
  for (const double max_extension : {1.0, 0.25, 5.0}) {
    TimeParams p;
    p.proof_volatility = true;
    p.max_extension = max_extension;
    p.smart_pruning_factor = 1e9;  // no stop signal: the extensions alone
    p.kld_threshold = 0;
    p.churn_kld_threshold = 0;
    TimeContext ctx;
    ctx.remaining_seconds = 600;
    const TimePlan plan = plan_decision(p, 0.01, 500, 0, true, ctx, std::nan(""));
    TimeManager tm(p);
    tm.end(0.9);
    tm.begin(plan, {1, 1}, 2);
    std::uint64_t work = 0;
    TimeVerdict v;
    std::uint32_t root = 2;
    for (int step = 0; step < 100000; ++step) {
      work += 8;
      root += 8;
      RootSnapshot r = snap(work, {root / 2 + (step % 2 == 0 ? 3u : 0u), root / 2});
      r.root_visits = root;
      r.root_evaluated = true;
      r.root_variance = 0.9;
      r.proof_channel = true;
      r.proof_gap = 2.0;
      r.root_q = -0.5;
      r.best_since = root - 1;
      r.top_value[0] = peaked(25);
      r.top_value[1] = peaked(25);
      v = tm.check(r, std::nan(""));
      CHECK(v.target_evaluations <= plan.soft_evaluations * (1 + max_extension) + 1e-9);
      CHECK(v.elapsed_seconds < plan.hard_seconds);
      if (v.stop != TimeStop::None) break;
    }
    CHECK(v.stop == TimeStop::SoftBudget || v.stop == TimeStop::HardBound);
    CHECK(v.elapsed_seconds < plan.hard_seconds);
  }
}

// ---- the hard bound ----------------------------------------------------------------

TEST(timeman_the_hard_bound_holds_on_a_virtual_clock) {
  // The evaluator costs three times the model's prior, and more at its deeper
  // leaves (more boards) than the root's shape predicts: the first batch runs
  // whatever the clock says (a move needs a visit), every later check predicts
  // from the measured scale - so a decision never ends past its hard bound by
  // more than one batch.
  for (const char* name : {"Standard", "Standard - Two Timelines"}) {
    TimeParams p;
    TimeManager tm(p);
    ManualClock clock;
    ChargingEvaluator ev(clock, 3 * p.cost_seconds_per_evaluation);
    Search search(variant(name), quiet_params());
    for (int decision = 0; decision < 4; ++decision) {
      TimeContext ctx;
      ctx.remaining_seconds = 3.0;
      const double t0 = clock.t;
      ev.largest_batch = 0;
      const TimedRun run = run_timed(search, ev, tm, ctx, &clock);
      CHECK(clock.t - t0 <= run.plan.hard_seconds + ev.largest_batch + 1e-12);
      CHECK(near(run.seconds, clock.t - t0, 1e-12));
      CHECK(run.evaluations > 0);
      CHECK(run.verdict.stop != TimeStop::None);
      // §15.4: the choice after any stop is among the eligible children.
      const std::vector<RootChild> children = search.root_children();
      std::vector<FinalCandidate> cands;
      for (const RootChild& c : children) {
        FinalCandidate f;
        f.visits = c.visits;
        f.q = c.q;
        f.variance = c.value_variance;
        f.lo = c.lo;
        f.hi = c.hi;
        cands.push_back(f);
      }
      const Choice choice = search.choose();
      CHECK(final_eligible(cands, choice.index, choice.reason, search.params()));
      search.check_invariants();
    }
    // The scale was measured: at least the evaluator's price at the root's shape.
    CHECK(tm.scale() >= 3 * p.cost_seconds_per_evaluation * (1 - 1e-9));
  }
}

TEST(timeman_the_hard_bound_holds_on_the_wall_clock_under_load) {
  // A correctness bound, generous: four threads spinning beside the search, an
  // evaluator taking 2 ms a batch; each decision ends within its hard bound
  // plus half a second.
  std::atomic<bool> stop{false};
  std::vector<std::thread> load;
  for (int i = 0; i < 4; ++i) {
    load.emplace_back([&stop] {
      volatile std::uint64_t x = 0;
      while (!stop.load(std::memory_order_relaxed)) x = x + 1;
    });
  }
  struct Join {
    std::atomic<bool>& stop;
    std::vector<std::thread>& threads;
    ~Join() {
      stop = true;
      for (std::thread& t : threads) t.join();
    }
  } join{stop, load};
  SleepyEvaluator ev(std::chrono::microseconds(2000));
  TimeManager tm;
  WallClock clock;
  Search search(variant("Standard - Two Timelines"), quiet_params());
  for (int decision = 0; decision < 3; ++decision) {
    TimeContext ctx;
    ctx.remaining_seconds = 4.0;
    const auto t0 = std::chrono::steady_clock::now();
    const TimedRun run = run_timed(search, ev, tm, ctx, &clock);
    const double spent = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(spent <= run.plan.hard_seconds + 0.5);
    CHECK(run.plan.hard_seconds <= 0.25 * (4.0 - 0.05) + 1e-9);
    CHECK(!search.deadline().has_value());  // the run's deadline is cleared after it
    search.advance(search.choose().index);
  }
}

TEST(timeman_the_deadline_stops_the_rules_engine_inside_a_batch) {
  // A 400-board position (random pseudo-legal play, synthetic). Past the
  // deadline the calls a batch makes stop at once and answer Unknown: E4's
  // detector under the budget the search gives it (Search::set_deadline's
  // time left), and the legality classification - exact, no node budget, so
  // only the clock stops it - inside the search's expansion and alone.
  synthetic::Spec spec;
  spec.target_boards = 400;
  spec.target_active = 10;
  spec.seed = 5;
  const synthetic::Game g = synthetic::play_until_reached(spec);
  REQUIRE(g.stop == synthetic::Stop::Target);
  REQUIRE(g.state.multiverse().board_count() >= 400);
  SearchBudget past;
  past.max_time = std::chrono::nanoseconds{0};  // what the search gives a check past its deadline
  const MateResult mate = detect_mate(g.state, Detector::Hypercuboid, past);
  CHECK(mate.verdict == Outcome::Unknown || mate.verdict == Outcome::Ongoing);  // never a terminal unproven
  SearchParams sp = quiet_params();
  sp.mate_nodes = 50'000;  // the root's own check, at construction, is bounded by nodes
  sp.legality = LegalityOptions::exact();
  Search search(g.state, sp);
  search.set_deadline(std::chrono::steady_clock::now() - std::chrono::seconds(1));
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  const auto t0 = std::chrono::steady_clock::now();
  Limits one;
  one.playouts = 1;
  search.run(*uniform, one);
  const double spent = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  CHECK(spent < 30.0);  // generous: the exact classification is otherwise unbounded
  CHECK(!search.root_terminal());
  CHECK(search.stats().unknown_edges > 0);
  CHECK(search.stats().playouts >= 1);
  // The legality classification alone, past its deadline: nothing beyond the
  // cheap tiers is settled.
  MoveList moves;
  generate_moves(g.state, moves);
  std::vector<MoveLegality> out;
  LegalityOptions late = LegalityOptions::exact();
  late.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  LegalityStats stats;
  classify_moves(g.state, std::span<const Move>(moves.data(), moves.size()), out, late, &stats);
  CHECK_EQ(stats.search_nodes, std::uint64_t{0});
  CHECK(std::count(out.begin(), out.end(), MoveLegality::Unknown) > 0);
}

// ---- determinism, the hook's refusal, match play ----------------------------------

TEST(timeman_the_model_clock_is_bit_reproducible) {
  // H15: no clock - the cost model's - so a timed search is node-limited.
  for (const bool synchronous : {false, true}) {
    std::vector<std::uint64_t> digests;
    std::vector<double> soft;
    for (int run = 0; run < 2; ++run) {
      SearchParams sp = quiet_params();
      sp.synchronous = synchronous;
      Search search(variant("Standard - Two Timelines"), sp);
      const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 7);
      TimeManager tm;
      TimeContext ctx;
      ctx.remaining_seconds = 2.0;
      ctx.increment_seconds = 0.1;
      std::uint64_t digest = 0;
      for (int decision = 0; decision < 3; ++decision) {
        const TimedRun r = run_timed(search, *random, tm, ctx);
        CHECK(r.evaluations > 0);
        soft.push_back(r.plan.soft_evaluations);
        digest ^= search.digest() + std::uint64_t{0x9E3779B97F4A7C15} * static_cast<std::uint64_t>(decision + 1) +
                  static_cast<std::uint64_t>(r.verdict.stop) + r.evaluations;
        ctx.remaining_seconds -= r.seconds;
        search.advance(search.choose().index);
      }
      digests.push_back(digest);
    }
    CHECK_EQ(digests[0], digests[1]);
    CHECK(std::equal(soft.begin(), soft.begin() + 3, soft.begin() + 3));
  }
}

TEST(timeman_the_proof_hook_is_refused_without_the_proof_channel) {
  TimeParams p;
  p.proof_volatility = true;
  TimeManager tm(p);
  Search search(variant("Standard"), quiet_params());
  TimeContext ctx;
  ctx.remaining_seconds = 10;
  CHECK_THROWS_WITH(tm.begin(search, ctx), std::invalid_argument, "issue 180");
  // With the channel on (test-only), it is read.
  SearchParams on = quiet_params();
  on.proof_channel = true;
  Search channel(variant("Standard"), on);
  CHECK_NOTHROW(tm.begin(channel, ctx));
  // Out-of-range contexts and parameters are refused.
  TimeManager plain;
  ctx.remaining_seconds = 0;
  CHECK_THROWS_AS(plain.begin(search, ctx), std::invalid_argument);
  TimeParams bad;
  bad.hard_fraction = 0;
  CHECK_THROWS_AS(TimeManager{bad}, std::invalid_argument);
  CHECK_THROWS_AS(plain.check(search, 0.0), std::logic_error);  // no decision begun
}

TEST(timeman_match_play_on_the_clock_end_to_end) {
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
    for (const Allocator arm : {Allocator::Integral, Allocator::FlatEvals, Allocator::FlatWallClock}) {
      PlayParams p;
      p.search = quiet_params();
      p.search.split = split;
      p.match_play = true;
      p.turn_cap.turns = 4;
      p.turn_cap.boards = 60;  // branching can hold the present back
      p.clock.enabled = true;
      p.clock.model_clock = true;  // deterministic: the cost model's prior prices each evaluation ...
      p.clock.time.cost_seconds_per_evaluation = 0.1;  // ... here a slow one: fewer evaluations a decision
      p.clock.base_seconds = 10;
      p.clock.increment_seconds = 0.1;
      p.clock.time.allocator = arm;
      const GameState start = variant("Standard - Two Timelines");
      const GameRecord g = play_game(start, *uniform, p);
      CHECK(g.end == GameEnd::TurnCap || g.end == GameEnd::Checkmate || g.end == GameEnd::Stalemate);
      CHECK_EQ(g.timed_decisions, g.decisions);
      CHECK_EQ(g.hard_overruns, std::uint64_t{0});
      CHECK(g.clock_remaining[0] > 0);
      CHECK(g.clock_remaining[1] > 0);
      // Every action legal, and the record replays to the final state.
      GameState s = g.start;
      MoveList scratch;
      for (const std::vector<Move>& action : g.actions) {
        for (const Move& m : action) {
          REQUIRE(is_pseudo_legal(s, m, scratch));
          s.apply(m);
        }
        REQUIRE(s.submission_status() == SubmitStatus::Safe);
        s.submit();
      }
      CHECK(s == g.final_state);
      const std::string text = tools::play_record(g, "Standard - Two Timelines");
      PgnReplay replay(read_pgn(text), PgnOptions{}, RuleOptions{});
      CHECK_NOTHROW(replay.play_all());
      CHECK(replay.state() == g.final_state);
      // Deterministic on the model clock.
      if (arm == Allocator::Integral) {
        const GameRecord again = play_game(start, *uniform, p);
        CHECK(again.actions == g.actions);
        CHECK_EQ(again.clock_remaining[0], g.clock_remaining[0]);
        CHECK_EQ(again.clock_remaining[1], g.clock_remaining[1]);
      }
    }
  }
  // A clock too short for one batch: White's flag falls on its first decision.
  PlayParams p;
  p.search = quiet_params();
  p.match_play = true;
  p.clock.enabled = true;
  p.clock.model_clock = true;
  p.clock.base_seconds = 1e-6;
  const GameRecord flag = play_game(variant("Standard"), *uniform, p);
  CHECK(flag.end == GameEnd::Flag);
  CHECK_EQ(flag.winner, 1);
  CHECK(flag.actions.empty());
  CHECK_EQ(flag.result(), std::string("*"));
  CHECK_EQ(flag.white_score(), 0.0);
  // Self-play takes no clock (its budgets are PCR's).
  PlayParams self = p;
  self.match_play = false;
  CHECK_THROWS_AS(play_game(variant("Standard"), *uniform, self), std::invalid_argument);
}

// ---- E8: difficulty-weighted allocation, behind its flag (a prototype) -------------------------

namespace {

// An effort distribution with 0.9 of its uncensored mass on bin `k`, the rest spread evenly
// over bins 0-14 (none censored) - plus `censored` on the censored bin.
std::array<float, kEffortBins> effort_at(int k, float censored = 0.0f) {
  std::array<float, kEffortBins> e{};
  const float rest = (1.0f - censored) * 0.1f / 15.0f;
  for (int b = 0; b < kEffortBins - 1; ++b) e[static_cast<std::size_t>(b)] = rest;
  e[static_cast<std::size_t>(k)] += (1.0f - censored) * 0.9f;
  e[kEffortBins - 1] = censored;
  return e;
}

bool same_plan(const TimePlan& a, const TimePlan& b) {
  return a.boards == b.boards && a.moves_left_turns == b.moves_left_turns && a.moves_left_given == b.moves_left_given &&
         a.our_actions == b.our_actions && a.decisions_left == b.decisions_left &&
         a.seconds_per_evaluation == b.seconds_per_evaluation && a.usable_seconds == b.usable_seconds &&
         a.share == b.share && a.action_evaluations == b.action_evaluations &&
         a.soft_evaluations == b.soft_evaluations && a.soft_seconds == b.soft_seconds &&
         a.hard_seconds == b.hard_seconds && a.e7_soft_evaluations == b.e7_soft_evaluations &&
         a.difficulty_applied == b.difficulty_applied && a.effort_bin == b.effort_bin &&
         a.wont_resolve == b.wont_resolve && a.difficulty_factor == b.difficulty_factor;
}

}  // namespace

TEST(timeman_e8_off_is_e7_bit_for_bit) {
  // With the flag off, an effort distribution given or not, every plan is E7's;
  // and a timed run's tree is the run's without the distribution (H15).
  const TimeParams off;
  CHECK(!off.difficulty_weighting);
  for (const double remaining : {0.3, 5.0, 120.0}) {
    for (const int boards : {1, 40, 150, 500}) {
      for (const int bin : {0, 8, 14}) {
        TimeContext with;
        with.remaining_seconds = remaining;
        with.increment_seconds = 0.1;
        with.effort = effort_at(bin, 0.3f);
        TimeContext without = with;
        without.effort.reset();
        TimeManager a(off);
        TimeManager b(off);
        const Search search(variant("Standard"), quiet_params());
        const TimePlan pa = a.begin(search, with);
        const TimePlan pb = b.begin(search, without);
        CHECK(same_plan(pa, pb));
        CHECK(!pa.difficulty_applied);
        CHECK(same_plan(plan_decision(off, 0.01, boards, 1, false, with, std::nan("")),
                        plan_decision(off, 0.01, boards, 1, false, without, std::nan(""))));
      }
    }
  }
  std::vector<std::uint64_t> digests;
  for (const bool given : {false, true}) {
    Search search(variant("Standard - Two Timelines"), quiet_params());
    const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 4);
    TimeManager tm;
    TimeContext ctx;
    ctx.remaining_seconds = 2.0;
    if (given) ctx.effort = effort_at(14, 0.9f);  // as hard and unresolved as it gets: read by nothing
    for (int decision = 0; decision < 2; ++decision) {
      run_timed(search, *random, tm, ctx);
      search.advance(search.choose().index);
    }
    digests.push_back(search.digest());
  }
  CHECK_EQ(digests[0], digests[1]);
}

TEST(timeman_e8_on_moves_the_plan_as_its_rule_says) {
  TimeParams p;
  p.difficulty_weighting = true;
  TimeContext ctx;
  ctx.remaining_seconds = 300;
  ctx.moves_left_turns = 40;
  const TimePlan e7 = plan_decision(p, 0.01, 100, 0, true, ctx, std::nan(""));
  // The reference: the prior's 256 visits, bin 8. A harder prediction (bin 10)
  // funds 2^2 = 4x E7's plan, an easier one (bin 6) a quarter - each clamped to
  // [0.25, 4]. (The floor and the hard bound's cap bind on smaller plans than
  // this one - 903.9 evaluations, a hard bound of 4132: they are timeman_e8_
  // never_plans_below_its_floor_nor_past_e7s_hard_bound's.)
  const TimePlan hard = apply_difficulty(p, e7, effort_at(10), 8.0);
  const TimePlan easy = apply_difficulty(p, e7, effort_at(6), 8.0);
  const TimePlan same = apply_difficulty(p, e7, effort_at(8), 8.0);
  CHECK_EQ(hard.effort_bin, 10);
  CHECK_EQ(easy.effort_bin, 6);
  CHECK(hard.difficulty_applied && !hard.wont_resolve);
  CHECK(near(same.soft_evaluations, e7.soft_evaluations, 1e-9));
  CHECK(near(hard.soft_evaluations, std::min(4 * e7.soft_evaluations, e7.hard_seconds / e7.seconds_per_evaluation), 1e-9));
  CHECK(near(easy.soft_evaluations, std::max(0.25 * e7.soft_evaluations, p.difficulty_floor_evaluations), 1e-9));
  CHECK(hard.soft_evaluations > e7.soft_evaluations);
  CHECK(easy.soft_evaluations < e7.soft_evaluations);
  // Guard 3: the hard bound is E7's, whatever the prediction.
  CHECK_EQ(hard.hard_seconds, e7.hard_seconds);
  CHECK_EQ(easy.hard_seconds, e7.hard_seconds);
  // Clamped to the factor caps. (The floor and hard-bound checks hold here without
  // those clamps too - x0.25 is 226 evaluations, x4 is 3615: see above.)
  for (int bin = 0; bin < kEffortBins - 1; ++bin) {
    for (const double reference : {0.0, 8.0, 14.0}) {
      const TimePlan w = apply_difficulty(p, e7, effort_at(bin), reference);
      CHECK(w.soft_evaluations >= p.difficulty_floor_evaluations - 1e-9);
      CHECK(w.soft_evaluations <= e7.hard_seconds / e7.seconds_per_evaluation + 1e-9);
      CHECK(w.difficulty_factor >= p.difficulty_min_factor && w.difficulty_factor <= p.difficulty_max_factor);
    }
  }
  // The quantile is an UPPER one: the same distribution funded at 0.8 more than at 0.5.
  std::array<float, kEffortBins> spread{};
  for (int b = 4; b < 12; ++b) spread[static_cast<std::size_t>(b)] = 0.125f;
  TimeParams median = p;
  median.effort_quantile = 0.5;
  CHECK(effort_quantile_bin(spread, 0.8) > effort_quantile_bin(spread, 0.5));
  CHECK(apply_difficulty(p, e7, spread, 8.0).soft_evaluations > apply_difficulty(median, e7, spread, 8.0).soft_evaluations);
  // The reference is this game's running mean of quantile bins, seeded by the prior.
  TimeManager tm(p);
  CHECK_EQ(tm.reference_bin(), 8.0);
  const Search search(variant("Standard"), quiet_params());
  ctx.effort = effort_at(12);
  tm.begin(search, ctx);
  CHECK(near(tm.reference_bin(), (4 * 8.0 + 12) / 5, 1e-12));
  tm.new_game();
  CHECK_EQ(tm.reference_bin(), 8.0);
  // Malformed distributions are refused.
  std::array<float, kEffortBins> bad{};
  CHECK_THROWS_AS(apply_difficulty(p, e7, bad, 8.0), std::invalid_argument);
  bad[3] = -1.0f;
  CHECK_THROWS_AS(apply_difficulty(p, e7, bad, 8.0), std::invalid_argument);
}

TEST(timeman_e8_wont_resolve_under_both_readings) {
  TimeParams p;
  p.difficulty_weighting = true;
  TimeContext ctx;
  ctx.remaining_seconds = 300;
  ctx.moves_left_turns = 40;
  const TimePlan e7 = plan_decision(p, 0.01, 100, 0, true, ctx, std::nan(""));
  const TimePlan a = apply_difficulty(p, e7, effort_at(3, 0.6f), 8.0);  // censored mass 0.6 >= 0.5
  CHECK(a.wont_resolve);
  CHECK(near(a.censored_mass, 0.6, 1e-6));
  CHECK(near(a.soft_evaluations, e7.hard_seconds / e7.seconds_per_evaluation, 1e-9));  // (A): the hard bound
  TimeParams b = p;
  b.wont_resolve_policy = WontResolve::CapAtPlan;
  const TimePlan cap = apply_difficulty(b, e7, effort_at(3, 0.6f), 8.0);
  CHECK(cap.wont_resolve);
  CHECK_EQ(cap.soft_evaluations, e7.soft_evaluations);  // (B): E7's plan, unweighted
  CHECK(!apply_difficulty(p, e7, effort_at(3, 0.4f), 8.0).wont_resolve);
  // Overlap with churn - the convergence-chasing extension - is not taken
  // under won't-resolve, and is otherwise; volatility still is.
  for (const bool wont : {false, true}) {
    TimeManager tm(p);
    TimePlan plan = plan_of(100);
    plan.wont_resolve = wont;
    tm.begin(plan, {50, 50, 0}, 100);
    RootSnapshot r = snap(100, {100, 90, 10});
    r.top_value[0] = peaked(25);
    r.top_value[1] = peaked(25);
    const TimeVerdict v = tm.check(r, std::nan(""));
    CHECK_EQ(v.extensions, wont ? 0u : kExtendOverlap);
    TimeManager vol(p);
    vol.begin(plan, {50, 50, 0}, 100);
    r.root_evaluated = true;
    r.root_variance = 0.5;
    CHECK((vol.check(r, std::nan("")).extensions & kExtendVolatility) != 0);
  }
}

TEST(timeman_e8_guards_an_easy_call_still_extends_and_the_bounds_stay) {
  // Guard 1: a wrong "easy" call still extends on instability and overlap.
  TimeParams p;
  p.difficulty_weighting = true;
  TimeContext ctx;
  ctx.remaining_seconds = 300;
  ctx.moves_left_turns = 40;
  const TimePlan e7 = plan_decision(p, 0.01, 100, 0, true, ctx, std::nan(""));
  const TimePlan easy = apply_difficulty(p, e7, effort_at(0), 8.0);
  CHECK(easy.soft_evaluations < e7.soft_evaluations);
  TimeManager tm(p);
  tm.begin(easy, {50, 50, 0}, 100);
  const auto target = static_cast<std::uint64_t>(std::ceil(easy.soft_evaluations));
  RootSnapshot r = snap(target, {100, 90, 10});
  r.best_since = 195;  // changed in the last visits
  r.top_value[0] = peaked(25);
  r.top_value[1] = peaked(25);
  const TimeVerdict v = tm.check(r, std::nan(""));
  CHECK((v.extensions & kExtendInstability) != 0);
  CHECK((v.extensions & kExtendOverlap) != 0);
  CHECK(v.stop == TimeStop::None);
  // Guard 3: the hard bound is the same flag on and off. (The ladder's budgets are
  // out of E8's reach by construction: TimeManager::begin takes the search const.)
  TimeParams off = p;
  off.difficulty_weighting = false;
  for (const int bin : {0, 7, 14}) {
    SearchParams sp = quiet_params();
    sp.hunt = true;
    sp.settling = true;
    sp.synchronous = true;
    Search on_search(variant("Standard"), sp);
    Search off_search(variant("Standard"), sp);
    TimeManager on_tm(p);
    TimeManager off_tm(off);
    ctx.effort = effort_at(bin, bin == 14 ? 0.7f : 0.0f);
    const TimePlan a = on_tm.begin(on_search, ctx);
    const TimePlan b = off_tm.begin(off_search, ctx);
    CHECK_EQ(a.hard_seconds, b.hard_seconds);
  }
}

TEST(timeman_e8_on_the_model_clock_is_reproducible) {
  std::vector<std::uint64_t> digests;
  std::vector<double> plans;
  for (int run = 0; run < 2; ++run) {
    TimeParams p;
    p.difficulty_weighting = true;
    Search search(variant("Standard - Two Timelines"), quiet_params());
    const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 9);
    TimeManager tm(p);
    TimeContext ctx;
    ctx.remaining_seconds = 3.0;
    for (int decision = 0; decision < 3; ++decision) {
      ctx.effort = effort_at(4 + 3 * decision);
      const TimedRun r = run_timed(search, *random, tm, ctx);
      plans.push_back(r.plan.soft_evaluations);
      ctx.remaining_seconds -= r.seconds;
      search.advance(search.choose().index);
    }
    digests.push_back(search.digest());
  }
  CHECK_EQ(digests[0], digests[1]);
  CHECK(std::equal(plans.begin(), plans.begin() + 3, plans.begin() + 3));
  CHECK(plans[2] != plans[0]);  // the predictions moved the plans
  // Guard 2 binds here: the first two decisions' easy calls (5.05 and 24.16 evaluations
  // weighted) are lifted to the floor's 32.
  CHECK_EQ(plans[0], 32.0);
  CHECK_EQ(plans[1], 32.0);
  CHECK(plans[2] > 32.0);
}

// ---- review round 1 (E11/E8): the rules pinned one by one ----------------------------------
// Each test below is a probe of the review of 05afde5..644801c (findings M5, M6, L7, L8), named
// by the mutant it kills; every one passed on 644801c and fails on its mutant.

TEST(timeman_smart_pruning_counts_only_what_the_hard_bound_leaves) {
  // T18: 50 of a 1000-evaluation plan spent (0.5 s at 0.01 s each), the hard bound 0.9 s - 40
  // more evaluations at most, so a lead of 50 is unbridgeable (the plan alone would leave 950).
  TimeManager tm;
  tm.begin(plan_of(1000, 0.9));
  CHECK(tm.check(snap(50, {70, 20}), std::nan("")).stop == TimeStop::SmartPruning);
  TimeManager wide;
  wide.begin(plan_of(1000));
  CHECK(wide.check(snap(50, {70, 20}), std::nan("")).stop == TimeStop::None);
}

TEST(timeman_a_single_child_stops_at_once) {
  // T20: one child - nothing to decide - from stop_min_evaluations on.
  CHECK(smart_pruning_stop(std::vector<std::uint32_t>{5}, 1e6, 1.0));
  TimeManager tm;
  tm.begin(plan_of(1000));
  CHECK(tm.check(snap(40, {40}), std::nan("")).stop == TimeStop::SmartPruning);
}

TEST(timeman_overlap_without_churn_does_not_extend_on_a_measured_kld) {
  // T22: the KLD MEASURED (not unmeasured, as the extension test's first case is), below the
  // churn threshold (2e-4 per visit) - 0, and 7.25e-5: KL((.5, .5) || (.56, .44)) over 100
  // visits - with identical categoricals: no extension, the soft bound stops.
  for (const std::vector<std::uint32_t>& now : {std::vector<std::uint32_t>{100, 100}, std::vector<std::uint32_t>{112, 88}}) {
    TimeManager tm;
    tm.begin(plan_of(100), {50, 50}, 100);
    RootSnapshot r = snap(100, now);
    r.top_value[0] = peaked(25);
    r.top_value[1] = peaked(25);
    const TimeVerdict v = tm.check(r, std::nan(""));
    CHECK(std::isfinite(v.kld_gain));
    CHECK(v.kld_gain < 2e-4);
    CHECK_EQ(v.extensions, 0u);
    CHECK(v.stop == TimeStop::SoftBudget);
  }
  // ... and at a churn above the threshold the same root extends.
  TimeManager tm;
  tm.begin(plan_of(100), {50, 50}, 100);
  RootSnapshot r = snap(100, {150, 50});
  r.top_value[0] = peaked(25);
  r.top_value[1] = peaked(25);
  CHECK_EQ(tm.check(r, std::nan("")).extensions, kExtendOverlap);
}

TEST(timeman_each_extension_kind_is_taken_once_a_decision) {
  // T23: volatility at the soft bound extends it once (100 -> 150); still volatile at 150, it is
  // not taken again - the soft bound stops.
  TimeManager tm;
  tm.begin(plan_of(100));
  RootSnapshot r = snap(100, {60, 40});
  r.root_evaluated = true;
  r.root_variance = 0.5;
  const TimeVerdict first = tm.check(r, std::nan(""));
  CHECK_EQ(first.target_evaluations, 150.0);
  CHECK(first.stop == TimeStop::None);
  RootSnapshot later = snap(150, {90, 60});
  later.root_evaluated = true;
  later.root_variance = 0.5;
  const TimeVerdict v = tm.check(later, std::nan(""));
  CHECK_EQ(v.target_evaluations, 150.0);
  CHECK_EQ(v.extensions, kExtendVolatility);
  CHECK(v.stop == TimeStop::SoftBudget);
}

TEST(timeman_flat_wall_clock_extends_by_time) {
  // T25: the FlatWallClock arm executes by the clock - its soft second spent, a volatile root
  // extends it to 1.5 s (the extension scales the soft time, not only the work).
  TimeParams p;
  p.allocator = Allocator::FlatWallClock;
  TimeManager tm(p);
  tm.begin(plan_of(100));  // 1 s of soft time
  RootSnapshot r = snap(60, {60, 40});
  r.root_evaluated = true;
  r.root_variance = 0.25;
  const TimeVerdict v = tm.check(r, 1.0);
  CHECK_EQ(v.extensions, kExtendVolatility);
  CHECK(v.stop == TimeStop::None);
  CHECK(tm.check(snap(70, {55, 55}), 1.4).stop == TimeStop::None);  // level: no lead to prune on
  CHECK(tm.check(snap(80, {60, 60}), 1.5).stop == TimeStop::SoftBudget);
}

TEST(timeman_a_new_game_forgets_the_falling_eval_reference) {
  // T26: the previous game's last root value is not this game's reference.
  TimeManager tm;
  tm.end(0.5);
  tm.new_game();
  tm.begin(plan_of(100));
  RootSnapshot r = snap(100, {60, 40});
  r.root_q = 0.3;  // 0.2 below the forgotten reference: a fall, were it remembered
  CHECK_EQ(tm.check(r, std::nan("")).extensions, 0u);
}

TEST(timeman_the_model_clock_prices_the_work_not_the_evaluations) {
  // T28: a mate-in-one root - most playouts end at the terminal child without an evaluation, so
  // the work is playouts / 4 (playouts_per_evaluation_cap), and run_timed's seconds on the model
  // clock are that work's price.
  Search search(game({"[3k4/8/3K4/8/8/8/8/7Q:0:1:w]"}, BoardSize{8, 8}), quiet_params());
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  TimeManager tm;
  TimeContext ctx;
  ctx.remaining_seconds = 30;
  const TimedRun r = run_timed(search, *uniform, tm, ctx);
  CHECK(static_cast<double>(r.playouts) / 4.0 > static_cast<double>(r.evaluations));  // the premise
  const double work = std::max(static_cast<double>(r.evaluations), static_cast<double>(r.playouts) / 4.0);
  CHECK(near(r.seconds, work * tm.seconds_per_evaluation(r.plan.boards), 1e-12));
}

TEST(timeman_the_snapshots_root_value_is_the_backed_up_mean) {
  // T40: the falling-eval signal reads the root's mean backed-up value, not its sum.
  Search search(variant("Standard"), quiet_params());
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 5);
  search.run(*random, Limits{64});
  const NodeView root = search.node(search.root());
  CHECK(root.weight > 1);
  CHECK(near(snapshot(search, 0, 0, TimeParams{}).root_q, root.value / root.weight, 1e-12));
}

TEST(timeman_a_root_childs_categorical_is_read_in_the_root_players_view) {
  // M05 (Search::root_child_value): flat form, every root child the opponent's turn start. After
  // Qxd8+ Black is a queen down: the material evaluator's categorical (the mover's view) is low
  // for Black, so in White's view - the root player's - its mean is high.
  SearchParams p = quiet_params();
  p.split = TurnSplit::Flat;
  Search search(game({"[3q3k/8/8/8/8/8/8/3Q3K:0:1:w]"}, BoardSize{8, 8}), p);
  const std::unique_ptr<Evaluator> material = make_evaluator(BuiltinEvaluator::Material);
  search.run(*material, Limits{64});
  const std::vector<RootChild> children = search.root_children();
  bool found = false;
  for (std::size_t i = 0; i < children.size(); ++i) {
    const RootChild& c = children[i];
    if (c.moves.size() != 1 || c.moves[0].captured.empty() || c.child == kNoNode) continue;
    const std::optional<std::array<float, kValueBins>> v = search.root_child_value(i);
    if (!v) continue;
    double mean = 0;
    for (int k = 0; k < kValueBins; ++k) mean += (*v)[static_cast<std::size_t>(k)] * value_bin_centre(k);
    CHECK(mean > 0.5);
    found = true;
  }
  CHECK(found);
}

TEST(timeman_our_actions_round_up) {
  // T11 (issue 454: A = ceil(M / 2)): odd and fractional moves-left - every earlier hand check
  // used an even M.
  TimeParams p;
  TimeContext ctx;
  ctx.remaining_seconds = 100;
  for (const auto& [turns, actions] : {std::pair{41.0, 21.0}, std::pair{37.3, 19.0}, std::pair{20.5, 11.0}}) {
    ctx.moves_left_turns = turns;
    CHECK_EQ(plan_decision(p, 0.01, 100, 0, true, ctx, std::nan("")).our_actions, actions);
  }
}

TEST(timeman_the_contexts_moves_left_comes_before_the_roots) {
  // T12: a context's prediction is read before the root's network output.
  TimeParams p;
  TimeContext ctx;
  ctx.remaining_seconds = 100;
  ctx.moves_left_turns = 40;
  const TimePlan plan = plan_decision(p, 0.01, 100, 0, true, ctx, 60.0);
  CHECK_EQ(plan.moves_left_turns, 40.0);
  CHECK_EQ(plan.our_actions, 20.0);
}

TEST(timeman_the_integral_reads_the_board_range_at_every_step) {
  // T14: from 40 boards over 30 actions the trajectory crosses the 50- and the 100-board ranges
  // of boards_per_action; the sum by hand, the range read at each step.
  TimeParams p;
  double by_hand = 0;
  double n = 40;
  const double c0 = cost_relative(p, 40);
  for (int k = 0; k < 30; ++k) {
    by_hand += cost_relative(p, n) / c0;
    n += 2 * (n < 50 ? 1.05 : n < 100 ? 1.25 : 1.77);
  }
  CHECK(n > 100);  // the premise: both boundaries crossed
  CHECK(near(integral_cost_sum(p, 40, 30, 2), by_hand, 1e-9 * by_hand));
}

TEST(timeman_e8_never_plans_below_its_floor_nor_past_e7s_hard_bound) {
  // T30 (guard 2): an easy call at factor 0.25 on a 40-evaluation plan gives 10, lifted to the
  // floor's 32. T31 (guard 3): a hard call at factor 4 on a 100-evaluation plan gives 400, cut to
  // the hard bound's 150 (1.5 s at 0.01 s each).
  TimeParams p;
  p.difficulty_weighting = true;
  CHECK(near(apply_difficulty(p, plan_of(40), effort_at(0), 8.0).soft_evaluations, 32.0, 1e-9));
  CHECK(near(apply_difficulty(p, plan_of(100, 1.5), effort_at(10), 8.0).soft_evaluations, 150.0, 1e-9));
}

TEST(timeman_no_batch_starts_that_would_end_past_the_hard_bound) {
  // L8(c): an evaluator twenty times dearer than the prior, a 30 s clock - the plan, made at the
  // prior, is far from its soft bound when the hard bound is reached; once the scale is measured
  // (after the first batch) each check predicts the next batch, so the decision ends INSIDE its
  // hard bound, strictly - the batch margin's work (a mutant without it, or predicting one
  // evaluation a batch, ends at 1.2079 s against 1.1732 s; 644801c at 1.1130 s). A virtual clock:
  // no wall-clock timing.
  TimeParams p;
  TimeManager tm(p);
  ManualClock clock;
  ChargingEvaluator ev(clock, 20 * p.cost_seconds_per_evaluation);
  Search search(variant("Standard"), quiet_params());
  TimeContext ctx;
  ctx.remaining_seconds = 30.0;
  const TimedRun run = run_timed(search, ev, tm, ctx, &clock);
  CHECK(run.verdict.stop == TimeStop::HardBound);  // the premise
  CHECK(clock.t <= run.plan.hard_seconds);
}
