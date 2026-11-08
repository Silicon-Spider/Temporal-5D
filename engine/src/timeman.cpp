// Time management (t5d/timeman.hpp; engine stage E11).

#include "t5d/timeman.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "t5d/ladder.hpp"

namespace t5d::mcts {

const char* allocator_name(Allocator a) noexcept {
  switch (a) {
    case Allocator::Integral: return "integral";
    case Allocator::FlatEvals: return "flat_evals";
    case Allocator::FlatWallClock: return "flat_wall_clock";
  }
  return "?";
}

const char* time_stop_name(TimeStop s) noexcept {
  switch (s) {
    case TimeStop::None: return "none";
    case TimeStop::SoftBudget: return "soft_budget";
    case TimeStop::SmartPruning: return "smart_pruning";
    case TimeStop::KldGain: return "kld_gain";
    case TimeStop::Solved: return "solved";
    case TimeStop::HardBound: return "hard_bound";
    case TimeStop::Search: return "search";
  }
  return "?";
}

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void need(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("TimeParams::") + what);
}

bool finite_at_least(double x, double low) noexcept { return std::isfinite(x) && x >= low; }

}  // namespace

void validate(const TimeParams& p) {
  need(static_cast<int>(p.allocator) <= static_cast<int>(Allocator::FlatWallClock),
       "allocator: Integral, FlatEvals or FlatWallClock");
  need(p.cost_tokens_per_board > 0 && p.cost_global_tokens >= 0 && p.cost_d_model > 0,
       "cost_tokens_per_board and cost_d_model: positive; cost_global_tokens: 0 or more");
  need(finite_at_least(p.cost_ffn_ratio, 0), "cost_ffn_ratio: a finite number, 0 or more");
  need(p.cost_reference_boards > 0, "cost_reference_boards: positive");
  need(std::isfinite(p.cost_seconds_per_evaluation) && p.cost_seconds_per_evaluation > 0,
       "cost_seconds_per_evaluation: positive and finite");
  need(std::isfinite(p.cost_ema) && p.cost_ema > 0 && p.cost_ema <= 1, "cost_ema: in (0, 1]");
  need(std::isfinite(p.cost_fixed_fraction) && p.cost_fixed_fraction >= 0 && p.cost_fixed_fraction < 1,
       "cost_fixed_fraction: in [0, 1)");
  need(finite_at_least(p.moves_left_floor_turns, 0), "moves_left_floor_turns: finite, 0 or more");
  need(std::isfinite(p.moves_left_default_turns) && p.moves_left_default_turns > 0,
       "moves_left_default_turns: positive and finite");
  for (const double b : p.boards_per_action) need(finite_at_least(b, 0), "boards_per_action: finite, 0 or more");
  need(p.boards_per_action_ranges[0] > 0 && p.boards_per_action_ranges[0] < p.boards_per_action_ranges[1],
       "boards_per_action_ranges: positive and ascending");
  need(finite_at_least(p.expected_optional_moves, 0), "expected_optional_moves: finite, 0 or more");
  need(finite_at_least(p.overhead_seconds, 0), "overhead_seconds: finite, 0 or more");
  need(finite_at_least(p.min_evaluations, 0), "min_evaluations: finite, 0 or more");
  need(finite_at_least(p.playouts_per_evaluation_cap, 1), "playouts_per_evaluation_cap: finite, 1 or more");
  need(finite_at_least(p.hard_multiple_low, 1) && finite_at_least(p.hard_multiple_high, 1),
       "hard_multiple_low and hard_multiple_high: finite, 1 or more");
  need(p.hard_low_boards >= 0 && p.hard_low_boards < p.hard_high_boards,
       "hard_low_boards and hard_high_boards: 0 or more, ascending");
  need(std::isfinite(p.hard_fraction) && p.hard_fraction > 0 && p.hard_fraction <= 1, "hard_fraction: in (0, 1]");
  need(finite_at_least(p.batch_margin, 0), "batch_margin: finite, 0 or more");
  need(finite_at_least(p.max_extension, 0), "max_extension: finite, 0 or more");
  need(finite_at_least(p.stop_min_evaluations, 0), "stop_min_evaluations: finite, 0 or more");
  need(std::isfinite(p.smart_pruning_factor) && p.smart_pruning_factor > 0, "smart_pruning_factor: positive");
  need(p.kld_interval >= 1, "kld_interval: 1 or more");
  need(finite_at_least(p.kld_threshold, 0), "kld_threshold: finite, 0 or more");
  need(std::isfinite(p.overlap_threshold) && p.overlap_threshold >= 0 && p.overlap_threshold <= 1,
       "overlap_threshold: in [0, 1]");
  need(finite_at_least(p.churn_kld_threshold, 0), "churn_kld_threshold: finite, 0 or more");
  need(finite_at_least(p.variance_threshold, 0), "variance_threshold: finite, 0 or more");
  need(finite_at_least(p.disagreement_margin, 0), "disagreement_margin: finite, 0 or more");
  need(finite_at_least(p.fall_margin, 0), "fall_margin: finite, 0 or more");
  need(std::isfinite(p.instability_window) && p.instability_window >= 0 && p.instability_window <= 1,
       "instability_window: in [0, 1]");
  for (const double e : {p.extend_overlap, p.extend_volatility, p.extend_disagreement, p.extend_fall,
                         p.extend_instability}) {
    need(finite_at_least(e, 0), "extend_*: finite, 0 or more");
  }
  need(std::isfinite(p.effort_quantile) && p.effort_quantile > 0 && p.effort_quantile <= 1,
       "effort_quantile: in (0, 1]");
  need(std::isfinite(p.wont_resolve_mass) && p.wont_resolve_mass >= 0 && p.wont_resolve_mass <= 1,
       "wont_resolve_mass: in [0, 1]");
  need(static_cast<int>(p.wont_resolve_policy) <= static_cast<int>(WontResolve::CapAtPlan),
       "wont_resolve_policy: HardBound or CapAtPlan");
  need(finite_at_least(p.difficulty_floor_evaluations, 0), "difficulty_floor_evaluations: finite, 0 or more");
  need(std::isfinite(p.difficulty_min_factor) && p.difficulty_min_factor > 0 &&
           std::isfinite(p.difficulty_max_factor) && p.difficulty_max_factor >= p.difficulty_min_factor,
       "difficulty_min_factor and difficulty_max_factor: positive, finite, min <= max");
  need(std::isfinite(p.difficulty_reference_visits) && p.difficulty_reference_visits >= 1,
       "difficulty_reference_visits: finite, 1 or more");
  need(finite_at_least(p.difficulty_reference_weight, 0), "difficulty_reference_weight: finite, 0 or more");
}

// ---- the pure functions -------------------------------------------------------------

double cost_shape(const TimeParams& p, double boards) noexcept {
  const double t = p.cost_tokens_per_board * std::max(boards, 0.0) + p.cost_global_tokens;
  return t * (4.0 * t + (8.0 + 4.0 * p.cost_ffn_ratio) * p.cost_d_model);
}

double cost_relative(const TimeParams& p, double boards) noexcept {
  const double f = p.cost_fixed_fraction;
  return f + (1.0 - f) * cost_shape(p, boards) / cost_shape(p, p.cost_reference_boards);
}

double boards_per_action(const TimeParams& p, double boards) noexcept {
  if (boards < p.boards_per_action_ranges[0]) return p.boards_per_action[0];
  if (boards < p.boards_per_action_ranges[1]) return p.boards_per_action[1];
  return p.boards_per_action[2];
}

double hard_multiple(const TimeParams& p, double boards) noexcept {
  if (boards <= p.hard_low_boards) return p.hard_multiple_low;
  if (boards >= p.hard_high_boards) return p.hard_multiple_high;
  const double t = (boards - p.hard_low_boards) / static_cast<double>(p.hard_high_boards - p.hard_low_boards);
  return p.hard_multiple_low + t * (p.hard_multiple_high - p.hard_multiple_low);
}

double integral_cost_sum(const TimeParams& p, double boards, double actions, double step) noexcept {
  const double c0 = cost_relative(p, boards);
  double n = boards;
  double sum = 0;
  for (double k = 0; k < actions; k += 1) {
    sum += cost_relative(p, n) / c0;
    n += step * boards_per_action(p, n);
  }
  return std::max(sum, 1.0);
}

double visit_kld(std::span<const std::uint32_t> prev, std::span<const std::uint32_t> cur) noexcept {
  double sp = 0;
  double sc = 0;
  for (const std::uint32_t v : prev) sp += v;
  for (const std::uint32_t v : cur) sc += v;
  if (sp <= 0) return std::numeric_limits<double>::quiet_NaN();  // nothing to compare with
  if (sc <= 0) return std::numeric_limits<double>::infinity();
  double kl = 0;
  for (std::size_t i = 0; i < prev.size(); ++i) {
    if (prev[i] == 0) continue;
    const double a = prev[i] / sp;
    const double b = i < cur.size() ? cur[i] / sc : 0.0;
    if (b <= 0) return std::numeric_limits<double>::infinity();
    kl += a * std::log(a / b);
  }
  return std::max(kl, 0.0);
}

double value_overlap(const std::array<float, kValueBins>& a, const std::array<float, kValueBins>& b) noexcept {
  double sa = 0;
  double sb = 0;
  for (int k = 0; k < kValueBins; ++k) {
    sa += std::max(0.0f, a[static_cast<std::size_t>(k)]);
    sb += std::max(0.0f, b[static_cast<std::size_t>(k)]);
  }
  if (sa <= 0 || sb <= 0) return 0.0;
  double o = 0;
  for (int k = 0; k < kValueBins; ++k) {
    const auto i = static_cast<std::size_t>(k);
    o += std::min(std::max(0.0f, a[i]) / sa, std::max(0.0f, b[i]) / sb);
  }
  return std::clamp(o, 0.0, 1.0);
}

bool smart_pruning_stop(std::span<const std::uint32_t> visits, double remaining_playouts, double factor) noexcept {
  std::uint32_t first = 0;
  std::uint32_t second = 0;
  std::size_t count = 0;
  for (const std::uint32_t v : visits) {
    ++count;
    if (v > first) {
      second = first;
      first = v;
    } else if (v > second) {
      second = v;
    }
  }
  if (first == 0) return false;
  if (count == 1) return true;  // one child: nothing to decide
  return static_cast<double>(first - second) > std::max(remaining_playouts, 0.0) * factor;
}

bool unstable(std::uint32_t root_visits, std::uint32_t best_since, double window, std::uint32_t min_visits) noexcept {
  if (root_visits < min_visits || best_since == 0 || best_since > root_visits) return false;
  return static_cast<double>(root_visits - best_since) < window * root_visits;
}

TimePlan plan_decision(const TimeParams& p, double scale, int boards, int mandatory_remaining, bool whole_action,
                       const TimeContext& ctx, double root_moves_left) {
  TimePlan out;
  out.boards = boards;
  double ml = ctx.moves_left_turns;
  if (!(std::isfinite(ml) && ml >= 0)) ml = root_moves_left;
  out.moves_left_given = std::isfinite(ml) && ml >= 0;
  out.moves_left_turns = std::max(out.moves_left_given ? ml : p.moves_left_default_turns, p.moves_left_floor_turns);
  // Our remaining actions: the turns alternate, ours first (this one).
  const double a = std::max(1.0, std::ceil(out.moves_left_turns / 2.0));
  out.our_actions = a;
  const double s = scale * cost_relative(p, boards);
  out.seconds_per_evaluation = s;
  // The increment after our last action is never usable; the overhead is paid by each.
  out.usable_seconds = std::max(0.0, ctx.remaining_seconds + ctx.increment_seconds * (a - 1) - p.overhead_seconds * a);
  if (p.allocator == Allocator::Integral) {
    out.share = 1.0 / integral_cost_sum(p, boards, a, 2.0);  // both players' actions between ours
  } else {
    out.share = 1.0 / a;
  }
  const double action_seconds = out.usable_seconds * out.share;
  out.action_evaluations = action_seconds / s;
  // A6: an action's decisions - its mandatory boards still to move, its submit
  // and the optional moves expected; the flat form decides the whole action.
  out.decisions_left = whole_action ? 1.0 : std::max(0, mandatory_remaining) + 1.0 + p.expected_optional_moves;
  double soft_seconds = action_seconds / out.decisions_left;
  double soft_evaluations = std::max(soft_seconds / s, p.min_evaluations);
  soft_seconds = soft_evaluations * s;
  const double cap = p.hard_fraction * std::max(0.0, ctx.remaining_seconds - p.overhead_seconds);
  out.hard_seconds = std::min(hard_multiple(p, boards) * soft_seconds, cap);
  if (soft_seconds > out.hard_seconds) {
    soft_seconds = out.hard_seconds;
    soft_evaluations = soft_seconds / s;
  }
  out.soft_seconds = soft_seconds;
  out.soft_evaluations = soft_evaluations;
  out.e7_soft_evaluations = soft_evaluations;
  return out;
}

// ---- E8 -----------------------------------------------------------------------------------

int effort_quantile_bin(const std::array<float, kEffortBins>& probs, double q) noexcept {
  // heads.effort_quantile: (cumsum(probs) < q).sum(), capped at the last bin -
  // the cumulative sum in float32, as torch's, over the distribution normalised.
  float total = 0;
  for (const float x : probs) total += std::max(0.0f, x);
  int below = 0;
  float cdf = 0;
  for (const float x : probs) {
    cdf += total > 0 ? std::max(0.0f, x) / total : 0.0f;
    if (static_cast<double>(cdf) < q) ++below;
  }
  return std::min(below, kEffortBins - 1);
}

TimePlan apply_difficulty(const TimeParams& p, const TimePlan& e7, const std::array<float, kEffortBins>& effort,
                          double reference_bin) {
  for (const float x : effort) {
    if (!std::isfinite(x) || x < 0) {
      throw std::invalid_argument("TimeContext::effort: probabilities, finite and 0 or more");
    }
  }
  double total = 0;
  for (const float x : effort) total += x;
  if (!(total > 0)) throw std::invalid_argument("TimeContext::effort: not all zero");
  TimePlan out = e7;
  out.e7_soft_evaluations = e7.soft_evaluations;
  out.difficulty_applied = true;
  out.effort_bin = effort_quantile_bin(effort, p.effort_quantile);
  out.censored_mass = effort[kEffortBins - 1] / total;
  out.reference_bin = reference_bin;
  const double s = e7.seconds_per_evaluation;
  const double hard_evaluations = e7.hard_seconds / s;
  out.wont_resolve = out.censored_mass >= p.wont_resolve_mass;
  double plan = e7.soft_evaluations;
  if (out.wont_resolve) {
    // §8.6: "plan for the hard bound and lean on LCB ... rather than chase convergence".
    plan = p.wont_resolve_policy == WontResolve::HardBound ? hard_evaluations : e7.soft_evaluations;
    out.difficulty_factor = plan / std::max(e7.soft_evaluations, 1e-300);
  } else {
    // The quantile's visits against the reference's - the censored bin the largest.
    const double factor = out.effort_bin >= kEffortBins - 1
                              ? p.difficulty_max_factor
                              : std::clamp(std::exp2(out.effort_bin - reference_bin), p.difficulty_min_factor,
                                           p.difficulty_max_factor);
    out.difficulty_factor = factor;
    plan = std::max(e7.soft_evaluations * factor, p.difficulty_floor_evaluations);  // guard 2
  }
  plan = std::min(plan, hard_evaluations);  // guard 3: the hard bound is E7's
  out.soft_evaluations = plan;
  out.soft_seconds = plan * s;
  return out;
}

// ---- the root as the signals read it ------------------------------------------------

RootSnapshot snapshot(const Search& search, std::uint64_t evaluations, std::uint64_t playouts,
                      const TimeParams& params) {
  RootSnapshot r;
  r.evaluations = evaluations;
  r.playouts = playouts;
  const NodeView root = search.node(search.root());
  r.root_visits = root.visits;
  r.root_q = root.weight > 0 ? root.value / root.weight : 0.0;
  r.root_variance = root.eval_variance;
  r.root_net_value = root.eval_mean;
  r.root_evaluated = root.evaluated;
  r.best_since = search.root_best_since();
  r.solved = search.root_solved();
  r.proof_channel = search.params().proof_channel;
  r.batch_size = search.params().synchronous ? 1 : std::max<std::size_t>(search.params().batch_size, 1);
  const std::vector<RootChild> children = search.root_children();
  r.visits.reserve(children.size());
  for (const RootChild& c : children) r.visits.push_back(c.visits);
  if (params.signals) {
    // The top two children by visits (ties: the lower index).
    std::size_t top[2] = {children.size(), children.size()};
    for (std::size_t i = 0; i < children.size(); ++i) {
      if (children[i].visits == 0) continue;
      if (top[0] == children.size() || children[i].visits > children[top[0]].visits) {
        top[1] = top[0];
        top[0] = i;
      } else if (top[1] == children.size() || children[i].visits > children[top[1]].visits) {
        top[1] = i;
      }
    }
    for (int k = 0; k < 2; ++k) {
      if (top[k] < children.size()) r.top_value[k] = search.root_child_value(top[k]);
    }
  }
  if (params.proof_volatility && r.proof_channel) {
    // The hook (issue 180): each proven verdict against the network's value of
    // what it proves - our mate at the root (+1 against the root's evaluation),
    // the opponent's after our principal line (-1 against that line's child's).
    for (const LadderVerdict& v : search.verdicts()) {
      if (v.status != ProofStatus::Proven) continue;
      double net = r.root_net_value;
      double proven = 1.0;
      if (v.job == LadderJob::PvSettling) {
        proven = -1.0;
        if (!v.path.empty()) {
          for (const RootChild& c : children) {
            if (c.submit == v.path.front().submit && c.moves == v.path.front().moves && c.evaluated) {
              net = c.network_wdl[0] - c.network_wdl[2];
              break;
            }
          }
        }
      }
      r.proof_gap = std::max(r.proof_gap, std::abs(proven - net));
    }
  }
  return r;
}

// ---- the time manager ------------------------------------------------------------------

TimeManager::TimeManager(TimeParams params) : params_(params) {
  validate(params_);
  scale_ = params_.cost_seconds_per_evaluation;
}

double TimeManager::seconds_per_evaluation(double boards) const noexcept {
  return scale_ * cost_relative(params_, boards);
}

double TimeManager::model_seconds() const noexcept { return work_ * seconds_per_evaluation(plan_.boards); }

void TimeManager::new_game() noexcept {
  last_value_.reset();
  effort_bins_sum_ = 0;
  effort_decisions_ = 0;
  active_ = false;
}

double TimeManager::reference_bin() const noexcept {
  const double w = params_.difficulty_reference_weight;
  const double prior = std::log2(params_.difficulty_reference_visits);
  const double n = w + effort_decisions_;
  return n > 0 ? (w * prior + effort_bins_sum_) / n : prior;
}

TimePlan TimeManager::begin(const Search& search, const TimeContext& ctx) {
  if (!(std::isfinite(ctx.remaining_seconds) && ctx.remaining_seconds > 0)) {
    throw std::invalid_argument("TimeContext::remaining_seconds: positive and finite");
  }
  if (!finite_at_least(ctx.increment_seconds, 0)) {
    throw std::invalid_argument("TimeContext::increment_seconds: finite, 0 or more");
  }
  if (!std::isnan(ctx.moves_left_turns) && !finite_at_least(ctx.moves_left_turns, 0)) {
    throw std::invalid_argument("TimeContext::moves_left_turns: finite and 0 or more, or NaN (none)");
  }
  if (params_.proof_volatility && !search.params().proof_channel) {
    throw std::invalid_argument(
        "TimeParams::proof_volatility: Hunt's and the settling scans' proofs may not drive time management "
        "while the proof channel is off (build-log spec issue 180, open; H41)");
  }
  const GameState& root = search.root_state();
  const NodeView rv = search.node(search.root());
  plan_ = plan_decision(params_, scale_, static_cast<int>(root.multiverse().board_count()), root.mandatory_count(),
                        search.params().split == TurnSplit::Flat, ctx,
                        rv.evaluated ? static_cast<double>(rv.eval_moves_left) : kNaN);
  if (params_.difficulty_weighting && ctx.effort) {
    // E8 (off by default): the plan weighted by the predicted difficulty; the
    // game's reference updated after - this decision is judged against the
    // ones before it.
    plan_ = apply_difficulty(params_, plan_, *ctx.effort, reference_bin());
    effort_bins_sum_ += plan_.effort_bin;
    effort_decisions_ += 1;
  }
  std::vector<std::uint32_t> visits;
  for (const RootChild& c : search.root_children()) visits.push_back(c.visits);
  begin(plan_, std::move(visits), rv.visits);
  const SearchStats& st = search.stats();
  e0_ = st.evaluations;
  p0_ = st.playouts;
  return plan_;
}

void TimeManager::begin(const TimePlan& plan, std::vector<std::uint32_t> visits, std::uint32_t root_visits) {
  plan_ = plan;
  e0_ = 0;
  p0_ = 0;
  target_ = plan_.soft_evaluations;
  extended_ = 0;
  taken_ = 0;
  work_ = 0;
  last_elapsed_ = 0;
  last_work_ = 0;
  kld_prev_ = std::move(visits);
  kld_prev_visits_ = root_visits;
  kld_gain_ = kNaN;
  verdict_ = TimeVerdict{};
  verdict_.target_evaluations = target_;
  active_ = true;
}

TimeVerdict TimeManager::check(const Search& search, double elapsed) {
  if (!active_) throw std::logic_error("TimeManager::check: no decision begun");
  const SearchStats& st = search.stats();
  return check(snapshot(search, st.evaluations - e0_, st.playouts - p0_, params_), elapsed);
}

TimeVerdict TimeManager::check(const RootSnapshot& root, double elapsed) {
  if (!active_) throw std::logic_error("TimeManager::check: no decision begun");
  const TimeParams& p = params_;
  const bool model = std::isnan(elapsed);
  if (!model && !finite_at_least(elapsed, 0)) {
    throw std::invalid_argument("TimeManager::check: elapsed seconds finite and 0 or more, or NaN (the model clock)");
  }
  const double work = std::max(static_cast<double>(root.evaluations),
                               static_cast<double>(root.playouts) / p.playouts_per_evaluation_cap);
  work_ = work;
  if (!model) {
    // The scale, measured: seconds per unit of work between checks, the
    // shape removed. The first observation replaces the prior.
    const double dw = work - last_work_;
    const double dt = elapsed - last_elapsed_;
    if (dw > 0 && dt > 0) {
      const double observed = dt / dw / cost_relative(p, plan_.boards);
      scale_ = observed_ ? (1.0 - p.cost_ema) * scale_ + p.cost_ema * observed : observed;
      scale_ = std::max(scale_, 1e-15);
      observed_ = true;
    }
    last_elapsed_ = elapsed;
    last_work_ = work;
  }
  const double s = seconds_per_evaluation(plan_.boards);
  TimeVerdict v;
  v.work = work;
  v.elapsed_seconds = model ? work * s : elapsed;
  v.predicted_batch_seconds = static_cast<double>(root.batch_size) * s;
  // KLDGain: every kld_interval root visits, the divergence per visit.
  if (root.root_visits >= kld_prev_visits_ + p.kld_interval) {
    if (root.visits.size() >= kld_prev_.size()) {
      // NaN where the earlier distribution was empty (a fresh root): not yet measured.
      kld_gain_ = visit_kld(kld_prev_, root.visits) / static_cast<double>(root.root_visits - kld_prev_visits_);
    } else {
      kld_gain_ = kNaN;  // a child was removed: the comparison starts again
    }
    kld_prev_ = root.visits;
    kld_prev_visits_ = root.root_visits;
  }
  v.kld_gain = kld_gain_;
  const auto finish = [&](TimeStop stop) {
    v.stop = stop;
    v.target_evaluations = target_;
    v.extensions = taken_;
    verdict_ = v;
    return v;
  };
  const bool visited = std::any_of(root.visits.begin(), root.visits.end(), [](std::uint32_t x) { return x > 0; });
  if (!visited) return finish(TimeStop::None);  // one batch must run: a move needs a visit
  if (root.solved && root.proof_channel) return finish(TimeStop::Solved);  // H41: only with the channel on
  if (v.elapsed_seconds + p.batch_margin * v.predicted_batch_seconds >= plan_.hard_seconds) {
    return finish(TimeStop::HardBound);
  }
  const double plan = plan_.soft_evaluations;
  const auto reached = [&] {
    if (p.allocator == Allocator::FlatWallClock) {
      const double ratio = plan > 0 ? target_ / plan : 1.0;
      return v.elapsed_seconds >= plan_.soft_seconds * ratio;
    }
    return work >= target_;
  };
  if (reached() && p.signals) {
    // The extensions: each kind at most once a decision, by its share of the
    // ORIGINAL plan, together at most max_extension of it (the spiral guard).
    const double cap = p.max_extension * plan;
    const auto take = [&](std::uint32_t kind, double share) {
      if ((taken_ & kind) != 0) return;
      const double add = std::min(share * plan, cap - extended_);
      if (add <= 0) return;
      taken_ |= kind;
      extended_ += add;
      target_ += add;
    };
    const bool net_alarm = root.root_evaluated && root.root_variance >= p.variance_threshold;
    const bool prover_alarm = p.proof_volatility && root.proof_channel && root.proof_gap >= p.disagreement_margin;
    if (prover_alarm && !net_alarm) take(kExtendDisagreement, p.extend_disagreement);
    if (net_alarm) take(kExtendVolatility, p.extend_volatility);
    if (unstable(root.root_visits, root.best_since, p.instability_window, p.instability_min_visits)) {
      take(kExtendInstability, p.extend_instability);
    }
    if (last_value_ && root.root_q < *last_value_ - p.fall_margin) take(kExtendFall, p.extend_fall);
    // E8: a won't-resolve decision does not chase convergence.
    if (!plan_.wont_resolve && root.top_value[0] && root.top_value[1] && std::isfinite(kld_gain_) &&
        kld_gain_ >= p.churn_kld_threshold &&
        value_overlap(*root.top_value[0], *root.top_value[1]) >= p.overlap_threshold) {
      take(kExtendOverlap, p.extend_overlap);
    }
  }
  if (reached()) return finish(TimeStop::SoftBudget);
  if (p.signals && work >= p.stop_min_evaluations) {
    double remaining = 0;
    if (p.allocator == Allocator::FlatWallClock) {
      const double ratio = plan > 0 ? target_ / plan : 1.0;
      remaining = (plan_.soft_seconds * ratio - v.elapsed_seconds) / s;
    } else {
      remaining = target_ - work;
    }
    remaining = std::min(remaining, (plan_.hard_seconds - v.elapsed_seconds) / s);
    const double per_evaluation =
        root.evaluations > 0 ? std::max(1.0, static_cast<double>(root.playouts) / static_cast<double>(root.evaluations)) : 1.0;
    if (smart_pruning_stop(root.visits, remaining * per_evaluation, p.smart_pruning_factor)) {
      return finish(TimeStop::SmartPruning);
    }
    if (std::isfinite(kld_gain_) && kld_gain_ < p.kld_threshold) return finish(TimeStop::KldGain);
  }
  return finish(TimeStop::None);
}

void TimeManager::end(double root_q) {
  last_value_ = root_q;
  active_ = false;
}

void TimeManager::end(const Search& search) {
  const NodeView root = search.node(search.root());
  if (root.weight > 0) {
    end(root.value / root.weight);
  } else {
    active_ = false;
  }
}

double WallClock::now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

TimedRun run_timed(Search& search, Evaluator& evaluator, TimeManager& tm, const TimeContext& ctx, Clock* clock) {
  TimedRun out;
  const SearchStats& st = search.stats();
  const std::uint64_t e0 = st.evaluations;
  const std::uint64_t p0 = st.playouts;
  const std::uint64_t b0 = st.batches;
  out.plan = tm.begin(search, ctx);
  const double t0 = clock != nullptr ? clock->now() : 0.0;
  // The hard bound inside a batch: the rules engine's calls stop at it.
  struct DeadlineReset {
    Search& s;
    ~DeadlineReset() { s.set_deadline(std::nullopt); }
  } reset{search};
  if (clock != nullptr && clock->steady()) {
    search.set_deadline(std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(out.plan.hard_seconds)));
  }
  // One slice: one batch's simulations (Search::run's loop, as a step-API
  // driver runs it for batch_size playouts - the same tree, H15); the
  // synchronous mode's batch is one leaf.
  Limits slice;
  slice.playouts = search.params().synchronous ? 1 : std::max<std::size_t>(search.params().batch_size, 1);
  for (;;) {
    const double elapsed = clock != nullptr ? clock->now() - t0 : std::numeric_limits<double>::quiet_NaN();
    out.verdict = tm.check(search, elapsed);
    if (out.verdict.stop != TimeStop::None) break;
    const StopReason r = search.run(evaluator, slice);
    if (r != StopReason::Playouts && r != StopReason::Evaluations) {
      out.search_stop = r;
      out.verdict.stop = r == StopReason::Finished && search.root_solved() && search.params().proof_channel
                             ? TimeStop::Solved
                             : TimeStop::Search;
      break;
    }
  }
  out.evaluations = st.evaluations - e0;
  out.playouts = st.playouts - p0;
  out.batches = st.batches - b0;
  if (clock != nullptr) {
    out.seconds = clock->now() - t0;
  } else {
    const double work = std::max(static_cast<double>(out.evaluations),
                                 static_cast<double>(out.playouts) / tm.params().playouts_per_evaluation_cap);
    out.seconds = work * tm.seconds_per_evaluation(out.plan.boards);
  }
  out.best_since = search.root_best_since();
  tm.end(search);
  return out;
}

}  // namespace t5d::mcts
