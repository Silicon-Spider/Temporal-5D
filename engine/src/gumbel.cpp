// gumbel.cpp - register E12 (Gumbel root search): the root's Gumbel sample and
// sequential halving (t5d/gumbel.hpp, whose header comment pins every form).
//
// Written from Danihelka et al., "Policy improvement by planning with Gumbel"
// (ICLR 2022), Karnin, Koren and Somekh's Sequential Halving (ICML 2013) and
// the Gumbel-top-k trick as the papers describe them; no code comes from any
// engine or library (H29).

#include "t5d/gumbel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace t5d::gumbel {

std::uint64_t mix64(std::uint64_t x) noexcept {
  x = (x ^ (x >> 30)) * std::uint64_t{0xBF58476D1CE4E5B9};
  x = (x ^ (x >> 27)) * std::uint64_t{0x94D049BB133111EB};
  return x ^ (x >> 31);
}

std::uint64_t stream_seed(std::uint64_t search_seed, std::uint64_t root_key) noexcept {
  return mix64(search_seed ^ mix64(root_key) ^ std::uint64_t{0x47554D42454C3132});  // "GUMBEL12"
}

double variate_of_bits(std::uint64_t bits) noexcept {
  // 52 bits and a half-step offset: u in (0, 1) strictly, so g is finite.
  const double u = (static_cast<double>(bits >> 12) + 0.5) * 0x1.0p-52;
  return -std::log(-std::log(u));
}

double variate(std::uint64_t stream, std::size_t index) noexcept {
  return variate_of_bits(mix64(stream + std::uint64_t{0x9E3779B97F4A7C15} * (static_cast<std::uint64_t>(index) + 1)));
}

Sample sample_order(std::span<const double> logits, std::uint64_t stream) {
  Sample s;
  s.g.resize(logits.size());
  s.order.reserve(logits.size());
  std::vector<std::size_t> closed;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double l = logits[i];
    if (std::isnan(l) || l == std::numeric_limits<double>::infinity()) {
      throw std::invalid_argument("gumbel: a logit is NaN or +infinity");
    }
    s.g[i] = variate(stream, i);
    if (l == -std::numeric_limits<double>::infinity()) {
      closed.push_back(i);
    } else {
      s.order.push_back(i);
    }
  }
  std::sort(s.order.begin(), s.order.end(), [&](std::size_t a, std::size_t b) {
    const double ka = s.g[a] + logits[a];
    const double kb = s.g[b] + logits[b];
    return ka != kb ? ka > kb : a < b;
  });
  s.order.insert(s.order.end(), closed.begin(), closed.end());
  return s;
}

std::vector<Phase> halving_schedule(std::size_t m, std::uint64_t n) {
  if (m == 0 || n == 0) throw std::invalid_argument("gumbel: the halving schedule needs m >= 1 and n >= 1");
  if (m == 1) return {Phase{1, n}};
  std::size_t phases = 0;  // ceil(log2 m)
  while ((std::size_t{1} << phases) < m) ++phases;
  std::vector<Phase> out;
  out.reserve(phases);
  std::size_t c = m;
  for (std::size_t r = 0; r < phases; ++r) {
    const std::uint64_t share = n / (static_cast<std::uint64_t>(phases) * static_cast<std::uint64_t>(c));
    out.push_back(Phase{c, std::max<std::uint64_t>(1, share)});
    c = (c + 1) / 2;
  }
  return out;
}

double sigma(double q, std::uint32_t max_visits, double c_visit, double c_scale) noexcept {
  return (c_visit + static_cast<double>(max_visits)) * c_scale * (q + 1.0) / 2.0;
}

double mixed_value(std::span<const Action> actions, double v_net) noexcept {
  std::uint64_t total = 0;
  double mass = 0;
  double weighted = 0;
  for (const Action& a : actions) {
    if (!a.open || a.visits == 0) continue;
    total += a.visits;
    mass += a.prior;
    weighted += a.prior * a.q;
  }
  if (total == 0 || !(mass > 0)) return v_net;
  const double n = static_cast<double>(total);
  return (v_net + n * weighted / mass) / (1.0 + n);
}

namespace {

bool is_open(std::span<const Action> actions, std::size_t edge) noexcept {
  return edge < actions.size() && actions[edge].open;
}

// A score as the halving and the choice order it: a NaN (a NaN q or prior from
// the caller) below every number, so that their comparisons are strict weak
// orders (sigma itself cannot overflow within the caps Root::make checks).
double ordered(double score) noexcept {
  return std::isnan(score) ? -std::numeric_limits<double>::infinity() : score;
}

std::uint32_t max_open_visits(std::span<const Action> actions) noexcept {
  std::uint32_t most = 0;
  for (const Action& a : actions) {
    if (a.open) most = std::max(most, a.visits);
  }
  return most;
}

}  // namespace

Root Root::make(std::vector<double> g_by_edge, std::size_t open_count, const Config& config) {
  if (!(std::isfinite(config.c_visit) && config.c_visit >= 0 && config.c_visit <= kMaxCVisit)) {
    throw std::invalid_argument("gumbel: c_visit must be finite, 0 or more and at most kMaxCVisit (1e6)");
  }
  if (!(std::isfinite(config.c_scale) && config.c_scale >= 0 && config.c_scale <= kMaxCScale)) {
    throw std::invalid_argument("gumbel: c_scale must be finite, 0 or more and at most kMaxCScale (1e6)");
  }
  Root r;
  const std::size_t open = std::min(open_count, g_by_edge.size());
  const std::size_t by_n = config.simulations < open ? static_cast<std::size_t>(config.simulations) : open;
  const std::size_t m = std::min(config.m, by_n);
  if (m == 0) return r;
  r.schedule_ = halving_schedule(m, config.simulations);
  r.considered_.reserve(g_by_edge.size());
  r.scratch_.reserve(g_by_edge.size());
  for (std::size_t e = 0; e < m; ++e) r.considered_.push_back(e);
  r.g_ = std::move(g_by_edge);
  r.open_count_ = open;
  r.active_ = true;
  r.m_ = m;
  r.shown_ = m;
  r.phase_ = 0;
  r.target_ = r.schedule_.front().visits;
  r.c_visit_ = config.c_visit;
  r.c_scale_ = config.c_scale;
  return r;
}

void Root::refill(std::span<const Action> actions) noexcept {
  // The first phase's candidates are the sample's first m open edges: a closed
  // one's place goes to the next open edge of the order.
  while (considered_.size() < m_ && shown_ < open_count_) {
    const std::size_t e = shown_++;
    if (is_open(actions, e)) considered_.push_back(e);  // shown_ only grows: edge order kept
  }
}

double Root::score(std::span<const Action> actions, std::size_t edge, double v_mix,
                   std::uint32_t max_visits) const noexcept {
  const Action& a = actions[edge];
  const double q = a.visits > 0 ? a.q : v_mix;
  return g(edge) + std::log(a.prior) + sigma(q, max_visits, c_visit_, c_scale_);
}

void Root::halve(std::span<const Action> actions, double v_net) noexcept {
  const double v_mix = mixed_value(actions, v_net);
  const std::uint32_t most = max_open_visits(actions);
  scratch_.clear();
  for (const std::size_t e : considered_) scratch_.emplace_back(ordered(score(actions, e, v_mix, most)), e);
  const std::size_t keep = std::min(schedule_[phase_ + 1].considered, considered_.size());
  std::sort(scratch_.begin(), scratch_.end(),
            [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
  considered_.clear();
  for (std::size_t k = 0; k < keep; ++k) considered_.push_back(scratch_[k].second);
  std::sort(considered_.begin(), considered_.end());
  ++phase_;
  target_ += schedule_[phase_].visits;
}

std::size_t Root::next(std::span<const Action> actions, double v_net) noexcept {
  if (!active_) return kNone;
  for (;;) {
    considered_.erase(std::remove_if(considered_.begin(), considered_.end(),
                                     [&](std::size_t e) { return !is_open(actions, e); }),
                      considered_.end());
    if (phase_ == 0) refill(actions);
    if (considered_.empty()) {
      // Every candidate closed: the open edges already shown, else the next.
      for (std::size_t e = 0; e < shown_; ++e) {
        if (is_open(actions, e)) considered_.push_back(e);
      }
      while (considered_.empty() && shown_ < open_count_) {
        const std::size_t e = shown_++;
        if (is_open(actions, e)) considered_.push_back(e);
      }
      if (considered_.empty()) return kNone;
    }
    std::size_t pick = considered_.front();
    for (const std::size_t e : considered_) {
      if (actions[e].visits + actions[e].pending < actions[pick].visits + actions[pick].pending) pick = e;
    }
    const bool last = phase_ + 1 >= schedule_.size();
    if (last || actions[pick].visits + actions[pick].pending < target_) return pick;
    for (const std::size_t e : considered_) {
      if (actions[e].visits < target_) return kWait;  // the phase's last visits are pending
    }
    halve(actions, v_net);
  }
}

std::size_t Root::best(std::span<const Action> actions, double v_net) const noexcept {
  if (!active_) return kNone;
  const double v_mix = mixed_value(actions, v_net);
  const std::uint32_t most = max_open_visits(actions);
  std::size_t found = kNone;
  double top = 0;
  const auto consider = [&](std::size_t e) {
    if (!is_open(actions, e)) return;
    const double s = ordered(score(actions, e, v_mix, most));
    if (found == kNone || s > top) {
      found = e;
      top = s;
    }
  };
  for (const std::size_t e : considered_) consider(e);
  if (found == kNone) {
    for (std::size_t e = 0; e < shown_; ++e) consider(e);
  }
  return found;
}

}  // namespace t5d::gumbel
