// Register E12 (Gumbel root search; t5d/gumbel.hpp): the pure parts - the
// variate, the Gumbel-top-k order, the halving schedule, sigma and the mixed
// value, the halving driven as a bandit - then the search's arm behind
// SearchParams::gumbel_root, and its C API.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "t5d/capi.h"
#include "t5d/capi_search.h"
#include "t5d/gumbel.hpp"
#include "t5d/legal.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/timeman.hpp"
#include "t5d/variants.hpp"
#include "test_alloc.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// A root's edges in the sample's order, as the search lays them out: edge k
// is the order's k-th index; its g, prior (the softmax of `logits`) and q.
struct Bandit {
  std::vector<gumbel::Action> actions;
  std::vector<double> g;
  std::vector<double> q;
  std::vector<std::size_t> original;  // edge -> index in `logits`
};

Bandit bandit(const std::vector<double>& logits, const std::vector<double>& q, std::uint64_t stream) {
  const gumbel::Sample s = gumbel::sample_order(logits, stream);
  double z = 0;
  for (const double l : logits) z += std::isfinite(l) ? std::exp(l) : 0.0;
  Bandit b;
  for (const std::size_t i : s.order) {
    gumbel::Action a;
    a.prior = std::isfinite(logits[i]) ? std::exp(logits[i]) / z : 0.0;
    a.open = a.prior > 0;
    b.actions.push_back(a);
    b.g.push_back(s.g[i]);
    b.q.push_back(q[i]);
    b.original.push_back(i);
  }
  return b;
}

std::size_t open_prefix(const Bandit& b) {
  std::size_t n = 0;
  while (n < b.actions.size() && b.actions[n].open) ++n;
  return n;
}

// One completed visit to `e`: its q is the action's fixed value.
void visit(Bandit& b, std::size_t e) {
  gumbel::Action& a = b.actions[e];
  ++a.visits;
  a.q = b.q[e];
}

// The test's own score of edge e (every form written out again).
double own_score(const Bandit& b, std::size_t e, double v_net, double c_visit, double c_scale) {
  std::uint32_t most = 0;
  for (const gumbel::Action& a : b.actions) {
    if (a.open) most = std::max(most, a.visits);
  }
  std::uint64_t total = 0;
  double mass = 0, weighted = 0;
  for (const gumbel::Action& a : b.actions) {
    if (!a.open || a.visits == 0) continue;
    total += a.visits;
    mass += a.prior;
    weighted += a.prior * a.q;
  }
  const double v_mix = total == 0 ? v_net : (v_net + static_cast<double>(total) * weighted / mass) / (1.0 + static_cast<double>(total));
  const gumbel::Action& a = b.actions[e];
  const double q = a.visits > 0 ? a.q : v_mix;
  return b.g[e] + std::log(a.prior) + (c_visit + static_cast<double>(most)) * c_scale * (q + 1.0) / 2.0;
}

}  // namespace

TEST(gumbel_variate_is_finite_pure_and_gumbel_distributed) {
  CHECK(std::abs(gumbel::variate_of_bits(0) - -3.6037789929704576) < 1e-12);
  CHECK(std::abs(gumbel::variate_of_bits(~std::uint64_t{0}) - 36.7368005696771) < 1e-9);
  CHECK(std::isfinite(gumbel::variate_of_bits(0)) && std::isfinite(gumbel::variate_of_bits(~std::uint64_t{0})));
  CHECK_EQ(gumbel::variate(7, 3), gumbel::variate(7, 3));
  CHECK(gumbel::variate(7, 3) != gumbel::variate(7, 4));
  CHECK(gumbel::variate(7, 3) != gumbel::variate(8, 3));
  // Gumbel(0, 1): mean the Euler-Mascheroni constant, variance pi^2 / 6.
  constexpr std::size_t kDraws = 200'000;
  double sum = 0, squares = 0;
  for (std::size_t i = 0; i < kDraws; ++i) {
    const double g = gumbel::variate(12345, i);
    sum += g;
    squares += g * g;
  }
  const double draws = static_cast<double>(kDraws);
  const double mean = sum / draws;
  const double variance = squares / draws - mean * mean;
  std::fprintf(stderr, "  %zu draws: mean %.4f, variance %.4f\n", kDraws, mean, variance);
  CHECK(std::abs(mean - 0.5772156649) < 0.01);
  CHECK(std::abs(variance - 1.6449340668) < 0.03);
  // The stream: a function of the seed and the root's key.
  CHECK_EQ(gumbel::stream_seed(1, 2), gumbel::stream_seed(1, 2));
  CHECK(gumbel::stream_seed(1, 2) != gumbel::stream_seed(2, 2));
  CHECK(gumbel::stream_seed(1, 2) != gumbel::stream_seed(1, 3));
  // SplitMix64's finaliser on known inputs (Python's selfplay.mix gives the same).
  CHECK_EQ(gumbel::mix64(0), std::uint64_t{0});
  CHECK_EQ(gumbel::mix64(1), std::uint64_t{0x5692161D100B05E5});
  CHECK_EQ(gumbel::mix64(2), std::uint64_t{0xDBD238973A2B148A});
  CHECK_EQ(gumbel::mix64(~std::uint64_t{0}), std::uint64_t{0xB4D055FCF2CBBD7B});  // the Python side pins these three too
}

TEST(gumbel_sample_order_is_a_permutation_with_closed_edges_last) {
  const std::vector<double> logits{0.1, -kInf, 1.5, -0.3, -kInf, 0.7, 0.0};
  const gumbel::Sample s = gumbel::sample_order(logits, 99);
  REQUIRE_EQ(s.order.size(), logits.size());
  std::vector<std::size_t> sorted = s.order;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) CHECK_EQ(sorted[i], i);
  for (std::size_t i = 0; i < logits.size(); ++i) CHECK_EQ(s.g[i], gumbel::variate(99, i));
  // Five open first, by g + logit non-increasing; then the closed in order.
  for (std::size_t k = 0; k + 1 < 5; ++k) {
    const std::size_t a = s.order[k], b = s.order[k + 1];
    CHECK(s.g[a] + logits[a] >= s.g[b] + logits[b]);
  }
  CHECK_EQ(s.order[5], std::size_t{1});
  CHECK_EQ(s.order[6], std::size_t{4});
  // Closing one index leaves the others' relative order unchanged (counter-based g).
  std::vector<double> closed = logits;
  closed[2] = -kInf;
  const gumbel::Sample t = gumbel::sample_order(closed, 99);
  std::vector<std::size_t> before, after;
  for (const std::size_t i : s.order) {
    if (i != 2 && std::isfinite(logits[i])) before.push_back(i);
  }
  for (const std::size_t i : t.order) {
    if (std::isfinite(closed[i])) after.push_back(i);
  }
  CHECK(before == after);
  for (const double bad : {std::nan(""), kInf}) {
    std::vector<double> l = logits;
    l[0] = bad;
    CHECK_THROWS_AS(gumbel::sample_order(l, 1), std::invalid_argument);
  }
}

TEST(gumbel_top_m_samples_without_replacement_from_the_softmax) {
  // The Gumbel-top-k trick: the first of the order is a sample from the
  // softmax, the first k a sample of k without replacement.
  const std::vector<double> p{0.5, 0.3, 0.15, 0.05};
  std::vector<double> logits;
  for (const double x : p) logits.push_back(std::log(x));
  constexpr std::size_t kSeeds = 40'000;
  std::vector<double> first(p.size(), 0.0);
  double starts_01 = 0, three_in_two = 0;
  for (std::uint64_t seed = 1; seed <= kSeeds; ++seed) {
    const gumbel::Sample s = gumbel::sample_order(logits, gumbel::mix64(seed));
    first[s.order[0]] += 1.0;
    starts_01 += s.order[0] == 0 && s.order[1] == 1 ? 1.0 : 0.0;
    three_in_two += s.order[0] == 3 || s.order[1] == 3 ? 1.0 : 0.0;
  }
  const double seeds = static_cast<double>(kSeeds);
  for (std::size_t i = 0; i < p.size(); ++i) CHECK(std::abs(first[i] / seeds - p[i]) < 0.01);
  CHECK(std::abs(starts_01 / seeds - 0.30) < 0.01);
  // P(3 among the first two) = 0.05 + 0.5 * 0.05/0.5 + 0.3 * 0.05/0.7 + 0.15 * 0.05/0.85.
  CHECK(std::abs(three_in_two / seeds - 0.1302521) < 0.01);
}

TEST(gumbel_halving_schedule_matches_hand_counts) {
  struct Case {
    std::size_t m;
    std::uint64_t n;
    std::vector<gumbel::Phase> want;
  };
  const std::vector<Case> cases{
      {16, 200, {{16, 3}, {8, 6}, {4, 12}, {2, 25}}},
      {16, 800, {{16, 12}, {8, 25}, {4, 50}, {2, 100}}},
      {16, 99, {{16, 1}, {8, 3}, {4, 6}, {2, 12}}},
      {16, 100, {{16, 1}, {8, 3}, {4, 6}, {2, 12}}},
      {16, 799, {{16, 12}, {8, 24}, {4, 49}, {2, 99}}},
      {20, 200, {{20, 2}, {10, 4}, {5, 8}, {3, 13}, {2, 20}}},
      {5, 50, {{5, 3}, {3, 5}, {2, 8}}},
      {3, 10, {{3, 1}, {2, 2}}},
      {4, 200, {{4, 25}, {2, 50}}},
      {2, 2, {{2, 1}}},
      {1, 7, {{1, 7}}},
      {16, 16, {{16, 1}, {8, 1}, {4, 1}, {2, 2}}},
  };
  for (const Case& c : cases) {
    const std::vector<gumbel::Phase> got = gumbel::halving_schedule(c.m, c.n);
    REQUIRE_EQ(got.size(), c.want.size());
    for (std::size_t r = 0; r < got.size(); ++r) {
      CHECK_EQ(got[r].considered, c.want[r].considered);
      CHECK_EQ(got[r].visits, c.want[r].visits);
    }
  }
  for (std::size_t m = 1; m <= 64; ++m) {
    std::size_t phases = 0;
    while ((std::size_t{1} << phases) < m) ++phases;
    for (std::uint64_t n = 1; n <= 2000; ++n) {
      const std::vector<gumbel::Phase> s = gumbel::halving_schedule(m, n);
      CHECK_EQ(s.size(), m == 1 ? std::size_t{1} : phases);
      CHECK_EQ(s.front().considered, m);
      std::uint64_t total = 0;
      for (std::size_t r = 0; r < s.size(); ++r) {
        CHECK(s[r].visits >= 1);
        if (r + 1 < s.size()) CHECK_EQ(s[r + 1].considered, (s[r].considered + 1) / 2);
        total += s[r].considered * s[r].visits;
      }
      if (m >= 2) CHECK_EQ(s.back().considered, std::size_t{2});
      if (n >= static_cast<std::uint64_t>(std::max<std::size_t>(phases, 1) * m)) CHECK(total <= n);
    }
  }
  CHECK_THROWS_AS(gumbel::halving_schedule(0, 5), std::invalid_argument);
  CHECK_THROWS_AS(gumbel::halving_schedule(5, 0), std::invalid_argument);
}

TEST(gumbel_sigma_and_mixed_value_are_the_target_sides) {
  // Danihelka et al.'s v_mix and sigma, spec §16's I21 form (q mapped onto
  // [0, 1]); python/t5d/train/completed_q.py pins the same numbers.
  std::vector<gumbel::Action> a(3);
  a[0] = {0.5, 3, 0, 0.2, true};
  a[1] = {0.3, 1, 0, -0.4, true};
  a[2] = {0.2, 0, 0, 0.0, true};
  const double v_mix = gumbel::mixed_value(a, 0.1);
  CHECK(std::abs(v_mix) < 1e-12);  // (0.1 + 4 * (0.5 * 0.2 + 0.3 * -0.4) / 0.8) / 5
  CHECK(std::abs(gumbel::sigma(0.2, 3, 50, 1) - 31.8) < 1e-12);
  CHECK(std::abs(gumbel::sigma(-0.4, 3, 50, 1) - 15.9) < 1e-12);
  CHECK(std::abs(gumbel::sigma(v_mix, 3, 50, 1) - 26.5) < 1e-12);
  CHECK(std::abs(gumbel::sigma(0.2, 3, 50, 2) - 63.6) < 1e-12);  // doubles with c_scale
  CHECK(gumbel::sigma(0.3, 3, 50, 1) > gumbel::sigma(0.2, 3, 50, 1));
  // Nothing visited: v_net. A closed action is left out of every sum.
  std::vector<gumbel::Action> none = a;
  for (gumbel::Action& x : none) x.visits = 0;
  CHECK_EQ(gumbel::mixed_value(none, 0.37), 0.37);
  std::vector<gumbel::Action> closed = a;
  closed[1].open = false;  // only action 0 visited: (0.1 + 3 * 0.2) / 4
  CHECK(std::abs(gumbel::mixed_value(closed, 0.1) - 0.175) < 1e-12);
}

TEST(gumbel_root_visits_survivors_equally_and_halves) {
  // Sequential halving as a bandit: 20 open actions, m 16, n 200 - phases of
  // 3, 6, 12 and 25 visits; at each halving the kept half is the test's own
  // top half by g + ln prior + sigma(q); the 6 visits the rounding leaves
  // over go on the last two equally; the 4 unsampled actions get none.
  std::vector<double> logits, q;
  for (std::size_t i = 0; i < 20; ++i) {
    logits.push_back(-0.15 * static_cast<double>(i));
    q.push_back(std::sin(1.7 * static_cast<double>(i)));
  }
  Bandit b = bandit(logits, q, 2024);
  gumbel::Root root = gumbel::Root::make(b.g, open_prefix(b), gumbel::Config{16, 200, 50.0, 1.0});
  REQUIRE(root.active());
  CHECK_EQ(root.m(), std::size_t{16});
  CHECK_EQ(root.shown(), std::size_t{16});
  const std::vector<std::uint64_t> targets{3, 9, 21, 46};
  int halvings = 0;
  for (int k = 0; k < 200; ++k) {
    const std::size_t phase = root.phase();
    const std::vector<std::size_t> members = root.considered();
    std::vector<std::pair<double, std::size_t>> expected;
    for (const std::size_t e : members) expected.emplace_back(own_score(b, e, 0.0, 50, 1), e);
    const std::size_t e = root.next(b.actions, 0.0);
    REQUIRE(e < b.actions.size());
    if (root.phase() != phase) {
      ++halvings;
      for (const std::size_t m : members) CHECK_EQ(std::uint64_t{b.actions[m].visits}, targets[phase]);
      std::sort(expected.begin(), expected.end(),
                [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
      std::vector<std::size_t> kept;
      for (std::size_t i = 0; i < root.schedule()[root.phase()].considered; ++i) kept.push_back(expected[i].second);
      std::sort(kept.begin(), kept.end());
      CHECK(kept == root.considered());
    }
    visit(b, e);
  }
  CHECK_EQ(halvings, 3);
  std::map<std::uint32_t, int> counts;
  for (const gumbel::Action& a : b.actions) ++counts[a.visits];
  CHECK_EQ(counts[0], 4);
  CHECK_EQ(counts[3], 8);
  CHECK_EQ(counts[9], 4);
  CHECK_EQ(counts[21], 2);
  CHECK_EQ(counts[49], 2);
  for (std::size_t e = 16; e < 20; ++e) CHECK_EQ(b.actions[e].visits, std::uint32_t{0});
  for (int k = 0; k < 10; ++k) visit(b, root.next(b.actions, 0.0));
  for (const std::size_t e : root.considered()) CHECK_EQ(b.actions[e].visits, std::uint32_t{54});
  // The move to play: the higher-scored of the last two.
  const std::vector<std::size_t> last = root.considered();
  REQUIRE_EQ(last.size(), std::size_t{2});
  const std::size_t best = root.best(b.actions, 0.0);
  const std::size_t other = best == last[0] ? last[1] : last[0];
  CHECK(best == last[0] || best == last[1]);
  CHECK(own_score(b, best, 0.0, 50, 1) >= own_score(b, other, 0.0, 50, 1));
}

TEST(gumbel_root_waits_for_pending_visits) {
  // m 4, n 8: phases of 1 visit for four, then 2 for two.
  std::vector<double> logits{0.0, -0.1, -0.2, -0.3}, q{0.1, 0.2, -0.3, 0.4};
  Bandit b = bandit(logits, q, 7);
  gumbel::Root root = gumbel::Root::make(b.g, 4, gumbel::Config{4, 8, 50.0, 1.0});
  REQUIRE_EQ(root.schedule().size(), std::size_t{2});
  for (std::size_t k = 0; k < 4; ++k) {
    const std::size_t e = root.next(b.actions, 0.0);
    REQUIRE(e < 4);
    CHECK_EQ(b.actions[e].pending, std::uint32_t{0});
    ++b.actions[e].pending;  // in flight
  }
  CHECK_EQ(root.next(b.actions, 0.0), gumbel::kWait);  // every visit of the phase pending
  for (std::size_t e = 0; e < 3; ++e) {  // three arrive
    --b.actions[e].pending;
    visit(b, e);
  }
  CHECK_EQ(root.next(b.actions, 0.0), gumbel::kWait);
  CHECK_EQ(root.phase(), std::size_t{0});
  --b.actions[3].pending;  // released uncounted (a collision): picked again
  CHECK_EQ(root.next(b.actions, 0.0), std::size_t{3});
  visit(b, 3);
  const std::size_t e = root.next(b.actions, 0.0);  // the phase ends: the last one begins
  CHECK_EQ(root.phase(), std::size_t{1});
  REQUIRE(e < 4);
  ++b.actions[e].pending;
  const std::size_t f = root.next(b.actions, 0.0);
  ++b.actions[f].pending;
  const std::size_t h = root.next(b.actions, 0.0);
  CHECK(h != gumbel::kWait);  // the last phase never waits
  CHECK(h < 4);
}

TEST(gumbel_root_refills_in_the_first_phase_and_drops_later) {
  std::vector<double> logits, q;
  for (std::size_t i = 0; i < 8; ++i) {
    logits.push_back(-0.2 * static_cast<double>(i));
    q.push_back(0.1 * static_cast<double>(i % 3));
  }
  Bandit b = bandit(logits, q, 11);
  gumbel::Root root = gumbel::Root::make(b.g, 8, gumbel::Config{4, 40, 50.0, 1.0});
  REQUIRE_EQ(root.shown(), std::size_t{4});
  // Phase 0: candidate 1 closes (found illegal at its first visit) - edge 4 takes its place.
  b.actions[1].open = false;
  const std::size_t e = root.next(b.actions, 0.0);
  CHECK(e < 8);
  CHECK_EQ(root.shown(), std::size_t{5});
  CHECK((root.considered() == std::vector<std::size_t>{0, 2, 3, 4}));
  // Through the first phase (5 visits each at m 4, n 40), then a member closes: only dropped.
  visit(b, e);
  while (root.phase() == 0) visit(b, root.next(b.actions, 0.0));
  REQUIRE_EQ(root.considered().size(), std::size_t{2});
  const std::size_t gone = root.considered().front();
  b.actions[gone].open = false;
  const std::size_t f = root.next(b.actions, 0.0);
  CHECK_EQ(root.considered().size(), std::size_t{1});
  CHECK(f != gone);
  CHECK_EQ(root.shown(), std::size_t{5});
  // Every member closed: the open edges already shown.
  b.actions[f].open = false;
  const std::size_t h = root.next(b.actions, 0.0);
  CHECK(h < root.shown() && b.actions[h].open);
  // No open edge at all: kNone.
  for (gumbel::Action& a : b.actions) a.open = false;
  CHECK_EQ(root.next(b.actions, 0.0), gumbel::kNone);
  CHECK_EQ(root.best(b.actions, 0.0), gumbel::kNone);
  // No open edge at the draw: an inactive Root.
  CHECK(!gumbel::Root::make({0.1, 0.2}, 0, gumbel::Config{16, 10, 50.0, 1.0}).active());
  CHECK(!gumbel::Root().active());
}

TEST(gumbel_choice_improves_on_the_prior_in_expectation) {
  // Five actions whose prior favours the worse ones: the halving's choice has
  // a higher mean q than a sample from the prior - and with c_scale 0 it is
  // one (the survivor is the first of the Gumbel order).
  const std::vector<double> p{0.1, 0.4, 0.3, 0.15, 0.05};
  const std::vector<double> q{0.9, -0.5, -0.5, 0.0, -0.9};
  std::vector<double> logits;
  for (const double x : p) logits.push_back(std::log(x));
  double prior_mean = 0;
  for (std::size_t i = 0; i < p.size(); ++i) prior_mean += p[i] * q[i];
  for (const double c_scale : {1.0, 0.0}) {
    constexpr std::size_t kSeeds = 20'000;
    double sum = 0, squares = 0;
    for (std::uint64_t seed = 1; seed <= kSeeds; ++seed) {
      Bandit b = bandit(logits, q, gumbel::mix64(seed));
      gumbel::Root root = gumbel::Root::make(b.g, 5, gumbel::Config{16, 50, 50.0, c_scale});
      for (int k = 0; k < 50; ++k) visit(b, root.next(b.actions, 0.0));
      const double chosen = b.q[root.best(b.actions, 0.0)];
      sum += chosen;
      squares += chosen * chosen;
    }
    const double seeds = static_cast<double>(kSeeds);
    const double mean = sum / seeds;
    const double se = std::sqrt((squares / seeds - mean * mean) / seeds);
    std::fprintf(stderr, "  c_scale %.0f: mean q %.4f (se %.4f), the prior's %.4f\n", c_scale, mean, se, prior_mean);
    if (c_scale > 0) {
      CHECK(mean > prior_mean + 0.5);
    } else {
      CHECK(std::abs(mean - prior_mean) < 3 * se);
    }
  }
}

// ---- the search's arm (SearchParams::gumbel_root) -----------------------------------------------------

namespace {

GameState standard() { return variant_start(*find_variant("Standard")); }

// A Gumbel search's parameters: the arm on, what it refuses off, the ladder off.
mcts::SearchParams gumbel_params(std::uint64_t simulations, std::uint32_t m = 16, std::uint64_t seed = 1) {
  mcts::SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.gumbel_root = true;
  p.gumbel_m = m;
  p.gumbel_simulations = simulations;
  p.asset_map = false;
  p.hunt = false;
  p.settling = false;
  p.seed = seed;
  return p;
}

std::vector<std::uint32_t> root_visits(const mcts::Search& s) {
  std::vector<std::uint32_t> v;
  for (const mcts::RootChild& c : s.root_children()) v.push_back(c.visits);
  std::sort(v.begin(), v.end());
  return v;
}

std::vector<std::uint32_t> multiset(std::initializer_list<std::pair<int, int>> counts) {
  std::vector<std::uint32_t> v;
  for (const auto& [visits, n] : counts) v.insert(v.end(), static_cast<std::size_t>(n), static_cast<std::uint32_t>(visits));
  std::sort(v.begin(), v.end());
  return v;
}

// Uniform logits; value +0.8 for White where timeline 0's head board has a White pawn on e4, -0.2
// otherwise - from the leaf's acting player's view.
class Planted final : public mcts::Evaluator {
 public:
  void evaluate(std::span<const mcts::EvalRequest> batch, std::span<mcts::Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      out[i].logits.assign(batch[i].expansion->moves().size(), 0.0f);
      const Board* head = batch[i].state->multiverse().head(0);
      const Piece e4 = head != nullptr ? head->at(4, 3) : Piece{};
      const bool planted = !e4.empty() && e4.kind() == PieceKind::Pawn && e4.colour() == Colour::White;
      const double white = planted ? 0.8 : -0.2;
      const double v = batch[i].expansion->acting_player() == Colour::White ? white : -white;
      out[i].wdl = mcts::wdl_of(v);
      out[i].value = mcts::two_hot(v);
    }
  }
};

// Fixed unequal logits (by candidate index) and equal values.
class Tilted final : public mcts::Evaluator {
 public:
  void evaluate(std::span<const mcts::EvalRequest> batch, std::span<mcts::Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const std::size_t n = batch[i].expansion->moves().size();
      out[i].logits.resize(n);
      for (std::size_t m = 0; m < n; ++m) out[i].logits[m] = -0.15f * static_cast<float>(m % 11);
      out[i].wdl = mcts::wdl_of(0.0);
      out[i].value = mcts::two_hot(0.0);
    }
  }
};

}  // namespace

TEST(mcts_gumbel_is_off_by_default_and_inert) {
  const mcts::SearchParams d{};
  CHECK(!d.gumbel_root);
  CHECK_EQ(d.gumbel_m, std::uint32_t{16});
  CHECK_EQ(d.gumbel_simulations, std::uint64_t{0});
  CHECK_EQ(d.gumbel_c_visit, 50.0);
  CHECK_EQ(d.gumbel_c_scale, 1.0);
  // Off, the other four change nothing: the same tree (digest) under every split.
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, 3);
  for (const mcts::TurnSplit split : {mcts::TurnSplit::BoardMove, mcts::TurnSplit::Tree, mcts::TurnSplit::Flat}) {
    mcts::SearchParams a;
    a.expand.square_features = SquareFeatures::None;
    a.split = split;
    a.hunt = a.settling = false;
    mcts::SearchParams b = a;
    b.gumbel_m = 3;
    b.gumbel_simulations = 77;
    b.gumbel_c_visit = 10;
    b.gumbel_c_scale = 2;
    mcts::Search x(standard(), a);
    mcts::Search y(standard(), b);
    x.run(*ev, mcts::Limits{64});
    y.run(*ev, mcts::Limits{64});
    CHECK_EQ(x.digest(), y.digest());
    CHECK(!y.gumbel_view().drawn);
    CHECK_THROWS_AS(y.choose_gumbel(), std::invalid_argument);
  }
}

TEST(mcts_gumbel_refusals_name_their_parameter) {
  const auto refused = [](const mcts::SearchParams& p, const char* name) {
    try {
      mcts::Search s(standard(), p);
    } catch (const std::invalid_argument& e) {
      if (std::string(e.what()).find(name) != std::string::npos) return true;
      std::fprintf(stderr, "  refused for another reason: %s\n", e.what());
      return false;
    }
    return false;
  };
  mcts::SearchParams p = gumbel_params(64);
  p.split = mcts::TurnSplit::Flat;
  CHECK(refused(p, "flat form"));
  p = gumbel_params(64);
  p.root_noise = true;
  CHECK(refused(p, "root_noise"));
  p = gumbel_params(64);
  p.root_policy_temperature = 1.25;
  CHECK(refused(p, "root_policy_temperature"));
  p = gumbel_params(64);
  p.asset_map = true;  // the default
  CHECK(refused(p, "asset_map"));
  p = gumbel_params(64);
  p.hunt_suggest = true;
  CHECK(refused(p, "hunt_suggest"));
  p = gumbel_params(64);
  p.proof_channel = true;
  CHECK(refused(p, "proof_channel"));
  p = gumbel_params(0);
  CHECK(refused(p, "gumbel_simulations"));
  p = gumbel_params(64, 0);
  CHECK(refused(p, "gumbel_m"));
  for (const double bad : {-1.0, std::nan("")}) {
    p = gumbel_params(64);
    p.gumbel_c_visit = bad;
    CHECK(refused(p, "gumbel_c_visit"));
    p = gumbel_params(64);
    p.gumbel_c_scale = bad;
    CHECK(refused(p, "gumbel_c_scale"));
  }
  // On a Gumbel search: match play's choice and the re-roots are refused, nothing changed.
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search s(standard(), gumbel_params(63));
  s.run(*ev, mcts::Limits{64});
  const std::uint64_t digest = s.digest();
  CHECK_THROWS_AS(s.choose(), std::invalid_argument);
  CHECK_THROWS_AS(s.choose_by_visits(0.9, 0.5), std::invalid_argument);
  CHECK_NOTHROW(s.choose_by_visits(0.0, 0.5));
  CHECK_THROWS_AS(s.advance(0), std::invalid_argument);
  const mcts::RootChild c = s.root_children().at(0);
  GameState next = standard();
  next.apply(c.moves.at(0));
  CHECK_THROWS_AS(s.advance_to(next, 1), std::invalid_argument);
  CHECK_THROWS_AS(s.advance_moves(c.moves), std::invalid_argument);
  CHECK_THROWS_AS(s.advance_action(c.moves), std::invalid_argument);
  CHECK_EQ(s.digest(), digest);
}

TEST(mcts_gumbel_root_children_are_the_sample_and_the_visits_the_schedules) {
  // The Standard start (20 moves), m 16, n 200 (Limits 201: the root's own evaluation is a
  // playout): the root's children are the sample - 16 of them, in the order of g + logit - and
  // their visits the schedule's: 3 for the eight dropped first, 9 for four, 21 for two, and 46 for
  // the last two plus the 6 the rounding leaves over (49).
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, 5);
  mcts::Search s(standard(), gumbel_params(200));
  s.run(*ev, mcts::Limits{201});
  const std::vector<mcts::RootChild> children = s.root_children();
  CHECK_EQ(children.size(), std::size_t{16});
  CHECK_EQ(s.node(s.root()).instantiated, std::size_t{16});
  CHECK_EQ(s.node(s.root()).visits, std::uint32_t{201});
  const gumbel::RootView view = s.gumbel_view();
  REQUIRE(view.drawn);
  REQUIRE_EQ(view.rows.size(), children.size());
  for (std::size_t i = 0; i + 1 < view.rows.size(); ++i) {
    CHECK(view.rows[i].g + view.rows[i].logit >= view.rows[i + 1].g + view.rows[i + 1].logit);
  }
  CHECK(root_visits(s) == multiset({{3, 8}, {9, 4}, {21, 2}, {49, 2}}));
  for (std::size_t i = children.size(); i < s.node(s.root()).edge_count; ++i) {  // outside the sample
    const mcts::EdgeView e = s.edge(s.root(), i);
    CHECK_EQ(e.visits, std::uint32_t{0});
    CHECK(e.child == mcts::kNoNode);
  }
  CHECK_NOTHROW(s.check_invariants());
  // The same seed: the same sample and tree; another seed: another sample.
  mcts::Search again(standard(), gumbel_params(200));
  again.run(*ev, mcts::Limits{201});
  CHECK_EQ(again.digest(), s.digest());
  mcts::Search other(standard(), gumbel_params(200, 16, 2));
  other.run(*ev, mcts::Limits{201});
  bool differs = false;
  const std::vector<mcts::RootChild> b = other.root_children();
  for (std::size_t i = 0; i < b.size(); ++i) differs = differs || !(b[i].moves == children[i].moves);
  CHECK(differs);
  // m 4: two phases, 25 visits for four, then 50 more for two.
  mcts::Search four(standard(), gumbel_params(200, 4));
  four.run(*ev, mcts::Limits{201});
  CHECK(root_visits(four) == multiset({{25, 2}, {75, 2}}));
}

TEST(mcts_gumbel_choice_is_the_argmax_over_the_survivors) {
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, 5);
  mcts::Search s(standard(), gumbel_params(200));
  s.run(*ev, mcts::Limits{201});
  const std::size_t chosen = s.choose_gumbel();
  const gumbel::RootView view = s.gumbel_view();
  const std::vector<mcts::RootChild> children = s.root_children();
  REQUIRE(chosen < view.rows.size());
  CHECK(view.rows[chosen].considered);
  std::uint32_t most = 0;
  for (const mcts::RootChild& c : children) most = std::max(most, c.visits);
  for (std::size_t i = 0; i < children.size(); ++i) {
    // The score from root_children() alone (every survivor visited here).
    if (children[i].visits == 0) continue;
    const double own = view.rows[i].g + std::log(children[i].prior) + (50.0 + most) * (children[i].q + 1.0) / 2.0;
    CHECK(std::abs(own - view.rows[i].score) < 1e-9);
    if (view.rows[i].considered) CHECK(view.rows[chosen].score >= view.rows[i].score);
  }
  // Cut at 4 simulations: unvisited survivors take v_mix - mixed_value over root_children() and
  // the root's own evaluation.
  mcts::Search cut(standard(), gumbel_params(200));
  cut.run(*ev, mcts::Limits{5});
  const gumbel::RootView v = cut.gumbel_view();
  std::vector<gumbel::Action> actions;
  for (const mcts::RootChild& c : cut.root_children()) actions.push_back({c.prior, c.visits, 0, c.q, true});
  const mcts::NodeView root = cut.node(cut.root());
  CHECK(std::abs(v.v_mix - gumbel::mixed_value(actions, root.eval_mean)) < 1e-12);
  for (std::size_t i = 0; i < v.rows.size(); ++i) {
    if (actions[i].visits == 0) CHECK_EQ(v.rows[i].completed_q, v.v_mix);
  }
  CHECK_NOTHROW(cut.choose_gumbel());
}

TEST(mcts_gumbel_finds_a_planted_move) {
  // e2-e4 is worth +0.8 to White, every other move -0.2; m 32 puts all 20 moves in the sample.
  Planted ev;
  const GameState s = standard();
  for (std::uint64_t seed = 1; seed <= 20; ++seed) {
    mcts::Search search(s, gumbel_params(200, 32, seed));
    search.run(ev, mcts::Limits{201});
    const std::vector<mcts::RootChild> children = search.root_children();
    const std::size_t chosen = search.choose_gumbel();
    const Move& m = children.at(chosen).moves.at(0);
    CHECK_EQ(int{m.from_square}, make_square(4, 1));
    CHECK_EQ(int{m.to_square}, make_square(4, 3));
    std::uint32_t most = 0;
    for (const mcts::RootChild& c : children) most = std::max(most, c.visits);
    CHECK(children[chosen].visits + 1 >= most);  // a finalist: the last two split 41 visits
  }
}

TEST(mcts_gumbel_with_one_simulation_plays_a_sample_of_the_prior) {
  // n 1: one candidate, the first of the Gumbel order - a sample from the root's prior. Over 400
  // seeds the three likeliest moves are chosen at their prior's rate, within 5 standard errors.
  Tilted ev;
  const GameState s = standard();
  mcts::SearchParams plain = gumbel_params(1);
  plain.gumbel_root = false;
  plain.widening = false;
  mcts::Search reference(s, plain);
  reference.run(ev, mcts::Limits{1});
  std::vector<mcts::RootChild> prior = reference.root_children();  // in prior order
  REQUIRE(prior.size() >= 3);
  std::vector<int> chosen(3, 0);
  constexpr std::uint64_t kSeeds = 400;
  const double seeds = static_cast<double>(kSeeds);
  for (std::uint64_t seed = 1; seed <= kSeeds; ++seed) {
    mcts::Search search(s, gumbel_params(1, 16, seed));
    search.run(ev, mcts::Limits{2});
    const std::vector<mcts::RootChild> children = search.root_children();
    CHECK_EQ(children.size(), std::size_t{1});
    const Move& m = children.at(search.choose_gumbel()).moves.at(0);
    for (std::size_t k = 0; k < 3; ++k) chosen[k] += m == prior[k].moves.at(0) ? 1 : 0;
  }
  for (std::size_t k = 0; k < 3; ++k) {
    const double p = prior[k].prior;
    const double rate = static_cast<double>(chosen[k]) / seeds;
    const double se = std::sqrt(p * (1 - p) / seeds);
    std::fprintf(stderr, "  move %zu: prior %.4f, chosen %.4f\n", k, p, rate);
    CHECK(std::abs(rate - p) < 5 * se);
  }
}

TEST(mcts_gumbel_verifies_unknown_candidates_at_their_first_visit) {
  // The White king in check from a rook on a1 (d1 and f1 illegal, d2, e2, f2 legal), no
  // legality budget in the expansion: candidates are verified at their first visit, an illegal
  // one removed and - in the first phase - its place given to the next of the sample. m 2: every
  // seed visits two legal moves; some seed's sample held an illegal one.
  const GameState s = t5d_test::game({"[k7/8/8/8/8/8/8/r3K3:0:1:w]"}, BoardSize{8, 8});
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  {
    // The precondition: the fast classification leaves an illegal move Unknown.
    MoveList listed;
    generate_moves(s, listed);
    const std::vector<Move> moves(listed.begin(), listed.end());
    std::vector<MoveLegality> fast;
    std::vector<MoveLegality> exact;
    classify_moves(s, moves, fast, LegalityOptions::budgeted(0, 0));
    classify_moves(s, moves, exact, LegalityOptions::exact());
    int hidden = 0;
    for (std::size_t i = 0; i < moves.size(); ++i) {
      if (fast[i] == MoveLegality::Unknown && exact[i] == MoveLegality::Illegal) ++hidden;
    }
    std::fprintf(stderr, "  %zu moves, %d illegal ones Unknown to the fast classification\n", moves.size(), hidden);
    REQUIRE(hidden > 0);
  }
  int replaced = 0;
  for (std::uint64_t seed = 1; seed <= 12; ++seed) {
    mcts::SearchParams p = gumbel_params(20, 2, seed);
    p.legality = LegalityOptions::budgeted(0, 0);
    mcts::Search search(s, p);
    search.run(*ev, mcts::Limits{21});
    int visited = 0;
    for (const mcts::RootChild& c : search.root_children()) {
      if (c.visits == 0) continue;
      ++visited;
      std::vector<MoveLegality> out;
      classify_moves(s, c.moves, out, LegalityOptions::exact());
      CHECK(out.at(0) == MoveLegality::Legal);
    }
    CHECK_EQ(visited, 2);
    replaced += search.stats().verified_illegal > 0 ? 1 : 0;
    CHECK_NOTHROW(search.check_invariants());
  }
  std::fprintf(stderr, "  %d of 12 samples held an illegal move, replaced\n", replaced);
  CHECK(replaced >= 1);
}

TEST(mcts_gumbel_batched_and_synchronous_allocate_alike) {
  // The schedule is exact either way: the halving waits for its phase's pending visits, ending
  // the batch short (at most once a phase) - the uniform evaluator, so the choice is the same too.
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::SearchParams batched = gumbel_params(200);
  batched.batch_size = 8;
  mcts::SearchParams sync = gumbel_params(200);
  sync.synchronous = true;
  mcts::Search a(standard(), batched);
  mcts::Search b(standard(), sync);
  a.run(*ev, mcts::Limits{201});
  b.run(*ev, mcts::Limits{201});
  CHECK(root_visits(a) == root_visits(b));
  CHECK(root_visits(a) == multiset({{3, 8}, {9, 4}, {21, 2}, {49, 2}}));
  CHECK_EQ(a.choose_gumbel(), b.choose_gumbel());
  CHECK_NOTHROW(a.check_invariants());
  const mcts::SearchStats& st = a.stats();
  const std::uint64_t full = (st.evaluations + 7) / 8;
  std::fprintf(stderr, "  batched: %llu batches for %llu evaluations, %llu collisions\n",
               static_cast<unsigned long long>(st.batches), static_cast<unsigned long long>(st.evaluations),
               static_cast<unsigned long long>(st.collisions));
  CHECK(st.batches >= full);
  CHECK(st.batches <= full + 4 + st.collisions + 1);  // four phases' waits
}

TEST(mcts_gumbel_under_the_tree_form_and_the_graph) {
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, 9);
  for (const bool graph : {false, true}) {
    for (const mcts::TurnSplit split : {mcts::TurnSplit::Tree, mcts::TurnSplit::BoardMove}) {
      if (!graph && split == mcts::TurnSplit::BoardMove) continue;
      mcts::SearchParams p = gumbel_params(200);
      p.split = split;
      p.graph = graph;
      mcts::Search s(standard(), p);
      s.run(*ev, mcts::Limits{201});
      CHECK(root_visits(s) == multiset({{3, 8}, {9, 4}, {21, 2}, {49, 2}}));
      CHECK_NOTHROW(s.check_invariants());
      CHECK(s.choose_gumbel() < s.root_children().size());
      bool widening = false;  // interior nodes keep PUCT and widening
      for (mcts::NodeId id = 0; id < s.node_count(); ++id) {
        const mcts::NodeView n = s.node(id);
        if (id != s.root() && n.expanded && n.instantiated < n.edge_count) widening = true;
      }
      CHECK(widening);
    }
  }
}

TEST(mcts_gumbel_one_move_roots_and_finished_roots) {
  // One legal move (Kxc2 the only one): m_eff 1, every visit to it, the choice it.
  const GameState one = t5d_test::game({"[3k/4/B1qK/4:0:1:w]"}, BoardSize{4, 4});
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search s(one, gumbel_params(20));
  s.run(*ev, mcts::Limits{21});
  REQUIRE_EQ(s.root_children().size(), std::size_t{1});
  CHECK_EQ(s.root_children()[0].visits, std::uint32_t{20});
  CHECK_EQ(s.choose_gumbel(), std::size_t{0});
  // A mated root: no draw, no choice.
  const GameState mated = t5d_test::game({"[8/8/8/8/8/5k2/6q1/7K:0:1:w]"}, BoardSize{8, 8});
  mcts::Search done(mated, gumbel_params(20));
  done.run(*ev, mcts::Limits{21});
  CHECK(!done.gumbel_view().drawn);
  CHECK_THROWS_AS(done.choose_gumbel(), std::invalid_argument);
}

TEST(mcts_gumbel_survives_every_allocation_failure) {
  // Issue 535's discipline at the draw: every allocation of the root's expansion (the first
  // delivery, where the sample is drawn) failing in turn - each fault thrown, the invariants
  // held, and the search run on to its cap draws the same sample and allocates the same visits.
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, 4);
  for (const mcts::TurnSplit split : {mcts::TurnSplit::BoardMove, mcts::TurnSplit::Tree}) {
    mcts::SearchParams p = gumbel_params(40);
    p.split = split;
    const auto first_batch = [&](mcts::Search& s) {
      const std::size_t n = s.gather(8);
      REQUIRE_EQ(n, std::size_t{1});
      std::vector<mcts::Evaluation> out(n);
      ev->evaluate(s.requests(), out);
      return out;
    };
    mcts::Search twin(standard(), p);
    twin.run(*ev, mcts::Limits{41});
    const gumbel::RootView want = twin.gumbel_view();
    long long total = 0;
    {
      mcts::Search s(standard(), p);
      const std::vector<mcts::Evaluation> out = first_batch(s);
      constexpr long long kFar = std::numeric_limits<long long>::max();
      const t5d_test::alloc::FailAfter far(kFar);
      s.deliver(out);
      total = kFar - t5d_test::alloc::fail_countdown;
    }
    std::size_t faults = 0, differ = 0, late = 0;
    for (long long k = 0; k < total; ++k) {
      mcts::Search s(standard(), p);
      const std::vector<mcts::Evaluation> out = first_batch(s);
      bool threw = false;
      bool fired = false;
      {
        const t5d_test::alloc::FailAfter fault(k);
        try {
          s.deliver(out);
        } catch (const std::bad_alloc&) {
          threw = true;
        }
        fired = t5d_test::alloc::fail_countdown < 0;
      }
      CHECK(threw || !fired);
      faults += threw ? 1u : 0u;
      CHECK_NOTHROW(s.check_invariants());
      // A fault after the expansion leaves the root's visit counted (the delivered leaf stays
      // delivered): the search is run on to the twin's 41 playouts in all.
      late += s.node(s.root()).expanded ? 1u : 0u;
      s.run(*ev, mcts::Limits{41 - s.stats().playouts});
      const gumbel::RootView got = s.gumbel_view();
      bool same = got.rows.size() == want.rows.size() && root_visits(s) == root_visits(twin);
      for (std::size_t i = 0; same && i < got.rows.size(); ++i) same = got.rows[i].g == want.rows[i].g;
      differ += same ? 0u : 1u;
    }
    std::fprintf(stderr, "  %s: %lld allocations in the root's delivery, %zu faults (%zu after the expansion), %zu "
                 "differ\n",
                 split == mcts::TurnSplit::Tree ? "tree form" : "board-move", total, faults, late, differ);
    CHECK(faults > 0);
    CHECK_EQ(differ, std::size_t{0});
  }
}

TEST(mcts_gumbel_is_refused_by_the_timed_decision_and_the_game_loop) {
  // Register E12: a timed decision is match play's, and the schedule is n simulations - refused at
  // TimeManager::begin (every timed decision's start: run_timed and t5d_timeman_begin), the
  // manager left as it was; and the C++ game loop does not drive the arm.
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search s(standard(), gumbel_params(63));
  mcts::TimeManager tm;
  mcts::TimeContext ctx;
  ctx.remaining_seconds = 60;
  CHECK_THROWS_AS(tm.begin(s, ctx), std::invalid_argument);
  CHECK_THROWS_AS(mcts::run_timed(s, *ev, tm, ctx), std::invalid_argument);
  CHECK_EQ(s.stats().playouts, std::uint64_t{0});
  mcts::Search plain(standard(), [] {
    mcts::SearchParams p = gumbel_params(63);
    p.gumbel_root = false;
    return p;
  }());
  CHECK_NOTHROW(tm.begin(plain, ctx));  // the same manager, untouched by the refusal
  mcts::PlayParams pp;
  pp.search = gumbel_params(63);
  CHECK_THROWS_AS(mcts::play_game(standard(), *ev, pp), std::invalid_argument);
}

// ---- the C API ---------------------------------------------------------------------------------------

TEST(capi_gumbel_fields_defaults_and_choice) {
  // engine ABI: five t5d_search_params fields in front of `expand` (still last), their defaults,
  // the layout the binding checks, t5d_search_choose_gumbel the C++ choice, and the refusals.
  t5d_search_params p;
  t5d_search_params_default(&p);
  CHECK_EQ(p.gumbel_root, 0);
  CHECK_EQ(p.gumbel_m, std::uint32_t{16});
  CHECK_EQ(p.gumbel_simulations, std::uint64_t{0});
  CHECK_EQ(p.gumbel_c_visit, 50.0);
  CHECK_EQ(p.gumbel_c_scale, 1.0);
  CHECK(offsetof(t5d_search_params, gumbel_c_scale) < offsetof(t5d_search_params, expand));
  for (const auto& [name, at] : {std::pair<const char*, std::size_t>{"gumbel_root", offsetof(t5d_search_params, gumbel_root)},
                                 {"gumbel_m", offsetof(t5d_search_params, gumbel_m)},
                                 {"gumbel_simulations", offsetof(t5d_search_params, gumbel_simulations)},
                                 {"gumbel_c_visit", offsetof(t5d_search_params, gumbel_c_visit)},
                                 {"gumbel_c_scale", offsetof(t5d_search_params, gumbel_c_scale)}}) {
    std::size_t got = 0;
    REQUIRE_EQ(t5d_search_layout("t5d_search_params", name, &got), T5D_OK);
    CHECK_EQ(got, at);
  }
  t5d_game* g = nullptr;
  REQUIRE_EQ(t5d_game_new_variant("Standard", T5D_RULES_STANDARD, &g, nullptr), T5D_OK);
  p.expand.square_features = T5D_FEATURES_NONE;
  p.gumbel_root = 1;
  p.gumbel_simulations = 63;
  p.hunt = 0;
  p.settling = 0;
  t5d_search* s = nullptr;
  char* error = nullptr;
  CHECK_EQ(t5d_search_new(g, 1, &p, &s, &error), T5D_ERR_ARGUMENT);  // asset_map on: refused, named
  CHECK(error != nullptr && std::string(error).find("asset_map") != std::string::npos);
  t5d_free(error);
  p.asset_map = 0;
  REQUIRE_EQ(t5d_search_new(g, 1, &p, &s, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_run(s, T5D_EVALUATOR_UNIFORM, 0, 64, nullptr), T5D_OK);
  std::size_t chosen = 99;
  REQUIRE_EQ(t5d_search_choose_gumbel(s, &chosen, nullptr), T5D_OK);
  mcts::SearchParams cpp = gumbel_params(63);
  mcts::Search twin(standard(), cpp);
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  twin.run(*ev, mcts::Limits{64});
  CHECK_EQ(chosen, twin.choose_gumbel());
  std::uint64_t digest = 0;
  REQUIRE_EQ(t5d_search_digest(s, &digest), T5D_OK);
  CHECK_EQ(digest, twin.digest());  // the same search through the C API
  std::size_t index = 0;
  int32_t reason = 0;
  CHECK_EQ(t5d_search_choose(s, &index, &reason, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_advance(s, 0, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_choose_gumbel(nullptr, &chosen, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_choose_gumbel(s, nullptr, nullptr), T5D_ERR_ARGUMENT);
  t5d_search_destroy(s);
  t5d_game_destroy(g);
}
