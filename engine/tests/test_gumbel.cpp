// Register E12 (Gumbel root search; t5d/gumbel.hpp): the pure parts - the
// variate, the Gumbel-top-k order, the halving schedule, sigma and the mixed
// value, the halving driven as a bandit - then the search's arm behind
// SearchParams::gumbel_root, and its C API.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "t5d/gumbel.hpp"
#include "test_framework.hpp"

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
