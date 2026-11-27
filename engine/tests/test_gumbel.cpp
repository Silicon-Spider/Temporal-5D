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

// ---- the pure parts' contracts, pinned (the mutation pass of 2026-11-27, TP-1) ------------------------

TEST(gumbel_stream_and_variate_are_the_headers_pinned_forms) {
  // G07-G11 (the stream's mixing, salt and XOR; the counter's index and odd constant): a record
  // derives an edge's g again from (search seed, root key, index) by gumbel.hpp's PINNED FORMS, so
  // the forms are pinned by value. The expected values are the header's formula written out again
  // in Python (python/tests/engine/test_gumbel_root.py, test_the_gumbel_streams_pinned_values,
  // checks the same numbers from the formula alone).
  CHECK_EQ(gumbel::stream_seed(1, 2), std::uint64_t{0xe34aec87d7dc0d1c});
  CHECK_EQ(gumbel::stream_seed(0, 0), std::uint64_t{0x0bfa9490a6f51b18});
  CHECK_EQ(gumbel::stream_seed(12345, 0xDEADBEEFCAFEF00Dull), std::uint64_t{0x814f7a0f7d709b5f});
  CHECK_EQ(gumbel::stream_seed(7, 0x0123456789ABCDEFull), std::uint64_t{0xaa6895ffc57d779c});
  const std::uint64_t a = gumbel::stream_seed(1, 2);
  CHECK(std::abs(gumbel::variate(a, 0) - 1.5411658734337765) < 1e-12);
  CHECK(std::abs(gumbel::variate(a, 5) - 1.5874060407463333) < 1e-12);
  const std::uint64_t b = gumbel::stream_seed(0, 0);
  CHECK(std::abs(gumbel::variate(b, 0) - 2.108013627453106) < 1e-12);
  CHECK(std::abs(gumbel::variate(b, 5) - 5.611003628944592) < 1e-12);
  const std::uint64_t c = gumbel::stream_seed(12345, 0xDEADBEEFCAFEF00Dull);
  CHECK(std::abs(gumbel::variate(c, 0) - -0.8048660621588094) < 1e-12);
  CHECK(std::abs(gumbel::variate(c, 5) - -1.3173491397342962) < 1e-12);
  // A sample's g, by index: the same numbers, whatever the logits.
  const std::uint64_t d = gumbel::stream_seed(7, 0x0123456789ABCDEFull);
  std::vector<double> logits(20, 0.0);
  logits[3] = -kInf;  // closed: its g is drawn all the same
  const gumbel::Sample s = gumbel::sample_order(logits, d);
  const double want[] = {-0.3186555669931809, -0.07414048585757446, -0.6194684835736681};
  for (std::size_t i = 0; i < 3; ++i) CHECK(std::abs(s.g[i] - want[i]) < 1e-12);
  CHECK(std::abs(s.g[19] - 0.7140964444518106) < 1e-12);
}

TEST(gumbel_sample_order_ties_go_to_the_lower_index) {
  // G13: an exact tie of g + logit (built from the stream's own variates) goes to the lower index.
  int ties = 0;
  for (std::uint64_t stream = 1; stream < 2000 && ties < 5; ++stream) {
    const double g0 = gumbel::variate(stream, 0), g1 = gumbel::variate(stream, 1);
    const double l1 = g0 - g1;
    if (g1 + l1 != g0) continue;
    ++ties;
    const gumbel::Sample s = gumbel::sample_order(std::vector<double>{0.0, l1}, stream);
    CHECK_EQ(s.order[0], std::size_t{0});
    CHECK_EQ(s.order[1], std::size_t{1});
  }
  REQUIRE(ties > 0);
}

TEST(gumbel_mixed_value_with_no_prior_mass_is_v_net) {
  // G29: v_mix is v_net where the visited actions carry no prior mass (the header's contract).
  std::vector<gumbel::Action> a(2);
  a[0] = {0.0, 3, 0, 0.5, true};
  a[1] = {0.0, 0, 0, 0.0, true};
  CHECK_EQ(gumbel::mixed_value(a, 0.25), 0.25);
}

TEST(gumbel_sigma_scale_counts_open_actions_only) {
  // G33: sigma's max_b N(b) is over the open actions only.
  gumbel::Root root = gumbel::Root::make({0.0, 5.0, 0.0}, 3, gumbel::Config{3, 30, 50.0, 1.0});
  REQUIRE(root.active());
  std::vector<gumbel::Action> a(3);
  a[0] = {0.5, 1, 0, 0.2, true};
  a[1] = {0.5, 1, 0, 0.1, true};
  a[2] = {0.5, 100, 0, 0.0, false};  // closed, the most visits
  // Open max 1: edge 0 scores ln .5 + 51 x 0.6 = 30.6 + ln .5, edge 1 5 + 51 x 0.55 = 33.05 + ln .5:
  // edge 1. With the closed 100 counted: 90 against 87.5 - edge 0.
  CHECK_EQ(root.best(a, 0.0), std::size_t{1});
}

TEST(gumbel_best_scores_an_unvisited_candidate_by_v_mix) {
  // G38, G59: an unvisited candidate's completed q is v_mix - not its q field, not v_net.
  gumbel::Root root = gumbel::Root::make({0.0, 40.0}, 2, gumbel::Config{2, 10, 50.0, 1.0});
  REQUIRE(root.active());
  std::vector<gumbel::Action> a(2);
  a[0] = {0.5, 1, 0, 1.0, true};   // visited once, q 1
  a[1] = {0.5, 0, 0, -1.0, true};  // unvisited: its q is not read
  // v_net -1: v_mix (-1 + 1 x 1) / 2 = 0. Edge 0 scores ln .5 + 51; edge 1 40 + ln .5 + 25.5: edge 1.
  // Scored by q -1 or by v_net -1, edge 1 has 40 + ln .5: edge 0.
  CHECK_EQ(root.best(a, -1.0), std::size_t{1});
}

TEST(gumbel_next_ties_go_to_the_first_candidate) {
  // G48: the candidate with the fewest visits and pending ones - ties: the first.
  gumbel::Root root = gumbel::Root::make({0.3, 0.2, 0.1, 0.0}, 4, gumbel::Config{4, 8, 50.0, 1.0});
  std::vector<gumbel::Action> a(4, gumbel::Action{0.25, 0, 0, 0.0, true});
  for (std::size_t want = 0; want < 4; ++want) {
    const std::size_t e = root.next(a, 0.0);
    CHECK_EQ(e, want);
    if (e < a.size()) ++a[e].pending;
  }
}

TEST(gumbel_with_every_candidate_closed_the_shown_then_the_next_open_edges_are_taken) {
  // G54, G55, G57: every member closed after the first phase - best() falls back to the open shown
  // edges; next() takes the open shown edges (showing nothing new), and only with none of those
  // the next edge of the order.
  std::vector<double> g{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
  gumbel::Root root = gumbel::Root::make(g, 8, gumbel::Config{4, 40, 50.0, 1.0});
  std::vector<gumbel::Action> a(8, gumbel::Action{0.125, 0, 0, 0.0, true});
  for (std::size_t e = 0; e < 8; ++e) a[e].q = 0.1 * static_cast<double>(e % 3);
  while (root.phase() == 0) {
    const std::size_t e = root.next(a, 0.0);
    REQUIRE(e < a.size());
    ++a[e].visits;
  }
  REQUIRE_EQ(root.shown(), std::size_t{4});
  const std::vector<std::size_t> members = root.considered();
  REQUIRE_EQ(members.size(), std::size_t{2});
  for (const std::size_t m : members) a[m].open = false;
  const std::size_t b = root.best(a, 0.0);  // G57
  CHECK(b < 4 && a[b].open);
  const std::size_t h = root.next(a, 0.0);  // G54
  CHECK(h < 4 && a[h].open);
  CHECK_EQ(root.shown(), std::size_t{4});
  for (std::size_t e = 0; e < 4; ++e) a[e].open = false;
  CHECK_EQ(root.next(a, 0.0), std::size_t{4});  // G55
  CHECK_EQ(root.shown(), std::size_t{5});
}

TEST(gumbel_halving_keeps_the_ceiling_of_half) {
  // G42: each halving keeps ceil(c / 2) - five candidates: 5, then 3, then 2.
  gumbel::Root root = gumbel::Root::make({0.5, 0.4, 0.3, 0.2, 0.1}, 5, gumbel::Config{5, 50, 50.0, 1.0});
  REQUIRE_EQ(root.schedule().size(), std::size_t{3});
  std::vector<gumbel::Action> a(5, gumbel::Action{0.2, 0, 0, 0.0, true});
  for (std::size_t e = 0; e < 5; ++e) a[e].q = 0.2 * static_cast<double>(e) - 0.4;
  std::vector<std::size_t> sizes{root.considered().size()};
  for (int k = 0; k < 50; ++k) {
    const std::size_t phase = root.phase();
    const std::size_t e = root.next(a, 0.0);
    REQUIRE(e < a.size());
    if (root.phase() != phase) sizes.push_back(root.considered().size());
    ++a[e].visits;
    a[e].q = 0.2 * static_cast<double>(e) - 0.4;
  }
  CHECK((sizes == std::vector<std::size_t>{5, 3, 2}));
}

TEST(gumbel_halving_keeps_only_the_members_left) {
  // G43: a halving keeps no more than the members left - three of four closed in the second of
  // three phases leave one member, and the halving keeps it alone.
  std::vector<double> g;  // g decreasing by 2, equal priors and q: the score order is the edge order
  for (int e = 0; e < 8; ++e) g.push_back(14.0 - 2.0 * e);
  gumbel::Root root = gumbel::Root::make(g, 8, gumbel::Config{8, 48, 50.0, 1.0});
  REQUIRE_EQ(root.schedule().size(), std::size_t{3});
  std::vector<gumbel::Action> a(8, gumbel::Action{0.125, 0, 0, 0.0, true});
  while (root.phase() == 0) {
    const std::size_t e = root.next(a, 0.0);
    REQUIRE(e < a.size());
    ++a[e].visits;
  }
  REQUIRE((root.considered() == std::vector<std::size_t>{0, 1, 2, 3}));
  a[0].open = a[2].open = a[3].open = false;  // edge 1 - the first phase's second best - is left
  while (root.phase() == 1) {
    const std::size_t e = root.next(a, 0.0);
    REQUIRE(e < a.size());
    ++a[e].visits;
  }
  CHECK((root.considered() == std::vector<std::size_t>{1}));
}

TEST(gumbel_refill_skips_a_closed_edge) {
  // G37: the first phase's refill skips an edge that closed before it was shown.
  gumbel::Root root =
      gumbel::Root::make({0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}, 8, gumbel::Config{4, 40, 50.0, 1.0});
  std::vector<gumbel::Action> a(8, gumbel::Action{0.125, 0, 0, 0.0, true});
  a[1].open = false;  // a candidate found illegal
  a[4].open = false;  // the next of the order, closed before it is shown
  const std::size_t e = root.next(a, 0.0);
  CHECK(e < a.size() && a[e].open);
  CHECK((root.considered() == std::vector<std::size_t>{0, 2, 3, 5}));
  CHECK_EQ(root.shown(), std::size_t{6});
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

// Logits a function of the move alone, a fixed submit logit (NaN: none given) and equal values;
// the first request's logits kept - a search's first evaluation is its root's.
class Keyed final : public mcts::Evaluator {
 public:
  explicit Keyed(float submit) : submit_(submit) {}
  static float of(const Move& m) {
    const int key = 7 * m.from_square + 3 * m.to_square + 5 * m.to_timeline + 2 * m.to_half_turn;
    return 0.31f * static_cast<float>(((key % 11) + 11) % 11) - 1.2f;
  }
  void evaluate(std::span<const mcts::EvalRequest> batch, std::span<mcts::Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const auto& moves = batch[i].expansion->moves();
      out[i].logits.resize(moves.size());
      for (std::size_t k = 0; k < moves.size(); ++k) out[i].logits[k] = of(moves[k]);
      out[i].submit_logit = submit_;
      out[i].wdl = mcts::wdl_of(0.0);
      out[i].value = mcts::two_hot(0.0);
      if (first_) root_logits = out[i].logits;
      first_ = false;
    }
  }
  std::vector<float> root_logits;

 private:
  float submit_;
  bool first_ = true;
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
    CHECK(y.gumbel_view().rows.empty());
    CHECK_THROWS_AS(y.choose_gumbel(), std::invalid_argument);
  }
  // M13 (the mutation pass of 2026-11-27): the comparison above is relative - a hook firing with
  // the flag off fires in both searches. Absolutely: off, the root's edges are in the as-built
  // order, the stable sort by prior (no draw reorders them), and there is no Gumbel view.
  for (const mcts::TurnSplit split : {mcts::TurnSplit::BoardMove, mcts::TurnSplit::Tree}) {
    mcts::SearchParams p = gumbel_params(63);
    p.gumbel_root = false;
    p.widening = false;  // every edge instantiated
    p.split = split;
    mcts::Search s(standard(), p);
    s.run(*ev, mcts::Limits{8});
    const std::vector<mcts::RootChild> c = s.root_children();
    REQUIRE(c.size() >= 20);
    bool distinct = false;
    for (std::size_t i = 0; i + 1 < c.size(); ++i) {
      CHECK(c[i].prior >= c[i + 1].prior);
      distinct = distinct || c[i].prior != c[i + 1].prior;
    }
    CHECK(distinct);  // the random evaluator's logits differ: the order means something
    CHECK(!s.gumbel_view().drawn);
    CHECK(s.gumbel_view().rows.empty());
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
  // SF-1 (the review of 2026-11-27): the root's logits are the network's raw ones, as I21 reads.
  p = gumbel_params(64);
  p.policy_temperature = 1.25;
  CHECK(refused(p, "Gumbel root search): policy_temperature must be 1"));
  p = gumbel_params(64);
  p.unknown_prior_scale = 0.5;
  CHECK(refused(p, "Gumbel root search): unknown_prior_scale must be 1"));
  p = gumbel_params(64);
  p.unknown_prior_scale = 0;  // unknown moves would never be candidates
  CHECK(refused(p, "unknown_prior_scale must be 1"));
  p = gumbel_params(0);
  CHECK(refused(p, "gumbel_simulations"));
  p = gumbel_params(64, 0);
  CHECK(refused(p, "gumbel_m"));
  for (const double bad : {-1.0, std::nan("")}) {
    p = gumbel_params(64);
    p.gumbel_c_visit = bad;
    CHECK(refused(p, "SearchParams::gumbel_c_visit"));
    p = gumbel_params(64);
    p.gumbel_c_scale = bad;
    CHECK(refused(p, "SearchParams::gumbel_c_scale"));
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

TEST(mcts_gumbel_root_logits_are_the_raw_logits_i21_reads) {
  // SF-1 (the review of 2026-11-27): under the arm each candidate's logit (gumbel_view's row:
  // ln of the root prior, which the draw, the halving and the choice read) is the network's raw
  // logit less one constant, the same for every candidate - and it is ln of the raw prior as
  // train.targets.raw_prior reads it for I21's target (a submit the network gave no logit:
  // submit_share, the moves (1 - submit_share) times their softmax; the generator gives its full
  // and cheap searches one submit_share, and the target reads the record's). Both splits; the
  // Standard start (no submit) and a mid-turn root with a legal submit, with and without the
  // network's submit logit.
  GameState mid = t5d_test::game({"[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]"}, BoardSize{4, 4}, RuleOptions{},
                                 Colour::White);
  t5d_test::play(mid, "(0T1)Ka1(0T1)a2");
  const double share = mcts::SearchParams{}.submit_share;
  int submits = 0;
  for (const mcts::TurnSplit split : {mcts::TurnSplit::BoardMove, mcts::TurnSplit::Tree}) {
    for (const bool middle : {false, true}) {
      for (const bool given : {true, false}) {
        constexpr float kSubmitLogit = 0.4f;
        Keyed ev(given ? kSubmitLogit : std::numeric_limits<float>::quiet_NaN());
        mcts::SearchParams p = gumbel_params(40, 32);
        p.split = split;
        mcts::Search s(middle ? mid : standard(), p, middle ? 1 : 0);
        s.run(ev, mcts::Limits{41});
        const gumbel::RootView view = s.gumbel_view();
        const std::vector<mcts::RootChild> children = s.root_children();
        REQUIRE(view.drawn);
        REQUIRE_EQ(view.rows.size(), children.size());
        REQUIRE_EQ(children.size(), s.node(s.root()).edge_count);  // m_eff covers every open edge
        bool offered = false;
        for (const mcts::RootChild& c : children) offered = offered || c.submit;
        submits += offered ? 1 : 0;
        CHECK(offered == middle);
        // The raw prior, train.targets.raw_prior's form, from the root's own evaluation.
        const std::vector<float>& z = ev.root_logits;
        REQUIRE_EQ(z.size() + (offered ? 1u : 0u), children.size());
        double top = given && offered ? double{kSubmitLogit} : -kInf;
        for (const float l : z) top = std::max(top, static_cast<double>(l));
        double sum = given && offered ? std::exp(double{kSubmitLogit} - top) : 0.0;
        for (const float l : z) sum += std::exp(static_cast<double>(l) - top);
        const double moves_share = offered && !given ? 1.0 - share : 1.0;
        double constant = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t i = 0; i < children.size(); ++i) {
          const mcts::RootChild& c = children[i];
          const double raw = c.submit ? double{kSubmitLogit} : static_cast<double>(Keyed::of(c.moves.at(0)));
          const double raw_prior = c.submit ? (given ? std::exp(raw - top) / sum : share)
                                            : moves_share * std::exp(raw - top) / sum;
          CHECK(std::abs(view.rows[i].logit - std::log(raw_prior)) < 1e-9);
          if (c.submit && !given) continue;  // no raw logit: submit_share stands for it
          if (std::isnan(constant)) constant = view.rows[i].logit - raw;
          CHECK(std::abs(view.rows[i].logit - raw - constant) < 1e-9);
        }
      }
    }
  }
  CHECK_EQ(submits, 4);
}

TEST(mcts_gumbel_constants_are_capped_so_sigma_stays_finite) {
  // EC-3 (the review of 2026-11-27): unbounded, (c_visit + max N) x c_scale overflowed - c_visit
  // 1e308 and c_scale 10 made a lost candidate's sigma NaN (infinity x 0) and best() play it. Each
  // constant is capped at 1e6 (a placeholder), refused above it whether or not the arm is on
  // (refusing an absurd value with the flag off is the reading); at both caps sigma is finite for
  // any 32-bit visit count.
  const auto refused = [](const mcts::SearchParams& p, const char* name) {
    try {
      mcts::Search s(standard(), p);
    } catch (const std::invalid_argument& e) {
      return std::string(e.what()).find(name) != std::string::npos;
    }
    return false;
  };
  for (const bool on : {true, false}) {
    mcts::SearchParams p = gumbel_params(63);
    p.gumbel_root = on;
    p.gumbel_c_visit = 1e308;
    p.gumbel_c_scale = 10;
    CHECK(refused(p, "SearchParams::gumbel_c_visit"));
    p.gumbel_c_visit = std::nextafter(gumbel::kMaxCVisit, kInf);
    p.gumbel_c_scale = 1;
    CHECK(refused(p, "SearchParams::gumbel_c_visit"));
    p.gumbel_c_visit = kInf;
    CHECK(refused(p, "SearchParams::gumbel_c_visit"));
    p.gumbel_c_visit = 50;
    p.gumbel_c_scale = std::nextafter(gumbel::kMaxCScale, kInf);
    CHECK(refused(p, "SearchParams::gumbel_c_scale"));
    p.gumbel_c_scale = kInf;
    CHECK(refused(p, "SearchParams::gumbel_c_scale"));
    p.gumbel_c_visit = gumbel::kMaxCVisit;
    p.gumbel_c_scale = gumbel::kMaxCScale;
    CHECK_NOTHROW(mcts::Search(standard(), p));
    p.gumbel_c_visit = 0;
    p.gumbel_c_scale = 0;
    CHECK_NOTHROW(mcts::Search(standard(), p));
  }
  CHECK_EQ(gumbel::kMaxCVisit, 1e6);
  CHECK_EQ(gumbel::kMaxCScale, 1e6);
  for (const double q : {-1.0, 0.0, 1.0}) {
    const double s = gumbel::sigma(q, std::numeric_limits<std::uint32_t>::max(), gumbel::kMaxCVisit, gumbel::kMaxCScale);
    CHECK(std::isfinite(s) && s >= 0 && s < 5e15);
  }
  // The pure part refuses them too (the case the review found: a lost and a won candidate).
  CHECK_THROWS_AS(gumbel::Root::make({0.0, 0.0}, 2, gumbel::Config{2, 2, 1e308, 10.0}), std::invalid_argument);
  CHECK_THROWS_AS(gumbel::Root::make({0.0, 0.0}, 2, gumbel::Config{2, 2, 50.0, std::nan("")}), std::invalid_argument);
  CHECK_THROWS_AS(gumbel::Root::make({0.0, 0.0}, 2, gumbel::Config{2, 2, -1.0, 1.0}), std::invalid_argument);
  gumbel::Root capped = gumbel::Root::make({0.0, 0.0}, 2, gumbel::Config{2, 2, gumbel::kMaxCVisit, gumbel::kMaxCScale});
  std::vector<gumbel::Action> a(2);
  a[0] = {0.5, 1, 0, -1.0, true};  // lost
  a[1] = {0.5, 1, 0, +1.0, true};  // won
  CHECK_EQ(capped.best(a, 0.0), std::size_t{1});
}

TEST(gumbel_nan_scores_order_last_in_the_choice_and_the_halving) {
  // EC-3's guard: a NaN score (a NaN q or prior from the caller) orders below every number, so
  // best() never plays it while another is open and the halving's sort keeps a strict weak order.
  gumbel::Root root = gumbel::Root::make({0.0, 0.0}, 2, gumbel::Config{2, 2, 50.0, 1.0});
  std::vector<gumbel::Action> a(2);
  a[0] = {0.5, 1, 0, std::nan(""), true};
  a[1] = {0.5, 1, 0, -1.0, true};
  CHECK_EQ(root.best(a, 0.0), std::size_t{1});
  std::swap(a[0], a[1]);
  CHECK_EQ(root.best(a, 0.0), std::size_t{0});
  a[0].q = a[1].q = std::nan("");  // every score NaN: still an open candidate, the first
  CHECK_EQ(root.best(a, 0.0), std::size_t{0});
  // The halving: four candidates a visit each, then two kept - never the NaN one, which has
  // the largest g (ordered by score it would be first).
  gumbel::Root four = gumbel::Root::make({9.0, 0.3, 0.2, 0.1}, 4, gumbel::Config{4, 8, 50.0, 1.0});
  REQUIRE_EQ(four.schedule().size(), std::size_t{2});
  std::vector<gumbel::Action> b(4, gumbel::Action{0.25, 0, 0, 0.0, true});
  for (int k = 0; k < 4; ++k) {
    const std::size_t e = four.next(b, 0.0);
    REQUIRE(e < 4);
    ++b[e].visits;
    b[e].q = e == 0 ? std::nan("") : 0.1 * static_cast<double>(e);
  }
  REQUIRE_EQ(four.next(b, 0.0), std::size_t{2});  // the halving ran; of the two kept, the first visited first
  CHECK((four.considered() == std::vector<std::size_t>{2, 3}));
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
  // the batch short (at most once a phase; a collision ends one too) - the uniform evaluator, so
  // the choice is the same too. Batches of 8 divide this search's phases (48, 48, 48): it never
  // waits, and no selection under the root collides.
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
  CHECK(st.batches <= full + 3);  // at most the three halvings' waits (the review of 2026-11-27, EC-2)
}

TEST(mcts_gumbel_waits_at_most_once_a_phase) {
  // EC-2 (the review of 2026-11-27; the correctness pass's rv_search_waits_per_phase, adopted,
  // its bound tightened): a phase's batch ends at a halving's wait at most once - before EC-1's
  // fix a collision counted as a visit in flight made a phase wait again (twice in phase 0 in all
  // 12 runs at gathers of 24). A collision under a Gumbel root now ends its batch (at most one a
  // batch), and every short batch ends at one or the other. Through the step API: gathers of 5,
  // 7, 24 and 32, the Random evaluator, seeds 1-6, both splits.
  int runs = 0, waits = 0;
  std::uint64_t collision_ends = 0;
  for (const mcts::TurnSplit split : {mcts::TurnSplit::BoardMove, mcts::TurnSplit::Tree}) {
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
      for (const std::size_t k : {std::size_t{5}, std::size_t{7}, std::size_t{24}, std::size_t{32}}) {
        const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, seed);
        mcts::SearchParams p = gumbel_params(200, 16, seed);
        p.split = split;
        mcts::Search s(standard(), p);
        std::map<std::size_t, int> per_phase;
        constexpr std::uint64_t kCap = 201;
        for (int b = 0; b < 1000 && s.stats().playouts < kCap; ++b) {
          const bool drawn = s.gumbel_view().drawn;
          const std::uint64_t p0 = s.stats().playouts;
          const std::uint64_t c0 = s.stats().collisions;
          const std::size_t n = s.gather(k, kCap - p0);
          const std::uint64_t done = s.stats().playouts - p0;
          const std::uint64_t coll = s.stats().collisions - c0;
          const bool cut = n + done >= k || n + done >= kCap - p0;  // full, or the cap's
          if (drawn) {
            CHECK(coll <= 1);
            const gumbel::RootView v = s.gumbel_view();
            // A halving's wait, read after the gather: every considered candidate's visits and
            // visits in flight at the target, one's visits below it, the phase not the last.
            bool wait = false;
            if (!cut && coll == 0 && v.phase + 1 < v.schedule.size()) {
              std::uint64_t target = 0;
              for (std::size_t r = 0; r <= v.phase; ++r) target += v.schedule[r].visits;
              bool all_out = true, some_back = false;
              for (std::size_t i = 0; i < v.rows.size(); ++i) {
                if (!v.rows[i].considered) continue;
                const mcts::EdgeView e = s.edge(s.root(), i);
                all_out = all_out && e.visits + e.virtual_loss >= target;
                some_back = some_back || e.visits < target;
              }
              wait = all_out && some_back;
            }
            if (wait) ++per_phase[v.phase];
            waits += wait ? 1 : 0;
            collision_ends += coll;
            CHECK(cut || wait || coll == 1);  // a short batch ends at a wait or a collision
          }
          if (n == 0) {
            REQUIRE(done > 0);  // a gather with no leaf completed visits: never stuck
            continue;
          }
          std::vector<mcts::Evaluation> out(n);
          ev->evaluate(s.requests(), out);
          s.deliver(out);
        }
        ++runs;
        CHECK_EQ(s.stats().playouts, kCap);
        CHECK(root_visits(s) == multiset({{3, 8}, {9, 4}, {21, 2}, {49, 2}}));
        for (const auto& [phase, w] : per_phase) CHECK(w <= 1);
        CHECK_NOTHROW(s.check_invariants());
      }
    }
  }
  std::fprintf(stderr, "  %d runs: %d halving waits, %llu batches ended by a collision\n", runs, waits,
               static_cast<unsigned long long>(collision_ends));
  CHECK_EQ(runs, 48);
  CHECK(waits > 0);
  CHECK(collision_ends > 0);
}

TEST(mcts_gumbel_batched_allocation_is_the_synchronous_one) {
  // EC-1 (the review of 2026-11-27; the correctness pass's rv_batched_vs_sync_random and
  // rv_last_phase_imbalance, adopted): batched, the visits are the synchronous search's. The
  // root's pick counts an edge's virtual loss as visits in flight, and a collision's never
  // becomes one - before the fix the last two split 48 and 50 where the synchronous search
  // gives 49 and 49 (seed 6, batches of 32; 10 of 20 seeds unequal at batches of 128); a
  // collision under a Gumbel root now ends the batch. Seeds 0-19, batches of 8, 32 and 128,
  // both splits:
  // - the Random evaluator (keyed by position): each candidate's visits, ranked, are the
  //   synchronous search's and the schedule's - the last two 49 and 49. Which candidates
  //   survive a halving may differ: it reads their values, and below the root a batch's leaves
  //   are chosen under virtual loss, not after each other's values (counted, not checked);
  // - the uniform evaluator (every value 0, so the halving keeps the same candidates): each
  //   candidate's visits, edge by edge, and the choice are the synchronous search's.
  int compared = 0, unequal = 0, other_survivors = 0;
  for (const bool uniform : {false, true}) {
    for (const mcts::TurnSplit split : {mcts::TurnSplit::BoardMove, mcts::TurnSplit::Tree}) {
      for (const std::size_t batch : {std::size_t{8}, std::size_t{32}, std::size_t{128}}) {
        for (std::uint64_t seed = 0; seed < 20; ++seed) {
          const std::unique_ptr<mcts::Evaluator> ev =
              mcts::make_evaluator(uniform ? mcts::BuiltinEvaluator::Uniform : mcts::BuiltinEvaluator::Random, seed);
          mcts::SearchParams b = gumbel_params(200, 16, seed);
          b.split = split;
          b.batch_size = batch;
          mcts::SearchParams y = b;
          y.synchronous = true;
          mcts::Search x(standard(), b);
          mcts::Search z(standard(), y);
          x.run(*ev, mcts::Limits{201});
          z.run(*ev, mcts::Limits{201});
          ++compared;
          const std::vector<mcts::RootChild> cx = x.root_children();
          const std::vector<mcts::RootChild> cz = z.root_children();
          REQUIRE_EQ(cx.size(), cz.size());
          bool same_ranked = root_visits(x) == root_visits(z);
          CHECK(root_visits(x) == multiset({{3, 8}, {9, 4}, {21, 2}, {49, 2}}));
          bool same_by_edge = true;
          for (std::size_t i = 0; i < cx.size(); ++i) {
            CHECK(cx[i].moves == cz[i].moves);  // one sample, one order
            same_by_edge = same_by_edge && cx[i].visits == cz[i].visits;
          }
          if (uniform) {
            same_ranked = same_ranked && same_by_edge && x.choose_gumbel() == z.choose_gumbel();
          } else {
            other_survivors += same_by_edge ? 0 : 1;
          }
          if (!same_ranked) {
            ++unequal;
            std::string d;
            for (std::size_t i = 0; i < cx.size(); ++i) d += std::to_string(cx[i].visits) + "/" + std::to_string(cz[i].visits) + " ";
            std::fprintf(stderr, "  %s %s batch %zu seed %llu: batched/sync %s\n", uniform ? "uniform" : "random",
                         split == mcts::TurnSplit::Tree ? "tree" : "board-move", batch,
                         static_cast<unsigned long long>(seed), d.c_str());
          }
          CHECK_NOTHROW(x.check_invariants());
        }
      }
    }
  }
  std::fprintf(stderr, "  %d pairs, %d unequal; under the random evaluator %d of 120 kept other survivors\n", compared,
               unequal, other_survivors);
  CHECK_EQ(compared, 240);
  CHECK_EQ(unequal, 0);
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

// ---- the search's arm, pinned (the mutation pass of 2026-11-27, TP-1) ---------------------------------

namespace {

// Counts the positions it is asked to evaluate.
class Counting final : public mcts::Evaluator {
 public:
  explicit Counting(mcts::Evaluator& inner) : inner_(inner) {}
  void evaluate(std::span<const mcts::EvalRequest> batch, std::span<mcts::Evaluation> out) override {
    calls += batch.size();
    inner_.evaluate(batch, out);
  }
  std::size_t calls = 0;

 private:
  mcts::Evaluator& inner_;
};

}  // namespace

TEST(mcts_gumbel_root_g_is_the_seed_and_root_keys_stream) {
  // M16: each root edge's g is variate(stream_seed(search seed, root key), its index in the
  // expansion's order) - the root key's part of the stream pinned in the search, not only in
  // gumbel::stream_seed.
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search s(standard(), gumbel_params(63, 16, 7));
  s.run(*ev, mcts::Limits{64});
  const gumbel::RootView v = s.gumbel_view();
  REQUIRE(v.drawn);
  const std::uint64_t stream = gumbel::stream_seed(7, s.node(s.root()).key);
  const std::size_t edges = s.node(s.root()).edge_count;
  for (const gumbel::Row& r : v.rows) {
    bool found = false;
    for (std::size_t j = 0; j < edges && !found; ++j) found = gumbel::variate(stream, j) == r.g;
    CHECK(found);
  }
}

TEST(mcts_gumbel_choice_refused_while_a_batch_waits) {
  // M33: no choice while a batch waits for its evaluations.
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search s(standard(), gumbel_params(63));
  s.run(*ev, mcts::Limits{20});
  REQUIRE(s.gather(4) > 0);
  CHECK_THROWS_AS(s.choose_gumbel(), std::invalid_argument);
  s.abandon();
  CHECK_NOTHROW(s.choose_gumbel());
}

TEST(mcts_gumbel_choice_indexes_root_children_after_a_removal) {
  // M36: the choice is an index into root_children(), also after a candidate before it was found
  // illegal (the king in check: d1 and f1 illegal, unknown to the expansion's zero budget).
  const GameState s = t5d_test::game({"[k7/8/8/8/8/8/8/r3K3:0:1:w]"}, BoardSize{8, 8});
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  int removed = 0;
  for (std::uint64_t seed = 1; seed <= 12; ++seed) {
    mcts::SearchParams p = gumbel_params(20, 2, seed);
    p.legality = LegalityOptions::budgeted(0, 0);
    mcts::Search search(s, p);
    search.run(*ev, mcts::Limits{21});
    if (search.stats().verified_illegal == 0) continue;
    ++removed;
    const std::vector<mcts::RootChild> children = search.root_children();
    const std::size_t c = search.choose_gumbel();
    REQUIRE(c < children.size());
    CHECK(children[c].visits > 0);
    CHECK(search.gumbel_view().rows.at(c).considered);
  }
  CHECK(removed > 0);
}

TEST(mcts_gumbel_a_refilled_candidate_is_instantiated_at_once) {
  // M23: a candidate the first phase's refill shows is a root child at once - after every single
  // playout, every visited root edge is within the instantiated prefix (root_children() lists it).
  const GameState s = t5d_test::game({"[k7/8/8/8/8/8/8/r3K3:0:1:w]"}, BoardSize{8, 8});
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  int refilled = 0;
  for (std::uint64_t seed = 1; seed <= 12; ++seed) {
    mcts::SearchParams p = gumbel_params(20, 2, seed);
    p.legality = LegalityOptions::budgeted(0, 0);
    p.synchronous = true;
    mcts::Search search(s, p);
    for (int k = 0; k < 21; ++k) {
      search.run(*ev, mcts::Limits{1});  // one more playout
      const mcts::NodeView root = search.node(search.root());
      for (std::size_t i = root.instantiated; i < root.edge_count; ++i) {
        CHECK_EQ(search.edge(search.root(), i).visits, std::uint32_t{0});
      }
    }
    refilled += search.stats().verified_illegal > 0 ? 1 : 0;
  }
  CHECK(refilled > 0);
}

TEST(mcts_gumbel_wait_releases_the_roots_virtual_visit) {
  // M28: a halving's wait releases the selection's virtual visit at the root. Batches of 5 do not
  // divide the first three phases' 48 visits, so a batch ends short at each of their ends (with
  // batches of 8 the phases end on batch boundaries and the search never waits).
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search s(standard(), gumbel_params(200));
  int short_batches = 0;
  for (int b = 0; b < 100 && s.stats().playouts < 201; ++b) {
    const std::uint64_t collisions = s.stats().collisions;
    const std::size_t n = s.gather(5, 201 - s.stats().playouts);
    if (b > 0 && n < 5 && s.stats().collisions == collisions && s.stats().playouts < 201 - n) ++short_batches;
    std::vector<mcts::Evaluation> out(n);
    ev->evaluate(s.requests(), out);
    s.deliver(out);
    CHECK_EQ(s.node(s.root()).virtual_loss, std::uint32_t{0});
  }
  std::fprintf(stderr, "  %d short batches without a collision\n", short_batches);
  CHECK(short_batches >= 3);  // the first three phases' ends
  CHECK_EQ(s.stats().playouts, std::uint64_t{201});
  CHECK_NOTHROW(s.check_invariants());
}

TEST(mcts_gumbel_a_batch_ends_at_a_halvings_wait) {
  // M29: the halving's wait ends the batch - a gather as large as can be asked for ends at the
  // first phase's end, not at its iteration limit (unbounded for SIZE_MAX: without the wait's
  // break the gather would never return). Each of the 16 candidates has 2 of its 3 visits; the
  // gather gives each its third (16 leaves, no collision), then waits.
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::Search s(standard(), gumbel_params(200));
  s.run(*ev, mcts::Limits{33});  // the root's own evaluation, then two visits to each candidate
  for (const mcts::RootChild& c : s.root_children()) REQUIRE_EQ(c.visits, std::uint32_t{2});
  const std::uint64_t collisions = s.stats().collisions;
  const std::size_t n = s.gather(std::numeric_limits<std::size_t>::max());
  CHECK_EQ(n, std::size_t{16});
  CHECK_EQ(s.stats().collisions, collisions);  // ended by the wait, not by a collision
  std::vector<mcts::Evaluation> out(n);
  ev->evaluate(s.requests(), out);
  s.deliver(out);
  for (const mcts::RootChild& c : s.root_children()) CHECK_EQ(c.visits, std::uint32_t{3});
  CHECK_EQ(s.gumbel_view().phase, std::size_t{0});  // the halving runs at the next pick
  CHECK_NOTHROW(s.check_invariants());
}

TEST(mcts_gumbel_timed_refusal_leaves_the_manager_unchanged) {
  // T02: TimeManager::begin refuses a Gumbel search before anything changes - compared with the
  // manager's state before the refused call, in a decision begun on a plain search (a refusal
  // made after beginning would have begun a decision on the 30-second clock).
  mcts::SearchParams plain = gumbel_params(63);
  plain.gumbel_root = false;
  mcts::Search other(standard(), plain);
  mcts::Search gumbel(standard(), gumbel_params(63));
  mcts::TimeManager tm;
  mcts::TimeContext ctx;
  ctx.remaining_seconds = 60;
  CHECK_THROWS_AS(tm.begin(gumbel, ctx), std::invalid_argument);  // a fresh manager
  CHECK(!tm.in_decision());
  CHECK_EQ(tm.plan().boards, 0);
  CHECK_EQ(tm.plan().usable_seconds, 0.0);
  tm.begin(other, ctx);
  REQUIRE(tm.in_decision());
  const mcts::TimePlan before = tm.plan();
  const double scale = tm.scale();
  mcts::TimeContext shorter;
  shorter.remaining_seconds = 30;
  CHECK_THROWS_AS(tm.begin(gumbel, shorter), std::invalid_argument);
  CHECK(tm.in_decision());
  const mcts::TimePlan after = tm.plan();
  CHECK_EQ(after.boards, before.boards);
  CHECK_EQ(after.usable_seconds, before.usable_seconds);
  CHECK_EQ(after.soft_evaluations, before.soft_evaluations);
  CHECK_EQ(after.soft_seconds, before.soft_seconds);
  CHECK_EQ(after.hard_seconds, before.hard_seconds);
  CHECK_EQ(tm.scale(), scale);
}

TEST(mcts_gumbel_memory_budget_counts_the_roots_buffers) {
  // The correctness pass's minor point (2026-11-27): memory_bytes() - Limits::bytes' measure -
  // counts a Gumbel root's buffers (the halving's and the view selection fills): after the root's
  // own evaluation alone the tree is the plain search's, byte for byte, and the difference is at
  // least the 20 edges' view and g. Off, an inactive Root holds nothing.
  CHECK_EQ(gumbel::Root{}.heap_bytes(), std::size_t{0});
  const gumbel::Root made = gumbel::Root::make(std::vector<double>(20, 0.0), 20, gumbel::Config{16, 200, 50.0, 1.0});
  CHECK(made.heap_bytes() >= 20 * (2 * sizeof(double) + sizeof(std::size_t)));
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  mcts::SearchParams on = gumbel_params(200);
  mcts::SearchParams off = on;
  off.gumbel_root = false;
  mcts::Search a(standard(), on);
  mcts::Search b(standard(), off);
  a.run(*ev, mcts::Limits{1});
  b.run(*ev, mcts::Limits{1});
  REQUIRE(a.gumbel_view().drawn);
  REQUIRE_EQ(a.node(a.root()).edge_count, std::size_t{20});
  CHECK(a.memory_bytes() >= b.memory_bytes() + 20 * (sizeof(gumbel::Action) + sizeof(double)));
}

TEST(mcts_gumbel_game_loop_refuses_before_any_evaluation) {
  // S01: play_game refuses the arm itself, up front - by name, before any evaluation (the loop's
  // later refusals, choose_by_visits' temperature or a re-root, come only after a search's work).
  const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Uniform);
  Counting counting(*ev);
  mcts::PlayParams pp;
  pp.search = gumbel_params(63);
  bool named = false;
  try {
    mcts::play_game(standard(), counting, pp);
  } catch (const std::invalid_argument& e) {
    named = std::string(e.what()).find("PlayParams::search.gumbel_root") != std::string::npos;
  }
  CHECK(named);
  CHECK_EQ(counting.calls, std::size_t{0});
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

TEST(capi_gumbel_fields_reach_the_search) {
  // C01, C03-C05 (the mutation pass of 2026-11-27): each Gumbel field of t5d_search_params
  // reaches the search - the C API's search is the C++ twin's with the same non-default m,
  // c_visit and c_scale, under the random evaluator (values differ, so sigma's form matters), and
  // each field at another value gives another tree (the precondition that makes the comparison
  // mean something). C06: gumbel_root is 0 or 1 - params_from refuses any other value of every
  // int32 flag of the ABI (boolean()), and this one is no exception.
  const std::uint64_t kSeed = 5;
  const auto twin_digest = [&](std::uint32_t m, double c_visit, double c_scale) {
    mcts::SearchParams cpp = gumbel_params(200, m);
    cpp.gumbel_c_visit = c_visit;
    cpp.gumbel_c_scale = c_scale;
    mcts::Search twin(standard(), cpp);
    const std::unique_ptr<mcts::Evaluator> ev = mcts::make_evaluator(mcts::BuiltinEvaluator::Random, kSeed);
    twin.run(*ev, mcts::Limits{201});
    return twin.digest();
  };
  const std::uint32_t m = 8;
  const double c_visit = 2.0, c_scale = 0.3;  // each field matters here (the mutation pass's grid)
  const std::uint64_t want = twin_digest(m, c_visit, c_scale);
  CHECK(twin_digest(16, c_visit, c_scale) != want);  // m not copied
  CHECK(twin_digest(m, 50.0, c_scale) != want);      // c_visit not copied
  CHECK(twin_digest(m, c_visit, 1.0) != want);       // c_scale not copied
  CHECK(twin_digest(m, c_scale, c_visit) != want);   // the two swapped
  t5d_search_params p;
  t5d_search_params_default(&p);
  p.expand.square_features = T5D_FEATURES_NONE;
  p.gumbel_root = 1;
  p.gumbel_simulations = 200;
  p.gumbel_m = m;
  p.gumbel_c_visit = c_visit;
  p.gumbel_c_scale = c_scale;
  p.hunt = 0;
  p.settling = 0;
  p.asset_map = 0;
  t5d_game* g = nullptr;
  REQUIRE_EQ(t5d_game_new_variant("Standard", T5D_RULES_STANDARD, &g, nullptr), T5D_OK);
  t5d_search* s = nullptr;
  REQUIRE_EQ(t5d_search_new(g, 1, &p, &s, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_run(s, T5D_EVALUATOR_RANDOM, kSeed, 201, nullptr), T5D_OK);
  std::uint64_t digest = 0;
  REQUIRE_EQ(t5d_search_digest(s, &digest), T5D_OK);
  CHECK_EQ(digest, want);
  t5d_search_destroy(s);
  for (const std::int32_t bad : {2, -1}) {
    p.gumbel_root = bad;
    char* error = nullptr;
    CHECK_EQ(t5d_search_new(g, 1, &p, &s, &error), T5D_ERR_ARGUMENT);
    CHECK(error != nullptr && std::string(error).find("gumbel_root must be 0 or 1") != std::string::npos);
    t5d_free(error);
  }
  t5d_game_destroy(g);
}
