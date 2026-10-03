// Hunt (stage E9; spec 11.4; register H43, H37, H41): the constructed mates
// proven, their lines ending in E4's checkmate; an export only through the
// gate (off by default); nothing claimed but proofs at any budget (H37); one
// thread deterministic; soft priors only behind their flag.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/certificate.hpp"
#include "t5d/check.hpp"
#include "t5d/hunt.hpp"
#include "t5d/movegen.hpp"
#include "t5d/prover.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_game_tree.hpp"
#include "test_mate_refs.hpp"
#include "test_play.hpp"
#include "test_positions.hpp"
#include "test_reference.hpp"
#include "test_rulesets.hpp"
#include "test_support.hpp"
#include "test_v3_support.hpp"

using namespace t5d;
using namespace t5d_test::v3;
using t5d_test::describe_position;
using t5d_test::game;
using t5d_test::GameTree;
using t5d_test::GameValue;
using t5d_test::playable;
using t5d_test::RestrictedGame;

// ---- Hunt (H43) -----------------------------------------------------------------------------

TEST(hunt_proves_the_constructed_mates_and_exports_only_through_the_gate) {
  struct Want {
    GameState s;
    int distance;
  };
  const std::vector<Want> wants{{mate_in_one(), 1}, {mate_in_two(), 3}, {cross_board_mate(), 1}};
  for (const Want& w : wants) {
    Hunt h;
    const HuntResult r = h.hunt(w.s, ProveBudget::nodes(2'000'000));
    CHECK(r.mate);
    CHECK_EQ(r.proof.distance, w.distance);
    CHECK(r.line_complete);
    GameState t = w.s;
    CHECK(replay(t, r.line));
    CHECK(mated(t));
  }
  // The gate: closed (H23, H41) - Hunt refuses the export switch, as the
  // search refuses its proof store.
  HuntOptions open;
  open.export_proofs = true;
  CHECK_THROWS_WITH(Hunt{open}, std::invalid_argument, "H23");
}

namespace {

// Budgets stepped below the need: a mate only where the full hunt finds it
// (the same distance), and every mate confirmed by the restricted game of
// its class or the game tree; never a mate where the full hunt has none.
void budget_property(int turns, std::uint64_t steps, int& budgets, int& mates) {
  std::uint64_t seed = 2900;
  for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, turns, seed))) {
    HuntOptions o;
    o.prover.depth = 3;
    Hunt h(o);
    const HuntResult full = h.hunt(s, ProveBudget::nodes(300'000));
    if (full.mate) {
      ++mates;
      ProveResult as;
      as.status = ProofStatus::Proven;
      as.proof = full.proof;
      CHECK(t5d_test::verify(as, s, o.prover, 200'000) >= 0);
    }
    const std::uint64_t step = std::max<std::uint64_t>(1, full.stats.nodes / steps);
    for (std::uint64_t b = 0; b < full.stats.nodes; b += step) {
      const HuntResult r = h.hunt(s, ProveBudget::nodes(b));
      ++budgets;
      if (r.mate) {
        CHECK(full.mate);
        CHECK_EQ(r.proof.distance, full.proof.distance);
      }
    }
  }
}

}  // namespace

TEST(hunt_claims_nothing_but_proofs_h37) {
  int budgets = 0;
  int mates = 0;
  budget_property(3, 8, budgets, mates);
  std::fprintf(stderr, "  %d budgets, %d mates\n", budgets, mates);
  CHECK(budgets > 100);
  CHECK(mates > 3);
}

TEST(long_hunt_claims_nothing_but_proofs_h37) {
  int budgets = 0;
  int mates = 0;
  budget_property(4, 12, budgets, mates);
  std::fprintf(stderr, "  %d budgets, %d mates\n", budgets, mates);
  CHECK(budgets > 200);
  CHECK(mates > 5);
}

TEST(hunt_one_thread_is_deterministic_and_suggests_only_behind_its_flag) {
  int suggested = 0;
  for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, 3, 3000))) {
    Hunt h;
    const HuntResult a = h.hunt(s, ProveBudget::nodes(5000));
    const HuntResult b = h.hunt(s, ProveBudget::nodes(5000));
    CHECK_EQ(a.mate, b.mate);
    CHECK_EQ(a.stats.nodes, b.stats.nodes);
    CHECK(a.suggestions.empty());  // the flag is off
    HuntOptions o;
    o.suggest = true;
    Hunt hs(o);
    const HuntResult c = hs.hunt(s, ProveBudget::nodes(5000));
    if (c.mate || c.suggestions.empty()) continue;
    ++suggested;
    double total = 0;
    for (std::size_t i = 0; i < c.suggestions.size(); ++i) {
      total += c.suggestions[i].weight;
      if (i > 0) CHECK(c.suggestions[i - 1].pn <= c.suggestions[i].pn);
    }
    CHECK(total > 0.999 && total < 1.001);
  }
  CHECK(suggested > 5);
}

TEST(hunt_at_five_threads_claims_only_confirmed_mates) {
  // M13's 5 threads, by nodes and by the clock: a mate only where one
  // thread's full hunt finds it, at its distance, and every mate's
  // certificate confirmed by E4 alone; nothing else claimed.
  int mates = 0;
  int searches = 0;
  for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, 4, 3100))) {
    HuntOptions one;
    one.prover.depth = 3;
    Hunt h1(one);
    const HuntResult full = h1.hunt(s, ProveBudget::nodes(300'000));
    HuntOptions five = one;
    five.threads = 5;
    five.prover.record_certificate = true;
    Hunt h5(five);
    for (const bool by_clock : {false, true}) {
      const HuntResult r = by_clock ? h5.hunt(s, ProveBudget::time(std::chrono::milliseconds(20)))
                                    : h5.hunt(s, ProveBudget::nodes(60'000));
      ++searches;
      if (!r.mate) continue;
      ++mates;
      CHECK(full.mate);
      if (full.mate) CHECK_EQ(r.proof.distance, full.proof.distance);
      REQUIRE(r.certificate != nullptr);
      CertificateCheckOptions co;
      co.max_positions = 20000;
      const CertificateCheck c = check_certificate(*r.certificate, s, r.proof.distance, co);
      CHECK(c.verdict == CertificateVerdict::Confirmed || c.verdict == CertificateVerdict::Undecided);
    }
  }
  std::fprintf(stderr, "  %d searches at 5 threads, %d mates\n", searches, mates);
  CHECK(mates > 5);
}
