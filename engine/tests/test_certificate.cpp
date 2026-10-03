// Stage E9 (verification v3, spec 11.4): the prover's additions for Guard and
// Hunt - the last-turn check filter changes no verdict, prove-only finds the
// full search's mates and claims no disproof (H43, H37) - and every proof's
// line and certificate confirmed by E4 alone (t5d/certificate.hpp), a doctored
// certificate not. Positions written from the rulebook or reached by random
// legal play from such positions.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/certificate.hpp"
#include "t5d/check.hpp"
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

// ---- the prover's additions -----------------------------------------------------------------

TEST(prover_check_filter_changes_no_answer) {
  // The literal class with and without the filter: the same verdict wherever
  // both decide, at depths 1 and 3; the filter's answers held to the
  // references too.
  int compared = 0;
  std::uint64_t cut = 0;
  t5d_test::Tally tally;
  tally.work_cap = 100'000;
  std::uint64_t seed = 2100;
  for (const RuleOptions& rules : {RuleOptions{}, guide_literal_q4()}) {
    for (const int depth : {1, 3}) {
      for (const GameState& s : playable(t5d_test::sparse_positions(rules, 1, 4, seed++))) {
        ProverOptions on = literal();
        on.depth = depth;
        on.check_filter = true;  // off by default: the test switches it on
        ProverOptions off = on;
        off.check_filter = false;
        Prover a(on);
        Prover b(off);
        const ProveResult ra = a.prove(s, ProveBudget::nodes(300'000));
        const ProveResult rb = b.prove(s, ProveBudget::nodes(300'000));
        cut += ra.stats.check_cut;
        if (!tally.add(ra, s, on)) CHECK(false);
        if (ra.status == ProofStatus::Unknown || rb.status == ProofStatus::Unknown) continue;
        ++compared;
        if (ra.status != rb.status ||
            (ra.status == ProofStatus::Proven && ra.proof.distance != rb.proof.distance)) {
          std::fprintf(stderr, "  filter %s, none %s at %s\n", proof_status_name(ra.status),
                       proof_status_name(rb.status), describe_position(s).c_str());
          CHECK(false);
        }
      }
    }
  }
  tally.print("check filter");
  std::fprintf(stderr, "  %d compared, %llu partial actions cut\n", compared, static_cast<unsigned long long>(cut));
  CHECK(compared > 50);
  // (No cut is required here: since the J6-twin fix, a position with a
  // non-branching jump leaves every action uncut, and these positions nearly
  // all have one.)
}

TEST(prover_prove_only_finds_the_full_searchs_mates) {
  // Prove-only never claims a disproof and records none; at the full
  // search's budget it proves every mate the full search proves, at the same
  // distance (its thresholds are df-pn's usual ones). The named alternative,
  // proof-number thresholds alone, claims nothing either.
  int proofs = 0;
  int exhausted = 0;
  int pn_proofs = 0;
  std::uint64_t seed = 2200;
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, 4, seed++))) {
      ProverOptions full = base;
      full.depth = 3;
      ProverOptions only = full;
      only.prove_only = true;
      Prover a(full);
      Prover b(only);
      const ProveResult rf = a.prove(s, ProveBudget::nodes(200'000));
      const ProveResult ro = b.prove(s, ProveBudget::nodes(200'000));
      // The named alternative, proof-number thresholds alone: no claim either.
      ProverOptions pn = only;
      pn.prove_only_pn_thresholds = true;
      Prover c(pn);
      const ProveResult rp = c.prove(s, ProveBudget::nodes(200'000));
      CHECK(rp.status != ProofStatus::Disproven);
      if (rp.status == ProofStatus::Proven) {
        ++pn_proofs;
        CHECK(rf.status != ProofStatus::Disproven);
      }
      CHECK(ro.status != ProofStatus::Disproven);
      CHECK_EQ(ro.stats.disproofs_recorded, std::uint64_t{0});
      if (ro.exhausted) {
        ++exhausted;
        CHECK(rf.status != ProofStatus::Proven);
      }
      if (rf.status == ProofStatus::Proven) {
        ++proofs;
        if (ro.status != ProofStatus::Proven) {
          std::fprintf(stderr, "  full %s in %llu nodes; prove-only %s%s in %llu (memory %d) at %s, class %s/%s\n",
                       to_string(rf.proof).c_str(), static_cast<unsigned long long>(rf.stats.nodes),
                       proof_status_name(ro.status), ro.exhausted ? " (exhausted)" : "",
                       static_cast<unsigned long long>(ro.stats.nodes), ro.stats.stopped_by_memory ? 1 : 0,
                       describe_position(s).c_str(), seed_class_name(base.seeds),
                       completion_class_name(base.completion));
        }
        CHECK(ro.status == ProofStatus::Proven);
        if (ro.status == ProofStatus::Proven) CHECK_EQ(ro.proof.distance, rf.proof.distance);
      }
    }
  }
  std::fprintf(stderr, "  %d mates found by both, %d roots exhausted by prove-only; proof numbers alone: %d mates\n",
               proofs, exhausted, pn_proofs);
  CHECK(proofs > 10);
}

TEST(prover_lines_and_certificates_are_confirmed_by_e4_alone) {
  // Every proof's line ends in E4's checkmate; every certificate is confirmed
  // by E4 alone (t5d/certificate.hpp) - and a doctored one is not.
  int confirmed = 0;
  int undecided = 0;
  int lines = 0;
  std::uint64_t seed = 2300;
  std::vector<GameState> positions{mate_in_one(), mate_in_two(), cross_board_mate()};
  for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, 4, seed))) positions.push_back(s);
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    for (const GameState& s : positions) {
      ProverOptions o = base;
      o.depth = 3;
      o.record_certificate = true;
      Prover p(o);
      const ProveResult r = p.prove(s, ProveBudget::nodes(300'000));
      if (r.status != ProofStatus::Proven) continue;
      CHECK(r.certificate != nullptr);
      if (r.line_complete) {
        ++lines;
        GameState t = s;
        CHECK(replay(t, r.line));
        CHECK(mated(t));
        CHECK_EQ(static_cast<int>(r.line.size()), r.proof.distance);
      }
      CertificateCheckOptions co;
      co.max_positions = 20000;
      const CertificateCheck c = check_certificate(*r.certificate, s, r.proof.distance, co);
      if (c.verdict == CertificateVerdict::Undecided) {
        ++undecided;
        continue;
      }
      if (c.verdict != CertificateVerdict::Confirmed) {
        std::fprintf(stderr, "  certificate %s (%s) at %s\n", certificate_verdict_name(c.verdict), c.why.c_str(),
                     describe_position(s).c_str());
      }
      CHECK(c.verdict == CertificateVerdict::Confirmed);
      ++confirmed;
      // Doctored: a distance too short, an empty certificate.
      if (r.proof.distance > 1) {
        CHECK(check_certificate(*r.certificate, s, r.proof.distance - 2, co).verdict != CertificateVerdict::Confirmed);
      }
      CHECK(check_certificate(ProofCertificate{}, s, r.proof.distance, co).verdict == CertificateVerdict::Incomplete);
    }
  }
  std::fprintf(stderr, "  %d certificates confirmed, %d undecided; %d complete lines\n", confirmed, undecided, lines);
  CHECK(confirmed > 10);
  CHECK(lines > 10);
}

// ---- the review's findings (stage E9's adversarial review) -------------------------------------

namespace {

// White (no royal) to move at T2w on -0 and +0; +1 (White's, ahead) and -1
// (Black's, head Black-coloured) beside them; Black's king on +1's boards.
// White mates in one only by a J6 twin: the jump (+0T2)Pb2(-0T2)b2 alone
// stalemates, but after (-0T2)Pc2(-0T2)c3 has played -0 the same jump branches
// to a new timeline (+2) and mates (the soundness review's position).
GameState twin_position() {
  return game({"[4/4/2P1/4:-0:1:w]", "[4/4/2P1/4:-0:1:b]", "[4/4/2P1/4:-0:2:w]", "[4/1p2/1P2/4:+0:1:w]",
               "[4/1p2/1P2/4:+0:1:b]", "[4/1p2/1P2/4:+0:2:w]", "[4/4/1k2/4:+1:1:b]", "[4/4/1k2/4:+1:2:w]",
               "[4/4/1k2/4:+1:2:b]", "[4/4/1k2/4:+1:3:w]", "[4/4/2P1/4:-1:2:w]", "[4/4/4/4:-1:2:b]"},
              BoardSize{4, 4}, RuleOptions{}, Colour::White);
}

// Black to move, checkmated, on 32 boards (the review's position): E4 says so
// in a few hundred nodes.
GameState no_action_position() {
  return game({"[1k1k/4/4/1KNK:-4:2:w]", "[1k1k/4/4/1K1K:-4:2:b]", "[kk2/4/2K1/2N1:-3:2:w]", "[kk2/4/N1K1/4:-3:2:b]",
               "[1kk1/4/4/B3:-2:2:w]",   "[1kk1/4/4/4:-2:2:b]",    "[kk2/4/2K1/2N1:-1:2:w]", "[kk2/4/4/2N1:-1:2:b]",
               "[1k2/4/4/2N1:-1:3:w]",   "[1k2/4/4/4:-1:3:b]",     "[1k2/4/4/1KN1:-0:1:w]",  "[1k2/4/2K1/2N1:-0:1:b]",
               "[2k1/4/2K1/2N1:-0:2:w]", "[2k1/4/N1K1/4:-0:2:b]",  "[4/4/N1K1/4:-0:3:w]",    "[4/4/NK2/4:-0:3:b]",
               "[2k1/4/4/B1K1:+0:1:w]",  "[2k1/4/4/B3:+0:1:b]",    "[1kk1/4/4/B3:+0:2:w]",   "[1kk1/4/4/BK2:+0:2:b]",
               "[2k1/4/4/BK2:+0:3:w]",   "[2k1/4/4/1K2:+0:3:b]",   "[1k2/4/4/1KNK:+1:1:b]",  "[4/4/4/1KNK:+1:2:w]",
               "[4/4/4/2NK:+1:2:b]",     "[2k1/4/2K1/B1N1:+2:2:b]", "[4/4/2K1/B1N1:+2:3:w]", "[4/3K/4/B1N1:+2:3:b]",
               "[4/4/N1KK/4:+3:3:b]",    "[kk2/4/B1K1/2N1:+4:2:b]", "[1kk1/4/4/B2N:+5:2:b]", "[2k1/4/2N1/BK2:+6:3:b]"},
              BoardSize{4, 4}, RuleOptions{}, Colour::Black);
}

}  // namespace

TEST(prover_check_filter_counts_j6_twins) {
  // The twin's mate in one, found with the filter on and off, at 1 and 5
  // threads; the game tree agrees; the certificate is confirmed by E4 alone.
  const GameState s = twin_position();
  for (const bool filter : {false, true}) {
    for (const int threads : {1, 5}) {
      ProverOptions o = literal();
      o.depth = 1;
      o.check_filter = filter;
      o.threads = threads;
      o.record_certificate = true;
      Prover p(o);
      const ProveResult r = p.prove(s, ProveBudget::nodes(1'000'000));
      CHECK(r.status == ProofStatus::Proven);
      if (r.status != ProofStatus::Proven) continue;
      CHECK_EQ(r.proof.distance, 1);
      CHECK(check_certificate(*r.certificate, s, 1).verdict == CertificateVerdict::Confirmed);
    }
  }
  GameTree tree(20000, 60000, 2'000'000);
  const GameValue v = tree.solve(s, 1);
  CHECK(v.kind == GameValue::Kind::Win && v.distance == 1);
}

TEST(prover_an_attacker_with_no_legal_action_is_decided_at_once) {
  const GameState s = no_action_position();
  CHECK(detect_mate(s, Detector::Hypercuboid, SearchBudget::nodes(1'000'000)).verdict == Outcome::Checkmate);
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    ProverOptions o = base;
    o.depth = 1;
    Prover p(o);
    const ProveResult r = p.prove(s, ProveBudget::nodes(50'000));
    CHECK(r.status == ProofStatus::Disproven);
    CHECK(r.stats.nodes < 20'000);
    const std::optional<Proof> stored = p.proofs().find(s);
    CHECK(stored.has_value() && stored->kind == Proof::Kind::Loss && stored->distance == 0);
  }
}

TEST(prover_certificates_cover_proofs_from_other_threads_and_searches) {
  // At 5 threads a node may be proven from another thread's export; a second
  // search of the same root reads the root from the store. Every certificate
  // is still complete: confirmed by E4 alone, never "incomplete".
  int confirmed = 0;
  int undecided = 0;
  std::vector<GameState> positions{mate_in_one(), mate_in_two(), cross_board_mate(), twin_position()};
  for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, 3, 2350))) positions.push_back(s);
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    for (const GameState& s : positions) {
      ProverOptions o = base;
      o.depth = 3;
      o.threads = 5;
      o.record_certificate = true;
      Prover p(o);
      for (int search = 0; search < 2; ++search) {
        const ProveResult r = p.prove(s, ProveBudget::nodes(100'000));
        if (r.status != ProofStatus::Proven) break;
        CertificateCheckOptions co;
        co.max_positions = 20000;
        const CertificateCheck c = check_certificate(*r.certificate, s, r.proof.distance, co);
        if (c.verdict == CertificateVerdict::Undecided) {
          ++undecided;
          continue;
        }
        if (c.verdict != CertificateVerdict::Confirmed) {
          std::fprintf(stderr, "  search %d: %s (%s) at %s\n", search, certificate_verdict_name(c.verdict),
                       c.why.c_str(), describe_position(s).c_str());
        }
        CHECK(c.verdict == CertificateVerdict::Confirmed);
        ++confirmed;
      }
    }
  }
  std::fprintf(stderr, "  %d certificates confirmed at 5 threads (first and second searches), %d undecided\n",
               confirmed, undecided);
  CHECK(confirmed > 10);
}

TEST(prover_the_attacker_probe_changes_no_answer) {
  // E4's question at the attacker's turns is efficiency only: with and
  // without it, the same verdict wherever both decide (both classes, depths
  // 1 and 3), and the probe's answers held to the references.
  int compared = 0;
  std::uint64_t tests = 0;
  t5d_test::Tally tally;
  tally.work_cap = 100'000;
  std::uint64_t seed = 2400;
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    for (const int depth : {1, 3}) {
      for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, 3, seed++))) {
        ProverOptions on = base;
        on.depth = depth;
        ProverOptions off = on;
        off.attacker_probe = false;
        Prover a(on);
        Prover b(off);
        const ProveResult ra = a.prove(s, ProveBudget::nodes(100'000));
        const ProveResult rb = b.prove(s, ProveBudget::nodes(100'000));
        tests += ra.stats.attacker_tests;
        CHECK_EQ(rb.stats.attacker_tests, std::uint64_t{0});
        if (!tally.add(ra, s, on)) CHECK(false);
        if (ra.status == ProofStatus::Unknown || rb.status == ProofStatus::Unknown) continue;
        ++compared;
        if (ra.status != rb.status ||
            (ra.status == ProofStatus::Proven && ra.proof.distance != rb.proof.distance)) {
          std::fprintf(stderr, "  probe %s, none %s at %s\n", proof_status_name(ra.status),
                       proof_status_name(rb.status), describe_position(s).c_str());
          CHECK(false);
        }
      }
    }
  }
  tally.print("attacker probe");
  std::fprintf(stderr, "  %d compared, %llu attacker turns probed\n", compared,
               static_cast<unsigned long long>(tests));
  CHECK(compared > 30);
  CHECK(tests > 0);
}
