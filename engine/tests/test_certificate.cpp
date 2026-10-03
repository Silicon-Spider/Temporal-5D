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

namespace {

// The literal class with and without the filter: the same verdict wherever
// both decide; the filter's answers held to the references too.
void filter_agrees(const std::vector<RuleOptions>& rulesets, const std::vector<int>& depths, int turns,
                   std::uint64_t nodes, int& compared) {
  std::uint64_t cut = 0;
  t5d_test::Tally tally;
  tally.work_cap = 100'000;
  std::uint64_t seed = 2100;
  for (const RuleOptions& rules : rulesets) {
    for (const int depth : depths) {
      for (const GameState& s : playable(t5d_test::sparse_positions(rules, 1, turns, seed++))) {
        ProverOptions on = literal();
        on.depth = depth;
        on.check_filter = true;  // off by default: the test switches it on
        ProverOptions off = on;
        off.check_filter = false;
        Prover a(on);
        Prover b(off);
        const ProveResult ra = a.prove(s, ProveBudget::nodes(nodes));
        const ProveResult rb = b.prove(s, ProveBudget::nodes(nodes));
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
}

}  // namespace

TEST(prover_check_filter_changes_no_answer) {
  // (That it cuts, and never a checking action: the test below.)
  int compared = 0;
  filter_agrees({RuleOptions{}}, {1, 3}, 2, 50'000, compared);
  CHECK(compared > 20);
}

TEST(long_prover_check_filter_changes_no_answer) {
  int compared = 0;
  filter_agrees({RuleOptions{}, guide_literal_q4()}, {1, 3}, 4, 300'000, compared);
  CHECK(compared > 50);
}

namespace {

// Prove-only never claims a disproof and records none; at the full search's
// budget it proves every mate the full search proves, at the same distance
// (its thresholds are df-pn's usual ones). The named alternative,
// proof-number thresholds alone, claims nothing either.
void prove_only_property(int turns, int min_proofs) {
  int proofs = 0;
  int exhausted = 0;
  int pn_proofs = 0;
  std::uint64_t seed = 2200;
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, turns, seed++))) {
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
  CHECK(proofs > min_proofs);
}

}  // namespace

TEST(prover_prove_only_finds_the_full_searchs_mates) { prove_only_property(2, 5); }

TEST(long_prover_prove_only_finds_the_full_searchs_mates) { prove_only_property(4, 20); }

namespace {

// Every proof's line ends in E4's checkmate; every certificate is confirmed
// by E4 alone (t5d/certificate.hpp) - and a doctored one is not.
void lines_property(int turns, int minimum) {
  int confirmed = 0;
  int undecided = 0;
  int lines = 0;
  std::uint64_t seed = 2300;
  std::vector<GameState> positions{mate_in_one(), mate_in_two(), cross_board_mate()};
  for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, turns, seed))) {
    positions.push_back(s);
  }
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
  CHECK(confirmed > minimum);
  CHECK(lines > minimum);
}

}  // namespace

TEST(prover_lines_and_certificates_are_confirmed_by_e4_alone) { lines_property(3, 8); }

TEST(long_prover_lines_and_certificates_are_confirmed_by_e4_alone) { lines_property(4, 10); }

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

namespace {

// E4's question at the attacker's turns is efficiency only: with and
// without it, the same verdict wherever both decide (both classes, depths 1
// and 3), and the probe's answers held to the references.
void probe_property(int turns, int minimum) {
  int compared = 0;
  std::uint64_t tests = 0;
  t5d_test::Tally tally;
  tally.work_cap = 100'000;
  std::uint64_t seed = 2400;
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    for (const int depth : {1, 3}) {
      for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, turns, seed++))) {
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
  CHECK(compared > minimum);
  CHECK(tests > 0);
}

}  // namespace

TEST(prover_the_attacker_probe_changes_no_answer) { probe_property(2, 20); }

TEST(long_prover_the_attacker_probe_changes_no_answer) { probe_property(4, 60); }

// ---- the check filter's soundness, against the reference (the review's M11) ----------------------

namespace {

struct FilterTally {
  int positions = 0;
  int applied = 0;
  int cutting = 0;          // the filter applies and no branch is assumed left: it may cut
  int checking = 0;         // legal actions that leave the defender in check
  int cut = 0;              // legal actions the filter cut (none of them checking)
  int by_void = 0;          // checking actions kept only by a void stop's capability
};

// At `s` (the attacker to move, its last half-turn): the literal class's
// tree with the filter keeps every legal action that leaves the defender in
// check (the reference's actions and R1's check status after each), and
// holds only legal actions.
void filter_keeps_every_check(const GameState& s, FilterTally& t) {
  const t5d_test::Reference ref = t5d_test::reference_actions(s, 4000);
  if (ref.overflow) return;
  ProverOptions o = literal();
  o.check_filter = true;
  Prover p(o);
  const CheckFilterView v = p.check_filter_view(s);
  const TreeActions kept = p.tree_actions(s, true);
  ++t.positions;
  if (v.applies) ++t.applied;
  if (v.applies && !v.branch) ++t.cutting;
  for (const BuiltAction& a : kept.actions) {
    const bool legal = std::any_of(ref.successors.begin(), ref.successors.end(),
                                   [&](const GameState& r) { return same_position(r, a.successor); });
    CHECK(legal);
  }
  for (const GameState& r : ref.successors) {
    const auto it = std::find_if(kept.actions.begin(), kept.actions.end(),
                                 [&](const BuiltAction& a) { return same_position(a.successor, r); });
    const bool check = in_check(r);
    if (!check) {
      if (it == kept.actions.end()) ++t.cut;
      continue;
    }
    ++t.checking;
    if (it == kept.actions.end()) {
      std::fprintf(stderr, "  the filter cut a checking action at %s\n", describe_position(s).c_str());
      CHECK(false);
      continue;
    }
    // Kept by a void stop alone: every capable move of the action capable
    // only through one (the turn start's moves; a twin is capable anyway).
    if (!v.applies || v.branch) continue;
    bool any = false;
    bool only_void = true;
    for (const Move& m : it->moves) {
      for (std::size_t i = 0; i < v.moves.size(); ++i) {
        if (!(v.moves[i] == m) || v.capable[i] == 0) continue;
        any = true;
        if (v.by_void[i] == 0) only_void = false;
      }
    }
    if (any && only_void) ++t.by_void;
  }
}

}  // namespace

// Three timelines, White (no royal) to move on each at T1: a rook on -1, pawns
// that only block each other's jumps (no non-branching jump exists - J6's
// rule would otherwise leave every action uncut), Black's king on +1's a4.
// Rook to a4 on -1, any move on 0 and +1: check along the timelines at the
// next turn start - through 0's new board, which only the move on 0 makes. No
// move of that action gives check alone: the rook's move is check-capable
// only through the line it stops at 0's void (the review's M11).
GameState void_stop_check() {
  return game({"[4/4/4/R2P:-1:1:w]", "[4/4/4/P2P:0:1:w]", "[k3/4/4/3P:+1:1:w]"}, BoardSize{4, 4}, RuleOptions{},
              Colour::White);
}

TEST(prover_check_filter_keeps_every_checking_action) {
  // The filter at the attacker's last half-turn, as the prover builds it,
  // against the reference: every legal action that leaves the defender in
  // check (R1, after submission) is kept; it cuts some that do not; and some
  // checks it keeps only through a line stopped at a void another move fills.
  FilterTally t;
  filter_keeps_every_check(void_stop_check(), t);
  CHECK(t.by_void > 0);
  filter_keeps_every_check(twin_position(), t);
  std::uint64_t seed = 5000;
  for (int round = 0; round < 2; ++round) {
    for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, 6, seed++))) {
      filter_keeps_every_check(s, t);
    }
  }
  std::fprintf(stderr, "  %d positions, the filter applied at %d (could cut at %d); %d checking actions kept, %d "
               "by a void stop alone; %d actions cut\n",
               t.positions, t.applied, t.cutting, t.checking, t.by_void, t.cut);
  CHECK(t.positions > 50);
  CHECK(t.checking > 100);
  // It cuts rarely - nearly every move attacks some place a board may come,
  // its capability through a void stop - but it does cut.
  CHECK(t.cut > 0);
}

TEST(prover_a_certificate_missing_any_defence_is_not_confirmed) {
  // Mates in three half-turns: the attacker's answer to each of the
  // defender's legal actions dropped from the certificate, one at a time -
  // never confirmed (the review's M5: a checker that skipped one defence
  // would confirm it).
  int doctored = 0;
  int certificates = 0;
  std::vector<GameState> positions{mate_in_two()};
  for (const GameState& s : playable(t5d_test::sparse_positions(RuleOptions{}, 1, 4, 2370))) positions.push_back(s);
  CertificateCheckOptions co;
  co.max_positions = 20000;
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    for (const GameState& s : positions) {
      ProverOptions o = base;
      o.depth = 3;
      o.record_certificate = true;
      Prover p(o);
      const ProveResult r = p.prove(s, ProveBudget::nodes(200'000));
      if (r.status != ProofStatus::Proven || r.proof.distance != 3) continue;
      if (check_certificate(*r.certificate, s, 3, co).verdict != CertificateVerdict::Confirmed) continue;
      const CertificateStep* first = r.certificate->find(s);
      REQUIRE(first != nullptr);
      GameState d = s;
      for (const Move& m : first->action) d.apply(m);
      d.submit(CheckRuling::none());
      const ActionList defences = list_legal_actions(d, 100000, Detector::Hypercuboid, SearchBudget::nodes(3'000'000));
      if (!defences.stats.exhaustive || defences.reached_cap || defences.actions.size() < 2) continue;
      ++certificates;
      for (const LegalAction& a : defences.actions) {
        ProofCertificate cut;
        bool dropped = false;
        for (const CertificateStep& step : r.certificate->steps()) {
          if (same_position(step.position, a.result)) {
            dropped = true;
            continue;
          }
          cut.add(step.position, step.action, step.distance);
        }
        if (!dropped) continue;
        ++doctored;
        const CertificateCheck c = check_certificate(cut, s, 3, co);
        if (c.verdict == CertificateVerdict::Confirmed) {
          std::fprintf(stderr, "  confirmed without the answer to a defence at %s\n", describe_position(s).c_str());
        }
        CHECK(c.verdict != CertificateVerdict::Confirmed);
      }
    }
  }
  std::fprintf(stderr, "  %d certificates, %d doctored ones refused\n", certificates, doctored);
  CHECK(certificates > 2);
  CHECK(doctored > 10);
}

TEST(prover_a_proof_from_another_provers_store_is_named_uncertifiable) {
  // A prover answering from a store another prover filled holds no
  // certificate for it: the check says so and names the position - never
  // confirmed (the review's finding 4).
  ConcurrentSolvedStore shared;
  ProverOptions o;
  o.depth = 3;
  o.record_certificate = true;
  Prover first(o, &shared);
  const GameState s = mate_in_two();
  REQUIRE(first.prove(s, ProveBudget::nodes(200'000)).status == ProofStatus::Proven);
  Prover second(o, &shared);
  const ProveResult r = second.prove(s, ProveBudget::nodes(200'000));
  REQUIRE(r.status == ProofStatus::Proven);
  REQUIRE(r.certificate != nullptr);
  const CertificateCheck c = check_certificate(*r.certificate, s, r.proof.distance);
  CHECK(c.verdict == CertificateVerdict::Incomplete);
  CHECK(c.why.find("H30 fingerprint") != std::string::npos);
  CHECK(c.why.find("cannot be certified") != std::string::npos);
  // The first prover's certificate is complete.
  const ProveResult again = first.prove(s, ProveBudget::nodes(200'000));
  REQUIRE(again.certificate != nullptr);
  CHECK(check_certificate(*again.certificate, s, again.proof.distance).verdict == CertificateVerdict::Confirmed);
}
