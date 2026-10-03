// The proof channel (t5d/prover.hpp; spec §11.1's redesign, part 2; register
// M14, H2, H15, H17, H19, H37, H39). Every position is written here from the
// rulebook or reached by random legal play from such positions - never taken
// from a game corpus or another implementation's test data.
//
// What is held, and against what:
//  * the semimove tree lists exactly the legal actions - the brute-force
//    reference's (test_reference.hpp) and E4's count - wherever the turn
//    decomposes;
//  * every proof is a mate the brute-force game tree (test_game_tree.hpp)
//    confirms within its distance, under every ruleset, at random budgets and
//    with 1-8 threads; and E4's exhaustive detector confirms the mates the
//    constructed positions describe;
//  * with the literal class (seeds All, completion Every) every disproof is the
//    game tree's "no mate within d" too; with the default class (checking
//    seeds, canonical completions) it is a reference of the test's own - the
//    restricted game, minimax over the class's actions computed independently
//    (the scanner's checking move made literal, the completion's definition
//    walked without any pruning) against every legal defence;
//  * no budget stop ever gives a proof or a disproof the references deny (H37:
//    every budget below the search's own need, on mates and non-mates);
//  * the stores get every proof and disproof at once (H19), and answer them.
//
// 5x5 boards unless said otherwise.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/prover.hpp"
#include "t5d/scanner.hpp"
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

using namespace t5d;
using t5d_test::game;
using t5d_test::GameTree;
using t5d_test::GameValue;
using t5d_test::checks_reference;
using t5d_test::CompletionRef;
using t5d_test::describe_position;
using t5d_test::playable;
using t5d_test::RestrictedGame;
using t5d_test::Tally;
using t5d_test::verify;

namespace {

constexpr BoardSize k5x5{5, 5};

RuleOptions guide_literal_q4() {
  RuleOptions r;
  r.moves_after_shift = false;  // Q4's Guide-literal reading: no move after the shift
  return r;
}

RuleOptions shift_rules() {
  RuleOptions r;
  r.moves_after_shift = true;  // Q4 as ruled (R8; the default from stage E7)
  return r;
}

ProverOptions literal() {
  ProverOptions o;
  o.seeds = SeedClass::All;
  o.completion = CompletionClass::Every;
  return o;
}


ProveResult prove(const GameState& s, const ProverOptions& o, std::uint64_t nodes = 2'000'000) {
  Prover p(o);
  return p.prove(s, ProveBudget::nodes(nodes));
}

// ---- positions (test_scanner.cpp's, with their derivations there) -----------------------

GameState mate_in_one(const RuleOptions& r) { return game({"[k4/4Q/1K3/5/5:0:1:w]"}, k5x5, r); }
GameState cross_board_mate(const RuleOptions& r) {
  return game({"[5/Q4/2p2/5/4K:-0:1:w]", "[k4/5/1K3/5/5:+0:1:w]"}, k5x5, r);
}
GameState stale_checkmate_in_one(const RuleOptions& r) {
  return game({"[k3K/5/5/5/4R:0:1:w]", "[5/5/5/4P/RR3:+1:1:w]"}, k5x5, r);
}
GameState mate_in_two(const RuleOptions& r) { return game({"[5/k4/4R/3R1/4K:0:1:w]"}, k5x5, r); }
GameState history_royal_mate(const RuleOptions& r) {
  return game({"[kp1p1/pp2N/5/5/4K:0:1:w]", "[kp1p1/pp2N/5/5/3K1:0:1:b]",
               "[kp3/pp1pN/5/5/3K1:0:2:w]", "[kp3/pp1pN/5/5/4K:0:2:b]",
               "[kp3/pp2N/3p1/5/4K:0:3:w]"},
              k5x5, r);
}


}  // namespace

// ---- the decomposition -------------------------------------------------------------------

TEST(prover_decomposes_only_where_order_does_not_matter) {
  CHECK(!decomposes(guide_literal_q4()));  // the shifting move last
  CHECK(decomposes(shift_rules()));
  RuleOptions r = shift_rules();
  r.activation = ActivationRule::Consecutive;
  CHECK(!decomposes(r));
  r = shift_rules();
  r.play_activates = true;
  CHECK(!decomposes(r));
  r = shift_rules();
  r.reactivation = Reactivation::Deferred;
  CHECK(!decomposes(r));
  r = shift_rules();
  r.castling_on_optional_boards = false;
  CHECK(!decomposes(r));
  r = shift_rules();
  r.mandatory = MandatoryBoards::AllActive;
  CHECK(decomposes(r));
}

namespace {

// The semimove tree's actions against the brute-force reference: the same
// distinct successors, and E4's count of them.
void semimove_tree_is_exact(const RuleOptions& rules, int games, int turns, std::uint64_t seed,
                            std::uint64_t limit,
                            int& positions, int& actions) {
  std::vector<GameState> all = t5d_test::sparse_positions(rules, games, turns, seed);
  for (GameState& s : t5d_test::variant_positions(rules, {"Standard - Two Timelines"}, 3, seed + 1)) {
    all.push_back(std::move(s));
  }
  Prover prover;
  for (const GameState& s : all) {
    // (The reference's own bookkeeping is quadratic in the successors: kept
    // to positions it can enumerate in a moment.)
    const t5d_test::Reference ref = t5d_test::reference_actions(s, limit);
    if (ref.overflow) continue;
    const std::vector<BuiltAction> got = prover.semimove_actions(s);
    ++positions;
    actions += static_cast<int>(got.size());
    bool ok = got.size() == ref.successors.size();
    std::unordered_map<std::uint64_t, std::vector<const GameState*>> by_key;
    for (const GameState& r : ref.successors) by_key[r.zobrist()].push_back(&r);
    for (const BuiltAction& a : got) {
      // Each one legal, through the state machine, and the reference's.
      GameState t = s;
      for (const Move& m : a.moves) t.apply(m);
      ok = ok && t.obligations_met() && royals_safe(t);
      t.submit(CheckRuling::none());
      ok = ok && same_position(t, a.successor);
      const auto it = by_key.find(a.successor.zobrist());
      ok = ok && it != by_key.end() &&
           std::any_of(it->second.begin(), it->second.end(),
                       [&](const GameState* r) { return same_position(*r, a.successor); });
    }
    const ActionCount n = count_legal_successors(s, 100000, Detector::Hypercuboid);
    ok = ok && n.stats.exhaustive && n.count == got.size();
    if (!ok) {
      std::fprintf(stderr, "  semimove tree %zu vs reference %zu at %s\n", got.size(),
                   ref.successors.size(), describe_position(s).c_str());
      CHECK(false);
    }
  }
}

}  // namespace

TEST(prover_semimove_tree_lists_exactly_the_legal_actions) {
  int positions = 0;
  int actions = 0;
  std::uint64_t seed = 300;
  for (const RuleOptions& base : t5d_test::every_ruleset()) {
    RuleOptions rules = base;
    rules.moves_after_shift = true;
    if (!decomposes(rules)) continue;
    semimove_tree_is_exact(rules, 1, 3, seed++, 5000, positions, actions);
  }
  std::fprintf(stderr, "  %d positions, %d actions\n", positions, actions);
  CHECK(positions > 200);
}

TEST(long_prover_semimove_tree_lists_exactly_the_legal_actions) {
  int positions = 0;
  int actions = 0;
  std::uint64_t seed = 400;
  for (const RuleOptions& base : t5d_test::every_ruleset()) {
    RuleOptions rules = base;
    rules.moves_after_shift = true;
    if (!decomposes(rules)) continue;
    semimove_tree_is_exact(rules, 4, 10, seed++, 30000, positions, actions);
  }
  std::fprintf(stderr, "  %d positions, %d actions\n", positions, actions);
  CHECK(positions > 2000);
}

TEST(prover_canonical_completion_is_its_definition) {
  // The prover's canonical completion of every move at the turn's start
  // against the definition walked with no pruning but the rulebook's lemma,
  // under every ruleset (both Q4 readings: under the Guide-literal one the
  // class is defined too, by the same walk).
  int seeds = 0;
  int completed = 0;
  int beyond = 0;
  CompletionRef ref;
  std::uint64_t seed = 500;
  for (const RuleOptions& base : t5d_test::every_ruleset()) {
    RuleOptions rules = base;
    rules.moves_after_shift = seed % 2 == 0;
    ProverOptions o;
    o.completion_max_nodes = 1'000'000;
    Prover prover(o);
    for (const GameState& s : t5d_test::sparse_positions(rules, 1, 4, seed++)) {
      MoveList moves;
      generate_moves(s, moves);
      for (const Move& m : moves) {
        ++seeds;
        const std::optional<BuiltAction> got = prover.canonical_completion(s, m);
        const std::optional<GameState> want = ref.complete(s, m);
        if (ref.capped) {
          ++beyond;
          continue;
        }
        if (got.has_value() != want.has_value() ||
            (got && !same_position(got->successor, *want))) {
          std::fprintf(stderr, "  completion of %s differs at %s\n", to_string(m, false).c_str(),
                       describe_position(s).c_str());
          CHECK(false);
        }
        if (got) ++completed;
      }
    }
  }
  std::fprintf(stderr, "  %d seeds, %d completed, %d beyond the reference's cap\n", seeds, completed,
               beyond);
  CHECK(completed > 1000);
}

// ---- constructed mates ---------------------------------------------------------------------

TEST(prover_constructed_mates_are_proven) {
  for (const RuleOptions& rules : {guide_literal_q4(), shift_rules()}) {
    struct Case {
      GameState s;
      int distance;
    };
    const std::vector<Case> cases{{mate_in_one(rules), 1},
                                  {cross_board_mate(rules), 1},
                                  {stale_checkmate_in_one(rules), 1},
                                  {history_royal_mate(rules), 1},
                                  {mate_in_two(rules), 3}};
    for (const Case& c : cases) {
      for (const ProverOptions& o : {ProverOptions{}, literal()}) {
        const ProveResult r = prove(c.s, o);
        CHECK(r.status == ProofStatus::Proven);
        CHECK(r.proof == Proof::win(c.distance));
        CHECK_EQ(verify(r, c.s, o), 1);
        // The proof's first action is a legal action.
        REQUIRE(!r.first_action.empty());
        GameState t = c.s;
        for (const Move& m : r.first_action) t.apply(m);
        CHECK(t.obligations_met());
        CHECK(royals_safe(t));
        t.submit();
        if (c.distance == 1) {
          CHECK(detect_mate(t, Detector::Naive).verdict == Outcome::Checkmate);
        }
      }
    }
  }
}

TEST(prover_a_shallower_depth_disproves_the_mate_in_two) {
  const GameState s = mate_in_two(shift_rules());
  for (ProverOptions o : {ProverOptions{}, literal()}) {
    o.depth = 2;
    const ProveResult r = prove(s, o);
    CHECK(r.status == ProofStatus::Disproven);
    CHECK_EQ(verify(r, s, o), 1);
    o.depth = 3;
    CHECK(prove(s, o).status == ProofStatus::Proven);
  }
}

TEST(prover_a_side_with_no_legal_action_cannot_mate) {
  // The side to move checkmated (E4's position) or stalemated: disproven.
  for (const GameState& s : {game({"[k4/5/5/5/R4:0:1:b]", "[5/5/5/5/RR3:+1:1:b]"}, k5x5, shift_rules()),
                             game({"[k4/5/5/5/5:0:1:b]", "[1R3/RR3/5/5/5:+1:2:w]"}, k5x5, shift_rules())}) {
    const ProveResult r = prove(s, literal());
    CHECK(r.status == ProofStatus::Disproven);
  }
}

TEST(prover_proofs_and_disproofs_go_to_the_stores_h19) {
  const GameState s = mate_in_two(shift_rules());
  MemorySolvedStore proofs;
  DisproofStore disproofs;
  {
    Prover p(ProverOptions{}, &proofs, &disproofs);
    const ProveResult r = p.prove(s, ProveBudget::nodes(2'000'000));
    REQUIRE(r.status == ProofStatus::Proven);
    CHECK(r.stats.proofs_exported > 0);
  }
  // The root's proof and the defender's losses below it are in the store.
  const std::optional<Proof> root = proofs.find(s);
  REQUIRE(root.has_value());
  CHECK(*root == Proof::win(3));
  bool loss = false;
  for (const MemorySolvedStore::Entry& e : proofs.entries()) {
    if (e.proof.kind == Proof::Kind::Loss) loss = true;
  }
  CHECK(loss);
  // A new prover over the same stores answers from them at once.
  Prover again(ProverOptions{}, &proofs, &disproofs);
  const ProveResult r2 = again.prove(s, ProveBudget::nodes(2'000'000));
  CHECK(r2.status == ProofStatus::Proven);
  CHECK(r2.stats.store_hits >= 1);
  CHECK(r2.stats.expansions == 0);
  // A disproof is recorded with its depth and class.
  ProverOptions o;
  o.depth = 2;
  Prover shallow(o, &proofs, &disproofs);
  // (The stores hold Win(3): not within 2, so the search runs.)
  const ProveResult r3 = shallow.prove(s, ProveBudget::nodes(2'000'000));
  CHECK(r3.status == ProofStatus::Disproven);
  CHECK_EQ(disproofs.find(s, DisproofClass{true, o.seeds, o.completion}), 2);
  CHECK_EQ(disproofs.find(s, DisproofClass{true, SeedClass::All, CompletionClass::Every}), -1);
}

// ---- random play: every proof and disproof checked ---------------------------------------------

namespace {

void random_play(const RuleOptions& rules, const ProverOptions& o, int games, int turns,
                 std::uint64_t seed, std::uint64_t nodes, Tally& tally) {
  for (const GameState& s : playable(t5d_test::sparse_positions(rules, games, turns, seed))) {
    const ProveResult r = prove(s, o, nodes);
    if (!tally.add(r, s, o)) CHECK(false);
  }
}

}  // namespace

TEST(prover_literal_class_agrees_with_the_game_tree) {
  // Seeds All, completion Every: exactly "mate within d", both ways.
  Tally tally;
  std::uint64_t seed = 600;
  for (const int depth : {1, 3}) {
    ProverOptions o = literal();
    o.depth = depth;
    random_play(shift_rules(), o, 1, 4, seed++, 200'000, tally);
    random_play(guide_literal_q4(), o, 1, 3, seed++, 200'000, tally);  // listings
  }
  tally.print("literal class");
  CHECK(tally.proven > 3);
  CHECK(tally.disproven > 20);
}

TEST(prover_default_class_agrees_with_its_references) {
  // Checking seeds, canonical completions: proofs against the game tree,
  // disproofs against the restricted game.
  Tally tally;
  std::uint64_t seed = 700;
  for (const int depth : {1, 3}) {
    ProverOptions o;
    o.depth = depth;
    o.completion_max_nodes = 1'000'000;
    random_play(shift_rules(), o, 1, 4, seed++, 200'000, tally);
  }
  tally.print("default class");
  CHECK(tally.proven > 3);
  CHECK(tally.disproven > 20);
}

TEST(prover_every_ruleset_proves_only_mates) {
  // Every ruleset, with and without Q4's moves after the shift: every proof
  // checked; disproofs within a small reference budget (the full checks are
  // the tests above and the long_ one).
  Tally tally;
  tally.work_cap = 20'000;
  std::uint64_t seed = 800;
  for (const RuleOptions& base : t5d_test::every_ruleset()) {
    if (base.moves_after_shift != RuleOptions{}.moves_after_shift) continue;  // both below
    for (const bool shift : {false, true}) {
      RuleOptions rules = base;
      rules.moves_after_shift = shift;
      ProverOptions o;
      o.depth = 3;
      random_play(rules, o, 1, 3, seed++, 20'000, tally);
      ProverOptions l = literal();
      l.depth = 1;
      random_play(rules, l, 1, 3, seed++, 20'000, tally);
    }
  }
  tally.print("every ruleset");
  CHECK(tally.proven > 10);
}

TEST(long_prover_random_play_agrees_with_the_references) {
  Tally literal_tally;
  Tally default_tally;
  std::uint64_t seed = 900;
  const auto start = std::chrono::steady_clock::now();
  const std::vector<RuleOptions> rulesets = t5d_test::every_ruleset();
  for (std::size_t i = 0; i < rulesets.size(); ++i) {
    const RuleOptions& rules = rulesets[i];  // Q4's other reading included: listings
    // The default ruleset at full size (about ten minutes); every alternative
    // at a quarter of the positions and budgets, so the test stays within
    // about half an hour.
    const bool full = i == 0;
    const int games = full ? 2 : 1;
    const int turns = full ? 8 : 6;
    const std::uint64_t nodes = full ? 2'000'000 : 500'000;
    literal_tally.work_cap = default_tally.work_cap = full ? 4'000'000 : 1'000'000;
    for (const int depth : {1, 3, 5}) {
      if (depth <= 3) {
        ProverOptions l = literal();
        l.depth = depth;
        random_play(rules, l, games, turns, seed++, nodes, literal_tally);
      }
      ProverOptions d;
      d.depth = depth;
      d.completion_max_nodes = 1'000'000;
      random_play(rules, d, games, turns, seed++, nodes, default_tally);
    }
    // Progress: a long test, so that a run can be followed.
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "  ruleset %zu/%zu (%s) done at %.0f s\n", i + 1, rulesets.size(),
                 describe(rules).c_str(), s);
  }
  literal_tally.print("literal class");
  default_tally.print("default class");
  CHECK(literal_tally.proven > 20);
  CHECK(default_tally.proven > 20);
}

// ---- budgets (H37) and threads ---------------------------------------------------------------

namespace {

// Every budget below the search's own need: a decided answer only where the
// references agree, never a proof the game tree denies.
void budget_property(const std::vector<GameState>& positions, const ProverOptions& o,
                     int& stops, int& checked) {
  for (const GameState& s : positions) {
    Prover full_prover(o);
    const ProveResult full = full_prover.prove(s, ProveBudget::nodes(200'000));
    if (full.status == ProofStatus::Unknown) continue;
    const std::uint64_t need = full.stats.nodes;
    const std::uint64_t step = std::max<std::uint64_t>(1, need / 60);
    for (std::uint64_t b = 0; b < need; b += step) {
      Prover p(o);
      const ProveResult r = p.prove(s, ProveBudget::nodes(b));
      ++checked;
      if (r.status == ProofStatus::Unknown) {
        ++stops;
        continue;
      }
      // Decided within a smaller budget: it must be the same answer.
      if (r.status != full.status) {
        std::fprintf(stderr, "  budget %llu: %s, the full search %s, at %s\n",
                     static_cast<unsigned long long>(b), proof_status_name(r.status),
                     proof_status_name(full.status), describe_position(s).c_str());
        CHECK(false);
      }
    }
  }
}

}  // namespace

TEST(prover_no_budget_stop_gives_a_verdict_h37) {
  std::vector<GameState> positions;
  for (const GameState& s : {mate_in_one(shift_rules()), cross_board_mate(shift_rules()),
                             mate_in_two(shift_rules()), stale_checkmate_in_one(shift_rules())}) {
    positions.push_back(s);
  }
  const std::vector<GameState> random = playable(t5d_test::sparse_positions(shift_rules(), 1, 4, 1000));
  positions.insert(positions.end(), random.begin(), random.begin() + std::min<std::ptrdiff_t>(20, static_cast<std::ptrdiff_t>(random.size())));
  int stops = 0;
  int checked = 0;
  budget_property(positions, ProverOptions{}, stops, checked);
  budget_property(positions, literal(), stops, checked);
  std::fprintf(stderr, "  %d budgets, %d stopped unknown\n", checked, stops);
  CHECK(stops > 100);
}

TEST(prover_one_thread_is_deterministic_h15) {
  for (const GameState& s : playable(t5d_test::sparse_positions(shift_rules(), 1, 4, 1100))) {
    const ProveResult a = prove(s, ProverOptions{}, 3000);
    const ProveResult b = prove(s, ProverOptions{}, 3000);
    CHECK(a.status == b.status);
    CHECK_EQ(a.stats.nodes, b.stats.nodes);
    CHECK_EQ(a.stats.expansions, b.stats.expansions);
    CHECK_EQ(a.stats.tree_nodes, b.stats.tree_nodes);
  }
}

namespace {

void threads_property(int threads, bool by_clock, std::uint64_t seed, Tally& tally) {
  std::vector<GameState> positions{mate_in_two(shift_rules()), cross_board_mate(shift_rules())};
  const std::vector<GameState> random = playable(t5d_test::sparse_positions(shift_rules(), 1, 3, seed));
  positions.insert(positions.end(), random.begin(), random.end());
  MemorySolvedStore proofs;  // shared by every search of this thread count
  DisproofStore disproofs;
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    ProverOptions o = base;
    o.threads = threads;
    o.depth = 3;
    for (const GameState& s : positions) {
      Prover p(o, &proofs, &disproofs);
      const ProveResult r = by_clock ? p.prove(s, ProveBudget::time(std::chrono::milliseconds(20)))
                                     : p.prove(s, ProveBudget::nodes(60'000));
      if (!tally.add(r, s, o)) CHECK(false);
      // And the same as one thread's, where both decide.
      ProverOptions one = o;
      one.threads = 1;
      Prover q(one);
      const ProveResult r1 = q.prove(s, ProveBudget::nodes(60'000));
      if (r.status != ProofStatus::Unknown && r1.status != ProofStatus::Unknown) {
        CHECK(r.status == r1.status);
      }
    }
  }
}

}  // namespace

TEST(prover_threads_agree_with_the_references) {
  Tally tally;
  tally.work_cap = 50'000;  // every proof checked; disproofs within this budget
  for (const int threads : {2, 5, 8}) {
    threads_property(threads, false, 1200 + static_cast<std::uint64_t>(threads), tally);
    threads_property(threads, true, 1300 + static_cast<std::uint64_t>(threads), tally);
  }
  tally.print("threads 2-8");
  CHECK(tally.proven > 4);
}
