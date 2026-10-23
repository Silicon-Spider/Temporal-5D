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
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "t5d/certificate.hpp"
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
  // The prover's own trees, as it expands them (expand_turn, expand_semi):
  // the defender's, and the attacker's in the literal class (the semimove
  // tree entered once - every legal action exactly once).
  ProverOptions literal;
  literal.seeds = SeedClass::All;
  literal.completion = CompletionClass::Every;
  for (const GameState& s : all) {
    // (The reference's own bookkeeping is quadratic in the successors: kept
    // to positions it can enumerate in a moment.)
    const t5d_test::Reference ref = t5d_test::reference_actions(s, limit);
    if (ref.overflow) continue;
    ++positions;
    const ActionCount n = count_legal_successors(s, 100000, Detector::Hypercuboid);
    std::unordered_map<std::uint64_t, std::vector<const GameState*>> by_key;
    for (const GameState& r : ref.successors) by_key[r.zobrist()].push_back(&r);
    for (const bool as_attacker : {false, true}) {
      Prover prover(as_attacker ? literal : ProverOptions{});
      const TreeActions got = prover.tree_actions(s, as_attacker);
      if (as_attacker) actions += static_cast<int>(got.actions.size());
      bool ok = got.actions.size() == ref.successors.size();
      for (const BuiltAction& a : got.actions) {
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
      ok = ok && n.stats.exhaustive && n.count == got.actions.size();
      // Every legal action - a move sequence, branches in the order made
      // (two orders of two branches may reach one position: two actions) -
      // is a path of the tree, once: the attacker's tree holds exactly the
      // reference's actions; the defender's, its witness besides.
      std::set<std::string> keys;
      const bool even = s.multiverse().even();
      for (const BuiltAction& a : got.actions) keys.insert(t5d_test::Reference::key(a.moves, even));
      for (const auto& [kept, again] : got.repeated) keys.insert(t5d_test::Reference::key(again, even));
      const std::size_t witness = !as_attacker && !ref.actions.empty() ? 1 : 0;
      ok = ok && keys == ref.actions && got.paths == ref.actions.size() + witness;
      if (!ok) {
        std::fprintf(stderr, "  %s tree %zu (%zu paths) vs reference %zu at %s\n", as_attacker ? "attacker" : "defender",
                     got.actions.size(), got.paths, ref.successors.size(), describe_position(s).c_str());
        for (std::size_t k = 0; k < got.repeated.size() && k < 4; ++k) {
          std::fprintf(stderr, "    repeated: %s | %s\n",
                       action_text(got.repeated[k].first, s.multiverse().even()).c_str(),
                       action_text(got.repeated[k].second, s.multiverse().even()).c_str());
        }
        CHECK(false);
      }
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
    semimove_tree_is_exact(rules, 1, 3, seed++, 2000, positions, actions);
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
  CHECK_EQ(disproofs.find(s, DisproofClass{true, o.seeds, o.completion, disproof_shape(o)}), 2);
  ProverOptions literal_o = o;
  literal_o.seeds = SeedClass::All;
  literal_o.completion = CompletionClass::Every;
  CHECK_EQ(disproofs.find(s, DisproofClass{true, SeedClass::All, CompletionClass::Every, disproof_shape(literal_o)}),
           -1);
  // Every class-shaping option is in the key: another completion cap, other
  // values, the check filter - another class, no disproof shared.
  for (int k = 0; k < 4; ++k) {
    ProverOptions other = o;
    if (k == 0) other.completion_max_nodes = 17;
    if (k == 1) other.values.value[1] += 1;
    if (k == 2) other.check_filter = !other.check_filter;
    if (k == 3) other.promotions_forcing = !other.promotions_forcing;
    CHECK(disproof_shape(other) != disproof_shape(o));
    CHECK_EQ(disproofs.find(s, DisproofClass{true, o.seeds, o.completion, disproof_shape(other)}), -1);
  }
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
    random_play(shift_rules(), o, 1, 2, seed++, 60'000, tally);
    random_play(guide_literal_q4(), o, 1, 2, seed++, 60'000, tally);  // listings
  }
  tally.print("literal class");
  CHECK(tally.proven > 2);
  CHECK(tally.disproven > 10);
}

TEST(long_prover_literal_class_agrees_with_the_game_tree) {
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

namespace {

// Every ruleset, with and without Q4's moves after the shift: every proof
// checked; disproofs within a small reference budget (the full checks are
// the tests above and the long_ ones).
void every_ruleset_property(int turns, std::uint64_t nodes, Tally& tally) {
  tally.work_cap = nodes;
  std::uint64_t seed = 800;
  for (const RuleOptions& base : t5d_test::every_ruleset()) {
    if (base.moves_after_shift != RuleOptions{}.moves_after_shift) continue;  // both below
    for (const bool shift : {false, true}) {
      RuleOptions rules = base;
      rules.moves_after_shift = shift;
      ProverOptions o;
      o.depth = 3;
      random_play(rules, o, 1, turns, seed++, nodes, tally);
      ProverOptions l = literal();
      l.depth = 1;
      random_play(rules, l, 1, turns, seed++, nodes, tally);
    }
  }
}

}  // namespace

TEST(prover_every_ruleset_proves_only_mates) {
  Tally tally;
  every_ruleset_property(2, 8'000, tally);
  tally.print("every ruleset");
  CHECK(tally.proven > 5);
}

TEST(long_prover_every_ruleset_proves_only_mates) {
  Tally tally;
  every_ruleset_property(3, 20'000, tally);
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
void budget_property(const std::vector<GameState>& positions, const ProverOptions& o, std::uint64_t steps,
                     int& stops, int& checked) {
  for (const GameState& s : positions) {
    Prover full_prover(o);
    const ProveResult full = full_prover.prove(s, ProveBudget::nodes(200'000));
    if (full.status == ProofStatus::Unknown) continue;
    const std::uint64_t need = full.stats.nodes;
    const std::uint64_t step = std::max<std::uint64_t>(1, need / steps);
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

namespace {

void no_budget_stop(std::ptrdiff_t random_positions, std::uint64_t steps, int& stops, int& checked) {
  std::vector<GameState> positions;
  for (const GameState& s : {mate_in_one(shift_rules()), cross_board_mate(shift_rules()),
                             mate_in_two(shift_rules()), stale_checkmate_in_one(shift_rules())}) {
    positions.push_back(s);
  }
  const std::vector<GameState> random = playable(t5d_test::sparse_positions(shift_rules(), 1, 4, 1000));
  positions.insert(positions.end(), random.begin(),
                   random.begin() + std::min<std::ptrdiff_t>(random_positions, static_cast<std::ptrdiff_t>(random.size())));
  budget_property(positions, ProverOptions{}, steps, stops, checked);
  budget_property(positions, literal(), steps, stops, checked);
}

}  // namespace

TEST(prover_no_budget_stop_gives_a_verdict_h37) {
  int stops = 0;
  int checked = 0;
  no_budget_stop(3, 20, stops, checked);
  std::fprintf(stderr, "  %d budgets, %d stopped unknown\n", checked, stops);
  CHECK(stops > 30);
}

TEST(long_prover_no_budget_stop_gives_a_verdict_h37) {
  int stops = 0;
  int checked = 0;
  no_budget_stop(20, 60, stops, checked);
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

void threads_property(int threads, bool by_clock, std::uint64_t seed, int turns, std::uint64_t nodes,
                      Tally& tally) {
  std::vector<GameState> positions{mate_in_two(shift_rules()), cross_board_mate(shift_rules())};
  const std::vector<GameState> random = playable(t5d_test::sparse_positions(shift_rules(), 1, turns, seed));
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
                                     : p.prove(s, ProveBudget::nodes(nodes));
      if (!tally.add(r, s, o)) CHECK(false);
      // And the same as one thread's, where both decide.
      ProverOptions one = o;
      one.threads = 1;
      Prover q(one);
      const ProveResult r1 = q.prove(s, ProveBudget::nodes(nodes));
      if (r.status != ProofStatus::Unknown && r1.status != ProofStatus::Unknown) {
        CHECK(r.status == r1.status);
      }
    }
  }
}

}  // namespace

TEST(prover_threads_agree_with_the_references) {
  // M13's 5 threads, by nodes (by the clock, and 2 and 8 threads: the long_ test).
  Tally tally;
  tally.work_cap = 10'000;  // every proof checked; disproofs within this budget
  threads_property(5, false, 1205, 1, 30'000, tally);
  tally.print("threads 5");
  CHECK(tally.proven > 2);
}

TEST(long_prover_threads_agree_with_the_references) {
  Tally tally;
  tally.work_cap = 50'000;  // every proof checked; disproofs within this budget
  for (const int threads : {2, 5, 8}) {
    threads_property(threads, false, 1200 + static_cast<std::uint64_t>(threads), 3, 60'000, tally);
    threads_property(threads, true, 1300 + static_cast<std::uint64_t>(threads), 3, 60'000, tally);
  }
  tally.print("threads 2-8");
  CHECK(tally.proven > 4);
}

// ---- H19 under TT churn (6f; §11.3): proofs never live only in the TT ---------------------
//
// §13.1 6f: "verify the proof-export invariant under TT churn for Hunt and the
// settling scans (§11.3)". The TT is a lossy cache of estimates; with 1 or 2
// entries it keeps nothing, so every fact a search relies on must be in the
// stores - proofs in the solved store, disproofs (class-relative) in the
// disproof store, the moment they are found. The settling scans are full
// df-pn (prove_only off) over external stores; their wrapper (E10's) is not
// on main, so these tests drive the prover as it would (build-log issue 402).

namespace {

// Every entry of a solved store against the brute-force game tree: a Win(k)
// mates within k, a Loss(k) is mated within k, a Draw is a stalemate now (at
// most `most` entries; the
// game tree's budgets small, so an entry it cannot reach is counted, not
// failed).
struct StoreCheck {
  int entries = 0;
  int confirmed = 0;
  int beyond = 0;
};
void check_store(const MemorySolvedStore& proofs, std::size_t most, StoreCheck& c) {
  std::size_t n = 0;
  for (const MemorySolvedStore::Entry& e : proofs.entries()) {
    if (n++ >= most) break;
    ++c.entries;
    // A proof, never an estimate or a disproof: a Win, a Loss, or a Draw - a
    // stalemate now (distance 0), which the prover records as a game fact.
    if (e.proof.kind == Proof::Kind::None || (e.proof.kind == Proof::Kind::Draw && e.proof.distance != 0)) {
      std::fprintf(stderr, "  the solved store holds %s at %s\n", to_string(e.proof).c_str(),
                   describe_position(e.position).c_str());
      CHECK(false);
      continue;
    }
    GameTree tree(5000, 20000, 40000);
    const GameValue v = tree.solve(e.position, e.proof.distance);
    if (v.kind == GameValue::Kind::Unknown) {
      ++c.beyond;
      continue;
    }
    const GameValue::Kind want = e.proof.kind == Proof::Kind::Win    ? GameValue::Kind::Win
                                 : e.proof.kind == Proof::Kind::Loss ? GameValue::Kind::Loss
                                                                     : GameValue::Kind::Draw;
    if (v.kind == want && v.distance <= e.proof.distance) {
      ++c.confirmed;
    } else {
      std::fprintf(stderr, "  stored %s, the game tree %s %d at %s\n", to_string(e.proof).c_str(),
                   t5d_test::name_of(v.kind), v.distance, describe_position(e.position).c_str());
      CHECK(false);
    }
  }
}

DisproofClass root_class(const ProverOptions& o) {
  return DisproofClass{true, o.seeds, o.completion, disproof_shape(o)};
}

// H19 at the root after a search over external stores: a proof is in the
// solved store, a disproof in the disproof store with its class and depth -
// or, where the root was found lost or stalemated outright, that fact in the
// solved store - and never a Win for a disproven root.
void exports_hold(const ProveResult& r, const GameState& s, const ProverOptions& o, const MemorySolvedStore& proofs,
                  const DisproofStore& disproofs) {
  if (r.status == ProofStatus::Proven) {
    CHECK(r.stats.proofs_exported >= 1 || r.stats.store_hits >= 1);
    const std::optional<Proof> p = proofs.find(s);
    REQUIRE(p.has_value());
    CHECK(p->kind == Proof::Kind::Win);
    CHECK(p->distance <= r.proof.distance);
  } else if (r.status == ProofStatus::Disproven) {
    const std::optional<Proof> p = proofs.find(s);
    CHECK(!p.has_value() || p->kind != Proof::Kind::Win);
    // Either the class-relative disproof, or a game fact that implies it: the
    // side to move proven lost (or stalemated) in the solved store - found
    // when the search met the root as the defender's and read back at the
    // root (make_turn's "the other side wins: no mate for A").
    if (disproofs.find(s, root_class(o)) < o.depth && !p.has_value()) {
      std::fprintf(stderr, "  disproven, no disproof recorded; the solved store holds %s at %s\n",
                   p ? to_string(*p).c_str() : "nothing", describe_position(s).c_str());
      CHECK(false);
    }
  }
}

struct Churn {
  Tally tally;
  StoreCheck store;
  int roots = 0;
  int compared = 0;   // decided under churn and with the default TT
  int answered = 0;   // re-asked of a fresh prover over the stores
  int related = 0;    // a proof's continuation, answered from the stores
  int sequences = 0;  // positions re-asked after a sequence shared the stores
};

// One position under the TT's churn: tt_entries 1 and 2 against the default.
void churn_position(const GameState& s, const ProverOptions& base, std::uint64_t nodes, Churn& c) {
  MemorySolvedStore p0;
  DisproofStore d0;
  Prover reference(base, &p0, &d0);
  const ProveResult r0 = reference.prove(s, ProveBudget::nodes(nodes));
  for (const std::size_t tt : {std::size_t{1}, std::size_t{2}}) {
    ProverOptions o = base;
    o.tt_entries = tt;
    MemorySolvedStore proofs;
    DisproofStore disproofs;
    Prover p(o, &proofs, &disproofs);
    const ProveResult r = p.prove(s, ProveBudget::nodes(nodes));
    ++c.roots;
    if (!c.tally.add(r, s, o)) CHECK(false);
    if (r.status != ProofStatus::Unknown && r0.status != ProofStatus::Unknown) {
      ++c.compared;
      if (r.status != r0.status) {
        std::fprintf(stderr, "  tt %zu: %s, the default TT %s at %s\n", tt, proof_status_name(r.status),
                     proof_status_name(r0.status), describe_position(s).c_str());
        CHECK(false);
      }
    }
    exports_hold(r, s, o, proofs, disproofs);
    check_store(proofs, 6, c.store);
    if (r.status == ProofStatus::Unknown) continue;
    // A fresh prover over the same stores (its TT empty) answers at once.
    Prover fresh(o, &proofs, &disproofs);
    const ProveResult again = fresh.prove(s, ProveBudget::nodes(nodes));
    ++c.answered;
    CHECK(again.status == r.status);
    CHECK(again.stats.store_hits >= 1);
    CHECK_EQ(again.stats.expansions, std::uint64_t{0});
    if (r.status == ProofStatus::Proven) CHECK(again.proof.distance <= r.proof.distance);
    // A related root ("on PV changes"): the proof's continuation - the
    // attacker to move again after its action and the longest defence - is
    // answered from the stores too.
    if (r.status == ProofStatus::Proven && r.line_complete && r.line.size() >= 3) {
      GameState t = s;
      bool ok = true;
      for (std::size_t i = 0; i < 2 && ok; ++i) {
        for (const Move& m : r.line[i]) t.apply(m);
        ok = t.obligations_met() && royals_safe(t);
        if (ok) t.submit();
      }
      REQUIRE(ok);
      Prover next(o, &proofs, &disproofs);
      const ProveResult rn = next.prove(t, ProveBudget::nodes(nodes), r.proof.distance - 2);
      ++c.related;
      CHECK(rn.status == ProofStatus::Proven);
      CHECK(rn.stats.store_hits >= 1);
      CHECK_EQ(rn.stats.expansions, std::uint64_t{0});
    }
  }
}

// A sequence of related roots - consecutive turn starts of random games -
// sharing one pair of stores under a 1-entry TT; then every root it decided,
// re-asked of a fresh prover, is answered from the stores alone.
void churn_sequence(const std::vector<GameState>& positions, const ProverOptions& base, std::uint64_t nodes,
                    Churn& c) {
  ProverOptions o = base;
  o.tt_entries = 1;
  MemorySolvedStore proofs;
  DisproofStore disproofs;
  std::vector<std::pair<const GameState*, ProveResult>> decided;
  for (const GameState& s : positions) {
    Prover p(o, &proofs, &disproofs);
    const ProveResult r = p.prove(s, ProveBudget::nodes(nodes));
    if (!c.tally.add(r, s, o)) CHECK(false);
    exports_hold(r, s, o, proofs, disproofs);
    if (r.status != ProofStatus::Unknown) decided.emplace_back(&s, r);
  }
  for (const auto& [s, r] : decided) {
    Prover fresh(o, &proofs, &disproofs);
    const ProveResult again = fresh.prove(*s, ProveBudget::nodes(nodes));
    ++c.sequences;
    CHECK(again.status == r.status);
    CHECK_EQ(again.stats.expansions, std::uint64_t{0});
  }
  check_store(proofs, 20, c.store);
}

void churn_property(int turns, std::uint64_t nodes, Churn& c) {
  c.tally.work_cap = nodes;
  ProverOptions dflt;
  dflt.depth = 3;
  ProverOptions lit = literal();
  lit.depth = 1;
  // The constructed mates, both Q4 readings, both classes, at depth 3.
  for (const RuleOptions& rules : {guide_literal_q4(), shift_rules()}) {
    for (const GameState& s : {mate_in_one(rules), cross_board_mate(rules), stale_checkmate_in_one(rules),
                               history_royal_mate(rules), mate_in_two(rules)}) {
      for (ProverOptions o : {ProverOptions{}, literal()}) {
        o.depth = 3;
        churn_position(s, o, 2'000'000, c);
      }
    }
  }
  // Random play under every ruleset (both Q4 readings).
  std::uint64_t seed = 7300;
  for (const RuleOptions& base : t5d_test::every_ruleset()) {
    if (base.moves_after_shift != RuleOptions{}.moves_after_shift) continue;  // both below
    for (const bool shift : {false, true}) {
      RuleOptions rules = base;
      rules.moves_after_shift = shift;
      const std::vector<GameState> positions = playable(t5d_test::sparse_positions(rules, 1, turns, seed++));
      for (const GameState& s : positions) {
        churn_position(s, dflt, nodes, c);
        churn_position(s, lit, nodes, c);
      }
      churn_sequence(positions, dflt, nodes, c);
    }
  }
}

void churn_print(const Churn& c) {
  c.tally.print("TT churn");
  std::fprintf(stderr,
               "  %d roots under churn; %d decided both ways and compared; %d re-asked from the stores, %d "
               "continuations and %d sequence roots answered from them; store entries %d (%d confirmed by the "
               "game tree, %d beyond it)\n",
               c.roots, c.compared, c.answered, c.related, c.sequences, c.store.entries, c.store.confirmed,
               c.store.beyond);
}

}  // namespace

TEST(prover_proofs_survive_tt_churn_h19) {
  Churn c;
  churn_property(2, 10'000, c);
  churn_print(c);
  CHECK(c.tally.proven > 20);
  CHECK(c.tally.disproven > 20);
  CHECK(c.compared > 50);
  CHECK(c.related > 2);
  CHECK(c.sequences > 20);
  CHECK(c.store.confirmed > 50);
}

TEST(long_prover_proofs_survive_tt_churn_h19) {
  Churn c;
  churn_property(5, 60'000, c);
  churn_print(c);
  CHECK(c.tally.proven > 60);
  CHECK(c.tally.disproven > 60);
  CHECK(c.compared > 200);
  CHECK(c.related > 4);
  CHECK(c.sequences > 60);
  CHECK(c.store.confirmed > 200);
}

namespace {

// M13's 5 threads sharing a 1-entry TT, by nodes and by the clock: no false
// proof (the references), every proof in the solved store, and every proof's
// certificate confirmed by E4 alone (fresh stores per search: a proof read
// from another prover's store could not be certified).
void threads_churn(int turns, std::uint64_t seed, std::uint64_t nodes, Tally& tally, int& certified) {
  std::vector<GameState> positions{mate_in_one(shift_rules()), mate_in_two(shift_rules()),
                                   cross_board_mate(shift_rules())};
  for (const GameState& s : playable(t5d_test::sparse_positions(shift_rules(), 1, turns, seed))) positions.push_back(s);
  for (const ProverOptions& base : {ProverOptions{}, literal()}) {
    ProverOptions o = base;
    o.threads = 5;
    o.tt_entries = 1;
    o.depth = 3;
    o.record_certificate = true;
    for (const GameState& s : positions) {
      for (const bool by_clock : {false, true}) {
        MemorySolvedStore proofs;
        DisproofStore disproofs;
        Prover p(o, &proofs, &disproofs);
        const ProveResult r = by_clock ? p.prove(s, ProveBudget::time(std::chrono::milliseconds(20)))
                                       : p.prove(s, ProveBudget::nodes(nodes));
        if (!tally.add(r, s, o)) CHECK(false);
        exports_hold(r, s, o, proofs, disproofs);
        if (r.status != ProofStatus::Proven) continue;
        REQUIRE(r.certificate != nullptr);
        CertificateCheckOptions co;
        co.max_positions = 20000;
        const CertificateCheck cc = check_certificate(*r.certificate, s, r.proof.distance, co);
        if (cc.verdict == CertificateVerdict::Confirmed) {
          ++certified;
        } else if (cc.verdict != CertificateVerdict::Undecided) {
          std::fprintf(stderr, "  certificate %s (%s) at %s\n", certificate_verdict_name(cc.verdict), cc.why.c_str(),
                       describe_position(s).c_str());
          CHECK(false);
        }
      }
    }
  }
}

}  // namespace

TEST(prover_threads_survive_tt_churn) {
  Tally tally;
  tally.work_cap = 10'000;  // every proof checked; disproofs within this budget
  int certified = 0;
  threads_churn(2, 7400, 30'000, tally, certified);
  tally.print("5 threads, 1-entry TT");
  std::fprintf(stderr, "  %d certificates confirmed by E4\n", certified);
  CHECK(tally.proven > 6);
  CHECK(certified > 6);
}

TEST(long_prover_threads_survive_tt_churn) {
  Tally tally;
  tally.work_cap = 50'000;
  int certified = 0;
  threads_churn(4, 7500, 100'000, tally, certified);
  tally.print("5 threads, 1-entry TT");
  std::fprintf(stderr, "  %d certificates confirmed by E4\n", certified);
  CHECK(tally.proven > 12);
  CHECK(certified > 12);
}
