// The main search (t5d/mcts.hpp; engine stage E6a): design spec §15.4's
// invariants under uniform and random evaluators - visit counts consistent
// through backup, virtual loss always released, solved bounds propagating and
// never contradicting, LCB and secure child choosing only among eligible
// children, re-rooting keeping only reachable nodes while state-keyed facts
// survive, the synchronous mode bit-reproducible, MCGS transposition hits
// returning identical evaluations - and the mechanics behind them: both D2
// arms, the legal-move children, dead ends, widening, noise, the turn cap,
// H37's "unknown is never terminal", the disabled proof store.
//
// The positions are written for these tests. Where a test rests on a rules
// fact (a mate, a stalemate, a forced loss), the fact is derived by hand in a
// comment where that is short, and is in every case RE-CHECKED by the test
// itself with E4's exhaustive detectors (t5d/search.hpp) - the rules are E4's
// and the gate's to prove; these tests prove the search.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/expand.hpp"
#include "t5d/fen.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"

using namespace t5d;
using namespace t5d::mcts;
using t5d_test::game;

namespace {

GameState variant(const char* name, const RuleOptions& rules = {}) {
  const Variant* v = find_variant(name);
  if (v == nullptr) throw std::logic_error(std::string("no variant ") + name);
  return variant_start(*v, rules);
}

// Search parameters that keep the tests fast: no square features (the
// built-in evaluators read none).
SearchParams fast(SearchParams p = {}) {
  p.expand.square_features = SquareFeatures::None;
  return p;
}

// An evaluator that forwards to another and records what it was asked.
class Recording final : public Evaluator {
 public:
  explicit Recording(Evaluator& inner) : inner_(inner) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    inner_.evaluate(batch, out);
    for (std::size_t i = 0; i < batch.size(); ++i) {
      keys.push_back(batch[i].key);
      seen.emplace(batch[i].key, out[i]);
      ++calls;
    }
  }
  void on_new_root(const GameState& root, std::uint64_t epoch) override {
    ++new_roots;
    inner_.on_new_root(root, epoch);
  }
  std::vector<std::uint64_t> keys;
  std::multimap<std::uint64_t, Evaluation> seen;
  std::size_t calls = 0;
  std::size_t new_roots = 0;

 private:
  Evaluator& inner_;
};

// An evaluator that throws on its n-th call.
class Failing final : public Evaluator {
 public:
  explicit Failing(int fail_on) : fail_on_(fail_on) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    if (++calls_ == fail_on_) throw std::runtime_error("the evaluator failed");
    for (std::size_t i = 0; i < batch.size(); ++i) {
      out[i].logits.assign(batch[i].expansion->moves().size(), 0.0f);
      out[i].value = two_hot(0.0);
    }
  }

 private:
  int fail_on_;
  int calls_ = 0;
};

// One move far more likely than the rest: a sharp policy.
class Sharp final : public Evaluator {
 public:
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      out[i].logits.assign(batch[i].expansion->moves().size(), 0.0f);
      if (!out[i].logits.empty()) out[i].logits[0] = 30.0f;
      out[i].value = two_hot(0.0);
    }
  }
};

Outcome verdict_of(const GameState& s) { return detect_mate(s, Detector::Hypercuboid).verdict; }

// Every configuration the invariants are checked under.
struct Config {
  const char* name;
  TurnSplit split;
  bool graph;
  bool widening;
  bool e2;
  bool e3;
  NodeKey key;
};
const std::vector<Config> kConfigs = {
    {"ply tree", TurnSplit::BoardMove, false, true, false, false, NodeKey::State},
    {"ply graph", TurnSplit::BoardMove, true, true, false, false, NodeKey::State},
    {"ply graph origins", TurnSplit::BoardMove, true, true, true, true, NodeKey::StateAndOrigins},
    {"ply tree, no widening, E2+E3", TurnSplit::BoardMove, false, false, true, true, NodeKey::State},
    {"turn tree", TurnSplit::WholeTurn, false, true, false, false, NodeKey::State},
    {"turn graph, E2+E3", TurnSplit::WholeTurn, true, true, true, true, NodeKey::State},
};

SearchParams params_of(const Config& c) {
  SearchParams p = fast();
  p.split = c.split;
  p.graph = c.graph;
  p.widening = c.widening;
  p.variance_scaled_cpuct = c.e2;
  p.uncertainty_weighted_backup = c.e2;
  p.variance_bonus = c.e3;
  p.node_key = c.key;
  // The whole-turn listings of a two-board turn run to hundreds of actions;
  // the invariants hold for a truncated list too (it is never proven lost).
  p.max_actions = 96;
  return p;
}

// The two positions most tests start from: the Standard start, and a
// position a few actions into Standard - Two Timelines (two boards to play
// each turn: the ply split's intra-turn transpositions).
GameState two_timelines_after(int actions) {
  GameState s = variant("Standard - Two Timelines");
  const std::unique_ptr<Evaluator> u = make_evaluator(BuiltinEvaluator::Uniform);
  for (int a = 0; a < actions; ++a) {
    const ActionList list = list_legal_actions(s, 1, Detector::Hypercuboid);
    if (list.actions.empty()) break;
    s = list.actions.front().result;
  }
  return s;
}

}  // namespace

// ---- §15.4: visit counts consistent through backup -------------------------------------

TEST(mcts_visit_counts_are_consistent_through_backup) {
  for (const Config& c : kConfigs) {
    for (const GameState& start : {variant("Standard"), two_timelines_after(2)}) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 7);
      Search search(start, params_of(c));
      search.run(*ev, Limits{120});
      CHECK_NOTHROW(search.check_invariants());
      const NodeView root = search.node(search.root());
      CHECK_EQ(static_cast<std::uint64_t>(root.visits), search.stats().playouts);
      // The root's visits are its own evaluation plus its children's.
      std::uint64_t through = 0;
      for (std::size_t i = 0; i < root.edge_count; ++i) through += search.edge(search.root(), i).visits;
      CHECK_EQ(static_cast<std::uint64_t>(root.visits), root.ends + through);
      CHECK(search.stats().playouts >= 120 || search.root_solved());
      if (c.graph) CHECK(search.stats().key_collisions == 0);
    }
  }
}

// ---- §15.4: virtual loss always released -------------------------------------------------

TEST(mcts_virtual_loss_is_released_by_delivery_and_abandon) {
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 3);
  SearchParams p = fast();
  p.batch_size = 8;
  Search search(variant("Standard"), p);
  search.run(*ev, Limits{40});
  // A batch in flight carries virtual loss on its paths.
  const std::size_t n = search.gather(8);
  REQUIRE(n > 0);
  CHECK(search.node(search.root()).virtual_loss > 0);
  CHECK_THROWS_AS(search.check_invariants(), std::logic_error);  // a batch is waiting
  search.abandon();
  CHECK_NOTHROW(search.check_invariants());
  // Gathered again and delivered: released, and backed up.
  const std::uint64_t before = search.stats().playouts;
  const std::size_t m = search.gather(8);
  REQUIRE(m > 0);
  std::vector<Evaluation> out(m);
  ev->evaluate(search.requests(), out);
  search.deliver(out);
  CHECK_NOTHROW(search.check_invariants());
  CHECK(search.stats().playouts >= before + m);
}

TEST(mcts_virtual_loss_is_released_when_the_evaluator_fails) {
  for (const Config& c : kConfigs) {
    Failing failing(3);
    Search search(two_timelines_after(1), params_of(c));
    CHECK_THROWS_AS(search.run(failing, Limits{200}), std::runtime_error);
    CHECK_NOTHROW(search.check_invariants());
    // ... and the search goes on afterwards.
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    search.run(*ev, Limits{30});
    CHECK_NOTHROW(search.check_invariants());
  }
}

TEST(mcts_a_malformed_evaluation_is_refused_and_released) {
  Search search(variant("Standard"), fast());
  const std::size_t n = search.gather(4);
  REQUIRE(n == 1);  // the root is the only leaf
  std::vector<Evaluation> out(1);
  out[0].logits.assign(3, 0.0f);  // the Standard start has 20 moves
  out[0].value = two_hot(0.0);
  CHECK_THROWS_AS(search.deliver(out), std::invalid_argument);
  CHECK_NOTHROW(search.check_invariants());
  CHECK_EQ(search.gather(4), std::size_t{1});
  out[0].logits.assign(20, 0.0f);
  out[0].value.fill(0.0f);  // all zero: no distribution
  CHECK_THROWS_AS(search.deliver(out), std::invalid_argument);
  CHECK_NOTHROW(search.check_invariants());
  CHECK_EQ(search.gather(4), std::size_t{1});
  out[0].value = two_hot(0.25);
  out[0].logits[2] = std::nanf("");
  CHECK_THROWS_AS(search.deliver(out), std::invalid_argument);
  CHECK_NOTHROW(search.check_invariants());
  CHECK_EQ(search.gather(4), std::size_t{1});
  out[0].logits[2] = 0.0f;
  CHECK_NOTHROW(search.deliver(out));
  CHECK(std::abs(search.node(search.root()).eval_mean - 0.25) < 1e-9);
}

// ---- §15.4: solved bounds propagating and never contradicting ---------------------------

namespace {

// White to move on one 5x5 board at T1: k a5; Q e4, K b3. Qb4 is mate - the
// queen checks a5 along the diagonal, a4 and b5 are the queen's, b4 is
// guarded by the king, and Black at T1 has no board of its own in the past to
// travel to (rulebook L4: a move lands on a board of the mover's colour). (Qa4
// and Qe5 mate too.)
GameState mate_in_one() { return game({"[k4/4Q/1K3/5/5:0:1:w]"}, BoardSize{5, 5}); }

// White to move on one 4x4 board: k d4; B a2, q c2, K d2. White is in check from
// the queen; c1, d1, c3 and d3 are covered, so Kxc2 is the only move - after
// which Black's king (c4 under the bishop, c3 and d3 under the king) has no
// move and is not in check: stalemate (C10, a draw).
GameState forced_draw() { return game({"[3k/4/B1qK/4:0:1:w]"}, BoardSize{4, 4}); }

// White to move on one 4x4 board: K a3; b d3, n d2, k b1. Every legal action
// of White's leaves Black a mate in one (found and checked by E4's detectors in
// the test; at T2 White could travel, and the detectors judge that too).
GameState forced_loss() { return game({"[4/K2b/3n/1k2:0:1:w]"}, BoardSize{4, 4}); }

// The premise of a forced loss for the side to move, checked with E4's
// detectors: every legal action leads to a position with a mating reply.
bool every_action_allows_mate(const GameState& s) {
  const ActionList acts = list_legal_actions(s, 64, Detector::Hypercuboid);
  if (!acts.stats.exhaustive || acts.actions.empty()) return false;
  for (const LegalAction& a : acts.actions) {
    const ActionList replies = list_legal_actions(a.result, 512, Detector::Hypercuboid);
    if (!replies.stats.exhaustive) return false;
    bool mate = false;
    for (const LegalAction& b : replies.actions) mate = mate || verdict_of(b.result) == Outcome::Checkmate;
    if (!mate) return false;
  }
  return true;
}

}  // namespace

TEST(mcts_solver_proves_a_mate_in_one_and_plays_it) {
  const GameState s = mate_in_one();
  // The premise, by E4's detectors: some legal action mates.
  bool premise = false;
  for (const LegalAction& a : list_legal_actions(s, 100, Detector::Hypercuboid).actions) {
    premise = premise || verdict_of(a.result) == Outcome::Checkmate;
  }
  REQUIRE(premise);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const bool graph : {false, true}) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
      SearchParams p = fast();
      p.split = split;
      p.graph = graph;
      Search search(s, p);
      search.run(*ev, Limits{2000});
      CHECK(search.root_solved());
      const NodeView root = search.node(search.root());
      CHECK_EQ(root.lo, 1);
      CHECK_EQ(root.hi, 1);
      CHECK(search.stats().playouts < 2000);  // the search stopped once the root was solved
      const Choice c = search.choose();
      CHECK(c.reason == ChoiceReason::ProvenWin);
      const std::vector<RootChild> children = search.root_children();
      const RootChild& chosen = children.at(c.index);
      REQUIRE(chosen.child != kNoNode);
      CHECK(search.node(chosen.child).terminal == Terminal::Checkmate);
      CHECK(verdict_of(*search.node(chosen.child).state) == Outcome::Checkmate);
      CHECK_NOTHROW(search.check_invariants());
      // Solved: more running consumes no evaluation.
      const std::uint64_t evaluations = search.stats().evaluations;
      search.run(*ev, Limits{50});
      CHECK_EQ(search.stats().evaluations, evaluations);
    }
  }
}

TEST(mcts_solver_proves_a_draw_when_every_child_is_drawn) {
  const GameState s = forced_draw();
  const ActionList acts = list_legal_actions(s, 16, Detector::Hypercuboid);
  REQUIRE(acts.stats.exhaustive);
  REQUIRE(!acts.actions.empty());
  for (const LegalAction& a : acts.actions) REQUIRE(verdict_of(a.result) == Outcome::Stalemate);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 5);
    SearchParams p = fast();
    p.split = split;
    Search search(s, p);
    search.run(*ev, Limits{200});
    CHECK(search.root_solved());
    CHECK_EQ(search.node(search.root()).lo, 0);
    CHECK_EQ(search.node(search.root()).hi, 0);
    const Choice c = search.choose();
    CHECK(c.reason == ChoiceReason::SecureChild);
    CHECK_NOTHROW(search.check_invariants());
  }
}

TEST(mcts_solver_proves_a_loss_when_every_child_is_lost) {
  const GameState s = forced_loss();
  REQUIRE(every_action_allows_mate(s));
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const bool graph : {false, true}) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
      SearchParams p = fast();
      p.split = split;
      p.graph = graph;
      Search search(s, p);
      search.run(*ev, Limits{20000});
      CHECK(search.root_solved());
      CHECK_EQ(search.node(search.root()).lo, -1);
      CHECK_EQ(search.node(search.root()).hi, -1);
      // Every root child is proven lost: final selection says so.
      const Choice c = search.choose();
      CHECK(c.reason == ChoiceReason::AllLost);
      for (const RootChild& child : search.root_children()) CHECK_EQ(child.hi, -1);
      CHECK_NOTHROW(search.check_invariants());
    }
  }
}

TEST(mcts_solver_bounds_never_contradict_in_play) {
  // Random play from small variants (where mates and stalemates come quickly)
  // searched at every turn start: the invariant check holds every node's
  // bounds against its children's, and a contradiction would have thrown.
  std::size_t solved_nodes = 0;
  t5d_test::Rng rng(11);
  for (const char* name : {"Focused - Just Kings", "Very Small - Open", "Very Small", "Standard"}) {
    for (int game_no = 0; game_no < 3; ++game_no) {
      GameState s = variant(name);
      for (int ply = 0; ply < 10; ++ply) {
        if (verdict_of(s) != Outcome::Ongoing) break;
        const Config& c = kConfigs[static_cast<std::size_t>(rng.below(static_cast<int>(kConfigs.size())))];
        const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Material);
        Search search(s, params_of(c));
        search.run(*ev, Limits{60});
        CHECK_NOTHROW(search.check_invariants());
        for (NodeId id = 0; id < search.node_count(); ++id) {
          const NodeView n = search.node(id);
          if (n.lo == n.hi) ++solved_nodes;
          CHECK(n.terminal == Terminal::None || !n.expanded);  // terminals are never evaluated
        }
        const ActionList acts = list_legal_actions(s, 64, Detector::Hypercuboid);
        if (acts.actions.empty()) break;
        s = acts.actions[static_cast<std::size_t>(rng.below(static_cast<int>(acts.actions.size())))].result;
      }
    }
  }
  CHECK(solved_nodes > 0);
}

// ---- H37: an unknown verdict is never a terminal ------------------------------------------

TEST(mcts_an_unknown_verdict_is_never_terminal) {
  // A detector with no budget answers Unknown everywhere: the mate in one is
  // then never scored as a mate, and nothing is solved.
  SearchParams p = fast();
  p.mate_nodes = 0;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  Search search(mate_in_one(), p);
  search.run(*ev, Limits{300});
  CHECK(!search.root_solved());
  CHECK(search.stats().unknown_verdicts > 0);
  CHECK_EQ(search.stats().checkmates, std::uint64_t{0});
  for (NodeId id = 0; id < search.node_count(); ++id) {
    CHECK(search.node(id).terminal == Terminal::None);
    CHECK(search.node(id).lo != search.node(id).hi);
  }
  CHECK_NOTHROW(search.check_invariants());
}

// ---- H19/H23: the proof store stays disabled -----------------------------------------------

TEST(mcts_the_proof_store_stays_disabled_until_the_gate_passes) {
  SearchParams p = fast();
  p.proof_store = true;
  CHECK_THROWS_WITH(Search(variant("Standard"), p), std::invalid_argument, "H23");
}

// ---- I11: the turn cap is a draw, never a proof ---------------------------------------------

TEST(mcts_the_turn_cap_scores_a_draw_and_proves_nothing) {
  SearchParams p = fast();
  p.turn_cap.turns = 1;  // every turn start after the first turn is capped
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Material);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    p.split = split;
    // From Black's first turn: every reply reaches turn 2's start - capped.
    GameState s = variant("Standard");
    s = list_legal_actions(s, 1, Detector::Hypercuboid).actions.front().result;
    Search search(s, p);
    search.run(*ev, Limits{100});
    CHECK(search.stats().turn_caps > 0);
    std::size_t capped = 0;
    for (NodeId id = 0; id < search.node_count(); ++id) {
      const NodeView n = search.node(id);
      if (n.terminal != Terminal::TurnCap) continue;
      ++capped;
      CHECK(n.lo != n.hi);                       // never a proof
      CHECK_EQ(n.value, 0.0);                    // scored as a draw
    }
    CHECK(capped > 0);
    CHECK(!search.root_solved());
    CHECK_NOTHROW(search.check_invariants());
  }
}

// ---- E5, E9: final selection chooses only among eligible children ---------------------------

TEST(mcts_final_selection_on_constructed_statistics) {
  SearchParams p;
  // A proven win is chosen however few its visits.
  {
    const std::vector<FinalCandidate> c = {{500, 0.4, 0.01, 0, -1, 1}, {3, 0.9, 0, 0, 1, 1}};
    const Choice ch = select_final(c, p);
    CHECK_EQ(ch.index, std::size_t{1});
    CHECK(ch.reason == ChoiceReason::ProvenWin);
  }
  // A proven loss is never chosen while another child is open.
  {
    const std::vector<FinalCandidate> c = {{900, 0.1, 0.01, 0, -1, -1}, {100, -0.2, 0.01, 0, -1, 1}};
    const Choice ch = select_final(c, p);
    CHECK_EQ(ch.index, std::size_t{1});
    CHECK(final_eligible(c, ch.index, ch.reason, p));
    CHECK(!final_eligible(c, 0, ch.reason, p));
  }
  // LCB: a high-Q child with thin evidence (under 15% of the most visits) loses
  // to the incumbent; above the floor, the bound decides.
  {
    p.final_rule = FinalRule::Lcb;
    const std::vector<FinalCandidate> c = {{1000, 0.30, 0.04, 0, -1, 1},
                                           {100, 0.90, 0.04, 0, -1, 1},
                                           {400, 0.36, 0.04, 0, -1, 1}};
    const Choice ch = select_final(c, p);
    CHECK(ch.reason == ChoiceReason::Lcb);
    CHECK(ch.index != std::size_t{1});  // below the 15% floor
    CHECK(final_eligible(c, ch.index, ch.reason, p));
    CHECK(!final_eligible(c, 1, ChoiceReason::Lcb, p));
    // 0.30 - 3 sqrt(0.04/1000) = 0.281 against 0.36 - 3 sqrt(0.04/400) = 0.33.
    CHECK_EQ(ch.index, std::size_t{2});
    // The optional categorical term widens the thinner child's bound most.
    SearchParams q = p;
    q.lcb_categorical_weight = 1.0;
    std::vector<FinalCandidate> d = c;
    d[2].eval_variance = 0.5;
    CHECK_EQ(select_final(d, q).index, std::size_t{0});
  }
  // Visits: the most visits.
  {
    SearchParams q = p;
    q.final_rule = FinalRule::Visits;
    const std::vector<FinalCandidate> c = {{10, 0.9, 0, 0, -1, 1}, {30, -0.5, 0, 0, -1, 1}};
    const Choice ch = select_final(c, q);
    CHECK_EQ(ch.index, std::size_t{1});
    CHECK(ch.reason == ChoiceReason::Visits);
  }
  // A solved draw among the eligible: the secure child - the largest lower
  // bound, 0 for the draw against Q - A/sqrt(n) for the open children.
  {
    // 0.1 - 1/sqrt(50) = -0.041 < 0: the draw.
    const std::vector<FinalCandidate> c = {{50, 0.1, 0.1, 0, -1, 1}, {8, 0.0, 0, 0, 0, 0}};
    const Choice ch = select_final(c, p);
    CHECK(ch.reason == ChoiceReason::SecureChild);
    CHECK_EQ(ch.index, std::size_t{1});
    // 0.3 - 1/sqrt(50) = 0.159 > 0: the open child is the more secure.
    const std::vector<FinalCandidate> d = {{50, 0.3, 0.1, 0, -1, 1}, {8, 0.0, 0, 0, 0, 0}};
    CHECK_EQ(select_final(d, p).index, std::size_t{0});
  }
  // Every child lost: the most visits.
  {
    const std::vector<FinalCandidate> c = {{5, -1, 0, 0, -1, -1}, {9, -1, 0, 0, -1, -1}};
    const Choice ch = select_final(c, p);
    CHECK(ch.reason == ChoiceReason::AllLost);
    CHECK_EQ(ch.index, std::size_t{1});
  }
  CHECK_THROWS_AS(select_final(std::vector<FinalCandidate>{{0, 0, 0, 0, -1, 1}}, p),
                  std::invalid_argument);
}

TEST(mcts_final_selection_chooses_only_eligible_children_at_random) {
  t5d_test::Rng rng(99);
  for (int trial = 0; trial < 3000; ++trial) {
    SearchParams p;
    p.final_rule = rng.below(2) == 0 ? FinalRule::Visits : FinalRule::Lcb;
    p.lcb_categorical_weight = rng.below(2) == 0 ? 0.0 : 0.5;
    const int n = 1 + rng.below(6);
    std::vector<FinalCandidate> c;
    bool any_visit = false;
    for (int i = 0; i < n; ++i) {
      FinalCandidate f;
      f.visits = static_cast<std::uint32_t>(rng.below(4) == 0 ? 0 : rng.below(1000));
      f.q = (rng.below(2001) - 1000) / 1000.0;
      f.variance = rng.below(100) / 100.0;
      f.eval_variance = rng.below(100) / 100.0;
      const int kind = rng.below(8);
      if (kind == 0) f.lo = f.hi = 1;
      if (kind == 1) f.lo = f.hi = -1;
      if (kind == 2) f.lo = f.hi = 0;
      if (kind == 3) f.lo = 0;
      any_visit = any_visit || f.visits > 0;
      c.push_back(f);
    }
    if (!any_visit) {
      CHECK_THROWS_AS(select_final(c, p), std::invalid_argument);
      continue;
    }
    const Choice ch = select_final(c, p);
    CHECK(final_eligible(c, ch.index, ch.reason, p));
    // No proven win left behind; no proven loss while an open child is eligible.
    bool win = false;
    bool open = false;
    for (const FinalCandidate& f : c) {
      win = win || (f.lo == 1 && f.visits > 0);
      open = open || (f.visits > 0 && f.hi > -1);
    }
    if (win) CHECK(ch.reason == ChoiceReason::ProvenWin);
    if (open) CHECK(c[ch.index].hi > -1);
  }
}

TEST(mcts_final_selection_in_searches_is_among_eligible_children) {
  for (const Config& c : kConfigs) {
    for (const FinalRule rule : {FinalRule::Visits, FinalRule::Lcb}) {
      SearchParams p = params_of(c);
      p.final_rule = rule;
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 21);
      Search search(two_timelines_after(2), p);
      search.run(*ev, Limits{80});
      const Choice ch = search.choose();
      std::vector<FinalCandidate> cands;
      for (const RootChild& r : search.root_children()) {
        FinalCandidate f;
        f.visits = r.visits;
        f.q = r.q;
        f.lo = r.lo;
        f.hi = r.hi;
        cands.push_back(f);
      }
      CHECK(final_eligible(cands, ch.index, ch.reason, p));
      CHECK(cands.at(ch.index).visits > 0);
    }
  }
}

// ---- C12, C13: re-rooting keeps only reachable nodes; state-keyed facts survive -----------

TEST(mcts_rerooting_keeps_only_reachable_nodes_and_cached_evaluations) {
  for (const Config& c : kConfigs) {
    const std::unique_ptr<Evaluator> inner = make_evaluator(BuiltinEvaluator::Random, 4);
    Recording ev(*inner);
    Search search(two_timelines_after(1), params_of(c));
    std::size_t hooks = 0;
    search.set_root_hook([&hooks](const GameState&, std::uint64_t) { ++hooks; });
    search.run(ev, Limits{200});
    const std::vector<RootChild> children = search.root_children();
    REQUIRE(children.size() >= 2);
    const Choice ch = search.choose();
    const NodeId kept = children[ch.index].child;
    REQUIRE(kept != kNoNode);
    // The nodes the chosen child reaches, by key, before re-rooting.
    std::multiset<std::uint64_t> reachable;
    std::set<NodeId> seen{kept};
    std::vector<NodeId> stack{kept};
    while (!stack.empty()) {
      const NodeId id = stack.back();
      stack.pop_back();
      reachable.insert(search.node(id).key);
      for (std::size_t i = 0; i < search.node(id).edge_count; ++i) {
        const NodeId ch2 = search.edge(id, i).child;
        if (ch2 != kNoNode && seen.insert(ch2).second) stack.push_back(ch2);
      }
    }
    const std::uint32_t kept_visits = search.node(kept).visits;
    // A sibling that will be deleted: its state's evaluation is a state-keyed fact.
    NodeId sibling = kNoNode;
    for (std::size_t i = 0; i < children.size(); ++i) {
      if (i != ch.index && children[i].child != kNoNode && search.node(children[i].child).expanded &&
          !seen.count(children[i].child)) {
        sibling = children[i].child;
      }
    }
    REQUIRE(sibling != kNoNode);
    const GameState sibling_state = *search.node(sibling).state;
    const std::uint64_t sibling_key = search.node(sibling).key;
    const std::size_t before = search.node_count();
    search.advance(ch.index);
    CHECK_EQ(hooks, std::size_t{1});
    CHECK_EQ(search.node_count(), reachable.size());
    CHECK(search.node_count() < before);
    std::multiset<std::uint64_t> after;
    for (NodeId id = 0; id < search.node_count(); ++id) after.insert(search.node(id).key);
    CHECK(after == reachable);
    CHECK_EQ(search.node(search.root()).visits, kept_visits);  // statistics carried
    CHECK_NOTHROW(search.check_invariants());
    CHECK(search.stats().nodes_collected == before - search.node_count());
    REQUIRE(search.cache() != nullptr);
    CHECK(search.cache()->find(sibling_key, 0, sibling_state, true) != nullptr);
    // Searching on: the evaluator is told of the new root, and never asked
    // about a state the cache holds.
    const std::size_t roots = ev.new_roots;
    search.run(ev, Limits{100});
    CHECK_EQ(ev.new_roots, roots + 1);
    std::map<std::uint64_t, std::size_t> per_key;
    for (const std::uint64_t k : ev.keys) ++per_key[k];
    if (c.graph) {
      for (const auto& [k, count] : per_key) CHECK_EQ(count, std::size_t{1});
    }
  }
}

TEST(mcts_pondering_searches_the_reply_and_reroots_on_it) {
  // Match play: our action, then the opponent's position searched (ponder),
  // then the opponent's actual reply - the tree re-rooted each time.
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    SearchParams p = fast();
    p.split = split;
    p.graph = split == TurnSplit::BoardMove;
    p.max_actions = 96;
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 8);
    const GameState start = two_timelines_after(2);
    Search search(start, p);
    std::vector<Move> ours;
    while (true) {
      search.run(*ev, Limits{40});
      const Choice ch = search.choose();
      const RootChild child = search.root_children().at(ch.index);
      ours.insert(ours.end(), child.moves.begin(), child.moves.end());
      search.advance(ch.index);
      if (search.root_turn_start()) break;
    }
    CHECK(search.root_state().to_move() != start.to_move());
    // Ponder: search the opponent's position.
    search.run(*ev, Limits{60});
    const std::uint64_t pondered = search.node(search.root()).visits;
    CHECK(pondered >= 60);
    // The reply: the first legal action, whatever the tree thought of it.
    const ActionList replies = list_legal_actions(search.root_state(), 1, Detector::Hypercuboid);
    REQUIRE(!replies.actions.empty());
    const GameState expected = replies.actions.front().result;
    search.advance_action(replies.actions.front().moves);
    CHECK(search.root_turn_start());
    CHECK(search.root_state() == expected);
    CHECK_NOTHROW(search.check_invariants());
    search.run(*ev, Limits{40});
    CHECK_NOTHROW(search.check_invariants());
  }
}

TEST(mcts_advance_to_a_state_the_tree_does_not_hold_starts_afresh) {
  Search search(variant("Standard"), fast());
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  search.run(*ev, Limits{10});
  // A reply the widening has not instantiated yet is still found among the
  // root's children; a state two decisions away is not, and starts afresh.
  GameState far = variant("Standard");
  far = list_legal_actions(far, 1, Detector::Hypercuboid).actions.front().result;
  far = list_legal_actions(far, 1, Detector::Hypercuboid).actions.front().result;
  search.advance_to(far, true);
  CHECK_EQ(search.node_count(), std::size_t{1});
  CHECK(search.root_state() == far);
  search.run(*ev, Limits{10});
  CHECK_NOTHROW(search.check_invariants());
  // An illegal move is refused.
  Move bogus{};
  CHECK_THROWS_AS(search.advance_moves(std::vector<Move>{bogus}), std::invalid_argument);
}

// ---- H15: the synchronous mode is bit-reproducible -----------------------------------------

TEST(mcts_synchronous_mode_is_bit_reproducible) {
  for (const Config& c : kConfigs) {
    for (const bool synchronous : {true, false}) {
      SearchParams p = params_of(c);
      p.synchronous = synchronous;
      p.root_noise = true;
      p.seed = 77;
      const auto digest_of = [&](std::uint64_t seed) {
        const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, seed);
        Search search(two_timelines_after(2), p);
        search.run(*ev, Limits{80});
        return search.digest();
      };
      const std::uint64_t a = digest_of(5);
      CHECK_EQ(a, digest_of(5));
      CHECK(a != digest_of(6));
    }
  }
}

// ---- C11: MCGS transposition hits return identical evaluations --------------------------------

TEST(mcts_graph_transpositions_are_shared_and_evaluated_once) {
  // Two Timelines: White plays both boards each turn, in either order - the
  // ply split's intra-turn transpositions (§8.5).
  for (const NodeKey key : {NodeKey::State, NodeKey::StateAndOrigins}) {
    SearchParams p = fast();
    p.graph = true;
    p.node_key = key;
    p.widening = false;
    const std::unique_ptr<Evaluator> inner = make_evaluator(BuiltinEvaluator::Random, 12);
    Recording ev(*inner);
    Search search(variant("Standard - Two Timelines"), p);
    search.run(ev, Limits{400});
    CHECK(search.stats().transposition_hits > 0);
    CHECK(search.stats().residual_backups > 0);
    // Each state is evaluated once: a transposition costs no evaluation.
    std::map<std::uint64_t, std::size_t> per_key;
    for (const std::uint64_t k : ev.keys) ++per_key[k];
    for (const auto& [k, count] : per_key) CHECK_EQ(count, std::size_t{1});
    CHECK_EQ(ev.calls, search.stats().evaluations);
    // Some node has two parents.
    std::map<NodeId, int> parents;
    for (NodeId id = 0; id < search.node_count(); ++id) {
      for (std::size_t i = 0; i < search.node(id).edge_count; ++i) {
        const NodeId ch = search.edge(id, i).child;
        if (ch != kNoNode) ++parents[ch];
      }
    }
    bool shared = false;
    for (const auto& [id, n] : parents) shared = shared || n > 1;
    CHECK(shared);
    CHECK_NOTHROW(search.check_invariants());
  }
}

TEST(mcts_tree_duplicates_get_bit_identical_evaluations_from_the_cache) {
  // A plain tree holds the transposed states as separate nodes; the state-keyed
  // cache answers the second from the first, bit for bit.
  SearchParams p = fast();
  p.widening = false;
  const std::unique_ptr<Evaluator> inner = make_evaluator(BuiltinEvaluator::Random, 12);
  Recording ev(*inner);
  Search search(variant("Standard - Two Timelines"), p);
  search.run(ev, Limits{400});
  CHECK(search.stats().cache_hits + search.stats().batch_duplicates > 0);
  std::map<std::uint64_t, std::vector<NodeId>> by_key;
  for (NodeId id = 0; id < search.node_count(); ++id) {
    if (search.node(id).expanded) by_key[search.node(id).key].push_back(id);
  }
  std::size_t duplicates = 0;
  for (const auto& [k, ids] : by_key) {
    for (std::size_t i = 1; i < ids.size(); ++i) {
      ++duplicates;
      const NodeView a = search.node(ids[0]);
      const NodeView b = search.node(ids[i]);
      CHECK(*a.state == *b.state);
      CHECK_EQ(a.eval_mean, b.eval_mean);
      CHECK_EQ(a.eval_variance, b.eval_variance);
      REQUIRE_EQ(a.edge_count, b.edge_count);
      for (std::size_t e = 0; e < a.edge_count; ++e) {
        CHECK_EQ(search.edge(ids[0], e).prior, search.edge(ids[i], e).prior);
      }
    }
  }
  CHECK(duplicates > 0);
  // The evaluator never saw a state twice.
  std::map<std::uint64_t, std::size_t> per_key;
  for (const std::uint64_t k : ev.keys) ++per_key[k];
  for (const auto& [k, count] : per_key) CHECK_EQ(count, std::size_t{1});
}

// ---- D2: what the children are, under each arm ----------------------------------------------

TEST(mcts_board_move_children_are_the_legal_moves) {
  SearchParams p = fast();
  p.widening = false;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 2);
  Search search(two_timelines_after(3), p);
  search.run(*ev, Limits{150});
  std::size_t checked = 0;
  for (NodeId id = 0; id < search.node_count() && checked < 25; ++id) {
    const NodeView n = search.node(id);
    if (!n.expanded || n.stuck) continue;
    ++checked;
    MoveList legal;
    legal_moves(*n.state, legal);
    std::vector<Move> edges;
    bool submit = false;
    for (std::size_t i = 0; i < n.edge_count; ++i) {
      const EdgeView e = search.edge(id, i);
      if (e.submit) {
        submit = true;
      } else {
        edges.push_back(*e.move);
        // (a) under the default budget: a legal move, or one flagged unknown.
        if (std::find(legal.begin(), legal.end(), *e.move) == legal.end()) CHECK(e.unknown || e.pruned);
      }
    }
    CHECK(edges.size() >= legal.size());
    for (const Move& m : legal) CHECK(std::find(edges.begin(), edges.end(), m) != edges.end());
    CHECK(!submit);  // the default Q4 reading: a finished action is submitted at once
    // Mid-turn nodes keep the mover; terminal detection only at turn starts.
    if (!n.turn_start) CHECK(n.terminal == Terminal::None);
  }
  CHECK(checked > 5);
}

TEST(mcts_board_move_children_are_exactly_the_legal_moves_under_exact_legality) {
  SearchParams p = fast();
  p.widening = false;
  p.legality = LegalityOptions::exact();
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 2);
  Search search(two_timelines_after(3), p);
  search.run(*ev, Limits{100});
  CHECK_EQ(search.stats().unknown_edges, std::uint64_t{0});
  std::size_t checked = 0;
  for (NodeId id = 0; id < search.node_count(); ++id) {
    const NodeView n = search.node(id);
    if (!n.expanded || n.stuck) continue;
    ++checked;
    MoveList legal;
    legal_moves(*n.state, legal);
    CHECK_EQ(n.edge_count, legal.size());
  }
  CHECK(checked > 5);
}

// ---- E5a's tiers in the search (build-log spec issue 46; parts (a)-(f)) -----------------

namespace {

// A turn start from seeded random legal play at which the FAST classification
// (t5d/legal.hpp: budgeted(0, 0)) leaves at least one move unknown that is in
// fact illegal and one that is in fact legal (the exact classification says).
GameState position_with_unknown_moves() {
  t5d_test::Rng rng(31);
  for (int game_no = 0; game_no < 40; ++game_no) {
    GameState s = variant(game_no % 2 == 0 ? "Standard - Two Timelines" : "Standard");
    for (int a = 0; a < 12; ++a) {
      MoveList moves;
      generate_moves(s, moves);
      std::vector<MoveLegality> fast_class;
      std::vector<MoveLegality> exact_class;
      classify_moves(s, moves, fast_class, LegalityOptions::budgeted(0, 0));
      classify_moves(s, moves, exact_class, LegalityOptions::exact());
      bool unknown_illegal = false;
      bool unknown_legal = false;
      for (std::size_t i = 0; i < moves.size(); ++i) {
        if (fast_class[i] != MoveLegality::Unknown) continue;
        unknown_illegal = unknown_illegal || exact_class[i] == MoveLegality::Illegal;
        unknown_legal = unknown_legal || exact_class[i] == MoveLegality::Legal;
      }
      if (unknown_illegal && unknown_legal) return s;
      const ActionList acts = list_legal_actions(s, 64, Detector::Hypercuboid);
      if (acts.actions.empty()) break;
      s = acts.actions[static_cast<std::size_t>(rng.below(static_cast<int>(acts.actions.size())))].result;
    }
  }
  throw std::logic_error("no position with unknown moves found");
}

}  // namespace

TEST(mcts_unknown_moves_are_kept_flagged_verified_and_pruned) {
  const GameState s = position_with_unknown_moves();
  MoveList pseudo;
  generate_moves(s, pseudo);
  std::vector<MoveLegality> exact_class;
  classify_moves(s, pseudo, exact_class, LegalityOptions::exact());
  const auto exact_of = [&](const Move& m) {
    for (std::size_t i = 0; i < pseudo.size(); ++i) {
      if (pseudo[i] == m) return exact_class[i];
    }
    throw std::logic_error("not a pseudo-legal move");
  };
  for (const bool graph : {false, true}) {
    SearchParams p = fast();
    p.graph = graph;
    p.widening = false;  // every child selectable at once: the unknown ones get picked
    p.legality = LegalityOptions::budgeted(0, 0);
    p.verify_legality = LegalityOptions::exact();
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    Search search(s, p);
    search.run(*ev, Limits{400});
    CHECK(search.stats().unknown_edges > 0);
    CHECK(search.stats().verified_illegal > 0);
    CHECK(search.stats().verified_legal > 0);
    const NodeView root = search.node(search.root());
    double sum = 0;
    for (std::size_t i = 0; i < root.edge_count; ++i) {
      const EdgeView e = search.edge(search.root(), i);
      const MoveLegality truth = exact_of(*e.move);
      CHECK(truth != MoveLegality::Unknown);
      // (a) a proved-illegal move never became an edge: an illegal one is an
      // unknown one, verified (or found a dead end) and pruned.
      if (truth == MoveLegality::Illegal) {
        CHECK(e.unknown || e.verified);
        CHECK(!e.verified || e.pruned);
        CHECK_EQ(e.visits, std::uint32_t{0});  // (c) nothing backed up through it
      }
      if (e.pruned) CHECK(truth == MoveLegality::Illegal);
      if (e.verified && !e.unknown) CHECK(truth == MoveLegality::Legal);
      if (!e.pruned) sum += e.prior;
    }
    CHECK(std::abs(sum - 1.0) < 1e-9);  // renormalised
    CHECK_NOTHROW(search.check_invariants());
    // RootChild says which children are known legal (f: the network's mask is
    // legal or unknown).
    for (const RootChild& c : search.root_children()) {
      CHECK(exact_of(c.moves.at(0)) == MoveLegality::Legal || c.legality == MoveLegality::Unknown);
    }
  }
}

TEST(mcts_an_unknown_edge_proves_nothing_until_verified) {
  // forced_draw: White's only legal move is Kxc2, which stalemates Black. The
  // fast classification refutes the bishop's moves (the queen's capture of d2
  // stands with the board passed) and leaves the king's moves unknown (royal
  // moves). Unverified, the stalemate behind Kxc2 bounds the root from above
  // but proves nothing (e); verified, it proves the draw.
  const GameState s = forced_draw();
  for (const bool verify : {false, true}) {
    SearchParams p = fast();
    p.widening = false;
    p.legality = LegalityOptions::budgeted(0, 0);
    p.verify_unknown = verify;
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    Search search(s, p);
    search.run(*ev, Limits{300});
    CHECK(search.stats().unknown_edges > 0);
    const NodeView root = search.node(search.root());
    CHECK_EQ(root.hi, 0);
    if (verify) {
      CHECK(search.root_solved());
      CHECK_EQ(root.lo, 0);
    } else {
      CHECK(!search.root_solved());
      CHECK_EQ(root.lo, -1);
      // Final selection sees no proof through the unknown edge either.
      for (const RootChild& c : search.root_children()) {
        if (c.legality == MoveLegality::Unknown) CHECK_EQ(c.lo, -1);
      }
    }
    // (e) everywhere: a node's lower bound rests on a child through a known edge.
    for (NodeId id = 0; id < search.node_count(); ++id) {
      const NodeView n = search.node(id);
      if (!n.expanded || n.terminal != Terminal::None || n.lo == -1) continue;
      bool supported = false;
      for (std::size_t i = 0; i < n.edge_count; ++i) {
        const EdgeView e = search.edge(id, i);
        if (e.pruned || e.unknown || e.child == kNoNode) continue;
        const NodeView c = search.node(e.child);
        const int lo = c.mover == n.mover ? c.lo : -c.hi;
        supported = supported || lo >= n.lo;
      }
      CHECK(supported);
    }
    CHECK_NOTHROW(search.check_invariants());
  }
}

TEST(mcts_verify_settles_a_root_child_before_it_is_played) {
  const GameState s = forced_draw();
  SearchParams p = fast();
  p.widening = false;
  p.legality = LegalityOptions::budgeted(0, 0);
  p.verify_unknown = false;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  Search search(s, p);
  search.run(*ev, Limits{20});
  std::size_t legal = 0;
  for (std::size_t i = 0; i < search.root_children().size();) {
    const MoveLegality v = search.verify(i);
    CHECK(v != MoveLegality::Unknown);
    if (v == MoveLegality::Legal) {
      ++legal;
      CHECK(search.root_children()[i].legality == MoveLegality::Legal);
      ++i;
    }
  }
  CHECK_EQ(legal, std::size_t{1});  // Kxc2
  CHECK_EQ(search.root_children().size(), std::size_t{1});
  CHECK_NOTHROW(search.check_invariants());
}

TEST(mcts_whole_turn_children_are_the_legal_actions) {
  SearchParams p = fast();
  p.split = TurnSplit::WholeTurn;
  p.widening = false;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 2);
  const GameState s = two_timelines_after(2);
  Search search(s, p);
  search.run(*ev, Limits{60});
  const ActionList acts = list_legal_actions(s, 4096, Detector::Hypercuboid);
  REQUIRE(acts.stats.exhaustive);
  const std::vector<RootChild> children = search.root_children();
  CHECK_EQ(children.size(), acts.actions.size());
  for (const RootChild& c : children) {
    GameState t = s;
    for (const Move& m : c.moves) {
      REQUIRE(is_pseudo_legal(t, m));
      t.apply(m);
    }
    CHECK(t.submission_status() == SubmitStatus::Safe);
  }
  // Priors sum to one.
  double sum = 0;
  for (const RootChild& c : children) sum += c.prior;
  CHECK(std::abs(sum - 1.0) < 1e-9);
  CHECK_NOTHROW(search.check_invariants());
}

TEST(mcts_whole_turn_listing_stopped_early_is_never_proven_lost) {
  SearchParams p = fast();
  p.split = TurnSplit::WholeTurn;
  p.max_actions = 1;  // every listing stops at its first action
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  Search search(forced_loss(), p);
  search.run(*ev, Limits{3000});
  CHECK(search.stats().truncated_listings > 0);
  CHECK(search.node(search.root()).hi == 1);  // an incomplete list proves no loss
  CHECK_NOTHROW(search.check_invariants());
}

TEST(mcts_board_move_dead_ends_are_pruned_not_scored) {
  // The test seam: pseudo-legal children. On one board every move is the whole
  // action, so a move that leaves the king capturable has no legal completion -
  // a dead end, pruned (§8.2), never backed up as a loss.
  SearchParams p = fast();
  p.legal_moves_only = false;
  p.widening = false;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  const GameState s = forced_draw();  // only Kxc2 is legal
  Search search(s, p);
  search.run(*ev, Limits{50});
  CHECK(search.stats().dead_ends > 0);
  MoveList legal;
  legal_moves(s, legal);
  REQUIRE_EQ(legal.size(), std::size_t{1});
  for (std::size_t i = 0; i < search.node(search.root()).edge_count; ++i) {
    const EdgeView e = search.edge(search.root(), i);
    const bool is_legal = *e.move == legal[0];
    CHECK_EQ(e.pruned, !is_legal);
    if (e.pruned) CHECK_EQ(e.visits, std::uint32_t{0});
  }
  CHECK(search.root_solved());  // the one legal child decides it: a draw
  CHECK_NOTHROW(search.check_invariants());
}

TEST(mcts_q4_other_reading_offers_submission_as_a_child) {
  // Under moves_after_shift the mover may go on after the present has shifted:
  // "submit" is then a child of its own (the board-move split).
  RuleOptions rules;
  rules.moves_after_shift = true;
  SearchParams p = fast();
  p.widening = false;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 3);
  Search search(variant("Standard", rules), p);
  search.run(*ev, Limits{120});
  std::size_t submit_nodes = 0;
  for (NodeId id = 0; id < search.node_count(); ++id) {
    const NodeView n = search.node(id);
    if (!n.expanded || n.turn_start) continue;
    for (std::size_t i = 0; i < n.edge_count; ++i) {
      if (search.edge(id, i).submit) ++submit_nodes;
    }
  }
  CHECK(submit_nodes > 0);
  CHECK_NOTHROW(search.check_invariants());
}

// ---- E1: widening, FPU, noise -------------------------------------------------------------

TEST(mcts_widening_follows_the_policy_entropy) {
  SearchParams p = fast();
  p.synchronous = true;  // one selection per batch: k(N) at every N
  // Uniform priors over the 20 moves of the Standard start: H = ln 20, so k0 =
  // max(2, 0.5 * 20) = 10.
  {
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    Search search(variant("Standard"), p);
    search.run(*ev, Limits{1});
    const NodeView root = search.node(search.root());
    CHECK(std::abs(root.k0 - 10.0) < 1e-9);
    CHECK_EQ(root.instantiated, std::size_t{10});  // k(0) = ceil(10 * 1^0.5)
    // Selections at N = 1 and N = 2 widen to ceil(10 sqrt 2) = 15, ceil(10 sqrt 3) = 18.
    search.run(*ev, Limits{2});
    CHECK_EQ(search.node(search.root()).instantiated, std::size_t{18});
  }
  // A sharp policy: H ~ 0, so k0 = the minimum, 2.
  {
    Sharp ev;
    Search search(variant("Standard"), p);
    search.run(ev, Limits{1});
    const NodeView root = search.node(search.root());
    CHECK(std::abs(root.k0 - 2.0) < 1e-6);
    CHECK_EQ(root.instantiated, std::size_t{2});  // ceil(2 * 1^0.5)
    // The children are in prior order: the sharp move first.
    CHECK(search.edge(search.root(), 0).prior > 0.99);
    search.run(ev, Limits{50});
    // Never a child beyond the instantiated ones.
    const NodeView after = search.node(search.root());
    for (std::size_t i = after.instantiated; i < after.edge_count; ++i) {
      CHECK_EQ(search.edge(search.root(), i).visits, std::uint32_t{0});
    }
  }
  // Without widening every child is instantiated at once.
  {
    p.widening = false;
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    Search search(variant("Standard"), p);
    search.run(*ev, Limits{1});
    CHECK_EQ(search.node(search.root()).instantiated, std::size_t{20});
  }
}

TEST(mcts_root_noise_is_seeded_and_only_at_the_root) {
  SearchParams p = fast();
  p.widening = false;
  p.root_noise = true;
  const auto priors = [&](std::uint64_t seed) {
    p.seed = seed;
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    Search search(variant("Standard"), p);
    search.run(*ev, Limits{30});
    std::vector<double> out;
    for (std::size_t i = 0; i < search.node(search.root()).edge_count; ++i) {
      out.push_back(search.edge(search.root(), i).prior);
    }
    // Below the root the priors are untouched: uniform.
    for (NodeId id = 1; id < search.node_count(); ++id) {
      const NodeView n = search.node(id);
      if (!n.expanded || n.edge_count == 0) continue;
      for (std::size_t i = 0; i < n.edge_count; ++i) {
        CHECK(std::abs(search.edge(id, i).prior - 1.0 / static_cast<double>(n.edge_count)) < 1e-12);
      }
    }
    return out;
  };
  const std::vector<double> a = priors(1);
  CHECK(a == priors(1));
  CHECK(a != priors(2));
  double sum = 0;
  for (const double x : a) sum += x;
  CHECK(std::abs(sum - 1.0) < 1e-9);
  CHECK(a.front() > a.back());  // sorted by the noised prior
}

TEST(mcts_value_distribution_helpers) {
  for (const double v : {-1.0, -0.63, 0.0, 0.25, 0.999, 1.0}) {
    const ValueMoments m = value_moments(two_hot(v));
    CHECK(std::abs(m.mean - v) < 1e-6);
    CHECK(m.variance < 0.0005);
  }
  std::array<float, kValueBins> flat{};
  flat.fill(1.0f);
  const ValueMoments m = value_moments(flat);
  CHECK(std::abs(m.mean) < 1e-12);
  CHECK(m.variance > 0.3);
  flat[3] = -1.0f;
  CHECK_THROWS_AS(value_moments(flat), std::invalid_argument);
  CHECK_EQ(value_bin_centre(0), -1.0);
  CHECK_EQ(value_bin_centre(25), 0.0);
  CHECK_EQ(value_bin_centre(50), 1.0);
}

TEST(mcts_node_keys_see_origins_only_when_asked) {
  // Two states that differ only in a timeline's branch origin: one Zobrist key,
  // two keys under the lineage reading (spec issue 30).
  const GameState start = variant("Standard");
  GameState a = start;
  for (int i = 0; i < 2; ++i) a = list_legal_actions(a, 1, Detector::Hypercuboid).actions.front().result;
  // Find a branching action, and replay it into a multiverse without origins.
  const ActionList acts = list_legal_actions(a, 4096, Detector::Hypercuboid);
  const LegalAction* branching = nullptr;
  for (const LegalAction& x : acts.actions) {
    for (const Move& m : x.moves) {
      if (m.branching()) branching = &x;
    }
  }
  REQUIRE(branching != nullptr);
  const GameState b = branching->result;
  std::vector<std::string> fen;
  for (const TimelineRef& t : b.multiverse().timelines()) {
    int h = t->first_half_turn();
    for (const BoardRef& board : t->boards()) {
      fen.push_back(fen_board_string(*board, t->position(), h, b.multiverse().even()));
      ++h;
    }
  }
  const GameState read = game_from_fen(fen, b.multiverse().board_size(), Parity::Odd, b.rules(), b.to_move());
  REQUIRE(same_position(read, b));
  REQUIRE(!(read == b));  // 5DFEN keeps no origin
  CHECK_EQ(node_key(read, NodeKey::State), node_key(b, NodeKey::State));
  CHECK(node_key(read, NodeKey::StateAndOrigins) != node_key(b, NodeKey::StateAndOrigins));
}

TEST(mcts_parameters_are_validated) {
  SearchParams p = fast();
  p.cpuct = -1;
  CHECK_THROWS_AS(Search(variant("Standard"), p), std::invalid_argument);
  p = fast();
  p.split = TurnSplit::WholeTurn;
  CHECK_THROWS_AS(Search(variant("Standard"), p, false), std::invalid_argument);
  p = fast();
  p.expand.pin = true;
  CHECK_THROWS_AS(Search(variant("Standard"), p), std::invalid_argument);
}
