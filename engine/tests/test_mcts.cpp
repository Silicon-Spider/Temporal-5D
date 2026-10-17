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
#include <array>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/expand.hpp"
#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_game_tree.hpp"
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
  bool proof_channel;  // off by rule (H41); the invariants hold either way
};
const std::vector<Config> kConfigs = {
    {"ply tree, solver", TurnSplit::BoardMove, false, true, false, false, NodeKey::State, true},
    {"ply graph", TurnSplit::BoardMove, true, true, false, false, NodeKey::State, false},
    {"ply graph origins, solver", TurnSplit::BoardMove, true, true, true, true, NodeKey::StateAndOrigins, true},
    {"ply tree, no widening, E2+E3", TurnSplit::BoardMove, false, false, true, true, NodeKey::State, false},
    {"turn tree, solver", TurnSplit::WholeTurn, false, true, false, false, NodeKey::State, true},
    {"turn graph, E2+E3", TurnSplit::WholeTurn, true, true, true, true, NodeKey::State, false},
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
  p.proof_channel = c.proof_channel;
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

TEST(mcts_a_batch_is_bounded_in_visits_of_every_kind) {
  // A batch of 8 is 8 visits: the leaves sent for evaluation plus the visits
  // completed without the evaluator (here the mate and the turn cap's
  // terminals). Never a run's worth of terminal visits behind one pending
  // leaf's virtual loss.
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  SearchParams p = fast();
  p.batch_size = 8;
  p.proof_channel = false;
  p.turn_cap.turns = 1;  // every turn start after Black's first action is capped
  Search search(game({"[k4/4Q/1K3/5/5:0:1:w]"}, BoardSize{5, 5}), p);
  std::size_t gathers = 0;
  std::size_t completed_without = 0;
  for (int i = 0; i < 100; ++i) {
    const std::uint64_t before = search.stats().playouts;
    const std::size_t n = search.gather(8, 1u << 30);
    const std::uint64_t done = search.stats().playouts - before;
    CHECK(done + n <= 8);
    completed_without += static_cast<std::size_t>(done);
    ++gathers;
    if (n == 0) {
      if (done == 0) break;
      continue;
    }
    std::vector<Evaluation> out(n);
    ev->evaluate(search.requests(), out);
    search.deliver(out);
  }
  CHECK(gathers > 10);
  CHECK(completed_without > 0);
  CHECK_NOTHROW(search.check_invariants());
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
  for (const bool after_shift : {true, false}) {  // Q4: R8's reading, and the Guide-literal one
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    const GameState start = game({"[k4/4Q/1K3/5/5:0:1:w]"}, BoardSize{5, 5}, rules);
    for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
      for (const bool graph : {false, true}) {
        const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
        SearchParams p = fast();
        p.proof_channel = true;  // the in-tree solver under test (off by rule, H41)
        p.split = split;
        p.graph = graph;
        Search search(start, p);
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
        // Under the ply split the mating move leaves White's action open (D8):
        // its node is White's, proven won, and its submit edge reaches the mate.
        NodeId mate = chosen.child;
        if (split == TurnSplit::BoardMove) {
          const NodeView mid = search.node(mate);
          CHECK(!mid.turn_start);
          CHECK(mid.mover == Colour::White);
          CHECK_EQ(mid.lo, 1);
          mate = kNoNode;
          for (std::size_t i = 0; i < mid.edge_count; ++i) {
            const EdgeView e = search.edge(chosen.child, i);
            if (e.submit) mate = e.child;
          }
          REQUIRE(mate != kNoNode);
        }
        CHECK(search.node(mate).terminal == Terminal::Checkmate);
        CHECK(verdict_of(*search.node(mate).state) == Outcome::Checkmate);
        CHECK_NOTHROW(search.check_invariants());
        // Solved: more running consumes no evaluation.
        const std::uint64_t evaluations = search.stats().evaluations;
        search.run(*ev, Limits{50});
        CHECK_EQ(search.stats().evaluations, evaluations);
      }
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
    p.proof_channel = true;  // the in-tree solver under test (off by rule, H41)
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
      p.proof_channel = true;  // the in-tree solver under test (off by rule, H41)
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

namespace {

// Small positions where Q4's two readings differ: a 4x4 board on timeline 0
// that White must play at T1, and White's own timeline +1 with a board at T2 -
// ahead of the present, so optional: after the shift R8 lets White play it
// before submitting, the Guide-literal reading does not. Kings and a few
// pieces at random (seeded); positions the state rules refuse are skipped.
std::vector<GameState> two_board_positions(const RuleOptions& rules, int count, std::uint64_t seed) {
  t5d_test::Rng rng(seed);
  const char pieces[] = {'Q', 'R', 'N', 'B', 'q', 'r', 'n', 'b'};
  std::vector<GameState> out;
  for (int attempt = 0; attempt < 4000 && static_cast<int>(out.size()) < count; ++attempt) {
    std::vector<std::string> boards;
    for (const char* place : {":0:1:w]", ":+1:2:w]"}) {
      std::string cells(16, '.');
      const auto put = [&](char c) {
        for (;;) {
          const int sq = rng.below(16);
          if (cells[static_cast<std::size_t>(sq)] == '.') {
            cells[static_cast<std::size_t>(sq)] = c;
            return;
          }
        }
      };
      put('K');
      put('k');
      const int extra = rng.below(3);
      for (int i = 0; i < extra; ++i) put(pieces[rng.below(8)]);
      std::string fen = "[";
      for (int rank = 3; rank >= 0; --rank) {
        int blank = 0;
        for (int file = 0; file < 4; ++file) {
          const char c = cells[static_cast<std::size_t>(rank * 4 + file)];
          if (c == '.') {
            ++blank;
            continue;
          }
          if (blank > 0) fen += std::to_string(blank);
          blank = 0;
          fen += c;
        }
        if (blank > 0) fen += std::to_string(blank);
        if (rank > 0) fen += '/';
      }
      boards.push_back(fen + place);
    }
    try {
      GameState s = game(boards, BoardSize{4, 4}, rules, Colour::White);
      if (verdict_of(s) != Outcome::Ongoing) continue;
      out.push_back(std::move(s));
    } catch (const std::exception&) {
    }
  }
  return out;
}

}  // namespace

TEST(mcts_with_the_solver_off_terminals_are_valued_and_nothing_else_is_proven) {
  // The default until the user rules (H23; spec §11.1 item 2): a terminal is
  // valued as its game result, but no bound propagates - no non-terminal node
  // is solved, the search runs every playout, and final selection takes no
  // proof (no ProvenWin, SecureChild or AllLost). The mate in one is still
  // found, by its value.
  CHECK(!SearchParams{}.proof_channel);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const bool graph : {false, true}) {
      for (int which = 0; which < 3; ++which) {
        const GameState s = which == 0 ? mate_in_one() : which == 1 ? forced_draw() : forced_loss();
        SearchParams p = fast();
        p.split = split;
        p.graph = graph;
        const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
        Search search(s, p);
        search.run(*ev, Limits{1500});
        CHECK_NOTHROW(search.check_invariants());
        CHECK_EQ(search.stats().playouts, std::uint64_t{1500});
        CHECK(!search.root_solved());
        CHECK(search.stats().checkmates + search.stats().stalemates > 0);
        for (NodeId id = 0; id < search.node_count(); ++id) {
          const NodeView n = search.node(id);
          if (n.terminal != Terminal::None) continue;
          CHECK_EQ(n.lo, -1);
          CHECK_EQ(n.hi, 1);
        }
        for (const RootChild& c : search.root_children()) {
          CHECK_EQ(c.lo, -1);
          CHECK_EQ(c.hi, 1);
        }
        const auto [index, reason] = search.choose();
        CHECK(reason == ChoiceReason::Lcb || reason == ChoiceReason::Visits);
        if (which == 0) {
          // The chosen line mates: its moves, then the submission.
          GameState t = s;
          for (const Move& m : search.root_children().at(index).moves) t.apply(m);
          if (split == TurnSplit::WholeTurn || t.obligations_met()) {
            REQUIRE(t.submission_status() == SubmitStatus::Safe);
            t.submit();
            CHECK(verdict_of(t) == Outcome::Checkmate);
          }
          CHECK(search.root_children().at(index).q > 0.5);
        }
      }
    }
  }
}

TEST(mcts_solver_agrees_with_the_brute_force_game_tree_under_both_q4_readings) {
  // The search's proofs against E7's brute-force game tree (test_game_tree.hpp:
  // the reference's enumeration of every move sequence, plain minimax), which
  // shares none of the search's machinery: never opposite verdicts, and every
  // mate in one (Win at distance 1) proved. Under the board-move split the
  // proofs run through mid-action nodes and explicit submit edges (D8).
  std::size_t compared = 0;
  std::size_t search_proofs = 0;
  std::size_t mates_in_one = 0;
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    std::vector<GameState> positions = two_board_positions(rules, 10, after_shift ? 41 : 43);
    positions.push_back(game({"[k4/4Q/1K3/5/5:0:1:w]"}, BoardSize{5, 5}, rules));
    positions.push_back(game({"[3k/4/B1qK/4:0:1:w]"}, BoardSize{4, 4}, rules));
    positions.push_back(game({"[4/K2b/3n/1k2:0:1:w]"}, BoardSize{4, 4}, rules));
    for (const GameState& s : positions) {
      t5d_test::GameTree tree;
      const t5d_test::GameValue truth = tree.solve(s, 2);
      if (truth.kind == t5d_test::GameValue::Kind::Unknown) continue;
      for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
        SearchParams p = fast();
        p.proof_channel = true;  // the in-tree solver under test (off by rule, H41)
        p.split = split;
        p.legality = LegalityOptions::exact();
        const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
        Search search(s, p);
        search.run(*ev, Limits{3000});
        CHECK_NOTHROW(search.check_invariants());
        ++compared;
        const NodeView root = search.node(search.root());
        using Kind = t5d_test::GameValue::Kind;
        if (root.lo == root.hi) ++search_proofs;
        if (root.lo == 1) CHECK(truth.kind == Kind::Win || truth.kind == Kind::Other);
        if (root.hi == -1) CHECK(truth.kind == Kind::Loss || truth.kind == Kind::Other);
        if (root.lo == 0 && root.hi == 0) CHECK(truth.kind != Kind::Win && truth.kind != Kind::Loss);
        if (truth.kind == Kind::Win) CHECK_EQ(root.hi, 1);
        if (truth.kind == Kind::Loss) CHECK_EQ(root.lo, -1);
        if (truth.kind == Kind::Win && truth.distance == 1) {
          ++mates_in_one;
          CHECK_EQ(root.lo, 1);  // complete for a mate in one
        }
      }
    }
  }
  CHECK(compared >= 20);
  CHECK(search_proofs > 0);
  CHECK(mates_in_one >= 4);
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
        SearchParams sp = params_of(c);
        sp.proof_channel = true;
        Search search(s, sp);
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
    const int sibling_moves = search.node(sibling).moves_made;
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
    CHECK(search.cache()->find(sibling_key, sibling_moves, 0, sibling_state, true) != nullptr);
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
  search.advance_to(far, 0);
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
    std::size_t submits = 0;
    for (std::size_t i = 0; i < n.edge_count; ++i) {
      const EdgeView e = search.edge(id, i);
      if (e.submit) {
        ++submits;
      } else {
        edges.push_back(*e.move);
        // (a) under the default budget: a legal move, or one flagged unknown.
        if (std::find(legal.begin(), legal.end(), *e.move) == legal.end()) CHECK(e.unknown || e.pruned);
      }
    }
    CHECK(edges.size() >= legal.size());
    for (const Move& m : legal) CHECK(std::find(edges.begin(), edges.end(), m) != edges.end());
    // D8: a submit edge exactly where submitting is possible (T5).
    CHECK_EQ(submits, n.state->obligations_met() ? std::size_t{1} : std::size_t{0});
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
    // The legal moves, and (D8) the submit edge where submitting is possible.
    const std::size_t submit = n.state->obligations_met() ? 1 : 0;
    CHECK_EQ(n.edge_count, legal.size() + submit);
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
  // but proves nothing (e); verified, it proves the draw - unless the
  // verification's own budget leaves it unknown, when it still proves nothing.
  const GameState s = forced_draw();
  for (const int mode : {0, 1, 2}) {  // unverified; verified exactly; verified with no budget
    const bool verify = mode != 0;
    SearchParams p = fast();
    p.proof_channel = true;  // the in-tree solver under test (off by rule, H41)
    p.widening = false;
    p.legality = LegalityOptions::budgeted(0, 0);
    p.verify_unknown = verify;
    if (mode == 2) p.verify_legality = LegalityOptions::budgeted(0, 0);
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    Search search(s, p);
    search.run(*ev, Limits{300});
    CHECK(search.stats().unknown_edges > 0);
    const NodeView root = search.node(search.root());
    CHECK_EQ(root.hi, 0);
    if (mode == 1) {
      CHECK(search.root_solved());
      CHECK_EQ(root.lo, 0);
    } else {
      CHECK(!search.root_solved());
      CHECK_EQ(root.lo, -1);
      // Final selection sees no proof through the unknown edge either: no
      // secure child, no proven anything - the ordinary rule.
      for (const RootChild& c : search.root_children()) {
        if (c.legality == MoveLegality::Unknown) CHECK_EQ(c.lo, -1);
      }
      const ChoiceReason reason = search.choose().reason;
      CHECK(reason == ChoiceReason::Lcb || reason == ChoiceReason::Visits);
    }
    if (mode == 2) {
      // Verified and still unknown: the edges stay flagged.
      CHECK(search.stats().verified_unknown > 0);
      std::uint64_t still_unknown = 0;
      for (NodeId id = 0; id < search.node_count(); ++id) {
        for (std::size_t i = 0; i < search.node(id).edge_count; ++i) {
          const EdgeView e = search.edge(id, i);
          if (e.verified && e.unknown) ++still_unknown;
        }
      }
      CHECK_EQ(still_unknown, search.stats().verified_unknown);
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
  p.max_actions = 4096;  // every action listed (the placeholder default is 256)
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
  p.proof_channel = true;  // the in-tree solver under test (off by rule, H41)
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

// ---- D8: submitting is an action (rulebook R8) --------------------------------------------

namespace {

// A position `actions` legal actions into a variant, under `rules`.
GameState played(const char* name, const RuleOptions& rules, int actions, std::uint64_t seed) {
  GameState s = variant(name, rules);
  t5d_test::Rng rng(seed);
  for (int a = 0; a < actions; ++a) {
    const ActionList list = list_legal_actions(s, 32, Detector::Hypercuboid);
    if (list.actions.empty()) break;
    s = list.actions[static_cast<std::size_t>(rng.below(static_cast<int>(list.actions.size())))].result;
  }
  return s;
}

}  // namespace

TEST(mcts_submit_is_an_edge_where_possible_legal_only_when_safe_never_automatic) {
  // Under the board-move split: a submit edge exactly where submitting is
  // possible (the obligations met, T5), pruned exactly where the submission
  // fails C4; only a submit edge changes the acting player, and it reaches a
  // turn start; a move never submits. Under both Q4 readings - R8's (moves may
  // follow the shift) and the Guide-literal one (none may: submit alone) - and
  // with the pseudo-legal test seam, whose children include moves that leave a
  // royal capturable (so submissions that are illegal).
  std::size_t submit_edges = 0;
  std::size_t illegal_submits = 0;
  std::size_t after_shift_moves = 0;
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    for (const bool seam : {false, true}) {
      for (const char* name : {"Standard", "Standard - Two Timelines", "Very Small - Open"}) {
        SearchParams p = fast();
        p.widening = false;
        p.legal_moves_only = !seam;
        const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 3);
        Search search(played(name, rules, 3, 7), p);
        search.run(*ev, Limits{150});
        CHECK_NOTHROW(search.check_invariants());
        for (NodeId id = 0; id < search.node_count(); ++id) {
          const NodeView n = search.node(id);
          if (!n.expanded || n.terminal != Terminal::None) continue;
          const GameState& s = *n.state;
          const bool possible = s.obligations_met();
          const bool legal = possible && s.submission_status() == SubmitStatus::Safe;
          CHECK_EQ(n.submit_possible, possible);
          CHECK_EQ(n.submit_legal, legal);
          std::size_t submits = 0;
          std::size_t moves = 0;
          for (std::size_t i = 0; i < n.edge_count; ++i) {
            const EdgeView e = search.edge(id, i);
            if (e.submit) {
              ++submits;
              ++submit_edges;
              CHECK_EQ(e.pruned, !legal);
              if (e.pruned) {
                ++illegal_submits;
                CHECK_EQ(e.visits, std::uint32_t{0});
                CHECK(e.child == kNoNode);
                CHECK_EQ(e.prior, 0.0);
              }
              if (e.child != kNoNode) {
                const NodeView c = search.node(e.child);
                CHECK(c.turn_start);
                CHECK(c.mover != n.mover);
              }
            } else {
              ++moves;
              if (e.child != kNoNode && !search.node(e.child).dead_end) {
                const NodeView c = search.node(e.child);
                CHECK(!c.turn_start);  // never submitted automatically
                CHECK(c.mover == n.mover);
              }
            }
          }
          CHECK_EQ(submits, possible ? std::size_t{1} : std::size_t{0});
          if (possible && !after_shift) CHECK_EQ(moves, std::size_t{0});  // Guide-literal: submit alone
          if (possible && after_shift) after_shift_moves += moves;
        }
      }
    }
  }
  CHECK(submit_edges > 0);
  CHECK(illegal_submits > 0);
  CHECK(after_shift_moves > 0);  // R8: moves offered beside the submit
}

TEST(mcts_node_keys_tell_apart_the_acting_player_and_what_was_played) {
  CHECK(SearchParams{}.node_key == NodeKey::StateAndOrigins);  // C23: lineage
  // After White's first move on the Standard board the obligations are met but
  // White still acts (D8); submitted, the same boards are Black's turn start.
  GameState mid = variant("Standard");
  t5d_test::play(mid, "(0T1)Pe2(0T1)e4");
  REQUIRE(mid.obligations_met());
  GameState submitted = mid;
  submitted.submit();
  for (const NodeKey k : {NodeKey::State, NodeKey::StateAndOrigins}) {
    CHECK(node_key(mid, k) != node_key(submitted, k));
  }
  // Two Timelines: White plays both boards; the two orders reach one state,
  // and one board played differs from the other one played.
  GameState a = variant("Standard - Two Timelines");
  GameState b = a;
  GameState c = a;
  t5d_test::play(a, "(-0T1)Pe2(-0T1)e4");
  t5d_test::play(c, "(+0T1)Pe2(+0T1)e4");
  CHECK(node_key(a, NodeKey::StateAndOrigins) != node_key(c, NodeKey::StateAndOrigins));
  t5d_test::play(a, "(+0T1)Pe2(+0T1)e4");
  t5d_test::play(b, "(+0T1)Pe2(+0T1)e4");
  t5d_test::play(b, "(-0T1)Pe2(-0T1)e4");
  CHECK(a == b);
  CHECK_EQ(node_key(a, NodeKey::StateAndOrigins), node_key(b, NodeKey::StateAndOrigins));
  // In a graph search the submit edge leads to a node of its own.
  SearchParams p = fast();
  p.graph = true;
  p.widening = false;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  Search search(variant("Standard"), p);
  search.run(*ev, Limits{80});
  std::size_t checked = 0;
  for (NodeId id = 0; id < search.node_count(); ++id) {
    const NodeView n = search.node(id);
    for (std::size_t i = 0; i < n.edge_count; ++i) {
      const EdgeView e = search.edge(id, i);
      if (!e.submit || e.child == kNoNode) continue;
      ++checked;
      CHECK(e.child != id);
      CHECK(search.node(e.child).key != n.key);
    }
  }
  CHECK(checked > 0);
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
  CHECK_THROWS_AS(Search(variant("Standard"), p, 1), std::invalid_argument);  // whole turns: from turn starts
  p = fast();
  p.expand.pin = true;
  CHECK_THROWS_AS(Search(variant("Standard"), p), std::invalid_argument);
}

// ---- MCGS identity: a turn start is never a state during an action ------------------------

namespace {

// Steers a search down chosen lines: the `preferred` moves logit 8, every
// other move -8, submitting 8 where it is possible; values 0. `late` is one
// of the preferred moves. Mode 1 plays it EARLY: wherever it is on offer, the
// other choices (preferred moves and submitting) get 6.5, not 8. Mode 2 plays it
// LATE: before turn 2 it gets 6.5. A ratio of e^1.5: both lines are searched,
// the favoured one reached first.
class Steer final : public Evaluator {
 public:
  Steer(std::vector<std::string> preferred, std::string late, int mode)
      : preferred_(std::move(preferred)), late_(std::move(late)), mode_(mode) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const GameState& s = *batch[i].state;
      const MoveList& moves = batch[i].expansion->moves();
      std::vector<std::string> text;
      bool late_on_offer = false;
      for (const Move& m : moves) {
        text.push_back(to_string(m, s.multiverse().even()));
        late_on_offer = late_on_offer || text.back() == late_;
      }
      const float other = mode_ == 1 && late_on_offer ? 6.5f : 8.0f;
      out[i].logits.resize(moves.size());
      for (std::size_t m = 0; m < moves.size(); ++m) {
        float logit = -8.0f;
        if (text[m] == late_) {
          logit = mode_ == 2 && turn_of(s.present()) < 2 ? 6.5f : 8.0f;
        } else if (std::find(preferred_.begin(), preferred_.end(), text[m]) != preferred_.end()) {
          logit = other;
        }
        out[i].logits[m] = logit;
      }
      out[i].submit_logit = batch[i].expansion->submit_possible() ? other : std::nanf("");
      out[i].value = two_hot(0.0);
    }
  }

 private:
  std::vector<std::string> preferred_;
  std::string late_;
  int mode_;
};

struct Meeting {
  GameState root;
  GameState at_turn_start;   // line A: White's turn start
  GameState during_action;   // line B: the same GameState, White's action in progress
  std::vector<Move> line_b;  // B's moves from the root (each followed by a submit but the last)
};

// Timeline 0 at T1 (White's: mandatory) and White's timeline +1 at T2 (ahead of
// the present: optional). Line A plays both of White's boards in one action
// (+1 first, so that it is a line under either Q4 reading) and submits, then
// Black plays 0 and submits: White's turn start. Line B plays 0 and submits,
// Black plays 0 and submits, then White plays +1 - the first move of an action
// whose 0 board is still to play. The lines meet in one GameState (the same
// boards, player and present); only whether an action is in progress tells
// them apart.
Meeting meeting(const char* zero, const char* plus_one, const RuleOptions& rules) {
  const GameState root = game({zero, plus_one}, BoardSize{4, 4}, rules, Colour::White);
  GameState a = root;
  t5d_test::play(a, "(+1T2)Ka1(+1T2)b1");
  t5d_test::play(a, "(0T1)Ka1(0T1)a2");
  a.submit();
  t5d_test::play(a, "(0T1)Kd4(0T1)d3");
  a.submit();
  GameState b = root;
  std::vector<Move> line_b;
  line_b.push_back(t5d_test::play(b, "(0T1)Ka1(0T1)a2"));
  b.submit();
  line_b.push_back(t5d_test::play(b, "(0T1)Kd4(0T1)d3"));
  b.submit();
  line_b.push_back(t5d_test::play(b, "(+1T2)Ka1(+1T2)b1"));
  return Meeting{root, std::move(a), std::move(b), std::move(line_b)};
}

const std::vector<std::string> kMeetingLines = {"(0T1)Ka1(0T1)a2", "(0T1)Kd4(0T1)d3", "(+1T2)Ka1(+1T2)b1"};
const char* const kLate = "(+1T2)Ka1(+1T2)b1";

// The nodes holding `s`: the lowest id as a turn start and during an action
// (kNoNode: none), and how many of each.
struct Holders {
  NodeId start = kNoNode;
  NodeId mid = kNoNode;
  std::size_t starts = 0;
  std::size_t mids = 0;
};

Holders holders(const Search& search, const GameState& s) {
  Holders h;
  for (NodeId id = 0; id < search.node_count(); ++id) {
    const NodeView n = search.node(id);
    if (!(*n.state == s)) continue;
    if (n.turn_start) {
      if (h.starts++ == 0) h.start = id;
    } else {
      if (h.mids++ == 0) h.mid = id;
    }
  }
  return h;
}

}  // namespace

TEST(mcts_graph_keeps_a_turn_start_apart_from_the_same_state_during_an_action) {
  // The review's repro: with I11's cap by boards set at the meeting state's
  // board count, line A's turn start is capped (a terminal) while line B's
  // state - the same GameState, an action in progress - is no turn start and
  // can be neither capped nor ended (C5). Merged into one node, MCGS made the
  // mid-action state terminal, and match play's choose() threw there. Each
  // line reached first in turn (the merged node took the first one's kind).
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    const Meeting m = meeting("[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]", rules);
    REQUIRE(m.at_turn_start == m.during_action);
    REQUIRE(!m.during_action.obligations_met());
    REQUIRE_EQ(node_key(m.at_turn_start, NodeKey::StateAndOrigins),
               node_key(m.during_action, NodeKey::StateAndOrigins));
    const int cap = static_cast<int>(m.at_turn_start.multiverse().board_count());
    for (const bool graph : {false, true}) {
      for (const int order : {1, 2}) {
        SearchParams p = fast();
        p.graph = graph;
        p.legality = LegalityOptions::exact();
        p.turn_cap.boards = cap;
        Search search(m.root, p);
        Steer steer(kMeetingLines, kLate, order);
        search.run(steer, Limits{1500});
        CHECK_NOTHROW(search.check_invariants());  // submit edges into turn starts only, moves into actions
        const Holders h = holders(search, m.at_turn_start);
        REQUIRE(h.start != kNoNode);
        REQUIRE(h.mid != kNoNode);
        CHECK_EQ(order == 1, h.start < h.mid);  // the line reached first
        if (graph) {
          CHECK_EQ(h.starts, std::size_t{1});
          CHECK_EQ(h.mids, std::size_t{1});
        }
        for (NodeId id = 0; id < search.node_count(); ++id) {
          const NodeView n = search.node(id);
          if (!(*n.state == m.at_turn_start)) continue;
          CHECK(n.terminal == (n.turn_start ? Terminal::TurnCap : Terminal::None));
          CHECK(!n.dead_end);  // during the action White can still play 0 and submit
        }
        if (graph) {
          // Match play down line B, line A's turn start in the graph.
          const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
          search.advance_moves(std::span<const Move>(&m.line_b[0], 1));
          search.run(*ev, Limits{50});
          search.advance_moves({});
          search.run(*ev, Limits{50});
          search.advance_moves(std::span<const Move>(&m.line_b[1], 1));
          search.run(*ev, Limits{50});
          search.advance_moves({});
          search.run(*ev, Limits{50});
          search.advance_moves(std::span<const Move>(&m.line_b[2], 1));
          CHECK(search.root_state() == m.during_action);
          CHECK(!search.root_turn_start());
          CHECK(!search.root_terminal());
          search.run(*ev, Limits{100});
          CHECK_NOTHROW((void)search.choose());
          CHECK_NOTHROW(search.check_invariants());
        }
      }
    }
  }
}

TEST(mcts_graph_never_prunes_a_mating_submission_for_a_dead_end_elsewhere) {
  // The review's second repro: a black knight on timeline 0 (d1) attacks +1's
  // b1. Line A's meeting state is White's turn start with every White action
  // leaving the king on +1 capturable: checkmate. Line B's - the same GameState
  // - is a state during White's action from which no legal completion exists:
  // a dead end, pruned. Merged, MCGS pruned Black's mating submission as a dead
  // end, or scored the dead end as checkmate.
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    const Meeting m = meeting("[3k/4/4/K2n:0:1:w]", "[3k/4/4/K3:+1:2:w]", rules);
    REQUIRE(m.at_turn_start == m.during_action);
    REQUIRE(verdict_of(m.at_turn_start) == Outcome::Checkmate);
    REQUIRE(can_finish_legally(m.during_action) == std::optional<bool>(false));
    // Black's state with the mating submission to make (line A).
    GameState black_mid = m.root;
    t5d_test::play(black_mid, "(+1T2)Ka1(+1T2)b1");
    t5d_test::play(black_mid, "(0T1)Ka1(0T1)a2");
    black_mid.submit();
    t5d_test::play(black_mid, "(0T1)Kd4(0T1)d3");
    REQUIRE(black_mid.submission_status() == SubmitStatus::Safe);
    for (const bool graph : {false, true}) {
      for (const int order : {1, 2}) {
        SearchParams p = fast();
        p.graph = graph;
        p.proof_channel = true;
        p.legal_moves_only = false;  // the pseudo-legal seam: B's move is offered, its child a dead end
        Search search(m.root, p);
        Steer steer(kMeetingLines, kLate, order);
        search.run(steer, Limits{1500});
        CHECK_NOTHROW(search.check_invariants());
        const Holders h = holders(search, m.at_turn_start);
        REQUIRE(h.start != kNoNode);
        REQUIRE(h.mid != kNoNode);
        CHECK_EQ(order == 1, h.start < h.mid);
        std::size_t mating_submits = 0;
        for (NodeId id = 0; id < search.node_count(); ++id) {
          const NodeView n = search.node(id);
          if (*n.state == m.at_turn_start) {
            CHECK(n.terminal == (n.turn_start ? Terminal::Checkmate : Terminal::None));
            CHECK_EQ(n.dead_end, !n.turn_start);
          }
          if (*n.state == black_mid && !n.turn_start && n.expanded) {
            for (std::size_t i = 0; i < n.edge_count; ++i) {
              const EdgeView e = search.edge(id, i);
              if (!e.submit) continue;
              ++mating_submits;
              CHECK(!e.pruned);
              if (e.child != kNoNode) {
                CHECK(search.node(e.child).terminal == Terminal::Checkmate);
                CHECK_EQ(n.lo, 1);  // Black's view: a proven win
              }
            }
          }
        }
        CHECK(mating_submits >= 1);
      }
    }
  }
}

// ---- A6: the moves made in the action, in a node's and an evaluation's identity -----------

namespace {

// Prefers the given moves (logit 8, every other move -8) and submitting (8),
// and values a leaf by the moves made in its action alone (register A6: the
// network sees the count) - so an evaluation served for another count than
// the leaf's shows. It records what it is asked, batch by batch.
class CountAware final : public Evaluator {
 public:
  explicit CountAware(std::vector<std::string> preferred) : preferred_(std::move(preferred)) {}
  static double value_of(int moves_made) { return -0.6 + 0.3 * std::min(moves_made, 4); }
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    batches.emplace_back();
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const EvalRequest& r = batch[i];
      if (r.moves_made != r.expansion->moves_made()) ++disagreements;
      batches.back().push_back(Asked{*r.state, r.moves_made});
      const MoveList& moves = r.expansion->moves();
      out[i].logits.resize(moves.size());
      for (std::size_t m = 0; m < moves.size(); ++m) {
        const std::string text = to_string(moves[m], r.state->multiverse().even());
        const bool preferred = std::find(preferred_.begin(), preferred_.end(), text) != preferred_.end();
        out[i].logits[m] = preferred ? 8.0f : -8.0f;
      }
      out[i].submit_logit = r.expansion->submit_possible() ? 8.0f : std::nanf("");
      out[i].value = two_hot(value_of(r.moves_made));
    }
  }
  struct Asked {
    GameState state;
    int moves_made = 0;
  };
  std::vector<std::vector<Asked>> batches;
  int disagreements = 0;

 private:
  std::vector<std::string> preferred_;
};

// Every expanded node's evaluation is its own count's (CountAware).
void check_own_evaluations(const Search& search) {
  for (NodeId id = 0; id < search.node_count(); ++id) {
    const NodeView n = search.node(id);
    if (!n.expanded) continue;
    if (std::abs(n.eval_mean - CountAware::value_of(n.moves_made)) >= 1e-6) {
      std::fprintf(stderr, "  node %u: %d moves made, evaluated %.6f\n", static_cast<unsigned>(id), n.moves_made,
                   n.eval_mean);
      CHECK(false);
    }
  }
}

// The nodes holding `s`, by their moves made.
std::map<int, std::vector<NodeId>> holders_by_count(const Search& search, const GameState& s) {
  std::map<int, std::vector<NodeId>> out;
  for (NodeId id = 0; id < search.node_count(); ++id) {
    const NodeView n = search.node(id);
    if (*n.state == s) out[n.moves_made].push_back(id);
  }
  return out;
}

std::size_t parents_of(const Search& search, NodeId child) {
  std::size_t parents = 0;
  for (NodeId id = 0; id < search.node_count(); ++id) {
    for (std::size_t i = 0; i < search.node(id).edge_count; ++i) parents += search.edge(id, i).child == child ? 1u : 0u;
  }
  return parents;
}

// White's 0 at T1 (mandatory), and +1 and -1 at T2 (White's and Black's first
// timelines, both active): line C plays 0, submits, Black plays 0 and submits,
// then White plays +1 and -1 - two moves into an action whose 0 is still owed;
// line D plays +1 and 0, submits, Black plays 0 and submits, then White plays
// -1 - one move. The same boards, the same player, the same present: one
// GameState, with 2 and 1 moves made in its action (the plan review's example
// of A6's count across turns).
struct Counted {
  GameState root;
  GameState two;   // line C's state
  GameState one;   // line D's
};

Counted counted(const RuleOptions& rules) {
  const GameState root =
      game({"[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]", "[3k/4/4/K3:-1:2:w]"}, BoardSize{4, 4}, rules, Colour::White);
  GameState c = root;
  t5d_test::play(c, "(0T1)Ka1(0T1)a2");
  c.submit();
  t5d_test::play(c, "(0T1)Kd4(0T1)d3");
  c.submit();
  t5d_test::play(c, "(+1T2)Ka1(+1T2)b1");
  t5d_test::play(c, "(-1T2)Ka1(-1T2)b1");
  GameState d = root;
  t5d_test::play(d, "(+1T2)Ka1(+1T2)b1");
  t5d_test::play(d, "(0T1)Ka1(0T1)a2");
  d.submit();
  t5d_test::play(d, "(0T1)Kd4(0T1)d3");
  d.submit();
  t5d_test::play(d, "(-1T2)Ka1(-1T2)b1");
  return Counted{root, std::move(c), std::move(d)};
}

const std::vector<std::string> kCountedLines = {"(0T1)Ka1(0T1)a2", "(0T1)Kd4(0T1)d3", "(+1T2)Ka1(+1T2)b1",
                                                "(-1T2)Ka1(-1T2)b1"};

}  // namespace

TEST(mcts_the_evaluation_cache_keeps_a_turn_start_apart_from_the_same_state_during_an_action) {
  // EQ-06 under A6 (§15.2: "same state + epoch -> same evaluation"): the
  // network sees the moves made in the action, so line A's turn start (none
  // made) and line B's state during White's action (one made) - one GameState
  // - are two inputs: two evaluations, two cache entries, each its own. Keyed
  // by the state and the epoch alone, the cache served one line's evaluation
  // for the other.
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    const Meeting m = meeting("[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]", rules);
    REQUIRE(m.at_turn_start == m.during_action);
    for (const bool graph : {false, true}) {
      SearchParams p = fast();
      p.graph = graph;
      p.legality = LegalityOptions::exact();
      Search search(m.root, p);
      CountAware ev(kMeetingLines);
      search.run(ev, Limits{1500});
      CHECK_NOTHROW(search.check_invariants());
      CHECK_EQ(ev.disagreements, 0);  // each request's count is its expansion's
      check_own_evaluations(search);
      const Holders h = holders(search, m.at_turn_start);
      REQUIRE(h.start != kNoNode);
      REQUIRE(h.mid != kNoNode);
      CHECK_EQ(search.node(h.start).moves_made, 0);
      CHECK_EQ(search.node(h.mid).moves_made, 1);
      const EvalCache* cache = search.cache();
      REQUIRE(cache != nullptr);
      const std::uint64_t key = search.node(h.start).key;
      const Evaluation* a = cache->find(key, 0, 0, m.at_turn_start, true);
      const Evaluation* b = cache->find(key, 1, 0, m.during_action, true);
      REQUIRE(a != nullptr);
      REQUIRE(b != nullptr);
      CHECK(std::abs(value_moments(a->value).mean - CountAware::value_of(0)) < 1e-6);
      CHECK(std::abs(value_moments(b->value).mean - CountAware::value_of(1)) < 1e-6);
      // The evaluator was asked for both inputs.
      std::set<int> asked;
      for (const std::vector<CountAware::Asked>& batch : ev.batches) {
        for (const CountAware::Asked& q : batch) {
          if (q.state == m.at_turn_start) asked.insert(q.moves_made);
        }
      }
      CHECK((asked == std::set<int>{0, 1}));
    }
  }
}

TEST(mcts_graph_keeps_equal_states_with_different_move_counts_apart) {
  // A6's count across turns (the plan review's example; build-log spec issue
  // 302): one GameState during White's action with 2 moves made (line C) and
  // with 1 (line D) - and as a turn start (line E). The network tells them
  // apart, so MCGS keeps them apart - its identity is the state with the moves
  // made in its action - while every ordering of one action's moves from one
  // turn start still merges (line C's +1 then -1, and -1 then +1, reach one
  // node).
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    const Counted c = counted(rules);
    REQUIRE(c.two == c.one);
    REQUIRE(!c.two.obligations_met());  // White's 0 still owed: an action in progress
    REQUIRE_EQ(node_key(c.two, NodeKey::StateAndOrigins), node_key(c.one, NodeKey::StateAndOrigins));
    for (const bool graph : {false, true}) {
      SearchParams p = fast();
      p.graph = graph;
      p.legality = LegalityOptions::exact();
      Search search(c.root, p);
      CountAware ev(kCountedLines);
      search.run(ev, Limits{3000});
      CHECK_NOTHROW(search.check_invariants());
      CHECK_EQ(ev.disagreements, 0);
      check_own_evaluations(search);
      // The counts met: 2 (line C), 1 (line D), and 0 - line E, White's first
      // action playing all three boards: after Black's reply, White's turn start
      // holds the same boards (a turn start, as in the meeting tests).
      const std::map<int, std::vector<NodeId>> by_count = holders_by_count(search, c.two);
      std::set<int> counts;
      for (const auto& [count, ids] : by_count) counts.insert(count);
      CHECK((counts == std::set<int>{0, 1, 2}));
      REQUIRE_EQ(by_count.count(1), std::size_t{1});
      REQUIRE_EQ(by_count.count(2), std::size_t{1});
      if (graph) {
        for (const auto& [count, ids] : by_count) CHECK_EQ(ids.size(), std::size_t{1});  // one node per count
        CHECK(by_count.at(1).front() != by_count.at(2).front());
        // Line C's two orders of +1 and -1: one node, two parents.
        CHECK(parents_of(search, by_count.at(2).front()) >= 2);
        CHECK(search.stats().transposition_hits > 0);
      }
      const EvalCache* cache = search.cache();
      REQUIRE(cache != nullptr);
      const std::uint64_t key = search.node(by_count.at(1).front()).key;
      const Evaluation* one = cache->find(key, 1, 0, c.one, true);
      const Evaluation* two = cache->find(key, 2, 0, c.two, true);
      REQUIRE(one != nullptr);
      REQUIRE(two != nullptr);
      CHECK(std::abs(value_moments(one->value).mean - CountAware::value_of(1)) < 1e-6);
      CHECK(std::abs(value_moments(two->value).mean - CountAware::value_of(2)) < 1e-6);
    }
  }
}

TEST(mcts_a_batch_asks_once_per_state_and_count) {
  // The batch's de-duplication matches the whole network input (A6): in a tree
  // (no merging) the meeting state is held by several nodes - line A's turn
  // start by two orders of White's first action, line B's mid-action state -
  // and a batch asks once per (state, moves made): never twice for one, never
  // once for both counts.
  const Meeting m = meeting("[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]", RuleOptions{});
  SearchParams p = fast();
  p.graph = false;
  p.legality = LegalityOptions::exact();
  p.batch_size = 32;
  Search search(m.root, p);
  CountAware ev(kMeetingLines);
  search.run(ev, Limits{3000});
  CHECK_NOTHROW(search.check_invariants());
  check_own_evaluations(search);
  std::size_t both_counts = 0;
  for (const std::vector<CountAware::Asked>& batch : ev.batches) {
    std::set<int> meeting_counts;
    for (std::size_t i = 0; i < batch.size(); ++i) {
      for (std::size_t j = i + 1; j < batch.size(); ++j) {
        CHECK(!(batch[i].moves_made == batch[j].moves_made && batch[i].state == batch[j].state));
      }
      if (batch[i].state == m.at_turn_start) meeting_counts.insert(batch[i].moves_made);
    }
    if (meeting_counts.size() == 2) ++both_counts;
  }
  CHECK(search.stats().batch_duplicates > 0);  // equal inputs did share a request
  CHECK(both_counts > 0);                      // equal states with two counts did not
  std::fprintf(stderr, "  %zu batches, %llu duplicates shared, %zu batches asking for both counts\n",
               ev.batches.size(), static_cast<unsigned long long>(search.stats().batch_duplicates), both_counts);
}

TEST(mcts_every_node_counts_the_moves_made_since_its_turn_start) {
  // §15.4's turn structure with A6's count, read from outside the search: from
  // the root's count, 0 below every submit edge (or whole action), the parent's
  // + 1 below every move - in a tree and in a graph (where every parent agrees),
  // from a turn start and from a root during an action; every request the
  // evaluator gets carries its node's count.
  GameState mid = two_timelines_after(2);
  t5d_test::play(mid, t5d_test::texts(mid).front());
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const bool graph : {false, true}) {
      for (const bool from_mid : {false, true}) {
        if (split == TurnSplit::WholeTurn && from_mid) continue;  // whole turns: from turn starts only
        SearchParams p = fast();
        p.split = split;
        p.graph = graph;
        p.max_actions = 64;
        const GameState root = from_mid ? mid : two_timelines_after(2);
        Search search(root, p, from_mid ? 1 : 0);
        CHECK_EQ(search.root_moves_made(), from_mid ? 1 : 0);
        CountAware ev({});
        search.run(ev, Limits{300});
        CHECK_NOTHROW(search.check_invariants());
        CHECK_EQ(ev.disagreements, 0);
        check_own_evaluations(search);
        std::size_t checked = 0;
        std::size_t above_one = 0;
        for (NodeId id = 0; id < search.node_count(); ++id) {
          const NodeView n = search.node(id);
          CHECK_EQ(n.turn_start, n.moves_made == 0);
          for (std::size_t i = 0; i < n.edge_count; ++i) {
            const EdgeView e = search.edge(id, i);
            if (e.child == kNoNode) continue;
            const int want = split == TurnSplit::WholeTurn || e.submit ? 0 : n.moves_made + 1;
            CHECK_EQ(search.node(e.child).moves_made, want);
            ++checked;
            above_one += want > 1 ? 1u : 0u;
          }
        }
        CHECK(checked > 20);
        if (split == TurnSplit::BoardMove) CHECK(above_one > 0);
      }
    }
  }
}

TEST(mcts_a_root_with_no_move_made_must_be_a_turn_start) {
  // A root's count is the caller's: 0 - a turn start - where the obligations are
  // met is refused (every turn starts with a board owed, T5), and so is a count
  // below 0 or, under whole turns, above 0; a state's count is not guessed.
  GameState s = variant("Standard");
  t5d_test::play(s, "(0T1)Pe2(0T1)e3");
  REQUIRE(s.obligations_met());
  CHECK_THROWS_WITH(Search(s, fast(), 0), std::invalid_argument, "T5");
  CHECK_THROWS_AS(Search(s, fast(), -1), std::invalid_argument);
  CHECK_NOTHROW(Search(s, fast(), 1));
  SearchParams whole = fast();
  whole.split = TurnSplit::WholeTurn;
  CHECK_THROWS_AS(Search(variant("Standard"), whole, 1), std::invalid_argument);
  // advance_to keeps the rule.
  Search search(variant("Standard"), fast());
  CHECK_THROWS_AS(search.advance_to(s, 0), std::invalid_argument);
  search.advance_to(s, 1);
  CHECK_EQ(search.root_moves_made(), 1);
  CHECK(!search.root_turn_start());
}

TEST(mcts_graph_dead_ends_have_no_legal_completion) {
  // The review's third repro (a fast legality budget, so moves stay unknown
  // until verified): every node marked a dead end truly has no legal
  // completion, and no move pruned for a dead-end child is legal.
  const GameState s = game({"[n3/n3/K3/3k:0:1:w]", "[B3/3k/K3/4:+1:2:w]"}, BoardSize{4, 4});
  for (const bool graph : {false, true}) {
    for (const BuiltinEvaluator kind : {BuiltinEvaluator::Uniform, BuiltinEvaluator::Random}) {
      SearchParams p = fast();
      p.graph = graph;
      p.proof_channel = true;
      p.legality = LegalityOptions::budgeted(0, 0);
      p.verify_legality = LegalityOptions::budgeted(0, 0);
      const std::unique_ptr<Evaluator> ev = make_evaluator(kind, 3);
      Search search(s, p);
      search.run(*ev, Limits{3000});
      CHECK_NOTHROW(search.check_invariants());
      std::size_t checked = 0;
      for (NodeId id = 0; id < search.node_count(); ++id) {
        const NodeView n = search.node(id);
        if (n.dead_end) {
          const t5d_test::Reference ref = t5d_test::reference_actions(*n.state, 200000);
          if (!ref.overflow) {
            ++checked;
            CHECK(ref.successors.empty());
          }
        }
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = search.edge(id, i);
          if (e.submit || !e.pruned || e.child == kNoNode || !search.node(e.child).dead_end) continue;
          std::vector<MoveLegality> leg;
          classify_moves(*n.state, std::span<const Move>(e.move, 1), leg, LegalityOptions::exact());
          CHECK(leg[0] == MoveLegality::Illegal);
        }
      }
      CHECK(checked > 0);
    }
  }
}

TEST(mcts_graph_invariants_hold_in_random_searches) {
  // Random two-board positions (intra-turn transpositions everywhere), random
  // evaluators, both Q4 readings, MCGS on: check_invariants holds every edge
  // into the right kind of node (a submit edge into a turn start, a move into a
  // state during an action) and no game end during an action; and no node
  // marked a dead end has a legal completion.
  std::size_t searches = 0;
  std::size_t checked = 0;
  std::size_t transpositions = 0;
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    const std::vector<GameState> positions = two_board_positions(rules, 6, after_shift ? 71 : 73);
    for (std::size_t i = 0; i < positions.size(); ++i) {
      for (const bool seam : {false, true}) {
        SearchParams p = fast();
        p.graph = true;
        p.proof_channel = true;
        p.legal_moves_only = !seam;
        p.legality = i % 2 == 0 ? LegalityOptions::budgeted(0, 0) : LegalityOptions::exact();
        p.turn_cap.boards = static_cast<int>(positions[i].multiverse().board_count()) + 3;
        const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 100 + i);
        Search search(positions[i], p);
        search.run(*ev, Limits{400});
        CHECK_NOTHROW(search.check_invariants());
        ++searches;
        transpositions += search.stats().transposition_hits;
        for (NodeId id = 0; id < search.node_count(); ++id) {
          const NodeView n = search.node(id);
          if (!n.dead_end) continue;
          const t5d_test::Reference ref = t5d_test::reference_actions(*n.state, 200000);
          if (ref.overflow) continue;
          ++checked;
          CHECK(ref.successors.empty());
        }
      }
    }
  }
  CHECK(searches >= 20);
  CHECK(checked > 0);
  CHECK(transpositions > 0);
}

// ---- What the mutation runs found unchecked (review C) ------------------------------------

namespace {

// Values by colour: `white` from White's view at every leaf - so +white from
// the acting player's view at White's nodes and -white at Black's - and
// uniform logits.
class ByColour final : public Evaluator {
 public:
  explicit ByColour(double white) : white_(white) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      out[i].logits.assign(batch[i].expansion->moves().size(), 0.0f);
      const bool white = batch[i].expansion->acting_player() == Colour::White;
      out[i].value = two_hot(white ? white_ : -white_);
    }
  }

 private:
  double white_;
};

// Uniform logits and one value distribution for every leaf.
class Fixed final : public Evaluator {
 public:
  explicit Fixed(std::array<float, kValueBins> value) : value_(value) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      out[i].logits.assign(batch[i].expansion->moves().size(), 0.0f);
      out[i].value = value_;
    }
  }

 private:
  std::array<float, kValueBins> value_;
};

}  // namespace

TEST(mcts_values_are_backed_up_from_the_acting_players_view) {
  // Every leaf worth +0.5 to White: every edge's mean, seen from its parent's
  // acting player, is +0.5 out of White's nodes and -0.5 out of Black's - the
  // sign flipping across a submit edge (or a whole-turn action) and nowhere
  // else. Under MCGS the residual corrections (Two Timelines' intra-turn
  // transpositions) must agree too.
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const bool graph : {false, true}) {
      SearchParams p = fast();
      p.split = split;
      p.graph = graph;
      p.max_actions = 96;
      ByColour ev(0.5);
      Search search(variant("Standard - Two Timelines"), p);
      search.run(ev, Limits{400});
      REQUIRE_EQ(search.stats().terminal_visits, std::uint64_t{0});
      CHECK_NOTHROW(search.check_invariants());
      if (graph && split == TurnSplit::BoardMove) CHECK(search.stats().residual_backups > 0);
      std::size_t flips = 0;
      for (NodeId id = 0; id < search.node_count(); ++id) {
        const NodeView n = search.node(id);
        const double mine = n.mover == Colour::White ? 0.5 : -0.5;
        if (n.weight > 0) CHECK(std::abs(n.value / n.weight - mine) < 1e-5);
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = search.edge(id, i);
          if (e.visits == 0) continue;
          CHECK(std::abs(e.value / e.weight - mine) < 1e-5);
          if (search.node(e.child).mover != n.mover) ++flips;
        }
      }
      CHECK(flips > 0);
    }
  }
}

TEST(mcts_the_submit_share_is_the_prior_without_a_submit_logit) {
  // With no submit logit from the evaluator (NaN: the built-ins), submit's
  // prior is submit_share where moves are open beside it (R8: White has played
  // its mandatory board 0, and may still play +1).
  for (const double share : {0.5, 0.3}) {
    SearchParams p = fast();
    p.submit_share = share;
    p.legality = LegalityOptions::exact();
    const Meeting m = meeting("[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]", RuleOptions{});
    GameState s = m.root;
    t5d_test::play(s, "(0T1)Ka1(0T1)a2");
    REQUIRE(s.obligations_met());
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    Search search(s, p, 1);  // one move made in the action (A6)
    search.run(*ev, Limits{20});
    const NodeView root = search.node(search.root());
    std::size_t submits = 0;
    double moves = 0;
    for (std::size_t i = 0; i < root.edge_count; ++i) {
      const EdgeView e = search.edge(search.root(), i);
      CHECK(!e.pruned);
      if (e.submit) {
        ++submits;
        CHECK(std::abs(e.prior - share) < 1e-12);
      } else {
        moves += e.prior;
      }
    }
    CHECK_EQ(submits, std::size_t{1});
    CHECK(std::abs(moves - (1 - share)) < 1e-12);
  }
}

TEST(mcts_the_turn_cap_binds_past_its_turn_and_at_its_board_count) {
  // I11's cap: a present PAST `turns` turns (at turn turns + 1), or `boards`
  // boards or more, at a turn start.
  GameState s = variant("Standard");
  for (int a = 0; a < 2; ++a) s = list_legal_actions(s, 1, Detector::Hypercuboid).actions.front().result;
  REQUIRE_EQ(turn_of(s.present()), 2);
  REQUIRE_EQ(s.multiverse().board_count(), std::size_t{3});
  CHECK(!(TurnCap{2, 0}.reached(s)));
  CHECK((TurnCap{1, 0}.reached(s)));
  CHECK((TurnCap{0, 3}.reached(s)));
  CHECK(!(TurnCap{0, 4}.reached(s)));
  CHECK(!(TurnCap{0, 0}.reached(s)));
}

TEST(mcts_the_evaluation_cache_is_per_epoch_and_exact) {
  const GameState a = variant("Standard");
  const GameState b = variant("Standard - Two Timelines");
  Evaluation ev;
  ev.logits.assign(20, 0.25f);
  ev.value = two_hot(0.5);
  EvalCache cache(16, std::size_t{1} << 20);
  cache.insert(7, 0, 0, a, ev);
  CHECK(cache.find(7, 0, 0, a, true) != nullptr);
  CHECK(cache.find(7, 0, 1, a, true) == nullptr);  // another epoch: a miss (§7.9)
  CHECK(cache.find(7, 0, 0, b, true) == nullptr);  // the same key, another state: a miss
  CHECK(cache.find(8, 0, 0, a, true) == nullptr);
  CHECK(cache.find(7, 1, 0, a, true) == nullptr);  // another count of moves made (A6): a miss
  // In searches sharing a cache: the root evaluated once per epoch.
  const SearchParams p = fast();
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  Recording rec(*uniform);
  const auto shared = std::make_shared<EvalCache>();
  {
    Search first(a, p, 0, shared);
    first.run(rec, Limits{1});
  }
  const std::size_t calls = rec.calls;
  CHECK_EQ(calls, std::size_t{1});
  {
    Search second(a, p, 0, shared);
    second.run(rec, Limits{1});
    CHECK_EQ(rec.calls, calls);  // a cache hit
    CHECK_EQ(second.stats().cache_hits, std::uint64_t{1});
  }
  {
    Search third(a, p, 0, shared);
    third.set_epoch(1);
    third.run(rec, Limits{1});
    CHECK_EQ(rec.calls, calls + 1);  // evaluated again under the new epoch
  }
}

TEST(mcts_graph_keys_that_collide_are_compared_exactly) {
  // Keys cut to 3 bits (the key_bits test seam): collisions everywhere. Every
  // edge still leads to the state its move (or submission) makes.
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    SearchParams p = fast();
    p.split = split;
    p.graph = true;
    p.key_bits = 3;
    p.max_actions = 64;
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 9);
    Search search(two_timelines_after(1), p);
    search.run(*ev, Limits{300});
    CHECK_NOTHROW(search.check_invariants());
    CHECK(search.stats().key_collisions > 0);
    std::size_t checked = 0;
    for (NodeId id = 0; id < search.node_count(); ++id) {
      const NodeView n = search.node(id);
      for (std::size_t i = 0; i < n.edge_count; ++i) {
        const EdgeView e = search.edge(id, i);
        if (e.child == kNoNode) continue;
        GameState t = *n.state;
        if (split == TurnSplit::WholeTurn) {
          for (const Move& m : e.action) t.apply(m);
          t.submit();
        } else if (e.submit) {
          t.submit();
        } else {
          t.apply(*e.move);
        }
        CHECK(*search.node(e.child).state == t);
        ++checked;
      }
    }
    CHECK(checked > 50);
  }
  SearchParams bad = fast();
  bad.key_bits = 0;
  CHECK_THROWS_AS(Search(variant("Standard"), bad), std::invalid_argument);
}

TEST(mcts_uncertainty_weighted_backup_down_weights_uncertain_values) {
  // E2: a leaf's backup weight is sigma_ref^2 / (sigma_ref^2 + var). Every leaf
  // here has variance 1 (half the mass at -1, half at +1): weight 0.16 / 1.16
  // per visit, below 1.
  std::array<float, kValueBins> split{};
  split[0] = 0.5f;
  split[kValueBins - 1] = 0.5f;
  Fixed ev(split);
  SearchParams p = fast();
  p.uncertainty_weighted_backup = true;
  Search search(variant("Standard"), p);
  search.run(ev, Limits{100});
  REQUIRE_EQ(search.stats().terminal_visits, std::uint64_t{0});
  const double r2 = p.variance_reference * p.variance_reference;
  const double w = r2 / (r2 + 1.0);
  const NodeView root = search.node(search.root());
  CHECK(std::abs(root.weight - w * root.visits) < 1e-6 * root.visits);
  for (std::size_t i = 0; i < root.edge_count; ++i) {
    const EdgeView e = search.edge(search.root(), i);
    if (e.visits > 0) CHECK(std::abs(e.weight - w * e.visits) < 1e-6 * e.visits);
  }
}

TEST(mcts_the_variance_bonus_never_falls_below_one) {
  // E3 multiplies U by clamp(1 + beta (sigma / sigma_ref - 1), 1, cap): with
  // every variance 0 (the uniform evaluator's value is one bin) the factor is
  // 1, and the search is the one without E3, bit for bit.
  std::uint64_t digests[2] = {0, 0};
  for (const bool e3 : {false, true}) {
    SearchParams p = fast();
    p.synchronous = true;
    p.variance_bonus = e3;
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
    Search search(variant("Standard"), p);
    search.run(*ev, Limits{300});
    digests[e3 ? 1 : 0] = search.digest();
  }
  CHECK_EQ(digests[0], digests[1]);
}

TEST(mcts_first_play_urgency_is_reduced_not_optimistic) {
  // FPU: an unvisited child scores Q(parent) - fpu_reduction * sqrt(visited
  // prior mass). With one child holding all the prior and every value 0, the
  // others (prior e^-30) score -0.25 and are never tried; an optimistic FPU
  // (+0.25) would try them once U of the first child fell below 0.25.
  SearchParams p = fast();
  p.synchronous = true;
  p.widening = false;
  Sharp ev;
  Search search(variant("Standard"), p);
  search.run(ev, Limits{400});
  const NodeView root = search.node(search.root());
  std::size_t visited = 0;
  for (std::size_t i = 0; i < root.edge_count; ++i) {
    if (search.edge(search.root(), i).visits > 0) ++visited;
  }
  CHECK_EQ(visited, std::size_t{1});
}

TEST(mcts_a_solved_child_is_visited_only_up_to_the_threshold) {
  // E9's visit threshold. White's Qd1-b3 stalemates Black ([k4/5/5/5/3QK]: the
  // queen on b3 takes a4, b4 and b5 from the king on a5 without giving check):
  // a child proven drawn. It is selected only while it has fewer than
  // solved_visit_cap visits, while ordinary children remain.
  const GameState s = game({"[k4/5/5/5/3QK:0:1:w]"}, BoardSize{5, 5});
  GameState t = s;
  const Move qb3 = t5d_test::play(t, "(0T1)Qd1(0T1)b3");
  t.submit();
  REQUIRE(verdict_of(t) == Outcome::Stalemate);
  SearchParams p = fast();
  p.proof_channel = true;
  p.synchronous = true;
  p.widening = false;
  p.turn_cap.turns = 1;  // nothing past Black's reply proven: the root stays open
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  Search search(s, p);
  search.run(*ev, Limits{3000});
  REQUIRE(!search.root_solved());
  bool found = false;
  for (const RootChild& c : search.root_children()) {
    if (c.moves.size() != 1 || !(c.moves[0] == qb3)) continue;
    found = true;
    CHECK_EQ(c.lo, 0);
    CHECK_EQ(c.hi, 0);
    CHECK(c.visits <= p.solved_visit_cap);
  }
  CHECK(found);
}

// ---- Call sequences (review B) ------------------------------------------------------------

TEST(mcts_advance_action_is_atomic) {
  // A failed advance_action changes nothing: a move that is not one, a
  // submission that is illegal, or an empty action at a turn start - the tree,
  // its statistics and its root are as they were.
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    SearchParams p = fast();
    p.split = split;
    p.max_actions = 64;
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 4);
    Search search(variant("Standard - Two Timelines"), p);
    search.run(*ev, Limits{100});
    const std::uint64_t digest = search.digest();
    const std::size_t nodes = search.node_count();
    const GameState root = search.root_state();
    // White's first legal action (two moves: both timelines are White's), and
    // a move of Black's reply.
    const ActionList white = list_legal_actions(root, 1, Detector::Hypercuboid);
    REQUIRE(!white.actions.empty());
    const std::vector<Move>& action = white.actions.front().moves;
    REQUIRE_EQ(action.size(), std::size_t{2});
    const ActionList black = list_legal_actions(white.actions.front().result, 1, Detector::Hypercuboid);
    REQUIRE(!black.actions.empty());
    const std::vector<Move> bad_move = {action[0], black.actions.front().moves[0]};
    CHECK_THROWS_AS(search.advance_action(bad_move), std::invalid_argument);
    // Every move fine, the submission not (a timeline left unplayed: T5).
    const std::vector<Move> unfinished = {action[0]};
    CHECK_THROWS_AS(search.advance_action(unfinished), std::invalid_argument);
    // Nothing at a turn start.
    CHECK_THROWS_AS(search.advance_action({}), std::invalid_argument);
    CHECK_EQ(search.digest(), digest);
    CHECK_EQ(search.node_count(), nodes);
    CHECK(search.root_state() == root);
    CHECK(search.root_turn_start());
    // The whole action goes through.
    search.advance_action(action);
    CHECK(search.root_state() == white.actions.front().result);
    CHECK(search.root_turn_start());
    CHECK_NOTHROW(search.check_invariants());
  }
}

TEST(mcts_the_epoch_does_not_change_under_a_waiting_batch) {
  // set_epoch between gather and deliver would file the batch's evaluations
  // under the new epoch: refused; after the batch, allowed.
  const auto cache = std::make_shared<EvalCache>();
  Search search(variant("Standard"), fast(), 0, cache);
  const std::size_t n = search.gather(1);
  REQUIRE_EQ(n, std::size_t{1});
  CHECK_THROWS_AS(search.set_epoch(5), std::invalid_argument);
  CHECK_EQ(search.epoch(), std::uint64_t{0});
  std::vector<Evaluation> ev(1);
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  uniform->evaluate(search.requests(), ev);
  search.deliver(ev);
  CHECK(cache->find(search.node(search.root()).key, 0, 0, variant("Standard"), true) != nullptr);
  CHECK(cache->find(search.node(search.root()).key, 0, 5, variant("Standard"), true) == nullptr);
  search.set_epoch(5);
  CHECK_EQ(search.epoch(), std::uint64_t{5});
}

namespace {

// Uniform answers; on its first call only, a submit logit of 25 for leaves
// where submitting is NOT possible (where the search never reads it) - so a
// stale value in a reused buffer would leak into a later leaf that reads it.
class StaleSubmit final : public Evaluator {
 public:
  explicit StaleSubmit(bool stale) : stale_(stale) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      out[i].logits.assign(batch[i].expansion->moves().size(), 0.0f);
      out[i].value = two_hot(0.0);
      if (stale_ && calls_ == 0 && !batch[i].expansion->submit_possible()) out[i].submit_logit = 25.0f;
    }
    ++calls_;
  }

 private:
  bool stale_;
  int calls_ = 0;
};

}  // namespace

TEST(mcts_run_resets_every_field_of_its_evaluation_buffers) {
  // run() reuses its Evaluation buffers between batches: a submit logit left
  // from an earlier leaf must not reach a later one. The meeting position: at
  // White's root no submit is possible; after (0T1)Ka1-a2 it is, beside +1's
  // moves - where a leaked logit of 25 would take submit's share.
  const Meeting m = meeting("[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]", RuleOptions{});
  std::uint64_t digests[2] = {0, 0};
  for (const bool stale : {false, true}) {
    SearchParams p = fast();
    p.synchronous = true;  // batches of one: the same buffer every time
    StaleSubmit ev(stale);
    Search search(m.root, p);
    search.run(ev, Limits{200});
    digests[stale ? 1 : 0] = search.digest();
  }
  CHECK_EQ(digests[0], digests[1]);
}

// ---- Validation and the C++ API's error kinds (reviews A and B) ---------------------------

TEST(mcts_every_real_parameter_must_be_finite) {
  // NaN and infinity poison selection (review A: virtual_loss = inf and
  // cpuct_variance_scale = NaN were accepted). Each is refused, by name.
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  struct Bad {
    const char* name;
    void (*set)(SearchParams&, double);
    double value;
  };
  const std::vector<Bad> bad = {
      {"virtual_loss", [](SearchParams& p, double v) { p.virtual_loss = v; }, inf},
      {"variance_reference", [](SearchParams& p, double v) { p.variance_reference = v; }, inf},
      {"dirichlet_alpha", [](SearchParams& p, double v) { p.dirichlet_alpha = v; }, inf},
      {"widening_min", [](SearchParams& p, double v) { p.widening_min = v; }, inf},
      {"cpuct_variance_scale", [](SearchParams& p, double v) { p.cpuct_variance_scale = v; }, nan},
      {"cpuct_variance_scale", [](SearchParams& p, double v) { p.cpuct_variance_scale = v; }, inf},
      {"policy_temperature", [](SearchParams& p, double v) { p.policy_temperature = v; }, inf},
      {"cpuct_scale_min/max", [](SearchParams& p, double v) { p.cpuct_scale_max = v; }, inf},
      {"variance_bonus_cap", [](SearchParams& p, double v) { p.variance_bonus_cap = v; }, inf},
      {"cpuct", [](SearchParams& p, double v) { p.cpuct = v; }, nan},
      {"submit_share", [](SearchParams& p, double v) { p.submit_share = v; }, nan},
      {"lcb_floor", [](SearchParams& p, double v) { p.lcb_floor = v; }, nan},
  };
  for (const Bad& b : bad) {
    SearchParams p = fast();
    b.set(p, b.value);
    try {
      Search search(variant("Standard"), p);
      CHECK(false);  // accepted
    } catch (const std::invalid_argument& e) {
      CHECK(std::string(e.what()).find(b.name) != std::string::npos);
    }
  }
  SearchParams p = fast();
  p.turn_cap.turns = -1;
  CHECK_THROWS_AS(Search(variant("Standard"), p), std::invalid_argument);
  CHECK_THROWS_AS((void)two_hot(nan), std::invalid_argument);
  CHECK_NOTHROW((void)two_hot(inf));  // clamped to +1
}

TEST(mcts_the_cpp_api_tells_the_callers_errors_apart) {
  // The caller's errors are std::invalid_argument (the C API's
  // T5D_ERR_ARGUMENT); any other logic_error is the search's own.
  Search search(variant("Standard"), fast());
  CHECK_THROWS_AS((void)search.choose(), std::invalid_argument);  // nothing visited yet
  CHECK_THROWS_AS((void)search.gather(0), std::invalid_argument);
  CHECK_THROWS_AS(search.deliver({}), std::invalid_argument);  // no batch waiting
  REQUIRE(search.gather(4) > 0);
  CHECK_THROWS_AS((void)search.gather(4), std::invalid_argument);  // one is waiting
  CHECK_THROWS_AS(search.advance(0), std::invalid_argument);
  search.abandon();
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  search.run(*ev, Limits{30});
  CHECK_THROWS_AS(search.advance(9999), std::invalid_argument);
  CHECK_THROWS_AS((void)search.verify(9999), std::invalid_argument);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS((void)search.choose_by_visits(nan, 0.5), std::invalid_argument);
  CHECK_THROWS_AS((void)search.choose_by_visits(1.0, nan), std::invalid_argument);
  CHECK_THROWS_AS((void)search.choose_by_visits(1.0, 2.0), std::invalid_argument);
  CHECK_NOTHROW((void)search.choose_by_visits(1.0, 0.25));
  // A huge max_leaves is bounded by the playouts, not overflowed.
  const std::size_t n = search.gather(std::numeric_limits<std::size_t>::max(), 16);
  CHECK(n <= 16);
  search.abandon();
  CHECK_NOTHROW(search.check_invariants());
}

TEST(mcts_a_deadline_stops_a_run) {
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  Search search(variant("Standard - Two Timelines"), fast());
  Limits l;
  l.playouts = std::uint64_t{1} << 40;
  l.seconds = 0.05;
  CHECK(search.run(*ev, l) == StopReason::Deadline);
  CHECK(search.stats().playouts > 0);
  l.seconds = -1;
  CHECK_THROWS_AS(search.run(*ev, l), std::invalid_argument);
  l.seconds = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(search.run(*ev, l), std::invalid_argument);
}

// ---- H18 as amended at CHECKPOINT 4: the acting player in every state key ------------------

TEST(mcts_h18_the_acting_player_is_in_every_key) {
  // Register H18 (amended): the ruleset ID and the acting player in every
  // state key - under Q4 a state whose present has shifted but whose action is
  // not yet submitted holds exactly the boards of the opponent's turn start
  // after the submission. The 5D Zobrist key itself, both node keys, the
  // evaluation cache's exact comparison and the MCGS graph keep them apart.
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;
    for (const char* name : {"Standard", "Standard - Two Timelines", "Very Small - Open"}) {
      const GameState start = variant(name, rules);
      // The acting player's first legal action, played and not submitted.
      GameState shifted = start;
      const ActionList acts = list_legal_actions(start, 1, Detector::Hypercuboid);
      REQUIRE(!acts.actions.empty());
      for (const Move& m : acts.actions.front().moves) shifted.apply(m);
      REQUIRE(shifted.obligations_met());
      REQUIRE(shifted.present() != start.present());  // the present has shifted
      GameState opponent = shifted;
      opponent.submit();
      // The same boards ...
      CHECK_EQ(fingerprint(shifted.multiverse()).fingerprint, fingerprint(opponent.multiverse()).fingerprint);
      CHECK(shifted.to_move() != opponent.to_move());
      // ... never the same key, nor the same state.
      CHECK(shifted.multiverse().zobrist() != opponent.multiverse().zobrist());  // t5d/zobrist: H18
      CHECK(shifted.zobrist() != opponent.zobrist());
      for (const NodeKey k : {NodeKey::State, NodeKey::StateAndOrigins}) {
        CHECK(node_key(shifted, k) != node_key(opponent, k));
      }
      CHECK(!(shifted == opponent));
      CHECK(!same_position(shifted, opponent));
      // The evaluation cache tells them apart even filed under one key.
      Evaluation ev;
      ev.logits.assign(4, 0.0f);
      ev.value = two_hot(0.5);
      EvalCache cache;
      const int made = static_cast<int>(acts.actions.front().moves.size());  // A6
      cache.insert(42, made, 0, shifted, ev);
      CHECK(cache.find(42, made, 0, shifted, true) != nullptr);
      CHECK(cache.find(42, made, 0, opponent, true) == nullptr);  // the count equal, the state not
      CHECK(cache.find(42, made, 0, opponent, false) == nullptr);
      CHECK(cache.find(42, 0, 0, opponent, true) == nullptr);
      // MCGS with keys cut to one bit (every key collides): the shifted root's
      // submit edge leads to a node of its own holding the opponent's turn
      // start, never back to the root, and every node's player is its state's.
      SearchParams p = fast();
      p.graph = true;
      p.key_bits = 1;
      p.widening = false;
      const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
      Search search(shifted, p, made);
      search.run(*uniform, Limits{120});
      CHECK_NOTHROW(search.check_invariants());
      std::size_t submits = 0;
      for (NodeId id = 0; id < search.node_count(); ++id) {
        const NodeView n = search.node(id);
        CHECK(n.mover == n.state->to_move());
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = search.edge(id, i);
          if (!e.submit || e.child == kNoNode) continue;
          ++submits;
          const NodeView c = search.node(e.child);
          CHECK(e.child != id);
          CHECK(c.mover != n.mover);
          if (id == search.root()) CHECK(*c.state == opponent);
        }
      }
      CHECK(submits > 0);
    }
  }
}
