// D2's tree form (t5d/mcts.hpp, TurnSplit::Tree; design spec §8.2 "Evaluating
// at turn boundaries"): the ply split's tree with the turn-start priors
// inherited inside a turn, the network called again only at the root, at turn
// starts and where an interaction gate fires.
//
//  * §15.2's row: "Tree form with every interaction gate firing = the
//    evaluate-every-node ply split" ("Inheritance is the tree form's only
//    difference") - under kGateAlways the search is the ply split's bit for
//    bit (H15's digest, the statistics, the root's children), on the named
//    variants' starts, random-play positions with many timelines and deep
//    travel, roots during an action and past the shift, several evaluators at
//    realistic scale, both legality settings and the pseudo-legal seam, every
//    tree and graph configuration, batched and synchronous, and across a
//    re-root.
//  * With no gate the network sees only the root and turn starts; with one
//    gate it sees exactly the mid-turn nodes whose incoming move fires it,
//    each gate's predicate recomputed here from the rules engine's state.
//  * The capture gate's reading on constructed positions; the inherited
//    logits (a J6 twin included) and the priors taken from them.
//  * §15.4: the invariants (turn structure, no game end during a turn, dead
//    ends pruned, never scored) and the parameters' checks.
//  * UNDER GRAPH SEARCH (§8.5 "The tree form under graph search", register C11
//    and D2; the 2026-11-15 review's decision 10, settling build-log spec
//    issue 301): the tests above run in a tree and in a graph alike
//    (TREE_AND_GRAPH_TEST), §15.2's row is held against the ply split's graph
//    search, and the graph's own: only evaluated nodes are graph nodes - the
//    gate on a move edge's own move decides, the nodes of moves that fire none
//    are path-local (one parent, in no table, never merged, duplicates of one
//    state included) - the network is asked once per evaluated state, a
//    merged node holds nothing of the path that made it, visits are conserved,
//    no edge leads back, and values are the acting player's.
//
// The positions are written for these tests or reached by seeded random play.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/check.hpp"
#include "t5d/expand.hpp"
#include "t5d/mcts.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"

using namespace t5d;
using namespace t5d::mcts;
using t5d_test::game;

// A test whose body runs twice, `graph` false and true: the tree form in a
// tree and under graph search (§8.5, C11: only evaluated nodes are graph
// nodes). The body sets SearchParams::graph from it; its counters and floors
// are each arm's own.
#define TREE_AND_GRAPH_TEST(name)                                                        \
  static void T5D_TEST_CAT(t5d_tree_and_graph_, name)(bool graph);                       \
  TEST(name) {                                                                           \
    for (const bool graph : {false, true}) T5D_TEST_CAT(t5d_tree_and_graph_, name)(graph); \
  }                                                                                      \
  static void T5D_TEST_CAT(t5d_tree_and_graph_, name)(bool graph)

namespace {

GameState variant(const char* name) {
  const Variant* v = find_variant(name);
  if (v == nullptr) throw std::logic_error(std::string("no variant ") + name);
  return variant_start(*v, RuleOptions{});
}

SearchParams fast(SearchParams p = {}) {
  p.expand.square_features = SquareFeatures::None;
  return p;
}

SearchParams tree(SearchParams p = fast()) {
  p.split = TurnSplit::Tree;
  return p;
}

// One root: a state and the moves made in its action (A6).
struct Root {
  std::string name;
  GameState state;
  int moves_made = 0;
};

// A move's coordinates: source, destination, promotion.
using Coord = std::tuple<int, int, int, int, int, int, int>;
Coord coord_of(const Move& m) {
  return Coord{m.from_timeline, m.from_half_turn, m.from_square, m.to_timeline,
               m.to_half_turn,  m.to_square,      m.promotes() ? static_cast<int>(m.promotion) + 1 : 0};
}

// `actions` random legal actions from `start` (seeded), stopping at a game end.
GameState random_play(GameState s, int actions, std::uint64_t seed) {
  t5d_test::Rng rng(seed);
  for (int a = 0; a < actions; ++a) {
    const ActionList list = list_legal_actions(s, 12, Detector::Hypercuboid);
    if (list.actions.empty()) break;
    s = list.actions[static_cast<std::size_t>(rng.below(static_cast<int>(list.actions.size())))].result;
  }
  return s;
}

// A root one move into a legal action from `turn_start` (seeded): the moves
// made 1 - past the shift where that move met the obligations.
Root mid_turn(const std::string& name, const GameState& turn_start, std::uint64_t seed) {
  const ActionList list = list_legal_actions(turn_start, 12, Detector::Hypercuboid);
  if (list.actions.empty()) return Root{name, turn_start, 0};
  t5d_test::Rng rng(seed);
  const LegalAction& a = list.actions[static_cast<std::size_t>(rng.below(static_cast<int>(list.actions.size())))];
  GameState s = turn_start;
  s.apply(a.moves.front());
  return Root{name + ", one move in", s, 1};
}

// The roots of §15.2's row: every named variant's start; random play from the
// multi-timeline ones (many timelines, deep travel) and the small ones; roots
// during an action.
std::vector<Root> roots() {
  std::vector<Root> out;
  for (const Variant& v : variants()) {
    out.push_back(Root{std::string(v.name), variant_start(v, RuleOptions{}), 0});
  }
  std::uint64_t seed = 1;
  for (const char* name : {"Standard - Two Timelines", "Misc - Timeline Battleground", "Misc - Timeline Tactician",
                           "Misc - Timeline Strategos", "Very Small - Open", "Standard"}) {
    for (const int actions : {4, 9}) {
      const GameState s = random_play(variant(name), actions, seed++);
      out.push_back(Root{std::string(name) + " after " + std::to_string(actions), s, 0});
      if (actions == 4) out.push_back(mid_turn(out.back().name, s, seed++));
    }
  }
  return out;
}

// Every statistic but the tree form's own counters and the timings.
bool same_stats(const mcts::SearchStats& a, const mcts::SearchStats& b) {
  return a.playouts == b.playouts && a.evaluations == b.evaluations && a.cache_hits == b.cache_hits &&
         a.batches == b.batches && a.collisions == b.collisions && a.batch_duplicates == b.batch_duplicates &&
         a.terminal_visits == b.terminal_visits && a.residual_backups == b.residual_backups &&
         a.transposition_hits == b.transposition_hits && a.key_collisions == b.key_collisions &&
         a.dead_ends == b.dead_ends && a.submit_edges == b.submit_edges && a.illegal_submits == b.illegal_submits &&
         a.unknown_edges == b.unknown_edges && a.verified_legal == b.verified_legal &&
         a.verified_illegal == b.verified_illegal && a.verified_unknown == b.verified_unknown &&
         a.mate_checks == b.mate_checks && a.unknown_verdicts == b.unknown_verdicts &&
         a.checkmates == b.checkmates && a.stalemates == b.stalemates && a.turn_caps == b.turn_caps &&
         a.stuck == b.stuck && a.generated_turns == b.generated_turns &&
         a.stopped_generators == b.stopped_generators &&
         a.nodes_created == b.nodes_created && a.nodes_collected == b.nodes_collected && a.reroots == b.reroots &&
         a.cache_purged == b.cache_purged && a.uncertainty_missing == b.uncertainty_missing &&
         a.last_stop == b.last_stop;
}

bool same_children(const Search& a, const Search& b) {
  const std::vector<RootChild> x = a.root_children();
  const std::vector<RootChild> y = b.root_children();
  if (x.size() != y.size()) return false;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (x[i].moves != y[i].moves || x[i].submit != y[i].submit || x[i].visits != y[i].visits ||
        x[i].prior != y[i].prior || x[i].prior_noise_free != y[i].prior_noise_free || x[i].q != y[i].q ||
        x[i].d != y[i].d || x[i].evaluated != y[i].evaluated || x[i].child_key != y[i].child_key ||
        x[i].lo != y[i].lo || x[i].hi != y[i].hi) {
      return false;
    }
  }
  return true;
}

// The ply split's search and the tree form's with every gate firing: one and
// the same, bit for bit - in a tree, and in a graph (`p.graph`: then every
// node is a graph node, none path-local). Returns the mid-turn leaves the tree
// form gated, and adds the transpositions it met to `transpositions`.
std::uint64_t compare(const Root& r, SearchParams p, Evaluator& ev, std::uint64_t playouts, const std::string& what,
                      std::uint64_t* transpositions = nullptr) {
  // With the proof channel on, the ladder's proofs (stage E10) enter each tree
  // when they arrive - asynchronous, so the two searches would differ by
  // timing alone (11.2's determinism debt); the synchronous mode's equality
  // with the ladder on is test_ladder.cpp's. The asset map stays on (it is a
  // function of the state).
  if (p.proof_channel) p.hunt = p.settling = false;
  p.split = TurnSplit::BoardMove;
  Search ply(r.state, p, r.moves_made);
  p.split = TurnSplit::Tree;
  p.tree_gates = kGateAlways;
  Search tree_form(r.state, p, r.moves_made);
  ply.run(ev, Limits{playouts});
  tree_form.run(ev, Limits{playouts});
  const auto fail = [&](const char* part) {
    ::t5d_test::record_failure(__FILE__, __LINE__, std::string("tree form with every gate firing != ply split (") +
                                                       part + "): " + what);
  };
  if (ply.digest() != tree_form.digest()) fail("digest");
  if (!same_stats(ply.stats(), tree_form.stats())) fail("statistics");
  if (!same_children(ply, tree_form)) fail("root children");
  CHECK_EQ(tree_form.stats().inherited_nodes, std::uint64_t{0});
  CHECK_NOTHROW(tree_form.check_invariants());
  std::uint64_t mid_turn_leaves = 0;
  for (NodeId id = 0; id < tree_form.node_count(); ++id) {
    const NodeView n = tree_form.node(id);
    CHECK(!n.inherited);
    CHECK(!n.path_local);
    if (n.evaluated && !n.turn_start && id != tree_form.root()) ++mid_turn_leaves;
  }
  if (!p.graph) CHECK_EQ(tree_form.stats().transposition_hits, std::uint64_t{0});
  if (transpositions != nullptr) *transpositions += tree_form.stats().transposition_hits;
  // Every mid-turn evaluation was a gate's (always), counted once per node -
  // answered by the network or by the cache.
  CHECK_EQ(tree_form.stats().gate_fired[4], mid_turn_leaves);
  for (std::size_t g = 0; g < 4; ++g) CHECK_EQ(tree_form.stats().gate_fired[g], std::uint64_t{0});
  // Persistence: re-rooted on the chosen child, they stay one search.
  if (!ply.root_terminal() && !ply.root_children().empty() && ply.choose().index < ply.root_children().size()) {
    const std::size_t pick = ply.choose().index;
    ply.advance(pick);
    tree_form.advance(pick);
    ply.run(ev, Limits{playouts / 2});
    tree_form.run(ev, Limits{playouts / 2});
    if (ply.digest() != tree_form.digest()) fail("digest after a re-root");
    if (!same_stats(ply.stats(), tree_form.stats())) fail("statistics after a re-root");
    CHECK_NOTHROW(tree_form.check_invariants());
  }
  return mid_turn_leaves;
}

// The configurations §15.2's row is checked under: every tree configuration of
// the search, both legality settings and the pseudo-legal seam, batched and
// synchronous - and under graph search (§8.5, C11), where with every gate
// firing every mid-turn node is evaluated, so every node is a graph node and
// the search is the ply split's graph search.
struct Variation {
  const char* name;
  void (*apply)(SearchParams&);
};
const std::vector<Variation> kVariations = {
    {"defaults", [](SearchParams&) {}},
    {"exact legality, solver", [](SearchParams& p) {
       p.legality = LegalityOptions::exact();
       p.proof_channel = true;
     }},
    {"no widening, E2+E3, origins", [](SearchParams& p) {
       p.widening = false;
       p.variance_scaled_cpuct = true;
       p.uncertainty_weighted_backup = true;
       p.variance_bonus = true;
       p.node_key = NodeKey::State;
     }},
    {"pseudo-legal children, solver", [](SearchParams& p) {
       p.legal_moves_only = false;
       p.proof_channel = true;
     }},
    {"synchronous, root noise", [](SearchParams& p) {
       p.synchronous = true;
       p.root_noise = true;
       p.seed = 11;
     }},
    {"no budget left (every move unknown), no cache", [](SearchParams& p) {
       p.legality = LegalityOptions::budgeted(0, 0);
       p.eval_cache_entries = 0;
       p.batch_size = 3;
     }},
    {"graph", [](SearchParams& p) { p.graph = true; }},
    {"graph, state key, no widening, E2+E3", [](SearchParams& p) {
       p.graph = true;
       p.node_key = NodeKey::State;
       p.widening = false;
       p.variance_scaled_cpuct = true;
       p.uncertainty_weighted_backup = true;
       p.variance_bonus = true;
     }},
    {"graph, exact legality, solver", [](SearchParams& p) {
       p.graph = true;
       p.legality = LegalityOptions::exact();
       p.proof_channel = true;
     }},
    {"graph, synchronous, root noise", [](SearchParams& p) {
       p.graph = true;
       p.synchronous = true;
       p.root_noise = true;
       p.seed = 11;
     }},
};

// The predicates of §8.2's gates, from the rules engine's state alone (the
// capture gate under its Any reading).
// "The boards still to move" (StillToMove): every playable board of the
// mover's, or the mandatory ones.
std::vector<int> still_to_move(const GameState& s, StillToMove still) {
  return still == StillToMove::Mandatory ? s.mandatory_timelines() : s.playable_timelines();
}

// R1's check status carried into an action (CheckChange::Passed): each board
// still to move passed - copied with the opponent to move, no en-passant
// square - and then a royal capture of the opponent's (t5d/check.hpp).
bool in_check_passed(const GameState& s, StillToMove still) {
  Multiverse mv = s.multiverse();
  for (const int p : still_to_move(s, still)) {
    BoardBuilder b(*mv.head(p));
    b.set_en_passant(kNoSquare);
    mv.append_board(p, b.publish());
  }
  return royal_capture_exists(mv, s.rules(), opposite(s.to_move()), s.inactive_timelines());
}

// (i) under the Mandatory reading: a non-branching jump onto a mandatory
// board - a jump onto an optional one (an inactive timeline's, T6) is not
// "between" the boards still to move.
bool predicate(std::uint32_t gate, const GameState& before, const Move& m, const GameState& after,
               CheckChange check = CheckChange::Passed, PresentShift present = PresentShift::Any,
               StillToMove still = StillToMove::Playable) {
  switch (gate) {
    case kGateBetween:
      return m.jump() && !m.branching() && (still == StillToMove::Playable || before.mandatory(m.to_timeline));
    case kGateCapture: return m.capture();
    case kGateCheck:
      return check == CheckChange::Passed ? in_check_passed(before, still) != in_check_passed(after, still)
                                          : royals_safe(before) != royals_safe(after);
    case kGatePresent:
      return present == PresentShift::Any ? before.present() != after.present() : after.present() < before.present();
    default: return true;
  }
}

}  // namespace

// ---- §15.2: the tree form with every gate firing is the ply split ------------------------

TEST(mcts_tree_form_with_every_gate_firing_equals_the_ply_split) {
  const std::vector<Root> all = roots();
  REQUIRE(all.size() >= 30);
  std::uint64_t gated = 0;
  std::size_t compared = 0;
  std::size_t graphs = 0;
  std::uint64_t transpositions = 0;
  std::size_t mid_roots = 0;
  for (std::size_t i = 0; i < all.size(); ++i) {
    const Root& r = all[i];
    if (outcome(r.state, standard_check_ruling()) != Outcome::Ongoing && r.moves_made == 0) continue;
    mid_roots += r.moves_made > 0 ? 1 : 0;
    // Two evaluators per root, at realistic scale (never all zeros: distinct
    // paths would agree trivially) - the random one's seeds vary - each under
    // a different configuration, so that every configuration meets many roots.
    const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 3 + i);
    const std::unique_ptr<Evaluator> material = make_evaluator(BuiltinEvaluator::Material);
    for (int k = 0; k < 2; ++k) {
      const Variation& v = kVariations[(i + static_cast<std::size_t>(k) * 3) % kVariations.size()];
      SearchParams p = fast();
      v.apply(p);
      Evaluator& ev = k == 0 ? *random : *material;
      gated += compare(r, p, ev, 48, r.name + ", " + v.name + (k == 0 ? ", random" : ", material"), &transpositions);
      ++compared;
      graphs += p.graph ? 1u : 0u;
    }
  }
  std::printf("  %zu comparisons, %zu of them under graph search, meeting %llu transpositions\n", compared, graphs,
              static_cast<unsigned long long>(transpositions));
  CHECK(compared >= 60);
  CHECK(mid_roots >= 6);
  CHECK(gated > 0);  // the comparison met mid-turn evaluations
  // ... and, under graph search, merges: the row's graph half is not a tree's.
  CHECK(graphs >= 20);
  CHECK(transpositions > 0);
}

TEST(mcts_tree_form_with_every_gate_firing_equals_the_ply_split_in_every_configuration) {
  // Every variation on the same two multi-board roots, with both evaluators.
  const std::vector<Root> rs = {
      Root{"Two Timelines after 3", random_play(variant("Standard - Two Timelines"), 3, 5), 0},
      Root{"Timeline Battleground after 2", random_play(variant("Misc - Timeline Battleground"), 2, 6), 0}};
  std::uint64_t transpositions = 0;
  std::size_t merged = 0;
  for (const Root& r : rs) {
    for (const Variation& v : kVariations) {
      for (const BuiltinEvaluator kind : {BuiltinEvaluator::Random, BuiltinEvaluator::Material}) {
        const std::unique_ptr<Evaluator> ev = make_evaluator(kind, 9);
        SearchParams p = fast();
        v.apply(p);
        // The graph variations run five times as long: with every node a leaf
        // (the ply split's graph), a turn's two orders first meet after some
        // 300 playouts here - and the row's graph half is to be held where the
        // graph merges, not where it is still a tree.
        std::uint64_t met = 0;
        compare(r, p, *ev, p.graph ? 320 : 64, r.name + ", " + v.name + ", " + evaluator_name(kind), &met);
        transpositions += met;
        merged += met > 0 ? 1u : 0u;
      }
    }
  }
  std::printf("  the row under graph search: %llu transpositions, in %zu of 16 comparisons\n",
              static_cast<unsigned long long>(transpositions), merged);
  // What the comparisons met when written (deterministic): 72 transpositions,
  // in 13 of the 16 graph comparisons; the floors about a third of that.
  CHECK(transpositions > 24);
  CHECK(merged >= 4);
}

TEST(mcts_tree_form_with_every_gate_firing_equals_the_ply_split_under_q4s_guide_literal_reading) {
  // The row under Q4's other reading (moves_after_shift = false: no move once
  // the present has shifted), which the tests above never meet - they run the
  // default rules (R8) only (the gate lane's done-means audit, 2026-10-24).
  // Multi-board roots from random play and one move into an action, every
  // variation, both evaluators.
  RuleOptions rules;
  rules.moves_after_shift = false;
  const auto start = [&](const char* name) { return variant_start(*find_variant(name), rules); };
  std::vector<Root> rs;
  std::uint64_t seed = 40;
  for (const char* name : {"Standard - Two Timelines", "Misc - Timeline Battleground", "Misc - Timeline Strategos",
                           "Very Small - Open"}) {
    const GameState s = random_play(start(name), 3, seed++);
    REQUIRE(!s.rules().moves_after_shift);
    rs.push_back(Root{std::string(name) + " after 3, Guide-literal", s, 0});
    rs.push_back(mid_turn(rs.back().name, s, seed++));
  }
  std::uint64_t gated = 0;
  std::size_t compared = 0;
  std::uint64_t transpositions = 0;
  for (std::size_t i = 0; i < rs.size(); ++i) {
    for (std::size_t k = 0; k < kVariations.size(); ++k) {
      const Variation& v = kVariations[k];
      const BuiltinEvaluator kind = (i + k) % 2 == 0 ? BuiltinEvaluator::Random : BuiltinEvaluator::Material;
      const std::unique_ptr<Evaluator> ev = make_evaluator(kind, 13 + i);
      SearchParams p = fast();
      v.apply(p);
      gated += compare(rs[i], p, *ev, 48, rs[i].name + ", " + v.name + ", " + evaluator_name(kind), &transpositions);
      ++compared;
    }
  }
  std::printf("  %zu comparisons, the graph variations meeting %llu transpositions\n", compared,
              static_cast<unsigned long long>(transpositions));
  CHECK_EQ(compared, rs.size() * kVariations.size());
  CHECK(gated > 0);  // the comparison met mid-turn evaluations
  CHECK(transpositions > 0);  // ... and, in the graph variations, merges
}

// ---- Inheritance: the network at the root, turn starts and gates only -----------------------

namespace {

// Records each request: its state, its count of moves made (and its key).
class Recording final : public Evaluator {
 public:
  explicit Recording(Evaluator& inner) : inner_(inner) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    inner_.evaluate(batch, out);
    for (const EvalRequest& r : batch) {
      requests.emplace_back(*r.state, r.moves_made);
      keys.push_back(r.key);
    }
  }
  std::vector<std::pair<GameState, int>> requests;
  std::vector<std::uint64_t> keys;  // requests[i]'s node key

 private:
  Evaluator& inner_;
};

std::vector<Root> multi_board_roots() {
  return {Root{"Two Timelines", variant("Standard - Two Timelines"), 0},
          Root{"Two Timelines after 4", random_play(variant("Standard - Two Timelines"), 4, 21), 0},
          Root{"Timeline Battleground", variant("Misc - Timeline Battleground"), 0},
          Root{"Timeline Tactician after 3", random_play(variant("Misc - Timeline Tactician"), 3, 22), 0},
          Root{"Timeline Strategos after 2", random_play(variant("Misc - Timeline Strategos"), 2, 23), 0}};
}

}  // namespace

TREE_AND_GRAPH_TEST(mcts_tree_form_without_gates_evaluates_only_the_root_and_turn_starts) {
  std::uint64_t transpositions = 0;
  for (const Root& r : multi_board_roots()) {
    const std::unique_ptr<Evaluator> inner = make_evaluator(BuiltinEvaluator::Random, 4);
    Recording ev(*inner);
    SearchParams p = tree();
    p.graph = graph;
    p.tree_gates = 0;
    p.eval_cache_entries = 0;
    Search s(r.state, p, r.moves_made);
    s.run(ev, Limits{300});
    CHECK_NOTHROW(s.check_invariants());
    // Every request a turn start, or the root.
    for (const auto& [state, made] : ev.requests) {
      CHECK(made == 0 || (made == r.moves_made && state == r.state));
    }
    CHECK_EQ(s.stats().evaluations, static_cast<std::uint64_t>(ev.requests.size()));
    std::uint64_t inherited = 0;
    std::uint64_t evaluated = 0;
    for (NodeId id = 0; id < s.node_count(); ++id) {
      const NodeView n = s.node(id);
      // Under graph search every node a move made is its path's own - no gate
      // fires - from the moment it is made, and no other node is (§8.5).
      CHECK_EQ(n.path_local, graph && id != s.root() && !n.turn_start);
      if (!n.expanded) continue;
      if (id == s.root() || n.turn_start) {
        CHECK(n.evaluated);
        CHECK(!n.inherited);
        ++evaluated;
      } else {
        CHECK(n.inherited);
        CHECK(!n.evaluated);
        CHECK_EQ(n.ends, std::uint32_t{0});  // the visits pass through it
        ++inherited;
      }
    }
    CHECK_EQ(s.stats().inherited_nodes, inherited);
    CHECK(inherited > 0);
    CHECK(evaluated >= s.stats().evaluations);  // a tree may hold one input twice (asked once per batch)
    if (graph) CHECK_EQ(evaluated, s.stats().evaluations);  // ... a graph holds it once: each node asked about once
    // A root child reports the network's evaluation only where the network
    // gave one - never an inherited child's borrowed value.
    std::size_t inherited_children = 0;
    for (const RootChild& c : s.root_children()) {
      if (c.child == kNoNode) continue;
      CHECK_EQ(c.evaluated, s.node(c.child).evaluated);
      inherited_children += s.node(c.child).inherited ? 1u : 0u;
    }
    if (r.moves_made == 0) CHECK(inherited_children > 0);
    for (const std::uint64_t g : s.stats().gate_fired) CHECK_EQ(g, std::uint64_t{0});
    transpositions += s.stats().transposition_hits;
  }
  // With no gate the orderings of a turn meet at its turn start (§8.5: "the
  // orderings of one turn still meet at the complete turn") - in a graph only.
  CHECK_EQ(transpositions > 0, graph);
}

TREE_AND_GRAPH_TEST(mcts_tree_form_evaluates_exactly_where_a_gate_fires) {
  // Each gate alone (the capture gate under its Any reading): a mid-turn node
  // is evaluated exactly when its incoming move fires it, by the predicate
  // recomputed here; else it is inherited. Positions from random play in
  // tactical variants, so that captures and checks occur. Under graph search
  // the assertion is per edge, as in a tree: the gate on an edge's own move
  // decides the kind of node it leads to - a graph node (evaluated; one node
  // however many gated moves reach its state) or a path-local one (inherited)
  // - whatever another path's move did (§8.5).
  std::vector<GameState> positions;
  std::uint64_t seed = 31;
  for (const char* name : {"Standard - Two Timelines", "Misc - Timeline Tactician", "Misc - Timeline Battleground",
                           "Very Small - Open", "Focused - Just Kings", "Misc - Timeline Strategos"}) {
    for (const int actions : {2, 7, 12}) positions.push_back(random_play(variant(name), actions, seed++));
  }
  // White in check (R1): the rook a4 of board 0 takes the king a1 once board 0
  // is passed. Moving the king off the a-file ends it; a move on +1 does not.
  positions.push_back(game({"[r3/4/4/K3:0:1:w]", "[3k/4/4/1K2:+1:1:w]"}, BoardSize{4, 4}));
  REQUIRE(in_check(positions.back()));
  // Where the two readings of "the boards still to move" differ: an optional
  // board, +2 (White's third timeline, inactive with Black's one: T6). The
  // rook a1 of 0 jumps to +2's a1 - between the boards still to move only
  // under the Playable reading - and a king d1 sits on +2 too.
  positions.push_back(game({"[k3/4/4/R2K:0:1:w]", "[k3/4/4/3K:+1:1:w]", "[k3/4/4/3K:+2:1:w]"}, BoardSize{4, 4}));
  REQUIRE_EQ(positions.back().mandatory_count(), 2);
  REQUIRE_EQ(positions.back().playable_timelines().size(), std::size_t{3});
  // ... and the king d1 of the optional +2 attacked along the rank by the
  // rook a1: "in check" with +2 passed (Playable), not with it left (Mandatory)
  // - so the king's step off the rank changes check under one reading only.
  positions.push_back(game({"[k3/4/4/R2K:0:1:w]", "[k3/4/4/3K:+1:1:w]", "[k3/4/4/r2K:+2:1:w]"}, BoardSize{4, 4}));
  REQUIRE(in_check_passed(positions.back(), StillToMove::Playable));
  REQUIRE(!in_check_passed(positions.back(), StillToMove::Mandatory));
  struct Case {
    std::uint32_t gate;
    CheckChange check;
    bool legal_moves_only;
    PresentShift present = PresentShift::Any;
    StillToMove still = StillToMove::Playable;
  };
  // C4's test as the state stands changes only on a move after which no legal
  // completion is left (or on its way back from one): the pseudo-legal seam
  // makes such moves children.
  for (const Case k : {Case{kGateBetween, CheckChange::Passed, true}, Case{kGateCapture, CheckChange::Passed, true},
                       Case{kGateCheck, CheckChange::Passed, true}, Case{kGateCheck, CheckChange::Submission, false},
                       Case{kGatePresent, CheckChange::Passed, true},
                       Case{kGatePresent, CheckChange::Passed, true, PresentShift::Rewind},
                       Case{kGateBetween, CheckChange::Passed, true, PresentShift::Any, StillToMove::Mandatory},
                       Case{kGateCheck, CheckChange::Passed, true, PresentShift::Any, StillToMove::Mandatory}}) {
    const std::uint32_t gate = k.gate;
    std::size_t fired = 0;
    std::size_t quiet = 0;
    std::size_t readings_apart = 0;
    for (const GameState& start : positions) {
      if (outcome(start, standard_check_ruling()) != Outcome::Ongoing) continue;
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Material);
      SearchParams p = tree();
      p.graph = graph;
      p.tree_gates = gate;
      p.tree_capture = CaptureReach::Any;
      p.tree_check = k.check;
      p.tree_present = k.present;
      p.tree_still_to_move = k.still;
      p.legal_moves_only = k.legal_moves_only;
      p.widening = false;
      Search s(start, p);
      s.run(*ev, Limits{250});
      CHECK_NOTHROW(s.check_invariants());
      std::uint64_t gated = 0;
      std::set<NodeId> gated_nodes;  // a graph node is evaluated once, however many edges lead to it
      for (NodeId id = 0; id < s.node_count(); ++id) {
        const NodeView n = s.node(id);
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = s.edge(id, i);
          if (e.submit || e.child == kNoNode || e.child == s.root()) continue;
          const NodeView c = s.node(e.child);
          if (c.turn_start) continue;
          const bool want = predicate(gate, *n.state, *e.move, *c.state, k.check, k.present, k.still);
          // Path-local from the moment it is made, exactly where no gate fires.
          CHECK_EQ(c.path_local, graph && !want);
          if (!c.expanded) continue;
          CHECK_EQ(c.evaluated, want);
          CHECK_EQ(c.inherited, !want);
          (want ? fired : quiet) += 1;
          gated += want && gated_nodes.insert(e.child).second ? 1u : 0u;
          // The two readings apart: the move fires under one and not the other.
          const StillToMove other = k.still == StillToMove::Mandatory ? StillToMove::Playable : StillToMove::Mandatory;
          if ((gate == kGateBetween || gate == kGateCheck) &&
              want != predicate(gate, *n.state, *e.move, *c.state, k.check, k.present, other)) {
            ++readings_apart;
          }
        }
      }
      const std::size_t bit = static_cast<std::size_t>(std::countr_zero(gate));
      CHECK_EQ(s.stats().gate_fired[bit], gated);
    }
    CHECK(fired > 0);
    CHECK(quiet > 0);
    // Under the Mandatory reading the search met moves the readings judge
    // apart (each case's positions above).
    if (k.still == StillToMove::Mandatory) CHECK(readings_apart > 0);
  }
}

TREE_AND_GRAPH_TEST(mcts_tree_form_natural_gates_are_the_union_of_the_four) {
  // §8.2's four together (the capture gate's Reach reading, the default): an
  // evaluated mid-turn node's move fires one of them; an inherited one's fires
  // neither the between, the check nor the present gate, and the capture
  // gate's reading is held on constructed positions below. Under graph search
  // every edge into an evaluated node is held to it - each of a merged node's
  // parents - and a path-local node is never evaluated.
  std::size_t evaluated = 0;
  std::size_t local = 0;
  for (const Root& r : multi_board_roots()) {
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 8);
    SearchParams p = tree();
    p.graph = graph;
    Search s(r.state, p, r.moves_made);
    s.run(*ev, Limits{300});
    CHECK_NOTHROW(s.check_invariants());
    for (NodeId id = 0; id < s.node_count(); ++id) {
      const NodeView n = s.node(id);
      for (std::size_t i = 0; i < n.edge_count; ++i) {
        const EdgeView e = s.edge(id, i);
        if (e.submit || e.child == kNoNode) continue;
        const NodeView c = s.node(e.child);
        if (!c.expanded || c.turn_start) continue;
        const bool hard = predicate(kGateBetween, *n.state, *e.move, *c.state) ||
                          predicate(kGateCheck, *n.state, *e.move, *c.state) ||
                          predicate(kGatePresent, *n.state, *e.move, *c.state);
        if (c.evaluated) CHECK(hard || e.move->capture());
        if (hard) CHECK(c.evaluated);
        CHECK_EQ(c.path_local, graph && !c.evaluated);
        evaluated += c.evaluated ? 1u : 0u;
        local += c.path_local ? 1u : 0u;
      }
    }
  }
  CHECK(evaluated > 0);
  CHECK_EQ(local > 0, graph);
}

namespace {

// Two mandatory boards, both White's at T1. On 0, the white rook c1 takes the
// black rook c3; the board +1 is still to move. A rook moves along one axis:
// along the timeline axis the black rook reaches c3 of +1's passed copy - in
// `reaching` it is empty; in `blocked` a black knight stands there.
const char* const kZero = "[k3/2r1/4/K1R1:0:1:w]";
GameState reaching() { return game({kZero, "[3k/4/4/K3:+1:1:w]"}, BoardSize{4, 4}); }
GameState blocked() { return game({kZero, "[3k/2n1/4/K3:+1:1:w]"}, BoardSize{4, 4}); }

// The child of `parent` through the edge whose move satisfies `pick`.
NodeId child_where(const Search& s, NodeId parent, bool (*pick)(const Move&)) {
  const NodeView n = s.node(parent);
  for (std::size_t i = 0; i < n.edge_count; ++i) {
    const EdgeView e = s.edge(parent, i);
    if (!e.submit && pick(*e.move)) return e.child;
  }
  return kNoNode;
}

bool the_capture(const Move& m) { return m.capture() && m.from_timeline == 0; }
bool a_king_move_on_plus_one(const Move& m) {
  return m.from_timeline == 1 && m.physical() && m.piece.kind() == PieceKind::King;
}

}  // namespace

TREE_AND_GRAPH_TEST(mcts_tree_form_capture_gate_reading) {
  for (const CaptureReach reading : {CaptureReach::Reach, CaptureReach::Any}) {
    for (const bool can_reach : {true, false}) {
      const GameState start = can_reach ? reaching() : blocked();
      REQUIRE_EQ(start.mandatory_count(), 2);
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
      SearchParams p = tree();
      p.graph = graph;
      p.tree_gates = kGateCapture;
      p.tree_capture = reading;
      p.widening = false;
      Search s(start, p);
      s.run(*ev, Limits{600});
      CHECK_NOTHROW(s.check_invariants());
      // Taken first: the board +1 is still to move.
      const NodeId taken = child_where(s, s.root(), the_capture);
      REQUIRE(taken != kNoNode);
      REQUIRE(s.node(taken).expanded);
      CHECK_EQ(s.node(taken).evaluated, reading == CaptureReach::Any || can_reach);
      // Taken after +1 is played: no board is left to reach.
      const NodeId king = child_where(s, s.root(), a_king_move_on_plus_one);
      REQUIRE(king != kNoNode);
      CHECK(s.node(king).inherited);  // a quiet move fires no capture gate
      const NodeId later = child_where(s, king, the_capture);
      REQUIRE(later != kNoNode);
      REQUIRE(s.node(later).expanded);
      CHECK_EQ(s.node(later).evaluated, reading == CaptureReach::Any);
      // Under graph search the reading decides the kind of node too: the
      // capture that fires no gate leads to a path-local node.
      CHECK_EQ(s.node(taken).path_local, graph && !s.node(taken).evaluated);
      CHECK_EQ(s.node(king).path_local, graph);
      CHECK_EQ(s.node(later).path_local, graph && reading != CaptureReach::Any);
    }
  }
}

TREE_AND_GRAPH_TEST(mcts_tree_form_inherits_its_parents_logits) {
  // Under graph search as in a tree: an inherited node is path-local, so the
  // parent read here is its only one - were inherited nodes merged, a second
  // parent's logits would disagree with what the first path to reach it left.
  //
  // No gate: every mid-turn node below the root inherits. Its move edges take
  // the parent's logits by coordinates - a branching jump onto a board played
  // since (J6) the logit of the non-branching jump it was - an unmatched move
  // the parent's lowest; its submit edge the parent's submit logit; and its
  // priors are their softmax (the submit share where no submit logit is
  // given). The rook a2 of board 0 can jump to a2 of board +1: after the king
  // moves on +1, the same jump branches. In `promoting` the pawn b3 of board 0
  // promotes on b4 to a queen, rook, bishop or knight (S8's set, as a
  // Promotions header gives it) - four moves told apart by their promotion
  // alone.
  const GameState start = game({"[k3/4/R3/K3:0:1:w]", "[3k/4/4/1K2:+1:1:w]"}, BoardSize{4, 4});
  REQUIRE_EQ(start.mandatory_count(), 2);
  RuleOptions four;
  four.promotions = PromotionSet::of({PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop, PieceKind::Knight});
  const GameState promoting = game({"[3k/1P2/4/K3:0:1:w]", "[3k/4/4/1K2:+1:1:w]"}, BoardSize{4, 4}, four);
  REQUIRE_EQ(promoting.mandatory_count(), 2);
  std::size_t twins = 0;
  std::size_t inherited = 0;
  std::size_t promotions = 0;
  std::size_t unmatched = 0;
  for (const GameState& s0 : {start, promoting, random_play(variant("Standard - Two Timelines"), 3, 41)}) {
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 5);
    SearchParams p = tree();
    p.graph = graph;
    p.tree_gates = 0;
    p.widening = false;
    p.asset_map = false;  // the inherited logits' softmax alone (steering: test_ladder.cpp)
    Search s(s0, p);
    s.run(*ev, Limits{500});
    CHECK_NOTHROW(s.check_invariants());
    using Coord = std::tuple<int, int, int, int, int, int, int>;
    const auto coord = [](const Move& m) {
      return Coord{m.from_timeline, m.from_half_turn, m.from_square, m.to_timeline,
                   m.to_half_turn, m.to_square, m.promotes() ? static_cast<int>(m.promotion) + 1 : 0};
    };
    for (NodeId id = 0; id < s.node_count(); ++id) {
      const NodeView n = s.node(id);
      for (std::size_t i = 0; i < n.edge_count; ++i) {
        const EdgeView pe = s.edge(id, i);
        if (pe.child == kNoNode || !s.node(pe.child).inherited) continue;
        ++inherited;
        const NodeId cid = pe.child;
        const NodeView c = s.node(cid);
        CHECK_EQ(c.path_local, graph);
        std::map<Coord, std::pair<double, bool>> parent;  // logit, branching
        double lowest = 0;
        bool first = true;
        double submit_logit = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t j = 0; j < n.edge_count; ++j) {
          const EdgeView e = s.edge(id, j);
          if (e.submit) continue;
          parent.emplace(coord(*e.move), std::make_pair(e.logit, e.move->branching()));
          lowest = first ? e.logit : std::min(lowest, e.logit);
          first = false;
        }
        // The submit logit the parent carries: its submit edge's, or (none)
        // read through the child's own submit edge below.
        for (std::size_t j = 0; j < n.edge_count; ++j) {
          if (s.edge(id, j).submit) submit_logit = s.edge(id, j).logit;
        }
        std::vector<double> logits;
        std::vector<double> priors;
        double submit_prior = -1;
        for (std::size_t j = 0; j < c.edge_count; ++j) {
          const EdgeView e = s.edge(cid, j);
          if (e.submit) {
            if (n.submit_possible) CHECK(e.logit == submit_logit || (std::isnan(e.logit) && std::isnan(submit_logit)));
            if (!e.pruned) submit_prior = e.prior;
            continue;
          }
          const auto it = parent.find(coord(*e.move));
          if (it != parent.end()) {
            CHECK_EQ(e.logit, it->second.first);
            if (e.move->branching() && !it->second.second) ++twins;
            if (e.move->promotes() && e.move->promotion != PieceKind::Queen) ++promotions;
          } else {
            CHECK_EQ(e.logit, lowest);
            ++unmatched;
          }
          if (!e.pruned) {
            logits.push_back(e.logit);
            priors.push_back(e.prior);
          }
        }
        // The softmax (temperature 1, below the root; the moves' share of 1
        // less a submit share).
        double z = 0;
        double top = -1e300;
        for (const double l : logits) top = std::max(top, l);
        for (const double l : logits) z += std::exp(l - top);
        double moves_share = 0;
        for (const double q : priors) moves_share += q;
        for (std::size_t j = 0; j < logits.size(); ++j) {
          CHECK(std::abs(priors[j] - moves_share * std::exp(logits[j] - top) / z) < 1e-12);
        }
        if (submit_prior >= 0) CHECK(std::abs(moves_share + submit_prior - 1.0) < 1e-12);
      }
    }
  }
  CHECK(inherited > 20);
  CHECK(twins > 0);       // the J6 twin met, and matched
  CHECK(promotions > 3);  // each promotion its own logit, under-promotions too
  // A move the parent did not have: legal only after another move of the
  // action (proved illegal as the parent's next move - no legal completion
  // begins with it - and legal once the other board is played).
  CHECK(unmatched > 0);
}

TREE_AND_GRAPH_TEST(mcts_tree_form_evaluates_a_carried_root_it_inherited) {
  // Match play carries the tree (C12). Advanced onto a child the tree form
  // inherited - a mid-turn node the network never saw - the search's root is
  // evaluated by the first simulation after the re-root, as a leaf, and never
  // again: its edges, their children and statistics kept, their logits and
  // priors and its value the network's - a fresh search's root's at that
  // state - and every edge already visited still instantiated. Under graph
  // search the child was path-local, and as the root it is a graph node
  // (entered in the table, which check_invariants reads; the root is always
  // the network's, and only evaluated nodes are graph nodes - §8.5).
  std::size_t carried = 0;
  for (const Root& r : multi_board_roots()) {
    for (const std::uint32_t gates : {std::uint32_t{0}, kGatesInteraction}) {
      for (const bool one_first : {true, false}) {
        const std::unique_ptr<Evaluator> inner = make_evaluator(BuiltinEvaluator::Random, 6);
        Recording ev(*inner);
        SearchParams p = tree();
        p.graph = graph;
        p.tree_gates = gates;
        p.eval_cache_entries = 0;
        Search s(r.state, p, r.moves_made);
        s.run(ev, Limits{300});
        const std::vector<RootChild> kids = s.root_children();
        std::size_t pick = kids.size();
        for (std::size_t i = 0; i < kids.size() && pick == kids.size(); ++i) {
          if (!kids[i].submit && kids[i].child != kNoNode && s.node(kids[i].child).inherited && kids[i].visits > 3) {
            pick = i;
          }
        }
        if (pick == kids.size()) continue;
        ++carried;
        // The carried node's edges, by move (submit: no move).
        struct Stat {
          std::uint32_t visits = 0;
          double weight = 0;
          double value = 0;
          bool child = false;
        };
        using Key = std::pair<bool, Coord>;
        std::map<Key, Stat> before;
        {
          const NodeId c = kids[pick].child;
          for (std::size_t i = 0; i < s.node(c).edge_count; ++i) {
            const EdgeView e = s.edge(c, i);
            before[Key{e.submit, e.submit ? Coord{} : coord_of(*e.move)}] =
                Stat{e.visits, e.weight, e.value, e.child != kNoNode};
          }
        }
        CHECK_EQ(s.node(kids[pick].child).path_local, graph);
        s.advance(pick);
        const NodeView carried_root = s.node(s.root());
        REQUIRE(carried_root.inherited);
        CHECK(!carried_root.evaluated);
        CHECK(!carried_root.path_local);  // the root: a graph node from the re-root on
        CHECK_NOTHROW(s.check_invariants());
        const GameState state = *carried_root.state;
        const int made = carried_root.moves_made;
        const std::uint32_t visits = carried_root.visits;
        ev.requests.clear();
        s.run(ev, Limits{one_first ? 1u : 64u});
        CHECK_NOTHROW(s.check_invariants());
        std::size_t asked = 0;
        for (const auto& [st, m] : ev.requests) asked += st == state && m == made ? 1u : 0u;
        CHECK_EQ(asked, std::size_t{1});
        const NodeView now = s.node(s.root());
        CHECK(now.evaluated);
        CHECK(!now.inherited);
        CHECK(!now.path_local);
        if (!one_first) {
          CHECK_EQ(now.visits, visits + 64);
          continue;
        }
        REQUIRE_EQ(ev.requests.size(), std::size_t{1});  // the first simulation: the root, and nothing else
        CHECK_EQ(now.visits, visits + 1);
        CHECK_EQ(now.ends, std::uint32_t{1});
        // Its value and its edges' logits and priors: a fresh root's.
        Search fresh(state, p, made);
        fresh.run(*inner, Limits{1});
        const NodeView f = fresh.node(fresh.root());
        CHECK_EQ(now.eval_mean, f.eval_mean);
        CHECK_EQ(now.eval_draw, f.eval_draw);
        CHECK_EQ(now.eval_variance, f.eval_variance);
        REQUIRE_EQ(now.edge_count, f.edge_count);
        std::map<Key, std::pair<double, double>> fresh_edges;  // logit, prior
        for (std::size_t i = 0; i < f.edge_count; ++i) {
          const EdgeView e = fresh.edge(fresh.root(), i);
          fresh_edges[Key{e.submit, e.submit ? Coord{} : coord_of(*e.move)}] = {e.logit, e.prior};
        }
        bool pruned = false;
        for (std::size_t i = 0; i < now.edge_count; ++i) pruned = pruned || s.edge(s.root(), i).pruned;
        double last_prior = 2;
        for (std::size_t i = 0; i < now.edge_count; ++i) {
          const EdgeView e = s.edge(s.root(), i);
          const Key key{e.submit, e.submit ? Coord{} : coord_of(*e.move)};
          REQUIRE(fresh_edges.count(key) == 1 && before.count(key) == 1);
          const auto [logit, prior] = fresh_edges[key];
          CHECK(e.logit == logit || (std::isnan(e.logit) && std::isnan(logit)));
          if (!pruned) CHECK(std::abs(e.prior - prior) < 1e-12);
          CHECK(e.prior <= last_prior);  // ordered by the new priors
          last_prior = e.prior;
          // The statistics carried.
          const Stat& b = before[key];
          CHECK_EQ(e.visits, b.visits);
          CHECK_EQ(e.weight, b.weight);
          CHECK_EQ(e.value, b.value);
          CHECK_EQ(e.child != kNoNode, b.child);
          if (e.visits > 0 || e.child != kNoNode) CHECK(i < now.instantiated);
        }
        // The search goes on through it; the network is not asked about it again.
        ev.requests.clear();
        s.run(ev, Limits{200});
        CHECK_NOTHROW(s.check_invariants());
        for (const auto& [st, m] : ev.requests) CHECK(!(st == state && m == made));
        CHECK(s.node(s.root()).evaluated);
      }
    }
  }
  CHECK(carried >= 10);
}

TREE_AND_GRAPH_TEST(mcts_tree_form_inherited_value_is_its_parents) {
  // tree_inherit_value: an inherited node's eval_* are the node's above it -
  // the one its priors came from (its own inherited ones in a run of them) -
  // and 0 without; the searches differ (the FPU base of its first children)
  // and both keep the invariants. (Under graph search "the node above it" is
  // still one node: an inherited node is path-local.)
  std::size_t below_turn_starts = 0;
  std::size_t checked = 0;
  bool differ = false;
  for (const Root& r : multi_board_roots()) {
    std::uint64_t digests[2] = {0, 0};
    for (const bool value : {true, false}) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 12);
      SearchParams p = tree();
      p.graph = graph;
      p.tree_gates = 0;
      p.tree_inherit_value = value;
      Search s(r.state, p, r.moves_made);
      s.run(*ev, Limits{300});
      CHECK_NOTHROW(s.check_invariants());
      for (NodeId id = 0; id < s.node_count(); ++id) {
        const NodeView n = s.node(id);
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = s.edge(id, i);
          if (e.child == kNoNode || !s.node(e.child).inherited) continue;
          const NodeView c = s.node(e.child);
          ++checked;
          if (value) {
            CHECK_EQ(c.eval_mean, n.eval_mean);
            CHECK_EQ(c.eval_draw, n.eval_draw);
            CHECK_EQ(c.eval_variance, n.eval_variance);
            if (id != s.root() && n.evaluated) ++below_turn_starts;  // a turn start's value, not the root's
          } else {
            CHECK_EQ(c.eval_mean, 0.0);
            CHECK_EQ(c.eval_draw, 0.0);
            CHECK_EQ(c.eval_variance, 0.0);
          }
        }
      }
      digests[value ? 1 : 0] = s.digest();
    }
    differ = differ || digests[0] != digests[1];
  }
  CHECK(checked > 100);
  CHECK(below_turn_starts > 0);
  CHECK(differ);
}

TREE_AND_GRAPH_TEST(mcts_tree_form_inherited_nodes_take_a_leafs_legality) {
  // An inherited node's children are a leaf's: its state's moves under the
  // same classification (E5a: legal_moves_only, legality) - a move left
  // unknown by the budget flagged unknown, every move with no budget at all;
  // under the pseudo-legal seam the illegal moves kept too, and the dead ends
  // they lead to found at nodes the network never sees.
  struct Setting {
    const char* name;
    void (*apply)(SearchParams&);
  };
  for (const Setting& v : {Setting{"defaults", [](SearchParams&) {}},
                           Setting{"no budget", [](SearchParams& p) { p.legality = LegalityOptions::budgeted(0, 0); }},
                           Setting{"pseudo-legal", [](SearchParams& p) { p.legal_moves_only = false; }}}) {
    const bool no_budget = std::string(v.name) == "no budget";
    const bool pseudo = std::string(v.name) == "pseudo-legal";
    std::size_t nodes = 0;
    std::size_t unknown = 0;
    std::size_t illegal_kept = 0;
    std::size_t dead_inherited = 0;
    for (const Root& r : multi_board_roots()) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Material);
      SearchParams p = tree();
      p.graph = graph;
      p.tree_gates = 0;
      v.apply(p);
      Search s(r.state, p, r.moves_made);
      s.run(*ev, Limits{300});
      CHECK_NOTHROW(s.check_invariants());
      for (NodeId id = 0; id < s.node_count(); ++id) {
        const NodeView n = s.node(id);
        if (n.dead_end && !n.expanded && !n.turn_start) ++dead_inherited;  // no gate: found by inheriting
        if (!n.inherited) continue;
        ++nodes;
        ExpandOptions o;
        o.window_k = 0;
        o.square_features = SquareFeatures::None;
        o.legal_moves = p.legal_moves_only;
        o.legality = p.legality;
        Expansion x;
        expand(*n.state, o, x, n.moves_made);
        std::map<Coord, MoveLegality> want;
        for (std::size_t i = 0; i < x.moves().size(); ++i) {
          want[coord_of(x.moves()[i])] = i < x.move_legality().size() ? x.move_legality()[i] : MoveLegality::Legal;
        }
        std::size_t moves = 0;
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = s.edge(id, i);
          if (e.submit) continue;
          ++moves;
          const auto it = want.find(coord_of(*e.move));
          REQUIRE(it != want.end());
          if (!e.verified) CHECK_EQ(e.unknown, it->second == MoveLegality::Unknown);
          unknown += e.unknown ? 1u : 0u;
        }
        CHECK_EQ(moves, want.size());
        if (pseudo) {
          // The legal moves alone are fewer where an illegal one is kept.
          ExpandOptions legal = o;
          legal.legal_moves = true;
          legal.legality = LegalityOptions::exact();
          Expansion y;
          expand(*n.state, legal, y, n.moves_made);
          illegal_kept += x.moves().size() - y.moves().size();
        }
      }
    }
    CHECK(nodes > 100);
    if (no_budget) CHECK(unknown > 100);
    if (pseudo) {
      CHECK(illegal_kept > 0);
      CHECK(dead_inherited > 0);
    }
  }
}

// ---- §15.4 under the tree form ---------------------------------------------------------------

TREE_AND_GRAPH_TEST(mcts_tree_form_keeps_the_turn_structure_and_prunes_dead_ends_unscored) {
  // The pseudo-legal seam makes dead ends; with no gate they are met at nodes
  // the network never sees - pruned, never scored, never evaluated - and the
  // invariants hold: a submit edge into a turn start, a move edge into a
  // node during an action, no game end during one, an inherited node's visits
  // all passing through it. (Under graph search with the solver on, too.)
  std::uint64_t dead = 0;
  for (const Root& r : multi_board_roots()) {
    for (const std::uint32_t gates : {std::uint32_t{0}, kGatesInteraction}) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Material);
      SearchParams p = tree();
      p.graph = graph;
      p.tree_gates = gates;
      p.legal_moves_only = false;
      p.proof_channel = true;
      p.hunt = p.settling = false;  // the ladder's proofs (E10) are test_ladder.cpp's
      Search s(r.state, p, r.moves_made);
      s.run(*ev, Limits{400});
      CHECK_NOTHROW(s.check_invariants());
      dead += s.stats().dead_ends;
      for (NodeId id = 0; id < s.node_count(); ++id) {
        const NodeView n = s.node(id);
        if (n.dead_end) {
          CHECK(!n.turn_start);
          if (gates == 0) CHECK(!n.evaluated);  // no mid-turn node is
          if (!n.expanded) CHECK_EQ(n.visits, std::uint32_t{0});  // found on expansion: nothing backed up
        }
        if (!n.turn_start) CHECK(n.terminal == Terminal::None);
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = s.edge(id, i);
          if (e.child != kNoNode && s.node(e.child).dead_end) CHECK(e.pruned);  // pruned, never selected again
        }
      }
    }
  }
  CHECK(dead > 0);
}

// ---- §8.5: the tree form under graph search (C11 with D2; the 2026-11-15 review) --------------

namespace {

// Whether the move from `before` to `after` fires one of the gates in `gates`
// (the capture gate under its Any reading, the other readings the defaults).
bool fires(std::uint32_t gates, const GameState& before, const Move& m, const GameState& after) {
  for (const std::uint32_t g : {kGatePresent, kGateBetween, kGateCapture, kGateCheck}) {
    if ((gates & g) != 0 && predicate(g, before, m, after)) return true;
  }
  return false;
}

// Each node's parents: the edges that lead to it.
std::vector<std::uint32_t> parents_of(const Search& s) {
  std::vector<std::uint32_t> parents(s.node_count(), 0);
  for (NodeId id = 0; id < s.node_count(); ++id) {
    for (std::size_t i = 0; i < s.node(id).edge_count; ++i) {
      const NodeId c = s.edge(id, i).child;
      if (c != kNoNode) ++parents[c];
    }
  }
  return parents;
}

// The nodes by what a graph node's identity is - the state and the moves made
// in its action (A6): grouped by key and count, then compared exactly.
std::vector<std::vector<NodeId>> by_identity(const Search& s) {
  std::map<std::pair<std::uint64_t, int>, std::vector<NodeId>> buckets;
  for (NodeId id = 0; id < s.node_count(); ++id) buckets[{s.node(id).key, s.node(id).moves_made}].push_back(id);
  std::vector<std::vector<NodeId>> groups;
  for (const auto& [key, ids] : buckets) {
    const std::size_t first = groups.size();
    for (const NodeId id : ids) {
      bool placed = false;
      for (std::size_t g = first; g < groups.size() && !placed; ++g) {
        if (*s.node(groups[g].front()).state == *s.node(id).state) {
          groups[g].push_back(id);
          placed = true;
        }
      }
      if (!placed) groups.push_back({id});
    }
  }
  return groups;
}

// An edge's name within its node: submit, or its move's coordinates.
using EdgeName = std::pair<bool, Coord>;
EdgeName name_of(const EdgeView& e) { return {e.submit, e.submit ? Coord{} : coord_of(*e.move)}; }

// A node's edges that lead to a child, by name.
std::map<EdgeName, NodeId> children_of(const Search& s, NodeId id) {
  std::map<EdgeName, NodeId> out;
  for (std::size_t i = 0; i < s.node(id).edge_count; ++i) {
    const EdgeView e = s.edge(id, i);
    if (e.child != kNoNode) out[name_of(e)] = e.child;
  }
  return out;
}

// Every leaf worth `white` to White - so +white from the acting player's view
// at White's nodes and -white at Black's - with uniform logits.
class ByColour final : public Evaluator {
 public:
  explicit ByColour(double white) : white_(white) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      out[i].logits.assign(batch[i].expansion->moves().size(), 0.0f);
      const double v = batch[i].expansion->acting_player() == Colour::White ? white_ : -white_;
      out[i].wdl = wdl_of(v);
      out[i].value = two_hot(v);
    }
  }

 private:
  double white_;
};

}  // namespace

TEST(mcts_tree_form_under_graph_search_merges_at_evaluated_nodes_only) {
  // §8.5: "under MCGS only evaluated nodes are graph nodes - turn starts,
  // complete turns and interaction points, each keyed by its state ... An
  // inherited inner node belongs to the path that made it and is never
  // merged." Each gate alone, the four together and none (the capture gate's
  // Any reading), on the multi-board roots, read through the views:
  //  (1) the gate on a move edge's OWN move decides the kind of node the edge
  //      leads to, from the moment the node is made - a gate fires: a graph
  //      node (evaluated once expanded); none: a path-local one;
  //  (2) a path-local node has one parent, is never evaluated, is inherited
  //      once expanded, and no visit ends at it;
  //  (3) a node with two parents or more is a graph node, and evaluated;
  //  (4) no two graph nodes hold one state with one count, while path-local
  //      nodes may (the duplicate inner nodes of §8.5) - and where two do, the
  //      same edge of each leads to ONE graph node (the orderings of a turn
  //      meet at the next evaluated node, at the latest the turn start) or to
  //      TWO path-local ones;
  //  (5) no cycle: every edge leads up a strict order - a move edge to a state
  //      with more boards, still inside the action; a submit edge from a node
  //      inside an action to a turn start with the same boards;
  //  (6) visits are conserved: a node's are the visits that ended at it plus
  //      its edges'; a path-local node's are its one edge's; a graph node's
  //      are its incoming edges' less the residual corrections that stopped at
  //      those edges - and those are all the corrections there were.
  // check_invariants holds the same of the search's own table: its entries
  // are the nodes that are not path-local, each identity once.
  std::size_t duplicate_pairs = 0;  // pairs of path-local nodes of one state and count
  std::size_t met_again = 0;        // ... one edge of each leading to one graph node
  std::size_t apart_again = 0;      // ... or to two path-local nodes
  std::size_t both_kinds = 0;       // states held by a graph node and by a path-local one at once
  for (const std::uint32_t gates :
       {std::uint32_t{0}, kGateBetween, kGateCapture, kGateCheck, kGatePresent, kGatesInteraction}) {
    std::uint64_t hits = 0;
    std::uint64_t inherited = 0;
    std::size_t local = 0;
    std::size_t merged = 0;
    for (const Root& r : multi_board_roots()) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 17);
      SearchParams p = tree();
      p.graph = true;
      p.tree_gates = gates;
      p.tree_capture = CaptureReach::Any;
      Search s(r.state, p, r.moves_made);
      s.run(*ev, Limits{300});
      CHECK_NOTHROW(s.check_invariants());
      CHECK_EQ(s.stats().key_collisions, std::uint64_t{0});
      hits += s.stats().transposition_hits;
      inherited += s.stats().inherited_nodes;
      const std::vector<std::uint32_t> parents = parents_of(s);
      std::vector<std::uint64_t> arriving(s.node_count(), 0);  // each node's incoming edges' visits
      for (NodeId id = 0; id < s.node_count(); ++id) {
        const NodeView n = s.node(id);
        if (n.path_local) {  // (2)
          ++local;
          CHECK_EQ(parents[id], std::uint32_t{1});
          CHECK(!n.evaluated && !n.turn_start && id != s.root());
          if (n.expanded) CHECK(n.inherited);
          CHECK_EQ(n.ends, std::uint32_t{0});
        } else {
          CHECK(!n.inherited);
        }
        if (parents[id] >= 2) {  // (3)
          ++merged;
          CHECK(!n.path_local);
          if (n.expanded) CHECK(n.evaluated);
        }
        const std::size_t boards = n.state->multiverse().board_count();
        std::uint64_t through = 0;
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = s.edge(id, i);
          through += e.visits;
          if (e.child == kNoNode) continue;
          const NodeView c = s.node(e.child);
          arriving[e.child] += e.visits;
          const std::size_t child_boards = c.state->multiverse().board_count();
          if (e.submit) {
            CHECK(!n.turn_start && c.turn_start && child_boards == boards);  // (5)
            CHECK(!c.path_local);  // a turn start: always a graph node
            continue;
          }
          CHECK(child_boards > boards && !c.turn_start);  // (5)
          const bool want = fires(gates, *n.state, *e.move, *c.state);  // (1)
          CHECK_EQ(c.path_local, !want);
          if (c.expanded) CHECK_EQ(c.evaluated, want);
          if (c.path_local) CHECK_EQ(c.visits, e.visits);  // (6)
        }
        CHECK_EQ(std::uint64_t{n.visits}, n.ends + through);  // (6)
      }
      CHECK_EQ(std::uint64_t{s.node(s.root()).visits}, s.stats().playouts);
      std::uint64_t stopped = 0;  // (6): the visits that stopped at an edge into a graph node
      for (NodeId id = 0; id < s.node_count(); ++id) {
        if (id == s.root()) continue;
        CHECK(arriving[id] >= s.node(id).visits);
        stopped += arriving[id] - s.node(id).visits;
      }
      CHECK_EQ(stopped, s.stats().residual_backups);
      for (const std::vector<NodeId>& group : by_identity(s)) {  // (4)
        std::vector<NodeId> locals;
        std::size_t graph_nodes = 0;
        for (const NodeId id : group) {
          if (s.node(id).path_local) {
            locals.push_back(id);
          } else {
            ++graph_nodes;
          }
        }
        CHECK(graph_nodes <= 1);
        if (graph_nodes == 1 && !locals.empty()) ++both_kinds;
        for (std::size_t a = 0; a < locals.size(); ++a) {
          for (std::size_t b = a + 1; b < locals.size(); ++b) {
            ++duplicate_pairs;
            const std::map<EdgeName, NodeId> x = children_of(s, locals[a]);
            const std::map<EdgeName, NodeId> y = children_of(s, locals[b]);
            for (const auto& [name, child] : x) {
              const auto other = y.find(name);
              if (other == y.end()) continue;
              // One state and one move: one verdict of the gate, for both.
              REQUIRE_EQ(s.node(child).path_local, s.node(other->second).path_local);
              if (s.node(child).path_local) {
                CHECK_NE(child, other->second);
                ++apart_again;
              } else {
                CHECK_EQ(child, other->second);
                ++met_again;
              }
            }
          }
        }
      }
    }
    // Every setting met merges, and nodes that are never merged.
    CHECK(hits > 0);
    CHECK(merged > 0);
    CHECK(inherited > 0);
    CHECK(local > 0);
  }
  std::printf("  tree form under graph search: %zu pairs of path-local nodes of one state, an edge of each leading "
              "to one graph node %zu times and to two path-local nodes %zu times; %zu states held by both kinds of "
              "node\n",
              duplicate_pairs, met_again, apart_again, both_kinds);
  // What the searches met when written (deterministic): 744 pairs, an edge of
  // each leading to one graph node 617 times and to two path-local nodes 164
  // times, 36 states held by both kinds of node; the floors about a third.
  CHECK(duplicate_pairs > 240);
  CHECK(met_again > 200);
  CHECK(apart_again > 50);
  CHECK(both_kinds > 10);
}

TEST(mcts_tree_form_under_graph_search_the_gate_on_an_edges_own_move_decides) {
  // The case §8.5 does not name: one state reached by a move that fires a gate
  // and by a move that fires none. The gate on the edge's own move decides,
  // and nothing else - so the state is held by an evaluated graph node AND by
  // a path-local inherited one, never merged ("whether it was evaluated at all
  // depends on the move that reached it"); each edge's child is a function of
  // the edge, not of which path the search tried first. On `reaching()`, the
  // capture gate alone (its Any reading): the rook takes on board 0 (fires),
  // the king steps on +1 (quiet). King then capture: a graph node. Capture
  // then king: a path-local one, of the same state and count. Both submit to
  // one turn start.
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
  SearchParams p = tree();
  p.graph = true;
  p.tree_gates = kGateCapture;
  p.tree_capture = CaptureReach::Any;
  p.widening = false;
  Search s(reaching(), p);
  s.run(*ev, Limits{600});
  CHECK_NOTHROW(s.check_invariants());
  NodeId king = kNoNode;
  Coord step{};
  for (std::size_t i = 0; i < s.node(s.root()).edge_count && king == kNoNode; ++i) {
    const EdgeView e = s.edge(s.root(), i);
    if (e.submit || !a_king_move_on_plus_one(*e.move)) continue;
    king = e.child;
    step = coord_of(*e.move);
  }
  const NodeId taken = child_where(s, s.root(), the_capture);
  REQUIRE(king != kNoNode && taken != kNoNode);
  REQUIRE(s.node(king).expanded && s.node(taken).expanded);
  CHECK(s.node(king).path_local && s.node(king).inherited);
  CHECK(!s.node(taken).path_local && s.node(taken).evaluated);
  const NodeId by_capture = child_where(s, king, the_capture);  // the king's step, then the capture
  NodeId by_step = kNoNode;                                     // the capture, then the same step
  for (std::size_t i = 0; i < s.node(taken).edge_count; ++i) {
    const EdgeView e = s.edge(taken, i);
    if (!e.submit && coord_of(*e.move) == step) by_step = e.child;
  }
  REQUIRE(by_capture != kNoNode && by_step != kNoNode);
  const NodeView gated = s.node(by_capture);
  const NodeView quiet = s.node(by_step);
  REQUIRE(*gated.state == *quiet.state);
  CHECK_EQ(gated.moves_made, 2);
  CHECK_EQ(quiet.moves_made, 2);
  CHECK_NE(by_capture, by_step);
  CHECK(!gated.path_local);
  CHECK(quiet.path_local);
  REQUIRE(gated.expanded && quiet.expanded);
  CHECK(gated.evaluated && !gated.inherited);
  CHECK(quiet.inherited && !quiet.evaluated);
  // The complete turn, submitted: one node, whichever of the two the path took.
  const std::map<EdgeName, NodeId> x = children_of(s, by_capture);
  const std::map<EdgeName, NodeId> y = children_of(s, by_step);
  const EdgeName submit{true, Coord{}};
  REQUIRE(x.count(submit) == 1 && y.count(submit) == 1);
  CHECK_EQ(x.at(submit), y.at(submit));
  CHECK(s.node(x.at(submit)).turn_start && !s.node(x.at(submit)).path_local);
}

TEST(mcts_tree_form_under_graph_search_carries_the_graph_across_reroots) {
  // Match play carries the tree (C12). At each re-root the graph's table is
  // made again from the kept nodes - the path-local ones left out, the new
  // root entered even where it was one (the root is the network's, so a graph
  // node) - and the search goes on merging at evaluated nodes only: along a
  // line of six decisions from Two Timelines, re-rooted on the chosen child
  // each time (path-local children and graph nodes both), check_invariants
  // reads the table after every step, later transpositions find the carried
  // nodes, no two graph nodes come to hold one state and count, and a
  // path-local node keeps its one parent.
  std::size_t onto_local = 0;
  std::size_t onto_graph = 0;
  for (const std::uint32_t gates : {kGatesInteraction, std::uint32_t{0}}) {
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 23);
    SearchParams p = tree();
    p.graph = true;
    p.tree_gates = gates;
    Search s(variant("Standard - Two Timelines"), p);
    s.run(*ev, Limits{300});
    std::uint64_t later_hits = 0;
    for (int step = 0; step < 6 && !s.root_terminal(); ++step) {
      CHECK_NOTHROW(s.check_invariants());
      const std::size_t pick = s.choose().index;
      const NodeId child = s.root_children().at(pick).child;
      REQUIRE(child != kNoNode);
      (s.node(child).path_local ? onto_local : onto_graph) += 1;
      const std::uint64_t hits = s.stats().transposition_hits;
      s.advance(pick);
      CHECK(!s.node(s.root()).path_local);
      CHECK_NOTHROW(s.check_invariants());
      s.run(*ev, Limits{200});
      CHECK_NOTHROW(s.check_invariants());
      later_hits += s.stats().transposition_hits - hits;
      const std::vector<std::uint32_t> parents = parents_of(s);
      for (NodeId id = 0; id < s.node_count(); ++id) {
        if (s.node(id).path_local) CHECK_EQ(parents[id], std::uint32_t{1});
      }
      for (const std::vector<NodeId>& group : by_identity(s)) {
        std::size_t graph_nodes = 0;
        for (const NodeId id : group) graph_nodes += s.node(id).path_local ? 0u : 1u;
        CHECK(graph_nodes <= 1);
      }
    }
    CHECK(later_hits > 0);
    CHECK_EQ(s.stats().key_collisions, std::uint64_t{0});
  }
  CHECK(onto_local > 0);
  CHECK(onto_graph > 0);
}

TEST(mcts_tree_form_under_graph_search_asks_the_network_once_per_evaluated_state) {
  // §15.2's "same state + epoch -> same evaluation", with no evaluation cache
  // to answer for the graph: under graph search no (state, moves made) is
  // requested twice - every request is the root's or that of one graph node,
  // evaluated and not path-local, and there is one request per evaluated node;
  // with no gate every request but the root's is a turn start's. The same
  // searches in a tree do ask about some state twice (a turn's two orders
  // reach two nodes), so the assertion can fail.
  std::size_t tree_repeats = 0;
  for (const bool graph : {true, false}) {
    for (const std::uint32_t gates : {std::uint32_t{0}, kGatesInteraction}) {
      for (const Root& r : multi_board_roots()) {
        const std::unique_ptr<Evaluator> inner = make_evaluator(BuiltinEvaluator::Random, 14);
        Recording ev(*inner);
        SearchParams p = tree();
        p.graph = graph;
        p.tree_gates = gates;
        p.eval_cache_entries = 0;
        Search s(r.state, p, r.moves_made);
        s.run(ev, Limits{300});
        CHECK_NOTHROW(s.check_invariants());
        std::size_t repeats = 0;
        std::map<std::pair<std::uint64_t, int>, std::vector<std::size_t>> asked;
        for (std::size_t i = 0; i < ev.requests.size(); ++i) {
          std::vector<std::size_t>& same_key = asked[{ev.keys[i], ev.requests[i].second}];
          for (const std::size_t j : same_key) repeats += ev.requests[j].first == ev.requests[i].first ? 1u : 0u;
          same_key.push_back(i);
        }
        if (!graph) {
          tree_repeats += repeats;
          continue;
        }
        CHECK_EQ(repeats, std::size_t{0});
        std::map<std::pair<std::uint64_t, int>, std::vector<NodeId>> holders;  // the evaluated graph nodes
        std::uint64_t evaluated = 0;
        for (NodeId id = 0; id < s.node_count(); ++id) {
          const NodeView n = s.node(id);
          if (!n.evaluated) continue;
          ++evaluated;
          CHECK(!n.path_local);
          holders[{n.key, n.moves_made}].push_back(id);
        }
        for (std::size_t i = 0; i < ev.requests.size(); ++i) {
          const auto& [state, made] = ev.requests[i];
          bool held = false;
          for (const NodeId id : holders[{ev.keys[i], made}]) held = held || *s.node(id).state == state;
          CHECK(held);
          if (gates == 0) CHECK(made == 0 || (made == r.moves_made && state == r.state));
        }
        CHECK_EQ(evaluated, static_cast<std::uint64_t>(ev.requests.size()));
        CHECK_EQ(s.stats().evaluations, evaluated);
      }
    }
  }
  std::printf("  the same searches in a tree asked about a state again %zu times\n", tree_repeats);
  CHECK(tree_repeats > 0);
}

TEST(mcts_tree_form_under_graph_search_a_merged_node_is_its_states_own) {
  // §14.4 refuses path-dependent evaluations, and §8.5 keeps the tree form
  // clear of them by merging at evaluated nodes only. So a node two paths
  // reach holds nothing of either: its value, its edges and their logits are
  // those of a fresh search's root at its state and count (the same evaluator,
  // one playout) - never the logits an inherited node takes from the parent
  // that reached it first. Up to 20 merged nodes a search, mid-turn ones (a
  // gate's) and turn starts.
  std::size_t compared = 0;
  std::size_t mid_turn = 0;
  for (const std::uint32_t gates : {kGatesInteraction, std::uint32_t{0}}) {
    for (const Root& r : multi_board_roots()) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 19);
      SearchParams p = tree();
      p.graph = true;
      p.tree_gates = gates;
      Search s(r.state, p, r.moves_made);
      s.run(*ev, Limits{400});
      CHECK_NOTHROW(s.check_invariants());
      const std::vector<std::uint32_t> parents = parents_of(s);
      std::size_t taken = 0;
      for (NodeId id = 0; id < s.node_count() && taken < 20; ++id) {
        const NodeView n = s.node(id);
        if (parents[id] < 2 || !n.expanded) continue;
        ++taken;
        ++compared;
        mid_turn += n.turn_start ? 0u : 1u;
        CHECK(n.evaluated && !n.inherited && !n.path_local);
        Search fresh(*n.state, p, n.moves_made);
        fresh.run(*ev, Limits{1});
        const NodeView f = fresh.node(fresh.root());
        REQUIRE(f.evaluated);
        CHECK_EQ(n.eval_mean, f.eval_mean);
        CHECK_EQ(n.eval_draw, f.eval_draw);
        CHECK_EQ(n.eval_variance, f.eval_variance);
        std::map<EdgeName, double> own;  // the fresh root's logits
        for (std::size_t i = 0; i < f.edge_count; ++i) {
          const EdgeView e = fresh.edge(fresh.root(), i);
          own[name_of(e)] = e.logit;
        }
        // (An unknown edge verified illegal is removed: the searched node may
        // have lost edges the fresh root still holds.)
        if (s.stats().verified_illegal == 0) CHECK_EQ(n.edge_count, f.edge_count);
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = s.edge(id, i);
          const auto it = own.find(name_of(e));
          REQUIRE(it != own.end());
          CHECK(e.logit == it->second || (std::isnan(e.logit) && std::isnan(it->second)));
        }
      }
    }
  }
  std::printf("  merged nodes compared with a fresh root: %zu, %zu of them mid-turn\n", compared, mid_turn);
  // What the searches met when written (deterministic): 138 merged nodes, 55
  // of them mid-turn; the floors about a third of that.
  CHECK(compared >= 45);
  CHECK(mid_turn >= 18);
}

TEST(mcts_tree_form_under_graph_search_backs_up_from_the_acting_players_view) {
  // mcts_values_are_backed_up_from_the_acting_players_view (test_mcts.cpp) for
  // the tree form under graph search. Every leaf worth +0.5 to White: every
  // node's mean, and every edge's seen from its parent's acting player, is
  // +0.5 at White's nodes and -0.5 at Black's - through the inherited nodes
  // (which back up nothing of their own), the merges and the residual
  // corrections - the sign flipping across a submit edge and nowhere else.
  for (const std::uint32_t gates : {kGatesInteraction, std::uint32_t{0}}) {
    SearchParams p = tree();
    p.graph = true;
    p.tree_gates = gates;
    ByColour ev(0.5);
    Search s(variant("Standard - Two Timelines"), p);
    s.run(ev, Limits{600});
    REQUIRE_EQ(s.stats().terminal_visits, std::uint64_t{0});
    CHECK_NOTHROW(s.check_invariants());
    CHECK(s.stats().transposition_hits > 0);
    CHECK(s.stats().residual_backups > 0);
    CHECK(s.stats().inherited_nodes > 0);
    std::size_t flips = 0;
    std::size_t through_inherited = 0;
    for (NodeId id = 0; id < s.node_count(); ++id) {
      const NodeView n = s.node(id);
      const double mine = n.mover == Colour::White ? 0.5 : -0.5;
      if (n.weight > 0) CHECK(std::abs(n.value / n.weight - mine) < 1e-5);
      through_inherited += n.inherited && n.weight > 0 ? 1u : 0u;
      for (std::size_t i = 0; i < n.edge_count; ++i) {
        const EdgeView e = s.edge(id, i);
        if (e.visits == 0) continue;
        CHECK(std::abs(e.value / e.weight - mine) < 1e-5);
        if (s.node(e.child).mover != n.mover) {
          CHECK(e.submit);
          ++flips;
        }
      }
    }
    CHECK(flips > 0);
    CHECK(through_inherited > 0);
  }
}

TEST(mcts_tree_form_parameters_are_checked) {
  // Under graph search the tree form is no longer refused (§8.5, the
  // 2026-11-15 review settling spec issue 301): the combination is made and
  // runs, whatever the gates.
  for (const std::uint32_t gates : {kGatesInteraction, std::uint32_t{0}, kGateAlways}) {
    SearchParams p = tree();
    p.graph = true;
    p.tree_gates = gates;
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 3);
    Search s(variant("Standard - Two Timelines"), p);
    CHECK(s.run(*ev, Limits{50}) == StopReason::Playouts);
    CHECK_EQ(s.stats().playouts, std::uint64_t{50});
    CHECK_NOTHROW(s.check_invariants());
  }
  SearchParams q = tree();
  q.tree_gates = 32;
  CHECK_THROWS_WITH(Search(variant("Standard"), q), std::invalid_argument, "tree_gates");
  // 6m: without its parent's value an inherited node has no uncertainty
  // signal, so a term that reads one refuses the combination.
  for (int term = 0; term < 4; ++term) {
    SearchParams u = tree();
    u.tree_inherit_value = false;
    if (term == 0) u.variance_scaled_cpuct = true;
    if (term == 1) u.uncertainty_weighted_backup = true;
    if (term == 2) u.variance_bonus = true;
    if (term == 3) {
      u.final_rule = FinalRule::Lcb;
      u.lcb_categorical_weight = 0.5;
    }
    CHECK_THROWS_WITH(Search(variant("Standard"), u), std::invalid_argument, "tree_inherit_value");
    u.tree_inherit_value = true;
    CHECK_NOTHROW(Search(variant("Standard"), u));
    u.tree_inherit_value = false;
    u.split = TurnSplit::BoardMove;  // read by no other arm
    CHECK_NOTHROW(Search(variant("Standard"), u));
  }
  SearchParams none = tree();
  none.tree_inherit_value = false;  // nothing reads the signal
  CHECK_NOTHROW(Search(variant("Standard"), none));
  // The tree form's parameters are read by no other arm: the ply split under
  // MCGS with them set is the ply split.
  SearchParams r = fast();
  r.graph = true;
  r.tree_gates = 0;
  CHECK_NOTHROW(Search(variant("Standard"), r));
}

TREE_AND_GRAPH_TEST(mcts_tree_form_sees_more_turns_per_evaluation) {
  // Every simulation still ends at one evaluated leaf (or a game end), so at
  // equal playouts both arms ask the network about as often; the saving is in
  // what an evaluation buys (§8.2: "a turn with k mandatory boards costs k
  // evaluations along every path"). At an equal evaluation budget, no gate
  // spends every evaluation but the root's on a turn start, the ply split
  // some on the states between - so the tree form reaches more turn starts;
  // and each node it inherited is one the ply split would have evaluated.
  // (All three arms in trees, then all three in graphs: there every
  // evaluation is a distinct state's, and the comparison stands.)
  for (const Root& r : multi_board_roots()) {
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 2);
    std::vector<std::uint64_t> turn_starts;
    for (const int arm : {0, 1, 2}) {
      SearchParams p = arm == 0 ? fast() : tree();
      p.graph = graph;
      if (arm == 2) p.tree_gates = 0;
      p.eval_cache_entries = 0;
      p.synchronous = true;
      Search s(r.state, p, r.moves_made);
      Limits l;
      l.playouts = 100000;
      l.evaluations = 200;
      s.run(*ev, l);
      CHECK_NOTHROW(s.check_invariants());
      std::uint64_t n = 0;
      for (NodeId id = 0; id < s.node_count(); ++id) n += s.node(id).evaluated && s.node(id).turn_start ? 1u : 0u;
      turn_starts.push_back(n);
      if (arm > 0) CHECK(s.stats().inherited_nodes > 0);
      if (arm == 2) CHECK_EQ(n + (r.moves_made > 0 ? 1u : 0u), s.stats().evaluations);
    }
    CHECK(turn_starts[2] > turn_starts[0]);
  }
}
