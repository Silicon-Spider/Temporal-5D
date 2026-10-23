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
//    tree configuration, batched and synchronous, and across a re-root.
//  * With no gate the network sees only the root and turn starts; with one
//    gate it sees exactly the mid-turn nodes whose incoming move fires it,
//    each gate's predicate recomputed here from the rules engine's state.
//  * The capture gate's reading on constructed positions; the inherited
//    logits (a J6 twin included) and the priors taken from them.
//  * §15.4: the invariants (turn structure, no game end during a turn, dead
//    ends pruned, never scored) and the refusal under MCGS (spec issue 301).
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
// the same, bit for bit. Returns the mid-turn leaves the tree form gated.
std::uint64_t compare(const Root& r, SearchParams p, Evaluator& ev, std::uint64_t playouts, const std::string& what) {
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
    if (n.evaluated && !n.turn_start && id != tree_form.root()) ++mid_turn_leaves;
  }
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
// the search (graph search is refused with the tree form, issue 301), both
// legality settings and the pseudo-legal seam, batched and synchronous.
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
      gated += compare(r, p, ev, 48, r.name + ", " + v.name + (k == 0 ? ", random" : ", material"));
      ++compared;
    }
  }
  CHECK(compared >= 60);
  CHECK(mid_roots >= 6);
  CHECK(gated > 0);  // the comparison met mid-turn evaluations
}

TEST(mcts_tree_form_with_every_gate_firing_equals_the_ply_split_in_every_configuration) {
  // Every variation on the same two multi-board roots, with both evaluators.
  const std::vector<Root> rs = {
      Root{"Two Timelines after 3", random_play(variant("Standard - Two Timelines"), 3, 5), 0},
      Root{"Timeline Battleground after 2", random_play(variant("Misc - Timeline Battleground"), 2, 6), 0}};
  for (const Root& r : rs) {
    for (const Variation& v : kVariations) {
      for (const BuiltinEvaluator kind : {BuiltinEvaluator::Random, BuiltinEvaluator::Material}) {
        const std::unique_ptr<Evaluator> ev = make_evaluator(kind, 9);
        SearchParams p = fast();
        v.apply(p);
        compare(r, p, *ev, 64, r.name + ", " + v.name + ", " + evaluator_name(kind));
      }
    }
  }
}

// ---- Inheritance: the network at the root, turn starts and gates only -----------------------

namespace {

// Records each request: its state, its count of moves made.
class Recording final : public Evaluator {
 public:
  explicit Recording(Evaluator& inner) : inner_(inner) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    inner_.evaluate(batch, out);
    for (const EvalRequest& r : batch) requests.emplace_back(*r.state, r.moves_made);
  }
  std::vector<std::pair<GameState, int>> requests;

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

TEST(mcts_tree_form_without_gates_evaluates_only_the_root_and_turn_starts) {
  for (const Root& r : multi_board_roots()) {
    const std::unique_ptr<Evaluator> inner = make_evaluator(BuiltinEvaluator::Random, 4);
    Recording ev(*inner);
    SearchParams p = tree();
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
  }
}

TEST(mcts_tree_form_evaluates_exactly_where_a_gate_fires) {
  // Each gate alone (the capture gate under its Any reading): a mid-turn node
  // is evaluated exactly when its incoming move fires it, by the predicate
  // recomputed here; else it is inherited. Positions from random play in
  // tactical variants, so that captures and checks occur.
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
      for (NodeId id = 0; id < s.node_count(); ++id) {
        const NodeView n = s.node(id);
        for (std::size_t i = 0; i < n.edge_count; ++i) {
          const EdgeView e = s.edge(id, i);
          if (e.submit || e.child == kNoNode || e.child == s.root()) continue;
          const NodeView c = s.node(e.child);
          if (!c.expanded || c.turn_start) continue;
          const bool want = predicate(gate, *n.state, *e.move, *c.state, k.check, k.present, k.still);
          CHECK_EQ(c.evaluated, want);
          CHECK_EQ(c.inherited, !want);
          (want ? fired : quiet) += 1;
          gated += want ? 1 : 0;
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

TEST(mcts_tree_form_natural_gates_are_the_union_of_the_four) {
  // §8.2's four together (the capture gate's Reach reading, the default): an
  // evaluated mid-turn node's move fires one of them; an inherited one's fires
  // neither the between, the check nor the present gate, and the capture
  // gate's reading is held on constructed positions below.
  for (const Root& r : multi_board_roots()) {
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 8);
    Search s(r.state, tree(), r.moves_made);
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
      }
    }
  }
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

TEST(mcts_tree_form_capture_gate_reading) {
  for (const CaptureReach reading : {CaptureReach::Reach, CaptureReach::Any}) {
    for (const bool can_reach : {true, false}) {
      const GameState start = can_reach ? reaching() : blocked();
      REQUIRE_EQ(start.mandatory_count(), 2);
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Uniform);
      SearchParams p = tree();
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
    }
  }
}

TEST(mcts_tree_form_inherits_its_parents_logits) {
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
    p.tree_gates = 0;
    p.widening = false;
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

TEST(mcts_tree_form_evaluates_a_carried_root_it_inherited) {
  // Match play carries the tree (C12). Advanced onto a child the tree form
  // inherited - a mid-turn node the network never saw - the search's root is
  // evaluated by the first simulation after the re-root, as a leaf, and never
  // again: its edges, their children and statistics kept, their logits and
  // priors and its value the network's - a fresh search's root's at that
  // state - and every edge already visited still instantiated.
  std::size_t carried = 0;
  for (const Root& r : multi_board_roots()) {
    for (const std::uint32_t gates : {std::uint32_t{0}, kGatesInteraction}) {
      for (const bool one_first : {true, false}) {
        const std::unique_ptr<Evaluator> inner = make_evaluator(BuiltinEvaluator::Random, 6);
        Recording ev(*inner);
        SearchParams p = tree();
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
        s.advance(pick);
        const NodeView carried_root = s.node(s.root());
        REQUIRE(carried_root.inherited);
        CHECK(!carried_root.evaluated);
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

TEST(mcts_tree_form_inherited_value_is_its_parents) {
  // tree_inherit_value: an inherited node's eval_* are the node's above it -
  // the one its priors came from (its own inherited ones in a run of them) -
  // and 0 without; the searches differ (the FPU base of its first children)
  // and both keep the invariants.
  std::size_t below_turn_starts = 0;
  std::size_t checked = 0;
  bool differ = false;
  for (const Root& r : multi_board_roots()) {
    std::uint64_t digests[2] = {0, 0};
    for (const bool value : {true, false}) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 12);
      SearchParams p = tree();
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

TEST(mcts_tree_form_inherited_nodes_take_a_leafs_legality) {
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

TEST(mcts_tree_form_keeps_the_turn_structure_and_prunes_dead_ends_unscored) {
  // The pseudo-legal seam makes dead ends; with no gate they are met at nodes
  // the network never sees - pruned, never scored, never evaluated - and the
  // invariants hold: a submit edge into a turn start, a move edge into a
  // node during an action, no game end during one, an inherited node's visits
  // all passing through it.
  std::uint64_t dead = 0;
  for (const Root& r : multi_board_roots()) {
    for (const std::uint32_t gates : {std::uint32_t{0}, kGatesInteraction}) {
      const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Material);
      SearchParams p = tree();
      p.tree_gates = gates;
      p.legal_moves_only = false;
      p.proof_channel = true;
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

TEST(mcts_tree_form_is_refused_under_graph_search) {
  SearchParams p = tree();
  p.graph = true;
  CHECK_THROWS_WITH(Search(variant("Standard"), p), std::invalid_argument, "301");
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

TEST(mcts_tree_form_sees_more_turns_per_evaluation) {
  // Every simulation still ends at one evaluated leaf (or a game end), so at
  // equal playouts both arms ask the network about as often; the saving is in
  // what an evaluation buys (§8.2: "a turn with k mandatory boards costs k
  // evaluations along every path"). At an equal evaluation budget, no gate
  // spends every evaluation but the root's on a turn start, the ply split
  // some on the states between - so the tree form reaches more turn starts;
  // and each node it inherited is one the ply split would have evaluated.
  for (const Root& r : multi_board_roots()) {
    const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 2);
    std::vector<std::uint64_t> turn_starts;
    for (const int arm : {0, 1, 2}) {
      SearchParams p = arm == 0 ? fast() : tree();
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
