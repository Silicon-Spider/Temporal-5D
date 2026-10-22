// t5d-tree-count: D2's tree form against the ply split on turn starts of
// recorded games - evaluation COUNTS, never timings (design spec §8.2,
// "Evaluating at turn boundaries"; register D2; docs/MEASUREMENTS.md). What
// the tree form buys is turns per evaluation: the ply split spends evaluations
// on the states between a turn's moves, the tree form only where an
// interaction gate fires. The counts are a function of the search's
// parameters and the positions alone - deterministic, the same on any machine
// and under any load.
//
//   t5d-tree-count RECORD ...        (--help: this text)
//
// POSITIONS. The records are read in the order given (a shell's sorted glob:
// build/t0db/*.5dpgn) and replayed under the standard rules; from each, at
// most one turn start is taken - the first, at action 6 or later and not a
// game end, that fills a class still wanting one: two mandatory boards (3
// wanted; the k-th of them with at least 20k boards), three or more (3), one
// with 30 boards or more (1). Records the engine refuses are skipped and
// counted. Taking stops once every class is filled.
//
// ARMS, each a fresh search from the turn start with the built-in random
// evaluator (seed 1), synchronous (one leaf per evaluation), no evaluation
// cache, square features off (the built-in evaluators read none), every other
// parameter the search's default:
//
//   ply split                    TurnSplit::BoardMove
//   tree form, natural gates     TurnSplit::Tree, §8.2's four gates (the defaults)
//   tree form, no gates          TurnSplit::Tree, tree_gates 0
//   tree, natural, rewind only   TurnSplit::Tree, tree_present Rewind
//
// Each arm runs twice: to 400 EVALUATIONS (playouts unbounded) - the turn
// starts and mid-turn nodes the network evaluated, the nodes inherited, and
// the deepest evaluated turn start (submit edges from the root) - and to 400
// PLAYOUTS - the evaluations made, the nodes inherited and the mid-turn leaves
// by the gate that fired first (between, capture, check, present).
//
// Not one of the truth's sources (CMakeLists.txt, T5D_TRUTH_SOURCES): it
// decides no verdict.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "t5d/mcts.hpp"
#include "t5d/pgn.hpp"
#include "t5d/search.hpp"

using namespace t5d;
using namespace t5d::mcts;

namespace {

struct Pick {
  std::string name;
  GameState state;
};

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Row {
  std::uint64_t evaluations = 0;
  std::uint64_t playouts = 0;
  std::uint64_t turn_starts_evaluated = 0;
  std::uint64_t mid_turn_evaluated = 0;
  std::uint64_t inherited = 0;
  std::uint64_t deepest = 0;
  std::array<std::uint64_t, 5> gates{};
};

constexpr int kArms = 4;
const char* const kArmNames[kArms] = {"ply split", "tree form, natural gates", "tree form, no gates",
                                      "tree, natural, rewind only"};

Row count(const GameState& s, int arm, const Limits& limits) {
  SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.eval_cache_entries = 0;
  p.synchronous = true;
  if (arm > 0) p.split = TurnSplit::Tree;
  if (arm == 2) p.tree_gates = 0;
  if (arm == 3) p.tree_present = PresentShift::Rewind;
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 1);
  Search search(s, p);
  search.run(*ev, limits);
  Row r;
  r.evaluations = search.stats().evaluations;
  r.playouts = search.stats().playouts;
  r.inherited = search.stats().inherited_nodes;
  r.gates = search.stats().gate_fired;
  // Breadth-first over the tree: the evaluated nodes, and how many turns deep
  // (submit edges from the root) the deepest evaluated turn start lies.
  std::vector<std::pair<NodeId, std::uint64_t>> queue{{search.root(), 0}};
  for (std::size_t i = 0; i < queue.size(); ++i) {
    const auto [id, depth] = queue[i];
    const NodeView n = search.node(id);
    if (n.evaluated) (n.turn_start ? r.turn_starts_evaluated : r.mid_turn_evaluated) += 1;
    if (n.turn_start && n.evaluated && depth > r.deepest) r.deepest = depth;
    for (std::size_t e = 0; e < n.edge_count; ++e) {
      const EdgeView edge = search.edge(id, e);
      if (edge.child != kNoNode) queue.emplace_back(edge.child, depth + (edge.submit ? 1 : 0));
    }
  }
  return r;
}

unsigned long long u(std::uint64_t v) { return static_cast<unsigned long long>(v); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
    std::fputs("usage: t5d-tree-count RECORD ...   (D2's tree form against the ply split: evaluation counts)\n",
               argc < 2 ? stderr : stdout);
    return argc < 2 ? 2 : 0;
  }
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<Pick> picks;
  int want2 = 3;
  int want3 = 3;
  int want1 = 1;
  std::size_t refused = 0;
  for (int i = 1; i < argc && (want2 > 0 || want3 > 0 || want1 > 0); ++i) {
    try {
      PgnReplay replay(read_pgn(slurp(argv[i])));
      std::size_t action = 0;
      while (!replay.finished()) {
        const GameState& s = replay.state();
        const int mandatory = s.mandatory_count();
        const std::size_t boards = s.multiverse().board_count();
        if (action >= 6 && outcome(s, standard_check_ruling()) == Outcome::Ongoing) {
          const std::string name = std::string(argv[i]) + " action " + std::to_string(action) + ", " +
                                   std::to_string(boards) + " boards, " + std::to_string(mandatory) + " mandatory, " +
                                   std::to_string(s.playable_timelines().size()) + " playable";
          if (mandatory == 2 && want2 > 0 && boards >= 20 * static_cast<std::size_t>(4 - want2)) {
            picks.push_back(Pick{name, s});
            --want2;
            break;
          }
          if (mandatory >= 3 && want3 > 0) {
            picks.push_back(Pick{name, s});
            --want3;
            break;
          }
          if (mandatory == 1 && want1 > 0 && boards >= 30) {
            picks.push_back(Pick{name, s});
            --want1;
            break;
          }
        }
        replay.play_next();
        ++action;
      }
    } catch (const std::exception&) {
      ++refused;  // a record the engine refuses (or cannot read on): skipped
    }
  }
  std::printf("t5d-tree-count: %zu turn starts; %zu records skipped; random evaluator seed 1, synchronous, no cache\n",
              picks.size(), refused);
  for (const Pick& p : picks) {
    std::printf("%s\n", p.name.c_str());
    for (int arm = 0; arm < kArms; ++arm) {
      Limits by_evaluations;
      by_evaluations.playouts = 1000000;
      by_evaluations.evaluations = 400;
      const Row a = count(p.state, arm, by_evaluations);
      const Row b = count(p.state, arm, Limits{400});
      std::printf(
          "  %-26s 400 evals: %4llu playouts, turn starts evaluated %4llu, mid-turn evaluated %4llu, inherited %4llu, "
          "deepest evaluated turn start %llu turns | 400 playouts: %4llu evals, %4llu inherited, gates b/c/k/p "
          "%llu/%llu/%llu/%llu\n",
          kArmNames[arm], u(a.playouts), u(a.turn_starts_evaluated), u(a.mid_turn_evaluated), u(a.inherited),
          u(a.deepest), u(b.evaluations), u(b.inherited), u(b.gates[0]), u(b.gates[1]), u(b.gates[2]),
          u(b.gates[3]));
    }
  }
  return 0;
}
