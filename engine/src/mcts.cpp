// The main search (t5d/mcts.hpp; engine stage E6a). Written from the design
// spec (§8.1-8.7, §11.1, §11.2, §15.4) and the papers it cites - PUCT
// (AlphaZero), progressive widening, MCTS-Solver (Winands, Björnsson and Saito
// 2008), score-bounded MCTS (Cazenave and Saffidine 2011), Monte-Carlo graph
// search with residual backups (Czech, Korus and Kersting 2021), KataGo's
// variance-scaled cPUCT and uncertainty-weighted playouts and its LCB move
// selection - and the Chess Programming Wiki. No code from Lc0, KataGo or any
// other engine (register H29).

#include "t5d/mcts.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/types.hpp"

namespace t5d::mcts {

namespace detail {

constexpr std::size_t kNoEdge = std::numeric_limits<std::size_t>::max();

constexpr std::uint64_t mix64(std::uint64_t z) noexcept {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// SplitMix64: a small, portable generator (root noise).
struct SplitMix {
  std::uint64_t state;
  std::uint64_t next() noexcept {
    state += 0x9E3779B97F4A7C15ull;
    return mix64(state);
  }
  /// In [0, 1).
  double uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

double standard_normal(SplitMix& rng) {
  const double u1 = std::max(rng.uniform(), 1e-300);
  const double u2 = rng.uniform();
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// Gamma(shape, 1): Marsaglia and Tsang's method; shape < 1 by boosting.
double gamma_sample(double shape, SplitMix& rng) {
  if (shape < 1.0) {
    const double u = std::max(rng.uniform(), 1e-300);
    return gamma_sample(shape + 1.0, rng) * std::pow(u, 1.0 / shape);
  }
  const double d = shape - 1.0 / 3.0;
  const double c = 1.0 / std::sqrt(9.0 * d);
  for (;;) {
    const double x = standard_normal(rng);
    double v = 1.0 + c * x;
    if (v <= 0) continue;
    v = v * v * v;
    const double u = std::max(rng.uniform(), 1e-300);
    if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
    if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
  }
}

// One child slot of a node. Its statistics are the values backed up THROUGH
// this edge, in the parent's perspective (MCGS: path statistics; in a tree they
// equal the child's own).
struct Edge {
  Move move{};                // board-move: the move
  std::uint32_t action = 0;   // whole-turn: the action's index in the node's list
  bool submit = false;        // board-move, Q4's other reading: submit the action
  bool pruned = false;        // a dead end (no legal completion), or verified illegal
  bool unknown = false;       // (a) the expansion's budget left the move's legality open
  bool verified = false;      // (c) verified once
  double prior = 0;
  NodeId child = kNoNode;     // created on first selection
  std::uint32_t visits = 0;
  std::uint32_t vl = 0;       // virtual visits of paths waiting in the batch
  double w = 0;               // sum of backup weights
  double wq = 0;              // weighted sum of values
  double wq2 = 0;             // weighted sum of squares
  double wd = 0;              // weighted sum of draw probabilities (D3)
};

// A heap block's bookkeeping, about (malloc's header and alignment), and a
// make_shared control block's.
constexpr std::size_t kHeapBlock = 16;
constexpr std::size_t kControlBlock = 16;

// What a state holds on the heap that the state it came from does not: its
// own list of timeline pointers and, for each board its last step made (a
// move, or a whole action), the board and the timeline object holding it
// (everything older is shared, t5d/multiverse.hpp).
inline std::size_t state_heap_bytes(const GameState& state, std::size_t new_boards) noexcept {
  return static_cast<std::size_t>(state.multiverse().timeline_count()) * sizeof(TimelineRef) + kHeapBlock +
         std::max<std::size_t>(new_boards, 1) *
             (sizeof(Timeline) + kControlBlock + kHeapBlock + sizeof(Board) + kControlBlock + kHeapBlock);
}

struct Node {
  explicit Node(GameState s) : state(std::move(s)) {}

  GameState state;
  std::uint64_t key = 0;
  Colour mover = Colour::White;
  std::int32_t moves_made = 0;  // A6: the moves made in its action - part of its identity
  bool turn_start = true;       // moves_made == 0
  Terminal terminal = Terminal::None;
  bool expanded = false;
  bool pending = false;       // waiting in the batch for its evaluation
  bool stuck = false;         // a turn start with no child and no verdict
  bool dead_end = false;      // a partial turn with no legal completion
  bool children_complete = true;
  bool submit_possible = false; // board-move, D8: a submit edge (obligations met, T5)
  bool submit_legal = false;    // ... legal (C4), else kept pruned
  bool has_eval = false;
  std::int8_t lo = -1;        // solver bounds, the mover's view
  std::int8_t hi = 1;
  std::uint32_t visits = 0;
  std::uint32_t ends = 0;     // visits that ended here
  std::uint32_t vl = 0;
  double w = 0;
  double wq = 0;
  double wq2 = 0;
  double wd = 0;              // weighted sum of draw probabilities (D3)
  double ends_w = 0;
  double ends_wq = 0;
  double ends_wd = 0;
  double eval_mean = 0;       // D3: its evaluation's W - L
  double eval_draw = 0;       // ... and D
  double eval_var = 0;        // ... and uncertainty signal (SearchParams::uncertainty_source)
  float eval_error = std::numeric_limits<float>::quiet_NaN();       // the error head's output
  float eval_moves_left = std::numeric_limits<float>::quiet_NaN();  // moves-left (never backed up)
  double k0 = 1;
  std::size_t instantiated = 0;
  std::vector<Edge> edges;
  // Whole-turn: the actions, their moves concatenated; offsets.size() = actions + 1
  // once listed (empty: not listed yet).
  std::vector<Move> action_moves;
  std::vector<std::uint32_t> action_offsets;
  std::size_t bytes = 0;      // memory_bytes' count for this node (account)
  std::size_t new_boards = 1; // the boards the edge into it made (its state's own)

  std::size_t action_count() const noexcept {
    return action_offsets.empty() ? 0 : action_offsets.size() - 1;
  }
  std::span<const Move> action(std::size_t a) const noexcept {
    return {action_moves.data() + action_offsets[a],
            static_cast<std::size_t>(action_offsets[a + 1] - action_offsets[a])};
  }
};

struct Step {
  NodeId node = kNoNode;
  std::size_t edge = kNoEdge;  // the edge of the previous node that leads here
};

struct Pending {
  std::vector<Step> path;
  std::size_t slot = 0;
  NodeId leaf = kNoNode;
  std::size_t request = 0;  // its evaluation: one per distinct (state, count) in the batch (A6)
};

enum class Selected : std::uint8_t { Pending, Done, Collision, DeadEnd };

int sign_between(const Node& parent, const Node& child) noexcept {
  return parent.mover == child.mover ? 1 : -1;
}

// A child's bounds in its parent's view.
std::pair<int, int> bounds_in_parent(const Node& parent, const Node& child) noexcept {
  if (sign_between(parent, child) > 0) return {child.lo, child.hi};
  return {-child.hi, -child.lo};
}

double terminal_value(Terminal t) noexcept { return t == Terminal::Checkmate ? -1.0 : 0.0; }

// D3: a game end's draw probability - 0 for a mate, 1 for a stalemate and the
// turn cap's convention (I11: a draw).
double terminal_draw(Terminal t) noexcept { return t == Terminal::Checkmate ? 0.0 : 1.0; }

bool same_state(const GameState& a, const GameState& b, bool compare_origins) noexcept {
  return compare_origins ? a == b : same_position(a, b);
}

std::uint64_t combine(std::uint64_t h, std::uint64_t v) noexcept {
  return mix64(h ^ (v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2)));
}

std::uint64_t bits_of(double d) noexcept { return std::bit_cast<std::uint64_t>(d); }

double seconds_since(std::chrono::steady_clock::time_point t0) noexcept {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace detail

using detail::Edge;
using detail::kNoEdge;
using detail::Node;
using detail::Pending;
using detail::Selected;
using detail::Step;

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

bool TurnCap::reached(const GameState& turn_start) const noexcept {
  if (turns > 0 && turn_of(turn_start.present()) > turns) return true;
  if (boards > 0 &&
      turn_start.multiverse().board_count() >= static_cast<std::size_t>(boards)) {
    return true;
  }
  return false;
}

ValueMoments value_moments(const std::array<float, kValueBins>& value) {
  double sum = 0;
  for (const float p : value) {
    if (!std::isfinite(p) || p < 0) {
      throw std::invalid_argument("a value distribution holds a negative or non-finite probability");
    }
    sum += static_cast<double>(p);
  }
  if (!(sum > 0)) throw std::invalid_argument("a value distribution sums to zero");
  double mean = 0;
  for (int k = 0; k < kValueBins; ++k) {
    mean += static_cast<double>(value[static_cast<std::size_t>(k)]) / sum * value_bin_centre(k);
  }
  double var = 0;
  for (int k = 0; k < kValueBins; ++k) {
    const double d = value_bin_centre(k) - mean;
    var += static_cast<double>(value[static_cast<std::size_t>(k)]) / sum * d * d;
  }
  return ValueMoments{mean, var};
}

WdlValue wdl_value(const std::array<float, kWdl>& wdl) {
  double sum = 0;
  for (const float p : wdl) {
    if (!std::isfinite(p) || p < 0) {
      throw std::invalid_argument("a WDL holds a negative or non-finite probability");
    }
    sum += static_cast<double>(p);
  }
  if (!(sum > 0)) throw std::invalid_argument("a WDL sums to zero");
  return WdlValue{(static_cast<double>(wdl[0]) - static_cast<double>(wdl[2])) / sum,
                  static_cast<double>(wdl[1]) / sum};
}

std::array<float, kWdl> wdl_of(double v) {
  if (std::isnan(v)) throw std::invalid_argument("wdl_of: the value is NaN");
  const double c = std::clamp(v, -1.0, 1.0);
  return {static_cast<float>((1.0 + c) * 0.5), 0.0f, static_cast<float>((1.0 - c) * 0.5)};
}

std::array<float, kValueBins> two_hot(double v) {
  if (std::isnan(v)) throw std::invalid_argument("two_hot: the value is NaN");
  std::array<float, kValueBins> out{};
  const double x = (std::clamp(v, -1.0, 1.0) + 1.0) * 0.5 * (kValueBins - 1);
  const int lower = std::clamp(static_cast<int>(std::floor(x)), 0, kValueBins - 1);
  const int upper = std::min(lower + 1, kValueBins - 1);
  const double frac = x - lower;
  out[static_cast<std::size_t>(lower)] += static_cast<float>(1.0 - frac);
  out[static_cast<std::size_t>(upper)] += static_cast<float>(frac);
  return out;
}

std::uint64_t node_key(const GameState& state, NodeKey keying) noexcept {
  std::uint64_t key = state.zobrist();
  if (keying == NodeKey::StateAndOrigins) {
    // Each timeline's branch origin (the lineage reading of spec issue 30):
    // not in the 5D Zobrist key, which holds only what the rules read.
    for (const TimelineRef& t : state.multiverse().timelines()) {
      if (!t->origin().has_value()) continue;
      const BoardCoord origin = t->origin().value();  // no optional copied (g++ 13/14 warn)
      const auto p = static_cast<std::uint64_t>(static_cast<std::uint32_t>(t->position()));
      const auto ot = static_cast<std::uint64_t>(static_cast<std::uint32_t>(origin.timeline));
      const auto oh = static_cast<std::uint64_t>(static_cast<std::uint32_t>(origin.half_turn));
      key = detail::combine(key, detail::mix64((p << 32) ^ 0x6F726967696E7321ull) ^
                                     detail::mix64((ot << 32) | oh));
    }
  }
  return key;
}

const char* stop_reason_name(StopReason reason) noexcept {
  switch (reason) {
    case StopReason::Playouts: return "playouts";
    case StopReason::Evaluations: return "evaluations";
    case StopReason::Finished: return "finished";
    case StopReason::Exhausted: return "exhausted";
    case StopReason::Nodes: return "nodes";
    case StopReason::Memory: return "memory";
    case StopReason::Deadline: return "deadline";
  }
  return "?";
}

const char* choice_reason_name(ChoiceReason reason) noexcept {
  switch (reason) {
    case ChoiceReason::ProvenWin: return "proven_win";
    case ChoiceReason::SecureChild: return "secure_child";
    case ChoiceReason::Lcb: return "lcb";
    case ChoiceReason::Visits: return "visits";
    case ChoiceReason::AllLost: return "all_lost";
  }
  return "?";
}

std::vector<double> dirichlet_alphas(std::span<const double> priors, const SearchParams& params) {
  if (priors.empty()) throw std::invalid_argument("dirichlet_alphas: no open edge");
  const std::size_t n = priors.size();
  const double count = static_cast<double>(n);
  if (!(params.dirichlet_concentration > 0)) {
    // The per-edge alpha (the arm before CHECKPOINT 7), exactly.
    if (!params.dirichlet_shaping) return std::vector<double>(n, params.dirichlet_alpha);
  }
  const double total =
      params.dirichlet_concentration > 0 ? params.dirichlet_concentration : params.dirichlet_alpha * count;
  std::vector<double> out(n, total / count);
  if (!params.dirichlet_shaping) return out;
  // The shaping (a prototype): the share s spread by how far each edge's
  // capped log-prior lies above the mean - "moves the prior ranks above most
  // others" - the rest uniformly; uniform where nothing lies above the mean.
  std::vector<double> r(n);
  double mean = 0;
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = std::log(std::max(std::min(priors[i], params.dirichlet_shaping_cap), 1e-300));
    mean += r[i];
  }
  mean /= count;
  double sum = 0;
  for (double& x : r) {
    x = std::max(0.0, x - mean);
    sum += x;
  }
  if (!(sum > 0)) return out;
  const double s = params.dirichlet_shaping_share;
  for (std::size_t i = 0; i < n; ++i) out[i] = total * ((1.0 - s) / count + s * r[i] / sum);
  return out;
}

// ---------------------------------------------------------------------------
// Final selection (E5, E9)
// ---------------------------------------------------------------------------

namespace {

double lcb_of(const FinalCandidate& c, const SearchParams& p) {
  const double n = std::max<double>(c.visits, 1.0);
  const double var = std::max(c.variance, 0.0) + p.lcb_categorical_weight * std::max(c.eval_variance, 0.0);
  return c.q - p.lcb_z * std::sqrt(var / n);
}

double secure_of(const FinalCandidate& c, const SearchParams& p) {
  if (c.solved()) return static_cast<double>(c.lo);
  return c.q - p.secure_child_a / std::sqrt(std::max<double>(c.visits, 1.0));
}

bool eligible_basic(const FinalCandidate& c) noexcept { return c.visits > 0 && c.hi > -1; }

}  // namespace

bool final_eligible(std::span<const FinalCandidate> children, std::size_t i, ChoiceReason reason,
                    const SearchParams& params) {
  if (i >= children.size()) return false;
  const FinalCandidate& c = children[i];
  switch (reason) {
    case ChoiceReason::ProvenWin: return c.lo == 1 && c.visits > 0;
    case ChoiceReason::SecureChild:
    case ChoiceReason::Visits: return eligible_basic(c);
    case ChoiceReason::Lcb: {
      if (!eligible_basic(c)) return false;
      std::uint32_t most = 0;
      for (const FinalCandidate& d : children) {
        if (eligible_basic(d)) most = std::max(most, d.visits);
      }
      return static_cast<double>(c.visits) >= params.lcb_floor * static_cast<double>(most);
    }
    case ChoiceReason::AllLost: return c.visits > 0;
  }
  return false;
}

Choice select_final(std::span<const FinalCandidate> children, const SearchParams& params) {
  const std::size_t n = children.size();
  // 1. A proven win is played at once (the most visited of them).
  std::size_t best = kNoEdge;
  for (std::size_t i = 0; i < n; ++i) {
    const FinalCandidate& c = children[i];
    if (c.lo == 1 && c.visits > 0 && (best == kNoEdge || c.visits > children[best].visits)) best = i;
  }
  if (best != kNoEdge) return Choice{best, ChoiceReason::ProvenWin};
  // 2. Children proven lost are ineligible unless all are.
  bool any_eligible = false;
  bool any_solved = false;
  for (const FinalCandidate& c : children) {
    if (!eligible_basic(c)) continue;
    any_eligible = true;
    any_solved = any_solved || c.solved();
  }
  if (!any_eligible) {
    for (std::size_t i = 0; i < n; ++i) {
      if (children[i].visits > 0 && (best == kNoEdge || children[i].visits > children[best].visits)) {
        best = i;
      }
    }
    if (best == kNoEdge) throw std::invalid_argument("final selection: no child has a visit");
    return Choice{best, ChoiceReason::AllLost};
  }
  // 3. A solved eligible child: the secure child (the largest lower bound).
  if (any_solved) {
    double best_value = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const FinalCandidate& c = children[i];
      if (!eligible_basic(c)) continue;
      const double v = secure_of(c, params);
      if (best == kNoEdge || v > best_value ||
          (v == best_value && c.visits > children[best].visits)) {
        best = i;
        best_value = v;
      }
    }
    return Choice{best, ChoiceReason::SecureChild};
  }
  // 4. The rule.
  if (params.final_rule == FinalRule::Visits) {
    for (std::size_t i = 0; i < n; ++i) {
      const FinalCandidate& c = children[i];
      if (!eligible_basic(c)) continue;
      if (best == kNoEdge || c.visits > children[best].visits ||
          (c.visits == children[best].visits && c.q > children[best].q)) {
        best = i;
      }
    }
    return Choice{best, ChoiceReason::Visits};
  }
  double best_value = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!final_eligible(children, i, ChoiceReason::Lcb, params)) continue;
    const FinalCandidate& c = children[i];
    const double v = lcb_of(c, params);
    if (best == kNoEdge || v > best_value || (v == best_value && c.visits > children[best].visits)) {
      best = i;
      best_value = v;
    }
  }
  return Choice{best, ChoiceReason::Lcb};
}

// ---------------------------------------------------------------------------
// The evaluation cache
// ---------------------------------------------------------------------------

const Evaluation* EvalCache::find(std::uint64_t key, int moves_made, std::uint64_t epoch, const GameState& state,
                                  bool compare_origins) const {
  const auto [first, last] = entries_.equal_range(key);
  for (auto it = first; it != last; ++it) {
    if (it->second.epoch == epoch && it->second.moves_made == moves_made &&
        detail::same_state(it->second.state, state, compare_origins)) {
      return &it->second.evaluation;
    }
  }
  return nullptr;
}

std::size_t EvalCache::entry_bytes(const GameState& state, const Evaluation& evaluation,
                                   std::size_t new_boards) noexcept {
  // The map's node (key, entry, link), the logits' block, and the state's own
  // heap - what the entry keeps alive once its search is gone.
  return sizeof(std::uint64_t) + sizeof(Entry) + sizeof(void*) + detail::kHeapBlock +
         evaluation.logits.capacity() * sizeof(float) + detail::kHeapBlock +
         detail::state_heap_bytes(state, new_boards);
}

void EvalCache::insert(std::uint64_t key, int moves_made, std::uint64_t epoch, const GameState& state,
                       const Evaluation& evaluation, std::size_t new_boards) {
  if (entries_.size() >= capacity_) return;
  if (find(key, moves_made, epoch, state, true) != nullptr) return;
  Entry e{state, moves_made, epoch, evaluation, 0};
  e.evaluation.logits.shrink_to_fit();
  e.bytes = entry_bytes(e.state, e.evaluation, new_boards);
  if (bytes_ + e.bytes > max_bytes_) return;
  bytes_ += e.bytes;
  entries_.emplace(key, std::move(e));
}

std::size_t EvalCache::purge_fewer_boards(std::size_t boards) {
  std::size_t removed = 0;
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (it->second.state.multiverse().board_count() < boards) {
      bytes_ -= it->second.bytes;
      it = entries_.erase(it);
      ++removed;
    } else {
      ++it;
    }
  }
  return removed;
}

// ---------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------

struct Search::Impl {
  SearchParams params_;
  std::vector<Node> nodes_;
  NodeId root_ = 0;
  std::unordered_map<std::uint64_t, NodeId> table_;  // MCGS
  std::shared_ptr<EvalCache> cache_;
  std::uint64_t epoch_ = 0;
  SearchStats stats_;
  std::vector<Pending> pending_;
  std::vector<std::vector<Step>> collided_;
  std::vector<EvalRequest> requests_;
  std::size_t unique_requests_ = 0;
  std::vector<std::unique_ptr<Expansion>> slots_;
  std::function<void(const GameState&, std::uint64_t)> hook_;
  std::uint64_t root_generation_ = 1;
  std::uint64_t notified_generation_ = 0;
  bool noise_applied_ = false;
  // I15's root side table: each root edge's prior before the noise (after the
  // temperatures, the unknown-move scale and the submit share), in the root's
  // edge order - filled by the root's own expansion; empty for a carried root
  // (expanded as a child: its prior is that already).
  std::vector<double> root_prior_noise_free_;
  // ... and the Dirichlet concentration the root noise drew each edge's
  // variate with (E1: dirichlet_alphas over the open edges; a pruned edge's
  // unused one), in the same order - empty when no noise was drawn.
  std::vector<double> root_noise_alpha_;
  // D4's effort label (I15): the root edge with the most visits, and the root's
  // visits when it became so ("visits until the decision stabilises", §8.6).
  std::size_t best_edge_ = kNoEdge;
  std::uint32_t best_since_ = 0;
  std::vector<Step> path_;
  std::size_t tree_bytes_ = 0;  // the nodes' counts (Node::bytes), summed

  Impl(GameState root, const SearchParams& params, int moves_made, std::shared_ptr<EvalCache> cache)
      : params_(params) {
    validate();
    check_moves_made(root, moves_made);
    if (cache) {
      cache_ = std::move(cache);
    } else if (params_.eval_cache_entries > 0) {
      cache_ = std::make_shared<EvalCache>(params_.eval_cache_entries, params_.eval_cache_bytes);
    }
    root_ = make_node(std::move(root), moves_made);
    register_node(root_);
    purge_cache();
  }

  // A root's count of moves made in its action (A6): 0 or more; 0 - a turn
  // start - only where the obligations are not met (every turn begins with the
  // present on the mover's colour, T5, so with a mandatory board); under the
  // whole-turn split, 0 only.
  void check_moves_made(const GameState& root, int moves_made) const {
    if (moves_made < 0) {
      throw std::invalid_argument("search: moves_made (the moves made in the root's action, A6) must be 0 or more, "
                                  "not " + std::to_string(moves_made));
    }
    if (params_.split == TurnSplit::WholeTurn && moves_made != 0) {
      throw std::invalid_argument("the whole-turn split searches from turn starts only (moves_made 0)");
    }
    if (moves_made == 0 && root.obligations_met()) {
      throw std::invalid_argument(
          "search: a root with no move made in its action is a turn start, and a turn start's present lies on "
          "the mover's colour (T5) - this root's obligations are met, so its action has moves: give their count (A6)");
    }
  }

  // The evaluation cache's entries behind the root, gone for good.
  void purge_cache() {
    if (cache_) stats_.cache_purged += cache_->purge_fewer_boards(nodes_[root_].state.multiverse().board_count());
  }

  // A node's memory: itself, its edges, its actions' moves, its state's own heap.
  static std::size_t node_bytes(const Node& n) noexcept {
    return sizeof(Node) + n.edges.capacity() * sizeof(Edge) + n.action_moves.capacity() * sizeof(Move) +
           n.action_offsets.capacity() * sizeof(std::uint32_t) + 3 * detail::kHeapBlock +
           detail::state_heap_bytes(n.state, n.new_boards);
  }

  void account(Node& n) noexcept {
    const std::size_t b = node_bytes(n);
    tree_bytes_ = tree_bytes_ - n.bytes + b;
    n.bytes = b;
  }

  std::size_t memory_bytes() const noexcept {
    // The MCGS table: a node per entry (key, id, link) and its bucket.
    constexpr std::size_t kTableEntry = sizeof(std::uint64_t) + sizeof(NodeId) + 2 * sizeof(void*) + detail::kHeapBlock;
    return tree_bytes_ + (nodes_.capacity() - nodes_.size()) * sizeof(Node) + table_.size() * kTableEntry;
  }

  void validate() const {
    const SearchParams& p = params_;
    if (p.proof_store) {
      throw std::invalid_argument(
          "SearchParams::proof_store: the proof channel is disabled - no proof may enter a "
          "solved store until the generator has passed the differential-testing gate (register "
          "H23; AGENTS.md, \"The proof channel\")");
    }
    // Every real parameter finite (NaN and infinity poison selection), the
    // offending one named.
    const auto require = [](bool ok, const char* name, const char* range) {
      if (!ok) throw std::invalid_argument(std::string("SearchParams::") + name + ": " + range);
    };
    const auto nonneg = [](double v) { return std::isfinite(v) && v >= 0; };
    const auto positive = [](double v) { return std::isfinite(v) && v > 0; };
    const auto unit = [](double v) { return v >= 0 && v <= 1; };  // NaN fails
    require(nonneg(p.cpuct), "cpuct", "finite, >= 0");
    require(nonneg(p.widening_alpha), "widening_alpha", "finite, >= 0");
    require(nonneg(p.widening_scale), "widening_scale", "finite, >= 0");
    require(std::isfinite(p.widening_min) && p.widening_min >= 1.0, "widening_min", "finite, >= 1");
    require(nonneg(p.fpu_reduction), "fpu_reduction", "finite, >= 0");
    require(positive(p.dirichlet_alpha), "dirichlet_alpha", "finite, > 0");
    require(unit(p.dirichlet_epsilon), "dirichlet_epsilon", "0 to 1");
    require(positive(p.policy_temperature), "policy_temperature", "finite, > 0");
    require(nonneg(p.dirichlet_concentration), "dirichlet_concentration", "finite, >= 0 (0: dirichlet_alpha per edge)");
    require(unit(p.dirichlet_shaping_share), "dirichlet_shaping_share", "0 to 1");
    require(p.dirichlet_shaping_cap > 0 && p.dirichlet_shaping_cap <= 1, "dirichlet_shaping_cap", "in (0, 1]");
    require(positive(p.root_policy_temperature), "root_policy_temperature", "finite, > 0");
    require(nonneg(p.cpuct_variance_scale), "cpuct_variance_scale", "finite, >= 0");
    require(positive(p.variance_reference), "variance_reference", "finite, > 0");
    require(nonneg(p.variance_prior_weight), "variance_prior_weight", "finite, >= 0");
    require(positive(p.cpuct_scale_min) && std::isfinite(p.cpuct_scale_max) && p.cpuct_scale_min <= p.cpuct_scale_max,
            "cpuct_scale_min/max", "finite, 0 < min <= max");
    require(std::isfinite(p.variance_bonus_cap) && p.variance_bonus_cap >= 1.0, "variance_bonus_cap", "finite, >= 1");
    require(nonneg(p.variance_bonus_scale), "variance_bonus_scale", "finite, >= 0");
    require(nonneg(p.residual_epsilon), "residual_epsilon", "finite, >= 0");
    require(unit(p.lcb_floor), "lcb_floor", "0 to 1");
    require(nonneg(p.lcb_z), "lcb_z", "finite, >= 0");
    require(nonneg(p.lcb_categorical_weight), "lcb_categorical_weight", "finite, >= 0");
    require(nonneg(p.secure_child_a), "secure_child_a", "finite, >= 0");
    require(positive(p.virtual_loss), "virtual_loss", "finite, > 0");
    require(unit(p.submit_share), "submit_share", "0 to 1");
    require(nonneg(p.unknown_prior_scale), "unknown_prior_scale", "finite, >= 0");
    require(p.turn_cap.turns >= 0 && p.turn_cap.boards >= 0, "turn_cap", "0 (none) or more");
    if (p.key_bits < 1 || p.key_bits > 64) throw std::invalid_argument("SearchParams::key_bits: 1 to 64");
    if (p.expand.pin || p.expand.check || p.expand.travel_threat) {
      throw std::invalid_argument("SearchParams::expand: pin, check and travel-threat are not built");
    }
  }

  // -- nodes ------------------------------------------------------------------

  bool compare_origins() const noexcept { return params_.node_key == NodeKey::StateAndOrigins; }

  // The node key, cut to key_bits bits (a test seam: collisions on purpose).
  std::uint64_t key_of(const GameState& s) const noexcept {
    const std::uint64_t k = node_key(s, params_.node_key);
    return params_.key_bits >= 64 ? k : k & ((std::uint64_t{1} << params_.key_bits) - 1);
  }

  NodeId make_node(GameState state, int moves_made, std::size_t new_boards = 1) {
    Node n(std::move(state));
    n.new_boards = new_boards;
    n.key = key_of(n.state);
    n.mover = n.state.to_move();
    n.moves_made = moves_made;
    n.turn_start = moves_made == 0;
    if (n.turn_start) classify(n);
    nodes_.push_back(std::move(n));
    ++stats_.nodes_created;
    account(nodes_.back());
    return static_cast<NodeId>(nodes_.size() - 1);
  }

  // MCGS identity: the state key AND the moves made in the action in progress
  // (A6). The network sees the count, and no state records it: equal states
  // with different counts are different inputs, so never one node (§15.2: a
  // transposition's evaluation must be the node's). The count subsumes
  // whether an action is in progress - a turn start and a state during an
  // action can hold the same GameState (an optional board played at the end of
  // one action, or at the start of the next), yet only a turn start can end the
  // game (C5) or be capped (I11), and only a partial action can be a dead end.
  // Every ordering of one action's moves from its turn start gives the same
  // count, so the merges inside an action are kept.
  static std::uint64_t identity(std::uint64_t key, int moves_made) noexcept {
    if (moves_made == 0) return detail::combine(key, 0x7475726E5F737461ull);
    return detail::combine(key, detail::mix64(0x6D6F7665735F6D61ull + static_cast<std::uint64_t>(moves_made)));
  }

  void register_node(NodeId id) {
    if (params_.graph) table_.emplace(identity(nodes_[id].key, nodes_[id].moves_made), id);
  }

  // The node of the graph holding (state, moves made in its action), compared
  // exactly; kNoNode.
  NodeId find_node(std::uint64_t key, const GameState& s, int moves_made) const {
    const auto it = table_.find(identity(key, moves_made));
    if (it == table_.end()) return kNoNode;
    const Node& n = nodes_[it->second];
    return n.moves_made == moves_made && detail::same_state(n.state, s, compare_origins()) ? it->second : kNoNode;
  }

  // A turn start: the game's end, by E4's exhaustive detector only (H37), or
  // the turn cap's convention (I11).
  void classify(Node& n) {
    ++stats_.mate_checks;
    SearchBudget budget;
    budget.max_nodes = params_.mate_nodes;
    budget.max_bytes = params_.mate_bytes;
    const auto t0 = std::chrono::steady_clock::now();
    const MateResult r = detect_mate(n.state, params_.detector, budget);
    stats_.seconds_mate += detail::seconds_since(t0);
    switch (r.verdict) {
      case Outcome::Checkmate:
        n.terminal = Terminal::Checkmate;
        n.lo = n.hi = -1;
        ++stats_.checkmates;
        return;
      case Outcome::Stalemate:
        n.terminal = Terminal::Stalemate;
        n.lo = n.hi = 0;
        ++stats_.stalemates;
        return;
      case Outcome::Unknown: ++stats_.unknown_verdicts; break;  // never a terminal
      case Outcome::Ongoing: break;
    }
    if (params_.turn_cap.reached(n.state)) {
      n.terminal = Terminal::TurnCap;  // a convention: value 0, bounds left open
      ++stats_.turn_caps;
    }
  }

  bool solved(const Node& n) const noexcept { return params_.proof_channel && n.lo == n.hi; }

  double node_q(const Node& n) const noexcept {
    if (n.terminal != Terminal::None) return detail::terminal_value(n.terminal);
    if (solved(n)) return n.lo;
    if (n.w > 0) return n.wq / n.w;
    return n.has_eval ? n.eval_mean : 0.0;
  }

  // D3: the node's draw probability, as node_q its value (a solved node's: 1
  // for a proven draw, 0 for a proven result).
  double node_d(const Node& n) const noexcept {
    if (n.terminal != Terminal::None) return detail::terminal_draw(n.terminal);
    if (solved(n)) return n.lo == 0 ? 1.0 : 0.0;
    if (n.w > 0) return n.wd / n.w;
    return n.has_eval ? n.eval_draw : 0.0;
  }

  bool root_finished() const noexcept {
    const Node& r = nodes_[root_];
    return r.terminal != Terminal::None || solved(r) || r.stuck || r.dead_end;
  }

  // The state an edge leads to, and the moves made in its action (A6, §15.4's
  // turn structure): 0 - the opponent's turn start - after a submit edge or a
  // whole action; the parent's count + 1 after a move (D8: nothing is submitted
  // automatically). `illegal` is set for an edge whose action can no longer be
  // legal (kept for a caller's checks; no edge made by the search is such).
  GameState child_state(NodeId pid, std::size_t ei, int& moves_made, bool& illegal) const {
    const Node& p = nodes_[pid];
    const Edge& e = p.edges[ei];
    GameState s = p.state;
    illegal = false;
    if (params_.split == TurnSplit::WholeTurn) {
      for (const Move& m : p.action(e.action)) s.apply(m);
      if (s.submission_status() != SubmitStatus::Safe) {
        throw std::logic_error("search: a listed legal action cannot be submitted");
      }
      s.submit();
      moves_made = 0;
      return s;
    }
    if (e.submit) {
      if (s.submission_status() != SubmitStatus::Safe) {
        illegal = true;  // C4 fails: never followed (the edge is pruned)
        moves_made = p.moves_made;
        return s;
      }
      s.submit();
      moves_made = 0;
      return s;
    }
    s.apply(e.move);
    moves_made = p.moves_made + 1;
    return s;
  }

  // The child node of edge (pid, ei), created (or, MCGS, found) on demand;
  // kNoNode for a dead end (the edge is pruned).
  NodeId ensure_child(NodeId pid, std::size_t ei) {
    {
      const Edge& e = nodes_[pid].edges[ei];
      if (e.pruned) return kNoNode;
      if (e.child != kNoNode) {
        if (nodes_[e.child].dead_end) {
          prune(pid, ei);
          return kNoNode;
        }
        return e.child;
      }
    }
    int moves_made = 0;
    bool illegal = false;
    GameState s = child_state(pid, ei, moves_made, illegal);
    if (illegal) {
      prune(pid, ei);
      ++stats_.dead_ends;
      return kNoNode;
    }
    NodeId id = kNoNode;
    const std::size_t made = s.multiverse().board_count() - nodes_[pid].state.multiverse().board_count();
    if (params_.graph) {
      const std::uint64_t key = key_of(s);
      const NodeId found = find_node(key, s, moves_made);
      if (found != kNoNode) {
        id = found;
        ++stats_.transposition_hits;
      } else if (table_.count(identity(key, moves_made)) != 0) {
        ++stats_.key_collisions;  // kept apart: an unshared node
        id = make_node(std::move(s), moves_made, made);
      } else {
        id = make_node(std::move(s), moves_made, made);
        register_node(id);
      }
    } else {
      id = make_node(std::move(s), moves_made, made);
    }
    nodes_[pid].edges[ei].child = id;
    if (nodes_[id].dead_end) {
      prune(pid, ei);
      return kNoNode;
    }
    return id;
  }

  // -- the leaves' expansion: THE ONE CALL SITE OF THE LEGAL-MOVES FILTER -----------

  // The network's input for a leaf, and (board-move split) its children. Under
  // the board-move split the moves are the LEGAL moves - E5a's tiered
  // classification through ExpandOptions::legal_moves (t5d/legal.hpp), under
  // the budget SearchParams::legality: proved-illegal moves dropped, legal and
  // unknown ones kept (Expansion::move_legality says which); under the
  // whole-turn split, the pseudo-legal moves the action prior is read from.
  void leaf_expansion(const Node& n, Expansion& out) {
    const auto t0 = std::chrono::steady_clock::now();
    ExpandOptions o = params_.expand;
    o.legal_moves = params_.split == TurnSplit::BoardMove && params_.legal_moves_only;
    o.legality = params_.legality;
    expand(n.state, o, out, n.moves_made);  // A6: the node's count, from its path
    stats_.seconds_expand += detail::seconds_since(t0);
  }

  // (c) An edge's move classified under `options`: Legal, Illegal or Unknown.
  MoveLegality classify_edge(NodeId pid, std::size_t ei, const LegalityOptions& options) const {
    const Node& p = nodes_[pid];
    const Edge& e = p.edges[ei];
    if (e.submit || params_.split == TurnSplit::WholeTurn) return MoveLegality::Legal;
    std::vector<MoveLegality> out;
    classify_moves(p.state, std::span<const Move>(&e.move, 1), out, options);
    return out.at(0);
  }

  // Removes an edge from selection, renormalising its siblings' priors (at the
  // root the side table's noise-free priors too, and the effort label's most
  // visited edge found again if it was this one).
  void prune(NodeId pid, std::size_t ei) {
    Node& p = nodes_[pid];
    if (p.edges[ei].pruned) return;
    p.edges[ei].pruned = true;
    double sum = 0;
    for (const Edge& e : p.edges) {
      if (!e.pruned) sum += e.prior;
    }
    if (sum > 0) {
      for (Edge& e : p.edges) {
        if (!e.pruned) e.prior /= sum;
      }
    }
    if (pid != root_) return;
    if (root_prior_noise_free_.size() == p.edges.size()) {
      double total = 0;
      for (std::size_t i = 0; i < p.edges.size(); ++i) total += p.edges[i].pruned ? 0.0 : root_prior_noise_free_[i];
      if (total > 0) {
        for (std::size_t i = 0; i < p.edges.size(); ++i) {
          if (!p.edges[i].pruned) root_prior_noise_free_[i] /= total;
        }
      }
    }
    if (best_edge_ == ei) reset_best();
  }

  // D4's effort label: a root edge's visit counted - it becomes the most
  // visited when it has strictly more visits than the incumbent.
  void note_root_visit(std::size_t ei) {
    const Node& r = nodes_[root_];
    if (best_edge_ == kNoEdge || (ei != best_edge_ && r.edges[ei].visits > r.edges[best_edge_].visits)) {
      best_edge_ = ei;
      best_since_ = r.visits;
    }
  }

  // The most visited unpruned root edge now (the lowest index of a tie), as of
  // the root's visits now; none without a visited edge.
  void reset_best() {
    const Node& r = nodes_[root_];
    best_edge_ = kNoEdge;
    best_since_ = r.visits;
    for (std::size_t i = 0; i < r.edges.size(); ++i) {
      const Edge& e = r.edges[i];
      if (e.pruned || e.visits == 0) continue;
      if (best_edge_ == kNoEdge || e.visits > r.edges[best_edge_].visits) best_edge_ = i;
    }
    if (best_edge_ == kNoEdge) best_since_ = 0;
  }

  // (c) Verifies an unknown edge once: Illegal prunes it.
  MoveLegality verify_edge(NodeId pid, std::size_t ei, const LegalityOptions& options) {
    const auto t0 = std::chrono::steady_clock::now();
    const MoveLegality v = classify_edge(pid, ei, options);
    stats_.seconds_verify += detail::seconds_since(t0);
    Edge& e = nodes_[pid].edges[ei];
    e.verified = true;
    switch (v) {
      case MoveLegality::Illegal:
        ++stats_.verified_illegal;
        prune(pid, ei);
        break;
      case MoveLegality::Legal:
        ++stats_.verified_legal;
        e.unknown = false;
        break;
      case MoveLegality::Unknown: ++stats_.verified_unknown; break;
    }
    return v;
  }

  void list_actions(Node& n) {
    SearchBudget budget;
    budget.max_nodes = params_.listing_nodes;
    budget.max_bytes = params_.mate_bytes;
    const auto t0 = std::chrono::steady_clock::now();
    const ActionList list =
        list_legal_actions(n.state, params_.max_actions, params_.detector, budget);
    stats_.seconds_listing += detail::seconds_since(t0);
    n.action_moves.clear();
    n.action_offsets.assign(1, 0);
    for (const LegalAction& a : list.actions) {
      n.action_moves.insert(n.action_moves.end(), a.moves.begin(), a.moves.end());
      n.action_offsets.push_back(static_cast<std::uint32_t>(n.action_moves.size()));
    }
    n.action_moves.shrink_to_fit();
    n.action_offsets.shrink_to_fit();
    n.children_complete = list.stats.exhaustive;
    if (!n.children_complete) ++stats_.truncated_listings;
    account(n);
  }

  Expansion& slot(std::size_t i) {
    while (slots_.size() <= i) slots_.push_back(std::make_unique<Expansion>());
    return *slots_[i];
  }

  // -- expanding a node with its evaluation ----------------------------------------

  // 6m (register I10; build-log spec issue 149): whether E2, E3 or E5's
  // optional term reads a node's uncertainty signal, and whether an
  // evaluation lacks the one uncertainty_source selects.
  bool uncertainty_read() const noexcept {
    return params_.variance_scaled_cpuct || params_.uncertainty_weighted_backup || params_.variance_bonus ||
           (params_.final_rule == FinalRule::Lcb && params_.lcb_categorical_weight > 0);
  }
  bool lacks_uncertainty(const Evaluation& ev) const noexcept {
    return params_.uncertainty_source == UncertaintySource::ErrorHead ? std::isnan(ev.error) : !ev.value;
  }

  void check_evaluation(const Evaluation& ev, std::size_t moves) const {
    if (ev.logits.size() != moves) {
      throw std::invalid_argument("an evaluation has " + std::to_string(ev.logits.size()) +
                                  " logits for " + std::to_string(moves) + " candidate moves");
    }
    for (const float l : ev.logits) {
      if (!std::isfinite(l)) throw std::invalid_argument("an evaluation holds a non-finite logit");
    }
    if (std::isinf(ev.submit_logit)) {
      throw std::invalid_argument("an evaluation holds an infinite submit logit");
    }
    (void)wdl_value(ev.wdl);
    if (ev.value) (void)value_moments(*ev.value);
    if (!std::isnan(ev.error) && !(std::isfinite(ev.error) && ev.error >= 0)) {
      throw std::invalid_argument("an evaluation's error head output is negative or infinite (NaN: none given)");
    }
    if (!std::isnan(ev.moves_left) && !(std::isfinite(ev.moves_left) && ev.moves_left >= 0)) {
      throw std::invalid_argument("an evaluation's moves-left is negative or infinite (NaN: none given)");
    }
    // 6m: a signal that is read must be given - refused, not read as 0 (a
    // missing signal taken as no variance would halve E2's cPUCT and give the
    // backup its largest weight: a 6m arm run silently wrong).
    if (uncertainty_read() && lacks_uncertainty(ev)) {
      throw std::invalid_argument(
          params_.uncertainty_source == UncertaintySource::ErrorHead
              ? "an evaluation has no error head output (NaN), the uncertainty signal uncertainty_source selects, "
                "while E2, E3 or E5's term reads it"
              : "an evaluation has no categorical, the uncertainty signal uncertainty_source selects, "
                "while E2, E3 or E5's term reads it");
    }
  }

  void expand_node(NodeId id, const Evaluation& ev, const Expansion& x) {
    Node& n = nodes_[id];
    const MoveList& moves = x.moves();
    check_evaluation(ev, moves.size());
    // D3: the value search uses is the WDL's; the uncertainty signal is the
    // categorical's variance or the error head's output (6m) - 0 without one,
    // which only a search that reads no signal accepts (check_evaluation), and
    // counted.
    const WdlValue v = wdl_value(ev.wdl);
    n.eval_mean = v.q;
    n.eval_draw = v.d;
    n.eval_error = ev.error;
    n.eval_moves_left = ev.moves_left;
    if (params_.uncertainty_source == UncertaintySource::ErrorHead) {
      n.eval_var = std::isnan(ev.error) ? 0.0 : static_cast<double>(ev.error);
    } else {
      n.eval_var = ev.value ? value_moments(*ev.value).variance : 0.0;
    }
    if (lacks_uncertainty(ev)) ++stats_.uncertainty_missing;
    n.has_eval = true;
    n.edges.clear();
    std::vector<double> logits;
    if (params_.split == TurnSplit::BoardMove) {
      n.edges.reserve(moves.size() + (n.submit_possible ? std::size_t{1} : std::size_t{0}));
      const std::span<const MoveLegality> legality = x.move_legality();
      for (std::size_t i = 0; i < moves.size(); ++i) {
        Edge e;
        e.move = moves[i];
        // (a) Kept unknown moves are flagged (proved-illegal ones were dropped).
        e.unknown = i < legality.size() && legality[i] == MoveLegality::Unknown;
        if (e.unknown) ++stats_.unknown_edges;
        n.edges.push_back(e);
        logits.push_back(static_cast<double>(ev.logits[i]));
      }
      // D8: submitting is an action wherever it is possible (T5), legal only
      // where C4 holds - an illegal submit is kept, pruned, with no prior.
      if (n.submit_possible) {
        Edge e;
        e.submit = true;
        e.pruned = !n.submit_legal;
        n.edges.push_back(e);
        ++stats_.submit_edges;
        if (e.pruned) ++stats_.illegal_submits;
        // The evaluator's submit logit; none given (NaN): submit_share, below.
        logits.push_back(std::isnan(ev.submit_logit) ? 0.0 : static_cast<double>(ev.submit_logit));
      }
    } else {
      // The action prior from the moves' logits (ActionPrior). A move of an
      // action is matched to the candidate with its source, destination and
      // promotion: a J6 branch onto a board played earlier in the action has
      // the coordinates of the non-branching jump generated at the turn start.
      using Coord = std::tuple<int, int, int, int, int, int, int>;
      std::map<Coord, std::size_t> index;
      const auto coord = [](const Move& mv) {
        return Coord{mv.from_timeline, mv.from_half_turn, mv.from_square, mv.to_timeline,
                     mv.to_half_turn, mv.to_square, mv.promotes() ? index_of(mv.promotion) + 1 : 0};
      };
      double lowest = 0;
      for (std::size_t i = 0; i < moves.size(); ++i) {
        index.emplace(coord(moves[i]), i);
        lowest = i == 0 ? static_cast<double>(ev.logits[i])
                        : std::min(lowest, static_cast<double>(ev.logits[i]));
      }
      const std::size_t count = n.action_count();
      n.edges.reserve(count);
      for (std::size_t a = 0; a < count; ++a) {
        double sum = 0;
        const std::span<const Move> action = n.action(a);
        for (const Move& mv : action) {
          const auto it = index.find(coord(mv));
          sum += it != index.end() ? static_cast<double>(ev.logits[it->second]) : lowest;
        }
        const double logit = params_.action_prior == ActionPrior::SumLogits || action.empty()
                                 ? sum
                                 : sum / static_cast<double>(action.size());
        Edge e;
        e.action = static_cast<std::uint32_t>(a);
        n.edges.push_back(e);
        logits.push_back(logit);
      }
    }
    // The prior: a softmax at the policy temperature over the edges that may
    // be taken (an illegal submit gets none); its entropy sets k0. A legal
    // submit without the evaluator's logit takes submit_share and leaves the
    // rest to the moves (D8's logit is the network's; this is the placeholder).
    const bool shared_submit = params_.split == TurnSplit::BoardMove && n.submit_legal &&
                               std::isnan(ev.submit_logit) && !n.edges.empty() &&
                               n.edges.back().submit;
    const auto in_softmax = [&](std::size_t i) {
      return !n.edges[i].pruned && !(shared_submit && n.edges[i].submit);
    };
    double entropy = 0;
    std::size_t open = 0;
    for (std::size_t i = 0; i < n.edges.size(); ++i) open += in_softmax(i) ? 1u : 0u;
    if (open > 0) {
      // I19: the root's own expansion takes the root policy temperature too,
      // before the noise (§8.4); every node takes policy_temperature.
      const double t = id == root_ ? params_.policy_temperature * params_.root_policy_temperature
                                   : params_.policy_temperature;
      double top = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < n.edges.size(); ++i) {
        if (in_softmax(i)) top = std::max(top, logits[i] / t);
      }
      double sum = 0;
      for (std::size_t i = 0; i < n.edges.size(); ++i) {
        if (!in_softmax(i)) {
          n.edges[i].prior = 0;
          continue;
        }
        n.edges[i].prior = std::exp(logits[i] / t - top);
        // (b) An unknown move's prior, scaled (1: its own prior).
        if (n.edges[i].unknown) n.edges[i].prior *= params_.unknown_prior_scale;
        sum += n.edges[i].prior;
      }
      for (std::size_t i = 0; i < n.edges.size(); ++i) {
        if (!in_softmax(i)) continue;
        Edge& e = n.edges[i];
        e.prior = sum > 0 ? e.prior / sum : 1.0 / static_cast<double>(open);
      }
    }
    if (shared_submit) {
      const double share = open > 0 ? params_.submit_share : 1.0;
      for (Edge& e : n.edges) {
        if (e.submit) {
          e.prior = share;
        } else if (!e.pruned) {
          e.prior *= 1.0 - share;
        }
      }
    }
    for (const Edge& e : n.edges) {
      if (e.prior > 0) entropy -= e.prior * std::log(e.prior);
    }
    n.k0 = std::max(params_.widening_min, params_.widening_scale * std::exp(entropy));
    if (id == root_) {
      std::vector<double> before;
      before.reserve(n.edges.size());
      for (const Edge& e : n.edges) before.push_back(e.prior);
      root_noise_alpha_.clear();
      if (params_.root_noise && !noise_applied_ && n.edges.size() > 1) {
        apply_noise(n);
        noise_applied_ = true;
      }
      // The edges sorted as below, the side table with them (a stable sort of
      // the indices orders them exactly as one of the edges).
      std::vector<std::size_t> order(n.edges.size());
      for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(),
                       [&n](std::size_t a, std::size_t b) { return n.edges[a].prior > n.edges[b].prior; });
      std::vector<Edge> sorted;
      sorted.reserve(n.edges.size());
      root_prior_noise_free_.clear();
      std::vector<double> alphas;
      for (const std::size_t i : order) {
        sorted.push_back(n.edges[i]);
        root_prior_noise_free_.push_back(before[i]);
        if (root_noise_alpha_.size() == order.size()) alphas.push_back(root_noise_alpha_[i]);
      }
      root_noise_alpha_ = std::move(alphas);
      n.edges = std::move(sorted);
      best_edge_ = kNoEdge;
      best_since_ = 0;
    } else {
      std::stable_sort(n.edges.begin(), n.edges.end(),
                       [](const Edge& a, const Edge& b) { return a.prior > b.prior; });
    }
    n.expanded = true;
    n.pending = false;
    n.instantiated = 0;
    account(n);
    widen(n);
    if (n.edges.empty()) {
      n.stuck = true;
      ++stats_.stuck;
    }
  }

  // Root Dirichlet noise (§8.4, self-play): P = (1 - eps) P + eps Dir(alpha),
  // alpha_i from dirichlet_alphas over the open edges (E1, I19: the total
  // concentration split across them). A pruned edge draws a gamma variate too,
  // unused (the open edges' mean alpha; dirichlet_alpha in the arm before
  // CHECKPOINT 7, whose draws are so exactly what they were).
  void apply_noise(Node& n) {
    std::vector<double> open_priors;
    for (const Edge& e : n.edges) {
      if (!e.pruned) open_priors.push_back(e.prior);
    }
    if (open_priors.empty()) return;
    const std::vector<double> alphas = dirichlet_alphas(open_priors, params_);
    double total = 0;
    for (const double a : alphas) total += a;
    // The arm before CHECKPOINT 7 draws dirichlet_alpha for every edge, exactly.
    const double pruned_alpha = params_.dirichlet_concentration > 0 || params_.dirichlet_shaping
                                    ? total / static_cast<double>(alphas.size())
                                    : params_.dirichlet_alpha;
    detail::SplitMix rng{params_.seed ^ detail::mix64(n.key)};
    std::vector<double> eta(n.edges.size());
    std::vector<double> drawn(n.edges.size());
    double sum = 0;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n.edges.size(); ++i) {
      const double alpha = n.edges[i].pruned ? pruned_alpha : alphas[k++];
      drawn[i] = alpha;
      eta[i] = detail::gamma_sample(alpha, rng);
      sum += eta[i];
    }
    root_noise_alpha_ = std::move(drawn);
    if (!(sum > 0)) return;
    double open = 0;  // the noise goes to the edges that may be taken
    for (std::size_t i = 0; i < n.edges.size(); ++i) open += n.edges[i].pruned ? 0.0 : eta[i];
    if (!(open > 0)) return;
    const double eps = params_.dirichlet_epsilon;
    for (std::size_t i = 0; i < n.edges.size(); ++i) {
      if (n.edges[i].pruned) continue;
      n.edges[i].prior = (1 - eps) * n.edges[i].prior + eps * eta[i] / open;
    }
  }

  // Progressive widening: k(N) = ceil(k0 (N + 1)^alpha) children instantiated.
  void widen(Node& n) const {
    if (!params_.widening) {
      n.instantiated = n.edges.size();
      return;
    }
    const double k = std::ceil(n.k0 * std::pow(static_cast<double>(n.visits) + 1.0, params_.widening_alpha));
    const double capped = std::min(k, static_cast<double>(n.edges.size()));
    n.instantiated = std::max(n.instantiated, static_cast<std::size_t>(capped));
  }

  // -- selection -------------------------------------------------------------------

  // A variance blended from the categorical prior (weight w0) and the
  // empirical variance of n backed-up values.
  double blended_variance(double prior_var, double w, double wq, double wq2,
                          std::uint32_t n) const noexcept {
    double emp = 0;
    if (w > 0) {
      const double mean = wq / w;
      emp = std::max(0.0, wq2 / w - mean * mean);
    }
    const double w0 = params_.variance_prior_weight;
    const double total = w0 + static_cast<double>(n);
    if (!(total > 0)) return 0;
    return (w0 * prior_var + static_cast<double>(n) * emp) / total;
  }

  double cpuct_scale(const Node& p) const noexcept {
    if (!params_.variance_scaled_cpuct) return 1.0;
    const double var = blended_variance(p.eval_var, p.w, p.wq, p.wq2, p.visits);
    const double s = 1.0 + params_.cpuct_variance_scale *
                               (std::sqrt(var) / params_.variance_reference - 1.0);
    return std::clamp(s, params_.cpuct_scale_min, params_.cpuct_scale_max);
  }

  double variance_bonus(const Node& child, const Edge& e) const noexcept {
    const double var = blended_variance(child.eval_var, e.w, e.wq, e.wq2, e.visits);
    const double f = 1.0 + params_.variance_bonus_scale *
                               (std::sqrt(var) / params_.variance_reference - 1.0);
    return std::clamp(f, 1.0, params_.variance_bonus_cap);
  }

  double backup_weight(double variance) const noexcept {
    if (!params_.uncertainty_weighted_backup) return 1.0;
    const double r2 = params_.variance_reference * params_.variance_reference;
    return r2 / (r2 + std::max(variance, 0.0));
  }

  // PUCT over the instantiated children. Three classes, the first non-empty
  // wins: ordinary children; solved children that have reached the visit cap;
  // children proven lost. Widens by one when no ordinary child is left.
  std::size_t select_child(NodeId pid) {
    for (;;) {
      // A child found to be a dead end since: pruned first (its siblings'
      // priors renormalised), then the scores.
      {
        const Node& q = nodes_[pid];
        for (std::size_t i = 0; i < q.edges.size(); ++i) {
          const Edge& e = q.edges[i];
          if (!e.pruned && e.child != kNoNode && nodes_[e.child].dead_end) prune(pid, i);
        }
      }
      Node& p = nodes_[pid];
      widen(p);
      const double vlw_parent = static_cast<double>(p.vl) * params_.virtual_loss;
      const double sqrt_n = std::sqrt(std::max(static_cast<double>(p.visits) + vlw_parent, 1.0));
      const double c = params_.cpuct * cpuct_scale(p);
      double visited_mass = 0;
      for (std::size_t i = 0; i < p.instantiated; ++i) {
        if (p.edges[i].visits + p.edges[i].vl > 0) visited_mass += p.edges[i].prior;
      }
      const double fpu = node_q(p) - params_.fpu_reduction * std::sqrt(visited_mass);
      std::array<std::size_t, 3> best{kNoEdge, kNoEdge, kNoEdge};
      std::array<double, 3> best_score{};
      for (std::size_t i = 0; i < p.instantiated; ++i) {
        const Edge& e = p.edges[i];
        if (e.pruned) continue;
        const Node* child = e.child == kNoNode ? nullptr : &nodes_[e.child];
        const double vlw = static_cast<double>(e.vl) * params_.virtual_loss;
        std::size_t cls = 0;
        double q = 0;
        if (child != nullptr && solved(*child)) {
          q = detail::sign_between(p, *child) * child->lo;
          if (q <= -1.0) {
            cls = 2;
          } else if (e.visits >= params_.solved_visit_cap) {
            cls = 1;
          }
        } else {
          q = e.w + vlw > 0 ? (e.wq - vlw) / (e.w + vlw) : fpu;
          if (child != nullptr && params_.proof_channel) {
            const auto [lo, hi] = detail::bounds_in_parent(p, *child);
            q = std::clamp(q, static_cast<double>(lo), static_cast<double>(hi));
          }
        }
        double u = c * e.prior * sqrt_n / (1.0 + static_cast<double>(e.visits) + vlw);
        if (params_.variance_bonus && child != nullptr && child->has_eval) {
          u *= variance_bonus(*child, e);
        }
        const double score = q + u;
        if (best[cls] == kNoEdge || score > best_score[cls]) {
          best[cls] = i;
          best_score[cls] = score;
        }
      }
      if (best[0] != kNoEdge) return best[0];
      if (p.instantiated < p.edges.size()) {
        ++p.instantiated;  // no ordinary child left: widen (the solver needs every child)
        continue;
      }
      if (best[1] != kNoEdge) return best[1];
      return best[2];
    }
  }

  // MCGS: whether to back up a correction through edge (pid, ei) into the
  // existing node cid instead of descending (Czech et al.: the edge's Q lags
  // the node's by more than epsilon; an edge never crossed takes the node's
  // value). D3: the draw probability is part of the value, so a D that lags by
  // more than epsilon is corrected the same way (the one epsilon for both).
  bool needs_residual(NodeId pid, std::size_t ei, NodeId cid) const noexcept {
    const Node& c = nodes_[cid];
    if (!c.expanded || c.terminal != Terminal::None || solved(c) || c.stuck || c.visits == 0) {
      return false;
    }
    const Edge& e = nodes_[pid].edges[ei];
    const double vc = detail::sign_between(nodes_[pid], c) * node_q(c);
    if (e.visits == 0 || !(e.w > 0)) return true;
    return std::abs(e.wq / e.w - vc) > params_.residual_epsilon ||
           std::abs(e.wd / e.w - node_d(c)) > params_.residual_epsilon;
  }

  // -- backup ------------------------------------------------------------------------

  void release(const std::vector<Step>& path) noexcept {
    for (std::size_t i = 0; i < path.size(); ++i) {
      Node& n = nodes_[path[i].node];
      if (n.vl > 0) --n.vl;
      if (i > 0) {
        Edge& e = nodes_[path[i - 1].node].edges[path[i].edge];
        if (e.vl > 0) --e.vl;
      }
    }
  }

  // Backs up `v` (the last node's view) and the draw probability `d` (the
  // same from either side, D3) along `path`, releasing its virtual loss; the
  // visit ends at the last node if `ends_here`.
  void backup(const std::vector<Step>& path, double v, double d, double w, bool ends_here) {
    for (std::size_t i = path.size(); i-- > 0;) {
      Node& n = nodes_[path[i].node];
      n.visits += 1;
      n.w += w;
      n.wq += w * v;
      n.wq2 += w * v * v;
      n.wd += w * d;
      if (i + 1 == path.size() && ends_here) {
        n.ends += 1;
        n.ends_w += w;
        n.ends_wq += w * v;
        n.ends_wd += w * d;
      }
      if (n.vl > 0) --n.vl;
      if (i > 0) {
        Node& parent = nodes_[path[i - 1].node];
        Edge& e = parent.edges[path[i].edge];
        if (parent.mover != n.mover) v = -v;
        e.visits += 1;
        e.w += w;
        e.wq += w * v;
        e.wq2 += w * v * v;
        e.wd += w * d;
        if (e.vl > 0) --e.vl;
      }
    }
    if (params_.proof_channel) {
      for (std::size_t i = path.size(); i-- > 0;) update_bounds(path[i].node);
    }
    if (path.size() >= 2 && path[0].node == root_) note_root_visit(path[1].edge);
    ++stats_.playouts;
  }

  void complete_residual(const std::vector<Step>& path, std::size_t ei, NodeId cid) {
    const NodeId pid = path.back().node;
    const double vc = detail::sign_between(nodes_[pid], nodes_[cid]) * node_q(nodes_[cid]);
    const double dc = node_d(nodes_[cid]);
    Edge& e = nodes_[pid].edges[ei];
    const double r = std::clamp(vc * (e.w + 1.0) - e.wq, -1.0, 1.0);
    const double rd = std::clamp(dc * (e.w + 1.0) - e.wd, 0.0, 1.0);  // D3: the draw's correction
    e.visits += 1;
    e.w += 1.0;
    e.wq += r;
    e.wq2 += r * r;
    e.wd += rd;
    ++stats_.residual_backups;
    backup(path, r, rd, 1.0, false);
    if (pid == root_) note_root_visit(ei);  // the edge crossed is the root's own
  }

  // MCTS-Solver with draws: lo = the best child's lower bound; hi = the best
  // child's upper bound once every legal child exists and the list is complete.
  // Bounds only tighten; a contradiction is a bug.
  void update_bounds(NodeId id) {
    Node& p = nodes_[id];
    if (!p.expanded || p.terminal != Terminal::None || p.stuck || p.dead_end) return;
    int lo = -1;
    int hi = -1;
    bool all_known = p.children_complete;
    bool any = false;
    for (const Edge& e : p.edges) {
      if (e.pruned) continue;
      if (e.child == kNoNode) {
        all_known = false;
        continue;
      }
      const Node& c = nodes_[e.child];
      if (c.dead_end) continue;
      any = true;
      const auto [clo, chi] = detail::bounds_in_parent(p, c);
      // (e) A child through an unknown edge (its move may be illegal) never
      // proves its parent anything; as part of a superset of the legal
      // children it still bounds the parent from above.
      if (!e.unknown) lo = std::max(lo, clo);
      hi = std::max(hi, chi);
    }
    if (!any) return;
    if (!all_known) hi = 1;
    const int new_lo = std::max<int>(p.lo, lo);
    const int new_hi = std::min<int>(p.hi, hi);
    if (new_lo > new_hi) {
      throw std::logic_error("search: solver bounds contradict at a node (" +
                             std::to_string(new_lo) + " > " + std::to_string(new_hi) + ")");
    }
    p.lo = static_cast<std::int8_t>(new_lo);
    p.hi = static_cast<std::int8_t>(new_hi);
  }

  void mark_dead_end(const std::vector<Step>& path) {
    const NodeId id = path.back().node;
    nodes_[id].dead_end = true;
    ++stats_.dead_ends;
    if (path.size() >= 2) prune(path[path.size() - 2].node, path.back().edge);
  }

  // -- one simulation's selection -------------------------------------------------------

  Selected select_once() {
    path_.clear();
    path_.push_back(Step{root_, kNoEdge});
    nodes_[root_].vl += 1;
    NodeId cur = root_;
    for (;;) {
      Node& n = nodes_[cur];
      if (n.terminal != Terminal::None) {
        ++stats_.terminal_visits;
        backup(path_, detail::terminal_value(n.terminal), detail::terminal_draw(n.terminal), 1.0, true);
        return Selected::Done;
      }
      if (solved(n)) {
        ++stats_.terminal_visits;
        backup(path_, n.lo, n.lo == 0 ? 1.0 : 0.0, 1.0, true);
        return Selected::Done;
      }
      if (n.stuck) {
        backup(path_, n.eval_mean, n.eval_draw, backup_weight(n.eval_var), true);
        return Selected::Done;
      }
      if (!n.expanded) {
        if (n.pending) {
          ++stats_.collisions;
          collided_.push_back(path_);  // its virtual loss stays until the batch ends
          return Selected::Collision;
        }
        return leaf(cur);
      }
      const std::size_t ei = select_child(cur);
      if (ei == kNoEdge) {
        // No child left: every one is a dead end.
        if (!n.turn_start && cur != root_) {
          mark_dead_end(path_);
          release(path_);
          return Selected::DeadEnd;
        }
        n.stuck = true;
        ++stats_.stuck;
        backup(path_, node_q(n), node_d(n), 1.0, true);
        return Selected::Done;
      }
      // (c) The first time an unknown edge is picked, it is verified under the
      // larger budget. Illegal: the edge goes (its siblings' priors
      // renormalised) and selection is repeated here - nothing backed up.
      if (params_.verify_unknown) {
        const Edge& e = nodes_[cur].edges[ei];
        if (e.unknown && !e.verified &&
            verify_edge(cur, ei, params_.verify_legality) == MoveLegality::Illegal) {
          continue;
        }
      }
      const NodeId child = ensure_child(cur, ei);
      if (child == kNoNode) {
        release(path_);
        return Selected::DeadEnd;
      }
      if (params_.graph && needs_residual(cur, ei, child)) {
        complete_residual(path_, ei, child);
        return Selected::Done;
      }
      path_.push_back(Step{child, ei});  // first: a failed push_back adds no virtual loss
      nodes_[cur].edges[ei].vl += 1;
      nodes_[child].vl += 1;
      cur = child;
    }
  }

  // A leaf to evaluate: its expansion now (the children it will have); an
  // evaluation from the cache completes the visit at once.
  Selected leaf(NodeId id) {
    Expansion& x = slot(pending_.size());
    {
      Node& n = nodes_[id];
      leaf_expansion(n, x);
      if (params_.split == TurnSplit::WholeTurn) {
        if (n.action_offsets.empty()) list_actions(n);
      } else {
        // D8: submitting is possible once the obligations are met (T5) and
        // legal when C4 holds - the expansion's own answers (exact).
        n.submit_possible = x.submit_possible();
        n.submit_legal = x.submit_legal();
      }
    }
    Node& n = nodes_[id];
    // The children that may be taken: the kept moves, and a legal submit.
    const std::size_t children = params_.split == TurnSplit::WholeTurn
                                     ? n.action_count()
                                     : x.moves().size() + (n.submit_legal ? std::size_t{1} : std::size_t{0});
    if (children == 0 && !n.turn_start && id != root_) {
      // A partial action whose remaining moves and submission are all illegal:
      // no legal completion, so no game end either (C5 judges turn starts) - a
      // dead end, pruned (§8.2).
      mark_dead_end(path_);
      release(path_);
      return Selected::DeadEnd;
    }
    // The cache and the batch match the whole network input: the state and the
    // moves made in its action (A6), at the epoch.
    if (cache_) {
      if (const Evaluation* ev = cache_->find(n.key, n.moves_made, epoch_, n.state, compare_origins())) {
        const Evaluation copy = *ev;
        expand_node(id, copy, x);
        ++stats_.cache_hits;
        const Node& done = nodes_[id];
        backup(path_, done.eval_mean, done.eval_draw, backup_weight(done.eval_var), true);
        return Selected::Done;
      }
    }
    // One evaluation per distinct input in the batch: a tree can hold one
    // state with one count in two nodes (a transposition it does not merge).
    std::size_t request = unique_requests_;
    for (const Pending& other : pending_) {
      const Node& o = nodes_[other.leaf];
      if (o.key == n.key && o.moves_made == n.moves_made && detail::same_state(o.state, n.state, compare_origins())) {
        request = other.request;
        ++stats_.batch_duplicates;
        break;
      }
    }
    pending_.push_back(Pending{path_, pending_.size(), id, request});  // first: it may throw
    if (request == unique_requests_) ++unique_requests_;
    n.pending = true;
    return Selected::Pending;
  }

  std::size_t gather(std::size_t max_leaves, std::uint64_t max_playouts) {
    if (!pending_.empty() || !collided_.empty()) {
      throw std::invalid_argument("search: a batch is waiting for its evaluations");
    }
    if (max_leaves == 0) throw std::invalid_argument("gather: max_leaves must be 1 or more");
    requests_.clear();
    unique_requests_ = 0;
    if (params_.synchronous) max_leaves = 1;
    std::uint64_t done = 0;
    std::size_t collisions = 0;
    std::size_t iterations = 0;
    // Saturating: max_leaves may be anything up to SIZE_MAX.
    const std::size_t iteration_limit =
        max_leaves > (std::numeric_limits<std::size_t>::max() - 1024) / 64 ? std::numeric_limits<std::size_t>::max()
                                                                           : 64 * max_leaves + 1024;
    // A batch is `max_leaves` visits: the leaves sent for evaluation AND the
    // visits completed without the evaluator (terminals, solved nodes, cache
    // hits, MCGS residuals). Counted apart, the latter piled up without bound
    // behind a pending leaf's virtual loss - a whole run's playouts down a line
    // of terminals while the one pending line starved.
    try {
      while (pending_.size() + done < max_leaves && pending_.size() + done < max_playouts &&
             iterations++ < iteration_limit) {
        if (root_finished()) break;
        const Selected s = select_once();
        if (s == Selected::Done) {
          ++done;
        } else if (s == Selected::Collision) {
          if (++collisions >= max_leaves) break;
        }
      }
    } catch (...) {
      // The batch is abandoned and the path being selected released: outside
      // a batch no virtual loss exists, so every count is cleared.
      abandon();
      clear_virtual_loss();
      throw;
    }
    for (const Pending& pd : pending_) {
      if (pd.request != requests_.size()) continue;  // a duplicate: its first leaf asks
      const Node& n = nodes_[pd.leaf];
      requests_.push_back(EvalRequest{&n.state, slots_[pd.slot].get(), n.key, epoch_, n.moves_made});
    }
    return requests_.size();
  }

  void clear_virtual_loss() noexcept {
    for (Node& n : nodes_) {
      n.vl = 0;
      for (Edge& e : n.edges) e.vl = 0;
    }
    path_.clear();
  }

  void abandon() noexcept {
    for (const Pending& pd : pending_) {
      release(pd.path);
      nodes_[pd.leaf].pending = false;
    }
    for (const std::vector<Step>& path : collided_) release(path);
    pending_.clear();
    collided_.clear();
    requests_.clear();
    unique_requests_ = 0;
  }

  void deliver(std::span<const Evaluation> evaluations) {
    if (pending_.empty()) throw std::invalid_argument("deliver: no batch is waiting");
    if (evaluations.size() != requests_.size()) {
      const std::size_t want = requests_.size();
      abandon();
      throw std::invalid_argument("deliver: " + std::to_string(evaluations.size()) +
                                  " evaluations for " + std::to_string(want) + " requests");
    }
    try {
      for (const Pending& pd : pending_) {
        check_evaluation(evaluations[pd.request], slots_[pd.slot]->moves().size());
      }
    } catch (...) {
      abandon();
      throw;
    }
    std::size_t delivered = 0;
    try {
      for (const Pending& pd : pending_) {
        const Evaluation& ev = evaluations[pd.request];
        expand_node(pd.leaf, ev, *slots_[pd.slot]);
        const Node& n = nodes_[pd.leaf];
        backup(pd.path, n.eval_mean, n.eval_draw, backup_weight(n.eval_var), true);
        ++delivered;
        // A duplicate is not stored twice.
        if (cache_) cache_->insert(n.key, n.moves_made, epoch_, n.state, ev, n.new_boards);
      }
    } catch (...) {
      // What was delivered stays; the rest of the batch is abandoned.
      pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(delivered));
      abandon();
      clear_virtual_loss();
      throw;
    }
    stats_.evaluations += requests_.size();
    ++stats_.batches;
    for (const std::vector<Step>& path : collided_) release(path);
    pending_.clear();
    collided_.clear();
    requests_.clear();
    unique_requests_ = 0;
  }

  // -- the root's children ------------------------------------------------------------

  std::vector<std::size_t> root_edges() const {
    std::vector<std::size_t> out;
    const Node& r = nodes_[root_];
    for (std::size_t i = 0; i < r.instantiated; ++i) {
      const Edge& e = r.edges[i];
      if (e.pruned || (e.child != kNoNode && nodes_[e.child].dead_end)) continue;
      out.push_back(i);
    }
    return out;
  }

  double edge_q(const Node& p, const Edge& e) const noexcept {
    if (e.child != kNoNode && solved(nodes_[e.child])) {
      return detail::sign_between(p, nodes_[e.child]) * nodes_[e.child].lo;
    }
    return e.w > 0 ? e.wq / e.w : 0.0;
  }

  // D3: an edge's backed-up draw probability (a solved child's exact one).
  double edge_d(const Edge& e) const noexcept {
    if (e.child != kNoNode && solved(nodes_[e.child])) return nodes_[e.child].lo == 0 ? 1.0 : 0.0;
    return e.w > 0 ? e.wd / e.w : 0.0;
  }

  std::vector<FinalCandidate> root_candidates(const std::vector<std::size_t>& edges) const {
    std::vector<FinalCandidate> out;
    const Node& r = nodes_[root_];
    for (const std::size_t i : edges) {
      const Edge& e = r.edges[i];
      FinalCandidate c;
      c.visits = e.visits;
      c.q = edge_q(r, e);
      if (e.w > 0) {
        const double mean = e.wq / e.w;
        c.variance = std::max(0.0, e.wq2 / e.w - mean * mean);
      }
      if (e.child != kNoNode) {
        const Node& child = nodes_[e.child];
        c.eval_variance = child.has_eval ? child.eval_var : 0.0;
        if (params_.proof_channel) {
          const auto [lo, hi] = detail::bounds_in_parent(r, child);
          c.lo = e.unknown ? -1 : lo;  // (e) no proof through an unknown edge
          c.hi = hi;
        }
      }
      out.push_back(c);
    }
    return out;
  }

  // -- persistence ----------------------------------------------------------------------

  // Keeps the nodes the new root reaches (breadth-first, renumbered), deletes
  // the rest: garbage collection of path-dependent structure (§8.7).
  void reroot(NodeId new_root) {
    if (!pending_.empty() || !collided_.empty()) {
      throw std::invalid_argument("search: cannot re-root while a batch is waiting");
    }
    std::vector<NodeId> order;
    std::vector<NodeId> remap(nodes_.size(), kNoNode);
    order.push_back(new_root);
    remap[new_root] = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      for (const Edge& e : nodes_[order[i]].edges) {
        if (e.child != kNoNode && remap[e.child] == kNoNode) {
          remap[e.child] = static_cast<NodeId>(order.size());
          order.push_back(e.child);
        }
      }
    }
    std::vector<Node> kept;
    kept.reserve(order.size());
    for (const NodeId id : order) kept.push_back(std::move(nodes_[id]));
    for (Node& n : kept) {
      for (Edge& e : n.edges) {
        if (e.child != kNoNode) e.child = remap[e.child];
      }
    }
    stats_.nodes_collected += nodes_.size() - kept.size();
    nodes_ = std::move(kept);
    root_ = 0;
    tree_bytes_ = 0;
    for (const Node& n : nodes_) tree_bytes_ += n.bytes;
    purge_cache();
    table_.clear();
    for (NodeId id = 0; id < nodes_.size(); ++id) register_node(id);
    // Root noise belongs to a fresh root's expansion (self-play); a carried
    // root (match play, where noise is off) gets none - nor a side table, and
    // its effort label starts here (the history above it is not kept).
    noise_applied_ = nodes_[0].expanded;
    root_prior_noise_free_.clear();
    root_noise_alpha_.clear();
    reset_best();
    ++stats_.reroots;
    notify_root();
  }

  void notify_root() {
    ++root_generation_;
    if (hook_) hook_(nodes_[root_].state, epoch_);
  }

  // -- invariants -------------------------------------------------------------------

  void fail(const std::string& what, NodeId id) const {
    throw std::logic_error("search invariant: " + what + " (node " + std::to_string(id) + ")");
  }

  void check_invariants() const {
    if (!pending_.empty() || !collided_.empty()) {
      throw std::logic_error("search invariant: checked while a batch is waiting");
    }
    // Reachability: every node from the root.
    std::vector<std::uint8_t> seen(nodes_.size(), 0);
    std::vector<NodeId> stack{root_};
    seen[root_] = 1;
    std::vector<std::uint32_t> parents(nodes_.size(), 0);
    while (!stack.empty()) {
      const NodeId id = stack.back();
      stack.pop_back();
      for (const Edge& e : nodes_[id].edges) {
        if (e.child == kNoNode) continue;
        if (e.child >= nodes_.size()) fail("an edge points past the nodes", id);
        ++parents[e.child];
        if (!seen[e.child]) {
          seen[e.child] = 1;
          stack.push_back(e.child);
        }
      }
    }
    for (NodeId id = 0; id < nodes_.size(); ++id) {
      if (!seen[id]) fail("a node the root does not reach", id);
      if (!params_.graph && id != root_ && parents[id] != 1) fail("a tree node with " + std::to_string(parents[id]) + " parents", id);
    }
    for (NodeId id = 0; id < nodes_.size(); ++id) {
      const Node& n = nodes_[id];
      if (n.vl != 0) fail("virtual loss not released on a node", id);
      if (n.pending) fail("a node still marked pending", id);
      std::uint64_t through = 0;
      double through_wq = 0;
      double through_w = 0;
      double through_wd = 0;
      for (const Edge& e : n.edges) {
        if (e.vl != 0) fail("virtual loss not released on an edge", id);
        through += e.visits;
        through_w += e.w;
        through_wq += e.wq;
        through_wd += e.wd;
        if (e.visits > 0 && e.child == kNoNode) fail("a visited edge without a child", id);
        if (!params_.graph && e.child != kNoNode && nodes_[e.child].visits != e.visits) {
          fail("a tree edge's visits differ from its child's", id);
        }
        // An action ends only by a submit edge (D8), and a submit edge always ends
        // one: whether a node is a turn start follows from the edge into it alone,
        // in a graph as in a tree. And the moves made in the action (A6) count
        // the move edges since: 0 after a submit edge or a whole action, the
        // parent's + 1 after a move (§15.4's turn structure).
        if (e.child != kNoNode && params_.split == TurnSplit::BoardMove &&
            nodes_[e.child].turn_start != e.submit) {
          fail(e.submit ? "a submit edge into a node during an action" : "a move edge into a turn start", id);
        }
        if (e.child != kNoNode && params_.split == TurnSplit::WholeTurn && !nodes_[e.child].turn_start) {
          fail("a whole-turn action into a node during an action", id);
        }
        if (e.child != kNoNode) {
          const int want = params_.split == TurnSplit::WholeTurn || e.submit ? 0 : n.moves_made + 1;
          if (nodes_[e.child].moves_made != want) {
            const char* edge = params_.split == TurnSplit::WholeTurn ? "action" : e.submit ? "submit" : "move";
            fail("a child with " + std::to_string(nodes_[e.child].moves_made) + " moves made in its action where its " +
                     edge + " edge gives " + std::to_string(want),
                 id);
          }
        }
      }
      if (n.moves_made < 0 || n.turn_start != (n.moves_made == 0)) {
        fail("a turn start's count of moves made is not 0, or a partial action's is", id);
      }
      if (!n.turn_start && n.terminal != Terminal::None) fail("a game end during an action (C5)", id);
      if (n.turn_start && n.dead_end) fail("a turn start marked a dead end", id);
      if (static_cast<std::uint64_t>(n.visits) != n.ends + through) {
        fail("visits " + std::to_string(n.visits) + " != ends " + std::to_string(n.ends) +
                 " + child visits " + std::to_string(through),
             id);
      }
      const double scale = 1.0 + std::abs(n.w);
      if (std::abs(n.w - (n.ends_w + through_w)) > 1e-9 * scale ||
          std::abs(n.wq - (n.ends_wq + through_wq)) > 1e-9 * scale) {
        fail("backed-up values inconsistent with the children's", id);
      }
      if (std::abs(n.wd - (n.ends_wd + through_wd)) > 1e-9 * scale) {
        fail("backed-up draw probabilities inconsistent with the children's (D3)", id);
      }
      if (n.wd < -1e-9 * scale || n.wd > n.w + 1e-9 * scale) {
        fail("a backed-up draw probability outside [0, 1] (D3)", id);
      }
      if (n.lo > n.hi) fail("solver bounds contradict", id);
      if (n.terminal == Terminal::Checkmate && (n.lo != -1 || n.hi != -1)) fail("a checkmate not proven lost", id);
      if (n.terminal == Terminal::Stalemate && (n.lo != 0 || n.hi != 0)) fail("a stalemate not proven drawn", id);
      if (n.terminal == Terminal::TurnCap && n.lo == n.hi) fail("the turn cap's convention taken as a proof", id);
      if (!params_.proof_channel || !n.expanded || n.terminal != Terminal::None) continue;
      // Never contradict the children: the node cannot be proven better than its
      // best child can be, nor worse than its best child is proven to be.
      int best_lo = -1;
      int best_hi = -1;
      bool any = false;
      for (const Edge& e : n.edges) {
        if (e.pruned || e.child == kNoNode || nodes_[e.child].dead_end) continue;
        any = true;
        const auto [lo, hi] = detail::bounds_in_parent(n, nodes_[e.child]);
        best_lo = std::max(best_lo, lo);
        best_hi = std::max(best_hi, hi);
      }
      if (!any) continue;
      bool uncreated = !n.children_complete;
      for (const Edge& e : n.edges) uncreated = uncreated || (!e.pruned && e.child == kNoNode);
      if (n.hi < best_lo) fail("a node proven worse than a child guarantees", id);
      if (!uncreated && n.lo > best_hi) fail("a node proven better than any child can be", id);
    }
  }

  std::uint64_t digest() const {
    std::vector<NodeId> order{root_};
    std::vector<NodeId> remap(nodes_.size(), kNoNode);
    remap[root_] = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      for (const Edge& e : nodes_[order[i]].edges) {
        if (e.child != kNoNode && remap[e.child] == kNoNode) {
          remap[e.child] = static_cast<NodeId>(order.size());
          order.push_back(e.child);
        }
      }
    }
    std::uint64_t h = 0x5D5D5D5D5D5D5D5Dull;
    using detail::bits_of;
    using detail::combine;
    for (const NodeId id : order) {
      const Node& n = nodes_[id];
      h = combine(h, n.key);
      h = combine(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(n.moves_made)));  // A6: identity
      h = combine(h, (static_cast<std::uint64_t>(n.visits) << 32) | n.ends);
      h = combine(h, bits_of(n.w));
      h = combine(h, bits_of(n.wq));
      h = combine(h, bits_of(n.wq2));
      h = combine(h, bits_of(n.wd));
      h = combine(h, bits_of(n.eval_mean));
      h = combine(h, bits_of(n.eval_draw));
      h = combine(h, bits_of(n.eval_var));
      h = combine(h, bits_of(n.k0));
      h = combine(h, static_cast<std::uint64_t>(n.lo + 2) | (static_cast<std::uint64_t>(n.hi + 2) << 8) |
                         (static_cast<std::uint64_t>(n.terminal) << 16) |
                         (static_cast<std::uint64_t>(n.expanded) << 24) |
                         (static_cast<std::uint64_t>(n.stuck) << 25) |
                         (static_cast<std::uint64_t>(n.dead_end) << 26) |
                         (static_cast<std::uint64_t>(n.instantiated) << 32));
      for (const Edge& e : n.edges) {
        h = combine(h, bits_of(e.prior));
        h = combine(h, (static_cast<std::uint64_t>(e.visits) << 32) |
                           (e.child == kNoNode ? std::uint64_t{0xFFFFFFFF} : std::uint64_t{remap[e.child]}));
        h = combine(h, bits_of(e.w));
        h = combine(h, bits_of(e.wq));
        h = combine(h, bits_of(e.wq2));
        h = combine(h, bits_of(e.wd));
        h = combine(h, static_cast<std::uint64_t>(e.pruned) | (static_cast<std::uint64_t>(e.submit) << 1) |
                           (static_cast<std::uint64_t>(e.action) << 8));
      }
    }
    return h;
  }
};

// ---------------------------------------------------------------------------
// Search: the public face
// ---------------------------------------------------------------------------

Search::Search(GameState root, SearchParams params, int moves_made, std::shared_ptr<EvalCache> cache)
    : impl_(std::make_unique<Impl>(std::move(root), params, moves_made, std::move(cache))) {}
Search::~Search() = default;
Search::Search(Search&&) noexcept = default;
Search& Search::operator=(Search&&) noexcept = default;

const SearchParams& Search::params() const noexcept { return impl_->params_; }
const SearchStats& Search::stats() const noexcept { return impl_->stats_; }
const EvalCache* Search::cache() const noexcept { return impl_->cache_.get(); }
std::shared_ptr<EvalCache> Search::shared_cache() const noexcept { return impl_->cache_; }
std::size_t Search::memory_bytes() const noexcept { return impl_->memory_bytes(); }

StopReason Search::run(Evaluator& evaluator, const Limits& limits) {
  Impl& s = *impl_;
  const auto t0 = std::chrono::steady_clock::now();
  const auto stop = [&s](StopReason r) {
    s.stats_.last_stop = r;
    return r;
  };
  if (std::isnan(limits.seconds) || limits.seconds < 0) throw std::invalid_argument("Limits::seconds: 0 (none) or more");
  if (s.notified_generation_ != s.root_generation_) {
    evaluator.on_new_root(s.nodes_[s.root_].state, s.epoch_);
    s.notified_generation_ = s.root_generation_;
  }
  const std::uint64_t p0 = s.stats_.playouts;
  const std::uint64_t e0 = s.stats_.evaluations;
  std::vector<Evaluation> out;
  for (;;) {
    const std::uint64_t played = s.stats_.playouts - p0;
    const std::uint64_t evaluated = s.stats_.evaluations - e0;
    if (played >= limits.playouts) return stop(StopReason::Playouts);
    if (evaluated >= limits.evaluations) return stop(StopReason::Evaluations);
    if (s.root_finished()) return stop(StopReason::Finished);
    if (s.nodes_.size() >= limits.nodes) return stop(StopReason::Nodes);
    if (s.memory_bytes() >= limits.bytes) return stop(StopReason::Memory);
    if (limits.seconds > 0 && detail::seconds_since(t0) >= limits.seconds) return stop(StopReason::Deadline);
    const std::uint64_t batch = std::max<std::size_t>(s.params_.batch_size, 1);
    const auto want = static_cast<std::size_t>(std::min(batch, limits.evaluations - evaluated));
    const std::uint64_t before = s.stats_.playouts;
    const std::size_t n = s.gather(want, limits.playouts - played);
    if (n == 0) {
      if (s.stats_.playouts == before) return stop(StopReason::Exhausted);  // nothing more can be done
      continue;
    }
    out.resize(n);
    for (Evaluation& e : out) {  // reused buffers: every field reset
      e.logits.clear();
      e.submit_logit = std::numeric_limits<float>::quiet_NaN();
      e.wdl.fill(0.0f);
      e.value.reset();
      e.error = std::numeric_limits<float>::quiet_NaN();
      e.moves_left = std::numeric_limits<float>::quiet_NaN();
    }
    try {
      evaluator.evaluate(s.requests_, out);
    } catch (...) {
      s.abandon();
      throw;
    }
    s.deliver(out);
  }
}

std::size_t Search::gather(std::size_t max_leaves, std::uint64_t max_playouts) {
  return impl_->gather(max_leaves, max_playouts);
}
std::span<const EvalRequest> Search::requests() const noexcept { return impl_->requests_; }
void Search::deliver(std::span<const Evaluation> evaluations) { impl_->deliver(evaluations); }
void Search::abandon() noexcept { impl_->abandon(); }

NodeId Search::root() const noexcept { return impl_->root_; }
const GameState& Search::root_state() const noexcept { return impl_->nodes_[impl_->root_].state; }
int Search::root_moves_made() const noexcept { return impl_->nodes_[impl_->root_].moves_made; }
bool Search::root_turn_start() const noexcept { return impl_->nodes_[impl_->root_].turn_start; }
bool Search::root_terminal() const noexcept {
  return impl_->nodes_[impl_->root_].terminal != Terminal::None;
}
bool Search::root_solved() const noexcept { return impl_->solved(impl_->nodes_[impl_->root_]); }

std::vector<RootChild> Search::root_children() const {
  const Impl& s = *impl_;
  const Node& r = s.nodes_[s.root_];
  std::vector<RootChild> out;
  for (const std::size_t i : s.root_edges()) {
    const Edge& e = r.edges[i];
    RootChild c;
    if (s.params_.split == TurnSplit::WholeTurn) {
      const std::span<const Move> a = r.action(e.action);
      c.moves.assign(a.begin(), a.end());
    } else if (!e.submit) {
      c.moves.push_back(e.move);
    }
    c.submit = e.submit;
    c.legality = e.unknown ? MoveLegality::Unknown : MoveLegality::Legal;
    c.prior = e.prior;
    c.visits = e.visits;
    c.q = s.edge_q(r, e);
    c.d = s.edge_d(e);
    c.child = e.child;
    // I15's side table.
    c.prior_noise_free = s.root_prior_noise_free_.size() == r.edges.size() ? s.root_prior_noise_free_[i] : e.prior;
    c.noise_alpha = s.root_noise_alpha_.size() == r.edges.size() ? s.root_noise_alpha_[i] : 0.0;
    if (e.w > 0) {
      const double mean = e.wq / e.w;
      c.value_variance = std::max(0.0, e.wq2 / e.w - mean * mean);
    }
    if (e.child != kNoNode) {
      const Node& n = s.nodes_[e.child];
      c.child_key = n.key;
      if (s.params_.variance_bonus && n.has_eval) c.exploration_scale = s.variance_bonus(n, e);
    }
    if (e.child != kNoNode && s.nodes_[e.child].has_eval) {
      // The child's own evaluation, the network's, in the root player's view.
      const Node& n = s.nodes_[e.child];
      const double q = detail::sign_between(r, n) * n.eval_mean;
      c.evaluated = true;
      c.network_wdl = {(1.0 - n.eval_draw + q) * 0.5, n.eval_draw, (1.0 - n.eval_draw - q) * 0.5};
      c.network_moves_left = n.eval_moves_left;
      c.network_error = n.eval_error;
    }
    if (e.child != kNoNode && s.params_.proof_channel) {
      const auto [lo, hi] = detail::bounds_in_parent(r, s.nodes_[e.child]);
      c.lo = e.unknown ? -1 : lo;  // (e) no proof through an unknown edge
      c.hi = hi;
    }
    out.push_back(std::move(c));
  }
  return out;
}

MoveLegality Search::verify(std::size_t index, const LegalityOptions& options) {
  Impl& s = *impl_;
  if (!s.pending_.empty()) throw std::invalid_argument("search: cannot verify while a batch is waiting");
  const std::vector<std::size_t> edges = s.root_edges();
  if (index >= edges.size()) {
    throw std::invalid_argument("verify: root child " + std::to_string(index) + " of " +
                                std::to_string(edges.size()));
  }
  const Edge& e = s.nodes_[s.root_].edges[edges[index]];
  if (!e.unknown) return MoveLegality::Legal;
  return s.verify_edge(s.root_, edges[index], options);
}

Choice Search::choose() const {
  const std::vector<std::size_t> edges = impl_->root_edges();
  const std::vector<FinalCandidate> c = impl_->root_candidates(edges);
  bool any = false;
  for (const FinalCandidate& f : c) any = any || f.visits > 0;
  if (!any) throw std::invalid_argument("search: the root has no visited child to choose");
  return select_final(c, impl_->params_);
}

std::size_t Search::choose_by_visits(double temperature, double random) const {
  if (!std::isfinite(temperature)) throw std::invalid_argument("choose_by_visits: the temperature is not finite");
  if (!(random >= 0 && random <= 1)) throw std::invalid_argument("choose_by_visits: random must be in [0, 1]");
  const std::vector<std::size_t> edges = impl_->root_edges();
  const std::vector<FinalCandidate> c = impl_->root_candidates(edges);
  // A proven win is played; children proven lost are left out unless all are.
  std::size_t best = kNoEdge;
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (c[i].lo == 1 && c[i].visits > 0 && (best == kNoEdge || c[i].visits > c[best].visits)) best = i;
  }
  if (best != kNoEdge) return best;
  bool any_alive = false;
  for (const FinalCandidate& f : c) any_alive = any_alive || (f.visits > 0 && f.hi > -1);
  const auto usable = [&](const FinalCandidate& f) {
    return f.visits > 0 && (!any_alive || f.hi > -1);
  };
  if (temperature <= 0) {
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (usable(c[i]) && (best == kNoEdge || c[i].visits > c[best].visits)) best = i;
    }
  } else {
    double total = 0;
    std::vector<double> weight(c.size(), 0.0);
    std::uint32_t most = 0;
    for (const FinalCandidate& f : c) most = usable(f) ? std::max(most, f.visits) : most;
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (!usable(c[i])) continue;
      weight[i] = std::pow(static_cast<double>(c[i].visits) / static_cast<double>(most), 1.0 / temperature);
      total += weight[i];
    }
    double x = std::clamp(random, 0.0, 1.0) * total;
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (weight[i] <= 0) continue;
      best = i;
      if (x < weight[i]) break;
      x -= weight[i];
    }
  }
  if (best == kNoEdge) throw std::invalid_argument("search: the root has no visited child to choose");
  return best;
}

void Search::advance(std::size_t index) {
  Impl& s = *impl_;
  if (!s.pending_.empty()) throw std::invalid_argument("search: cannot advance while a batch is waiting");
  const std::vector<std::size_t> edges = s.root_edges();
  if (index >= edges.size()) {
    throw std::invalid_argument("advance: root child " + std::to_string(index) + " of " +
                                std::to_string(edges.size()));
  }
  const NodeId c = s.ensure_child(s.root_, edges[index]);
  if (c == kNoNode) throw std::invalid_argument("advance: that child is a dead end (its move is illegal)");
  s.reroot(c);
}

void Search::advance_to(const GameState& target, int moves_made) {
  Impl& s = *impl_;
  if (!s.pending_.empty()) throw std::invalid_argument("search: cannot advance while a batch is waiting");
  s.check_moves_made(target, moves_made);
  const std::size_t count = s.nodes_[s.root_].edges.size();
  for (std::size_t i = 0; i < count; ++i) {
    const Edge& e = s.nodes_[s.root_].edges[i];
    if (e.pruned) continue;
    if (e.child != kNoNode) {
      const Node& c = s.nodes_[e.child];
      if (c.moves_made == moves_made && c.state == target) {
        s.reroot(e.child);
        return;
      }
      continue;
    }
    int made = 0;
    bool illegal = false;
    const GameState child = s.child_state(s.root_, i, made, illegal);
    if (!illegal && made == moves_made && child == target) {
      const NodeId c = s.ensure_child(s.root_, i);
      if (c != kNoNode) {
        s.reroot(c);
        return;
      }
    }
  }
  if (s.params_.graph) {
    // The same (state, moves made in its action) elsewhere in the graph - never
    // a turn start for a partial action, nor the reverse, nor one count for
    // another (A6).
    const NodeId found = s.find_node(s.key_of(target), target, moves_made);
    if (found != kNoNode && s.nodes_[found].state == target) {
      s.reroot(found);
      return;
    }
  }
  const NodeId fresh = s.make_node(target, moves_made);
  s.reroot(fresh);
}

void Search::advance_moves(std::span<const Move> moves) {
  const Impl& s = *impl_;
  GameState target = root_state();
  MoveList scratch;
  const auto play = [&](const Move& m) {
    if (!target.may_move() || !is_pseudo_legal(target, m, scratch)) {
      throw std::invalid_argument("advance: " + to_string(m, target.multiverse().even()) +
                                  " is not a move of the root's player now");
    }
    target.apply(m);
  };
  const auto submit = [&] {
    if (target.submission_status() != SubmitStatus::Safe) {
      throw std::invalid_argument("advance: the action cannot be submitted legally (T5, C4)");
    }
    target.submit();
  };
  int moves_made = 0;  // A6: 0 after a submission, the root's + 1 after a move
  if (s.params_.split == TurnSplit::WholeTurn) {
    for (const Move& m : moves) play(m);
    submit();
  } else if (moves.empty()) {
    submit();
  } else if (moves.size() == 1) {
    play(moves[0]);  // never submitted with it (D8): submitting is its own decision
    moves_made = s.nodes_[s.root_].moves_made + 1;
  } else {
    throw std::invalid_argument("advance_moves: the board-move split takes one move per decision");
  }
  advance_to(target, moves_made);
}

void Search::advance_action(std::span<const Move> moves) {
  const Impl& s = *impl_;
  if (!s.pending_.empty()) throw std::invalid_argument("search: cannot advance while a batch is waiting");
  if (moves.empty() && root_turn_start()) {
    throw std::invalid_argument("advance_action: an action from a turn start has at least one move (T5)");
  }
  // The whole action, checked on a copy: nothing is re-rooted unless every
  // move can be played and the submission is legal.
  GameState t = root_state();
  MoveList scratch;
  for (const Move& m : moves) {
    if (!t.may_move() || !is_pseudo_legal(t, m, scratch)) {
      throw std::invalid_argument("advance_action: " + to_string(m, t.multiverse().even()) +
                                  " is not a move of the acting player there");
    }
    t.apply(m);
  }
  if (t.submission_status() != SubmitStatus::Safe) {
    throw std::invalid_argument("advance_action: the action cannot be submitted legally (T5, C4)");
  }
  if (s.params_.split == TurnSplit::WholeTurn) {
    advance_moves(moves);
    return;
  }
  for (const Move& m : moves) advance_moves(std::span<const Move>(&m, 1));
  advance_moves({});
}

std::uint64_t Search::epoch() const noexcept { return impl_->epoch_; }
void Search::set_epoch(std::uint64_t epoch) {
  if (epoch == impl_->epoch_) return;
  if (!impl_->pending_.empty()) {
    throw std::invalid_argument(
        "set_epoch: a batch is waiting - its evaluations belong to the current epoch (deliver or "
        "abandon it first)");
  }
  impl_->epoch_ = epoch;
  impl_->notify_root();
}
void Search::set_root_hook(std::function<void(const GameState&, std::uint64_t)> hook) {
  impl_->hook_ = std::move(hook);
}

std::size_t Search::node_count() const noexcept { return impl_->nodes_.size(); }

NodeView Search::node(NodeId id) const {
  const Impl& s = *impl_;
  if (id >= s.nodes_.size()) throw std::out_of_range("node " + std::to_string(id));
  const Node& n = s.nodes_[id];
  NodeView v;
  v.state = &n.state;
  v.key = n.key;
  v.mover = n.mover;
  v.moves_made = n.moves_made;
  v.turn_start = n.turn_start;
  v.terminal = n.terminal;
  v.expanded = n.expanded;
  v.stuck = n.stuck;
  v.dead_end = n.dead_end;
  v.children_complete = n.children_complete;
  v.submit_possible = n.submit_possible;
  v.submit_legal = n.submit_legal;
  v.lo = n.lo;
  v.hi = n.hi;
  v.visits = n.visits;
  v.ends = n.ends;
  v.virtual_loss = n.vl;
  v.weight = n.w;
  v.value = n.wq;
  v.value_sq = n.wq2;
  v.draw = n.wd;
  v.eval_mean = n.eval_mean;
  v.eval_draw = n.eval_draw;
  v.eval_variance = n.eval_var;
  v.eval_error = n.eval_error;
  v.eval_moves_left = n.eval_moves_left;
  v.evaluated = n.has_eval;
  v.k0 = n.k0;
  v.instantiated = n.instantiated;
  v.edge_count = n.edges.size();
  return v;
}

EdgeView Search::edge(NodeId id, std::size_t index) const {
  const Impl& s = *impl_;
  if (id >= s.nodes_.size() || index >= s.nodes_[id].edges.size()) {
    throw std::out_of_range("edge " + std::to_string(index) + " of node " + std::to_string(id));
  }
  const Node& n = s.nodes_[id];
  const Edge& e = n.edges[index];
  EdgeView v;
  if (s.params_.split == TurnSplit::WholeTurn) {
    v.action = n.action(e.action);
  } else if (!e.submit) {
    v.move = &e.move;
  }
  v.submit = e.submit;
  v.pruned = e.pruned;
  v.unknown = e.unknown;
  v.verified = e.verified;
  v.prior = e.prior;
  v.child = e.child;
  v.visits = e.visits;
  v.virtual_loss = e.vl;
  v.weight = e.w;
  v.value = e.wq;
  v.value_sq = e.wq2;
  v.draw = e.wd;
  return v;
}

double Search::root_cpuct() const noexcept {
  const Impl& s = *impl_;
  return s.params_.cpuct * s.cpuct_scale(s.nodes_[s.root_]);
}

std::uint32_t Search::root_best_since() const noexcept { return impl_->best_since_; }

std::uint64_t Search::digest() const { return impl_->digest(); }
void Search::check_invariants() const { impl_->check_invariants(); }

}  // namespace t5d::mcts
