// t5d/mcts.hpp - the main search (engine stage E6a): PUCT over a tree, or a
// graph, of game states, fed by a batched evaluator (design spec §8.1-8.7,
// §11.1's proof channel - the consumer side - §11.2's synchronous debug mode,
// §15.4's invariants; register D2, E1-E9, C11-C14, H2, H15, H19, I11).
//
// WHAT A NODE IS. A node holds a GameState (t5d/state.hpp) and its player to
// move - the player whose ACTION is in progress. D2 (multi-move turns) is
// undecided, so both of its arms are built (SearchParams::split):
//
//  * TurnSplit::BoardMove, "each board-move as its own ply" (§8.2). A node is
//    a state at a turn's start or during an action; its children are the
//    LEGAL MOVES - those after which the action can still be finished legally
//    (t5d/search.hpp: legal_moves; rulebook R1 judges legality per action, so
//    a move into check mid-turn is legal) - and SUBMIT: turns have variable
//    length (rulebook R8, register D8), so submitting is an action of its own.
//    A submit edge exists exactly where submitting is POSSIBLE (the action's
//    obligations are met, T5: Expansion::submit_possible), and is legal
//    exactly where the submission passes C4's test (Expansion::submit_legal,
//    exact); an illegal one is kept but pruned - never selected. Nothing is
//    submitted automatically: after the move that shifts the present the node
//    still belongs to the same acting player, and offers submit beside any
//    further move on that player's remaining playable boards (by default,
//    R8; under Q4's Guide-literal reading no further move, so submit alone).
//    Only a submit edge changes the acting player. Mid-turn check is legal,
//    and terminal detection runs only at turn boundaries (§8.2).
//
//    THE ENDS, from the rulebook. At a turn's start, no legal action is the
//    game's end (C5): checkmate if the side to move is in check (R1's status),
//    otherwise stalemate (C10, a draw; C13: no royal, never in check) - decided
//    only by E4's exhaustive detector (H37). DURING an action there is no game
//    end: C5 speaks of the start of a turn. A partial action whose remaining
//    moves and submission are all illegal has no legal completion, so the
//    move that reached it was not a legal move (R1: a move is legal when a
//    legal action continues with it); it is a DEAD END, pruned and never
//    scored (§8.2: "a dead end to prune, not a loss"). With the exact legality
//    classification it is never reached; with a budget (an unknown move) or
//    the pseudo-legal test seam (SearchParams::legal_moves_only = false) it is
//    found when the node is expanded, and pruned. After the shift with a legal
//    submission the action can always be finished, whatever the further moves.
//  * TurnSplit::WholeTurn: a node is always a turn's start, and its children
//    are the legal ACTIONS (t5d/search.hpp: list_legal_actions, E4's
//    enumeration). The network's policy is over single moves (D1), so an
//    action's prior is derived from the logits of its moves
//    (SearchParams::action_prior; §8.2 does not say how). The listing stops at
//    SearchParams::max_actions; a node whose listing stopped has incomplete
//    children and is never proven lost.
//
// HOW LEGALITY ENTERS. Through one call site (mcts.cpp, leaf_expansion): the
// expansion of a leaf (t5d/expand.hpp) with ExpandOptions::legal_moves under
// the ply split, classified by engine stage E5a's tiers (t5d/legal.hpp) under
// a BUDGET (build-log spec issue 46; the orchestrator's policy, each part a
// parameter): (a) a move proved illegal is dropped, a legal or unknown one
// kept, the unknown ones flagged; (b) an unknown move gets its prior
// (unknown_prior_scale); (c) the first time selection picks an unknown edge it
// is verified under a larger budget (verify_legality) - illegal: the edge is
// removed, its siblings' priors renormalised, selection repeated, nothing
// backed up; legal: the flag goes; still unknown: the search proceeds with the
// edge flagged, never on a proof line; (d) a node is terminal only on an exact
// verdict of E4's detectors (H37), never because its known moves ran out; (e)
// the solver takes only exact verdicts: a child reached through an unknown edge
// never raises its parent's lower bound (it may still bound it from above - a
// superset of the legal children is a sound upper bound); (f) for training,
// the policy target is the root's visit counts and the network's move mask
// the legal and unknown moves (RootChild::legality). Under the whole-turn
// split the expansion offers the pseudo-legal moves (the network's
// candidates, over which every legal action's moves are found) and
// list_legal_actions the children, which are legal.
//
// TERMINAL VERDICTS come ONLY from E4's exhaustive detectors (detect_mate,
// H37), at turn starts: checkmate (the mover loses, -1) and stalemate (a draw,
// rulebook C10, 0). An Unknown verdict (a budget stopped the detector) is never
// a terminal. Self-play's turn cap (I11, SearchParams::turn_cap) is a
// convention defined from the state, scored as a draw and never a proof.
//
// VALUES are from the perspective of the node's ACTING PLAYER (register D8:
// the side whose action is in progress, until it submits - after the present
// has shifted inside the action it is no longer the colour to move on the
// present), in [-1, 1]. The evaluator's value is a 51-bin categorical
// distribution (D3) over the bin centres (2k - 50) / 50, from the leaf's
// acting player. Backing up across an edge flips the sign iff the two nodes'
// acting players differ - under the ply split, only across a submit edge.
//
// SELECTION (E1, committed): PUCT, U = c * P * sqrt(N) / (1 + n), over the
// INSTANTIATED children only: the children sorted by prior, the first
// k(N) = ceil(k0 * (N + 1)^alpha) instantiated (progressive widening), k0 set
// per node from the policy entropy, k0 = max(k_min, scale * exp(H)) - sharp
// policies narrow, flat ones widen. FPU reduction for unvisited children
// (Q_parent - fpu * sqrt(visited prior mass)). Root Dirichlet noise for
// self-play. E2 (provisional, flags): cPUCT scaled by the node's value
// standard deviation (the categorical variance as a prior, blended with the
// backed-up values'), and uncertainty-weighted backups. E3 (prototype, flag):
// a per-child variance bonus, multiplicative on U, capped, never below 1.
//
// VIRTUAL LOSS (§15.4): leaves are gathered in batches; each path carries one
// virtual visit, counted as a loss, on every node and edge it crosses, and the
// virtual loss is ALWAYS released - by the backup, by abandon(), or when an
// evaluator throws.
//
// MCGS (C11, prototype; SearchParams::graph): nodes keyed by the state - the
// 5D Zobrist key (H18), which carries the ruleset ID, every board in its place
// (so what has been played inside the action: a played board has its
// successor) and the acting player (a state after the shift and the same
// boards after submission differ in it) - and, since C23 ruled a board's
// causal past to be its ancestry (lineage; spec issue 30), each timeline's
// branch origin (NodeKey::StateAndOrigins, the default) - with an exact
// comparison on every key match.
// Statistics live on the edges (path statistics); a transposition reached
// through an edge whose Q lags the node's by more than epsilon backs up the
// residual correction instead of descending (Czech, Korus and Kersting 2021).
// Evaluations are a function of the state and the refresh epoch only
// (EvalCache), so a transposition is a zero-cost evaluation.
//
// THE PROOF CHANNEL'S CONSUMER (H2, E9), behind SearchParams::solver - OFF by
// default until the user rules (spec §11.1 item 2 counts injecting solved
// bounds into the tree as the proof channel; H23 keeps that disabled until
// the generator passes the gate). With it on, score bounds [lo, hi] per node
// propagate MCTS-Solver style (Winands et al. 2008; with draws, Cazenave and
// Saffidine's score-bounded MCTS): a child proven won for the parent makes the
// parent proven won; all children proven lost (every legal child created, the
// list complete) makes it proven lost; bounds only tighten and never
// contradict (a contradiction throws). A solved node leaves selection and
// consumes no evaluation: a visit backs up its exact value. A solved child is
// selected only until it has solved_visit_cap visits (the visit threshold
// for solved-node selection). The bounds come only from the engine's own
// exhaustive verdicts at its own leaves - ordinary search. A PERSISTENT SOLVED
// STORE (H19's export) is not built: proofs may not enter one until the
// generator has passed the differential-testing gate (H23; AGENTS.md, "The
// proof channel"), so SearchParams::proof_store must stay false - the search
// refuses to start otherwise.
//
// FINAL SELECTION (E5, E9): with the solver on, a proven win is played at
// once; children proven lost are ineligible unless all are; when an eligible
// child is solved, the
// secure child (the largest lower bound: exact for solved children,
// Q - A / sqrt(n) otherwise); else visits (the baseline) or the LCB rule -
// among children with at least lcb_floor of the most-visited child's visits,
// the largest Q - z * sqrt(var / n), var optionally plus the child's
// categorical variance. Under MCGS the visits are the edges' (path visits).
// No depth factor (E6, refused).
//
// PERSISTENCE (C12-C14): advance() re-roots on the child a move (or action)
// reaches, carries its statistics and deletes every node the new root cannot
// reach (garbage collection of path-dependent structure). State-keyed facts -
// the evaluation cache - survive; an invalidation hook tells an evaluator
// holding cached K/V that the root changed. Pondering is searching the
// post-move position and advancing on the reply. Self-play runs fresh searches
// per move (t5d/selfplay.hpp).
//
// DETERMINISM (H15): the search is single-threaded; with a deterministic
// evaluator every run from the same state, parameters and seed is identical,
// bit for bit (digest()). SearchParams::synchronous is the debug mode: batches
// of one leaf, every leaf backed up before the next is selected.

#ifndef T5D_MCTS_HPP
#define T5D_MCTS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "t5d/expand.hpp"
#include "t5d/legal.hpp"
#include "t5d/move.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"

namespace t5d::mcts {

/// D3: the value head's bins, centres (2k - (kValueBins - 1)) / (kValueBins - 1).
inline constexpr int kValueBins = 51;

/// The centre of value bin k.
constexpr double value_bin_centre(int k) noexcept {
  return (2.0 * k - (kValueBins - 1)) / static_cast<double>(kValueBins - 1);
}

using NodeId = std::uint32_t;
inline constexpr NodeId kNoNode = std::numeric_limits<NodeId>::max();

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// D2 - undecided; both arms (ablations 6g, 6k).
enum class TurnSplit : std::uint8_t {
  BoardMove = 0,  ///< (a) each board-move its own ply
  WholeTurn = 1,  ///< (b) children are whole legal actions
};

/// How an action's prior is derived from its moves' logits (whole-turn split;
/// §8.2 is silent). The softmax is then taken over the listed actions.
enum class ActionPrior : std::uint8_t {
  SumLogits = 0,   ///< logit(action) = sum of its moves' logits: the product of
                   ///< independent per-move factors
  MeanLogits = 1,  ///< the mean: no penalty on actions of more moves
};

/// The MCGS node key (C11, H18; build-log spec issue 30).
enum class NodeKey : std::uint8_t {
  State = 0,            ///< the 5D Zobrist key (ruleset ID included); origins ignored
  StateAndOrigins = 1,  ///< also each timeline's branch origin (the lineage reading)
};

/// The final-selection rule outside solved positions (E5).
enum class FinalRule : std::uint8_t {
  Visits = 0,  ///< the baseline: the most visits
  Lcb = 1,     ///< among children above the visit floor, the largest LCB of Q
};

/// Self-play's turn cap (I11): a convention defined from the state - a turn
/// start whose present has reached `turns` turns, or whose multiverse holds
/// `boards` boards, ends the game as a draw. 0 = no limit. Not a game rule.
struct TurnCap {
  int turns = 0;
  int boards = 0;
  bool enabled() const noexcept { return turns > 0 || boards > 0; }
  /// Whether the cap binds at this turn start.
  bool reached(const GameState& turn_start) const noexcept;
};

struct SearchParams {
  // ---- D2: multi-move turns (undecided) --------------------------------------
  TurnSplit split = TurnSplit::BoardMove;  ///< placeholder: D2 is undecided
  ActionPrior action_prior = ActionPrior::SumLogits;  ///< whole-turn: placeholder
  /// Whole-turn: the most actions listed per node (list_legal_actions' cap).
  std::size_t max_actions = 4096;
  /// Whole-turn: the listing's node budget; a stopped listing is incomplete.
  std::uint64_t listing_nodes = 2'000'000;
  /// Board-move: when the evaluator gives no submit logit (Evaluation::
  /// submit_logit NaN), submit's share of the prior where further moves are
  /// open too (the moves share the rest, in their own proportions). D8 gives
  /// the network a submit logit of its own (stage N5); until then this
  /// placeholder stands in - and decides how long untrained actions are: a
  /// submit logit of 0 beside N flat move logits would give submit 1/(N+1),
  /// and actions that play nearly every optional board (R8). Placeholder 0.5.
  double submit_share = 0.5;
  /// Board-move: children are the legal moves (true, the rule). False is a
  /// test seam - pseudo-legal children, whose dead ends are pruned.
  bool legal_moves_only = true;
  /// (a) Board-move: the expansion's legality classification (t5d/legal.hpp):
  /// proved-illegal moves dropped, unknown ones kept and flagged. Placeholder:
  /// E5a's measured budget (64 checks, 2,000 search nodes);
  /// LegalityOptions::exact() leaves nothing unknown.
  LegalityOptions legality = LegalityOptions::budgeted(64, 2000);
  /// (b) An unknown move's prior is multiplied by this (then renormalised).
  /// 1: it gets its prior. Placeholder.
  double unknown_prior_scale = 1.0;
  /// (c) Verify an unknown edge the first time selection picks it.
  bool verify_unknown = true;
  /// (c) ... under this budget. Placeholder.
  LegalityOptions verify_legality = LegalityOptions::budgeted(4096, 200'000);

  // ---- E1: PUCT, top-k instantiation, entropy-adaptive widening, FPU, noise -----
  double cpuct = 1.5;                 ///< placeholder
  bool widening = true;               ///< false: every child instantiated (plain PUCT)
  double widening_alpha = 0.5;        ///< k(N) = ceil(k0 (N + 1)^alpha); placeholder
  double widening_scale = 0.5;        ///< k0 = max(widening_min, scale * exp(H)); placeholder
  double widening_min = 2.0;          ///< placeholder
  double fpu_reduction = 0.25;        ///< placeholder
  bool root_noise = false;            ///< root Dirichlet noise (self-play only)
  double dirichlet_alpha = 0.3;       ///< per child; placeholder
  double dirichlet_epsilon = 0.25;    ///< placeholder
  double policy_temperature = 1.0;    ///< softmax temperature on the logits

  // ---- E2 (provisional): variance-scaled cPUCT, uncertainty-weighted backups ---
  bool variance_scaled_cpuct = false;
  double cpuct_variance_scale = 0.85;  ///< lambda: c *= 1 + lambda (sigma / sigma_ref - 1)
  double variance_reference = 0.4;     ///< sigma_ref, a value standard deviation
  double variance_prior_weight = 2.0;  ///< the categorical variance's weight, in visits
  double cpuct_scale_min = 0.5;
  double cpuct_scale_max = 2.0;
  bool uncertainty_weighted_backup = false;  ///< w = sigma_ref^2 / (sigma_ref^2 + var)

  // ---- E3 (prototype): per-child variance bonus ----------------------------------
  bool variance_bonus = false;
  double variance_bonus_scale = 1.0;  ///< beta: U *= clamp(1 + beta (sigma_c / sigma_ref - 1), 1, cap)
  double variance_bonus_cap = 2.0;

  // ---- C11 (prototype): MCGS --------------------------------------------------------
  bool graph = false;
  /// The lineage reading (C23, ruled): states that differ in a branch origin
  /// are different positions for the network, so different nodes.
  NodeKey node_key = NodeKey::StateAndOrigins;
  double residual_epsilon = 0.05;     ///< placeholder
  /// A test seam, not mirrored in the C API: node keys cut to their low
  /// `key_bits` bits (1-64), so that keys collide and the exact comparison
  /// behind every key match - the graph's and the evaluation cache's - is
  /// exercised. 64 in play.
  int key_bits = 64;

  // ---- H2, E9: the in-tree solver ------------------------------------------------
  /// Propagating proven bounds up the tree, and the decisions taken from them
  /// (a proven child's exact Q, the visit threshold, proven-lost children
  /// ineligible, a proven win played, the secure child, a solved root ending
  /// the search). OFF until the user rules: spec §11.1 item 2 names "inject
  /// solved bounds into the tree" as part of the proof channel, and H23 keeps
  /// the proof channel disabled until the generator has passed the gate. With
  /// it off, a terminal (C5 at a turn start, by E4's exhaustive detector; I11's
  /// cap) is still valued as its game result - ordinary MCTS - and nothing
  /// else is proven. Turning it on is this one line.
  bool solver = false;
  /// The visit threshold for solved-node selection: a solved child is selected
  /// only while it has fewer visits (unless nothing else can be). Placeholder.
  std::uint32_t solved_visit_cap = 8;
  /// H19's export to a persistent solved store. MUST stay false: no proof may
  /// enter a solved store until the generator has passed the gate (H23). The
  /// search throws std::invalid_argument if it is set.
  bool proof_store = false;

  // ---- E5, E9: final selection ----------------------------------------------------
  FinalRule final_rule = FinalRule::Lcb;
  double lcb_floor = 0.15;              ///< "~15% of the max-visit child"
  double lcb_z = 3.0;                   ///< "a few standard errors"; placeholder
  double lcb_categorical_weight = 0.0;  ///< the optional categorical term (0: off)
  double secure_child_a = 1.0;          ///< secure child: Q - A / sqrt(n); placeholder

  // ---- leaves -----------------------------------------------------------------------
  std::size_t batch_size = 8;   ///< leaves per evaluator call
  double virtual_loss = 1.0;    ///< virtual visits per path, each a loss
  /// H15: the synchronous debug mode - batches of one, deterministic.
  bool synchronous = false;
  std::uint64_t seed = 1;       ///< root noise
  /// The leaves' network input: window, features, relations. legal_moves is
  /// set by the split (above), whatever it says here.
  ExpandOptions expand{};

  // ---- terminal detection (H37) ------------------------------------------------------
  Detector detector = Detector::Hypercuboid;
  std::uint64_t mate_nodes = 2'000'000;              ///< a stopped detector: Unknown
  std::size_t mate_bytes = std::size_t{512} << 20;  ///< its memory ceiling
  TurnCap turn_cap{};  ///< I11, self-play only (off by default)

  // ---- state-keyed evaluations ---------------------------------------------------------
  std::size_t eval_cache_entries = std::size_t{1} << 16;  ///< 0: no cache
  std::size_t eval_cache_bytes = std::size_t{256} << 20;  ///< its memory ceiling (estimated)
};

/// What a search may spend in one run().
struct Limits {
  std::uint64_t playouts = 800;  ///< completed simulations (visits to the root)
  std::uint64_t evaluations = std::numeric_limits<std::uint64_t>::max();
};

// ---------------------------------------------------------------------------
// The evaluator
// ---------------------------------------------------------------------------

/// One leaf to evaluate: everything the network takes from it is in the
/// expansion - per-board and per-square features, the candidate moves the
/// policy is over (expansion->moves(), in order), the move graph, typed
/// relations, W(k)/E(k). Pointers are valid until the batch is delivered.
struct EvalRequest {
  const GameState* state = nullptr;
  /// Its global inputs include D8's: the acting player
  /// (expansion->acting_player()) and the submit action
  /// (expansion->submit_possible(), submit_legal()).
  const Expansion* expansion = nullptr;
  std::uint64_t key = 0;    ///< the node key
  std::uint64_t epoch = 0;  ///< the refresh epoch (§7.9)
  // Not here: whether the leaf is a turn start - a path property the network
  // must not see (an evaluation is a function of the state and the epoch).
};

/// The network's answer for one leaf.
struct Evaluation {
  /// One logit per candidate move (expansion->moves().size()).
  std::vector<float> logits;
  /// D8: the submit action's logit (under the ply split, read from a global
  /// register), used when the leaf's submit is possible; NaN - none given -
  /// stands for SearchParams::submit_share. Must not be infinite.
  float submit_logit = std::numeric_limits<float>::quiet_NaN();
  /// Probabilities over kValueBins bins (normalised by the search), from the
  /// leaf's acting player.
  std::array<float, kValueBins> value{};
};

/// The mean and variance of a value distribution (normalised first). Throws
/// std::invalid_argument for a negative, non-finite or all-zero one.
struct ValueMoments {
  double mean = 0;
  double variance = 0;
};
ValueMoments value_moments(const std::array<float, kValueBins>& value);

/// A distribution with all its mass on `v` in [-1, 1], split between the two
/// nearest centres (mean exactly v).
std::array<float, kValueBins> two_hot(double v);

class Evaluator {
 public:
  virtual ~Evaluator() = default;
  /// Fills out[i] for batch[i]. May throw: the search then releases the
  /// batch's virtual loss and rethrows.
  virtual void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) = 0;
  /// C12: the search re-rooted (or the epoch changed); an evaluator that holds
  /// cached K/V for the old root's conditioning invalidates it here.
  virtual void on_new_root(const GameState& /*root*/, std::uint64_t /*epoch*/) {}
};

/// Built-in evaluators, for tests, invariants and the tools.
///  * uniform: every logit 0, value exactly 0;
///  * random: logits and a value distribution pseudo-random but a function of
///    (seed, node key, epoch) only - so evaluation stays a function of the state;
///  * material: value tanh(material balance / 10) over every timeline's head
///    board, from the player to move; capture logits 0.5 x the captured value.
enum class BuiltinEvaluator : std::uint8_t { Uniform = 0, Random = 1, Material = 2 };
std::unique_ptr<Evaluator> make_evaluator(BuiltinEvaluator kind, std::uint64_t seed = 1);
const char* evaluator_name(BuiltinEvaluator kind) noexcept;

/// The material values the material evaluator counts (royals 0).
double material_value(PieceKind kind) noexcept;

// ---------------------------------------------------------------------------
// Final selection, as a pure function (testable on constructed statistics)
// ---------------------------------------------------------------------------

/// One root child as final selection sees it, from the root player's view.
struct FinalCandidate {
  std::uint32_t visits = 0;   ///< path visits
  double q = 0;               ///< mean backed-up value
  double variance = 0;        ///< variance of the backed-up values
  double eval_variance = 0;   ///< the child's categorical variance (0 if none)
  int lo = -1;                ///< solver bounds
  int hi = 1;
  bool solved() const noexcept { return lo == hi; }
};

enum class ChoiceReason : std::uint8_t {
  ProvenWin = 0,    ///< a child proven won
  SecureChild = 1,  ///< an eligible child is solved: the largest lower bound
  Lcb = 2,          ///< the LCB rule
  Visits = 3,       ///< the most visits
  AllLost = 4,      ///< every child proven lost: the most visits
};
const char* choice_reason_name(ChoiceReason reason) noexcept;

struct Choice {
  std::size_t index = 0;
  ChoiceReason reason = ChoiceReason::Visits;
};

/// E5, E9 (see the header comment). Throws std::invalid_argument if no child
/// has a visit (nothing to choose from).
Choice select_final(std::span<const FinalCandidate> children, const SearchParams& params);

/// Whether candidate i is eligible under select_final's rules for `reason`
/// (the tests' check that a choice is among the eligible children).
bool final_eligible(std::span<const FinalCandidate> children, std::size_t i,
                    ChoiceReason reason, const SearchParams& params);

// ---------------------------------------------------------------------------
// The evaluation cache: state-keyed evaluations (C11, §8.7)
// ---------------------------------------------------------------------------

class EvalCache {
 public:
  /// At most `capacity` entries and about `max_bytes` bytes (an entry's
  /// logits and its state's timeline list, estimated).
  explicit EvalCache(std::size_t capacity = std::size_t{1} << 16,
                     std::size_t max_bytes = std::size_t{256} << 20)
      : capacity_(capacity), max_bytes_(max_bytes) {}
  /// The cached evaluation of `state` at `epoch`, compared exactly (keys may
  /// collide), or nullptr.
  const Evaluation* find(std::uint64_t key, std::uint64_t epoch, const GameState& state,
                         bool compare_origins) const;
  /// Stores it (unless full: entries are never evicted - a deterministic cache).
  void insert(std::uint64_t key, std::uint64_t epoch, const GameState& state,
              const Evaluation& evaluation);
  std::size_t size() const noexcept { return entries_.size(); }
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t bytes() const noexcept { return bytes_; }
  void clear() noexcept {
    entries_.clear();
    bytes_ = 0;
  }

 private:
  struct Entry {
    GameState state;
    std::uint64_t epoch;
    Evaluation evaluation;
  };
  std::size_t capacity_;
  std::size_t max_bytes_;
  std::size_t bytes_ = 0;
  std::unordered_multimap<std::uint64_t, Entry> entries_;
};

// ---------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------

/// What the search has done (since construction; advance() keeps counting).
struct SearchStats {
  std::uint64_t playouts = 0;          ///< completed simulations
  std::uint64_t evaluations = 0;       ///< leaves sent to the evaluator
  std::uint64_t cache_hits = 0;        ///< leaves answered from the evaluation cache
  std::uint64_t batches = 0;
  std::uint64_t collisions = 0;        ///< a leaf already waiting in the batch
  std::uint64_t batch_duplicates = 0;  ///< a leaf whose state another leaf of the batch has
  std::uint64_t terminal_visits = 0;   ///< visits ending on a terminal or solved node
  std::uint64_t residual_backups = 0;  ///< MCGS: corrections backed up
  std::uint64_t transposition_hits = 0;  ///< MCGS: an edge linked to an existing node
  std::uint64_t key_collisions = 0;    ///< equal keys, different states
  std::uint64_t dead_ends = 0;         ///< partial turns with no legal completion, pruned
  std::uint64_t submit_edges = 0;      ///< D8: submit edges made (submitting possible)
  std::uint64_t illegal_submits = 0;   ///< ... of which illegal (C4), kept pruned
  std::uint64_t unknown_edges = 0;     ///< (a) edges whose move the expansion left unknown
  std::uint64_t verified_legal = 0;    ///< (c) unknown edges verified legal
  std::uint64_t verified_illegal = 0;  ///< (c) ... verified illegal and removed
  std::uint64_t verified_unknown = 0;  ///< (c) ... still unknown after verification
  std::uint64_t mate_checks = 0;       ///< turn starts given to the detector
  std::uint64_t unknown_verdicts = 0;  ///< ... that answered Unknown (H37: not terminal)
  std::uint64_t checkmates = 0;        ///< terminal nodes found
  std::uint64_t stalemates = 0;
  std::uint64_t turn_caps = 0;         ///< nodes the turn cap ended (I11)
  std::uint64_t stuck = 0;             ///< turn starts with no child and no verdict
  std::uint64_t truncated_listings = 0;  ///< whole-turn: the listing stopped early
  std::uint64_t nodes_created = 0;
  std::uint64_t nodes_collected = 0;   ///< deleted by re-rooting
  std::uint64_t reroots = 0;
  // Wall-clock seconds in the rules engine's calls (not part of digest()):
  double seconds_mate = 0;     ///< turn starts' terminal checks (detect_mate)
  double seconds_expand = 0;   ///< leaves' expansions (legality classification included)
  double seconds_listing = 0;  ///< whole-turn: the actions listed
  double seconds_verify = 0;   ///< (c) verifications of unknown edges
};

/// A node as the tests and tools read it.
enum class Terminal : std::uint8_t {
  None = 0,
  Checkmate,  ///< the player to move is checkmated: -1
  Stalemate,  ///< a draw (C10): 0
  TurnCap,    ///< I11's convention: a draw, never a proof
};

struct EdgeView {
  const Move* move = nullptr;  ///< board-move: the move (nullptr for submit)
  std::span<const Move> action;  ///< whole-turn: the action's moves (playable order)
  bool submit = false;
  bool pruned = false;
  bool unknown = false;   ///< the move's legality is unknown (a budget left it undecided)
  bool verified = false;  ///< verification has run on it (c)
  double prior = 0;
  NodeId child = kNoNode;
  std::uint32_t visits = 0;
  std::uint32_t virtual_loss = 0;
  double weight = 0;    ///< sum of backup weights
  double value = 0;     ///< weighted sum of backed-up values (the parent's view)
  double value_sq = 0;  ///< weighted sum of squares
};

struct NodeView {
  const GameState* state = nullptr;
  std::uint64_t key = 0;
  Colour mover = Colour::White;
  bool turn_start = true;
  Terminal terminal = Terminal::None;
  bool expanded = false;
  bool stuck = false;
  bool dead_end = false;
  bool children_complete = true;
  bool submit_possible = false;  ///< D8, once expanded: a submit edge exists
  bool submit_legal = false;     ///< ... and is legal (C4)
  int lo = -1;
  int hi = 1;
  std::uint32_t visits = 0;
  std::uint32_t ends = 0;  ///< visits that ended here (its evaluation, terminal visits)
  std::uint32_t virtual_loss = 0;
  double weight = 0;
  double value = 0;
  double value_sq = 0;
  double eval_mean = 0;
  double eval_variance = 0;
  double k0 = 0;
  std::size_t instantiated = 0;
  std::size_t edge_count = 0;
};

/// One root child for the caller: its moves and statistics (root player's view).
struct RootChild {
  std::vector<Move> moves;  ///< one move (board-move), an action, or none (submit)
  bool submit = false;
  /// Legal, or Unknown (the move's legality undecided by the expansion's
  /// budget): never Illegal - those are removed. Actions and submit: Legal.
  MoveLegality legality = MoveLegality::Legal;
  double prior = 0;
  std::uint32_t visits = 0;  ///< (f) the policy target
  double q = 0;
  int lo = -1;  ///< bounds; a child through an unknown edge proves nothing (lo -1)
  int hi = 1;
  NodeId child = kNoNode;
};

class Search {
 public:
  /// A search from `root`. `turn_start`: no move has been made yet in the
  /// root's action (the whole-turn split requires it). Throws
  /// std::invalid_argument for inconsistent parameters (proof_store among them).
  explicit Search(GameState root, SearchParams params = {}, bool turn_start = true,
                  std::shared_ptr<EvalCache> cache = nullptr);
  ~Search();
  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;
  Search(Search&&) noexcept;
  Search& operator=(Search&&) noexcept;

  const SearchParams& params() const noexcept;
  const SearchStats& stats() const noexcept;
  const EvalCache* cache() const noexcept;

  // -- running -----------------------------------------------------------------

  /// Runs simulations until `limits` (counted from this call) or until the
  /// root is solved or terminal: gather, evaluate, deliver.
  void run(Evaluator& evaluator, const Limits& limits);

  // -- the step API (the C API's external evaluator) ----------------------------

  /// Selects up to `max_leaves` leaves needing evaluation (virtual loss on each
  /// path), completing on the way every simulation that needs none (terminal
  /// and solved nodes, cache hits, MCGS corrections) as long as fewer than
  /// `max_playouts` simulations have been completed or gathered in this call.
  /// Returns the number of evaluations wanted - requests().size(): one per
  /// distinct state among the leaves (a tree may hold one state in two
  /// nodes). Precondition: none is waiting.
  std::size_t gather(std::size_t max_leaves,
                     std::uint64_t max_playouts = std::numeric_limits<std::uint64_t>::max());
  /// The evaluations wanted, in order.
  std::span<const EvalRequest> requests() const noexcept;
  /// Their evaluations, one per request, in order: expands the leaves, backs
  /// up, releases the virtual loss. Throws std::invalid_argument (releasing
  /// the virtual loss, nothing backed up) for a malformed evaluation.
  void deliver(std::span<const Evaluation> evaluations);
  /// Releases the waiting leaves' virtual loss without backing anything up.
  void abandon() noexcept;

  // -- the root ---------------------------------------------------------------------

  NodeId root() const noexcept;
  const GameState& root_state() const noexcept;
  bool root_turn_start() const noexcept;
  /// The root is terminal (its game is over) or solved.
  bool root_terminal() const noexcept;
  bool root_solved() const noexcept;
  /// The root's instantiated children (in prior order).
  std::vector<RootChild> root_children() const;
  /// E5/E9: the child to play, by index into root_children(). Throws
  /// std::logic_error if the root has no visited child.
  Choice choose() const;
  /// Self-play: a child sampled from the visit distribution at temperature t
  /// (t <= 0: the most visits, ties to the lower index), with `random` a
  /// uniform number in [0, 1).
  std::size_t choose_by_visits(double temperature, double random) const;
  /// Classifies root child `index`'s move under `options` (a move must be known
  /// legal before it is played): Illegal removes the child (its siblings'
  /// priors renormalised; indices of root_children() change), Legal clears its
  /// flag. Actions and submit are Legal. Precondition: nothing waiting.
  MoveLegality verify(std::size_t index, const LegalityOptions& options = LegalityOptions::exact());

  // -- persistence (C12, C13) --------------------------------------------------------

  /// Re-roots on root child `index` (of root_children()): carries its subtree's
  /// statistics, deletes every node the new root cannot reach; the evaluation
  /// cache survives; the evaluator hook is told. Precondition: nothing waiting.
  void advance(std::size_t index);
  /// Re-roots on the state reached by `moves` - one decision: under the
  /// board-move split one move (never submitted with it, D8) or none (submit),
  /// under the whole-turn split one action, submitted - whether or not the
  /// tree holds it.
  void advance_moves(std::span<const Move> moves);
  /// Re-roots on the turn start a whole action reaches - its moves in a
  /// playable order (the opponent's reply, at match play): under the
  /// board-move split one decision per move, then the submit decision; under
  /// the whole-turn split advance_moves(moves).
  void advance_action(std::span<const Move> moves);
  /// Re-roots on `target`, a state one decision from the root (a fresh root if
  /// the tree does not hold it). `turn_start` as for the constructor.
  void advance_to(const GameState& target, bool turn_start);
  /// The refresh epoch (§7.9): evaluations are cached per (state, epoch).
  std::uint64_t epoch() const noexcept;
  void set_epoch(std::uint64_t epoch);
  /// C12's invalidation hook: called with the new root and the epoch on every
  /// re-root and epoch change (after the evaluator's own on_new_root, when run()
  /// is given one). Replaces any earlier hook.
  void set_root_hook(std::function<void(const GameState&, std::uint64_t)> hook);

  // -- introspection (tests, tools) ---------------------------------------------------

  std::size_t node_count() const noexcept;
  NodeView node(NodeId id) const;
  EdgeView edge(NodeId id, std::size_t index) const;
  /// H15: a digest of the whole graph - every node's and edge's statistics, in
  /// a canonical order - equal for bit-identical searches.
  std::uint64_t digest() const;
  /// Checks the §15.4 invariants that hold between batches - visit counts
  /// consistent through backup, no virtual loss left, solver bounds sound and
  /// consistent with the children's, every node reachable from the root;
  /// throws std::logic_error naming the first that fails.
  void check_invariants() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The node key of a state under the given keying (the 5D Zobrist key, with
/// each timeline's branch origin mixed in for NodeKey::StateAndOrigins).
std::uint64_t node_key(const GameState& state, NodeKey keying) noexcept;

}  // namespace t5d::mcts

#endif  // T5D_MCTS_HPP
