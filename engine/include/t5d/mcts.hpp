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
//    found when the node is expanded, and pruned. After the shift a legal
//    submission finishes the action at once - but a further optional move
//    (R8) can still expose a royal, and such a move may lead to a dead end
//    like any other.
//  * TurnSplit::WholeTurn: a node is always a turn's start, and its children
//    are the legal ACTIONS (t5d/search.hpp: list_legal_actions, E4's
//    enumeration). The network's policy is over single moves (D1), so an
//    action's prior is derived from the logits of its moves
//    (SearchParams::action_prior; §8.2 does not say how). The listing stops at
//    SearchParams::max_actions; a node whose listing stopped has incomplete
//    children and is never proven lost. D8's submit is no separate decision
//    here (an action is listed submitted), so the evaluator's submit logit is
//    not read and the C API reports submit_possible 0: the whole-turn arm has
//    no end-of-turn token (a spec issue, not built).
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
// the policy target is the root's visit counts (RootChild::visits:
// root_children() lists the instantiated children, the others have none), and
// the network's move mask is a leaf's legal and unknown moves - every
// candidate the expansion kept, Expansion::move_legality (the C API's
// t5d_leaf.legality), not the root's children. Under the whole-turn
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
// present), in [-1, 1]. Backing up across an edge flips the sign iff the two
// nodes' acting players differ - under the ply split, only across a submit
// edge. The evaluator's value heads (register D3, corrected at CHECKPOINT 7),
// all from the leaf's acting player:
//  * WDL - the value search uses: a leaf's value is W - L (the three
//    normalised), and its draw probability D is backed up beside it, per node
//    and per edge, as Lc0 backs up D (a mate scores D 0, a stalemate and the
//    turn cap D 1) - so a root child's search value is (Q, D) and a WDL;
//  * the 51-bin categorical over the root search value, bin centres
//    (2k - 50) / 50 - optional; with the error head's output (optional) it is
//    one of the two uncertainty signals 6m chooses between
//    (SearchParams::uncertainty_source; register I10): the chosen one is the
//    node's variance that E2, E3 and E5's optional term read;
//  * moves-left (D4; optional) - recorded per node, never backed up: what a
//    moves-left count adds per edge waits on the unit of "turn" (build-log
//    spec issue 305), so the records keep each root child's network output
//    (RootChild::network_moves_left) and the targets add the offset.
//
// SELECTION (E1, committed): PUCT, U = c * P * sqrt(N) / (1 + n), over the
// INSTANTIATED children only: the children sorted by prior, the first
// k(N) = ceil(k0 * (N + 1)^alpha) instantiated (progressive widening), k0 set
// per node from the policy entropy, k0 = max(k_min, scale * exp(H)) - sharp
// policies narrow, flat ones widen. FPU reduction for unvisited children
// (Q_parent - fpu * sqrt(visited prior mass)). Root Dirichlet noise for
// self-play. E2 (provisional, flags): cPUCT scaled by the node's value
// standard deviation (the uncertainty signal - the categorical variance by
// default - as a prior, blended with the backed-up values'), and
// uncertainty-weighted backups. E3 (prototype, flag):
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
// comparison on every key match. A node's identity adds the MOVES MADE in its
// action (register A6, build-log spec issue 302): the network sees the count
// and no state records it, so equal states with different counts are
// different inputs, and different nodes. The count subsumes whether a turn is
// in progress (0: a turn start - only a turn start can end the game, C5, or be
// capped, I11; only a partial action can be a dead end). Every ordering of one
// action's moves from one turn start reaches its state with the same count, so
// MCGS still merges them; only states reached across turns with different
// counts stay apart.
// Statistics live on the edges (path statistics); a transposition reached
// through an edge whose Q lags the node's by more than epsilon backs up the
// residual correction instead of descending (Czech, Korus and Kersting 2021).
// Evaluations are a function of the state, the moves made in its action and
// the refresh epoch (EvalCache: §15.2's "same state + epoch -> same
// evaluation", the count part of what the state means to the network), so a
// transposition is a zero-cost evaluation.
//
// THE PROOF CHANNEL'S CONSUMER (H2, E9), behind SearchParams::proof_channel -
// OFF by rule (H41: the search's own solver is part of the proof channel;
// solved-status propagation and export stay off until the gate passes, the
// search's soundness fixes land with tests, and the proof channel is signed
// off; game-end scoring stays on). With it on, score bounds [lo, hi] per node
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
// FINAL SELECTION (E5, E9): with the proof channel on, a proven win is played at
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

/// D3: the WDL head's outputs - win, draw, loss, in that order.
inline constexpr int kWdl = 3;

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

/// The uncertainty signal search consumes (register I10; ablation 6m): "the
/// categorical variance vs the error head - both trained from the start
/// (§8.3)". The chosen one is each node's variance - what E2, E3 and E5's
/// optional term read.
enum class UncertaintySource : std::uint8_t {
  Categorical = 0,  ///< the 51-bin categorical's variance (0 where none was given)
  ErrorHead = 1,    ///< the error head's output (0 where none was given)
};

/// The final-selection rule outside solved positions (E5).
enum class FinalRule : std::uint8_t {
  Visits = 0,  ///< the baseline: the most visits
  Lcb = 1,     ///< among children above the visit floor, the largest LCB of Q
};

/// Self-play's turn cap (I11): a convention defined from the state - a turn
/// start whose present is PAST `turns` turns (turn turns + 1 or later: `turns`
/// whole turns have been played), or whose multiverse holds `boards` boards or
/// more, ends the game as a draw. 0 = no limit. Not a game rule.
struct TurnCap {
  int turns = 0;
  int boards = 0;
  bool enabled() const noexcept { return turns > 0 || boards > 0; }
  /// Whether the cap binds at this turn start: its present past `turns`
  /// turns, or `boards` boards or more.
  bool reached(const GameState& turn_start) const noexcept;
};

struct SearchParams {
  // ---- D2: multi-move turns (undecided) --------------------------------------
  TurnSplit split = TurnSplit::BoardMove;  ///< placeholder: D2 is undecided
  ActionPrior action_prior = ActionPrior::SumLogits;  ///< whole-turn: placeholder
  /// Whole-turn: the most actions listed per node (list_legal_actions' cap).
  /// Every listed action is stored with its node (its moves and an edge), so
  /// this bounds a node's memory: at 4,096 a node of a 100-board position held
  /// about 1.5 MB (review B). Placeholder 256 (build-log spec issue).
  std::size_t max_actions = 256;
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
  /// 6m (register I10): which signal is a node's variance for E2, E3 and E5's
  /// optional term. The register's default reading: the categorical.
  UncertaintySource uncertainty_source = UncertaintySource::Categorical;

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

  // ---- H2, E9, H41: the in-tree solver - the proof channel's switch ---------------
  /// THE PROOF CHANNEL'S SWITCH in the search (register H41, ruled: "the
  /// search's own solver is part of the proof channel: game-end scoring stays
  /// on; solved-status propagation and export are off until the gate passes,
  /// the search's soundness fixes land with tests, and the proof channel is
  /// signed off"; spec §11.1 item 2; H23). On, it propagates proven bounds up
  /// the tree - nodes marked solved, siblings pruned, secure-child selection
  /// and the other decisions taken from proven bounds (a proven child's exact
  /// Q, the visit threshold, proven-lost children ineligible, a proven win
  /// played, a solved root ending the search). OFF by rule until those three
  /// conditions hold. Off, a game-end position (C5 at a turn start, by E4's
  /// exhaustive detector; I11's cap) is still scored as its result - ordinary
  /// MCTS - and nothing else is proven. Export is proof_store, below.
  bool proof_channel = false;
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
  /// The optional term (0: off) - the child's uncertainty signal
  /// (uncertainty_source: the categorical's variance by default).
  double lcb_categorical_weight = 0.0;
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
  /// The node budget: run() stops, between batches, once the graph holds
  /// this many nodes (not counted from the call: the graph's size).
  std::uint64_t nodes = std::numeric_limits<std::uint64_t>::max();
  /// The memory budget: run() stops, between batches, once memory_bytes()
  /// reaches it (one batch may pass it: at most batch_size leaves expanded).
  std::uint64_t bytes = std::numeric_limits<std::uint64_t>::max();
  /// A deadline in wall-clock seconds from the call, checked between batches;
  /// 0: none. Not deterministic - for tools and match play, never training.
  double seconds = 0;
};

/// Why run() stopped.
enum class StopReason : std::uint8_t {
  Playouts = 0,  ///< Limits::playouts reached
  Evaluations,   ///< Limits::evaluations reached
  Finished,      ///< the root is terminal or solved (or stuck): nothing to search
  Exhausted,     ///< no simulation could be made
  Nodes,         ///< Limits::nodes reached
  Memory,        ///< Limits::bytes reached
  Deadline,      ///< Limits::seconds passed
};
const char* stop_reason_name(StopReason reason) noexcept;

// ---------------------------------------------------------------------------
// The evaluator
// ---------------------------------------------------------------------------

/// One leaf to evaluate: everything the network takes from it is in the
/// expansion - per-board and per-square features, the candidate moves the
/// policy is over (expansion->moves(), in order), the move graph, typed
/// relations, W(k)/E(k). Pointers are valid until the batch is delivered.
struct EvalRequest {
  const GameState* state = nullptr;
  /// Its global inputs include D8's - the acting player
  /// (expansion->acting_player()) and the submit action
  /// (expansion->submit_possible(), submit_legal()) - and A6's turn state: the
  /// mandatory and optional boards and the moves made in the action
  /// (expansion->mandatory_remaining(), optional_playable(), moves_made()).
  const Expansion* expansion = nullptr;
  /// The state's node key - not the request's identity, which is (key,
  /// moves_made, epoch) with the state compared exactly: equal states with
  /// different counts share a key (A6).
  std::uint64_t key = 0;
  std::uint64_t epoch = 0;  ///< the refresh epoch (§7.9)
  /// The moves made in the leaf's action (A6; = expansion->moves_made(); 0: a
  /// turn start) - an input of the network's that no state records, so leaves
  /// with equal states and different counts are evaluated, and cached, apart.
  int moves_made = 0;
};

/// The network's answer for one leaf.
struct Evaluation {
  /// One logit per candidate move (expansion->moves().size()).
  std::vector<float> logits;
  /// D8: the submit action's logit (under the ply split, read from the pooled
  /// global registers), used when the leaf's submit is possible; NaN - none given -
  /// stands for SearchParams::submit_share. Must not be infinite.
  float submit_logit = std::numeric_limits<float>::quiet_NaN();
  /// D3: the WDL head - the value search uses. Probabilities of a win, a draw
  /// and a loss for the leaf's acting player (kWdl, in that order; normalised
  /// by the search - non-negative, finite, not all zero). The leaf's value is
  /// W - L and its draw probability D (wdl_value).
  std::array<float, kWdl> wdl{};
  /// D3: the 51-bin categorical over the root search value - probabilities over
  /// kValueBins bins (normalised by the search), from the leaf's acting player;
  /// empty: none given. Its variance is the node's uncertainty signal under
  /// UncertaintySource::Categorical (0 without one); its mean is never backed up.
  std::optional<std::array<float, kValueBins>> value;
  /// D3: the error head's output - the predicted squared gap between the value
  /// and the short-term value, >= 0; NaN: none given. The node's uncertainty
  /// signal under UncertaintySource::ErrorHead (0 without one).
  float error = std::numeric_limits<float>::quiet_NaN();
  /// D4: the moves-left head's output, in the network's unit, >= 0; NaN: none
  /// given. Recorded on the node, never backed up (see VALUES above).
  float moves_left = std::numeric_limits<float>::quiet_NaN();
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

/// A WDL head's value (normalised first): q = (W - L) / (W + D + L) and the
/// draw probability d = D / (W + D + L). Throws std::invalid_argument for a
/// negative, non-finite or all-zero one.
struct WdlValue {
  double q = 0;
  double d = 0;
};
WdlValue wdl_value(const std::array<float, kWdl>& wdl);

/// A WDL with no draw whose value is `v` (clamped to [-1, 1]):
/// {(1 + v) / 2, 0, (1 - v) / 2} - the built-in material evaluator's, and the
/// tests'. Throws std::invalid_argument for NaN.
std::array<float, kWdl> wdl_of(double v);

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

/// Built-in evaluators, for tests, invariants and the tools. Each gives the
/// WDL and the categorical (no error head, no moves-left):
///  * uniform: every logit 0; WDL {1, 1, 1} (value exactly 0, draw 1/3) and the
///    categorical's mass on 0;
///  * random: logits, a WDL and a value distribution pseudo-random but a
///    function of (seed, node key, epoch) only - so evaluation stays a function
///    of the state;
///  * material: value v = tanh(material balance / 10) over every timeline's
///    head board, from the player to move - WDL wdl_of(v), categorical
///    two_hot(v); capture logits 0.5 x the captured value.
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
  double eval_variance = 0;   ///< the child's uncertainty signal (uncertainty_source; 0 if none)
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
// The evaluation cache: evaluations keyed by the state, the moves made in its
// action (A6) and the refresh epoch (C11, C14, §8.7). It has no lock, like
// the search (single-threaded, H15): the searches that share one (C14's
// fresh search per move) must be run one at a time - the caller's duty;
// nothing checks it.
// ---------------------------------------------------------------------------

class EvalCache {
 public:
  /// At most `capacity` entries and about `max_bytes` bytes - each entry
  /// counted for what it alone keeps alive once its search is gone: the
  /// map's node, its logits, its state's list of timelines and the boards
  /// (with their timeline objects) the step into the state made (entry_bytes;
  /// checked against the heap in the tests).
  explicit EvalCache(std::size_t capacity = std::size_t{1} << 16,
                     std::size_t max_bytes = std::size_t{256} << 20)
      : capacity_(capacity), max_bytes_(max_bytes) {}
  /// The cached evaluation of `state`, with `moves_made` moves made in its
  /// action (A6: the network sees the count), at `epoch` - every part compared
  /// exactly (keys may collide) - or nullptr.
  const Evaluation* find(std::uint64_t key, int moves_made, std::uint64_t epoch, const GameState& state,
                         bool compare_origins) const;
  /// Stores it (unless full); `new_boards`: the boards the step into the
  /// state made - its own, the rest shared with its predecessors.
  void insert(std::uint64_t key, int moves_made, std::uint64_t epoch, const GameState& state,
              const Evaluation& evaluation, std::size_t new_boards = 1);
  /// Removes every entry whose state has fewer than `boards` boards, and
  /// returns how many. Every move adds a board and none is ever removed, so
  /// such a state is behind a root of `boards` boards for good: no search
  /// from that root, or after it in the same game, can reach it again. (A
  /// cache shared by searches of different games would lose live entries.)
  /// The search calls it with its root's board count on creation and on
  /// every re-root: the cache's eviction, deterministic.
  std::size_t purge_fewer_boards(std::size_t boards);
  /// What an entry is counted for (see the constructor).
  static std::size_t entry_bytes(const GameState& state, const Evaluation& evaluation,
                                 std::size_t new_boards = 1) noexcept;
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
    int moves_made;  // A6: the moves made in the state's action
    std::uint64_t epoch;
    Evaluation evaluation;
    std::size_t bytes;  // entry_bytes, as counted on insertion
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
  std::uint64_t batch_duplicates = 0;  ///< a leaf whose state and count (A6) another leaf of the batch has
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
  std::uint64_t cache_purged = 0;      ///< cache entries behind the root, purged (EvalCache)
  StopReason last_stop = StopReason::Playouts;  ///< why the last run() stopped
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
  double draw = 0;      ///< weighted sum of backed-up draw probabilities (D3)
};

struct NodeView {
  const GameState* state = nullptr;
  std::uint64_t key = 0;
  Colour mover = Colour::White;
  int moves_made = 0;  ///< A6: the moves made in its action - part of its identity
  bool turn_start = true;  ///< moves_made == 0
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
  double draw = 0;           ///< weighted sum of backed-up draw probabilities (D3)
  double eval_mean = 0;      ///< its evaluation's value, W - L (D3)
  double eval_draw = 0;      ///< ... and draw probability
  double eval_variance = 0;  ///< ... and uncertainty signal (uncertainty_source)
  float eval_error = std::numeric_limits<float>::quiet_NaN();       ///< the error head's output (NaN: none)
  float eval_moves_left = std::numeric_limits<float>::quiet_NaN();  ///< moves-left (NaN: none)
  bool evaluated = false;    ///< it has an evaluation (eval_*)
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
  double q = 0;  ///< the backed-up value, W - L (D3)
  int lo = -1;  ///< bounds; a child through an unknown edge proves nothing (lo -1)
  int hi = 1;
  NodeId child = kNoNode;
  /// D3: the backed-up draw probability (0 before a visit) - with q, the
  /// child's search WDL: W = (1 - d + q) / 2, L = (1 - d - q) / 2.
  double d = 0;
  /// The child node's own evaluation, the network's (I13's and I15's memo; the
  /// root player's view): whether it has one; its WDL (win, draw, loss); its
  /// moves-left and error head's outputs (NaN: none given).
  bool evaluated = false;
  std::array<double, kWdl> network_wdl{};
  double network_moves_left = std::numeric_limits<double>::quiet_NaN();
  double network_error = std::numeric_limits<double>::quiet_NaN();
};

class Search {
 public:
  /// A search from `root`, with `moves_made` moves already made in the root's
  /// action (register A6; 0: a turn start - the whole-turn split requires it).
  /// Throws std::invalid_argument for inconsistent parameters (proof_store
  /// among them), a negative count, or a count of 0 where the root's
  /// obligations are met (a turn start's present lies on the mover's colour,
  /// T5 - such a root's action has moves).
  explicit Search(GameState root, SearchParams params = {}, int moves_made = 0,
                  std::shared_ptr<EvalCache> cache = nullptr);
  /// Before A6 the third argument was a flag, "turn start"; a bool is no
  /// count, so passing one is a compile error rather than a count of 0 or 1.
  Search(GameState root, SearchParams params, bool turn_start, std::shared_ptr<EvalCache> cache = nullptr) = delete;
  ~Search();
  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;
  Search(Search&&) noexcept;
  Search& operator=(Search&&) noexcept;

  const SearchParams& params() const noexcept;
  const SearchStats& stats() const noexcept;
  const EvalCache* cache() const noexcept;

  // -- running -----------------------------------------------------------------

  /// Runs simulations until `limits` (playouts and evaluations counted from
  /// this call; nodes and bytes the graph's) or until the root is solved or
  /// terminal: gather, evaluate, deliver. Returns why it stopped (also
  /// stats().last_stop).
  StopReason run(Evaluator& evaluator, const Limits& limits);
  /// The graph's memory, estimated: every node with its edges, its actions'
  /// moves and its state's own heap (counted as EvalCache::entry_bytes counts
  /// a state), and the MCGS table - checked against the heap in the tests.
  /// The evaluation cache is not included (it has its own ceiling).
  std::size_t memory_bytes() const noexcept;
  /// The evaluation cache, for another search to share (C14's self-play: a
  /// fresh search per move, the cache carried); null if none.
  std::shared_ptr<EvalCache> shared_cache() const noexcept;

  // -- the step API (the C API's external evaluator) ----------------------------

  /// Selects up to `max_leaves` leaves needing evaluation (virtual loss on each
  /// path), completing on the way every simulation that needs none (terminal
  /// and solved nodes, cache hits, MCGS corrections) as long as fewer than
  /// `max_playouts` simulations have been completed or gathered in this call.
  /// Returns the number of evaluations wanted - requests().size(): one per
  /// distinct (state, moves made in its action) among the leaves (a tree may
  /// hold one in two nodes; one state with two counts is two inputs, A6, and
  /// asked twice). Precondition: none is waiting. Exception-safe: if anything throws
  /// (std::bad_alloc included), the batch is abandoned and no virtual loss is
  /// left before the exception leaves.
  std::size_t gather(std::size_t max_leaves,
                     std::uint64_t max_playouts = std::numeric_limits<std::uint64_t>::max());
  /// The evaluations wanted, in order.
  std::span<const EvalRequest> requests() const noexcept;
  /// Their evaluations, one per request, in order: expands the leaves, backs
  /// up, releases the virtual loss. Throws std::invalid_argument (releasing
  /// the virtual loss, nothing backed up) for a malformed evaluation. If
  /// anything else throws part-way (std::bad_alloc), the leaves already
  /// delivered stay delivered and the rest of the batch is abandoned.
  void deliver(std::span<const Evaluation> evaluations);
  /// Releases the waiting leaves' virtual loss without backing anything up.
  void abandon() noexcept;

  // -- the root ---------------------------------------------------------------------

  NodeId root() const noexcept;
  const GameState& root_state() const noexcept;
  /// The moves made in the root's action (A6); root_turn_start() is its being 0.
  int root_moves_made() const noexcept;
  bool root_turn_start() const noexcept;
  /// The root is terminal (its game is over) or solved.
  bool root_terminal() const noexcept;
  bool root_solved() const noexcept;
  /// The root's instantiated children (in prior order).
  std::vector<RootChild> root_children() const;
  /// E5/E9: the child to play, by index into root_children(). Throws
  /// std::invalid_argument if the root has no visited child.
  Choice choose() const;
  /// Self-play: a child sampled from the visit distribution at temperature t
  /// (t <= 0: the most visits, ties to the lower index), with `random` a
  /// uniform number in [0, 1). Throws std::invalid_argument for a temperature
  /// that is not finite or a `random` outside [0, 1].
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
  /// the whole-turn split advance_moves(moves). Atomic: the whole action is
  /// checked first (every move pseudo-legal where it is made, the submission
  /// legal: T5, C4) and std::invalid_argument thrown with nothing changed if
  /// it fails. From a turn start an action has at least one move; from a
  /// state during an action, no moves means submitting it.
  void advance_action(std::span<const Move> moves);
  /// Re-roots on `target`, a state one decision from the root (a fresh root if
  /// the tree does not hold it), with `moves_made` moves made in its action as
  /// for the constructor (a move: the root's count + 1; a submission or a
  /// whole action: 0).
  void advance_to(const GameState& target, int moves_made);
  /// The flag it took before A6 (see the constructor's): a compile error.
  void advance_to(const GameState& target, bool turn_start) = delete;
  /// The refresh epoch (§7.9): evaluations are cached per (state, moves made in
  /// its action, epoch) (A6).
  std::uint64_t epoch() const noexcept;
  /// Throws std::invalid_argument while a batch is waiting (its evaluations
  /// belong to the epoch they were asked under): deliver or abandon it first.
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
