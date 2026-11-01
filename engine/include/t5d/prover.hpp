// t5d/prover.hpp - the proof channel (design spec §11.1, "The redesign after
// 6b (CHECKPOINT 3)", part 2; register M14, H1, H2, H15, H17, H18, H19, H37,
// H39): depth-limited df-pn over the semimove-decomposed turn tree, with
// incrementally completed turns. engine/README.md "Stage E8" sets out the
// design in full; in brief:
//
// THE QUESTION. At a root - a turn start; its side to move the ATTACKER A -
// does A force checkmate within `depth` half-turns (5: A, D, A, D, A; M15)
// by its FORCING CLASS of actions, whatever the defender D does? The answer
// is PROVEN (a real-game fact: Win(k), k <= depth, exported to the solved
// store at once, H19), DISPROVEN (no such mate within the class: recorded in
// the disproof store, with the class) or UNKNOWN (the budget ran out first).
// Never "mate" on an exhausted budget (H37): a node is proven only from exact
// sub-results - E4's exhaustive mate verdicts, or every defence refuted.
//
// THE CLASS (ProverOptions; the spec defines no forcing set, so each part is a
// parameter with a placeholder default). A's actions begin with a SEED -
// Checks (default: a checking move by the scanner's definition (b): a piece
// of A's on a board the move creates, that board passed, can capture a royal)
// / Forcing (the scanner's forcing set: captures, checks, branch threats) /
// All - and are finished by a COMPLETION: Canonical (default: one action per
// seed, the first legal completion in a fixed preference order) / Every
// (every legal completion). D gets EVERY legal action. With All and Every the
// question is exactly "mate within depth half-turns".
//
// THE TREE. Turn nodes (turn starts) and semimove nodes (partial actions:
// one of the mover's playable boards at a time, "play nothing here" or one
// move from it). A's nodes are OR nodes, D's AND nodes; a SUBMIT edge ends a
// half-turn. Under the rules where move order does not matter beyond the
// creations' order (Q4 "moves after the shift", count-rule activity,
// Rewinds, no play activation, castling on optional boards: decomposes()),
// a turn is two phases - the non-branching moves board by board in a fixed
// order, then branches in creation order and the submission - which holds
// every legal action exactly once; otherwise a turn node's children are E4's
// listing of the legal actions. Dead partial actions are cut where they
// arise: a STANDING CAPTURE (a royal of the mover's capturable from one of
// the opponent's playable boards - it lasts to the end of the action) kills a
// node at once (asked of incremental royal-attack maps, t5d/attacks.hpp; under
// Q18's alternative only a capture that crosses no void stands); a pass on a
// mandatory board is offered only where a later jump onto it, a branch from
// it or a relief of the present could make it good, and a partial action
// dies once such a pass is beyond help. An AND turn node is
// first asked E4's question: a quick escape search, then E4's hypercuboid
// detector (checkmate: proven; stalemate: disproven; a legal action: its
// successor searched first).
//
// THE SEARCH. Plain df-pn (Nagai) with proof and disproof numbers and
// thresholds, an explicit tree per thread, and a shared lockless
// transposition table of (pn, dn) ESTIMATES for turn nodes (5D Zobrist key
// with the ruleset ID, H18; the role; the depth left). Solved facts never live
// only there (§11.3): proofs go to the solved store, disproofs to the
// disproof store, the moment they are found, both compared in full on a key
// match. GHI does not arise (H39). With more than one thread (M13: 5), every
// thread searches the root over a tree of its own, sharing the TT and the
// stores; a helper breaks ties by a per-thread hash and widens its thresholds
// (1 + epsilon), and the first thread to decide the root stops the others.
// One thread with a node budget is deterministic (H15).
//
// NODES. One unit for all the work: a node expanded, a move applied, an
// attack-map query, a node of an E4 search.
//
// STAGE E9 (verification v3, spec 11.4: Hunt, H43, and Guard, H42, are built
// on this prover):
//  * PROVE-ONLY (ProverOptions::prove_only, Hunt): the search stops at its
//    budget and claims nothing - a root it exhausts is reported Unknown
//    (`exhausted`), never Disproven; it neither reads nor writes the caller's
//    disproof store (no disproof is recorded or shared, none from outside
//    drives its pruning) - the disproofs its own search finds are kept in a
//    private memo, emptied at each search, so that a transposition it has
//    exhausted is not searched again. Its thresholds are df-pn's usual two:
//    measured against proof-number thresholds alone (a child's disproof
//    threshold infinite, prove_only_pn_thresholds), the usual ones found
//    every mate the full search found, at the same cost, where proof numbers
//    alone took more nodes and missed one of 26 (tests/test_certificate.cpp,
//    prover_prove_only_finds_the_full_searchs_mates).
//  * THE LITERAL CLASS (seeds All, completion Every) enters the attacker's
//    semimove tree once, as the defender's: every legal action exactly once
//    (where turns decompose), not once per its first move.
//  * THE LAST-TURN CHECK FILTER (check_filter; completion Every): checkmate
//    needs check (R1), so at the attacker's last half-turn a partial action
//    that can no longer leave the defender in check is cut. From incremental
//    attack maps at the turn start, a move is CHECK-CAPABLE when, played
//    alone, it gives check or makes a line stop at void where another of the
//    attacker's moves could make a board (a line through several new boards
//    is opened by the first of them); a move the turn start does not offer
//    (a jump turned into a branching J6 twin mid-action) is check-capable;
//    and while any non-branching jump exists, a check-capable branch is
//    assumed left to make - its twin lands on a new timeline the maps never
//    saw. A partial action is cut when no check-capable move is in it and
//    none is left to make. (Before the review's fix the twins were not
//    counted: a twin's mate was cut - a false disproof.) The check-capable
//    boards come first in the turn's order. Off where attack maps recompute
//    (Q8, Q18, Q20's alternatives) and where a capture stands already - and
//    OFF BY DEFAULT: on real positions it cut little and cost more than it
//    saved (ProverOptions::check_filter). Whatever the filter, the
//    defender's turn at the horizon is decided not in check without E4.
//  * THE PROOF'S LINE (Proven): the attacker's proving actions and the
//    defender's longest defences, to the mate; and on request its
//    CERTIFICATE - every attacker position this prover proved by expanding
//    it, in any thread and any of its searches, with its proving action
//    (recorded as each proof is exported), so that a proof read back from the
//    solved store is certified too - which check_certificate
//    (t5d/certificate.hpp) verifies against E4 alone. A proof the prover
//    only read from a store it was given (another prover's) cannot be
//    certified: the check says which position is missing.
//  * AN ATTACKER WITH NO LEGAL ACTION: an attacker's turn is first asked
//    E4's question - a quick search for one legal action, then E4's
//    exhaustive detector where it finds none - so a mated or stalemated
//    attacker is decided at once, not by exhausting its semimove tree.

#ifndef T5D_PROVER_HPP
#define T5D_PROVER_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/scanner.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"

namespace t5d {

/// Which moves A's forcing actions may begin with (PLACEHOLDER default;
/// spec issue: §11.1 and §15.6 define no forcing set for the proof channel).
enum class SeedClass : std::uint8_t {
  /// Checking moves, the scanner's (b) (dlshogi's tsume rule: checks only).
  Checks = 0,
  /// The scanner's forcing set: captures, checks, branch threats (and
  /// promotions if promotions_forcing).
  Forcing,
  /// Every pseudo-legal move.
  All,
};

/// How a seed becomes A's action (PLACEHOLDER default; same spec issue).
enum class CompletionClass : std::uint8_t {
  /// One action per seed: the first legal completion in the preference
  /// order (engine/README.md "How a turn is completed incrementally").
  Canonical = 0,
  /// Every legal completion.
  Every,
};

const char* seed_class_name(SeedClass c) noexcept;
const char* completion_class_name(CompletionClass c) noexcept;

struct ProverOptions {
  /// Half-turns: the attacker's mate must come within this many (5, M15).
  int depth = 5;
  SeedClass seeds = SeedClass::Checks;
  CompletionClass completion = CompletionClass::Canonical;
  /// Forcing seeds: the scanner's branch-threat definition and values (its
  /// placeholders, t5d/scanner.hpp), and the quietness order of completions.
  PieceValues values;
  BranchThreat branch_threat = BranchThreat::ArrivalCapture;
  int branch_threat_value = 300;
  bool promotions_forcing = false;
  /// The canonical completion's depth-first search: at most this many steps
  /// per seed (PLACEHOLDER: the scanner's fallback cap); beyond it the seed
  /// has no action in the class.
  std::uint64_t completion_max_nodes = 2000;
  /// An AND turn node's quick escape search before E4's detector.
  std::uint64_t escape_probe_nodes = 256;
  /// E5a's fast tiers (t5d/legal.hpp) at semimove nodes: moves proved
  /// illegal are not made.
  bool legality_tiers = false;
  /// Turns that do not decompose: E4's listing of the legal actions, at most
  /// this many (more: the node cannot be proven - AND - or disproven - OR).
  std::size_t listing_cap = 4096;
  /// Transposition-table entries (estimates; a power of two).
  std::size_t tt_entries = std::size_t{1} << 18;
  /// The memory ceiling of each E4 search.
  std::size_t detector_max_bytes = std::size_t{256} << 20;
  /// Each thread's tree: at most this many nodes (then the search stops:
  /// unknown).
  std::size_t max_tree_nodes = std::size_t{2} << 20;
  /// Search threads (M13's budget: 5). 1: deterministic (H15).
  int threads = 1;
  /// Record every turn node's disproof in the disproof store (else the
  /// root's only).
  bool record_all_disproofs = true;
  /// Prove-only (Hunt, H43; see the header comment).
  bool prove_only = false;
  /// Prove-only's thresholds: df-pn's usual two (false, the default: they found
  /// more mates for fewer nodes) or proof numbers only (true; the search still
  /// claims no disproof either way).
  bool prove_only_pn_thresholds = false;
  /// The last-turn check filter (see the header comment). OFF by default.
  /// Stage E9's first filter missed J6 twins (a false disproof, found by the
  /// review; fixed, with its position as a test) - and was measured off
  /// before that: on 6b's first 120 roots (Guard, 50 ms, 5 threads) it
  /// cut little (no partial action under 50 boards, 125 at 50-99) and cost
  /// more than it saved (depth 1 completed at 82.4% with it, 83.5% without,
  /// under 50 boards): building it advances a map per move, and in positions
  /// of several timelines nearly every move is check-capable, a new board's
  /// passed copy sending lines across the timelines beside it.
  bool check_filter = false;
  /// Collect the proof's certificate (tools and tests): every attacker
  /// position proven by expansion, kept for the prover's lifetime.
  bool record_certificate = false;
  /// Ask E4's question at the attacker's turns too (one legal action, by a
  /// quick search, else E4's exhaustive detector): an attacker with no legal
  /// action is decided at once. Efficiency only - the answers are the same
  /// either way (a test checks it); off reproduces stage E9's first prover,
  /// for measurement.
  bool attacker_probe = true;
};

/// Prover::tree_actions: the actions distinct by successor, and how many
/// paths of the tree reached a submission (more than actions.size(): some
/// action was enumerated twice).
struct TreeActions {
  std::vector<BuiltAction> actions;
  std::size_t paths = 0;
  /// Each path dropped as a repeat: the kept path's moves, and its own.
  std::vector<std::pair<std::vector<Move>, std::vector<Move>>> repeated;
};

/// Prover::check_filter_view: the last-turn check filter at a turn start
/// (the attacker to move), as the prover builds it. `applies` false: the
/// filter does not apply there (maps recompute, a capture stands). Else, per
/// move of `moves` (generate_moves' order): check-capable, and capable
/// through a void stop alone; and whether a check-capable branch is assumed
/// left (J6's twins, or a capable branching move).
struct CheckFilterView {
  bool applies = false;
  MoveList moves;
  std::vector<std::uint8_t> capable;
  std::vector<std::uint8_t> by_void;
  bool branch = false;
};

// ---------------------------------------------------------------------------
// Proof certificates
// ---------------------------------------------------------------------------

/// One proven position of the attacker's (its turn start) and the action that
/// proves it: mate within `distance` half-turns.
struct CertificateStep {
  GameState position;
  std::vector<Move> action;  // in a playable order
  int distance = 0;
};

/// A proof's certificate: for each proven position of the attacker's met in
/// the proof, the proving action (the least distance kept). Positions are
/// matched by Zobrist key and a full comparison.
class ProofCertificate {
 public:
  void add(const GameState& position, std::vector<Move> action, int distance);
  const CertificateStep* find(const GameState& position) const;
  std::size_t size() const noexcept { return steps_.size(); }
  const std::vector<CertificateStep>& steps() const noexcept { return steps_; }

 private:
  std::vector<CertificateStep> steps_;
  std::unordered_multimap<std::uint64_t, std::size_t> index_;
};

/// Whether the turn decomposes into semimoves exactly under `rules` (see the
/// header comment).
bool decomposes(const RuleOptions& rules) noexcept;

struct ProveBudget {
  std::uint64_t max_nodes = std::numeric_limits<std::uint64_t>::max();
  std::optional<std::chrono::nanoseconds> max_time;
  /// Stage E10: the caller's stop - a budget stop like the clock's
  /// (ProverStats::stopped_by_cancel), read where the budget is spent; the
  /// search stops at its next node, claiming nothing (H37). The verification
  /// ladder (t5d/ladder.hpp) sets it when the decision a job serves ends: no
  /// job outlives its decision. Null: none.
  const std::atomic<bool>* cancel = nullptr;
  static ProveBudget nodes(std::uint64_t n) noexcept {
    ProveBudget b;
    b.max_nodes = n;
    return b;
  }
  static ProveBudget time(std::chrono::nanoseconds t) noexcept {
    ProveBudget b;
    b.max_time = t;
    return b;
  }
};

// ---------------------------------------------------------------------------
// Disproofs ("no mate within d", with the class)
// ---------------------------------------------------------------------------

/// Every option that shapes the attacker's class, hashed: which seeds and
/// completions, and what chooses them - the canonical completion's step cap
/// and quietness values, the branch-threat definition, promotions as forcing,
/// E5a's tiers, the check filter. Two provers whose options differ in any of
/// them never share a disproof (the review's finding: a disproof is a fact
/// about its class, and only about it).
std::uint64_t disproof_shape(const ProverOptions& options) noexcept;

/// A disproof's class: whose turn it is and A's forcing class. `attacker`
/// true: "the side to move cannot force mate within the depth by the class";
/// false: "the side to move is not mated within the depth by the opponent's
/// class". `shape`: disproof_shape() of the options that found it.
struct DisproofClass {
  bool attacker = true;
  SeedClass seeds = SeedClass::Checks;
  CompletionClass completion = CompletionClass::Canonical;
  std::uint64_t shape = 0;
  friend bool operator==(const DisproofClass&, const DisproofClass&) noexcept = default;
};

/// Where disproofs go the moment they are found: keyed by position - the
/// Zobrist key with the ruleset ID (H18) and a full comparison on a match -
/// and class, holding the greatest depth disproven. Safe for concurrent use:
/// 64 shards by key, each behind its own ShardLock (t5d/scanner.hpp).
class DisproofStore {
 public:
  DisproofStore();
  ~DisproofStore();
  DisproofStore(const DisproofStore&) = delete;
  DisproofStore& operator=(const DisproofStore&) = delete;
  /// The greatest depth disproven for `position` in `cls`; -1 if none.
  int find(const GameState& position, const DisproofClass& cls) const;
  void put(const GameState& position, const DisproofClass& cls, int depth);
  std::size_t size() const;
  void clear();

 private:
  struct Shards;
  std::unique_ptr<Shards> shards_;
};

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

enum class ProofStatus : std::uint8_t { Unknown = 0, Proven, Disproven };
const char* proof_status_name(ProofStatus s) noexcept;

struct ProverStats {
  std::uint64_t nodes = 0;            // the budget's unit (every thread's)
  std::uint64_t tree_nodes = 0;       // nodes made, every thread
  std::uint64_t turn_nodes = 0;
  std::uint64_t expansions = 0;
  std::uint64_t applies = 0;          // moves applied
  std::uint64_t attack_queries = 0;   // standing-capture and C4 tests
  std::uint64_t dead = 0;             // partial actions cut by a standing capture
  std::uint64_t passes_cut = 0;       // passes on mandatory boards not offered
  std::uint64_t tier_cut = 0;         // moves E5a proved illegal
  std::uint64_t seeds = 0;            // seeds of A's class
  std::uint64_t completions = 0;      // canonical completions found
  std::uint64_t completion_failures = 0;
  std::uint64_t completion_steps = 0;
  std::uint64_t mate_tests = 0;       // AND turn nodes asked E4's question
  std::uint64_t escape_probe_hits = 0;
  std::uint64_t checkmates = 0;
  std::uint64_t stalemates = 0;
  std::uint64_t detector_nodes = 0;
  std::uint64_t listings = 0;         // turns that did not decompose
  std::uint64_t tt_hits = 0;
  std::uint64_t store_hits = 0;
  std::uint64_t proofs_exported = 0;
  std::uint64_t disproofs_recorded = 0;
  std::uint64_t check_cut = 0;        // partial actions cut by the last-turn check filter
  std::uint64_t not_in_check = 0;     // horizon turns decided not in check
  std::uint64_t attacker_tests = 0;   // attacker turns asked for one legal action first
  bool stopped_by_nodes = false;
  bool stopped_by_time = false;
  bool stopped_by_cancel = false;     // ProveBudget::cancel (stage E10)
  bool stopped_by_memory = false;     // E4's ceiling or the tree's
  double seconds = 0;
};

struct ProveResult {
  ProofStatus status = ProofStatus::Unknown;
  /// Proven: Win(k), the attacker mates within k half-turns.
  Proof proof;
  /// The depth asked.
  int depth = 0;
  /// Proven: the attacker's first action of the proof, in a playable order -
  /// the reporting thread's (see `line`); empty only where no thread's tree
  /// holds the root's children and no certificate holds its action (every
  /// thread read the root's proof from a store).
  std::vector<Move> first_action;
  /// The root's children in the main thread's tree - A's actions in the
  /// class (Canonical), or its seeds (Every; one child, the semimove tree,
  /// for the literal class); -1 if the root was not expanded (solved from a
  /// store, or no budget).
  int root_actions = -1;
  int threads = 1;
  /// Prove-only: the root was exhausted - no mate within the class - which
  /// prove-only does not claim (status stays Unknown).
  bool exhausted = false;
  /// Proven: the proof's line - the attacker's action, the defender's
  /// longest defence, ..., the attacker's mating action - from the tree of a
  /// thread that proved the root at the least distance: the first whose tree
  /// holds the whole line, else the longest line (Lazy SMP: a thread that
  /// read the root's proof from another thread's export has no tree beneath
  /// its root). `line_complete` false: it stops where a node was solved from
  /// a store (another thread's export, an earlier search's) and the tree
  /// holds no continuation. At depth 1 it is complete whenever the store did
  /// not hold the root's proof already: the thread that proved the root first
  /// searched it, and its line ends at the mated defender.
  std::vector<std::vector<Move>> line;
  bool line_complete = false;
  /// record_certificate, Proven: the prover's certificate so far (every
  /// thread's, every search's).
  std::shared_ptr<const ProofCertificate> certificate;
  /// The main thread's root children (an action, or a first move for
  /// completion Every), with their proof and disproof numbers - Hunt's soft
  /// priors are read from them.
  struct RootChild {
    std::vector<Move> moves;
    std::uint64_t pn = 1;
    std::uint64_t dn = 1;
  };
  std::vector<RootChild> root_children;
  ProverStats stats;
};

// ---------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------

class Prover {
 public:
  /// `proofs`: the solved store proofs are exported to and read from (none: a
  /// store of the prover's own); `disproofs` likewise.
  explicit Prover(const ProverOptions& options = {}, SolvedStore* proofs = nullptr,
                  DisproofStore* disproofs = nullptr);
  ~Prover();
  Prover(const Prover&) = delete;
  Prover& operator=(const Prover&) = delete;

  /// The question at `root` (a turn start). Rethrows the first exception a
  /// thread threw (std::logic_error for a contradiction in the rules code).
  ProveResult prove(const GameState& root, const ProveBudget& budget);
  /// The same within `depth` half-turns instead of options().depth (Guard's
  /// iterative deepening: one prover, its TT and stores, every depth).
  ProveResult prove(const GameState& root, const ProveBudget& budget, int depth);

  /// Empties the transposition table (not the stores).
  void clear();

  const ProverOptions& options() const noexcept;
  SolvedStore& proofs() noexcept;
  DisproofStore& disproofs() noexcept;

  // -- the pieces, for tools and tests (no budget) ---------------------------

  /// A's actions at `turn_start` in the class (seeds completed; Every: every
  /// legal completion of every seed), distinct by successor: tree_actions'
  /// as the attacker.
  std::vector<BuiltAction> attacker_actions(const GameState& turn_start);
  /// The canonical completion of `seed` (pseudo-legal at `turn_start`).
  std::optional<BuiltAction> canonical_completion(const GameState& turn_start, const Move& seed);
  /// The actions the prover's own tree holds at `turn_start` for the side
  /// to move as the ATTACKER (this prover's class; the literal class: the
  /// semimove tree entered once; with the check filter: what survives it at
  /// a last turn - the turn is searched as one with a half-turn left) or as
  /// the DEFENDER (its semimove tree and its witness; none where it is
  /// mated or stalemated): the production expansion (expand_turn,
  /// expand_semi), every node expanded, no budget.
  TreeActions tree_actions(const GameState& turn_start, bool attacker);
  /// The last-turn check filter at `turn_start` (tests): see CheckFilterView.
  CheckFilterView check_filter_view(const GameState& turn_start);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace t5d

#endif  // T5D_PROVER_HPP
