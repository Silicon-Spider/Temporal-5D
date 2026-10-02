// t5d/scanner.hpp - the tactical scanner (design spec §11.1, §11.2, §11.3;
// register H1-H6, H14, H15, H18, H19; experiment 6b, M8).
//
// A brute-force, node-capped alpha-beta search over FORCING actions, with a
// material-count evaluation (H6 v1). It is the verification layer's
// "differently wrong" auditor: no learned component, net-free move ordering.
// Rule IDs are the rulebook's (docs/rules/README.md).
//
// THE UNIT. A node is a TURN START (a position at the start of a player's
// turn); an edge is one whole legal action (C5: the moves of a turn, submitted
// - the present shifted, T5, every royal of the mover's safe, C4). Depth is
// counted in FORCING HALF-TURNS: one side's whole action (spec §15.6); the
// number of board moves along a line is reported beside it ("board-move
// depth", §8.2). Check status (R1), terminal verdicts (C5) and stand-pat live
// at turn starts, so terminal detection runs only at turn boundaries, by E4's
// exhaustive detectors (t5d/search.hpp). Within an action the moves are chosen
// by the forcing generator below (a seed move and a completion) or, in check,
// enumerated by E4. (The alternative unit, one board move per ply with the
// sign flipping at submission - D2's ply split - is not built: it would give
// the same tree with mid-action nodes, for which neither check status nor
// stand-pat is defined.)
//
// THE FORCING SET (§11.1 "defining quiet" - the spec defines no exact set; this
// is the engine's, each class a parameter). A move m of the side to move S,
// pseudo-legal at the turn start, is FORCING iff
//  (a) it captures (Move::capture(): within a board, across timelines,
//      through time, en passant); or
//  (b) it CHECKS: some piece of S's standing on a board m creates, that board
//      advanced by a pass (an unchanged copy with S to move again, as R1's
//      check status advances the opponent's boards), can capture a royal of
//      the opponent's - by check.hpp's royal-capture rules: royals on history
//      boards count (C2), branching captures count (R1). A deliberate
//      approximation of R1's status at the opponent's next turn start, which
//      passes only the active boards in the present: it may call a move
//      checking whose new board lies ahead of the present (C7), and it does
//      not see a check given only by a piece on a board m did not create
//      (a slide newly bridged through m's passed board, or a royal copied by
//      a branch and attacked from elsewhere); or
//  (c) it BRANCHES and the new timeline holds an IMMEDIATE THREAT: the spec
//      does not define one, so it is ScannerOptions::branch_threat - by
//      default the arriving piece, on the new timeline's board advanced by a
//      pass, can capture an enemy piece worth at least branch_threat_value; or
//  (d) optionally (promotions_forcing) it promotes.
// A FORCING ACTION is a legal action that begins with a forcing move (its
// SEED) and finishes the turn's obligations by the COMPLETION rule (a
// parameter: by default, on each mandatory board left, the quietest move -
// see Completion), falling back to the first legal finish E4's detector finds
// when the rule's moves leave a royal capturable or a board unplayable. A
// position is QUIET when its side to move has no forcing move.
//
// THE SEARCH, at a turn start s with d forcing half-turns left:
//  * the solved store first (H19): a proof found before is the answer;
//  * IN CHECK (R1): ALL legal actions, so that a mate found is a proof (H2),
//    in two stages. First up to evasion_probe SEEDS - pseudo-legal moves in an
//    evasion order (the TT's move, a capture of the checking piece, a royal's
//    move, a block of the check's line, other captures, the rest by
//    histories) - each made into a legal action by the completion rule
//    (built through the state machine and tested, C4; a seed it leaves
//    unsafe is passed over) and searched as soon as it is built: a cutoff
//    ends the node, resolved - a legal action scoring at least beta bounds
//    the node whatever the others do. No cutoff: E4's enumeration
//    (list_legal_actions) and the successors not yet searched. None,
//    exhaustively: checkmate. More than evasion_cap: the node is CAPPED - the
//    actions searched are an estimate, and the iteration is not fully
//    resolved. A Loss proof needs the enumeration exhaustive and every child
//    of both stages a proven win of the opponent's. At d = 0 only "is this
//    a mate?" is asked: one legal action from the seeds answers no (an
//    in-check leaf, an estimate); none of them, and E4's mate detector
//    decides (checkmate, or an in-check leaf);
//  * otherwise: stand-pat (the static eval; estimate channel only), then the
//    forcing actions in stages - captures (MVV-LVA), checks, branch threats -
//    each built when it is searched. No forcing action: a QUIET node (E4's
//    detector tells stalemate from a quiet position, quiet_stalemate_test).
//    At d = 0 the static eval.
// Negamax alpha-beta (fail-soft); no reductions, no null move, no futility, no
// ProbCut, no delta pruning (H5; the scanner's pruning waits on 6b, M5/M8):
// alpha-beta's own cutoffs are the only pruning, and stand-pat is the only
// estimate-channel rule.
//
// TWO CHANNELS (H2). Every result carries an estimate (a score) and, apart, a
// PROOF: Win(d) - the side to move mates within d half-turns, whatever the
// defence; Loss(d) - it is mated within d; Draw(0) - stalemate now (C13). A
// proof is built only from: E4's exhaustive verdicts at turn starts; "some
// child is a proven Loss" (Win); "EVERY legal action, enumerated exhaustively
// by E4, leads to a proven Win of the opponent" (Loss). So a node cap hit
// inside a proof line gives no proof (H37's spirit), stand-pat never yields
// one, and an estimate is never promoted. Proofs are exported to the solved
// store the moment they are found (H19) - the transposition table holds
// estimates only, so a proof never lives only in it.
//
// The TT is keyed by GameState::zobrist() - the 5D Zobrist key with the
// ruleset ID (H18) - with a fixed, deterministic replacement policy (always
// replace); move ordering is net-free (H4): MVV-LVA, killers, history and
// continuation history.
//
// THREADS (H15, H17). With one thread (ScannerOptions::threads, the default)
// the scanner is deterministic: with a node budget (no time limit) the same
// position, options and store give the same search, node for node - the
// synchronous debug mode. With more, LAZY SMP: every thread searches the
// same root by iterative deepening, and they share only the TT and the
// solved store. The TT is lockless (a slot holds its entry and the key
// XORed with it, so a slot being written reads as a miss) and holds
// estimates only; proofs go to the solved store the moment they are found
// (H19), which compares positions in full on a key match (§11.3's hashing
// trap) and refuses two kinds of proof for one position - so a thread's
// result is an estimate unless it is a proof, built as above, and the
// threads' proofs agree. Killers, history and continuation history are per
// thread (§11.3), carried from scan to scan until clear() (between games).
// Diversity: helper threads search staggered blocks of depths (ahead of the
// main thread) and perturb their order of quiet moves by a per-thread hash.
// Each helper searches its own copy of the root, rebuilt board by board, so
// that the threads do not contend on shared reference counts (a state that
// cannot be rebuilt from its boards is shared: stats.shared_roots). A
// thread that proves the root or exhausts its tree stops the others; the
// node budget is shared (each thread adds its nodes in batches of 256, so
// several threads may overrun it by a few hundred nodes each), and so is
// the clock. The result is the deepest iteration any thread completed.
//
// NODES. One node is one move applied by the scanner (GameState::apply), one
// royal-capture scan it makes (check status, C4's test, a move's check test),
// or one node of an E4 search it calls (that search's own unit: a move
// applied or an action tested) - so the budget counts all the work in one
// unit. Tree positions (turn starts visited) are counted apart.

#ifndef T5D_SCANNER_HPP
#define T5D_SCANNER_HPP

#include <array>
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
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/types.hpp"

namespace t5d {

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Piece values in centipawns for the v1 material eval (H6). The spec gives
/// none for 5D: P N B R Q take the chess convention (100, 300, 300, 500, 900);
/// U, D, S, W, C are PLACEHOLDERS (a project parameter table, to settle) - the
/// unicorn 250 and the dragon 200 (they cannot move within a board), the
/// princess 700 (a rook and a bishop in one piece, without the queen's 3- and
/// 4-axis lines), the brawn 125 (a pawn with more captures), the common king
/// 300 (a king's moves, not royal). Royal pieces (K, Y) are worth 0: they are
/// never captured, and mate is the search's.
struct PieceValues {
  std::array<int, kPieceKindCount> value{100, 300, 300, 500, 900, 0, 250, 200, 700, 125, 300, 0};
  int of(PieceKind k) const noexcept { return value[static_cast<std::size_t>(index_of(k))]; }
};

/// How a forcing action's seed is completed (its other moves - those the
/// turn's obligations need, T5, T6). A cheap rule: the spec leaves it open.
enum class Completion : std::uint8_t {
  /// On each mandatory board left (ascending timeline), a move within the
  /// board that captures nothing, of the least valuable piece (generator
  /// order among equals); else any move from it. Then C4; if a royal is left
  /// capturable or a board cannot be played, E4's first legal finish.
  Quiet = 0,
  /// The same, but the best capture on each board first (MVV-LVA).
  Greedy,
  /// E4's first legal finish, always.
  Detector,
};

/// What makes a branching move's new timeline hold an "immediate threat" (§11.1
/// names the class and defines nothing: a parameter, spec issue).
enum class BranchThreat : std::uint8_t {
  /// No such class (captures and checks only).
  None = 0,
  /// The arriving piece, on the new timeline's board advanced by a pass, can
  /// capture an enemy piece worth at least branch_threat_value.
  ArrivalCapture,
  /// Any piece of the mover's on that board can.
  AnyCapture,
};

/// Stand-pat (§11.1; 6b measures whether it is valid under 5D's obligations).
enum class StandPat : std::uint8_t {
  On = 0,
  Off,
  /// Off where the side to move has at least `stand_pat_bind_boards`
  /// mandatory boards (§15.6's 5-20% band: "disabled where obligations bind").
  OffWhereObligationsBind,
};

struct ScannerOptions {
  PieceValues values;

  // -- the forcing set (see the header comment) ----------------------------------
  /// (b): checking moves are forcing.
  bool checks = true;
  /// (c): PLACEHOLDER default - the arriving piece's capture threat.
  BranchThreat branch_threat = BranchThreat::ArrivalCapture;
  /// (c): the least value of the threatened piece: PLACEHOLDER (a minor piece).
  int branch_threat_value = 300;
  /// (d): promotions are forcing (the spec's list has none; off).
  bool promotions_forcing = false;
  /// How a seed is completed.
  Completion completion = Completion::Quiet;
  /// The most nodes E4's search for a seed's first legal finish may spend
  /// (the completion's fallback); beyond it the seed is dropped - an action
  /// left out of a forcing set, which only the estimate channel sees.
  std::uint64_t fallback_max_nodes = 2000;

  // -- the search -------------------------------------------------------------------------
  StandPat stand_pat = StandPat::On;
  int stand_pat_bind_boards = 2;
  /// In check: the most legal actions enumerated; beyond it the node is
  /// capped (searched over the first evasion_cap + 1 that E4 listed - an
  /// estimate; no Loss proof; the iteration not fully resolved).
  std::size_t evasion_cap = 512;
  /// In check, before E4's enumeration: the most seeds (pseudo-legal moves,
  /// in an evasion order) the scanner makes into legal actions by the
  /// completion rule and searches one at a time, so that a cutoff ends the
  /// node with no enumeration; at the horizon, the most it tries for one
  /// legal action before E4's mate detector (see the header comment). 0: E4
  /// first, always (E5b's behaviour).
  std::size_t evasion_probe = 64;
  /// At an interior node with no forcing action: E4's detector tells a
  /// stalemate (an exact draw) from a quiet position.
  bool quiet_stalemate_test = true;
  /// Iterative deepening's last depth, in forcing half-turns.
  int max_depth = 16;
  /// Transposition-table entries (a power of two; at least 1).
  std::size_t tt_entries = std::size_t{1} << 16;
  /// The memory ceiling of each E4 search the scanner calls (H37's, smaller).
  std::size_t detector_max_bytes = std::size_t{256} << 20;
  /// Measurement only (6b): classify every horizon leaf and stand-pat cutoff
  /// as quiet or not (has_forcing_move). Its work is not charged to the budget
  /// and not counted in the nodes (stats.measurement_nodes), so the search is
  /// the same with or without it.
  bool classify_leaves = false;
  /// Search threads (Lazy SMP, H17; see THREADS in the header comment). 1:
  /// the synchronous, deterministic mode (H15). The scanner's design budget
  /// is 5 (M13: 5 of the 8 cores per GPU).
  int threads = 1;
};

/// What a scan may spend (H3: a node cap, not a depth cap). A time limit makes
/// the scan depend on the machine; leave it out for the deterministic mode
/// (H15).
struct ScanBudget {
  std::uint64_t max_nodes = std::numeric_limits<std::uint64_t>::max();
  std::optional<std::chrono::nanoseconds> max_time;
  static ScanBudget nodes(std::uint64_t n) noexcept {
    ScanBudget b;
    b.max_nodes = n;
    return b;
  }
};

// ---------------------------------------------------------------------------
// Proofs and the solved store (H2, H19)
// ---------------------------------------------------------------------------

/// A game-theoretic fact about a position, for its side to move: exact, never
/// an estimate.
struct Proof {
  enum class Kind : std::uint8_t { None = 0, Win, Loss, Draw };
  Kind kind = Kind::None;
  /// Win: the side to move mates within `distance` half-turns (1: its action
  /// mates). Loss: it is mated within `distance` (0: checkmated now). Draw:
  /// 0 - stalemate now.
  int distance = 0;

  bool proven() const noexcept { return kind != Kind::None; }
  static Proof win(int d) noexcept { return Proof{Kind::Win, d}; }
  static Proof loss(int d) noexcept { return Proof{Kind::Loss, d}; }
  static Proof draw() noexcept { return Proof{Kind::Draw, 0}; }
  friend bool operator==(const Proof&, const Proof&) noexcept = default;
};

/// "win 3", "loss 0", "draw", "none".
std::string to_string(const Proof& proof);

/// The permanent store proofs are exported to the moment they are found
/// (H19), keyed by position - the Zobrist key with the ruleset ID (H18), and
/// a full comparison on a key match. Never lossy.
class SolvedStore {
 public:
  virtual ~SolvedStore() = default;
  virtual std::optional<Proof> find(const GameState& position) const = 0;
  /// Records `proof` (a shorter distance of the same kind replaces a longer
  /// one). Two proofs of different kinds for one position are a contradiction
  /// - a false proof somewhere - and throw std::logic_error.
  virtual void put(const GameState& position, const Proof& proof) = 0;
  /// Whether find and put may be called from several threads at once. A
  /// scanner with more than one thread serialises its calls to a store that
  /// says no.
  virtual bool concurrent() const noexcept { return false; }
};

/// A minimal in-memory solved store.
class MemorySolvedStore final : public SolvedStore {
 public:
  std::optional<Proof> find(const GameState& position) const override;
  void put(const GameState& position, const Proof& proof) override;
  std::size_t size() const noexcept { return entries_.size(); }
  /// Every entry, in insertion order.
  struct Entry {
    GameState position;
    Proof proof;
  };
  const std::vector<Entry>& entries() const noexcept { return entries_; }
  void clear() noexcept;

 private:
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> index_;
  std::vector<Entry> entries_;
};

/// The solved store the threads of a scan share (Lazy SMP, H17, H19): the
/// same contract as MemorySolvedStore - keyed by the Zobrist key with the
/// ruleset ID (H18), same_position() on a key match (§11.3's hashing trap),
/// proofs only, never lossy, a contradiction throws std::logic_error - and
/// safe for concurrent use: the positions are spread over shards by key,
/// each behind a reader-writer lock, so a probe never waits on a probe. A
/// scanner's own store is one of these.
class ConcurrentSolvedStore final : public SolvedStore {
 public:
  ConcurrentSolvedStore();
  ~ConcurrentSolvedStore() override;
  ConcurrentSolvedStore(const ConcurrentSolvedStore&) = delete;
  ConcurrentSolvedStore& operator=(const ConcurrentSolvedStore&) = delete;
  std::optional<Proof> find(const GameState& position) const override;
  void put(const GameState& position, const Proof& proof) override;
  bool concurrent() const noexcept override { return true; }
  std::size_t size() const;
  void clear();

 private:
  struct Shards;
  std::unique_ptr<Shards> shards_;
};

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// The leaves of one iteration's tree: the nodes whose value is the static
/// eval, and the terminal ones.
struct LeafStats {
  std::uint64_t horizon = 0;          // depth exhausted, not in check
  std::uint64_t horizon_in_check = 0; // depth exhausted in check (no mate: an estimate)
  std::uint64_t stand_pat_cutoffs = 0;
  std::uint64_t quiet = 0;            // no forcing action
  std::uint64_t terminal = 0;         // checkmate or stalemate, proven
  std::uint64_t solved = 0;           // answered by the solved store
  /// classify_leaves only: of the horizon leaves and stand-pat cutoffs, those
  /// whose side to move has no forcing move.
  std::uint64_t horizon_quiet = 0;
  std::uint64_t stand_pat_quiet = 0;

  std::uint64_t total() const noexcept {
    return horizon + horizon_in_check + stand_pat_cutoffs + quiet + terminal + solved;
  }
  /// The leaves that reached a quiet position or a resolved one.
  std::uint64_t reached_quiet() const noexcept {
    return quiet + terminal + solved + horizon_quiet + stand_pat_quiet;
  }
};

/// One iteration of iterative deepening.
struct IterationStats {
  int depth = 0;
  /// It finished within the budget.
  bool completed = false;
  /// An in-check node had more than evasion_cap legal actions: searched over
  /// a subset, so the tree was not fully resolved.
  bool capped = false;
  /// No node met the depth limit: every deeper iteration is the same tree.
  bool exhausted = false;
  int score = 0;
  Proof proof;
  /// Cumulative at the iteration's end (all iterations so far).
  std::uint64_t nodes = 0;
  std::uint64_t positions = 0;
  /// The most board moves along any line searched; along the principal line.
  int max_board_moves = 0;
  int pv_board_moves = 0;
  int pv_half_turns = 0;
  LeafStats leaves;
  double seconds = 0;  // cumulative wall time
};

struct ScanStats {
  std::uint64_t nodes = 0;
  std::uint64_t positions = 0;
  std::uint64_t applies = 0;
  std::uint64_t scans = 0;
  std::uint64_t detector_calls = 0;
  std::uint64_t detector_nodes = 0;
  /// detector_nodes by purpose: listing the legal actions in check; a
  /// completion's fallback (a seed's first legal finish); the mate test at
  /// an in-check horizon leaf; the stalemate test at a quiet node.
  std::uint64_t evasion_nodes = 0;
  std::uint64_t fallback_nodes = 0;
  std::uint64_t horizon_mate_nodes = 0;
  std::uint64_t stalemate_test_nodes = 0;
  /// The profile of the two exhaustive paths (6b, M12) and the fallbacks:
  /// in-check interior nodes, those that failed high (a cutoff), the E4
  /// listings made there and the actions they listed; the horizon mate tests
  /// and their checkmate verdicts; and the wall time of each kind of E4 call.
  std::uint64_t evasion_visits = 0;
  std::uint64_t evasion_fail_highs = 0;
  std::uint64_t evasion_lists = 0;
  std::uint64_t evasion_listed = 0;
  std::uint64_t horizon_mate_tests = 0;
  std::uint64_t horizon_mates = 0;
  std::uint64_t horizon_mate_nodes_on_mates = 0;  // of horizon_mate_nodes
  double evasion_seconds = 0;
  double horizon_mate_seconds = 0;
  double fallback_seconds = 0;
  /// The in-check stage (ScannerOptions::evasion_probe): seeds tried, the
  /// nodes and wall time spent making them into actions, the legal actions
  /// they made and searched (interior nodes), the interior nodes it resolved
  /// by a cutoff, and the horizon mate tests it answered (no mate).
  std::uint64_t probe_seeds = 0;
  std::uint64_t probe_nodes = 0;
  double probe_seconds = 0;
  std::uint64_t probe_actions = 0;
  std::uint64_t probe_cutoffs = 0;
  std::uint64_t horizon_probe_hits = 0;
  /// Lazy SMP: helper threads that searched the caller's root itself, shared,
  /// because it could not be rebuilt from its boards (see THREADS).
  std::uint64_t shared_roots = 0;
  std::uint64_t tt_hits = 0;
  std::uint64_t solved_hits = 0;
  std::uint64_t proofs_exported = 0;
  std::uint64_t completion_fallbacks = 0;
  std::uint64_t capped_nodes = 0;
  /// classify_leaves' work, not charged to the budget, and its wall time
  /// (for a caller timing the search without it).
  std::uint64_t measurement_nodes = 0;
  double measurement_seconds = 0;
  bool stopped_by_nodes = false;
  bool stopped_by_time = false;
  bool stopped_by_memory = false;
  double seconds = 0;
};

struct ScanResult {
  /// The main thread's iterations (the only thread's in the deterministic
  /// mode). Each iteration's `nodes` counts every thread's.
  std::vector<IterationStats> iterations;
  /// The deepest iteration that completed within the budget, not capped - by
  /// any thread: the forcing tree to that many half-turns was fully resolved
  /// (0: none).
  int resolved_depth = 0;
  /// The deepest that completed, capped or not (any thread).
  int completed_depth = 0;
  /// Some completed iteration was exhausted (resolved at every depth).
  bool exhausted = false;
  /// The threads that searched.
  int threads = 1;
  /// The root's proof (none if not proven).
  Proof proof;
  /// The root's estimate after the deepest completed iteration (the side to
  /// move's view, centipawns, or a mate score beyond kScanMate - 1000).
  int score = 0;
  /// The best action of that iteration, its moves in a playable order (empty
  /// if none).
  std::vector<Move> best_action;
  /// Every thread's work, summed (seconds: the scan's wall time).
  ScanStats stats;
};

/// Scores: a mate in n half-turns from the root scores kScanMate - n.
inline constexpr int kScanMate = 1'000'000;

// ---------------------------------------------------------------------------
// The scanner
// ---------------------------------------------------------------------------

/// One action the forcing generator built: its moves in a playable order and
/// the successor (submitted: the opponent to move).
struct BuiltAction {
  std::vector<Move> moves;
  GameState successor;
};

class Scanner {
 public:
  /// `store`: the solved store proofs are exported to (and read from); none:
  /// a store of the scanner's own (store()).
  explicit Scanner(const ScannerOptions& options = {}, SolvedStore* store = nullptr);
  ~Scanner();
  Scanner(const Scanner&) = delete;
  Scanner& operator=(const Scanner&) = delete;

  /// Iterative deepening from `root` (a turn start) to options().max_depth
  /// forcing half-turns, within `budget`, by options().threads threads;
  /// stops early at a root proof or an exhausted tree. The TT and the
  /// histories carry over from earlier scans until clear(). Rethrows the
  /// first exception a thread threw (std::logic_error for a contradiction).
  ScanResult scan(const GameState& root, const ScanBudget& budget);

  /// Empties the TT and every thread's move-ordering tables (between games,
  /// §11.3) - not the solved store.
  void clear();

  const ScannerOptions& options() const noexcept;
  SolvedStore& store() noexcept;

  // -- the pieces, for tools and tests (no budget; they count nothing) ---------

  /// The static eval (H6 v1): the side to move's material on every
  /// timeline's latest board minus the opponent's, in centipawns.
  int evaluate(const GameState& state) const;

  /// The forcing moves at a turn start (see the header comment), in generator
  /// order.
  MoveList forcing_moves(const GameState& turn_start);
  /// Whether the side to move has a forcing move: not quiet.
  bool has_forcing_move(const GameState& turn_start);
  /// Whether `move` (pseudo-legal at `turn_start`) is a checking move - (b).
  bool checks(const GameState& turn_start, const Move& move);

  /// The legal action that begins with `seed` (pseudo-legal at `turn_start`)
  /// and is finished by the completion rule; nullopt if no legal action
  /// begins with it (or E4's search could not tell within its memory
  /// ceiling).
  std::optional<BuiltAction> complete(const GameState& turn_start, const Move& seed);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Stand-pat validity (6b, §15.6; operationalised before it was measured)
// ---------------------------------------------------------------------------

/// At a QUIET turn start Q (not in check; no forcing move) with static eval e0
/// for its side to move S: stand-pat assumes S can decline - play some action
/// that loses nothing. It is INVALID at Q when every legal action of S's
/// LOSES MATERIAL to the opponent's best forcing reply: for each action A, the
/// scanner's own forcing search from Q_A (the opponent to move) to
/// `reply_depth` forcing half-turns, stand-pat on, within `reply_nodes` nodes,
/// gives S less than e0. An action HOLDS when it gives S at least e0.
/// The actions: all of S's legal actions when there are at most
/// `max_actions` (E4's listing - EXHAUSTIVE); otherwise a SAMPLE - every
/// pseudo-legal move at Q as a seed, completed by the scanner's completion
/// rule, the first `max_actions` distinct legal ones (an invalid verdict is
/// then an upper bound: an action outside the sample might hold).
struct StandPatCheck {
  int max_actions = 32;
  int reply_depth = 2;
  std::uint64_t reply_nodes = 4000;
};

struct StandPatVerdict {
  enum class Kind : std::uint8_t {
    NotQuiet = 0,  // in check, or a forcing move exists: not asked
    NoAction,      // no legal action (a terminal position)
    Valid,         // some action holds
    Invalid,       // every action examined loses
    Undetermined,  // none holds, but some reply search ran out of nodes
  };
  Kind kind = Kind::NotQuiet;
  bool exhaustive = false;  // every legal action examined
  int eval = 0;             // e0
  int actions = 0;          // examined
  int holding = 0;
  int losing = 0;
  int unknown = 0;
  int mandatory_boards = 0; // at Q
  std::uint64_t nodes = 0;
};

StandPatVerdict stand_pat_validity(const GameState& position, const ScannerOptions& options,
                                   const StandPatCheck& check = {});

}  // namespace t5d

#endif  // T5D_SCANNER_HPP
