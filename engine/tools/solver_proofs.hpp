// tools/solver_proofs.hpp - the search's solver read back, and held to df-pn
// (register H41: the search's own solver "is part of the proof channel" and
// its proofs "are cross-checked against df-pn and oracle mate verdicts";
// design spec §11.1, §15.4). Shared by the engine's tests (tests/
// test_solver_refs.hpp), t5d-solver-check (the offline mode) and
// t5d-proof-records (the proof lines written as 5dpgn records).
//
// THE CLAIMS. With SearchParams::proof_channel on (test-only and offline:
// H41 keeps it off in play, and nothing here is a store - no proof goes
// anywhere but into the caller's report) the search keeps score bounds
// [lo, hi] per node, each in its own mover's view. The search records no
// proof distance; it is derived here from the solved subtree, edge by edge:
//  * a CHECKMATE node (E4's verdict) is Loss(0); a STALEMATE node Draw;
//  * a node proven won (lo = hi = 1) is Win(k), k the least over its
//    children proven lost for it (in its view): through a move edge (the same
//    mover, mid-action - the ply split and the tree form) the child's own
//    Win(k); through a submit edge or a whole turn (the opponent to move) the
//    child's Loss(j), k = j + 1;
//  * a node proven lost (lo = hi = -1) is Loss(k), k the greatest over its
//    children - every legal one, all proven lost for it: a same-mover child's
//    Loss(k), an opponent's Win(j) as j + 1;
//  * a node proven drawn (lo = hi = 0) is Draw.
// Pruned edges and dead-end children are skipped, and an edge whose move's
// legality is unknown proves nothing - exactly as the search's own bound
// update reads them. A solved node whose children do not support its bound
// is an error of the derivation's (std::logic_error): the search's
// check_invariants holds the bounds themselves.
//
// THE COMPARISON. At each solved TURN START (a claim, mid-action nodes are
// steps inside it), df-pn (t5d/prover.hpp) in its LITERAL class - seeds All,
// completion Every: exactly "mate within d" - one thread, a node budget,
// deterministic, its own stores:
//  * Win(k): df-pn at depth k proves Win(j <= k) or answers Unknown -
//    never Disproven;
//  * Loss(k), k >= 2 (Loss(0) is E4's checkmate, the search's own terminal):
//    at every legal successor (E4's exhaustive listing) df-pn at depth k - 1
//    proves the opponent's mate or answers Unknown - never Disproven; and at
//    the node itself df-pn at depth k proves nothing (the side to move has
//    no mate where it is mated whatever it does);
//  * Draw: df-pn at the node proves nothing (draw_depth), and the opponent's
//    mate is not proven at every successor (draw_depth);
//  * a df-pn proof at a solved node means the solver says Win (the second
//    halves of the Loss and Draw rules).
// "Where df-pn decides": Unknown gives no verdict and is counted (as 6b''
// did). Every df-pn proof is certified by check_certificate (t5d/
// certificate.hpp: E4 alone); a refuted or incomplete certificate is an
// error of the prover's, reported beside the contradictions.

#ifndef T5D_TOOLS_SOLVER_PROOFS_HPP
#define T5D_TOOLS_SOLVER_PROOFS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "t5d/certificate.hpp"
#include "t5d/mcts.hpp"
#include "t5d/prover.hpp"
#include "t5d/state.hpp"

namespace t5d::tools {

/// A solved node's claim, in its mover's view.
struct SolverClaim {
  enum class Kind : std::uint8_t { Win = 0, Loss, Draw };
  mcts::NodeId node = mcts::kNoNode;
  Kind kind = Kind::Draw;
  /// Win: half-turns to the mate along the solver's proof (the least over
  /// its winning children); Loss: the longest defence's; Draw: 0.
  int distance = 0;
  /// A game end (E4's checkmate or stalemate at the node).
  bool terminal = false;
  bool turn_start = true;
};

const char* claim_kind_name(SolverClaim::Kind k) noexcept;
/// "win 3", "loss 2", "draw".
std::string to_string(const SolverClaim& c);

/// The claims of a finished search: every solved node's (proof channel on),
/// the turn starts' and the mid-action nodes'. Throws std::logic_error where
/// a solved node's children do not support its bound.
std::vector<SolverClaim> solver_claims(const mcts::Search& search);

/// One node's claim (it must be solved).
SolverClaim solver_claim(const mcts::Search& search, mcts::NodeId id);

/// The edge from `id` that proves its Win claim at the claim's distance (the
/// first such, in edge order); its child is the next node of the proof line.
/// Throws std::logic_error if `id` is not proven won.
std::size_t winning_edge(const mcts::Search& search, mcts::NodeId id);

struct DfpnCheckOptions {
  /// df-pn's budget per question (one thread: deterministic, H15).
  std::uint64_t nodes = 200'000;
  /// Its transposition table (estimates only, H19).
  std::size_t tt_entries = std::size_t{1} << 16;
  /// The depth the Draw rule asks at.
  int draw_depth = 3;
  /// Claims deeper than this are not asked (counted: too_deep).
  int max_depth = 9;
  /// E4's listing of a Loss or Draw node's successors: at most this many
  /// actions, within listing_nodes (more, or a stop: undecided).
  std::size_t listing_cap = 4096;
  std::uint64_t listing_nodes = 2'000'000;
  /// Certify every df-pn proof with check_certificate (E4 alone).
  bool certify = true;
  CertificateCheckOptions certificate{};
};

/// What the comparison found. A COMPARISON is one rule applied: a Win
/// claim's, a Loss claim's at one successor or at the node itself, a Draw
/// claim's at the node or over its successors; DECIDED where df-pn's answer
/// (or answers) decided it, either way.
struct DfpnTally {
  std::size_t claims = 0;          // solved turn starts compared
  std::size_t wins = 0;
  std::size_t losses = 0;
  std::size_t draws = 0;
  std::size_t terminals = 0;       // game ends (E4's verdicts): not asked
  std::size_t too_deep = 0;        // claims beyond max_depth: not asked
  std::size_t comparisons = 0;
  std::size_t decided = 0;
  std::size_t agreed = 0;          // decided, and df-pn agrees
  std::size_t unknown = 0;         // comparisons df-pn left undecided
  std::size_t listing_undecided = 0;
  std::size_t questions = 0;       // df-pn questions asked (memo hits included)
  std::size_t searched = 0;        // ... and searched (not from the memo)
  std::size_t proofs = 0;          // df-pn proofs among the questions searched
  std::size_t disproofs = 0;
  std::size_t certified = 0;       // proofs whose certificate E4 confirmed
  std::size_t certificate_undecided = 0;
  std::vector<std::string> contradictions;        // the solver against df-pn
  std::vector<std::string> certificate_failures;  // df-pn's proof against E4
  void add(const DfpnTally& other);
  /// One line: the counts.
  std::string summary() const;
};

/// df-pn's answers, memoised by position and depth (a position the search's
/// tree holds twice, or another split's search reaches, is asked once).
class DfpnChecker {
 public:
  explicit DfpnChecker(const DfpnCheckOptions& options = {});
  /// Compares the claim of the node at `state` (a turn start) with df-pn,
  /// adding to `tally`.
  void check(const GameState& state, const SolverClaim& claim, DfpnTally& tally);
  /// Every solved turn start of `search`.
  void check_search(const mcts::Search& search, DfpnTally& tally);
  const DfpnCheckOptions& options() const noexcept { return options_; }

  struct Answer {
    ProofStatus status = ProofStatus::Unknown;
    int distance = 0;  // Proven: Win(distance)
  };
  /// df-pn's literal-class answer at `state` within `depth` (memoised;
  /// proofs certified once, when searched).
  Answer ask(const GameState& state, int depth, DfpnTally& tally);

 private:
  struct Memo {
    GameState state;
    int depth;
    Answer answer;
  };
  DfpnCheckOptions options_;
  std::unordered_map<std::uint64_t, std::vector<Memo>> memo_;
};

}  // namespace t5d::tools

#endif  // T5D_TOOLS_SOLVER_PROOFS_HPP
