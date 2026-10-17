// t5d/guard.hpp - GUARD (stage E9; verification v3, design spec 11.4; register
// H42, with H37 and H41): before the chosen action is played, can the
// opponent force mate against it?
//
// THE QUESTION. At a turn start, our chosen ACTION - complete and legal,
// about to be submitted - is submitted (under the ply split Guard checks the
// action at its submission: the opponent moves only after it; a reading,
// build-log spec issue 97), and at the opponent's turn start that follows Guard
// asks, depth by depth from 1 (iterative deepening): can the opponent force
// checkmate within d of THEIR actions - 2d - 1 half-turns, theirs, ours, ...,
// theirs - whatever we answer? Depth 1 is "for every opponent action that
// leaves us in check (R1), one legal escape exists": existence only - a found
// escape ends that branch (a quick escape search, E4's detector only where it
// finds none) - never the full listing of our legal actions.
//
// THE SEARCH. The prover (t5d/prover.hpp) with the opponent as its attacker
// and the literal class - every opponent action (seeds All, completion
// Every), enumerated once each in its semimove tree; our turns are AND nodes,
// decided by one refuting reply. At the horizon a turn of ours not in check
// is decided without E4. The prover's last-turn check filter is OFF, as in
// every prover (GuardOptions::prover.check_filter; measured not to pay). Where
// turns do not decompose, E4's capped listing: a capped listing never lets a
// depth complete (it cannot be disproven) - never a false "safe".
//
// THE ANSWER. The deepest depth that COMPLETED - "safe to depth d", exact:
// every opponent action within it refuted (a depth stopped by the budget is
// not claimed, H37) - and, where a depth proved a mate, the REFUTATION: the
// opponent's forcing line (their action, our longest defence, ..., their
// mating action), a proof - exact sub-results only, never a budget stop
// (H37). A refutation rejects the move and triggers a re-search from spec
// 8.6's reserve: the search does that (wired with the search core, stage E6;
// GuardHook is the hook).
//
// THE BUDGET. One budget for all the depths - nodes and/or wall clock (6b'':
// 50 ms) - each depth getting what the earlier ones left. No complexity model
// sets the depth (spec 11.4): it goes deeper where the position is simple.

#ifndef T5D_GUARD_HPP
#define T5D_GUARD_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/prover.hpp"
#include "t5d/state.hpp"

namespace t5d {

/// Guard's deepest depth: 2 x 50 - 1 half-turns, within the prover's limit.
inline constexpr int kGuardMaxDepth = 50;

struct GuardOptions {
  /// The deepest depth asked, in the opponent's actions (PLACEHOLDER: the
  /// budget, not this, is what normally stops it). Clamped to 1..50.
  int max_depth = 4;
  /// Search threads (6b'': M13's 5). 1: deterministic with a node budget
  /// (H15).
  int threads = 1;
  /// The prover's options; Guard sets the class (every opponent action), the
  /// depth and the threads, and never prove-only (a depth completes only by
  /// disproving the opponent's mate). Where turns do not decompose, the
  /// opponent's actions are E4's listing: Guard lists up to 65,536 of them
  /// (a capped listing never lets a depth complete).
  ProverOptions prover = [] {
    ProverOptions o;
    o.listing_cap = std::size_t{1} << 16;
    return o;
  }();
};

struct GuardDepth {
  int depth = 0;  // the opponent's actions
  ProofStatus status = ProofStatus::Unknown;
  std::uint64_t nodes = 0;
  double seconds = 0;
};

struct GuardResult {
  /// The deepest depth (the opponent's actions) every opponent line of which
  /// was refuted: "safe to depth completed_depth". 0: not even depth 1.
  int completed_depth = 0;
  /// A depth proved a forced mate against the action.
  bool refuted = false;
  /// The opponent's actions to mate (refuted).
  int refutation_depth = 0;
  /// The opponent's forcing line (refuted): their action, our longest
  /// defence, ..., their mating action (t5d/prover.hpp, ProveResult::line).
  /// `line_complete` false: it stops where the reporting thread's tree holds
  /// no continuation (a node another thread, or an earlier depth, solved and
  /// exported) - never at depth 1, whose stores start empty. Whatever the
  /// line, the refutation is the proof's: refuted and refutation_depth.
  std::vector<std::vector<Move>> line;
  bool line_complete = false;
  /// The budget stopped the last depth tried.
  bool stopped = false;
  std::vector<GuardDepth> depths;
  std::uint64_t nodes = 0;
  double seconds = 0;
  /// The prover's statistics, summed over the depths (counts only).
  ProverStats stats;
  /// prover.record_certificate, refuted: the refutation's certificate (the
  /// opponent's mate from the position after the action), for
  /// check_certificate (t5d/certificate.hpp).
  std::shared_ptr<const ProofCertificate> certificate;
};

/// The prover's options Guard runs with: `options.prover` with Guard's
/// class (seeds All, completion Every), never prove-only, the threads, and
/// the depth of its deepest depth (2 x max_depth - 1, max_depth clamped).
ProverOptions guard_prover_options(const GuardOptions& options);

/// Guard's question for `action` (complete, legal - throws
/// std::invalid_argument otherwise) at the turn start `turn_start`.
GuardResult guard(const GameState& turn_start, const std::vector<Move>& action, const ProveBudget& budget,
                  const GuardOptions& options = {});

/// The search's hook (wired with the search core, stage E6): called with the
/// result when Guard refutes the chosen action - the search rejects the move
/// and re-searches from spec 8.6's reserve, the refutation injected as a
/// solved bound once the proof channel is enabled (H41).
struct GuardHook {
  std::function<void(const GuardResult&)> on_refutation;
};

/// guard(), then the hook on a refutation.
GuardResult guard(const GameState& turn_start, const std::vector<Move>& action, const ProveBudget& budget,
                  const GuardOptions& options, const GuardHook& hook);

}  // namespace t5d

#endif  // T5D_GUARD_HPP
