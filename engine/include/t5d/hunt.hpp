// t5d/hunt.hpp - HUNT (stage E9; verification v3, design spec 11.4; register
// H43, with H37, H41 and H23): at the root, do WE have a forced mate?
//
// THE QUESTION. At a turn start, its side to move ours: a forced mate within
// `depth` half-turns (5 for 6b'': ours, theirs, ours, theirs, ours) by our
// forcing class (the prover's placeholder default: checking seeds, canonical
// completions; ProverOptions), whatever they answer - every legal action of
// theirs.
//
// PROVE-ONLY. df-pn (t5d/prover.hpp) in prove-only mode: it stops at its
// budget and CLAIMS NOTHING - no disproof, no "unknown" as a verdict, and a
// class it exhausts is "no mate found" like a budget stop. What prove-only
// changes in the prover: a root it exhausts is reported Unknown, never
// Disproven; it neither writes nor reads the shared disproof store (no
// disproof is recorded, none from outside prunes it) - its own search's
// disproofs stay in a private memo for its transpositions; and it is never
// run on to settle a root, so it never spends time completing a disproof the
// way the settling scans must. Its thresholds stay df-pn's usual two: on the
// tests' positions proof-number thresholds alone found fewer mates for more
// nodes (ProverOptions::prove_only_pn_thresholds, the named alternative).
//
// A FOUND MATE is a proof (H37: exact sub-results only, E4's exhaustive
// verdicts at the leaves), returned to the caller - and NEVER exported to a
// permanent solved store: no proof may enter one until the generator has
// passed the gate (H23), the search's soundness fixes have landed and the
// proof channel is signed off (H41). HuntOptions::export_proofs is that
// switch, enforced as the search's proof_store is (t5d/mcts.hpp): Hunt
// refuses to be built with it set. The export is not built.
//
// SOFT PRIORS, behind a flag (HuntOptions::suggest, OFF). Without a proof,
// Hunt can suggest its most promising attacking actions - the root's
// children with the least proof numbers, weighted by 1 / pn - as soft
// priors for the search. Spec 12.2 binds their use: a position where a
// suggestion steered the search is EXCLUDED from the policy targets (or its
// steered visits are recorded and down-weighted - the open choice of 12.2),
// so Hunt's material-greedy bias never trains the network. The bookkeeping
// is built (stage E10: t5d/ladder.hpp and the search's steered visits,
// RootChild::steered_visits); the search's flag (SearchParams::hunt_suggest)
// stays off.
//
// WIRED (stage E10, t5d/ladder.hpp): the verification ladder asks Hunt at a
// decision's root, a turn start under 100 boards (6b''), with 6b''s budget;
// a found mate reaches the search only behind the proof channel (H41).

#ifndef T5D_HUNT_HPP
#define T5D_HUNT_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/prover.hpp"
#include "t5d/scanner.hpp"
#include "t5d/state.hpp"

namespace t5d {

struct HuntOptions {
  /// The prover's options: the class and the depth (5) are Hunt's; Hunt sets
  /// prove-only and the threads.
  ProverOptions prover;
  /// Search threads (6b'': M13's 5). 1: deterministic with a node budget.
  int threads = 1;
  /// H41's gate: export a found mate to a permanent solved store. MUST stay
  /// false: no proof may enter a solved store until the generator has passed
  /// the gate (H23) and the proof channel is signed off (H41). Hunt's
  /// constructor throws std::invalid_argument if it is set.
  bool export_proofs = false;
  /// Soft priors (H43, spec 12.2's exclusions): OFF.
  bool suggest = false;
  std::size_t max_suggestions = 8;
};

struct HuntSuggestion {
  std::vector<Move> action;  // an attacking action of the class
  double weight = 0;         // 1 / pn, normalised over the suggestions
  std::uint64_t pn = 0;
};

struct HuntResult {
  /// A forced mate was proven: Win(k), k <= depth.
  bool mate = false;
  Proof proof;
  /// The proof's first action, in a playable order (ProveResult::first_action)
  /// - what the search plays (stage E10, behind the proof channel).
  std::vector<Move> first_action;
  /// The proof's line (t5d/prover.hpp ProveResult::line).
  std::vector<std::vector<Move>> line;
  bool line_complete = false;
  /// suggest, no mate: the suggestions, most promising first.
  std::vector<HuntSuggestion> suggestions;
  ProverStats stats;
  /// prover.record_certificate, mate: the proof's certificate, for
  /// check_certificate (t5d/certificate.hpp).
  std::shared_ptr<const ProofCertificate> certificate;
};

/// The prover's options Hunt runs with: `options.prover`, prove-only, the
/// threads.
ProverOptions hunt_prover_options(const HuntOptions& options);

class Hunt {
 public:
  /// Throws std::invalid_argument if options.export_proofs is set (H23, H41).
  explicit Hunt(const HuntOptions& options = {});
  /// The question at `root` (a turn start). A fresh prover each time.
  HuntResult hunt(const GameState& root, const ProveBudget& budget);
  const HuntOptions& options() const noexcept { return options_; }

 private:
  HuntOptions options_;
};

}  // namespace t5d

#endif  // T5D_HUNT_HPP
