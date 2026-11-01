// Hunt (see t5d/hunt.hpp).

#include "t5d/hunt.hpp"

#include <algorithm>
#include <stdexcept>

namespace t5d {

ProverOptions hunt_prover_options(const HuntOptions& options) {
  ProverOptions po = options.prover;
  po.prove_only = true;
  po.threads = std::max(1, options.threads);
  return po;
}

Hunt::Hunt(const HuntOptions& options) : options_(options) {
  if (options_.export_proofs) {
    throw std::invalid_argument(
        "HuntOptions::export_proofs: the proof channel is disabled - no proof may enter a solved store until "
        "the generator has passed the gate (H23) and the proof channel is signed off (H41)");
  }
}

HuntResult Hunt::hunt(const GameState& root, const ProveBudget& budget) {
  const ProverOptions po = hunt_prover_options(options_);
  // The prover's own stores: its working memory (H19: no proof only in the
  // TT). Nothing reaches a permanent store (H23, H41).
  Prover prover(po);
  const ProveResult r = prover.prove(root, budget);
  HuntResult out;
  out.stats = r.stats;
  if (r.status == ProofStatus::Proven) {
    out.mate = true;
    out.proof = r.proof;
    out.first_action = r.first_action;
    out.line = r.line;
    out.line_complete = r.line_complete;
    out.certificate = r.certificate;
    return out;
  }
  if (!options_.suggest) return out;
  // Soft priors: the root's open children, the least proof number first.
  std::vector<const ProveResult::RootChild*> open;
  for (const ProveResult::RootChild& c : r.root_children) {
    if (c.dn == 0 || c.pn == 0 || c.moves.empty()) continue;  // disproven, or an empty edge
    if (c.pn >= (std::uint64_t{1} << 39)) continue;            // never provable here
    open.push_back(&c);
  }
  std::stable_sort(open.begin(), open.end(),
                   [](const ProveResult::RootChild* a, const ProveResult::RootChild* b) { return a->pn < b->pn; });
  if (open.size() > options_.max_suggestions) open.resize(options_.max_suggestions);
  double total = 0;
  for (const ProveResult::RootChild* c : open) total += 1.0 / static_cast<double>(c->pn);
  for (const ProveResult::RootChild* c : open) {
    out.suggestions.push_back(HuntSuggestion{c->moves, (1.0 / static_cast<double>(c->pn)) / total, c->pn});
  }
  return out;
}

}  // namespace t5d
