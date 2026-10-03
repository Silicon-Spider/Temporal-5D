// Hunt (see t5d/hunt.hpp).

#include "t5d/hunt.hpp"

#include <algorithm>

namespace t5d {

Hunt::Hunt(const HuntOptions& options, SolvedStore* solved) : options_(options), solved_(solved) {}

HuntResult Hunt::hunt(const GameState& root, const ProveBudget& budget) {
  ProverOptions po = options_.prover;
  po.prove_only = true;
  po.threads = std::max(1, options_.threads);
  // The prover's own stores: its working memory (H19: no proof only in the
  // TT). The permanent store is reached through the gate alone.
  Prover prover(po);
  const ProveResult r = prover.prove(root, budget);
  HuntResult out;
  out.stats = r.stats;
  if (r.status == ProofStatus::Proven) {
    out.mate = true;
    out.proof = r.proof;
    out.line = r.line;
    out.line_complete = r.line_complete;
    out.certificate = r.certificate;
    if (options_.export_proofs && solved_ != nullptr) {
      solved_->put(root, r.proof);
      out.exported = true;
    }
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
