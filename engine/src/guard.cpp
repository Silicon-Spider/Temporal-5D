// Guard (see t5d/guard.hpp).

#include "t5d/guard.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "t5d/check.hpp"
#include "t5d/movegen.hpp"

namespace t5d {

namespace {

void add(ProverStats& into, const ProverStats& x) {
  into.nodes += x.nodes;
  into.tree_nodes += x.tree_nodes;
  into.turn_nodes += x.turn_nodes;
  into.expansions += x.expansions;
  into.applies += x.applies;
  into.attack_queries += x.attack_queries;
  into.dead += x.dead;
  into.passes_cut += x.passes_cut;
  into.mate_tests += x.mate_tests;
  into.escape_probe_hits += x.escape_probe_hits;
  into.checkmates += x.checkmates;
  into.stalemates += x.stalemates;
  into.detector_nodes += x.detector_nodes;
  into.listings += x.listings;
  into.store_hits += x.store_hits;
  into.check_cut += x.check_cut;
  into.not_in_check += x.not_in_check;
  into.stopped_by_nodes = into.stopped_by_nodes || x.stopped_by_nodes;
  into.stopped_by_time = into.stopped_by_time || x.stopped_by_time;
  into.stopped_by_memory = into.stopped_by_memory || x.stopped_by_memory;
  into.seconds += x.seconds;
}

}  // namespace

GuardResult guard(const GameState& turn_start, const std::vector<Move>& action, const ProveBudget& budget,
                  const GuardOptions& options) {
  const auto start = std::chrono::steady_clock::now();
  // The action: legal, as it is about to be submitted.
  GameState after = turn_start;
  MoveList scratch;
  for (const Move& m : action) {
    if (!after.may_move() || !is_pseudo_legal(after, m, scratch)) {
      throw std::invalid_argument("guard: a move of the action is not pseudo-legal where it is made");
    }
    after.apply(m);
  }
  if (!after.obligations_met()) throw std::invalid_argument("guard: the action leaves an obligation (T5)");
  if (!royals_safe(after)) throw std::invalid_argument("guard: the action leaves a royal capturable (C4)");
  after.submit(CheckRuling::none());

  GuardResult out;
  const int max_depth = std::max(1, options.max_depth);
  ProverOptions po = options.prover;
  po.seeds = SeedClass::All;            // every opponent action
  po.completion = CompletionClass::Every;
  po.prove_only = false;
  po.threads = std::max(1, options.threads);
  po.depth = 2 * max_depth - 1;
  ConcurrentSolvedStore proofs;
  DisproofStore disproofs;
  Prover prover(po, &proofs, &disproofs);
  std::uint64_t used = 0;
  for (int d = 1; d <= max_depth; ++d) {
    ProveBudget b;
    if (budget.max_nodes <= used) {
      out.stopped = true;
      break;
    }
    b.max_nodes = budget.max_nodes - used;
    if (budget.max_time) {
      const auto spent = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
      if (spent >= *budget.max_time) {
        out.stopped = true;
        break;
      }
      b.max_time = *budget.max_time - spent;
    }
    const ProveResult r = prover.prove(after, b, 2 * d - 1);
    used += r.stats.nodes;
    add(out.stats, r.stats);
    out.depths.push_back(GuardDepth{d, r.status, r.stats.nodes, r.stats.seconds});
    if (r.status == ProofStatus::Proven) {
      out.refuted = true;
      out.refutation_depth = (r.proof.distance + 1) / 2;
      out.line = r.line;
      out.line_complete = r.line_complete;
      break;
    }
    if (r.status == ProofStatus::Unknown) {
      out.stopped = true;
      break;
    }
    out.completed_depth = d;  // disproven: no forced mate within d of their actions
  }
  out.nodes = used;
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return out;
}

GuardResult guard(const GameState& turn_start, const std::vector<Move>& action, const ProveBudget& budget,
                  const GuardOptions& options, const GuardHook& hook) {
  GuardResult r = guard(turn_start, action, budget, options);
  if (r.refuted && hook.on_refutation) hook.on_refutation(r);
  return r;
}

}  // namespace t5d
