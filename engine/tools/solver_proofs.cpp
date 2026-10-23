// The search's solver read back, and held to df-pn (solver_proofs.hpp).

#include "solver_proofs.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/fen.hpp"
#include "t5d/search.hpp"

namespace t5d::tools {

namespace {

using mcts::EdgeView;
using mcts::kNoNode;
using mcts::NodeId;
using mcts::NodeView;
using mcts::Search;
using mcts::Terminal;

bool solved(const NodeView& n) noexcept { return n.lo == n.hi; }

// A child's bounds in its parent's view (the search's own rule).
std::pair<int, int> in_parent(const NodeView& parent, const NodeView& child) noexcept {
  if (parent.mover == child.mover) return {child.lo, child.hi};
  return {-child.hi, -child.lo};
}

// The edges a bound is read through: not pruned, a child made, the child no
// dead end. `proves`: and its move's legality known (an unknown edge bounds
// from above only).
bool counts(const Search& s, const EdgeView& e, bool proves) {
  if (e.pruned || e.child == kNoNode) return false;
  if (proves && e.unknown) return false;
  return !s.node(e.child).dead_end;
}

class Deriver {
 public:
  explicit Deriver(const Search& s) : s_(s) {}

  SolverClaim claim(NodeId id) {
    if (const auto it = memo_.find(id); it != memo_.end()) return it->second;
    const SolverClaim c = compute(id);
    memo_.emplace(id, c);
    return c;
  }

  // The edge proving a Win claim at its distance (first in edge order).
  std::size_t winning_edge(NodeId id) {
    const NodeView n = s_.node(id);
    if (!solved(n) || n.lo != 1) throw std::logic_error("solver claims: node " + std::to_string(id) + " is not proven won");
    const SolverClaim c = claim(id);
    for (std::size_t i = 0; i < n.edge_count; ++i) {
      const EdgeView e = s_.edge(id, i);
      if (!counts(s_, e, true)) continue;
      const NodeView child = s_.node(e.child);
      if (in_parent(n, child).first != 1) continue;
      if (via(n, e.child) == c.distance) return i;
    }
    throw std::logic_error("solver claims: no edge proves node " + std::to_string(id));
  }

 private:
  // The distance through `child` in the parent's view: a same-mover child's
  // own, an opponent's plus the half-turn the submission ends.
  int via(const NodeView& parent, NodeId child) {
    const SolverClaim c = claim(child);
    return s_.node(child).mover == parent.mover ? c.distance : c.distance + 1;
  }

  SolverClaim compute(NodeId id) {
    const NodeView n = s_.node(id);
    SolverClaim c;
    c.node = id;
    c.turn_start = n.turn_start;
    if (!solved(n)) throw std::logic_error("solver claims: node " + std::to_string(id) + " is not solved");
    if (n.terminal == Terminal::Checkmate) {
      c.kind = SolverClaim::Kind::Loss;
      c.terminal = true;
      return c;
    }
    if (n.terminal == Terminal::Stalemate) {
      c.kind = SolverClaim::Kind::Draw;
      c.terminal = true;
      return c;
    }
    if (n.terminal == Terminal::TurnCap) {
      throw std::logic_error("solver claims: the turn cap's convention solved at node " + std::to_string(id));
    }
    if (n.lo == 0) {
      c.kind = SolverClaim::Kind::Draw;
      return c;
    }
    std::optional<int> best;
    bool any = false;
    for (std::size_t i = 0; i < n.edge_count; ++i) {
      const EdgeView e = s_.edge(id, i);
      if (n.lo == 1) {
        if (!counts(s_, e, true)) continue;
        if (in_parent(n, s_.node(e.child)).first != 1) continue;
        const int d = via(n, e.child);
        if (!best || d < *best) best = d;
      } else {
        if (e.pruned) continue;
        if (e.child == kNoNode) {
          throw std::logic_error("solver claims: node " + std::to_string(id) + " proven lost with a child not made");
        }
        if (s_.node(e.child).dead_end) continue;
        any = true;
        if (in_parent(n, s_.node(e.child)).second != -1) {
          throw std::logic_error("solver claims: node " + std::to_string(id) + " proven lost with a child not lost");
        }
        const int d = via(n, e.child);
        if (!best || d > *best) best = d;
      }
    }
    if (n.lo == -1 && !n.children_complete) {
      throw std::logic_error("solver claims: node " + std::to_string(id) + " proven lost, its children incomplete");
    }
    if (!best || (n.lo == -1 && !any)) {
      throw std::logic_error("solver claims: no child supports node " + std::to_string(id) + "'s bound");
    }
    c.kind = n.lo == 1 ? SolverClaim::Kind::Win : SolverClaim::Kind::Loss;
    c.distance = *best;
    return c;
  }

  const Search& s_;
  std::unordered_map<NodeId, SolverClaim> memo_;
};

std::string position_text(const GameState& s) {
  std::string out = s.to_move() == Colour::White ? "white to move:" : "black to move:";
  const Multiverse& mv = s.multiverse();
  mv.for_each_board([&](int p, int h, const Board& b) {
    out += ' ';
    out += fen_board_string(b, p, h, mv.even());
  });
  out += " size " + to_string(mv.board_size()) + " rules " + describe(s.rules());
  return out;
}

}  // namespace

const char* claim_kind_name(SolverClaim::Kind k) noexcept {
  switch (k) {
    case SolverClaim::Kind::Win: return "win";
    case SolverClaim::Kind::Loss: return "loss";
    case SolverClaim::Kind::Draw: return "draw";
  }
  return "?";
}

std::string to_string(const SolverClaim& c) {
  if (c.kind == SolverClaim::Kind::Draw) return "draw";
  return std::string(claim_kind_name(c.kind)) + " " + std::to_string(c.distance);
}

std::vector<SolverClaim> solver_claims(const Search& search) {
  Deriver d(search);
  std::vector<SolverClaim> out;
  for (NodeId id = 0; id < search.node_count(); ++id) {
    const NodeView n = search.node(id);
    if (n.state == nullptr || !solved(n) || n.dead_end) continue;
    out.push_back(d.claim(id));
  }
  return out;
}

SolverClaim solver_claim(const Search& search, NodeId id) { return Deriver(search).claim(id); }

std::size_t winning_edge(const Search& search, NodeId id) { return Deriver(search).winning_edge(id); }

void DfpnTally::add(const DfpnTally& o) {
  claims += o.claims;
  wins += o.wins;
  losses += o.losses;
  draws += o.draws;
  terminals += o.terminals;
  too_deep += o.too_deep;
  comparisons += o.comparisons;
  decided += o.decided;
  agreed += o.agreed;
  unknown += o.unknown;
  listing_undecided += o.listing_undecided;
  questions += o.questions;
  searched += o.searched;
  proofs += o.proofs;
  disproofs += o.disproofs;
  certified += o.certified;
  certificate_undecided += o.certificate_undecided;
  contradictions.insert(contradictions.end(), o.contradictions.begin(), o.contradictions.end());
  certificate_failures.insert(certificate_failures.end(), o.certificate_failures.begin(), o.certificate_failures.end());
}

std::string DfpnTally::summary() const {
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "%zu claims (%zu win, %zu loss, %zu draw; %zu game ends and %zu too deep not asked); "
                "%zu comparisons, %zu decided, %zu agreed, %zu unknown, %zu listings undecided; "
                "%zu contradictions; df-pn %zu questions (%zu searched): %zu proofs (%zu certified by E4, "
                "%zu undecided), %zu disproofs; %zu certificate failures",
                claims, wins, losses, draws, terminals, too_deep, comparisons, decided, agreed, unknown,
                listing_undecided, contradictions.size(), questions, searched, proofs, certified,
                certificate_undecided, disproofs, certificate_failures.size());
  return buf;
}

DfpnChecker::DfpnChecker(const DfpnCheckOptions& options) : options_(options) {}

DfpnChecker::Answer DfpnChecker::ask(const GameState& state, int depth, DfpnTally& tally) {
  ++tally.questions;
  const std::uint64_t key = state.zobrist() * 1315423911ULL + static_cast<std::uint64_t>(depth);
  std::vector<Memo>& slot = memo_[key];
  for (const Memo& m : slot) {
    if (m.depth == depth && same_position(m.state, state)) return m.answer;
  }
  ++tally.searched;
  ProverOptions o;
  o.seeds = SeedClass::All;
  o.completion = CompletionClass::Every;
  o.depth = depth;
  o.threads = 1;
  o.tt_entries = options_.tt_entries;
  o.record_certificate = options_.certify;
  Prover prover(o);
  const ProveResult r = prover.prove(state, ProveBudget::nodes(options_.nodes));
  Answer a;
  a.status = r.status;
  if (r.status == ProofStatus::Proven) {
    ++tally.proofs;
    a.distance = r.proof.distance;
    if (r.proof.kind != Proof::Kind::Win || r.proof.distance > depth) {
      tally.certificate_failures.push_back("df-pn proved " + to_string(r.proof) + " within depth " +
                                           std::to_string(depth) + " at " + position_text(state));
    }
    if (options_.certify) {
      if (r.certificate == nullptr) {
        tally.certificate_failures.push_back("df-pn's proof has no certificate at " + position_text(state));
      } else {
        const CertificateCheck c = check_certificate(*r.certificate, state, r.proof.distance, options_.certificate);
        if (c.verdict == CertificateVerdict::Confirmed) {
          ++tally.certified;
        } else if (c.verdict == CertificateVerdict::Undecided) {
          ++tally.certificate_undecided;
        } else {
          tally.certificate_failures.push_back(std::string("df-pn's certificate ") +
                                               certificate_verdict_name(c.verdict) + " (" + c.why + ") for " +
                                               to_string(r.proof) + " at " + position_text(state));
        }
      }
    }
  } else if (r.status == ProofStatus::Disproven) {
    ++tally.disproofs;
  }
  slot.push_back(Memo{state, depth, a});
  return a;
}

void DfpnChecker::check(const GameState& state, const SolverClaim& claim, DfpnTally& tally) {
  ++tally.claims;
  (claim.kind == SolverClaim::Kind::Win ? tally.wins : claim.kind == SolverClaim::Kind::Loss ? tally.losses
                                                                                              : tally.draws) += 1;
  if (claim.terminal) {
    ++tally.terminals;
    return;
  }
  const int depth = claim.kind == SolverClaim::Kind::Draw ? options_.draw_depth : claim.distance;
  if (claim.kind == SolverClaim::Kind::Loss && claim.distance < 2) {
    // Not a game end, so the opponent's mate is at least one action away.
    tally.contradictions.push_back("the solver says " + to_string(claim) + " at a position that is no game end: " +
                                   position_text(state));
    return;
  }
  if (depth > options_.max_depth) {
    ++tally.too_deep;
    return;
  }
  const auto contradict = [&](const std::string& what) {
    tally.contradictions.push_back("the solver says " + to_string(claim) + ", df-pn " + what + " at " +
                                   position_text(state));
  };
  // One comparison: `decided` and `agrees` as df-pn's answer settles it.
  const auto compare = [&](bool decided, bool agrees, const std::string& what) {
    ++tally.comparisons;
    if (!decided) {
      ++tally.unknown;
      return;
    }
    ++tally.decided;
    if (agrees) {
      ++tally.agreed;
    } else {
      contradict(what);
    }
  };
  if (claim.kind == SolverClaim::Kind::Win) {
    const Answer a = ask(state, claim.distance, tally);
    compare(a.status != ProofStatus::Unknown, a.status == ProofStatus::Proven && a.distance <= claim.distance,
            a.status == ProofStatus::Disproven ? "disproves a mate within " + std::to_string(claim.distance)
                                               : "proves win " + std::to_string(a.distance));
    return;
  }
  // Loss and Draw: no df-pn proof for the side to move at the node.
  const Answer here = ask(state, depth, tally);
  compare(here.status != ProofStatus::Unknown, here.status == ProofStatus::Disproven,
          "proves the side to move's mate (win " + std::to_string(here.distance) + ")");
  // The successors: E4's exhaustive listing.
  SearchBudget b = SearchBudget::nodes(options_.listing_nodes);
  const ActionList list = list_legal_actions(state, options_.listing_cap, Detector::Hypercuboid, b);
  if (!list.stats.exhaustive || list.reached_cap) {
    ++tally.listing_undecided;
    return;
  }
  if (list.actions.empty()) {
    // No legal action: the node is a game end the search did not mark - its
    // claim would rest on no child at all.
    contradict("finds no legal action (E4)");
    return;
  }
  if (claim.kind == SolverClaim::Kind::Loss) {
    for (const LegalAction& act : list.actions) {
      const Answer a = ask(act.result, claim.distance - 1, tally);
      compare(a.status != ProofStatus::Unknown, a.status == ProofStatus::Proven,
              "disproves the opponent's mate within " + std::to_string(claim.distance - 1) + " after an action");
    }
    return;
  }
  // Draw: the opponent's mate within draw_depth is not proven after every action.
  bool disproof = false;
  bool unknown = false;
  for (const LegalAction& act : list.actions) {
    const Answer a = ask(act.result, options_.draw_depth, tally);
    if (a.status == ProofStatus::Disproven) {
      disproof = true;
      break;
    }
    if (a.status == ProofStatus::Unknown) unknown = true;
  }
  const bool all_proven = !disproof && !unknown;
  compare(disproof || all_proven, disproof, "proves the opponent's mate after every action");
}

void DfpnChecker::check_search(const Search& search, DfpnTally& tally) {
  for (const SolverClaim& c : solver_claims(search)) {
    if (!c.turn_start) continue;
    check(*search.node(c.node).state, c, tally);
  }
}

}  // namespace t5d::tools
