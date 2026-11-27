// MT-03's store checks (moved here from test_prover.cpp, unchanged, for the
// settling scans' churn test through the wired ladder too - build-log spec
// issue 402's part 5): a solved store's entries held to the brute-force game
// tree (a Draw to E4's stalemate verdict), a stored Loss confirmed successor
// by successor, and H19 at a root after a search over external stores.

#ifndef T5D_TEST_STORE_CHECKS_HPP
#define T5D_TEST_STORE_CHECKS_HPP

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "t5d/prover.hpp"
#include "t5d/scanner.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_game_tree.hpp"
#include "test_mate_refs.hpp"
#include "test_reference.hpp"

namespace t5d_test {

using namespace t5d;

// E4's verdict at a position: the stalemate a Draw entry must be.
inline bool stalemate_now(const GameState& s) {
  SearchBudget b = SearchBudget::nodes(2'000'000);
  return detect_mate(s, Detector::Hypercuboid, b).verdict == Outcome::Stalemate;
}

// Every entry of a solved store: a Win(k) mates within k and a Loss(k) is
// mated within k by the brute-force game tree, a Draw is a stalemate now by
// E4's verdict (at most `most` entries; the game tree's budgets are small, so
// a Win or Loss it cannot reach - the tree overflowed - is counted "beyond",
// not failed, and the tests bound that count). An Unknown WITHOUT overflow is
// a refutation: the tree saw every line within the distance and found no such
// result - a failure (the review of 2026-11-09, L3: every Unknown counted
// "beyond", so a false Win or Loss distance whose refutation the tree explored
// completely never failed). (MT-03's review, 2026-11-01: a Draw was judged by
// the game tree at depth 0, which answers Unknown wherever a legal action
// exists, so a false Draw counted "beyond", never failed.)
struct StoreCheck {
  int entries = 0;
  int confirmed = 0;
  int beyond = 0;
};
inline void check_store(const MemorySolvedStore& proofs, std::size_t most, StoreCheck& c) {
  std::size_t n = 0;
  for (const MemorySolvedStore::Entry& e : proofs.entries()) {
    if (n++ >= most) break;
    ++c.entries;
    // A proof, never an estimate or a disproof: a Win, a Loss, or a Draw - a
    // stalemate now (distance 0), which the prover records as a game fact.
    if (e.proof.kind == Proof::Kind::None || (e.proof.kind == Proof::Kind::Draw && e.proof.distance != 0)) {
      std::fprintf(stderr, "  the solved store holds %s at %s\n", to_string(e.proof).c_str(),
                   describe_position(e.position).c_str());
      CHECK(false);
      continue;
    }
    if (e.proof.kind == Proof::Kind::Draw) {
      if (stalemate_now(e.position)) {
        ++c.confirmed;
      } else {
        std::fprintf(stderr, "  stored draw, E4 finds no stalemate at %s\n", describe_position(e.position).c_str());
        CHECK(false);
      }
      continue;
    }
    GameTree tree(5000, 20000, 40000);
    const GameValue v = tree.solve(e.position, e.proof.distance);
    if (v.kind == GameValue::Kind::Unknown) {
      if (tree.overflowed()) {
        ++c.beyond;
      } else {
        std::fprintf(stderr, "  the solved store holds %s, which the game tree refutes within its distance at %s\n",
                     to_string(e.proof).c_str(), describe_position(e.position).c_str());
        CHECK(false);
      }
      continue;
    }
    const GameValue::Kind want = e.proof.kind == Proof::Kind::Win ? GameValue::Kind::Win : GameValue::Kind::Loss;
    if (v.kind == want && v.distance <= e.proof.distance) {
      ++c.confirmed;
    } else {
      std::fprintf(stderr, "  stored %s, the game tree %s %d at %s\n", to_string(e.proof).c_str(),
                   t5d_test::name_of(v.kind), v.distance, describe_position(e.position).c_str());
      CHECK(false);
    }
  }
}

// A stored Loss(k) of the side to move, checked: a checkmate now for k 0
// (E4's verdict); else the side to move has legal successors - every one, by
// the brute-force reference (every move order: under Q3's play activation E4's
// listing keeps one state per action key and could miss the escape - the
// review of 2026-11-09, H2) - and after each the opponent mates within k - 1
// by the game tree, asked successor by successor with a tree of its own, so
// that a wide root does not exhaust one tree's budget (a Q6 root of 4 boards
// and 8 actions did, in the long twin: every successor Win(1), the whole root
// Unknown). Too many sequences to enumerate: unconfirmed.
inline bool loss_confirmed(const GameState& s, int k) {
  if (k == 0) {
    SearchBudget b = SearchBudget::nodes(2'000'000);
    return detect_mate(s, Detector::Hypercuboid, b).verdict == Outcome::Checkmate;
  }
  if (k < 2) return false;  // the opponent needs an action of its own
  const t5d_test::Reference ref = t5d_test::reference_actions(s, 2'000'000);
  if (ref.overflow || ref.successors.empty()) return false;
  for (const GameState& next : ref.successors) {
    GameTree tree(200'000, 2'000'000, 20'000'000);
    const GameValue v = tree.solve(next, k - 1);
    if (v.kind != GameValue::Kind::Win || v.distance > k - 1) return false;
  }
  return true;
}

inline DisproofClass root_class(const ProverOptions& o) {
  return DisproofClass{true, o.seeds, o.completion, disproof_shape(o)};
}

// H19 at the root after a search over external stores: a proof is in the
// solved store, a disproof in the disproof store with its class and depth -
// or, where the root was found lost or stalemated outright, that fact in the
// solved store, checked: a Loss the game tree confirms after each of E4's
// listed actions (loss_confirmed), a stalemate by E4's verdict - and never a
// Win for a disproven root. (MT-03's review, 2026-11-01: any solved-store
// entry other than a Win used to stand in for the disproof unchecked.)
// `facts` counts the roots that relied on a fact.
inline void exports_hold(const ProveResult& r, const GameState& s, const ProverOptions& o, const MemorySolvedStore& proofs,
                  const DisproofStore& disproofs, int* facts = nullptr) {
  if (r.status == ProofStatus::Proven) {
    CHECK(r.stats.proofs_exported >= 1 || r.stats.store_hits >= 1);
    const std::optional<Proof> p = proofs.find(s);
    REQUIRE(p.has_value());
    CHECK(p->kind == Proof::Kind::Win);
    CHECK(p->distance <= r.proof.distance);
  } else if (r.status == ProofStatus::Disproven) {
    const std::optional<Proof> p = proofs.find(s);
    CHECK(!p.has_value() || p->kind != Proof::Kind::Win);
    // Either the class-relative disproof at the search's depth, or a game fact
    // that implies it: the side to move proven lost (or stalemated) in the
    // solved store - found when the search met the root as the defender's and
    // read back at the root (make_turn's "the other side wins: no mate for A").
    if (disproofs.find(s, root_class(o)) >= o.depth) return;
    bool fact = false;
    if (p && p->kind == Proof::Kind::Loss) {
      fact = loss_confirmed(s, p->distance);
    } else if (p && p->kind == Proof::Kind::Draw) {
      fact = stalemate_now(s);
    }
    if (fact) {
      if (facts != nullptr) ++*facts;
      return;
    }
    std::fprintf(stderr, "  disproven, no disproof recorded and no confirmed fact; the solved store holds %s at %s\n",
                 p ? to_string(*p).c_str() : "nothing", describe_position(s).c_str());
    CHECK(false);
  }
}

}  // namespace t5d_test

#endif  // T5D_TEST_STORE_CHECKS_HPP
