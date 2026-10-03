// The mate references of the proof-channel tests (moved here from
// test_prover.cpp in stage E9, unchanged, for Guard's and Hunt's tests too):
// the scanner's checking move made literal, the canonical completion by its
// definition, the restricted game (mate within d by the attacker's class
// against every legal defence, successors from the brute-force reference or
// E4's listing), and a result's check against them and the game tree.

#ifndef T5D_TEST_MATE_REFS_HPP
#define T5D_TEST_MATE_REFS_HPP

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/prover.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "test_game_tree.hpp"
#include "test_reference.hpp"

namespace t5d_test {

using namespace t5d;

inline std::string describe_position(const GameState& s) {
  std::string out = s.to_move() == Colour::White ? "white to move:" : "black to move:";
  const Multiverse& mv = s.multiverse();
  mv.for_each_board([&](int p, int h, const Board& b) {
    out += ' ';
    out += fen_board_string(b, p, h, mv.even());
  });
  out += " size " + to_string(mv.board_size()) + " rules " + describe(s.rules());
  return out;
}

// ---- the restricted game: the default class, independently ----------------------------------

// The scanner's (b) made literal (test_scanner.cpp's checks_reference): the
// move made, every board it created passed, and a royal capture from those
// copies among every capture check.hpp finds.
inline bool checks_reference(const GameState& s, const Move& m) {
  GameState t = s;
  t.apply(m);
  Multiverse mv = t.multiverse();
  std::vector<int> created{m.from_timeline};
  if (m.jump()) {
    created.push_back(!m.branching() ? m.to_timeline
                                     : (s.to_move() == Colour::White ? mv.max_timeline()
                                                                     : mv.min_timeline()));
  }
  std::vector<std::pair<int, int>> places;
  for (const int p : created) {
    BoardBuilder pass(*mv.head(p));
    pass.set_en_passant(kNoSquare);
    mv.append_board(p, pass.publish());
    places.emplace_back(p, mv.timeline(p)->last_half_turn());
  }
  return for_each_royal_capture(mv, s.rules(), s.to_move(), t.inactive_timelines(),
                                [&](const RoyalCapture& c) {
                                  return std::find(places.begin(), places.end(),
                                                   std::make_pair(c.from_timeline,
                                                                  c.from_half_turn)) != places.end();
                                });
}

// The canonical completion by its definition (engine/README.md "How a turn
// is completed incrementally"), walked without pruning: axes by ascending
// timeline; on each still-playable one, a pass first if the board is not
// mandatory, then its non-branching moves quietest first (the scanner's
// rank: captures last, jumps before them, by the moving piece's value,
// royals late; generator order among equals), a pass last if it is; then
// the branch phase - submit if the obligations are met and every royal is
// safe, else each branching move in generator order. The first legal leaf.
// Its only pruning is the rulebook's lemma (E5a's tier 0) tested from scratch:
// a royal capture standing in a partial action (royals_safe false) lasts to
// its end (not under Q18's alternative). Capped: beyond `limit` steps it
// gives up (`capped`), and the test compares nothing.
struct CompletionRef {
  PieceValues values;
  std::uint64_t limit = 300000;
  mutable std::uint64_t steps = 0;
  mutable bool capped = false;
  long rank(const Move& m) const {
    long r = m.piece.royal() ? 100000L : static_cast<long>(values.of(m.piece.kind()));
    if (!m.physical()) r += 1000000L;
    if (m.capture()) r += 2000000L;
    return r;
  }
  bool walk(const GameState& t, const std::vector<int>& axes, std::size_t j, bool branch_phase,
            std::vector<Move>& path, std::unique_ptr<GameState>& out) const {
    if (++steps > limit) {
      capped = true;
      return false;
    }
    if (!t.rules().sliders_cross_void && !royals_safe(t)) return false;
    if (!branch_phase) {
      while (j < axes.size() && !t.playable_by_mover(axes[j])) ++j;
      if (j < axes.size()) {
        const int a = axes[j];
        MoveList all;
        if (t.may_move()) append_board_moves(t, a, all);
        std::vector<Move> plain;
        for (const Move& m : all) {
          if (!m.branching()) plain.push_back(m);
        }
        std::stable_sort(plain.begin(), plain.end(),
                         [&](const Move& x, const Move& y) { return rank(x) < rank(y); });
        const bool mandatory = t.mandatory(a);
        if (!mandatory && walk(t, axes, j + 1, false, path, out)) return true;
        for (const Move& m : plain) {
          GameState u = t;
          u.apply(m);
          path.push_back(m);
          if (walk(u, axes, j + 1, false, path, out)) return true;
          path.pop_back();
        }
        return mandatory && walk(t, axes, j + 1, false, path, out);
      }
    }
    if (t.obligations_met() && royals_safe(t)) {
      out = std::make_unique<GameState>(t);
      return true;
    }
    if (!t.may_move()) return false;
    MoveList all;
    generate_moves(t, all);
    for (const Move& b : all) {
      if (!b.branching()) continue;
      GameState u = t;
      u.apply(b);
      path.push_back(b);
      if (walk(u, axes, axes.size(), true, path, out)) return true;
      path.pop_back();
    }
    return false;
  }
  std::optional<GameState> complete(const GameState& s, const Move& seed) const {
    steps = 0;
    capped = false;
    GameState t = s;
    t.apply(seed);
    std::vector<int> axes = s.playable_timelines();
    std::vector<Move> path{seed};
    std::unique_ptr<GameState> out;
    if (!walk(t, axes, 0, false, path, out)) return std::nullopt;
    out->submit(CheckRuling::none());
    return GameState(*out);
  }
};

// Whether the attacker forces mate within `depth` half-turns in the
// restricted game: its options the canonical completions of its checking
// seeds, the defender's every legal action (the brute-force reference). -1:
// the reference overflowed.
class RestrictedGame {
 public:
  // `literal`: A's options are every legal action (the question "mate within
  // d" itself); else the default class's, computed as above.
  RestrictedGame(Colour attacker, bool literal, std::uint64_t cap)
      : attacker_(attacker), literal_(literal), cap_(cap) {}
  int mates(const GameState& s, int depth) {
    const std::uint64_t key = s.zobrist() * 131 + static_cast<std::uint64_t>(depth);
    for (const Memo& m : memo_[key]) {
      if (same_position(m.state, s)) return m.value;
    }
    const int v = compute(s, depth);
    memo_[key].push_back(Memo{s, v});
    return v;
  }
  std::uint64_t work = 0;
  int by_brute_force = 0;  // successor sets from the brute-force reference
  int by_e4 = 0;           // ... and from E4's exhaustive listing

 private:
  struct Memo {
    GameState state;
    int value;
  };
  // E4's node budget within what is left of the reference's work (a node is
  // an eighth of a unit): the cap bounds a call, not only the next one.
  std::uint64_t e4_nodes(std::uint64_t most) const {
    const std::uint64_t left = work < cap_ ? cap_ - work : 0;
    return std::max<std::uint64_t>(1000, std::min<std::uint64_t>(most, 8 * left));
  }
  // Every legal successor: the brute-force reference's enumeration where it
  // fits, else E4's exhaustive listing (the hypercuboid detector - another
  // algorithm than the prover's semimove tree, itself held to the
  // reference). False: neither could list them.
  bool successors(const GameState& s, std::vector<GameState>& out) {
    out.clear();
    const t5d_test::Reference r = t5d_test::reference_actions(s, 5000);
    work += r.sequences;
    if (!r.overflow) {
      ++by_brute_force;
      out = r.successors;
      return true;
    }
    const ActionList l = list_legal_actions(s, 200000, Detector::Hypercuboid, SearchBudget::nodes(e4_nodes(3'000'000)));
    work += l.stats.nodes / 8;
    if (!l.stats.exhaustive || l.reached_cap) return false;
    ++by_e4;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> seen;
    for (const LegalAction& a : l.actions) {
      std::vector<std::size_t>& slot = seen[a.result.zobrist()];
      if (std::any_of(slot.begin(), slot.end(), [&](std::size_t i) { return same_position(out[i], a.result); })) {
        continue;
      }
      slot.push_back(out.size());
      out.push_back(a.result);
    }
    return true;
  }
  int compute(const GameState& s, int depth) {
    if (work > cap_) return -1;
    if (s.to_move() == attacker_) {
      if (depth <= 0) return 0;
      bool unknown = false;
      if (literal_) {
        std::vector<GameState> next;
        if (!successors(s, next)) return -1;
        for (const GameState& c : next) {
          const int v = mates(c, depth - 1);
          if (v == 1) return 1;
          if (v < 0) unknown = true;
        }
        return unknown ? -1 : 0;
      }
      MoveList moves;
      generate_moves(s, moves);
      for (const Move& m : moves) {
        if (!checks_reference(s, m)) continue;
        ++work;
        ref_.limit = std::max<std::uint64_t>(1000, std::min<std::uint64_t>(300000, cap_ > work ? cap_ - work : 0));
        const std::optional<GameState> next = ref_.complete(s, m);
        work += ref_.steps;
        if (ref_.capped) unknown = true;
        if (!next) continue;
        const int v = mates(*next, depth - 1);
        if (v == 1) return 1;
        if (v < 0) unknown = true;
      }
      return unknown ? -1 : 0;
    }
    if (depth <= 0) {
      // Mated now, or not: E4's verdict (exhaustive, H37).
      const MateResult m = detect_mate(s, Detector::Hypercuboid, SearchBudget::nodes(e4_nodes(2'000'000)));
      work += m.stats.nodes / 8;
      if (m.verdict == Outcome::Unknown) return -1;
      return m.verdict == Outcome::Checkmate ? 1 : 0;
    }
    std::vector<GameState> next;
    if (!successors(s, next)) return -1;
    if (next.empty()) return in_check(s) ? 1 : 0;  // checkmate, or stalemate
    bool unknown = false;
    for (const GameState& c : next) {
      const int v = mates(c, depth - 1);
      if (v == 0) return 0;
      if (v < 0) unknown = true;
    }
    return unknown ? -1 : 1;
  }
  Colour attacker_;
  bool literal_;
  std::uint64_t cap_;
  CompletionRef ref_;
  std::unordered_map<std::uint64_t, std::vector<Memo>> memo_;
};

// Checks a result: a proof against the brute-force game tree (the mate
// within its distance), or, where the tree runs out, against the literal
// mate reference; a disproof against the mate reference of its class
// (literal class: every legal action; default class: the class's actions).
// 1 agrees, -1 contradicts, 0 no reference could decide.
inline int verify(const ProveResult& r, const GameState& s, const ProverOptions& o,
           std::uint64_t cap = 300'000) {
  if (r.status == ProofStatus::Unknown) return 1;
  const bool literal_class = o.seeds == SeedClass::All && o.completion == CompletionClass::Every;
  const bool default_class = o.seeds == SeedClass::Checks && o.completion == CompletionClass::Canonical;
  if (r.status == ProofStatus::Proven) {
    if (r.proof.kind != Proof::Kind::Win || r.proof.distance > o.depth) return -1;
    if (default_class) {
      // A mate by the class's own actions against every defence is a mate:
      // the cheapest exact check (its actions computed independently).
      RestrictedGame g(s.to_move(), false, cap);
      const int m = g.mates(s, r.proof.distance);
      if (m >= 0) return m == 1 ? 1 : -1;
    }
    GameTree tree(20000, 60000, std::min<std::uint64_t>(2'000'000, 20 * cap));
    const GameValue v = tree.solve(s, r.proof.distance);
    if (v.kind != GameValue::Kind::Unknown) {
      return v.kind == GameValue::Kind::Win && v.distance <= r.proof.distance ? 1 : -1;
    }
    RestrictedGame g(s.to_move(), true, cap);
    const int m = g.mates(s, r.proof.distance);
    return m < 0 ? 0 : m == 1 ? 1 : -1;
  }
  if (!literal_class && !default_class) return 1;
  RestrictedGame g(s.to_move(), literal_class, cap);
  const int m = g.mates(s, o.depth);
  return m < 0 ? 0 : m == 1 ? -1 : 1;
}

struct Tally {
  int proven = 0;
  int disproven = 0;
  int unknown = 0;
  int confirmed = 0;
  int undecided = 0;
  std::uint64_t work_cap = 300'000;  // each reference's work
  bool add(const ProveResult& r, const GameState& s, const ProverOptions& o) {
    (r.status == ProofStatus::Proven ? proven
     : r.status == ProofStatus::Disproven ? disproven
                                          : unknown) += 1;
    const int v = verify(r, s, o, work_cap);
    if (v == 1 && r.status != ProofStatus::Unknown) ++confirmed;
    if (v == 0) ++undecided;
    if (v < 0) {
      std::fprintf(stderr, "  CONTRADICTED: %s %s at %s (class %s/%s, depth %d)\n",
                   proof_status_name(r.status), to_string(r.proof).c_str(),
                   describe_position(s).c_str(), seed_class_name(o.seeds),
                   completion_class_name(o.completion), o.depth);
    }
    return v >= 0;
  }
  void print(const char* what) const {
    std::fprintf(stderr, "  %s: %d proven, %d disproven, %d unknown; %d confirmed, %d beyond the references\n",
                 what, proven, disproven, unknown, confirmed, undecided);
  }
};

// Positions with a legal action for the side to move, from random play.
inline std::vector<GameState> playable(std::vector<GameState> in) {
  std::vector<GameState> out;
  for (GameState& s : in) {
    const MateResult m = detect_mate(s, Detector::Hypercuboid, SearchBudget::nodes(200000));
    if (m.verdict == Outcome::Ongoing) out.push_back(std::move(s));
  }
  return out;
}

}  // namespace t5d_test

#endif  // T5D_TEST_MATE_REFS_HPP
