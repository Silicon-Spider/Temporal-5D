// The hypercuboid detector (see t5d/search.hpp). Its design is the engine
// README's "The hypercuboid detector: design", implemented from that note.
//
// The method - a turn as one semimove per timeline, "play nothing here" among
// them, searched as a product space - originates in penteract's cwmtt and is
// adapted in ftxi's 5dchess_engine (design spec §11.0 item 3). This
// implementation was designed from the spec's description and the rulebook
// (docs/rules/README.md) alone: neither oracle's source or documentation was
// read for it, so that both stay independent checks of it (AGENTS.md).
//
// In brief: the axes are the mover's playable boards at the turn's start; a
// point gives each axis one value (Pass, a move within the board, one half of
// a jump between two axes, or a branching jump); the unexplored space is a
// stack of disjoint boxes; each step judges one point of a box - through the
// state machine itself - and removes from the box a region (a box) that
// contains the point and provably no legal action.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "search_internal.hpp"
#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"

namespace t5d::detail {

namespace {

// ---- values and axes ---------------------------------------------------------------------

enum class Kind : std::uint8_t { Pass, Physical, Depart, Arrive, Branch };

constexpr std::size_t kMaxChanges = 4;  // castling changes four squares, the most

struct Value {
  Kind kind = Kind::Pass;
  int move = -1;           // into Hyper::moves_
  // Depart/Arrive: the other axis of the jump, and the matching value there.
  // Branch: the axis that must be played first (J6), or -1.
  int partner = -1;
  int partner_value = -1;
  // Branch: its new board lies before the present - only such a creation can
  // rewind the present (T8).
  bool rewinds = false;
  // The board this value leaves on the axis's timeline (its successor), as
  // changes to the axis's playable board. Pass leaves none.
  std::array<std::int8_t, kMaxChanges> squares{};
  std::array<Piece, kMaxChanges> pieces{};
  std::size_t changes = 0;
};

struct Axis {
  int timeline = 0;
  int head = 0;  // the playable board's half-turn
  const Board* board = nullptr;
  bool required = false;  // mandatory at the turn's start (T6, Q5)
  std::vector<Value> values;  // values[0] is Pass
  std::size_t offset = 0;     // its first word in a box
  std::size_t words = 0;
};

// ---- boxes: one subset of values per axis, as bits ------------------------------------------

using Box = std::vector<std::uint64_t>;
using Bits = std::vector<std::uint64_t>;  // one axis's subset

inline bool has(const Box& box, const Axis& a, int v) {
  const auto u = static_cast<std::size_t>(v);
  return ((box[a.offset + u / 64] >> (u % 64)) & 1u) != 0;
}
inline bool bit(const Bits& bits, int v) {
  const auto u = static_cast<std::size_t>(v);
  return ((bits[u / 64] >> (u % 64)) & 1u) != 0;
}
inline void set_bit(Bits& bits, int v) {
  const auto u = static_cast<std::size_t>(v);
  bits[u / 64] |= std::uint64_t{1} << (u % 64);
}

Bits all_values(const Axis& a) {
  Bits out(a.words, 0);
  for (std::size_t v = 0; v < a.values.size(); ++v) out[v / 64] |= std::uint64_t{1} << (v % 64);
  return out;
}
Bits only(const Axis& a, int v) {
  Bits out(a.words, 0);
  set_bit(out, v);
  return out;
}
std::size_t count_in_box(const Box& box, const Axis& a) {
  std::size_t n = 0;
  for (std::size_t w = 0; w < a.words; ++w) {
    n += static_cast<std::size_t>(std::popcount(box[a.offset + w]));
  }
  return n;
}
std::size_t count_common(const Box& box, const Axis& a, const Bits& bits) {
  std::size_t n = 0;
  for (std::size_t w = 0; w < a.words; ++w) {
    n += static_cast<std::size_t>(std::popcount(box[a.offset + w] & bits[w]));
  }
  return n;
}

// A region: a box given by its constrained axes (every other axis free).
struct Region {
  std::vector<int> axes;
  std::vector<Bits> allowed;

  void restrict(int axis, const Bits& bits) {
    for (std::size_t i = 0; i < axes.size(); ++i) {
      if (axes[i] == axis) {
        for (std::size_t w = 0; w < bits.size(); ++w) allowed[i][w] &= bits[w];
        return;
      }
    }
    axes.push_back(axis);
    allowed.push_back(bits);
  }
  void clear() {
    axes.clear();
    allowed.clear();
  }
};

// The remainder of `box` once `region` is removed: one box per constrained axis
// (the region's subsets on the axes before it, the box's minus the region's on
// it, the box's elsewhere), each pushed if it holds a point.
void subtract(const Box& box, const Region& region, const std::vector<Axis>& axes,
              std::vector<Box>& out) {
  Box prefix = box;
  for (std::size_t i = 0; i < region.axes.size(); ++i) {
    const Axis& a = axes[static_cast<std::size_t>(region.axes[i])];
    const Bits& r = region.allowed[i];
    Box rest = prefix;
    bool rest_empty = true;
    bool inside_empty = true;
    for (std::size_t w = 0; w < a.words; ++w) {
      rest[a.offset + w] = prefix[a.offset + w] & ~r[w];
      prefix[a.offset + w] &= r[w];
      if (rest[a.offset + w] != 0) rest_empty = false;
      if (prefix[a.offset + w] != 0) inside_empty = false;
    }
    if (!rest_empty) out.push_back(std::move(rest));
    if (inside_empty) return;  // the rest of the box lies outside the region
  }
}

// ---- the search -------------------------------------------------------------------------------

class Hyper {
 public:
  explicit Hyper(SearchRun& run)
      : run_(run),
        s0_(run.start),
        rules_(run.start.rules()),
        budget_(run.budget, run.stats),
        mover_(run.start.to_move()) {}

  void search() {
    bool stopped = !build();
    std::vector<Box> stack;
    if (!stopped) stack.push_back(full_box());
    Region region;
    while (!stopped && !stack.empty()) {
      if (!budget_.hold(run_.successors.size() + stack.size())) {
        stopped = true;
        break;
      }
      Box box = std::move(stack.back());
      stack.pop_back();
      if (!propagate(box)) continue;  // no point of it can be an action
      const std::vector<int> x = choose(box);
      region.clear();
      if (!judge(x, box, region)) {
        stopped = true;
        break;
      }
      for (std::size_t i = 0; i < region.axes.size(); ++i) {
        if (!bit(region.allowed[i], x[static_cast<std::size_t>(region.axes[i])])) {
          throw std::logic_error("hypercuboid: a region that does not hold its own point");
        }
      }
      ++run_.stats.regions;
      subtract(box, region, axes_, stack);
    }
    // Exhaustive: every box was searched - no budget stop, no early stop at
    // a legal action (a search for one) or at the cap.
    run_.stats.exhaustive = !stopped && stack.empty();
    run_.stats.seconds = budget_.seconds();
  }

 private:
  // -- set-up --------------------------------------------------------------------------------

  // The axes, their values and the successors' changes. False if the budget
  // stopped it.
  bool build() {
    const Multiverse& mv = s0_.multiverse();
    min_timeline_ = mv.min_timeline();
    max_timeline_ = mv.max_timeline();
    axis_of_.assign(static_cast<std::size_t>(max_timeline_ - min_timeline_ + 1), -1);
    for (const TimelineRef& t : mv.timelines()) {
      if (to_move_at(t->last_half_turn()) != mover_) continue;
      Axis a;
      a.timeline = t->position();
      a.head = t->last_half_turn();
      a.board = t->head().get();
      a.required = s0_.mandatory(a.timeline);
      a.values.push_back(Value{});  // Pass
      axis_of_[static_cast<std::size_t>(a.timeline - min_timeline_)] =
          static_cast<int>(axes_.size());
      axes_.push_back(std::move(a));
    }
    fast_ = fast_rules();
    const Colour opponent = opposite(mover_);
    first_creation_active_ =
        mv.created_by(opponent) + credit(opponent) >= mv.created_by(mover_) + credit(mover_);
    // The candidate moves, the two activity-dependent restrictions lifted (Q21's
    // castling on optional boards, Q20's landing on an inactive timeline's
    // history): whether a move is really allowed when it is played is the
    // state machine's to say, at realisation.
    RuleOptions relaxed = rules_;
    relaxed.castling_on_optional_boards = true;
    relaxed.inactive_history_is_destination = true;
    MoveList list;
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      list.clear();
      append_moves_of(mv, relaxed, mover_, axes_[i].timeline, list, s0_.inactive_timelines());
      for (const Move& m : list) {
        if (!add_move(static_cast<int>(i), m)) return false;
      }
    }
    std::size_t words = 0;
    for (Axis& a : axes_) {
      a.offset = words;
      a.words = (a.values.size() + 63) / 64;
      words += a.words;
    }
    words_ = words;
    return true;
  }

  // Whether the realisation's fast scheme is exact: activity changes only at
  // creations, by the count rule, nothing can be reactivated (T11: only the
  // opponent's inactive timelines could be), and Q4's default holds.
  bool fast_rules() const {
    if (rules_.moves_after_shift || rules_.activation != ActivationRule::Count ||
        rules_.play_activates || rules_.reactivation != Reactivation::Rewinds) {
      return false;
    }
    const Colour opponent = opposite(mover_);
    for (const int p : s0_.inactive_timelines()) {
      if (s0_.multiverse().timeline(p)->creator() == opponent) return false;
    }
    return true;
  }

  // Q2's credit (T10), as the state applies it.
  int credit(Colour c) const {
    return rules_.initial_credit == InitialTimelineCredit::MinusZeroToBlack &&
                   s0_.multiverse().even() && c == Colour::Black
               ? 1
               : 0;
  }

  int axis_of(int timeline) const {
    if (timeline < min_timeline_ || timeline > max_timeline_) return -1;
    return axis_of_[static_cast<std::size_t>(timeline - min_timeline_)];
  }

  // The changes `next` makes to axis a's playable board.
  static void set_changes(Value& v, const Board& head, const Board& next) {
    v.changes = 0;
    for (int sq = 0; sq < kSquareCount; ++sq) {
      if (head.at(sq) == next.at(sq)) continue;
      if (v.changes == kMaxChanges) throw std::logic_error("hypercuboid: a move changes too much");
      v.squares[v.changes] = static_cast<std::int8_t>(sq);
      v.pieces[v.changes] = next.at(sq);
      ++v.changes;
    }
  }

  bool add_move(int i, const Move& m) {
    Axis& a = axes_[static_cast<std::size_t>(i)];
    const int index = static_cast<int>(moves_.size());
    moves_.push_back(m);
    if (m.branching()) {
      // Plays its source alone (J5): the source loses the piece.
      Value v;
      v.kind = Kind::Branch;
      v.move = index;
      v.rewinds = first_creation_active_ && m.to_half_turn + 1 < s0_.present();
      v.changes = 1;
      v.squares[0] = static_cast<std::int8_t>(m.from_square);
      v.pieces[0] = Piece{};
      a.values.push_back(v);
      return true;
    }
    // A move within the board, or a jump onto another axis's board: applied to
    // the turn's start for the boards it leaves.
    if (!budget_.spend()) return false;
    GameState t = s0_;
    t.apply(m);
    if (m.physical()) {
      Value v;
      v.kind = Kind::Physical;
      v.move = index;
      set_changes(v, *a.board, *t.multiverse().head(a.timeline));
      a.values.push_back(v);
      return true;
    }
    const int j = axis_of(m.to_timeline);
    if (j < 0) throw std::logic_error("hypercuboid: a jump onto a board that is no axis");
    Axis& b = axes_[static_cast<std::size_t>(j)];
    Value depart;
    depart.kind = Kind::Depart;
    depart.move = index;
    depart.partner = j;
    depart.partner_value = static_cast<int>(b.values.size());
    set_changes(depart, *a.board, *t.multiverse().head(a.timeline));
    Value arrive;
    arrive.kind = Kind::Arrive;
    arrive.move = index;
    arrive.partner = i;
    arrive.partner_value = static_cast<int>(a.values.size());
    set_changes(arrive, *b.board, *t.multiverse().head(b.timeline));
    a.values.push_back(depart);
    b.values.push_back(arrive);
    // J6: the same jump, made after axis j's board was played, lands on a
    // history board and branches. Its source loses the piece, as before.
    Move twin = m;
    twin.flags = static_cast<std::uint8_t>(twin.flags | Move::kBranching);
    Value branch = depart;
    branch.kind = Kind::Branch;
    branch.move = static_cast<int>(moves_.size());
    branch.partner = j;
    branch.partner_value = -1;
    branch.rewinds = first_creation_active_ && m.to_half_turn + 1 < s0_.present();
    moves_.push_back(twin);
    a.values.push_back(branch);
    return true;
  }

  Box full_box() const {
    Box box(words_, 0);
    for (const Axis& a : axes_) {
      const Bits all = all_values(a);
      for (std::size_t w = 0; w < a.words; ++w) box[a.offset + w] = all[w];
    }
    return box;
  }

  // -- narrowing a box ------------------------------------------------------------------------

  static bool only_pass(const Box& box, const Axis& a) {
    if (!has(box, a, 0)) return false;
    for (std::size_t w = 0; w < a.words; ++w) {
      const std::uint64_t rest = w == 0 ? box[a.offset] & ~std::uint64_t{1} : box[a.offset + w];
      if (rest != 0) return false;
    }
    return true;
  }
  static void drop(Box& box, const Axis& a, std::size_t v) {
    box[a.offset + v / 64] &= ~(std::uint64_t{1} << (v % 64));
  }

  // Removes from `box` the values no valid point of it can take, until none is
  // left to remove; false if the box then holds no point that could be an
  // action. Each removal is a region of the design's kinds: a jump's half
  // whose other half is gone (inconsistent), a J6 branch whose board can no
  // longer be played, and - under the fast scheme's rules - Pass on a
  // mandatory board when no value left could rewind the present (T5; under
  // Q5's alternative, Pass on any active board).
  bool propagate(Box& box) const {
    const std::size_t k = axes_.size();
    for (bool changed = true; changed;) {
      changed = false;
      for (std::size_t i = 0; i < k; ++i) {
        const Axis& a = axes_[i];
        for (std::size_t v = 1; v < a.values.size(); ++v) {
          if (!has(box, a, static_cast<int>(v))) continue;
          const Value& val = a.values[v];
          bool gone = false;
          if (val.kind == Kind::Depart || val.kind == Kind::Arrive) {
            gone = !has(box, axes_[static_cast<std::size_t>(val.partner)], val.partner_value);
          } else if (val.kind == Kind::Branch && val.partner >= 0) {
            gone = only_pass(box, axes_[static_cast<std::size_t>(val.partner)]);
          }
          if (gone) {
            drop(box, a, v);
            changed = true;
          }
        }
      }
      if (fast_) {
        bool rewind = false;
        if (rules_.mandatory == MandatoryBoards::InPresent) {
          for (std::size_t i = 0; i < k && !rewind; ++i) {
            const Axis& a = axes_[i];
            for (std::size_t v = 1; v < a.values.size() && !rewind; ++v) {
              rewind = a.values[v].rewinds && has(box, a, static_cast<int>(v));
            }
          }
        }
        if (!rewind) {
          for (std::size_t i = 0; i < k; ++i) {
            const Axis& a = axes_[i];
            if (!a.required || !has(box, a, 0)) continue;
            drop(box, a, 0);
            changed = true;
          }
        }
      }
      for (std::size_t i = 0; i < k; ++i) {
        if (count_in_box(box, axes_[i]) == 0) return false;
      }
    }
    return true;
  }

  // -- choosing a point -------------------------------------------------------------------------

  // A point of `box`, consistent where the box allows: the halves of a jump
  // together, a J6 branch's board played. Required boards are played if they
  // can be, optional ones left (fewer moves, fewer ways to fail).
  std::vector<int> choose(const Box& box) const {
    const std::size_t k = axes_.size();
    std::vector<int> x(k, -1);
    std::vector<int> forced(k, -1);
    std::vector<char> must_play(k, 0);
    for (std::size_t i = 0; i < k; ++i) {
      const Axis& a = axes_[i];
      if (forced[i] >= 0 && has(box, a, forced[i])) {
        x[i] = forced[i];
        continue;
      }
      const auto fits = [&](int v) {
        if (!has(box, a, v)) return false;
        const Value& val = a.values[static_cast<std::size_t>(v)];
        switch (val.kind) {
          case Kind::Pass:
            return must_play[i] == 0;
          case Kind::Depart:
          case Kind::Arrive: {
            const auto j = static_cast<std::size_t>(val.partner);
            if (j < i) return x[j] == val.partner_value;
            return forced[j] < 0 && has(box, axes_[j], val.partner_value);
          }
          case Kind::Branch:
            if (val.partner >= 0 && static_cast<std::size_t>(val.partner) < i) {
              return x[static_cast<std::size_t>(val.partner)] != 0;
            }
            return true;
          case Kind::Physical:
            return true;
        }
        return true;
      };
      const int n = static_cast<int>(a.values.size());
      int pick = -1;
      const bool play_first = a.required || must_play[i] != 0;
      if (!play_first && fits(0)) pick = 0;
      for (int v = 1; pick < 0 && v < n; ++v) {
        if (fits(v)) pick = v;
      }
      if (pick < 0 && fits(0)) pick = 0;
      for (int v = 0; pick < 0 && v < n; ++v) {
        if (has(box, a, v)) pick = v;  // nothing fits: an inconsistent point
      }
      x[i] = pick;
      const Value& val = a.values[static_cast<std::size_t>(pick)];
      if ((val.kind == Kind::Depart || val.kind == Kind::Arrive) &&
          static_cast<std::size_t>(val.partner) > i) {
        forced[static_cast<std::size_t>(val.partner)] = val.partner_value;
      }
      if (val.kind == Kind::Branch && val.partner >= 0 && static_cast<std::size_t>(val.partner) > i) {
        must_play[static_cast<std::size_t>(val.partner)] = 1;
      }
    }
    return x;
  }

  // -- judging a point ----------------------------------------------------------------------------

  const Value& value_of(const std::vector<int>& x, std::size_t i) const {
    return axes_[i].values[static_cast<std::size_t>(x[i])];
  }

  void single_point(const std::vector<int>& x, Region& region) const {
    region.clear();
    for (std::size_t i = 0; i < axes_.size(); ++i) region.restrict(static_cast<int>(i), only(axes_[i], x[i]));
  }

  // Judges x and sets `region` (it holds x and no legal action). False: stop
  // the search (the budget, or the cap reached - a search for one stops at
  // its first legal action).
  bool judge(const std::vector<int>& x, const Box& box, Region& region) {
    ++run_.stats.points;
    const std::size_t k = axes_.size();
    // (1) Consistency: a jump's halves together.
    for (std::size_t i = 0; i < k; ++i) {
      const Value& v = value_of(x, i);
      if (v.kind != Kind::Depart && v.kind != Kind::Arrive) continue;
      const auto j = static_cast<std::size_t>(v.partner);
      if (x[j] == v.partner_value) continue;
      region.restrict(static_cast<int>(i), only(axes_[i], x[i]));
      Bits others = all_values(axes_[j]);
      others[static_cast<std::size_t>(v.partner_value) / 64] &=
          ~(std::uint64_t{1} << (static_cast<std::size_t>(v.partner_value) % 64));
      region.restrict(static_cast<int>(j), others);
      return true;
    }
    // (2) J6: a branch onto another axis's board needs that board played.
    for (std::size_t i = 0; i < k; ++i) {
      const Value& v = value_of(x, i);
      if (v.kind != Kind::Branch || v.partner < 0) continue;
      const auto j = static_cast<std::size_t>(v.partner);
      if (x[j] != 0) continue;
      region.restrict(static_cast<int>(i), only(axes_[i], x[i]));
      region.restrict(static_cast<int>(j), only(axes_[j], 0));
      return true;
    }
    // (2b) J6 in a cycle: branches each needing the next one's board played
    // first - no order plays them. Every combination of such branches around
    // the same cycle of boards fails alike.
    for (std::size_t i = 0; i < k; ++i) {
      std::vector<std::size_t> cycle{i};
      std::size_t at = i;
      bool closed = false;
      while (value_of(x, at).kind == Kind::Branch && value_of(x, at).partner >= 0) {
        at = static_cast<std::size_t>(value_of(x, at).partner);
        if (at == i) {
          closed = true;
          break;
        }
        if (std::find(cycle.begin(), cycle.end(), at) != cycle.end()) break;
        cycle.push_back(at);
      }
      if (!closed) continue;
      for (std::size_t c = 0; c < cycle.size(); ++c) {
        const Axis& a = axes_[cycle[c]];
        const int next = static_cast<int>(cycle[(c + 1) % cycle.size()]);
        Bits bits(a.words, 0);
        for (std::size_t v = 0; v < a.values.size(); ++v) {
          if (a.values[v].kind == Kind::Branch && a.values[v].partner == next) {
            set_bit(bits, static_cast<int>(v));
          }
        }
        region.restrict(static_cast<int>(cycle[c]), bits);
      }
      return true;
    }
    // (3) T5: the present must shift (under the fast scheme's rules).
    if (fast_ && t5_region(x, region)) return true;
    // (4) Realisation and (5) royal safety (C4).
    bool legal = false;
    bool stop = false;
    bool eliminated = false;
    bool realised = false;
    const auto visit = [&](const GameState& final_state, const std::vector<Move>& seq) {
      realised = true;
      if (!budget_.spend()) {
        stop = true;
        return false;
      }
      ++run_.stats.actions_tested;
      if (royals_safe(final_state)) {
        legal = true;
        if (!run_.record(final_state, seq)) {
          stop = true;
          return false;
        }
        return true;
      }
      if (!legal && !eliminated && capture_region(x, box, final_state, seq, region)) {
        eliminated = true;  // every order fails alike: no need to try the others
        return false;
      }
      return true;
    };
    const bool complete = fast_ ? realise_fast(x, visit) : realise_general(x, visit);
    if (stop || budget_.stopped()) return false;
    (void)complete;
    (void)realised;
    if (legal || !eliminated) single_point(x, region);
    return true;
  }

  // The present never shifts (T5) - with the fast scheme's rules: under T6's
  // default, a mandatory board left unplayed and no creation that could rewind
  // the present (only an active one whose board lies before it, T8); under
  // Q5's alternative, an active board left unplayed.
  bool t5_region(const std::vector<int>& x, Region& region) const {
    const std::size_t k = axes_.size();
    if (rules_.mandatory == MandatoryBoards::AllActive) {
      for (std::size_t i = 0; i < k; ++i) {
        if (axes_[i].required && x[i] == 0) {
          region.restrict(static_cast<int>(i), only(axes_[i], 0));
          return true;
        }
      }
      return false;
    }
    for (std::size_t i = 0; i < k; ++i) {
      if (value_of(x, i).rewinds) return false;
    }
    for (std::size_t i = 0; i < k; ++i) {
      if (!axes_[i].required || x[i] != 0) continue;
      region.restrict(static_cast<int>(i), only(axes_[i], 0));
      for (std::size_t j = 0; j < k; ++j) {
        const Axis& a = axes_[j];
        Bits no_rewind = all_values(a);
        bool any = false;
        for (std::size_t v = 0; v < a.values.size(); ++v) {
          if (!a.values[v].rewinds) continue;
          no_rewind[v / 64] &= ~(std::uint64_t{1} << (v % 64));
          any = true;
        }
        if (any) region.restrict(static_cast<int>(j), no_rewind);
      }
      return true;
    }
    return false;
  }

  // -- realisation --------------------------------------------------------------------------------

  using Visit = std::function<bool(const GameState&, const std::vector<Move>&)>;

  // Plays `seq` from the turn's start: every move pseudo-legal where it is
  // played and allowed by the turn rules, the obligations met at the end.
  // nullopt if not; `stopped` if the budget ran out.
  std::optional<GameState> play(const std::vector<Move>& seq, bool& stopped) {
    GameState s = s0_;
    for (const Move& m : seq) {
      if (!s.may_move() || !is_pseudo_legal(s, m, scratch_)) return std::nullopt;
      if (!budget_.spend()) {
        stopped = true;
        return std::nullopt;
      }
      s.apply(m);
    }
    if (!s.obligations_met()) return std::nullopt;
    return s;
  }

  // The fast scheme (exact under fast_rules()): for each order of the
  // branching moves - J6's order respected - the other moves by ascending
  // source timeline, then the branching ones, then a last move L, tried for
  // each candidate L. Returns false if stopped.
  bool realise_fast(const std::vector<int>& x, const Visit& visit) {
    const std::size_t k = axes_.size();
    std::vector<int> plain;  // moves that create no timeline
    std::vector<std::size_t> branch_axes;
    for (std::size_t i = 0; i < k; ++i) {
      const Value& v = value_of(x, i);
      if (v.kind == Kind::Physical || v.kind == Kind::Depart) plain.push_back(v.move);
      if (v.kind == Kind::Branch) branch_axes.push_back(i);
    }
    // Which axes a move plays: its source, and a jump's destination.
    const auto plays_axis = [&](int move, int axis) {
      const Move& m = moves_[static_cast<std::size_t>(move)];
      return axis_of(m.from_timeline) == axis || (!m.physical() && !m.branching() &&
                                                  axis_of(m.to_timeline) == axis);
    };
    std::vector<std::size_t> order(branch_axes.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::vector<Move> seq;
    bool stopped = false;
    do {
      // J6 between branches: a branch onto an axis whose own move branches
      // comes after that move.
      bool valid = true;
      for (std::size_t p = 0; p < order.size() && valid; ++p) {
        const Value& b = value_of(x, branch_axes[order[p]]);
        if (b.partner < 0) continue;
        const Value& needed = value_of(x, static_cast<std::size_t>(b.partner));
        if (needed.kind != Kind::Branch) continue;
        for (std::size_t q = p + 1; q < order.size(); ++q) {
          if (branch_axes[order[q]] == static_cast<std::size_t>(b.partner)) valid = false;
        }
      }
      if (!valid) continue;
      // The candidates for the last move: the last branch, then each plain
      // move no branch needs played before it (J6).
      std::vector<int> candidates;
      if (!order.empty()) candidates.push_back(-1);
      for (std::size_t e = plain.size(); e-- > 0;) {
        bool needed = false;
        for (const std::size_t ba : branch_axes) {
          const Value& b = value_of(x, ba);
          if (b.partner >= 0 && plays_axis(plain[e], b.partner)) needed = true;
        }
        if (!needed) candidates.push_back(static_cast<int>(e));
      }
      for (const int last : candidates) {
        seq.clear();
        for (std::size_t e = 0; e < plain.size(); ++e) {
          if (static_cast<int>(e) != last) seq.push_back(moves_[static_cast<std::size_t>(plain[e])]);
        }
        for (std::size_t p = 0; p < order.size(); ++p) {
          if (last == -1 && p + 1 == order.size()) continue;
          seq.push_back(moves_[static_cast<std::size_t>(value_of(x, branch_axes[order[p]]).move)]);
        }
        if (last == -1) {
          seq.push_back(moves_[static_cast<std::size_t>(value_of(x, branch_axes[order.back()]).move)]);
        } else {
          seq.push_back(moves_[static_cast<std::size_t>(plain[static_cast<std::size_t>(last)])]);
        }
        std::optional<GameState> done = play(seq, stopped);
        if (stopped) return false;
        if (!done) continue;
        if (!visit(*done, seq)) return false;
        break;  // every working last move reaches the same state
      }
    } while (std::next_permutation(order.begin(), order.end()));
    return true;
  }

  // The general realisation (exact under any rules): every order of the
  // point's moves, depth-first, with a table of the states reached; each
  // distinct final state with the obligations met is visited once. Returns
  // false if stopped.
  bool realise_general(const std::vector<int>& x, const Visit& visit) {
    std::vector<Move> moves;
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      const Value& v = value_of(x, i);
      if (v.kind == Kind::Physical || v.kind == Kind::Depart || v.kind == Kind::Branch) {
        moves.push_back(moves_[static_cast<std::size_t>(v.move)]);
      }
    }
    PositionSet seen;
    PositionSet finals;
    std::vector<char> used(moves.size(), 0);
    std::vector<Move> path;
    bool stop = false;
    std::function<void(const GameState&)> dfs = [&](const GameState& s) {
      if (path.size() == moves.size()) {
        if (s.obligations_met() && finals.insert(s).second && !visit(s, path)) stop = true;
        return;
      }
      if (!s.may_move()) return;
      for (std::size_t m = 0; m < moves.size() && !stop; ++m) {
        if (used[m] != 0 || !is_pseudo_legal(s, moves[m], scratch_)) continue;
        if (!budget_.spend()) {
          stop = true;
          return;
        }
        GameState child = s;
        child.apply(moves[m]);
        if (!seen.insert(child).second) continue;
        used[m] = 1;
        path.push_back(moves[m]);
        dfs(child);
        path.pop_back();
        used[m] = 0;
      }
    };
    dfs(s0_);
    return !stop;
  }

  // -- the region of a royal capture ------------------------------------------------------------

  // The piece axis a's value v leaves on `square` of its successor board.
  Piece content(const Axis& a, int v, int square) const {
    const Value& val = a.values[static_cast<std::size_t>(v)];
    for (std::size_t c = 0; c < val.changes; ++c) {
      if (val.squares[c] == square) return val.pieces[c];
    }
    return a.board->at(square);
  }

  // The piece a branching move's new board holds on `square`: the
  // destination board's (J5: a copy of it), or the arriving piece. Compared
  // without the unmoved flag, which no capture reads.
  Piece new_board_piece(const Move& m, int square) const {
    if (square == m.to_square) {
      return m.promotes() ? Piece(m.promotion, mover_) : m.piece.as_moved();
    }
    return s0_.multiverse().board_at(m.to_timeline, m.to_half_turn)->at(square).as_moved();
  }

  // From a capture of a royal in `final_state` (x's realisation by `seq`), the
  // region of the points that share it in every realisation (the README's
  // design): the capture rests on its places; a place on an axis's successor
  // board constrains that axis to the values that leave the same pieces
  // there. A capture made within one board a branch created depends on that
  // board alone (a copy of its destination, and the arriving piece), whatever
  // its timeline's number: it constrains the branching axis to the branches
  // that leave the same pieces there. Any other capture touching a created
  // timeline is used only when x creates exactly one (its number is then
  // fixed). Of the captures, the one removing most of `box`. False if no
  // capture qualifies.
  bool capture_region(const std::vector<int>& x, const Box& box, const GameState& final_state,
                      const std::vector<Move>& seq, Region& out) {
    const std::size_t k = axes_.size();
    int sole_branch = -1;
    int branches = 0;
    for (std::size_t i = 0; i < k; ++i) {
      if (value_of(x, i).kind == Kind::Branch) {
        ++branches;
        sole_branch = static_cast<int>(i);
      }
    }
    if (branches != 1) sole_branch = -1;
    // A single creation's timeline: outermost on the mover's side (L7).
    const int created = mover_ == Colour::White ? max_timeline_ + 1 : min_timeline_ - 1;
    // Which axis's branch created each new timeline, in this realisation: the
    // j-th creation takes the j-th position out from the mover's side (L7, J9).
    std::vector<std::pair<int, int>> creators;  // (timeline, axis)
    for (const Move& m : seq) {
      if (!m.branching()) continue;
      const int j = static_cast<int>(creators.size()) + 1;
      creators.emplace_back(mover_ == Colour::White ? max_timeline_ + j : min_timeline_ - j,
                            axis_of(m.from_timeline));
    }
    const auto creator_axis = [&](int timeline) {
      for (const auto& [t, a] : creators) {
        if (t == timeline) return a;
      }
      return -1;
    };
    const Multiverse& mv = final_state.multiverse();
    double best = -std::numeric_limits<double>::infinity();
    bool found = false;
    int examined = 0;
    int usable = 0;
    Region r;
    for_each_royal_capture(
        mv, rules_, opposite(mover_), final_state.inactive_timelines(),
        [&](const RoyalCapture& c) {
          // Enough to choose from (captures that do not qualify are not
          // counted as usable).
          if (++examined > 512 || usable >= 24) return true;
          // Q20's alternative: landing on a history board depends on activity.
          const Timeline* target = mv.timeline(c.to_timeline);
          if (!rules_.inactive_history_is_destination &&
              (target == nullptr || target->last_half_turn() != c.to_half_turn)) {
            return false;
          }
          r.clear();
          bool touches_new = false;
          std::vector<CapturePlace> places = capture_places(c);
          // Within one new board: its creating axis's branches that leave the
          // same pieces on the capture's squares.
          if (c.step.physical() &&
              (c.from_timeline < min_timeline_ || c.from_timeline > max_timeline_)) {
            const int ai = creator_axis(c.from_timeline);
            if (ai < 0) throw std::logic_error("hypercuboid: a new timeline without its creator");
            const Axis& a = axes_[static_cast<std::size_t>(ai)];
            const Board* board = mv.board_at(c.from_timeline, c.from_half_turn);
            const Move& made = moves_[static_cast<std::size_t>(value_of(x, static_cast<std::size_t>(ai)).move)];
            Bits allowed(a.words, 0);
            for (std::size_t v = 1; v < a.values.size(); ++v) {
              if (a.values[v].kind != Kind::Branch) continue;
              const Move& m = moves_[static_cast<std::size_t>(a.values[v].move)];
              bool ok = true;
              for (std::size_t p = 0; p < places.size() && ok; ++p) {
                const Piece seen = board->at(places[p].square).as_moved();
                if (v == static_cast<std::size_t>(x[static_cast<std::size_t>(ai)]) &&
                    new_board_piece(made, places[p].square) != seen) {
                  throw std::logic_error("hypercuboid: a new board differs from its branch");
                }
                const Piece here = new_board_piece(m, places[p].square);
                const bool crossed = p != 0 && p + 1 != places.size();
                ok = crossed ? (here.empty() || here == seen) : here == seen;
              }
              if (ok) set_bit(allowed, static_cast<int>(v));
            }
            r.restrict(ai, allowed);
            places.clear();
          }
          for (std::size_t p = 0; p < places.size(); ++p) {
            const CapturePlace& place = places[p];
            if (place.timeline < min_timeline_ || place.timeline > max_timeline_) {
              if (sole_branch < 0 || place.timeline != created) return false;
              touches_new = true;
              continue;
            }
            const int ai = axis_of(place.timeline);
            if (ai < 0) continue;  // no axis: its boards never change
            const Axis& a = axes_[static_cast<std::size_t>(ai)];
            if (place.half_turn != a.head + 1) continue;  // not the successor: never changes
            const bool crossed = p != 0 && p + 1 != places.size();
            const Board* board = mv.board_at(place.timeline, place.half_turn);
            const int xi = x[static_cast<std::size_t>(ai)];
            // What x leaves there must be what the capture saw.
            if ((board == nullptr) != (xi == 0) ||
                (board != nullptr && board->at(place.square) != content(a, xi, place.square))) {
              throw std::logic_error("hypercuboid: a successor board differs from its value");
            }
            Bits allowed(a.words, 0);
            const Piece seen = board != nullptr ? board->at(place.square).as_moved() : Piece{};
            for (int v = 0; v < static_cast<int>(a.values.size()); ++v) {
              bool ok = false;
              if (v == 0) {
                ok = crossed && rules_.sliders_cross_void;  // void, crossed (Q18)
              } else {
                const Piece here = content(a, v, place.square).as_moved();
                ok = crossed ? (here.empty() || here == seen) : here == seen;
              }
              if (ok) set_bit(allowed, v);
            }
            r.restrict(ai, allowed);
          }
          if (touches_new) {
            const auto sb = static_cast<std::size_t>(sole_branch);
            r.restrict(sole_branch, only(axes_[sb], x[sb]));
            for (std::size_t j = 0; j < k; ++j) {
              if (j == sb) continue;
              const Axis& a = axes_[j];
              Bits no_branch = all_values(a);
              bool any = false;
              for (std::size_t v = 0; v < a.values.size(); ++v) {
                if (a.values[v].kind != Kind::Branch) continue;
                no_branch[v / 64] &= ~(std::uint64_t{1} << (v % 64));
                any = true;
              }
              if (any) r.restrict(static_cast<int>(j), no_branch);
            }
          }
          ++usable;
          // The share of the box the region removes (log of a product).
          double share = 0;
          for (std::size_t i = 0; i < r.axes.size(); ++i) {
            const Axis& a = axes_[static_cast<std::size_t>(r.axes[i])];
            const auto in_box = static_cast<double>(count_in_box(box, a));
            const auto common = static_cast<double>(count_common(box, a, r.allowed[i]));
            share += std::log(common / in_box);
          }
          if (share > best) {
            best = share;
            out = r;
            found = true;
          }
          return false;
        });
    return found;
  }

  SearchRun& run_;
  const GameState& s0_;
  const RuleOptions& rules_;
  Budgeter budget_;
  Colour mover_;
  std::vector<Axis> axes_;
  std::vector<Move> moves_;
  std::vector<int> axis_of_;
  int min_timeline_ = 0;
  int max_timeline_ = 0;
  bool fast_ = false;
  bool first_creation_active_ = false;
  std::size_t words_ = 0;
  MoveList scratch_;
};

}  // namespace

void run_hypercuboid(SearchRun& run) { Hyper(run).search(); }

}  // namespace t5d::detail
