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
// contains the point and provably no legal action. Outside the general
// regime an abstract check first asks whether any point of the box can be
// played in some order at all (feasible()): a box where none can is removed
// whole, and otherwise the check's skeleton picks the point.

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
#include <string>
#include <unordered_set>
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
  // Branch: whether this creation could be what shifts the present to the
  // opponent with a mandatory board unplayed (T8 rewinding, T12 reactivation)
  // - a superset of the creations that can; see Hyper::build.
  bool rewind_capable = false;
  // Physical: a castling (whose legality can depend on the present, Q21).
  bool castle = false;
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
inline void clear_bit(Bits& bits, std::size_t v) {
  bits[v / 64] &= ~(std::uint64_t{1} << (v % 64));
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
// The values of `a` that `keep` accepts.
template <class Keep>
Bits values_where(const Axis& a, Keep keep) {
  Bits out(a.words, 0);
  for (std::size_t v = 0; v < a.values.size(); ++v) {
    if (keep(a.values[v])) set_bit(out, static_cast<int>(v));
  }
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

constexpr int kNever = std::numeric_limits<int>::max();

// The most steps one abstract check may take (see Hyper::feasible); beyond
// it the check answers "unknown" and the search goes on without it.
constexpr std::size_t kAbstractSteps = 1u << 12;

// How a point's moves may be ordered - which realisation is exact.
enum class Regime : std::uint8_t {
  // Q4's default, and activity changed by creations alone, deterministically
  // in their number (count or consecutive activation, Rewinds; T11/T12
  // reactivation included); castling allowed on optional boards.
  Ordered,
  // Q4's other reading (moves after the shift), otherwise as Ordered.
  FreeOrder,
  // Anything else (Q3's play activation, Q6's deferral, castling on
  // mandatory boards only): every order is searched.
  General,
};

// ---- the search -------------------------------------------------------------------------------

class Hyper {
 public:
  explicit Hyper(SearchRun& run)
      : run_(run),
        s0_(run.start),
        rules_(run.start.rules()),
        budget_(run.budget, run.stats, position_bytes(run.start)),
        mover_(run.start.to_move()) {}

  void search() {
    // A state whose action is over (its obligations met and, Q4's default, no
    // move allowed - a search started after the moves of one): its only
    // completion is to stop there.
    if (s0_.obligations_met() && !s0_.may_move()) {
      bool done = budget_.spend();
      if (done) {
        ++run_.stats.actions_tested;
        if (royals_safe(s0_)) done = run_.record(s0_, {});
      }
      run_.stats.exhaustive = done;
      run_.stats.seconds = budget_.seconds();
      return;
    }
    bool stopped = !build();
    std::vector<Box> stack;
    if (!stopped) stack.push_back(full_box());
    Region region;
    while (!stopped && !stack.empty()) {
      if (!budget_.hold(run_.held() + stack.size(), run_.held_bytes())) {
        stopped = true;
        break;
      }
      Box box = std::move(stack.back());
      stack.pop_back();
      if (!propagate(box)) continue;  // no point of it can be an action
      std::vector<int> x;
      if (regime_ != Regime::General) {
        const Feasibility f = feasible(box, abstract_);
        // Its work counts against the budget: a node for the box, and one per
        // step of the check (a step costs a few microseconds at a dozen boards,
        // about what a node of the realisation does).
        if (!budget_.spend_many(1 + abstract_.steps)) {
          stopped = true;
          break;
        }
        if (f == Feasibility::No) {
          ++run_.stats.regions;  // the whole box: no skeleton of it is realisable
          ++run_.stats.unrealisable_boxes;
          continue;
        }
        if (f == Feasibility::Yes) x = from_witness(abstract_);
        if (f == Feasibility::Unknown) ++run_.stats.abstract_gave_up;
      }
      if (x.empty()) x = choose(box);
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
    present_ = s0_.present();
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
    regime_ = regime();
    in_present_ = rules_.mandatory == MandatoryBoards::InPresent;
    for (const Axis& a : axes_) heads_.push_back(a.head);
    std::sort(heads_.begin(), heads_.end());
    heads_.erase(std::unique(heads_.begin(), heads_.end()), heads_.end());
    // T11: the mover's j-th creation reactivates the opponent's j-th
    // earliest-created timeline still inactive - fixed at the turn's start, as
    // the opponent creates nothing during the mover's action.
    const Colour opponent = opposite(mover_);
    for (const int p : s0_.inactive_timelines()) {
      const Timeline* t = mv.timeline(p);
      if (t->creator() == opponent) reactivated_.push_back(t);
    }
    std::sort(reactivated_.begin(), reactivated_.end(), [](const Timeline* a, const Timeline* b) {
      return a->creation_order() < b->creation_order();
    });
    // A reactivation that could itself put the present on the opponent's colour
    // before the present (T12): a reactivated board of the opponent's there, or
    // one of the mover's there that the mover played first.
    for (const Timeline* t : reactivated_) {
      const int h = t->last_half_turn();
      if ((to_move_at(h) == opponent && h < present_) ||
          (to_move_at(h) == mover_ && h + 1 < present_)) {
        reactivation_threat_ = true;
      }
    }
    // The abstract check's view of the present (T4): from when each axis's
    // timeline counts for it - the start, the r-th creation (T11), or never -
    // and the earliest counting head among the other timelines (all of the
    // opponent's colour) after r creations. (Outside the general regime no
    // timeline is deferred (Q6), so the active ones are the counting ones.)
    counts_from_.assign(axes_.size(), kNever);
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      if (s0_.active(axes_[i].timeline)) counts_from_[i] = 0;
    }
    o_base_.assign(reactivated_.size() + 1, kNever);
    for (const TimelineRef& t : mv.timelines()) {
      if (axis_of(t->position()) >= 0 || !s0_.active(t->position())) continue;
      o_base_[0] = std::min(o_base_[0], t->last_half_turn());
    }
    for (std::size_t r = 0; r < reactivated_.size(); ++r) {
      const int ai = axis_of(reactivated_[r]->position());
      o_base_[r + 1] = o_base_[r];
      if (ai >= 0) {
        counts_from_[static_cast<std::size_t>(ai)] = static_cast<int>(r) + 1;
      } else {
        o_base_[r + 1] = std::min(o_base_[r + 1], reactivated_[r]->last_half_turn());
      }
    }
    // The creations that can be active (T10; Q1): the largest number b such
    // that an action's b-th creation is active.
    if (rules_.activation == ActivationRule::Count) {
      max_last_active_ = mv.created_by(opponent) + credit(opponent) - mv.created_by(mover_) -
                         credit(mover_) + 1;
    } else {
      max_last_active_ = s0_.last_creator() == mover_ ? 0 : 1;
    }
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

  Regime regime() const {
    if (rules_.play_activates || rules_.reactivation != Reactivation::Rewinds ||
        !rules_.castling_on_optional_boards) {
      return Regime::General;
    }
    return rules_.moves_after_shift ? Regime::FreeOrder : Regime::Ordered;
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

  // Where a created board at half-turn c lies among the boards of the mover's
  // that can count for the present - every axis's head (a reactivated one's
  // included): the number of them before it. That is all of c the state
  // machine reads: the present is the earliest counting head (T4), on the
  // mover's colour while a head of the mover's precedes every counting one of
  // the opponent's (T5, T8). Under Q5's alternative the obligations do not
  // read the present at all (0).
  int threshold_class(int c) const {
    if (!in_present_) return 0;
    return static_cast<int>(std::lower_bound(heads_.begin(), heads_.end(), c) - heads_.begin());
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

  // A branching move's flag (see Value): its creation's board lies at the
  // half-turn after its destination. Only a creation with its board before
  // the present can rewind it (T8) - and an active one (T10) - unless a
  // reactivation can (T12).
  void set_branch_flags(Value& v, const Move& m) const {
    const int c = m.to_half_turn + 1;
    v.rewind_capable = reactivation_threat_ || (max_last_active_ >= 1 && c < present_);
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
      set_branch_flags(v, m);
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
      v.castle = m.castle();
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
    set_branch_flags(branch, twin);
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
  // longer be played, and - unless the general regime - Pass on a mandatory
  // board when no value left could shift the present without it (T5, T8, T12;
  // under Q5's alternative, Pass on any active board).
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
      if (regime_ != Regime::General) {
        bool rewind = false;
        if (in_present_) {
          for (std::size_t i = 0; i < k && !rewind; ++i) {
            const Axis& a = axes_[i];
            for (std::size_t v = 1; v < a.values.size() && !rewind; ++v) {
              rewind = a.values[v].rewind_capable && has(box, a, static_cast<int>(v));
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

  // -- the abstract check: can any point of a box be realised? ------------------------------------
  //
  // Under the Ordered and FreeOrder regimes whether a point is realisable
  // depends only on its SKELETON: which axes it passes, which it plays with a
  // move that creates no timeline (a plain move), and of each branch its J6
  // target and threshold class (unrealised_region's (e) gives the reasons).
  // feasible() searches the skeletons a box allows, creation by creation, in
  // the normal form the realisation uses - plain moves as late as J6 allows:
  // after every creation (one of them last), or all just before the last
  // creation - reading only the counting heads (T4, T5, T8, T10-T12). It
  // over-approximates: it ignores Q20's activity test, and takes each plain
  // move as playing one board (exact for jumps, the README shows). So "no
  // skeleton" proves the box holds no action, and a skeleton found guides the
  // choice of the point judged.

  enum class Feasibility : std::uint8_t { No, Yes, Unknown };

  struct Assign {
    int axis;
    int value;  // -1: a plain move, chosen when the point is built
  };

  struct Abstract {
    std::vector<char> pass_ok, plain_ok;
    std::vector<int> physical;                            // a Physical value in the box, or -1
    std::vector<std::vector<std::pair<int, int>>> jumps;  // (value, other axis), one per other axis
    std::vector<std::vector<int>> kinds;                  // branch values, one per (target, class)
    std::unordered_set<std::string> failed;               // states searched without success
    std::size_t steps = 0;
    bool unknown = false;
    std::vector<Assign> witness;
  };

  int o_base(int k) const {
    return o_base_[std::min(static_cast<std::size_t>(k), o_base_.size() - 1)];
  }

  // Whether the obligations are met (T5, Q5) with the axes in `played`
  // played, after k creations whose earliest active new board is minc.
  bool met(const Bits& played, int k, int minc) const {
    int min_m = kNever;
    int min_o = std::min(o_base(k), minc);
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      if (counts_from_[i] > k) continue;
      if (bit(played, static_cast<int>(i))) {
        min_o = std::min(min_o, axes_[i].head + 1);
      } else {
        min_m = std::min(min_m, axes_[i].head);
      }
    }
    if (!in_present_) return min_m == kNever;  // Q5: every counting board of the mover's played
    return min_o < min_m;                       // T5: the present on the opponent's colour
  }

  Feasibility feasible(const Box& box, Abstract& ab) const {
    const std::size_t k = axes_.size();
    ab.pass_ok.assign(k, 0);
    ab.plain_ok.assign(k, 0);
    ab.physical.assign(k, -1);
    ab.jumps.assign(k, {});
    ab.kinds.assign(k, {});
    ab.failed.clear();
    ab.steps = 0;
    ab.unknown = false;
    ab.witness.clear();
    std::vector<std::pair<int, int>> seen;  // (target, class) of the branches kept
    for (std::size_t i = 0; i < k; ++i) {
      const Axis& a = axes_[i];
      seen.clear();
      for (std::size_t v = 0; v < a.values.size(); ++v) {
        if (!has(box, a, static_cast<int>(v))) continue;
        const Value& w = a.values[v];
        switch (w.kind) {
          case Kind::Pass: ab.pass_ok[i] = 1; break;
          case Kind::Physical:
            ab.plain_ok[i] = 1;
            if (ab.physical[i] < 0) ab.physical[i] = static_cast<int>(v);
            break;
          case Kind::Depart:
          case Kind::Arrive: {
            ab.plain_ok[i] = 1;
            auto& js = ab.jumps[i];
            if (std::none_of(js.begin(), js.end(), [&](const auto& j) { return j.second == w.partner; })) {
              js.emplace_back(static_cast<int>(v), w.partner);
            }
            break;
          }
          case Kind::Branch: {
            const std::pair<int, int> key{
                w.partner, threshold_class(moves_[static_cast<std::size_t>(w.move)].to_half_turn + 1)};
            if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
              seen.push_back(key);
              ab.kinds[i].push_back(static_cast<int>(v));
            }
            break;
          }
        }
      }
    }
    Bits played((k + 63) / 64, 0);
    if (abstract_step(ab, played, 0, kNever)) return Feasibility::Yes;
    return ab.unknown ? Feasibility::Unknown : Feasibility::No;
  }

  // From a state the action can go on from (Ordered: obligations unmet):
  // whether some skeleton of the box completes it. On success, ab.witness
  // holds the skeleton's moves.
  bool abstract_step(Abstract& ab, const Bits& played, int k, int minc) const {
    if (++ab.steps > kAbstractSteps) {
      ab.unknown = true;
      return false;
    }
    if (abstract_end(ab, played, k, minc)) return true;
    const bool ordered = regime_ == Regime::Ordered;
    std::vector<std::vector<Assign>> options;
    for (std::size_t a = 0; a < axes_.size(); ++a) {
      if (bit(played, static_cast<int>(a))) continue;
      for (const int v : ab.kinds[a]) {
        const Value& bv = axes_[a].values[static_cast<std::size_t>(v)];
        const int c = moves_[static_cast<std::size_t>(bv.move)].to_half_turn + 1;
        // J6: the board it lands on played first - by an earlier move, or by
        // a plain move made just before it (one board, or a jump's two).
        options.clear();
        if (bv.partner < 0 || bit(played, bv.partner)) {
          options.emplace_back();
        } else {
          const auto p = static_cast<std::size_t>(bv.partner);
          if (ab.physical[p] >= 0) options.push_back({{bv.partner, ab.physical[p]}});
          for (const auto& [jv, q] : ab.jumps[p]) {
            if (bit(played, q) || q == static_cast<int>(a)) continue;
            options.push_back({{bv.partner, jv},
                               {q, axes_[p].values[static_cast<std::size_t>(jv)].partner_value}});
          }
        }
        for (const std::vector<Assign>& early : options) {
          Bits before = played;
          for (const Assign& e : early) set_bit(before, e.axis);
          if (ordered && !early.empty() && met(before, k, minc)) continue;
          const int next_minc = k + 1 <= max_last_active_ ? std::min(minc, c) : minc;
          if (ordered && abstract_last(ab, before, k, minc, static_cast<int>(a), next_minc)) {
            ab.witness.insert(ab.witness.end(), early.begin(), early.end());
            ab.witness.push_back({static_cast<int>(a), v});
            return true;
          }
          Bits after = before;
          set_bit(after, static_cast<int>(a));
          if (ordered && met(after, k + 1, next_minc)) continue;  // the action ended: abstract_last's case
          std::string key(reinterpret_cast<const char*>(after.data()), after.size() * sizeof(std::uint64_t));
          key.append(reinterpret_cast<const char*>(&next_minc), sizeof next_minc);
          const int k1 = k + 1;
          key.append(reinterpret_cast<const char*>(&k1), sizeof k1);
          if (ab.failed.count(key) != 0) continue;
          if (abstract_step(ab, after, k + 1, next_minc)) {
            ab.witness.insert(ab.witness.end(), early.begin(), early.end());
            ab.witness.push_back({static_cast<int>(a), v});
            return true;
          }
          if (ab.unknown) return false;
          ab.failed.insert(std::move(key));
        }
      }
    }
    return false;
  }

  // The end after the creations (Ordered: plain moves after them, one last).
  bool abstract_end(Abstract& ab, const Bits& played, int k, int minc) const {
    const std::size_t n = axes_.size();
    Bits forced = played;  // with the boards that cannot be passed
    Bits most = played;    // with every board a plain move can play
    std::vector<int> must;
    for (std::size_t i = 0; i < n; ++i) {
      if (bit(played, static_cast<int>(i))) continue;
      if (ab.pass_ok[i] == 0) {
        if (ab.plain_ok[i] == 0) return false;
        set_bit(forced, static_cast<int>(i));
        must.push_back(static_cast<int>(i));
      }
      if (ab.plain_ok[i] != 0) set_bit(most, static_cast<int>(i));
    }
    if (!met(most, k, minc)) return false;
    const auto plains = [&](const Bits& set) {
      for (std::size_t i = 0; i < n; ++i) {
        if (bit(set, static_cast<int>(i)) && !bit(played, static_cast<int>(i))) {
          ab.witness.push_back({static_cast<int>(i), -1});
        }
      }
    };
    if (regime_ == Regime::FreeOrder) {
      plains(most);
      return true;
    }
    if (!met(forced, k, minc)) {
      // Add boards, the counting ones earliest first, until the obligations
      // are met: the one that met them is the last move.
      std::vector<int> order;
      for (std::size_t i = 0; i < n; ++i) {
        if (bit(most, static_cast<int>(i)) && !bit(forced, static_cast<int>(i)) &&
            counts_from_[i] <= k) {
          order.push_back(static_cast<int>(i));
        }
      }
      std::sort(order.begin(), order.end(), [&](int x, int y) {
        return axes_[static_cast<std::size_t>(x)].head < axes_[static_cast<std::size_t>(y)].head;
      });
      Bits set = forced;
      for (const int i : order) {
        set_bit(set, i);
        if (met(set, k, minc)) break;
      }
      plains(set);
      return true;
    }
    for (const int u : must) {
      Bits set = forced;
      clear_bit(set, static_cast<std::size_t>(u));
      if (!met(set, k, minc)) {
        plains(forced);
        return true;
      }
    }
    return false;
  }

  // Ordered: whether the next creation, by axis a (the earliest active new
  // board then next_minc), can be the action's last move, every plain move
  // made before it. Of the sets of plain moves after which the obligations
  // are still unmet, every one lies within one of the sets S_z below (z an
  // unplayed counting board of the mover's that keeps them unmet), and
  // meeting them afterwards only gets easier with more boards played.
  bool abstract_last(Abstract& ab, const Bits& before, int k, int minc, int a, int next_minc) const {
    const std::size_t n = axes_.size();
    Bits most((n + 63) / 64, 0);
    std::vector<int> must;
    for (std::size_t i = 0; i < n; ++i) {
      if (bit(before, static_cast<int>(i)) || static_cast<int>(i) == a) continue;
      if (ab.pass_ok[i] == 0) {
        if (ab.plain_ok[i] == 0) return false;
        must.push_back(static_cast<int>(i));
      }
      if (ab.plain_ok[i] != 0) set_bit(most, static_cast<int>(i));
    }
    for (std::size_t z = 0; z < n; ++z) {
      if (bit(before, static_cast<int>(z)) || counts_from_[z] > k) continue;
      if (static_cast<int>(z) != a && ab.pass_ok[z] == 0) continue;  // must be played
      Bits set = most;
      clear_bit(set, z);
      if (in_present_) {
        for (std::size_t i = 0; i < n; ++i) {
          if (counts_from_[i] <= k && axes_[i].head < axes_[z].head) clear_bit(set, i);
        }
      }
      if (std::any_of(must.begin(), must.end(), [&](int i) { return !bit(set, i); })) continue;
      Bits all = before;
      for (std::size_t w = 0; w < all.size(); ++w) all[w] |= set[w];
      if (met(all, k, minc)) continue;
      set_bit(all, a);
      if (!met(all, k + 1, next_minc)) continue;
      for (std::size_t i = 0; i < n; ++i) {
        if (bit(set, static_cast<int>(i))) ab.witness.push_back({static_cast<int>(i), -1});
      }
      return true;
    }
    return false;
  }

  // The point of the box the witness describes: its branches and early
  // moves as given, a plain move on each of its other played boards (a
  // physical one, else a jump between two of them), Pass elsewhere. Empty if
  // the box has no such point (the check's over-approximation).
  std::vector<int> from_witness(const Abstract& ab) const {
    const std::size_t k = axes_.size();
    std::vector<int> x(k, 0);
    std::vector<char> plain(k, 0);
    for (const Assign& s : ab.witness) {
      if (s.value >= 0) {
        x[static_cast<std::size_t>(s.axis)] = s.value;
      } else {
        plain[static_cast<std::size_t>(s.axis)] = 1;
      }
    }
    for (std::size_t i = 0; i < k; ++i) {
      if (plain[i] == 0 || x[i] != 0) continue;
      if (ab.physical[i] >= 0) {
        x[i] = ab.physical[i];
        continue;
      }
      bool done = false;
      for (const auto& [v, q] : ab.jumps[i]) {
        const auto j = static_cast<std::size_t>(q);
        if (plain[j] == 0 || x[j] != 0) continue;
        x[i] = v;
        x[j] = axes_[i].values[static_cast<std::size_t>(v)].partner_value;
        done = true;
        break;
      }
      if (!done) return {};
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

  // The share of `box` a region removes (the log of a product of fractions).
  double share(const Box& box, const Region& r) const {
    double s = 0;
    for (std::size_t i = 0; i < r.axes.size(); ++i) {
      const Axis& a = axes_[static_cast<std::size_t>(r.axes[i])];
      const auto in_box = static_cast<double>(count_in_box(box, a));
      const auto common = static_cast<double>(count_common(box, a, r.allowed[i]));
      s += std::log(common / in_box);
    }
    return s;
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
      clear_bit(others, static_cast<std::size_t>(v.partner_value));
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
    if (j6_cycle_region(x, region)) return true;
    // (3) Realisation and royal safety (C4).
    bool legal = false;
    bool stop = false;
    bool eliminated = false;
    bool realised = false;
    // Where no capture's region holds for every creation order (a capture
    // touching the timelines x creates, when it creates two or more), each
    // order's capture constrains the axes x plays with moves that create no
    // timeline, x's branches and passes kept (order_capture): outside the
    // general regime, and unless Q20's alternative makes a landing depend on
    // activity.
    bool by_order = regime_ != Regime::General && rules_.inactive_history_is_destination;
    Region order_region;
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
      if (!legal && !eliminated) {
        if (capture_region(x, box, final_state, seq, region)) {
          eliminated = true;  // every order fails alike: no need to try the others
          return false;
        }
        if (by_order) by_order = order_capture(x, box, final_state, order_region);
      }
      return true;
    };
    bool stopped = false;
    switch (regime_) {
      case Regime::Ordered: realise_ordered(x, visit, stopped); break;
      case Regime::FreeOrder: realise_free(x, visit, stopped); break;
      case Regime::General: realise_general(x, visit, stopped); break;
    }
    if (stop || stopped || budget_.stopped()) return false;
    if (legal) {
      single_point(x, region);
    } else if (realised && !eliminated) {
      // Every order of x was played and left a royal capturable.
      if (by_order) {
        region = order_region;
        keep_skeleton(x, region);
        ++run_.stats.order_regions;
      } else {
        single_point(x, region);
      }
    } else if (!realised) {
      ++run_.stats.unrealised;
      unrealised_region(x, box, region);
    }
    if (region.axes.size() == k &&
        std::all_of(region.allowed.begin(), region.allowed.end(), [](const Bits& b) {
          std::size_t n = 0;
          for (const std::uint64_t w : b) n += static_cast<std::size_t>(std::popcount(w));
          return n == 1;
        })) {
      ++run_.stats.single_points;
    }
    return true;
  }

  bool j6_cycle_region(const std::vector<int>& x, Region& region) const {
    const std::size_t k = axes_.size();
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
        region.restrict(static_cast<int>(cycle[c]), values_where(a, [&](const Value& v) {
                          return v.kind == Kind::Branch && v.partner == next;
                        }));
      }
      return true;
    }
    return false;
  }

  // The region of a point no order realises - the largest of those that apply:
  //  (e) Ordered and FreeOrder: its skeleton class (see (e) below and
  //      feasible());
  //  (a) every point differing from x only by other physical moves (castling
  //      apart) on the axes where x makes one: such a move plays its own board
  //      alone and no other move reads what it leaves there, so which one is
  //      made changes no order's validity;
  //  (T5) a mandatory board on Pass and no creation that could shift the
  //      present (unless the general regime; under Q5's alternative, an
  //      active board on Pass);
  //  else the single point.
  void unrealised_region(const std::vector<int>& x, const Box& box, Region& out) const {
    const std::size_t k = axes_.size();
    single_point(x, out);
    double best = share(box, out);
    Region r;
    const auto consider = [&]() {
      const double s = share(box, r);
      if (s > best) {
        best = s;
        out = r;
      }
    };
    // (e) Ordered and FreeOrder: whether a point is realisable depends only on
    // which axes are on Pass, which boards its plain moves play (they are made
    // after every creation, or just before the last, unless J6 needs them
    // earlier), and of each branch its J6 target and where its new board lies
    // against the boards of the mover's that can count for the present (the
    // present's, the reactivated ones'): a creation changes the present's
    // colour by those comparisons alone (T4, T8, T12), and its activity by its
    // place in the creation order (T10). So: Pass fixed; each branch axis on
    // the branches of the same target and class (and, under Q20's
    // alternative, destination timeline); each plain axis that no branch of x
    // needs played on any plain value - except where J6 needs it, which keeps
    // its kind of move.
    if (regime_ != Regime::General) {
      const std::vector<char> target = j6_targets(x);
      r.clear();
      for (std::size_t i = 0; i < k; ++i) {
        const Value& v = value_of(x, i);
        const Axis& a = axes_[i];
        const auto plain = [](const Value& w) { return plain_kind(w); };
        if (v.kind == Kind::Branch) {
          const Move& m = moves_[static_cast<std::size_t>(v.move)];
          const int cls = threshold_class(m.to_half_turn + 1);
          r.restrict(static_cast<int>(i), values_where(a, [&](const Value& w) {
                       if (w.kind != Kind::Branch || w.partner != v.partner) return false;
                       const Move& n = moves_[static_cast<std::size_t>(w.move)];
                       if (threshold_class(n.to_half_turn + 1) != cls) return false;
                       return rules_.inactive_history_is_destination || n.to_timeline == m.to_timeline;
                     }));
        } else if (plain(v) && target[i] == 0) {
          r.restrict(static_cast<int>(i), values_where(a, plain));
        } else if (v.kind == Kind::Physical) {
          r.restrict(static_cast<int>(i), values_where(a, [](const Value& w) {
                       return w.kind == Kind::Physical;
                     }));
        } else {
          r.restrict(static_cast<int>(i), only(a, x[i]));
        }
      }
      consider();
    }
    // (a)
    r.clear();
    bool any_physical = false;
    for (std::size_t i = 0; i < k; ++i) {
      const Value& v = value_of(x, i);
      if (v.kind == Kind::Physical && !v.castle) {
        any_physical = true;
        r.restrict(static_cast<int>(i), values_where(axes_[i], [](const Value& w) {
                     return w.kind == Kind::Physical && !w.castle;
                   }));
      } else {
        r.restrict(static_cast<int>(i), only(axes_[i], x[i]));
      }
    }
    if (any_physical) consider();
    std::vector<std::size_t> branch_axes;
    for (std::size_t i = 0; i < k; ++i) {
      if (value_of(x, i).kind == Kind::Branch) branch_axes.push_back(i);
    }
    // Which mandatory axes x leaves on Pass.
    std::vector<std::size_t> passed;
    for (std::size_t i = 0; i < k; ++i) {
      if (axes_[i].required && x[i] == 0) passed.push_back(i);
    }
    // (T5)
    if (regime_ != Regime::General && !passed.empty()) {
      if (!in_present_) {
        r.clear();
        r.restrict(static_cast<int>(passed.front()), only(axes_[passed.front()], 0));
        consider();
      } else if (std::none_of(branch_axes.begin(), branch_axes.end(), [&](std::size_t i) {
                   return value_of(x, i).rewind_capable;
                 })) {
        r.clear();
        r.restrict(static_cast<int>(passed.front()), only(axes_[passed.front()], 0));
        for (std::size_t i = 0; i < k; ++i) {
          const Axis& a = axes_[i];
          const bool any = std::any_of(a.values.begin(), a.values.end(),
                                       [](const Value& w) { return w.rewind_capable; });
          if (any) {
            r.restrict(static_cast<int>(i),
                       values_where(a, [](const Value& w) { return !w.rewind_capable; }));
          }
        }
        consider();
      }
    }
  }

  static bool plain_kind(const Value& w) {
    return w.kind == Kind::Physical || w.kind == Kind::Depart || w.kind == Kind::Arrive;
  }

  // The axes x's branches need played first (J6): their targets, and the
  // other board of a jump that plays one (it is played as early).
  std::vector<char> j6_targets(const std::vector<int>& x) const {
    std::vector<char> target(axes_.size(), 0);
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      const Value& v = value_of(x, i);
      if (v.kind != Kind::Branch || v.partner < 0) continue;
      target[static_cast<std::size_t>(v.partner)] = 1;
      const Value& t = value_of(x, static_cast<std::size_t>(v.partner));
      if ((t.kind == Kind::Depart || t.kind == Kind::Arrive) && t.partner >= 0) {
        target[static_cast<std::size_t>(t.partner)] = 1;
      }
    }
    return target;
  }

  // Restricts `r` to the points with x's skeleton and x's very branches: x's
  // branches and passes kept, every other axis on a move that creates no
  // timeline (a J6 target on its kind of move, as in unrealised_region's
  // (e)). Outside the general regime such a point is played in exactly the
  // orders x is, creating the same boards under the same numbers.
  void keep_skeleton(const std::vector<int>& x, Region& r) const {
    const std::vector<char> target = j6_targets(x);
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      const Value& v = value_of(x, i);
      const Axis& a = axes_[i];
      if (v.kind == Kind::Branch || v.kind == Kind::Pass) {
        r.restrict(static_cast<int>(i), only(a, x[i]));
      } else if (target[i] == 0) {
        r.restrict(static_cast<int>(i), values_where(a, [](const Value& w) { return plain_kind(w); }));
      } else if (v.kind == Kind::Physical) {
        r.restrict(static_cast<int>(i),
                   values_where(a, [](const Value& w) { return w.kind == Kind::Physical; }));
      } else {
        r.restrict(static_cast<int>(i), only(a, x[i]));
      }
    }
  }

  // A capture of a royal in `final_state` - x played in one creation order -
  // as constraints on the axes x plays with moves that create no timeline:
  // the values that leave the capture's pieces in place on their successor
  // boards (as capture_region's). With x's skeleton and branches kept
  // (keep_skeleton), every point that meets them is played in this order with
  // the same new boards, and has the capture. Of the captures, the one that
  // removes most of the box; intersected into `out`. False if none.
  bool order_capture(const std::vector<int>& x, const Box& box, const GameState& final_state,
                     Region& out) const {
    const Multiverse& mv = final_state.multiverse();
    double best = -std::numeric_limits<double>::infinity();
    bool found = false;
    int examined = 0;
    Region r;
    Region chosen;
    for_each_royal_capture(
        mv, rules_, opposite(mover_), final_state.inactive_timelines(), [&](const RoyalCapture& c) {
          if (++examined > 512) return true;
          r.clear();
          const std::vector<CapturePlace> places = capture_places(c);
          for (std::size_t p = 0; p < places.size(); ++p) {
            const CapturePlace& place = places[p];
            // A created board: fixed with x's branches and the order.
            if (place.timeline < min_timeline_ || place.timeline > max_timeline_) continue;
            const int ai = axis_of(place.timeline);
            if (ai < 0) continue;
            const Axis& a = axes_[static_cast<std::size_t>(ai)];
            if (place.half_turn != a.head + 1) continue;
            const int xi = x[static_cast<std::size_t>(ai)];
            if (!plain_kind(a.values[static_cast<std::size_t>(xi)])) continue;  // kept as it is
            const Board* board = mv.board_at(place.timeline, place.half_turn);
            if (board == nullptr || board->at(place.square) != content(a, xi, place.square)) {
              throw std::logic_error("hypercuboid: a successor board differs from its value");
            }
            const bool crossed = p != 0 && p + 1 != places.size();
            const Piece seen = board->at(place.square).as_moved();
            Bits allowed(a.words, 0);
            for (std::size_t v = 0; v < a.values.size(); ++v) {
              if (!plain_kind(a.values[v])) continue;
              const Piece here = content(a, static_cast<int>(v), place.square).as_moved();
              if (crossed ? (here.empty() || here == seen) : here == seen) set_bit(allowed, static_cast<int>(v));
            }
            r.restrict(ai, allowed);
          }
          const double s = share(box, r);
          if (s > best) {
            best = s;
            chosen = r;
            found = true;
          }
          return false;
        });
    if (!found) return false;
    for (std::size_t i = 0; i < chosen.axes.size(); ++i) out.restrict(chosen.axes[i], chosen.allowed[i]);
    return true;
  }

  // -- realisation --------------------------------------------------------------------------------

  using Visit = std::function<bool(const GameState&, const std::vector<Move>&)>;

  // Plays move `m` on `s` if the turn rules allow a move now and it is
  // pseudo-legal there, appending it to `seq`. False if not; `stopped` if the
  // budget ran out.
  bool play(GameState& s, const Move& m, std::vector<Move>& seq, bool& stopped) {
    if (!s.may_move() || !is_pseudo_legal(s, m, scratch_)) return false;
    if (!budget_.spend()) {
      stopped = true;
      return false;
    }
    s.apply(m);
    seq.push_back(m);
    return true;
  }

  // A point's moves: those that create no timeline (in axis order, each with
  // the axes it plays) and its branches.
  struct Plan {
    std::vector<int> plain;                   // indices into moves_
    std::vector<std::vector<int>> plain_axes;  // the axes each plays
    std::vector<std::size_t> branch_axes;
  };

  Plan plan_of(const std::vector<int>& x) const {
    Plan p;
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      const Value& v = value_of(x, i);
      if (v.kind == Kind::Physical) {
        p.plain.push_back(v.move);
        p.plain_axes.push_back({static_cast<int>(i)});
      } else if (v.kind == Kind::Depart) {
        p.plain.push_back(v.move);
        p.plain_axes.push_back({static_cast<int>(i), v.partner});
      } else if (v.kind == Kind::Branch) {
        p.branch_axes.push_back(i);
      }
    }
    return p;
  }

  // Ordered (exact under Regime::Ordered). With the creations' order fixed,
  // whether the obligations are met after some moves depends only on which
  // boards were played and how many creations were made; with that number
  // fixed it can only turn from unmet to met as boards are played. So a move
  // that creates no timeline is best made as late as J6 allows - just before
  // the first branch that needs its board played, else after the last branch
  // (shape B: one of them last), or, when the last branch must be the last
  // move, just before it (shape A). A depth-first search over the creation
  // orders, stopping wherever a move ends the action with moves still to make
  // (Q4), visits each realisable order once.
  void realise_ordered(const std::vector<int>& x, const Visit& visit, bool& stopped) {
    const Plan p = plan_of(x);
    std::vector<char> plain_done(p.plain.size(), 0);
    std::vector<char> branch_done(p.branch_axes.size(), 0);
    std::vector<Move> seq;
    bool go_on = true;
    ordered_step(x, p, s0_, plain_done, branch_done, p.branch_axes.size(), seq, visit, stopped,
                 go_on);
  }

  // The plain move that plays axis `a`, or -1.
  static int plain_playing(const Plan& p, int a) {
    for (std::size_t q = 0; q < p.plain.size(); ++q) {
      if (std::find(p.plain_axes[q].begin(), p.plain_axes[q].end(), a) != p.plain_axes[q].end()) {
        return static_cast<int>(q);
      }
    }
    return -1;
  }

  // Plays the plain moves not yet made, ascending - all of them, or all but
  // `last` (then `last` after them) - and checks the end: obligations met. On
  // success, visits. Returns whether this shape realised the order.
  bool finish(const Plan& p, const GameState& s, const std::vector<char>& plain_done,
              std::vector<Move>& seq, int last, const Visit& visit, bool& stopped, bool& go_on) {
    GameState t = s;
    const std::size_t mark = seq.size();
    bool ok = true;
    for (std::size_t q = 0; q < p.plain.size() && ok; ++q) {
      if (plain_done[q] != 0 || static_cast<int>(q) == last) continue;
      ok = play(t, moves_[static_cast<std::size_t>(p.plain[q])], seq, stopped);
    }
    if (ok && last >= 0) ok = play(t, moves_[static_cast<std::size_t>(p.plain[static_cast<std::size_t>(last)])], seq, stopped);
    if (ok && !stopped && t.obligations_met()) {
      if (!visit(t, seq)) go_on = false;
      seq.resize(mark);
      return true;
    }
    seq.resize(mark);
    return false;
  }

  // Shape B's end: the plain moves left after the last creation, one of them
  // last. The candidates for the last move: those that play a board the
  // mover must still play, first.
  bool finish_with_last(const Plan& p, const GameState& s, const std::vector<char>& plain_done,
                        std::vector<Move>& seq, const Visit& visit, bool& stopped, bool& go_on) {
    std::vector<int> left;
    for (std::size_t q = 0; q < p.plain.size(); ++q) {
      if (plain_done[q] == 0) left.push_back(static_cast<int>(q));
    }
    if (left.empty()) return false;
    std::stable_sort(left.begin(), left.end(), [&](int a, int b) {
      const auto must = [&](int q) {
        for (const int ax : p.plain_axes[static_cast<std::size_t>(q)]) {
          if (s.mandatory(axes_[static_cast<std::size_t>(ax)].timeline)) return 0;
        }
        return 1;
      };
      return must(a) < must(b);
    });
    for (const int last : left) {
      if (finish(p, s, plain_done, seq, last, visit, stopped, go_on)) return true;
      if (stopped || !go_on) return false;
    }
    return false;
  }

  void ordered_step(const std::vector<int>& x, const Plan& p, const GameState& s,
                    std::vector<char>& plain_done, std::vector<char>& branch_done,
                    std::size_t left, std::vector<Move>& seq, const Visit& visit, bool& stopped,
                    bool& go_on) {
    if (left == 0) {  // no branch at all: shape B
      finish_with_last(p, s, plain_done, seq, visit, stopped, go_on);
      return;
    }
    for (std::size_t b = 0; b < p.branch_axes.size() && go_on && !stopped; ++b) {
      if (branch_done[b] != 0) continue;
      const Value& bv = value_of(x, p.branch_axes[b]);
      // J6: the board it lands on played first - by a branch made before it,
      // or by a plain move made just before it.
      int needed = -1;
      if (bv.partner >= 0) {
        const Value& tv = value_of(x, static_cast<std::size_t>(bv.partner));
        if (tv.kind == Kind::Branch) {
          const auto at = std::find(p.branch_axes.begin(), p.branch_axes.end(),
                                    static_cast<std::size_t>(bv.partner));
          if (branch_done[static_cast<std::size_t>(at - p.branch_axes.begin())] == 0) continue;
        } else {
          needed = plain_playing(p, bv.partner);
        }
      }
      GameState t = s;
      const std::size_t mark = seq.size();
      bool ok = true;
      bool played_needed = false;
      if (needed >= 0 && plain_done[static_cast<std::size_t>(needed)] == 0) {
        ok = play(t, moves_[static_cast<std::size_t>(p.plain[static_cast<std::size_t>(needed)])],
                  seq, stopped);
        played_needed = ok;
        if (ok) plain_done[static_cast<std::size_t>(needed)] = 1;
      }
      const Move& branch = moves_[static_cast<std::size_t>(bv.move)];
      if (ok && left == 1) {
        // Shape A: every plain move left, then this branch last.
        GameState a = t;
        const std::size_t mark_a = seq.size();
        bool ok_a = true;
        for (std::size_t q = 0; q < p.plain.size() && ok_a; ++q) {
          if (plain_done[q] != 0) continue;
          ok_a = play(a, moves_[static_cast<std::size_t>(p.plain[q])], seq, stopped);
        }
        if (ok_a) ok_a = play(a, branch, seq, stopped);
        bool done = false;
        if (ok_a && !stopped && a.obligations_met()) {
          if (!visit(a, seq)) go_on = false;
          done = true;
        }
        seq.resize(mark_a);
        // Shape B: this branch, then the plain moves left, one of them last.
        if (!done && go_on && !stopped && play(t, branch, seq, stopped)) {
          branch_done[b] = 1;
          finish_with_last(p, t, plain_done, seq, visit, stopped, go_on);
          branch_done[b] = 0;
        }
      } else if (ok && play(t, branch, seq, stopped) && t.may_move()) {
        branch_done[b] = 1;
        ordered_step(x, p, t, plain_done, branch_done, left - 1, seq, visit, stopped, go_on);
        branch_done[b] = 0;
      }
      if (played_needed) plain_done[static_cast<std::size_t>(needed)] = 0;
      seq.resize(mark);
    }
  }

  // FreeOrder (exact under Regime::FreeOrder): moves may follow the shift, so
  // any order the J6 needs allow is valid; the plain moves first, then the
  // branches in each creation order.
  void realise_free(const std::vector<int>& x, const Visit& visit, bool& stopped) {
    const Plan p = plan_of(x);
    GameState s = s0_;
    std::vector<Move> seq;
    for (const int q : p.plain) {
      if (!play(s, moves_[static_cast<std::size_t>(q)], seq, stopped)) return;
    }
    std::vector<char> done(p.branch_axes.size(), 0);
    bool go_on = true;
    free_step(x, p, s, done, p.branch_axes.size(), seq, visit, stopped, go_on);
  }

  void free_step(const std::vector<int>& x, const Plan& p, const GameState& s,
                 std::vector<char>& done, std::size_t left, std::vector<Move>& seq,
                 const Visit& visit, bool& stopped, bool& go_on) {
    if (left == 0) {
      if (s.obligations_met() && !visit(s, seq)) go_on = false;
      return;
    }
    for (std::size_t b = 0; b < p.branch_axes.size() && go_on && !stopped; ++b) {
      if (done[b] != 0) continue;
      GameState t = s;
      const std::size_t mark = seq.size();
      if (play(t, moves_[static_cast<std::size_t>(value_of(x, p.branch_axes[b]).move)], seq,
               stopped)) {
        done[b] = 1;
        free_step(x, p, t, done, left - 1, seq, visit, stopped, go_on);
        done[b] = 0;
      }
      seq.resize(mark);
    }
  }

  // General (exact under any rules): every order of the point's moves,
  // depth-first, with a table of the states reached (each with the creations
  // made, in order). Between two creations the plain moves other than castling
  // commute and the obligations only turn from unmet to met as boards are
  // played, so they are made in ascending order - but for the action's very
  // last move, which may be any. A castling is exempt: under Q21's alternative
  // it is legal only while its board is mandatory, and under Q3 with Q4's
  // other reading a plain move can rewind the present off that board, so a
  // castling may have to precede moves of lower index (found by review,
  // 2026-10-02). Each distinct final (state, creation order) with the
  // obligations met is visited once.
  void realise_general(const std::vector<int>& x, const Visit& visit, bool& stopped) {
    std::vector<Move> moves;
    std::vector<char> plain;
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      const Value& v = value_of(x, i);
      if (v.kind == Kind::Physical || v.kind == Kind::Depart || v.kind == Kind::Branch) {
        moves.push_back(moves_[static_cast<std::size_t>(v.move)]);
        plain.push_back(v.kind == Kind::Branch ? 0 : 1);
      }
    }
    TaggedPositionSet seen;
    TaggedPositionSet finals;
    std::vector<char> used(moves.size(), 0);
    std::vector<Move> path;
    std::string tag;
    bool go_on = true;
    std::function<void(const GameState&, std::size_t, int)> dfs = [&](const GameState& s,
                                                                     std::size_t made,
                                                                     int last_plain) {
      if (made == moves.size()) {
        if (s.obligations_met() && finals.insert(s, tag).second && !visit(s, path)) go_on = false;
        return;
      }
      if (!s.may_move()) return;
      for (std::size_t m = 0; m < moves.size() && go_on && !stopped; ++m) {
        if (used[m] != 0) continue;
        const bool final_move = made + 1 == moves.size();
        const bool castle = moves[m].castle();
        if (plain[m] != 0 && !castle && static_cast<int>(m) < last_plain && !final_move) continue;
        if (!is_pseudo_legal(s, moves[m], scratch_)) continue;
        if (!budget_.spend()) {
          stopped = true;
          return;
        }
        GameState child = s;
        child.apply(moves[m]);
        const std::size_t tag_size = tag.size();
        if (plain[m] == 0) {
          tag += to_string(moves[m], false);
          tag += '|';
        }
        if (seen.insert(child, tag).second) {
          if (!budget_.hold(run_.held() + seen.size() + finals.size(), run_.held_bytes())) {
            stopped = true;
            return;
          }
          used[m] = 1;
          path.push_back(moves[m]);
          // A castling leaves the ascending order where it was.
          dfs(child, made + 1, plain[m] == 0 ? -1 : (castle ? last_plain : static_cast<int>(m)));
          path.pop_back();
          used[m] = 0;
        }
        tag.resize(tag_size);
      }
    };
    dfs(s0_, 0, -1);
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
              const bool any = std::any_of(a.values.begin(), a.values.end(),
                                           [](const Value& w) { return w.kind == Kind::Branch; });
              if (any) {
                r.restrict(static_cast<int>(j), values_where(a, [](const Value& w) {
                             return w.kind != Kind::Branch;
                           }));
              }
            }
          }
          ++usable;
          const double s = share(box, r);
          if (s > best) {
            best = s;
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
  std::vector<int> heads_;  // the axes' heads, ascending, distinct
  std::vector<int> counts_from_;  // per axis: counts for the present from this creation on
  std::vector<int> o_base_;       // per number of creations: see build()
  int min_timeline_ = 0;
  int max_timeline_ = 0;
  int present_ = 0;
  Regime regime_ = Regime::General;
  bool in_present_ = true;
  std::vector<const Timeline*> reactivated_;
  bool reactivation_threat_ = false;
  int max_last_active_ = 0;
  std::size_t words_ = 0;
  MoveList scratch_;
  Abstract abstract_;
};

}  // namespace

void run_hypercuboid(SearchRun& run) { Hyper(run).search(); }

}  // namespace t5d::detail
