// The legal moves, in tiers (see t5d/legal.hpp for the tiers and why each is
// sound). Rule IDs refer to the rulebook, docs/rules/README.md.

#include "t5d/legal.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "search_internal.hpp"
#include "t5d/board.hpp"
#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"

namespace t5d {

namespace {

// A move that creates no timeline and moves no royal piece (nor makes one):
// tier 1's moves.
bool quiet(const Move& m) noexcept {
  if (m.branching() || m.castle() || m.piece.royal()) return false;
  return !(m.promotes() && is_royal(m.promotion));
}

// The boards a move plays: its source, and a non-branching jump's destination.
std::pair<int, std::optional<int>> boards_of(const Move& m) noexcept {
  if (m.physical() || m.branching()) return {m.from_timeline, std::nullopt};
  return {m.from_timeline, m.to_timeline};
}

// Tier 0's lemma: a royal capture standing in `s` persists to the end of its
// action - unless, under Q18's alternative, it crosses a void place that a
// later move could fill with a board.
bool standing_capture(const GameState& s) {
  const Colour opponent = opposite(s.to_move());
  const Multiverse& mv = s.multiverse();
  if (!s.rules().sliders_cross_void) {
    return royal_capture_exists(mv, s.rules(), opponent, s.inactive_timelines());
  }
  return for_each_royal_capture(mv, s.rules(), opponent, s.inactive_timelines(),
                                [&](const RoyalCapture& c) {
                                  for (const CapturePlace& p : capture_places(c)) {
                                    if (mv.board_at(p.timeline, p.half_turn) == nullptr) return false;
                                  }
                                  return true;
                                });
}

// A search for a legal finish of the action at `s`: Legal (with the finish),
// Illegal (exhaustive, none) or Unknown (the budget or the memory ceiling).
struct Finish {
  MoveLegality status = MoveLegality::Unknown;
  std::vector<Move> moves;
  std::uint64_t nodes = 0;
};

Finish search_finish(const GameState& s, std::uint64_t nodes, std::size_t max_bytes,
                     Detector detector) {
  Finish f;
  if (s.obligations_met()) {
    if (royals_safe(s)) {
      f.status = MoveLegality::Legal;
      return f;
    }
    if (!s.may_move()) {
      f.status = MoveLegality::Illegal;
      return f;
    }
  }
  SearchBudget b = SearchBudget::nodes(nodes);
  b.max_bytes = max_bytes;
  detail::SearchRun r(s, b, 1, detail::Unit::Successors, false);
  if (detector == Detector::Naive) {
    detail::run_naive(r);
  } else {
    detail::run_hypercuboid(r);
  }
  f.nodes = r.stats.nodes;
  if (r.found() > 0) {
    f.status = MoveLegality::Legal;
    f.moves = r.actions.front();
  } else if (r.stats.exhaustive) {
    f.status = MoveLegality::Illegal;
  }
  return f;
}

constexpr int kNever = std::numeric_limits<int>::max();

// The present (T4, T5) after ONE creation and moves within boards - tier 1's
// branches - under the rules the regimes of the hypercuboid call Ordered and
// FreeOrder (activity changed by creations alone; Rewinds): which playable
// boards of the mover's count for the present, from the start or from the
// first creation on (T11: the opponent's earliest-created inactive timeline
// is reactivated); the earliest counting head of the opponent's colour on the
// other timelines; and whether the first creation is active (T10, Q1).
class PresentModel {
 public:
  PresentModel(const GameState& s, const std::vector<int>& axes) {
    const Multiverse& mv = s.multiverse();
    const RuleOptions& rules = s.rules();
    const Colour mover = s.to_move();
    const Colour opponent = opposite(mover);
    in_present_ = rules.mandatory == MandatoryBoards::InPresent;
    const Timeline* r1 = nullptr;
    for (const int p : s.inactive_timelines()) {
      const Timeline* t = mv.timeline(p);
      if (t->creator() == opponent && (r1 == nullptr || t->creation_order() < r1->creation_order())) {
        r1 = t;
      }
    }
    for (const int p : axes) {
      heads_.push_back(mv.timeline(p)->last_half_turn());
      counts_.push_back(s.active(p) || (r1 != nullptr && r1->position() == p) ? 1 : 0);
    }
    for (const TimelineRef& t : mv.timelines()) {
      if (to_move_at(t->last_half_turn()) == mover || !s.active(t->position())) continue;
      o_ = std::min(o_, t->last_half_turn());
    }
    if (r1 != nullptr && to_move_at(r1->last_half_turn()) != mover) {
      o_ = std::min(o_, r1->last_half_turn());
    }
    if (rules.activation == ActivationRule::Count) {
      const auto credit = [&](Colour c) {
        return rules.initial_credit == InitialTimelineCredit::MinusZeroToBlack && mv.even() &&
                       c == Colour::Black
                   ? 1
                   : 0;
      };
      first_active_ = mv.created_by(opponent) + credit(opponent) >= mv.created_by(mover) + credit(mover);
    } else {
      first_active_ = !(s.last_creator() == mover);
    }
  }

  // Whether the obligations are met after the creation (its new board at
  // half-turn c) with the axes marked in `played` played.
  bool met(const std::vector<char>& played, int c) const {
    int min_m = kNever;
    int min_o = first_active_ ? std::min(o_, c) : o_;
    for (std::size_t i = 0; i < heads_.size(); ++i) {
      if (counts_[i] == 0) continue;
      if (played[i] != 0) {
        min_o = std::min(min_o, heads_[i] + 1);
      } else {
        min_m = std::min(min_m, heads_[i]);
      }
    }
    return in_present_ ? min_o < min_m : min_m == kNever;
  }

 private:
  std::vector<int> heads_;
  std::vector<char> counts_;
  int o_ = kNever;
  bool first_active_ = false;
  bool in_present_ = true;
};

class Classifier {
 public:
  Classifier(const GameState& s, std::span<const Move> moves, const LegalityOptions& budget,
             LegalityStats& stats)
      : s_(s),
        rules_(s.rules()),
        mover_(s.to_move()),
        moves_(moves),
        budget_(budget),
        stats_(stats),
        min_timeline_(s.multiverse().min_timeline()) {}

  void run(std::vector<MoveLegality>& out) {
    const std::size_t n = moves_.size();
    out.assign(n, MoveLegality::Unknown);
    stats_.moves += n;
    if (n == 0) return;
    // Tier 0.
    if (standing_capture(s_)) {
      std::fill(out.begin(), out.end(), MoveLegality::Illegal);
      stats_.standing_at_start += n;
      return;
    }
    prepare();
    std::vector<char> done(n, 0);
    // Where the stripped multiverse keeps a capture: one search from the state.
    // No legal finish at all - nothing is legal; one found (a0_) - its moves
    // that can lead it are legal, and with no creation in it, each board is
    // stripped on its own around it.
    prepare_passed();
    if (!bulk_ && tier1_possible_) {
      // First a finish made greedily - a move within the board on each board
      // to play - then, if none is found that way, a search.
      if (std::optional<std::vector<Move>> g = greedy_finish()) {
        a0_ = std::move(*g);
        ++stats_.greedy;
      } else {
        const Finish f = search(s_);
        searched_root_ = true;
        if (f.status == MoveLegality::Illegal) {
          for (std::size_t i = 0; i < n; ++i) {
            out[i] = MoveLegality::Illegal;
            ++stats_.no_finish_at_all;
          }
          return;
        }
        if (f.status == MoveLegality::Legal) a0_ = f.moves;
      }
      if (a0_) {
        learn(*a0_);
        prepare_around_a0();
      }
    }
    // Tier 1.
    for (std::size_t i = 0; i < n; ++i) {
      if (bulk_certifies(moves_[i]) || around_a0_certifies(moves_[i])) {
        out[i] = MoveLegality::Legal;
        done[i] = 1;
        ++stats_.bulk_quiet;
      }
    }
    if (branches_allowed_) {
      for (std::size_t i = 0; i < n; ++i) {
        if (done[i] == 0 && bulk_certifies_branch(moves_[i])) {
          out[i] = MoveLegality::Legal;
          done[i] = 1;
          ++stats_.bulk_branches;
        }
      }
    }
    // Refutations by a capture standing once a board is played as it is.
    if (std::find(done.begin(), done.end(), 0) != done.end()) {
      for (std::size_t i = 0; i < n; ++i) {
        if (done[i] == 0 && refuted_by_passed(moves_[i])) {
          out[i] = MoveLegality::Illegal;
          done[i] = 1;
          ++stats_.refuted_in_bulk;
        }
      }
    }
    // Tier 2.
    for (std::size_t i = 0; i < n; ++i) {
      if (done[i] != 0) continue;
      if (known_legal(moves_[i])) {
        out[i] = MoveLegality::Legal;
        done[i] = 1;
        ++stats_.learned;
        continue;
      }
      if (!take_check()) break;  // the budget: the rest left to the search tier
      GameState after = s_;
      after.apply(moves_[i]);
      const MoveLegality t = tier2(moves_[i], after);
      if (t == MoveLegality::Legal) {
        out[i] = t;
        done[i] = 1;
        ++stats_.finished;
      } else if (t == MoveLegality::Illegal) {
        out[i] = t;
        done[i] = 1;
        ++stats_.standing_after;
      }
    }
    // Tier 3.
    const bool any_legal = std::find(out.begin(), out.end(), MoveLegality::Legal) != out.end();
    if (!any_legal && !searched_root_ && std::find(done.begin(), done.end(), 0) != done.end()) {
      // One search from the state: an action with no legal finish at all has
      // no legal move.
      const Finish f = search(s_);
      searched_root_ = true;
      if (f.status == MoveLegality::Illegal) {
        for (std::size_t i = 0; i < n; ++i) {
          if (done[i] != 0) continue;
          out[i] = MoveLegality::Illegal;
          done[i] = 1;
          ++stats_.no_finish_at_all;
        }
      } else if (f.status == MoveLegality::Legal) {
        learn(f.moves);
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (done[i] != 0) continue;
      if (known_legal(moves_[i])) {
        out[i] = MoveLegality::Legal;
        ++stats_.learned;
        continue;
      }
      if (spent_ >= budget_.search_nodes) {
        ++stats_.unknown;
        continue;
      }
      GameState after = s_;
      after.apply(moves_[i]);
      const Finish f = search(after);
      out[i] = f.status;
      if (f.status == MoveLegality::Legal) {
        ++stats_.searched_legal;
        std::vector<Move> finish{moves_[i]};
        finish.insert(finish.end(), f.moves.begin(), f.moves.end());
        learn(finish);
      } else if (f.status == MoveLegality::Illegal) {
        ++stats_.searched_illegal;
      } else {
        ++stats_.unknown;
      }
    }
  }

 private:
  // -- set-up: the playable boards, the quiet moves, the stripped multiverse --

  std::size_t index(int timeline) const { return static_cast<std::size_t>(timeline - min_timeline_); }

  void prepare() {
    const Multiverse& mv = s_.multiverse();
    const std::size_t width = static_cast<std::size_t>(mv.max_timeline() - min_timeline_ + 1);
    in_x_.assign(width, 0);
    kept_.assign(width, 0);
    quiet_moves_.assign(width, 0);
    filler_.assign(width, -1);
    for (const int p : s_.playable_timelines()) boards_.push_back(p);
    mandatory_ = s_.mandatory_timelines();
    // Tier 1 needs: no slide across void (Q18), activity changed by creations
    // alone (no Q3 activation by play) - the stripped superset and the quiet
    // finishes rest on both - and the action still open (Q4: a move may be made).
    if (!s_.may_move() || rules_.sliders_cross_void || rules_.play_activates) {
      fillers();
      return;
    }
    tier1_possible_ = true;
    for (const int p : boards_) in_x_[index(p)] = 1;
    // Strip, scan, and refine until the stripped multiverse holds no capture
    // (or no refinement can remove the one it holds).
    for (std::size_t round = 0; round < 256; ++round) {
      Multiverse stripped = strip();
      ++stats_.scans;
      std::optional<RoyalCapture> capture;
      for_each_royal_capture(stripped, rules_, opposite(mover_), s_.inactive_timelines(),
                             [&](const RoyalCapture& c) {
                               capture = c;
                               return true;
                             });
      if (!capture) {
        bulk_ = true;
        stripped_ = std::move(stripped);
        break;
      }
      if (!refine(*capture)) break;
    }
    stats_.bulk = bulk_;
    fillers();
    // Branches: one more board, on a new timeline; under Q20's alternative a
    // creation can change whether a capture may land (activity), which the
    // stripped multiverse does not follow.
    branches_allowed_ = bulk_ && rules_.inactive_history_is_destination &&
                        rules_.reactivation == Reactivation::Rewinds;
    if (branches_allowed_) {
      model_.emplace(s_, boards_);
      axis_.assign(in_x_.size(), 0);
      for (std::size_t i = 0; i < boards_.size(); ++i) axis_[index(boards_[i])] = i;
      played_.assign(boards_.size(), 0);
    }
    // The boards an action must still play, and whether each has a quiet move
    // within the tier: one that is not, in `blocked_`.
    for (const int r : mandatory_) {
      if (in_x_[index(r)] == 0 || quiet_moves_[index(r)] == 0) blocked_.push_back(r);
    }
  }

  // The stripped multiverse of the boards in X (see the header comment).
  Multiverse strip() const {
    Multiverse mv = s_.multiverse();
    for (const int p : boards_) {
      if (in_x_[index(p)] == 0) continue;
      const Board& head = *mv.head(p);
      BoardBuilder b(head);
      b.set_en_passant(kNoSquare);
      const std::uint64_t kept = kept_[index(p)];
      for (int sq = 0; sq < kSquareCount; ++sq) {
        const Piece q = head.at(sq);
        if (q.empty() || q.colour() != mover_ || q.royal()) continue;
        if (((kept >> sq) & 1u) != 0) continue;
        b.remove(sq);
      }
      mv.append_board(p, b.publish());
    }
    return mv;
  }

  // Removes the capture from the next stripped multiverse: keeps a blocker
  // of the mover's that its line crosses, else drops an optional board it
  // rests on. False if neither can.
  bool refine(const RoyalCapture& c) {
    const std::vector<CapturePlace> places = capture_places(c);
    const Multiverse& mv = s_.multiverse();
    const auto stripped_board = [&](const CapturePlace& place) {
      if (place.timeline < min_timeline_ || place.timeline > mv.max_timeline()) return false;
      if (in_x_[index(place.timeline)] == 0) return false;
      return place.half_turn == mv.timeline(place.timeline)->last_half_turn() + 1;
    };
    for (std::size_t i = 1; i + 1 < places.size(); ++i) {
      const CapturePlace& place = places[i];
      if (!stripped_board(place)) continue;
      const Piece q = mv.head(place.timeline)->at(place.square);
      if (!q.empty() && q.colour() == mover_ && !q.royal()) {
        kept_[index(place.timeline)] |= std::uint64_t{1} << place.square;
        ++stats_.kept_blockers;
        return true;
      }
    }
    for (const CapturePlace& place : places) {
      if (!stripped_board(place)) continue;
      if (std::find(mandatory_.begin(), mandatory_.end(), place.timeline) != mandatory_.end()) continue;
      in_x_[index(place.timeline)] = 0;
      ++stats_.dropped_boards;
      return true;
    }
    return false;
  }

  // Per board: the quiet moves within the board whose piece tier 1 did not
  // keep (counted, for tier 1's finishes), and one move within the board to
  // finish with in tier 2 (a quiet one, of a piece not kept if possible).
  void fillers() {
    for (std::size_t i = 0; i < moves_.size(); ++i) {
      const Move& m = moves_[i];
      if (!m.physical() || !quiet(m)) continue;
      const std::size_t t = index(m.from_timeline);
      const bool kept = ((kept_[t] >> m.from_square) & 1u) != 0;
      if (!kept) ++quiet_moves_[t];
      if (filler_[t] < 0 || (!kept && ((kept_[t] >> moves_[static_cast<std::size_t>(filler_[t])].from_square) & 1u) != 0)) {
        filler_[t] = static_cast<int>(i);
      }
    }
  }

  // -- tier 1 ---------------------------------------------------------------------

  bool in_tier(const Move& m) const {
    const std::size_t t = index(m.from_timeline);
    return in_x_[t] != 0 && ((kept_[t] >> m.from_square) & 1u) == 0;
  }

  // A quiet move: safe (the stripped multiverse), and the boards still to play
  // can be played by quiet moves within the board. Under the creation-free
  // rules of a quiet action the present moves only when every board in it has
  // been played (T5): those are the mandatory boards, whatever is played
  // besides, and playing them all meets the obligations.
  bool bulk_certifies(const Move& m) const {
    if (!bulk_ || !quiet(m) || !in_tier(m)) return false;
    const auto [a, b] = boards_of(m);
    if (b && in_x_[index(*b)] == 0) return false;
    for (const int r : blocked_) {
      if (r != a && (!b || r != *b)) return false;
    }
    return true;
  }

  // A branching move of a piece that is not royal: its new board (a copy of
  // its destination with the piece arrived, on the outermost new timeline)
  // added to the stripped multiverse - one scan per destination board - and
  // its effect on the present from the state machine: it ends the action, or
  // the boards then to play can be played by quiet moves within the board.
  bool bulk_certifies_branch(const Move& m) {
    if (!m.branching() || m.piece.royal() || !in_tier(m)) return false;
    if (m.promotes() && is_royal(m.promotion)) return false;
    if (!destination_safe(m)) return false;
    // After the branch the obligations are met, or playing every board of X
    // with a quiet move within the board meets them (they turn from unmet to
    // met as boards are played, with no further creation: the board that meets
    // them is the last move).
    const std::size_t a = axis_[index(m.from_timeline)];
    std::vector<char>& played = played_;
    std::fill(played.begin(), played.end(), 0);
    played[a] = 1;
    const int c = m.to_half_turn + 1;
    if (model_->met(played, c)) return true;
    for (std::size_t i = 0; i < boards_.size(); ++i) {
      const std::size_t t = index(boards_[i]);
      if (in_x_[t] != 0 && quiet_moves_[t] > 0) played[i] = 1;
    }
    return model_->met(played, c);
  }

  bool destination_safe(const Move& m) {
    const std::pair<int, int> key{m.to_timeline, m.to_half_turn};
    const auto it = destinations_.find(key);
    if (it != destinations_.end()) return it->second;
    if (!take_check()) return false;  // the budget: left to the later tiers
    Multiverse mv = *stripped_;
    const Board& d = *s_.multiverse().board_at(m.to_timeline, m.to_half_turn);
    BoardBuilder b(d);
    b.set_en_passant(kNoSquare);
    for (int sq = 0; sq < kSquareCount; ++sq) {
      const Piece q = d.at(sq);
      if (!q.empty() && q.colour() == mover_ && !q.royal()) b.remove(sq);
    }
    NewTimeline nt;
    nt.position = mover_ == Colour::White ? mv.max_timeline() + 1 : mv.min_timeline() - 1;
    nt.first_half_turn = m.to_half_turn + 1;
    nt.boards = {b.publish()};
    nt.creator = mover_;
    nt.creation_order = mv.created_by(mover_) + 1;
    mv.add_timeline(std::move(nt));
    ++stats_.scans;
    const bool safe = !royal_capture_exists(mv, rules_, opposite(mover_), s_.inactive_timelines());
    destinations_.emplace(key, safe);
    return safe;
  }

  // A legal finish a0_ with no creation in it, ending in f0_: each board b
  // stripped on its own in f0_ - its successor there (or a new successor, if
  // a0_ does not play b) replaced by b with the mover's pieces other than its
  // royals and kept blockers removed. f0_ holds no capture, so a capture
  // there rests on b's board: kept blockers of b's remove what they can. A
  // quiet move within b (of a piece not kept), in place of a0_'s move there,
  // then ends - after a0_'s other moves - in a position whose captures are
  // among those of the stripped one: none.
  void prepare_around_a0() {
    GameState f = s_;
    for (const Move& m : *a0_) f.apply(m);
    f0_ = f.multiverse();
    // How a0_ plays each board: 1 within the board, 2 by a jump or a branch
    // (those boards are left out: another move there would change what is
    // created, or leave a jump's other board unplayed).
    a0_board_.assign(in_x_.size(), 0);
    for (const Move& m : *a0_) {
      const auto [a, b] = boards_of(m);
      a0_board_[index(a)] = (b || m.branching()) ? 2 : 1;
      if (b) a0_board_[index(*b)] = 2;
    }
    around_.assign(in_x_.size(), 0);
    around_kept_.assign(in_x_.size(), 0);
    for (const int p : boards_) {
      const std::size_t t = index(p);
      if (a0_board_[t] == 2) continue;
      // Whether a quiet move within b, then a0_'s other moves, is a finish:
      // that depends on the boards played and the creations (the present,
      // T4-T12), not on which quiet move it is - replayed once, with one.
      if (!finish_around_a0(p)) continue;
      for (int round = 0; round < 32; ++round) {
        Multiverse mv = *f0_;
        const Board& head = *s_.multiverse().head(p);
        BoardBuilder b(head);
        b.set_en_passant(kNoSquare);
        for (int sq = 0; sq < kSquareCount; ++sq) {
          const Piece q = head.at(sq);
          if (q.empty() || q.colour() != mover_ || q.royal()) continue;
          if (((around_kept_[t] >> sq) & 1u) != 0) continue;
          b.remove(sq);
        }
        if (a0_board_[t] == 1) {
          mv.replace_head(p, b.publish());
        } else {
          mv.append_board(p, b.publish());
        }
        ++stats_.scans;
        std::optional<RoyalCapture> capture;
        for_each_royal_capture(mv, rules_, opposite(mover_), s_.inactive_timelines(),
                               [&](const RoyalCapture& c) {
                                 capture = c;
                                 return true;
                               });
        if (!capture) {
          around_[t] = 1;
          break;
        }
        // Keep a blocker of the mover's on b that the capture's line crosses.
        bool kept = false;
        const std::vector<CapturePlace> places = capture_places(*capture);
        for (std::size_t i = 1; i + 1 < places.size() && !kept; ++i) {
          const CapturePlace& place = places[i];
          if (place.timeline != p || place.half_turn != head_half_turn(p) + 1) continue;
          const Piece q = head.at(place.square);
          if (!q.empty() && q.colour() == mover_ && !q.royal()) {
            around_kept_[t] |= std::uint64_t{1} << place.square;
            ++stats_.kept_blockers;
            kept = true;
          }
        }
        if (!kept) break;
      }
    }
  }

  int head_half_turn(int p) const { return s_.multiverse().timeline(p)->last_half_turn(); }

  bool finish_around_a0(int p) {
    const std::size_t t = index(p);
    const Move* first = nullptr;
    for (const Move& m : moves_) {
      if (m.from_timeline == p && m.physical() && quiet(m)) {
        first = &m;
        break;
      }
    }
    if (first == nullptr) return false;
    (void)t;
    GameState g = s_;
    g.apply(*first);
    for (const Move& x : *a0_) {
      if (x.from_timeline == p) continue;
      if (g.obligations_met()) break;  // met before a0_'s end: finished here
      if (!g.may_move() || !is_pseudo_legal(g, x, scratch_)) return false;
      g.apply(x);
    }
    return g.obligations_met();
  }

  // Per board of the mover's: a capture standing once the board is played
  // with its pieces unchanged (a passed copy), or none. Moves that change none
  // of its squares on that board leave it standing (tier 0's lemma): refuted.
  void prepare_passed() {
    passed_.assign(in_x_.size(), {});
    passed_found_.assign(in_x_.size(), 0);
    const Multiverse& base = s_.multiverse();
    for (const int p : boards_) {
      Multiverse mv = base;
      BoardBuilder b(*base.head(p));
      b.set_en_passant(kNoSquare);
      mv.append_board(p, b.publish());
      ++stats_.scans;
      std::optional<RoyalCapture> capture;
      for_each_royal_capture(mv, rules_, opposite(mover_), s_.inactive_timelines(), [&](const RoyalCapture& c) {
        // Under Q18's alternative a capture across void could be blocked by a
        // board made later: not taken.
        if (rules_.sliders_cross_void) {
          for (const CapturePlace& place : capture_places(c)) {
            if (mv.board_at(place.timeline, place.half_turn) == nullptr) return false;
          }
        }
        capture = c;
        return true;
      });
      if (!capture) continue;
      std::uint64_t squares = 0;
      for (const CapturePlace& place : capture_places(*capture)) {
        if (place.timeline == p && place.half_turn == head_half_turn(p) + 1) {
          squares |= std::uint64_t{1} << place.square;
        }
      }
      passed_[index(p)] = squares;
      passed_found_[index(p)] = 1;
    }
  }

  // A finish made greedily: on each board the action must play, a move within
  // the board - where the board played as it is would leave a capture
  // standing, one that changes a square the capture rests on (a check's
  // answer); elsewhere a quiet one - replayed and tested (C4). A capture in
  // the end position names a board whose move is then changed; a few tries,
  // each a check of the budget (a replay and a scan, as tier 2's).
  std::optional<std::vector<Move>> greedy_finish() {
    if (!s_.may_move()) return std::nullopt;
    std::vector<std::vector<int>> options(mandatory_.size());
    for (std::size_t k = 0; k < mandatory_.size(); ++k) {
      const int r = mandatory_[k];
      const std::size_t t = index(r);
      std::vector<int> fixing;
      std::vector<int> quiet_ones;
      for (std::size_t i = 0; i < moves_.size(); ++i) {
        const Move& m = moves_[i];
        if (m.from_timeline != r || !m.physical()) continue;
        if (passed_found_[t] != 0 && (changed_on(m, r) & passed_[t]) != 0) {
          fixing.push_back(static_cast<int>(i));
        } else if (passed_found_[t] == 0 && quiet(m)) {
          quiet_ones.push_back(static_cast<int>(i));
        }
      }
      options[k] = passed_found_[t] != 0 ? fixing : quiet_ones;
      if (options[k].empty()) return std::nullopt;
    }
    std::vector<std::size_t> pick(mandatory_.size(), 0);
    for (int attempt = 0; attempt < 8; ++attempt) {
      if (!take_check()) return std::nullopt;
      GameState g = s_;
      std::vector<Move> seq;
      bool ok = true;
      for (std::size_t k = 0; k < mandatory_.size() && ok; ++k) {
        const Move& m = moves_[static_cast<std::size_t>(options[k][pick[k]])];
        if (!g.may_move() || !is_pseudo_legal(g, m, scratch_)) {
          ok = false;
          break;
        }
        g.apply(m);
        seq.push_back(m);
      }
      if (!ok || !g.obligations_met()) return std::nullopt;
      const std::optional<RoyalCapture> c = submission_witness(g);
      if (!c) return seq;
      // Change the move on a board the capture rests on, if it has another.
      bool changed = false;
      for (const CapturePlace& place : capture_places(*c)) {
        const auto it = std::find(mandatory_.begin(), mandatory_.end(), place.timeline);
        if (it == mandatory_.end()) continue;
        const auto k = static_cast<std::size_t>(it - mandatory_.begin());
        if (pick[k] + 1 < options[k].size()) {
          ++pick[k];
          changed = true;
          break;
        }
      }
      if (!changed) return std::nullopt;
    }
    return std::nullopt;
  }

  // The squares of board p a move changes there: where it leaves, where it
  // lands, a castling rook's squares, an en passant victim.
  static std::uint64_t changed_on(const Move& m, int p) {
    std::uint64_t out = 0;
    const auto put = [&](int sq) {
      if (sq >= 0 && sq < kSquareCount) out |= std::uint64_t{1} << sq;
    };
    if (m.from_timeline == p) put(m.from_square);
    const bool lands_here = m.physical() ? m.from_timeline == p : (!m.branching() && m.to_timeline == p);
    if (lands_here) put(m.to_square);
    if (m.from_timeline == p && (m.castle() || m.en_passant())) {
      put(m.aux_square);
      if (m.castle()) {
        // The rook lands on the square the king crosses.
        put(m.from_square + (m.to_square > m.from_square ? 1 : -1));
      }
    }
    return out;
  }

  bool refuted_by_passed(const Move& m) const {
    for (const int p : {m.from_timeline, (!m.physical() && !m.branching()) ? m.to_timeline : m.from_timeline}) {
      const std::size_t t = index(p);
      if (passed_found_[t] != 0 && (changed_on(m, p) & passed_[t]) == 0) return true;
    }
    return false;
  }

  // A quiet move within a board stripped around a0_ (not of a kept piece).
  bool around_a0_certifies(const Move& m) const {
    if (!f0_ || !m.physical() || !quiet(m)) return false;
    const std::size_t t = index(m.from_timeline);
    return around_[t] != 0 && ((around_kept_[t] >> m.from_square) & 1u) == 0;
  }

  // -- tier 2 ---------------------------------------------------------------------

  MoveLegality tier2(const Move& m, const GameState& after) {
    if (after.obligations_met() && royals_safe(after)) return MoveLegality::Legal;
    if (standing_capture(after)) return MoveLegality::Illegal;
    if (!after.may_move()) return MoveLegality::Illegal;  // met, unsafe, and no move may follow
    if (modified_certifies(m, after)) {
      ++stats_.modified;
      return MoveLegality::Legal;
    }
    // A finish: a0_'s moves on the boards the move did not play, in a0_'s
    // order, replayed.
    if (a0_) {
      const auto [a, b] = boards_of(m);
      GameState t = after;
      bool ok = true;
      for (const Move& x : *a0_) {
        const auto [xa, xb] = boards_of(x);
        if (xa == a || (b && xa == *b) || (xb && (*xb == a || (b && *xb == *b)))) continue;
        if (!t.may_move() || !is_pseudo_legal(t, x, scratch_)) {
          ok = false;
          break;
        }
        t.apply(x);
      }
      if (ok && t.obligations_met() && royals_safe(t)) return MoveLegality::Legal;
    }
    // A finish: a move within the board on each board still to play.
    GameState t = after;
    std::vector<Move> played;
    for (const int r : after.mandatory_timelines()) {
      if (r < min_timeline_ || r > s_.multiverse().max_timeline()) return MoveLegality::Unknown;
      const int f = filler_[index(r)];
      if (f < 0 || !t.may_move()) return MoveLegality::Unknown;
      const Move& filler = moves_[static_cast<std::size_t>(f)];
      if (!is_pseudo_legal(t, filler, scratch_)) return MoveLegality::Unknown;
      t.apply(filler);
      played.push_back(filler);
    }
    if (t.obligations_met() && royals_safe(t)) return MoveLegality::Legal;
    return MoveLegality::Unknown;
  }

  // A move tier 1 does not settle - a royal's, a kept piece's, a castling, a
  // move on a board out of X - with the stripped multiverse: its own successor
  // boards (and new board), stripped like the others but for the royals and
  // the kept pieces still in place, and the boards then to play by quiet
  // moves within the board (the obligations then met with them, or already).
  bool modified_certifies(const Move& m, const GameState& after) {
    if (!bulk_) return false;
    if (m.branching() && (!rules_.inactive_history_is_destination || rules_.play_activates)) return false;
    if (!after.obligations_met()) {
      for (const int r : after.mandatory_timelines()) {
        if (r < min_timeline_ || r > s_.multiverse().max_timeline()) return false;
        if (in_x_[index(r)] == 0 || quiet_moves_[index(r)] == 0) return false;
      }
    }
    Multiverse mv = *stripped_;
    const auto strip_real = [&](const Board& real, std::uint64_t kept) {
      BoardBuilder b(real);
      b.set_en_passant(kNoSquare);
      for (int sq = 0; sq < kSquareCount; ++sq) {
        const Piece q = real.at(sq);
        if (q.empty() || q.colour() != mover_ || q.royal()) continue;
        if (((kept >> sq) & 1u) != 0) continue;
        b.remove(sq);
      }
      return b.publish();
    };
    const auto [a, b] = boards_of(m);
    for (const std::optional<int> q : {std::optional<int>(a), b}) {
      if (!q) continue;
      const BoardRef board = strip_real(*after.multiverse().head(*q), kept_[index(*q)]);
      if (in_x_[index(*q)] != 0) {
        mv.replace_head(*q, board);
      } else {
        mv.append_board(*q, board);
      }
    }
    if (m.branching()) {
      const Multiverse& amv = after.multiverse();
      const int p = mover_ == Colour::White ? amv.max_timeline() : amv.min_timeline();
      const Timeline* t = amv.timeline(p);
      NewTimeline nt;
      nt.position = p;
      nt.first_half_turn = t->first_half_turn();
      nt.boards = {strip_real(*t->head(), 0)};
      nt.creator = mover_;
      nt.creation_order = mv.created_by(mover_) + 1;
      mv.add_timeline(std::move(nt));
    }
    ++stats_.scans;
    return !royal_capture_exists(mv, rules_, opposite(mover_), after.inactive_timelines());
  }

  // -- tier 3 ---------------------------------------------------------------------

  bool take_check() {
    if (checks_ >= budget_.checks) return false;
    ++checks_;
    return true;
  }

  Finish search(const GameState& s) {
    if (spent_ >= budget_.search_nodes) return Finish{};
    const Finish f =
        search_finish(s, budget_.search_nodes - spent_, budget_.max_bytes, budget_.detector);
    spent_ += f.nodes;
    stats_.search_nodes += f.nodes;
    return f;
  }

  // A legal finish (from the state) found: each of its moves that can lead it
  // - played first, the rest in the order found, through the state machine to
  // a legal end - begins a legal action.
  void learn(const std::vector<Move>& finish) {
    std::vector<Move> order;
    for (std::size_t i = 0; i < finish.size(); ++i) {
      if (known_legal(finish[i])) continue;
      order.assign(1, finish[i]);
      for (std::size_t j = 0; j < finish.size(); ++j) {
        if (j != i) order.push_back(finish[j]);
      }
      if (i == 0 || replays_legally(order)) known_.push_back(finish[i]);
    }
  }

  bool replays_legally(const std::vector<Move>& order) {
    GameState t = s_;
    for (const Move& m : order) {
      if (!t.may_move() || !is_pseudo_legal(t, m, scratch_)) return false;
      t.apply(m);
    }
    return t.obligations_met() && royals_safe(t);
  }

  bool known_legal(const Move& m) const {
    return std::find(known_.begin(), known_.end(), m) != known_.end();
  }

  const GameState& s_;
  const RuleOptions& rules_;
  Colour mover_;
  std::span<const Move> moves_;
  const LegalityOptions& budget_;
  LegalityStats& stats_;
  int min_timeline_;

  std::vector<int> boards_;           // the mover's playable boards
  std::vector<int> mandatory_;
  std::vector<char> in_x_;            // tier 1's boards
  std::vector<std::uint64_t> kept_;   // tier 1's kept blockers, by square
  std::vector<int> quiet_moves_;      // quiet moves within the board, not kept
  std::vector<int> filler_;           // tier 2's move within the board, or -1
  std::vector<int> blocked_;          // mandatory boards tier 1 cannot finish
  bool bulk_ = false;
  bool branches_allowed_ = false;
  std::optional<Multiverse> stripped_;
  bool tier1_possible_ = false;
  bool searched_root_ = false;
  std::optional<std::vector<Move>> a0_;       // a legal finish from the state
  std::optional<Multiverse> f0_;              // where a0_ ends (no creation in it)
  std::vector<char> a0_board_;                // by timeline index: how a0_ plays it
  std::vector<char> around_;                  // boards stripped clean around a0_
  std::vector<std::uint64_t> around_kept_;    // their kept blockers
  std::vector<std::uint64_t> passed_;         // by board: a passed capture's squares there
  std::vector<char> passed_found_;
  std::optional<PresentModel> model_;
  std::vector<std::size_t> axis_;      // by timeline index: the board's place in boards_
  std::vector<char> played_;
  std::map<std::pair<int, int>, bool> destinations_;
  std::vector<Move> known_;
  std::uint64_t spent_ = 0;
  std::uint64_t checks_ = 0;
  MoveList scratch_;
};

}  // namespace

void classify_moves(const GameState& state, std::span<const Move> moves,
                    std::vector<MoveLegality>& out, const LegalityOptions& budget,
                    LegalityStats* stats) {
  LegalityStats local;
  Classifier(state, moves, budget, stats != nullptr ? *stats : local).run(out);
}

}  // namespace t5d
