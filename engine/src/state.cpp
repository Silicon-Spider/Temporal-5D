// The game state machine: activity, the present, move application and
// submission (see t5d/state.hpp). Rule IDs refer to docs/rules/README.md.

#include "t5d/state.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#include "t5d/board.hpp"
#include "t5d/geometry.hpp"
#include "t5d/timeline.hpp"
#include "t5d/zobrist.hpp"

namespace t5d {

namespace {

// Every half-turn index stays within [0, kMaxHalfTurn] (as in multiverse.cpp).
constexpr int kMaxHalfTurn = 2 * kMaxTurn + 1;

std::string colour_name(Colour c) { return c == Colour::White ? "White" : "Black"; }

std::string describe_board(const Multiverse& mv, int p, int h) {
  return "(" + timeline_label_string(p, mv.even()) + "T" + std::to_string(turn_of(h)) + ", " +
         (to_move_at(h) == Colour::White ? "white" : "black") + " to move)";
}

[[noreturn]] void bad_move(const Multiverse& mv, const Move& m, const std::string& why) {
  throw std::invalid_argument("move " + to_string(m, mv.even()) + ": " + why);
}

// Q2's credit: under MinusZeroToBlack, in a game with an even number of
// starting timelines, Black counts as having created one timeline already.
int credit_of(const Multiverse& mv, const RuleOptions& r, Colour c) noexcept {
  return (r.initial_credit == InitialTimelineCredit::MinusZeroToBlack && mv.even() &&
          c == Colour::Black)
             ? 1
             : 0;
}

// T10's count rule as a closed form: initial timelines are active; the n-th
// timeline a player created is active iff the opponent has created at least
// n-1 (each count with Q2's credit). T11's reactivation keeps this true at
// every moment, so it is also the activity of a position read from 5DFEN.
bool count_rule_active(const Multiverse& mv, const RuleOptions& r, const Timeline& t) noexcept {
  if (!t.creator()) return true;
  const Colour c = *t.creator();
  const Colour o = opposite(c);
  return mv.created_by(o) + credit_of(mv, r, o) >= t.creation_order() + credit_of(mv, r, c) - 1;
}

std::vector<int> count_rule_inactive(const Multiverse& mv, const RuleOptions& r) {
  std::vector<int> out;
  for (const TimelineRef& t : mv.timelines()) {
    if (!count_rule_active(mv, r, *t)) out.push_back(t->position());  // ascending
  }
  return out;
}

bool central_timelines_initial(const Multiverse& mv) noexcept {
  const auto initial = [&](int p) {
    const Timeline* t = mv.timeline(p);
    return t != nullptr && !t->creator().has_value();
  };
  return mv.even() ? initial(-1) && initial(0) : initial(0);
}

bool is_pawnlike(PieceKind k) noexcept { return k == PieceKind::Pawn || k == PieceKind::Brawn; }

int last_rank(Colour c, int height) noexcept { return c == Colour::White ? height - 1 : 0; }

std::size_t colour_index(int half_turn) noexcept {
  return static_cast<std::size_t>(index_of(to_move_at(half_turn)));
}

// Q1: the one rule that reads who made the last creation.
bool consecutive_rule(const RuleOptions& r) noexcept {
  return r.activation == ActivationRule::Consecutive;
}

}  // namespace

// ---- construction ----------------------------------------------------------------

GameState::GameState(Multiverse multiverse, const RuleOptions& rules)
    : mv_(std::move(multiverse)), rules_(rules) {
  rules_.validate();
  if (mv_.empty()) throw std::invalid_argument("a game state needs at least one timeline");
  if (!central_timelines_initial(mv_)) {
    throw std::invalid_argument(
        std::string("a game state needs its central timeline") +
        (mv_.even() ? "s \"-0\" and \"+0\"" : " \"0\"") +
        ", as initial timelines (no creator): every game starts with them (L7, L8)");
  }
  mv_.set_ruleset(rules_.ruleset_id());
  inactive_ = count_rule_inactive(mv_, rules_);
  rescan();
  settle();
  if (present_colour() != to_move()) {
    throw std::invalid_argument(
        "the present is on " + colour_name(present_colour()) + "'s colour (half-turn column " +
        std::to_string(present_) + ", turn " + std::to_string(turn_of(present_)) + ") but " +
        colour_name(to_move()) +
        " is to move: every turn begins with the present on the mover's colour (T4, T5)");
  }
}

// ---- activity ------------------------------------------------------------------------

bool GameState::active(int p) const noexcept {
  return mv_.has_timeline(p) && !std::binary_search(inactive_.begin(), inactive_.end(), p);
}

bool GameState::deferred(int p) const noexcept {
  for (const Deferral& d : deferred_) {
    if (d.timeline == p) return true;
  }
  return false;
}

bool GameState::counts_for_present(int p) const noexcept { return active(p) && !deferred(p); }

void GameState::set_inactive(int p, bool inactive) {
  const auto it = std::lower_bound(inactive_.begin(), inactive_.end(), p);
  const bool listed = it != inactive_.end() && *it == p;
  if (inactive && !listed) inactive_.insert(it, p);
  if (!inactive && listed) inactive_.erase(it);
}

int GameState::credited_count(Colour c) const noexcept {
  return mv_.created_by(c) + credit_of(mv_, rules_, c);
}

int GameState::credited_order(Colour creator, int order) const noexcept {
  return order + credit_of(mv_, rules_, creator);
}

void GameState::on_created(int p, Colour creator) {
  const Timeline* t = mv_.timeline(p);
  // T10 / Q1: the new timeline's own activity.
  bool active_now = true;
  if (rules_.activation == ActivationRule::Count) {
    active_now = credited_count(opposite(creator)) >= credited_order(creator, t->creation_order()) - 1;
  } else {
    active_now = !(last_creator_.has_value() && *last_creator_ == creator);
  }
  if (active_now) {
    count_in(p);  // T8: an active branch behind the present rewinds it
  } else {
    set_inactive(p, true);  // T8: an inactive one never moves the present
  }
  // T11: the creation activates the opponent's earliest-created timeline that
  // is still inactive, if there is one - exactly one.
  const Colour other = opposite(creator);
  int earliest = 0;
  int earliest_order = INT_MAX;
  for (const int q : inactive_) {
    const Timeline* u = mv_.timeline(q);
    if (u->creator() == other && u->creation_order() < earliest_order) {
      earliest = q;
      earliest_order = u->creation_order();
    }
  }
  if (earliest_order != INT_MAX) {
    set_inactive(earliest, false);
    if (rules_.reactivation == Reactivation::Deferred && playable_by_mover(earliest)) {
      // Q6 Deferred: a reactivated board of the mover's own colour obliges
      // the mover only from the mover's next turn on.
      const Deferral d{earliest, creator};
      deferred_.insert(std::lower_bound(deferred_.begin(), deferred_.end(), d,
                                        [](const Deferral& a, const Deferral& b) {
                                          return a.timeline < b.timeline;
                                        }),
                       d);
    } else {
      count_in(earliest);  // T12: rewinding by reactivation
    }
  }
  last_creator_ = creator;
}

void GameState::on_played(int p, Colour mover) {
  // Q3 (play activates): a move by the player who did not create an inactive
  // timeline, on or onto its playable board, activates it. (An inactive
  // timeline is never deferred.)
  if (rules_.play_activates && !active(p)) {
    const Timeline* t = mv_.timeline(p);
    if (t->creator().has_value() && *t->creator() != mover) {
      set_inactive(p, false);
      count_in(p);
    }
  }
  // Q6 Deferred: a deferred board that its owner plays obliges no one; the
  // timeline counts for the present from now on (a deferred one is active).
  const auto it = std::find_if(deferred_.begin(), deferred_.end(),
                               [p](const Deferral& d) { return d.timeline == p; });
  if (it != deferred_.end()) {
    deferred_.erase(it);
    count_in(p);
  }
}

// ---- the present and the action ---------------------------------------------------------

void GameState::count_in(int p) noexcept { head_added(mv_.timeline(p)->last_half_turn()); }

void GameState::head_added(int h) noexcept {
  ++heads_[colour_index(h)];
  if (at_present_ == 0) return;  // the present is stale: settle() rescans
  if (h < present_) {
    present_ = h;
    at_present_ = 1;
  } else if (h == present_) {
    ++at_present_;
  }
}

void GameState::head_removed(int h) noexcept {
  --heads_[colour_index(h)];
  // The last counting head in the present's column leaves it: the present
  // moves on (T5, T9), to wherever the earliest remaining head is - rescanned.
  if (at_present_ != 0 && h == present_) --at_present_;
}

void GameState::rescan() noexcept {
  int present = INT_MAX;
  int at = 0;
  std::array<int, 2> heads{};
  for (const TimelineRef& t : mv_.timelines()) {
    if (!counts_for_present(t->position())) continue;
    const int h = t->last_half_turn();
    ++heads[colour_index(h)];
    if (h < present) {
      present = h;
      at = 1;
    } else if (h == present) {
      ++at;
    }
  }
  // The central timelines are initial, so always active and never deferred:
  // the present always exists.
  present_ = present;
  at_present_ = at;
  heads_ = heads;
}

void GameState::settle() noexcept {
  if (at_present_ == 0) rescan();
  const std::size_t mover = static_cast<std::size_t>(index_of(to_move()));
  if (rules_.mandatory == MandatoryBoards::AllActive) {
    mandatory_count_ = heads_[mover];  // Q5: every counting board of the mover's
  } else {
    // T6: the counting boards in the present - the mover's iff the present is
    // on the mover's colour.
    mandatory_count_ = colour_index(present_) == mover ? at_present_ : 0;
  }
}

bool GameState::playable_by_mover(int p) const noexcept {
  const Timeline* t = mv_.timeline(p);
  return t != nullptr && to_move_at(t->last_half_turn()) == to_move();
}

bool GameState::mandatory(int p) const noexcept {
  if (!counts_for_present(p) || !playable_by_mover(p)) return false;
  if (rules_.mandatory == MandatoryBoards::AllActive) return true;
  return mv_.timeline(p)->last_half_turn() == present_;
}

std::vector<int> GameState::mandatory_timelines() const {
  std::vector<int> out;
  for (const TimelineRef& t : mv_.timelines()) {
    if (mandatory(t->position())) out.push_back(t->position());
  }
  return out;
}

std::vector<int> GameState::playable_timelines() const {
  std::vector<int> out;
  for (const TimelineRef& t : mv_.timelines()) {
    if (playable_by_mover(t->position())) out.push_back(t->position());
  }
  return out;
}

// ---- moves ----------------------------------------------------------------------------------

void GameState::apply(const Move& m) {
  if (!may_move()) {
    throw std::logic_error(
        "move " + to_string(m, mv_.even()) +
        ": no move may be made now - the action's obligations are met (the present has shifted "
        "to the opponent), so it must be submitted (T5) - under Q4's Guide-literal reading "
        "(moves_after_shift = false; the default, R8, allows moves until submission)");
  }
  const Colour mover = to_move();
  const BoardSize size = mv_.board_size();

  // -- the source: a piece of the mover's on a playable board of the mover's (L5)
  const Timeline* src_tl = mv_.timeline(m.from_timeline);
  if (src_tl == nullptr) bad_move(mv_, m, "there is no source timeline");
  if (m.from_half_turn != src_tl->last_half_turn()) {
    bad_move(mv_, m, "the source board is not its timeline's playable board (L5)");
  }
  if (to_move_at(m.from_half_turn) != mover) {
    bad_move(mv_, m, "the source board is not " + colour_name(mover) + "'s (L5)");
  }
  if (!size.contains_square(m.from_square) || !size.contains_square(m.to_square)) {
    bad_move(mv_, m, "a square is off the board");
  }
  // Boards are held by reference: every board read here stays in the
  // multiverse (as history, or untouched) through the edits below.
  const Board& src = *src_tl->head();
  const Piece piece = src.at(m.from_square);
  if (piece.empty() || piece.colour() != mover) {
    bad_move(mv_, m, "no piece of " + colour_name(mover) + "'s on the source square");
  }
  if (piece != m.piece) bad_move(mv_, m, "the move's piece is not the one on the source square");

  // -- the destination: an existing board of the mover's colour (J2, L4, L6)
  const bool physical = m.physical();
  const Timeline* dst_tl = physical ? src_tl : mv_.timeline(m.to_timeline);
  if (dst_tl == nullptr || !dst_tl->contains(m.to_half_turn)) {
    bad_move(mv_, m, "there is no board at the destination (J2, L6)");
  }
  if (to_move_at(m.to_half_turn) != mover) {
    bad_move(mv_, m, "the destination board is not of the mover's colour (L4)");
  }
  // J3: a move between boards branches iff it lands on a history board (L5)
  // - or, under the Quick Rules' literal reading (earlier_playable_jump_
  // branches), whenever it lands at an earlier turn than it started.
  const bool history = !physical && m.to_half_turn != dst_tl->last_half_turn();
  const bool branches = history || (!physical && rules_.earlier_playable_jump_branches &&
                                    m.to_half_turn < m.from_half_turn);
  if (branches != m.branching()) {
    bad_move(mv_, m, "the branching flag disagrees with the destination board (J3)");
  }
  if (history && !rules_.turn_zero_reachable && turn_of(m.to_half_turn) == 0) {
    bad_move(mv_, m, "a turn-zero board is not a destination under this ruleset (Q8)");
  }
  if (history && !rules_.inactive_history_is_destination && !active(m.to_timeline)) {
    bad_move(mv_, m, "a history board of an inactive timeline is not a destination under "
                     "this ruleset (Q20)");
  }
  const Board& dst = physical ? src : *dst_tl->board_at(m.to_half_turn);
  const Piece target = dst.at(m.to_square);
  const bool pawnlike = is_pawnlike(piece.kind());
  const int height = size.height;

  // -- the special cases
  if (m.castle()) {
    // S4: only a royal king castles (a common king never carries the unmoved
    // flag, which 5DFEN tracks only on pawns, rooks, kings and brawns).
    const bool king_ok = piece.kind() == PieceKind::King;
    const Piece rook = size.contains_square(m.aux_square) ? src.at(m.aux_square) : Piece{};
    const int df = file_of(m.to_square) - file_of(m.from_square);
    const int dr = file_of(m.aux_square) - file_of(m.from_square);
    // The rook lands on the square the king crosses, which must be free.
    const int crossed_file = file_of(m.from_square) + (df > 0 ? 1 : -1);
    if (!physical || !king_ok || !piece.unmoved() || rook.empty() ||
        !src.at(crossed_file, rank_of(m.from_square)).empty() ||
        rook.kind() != PieceKind::Rook || rook.colour() != mover || !rook.unmoved() ||
        rank_of(m.to_square) != rank_of(m.from_square) ||
        rank_of(m.aux_square) != rank_of(m.from_square) || (df != 2 && df != -2) ||
        (dr > 0) != (df > 0) || (target != Piece{} && m.to_square != m.aux_square) ||
        !m.captured.empty() || m.capture() || m.promotes() || m.en_passant() ||
        m.double_step()) {
      bad_move(mv_, m, "not a castling move of an unmoved king and rook (S1)");
    }
  } else if (m.en_passant()) {
    // S7 / Q16: pawns - and brawns, unless the ruleset excludes them.
    const auto takes_part = [&](Piece x) {
      return !x.empty() && (x.kind() == PieceKind::Pawn ||
                            (rules_.en_passant_brawns && x.kind() == PieceKind::Brawn));
    };
    const int fy = geometry::forward_rank(mover);
    const int victim_square = make_square(file_of(m.to_square), rank_of(m.to_square) - fy);
    const Piece victim = src.at(victim_square);
    if (!physical || !takes_part(piece) || src.en_passant() != m.to_square || !target.empty() ||
        m.aux_square != victim_square || !takes_part(victim) || victim.colour() == mover ||
        victim != m.captured || !m.capture() || m.double_step()) {
      bad_move(mv_, m, "not an en passant capture on this board (S7)");
    }
  } else {
    if (m.aux_square != kNoSquare) bad_move(mv_, m, "an auxiliary square on an ordinary move");
    if (target != m.captured) bad_move(mv_, m, "the captured piece is not the one there");
    if (!target.empty() && target.colour() == mover) {
      bad_move(mv_, m, "the destination holds a piece of the mover's own");
    }
    if (m.capture() != !target.empty()) bad_move(mv_, m, "the capture flag disagrees");
  }
  // S8: a pawn that reaches its last rank promotes - and a brawn, unless the
  // ruleset says otherwise (Q17: brawns_promote; brawn_cross_board_promotion
  // for an arrival there from another board).
  const bool can_promote =
      piece.kind() == PieceKind::Pawn ||
      (piece.kind() == PieceKind::Brawn && rules_.brawns_promote &&
       (physical || rules_.brawn_cross_board_promotion));
  const bool promotes =
      can_promote && rank_of(m.to_square) == last_rank(mover, height) && !m.castle();
  if (m.promotes() != promotes) bad_move(mv_, m, "the promotion flag disagrees (S8)");
  if (promotes && !rules_.promotions.contains(m.promotion)) {
    bad_move(mv_, m, "promotion to a piece outside this ruleset's promotion set (S8)");
  }
  if (m.double_step()) {
    // 4.6 (Q15): two steps forward along the rank or the timeline axis, by a
    // pawn or brawn with the right, through an empty square onto an empty
    // one. Checked in full: the en-passant square it may create rests on it.
    const int home = mover == Colour::White ? 1 : height - 2;
    const bool right = rules_.double_step_right == DoubleStepRight::UnmovedFlag
                           ? piece.unmoved()
                           : rank_of(m.from_square) == home;
    bool ok = pawnlike && right && !m.capture() && target.empty() && !m.castle() &&
              !m.en_passant();
    if (ok && physical) {
      const int fy = geometry::forward_rank(mover);
      ok = file_of(m.to_square) == file_of(m.from_square) &&
           rank_of(m.to_square) - rank_of(m.from_square) == 2 * fy &&
           src.at(file_of(m.from_square), rank_of(m.from_square) + fy).empty();
    } else if (ok) {
      const int fl = geometry::forward_timeline(mover);
      ok = rules_.timeline_double_step && m.to_square == m.from_square &&
           m.to_half_turn == m.from_half_turn && m.to_timeline - m.from_timeline == 2 * fl;
      if (ok) {
        // Through the same square on the in-between timeline's board - or,
        // under double_step_over_void, past a place with no board (J8).
        const Board* between = mv_.board_at(m.from_timeline + fl, m.from_half_turn);
        ok = between != nullptr ? between->at(m.from_square).empty()
                                : rules_.double_step_over_void;
      }
    }
    if (!ok) bad_move(mv_, m, "not a pawn's or brawn's double step (4.6, Q15)");
  }

  // -- the limits of the multiverse (a game would need ~2^20 turns or
  //    timelines to reach them); checked before anything changes
  if (m.from_half_turn + 1 > kMaxHalfTurn || m.to_half_turn + 1 > kMaxHalfTurn) {
    bad_move(mv_, m, "the half-turn index would leave its range");
  }
  const int new_position = mover == Colour::White ? mv_.max_timeline() + 1 : mv_.min_timeline() - 1;
  if (branches &&
      (new_position > kMaxTimelinePosition || new_position < min_timeline_position(mv_.even()))) {
    bad_move(mv_, m, "the new timeline's index would leave its range");
  }

  // -- the piece that arrives: promoted, or the moving piece without its
  //    unmoved flag (Q15: unless a pawn or brawn left its board under
  //    jumps_keep_unmoved)
  Piece placed = piece.as_moved();
  if (promotes) {
    placed = Piece(m.promotion, mover);
  } else if (!physical && pawnlike && rules_.jumps_keep_unmoved) {
    placed = piece;
  }

  // -- the present's bookkeeping: which played timelines count now (T4)
  const bool advances_destination = !physical && !branches;  // J4
  const bool src_counted = counts_for_present(m.from_timeline);
  const bool dst_counted = advances_destination && counts_for_present(m.to_timeline);
  const int dst_head = advances_destination ? dst_tl->last_half_turn() : 0;

  // -- build every new board before changing anything
  if (physical) {
    BoardBuilder b(src);
    b.set_en_passant(kNoSquare);
    if (m.castle()) {
      const Piece rook = src.at(m.aux_square);
      const int dir = file_of(m.to_square) > file_of(m.from_square) ? 1 : -1;
      const int rook_to = make_square(file_of(m.from_square) + dir, rank_of(m.from_square));
      b.remove(m.from_square);
      b.remove(m.aux_square);
      b.put(m.to_square, piece.as_moved());
      b.put(rook_to, rook.as_moved());
    } else {
      if (m.en_passant()) b.remove(m.aux_square);
      b.remove(m.from_square);
      b.put(m.to_square, placed);
      if (m.double_step()) {
        // S7: en passant within this board, by the opponent, from the board
        // this move creates - recorded only where it can be taken: the piece
        // that arrived is still a pawn (or brawn) - not promoted - and an
        // enemy pawn (or brawn) stands beside it. So equal positions keep
        // equal keys.
        const auto takes_part = [&](Piece x) {
          return !x.empty() && (x.kind() == PieceKind::Pawn ||
                                (rules_.en_passant_brawns && x.kind() == PieceKind::Brawn));
        };
        bool takeable = false;
        for (const int df : {-1, 1}) {
          const Piece neighbour = b.at(file_of(m.to_square) + df, rank_of(m.to_square));
          if (takes_part(neighbour) && neighbour.colour() != mover) takeable = true;
        }
        if (takes_part(placed) && takeable) {
          b.set_en_passant(make_square(file_of(m.from_square),
                                       (rank_of(m.from_square) + rank_of(m.to_square)) / 2));
        }
      }
    }
    mv_.append_board(m.from_timeline, b.publish());
  } else {
    BoardBuilder s(src);
    s.set_en_passant(kNoSquare);
    s.remove(m.from_square);
    BoardBuilder d(dst);
    d.set_en_passant(kNoSquare);
    if (branches && !rules_.branch_copies_keep_unmoved) {
      // Q15/Q21 alternative: the copy's pieces lose their unmoved flags.
      for (std::uint64_t bits = dst.unmoved(); bits != 0; bits &= bits - 1) {
        const int sq = std::countr_zero(bits);
        d.put(sq, d.at(sq).as_moved());
      }
    }
    d.put(m.to_square, placed);
    BoardRef source_next = s.publish();
    BoardRef arrival = d.publish();
    mv_.append_board(m.from_timeline, std::move(source_next));
    if (advances_destination) {
      // J4: a non-branching jump - the destination timeline advances too.
      mv_.append_board(m.to_timeline, std::move(arrival));
    } else {
      // J5: a branching jump - a new timeline, outermost on the mover's side
      // (L7), whose first board is the destination's half-step successor. The
      // destination itself never changes.
      NewTimeline nt;
      nt.position = new_position;
      nt.first_half_turn = m.to_half_turn + 1;
      nt.boards.push_back(std::move(arrival));
      nt.creator = mover;
      nt.creation_order = mv_.created_by(mover) + 1;
      nt.origin = BoardCoord{m.to_timeline, m.to_half_turn};
      mv_.add_timeline(std::move(nt));
    }
  }

  // -- activity (T10, T11, Q1-Q3, Q6) and the present (T4), brought up to date
  if (src_counted) {
    head_removed(m.from_half_turn);
    head_added(m.from_half_turn + 1);
  }
  if (dst_counted) {
    head_removed(dst_head);
    head_added(dst_head + 1);
  }
  on_played(m.from_timeline, mover);
  if (advances_destination) on_played(m.to_timeline, mover);
  if (!physical && !advances_destination) on_created(new_position, mover);
  settle();
}

// ---- submission ----------------------------------------------------------------------------

SubmitStatus GameState::submission_status(const CheckRuling& ruling) const {
  if (!obligations_met()) return SubmitStatus::ObligationsUnmet;
  if (ruling.royals_safe == nullptr) return SubmitStatus::Unchecked;
  switch (ruling.royals_safe(*this)) {
    case Verdict::Yes: return SubmitStatus::Safe;
    case Verdict::No: return SubmitStatus::Unsafe;
    case Verdict::Unknown: break;
  }
  return SubmitStatus::Unchecked;
}

void GameState::submit(const CheckRuling& ruling) {
  const SubmitStatus status = submission_status(ruling);
  if (status == SubmitStatus::ObligationsUnmet) {
    std::string boards;
    for (const int p : mandatory_timelines()) {
      if (!boards.empty()) boards += ", ";
      boards += describe_board(mv_, p, mv_.timeline(p)->last_half_turn());
    }
    throw std::logic_error(
        "cannot submit: the present has not shifted to the opponent (T5); mandatory boards "
        "left: " + boards);
  }
  if (status == SubmitStatus::Unsafe) {
    throw std::logic_error("cannot submit: a royal piece would be left capturable (C4)");
  }
  const Colour next = opposite(to_move());
  mv_.set_to_move(next);
  // Q6 Deferred: the deferrals of the player whose turn now begins end; those
  // timelines (active) count for the present from now on.
  for (std::size_t i = 0; i < deferred_.size();) {
    if (deferred_[i].owner == next) {
      const int p = deferred_[i].timeline;
      deferred_.erase(deferred_.begin() + static_cast<std::ptrdiff_t>(i));
      count_in(p);
    } else {
      ++i;
    }
  }
  settle();
}

// ---- key and equality --------------------------------------------------------------------

std::uint64_t GameState::zobrist() const noexcept {
  const std::uint64_t base = mv_.zobrist();
  if (!rules_.history_dependent_activity()) return base;
  // Activity is part of the state under these rules: fold it in.
  const auto& k = zobrist::detail::kKeys;
  std::uint64_t x = zobrist::mr_round(k.activity_seed, base);
  x = zobrist::mr_round(x, static_cast<std::uint64_t>(inactive_.size()));
  for (const int p : inactive_) {
    x = zobrist::mr_round(
        x, zobrist::avalanche(std::uint64_t{static_cast<std::uint32_t>(p)} ^ k.inactive_salt));
  }
  x = zobrist::mr_round(x, static_cast<std::uint64_t>(deferred_.size()));
  for (const Deferral& d : deferred_) {
    const std::uint64_t packed = (std::uint64_t{static_cast<std::uint32_t>(d.timeline)} << 1) |
                                 static_cast<std::uint64_t>(index_of(d.owner));
    x = zobrist::mr_round(x, zobrist::avalanche(packed ^ k.deferred_salt));
  }
  // The last creator is read by Q1's Consecutive reading alone.
  const std::optional<Colour> last = consecutive_rule(rules_) ? last_creator_ : std::nullopt;
  const std::size_t c = last ? 1 + static_cast<std::size_t>(index_of(*last)) : 0;
  x = zobrist::mr_round(x, k.creator[c] ^ k.last_creator_salt);
  return zobrist::avalanche(x);
}

bool operator==(const GameState& a, const GameState& b) noexcept {
  return a.rules_ == b.rules_ && a.inactive_ == b.inactive_ && a.deferred_ == b.deferred_ &&
         (!consecutive_rule(a.rules_) || a.last_creator_ == b.last_creator_) && a.mv_ == b.mv_;
}

// ---- free functions ------------------------------------------------------------------------

int initial_present(const Multiverse& mv, const RuleOptions& rules) {
  int present = INT_MAX;
  for (const TimelineRef& t : mv.timelines()) {
    if (count_rule_active(mv, rules, *t)) present = std::min(present, t->last_half_turn());
  }
  if (present == INT_MAX) {
    throw std::invalid_argument("no active timeline: the present is undefined (T4)");
  }
  return present;
}

GameState game_from_fen(std::span<const std::string_view> boards, BoardSize size, Parity parity,
                        const RuleOptions& rules, std::optional<Colour> to_move) {
  rules.validate();
  Multiverse mv = multiverse_from_fen(boards, size, parity, rules.ruleset_id(),
                                      to_move.value_or(Colour::White));
  if (!to_move) {
    if (!central_timelines_initial(mv)) {
      // Let the constructor say what is wrong.
      return GameState(std::move(mv), rules);
    }
    mv.set_to_move(to_move_at(initial_present(mv, rules)));
  }
  return GameState(std::move(mv), rules);
}

GameState game_from_fen(const std::vector<std::string>& boards, BoardSize size, Parity parity,
                        const RuleOptions& rules, std::optional<Colour> to_move) {
  std::vector<std::string_view> views(boards.begin(), boards.end());
  return game_from_fen(std::span<const std::string_view>(views), size, parity, rules, to_move);
}

Outcome outcome(const GameState& turn_start, const CheckRuling& ruling) {
  if (ruling.has_legal_action == nullptr) return Outcome::Unknown;
  switch (ruling.has_legal_action(turn_start)) {
    case Verdict::Yes: return Outcome::Ongoing;
    case Verdict::Unknown: return Outcome::Unknown;
    case Verdict::No: break;
  }
  if (ruling.in_check == nullptr) return Outcome::Unknown;
  switch (ruling.in_check(turn_start)) {
    case Verdict::Yes: return Outcome::Checkmate;
    case Verdict::No: return Outcome::Stalemate;
    case Verdict::Unknown: break;
  }
  return Outcome::Unknown;
}

}  // namespace t5d
