// Check status and submission legality (see t5d/check.hpp). Rule IDs refer to
// the rulebook, docs/rules/README.md; the definitions are its ruling R1
// (register H36). Written from the rulebook and the design spec; no other
// implementation was consulted.

#include "t5d/check.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "t5d/board.hpp"
#include "t5d/search.hpp"
#include "t5d/timeline.hpp"

namespace t5d {

namespace {

// ---- the scan: the move generator's capture semantics, one capture at a time ------
//
// The generator's rules (movegen.cpp) restated for captures only: what a piece
// of the attacker's can capture from its board. Each function returns whether
// the visitor asked to stop.

struct Scan {
  const Multiverse& mv;
  const RuleOptions& rules;
  std::span<const int> inactive;
  Colour attacker;
  BoardSize size;
  // The bounds of every board (timelines are contiguous; half-turns >= 0): a
  // slide that leaves them can never meet a board again.
  int min_timeline = 0;
  int max_timeline = -1;
  int max_half_turn = -1;
  // The source: timeline p's playable board, at half-turn h.
  int p = 0;
  int h = 0;
  const Board* board = nullptr;
};

// A board at a place, or void (board == nullptr), and whether it is a
// history board (L5: not the latest of its timeline).
struct Spot {
  const Board* board = nullptr;
  bool history = false;
  int half_turn = 0;
};

inline Spot spot_at(const Scan& s, int timeline, int half_turn) noexcept {
  if (timeline < s.min_timeline || timeline > s.max_timeline) return Spot{};
  const Timeline* t = s.mv.timeline(timeline);
  if (!t->contains(half_turn)) return Spot{};
  return Spot{t->board_at(half_turn), half_turn != t->last_half_turn(), half_turn};
}

// Whether a piece may land on `spot` (J2): any board but, under the ruleset's
// alternatives, a turn-zero history board (Q8) or a history board of an
// inactive timeline (Q20).
inline bool landable(const Scan& s, const Spot& spot, int timeline) noexcept {
  if (!spot.history) return true;
  if (!s.rules.turn_zero_reachable && turn_of(spot.half_turn) == 0) return false;
  if (!s.rules.inactive_history_is_destination &&
      std::binary_search(s.inactive.begin(), s.inactive.end(), timeline)) {
    return false;
  }
  return true;
}

// J3: a move between boards branches iff it lands on a history board - or,
// under the Quick Rules' literal reading, at an earlier turn than it started.
inline bool branches(const Scan& s, const Spot& spot, bool physical) noexcept {
  if (physical) return false;
  return spot.history || (s.rules.earlier_playable_jump_branches && spot.half_turn < s.h);
}

inline RoyalCapture capture_of(const Scan& s, int from, Piece attacker, Vec4 step, int steps,
                               int to_timeline, const Spot& spot, int to, Piece royal) noexcept {
  RoyalCapture c;
  c.attacker = attacker;
  c.from_timeline = s.p;
  c.from_half_turn = s.h;
  c.from_square = from;
  c.royal = royal;
  c.to_timeline = to_timeline;
  c.to_half_turn = spot.half_turn;
  c.to_square = to;
  c.step = step;
  c.steps = steps;
  c.branching = branches(s, spot, step.physical());
  return c;
}

// A leap, a king's step or a pawn's capture: only the destination matters
// (4.4, 4.5, 4.6, 4.7; J8: knights leap over void and pieces alike).
template <class Visit>
bool leap(const Scan& s, int from, Piece piece, Vec4 d, Visit& visit) {
  const int x = file_of(from) + d.x;
  const int y = rank_of(from) + d.y;
  if (!s.size.contains(x, y)) return false;
  const Spot spot =
      d.physical() ? Spot{s.board, false, s.h} : spot_at(s, s.p + d.l, s.h + 2 * d.t);
  if (spot.board == nullptr || !landable(s, spot, s.p + d.l)) return false;  // J2, L6
  const int to = make_square(x, y);
  const Piece occupant = spot.board->at(to);
  if (occupant.empty() || occupant.colour() == s.attacker || !occupant.royal()) return false;
  return visit(capture_of(s, from, piece, d, 1, s.p + d.l, spot, to, occupant));
}

// A slide (J7, J8; Q18, Q19): one unit per moved axis per step, over boards of
// the attacker's colour; every square crossed empty; void blocks unless
// sliders_cross_void; the slide ends on the first piece.
template <class Visit>
bool slide(const Scan& s, int from, Piece piece, Vec4 d, Visit& visit) {
  const bool physical = d.physical();
  // Q19: only a slide straight along T on its own square and timeline can meet
  // the piece's own earlier self.
  const bool past_self_passes = !s.rules.past_self_blocks && d.x == 0 && d.y == 0 && d.l == 0;
  int x = file_of(from);
  int y = rank_of(from);
  int l = s.p;
  int ht = s.h;
  for (int n = 1;; ++n) {
    x += d.x;
    y += d.y;
    l += d.l;
    ht += 2 * d.t;
    if (!s.size.contains(x, y)) return false;
    Spot spot{s.board, false, s.h};
    if (!physical) {
      if (l < s.min_timeline || l > s.max_timeline || ht < 0 || ht > s.max_half_turn) {
        return false;  // beyond every board
      }
      spot = spot_at(s, l, ht);
      if (spot.board == nullptr) {
        if (s.rules.sliders_cross_void) continue;  // Q18's dissent
        return false;                              // J8: void blocks
      }
    }
    const int to = make_square(x, y);
    const Piece occupant = spot.board->at(to);
    if (occupant.empty()) continue;
    if (occupant.colour() == s.attacker) {
      if (past_self_passes && occupant.kind() == piece.kind()) continue;
      return false;
    }
    if (!occupant.royal() || !landable(s, spot, l)) return false;
    return visit(capture_of(s, from, piece, d, n, l, spot, to, occupant));
  }
}

constexpr Vec4 v4(int x, int y, int l, int t) noexcept {
  return Vec4{static_cast<std::int8_t>(x), static_cast<std::int8_t>(y),
              static_cast<std::int8_t>(l), static_cast<std::int8_t>(t)};
}

// Every capture a piece can make (only royal ones reach the visitor).
template <class Visit>
bool piece_captures(const Scan& s, int square, Piece piece, Visit& visit) {
  switch (piece.kind()) {
    case PieceKind::Pawn:
    case PieceKind::Brawn: {
      // 4.6: forward rank and +-1 file; forward timeline and +-1 turn. 4.7: a
      // brawn also forward rank and +-1 turn, forward timeline and +-1 file,
      // and - Q14, the Guide's literal reading - the two forward axes
      // together. Forward moves never capture; en passant takes only a pawn
      // or a brawn, never a royal (S7).
      const int fy = geometry::forward_rank(s.attacker);
      const int fl = geometry::forward_timeline(s.attacker);
      const Vec4 pawn[] = {v4(-1, fy, 0, 0), v4(1, fy, 0, 0), v4(0, 0, fl, -1), v4(0, 0, fl, 1)};
      for (const Vec4 v : pawn) {
        if (leap(s, square, piece, v, visit)) return true;
      }
      if (piece.kind() == PieceKind::Brawn) {
        const Vec4 brawn[] = {v4(0, fy, 0, -1), v4(0, fy, 0, 1), v4(-1, 0, fl, 0),
                              v4(1, 0, fl, 0)};
        for (const Vec4 v : brawn) {
          if (leap(s, square, piece, v, visit)) return true;
        }
        if (s.rules.brawn_rank_timeline_capture && leap(s, square, piece, v4(0, fy, fl, 0), visit)) {
          return true;
        }
      }
      return false;
    }
    case PieceKind::Knight:
      for (const Vec4 v : geometry::knight_leaps()) {
        if (leap(s, square, piece, v, visit)) return true;
      }
      return false;
    case PieceKind::King:
    case PieceKind::CommonKing:
      // 4.4, 4.12: one step along any set of axes. Castling never captures.
      for (const Vec4 v : geometry::all_directions()) {
        if (leap(s, square, piece, v, visit)) return true;
      }
      return false;
    default:
      for (const Vec4 v : geometry::slider_directions(piece.kind())) {
        if (slide(s, square, piece, v, visit)) return true;
      }
      return false;
  }
}

// Whether `colour` has a royal piece anywhere (C13: without one it can be
// neither captured nor checked).
bool has_royal(const Multiverse& mv, Colour colour) {
  for (const TimelineRef& t : mv.timelines()) {
    for (const BoardRef& b : t->boards()) {
      if (b->royals(colour) != 0) return true;
    }
  }
  return false;
}

template <class Visit>
bool scan(const Multiverse& mv, const RuleOptions& rules, Colour attacker,
          std::span<const int> inactive, Visit& visit) {
  if (mv.empty() || !has_royal(mv, opposite(attacker))) return false;
  Scan s{mv, rules, inactive, attacker, mv.board_size(), mv.min_timeline(), mv.max_timeline(),
         mv.latest_half_turn(), 0, 0, nullptr};
  for (const TimelineRef& t : mv.timelines()) {
    // L5: the attacker's playable boards - the latest of each timeline, with
    // the attacker to move on it.
    if (to_move_at(t->last_half_turn()) != attacker) continue;
    s.p = t->position();
    s.h = t->last_half_turn();
    s.board = t->head().get();
    for (std::uint64_t bits = s.board->pieces(attacker); bits != 0; bits &= bits - 1) {
      const int square = std::countr_zero(bits);
      if (piece_captures(s, square, s.board->at(square), visit)) return true;
    }
  }
  return false;
}

std::string piece_word(PieceKind k) {
  switch (k) {
    case PieceKind::Pawn: return "pawn";
    case PieceKind::Knight: return "knight";
    case PieceKind::Bishop: return "bishop";
    case PieceKind::Rook: return "rook";
    case PieceKind::Queen: return "queen";
    case PieceKind::King: return "king";
    case PieceKind::Unicorn: return "unicorn";
    case PieceKind::Dragon: return "dragon";
    case PieceKind::Princess: return "princess";
    case PieceKind::Brawn: return "brawn";
    case PieceKind::CommonKing: return "common king";
    case PieceKind::RoyalQueen: return "royal queen";
  }
  return "piece";
}

std::string board_text(int timeline, int half_turn, bool even) {
  return "(" + timeline_label_string(timeline, even) + "T" + std::to_string(turn_of(half_turn)) +
         ")";
}

// The hooks of the engine's check ruling (t5d/state.hpp).
Verdict royals_safe_hook(const GameState& s) { return royals_safe(s) ? Verdict::Yes : Verdict::No; }
Verdict in_check_hook(const GameState& s) { return in_check(s) ? Verdict::Yes : Verdict::No; }
// C5, exhaustively (no budget: never Unknown - H37), by the hypercuboid
// detector (the faster near mate).
Verdict has_legal_action_hook(const GameState& s) {
  switch (detect_mate(s, Detector::Hypercuboid).verdict) {
    case Outcome::Ongoing: return Verdict::Yes;
    case Outcome::Checkmate:
    case Outcome::Stalemate: return Verdict::No;
    case Outcome::Unknown: break;
  }
  return Verdict::Unknown;
}

}  // namespace

// ---- captures -----------------------------------------------------------------------

std::vector<CapturePlace> capture_places(const RoyalCapture& c) {
  std::vector<CapturePlace> out;
  out.reserve(static_cast<std::size_t>(c.steps) + 1);
  const int x0 = file_of(c.from_square);
  const int y0 = rank_of(c.from_square);
  out.push_back(CapturePlace{c.from_timeline, c.from_half_turn, c.from_square});
  for (int i = 1; i < c.steps; ++i) {
    out.push_back(CapturePlace{c.from_timeline + i * c.step.l, c.from_half_turn + 2 * i * c.step.t,
                               make_square(x0 + i * c.step.x, y0 + i * c.step.y)});
  }
  out.push_back(CapturePlace{c.to_timeline, c.to_half_turn, c.to_square});
  return out;
}

bool royal_capture_exists(const Multiverse& multiverse, const RuleOptions& rules, Colour attacker,
                          std::span<const int> inactive_timelines) {
  const auto stop = [](const RoyalCapture&) { return true; };
  return scan(multiverse, rules, attacker, inactive_timelines, stop);
}

bool for_each_royal_capture(const Multiverse& multiverse, const RuleOptions& rules,
                            Colour attacker, std::span<const int> inactive_timelines,
                            const std::function<bool(const RoyalCapture&)>& visit) {
  const auto call = [&](const RoyalCapture& c) { return visit(c); };
  return scan(multiverse, rules, attacker, inactive_timelines, call);
}

std::optional<RoyalCapture> find_royal_capture(const Multiverse& multiverse,
                                               const RuleOptions& rules, Colour attacker,
                                               std::span<const int> inactive_timelines) {
  std::optional<RoyalCapture> found;
  const auto keep = [&](const RoyalCapture& c) {
    found = c;
    return true;
  };
  scan(multiverse, rules, attacker, inactive_timelines, keep);
  return found;
}

// ---- check status --------------------------------------------------------------------

std::vector<int> passed_timelines(const GameState& s) {
  std::vector<int> out;
  for (const TimelineRef& t : s.multiverse().timelines()) {
    const int p = t->position();
    // R1: ACTIVE boards IN THE PRESENT (T4) - the mover's, as the present is
    // on the mover's colour at a turn's start.
    if (!s.active(p) || t->last_half_turn() != s.present() || !s.playable_by_mover(p)) continue;
    // Q6's alternative defers a timeline only while its board is its owner's,
    // and every deferral of the mover's ends as the mover's turn begins: at a
    // turn's start no deferred timeline is the mover's to play.
    const auto& deferred = s.deferred_timelines();
    if (std::any_of(deferred.begin(), deferred.end(),
                    [p](const GameState::Deferral& d) { return d.timeline == p; })) {
      throw std::logic_error("passed_timelines: a deferred timeline of the mover's (Q6) - not a "
                             "turn's start");
    }
    out.push_back(p);
  }
  return out;
}

Multiverse passed_multiverse(const GameState& s) {
  Multiverse mv = s.multiverse();
  for (const int p : passed_timelines(s)) {
    // A null move (C6, "a passed board is read as advanced unchanged"): the same
    // pieces, flags and all; no en-passant square, as no double step was made.
    BoardBuilder b(*mv.head(p));
    b.set_en_passant(kNoSquare);
    mv.append_board(p, b.publish());
  }
  return mv;
}

bool in_check(const GameState& s) {
  const Colour opponent = opposite(s.to_move());
  if (!has_royal(s.multiverse(), s.to_move())) return false;  // C13
  return royal_capture_exists(passed_multiverse(s), s.rules(), opponent, s.inactive_timelines());
}

std::optional<RoyalCapture> check_witness(const GameState& s) {
  return find_royal_capture(passed_multiverse(s), s.rules(), opposite(s.to_move()),
                            s.inactive_timelines());
}

// ---- submission legality --------------------------------------------------------------

bool royals_safe(const GameState& s) {
  return !royal_capture_exists(s.multiverse(), s.rules(), opposite(s.to_move()),
                               s.inactive_timelines());
}

std::optional<RoyalCapture> submission_witness(const GameState& s) {
  return find_royal_capture(s.multiverse(), s.rules(), opposite(s.to_move()),
                            s.inactive_timelines());
}

std::string describe(const RoyalCapture& c, bool even) {
  return "the " + piece_word(c.attacker.kind()) + " on " +
         board_text(c.from_timeline, c.from_half_turn, even) + square_name(c.from_square) +
         " can capture the " + piece_word(c.royal.kind()) + " on " +
         board_text(c.to_timeline, c.to_half_turn, even) + square_name(c.to_square) +
         (c.branching ? " (a capture that branches)" : "");
}

// ---- the engine's check ruling ----------------------------------------------------------

CheckRuling standard_check_ruling() noexcept {
  CheckRuling r;
  r.royals_safe = &royals_safe_hook;
  r.in_check = &in_check_hook;
  r.has_legal_action = &has_legal_action_hook;
  return r;
}

}  // namespace t5d
