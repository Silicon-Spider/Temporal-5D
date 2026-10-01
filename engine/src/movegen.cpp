// Pseudo-legal move generation (see t5d/movegen.hpp). Rule IDs refer to the
// rules reference, docs/rules/README.md. Written from that reference and the
// design spec; no other implementation was consulted.

#include "t5d/movegen.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>

#include "t5d/geometry.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/rules.hpp"

namespace t5d {

namespace {

// ---- in-board attack patterns (for castling's S2 test) ---------------------------

struct Step2 {
  int x;
  int y;
};

constexpr std::array<Step2, 4> kOrthogonal = {{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
constexpr std::array<Step2, 4> kDiagonal = {{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}}};
constexpr std::array<Step2, 8> kKnight2D = {
    {{-2, -1}, {-2, 1}, {2, -1}, {2, 1}, {-1, -2}, {1, -2}, {-1, 2}, {1, 2}}};
constexpr std::array<Step2, 8> kKing2D = {
    {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};

constexpr bool slides_orthogonally(PieceKind k) noexcept {
  return k == PieceKind::Rook || k == PieceKind::Queen || k == PieceKind::RoyalQueen ||
         k == PieceKind::Princess;
}
constexpr bool slides_diagonally(PieceKind k) noexcept {
  return k == PieceKind::Bishop || k == PieceKind::Queen || k == PieceKind::RoyalQueen ||
         k == PieceKind::Princess;
}

// ---- the generator's context ---------------------------------------------------------

// Bounds of every board in the multiverse: a slide that leaves them can never
// meet a board again (timelines are contiguous; half-turns are >= 0). O(1):
// the multiverse keeps its latest half-turn.
struct Bounds {
  int min_timeline = 0;
  int max_timeline = -1;
  int max_half_turn = -1;
};

Bounds bounds_of(const Multiverse& mv) noexcept {
  Bounds b;
  if (mv.empty()) return b;
  b.min_timeline = mv.min_timeline();
  b.max_timeline = mv.max_timeline();
  b.max_half_turn = mv.latest_half_turn();
  return b;
}

// The board at a place, and whether it is a history board (L5: not the
// latest of its timeline).
struct Place {
  const Board* board = nullptr;  // nullptr: void (L6)
  bool history = false;
  int half_turn = 0;
};

struct Gen {
  // The state whose action the moves belong to; nullptr for append_moves_of
  // (no action: the optional-board castling option is not applied).
  const GameState* state;
  const Multiverse& mv;
  const RuleOptions& rules;
  MoveList& out;
  Colour mover;
  BoardSize size;
  Bounds bounds;
  // The inactive timelines, ascending: read only under
  // inactive_history_is_destination = false (Q20).
  std::span<const int> inactive;
  // Whether the ruleset leaves every landing, branching and cross-board
  // promotion option at its default (plain_rules): then the hot functions'
  // kPlain instantiation runs, which reads none of them.
  bool plain = true;
  // The source: timeline p's playable board, at half-turn h.
  int p = 0;
  int h = 0;
  const Board* board = nullptr;
};

inline Place place_at(const Gen& g, int timeline, int half_turn) noexcept {
  if (timeline < g.bounds.min_timeline || timeline > g.bounds.max_timeline) return Place{};
  const Timeline* t = g.mv.timeline(timeline);
  if (!t->contains(half_turn)) return Place{};
  return Place{t->board_at(half_turn), half_turn != t->last_half_turn(), half_turn};
}

inline Place place_after(const Gen& g, Vec4 v) noexcept {
  if (v.physical()) return Place{g.board, false, g.h};
  return place_at(g, g.p + v.l, g.h + 2 * v.t);
}

// The functions below that probe places are compiled twice. kPlain = true is
// for a ruleset that leaves the options read per probe at their defaults -
// turn_zero_reachable (Q8), inactive_history_is_destination (Q20),
// earlier_playable_jump_branches (J3) and brawn_cross_board_promotion (Q17) -
// as the standard rules do: that instantiation reads none of them, so they
// cost the standard rules nothing. kPlain = false reads them.
constexpr bool plain_rules(const RuleOptions& r) noexcept {
  return r.turn_zero_reachable && r.inactive_history_is_destination &&
         !r.earlier_playable_jump_branches && r.brawn_cross_board_promotion;
}

// Whether a piece may land on the board at `place` on `timeline`: any board
// but, under the ruleset's alternatives, a turn-zero history board (Q8) or a
// history board of an inactive timeline (Q20).
template <bool kPlain>
inline bool landable([[maybe_unused]] const Gen& g, [[maybe_unused]] const Place& place,
                     [[maybe_unused]] int timeline) noexcept {
  if constexpr (kPlain) {
    return true;
  } else {
    if (!place.history) return true;
    if (!g.rules.turn_zero_reachable && turn_of(place.half_turn) == 0) return false;
    if (!g.rules.inactive_history_is_destination &&
        std::binary_search(g.inactive.begin(), g.inactive.end(), timeline)) {
      return false;
    }
    return true;
  }
}

template <bool kPlain>
inline Move make_move(const Gen& g, int from, Piece piece, const Place& place, int timeline,
                      int to, Piece captured) noexcept {
  Move m;
  m.from_timeline = g.p;
  m.from_half_turn = g.h;
  m.to_timeline = timeline;
  m.to_half_turn = place.half_turn;
  m.from_square = static_cast<std::int8_t>(from);
  m.to_square = static_cast<std::int8_t>(to);
  m.piece = piece;
  m.captured = captured;
  // J3: a move branches iff it lands on a history board - or, under the Quick
  // Rules' literal reading (earlier_playable_jump_branches), at any earlier
  // turn than it started.
  bool branches = place.history;
  if constexpr (!kPlain) {
    branches = branches || (g.rules.earlier_playable_jump_branches && place.half_turn < g.h);
  }
  m.flags = static_cast<std::uint8_t>((captured.empty() ? 0 : Move::kCapture) |
                                      (branches ? Move::kBranching : 0));
  return m;
}

// A pawn's or brawn's move: one per promotion piece when it promotes (S8),
// else one.
inline void push_pawnlike(Gen& g, Move m, bool promotes) {
  if (!promotes) {
    g.out.push_back(m);
    return;
  }
  m.flags = static_cast<std::uint8_t>(m.flags | Move::kPromotion);
  for (const PieceKind k : PromotionSet::kOrder) {
    if (!g.rules.promotions.contains(k)) continue;
    m.promotion = k;
    g.out.push_back(m);
  }
}

// ---- sliders (J7, J8; Q18, Q19) --------------------------------------------------------

template <bool kPlain>
void slide(Gen& g, int from, Piece piece, Vec4 d) {
  const bool physical = d.physical();
  // Q19: only a slide straight back in time on its own square and timeline can
  // meet the piece's own earlier self.
  const bool past_self_passes = !g.rules.past_self_blocks && d.x == 0 && d.y == 0 && d.l == 0;
  int x = file_of(from);
  int y = rank_of(from);
  int l = g.p;
  int ht = g.h;
  for (;;) {
    x += d.x;
    y += d.y;
    l += d.l;
    ht += 2 * d.t;
    if (!g.size.contains(x, y)) return;
    Place place{g.board, false, g.h};
    if (!physical) {
      if (l < g.bounds.min_timeline || l > g.bounds.max_timeline || ht < 0 ||
          ht > g.bounds.max_half_turn) {
        return;  // beyond every board
      }
      place = place_at(g, l, ht);
      if (place.board == nullptr) {
        if (g.rules.sliders_cross_void) continue;  // Q18's dissent
        return;                                    // J8: void blocks
      }
    }
    const int to = make_square(x, y);
    const Piece occupant = place.board->at(to);
    if (occupant.empty()) {
      if (landable<kPlain>(g, place, l)) {
        g.out.push_back(make_move<kPlain>(g, from, piece, place, l, to, Piece{}));
      }
      continue;
    }
    if (occupant.colour() == g.mover) {
      if (past_self_passes && occupant.kind() == piece.kind()) continue;
      return;
    }
    if (landable<kPlain>(g, place, l)) {
      g.out.push_back(make_move<kPlain>(g, from, piece, place, l, to, occupant));
    }
    return;
  }
}

// ---- leapers and steppers (4.4, 4.5, 4.12) ------------------------------------------

template <bool kPlain>
void leap(Gen& g, int from, Piece piece, Vec4 d) {
  const int x = file_of(from) + d.x;
  const int y = rank_of(from) + d.y;
  if (!g.size.contains(x, y)) return;
  const Place place = place_after(g, d);
  // J2, L6: no landing on void.
  if (place.board == nullptr || !landable<kPlain>(g, place, g.p + d.l)) return;
  const int to = make_square(x, y);
  const Piece occupant = place.board->at(to);
  if (!occupant.empty() && occupant.colour() == g.mover) return;
  g.out.push_back(make_move<kPlain>(g, from, piece, place, g.p + d.l, to, occupant));
}

// ---- pawns and brawns (4.6, 4.7, S7, S8; Q14-Q17) ------------------------------------

inline bool takes_part_in_en_passant(const RuleOptions& r, PieceKind k) noexcept {
  return k == PieceKind::Pawn || (r.en_passant_brawns && k == PieceKind::Brawn);
}

template <bool kPlain>
void pawnlike(Gen& g, int from, Piece piece) {
  const bool brawn = piece.kind() == PieceKind::Brawn;
  const int fy = geometry::forward_rank(g.mover);
  const int fl = geometry::forward_timeline(g.mover);
  const int x0 = file_of(from);
  const int y0 = rank_of(from);
  const int last = g.mover == Colour::White ? g.size.height - 1 : 0;
  const bool can_promote = !brawn || g.rules.brawns_promote;
  // S8: on reaching the last rank (Q17: a brawn arriving there from another
  // board only under brawn_cross_board_promotion).
  const auto promotes = [&](int y, [[maybe_unused]] bool across) {
    if constexpr (kPlain) {
      return can_promote && y == last;
    } else {
      return can_promote && y == last &&
             (!across || !brawn || g.rules.brawn_cross_board_promotion);
    }
  };
  const int home = g.mover == Colour::White ? 1 : g.size.height - 2;
  const bool double_right = g.rules.double_step_right == DoubleStepRight::UnmovedFlag
                                ? piece.unmoved()
                                : y0 == home;

  // Forward along the rank, within the board: one step, or two (4.6).
  if (g.size.contains(x0, y0 + fy) && g.board->at(x0, y0 + fy).empty()) {
    const Place here{g.board, false, g.h};
    const int to1 = make_square(x0, y0 + fy);
    push_pawnlike(g, make_move<kPlain>(g, from, piece, here, g.p, to1, Piece{}),
                  promotes(y0 + fy, false));
    if (double_right && g.size.contains(x0, y0 + 2 * fy) &&
        g.board->at(x0, y0 + 2 * fy).empty()) {
      Move m = make_move<kPlain>(g, from, piece, here, g.p, make_square(x0, y0 + 2 * fy), Piece{});
      m.flags = static_cast<std::uint8_t>(m.flags | Move::kDoubleStep);
      push_pawnlike(g, m, promotes(y0 + 2 * fy, false));
    }
  }

  // Forward along the timeline: the same square and half-turn on the
  // forward-adjacent timeline (one step), or the next one (two steps, through
  // the in-between timeline's board). Branches when the board is history (J3).
  const Place one = place_at(g, g.p + fl, g.h);
  const bool one_empty = one.board != nullptr && one.board->at(from).empty();
  if (one_empty && landable<kPlain>(g, one, g.p + fl)) {
    push_pawnlike(g, make_move<kPlain>(g, from, piece, one, g.p + fl, from, Piece{}),
                  promotes(y0, true));
  }
  if (double_right && g.rules.timeline_double_step) {
    const bool passable = one.board != nullptr ? one_empty : g.rules.double_step_over_void;
    if (passable) {
      const Place two = place_at(g, g.p + 2 * fl, g.h);
      if (two.board != nullptr && landable<kPlain>(g, two, g.p + 2 * fl) &&
          two.board->at(from).empty()) {
        Move m = make_move<kPlain>(g, from, piece, two, g.p + 2 * fl, from, Piece{});
        m.flags = static_cast<std::uint8_t>(m.flags | Move::kDoubleStep);
        push_pawnlike(g, m, promotes(y0, true));
      }
    }
  }

  // Captures: only onto an enemy piece.
  const auto capture = [&](Vec4 v) {
    const int x = x0 + v.x;
    const int y = y0 + v.y;
    if (!g.size.contains(x, y)) return;
    const Place place = place_after(g, v);
    if (place.board == nullptr || !landable<kPlain>(g, place, g.p + v.l)) return;
    const int to = make_square(x, y);
    const Piece occupant = place.board->at(to);
    if (occupant.empty() || occupant.colour() == g.mover) return;
    push_pawnlike(g, make_move<kPlain>(g, from, piece, place, g.p + v.l, to, occupant),
                  promotes(y, !v.physical()));
  };
  const auto v4 = [](int x, int y, int l, int t) {
    return Vec4{static_cast<std::int8_t>(x), static_cast<std::int8_t>(y),
                static_cast<std::int8_t>(l), static_cast<std::int8_t>(t)};
  };
  // Pawn and brawn alike: forward rank and +-1 file; forward timeline and +-1 turn.
  capture(v4(-1, fy, 0, 0));
  capture(v4(1, fy, 0, 0));
  capture(v4(0, 0, fl, -1));
  capture(v4(0, 0, fl, 1));
  if (brawn) {
    // 4.7: forward rank and +-1 turn; forward timeline and +-1 file.
    capture(v4(0, fy, 0, -1));
    capture(v4(0, fy, 0, 1));
    capture(v4(-1, 0, fl, 0));
    capture(v4(1, 0, fl, 0));
    // Q14: forward rank and forward timeline together (the Guide's literal
    // reading, the default).
    if (g.rules.brawn_rank_timeline_capture) capture(v4(0, fy, fl, 0));
  }

  // En passant (S7): within this board only, onto the square a double step
  // passed on the move that created this board.
  const int ep = g.board->en_passant();
  if (ep != kNoSquare && takes_part_in_en_passant(g.rules, piece.kind()) &&
      rank_of(ep) == y0 + fy && (file_of(ep) == x0 - 1 || file_of(ep) == x0 + 1) &&
      g.board->at(ep).empty()) {
    const int victim_square = make_square(file_of(ep), rank_of(ep) - fy);
    const Piece victim = g.board->at(victim_square);
    if (!victim.empty() && victim.colour() != g.mover &&
        takes_part_in_en_passant(g.rules, victim.kind())) {
      Move m = make_move<kPlain>(g, from, piece, Place{g.board, false, g.h}, g.p, ep, victim);
      m.flags = static_cast<std::uint8_t>(m.flags | Move::kEnPassant);
      m.aux_square = static_cast<std::int8_t>(victim_square);
      push_pawnlike(g, m, promotes(rank_of(ep), false));
    }
  }
}

// ---- castling (S1-S5; Q21) ----------------------------------------------------------------

template <bool kPlain>
void castling(Gen& g, int from, Piece king) {
  // S4: only a royal king castles - never a common king (which, besides, never
  // carries the unmoved flag: 5DFEN tracks it on pawns, rooks, kings and
  // brawns only) and never a royal queen.
  if (king.kind() != PieceKind::King) return;
  if (!king.unmoved()) return;  // S2: the king has not moved
  if (!g.rules.castling_on_optional_boards && g.state != nullptr && !g.state->mandatory(g.p)) {
    return;
  }
  const int kx = file_of(from);
  const int ky = rank_of(from);
  if (!g.rules.castling_any_rank && ky != (g.mover == Colour::White ? 0 : g.size.height - 1)) {
    return;  // S5's alternative
  }
  const Colour enemy = opposite(g.mover);
  if (attacked_on_board(*g.board, from, enemy)) return;  // S2: the king's square
  const int min_distance = g.rules.castling_rook_two_away ? 2 : 3;
  for (const int dir : {-1, 1}) {
    // The rook: an unmoved rook of the mover's on the king's rank, at least
    // two squares away (S5); with the squares between required empty (S2), the
    // first piece in that direction.
    int rook_x = -1;
    for (int x = kx + dir; x >= 0 && x < g.size.width; x += dir) {
      const Piece q = g.board->at(x, ky);
      if (q.empty()) continue;
      if (q.kind() == PieceKind::Rook && q.colour() == g.mover && q.unmoved() &&
          (x - kx) * dir >= min_distance) {
        rook_x = x;
        break;
      }
      if (g.rules.castling_empty_between) break;  // a piece between king and rook
    }
    if (rook_x < 0) continue;
    // The king moves two squares towards the rook; the rook to the square the
    // king crossed (S1). A rook two squares away is landed on (S5, inferred).
    const int king_to = kx + 2 * dir;
    const int rook_to = kx + dir;
    if (!g.board->at(rook_to, ky).empty()) continue;
    if (king_to != rook_x && !g.board->at(king_to, ky).empty()) continue;
    // S2: the square the king crosses.
    if (attacked_on_board(*g.board, make_square(rook_to, ky), enemy)) continue;
    // S3's alternative: the king's destination.
    if (g.rules.castling_destination_test &&
        attacked_on_board(*g.board, make_square(king_to, ky), enemy)) {
      continue;
    }
    // Q21's alternative: the squares the rook crosses.
    if (g.rules.castling_rook_path_test) {
      bool attacked = false;
      for (int x = rook_x - dir; x != rook_to; x -= dir) {
        if (attacked_on_board(*g.board, make_square(x, ky), enemy)) attacked = true;
      }
      if (attacked) continue;
    }
    Move m = make_move<kPlain>(g, from, king, Place{g.board, false, g.h}, g.p,
                       make_square(king_to, ky), Piece{});
    m.flags = static_cast<std::uint8_t>(m.flags | Move::kCastle);
    m.aux_square = static_cast<std::int8_t>(make_square(rook_x, ky));
    g.out.push_back(m);
  }
}

// ---- dispatch ------------------------------------------------------------------------------

template <bool kPlain>
void piece_moves(Gen& g, int square, Piece piece) {
  switch (piece.kind()) {
    case PieceKind::Pawn:
    case PieceKind::Brawn:
      pawnlike<kPlain>(g, square, piece);
      break;
    case PieceKind::Knight:
      for (const Vec4 v : geometry::knight_leaps()) leap<kPlain>(g, square, piece, v);
      break;
    case PieceKind::King:
      for (const Vec4 v : geometry::all_directions()) leap<kPlain>(g, square, piece, v);
      castling<kPlain>(g, square, piece);
      break;
    case PieceKind::CommonKing:
      for (const Vec4 v : geometry::all_directions()) leap<kPlain>(g, square, piece, v);
      break;
    default:
      for (const Vec4 v : geometry::slider_directions(piece.kind())) {
        slide<kPlain>(g, square, piece, v);
      }
      break;
  }
}

void any_piece_moves(Gen& g, int square, Piece piece) {
  if (g.plain) {
    piece_moves<true>(g, square, piece);
  } else {
    piece_moves<false>(g, square, piece);
  }
}

// Fills `g`'s source fields for timeline p; false if p's playable board is not
// the mover's.
bool set_source(Gen& g, const Timeline* t) noexcept {
  if (t == nullptr || to_move_at(t->last_half_turn()) != g.mover) return false;
  g.p = t->position();
  g.h = t->last_half_turn();
  g.board = t->head().get();
  return true;
}

template <bool kPlain>
void board_moves_of(Gen& g) {
  for (std::uint64_t bits = g.board->pieces(g.mover); bits != 0; bits &= bits - 1) {
    const int square = std::countr_zero(bits);
    piece_moves<kPlain>(g, square, g.board->at(square));
  }
}

void board_moves(Gen& g) {
  if (g.plain) {
    board_moves_of<true>(g);
  } else {
    board_moves_of<false>(g);
  }
}

Gen make_gen(const GameState* state, const Multiverse& mv, const RuleOptions& rules,
             Colour mover, std::span<const int> inactive, MoveList& out) {
  return Gen{state, mv, rules, out, mover, mv.board_size(), bounds_of(mv), inactive,
             plain_rules(rules), 0, 0, nullptr};
}

Gen make_gen(const GameState& state, MoveList& out) {
  return make_gen(&state, state.multiverse(), state.rules(), state.to_move(),
                  state.inactive_timelines(), out);
}

}  // namespace

// ---- public ----------------------------------------------------------------------------------

void generate_moves(const GameState& state, MoveList& out) {
  out.clear();
  if (!state.may_move()) return;
  Gen g = make_gen(state, out);
  for (const TimelineRef& t : g.mv.timelines()) {
    if (set_source(g, t.get())) board_moves(g);
  }
}

void append_board_moves(const GameState& state, int timeline, MoveList& out) {
  if (!state.may_move()) return;
  Gen g = make_gen(state, out);
  if (set_source(g, g.mv.timeline(timeline))) board_moves(g);
}

void append_piece_moves(const GameState& state, int timeline, int square, MoveList& out) {
  if (!state.may_move()) return;
  Gen g = make_gen(state, out);
  if (!set_source(g, g.mv.timeline(timeline)) || !g.size.contains_square(square)) return;
  const Piece piece = g.board->at(square);
  if (piece.empty() || piece.colour() != g.mover) return;
  any_piece_moves(g, square, piece);
}

void append_moves_of(const Multiverse& multiverse, const RuleOptions& rules, Colour colour,
                     int timeline, MoveList& out, std::span<const int> inactive_timelines) {
  if (multiverse.empty()) return;
  Gen g = make_gen(nullptr, multiverse, rules, colour, inactive_timelines, out);
  if (set_source(g, multiverse.timeline(timeline))) board_moves(g);
}

bool is_pseudo_legal(const GameState& state, const Move& move, MoveList& scratch) {
  scratch.clear();
  append_piece_moves(state, move.from_timeline, move.from_square, scratch);
  return std::find(scratch.begin(), scratch.end(), move) != scratch.end();
}

bool is_pseudo_legal(const GameState& state, const Move& move) {
  MoveList scratch;
  scratch.reserve(128);
  return is_pseudo_legal(state, move, scratch);
}

std::size_t count_moves(const GameState& state, MoveList& scratch) {
  generate_moves(state, scratch);
  return scratch.size();
}

std::size_t count_moves(const GameState& state) {
  MoveList scratch;
  scratch.reserve(512);
  return count_moves(state, scratch);
}

bool attacked_on_board(const Board& board, int square, Colour by) {
  const int x0 = file_of(square);
  const int y0 = rank_of(square);
  const auto holds = [&](int x, int y, auto&& kind_ok) {
    const Piece q = board.at(x, y);  // empty off the board
    return !q.empty() && q.colour() == by && kind_ok(q.kind());
  };
  // Pawns and brawns capture one step forward (theirs) and one file sideways
  // within a board (4.6, 4.7), so an attacker stands one rank behind.
  const int fy = geometry::forward_rank(by);
  for (const int dx : {-1, 1}) {
    if (holds(x0 + dx, y0 - fy, [](PieceKind k) {
          return k == PieceKind::Pawn || k == PieceKind::Brawn;
        })) {
      return true;
    }
  }
  for (const Step2 s : kKnight2D) {
    if (holds(x0 + s.x, y0 + s.y, [](PieceKind k) { return k == PieceKind::Knight; })) return true;
  }
  for (const Step2 s : kKing2D) {
    if (holds(x0 + s.x, y0 + s.y, [](PieceKind k) {
          return k == PieceKind::King || k == PieceKind::CommonKing;
        })) {
      return true;
    }
  }
  const BoardSize size = board.size();
  const auto ray = [&](Step2 s, bool (*kind_ok)(PieceKind) noexcept) {
    for (int x = x0 + s.x, y = y0 + s.y; size.contains(x, y); x += s.x, y += s.y) {
      const Piece q = board.at(x, y);
      if (q.empty()) continue;
      return q.colour() == by && kind_ok(q.kind());
    }
    return false;
  };
  for (const Step2 s : kOrthogonal) {
    if (ray(s, &slides_orthogonally)) return true;
  }
  for (const Step2 s : kDiagonal) {
    if (ray(s, &slides_diagonally)) return true;
  }
  return false;
}

}  // namespace t5d
