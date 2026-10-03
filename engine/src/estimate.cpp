// The estimate channel (see t5d/estimate.hpp). Rule IDs refer to the
// rulebook, docs/rules/README.md. Negamax alpha-beta, stand-pat, MVV-LVA and
// delta pruning are the textbook quiescence techniques (Chess Programming
// Wiki), written afresh. The captures come from all-capture maps
// (t5d/attacks.hpp), advanced move by move: each local move rescans only the
// boards it made and the lines that stopped where they stand. The threats and
// local evasions are the generator's own moves (append_moves_of). A move is
// applied to boards as the state machine applies them (t5d/state.hpp), without
// its bookkeeping.

#include "t5d/estimate.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <utility>
#include <vector>

#include "t5d/attacks.hpp"
#include "t5d/board.hpp"
#include "t5d/geometry.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"

namespace t5d {

namespace {

constexpr int kInfinity = 1'000'000;

int board_material(const Board& b, Colour c, const PieceValues& v) {
  int total = 0;
  for (int k = 0; k < kPieceKindCount; ++k) {
    const int val = v.value[static_cast<std::size_t>(k)];
    if (val == 0) continue;
    total += val * std::popcount(b.pieces(c, static_cast<PieceKind>(k)));
  }
  return total;
}

// The local position: boards, the material on every timeline's latest board,
// each side's captures, and the boards the last move created (the opponent
// must answer on them or leave them passed - so that a threat there carries).
struct Local {
  explicit Local(const Multiverse& m) : mv(m) {}
  Multiverse mv;
  std::array<int, 2> material{};
  std::array<RoyalAttackMap, 2> captures;  // by colour
  std::array<bool, 2> current{};           // captures[c] describes mv (else advance it first)
  std::array<int, 2> fresh{};              // timelines whose head the last move created
  int fresh_count = 0;
};

class Search {
 public:
  Search(const GameState& s, const EstimateOptions& o, EstimateResult& out)
      : o_(o), rules_(s.rules()), inactive_(s.inactive_timelines()), out_(out) {}

  Local make(const Multiverse& mv) {
    Local l(mv);
    for (const TimelineRef& t : mv.timelines()) {
      for (const Colour c : {Colour::White, Colour::Black}) {
        l.material[static_cast<std::size_t>(index_of(c))] += board_material(*t->head(), c, o_.values);
      }
    }
    // The maps are built when a node first needs them (captures_of).
    return l;
  }

  int run(Local& root, Colour side) { return qs(root, side, -kInfinity, kInfinity, 0); }

 private:
  int value(Piece p) const { return p.empty() ? 0 : o_.values.of(p.kind()); }
  // Work: a board scanned (a map's source, a board's moves generated) is a
  // node, as a position searched is.
  void scanned(std::size_t boards) {
    out_.boards_scanned += boards;
    out_.nodes += boards;
  }
  long mvv_lva(const Move& m) const {
    return static_cast<long>(value(m.captured)) * 64 - (m.piece.royal() ? 0 : value(m.piece) / 16);
  }

  static Piece arrived(const Move& m) {
    return m.promotes() ? Piece(m.promotion, m.piece.colour()) : m.piece.as_moved();
  }

  // Whether a royal of `defender` on `b` is attacked from within `b` by the
  // other colour (movegen's attacked_on_board: the board's own lines).
  static bool royal_attacked(const Board& b, Colour defender) {
    for (std::uint64_t r = b.royals(defender); r != 0; r &= r - 1) {
      if (attacked_on_board(b, std::countr_zero(r), opposite(defender))) return true;
    }
    return false;
  }

  // A capture of the map as the generator's move: flags, and the promotion
  // (S8; Q17 for brawns) to the first piece of the ruleset's set.
  Move as_move(const RoyalCapture& c, int height) const {
    Move m;
    m.from_timeline = c.from_timeline;
    m.from_half_turn = c.from_half_turn;
    m.to_timeline = c.to_timeline;
    m.to_half_turn = c.to_half_turn;
    m.from_square = static_cast<std::int8_t>(c.from_square);
    m.to_square = static_cast<std::int8_t>(c.to_square);
    m.piece = c.attacker;
    m.captured = c.royal;  // all-capture maps: the piece taken
    m.flags = static_cast<std::uint8_t>(Move::kCapture | (c.branching ? Move::kBranching : 0));
    const PieceKind k = c.attacker.kind();
    const bool physical = c.step.physical();
    const bool can_promote =
        k == PieceKind::Pawn || (k == PieceKind::Brawn && rules_.brawns_promote &&
                                 (physical || rules_.brawn_cross_board_promotion));
    const int last = c.attacker.colour() == Colour::White ? height - 1 : 0;
    if (can_promote && rank_of(c.to_square) == last) {
      for (const PieceKind p : PromotionSet::kOrder) {
        if (!rules_.promotions.contains(p)) continue;
        m.flags = static_cast<std::uint8_t>(m.flags | Move::kPromotion);
        m.promotion = p;
        break;
      }
    }
    return m;
  }

  // Replaces timeline p's head by `board` (appended), keeping the material.
  void append(Local& pos, int p, BoardRef board) {
    const Board& old = *pos.mv.head(p);
    for (const Colour c : {Colour::White, Colour::Black}) {
      int& m = pos.material[static_cast<std::size_t>(index_of(c))];
      m += board_material(*board, c, o_.values) - board_material(old, c, o_.values);
    }
    pos.mv.append_board(p, std::move(board));
  }

  // The move's boards (T2, J4, J5), as the state machine builds them; a jump
  // onto a board no longer the head of its timeline branches. Then the
  // boards the opponent's last move created and this move did not play are
  // passed (advanced unchanged): what was threatened there carries. Then
  // both sides' captures, advanced.
  void play(Local& pos, const Move& m) {
    Multiverse& mv = pos.mv;
    const Colour mover = m.piece.colour();
    const Board& src = *mv.timeline(m.from_timeline)->head();
    const Piece placed = arrived(m);
    std::array<int, 2> created{m.from_timeline, 0};
    int n = 1;
    if (m.physical()) {
      BoardBuilder b(src);
      b.set_en_passant(kNoSquare);
      if (m.castle()) {
        const Piece rook = src.at(m.aux_square);
        const int dir = file_of(m.to_square) > file_of(m.from_square) ? 1 : -1;
        b.remove(m.from_square);
        b.remove(m.aux_square);
        b.put(m.to_square, m.piece.as_moved());
        b.put(make_square(file_of(m.from_square) + dir, rank_of(m.from_square)), rook.as_moved());
      } else {
        if (m.en_passant()) b.remove(m.aux_square);
        b.remove(m.from_square);
        b.put(m.to_square, placed);
      }
      append(pos, m.from_timeline, b.publish());
    } else {
      const Timeline* dt = mv.timeline(m.to_timeline);
      const Board& dst = *dt->board_at(m.to_half_turn);
      const bool branch = m.to_half_turn != dt->last_half_turn() ||
                          (rules_.earlier_playable_jump_branches && m.to_half_turn < m.from_half_turn);
      BoardBuilder s(src);
      s.set_en_passant(kNoSquare);
      s.remove(m.from_square);
      BoardBuilder d(dst);
      d.set_en_passant(kNoSquare);
      d.put(m.to_square, placed);
      BoardRef arrival = d.publish();
      append(pos, m.from_timeline, s.publish());
      if (!branch) {
        append(pos, m.to_timeline, std::move(arrival));
        created[1] = m.to_timeline;
      } else {
        NewTimeline nt;
        nt.position = mover == Colour::White ? mv.max_timeline() + 1 : mv.min_timeline() - 1;
        nt.first_half_turn = m.to_half_turn + 1;
        for (const Colour c : {Colour::White, Colour::Black}) {
          pos.material[static_cast<std::size_t>(index_of(c))] += board_material(*arrival, c, o_.values);
        }
        nt.boards.push_back(std::move(arrival));
        nt.creator = mover;
        nt.creation_order = mv.created_by(mover) + 1;
        created[1] = nt.position;
        mv.add_timeline(std::move(nt));
      }
      n = 2;
    }
    for (int i = 0; i < pos.fresh_count; ++i) {
      const int p = pos.fresh[static_cast<std::size_t>(i)];
      const Timeline* t = mv.timeline(p);
      if (t == nullptr || to_move_at(t->last_half_turn()) != mover) continue;  // played
      BoardBuilder pass(*t->head());
      pass.set_en_passant(kNoSquare);
      append(pos, p, pass.publish());
    }
    pos.fresh = created;
    pos.fresh_count = n;
    pos.current = {false, false};  // advanced when a node needs them
  }

  // The side's captures in `pos`, brought up to date (from wherever they were
  // last: the local multiverse only grows, so this is incremental).
  const RoyalAttackMap& captures_of(Local& pos, std::size_t side) {
    if (!pos.current[side]) {
      RoyalAttackMap& map = pos.captures[side];
      if (map.empty()) {
        map = RoyalAttackMap(pos.mv, rules_, side == 0 ? Colour::White : Colour::Black, inactive_,
                             CaptureTargets::All);
        scanned(map.source_count());
      } else {
        AttackMapStats stats;
        map = map.advanced(pos.mv, inactive_, &stats);
        scanned(stats.sources_scanned);
      }
      pos.current[side] = true;
    }
    return pos.captures[side];
  }

  // Whether a move within board `b` could check an enemy royal there - by
  // the moved piece's own pattern from its destination, or by uncovering a
  // line through its origin - ignoring what stands between (a cheap filter
  // before the exact test).
  static bool could_check(const Board& b, const Move& m, Piece arriving, Colour side) {
    const int ax = file_of(m.from_square);
    const int ay = rank_of(m.from_square);
    const int bx = file_of(m.to_square);
    const int by = rank_of(m.to_square);
    const PieceKind k = arriving.kind();
    const bool orth = k == PieceKind::Rook || k == PieceKind::Queen || k == PieceKind::RoyalQueen ||
                      k == PieceKind::Princess;
    const bool diag = k == PieceKind::Bishop || k == PieceKind::Queen || k == PieceKind::RoyalQueen ||
                      k == PieceKind::Princess;
    for (std::uint64_t r = b.royals(opposite(side)); r != 0; r &= r - 1) {
      const int sq = std::countr_zero(r);
      const int dx = file_of(sq) - bx;
      const int dy = rank_of(sq) - by;
      const int adx = dx < 0 ? -dx : dx;
      const int ady = dy < 0 ? -dy : dy;
      if (k == PieceKind::Knight && ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))) return true;
      if ((k == PieceKind::King || k == PieceKind::CommonKing) && adx <= 1 && ady <= 1) return true;
      if ((k == PieceKind::Pawn || k == PieceKind::Brawn) && adx == 1 && dy == geometry::forward_rank(side)) {
        return true;
      }
      if (orth && (dx == 0 || dy == 0)) return true;
      if (diag && adx == ady) return true;
      // A line uncovered through the origin.
      const int ox = file_of(sq) - ax;
      const int oy = rank_of(sq) - ay;
      if (ox == 0 || oy == 0 || ox == oy || ox == -oy) return true;
    }
    return false;
  }

  int qs(Local& pos, Colour side, int alpha, int beta, int ply) {
    ++out_.nodes;
    ++out_.positions;
    out_.depth = std::max(out_.depth, ply);
    const std::size_t me = static_cast<std::size_t>(index_of(side));
    const int eval = pos.material[me] - pos.material[1 - me];
    if (out_.nodes > o_.max_nodes) {
      out_.capped = true;
      return eval;
    }
    // A royal capture wins at once.
    const std::vector<RoyalCapture> caps = captures_of(pos, me).captures();
    for (const RoyalCapture& c : caps) {
      if (c.royal.royal()) return o_.royal_value - ply;
    }
    // A royal attacked on a board the opponent's last move created: a local
    // check, answered on that board or the royal falls.
    std::vector<int> checked;
    for (int i = 0; i < pos.fresh_count; ++i) {
      const int p = pos.fresh[static_cast<std::size_t>(i)];
      const Timeline* t = pos.mv.timeline(p);
      if (t == nullptr || to_move_at(t->last_half_turn()) != side) continue;
      if (royal_attacked(*t->head(), side)) checked.push_back(p);
    }
    const bool in_check = !checked.empty();
    int best = -kInfinity;
    if (!in_check) {
      best = eval;
      if (ply >= o_.max_plies || eval >= beta) return eval;
      alpha = std::max(alpha, eval);
    } else if (ply >= o_.max_plies) {
      return eval;
    }
    // Captures (MVV-LVA), then threats in the first plies, then the local
    // evasions in check.
    const int height = pos.mv.board_size().height;
    std::vector<Move> order;
    order.reserve(caps.size() + 8);
    for (const RoyalCapture& c : caps) order.push_back(as_move(c, height));
    std::stable_sort(order.begin(), order.end(),
                     [&](const Move& a, const Move& b) { return mvv_lva(a) > mvv_lva(b); });
    const std::size_t captures = order.size();
    if (ply < o_.threat_plies || in_check) {
      MoveList moves;
      for (const TimelineRef& t : pos.mv.timelines()) {
        if (to_move_at(t->last_half_turn()) != side) continue;
        const bool evasions = std::find(checked.begin(), checked.end(), t->position()) != checked.end();
        if (!evasions && ply >= o_.threat_plies) continue;
        moves.clear();
        append_moves_of(pos.mv, rules_, side, t->position(), moves, inactive_);
        scanned(1);
        const Board& here = *t->head();
        for (const Move& m : moves) {
          if (m.capture()) continue;  // among the captures already
          if (evasions) {
            order.push_back(m);
            continue;
          }
          // A threat: the board this move creates attacks an enemy royal on it.
          if (!m.physical() || m.castle() || here.royals(opposite(side)) == 0) continue;
          if (!could_check(here, m, arrived(m), side)) continue;
          BoardBuilder b(here);
          b.remove(m.from_square);
          b.put(m.to_square, arrived(m));
          const BoardRef created = b.publish();
          if (royal_attacked(*created, opposite(side))) order.push_back(m);
        }
      }
    }
    for (std::size_t i = 0; i < order.size(); ++i) {
      const Move& m = order[i];
      // Delta pruning (estimate channel only): a capture that cannot lift the
      // score to alpha.
      if (!in_check && i < captures && eval + value(m.captured) + o_.delta_margin <= alpha) continue;
      Local child = pos;
      play(child, m);
      const int score = -qs(child, opposite(side), -beta, -alpha, ply + 1);
      if (score > best) {
        best = score;
        if (ply == 0 && score > eval) out_.best = m;
      }
      alpha = std::max(alpha, score);
      if (alpha >= beta || out_.capped) break;
    }
    if (best == -kInfinity) return -(o_.royal_value - ply - 1);  // in check, no answer
    return best;
  }

  const EstimateOptions& o_;
  const RuleOptions& rules_;
  std::vector<int> inactive_;
  EstimateResult& out_;
};

}  // namespace

EstimateResult estimate(const GameState& position, const EstimateOptions& options) {
  EstimateResult out;
  Search search(position, options, out);
  Local root = search.make(position.multiverse());
  const Colour side = position.to_move();
  const std::size_t me = static_cast<std::size_t>(index_of(side));
  out.static_eval = root.material[me] - root.material[1 - me];
  out.score = search.run(root, side);
  const int swing = out.score - out.static_eval;
  out.live = swing >= options.live_margin || swing <= -options.live_margin;
  return out;
}

}  // namespace t5d
