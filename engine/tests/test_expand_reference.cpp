// expand() against a slow reference written here from the definitions, on
// random positions of the synthetic player (tools/synthetic.hpp) - several
// variants and board sizes, positions at the start of an action and in the
// middle of one, and several rulesets.
//
// The reference shares nothing with t5d/expand.hpp but the move list
// (generate_moves, itself cross-checked against per-piece generation): boards
// in the canonical order from Multiverse::for_each_board, W(k) and its bands
// from §10.4.3's definition, the lineage from the timelines, E(k) and
// src_extra(k) as sets, the move graph as a set, the features through the
// Piece API, and the defence and x-ray relations by a ray walker written from
// the rules reference - J7, J8, L4, 4.1-4.12, and the ruleset's alternatives
// that change what a ray reaches: Q8 (no turn-zero destination), Q14 (the
// brawn's rank-and-timeline capture), Q18 (sliders crossing void), Q19 (past
// selves passed), Q20 (no destination on an inactive timeline's history).

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "synthetic.hpp"
#include "t5d/expand.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/records.h"
#include "t5d/state.hpp"
#include "test_framework.hpp"

using namespace t5d;

namespace {

using Quad = std::array<std::int32_t, 4>;
using Quint = std::array<std::int32_t, 5>;

struct D4 {
  int x, y, l, t;
};

// The direction sets of the rules reference (4.0-4.12), written afresh: every
// non-zero vector of {-1, 0, 1}^4 moving along `axes` axes.
std::vector<D4> unit_vectors(int min_axes, int max_axes) {
  std::vector<D4> out;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int l = -1; l <= 1; ++l) {
        for (int t = -1; t <= 1; ++t) {
          const int axes = (x != 0) + (y != 0) + (l != 0) + (t != 0);
          if (axes >= min_axes && axes <= max_axes) out.push_back({x, y, l, t});
        }
      }
    }
  }
  return out;
}

std::vector<D4> knight_vectors() {
  std::vector<D4> out;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      for (const int sa : {-2, 2}) {
        for (const int sb : {-1, 1}) {
          int c[4] = {0, 0, 0, 0};
          c[a] = sa;
          c[b] = sb;
          out.push_back({c[0], c[1], c[2], c[3]});
        }
      }
    }
  }
  return out;
}

std::vector<D4> slider_vectors(PieceKind k) {
  switch (k) {
    case PieceKind::Rook: return unit_vectors(1, 1);
    case PieceKind::Bishop: return unit_vectors(2, 2);
    case PieceKind::Unicorn: return unit_vectors(3, 3);
    case PieceKind::Dragon: return unit_vectors(4, 4);
    case PieceKind::Princess: return unit_vectors(1, 2);
    case PieceKind::Queen:
    case PieceKind::RoyalQueen: return unit_vectors(1, 4);
    default: return {};
  }
}

// What the reference walker met, so that a pass under a ruleset is not an
// empty one.
struct WalkCoverage {
  long long past_self_passed = 0;  // Q19: a relation found beyond a past self
  long long void_crossed = 0;      // Q18: a relation found beyond void
  long long not_landable = 0;      // Q8, Q20: a relation dropped, its board no destination
};
WalkCoverage walk_coverage;

struct Reference {
  std::vector<std::pair<int, int>> boards;  // (p, h), canonical order
  std::map<std::pair<int, int>, int> index;
  std::vector<int> head, active, side, playable, parent;
  std::vector<std::uint8_t> piece, colour, moved;
  std::vector<MoveRecord> records;
  std::set<Quad> edges;
  std::set<Quint> relations;
  std::set<int> destinations, sources, origins;
  int origins_unknown = 0;
  // W(k) and its bands alone (W(k) without the origins), k = 0..kMaxK.
  static constexpr int kMaxK = 24;
  std::vector<std::set<int>> window, bands;
};

Reference reference(const GameState& s, bool attack, bool defence, bool xray) {
  Reference r;
  const Multiverse& mv = s.multiverse();
  const RuleOptions& rules = s.rules();
  const BoardSize size = mv.board_size();
  int min_h = 0;
  int max_h = 0;
  bool first = true;
  mv.for_each_board([&](int p, int h, const Board& b) {
    r.index[{p, h}] = static_cast<int>(r.boards.size());
    r.boards.emplace_back(p, h);
    const Timeline* t = mv.timeline(p);
    const bool is_head = h == t->last_half_turn();
    r.head.push_back(is_head ? 1 : 0);
    r.active.push_back(s.active(p) ? 1 : 0);
    r.side.push_back(h % 2);
    r.playable.push_back(is_head && to_move_at(h) == s.to_move() && s.may_move() ? 1 : 0);
    min_h = first ? h : std::min(min_h, h);
    max_h = first ? h : std::max(max_h, h);
    first = false;
    for (int rank = 0; rank < size.height; ++rank) {
      for (int file = 0; file < size.width; ++file) {
        const Piece q = b.at(file, rank);
        r.piece.push_back(static_cast<std::uint8_t>(q.empty() ? 0 : index_of(q.kind()) + 1));
        r.colour.push_back(static_cast<std::uint8_t>(q.empty() ? 0 : index_of(q.colour()) + 1));
        r.moved.push_back(static_cast<std::uint8_t>(
            !q.empty() && tracks_unmoved(q.kind()) && !q.unmoved() ? 1 : 0));
      }
    }
  });
  const auto cell = [&](int square) { return rank_of(square) * size.width + file_of(square); };
  const auto board_of = [&](int p, int h) { return r.index.at({p, h}); };

  // The lineage: a board's previous board on its timeline, or a timeline's
  // first board's branch origin (-1: an initial timeline, or not recorded).
  for (const auto& [p, h] : r.boards) {
    const Timeline* t = mv.timeline(p);
    if (h > t->first_half_turn()) {
      r.parent.push_back(board_of(p, h - 1));
    } else if (t->origin()) {
      r.parent.push_back(board_of(t->origin()->timeline, t->origin()->half_turn));
    } else {
      r.parent.push_back(-1);
    }
  }

  // W(k): every active timeline's head and the k boards behind it (its band),
  // and its branch origin (§10.4.3; the engine README's reading). An active
  // timeline created by branching whose origin is not recorded (5DFEN) is
  // counted.
  r.window.resize(Reference::kMaxK + 1);
  r.bands.resize(Reference::kMaxK + 1);
  for (const TimelineRef& t : mv.timelines()) {
    if (!s.active(t->position())) continue;
    if (t->origin()) {
      r.origins.insert(board_of(t->origin()->timeline, t->origin()->half_turn));
    } else if (t->creator()) {
      ++r.origins_unknown;
    }
  }
  for (int k = 0; k <= Reference::kMaxK; ++k) {
    const auto ki = static_cast<std::size_t>(k);
    for (const TimelineRef& t : mv.timelines()) {
      if (!s.active(t->position())) continue;
      const int from = std::max(t->first_half_turn(), t->last_half_turn() - k);
      for (int h = from; h <= t->last_half_turn(); ++h) {
        r.bands[ki].insert(board_of(t->position(), h));
      }
    }
    r.window[ki] = r.bands[ki];
    r.window[ki].insert(r.origins.begin(), r.origins.end());
  }

  MoveList moves;
  generate_moves(s, moves);
  for (const Move& m : moves) {
    MoveRecord rec{};
    rec.from_timeline = m.from_timeline;
    rec.from_half_turn = m.from_half_turn;
    rec.to_timeline = m.to_timeline;
    rec.to_half_turn = m.to_half_turn;
    rec.from_board = board_of(m.from_timeline, m.from_half_turn);
    rec.to_board = board_of(m.to_timeline, m.to_half_turn);
    rec.from_square = m.from_square;
    rec.to_square = m.to_square;
    rec.from_cell = static_cast<std::int8_t>(cell(m.from_square));
    rec.to_cell = static_cast<std::int8_t>(cell(m.to_square));
    rec.piece = static_cast<std::uint8_t>(index_of(m.piece.kind()) + 1);
    rec.captured =
        static_cast<std::uint8_t>(m.captured.empty() ? 0 : index_of(m.captured.kind()) + 1);
    rec.promotion = static_cast<std::uint8_t>(m.promotes() ? index_of(m.promotion) + 1 : 0);
    const bool travel = rec.to_board != rec.from_board;
    rec.flags = static_cast<std::uint8_t>(m.flags | (travel ? T5D_MOVE_TRAVEL : 0));
    r.records.push_back(rec);
    r.edges.insert({rec.from_board, rec.from_cell, rec.to_board, rec.to_cell});
    // §10.4.1: E is made of TRAVEL destinations; every move has a source.
    if (travel) r.destinations.insert(rec.to_board);
    r.sources.insert(rec.from_board);
    if (attack && m.capture()) {
      r.relations.insert({rec.from_board, rec.from_cell, rec.to_board,
                          m.en_passant() ? cell(m.aux_square) : rec.to_cell, kRelationAttack});
    }
  }

  // Defence and x-ray: walk from every piece of the mover's on the boards the
  // moves start from.
  if ((defence || xray) && s.may_move()) {
    const Colour mover = s.to_move();
    // A board's square: (exists, piece). Void and squares off the board do
    // not exist (L6).
    const auto piece_at = [&](int p, int h, int x, int y) -> std::pair<bool, Piece> {
      if (!size.contains(x, y)) return {false, Piece{}};
      const Board* b = mv.board_at(p, h);
      if (b == nullptr) return {false, Piece{}};
      return {true, b->at(x, y)};
    };
    // Whether a piece may land on the board (p, h): any board, but under Q8's
    // alternative not a turn-zero history board, and under Q20's not a history
    // board of an inactive timeline.
    const auto landable = [&](int p, int h) {
      if (h == mv.timeline(p)->last_half_turn()) return true;  // a head is no history
      if (!rules.turn_zero_reachable && turn_of(h) == 0) return false;
      if (!rules.inactive_history_is_destination && !s.active(p)) return false;
      return true;
    };
    for (const TimelineRef& t : mv.timelines()) {
      const int p = t->position();
      const int h = t->last_half_turn();
      if (to_move_at(h) != mover) continue;
      const Board& b = *t->head();
      const int from_board = board_of(p, h);
      for (int square = 0; square < kSquareCount; ++square) {
        const Piece q = b.at(square);
        if (q.empty() || q.colour() != mover) continue;
        const int x0 = file_of(square);
        const int y0 = rank_of(square);
        const auto record = [&](int tp, int th, int x, int y, int kind) {
          r.relations.insert(
              {from_board, cell(square), board_of(tp, th), y * size.width + x, kind});
        };
        for (const D4& d : slider_vectors(q.kind())) {
          // J7: any piece blocks - but under Q19's alternative a slide straight
          // back in time (along T alone) passes the piece's own past selves, of
          // its kind and colour; J8: void blocks - but under Q18's alternative a
          // slider passes it, never landing; L4: one turn is two half-turns.
          const bool along_time_only = d.x == 0 && d.y == 0 && d.l == 0;
          bool passed_self = false;
          bool crossed_void = false;
          int seen = 0;
          for (int n = 1;; ++n) {
            const int x = x0 + n * d.x;
            const int y = y0 + n * d.y;
            const int tp = p + n * d.l;
            const int th = h + 2 * n * d.t;
            if (!size.contains(x, y) || tp < mv.min_timeline() || tp > mv.max_timeline() ||
                th < min_h || th > max_h) {
              break;  // beyond every board
            }
            const auto [exists, there] = piece_at(tp, th, x, y);
            if (!exists) {
              if (!rules.sliders_cross_void) break;
              crossed_void = true;
              continue;
            }
            if (there.empty()) continue;
            if (!rules.past_self_blocks && along_time_only && there.colour() == mover &&
                there.kind() == q.kind()) {
              passed_self = true;
              continue;
            }
            const int kind = seen == 0 ? kRelationDefence : kRelationXRay;
            const bool wanted = seen == 0 ? defence && there.colour() == mover : xray;
            if (wanted) {
              if (landable(tp, th)) {
                record(tp, th, x, y, kind);
                walk_coverage.past_self_passed += passed_self ? 1 : 0;
                walk_coverage.void_crossed += crossed_void ? 1 : 0;
              } else {
                ++walk_coverage.not_landable;
              }
            }
            if (seen == 1 || !xray) break;
            seen = 1;
          }
        }
        if (!defence) continue;
        std::vector<D4> leaps;
        if (q.kind() == PieceKind::Knight) leaps = knight_vectors();
        if (q.kind() == PieceKind::King || q.kind() == PieceKind::CommonKing) {
          leaps = unit_vectors(1, 4);
        }
        if (q.kind() == PieceKind::Pawn || q.kind() == PieceKind::Brawn) {
          // 4.6, 4.7: forward rank +y for White, forward timeline -L for White.
          const int fy = mover == Colour::White ? 1 : -1;
          const int fl = mover == Colour::White ? -1 : 1;
          leaps = {{-1, fy, 0, 0}, {1, fy, 0, 0}, {0, 0, fl, -1}, {0, 0, fl, 1}};
          if (q.kind() == PieceKind::Brawn) {
            leaps.push_back({0, fy, 0, -1});
            leaps.push_back({0, fy, 0, 1});
            leaps.push_back({-1, 0, fl, 0});
            leaps.push_back({1, 0, fl, 0});
            // Q14: the Guide's literal reading (the default) adds forward rank
            // and forward timeline together.
            if (rules.brawn_rank_timeline_capture) leaps.push_back({0, fy, fl, 0});
          }
        }
        for (const D4& d : leaps) {
          const int tp = p + d.l;
          const int th = h + 2 * d.t;
          const auto [exists, there] = piece_at(tp, th, x0 + d.x, y0 + d.y);
          if (exists && !there.empty() && there.colour() == mover) {
            if (landable(tp, th)) {
              record(tp, th, x0 + d.x, y0 + d.y, kRelationDefence);
            } else {
              ++walk_coverage.not_landable;
            }
          }
        }
      }
    }
  }
  return r;
}

std::set<int> as_set(std::span<const std::int32_t> v) { return std::set<int>(v.begin(), v.end()); }

// Ascending, each board once.
bool strictly_ascending(std::span<const std::int32_t> v) {
  return std::adjacent_find(v.begin(), v.end(), [](std::int32_t a, std::int32_t b) {
           return a >= b;
         }) == v.end();
}

std::set<int> minus(const std::set<int>& a, const std::set<int>& b) {
  std::set<int> out;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(out, out.end()));
  return out;
}

// The least k with board i in sets[k], or kNoWindow.
std::int32_t least_k(const std::vector<std::set<int>>& sets, int i) {
  for (int k = 0; k <= Reference::kMaxK; ++k) {
    if (sets[static_cast<std::size_t>(k)].count(i) != 0) return k;
  }
  return kNoWindow;
}

int checked = 0;

// What the comparisons exercised, so that a pass is not an empty one.
struct Coverage {
  long long relations[8] = {};
  long long e_boards = 0;
  long long src_extra = 0;
  long long src_only = 0;
  long long branching = 0;
  long long inactive = 0;
  long long small_boards = 0;
  long long unknown_origins = 0;
  long long origins_on_inactive = 0;
  long long no_moves = 0;
};
Coverage coverage;

// The whole comparison, for one position.
void compare(const GameState& s, bool relations_on) {
  const Reference r = reference(s, relations_on, relations_on, relations_on);
  ExpandOptions o;
  o.attack = o.defence = o.xray = relations_on;
  for (const int k : {0, 3, 8}) {
    o.window_k = k;
    o.square_features = k == 3 ? SquareFeatures::Scope : SquareFeatures::All;
    Expansion x;
    expand(s, o, x);
    ++checked;

    // (f) boards
    REQUIRE_EQ(x.board_count(), r.boards.size());
    for (std::size_t i = 0; i < r.boards.size(); ++i) {
      CHECK_EQ(x.board_timelines()[i], r.boards[i].first);
      CHECK_EQ(x.board_half_turns()[i], r.boards[i].second);
      CHECK_EQ(static_cast<int>(x.board_is_head()[i]), r.head[i]);
      CHECK_EQ(static_cast<int>(x.board_active()[i]), r.active[i]);
      CHECK_EQ(static_cast<int>(x.board_side_to_move()[i]), r.side[i]);
      CHECK_EQ(static_cast<int>(x.board_playable()[i]), r.playable[i]);
      CHECK_EQ(static_cast<int>(x.board_parent()[i]), r.parent[i]);
      CHECK_EQ(x.board_index(r.boards[i].first, r.boards[i].second), static_cast<int>(i));
      const bool origin = r.origins.count(static_cast<int>(i)) != 0;
      CHECK_EQ(static_cast<int>(x.board_is_origin()[i]), origin ? 1 : 0);
      // The depths: the least k with the board in W(k), in its band.
      const std::int32_t depth = least_k(r.window, static_cast<int>(i));
      const std::int32_t band = least_k(r.bands, static_cast<int>(i));
      if (depth != kNoWindow || x.window_depth()[i] <= Reference::kMaxK) {
        CHECK_EQ(x.window_depth()[i], depth);
      }
      if (band != kNoWindow || x.band_depth()[i] <= Reference::kMaxK) {
        CHECK_EQ(x.band_depth()[i], band);
      }
      if (origin && r.active[i] == 0) ++coverage.origins_on_inactive;
      // One board alone, without the arrays: the same entries.
      const BoardInfo info = x.board_info(static_cast<int>(i));
      CHECK(info.timeline == r.boards[i].first && info.half_turn == r.boards[i].second &&
            info.head == (r.head[i] == 1) && info.active == (r.active[i] == 1) &&
            info.side_to_move == r.side[i] && info.playable == (r.playable[i] == 1) &&
            info.parent == r.parent[i] && info.origin == origin &&
            info.window_depth == x.window_depth()[i] && info.band_depth == x.band_depth()[i]);
    }
    CHECK_EQ(x.origin_count(), r.origins.size());
    CHECK_EQ(x.origins_unknown(), r.origins_unknown);
    // (f) squares
    const std::size_t cells = static_cast<std::size_t>(s.multiverse().board_size().square_count());
    REQUIRE_EQ(x.square_pieces().size(), x.feature_boards().size() * cells);
    for (std::size_t j = 0; j < x.feature_boards().size(); ++j) {
      const auto b = static_cast<std::size_t>(x.feature_boards()[j]);
      for (std::size_t c = 0; c < cells; ++c) {
        CHECK_EQ(x.square_pieces()[j * cells + c], r.piece[b * cells + c]);
        CHECK_EQ(x.square_colours()[j * cells + c], r.colour[b * cells + c]);
        CHECK_EQ(x.square_moved()[j * cells + c], r.moved[b * cells + c]);
      }
    }
    // (a) moves and records
    REQUIRE_EQ(x.move_records().size(), r.records.size());
    for (std::size_t i = 0; i < r.records.size(); ++i) {
      const MoveRecord a = x.move_records()[i];
      const MoveRecord& e = r.records[i];
      CHECK(a.from_timeline == e.from_timeline && a.from_half_turn == e.from_half_turn &&
            a.to_timeline == e.to_timeline && a.to_half_turn == e.to_half_turn &&
            a.from_board == e.from_board && a.to_board == e.to_board &&
            a.from_square == e.from_square && a.to_square == e.to_square &&
            a.from_cell == e.from_cell && a.to_cell == e.to_cell && a.piece == e.piece &&
            a.captured == e.captured && a.promotion == e.promotion && a.flags == e.flags);
    }
    // (b) the move graph: the same pairs, each once
    std::set<Quad> edges;
    const std::span<const std::int32_t> ev = x.move_edges();
    for (std::size_t i = 0; i < x.move_edge_count(); ++i) {
      edges.insert({ev[4 * i], ev[4 * i + 1], ev[4 * i + 2], ev[4 * i + 3]});
    }
    CHECK(edges == r.edges);
    CHECK_EQ(edges.size(), x.move_edge_count());
    // (c) W(k), (d) E(k), src_extra(k), src_only(k) - at the options' k and
    // others, against W(k) and against its bands alone
    const std::set<int> w = r.window[static_cast<std::size_t>(k)];
    CHECK(as_set(x.window()) == w);
    CHECK(as_set(x.e_boards()) == minus(r.destinations, w));
    CHECK(as_set(x.src_extra()) == minus(r.sources, w));
    CHECK(strictly_ascending(x.window()));
    CHECK(strictly_ascending(x.e_boards()));
    CHECK(strictly_ascending(x.src_extra()));
    CHECK(strictly_ascending(x.travel_destinations()));
    CHECK(strictly_ascending(x.source_boards()));
    CHECK(strictly_ascending(x.feature_boards()));
    CHECK(strictly_ascending(x.origins()));
    CHECK(as_set(x.travel_destinations()) == r.destinations);
    CHECK(as_set(x.source_boards()) == r.sources);
    for (const int kk : {0, 1, 4, 8, 16}) {
      for (const Window which : {Window::WithOrigins, Window::BandsOnly}) {
        const std::set<int>& wk = (which == Window::BandsOnly ? r.bands : r.window)
            [static_cast<std::size_t>(kk)];
        const std::set<int> e = minus(r.destinations, wk);
        const std::set<int> src = minus(r.sources, wk);
        CHECK_EQ(x.window_size(kk, which), wk.size());
        CHECK_EQ(x.e_count(kk, which), e.size());
        CHECK_EQ(x.src_extra_count(kk, which), src.size());
        CHECK_EQ(x.src_only_count(kk, which), minus(src, e).size());
        std::vector<std::int32_t> list;
        x.e_boards(kk, list, which);
        CHECK(as_set(list) == e && list.size() == e.size());
        x.window(kk, list, which);
        CHECK(as_set(list) == wk && strictly_ascending(list));
        x.src_extra(kk, list, which);
        CHECK(as_set(list) == src && list.size() == src.size());
      }
    }
    // (e) relations: the same, each once
    std::set<Quint> rel;
    const std::span<const std::int32_t> rv = x.relations();
    for (std::size_t i = 0; i < x.relation_count(); ++i) {
      rel.insert({rv[5 * i], rv[5 * i + 1], rv[5 * i + 2], rv[5 * i + 3], rv[5 * i + 4]});
    }
    CHECK(rel == r.relations);
    CHECK_EQ(rel.size(), x.relation_count());
    for (const Quint& q : rel) ++coverage.relations[q[4]];
    coverage.e_boards += static_cast<long long>(x.e_boards().size());
    coverage.src_extra += static_cast<long long>(x.src_extra().size());
    coverage.src_only += static_cast<long long>(x.src_only_count(k));
    for (const MoveRecord& m : x.move_records()) {
      coverage.branching += (m.flags & T5D_MOVE_BRANCHING) != 0;
    }
    coverage.inactive += x.active_timeline_count() < x.timeline_count() ? 1 : 0;
    coverage.small_boards += x.board_size().width < 8 ? 1 : 0;
    coverage.unknown_origins += x.origins_unknown() > 0 ? 1 : 0;
    coverage.no_moves += x.moves().empty() ? 1 : 0;
    // Scope features: exactly W + E + src_extra.
    if (o.square_features == SquareFeatures::Scope) {
      std::set<int> scope = w;
      for (const int b : r.destinations) scope.insert(b);
      for (const int b : r.sources) scope.insert(b);
      CHECK(as_set(x.feature_boards()) == scope);
    }
  }
}

// The moves generate_moves lists are those of every piece of every playable
// board, one by one (the reference's one shared input, cross-checked).
void check_moves_by_piece(const GameState& s) {
  MoveList all;
  generate_moves(s, all);
  MoveList by_piece;
  for (const TimelineRef& t : s.multiverse().timelines()) {
    for (int square = 0; square < kSquareCount; ++square) {
      append_piece_moves(s, t->position(), square, by_piece);
    }
  }
  CHECK(all == by_piece);
}

void run(const char* variant, std::uint64_t seed, int target_boards, int target_active,
         const RuleOptions& rules = {}, bool active_branches_only = true) {
  synthetic::Spec spec;
  spec.variant = variant;
  spec.rules = rules;
  spec.seed = seed;
  spec.target_boards = target_boards;
  spec.target_active = target_active;
  spec.branch_percent = 40;
  spec.active_branches_only = active_branches_only;
  const synthetic::Game g = synthetic::play(spec);
  REQUIRE(!g.actions.empty());
  GameState s = g.start;
  for (std::size_t a = 0; a < g.actions.size(); ++a) {
    if (a % 3 == 0) {
      compare(s, a % 2 == 0);
      check_moves_by_piece(s);
    }
    // Into the action: the position after its first move, mid-action.
    GameState mid = s;
    mid.apply(g.actions[a].front());
    if (a % 4 == 1) compare(mid, true);
    for (const Move& m : g.actions[a]) s.apply(m);
    s.submit(CheckRuling{});  // the synthetic games are pseudo-legal
  }
  compare(s, true);
}

}  // namespace

TEST(expand_agrees_with_the_reference_on_random_positions) {
  checked = 0;
  coverage = Coverage{};
  run("Standard", 11, 70, 4);
  run("Standard - Two Timelines", 12, 60, 4);  // an even game: labels -0, +0
  run("Standard - Turn Zero", 13, 50, 3);      // a T0 history board
  run("Misc - Timeline Battleground", 14, 60, 5);  // 5x5, three timelines, no origins
  run("Very Small - Open", 15, 50, 4);             // 4x4
  run("Focused - Just Brawns", 16, 50, 3);         // brawns
  run("Standard - Reversed Royalty", 17, 50, 3);   // royal queens, common kings
  // Branching whenever the dice say so: inactive timelines (T10), whose
  // boards have no band, whose playable boards are sources outside W.
  run("Standard", 18, 70, 0, {}, false);
  run("Standard - Two Timelines", 19, 60, 0, {}, false);
  CHECK(checked > 150);
  // Every part of the comparison had something to compare.
  CHECK(coverage.relations[kRelationAttack] > 100);
  CHECK(coverage.relations[kRelationDefence] > 100);
  CHECK(coverage.relations[kRelationXRay] > 100);
  CHECK(coverage.e_boards > 20);
  CHECK(coverage.branching > 100);
  CHECK(coverage.small_boards > 10);
  CHECK(coverage.unknown_origins > 5);
  CHECK(coverage.inactive > 10);
  CHECK(coverage.src_extra > 0);
  CHECK(coverage.src_only > 0);
  std::printf("  reference: %d expansions; relations %lld/%lld/%lld (attack/defence/x-ray); "
              "E boards %lld; src_extra %lld (src_only %lld); branching moves %lld; with inactive "
              "timelines %lld; small boards %lld; unknown origins %lld; origins on inactive "
              "timelines %lld; no moves %lld\n",
              checked, coverage.relations[kRelationAttack], coverage.relations[kRelationDefence],
              coverage.relations[kRelationXRay], coverage.e_boards, coverage.src_extra,
              coverage.src_only, coverage.branching, coverage.inactive, coverage.small_boards,
              coverage.unknown_origins, coverage.origins_on_inactive, coverage.no_moves);
}

TEST(expand_agrees_with_the_reference_when_moves_follow_the_shift) {
  // Q4's other reading: moves after the present has shifted, so positions in
  // mid-action where every board but the playable ones is done.
  RuleOptions r;
  r.moves_after_shift = true;
  checked = 0;
  run("Standard", 21, 60, 4, r);
  run("Standard - Two Timelines", 22, 50, 3, r);
  CHECK(checked > 40);
}

TEST(expand_agrees_with_the_reference_under_the_rulesets_alternatives) {
  // The relations under every alternative that changes what a ray or a leap
  // reaches - and the rest of the expansion with them: Q18 (void crossed),
  // Q19 (past selves passed), Q8 and Q20 (boards that are no destination;
  // inactive timelines to have some), Q14 (the brawn's capture), J3's literal
  // reading (Q20) with Q4's, and all of them at once.
  std::vector<std::pair<const char*, RuleOptions>> rulesets;
  {
    RuleOptions r;
    r.sliders_cross_void = true;
    rulesets.emplace_back("Q18", r);
  }
  {
    RuleOptions r;
    r.past_self_blocks = false;
    rulesets.emplace_back("Q19", r);
  }
  {
    RuleOptions r;
    r.turn_zero_reachable = false;
    r.inactive_history_is_destination = false;
    rulesets.emplace_back("Q8+Q20", r);
  }
  {
    RuleOptions r;
    r.brawn_rank_timeline_capture = false;
    r.earlier_playable_jump_branches = true;
    r.moves_after_shift = true;
    rulesets.emplace_back("Q14+J3+Q4", r);
  }
  {
    RuleOptions r;
    r.sliders_cross_void = true;
    r.past_self_blocks = false;
    r.turn_zero_reachable = false;
    r.inactive_history_is_destination = false;
    r.brawn_rank_timeline_capture = false;
    rulesets.emplace_back("all", r);
  }
  checked = 0;
  coverage = Coverage{};
  walk_coverage = WalkCoverage{};
  std::uint64_t seed = 60;
  for (const auto& [name, rules] : rulesets) {
    run("Standard - Turn Zero", seed++, 60, 0, rules, false);  // T0 boards, inactive timelines
    run("Focused - Just Brawns", seed++, 50, 0, rules, false);
    run("Standard", seed++, 70, 4, rules);
  }
  CHECK(checked > 200);
  CHECK(coverage.relations[kRelationDefence] > 100);
  CHECK(coverage.relations[kRelationXRay] > 100);
  CHECK(coverage.inactive > 10);
  // The alternatives were met, not just switched on.
  CHECK(walk_coverage.void_crossed > 0);
  CHECK(walk_coverage.past_self_passed > 0);
  CHECK(walk_coverage.not_landable > 0);
  std::printf("  rulesets: %d expansions; relations %lld/%lld/%lld; beyond void %lld, beyond a "
              "past self %lld, dropped as no destination %lld\n",
              checked, coverage.relations[kRelationAttack], coverage.relations[kRelationDefence],
              coverage.relations[kRelationXRay], walk_coverage.void_crossed,
              walk_coverage.past_self_passed, walk_coverage.not_landable);
}

TEST(expand_relations_never_change_the_moves_under_any_ruleset) {
  // Collecting defence and x-ray in the generator's pass (its kRel
  // instantiation) must give exactly generate_moves' list - under the
  // standard rules and under rulesets that take the generator's general path
  // (landing, branching and promotion options read per probe), past selves
  // passing (Q19) and slides crossing void (Q18) - and each relation once.
  std::vector<RuleOptions> rulesets(1);
  {
    RuleOptions r;
    r.sliders_cross_void = true;
    r.past_self_blocks = false;
    r.turn_zero_reachable = false;
    r.inactive_history_is_destination = false;
    rulesets.push_back(r);
  }
  {
    RuleOptions r;
    r.earlier_playable_jump_branches = true;
    r.brawn_cross_board_promotion = false;
    r.moves_after_shift = true;
    rulesets.push_back(r);
  }
  int positions = 0;
  for (std::size_t i = 0; i < rulesets.size(); ++i) {
    for (const char* variant : {"Standard - Turn Zero", "Focused - Just Brawns"}) {
      synthetic::Spec spec;
      spec.variant = variant;
      spec.rules = rulesets[i];
      spec.seed = 40 + i;
      spec.target_boards = 50;
      spec.target_active = 3;
      spec.branch_percent = 40;
      spec.active_branches_only = false;  // inactive timelines too (Q20's option)
      const synthetic::Game g = synthetic::play(spec);
      GameState s = g.start;
      MoveList plain;
      MoveList with;
      RelationList relations;
      for (const std::vector<Move>& action : g.actions) {
        for (const Move& m : action) {
          generate_moves(s, plain);
          generate_moves(s, with, relations, RelationKinds{true, true});
          CHECK(plain == with);
          std::set<std::array<int, 7>> distinct;
          for (const GeometricRelation& r : relations) {
            distinct.insert({r.from_timeline, r.from_half_turn, r.from_square, r.to_timeline,
                             r.to_half_turn, r.to_square, static_cast<int>(r.kind)});
          }
          CHECK_EQ(distinct.size(), relations.size());
          ++positions;
          s.apply(m);
        }
        s.submit(CheckRuling{});
      }
    }
  }
  CHECK(positions > 100);
}
