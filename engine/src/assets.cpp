// The superseded-board asset map (see t5d/assets.hpp). Written from the
// design spec (11.2 rung 4) and the rulebook; the quiescence is the textbook
// one (stand-pat, captures by MVV-LVA, every move in check).

#include "t5d/assets.hpp"

#include <algorithm>
#include <exception>
#include <utility>

#include "t5d/check.hpp"
#include "t5d/movegen.hpp"

namespace t5d {

namespace {

int material(const Board& b, Colour c, const PieceValues& values) {
  int sum = 0;
  for (int sq = 0; sq < kSquareCount; ++sq) {  // the 8x8 grid; off-board squares are empty
    const Piece p = b.at(sq);
    if (p.empty() || p.colour() != c || p.royal()) continue;
    sum += values.of(p.kind());
  }
  return sum;
}

struct Quiescence {
  const AssetOptions& o;
  std::uint64_t nodes = 0;

  int value(Piece p) const { return p.empty() ? 0 : o.values.of(p.kind()); }

  // The one-board position `s`, its side to move's view.
  int search(const GameState& s, int alpha, int beta, int ply) {
    ++nodes;
    const Colour me = s.to_move();
    const Board& b = *s.multiverse().timelines().front()->head();
    const int stand = material(b, me, o.values) - material(b, opposite(me), o.values);
    MoveList moves;
    generate_moves(s, moves);
    for (const Move& m : moves) {
      if (m.capture() && m.captured.royal()) return o.royal_value;
    }
    if (nodes >= o.qsearch_nodes || ply >= o.qsearch_plies) return stand;
    const bool check = in_check(s);
    if (!check) {
      if (stand >= beta) return stand;
      alpha = std::max(alpha, stand);
    }
    std::vector<const Move*> candidates;
    for (const Move& m : moves) {
      if (check || m.capture()) candidates.push_back(&m);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [&](const Move* x, const Move* y) {
      const int vx = value(x->captured) * 16 - value(x->piece) / 64;
      const int vy = value(y->captured) * 16 - value(y->piece) / 64;
      return vx > vy;
    });
    int best = check ? -o.royal_value : stand;
    for (const Move* m : candidates) {
      if (nodes >= o.qsearch_nodes) break;
      GameState t = s;
      t.apply(*m);
      if (!t.obligations_met()) continue;
      t.submit(CheckRuling::none());
      const int v = -search(t, -beta, -alpha, ply + 1);
      if (v > best) best = v;
      if (v > alpha) alpha = v;
      if (alpha >= beta) break;
    }
    return best;
  }
};

}  // namespace

BoardAsset assess_board(const BoardRef& board, int half_turn, BoardSize size, bool even, std::uint32_t ruleset,
                        const RuleOptions& rules, const AssetOptions& options) {
  BoardAsset a;
  a.traveller = to_move_at(half_turn);
  a.material = material(*board, a.traveller, options.values) -
               material(*board, opposite(a.traveller), options.values);
  a.value = a.material;
  try {
    // The board alone: one timeline, the traveller to move on it.
    Multiverse mv(size, even, ruleset);
    mv.set_to_move(a.traveller);
    NewTimeline t;
    t.position = 0;
    t.first_half_turn = half_turn;
    t.boards.push_back(board);
    mv.add_timeline(std::move(t));
    const GameState s(std::move(mv), rules);
    Quiescence q{options};
    a.value = q.search(s, -options.royal_value - 1, options.royal_value + 1, 0);
    a.nodes = q.nodes;
    MoveList moves;
    generate_moves(s, moves);
    a.royal_capture = std::any_of(moves.begin(), moves.end(),
                                  [](const Move& m) { return m.capture() && m.captured.royal(); });
  } catch (const std::exception&) {
    // A board the one-board position cannot hold: its material alone.
  }
  a.flagged = a.value >= options.flag_margin;
  if (a.flagged) {
    const double scale = static_cast<double>(std::max(1, options.boost_scale));
    a.boost = options.max_boost * std::min(1.0, static_cast<double>(a.value) / scale);
  }
  return a;
}

AssetMap::AssetMap(const AssetOptions& options, const RuleOptions& rules) : options_(options), rules_(rules) {}

const BoardAsset& AssetMap::lookup(const GameState& state, const BoardRef& board, int half_turn) {
  const Colour traveller = to_move_at(half_turn);
  const std::uint64_t key = board->content_hash() * 2 + static_cast<std::uint64_t>(index_of(traveller));
  const auto range = index_.equal_range(key);
  for (auto it = range.first; it != range.second; ++it) {
    const Entry& e = entries_[it->second];
    if (e.traveller == traveller && (e.board == board || *e.board == *board)) return e.asset;
  }
  const Multiverse& mv = state.multiverse();
  ++assessments_;
  entries_.push_back(Entry{board, traveller,
                           assess_board(board, half_turn, mv.board_size(), mv.even(), mv.ruleset(), rules_, options_)});
  index_.emplace(key, entries_.size() - 1);
  return entries_.back().asset;
}

std::size_t AssetMap::refresh(const GameState& state) {
  const std::uint64_t before = assessments_;
  for (const TimelineRef& t : state.multiverse().timelines()) {
    const int last = t->last_half_turn();
    for (int h = t->first_half_turn(); h < last; ++h) {  // every board but the head: superseded
      const BoardRef* b = t->board_ref_at(h);
      if (b != nullptr && *b) lookup(state, *b, h);
    }
  }
  return static_cast<std::size_t>(assessments_ - before);
}

const BoardAsset* AssetMap::asset(const GameState& state, int timeline, int half_turn) {
  const Timeline* t = state.multiverse().timeline(timeline);
  if (t == nullptr || half_turn >= t->last_half_turn()) return nullptr;
  const BoardRef* b = t->board_ref_at(half_turn);
  if (b == nullptr || !*b) return nullptr;
  return &lookup(state, *b, half_turn);
}

double AssetMap::boost(const GameState& state, const Move& move) {
  if (move.physical()) return 0;
  const BoardAsset* a = asset(state, move.to_timeline, move.to_half_turn);
  if (a == nullptr || !a->flagged || a->traveller != move.piece.colour()) return 0;
  return a->boost;
}

}  // namespace t5d
