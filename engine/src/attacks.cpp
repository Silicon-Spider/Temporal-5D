// Incremental royal-attack maps (see t5d/attacks.hpp). Rule IDs refer to the
// rulebook, docs/rules/README.md. The capture rules are check.hpp's - the move
// generator's restated for captures (J2, J7, J8, L4, Q8, Q14, Q18-Q20) - walked
// line by line so that each line's end is kept. Written from the rulebook and
// the design spec; no other implementation was consulted.

#include "t5d/attacks.hpp"

#include <algorithm>
#include <bit>
#include <utility>

#include "t5d/board.hpp"
#include "t5d/geometry.hpp"

namespace t5d {

// One playable board's record: the captures from it and the lines from it
// that stopped at void.
struct RoyalAttackMap::Source {
  struct Line {
    Piece piece;
    Vec4 step{};
    std::int8_t from_square = kNoSquare;
    std::uint8_t ordinal = 0;  // the line's place in piece_lines' order
    bool slide = false;
  };
  struct Capture {
    Line line;
    std::int16_t steps = 1;
    std::int32_t to_timeline = 0;
    std::int32_t to_half_turn = 0;
    std::int8_t to_square = kNoSquare;
    Piece royal;
  };
  // A line that stopped at void: the place, the square it would have
  // occupied there, and the step number it reached.
  struct Stop {
    std::int32_t timeline = 0;
    std::int32_t half_turn = 0;
    Line line;
    std::int16_t steps = 1;
    std::int8_t x = 0;
    std::int8_t y = 0;
  };

  int timeline = 0;
  int half_turn = 0;
  BoardRef board;
  std::vector<Capture> captures;
  std::vector<Stop> stops;  // sorted by (timeline, half_turn)
};

namespace {

using Source = RoyalAttackMap::Source;

bool place_less(const Source::Stop& a, const Source::Stop& b) noexcept {
  return a.timeline != b.timeline ? a.timeline < b.timeline : a.half_turn < b.half_turn;
}

constexpr Vec4 v4(int x, int y, int l, int t) noexcept {
  return Vec4{static_cast<std::int8_t>(x), static_cast<std::int8_t>(y),
              static_cast<std::int8_t>(l), static_cast<std::int8_t>(t)};
}

// The lines a piece's captures follow, in check.cpp's order (piece_captures):
// f(step, slide) for each.
template <class F>
void piece_lines(const RuleOptions& rules, Piece piece, F&& f) {
  const Colour c = piece.colour();
  switch (piece.kind()) {
    case PieceKind::Pawn:
    case PieceKind::Brawn: {
      const int fy = geometry::forward_rank(c);
      const int fl = geometry::forward_timeline(c);
      f(v4(-1, fy, 0, 0), false);
      f(v4(1, fy, 0, 0), false);
      f(v4(0, 0, fl, -1), false);
      f(v4(0, 0, fl, 1), false);
      if (piece.kind() == PieceKind::Brawn) {
        f(v4(0, fy, 0, -1), false);
        f(v4(0, fy, 0, 1), false);
        f(v4(-1, 0, fl, 0), false);
        f(v4(1, 0, fl, 0), false);
        if (rules.brawn_rank_timeline_capture) f(v4(0, fy, fl, 0), false);
      }
      return;
    }
    case PieceKind::Knight:
      for (const Vec4 v : geometry::knight_leaps()) f(v, false);
      return;
    case PieceKind::King:
    case PieceKind::CommonKing:
      for (const Vec4 v : geometry::all_directions()) f(v, false);
      return;
    default:
      for (const Vec4 v : geometry::slider_directions(piece.kind())) f(v, true);
      return;
  }
}

// The walk's context: a multiverse, the rules, and the source being walked.
struct Ctx {
  const Multiverse& mv;
  const RuleOptions& rules;
  std::span<const int> inactive;
  Colour attacker;
  BoardSize size;
  int min_timeline = 0;
  int max_timeline = -1;
  int max_half_turn = -1;
  int p = 0;  // the source's place
  int h = 0;
  const Board* board = nullptr;
};

Ctx make_ctx(const Multiverse& mv, const RuleOptions& rules, std::span<const int> inactive,
             Colour attacker) {
  Ctx c{mv, rules, inactive, attacker, mv.board_size(), 0, -1, -1, 0, 0, nullptr};
  if (!mv.empty()) {
    c.min_timeline = mv.min_timeline();
    c.max_timeline = mv.max_timeline();
    c.max_half_turn = mv.latest_half_turn();
  }
  return c;
}

struct Spot {
  const Board* board = nullptr;
  bool history = false;
  int half_turn = 0;
};

inline Spot spot_at(const Ctx& c, int timeline, int half_turn) noexcept {
  const Timeline* t = c.mv.timeline(timeline);
  if (t == nullptr || !t->contains(half_turn)) return Spot{};
  return Spot{t->board_at(half_turn), half_turn != t->last_half_turn(), half_turn};
}

// J2 with Q8 and Q20, as check.cpp's landable().
inline bool landable(const Ctx& c, const Spot& spot, int timeline) noexcept {
  if (!spot.history) return true;
  if (!c.rules.turn_zero_reachable && turn_of(spot.half_turn) == 0) return false;
  if (!c.rules.inactive_history_is_destination &&
      std::binary_search(c.inactive.begin(), c.inactive.end(), timeline)) {
    return false;
  }
  return true;
}

// Walks `line` from the cell (x, y) of place (l, ht), its step n, onwards (a
// slide past it while it is empty). `sink.capture(line, n, l, ht, square,
// royal)` and `sink.stop(line, n, x, y, l, ht)` hear how it ends; a true
// return asks the scan to stop. Returns that request.
template <class Sink>
bool walk(const Ctx& c, const Source::Line& line, int n, int x, int y, int l, int ht, Sink& sink) {
  const Vec4 d = line.step;
  const bool physical = d.physical();
  // Q19: only a slide straight along T on its own square and timeline can
  // meet the piece's own earlier self.
  const bool past_self_passes =
      line.slide && !c.rules.past_self_blocks && d.x == 0 && d.y == 0 && d.l == 0;
  for (;; ++n, x += d.x, y += d.y, l += d.l, ht += 2 * d.t) {
    if (!c.size.contains(x, y)) return false;  // off the board: for good
    Spot spot{c.board, false, c.h};
    if (!physical) {
      if (l < c.min_timeline || l > c.max_timeline || ht < 0 || ht > c.max_half_turn) {
        // Beyond every board: void now; a board may come to stand there.
        if (ht >= 0) return sink.stop(line, n, x, y, l, ht);
        return false;
      }
      spot = spot_at(c, l, ht);
      if (spot.board == nullptr) {
        if (line.slide && c.rules.sliders_cross_void) continue;  // Q18's dissent
        return sink.stop(line, n, x, y, l, ht);                  // J8: void blocks
      }
    }
    const int to = make_square(x, y);
    const Piece occupant = spot.board->at(to);
    if (occupant.empty()) {
      if (line.slide) continue;
      return false;
    }
    if (occupant.colour() == c.attacker) {
      if (past_self_passes && occupant.kind() == line.piece.kind()) continue;
      return false;
    }
    if (!occupant.royal() || !landable(c, spot, l)) return false;
    return sink.capture(line, n, l, spot.half_turn, to, occupant);
  }
}

// Every line of every piece of the source's, from its first step.
template <class Sink>
bool scan_source(const Ctx& c, Sink& sink) {
  for (std::uint64_t bits = c.board->pieces(c.attacker); bits != 0; bits &= bits - 1) {
    const int square = std::countr_zero(bits);
    const Piece piece = c.board->at(square);
    std::uint8_t ordinal = 0;
    bool stop = false;
    piece_lines(c.rules, piece, [&](Vec4 d, bool slide) {
      if (stop) return;
      const Source::Line line{piece, d, static_cast<std::int8_t>(square), ordinal++, slide};
      stop = walk(c, line, 1, file_of(square) + d.x, rank_of(square) + d.y, c.p + d.l, c.h + 2 * d.t,
                  sink);
    });
    if (stop) return true;
  }
  return false;
}

// A sink that records into a Source.
struct Recorder {
  Source& out;
  bool capture(const Source::Line& line, int n, int l, int ht, int to, Piece royal) {
    Source::Capture k;
    k.line = line;
    k.steps = static_cast<std::int16_t>(n);
    k.to_timeline = l;
    k.to_half_turn = ht;
    k.to_square = static_cast<std::int8_t>(to);
    k.royal = royal;
    out.captures.push_back(k);
    return false;
  }
  bool stop(const Source::Line& line, int n, int x, int y, int l, int ht) {
    Source::Stop s;
    s.timeline = l;
    s.half_turn = ht;
    s.line = line;
    s.steps = static_cast<std::int16_t>(n);
    s.x = static_cast<std::int8_t>(x);
    s.y = static_cast<std::int8_t>(y);
    out.stops.push_back(s);
    return false;
  }
};

// A sink that only asks whether a capture exists.
struct Finder {
  bool capture(const Source::Line&, int, int, int, int, Piece) { return true; }
  bool stop(const Source::Line&, int, int, int, int, int) { return false; }
};

std::shared_ptr<Source> scan_new(Ctx c, const Timeline& t) {
  auto s = std::make_shared<Source>();
  s->timeline = t.position();
  s->half_turn = t.last_half_turn();
  s->board = t.head();
  c.p = s->timeline;
  c.h = s->half_turn;
  c.board = s->board.get();
  Recorder r{*s};
  scan_source(c, r);
  std::sort(s->stops.begin(), s->stops.end(), place_less);
  return s;
}

bool same_board(const BoardRef& a, const BoardRef& b) noexcept {
  return a == b || (a != nullptr && b != nullptr && *a == *b);
}

// Whether `next` extends `base` - every board of base's in the same place with
// the same content - and, if so, the places of next's new boards whose
// half-turn the attacker's lines can visit (its colour's parity).
bool extends(const Multiverse& base, const Multiverse& next, Colour attacker,
             std::vector<std::pair<int, int>>& new_places) {
  new_places.clear();
  if (!(base.board_size() == next.board_size()) || base.even() != next.even()) return false;
  if (base.empty()) return false;
  if (next.empty()) return false;
  if (next.min_timeline() > base.min_timeline() || next.max_timeline() < base.max_timeline()) {
    return false;
  }
  const int parity = index_of(attacker);
  const auto add_new = [&](int p, int h) {
    if ((h & 1) == parity) new_places.emplace_back(p, h);
  };
  for (const TimelineRef& nt : next.timelines()) {
    const int p = nt->position();
    const Timeline* bt = base.timeline(p);
    if (bt == nullptr) {
      for (int h = nt->first_half_turn(); h <= nt->last_half_turn(); ++h) add_new(p, h);
      continue;
    }
    if (bt == nt.get()) continue;  // shared: unchanged
    if (bt->first_half_turn() != nt->first_half_turn() || nt->length() < bt->length()) return false;
    for (int h = bt->first_half_turn(); h <= bt->last_half_turn(); ++h) {
      if (!same_board(*bt->board_ref_at(h), *nt->board_ref_at(h))) return false;
    }
    for (int h = bt->last_half_turn() + 1; h <= nt->last_half_turn(); ++h) add_new(p, h);
  }
  return true;
}

}  // namespace

bool attack_maps_incremental(const RuleOptions& rules) noexcept {
  return !rules.sliders_cross_void && rules.turn_zero_reachable &&
         rules.inactive_history_is_destination;
}

RoyalAttackMap::RoyalAttackMap(std::shared_ptr<const Multiverse> mv, const RuleOptions& rules,
                               Colour attacker, std::vector<int> inactive)
    : mv_(std::move(mv)), rules_(rules), attacker_(attacker), inactive_(std::move(inactive)) {}

RoyalAttackMap::RoyalAttackMap(const Multiverse& multiverse, const RuleOptions& rules,
                               Colour attacker, std::span<const int> inactive_timelines)
    : mv_(std::make_shared<const Multiverse>(multiverse)),
      rules_(rules),
      attacker_(attacker),
      inactive_(inactive_timelines.begin(), inactive_timelines.end()) {
  scan_all();
}

void RoyalAttackMap::scan_all() {
  sources_.clear();
  capture_count_ = 0;
  if (!mv_ || mv_->empty()) return;
  const Ctx c = make_ctx(*mv_, rules_, inactive_, attacker_);
  for (const TimelineRef& t : mv_->timelines()) {
    if (to_move_at(t->last_half_turn()) != attacker_) continue;  // L5
    std::shared_ptr<Source> s = scan_new(c, *t);
    capture_count_ += s->captures.size();
    sources_.push_back(std::move(s));
  }
}

std::size_t RoyalAttackMap::stop_count() const noexcept {
  std::size_t n = 0;
  for (const auto& s : sources_) n += s->stops.size();
  return n;
}

void RoyalAttackMap::void_stops(std::vector<VoidStop>& out) const {
  for (const auto& s : sources_) {
    for (const Source::Stop& st : s->stops) {
      VoidStop v;
      v.source_timeline = s->timeline;
      v.source_half_turn = s->half_turn;
      v.timeline = st.timeline;
      v.half_turn = st.half_turn;
      v.steps = st.steps;
      v.from_square = st.line.from_square;
      v.ordinal = st.line.ordinal;
      out.push_back(v);
    }
  }
}

std::vector<RoyalCapture> RoyalAttackMap::captures() const {
  std::vector<RoyalCapture> out;
  if (!mv_) return out;
  for (const auto& s : sources_) {
    std::vector<const Source::Capture*> order;
    order.reserve(s->captures.size());
    for (const Source::Capture& k : s->captures) order.push_back(&k);
    std::sort(order.begin(), order.end(), [](const Source::Capture* a, const Source::Capture* b) {
      return a->line.from_square != b->line.from_square ? a->line.from_square < b->line.from_square
                                                        : a->line.ordinal < b->line.ordinal;
    });
    for (const Source::Capture* k : order) {
      RoyalCapture c;
      c.attacker = k->line.piece;
      c.from_timeline = s->timeline;
      c.from_half_turn = s->half_turn;
      c.from_square = k->line.from_square;
      c.royal = k->royal;
      c.to_timeline = k->to_timeline;
      c.to_half_turn = k->to_half_turn;
      c.to_square = k->to_square;
      c.step = k->line.step;
      c.steps = k->steps;
      // J3: a capture between boards branches iff it lands on a history board
      // - or, under the Quick Rules' literal reading, at an earlier turn.
      if (!c.step.physical()) {
        const Timeline* t = mv_->timeline(c.to_timeline);
        const bool history = t != nullptr && c.to_half_turn != t->last_half_turn();
        c.branching = history ||
                      (rules_.earlier_playable_jump_branches && c.to_half_turn < c.from_half_turn);
      }
      out.push_back(c);
    }
  }
  return out;
}

bool RoyalAttackMap::derive(const Multiverse& next, std::span<const int> inactive,
                            RoyalAttackMap* out, AttackMapStats* stats) const {
  AttackMapStats local;
  AttackMapStats& st = stats != nullptr ? *stats : local;
  st = AttackMapStats{};
  std::vector<std::pair<int, int>> places;
  const bool incremental =
      mv_ && attack_maps_incremental(rules_) && extends(*mv_, next, attacker_, places);
  if (!incremental) {
    st.recomputed = true;
    if (out != nullptr) {
      *out = RoyalAttackMap(std::make_shared<const Multiverse>(next), rules_, attacker_,
                            std::vector<int>(inactive.begin(), inactive.end()));
      out->scan_all();
      st.sources_scanned = out->sources_.size();
      return out->any();
    }
    if (next.empty()) return false;
    Ctx c = make_ctx(next, rules_, inactive, attacker_);
    Finder f;
    for (const TimelineRef& t : next.timelines()) {
      if (to_move_at(t->last_half_turn()) != attacker_) continue;
      ++st.sources_scanned;
      c.p = t->position();
      c.h = t->last_half_turn();
      c.board = t->head().get();
      if (scan_source(c, f)) return true;
    }
    return false;
  }

  // Incremental: next extends the map's multiverse.
  Ctx c = make_ctx(next, rules_, inactive, attacker_);
  std::vector<std::shared_ptr<const Source>> kept;  // still playable
  kept.reserve(sources_.size());
  for (const auto& s : sources_) {
    const Timeline* t = next.timeline(s->timeline);
    if (t != nullptr && t->last_half_turn() == s->half_turn) kept.push_back(s);
  }
  // A kept source's captures stand: no new board can block them.
  bool found = false;
  for (const auto& s : kept) {
    if (!s->captures.empty()) found = true;
  }
  if (found && out == nullptr) {
    st.sources_kept = kept.size();
    return true;
  }
  // The lines of kept sources that stopped where a new board stands.
  std::vector<std::shared_ptr<const Source>> updated;
  updated.reserve(kept.size());
  for (const auto& s : kept) {
    std::vector<std::size_t> hits;
    for (const auto& [p, h] : places) {
      Source::Stop key;
      key.timeline = p;
      key.half_turn = h;
      auto it = std::lower_bound(s->stops.begin(), s->stops.end(), key, place_less);
      for (; it != s->stops.end() && it->timeline == p && it->half_turn == h; ++it) {
        hits.push_back(static_cast<std::size_t>(it - s->stops.begin()));
      }
    }
    if (hits.empty()) {
      ++st.sources_kept;
      updated.push_back(s);
      continue;
    }
    c.p = s->timeline;
    c.h = s->half_turn;
    c.board = s->board.get();
    if (out == nullptr) {
      Finder f;
      for (const std::size_t i : hits) {
        const Source::Stop& stop = s->stops[i];
        ++st.lines_resumed;
        if (walk(c, stop.line, stop.steps, stop.x, stop.y, stop.timeline, stop.half_turn, f)) {
          return true;
        }
      }
      continue;
    }
    // Copy on write: this source's record changes.
    auto copy = std::make_shared<Source>();
    copy->timeline = s->timeline;
    copy->half_turn = s->half_turn;
    copy->board = s->board;
    copy->captures = s->captures;
    std::sort(hits.begin(), hits.end());
    copy->stops.reserve(s->stops.size());
    std::size_t next_hit = 0;
    for (std::size_t i = 0; i < s->stops.size(); ++i) {
      if (next_hit < hits.size() && hits[next_hit] == i) {
        ++next_hit;
        continue;
      }
      copy->stops.push_back(s->stops[i]);
    }
    Recorder r{*copy};
    for (const std::size_t i : hits) {
      const Source::Stop& stop = s->stops[i];
      ++st.lines_resumed;
      walk(c, stop.line, stop.steps, stop.x, stop.y, stop.timeline, stop.half_turn, r);
    }
    std::sort(copy->stops.begin(), copy->stops.end(), place_less);
    updated.push_back(std::move(copy));
  }
  // New sources: playable boards that were not sources (or not there).
  std::vector<std::shared_ptr<const Source>> fresh;
  for (const TimelineRef& t : next.timelines()) {
    if (to_move_at(t->last_half_turn()) != attacker_) continue;
    const bool is_kept = std::any_of(kept.begin(), kept.end(), [&](const auto& s) {
      return s->timeline == t->position();
    });
    if (is_kept) continue;
    ++st.sources_scanned;
    if (out == nullptr) {
      c.p = t->position();
      c.h = t->last_half_turn();
      c.board = t->head().get();
      Finder f;
      if (scan_source(c, f)) return true;
      continue;
    }
    fresh.push_back(scan_new(c, *t));
  }
  if (out == nullptr) return false;
  *out = RoyalAttackMap(std::make_shared<const Multiverse>(next), rules_, attacker_,
                        std::vector<int>(inactive.begin(), inactive.end()));
  out->sources_ = std::move(updated);
  out->sources_.insert(out->sources_.end(), fresh.begin(), fresh.end());
  std::sort(out->sources_.begin(), out->sources_.end(),
            [](const auto& a, const auto& b) { return a->timeline < b->timeline; });
  out->capture_count_ = 0;
  for (const auto& s : out->sources_) out->capture_count_ += s->captures.size();
  return out->any();
}

RoyalAttackMap RoyalAttackMap::advanced(const Multiverse& next,
                                        std::span<const int> inactive_timelines,
                                        AttackMapStats* stats) const {
  RoyalAttackMap out;
  derive(next, inactive_timelines, &out, stats);
  return out;
}

bool RoyalAttackMap::any_capture_in(const Multiverse& next, std::span<const int> inactive_timelines,
                                    AttackMapStats* stats) const {
  return derive(next, inactive_timelines, nullptr, stats);
}

}  // namespace t5d
