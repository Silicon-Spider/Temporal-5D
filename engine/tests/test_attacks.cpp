// Incremental royal-attack maps (t5d/attacks.hpp; spec §11.1's redesign, part
// 3; register M14, A4): every map - built, advanced, or asked without being
// built - must give check.hpp's from-scratch answer (for_each_royal_capture,
// royal_capture_exists): on random multiverses, on random extensions of them,
// and along random legal play with its passed multiverses, under every
// ruleset.

#include <cstdint>
#include <set>
#include <tuple>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "t5d/attacks.hpp"
#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_play.hpp"
#include "test_positions.hpp"
#include "test_rulesets.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

std::vector<RoyalCapture> scratch(const Multiverse& mv, const RuleOptions& rules, Colour attacker,
                                  std::span<const int> inactive) {
  std::vector<RoyalCapture> out;
  for_each_royal_capture(mv, rules, attacker, inactive, [&](const RoyalCapture& c) {
    out.push_back(c);
    return false;
  });
  return out;
}

std::string text(const std::vector<RoyalCapture>& cs) {
  std::string out;
  for (const RoyalCapture& c : cs) out += describe(c, false) + "; ";
  return out;
}

// A random multiverse (test_check.cpp's kind): 1-5 contiguous timelines, each
// 1-4 boards from a random half-turn, random pieces with royals plentiful.
Piece random_piece_with_royals(t5d_test::Rng& rng) {
  Piece piece = t5d_test::random_piece(rng);
  if (rng.chance(25)) {
    piece = Piece(rng.chance(70) ? PieceKind::King : PieceKind::RoyalQueen, piece.colour());
  }
  return piece;
}

BoardRef royal_rich_board(t5d_test::Rng& rng, BoardSize size, int density = 28) {
  BoardBuilder b(size);
  for (int sq = 0; sq < 64; ++sq) {
    if (!size.contains_square(sq) || !rng.chance(density)) continue;
    b.put(sq, random_piece_with_royals(rng));
  }
  return b.publish();
}

Multiverse random_multiverse(t5d_test::Rng& rng) {
  const BoardSize size{rng.between(2, 6), rng.between(2, 6)};
  const bool even = rng.chance(50);
  Multiverse mv(size, even);
  const int count = rng.between(1, 5);
  const int first = -rng.below(count);
  for (int i = 0; i < count; ++i) {
    NewTimeline nt;
    nt.position = first + i;
    nt.first_half_turn = rng.between(0, 6);
    const int length = rng.between(1, 4);
    for (int k = 0; k < length; ++k) nt.boards.push_back(royal_rich_board(rng, size));
    if (nt.position != 0 && !(even && nt.position == -1)) {
      nt.creator = nt.position > 0 ? Colour::White : Colour::Black;
      nt.creation_order = 1;
    }
    mv.add_timeline(std::move(nt));
  }
  return mv;
}

// Extends `mv` at random: boards appended to timelines (a random board, or the
// head changed on a square or two - so that most lines still meet what they
// met), or a new timeline at either end.
void extend_randomly(Multiverse& mv, t5d_test::Rng& rng) {
  const int edits = rng.between(1, 3);
  for (int e = 0; e < edits; ++e) {
    if (rng.chance(25)) {
      NewTimeline nt;
      const bool up = rng.chance(50);
      nt.position = up ? mv.max_timeline() + 1 : mv.min_timeline() - 1;
      nt.first_half_turn = rng.between(0, mv.latest_half_turn() + 2);
      const int length = rng.between(1, 2);
      for (int k = 0; k < length; ++k) nt.boards.push_back(royal_rich_board(rng, mv.board_size()));
      nt.creator = up ? Colour::White : Colour::Black;
      nt.creation_order = mv.created_by(*nt.creator) + 1;
      mv.add_timeline(std::move(nt));
      continue;
    }
    const int p = rng.between(mv.min_timeline(), mv.max_timeline());
    BoardRef next;
    if (rng.chance(40)) {
      next = royal_rich_board(rng, mv.board_size());
    } else {
      BoardBuilder b(*mv.head(p));
      for (int k = rng.between(1, 2); k > 0; --k) {
        const int sq = make_square(rng.below(mv.board_size().width), rng.below(mv.board_size().height));
        b.put(sq, rng.chance(50) ? Piece{} : random_piece_with_royals(rng));
      }
      next = b.publish();
    }
    mv.append_board(p, next);
  }
}

// The map against the scratch scan: the same captures, in the same order.
bool same(const RoyalAttackMap& map, const Multiverse& mv, const RuleOptions& rules, Colour attacker,
          std::span<const int> inactive, const char* where) {
  const std::vector<RoyalCapture> want = scratch(mv, rules, attacker, inactive);
  const std::vector<RoyalCapture> got = map.captures();
  const bool any = map.any();
  if (got == want && any == !want.empty()) return true;
  std::fprintf(stderr, "  %s (%s, %s): map [%s] scratch [%s]\n", where, describe(rules).c_str(),
               attacker == Colour::White ? "white" : "black", text(got).c_str(), text(want).c_str());
  return false;
}

}  // namespace

TEST(attacks_map_equals_the_scratch_scan_on_random_multiverses) {
  t5d_test::Rng rng(808);
  int captures = 0;
  for (const RuleOptions& rules : t5d_test::every_ruleset()) {
    for (int i = 0; i < 120; ++i) {
      const Multiverse mv = random_multiverse(rng);
      std::vector<int> inactive;
      for (const TimelineRef& t : mv.timelines()) {
        if (t->creator() && rng.chance(30)) inactive.push_back(t->position());
      }
      for (const Colour c : {Colour::White, Colour::Black}) {
        const RoyalAttackMap map(mv, rules, c, inactive);
        CHECK(same(map, mv, rules, c, inactive, "built"));
        CHECK_EQ(map.any(), royal_capture_exists(mv, rules, c, inactive));
        captures += static_cast<int>(map.capture_count());
      }
    }
  }
  CHECK(captures > 200);
}

TEST(attacks_advanced_equals_the_scratch_scan_on_random_extensions) {
  t5d_test::Rng rng(909);
  std::size_t incremental = 0;
  std::size_t recomputed = 0;
  std::size_t resumed = 0;
  std::size_t kept = 0;
  for (const RuleOptions& rules : t5d_test::every_ruleset()) {
    for (int i = 0; i < 40; ++i) {
      Multiverse mv = random_multiverse(rng);
      const std::vector<int> inactive;
      for (const Colour c : {Colour::White, Colour::Black}) {
        RoyalAttackMap map(mv, rules, c, inactive);
        Multiverse cur = mv;
        for (int step = 0; step < 6; ++step) {
          Multiverse next = cur;
          extend_randomly(next, rng);
          AttackMapStats st;
          const bool any = map.any_capture_in(next, inactive);
          RoyalAttackMap advanced = map.advanced(next, inactive, &st);
          if (!same(advanced, next, rules, c, inactive, "advanced")) {
            CHECK(false);
          }
          CHECK_EQ(any, royal_capture_exists(next, rules, c, inactive));
          (st.recomputed ? recomputed : incremental) += 1;
          resumed += st.lines_resumed;
          kept += st.sources_kept;
          if (attack_maps_incremental(rules)) CHECK(!st.recomputed);
          map = std::move(advanced);
          cur = std::move(next);
        }
        // Not an extension (a head replaced): recomputed, and still equal.
        Multiverse other = cur;
        const int p = other.min_timeline();
        other.replace_head(p, royal_rich_board(rng, other.board_size()));
        AttackMapStats st;
        const RoyalAttackMap r = map.advanced(other, inactive, &st);
        CHECK(same(r, other, rules, c, inactive, "replaced"));
        CHECK(st.recomputed || *other.head(p) == *cur.head(p));
      }
    }
  }
  std::fprintf(stderr, "  %zu incremental advances (%zu sources kept, %zu lines resumed), %zu recomputed\n",
               incremental, kept, resumed, recomputed);
  CHECK(resumed > 0);
  CHECK(kept > 0);
}

namespace {

// Random legal play, one move at a time: after every move the chained maps of
// both colours - each advanced from the map before the move - equal the
// scratch scan, and so does the map advanced to the passed multiverse of
// every turn start (R1's check status).
void maps_along_random_play(const RuleOptions& rules, int games, int turns, std::uint64_t seed,
                            int& checked, int& checks) {
  t5d_test::Rng rng(seed);
  std::vector<GameState> starts;
  for (const t5d_test::SparseStart& s : t5d_test::sparse_starts()) {
    starts.push_back(t5d_test::game(s.boards, s.size, rules));
  }
  for (const char* name : {"Standard", "Standard - Two Timelines", "Misc - Timeline Battleground"}) {
    if (const Variant* v = find_variant(name)) starts.push_back(variant_start(*v, rules));
  }
  for (const GameState& start : starts) {
    for (int g = 0; g < games; ++g) {
      GameState s = start;
      RoyalAttackMap white(s.multiverse(), rules, Colour::White, s.inactive_timelines());
      RoyalAttackMap black(s.multiverse(), rules, Colour::Black, s.inactive_timelines());
      for (int turn = 0; turn < turns; ++turn) {
        // R1: the mover's opponent, after the mover passes its present boards.
        const Multiverse passed = passed_multiverse(s);
        const RoyalAttackMap& opp = s.to_move() == Colour::White ? black : white;
        const bool status = opp.any_capture_in(passed, s.inactive_timelines());
        CHECK_EQ(status, in_check(s));
        if (status) ++checks;
        CHECK(same(opp.advanced(passed, s.inactive_timelines()), passed, rules, opposite(s.to_move()),
                   s.inactive_timelines(), "passed"));
        const std::optional<std::vector<Move>> a = t5d_test::random_legal_action(s, rng);
        if (!a) break;
        for (const Move& m : *a) {
          s.apply(m);
          white = white.advanced(s.multiverse(), s.inactive_timelines());
          black = black.advanced(s.multiverse(), s.inactive_timelines());
          CHECK(same(white, s.multiverse(), rules, Colour::White, s.inactive_timelines(), "play"));
          CHECK(same(black, s.multiverse(), rules, Colour::Black, s.inactive_timelines(), "play"));
          // C4 asked of the partial action.
          CHECK_EQ((s.to_move() == Colour::White ? black : white).any(), !royals_safe(s));
          ++checked;
        }
        s.submit();
      }
    }
  }
}

}  // namespace

TEST(attacks_chained_maps_equal_the_scratch_scan_along_random_play) {
  int checked = 0;
  int checks = 0;
  std::uint64_t seed = 70;
  for (const RuleOptions& rules : t5d_test::every_ruleset()) {
    maps_along_random_play(rules, 1, 6, seed++, checked, checks);
  }
  std::fprintf(stderr, "  %d moves, %d turn starts in check\n", checked, checks);
  CHECK(checked > 500);
  CHECK(checks > 10);
}

TEST(long_attacks_chained_maps_equal_the_scratch_scan_along_random_play) {
  int checked = 0;
  int checks = 0;
  std::uint64_t seed = 170;
  for (const RuleOptions& rules : t5d_test::every_ruleset()) {
    maps_along_random_play(rules, 4, 14, seed++, checked, checks);
  }
  std::fprintf(stderr, "  %d moves, %d turn starts in check\n", checked, checks);
  CHECK(checked > 5000);
}

namespace {

// The generator's captures of any enemy piece (en passant apart), as (from,
// to) places and squares.
using Key = std::tuple<int, int, int, int, int, int>;
std::set<Key> generated_captures(const Multiverse& mv, const RuleOptions& rules, Colour attacker,
                                 std::span<const int> inactive) {
  std::set<Key> out;
  MoveList moves;
  for (const TimelineRef& t : mv.timelines()) {
    moves.clear();
    append_moves_of(mv, rules, attacker, t->position(), moves, inactive);
    for (const Move& m : moves) {
      if (!m.capture() || m.en_passant()) continue;
      out.insert({m.from_timeline, m.from_half_turn, m.from_square, m.to_timeline, m.to_half_turn,
                  m.to_square});
    }
  }
  return out;
}

std::set<Key> mapped_captures(const RoyalAttackMap& map) {
  std::set<Key> out;
  for (const RoyalCapture& c : map.captures()) {
    out.insert({c.from_timeline, c.from_half_turn, c.from_square, c.to_timeline, c.to_half_turn,
                c.to_square});
  }
  return out;
}

}  // namespace

TEST(attacks_all_capture_maps_equal_the_generators_captures) {
  // CaptureTargets::All (the estimate channel's): every capture of an enemy
  // piece the generator makes, built and advanced, under every ruleset.
  t5d_test::Rng rng(1212);
  int captures = 0;
  for (const RuleOptions& rules : t5d_test::every_ruleset()) {
    for (int i = 0; i < 25; ++i) {
      Multiverse mv = random_multiverse(rng);
      const std::vector<int> inactive;
      for (const Colour c : {Colour::White, Colour::Black}) {
        RoyalAttackMap map(mv, rules, c, inactive, CaptureTargets::All);
        Multiverse cur = mv;
        for (int step = 0; step < 4; ++step) {
          const std::set<Key> want = generated_captures(cur, rules, c, inactive);
          CHECK(mapped_captures(map) == want);
          CHECK_EQ(map.capture_count(), want.size());
          captures += static_cast<int>(want.size());
          Multiverse next = cur;
          extend_randomly(next, rng);
          map = map.advanced(next, inactive);
          cur = std::move(next);
        }
      }
    }
  }
  CHECK(captures > 1000);
}

TEST(attacks_a_board_left_unplayed_keeps_its_record) {
  // Two timelines; White plays one board: the other playable board of
  // White's keeps its record (shared), the played one is dropped.
  const GameState s = t5d_test::game({"[k4/5/5/5/K3R:-0:1:w]", "[k4/5/5/5/K3R:+0:1:w]"}, BoardSize{5, 5});
  const RoyalAttackMap map(s.multiverse(), s.rules(), Colour::White);
  CHECK_EQ(map.source_count(), 2u);
  GameState t = s;
  t5d_test::play(t, "(-0T1)Re1(-0T1)e2");
  AttackMapStats st;
  const RoyalAttackMap next = map.advanced(t.multiverse(), {}, &st);
  CHECK(!st.recomputed);
  CHECK_EQ(st.sources_kept, 1u);
  CHECK_EQ(next.source_count(), 1u);
  CHECK(same(next, t.multiverse(), t.rules(), Colour::White, {}, "unplayed"));
}
