// Expansion on positions built by hand (t5d/expand.hpp): the window W(k), the
// travel destinations E(k) and src_extra(k), the move graph, the typed
// relations and the features. Every expectation is worked out here from the
// rules reference (docs/rules/README.md) and the design spec's definitions
// (§6, §10.4.1, §10.4.3) - board by board, move by move - never recorded from a
// run.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/expand.hpp"
#include "t5d/fen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/records.h"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;

namespace {

// A board with pieces written "Ra1" (white rook on a1), "pa6" (black pawn),
// "P*c2" (an unmoved white pawn).
BoardRef board(std::initializer_list<std::string_view> pieces, BoardSize size = {8, 8},
               int en_passant = kNoSquare) {
  BoardBuilder b(size);
  for (std::string_view text : pieces) {
    const char letter = text[0];
    const bool unmoved = text[1] == '*';
    const std::string_view square = text.substr(unmoved ? 2 : 1);
    const Colour colour = std::isupper(static_cast<unsigned char>(letter)) != 0 ? Colour::White
                                                                              : Colour::Black;
    b.put(square[0] - 'a', square[1] - '1', Piece(*kind_from_letter(letter), colour, unmoved));
  }
  if (en_passant != kNoSquare) b.set_en_passant(en_passant);
  return b.publish();
}

int sq(std::string_view name) { return make_square(name[0] - 'a', name[1] - '1'); }

// A timeline's boards from half-turn `first` to `last`: empty, but for those
// given.
std::vector<BoardRef> boards(int first, int last, const std::map<int, BoardRef>& given = {},
                             BoardSize size = {8, 8}) {
  const BoardRef empty = BoardBuilder(size).publish();
  std::vector<BoardRef> out;
  for (int h = first; h <= last; ++h) {
    const auto it = given.find(h);
    out.push_back(it != given.end() ? it->second : empty);
  }
  return out;
}

NewTimeline timeline(int position, int first, std::vector<BoardRef> bs,
                     std::optional<Colour> creator = std::nullopt, int order = 0,
                     std::optional<BoardCoord> origin = std::nullopt) {
  NewTimeline t;
  t.position = position;
  t.first_half_turn = first;
  t.boards = std::move(bs);
  // Assigned only when present: g++ 13 and 14 at -O2 warn, wrongly, that
  // copying a disengaged optional may read its uninitialised payload.
  if (creator.has_value()) t.creator = *creator;
  t.creation_order = order;
  if (origin.has_value()) t.origin = *origin;
  return t;
}

GameState state_of(std::vector<NewTimeline> timelines, Colour to_move = Colour::White,
                   const RuleOptions& rules = {}, BoardSize size = {8, 8}) {
  Multiverse mv(size, false);
  for (NewTimeline& t : timelines) mv.add_timeline(std::move(t));
  mv.set_to_move(to_move);
  return GameState(std::move(mv), rules);
}

Expansion expanded(const GameState& s, int k = 8, bool relations = false,
                   SquareFeatures features = SquareFeatures::All) {
  ExpandOptions o;
  o.window_k = k;
  o.square_features = features;
  o.attack = o.defence = o.xray = relations;
  Expansion x;
  expand(s, o, x);
  return x;
}

std::string bands_text(const Expansion& x, int k) {
  std::vector<std::int32_t> v;
  x.window(k, v, Window::BandsOnly);
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i != 0) out += ',';
    out += std::to_string(v[i]);
  }
  return out;
}

std::string text(std::span<const std::int32_t> v) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i != 0) out += ',';
    out += std::to_string(v[i]);
  }
  return out;
}

std::string window_text(const Expansion& x, int k) {
  std::vector<std::int32_t> v;
  x.window(k, v);
  return text(v);
}

std::string e_text(const Expansion& x, int k) {
  std::vector<std::int32_t> v;
  x.e_boards(k, v);
  return text(v);
}

std::string src_extra_text(const Expansion& x, int k) {
  std::vector<std::int32_t> v;
  x.src_extra(k, v);
  return text(v);
}

using Relation = std::array<std::int32_t, 5>;

std::set<Relation> relation_set(const Expansion& x) {
  std::set<Relation> out;
  const std::span<const std::int32_t> r = x.relations();
  for (std::size_t i = 0; i + 4 < r.size(); i += 5) {
    out.insert({r[i], r[i + 1], r[i + 2], r[i + 3], r[i + 4]});
  }
  return out;
}

}  // namespace

// ---- W(k): the window's mandatory core ----------------------------------------------------

TEST(expand_window_on_one_timeline_is_the_head_and_the_last_k_boards) {
  // Timeline 0, T1 white (h = 2) to T6 white (h = 12): eleven boards, board i
  // at h = 2 + i; White to move on the head.
  const GameState s = state_of({timeline(0, 2, boards(2, 12))});
  const Expansion x = expanded(s, 4);
  REQUIRE_EQ(x.board_count(), std::size_t{11});
  CHECK_EQ(text(x.window()), std::string("6,7,8,9,10"));  // h = 8..12
  CHECK_EQ(window_text(x, 0), std::string("10"));          // the head alone
  CHECK_EQ(window_text(x, 1), std::string("9,10"));
  // k beyond the history: every board.
  CHECK_EQ(window_text(x, 16), std::string("0,1,2,3,4,5,6,7,8,9,10"));
  CHECK_EQ(x.window_size(16), std::size_t{11});
  CHECK_EQ(x.window_size(100), std::size_t{11});
  for (std::size_t i = 0; i < 11; ++i) {
    CHECK_EQ(x.window_depth()[i], static_cast<std::int32_t>(10 - i));
    CHECK_EQ(x.board_timelines()[i], 0);
    CHECK_EQ(x.board_half_turns()[i], static_cast<std::int32_t>(2 + i));
    CHECK_EQ(x.board_side_to_move()[i], static_cast<std::uint8_t>(i & 1));
    CHECK_EQ(x.board_active()[i], std::uint8_t{1});
    CHECK_EQ(x.board_is_head()[i], static_cast<std::uint8_t>(i == 10 ? 1 : 0));
    // L5: the head is playable, the mover's (White, h even), a move may be made.
    CHECK_EQ(x.board_playable()[i], static_cast<std::uint8_t>(i == 10 ? 1 : 0));
  }
  CHECK_EQ(x.board_index(0, 2), 0);
  CHECK_EQ(x.board_index(0, 12), 10);
  CHECK_EQ(x.board_index(0, 13), -1);
  CHECK_EQ(x.board_index(1, 12), -1);
  // No pieces: no moves, so no E and no sources.
  CHECK(x.moves().empty());
  CHECK(x.e_boards().empty());
  CHECK(x.src_extra().empty());
  CHECK_EQ(x.timeline_count(), 1);
  CHECK_EQ(x.active_timeline_count(), 1);
  CHECK_EQ(x.present(), 12);
  CHECK(x.to_move() == Colour::White);
}

TEST(expand_window_takes_each_active_timelines_band_and_branch_origin) {
  // Timeline 0: h = 2..12. White's +1 branched from (0, h = 6): its first board
  // is that board's half-step successor, h = 7 (J5). Black's -1 branched from
  // (0, h = 7) (Black lands on a black board), first board h = 8. All three
  // heads at h = 12, all active (T10: each player has created one).
  const GameState s = state_of({
      timeline(0, 2, boards(2, 12)),
      timeline(1, 7, boards(7, 12), Colour::White, 1, BoardCoord{0, 6}),
      timeline(-1, 8, boards(8, 12), Colour::Black, 1, BoardCoord{0, 7}),
  });
  // Canonical order: -1 (h 8..12) = boards 0..4; 0 (h 2..12) = 5..15; +1
  // (h 7..12) = 16..21. The origins: (0, 6) = board 9, (0, 7) = board 10.
  const Expansion x = expanded(s, 0);
  REQUIRE_EQ(x.board_count(), std::size_t{22});
  CHECK_EQ(x.board_index(0, 6), 9);
  CHECK_EQ(x.board_index(0, 7), 10);
  // k = 0: the three heads and the two origins - (k + 2)T - 1 = 5 (§15.6's
  // arithmetic: T timelines at least k + 1 boards long, all but one branched).
  CHECK_EQ(text(x.window()), std::string("4,9,10,15,21"));
  CHECK_EQ(x.window_size(0), std::size_t{5});
  // k = 2: three boards per band and the two origins: (2 + 2) * 3 - 1 = 11.
  CHECK_EQ(window_text(x, 2), std::string("2,3,4,9,10,13,14,15,19,20,21"));
  // k = 4: -1's band is all its five boards; (4 + 2) * 3 - 1 = 17.
  CHECK_EQ(window_text(x, 4), std::string("0,1,2,3,4,9,10,11,12,13,14,15,17,18,19,20,21"));
  CHECK_EQ(x.window_size(4), std::size_t{17});
  // k = 5: -1 has only five boards (fewer than k + 1), and timeline 0's band
  // (h 7..12) now holds the origin (0, 7): counted once. 5 + 1 + 6 + 6 = 18.
  CHECK_EQ(window_text(x, 5), std::string("0,1,2,3,4,9,10,11,12,13,14,15,16,17,18,19,20,21"));
  CHECK_EQ(x.window_size(5), std::size_t{18});
  CHECK_EQ(x.window_size(16), std::size_t{22});
  // An origin is in the window at every k: its depth is 0.
  CHECK_EQ(x.window_depth()[9], 0);
  CHECK_EQ(x.window_depth()[10], 0);
  CHECK_EQ(x.window_depth()[11], 4);  // (0, 8): four half-turns behind its head
  CHECK_EQ(x.origins_unknown(), 0);
  CHECK_EQ(x.active_timeline_count(), 3);
  // The bands alone - W(k) without the branch origins (§10.4.3's queue lists
  // the origins after the "mandatory core"): k + 1 boards per active timeline.
  CHECK_EQ(bands_text(x, 0), std::string("4,15,21"));
  CHECK_EQ(x.window_size(0, Window::BandsOnly), std::size_t{3});
  CHECK_EQ(bands_text(x, 2), std::string("2,3,4,13,14,15,19,20,21"));
  CHECK_EQ(x.window_size(4, Window::BandsOnly), std::size_t{15});
  CHECK_EQ(x.origin_count(), std::size_t{2});
  for (std::size_t i = 0; i < 22; ++i) {
    CHECK_EQ(x.board_is_origin()[i], static_cast<std::uint8_t>(i == 9 || i == 10 ? 1 : 0));
  }
  CHECK_EQ(x.band_depth()[9], 6);  // (0, 6): six half-turns behind 0's head
  CHECK_EQ(x.band_depth()[10], 5);
  CHECK_EQ(x.band_depth()[15], 0);
  // The lineage (python/t5d/net/state.py, `parent`): the previous board of
  // its timeline, or a timeline's first board's branch origin.
  CHECK_EQ(x.board_parent()[0], 10);  // -1's first board (h 8): from (0, 7)
  CHECK_EQ(x.board_parent()[1], 0);
  CHECK_EQ(x.board_parent()[5], -1);  // timeline 0's first board: initial
  CHECK_EQ(x.board_parent()[6], 5);
  CHECK_EQ(x.board_parent()[16], 9);  // +1's first board (h 7): from (0, 6)
  CHECK_EQ(x.board_parent()[21], 20);
}

TEST(expand_window_gives_inactive_timelines_no_band) {
  // White has created +1 (from (0, 6)) and +2 (from (0, 8)), Black none: +2,
  // White's second, is inactive (T10: active iff Black has created at least
  // one). Boards: 0 = 0..10 (h 2..12), +1 = 11..16 (h 7..12), +2 = 17..20
  // (h 9..12).
  const GameState s = state_of({
      timeline(0, 2, boards(2, 12)),
      timeline(1, 7, boards(7, 12), Colour::White, 1, BoardCoord{0, 6}),
      timeline(2, 9, boards(9, 12), Colour::White, 2, BoardCoord{0, 8}),
  });
  REQUIRE(!s.active(2));
  const Expansion x = expanded(s, 0);
  CHECK_EQ(x.active_timeline_count(), 2);
  // The heads of 0 and +1 and +1's origin (0, 6) = board 4. Not +2's head,
  // and not +2's origin (0, 8) = board 6: an inactive timeline has no band.
  CHECK_EQ(text(x.window()), std::string("4,10,16"));
  CHECK_EQ(window_text(x, 2), std::string("4,8,9,10,14,15,16"));
  CHECK_EQ(x.window_size(100), std::size_t{17});  // every board but +2's four
  for (std::size_t i = 17; i < 21; ++i) {
    CHECK_EQ(x.window_depth()[i], kNoWindow);
    CHECK_EQ(x.board_active()[i], std::uint8_t{0});
  }
  // +2's head is White's, and playable (T6: inactive boards may be played).
  CHECK_EQ(x.board_playable()[20], std::uint8_t{1});
  // The largest k takes every band whole and still no board of +2: kNoWindow
  // (INT32_MAX) marks "in no window", and no k reaches it.
  CHECK_EQ(x.window_size(kNoWindow), std::size_t{17});
  CHECK_EQ(x.window_size(kNoWindow - 1), std::size_t{17});
  const Expansion all_k = expanded(s, kNoWindow);
  CHECK_EQ(all_k.window().size(), std::size_t{17});
  CHECK(std::none_of(all_k.window().begin(), all_k.window().end(),
                     [](std::int32_t b) { return b >= 17; }));
}

TEST(expand_window_takes_an_origin_on_an_inactive_timeline) {
  // White has created +1, +2 and +3, Black only -1: +3, White's third, is
  // inactive (T10: active iff Black has created at least two); -1, Black's
  // first, is active - and branched from (+3, h = 11), a board of the
  // inactive timeline (Black lands on a black board; -1's first board is its
  // half-step successor, h = 12). Canonical order: -1 (h 12) = board 0; 0
  // (h 2..12) = 1..11; +1 (h 7..12) = 12..17; +2 (h 9..12) = 18..21; +3
  // (h 11..12) = 22..23.
  const GameState s = state_of({
      timeline(0, 2, boards(2, 12)),
      timeline(1, 7, boards(7, 12), Colour::White, 1, BoardCoord{0, 6}),
      timeline(2, 9, boards(9, 12), Colour::White, 2, BoardCoord{0, 8}),
      timeline(3, 11, boards(11, 12), Colour::White, 3, BoardCoord{0, 10}),
      timeline(-1, 12, boards(12, 12), Colour::Black, 1, BoardCoord{3, 11}),
  });
  REQUIRE(!s.active(3));
  REQUIRE(s.active(-1));
  const Expansion x = expanded(s, 0);
  REQUIRE_EQ(x.board_count(), std::size_t{24});
  CHECK_EQ(x.board_index(3, 11), 22);
  // k = 0: the active heads (0, 11, 17, 21) and the active timelines' origins:
  // -1's (+3, 11) = board 22 - on an inactive timeline, in W all the same -
  // +1's (0, 6) = board 5, +2's (0, 8) = board 7. Not +3's head (23) nor its
  // origin (0, 10) = board 9.
  CHECK_EQ(text(x.window()), std::string("0,5,7,11,17,21,22"));
  CHECK_EQ(bands_text(x, 0), std::string("0,11,17,21"));
  CHECK_EQ(x.window_depth()[22], 0);
  CHECK_EQ(x.band_depth()[22], kNoWindow);
  CHECK_EQ(x.board_is_origin()[22], std::uint8_t{1});
  CHECK_EQ(x.board_is_origin()[9], std::uint8_t{0});
  CHECK_EQ(x.window_depth()[9], 2);  // in 0's band from k = 2 (h 10, head h 12)
  CHECK_EQ(x.window_depth()[23], kNoWindow);
  CHECK_EQ(x.origin_count(), std::size_t{3});
  CHECK_EQ(x.board_parent()[0], 22);
  CHECK_EQ(x.window_size(kNoWindow), std::size_t{23});  // all but +3's head
}

TEST(expand_counts_the_unknown_origins_of_a_position_read_from_5dfen) {
  // 5DFEN records no origins (t5d/fen.hpp): +n is White's n-th creation, -n
  // Black's. White has created +1, +2, +3 and Black -1, so +3 is inactive
  // (T10) and three ACTIVE timelines were created by branching with their
  // origins unknown: -1, +1, +2. None of their first boards has a parent.
  const std::vector<std::string> fen = {
      "[8/8/8/8/8/8/8/8:-1:3:w]", "[4k3/8/8/8/8/8/8/4K3:0:3:w]", "[8/8/8/8/8/8/8/8:1:3:w]",
      "[8/8/8/8/8/8/8/8:2:3:w]",  "[8/8/8/8/8/8/8/8:3:3:w]",
  };
  const GameState s = game_from_fen(fen, BoardSize{8, 8}, Parity::Odd);
  REQUIRE(!s.active(3));
  const Expansion x = expanded(s, 8);
  CHECK_EQ(x.timeline_count(), 5);
  CHECK_EQ(x.active_timeline_count(), 4);
  CHECK_EQ(x.origins_unknown(), 3);
  CHECK_EQ(x.origin_count(), std::size_t{0});
  for (std::size_t i = 0; i < x.board_count(); ++i) CHECK_EQ(x.board_parent()[i], -1);
  // W(8) is the four active heads: no origin to add.
  CHECK_EQ(x.window_size(8), std::size_t{4});
}

// ---- E(k): travel destinations outside the window ------------------------------------------

TEST(expand_e_holds_the_history_boards_a_rook_reaches_through_time) {
  // One timeline, h = 2..20 (boards 0..18); a white rook on a1 of the head,
  // every other board empty. Moves: a2-a8 and b1-h1 on the head (14), and back
  // in time along T, one turn (two half-turns) a step over White's boards
  // (L4): h = 18, 16, ..., 2 - nine history boards, each move branching (J3).
  const GameState s = state_of({timeline(0, 2, boards(2, 20, {{20, board({"Ra1"})}}))});
  const Expansion x = expanded(s, 4);
  REQUIRE_EQ(x.moves().size(), std::size_t{23});
  CHECK_EQ(x.move_edge_count(), std::size_t{23});
  int branching = 0;
  for (const MoveRecord& r : x.move_records()) {
    if ((r.flags & T5D_MOVE_BRANCHING) != 0) {
      ++branching;
      CHECK_EQ(r.flags, static_cast<std::uint8_t>(T5D_MOVE_BRANCHING | T5D_MOVE_TRAVEL));
      CHECK_EQ(r.to_board, x.board_index(r.to_timeline, r.to_half_turn));
    }
  }
  CHECK_EQ(branching, 9);
  // The history board at h lies 20 - h half-turns behind the head. The moves
  // on the head itself are no travel: the head is no travel destination.
  CHECK_EQ(text(x.travel_destinations()), std::string("0,2,4,6,8,10,12,14,16"));
  CHECK_EQ(text(x.e_boards()), std::string("0,2,4,6,8,10,12"));  // depth > 4: h <= 14
  CHECK_EQ(e_text(x, 0), std::string("0,2,4,6,8,10,12,14,16"));
  CHECK_EQ(x.e_count(0), std::size_t{9});
  CHECK_EQ(x.e_count(4), std::size_t{7});
  CHECK_EQ(x.e_count(8), std::size_t{5});
  CHECK_EQ(x.e_count(16), std::size_t{1});  // h = 2 only
  CHECK_EQ(x.e_count(18), std::size_t{0});
  // The only source is the head, in the window.
  CHECK_EQ(text(x.source_boards()), std::string("18"));
  CHECK_EQ(x.src_extra_count(0), std::size_t{0});
}

TEST(expand_e_holds_the_boards_a_knight_leaps_to) {
  // A white knight on d4 of the head (h = 20): eight leaps within the board;
  // through time, two squares along a file or rank and one turn back (h = 18:
  // b4, f4, d2, d6), or one square and two turns back (h = 16: c4, e4, d3, d5).
  const GameState s = state_of({timeline(0, 2, boards(2, 20, {{20, board({"Nd4"})}}))});
  const Expansion x = expanded(s, 2);
  CHECK_EQ(x.moves().size(), std::size_t{16});
  CHECK_EQ(text(x.travel_destinations()), std::string("14,16"));  // h = 16, 18 (not the head)
  CHECK_EQ(e_text(x, 0), std::string("14,16"));
  CHECK_EQ(text(x.e_boards()), std::string("14"));  // k = 2: h = 18 is in the window
  CHECK_EQ(x.e_count(4), std::size_t{0});
}

TEST(expand_e_holds_the_history_board_of_a_branching_capture) {
  // The rook of the earlier test, and a black pawn on a1 at h = 12: the slide
  // back in time takes it there and stops (J7). Boards: h = 2 + i.
  const GameState s =
      state_of({timeline(0, 2, boards(2, 20, {{20, board({"Ra1"})}, {12, board({"pa1"})}}))});
  const Expansion x = expanded(s, 4, true);
  REQUIRE_EQ(x.moves().size(), std::size_t{18});  // 14 on the board, h = 18, 16, 14, 12
  const auto capture = std::find_if(x.move_records().begin(), x.move_records().end(),
                                    [](const MoveRecord& r) { return r.captured != 0; });
  REQUIRE(capture != x.move_records().end());
  CHECK_EQ(capture->to_board, 10);  // the history board itself, not the board it creates
  CHECK_EQ(capture->to_half_turn, 12);
  CHECK_EQ(capture->captured, std::uint8_t{1});  // a pawn
  CHECK_EQ(capture->flags, static_cast<std::uint8_t>(T5D_MOVE_CAPTURE | T5D_MOVE_BRANCHING |
                                                    T5D_MOVE_TRAVEL));
  CHECK_EQ(text(x.e_boards()), std::string("10,12"));  // h = 12 and 14: depth 8 and 6
  CHECK_EQ(x.e_count(8), std::size_t{0});
  // The attack: from the rook's square to the pawn's, across boards.
  CHECK(relation_set(x).count(Relation{18, sq("a1"), 10, sq("a1"), kRelationAttack}) == 1);
}

TEST(expand_a_jump_onto_another_active_head_is_no_e_board) {
  // Timeline 0 (boards 0..10, h 2..12) and White's +1 from (0, 6) (boards
  // 11..16, h 7..12). A white rook on a1 of 0's head, a white pawn on a1 at
  // h = 10 (blocking the slide back in time: J7). The rook jumps to +1's head,
  // a playable board: no branch (J4), and the board is in the window.
  const GameState s = state_of({
      timeline(0, 2, boards(2, 12, {{12, board({"Ra1"})}, {10, board({"Pa1"})}})),
      timeline(1, 7, boards(7, 12), Colour::White, 1, BoardCoord{0, 6}),
  });
  const Expansion x = expanded(s, 0);
  REQUIRE_EQ(x.moves().size(), std::size_t{15});  // 14 on the board, the jump
  const auto jump =
      std::find_if(x.move_records().begin(), x.move_records().end(),
                   [](const MoveRecord& r) { return (r.flags & T5D_MOVE_TRAVEL) != 0; });
  REQUIRE(jump != x.move_records().end());
  CHECK_EQ(jump->from_board, 10);
  CHECK_EQ(jump->to_board, 16);
  CHECK_EQ(jump->to_timeline, 1);
  CHECK_EQ(jump->to_half_turn, 12);
  CHECK_EQ(jump->flags, static_cast<std::uint8_t>(T5D_MOVE_TRAVEL));
  CHECK_EQ(text(x.travel_destinations()), std::string("16"));  // not 10: moves within a board
  CHECK(x.e_boards().empty());
  CHECK_EQ(x.e_count(0), std::size_t{0});
}

TEST(expand_e_holds_boards_of_an_inactive_timeline) {
  // As in the inactive-band test: 0 (boards 0..10), +1 (11..16, h 7..12), +2
  // inactive (17..20, h 9..12). A white rook on a1 of +1's head (board 16):
  // 14 moves on the board; jumps to 0's head (board 10, in W) and to +2's head
  // (board 20, inactive: E at every k); back in time on +1 to h = 10 (board 14)
  // and h = 8 (board 12), branching; +1 has no board at h = 6.
  const GameState s = state_of({
      timeline(0, 2, boards(2, 12)),
      timeline(1, 7, boards(7, 12, {{12, board({"Ra1"})}}), Colour::White, 1, BoardCoord{0, 6}),
      timeline(2, 9, boards(9, 12), Colour::White, 2, BoardCoord{0, 8}),
  });
  const Expansion x = expanded(s, 2);
  CHECK_EQ(x.moves().size(), std::size_t{18});
  CHECK_EQ(text(x.travel_destinations()), std::string("10,12,14,20"));  // not its own 16
  CHECK_EQ(e_text(x, 0), std::string("12,14,20"));
  CHECK_EQ(text(x.e_boards()), std::string("12,20"));
  CHECK_EQ(e_text(x, 4), std::string("20"));
  CHECK_EQ(e_text(x, 16), std::string("20"));
  CHECK(x.src_extra().empty());
}

TEST(expand_src_extra_holds_sources_outside_the_window) {
  // The same timelines; only a moved white pawn on a8 of +2's head (board 20,
  // inactive). Its one move: forward along the timeline axis (White: towards
  // lower L, 4.6) to a8 of +1's head, a playable board - arriving on its last
  // rank, so promoting (S8). The source board is in no window and receives no
  // move: it is in src_extra, not in E.
  const GameState s = state_of({
      timeline(0, 2, boards(2, 12)),
      timeline(1, 7, boards(7, 12), Colour::White, 1, BoardCoord{0, 6}),
      timeline(2, 9, boards(9, 12, {{12, board({"Pa8"})}}), Colour::White, 2, BoardCoord{0, 8}),
  });
  const Expansion x = expanded(s, 4);
  REQUIRE_EQ(x.moves().size(), std::size_t{1});
  const MoveRecord r = x.move_records()[0];
  CHECK_EQ(r.from_board, 20);
  CHECK_EQ(r.to_board, 16);
  CHECK_EQ(r.promotion, std::uint8_t{5});  // a queen
  CHECK_EQ(r.flags, static_cast<std::uint8_t>(T5D_MOVE_PROMOTION | T5D_MOVE_TRAVEL));
  CHECK(x.e_boards().empty());
  CHECK_EQ(text(x.src_extra()), std::string("20"));
  for (const int k : {0, 4, 8, 16}) {
    CHECK_EQ(x.src_extra_count(k), std::size_t{1});
    CHECK_EQ(x.src_only_count(k), std::size_t{1});
    CHECK_EQ(src_extra_text(x, k), std::string("20"));
  }
  // The Scope features hold the source board too: W(4) = 0's band h 8..12
  // (boards 6..10) with +1's origin (0, 6) = 4, and +1's band h 8..12 (12..16);
  // E(4) empty; src_extra(4) = {20}.
  const Expansion scope = expanded(s, 4, false, SquareFeatures::Scope);
  CHECK_EQ(text(scope.feature_boards()), std::string("4,6,7,8,9,10,12,13,14,15,16,20"));
  REQUIRE_EQ(scope.square_pieces().size(), std::size_t{12 * 64});
  CHECK_EQ(scope.square_pieces()[11 * 64 + static_cast<std::size_t>(sq("a8"))], std::uint8_t{1});
}

TEST(expand_moves_within_a_band_less_board_leave_it_out_of_e) {
  // A white knight on c3 of +2's head (board 20, h 12; +2 inactive). Its eight
  // leaps within the board are no travel (§10.4.1: E is the out-of-window
  // TRAVEL destinations), so board 20 is in src_extra, not in E. Its travel
  // (two axes, +-2 and +-1, one of them L or T; White's boards only, L4):
  //  * one turn back on +2 (h 10, board 18) by a3, e3, c1, c5 - a history
  //    board of the inactive timeline, branching;
  //  * one timeline over to +1's head (h 12, board 16) by a3, e3, c1, c5;
  //  * two over to 0's head (board 10) by b3, d3, c2, c4;
  //  * two over and one turn back: (0, h 10) = board 8, by c3;
  //  * one over and two turns back: (+1, h 8) = board 12, by c3.
  // W(8): 0's band h 4..12 (boards 2..10; its origin (0, 6) = 4 inside) and
  // +1's whole band (11..16). E(8) = {18}; E(0) = {8, 12, 18}.
  const GameState s = state_of({
      timeline(0, 2, boards(2, 12)),
      timeline(1, 7, boards(7, 12), Colour::White, 1, BoardCoord{0, 6}),
      timeline(2, 9, boards(9, 12, {{12, board({"Nc3"})}}), Colour::White, 2, BoardCoord{0, 8}),
  });
  const Expansion x = expanded(s, 8);
  CHECK_EQ(x.moves().size(), std::size_t{22});  // 8 within, 4 + 4 + 4 + 1 + 1 travelling
  CHECK_EQ(text(x.travel_destinations()), std::string("8,10,12,16,18"));
  CHECK_EQ(text(x.e_boards()), std::string("18"));
  CHECK_EQ(e_text(x, 0), std::string("8,12,18"));
  CHECK_EQ(text(x.src_extra()), std::string("20"));
  for (const int k : {0, 4, 8, 16}) {
    CHECK_EQ(x.src_extra_count(k), std::size_t{1});
    CHECK_EQ(x.src_only_count(k), std::size_t{1});  // board 20: a source, no destination
  }
}

// ---- the move graph and the records ------------------------------------------------------------

TEST(expand_move_edges_merge_promotions) {
  // A moved white pawn on b7, a black rook on c8; promotions to Q, R, B, N
  // (S8, the ruleset's set): b8 and bxc8, four moves each - two edges, one
  // attack.
  RuleOptions r;
  r.promotions = PromotionSet::of({PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop,
                                   PieceKind::Knight});
  const GameState s = state_of({timeline(0, 2, {board({"Pb7", "rc8"})})}, Colour::White, r);
  const Expansion x = expanded(s, 8, true);
  REQUIRE_EQ(x.moves().size(), std::size_t{8});
  CHECK_EQ(x.move_edge_count(), std::size_t{2});
  const std::span<const std::int32_t> e = x.move_edges();
  CHECK_EQ(text(e), std::to_string(0) + "," + std::to_string(sq("b7")) + ",0," +
                        std::to_string(sq("b8")) + ",0," + std::to_string(sq("b7")) + ",0," +
                        std::to_string(sq("c8")));
  std::set<int> promotions;
  for (const MoveRecord& m : x.move_records()) promotions.insert(m.promotion);
  CHECK(promotions == std::set<int>({2, 3, 4, 5}));  // N B R Q
  CHECK_EQ(x.relation_count(), std::size_t{1});
  CHECK(relation_set(x).count(Relation{0, sq("b7"), 0, sq("c8"), kRelationAttack}) == 1);
}

TEST(expand_records_carry_squares_cells_and_pieces) {
  // A 5x5 board: a square is rank * 8 + file, a cell rank * 5 + file. A white
  // knight on b3: square 17, cell 11.
  const GameState s = state_of({timeline(0, 2, {board({"Nb3", "Ke1", "ka5"}, {5, 5})})},
                               Colour::White, {}, BoardSize{5, 5});
  const Expansion x = expanded(s);
  bool seen = false;
  for (const MoveRecord& r : x.move_records()) {
    if (r.piece != 2) continue;  // the knight
    seen = true;
    CHECK_EQ(r.from_square, static_cast<std::int8_t>(17));
    CHECK_EQ(r.from_cell, static_cast<std::int8_t>(11));
    CHECK_EQ(r.to_cell, static_cast<std::int8_t>(rank_of(r.to_square) * 5 + file_of(r.to_square)));
    CHECK_EQ(r.from_board, 0);
    CHECK_EQ(r.to_board, 0);
  }
  CHECK(seen);
}

// ---- typed relations (provisional, B11) ------------------------------------------------------

TEST(expand_typed_relations_on_one_board) {
  // White: rook a1, knight a3, pawn c2 (moved). Black: pawns b5 and a6.
  //  * the rook up the a-file: a2, then its own knight (defence a1-a3), then
  //    past it as if empty to the black pawn on a6 (x-ray a1-a6);
  //  * the knight: takes b5 (attack a3-b5), reaches its own pawn (defence
  //    a3-c2), c4 and b1;
  //  * the pawn: c3; nothing on b3 or d3 to defend.
  const GameState s = state_of({timeline(0, 2, {board({"Ra1", "Na3", "Pc2", "pb5", "pa6"})})});
  const Expansion x = expanded(s, 8, true);
  CHECK_EQ(x.moves().size(), std::size_t{12});
  const std::set<Relation> expected = {
      {0, sq("a3"), 0, sq("b5"), kRelationAttack},
      {0, sq("a1"), 0, sq("a3"), kRelationDefence},
      {0, sq("a3"), 0, sq("c2"), kRelationDefence},
      {0, sq("a1"), 0, sq("a6"), kRelationXRay},
  };
  CHECK(relation_set(x) == expected);
  CHECK_EQ(x.relation_count(), std::size_t{4});
  // Attacks come first.
  CHECK_EQ(x.relations()[4], kRelationAttack);
  // Each kind on its own.
  ExpandOptions o;
  o.defence = true;
  Expansion only;
  expand(s, o, only);
  CHECK_EQ(only.relation_count(), std::size_t{2});
  for (std::size_t i = 0; i < only.relation_count(); ++i) {
    CHECK_EQ(only.relations()[5 * i + 4], kRelationDefence);
  }
  // Off by default.
  CHECK_EQ(expanded(s).relation_count(), std::size_t{0});
}

TEST(expand_typed_relations_through_time) {
  // One timeline, h = 2..10 (boards 0..8, board h - 2); the white rook on a1
  // of the head. Back in time over White's boards: h = 8 empty (a move), h = 6
  // a white knight (defence), past it h = 4 empty and h = 2 a black rook
  // (x-ray).
  const GameState s = state_of({timeline(
      0, 2, boards(2, 10, {{10, board({"Ra1"})}, {6, board({"Na1"})}, {2, board({"ra1"})}}))});
  const Expansion x = expanded(s, 8, true);
  CHECK_EQ(x.moves().size(), std::size_t{15});
  const std::set<Relation> expected = {
      {8, sq("a1"), 4, sq("a1"), kRelationDefence},
      {8, sq("a1"), 0, sq("a1"), kRelationXRay},
  };
  CHECK(relation_set(x) == expected);
}

TEST(expand_relations_pass_past_selves_under_q19) {
  // One timeline, h = 2..10 (board h - 2); a white rook on a1 of the head
  // (board 8). Back in time: h = 8 a white rook (the piece's past self), h = 6
  // a white knight, h = 4 a white rook again, h = 2 a black rook.
  //  * J7 (the default): the past self at h = 8 blocks - defence of it - and
  //    the x-ray passes it to the knight.
  //  * Q19's alternative: a slide straight back in time passes pieces of its
  //    own kind and colour - the defence is of the knight, and the x-ray
  //    passes the knight AND the past self at h = 4, to the black rook.
  const auto position = [](const RuleOptions& rules) {
    return state_of({timeline(0, 2,
                              boards(2, 10,
                                     {{10, board({"Ra1"})},
                                      {8, board({"Ra1"})},
                                      {6, board({"Na1"})},
                                      {4, board({"Ra1"})},
                                      {2, board({"ra1"})}}))},
                    Colour::White, rules);
  };
  const Expansion blocks = expanded(position({}), 8, true);
  CHECK(relation_set(blocks) == std::set<Relation>({{8, sq("a1"), 6, sq("a1"), kRelationDefence},
                                                    {8, sq("a1"), 4, sq("a1"), kRelationXRay}}));
  RuleOptions q19;
  q19.past_self_blocks = false;
  const Expansion passes = expanded(position(q19), 8, true);
  CHECK(relation_set(passes) == std::set<Relation>({{8, sq("a1"), 4, sq("a1"), kRelationDefence},
                                                    {8, sq("a1"), 0, sq("a1"), kRelationXRay}}));
  CHECK_EQ(passes.moves().size(), blocks.moves().size());  // 14 on the board either way
}

TEST(expand_relations_reach_only_destinations_under_q8_and_q20) {
  // Q8's alternative: a turn-zero history board is no destination. Black to
  // move; one timeline, h = 1 (T0, black) .. 7 (board (h - 1)): a black rook
  // on a8 of the head (board 6), a black knight on a8 at h = 3 (board 2), a
  // white rook on a8 at h = 1 (board 0, T0).
  RuleOptions q8;
  q8.turn_zero_reachable = false;
  const auto t0 = [](const RuleOptions& rules) {
    return state_of(
        {timeline(0, 1, boards(1, 7, {{7, board({"ra8"})}, {3, board({"na8"})}, {1, board({"Ra8"})}}))},
        Colour::Black, rules);
  };
  const Relation defence{6, sq("a8"), 2, sq("a8"), kRelationDefence};
  const Relation xray{6, sq("a8"), 0, sq("a8"), kRelationXRay};
  CHECK(relation_set(expanded(t0({}), 8, true)) == std::set<Relation>({defence, xray}));
  CHECK(relation_set(expanded(t0(q8), 8, true)) == std::set<Relation>({defence}));
  // The defended piece itself on the T0 board: h = 1..3, the rook on the head
  // (h 3, board 2), its knight at h = 1 (board 0).
  const auto t0_defence = [](const RuleOptions& rules) {
    return state_of({timeline(0, 1, boards(1, 3, {{3, board({"ra8"})}, {1, board({"na8"})}}))},
                    Colour::Black, rules);
  };
  CHECK(relation_set(expanded(t0_defence({}), 8, true)) ==
        std::set<Relation>({{2, sq("a8"), 0, sq("a8"), kRelationDefence}}));
  CHECK(relation_set(expanded(t0_defence(q8), 8, true)).empty());

  // Q20's alternative: a history board of an inactive timeline is no
  // destination. 0 (h 2..12, boards 0..10), White's +1 from (0, 6) (h 7..12,
  // 11..16), White's second, +2, inactive, from (0, 4) (h 5..12, 17..24). A
  // white rook on a1 of +2's head (board 24; inactive boards may be played,
  // T6); back in time on +2 a white knight (h 10, board 22), then a black pawn
  // (h 6, board 18).
  RuleOptions q20;
  q20.inactive_history_is_destination = false;
  const auto inactive = [](const RuleOptions& rules) {
    return state_of(
        {
            timeline(0, 2, boards(2, 12)),
            timeline(1, 7, boards(7, 12), Colour::White, 1, BoardCoord{0, 6}),
            timeline(2, 5, boards(5, 12, {{12, board({"Ra1"})}, {10, board({"Na1"})}, {6, board({"pa1"})}}),
                     Colour::White, 2, BoardCoord{0, 4}),
        },
        Colour::White, rules);
  };
  REQUIRE(!inactive({}).active(2));
  const Expansion presumed = expanded(inactive({}), 8, true);
  CHECK(relation_set(presumed) ==
        std::set<Relation>({{24, sq("a1"), 22, sq("a1"), kRelationDefence},
                            {24, sq("a1"), 18, sq("a1"), kRelationXRay}}));
  const Expansion refused = expanded(inactive(q20), 8, true);
  CHECK(relation_set(refused).empty());
  // The moves are the same: the rook's slides along L land on heads.
  CHECK_EQ(refused.moves().size(), presumed.moves().size());
}

TEST(expand_en_passant_attacks_the_pawn_taken) {
  // A white pawn on e5, a black pawn on d5 that has just double-stepped: the
  // board's en-passant square is d6 (S7). The capture moves to d6 and takes d5.
  const GameState s =
      state_of({timeline(0, 2, {board({"Pe5", "pd5"}, {8, 8}, sq("d6"))})});
  const Expansion x = expanded(s, 8, true);
  const auto ep =
      std::find_if(x.move_records().begin(), x.move_records().end(),
                   [](const MoveRecord& r) { return (r.flags & T5D_MOVE_EN_PASSANT) != 0; });
  REQUIRE(ep != x.move_records().end());
  CHECK_EQ(ep->to_cell, static_cast<std::int8_t>(sq("d6")));
  CHECK_EQ(ep->captured, std::uint8_t{1});
  CHECK(relation_set(x) == std::set<Relation>({{0, sq("e5"), 0, sq("d5"), kRelationAttack}}));
}

TEST(expand_refuses_a_negative_k) {
  const GameState s = state_of({timeline(0, 2, {board({"Ra1"})})});
  Expansion x;
  ExpandOptions o;
  o.window_k = -1;
  CHECK_THROWS_AS(expand(s, o, x), std::invalid_argument);
}

// ---- features ---------------------------------------------------------------------------------

TEST(expand_square_features_use_the_networks_cells) {
  // 5x5: a white king on e1 without its unmoved marker (cell 4: has-moved 1),
  // a black queen on a5 (cell 20: queens carry no marker, has-moved 0), an
  // unmoved white pawn on c2 (cell 7: 0), a moved one on d2 (cell 8: 1), a
  // white knight on b3 (cell 11: 0).
  const GameState s =
      state_of({timeline(0, 2, {board({"Ke1", "qa5", "P*c2", "Pd2", "Nb3"}, {5, 5})})},
               Colour::White, {}, BoardSize{5, 5});
  const Expansion x = expanded(s);
  REQUIRE_EQ(x.square_pieces().size(), std::size_t{25});
  CHECK_EQ(text(x.feature_boards()), std::string("0"));
  const auto at = [&](std::size_t cell) {
    return std::array<int, 3>{x.square_pieces()[cell], x.square_colours()[cell],
                              x.square_moved()[cell]};
  };
  CHECK(at(4) == (std::array<int, 3>{6, 1, 1}));
  CHECK(at(20) == (std::array<int, 3>{5, 2, 0}));
  CHECK(at(7) == (std::array<int, 3>{1, 1, 0}));
  CHECK(at(8) == (std::array<int, 3>{1, 1, 1}));
  CHECK(at(11) == (std::array<int, 3>{2, 1, 0}));
  int pieces = 0;
  for (std::size_t c = 0; c < 25; ++c) pieces += x.square_pieces()[c] != 0 ? 1 : 0;
  CHECK_EQ(pieces, 5);
}

TEST(expand_square_features_of_every_piece_on_an_8x8_board) {
  // 8x8 boards take the sixteen-cells-at-a-time decoding; every piece a board
  // can hold - 12 kinds x 2 colours, and the unmoved pawn, rook, king and
  // brawn of each colour (the four kinds that carry the flag; t5d::Piece
  // drops it on the rest) - against the Piece API, in two arrangements so
  // that each piece sits in two different sixteen-cell blocks.
  std::vector<Piece> all;
  for (int code = 1; code <= Piece::kMaxCode; ++code) {
    const Piece p = Piece::from_code(code);
    if (p.code() == code) all.push_back(p);  // the codes a board can hold
  }
  REQUIRE_EQ(all.size(), std::size_t{32});
  BoardBuilder first({8, 8});
  BoardBuilder second({8, 8});
  for (std::size_t i = 0; i < all.size(); ++i) {
    first.put(static_cast<int>(2 * i), all[i]);        // even squares
    second.put(static_cast<int>(63 - 2 * i), all[i]);  // odd squares, reversed
  }
  // h 2 and h 3: Black is to move on the head.
  const GameState s =
      state_of({timeline(0, 2, {first.publish(), second.publish()})}, Colour::Black);
  const Expansion x = expanded(s, 8);
  REQUIRE_EQ(x.square_pieces().size(), std::size_t{2 * 64});
  int brawns_moved = 0;
  for (std::size_t b = 0; b < 2; ++b) {
    const Board& on = *s.multiverse().board_at(0, 2 + static_cast<int>(b));
    for (int square = 0; square < 64; ++square) {
      const Piece p = on.at(square);
      const std::size_t cell = b * 64 + static_cast<std::size_t>(square);
      const int piece = p.empty() ? 0 : index_of(p.kind()) + 1;
      const int colour = p.empty() ? 0 : index_of(p.colour()) + 1;
      const int moved = !p.empty() && tracks_unmoved(p.kind()) && !p.unmoved() ? 1 : 0;
      CHECK_EQ(static_cast<int>(x.square_pieces()[cell]), piece);
      CHECK_EQ(static_cast<int>(x.square_colours()[cell]), colour);
      CHECK_EQ(static_cast<int>(x.square_moved()[cell]), moved);
      if (!p.empty() && p.kind() == PieceKind::Brawn) brawns_moved += moved;
    }
  }
  CHECK_EQ(brawns_moved, 4);  // each colour's moved brawn, on both boards
}

TEST(expand_scope_features_are_those_of_the_window_e_and_src_extra) {
  // The rook through time (E(4) = boards 0..12, even), W(4) = boards 14..18.
  const GameState s = state_of({timeline(0, 2, boards(2, 20, {{20, board({"Ra1"})}}))});
  const Expansion all = expanded(s, 4, false, SquareFeatures::All);
  const Expansion scope = expanded(s, 4, false, SquareFeatures::Scope);
  CHECK_EQ(text(scope.feature_boards()), std::string("0,2,4,6,8,10,12,14,15,16,17,18"));
  REQUIRE_EQ(scope.square_pieces().size(), scope.feature_boards().size() * 64);
  for (std::size_t i = 0; i < scope.feature_boards().size(); ++i) {
    const auto b = static_cast<std::size_t>(scope.feature_boards()[i]);
    for (std::size_t c = 0; c < 64; ++c) {
      CHECK_EQ(scope.square_pieces()[i * 64 + c], all.square_pieces()[b * 64 + c]);
      CHECK_EQ(scope.square_moved()[i * 64 + c], all.square_moved()[b * 64 + c]);
    }
  }
  const Expansion none = expanded(s, 4, false, SquareFeatures::None);
  CHECK(none.feature_boards().empty());
  CHECK(none.square_pieces().empty());
  CHECK_EQ(none.moves().size(), all.moves().size());
}

TEST(expand_reuses_one_expansion_for_different_positions) {
  // A larger position, then a smaller one, then the larger again: every
  // array describes the position of the latest call.
  const GameState big = state_of({timeline(0, 2, boards(2, 20, {{20, board({"Ra1"})}}))});
  const GameState small = state_of({timeline(0, 2, {board({"Nb1"})})});
  Expansion x;
  ExpandOptions o;
  o.attack = o.defence = o.xray = true;
  expand(big, o, x);
  CHECK_EQ(x.board_half_turns().size(), std::size_t{19});  // the per-board arrays, computed
  expand(small, o, x);
  CHECK_EQ(x.board_count(), std::size_t{1});
  // Computed again for the new position, not left from the last one.
  CHECK_EQ(x.board_half_turns().size(), std::size_t{1});
  CHECK_EQ(x.board_is_head()[0], std::uint8_t{1});
  CHECK_EQ(x.board_playable()[0], std::uint8_t{1});
  CHECK_EQ(x.moves().size(), std::size_t{3});  // a3, c3, d2
  CHECK_EQ(x.move_records().size(), std::size_t{3});
  CHECK_EQ(x.move_edge_count(), std::size_t{3});
  CHECK_EQ(x.square_pieces().size(), std::size_t{64});
  CHECK(x.e_boards().empty());
  expand(big, o, x);
  CHECK_EQ(x.board_count(), std::size_t{19});
  CHECK_EQ(x.moves().size(), std::size_t{23});
  CHECK_EQ(x.e_count(4), std::size_t{7});
}

TEST(expand_the_acting_player_and_the_submit_action_d8) {
  // Register D8 (rulebook R8): turns have variable length, so submitting is an
  // action - possible once the obligations are met (T5), legal when C4 holds
  // too - and the acting player is an input of its own.
  //
  // White rooks on a1 of 0 (T1, the present: mandatory) and of +1 (T2, ahead:
  // optional - +1 is White's first creation, active, T10). No royal piece, so
  // every submission is safe (C13).
  GameState s = t5d_test::game({"[5/5/5/5/R4:0:1:w]", "[5/5/5/5/R4:+1:2:w]"}, BoardSize{5, 5});
  Expansion x;
  expand(s, ExpandOptions{}, x);
  CHECK(x.acting_player() == Colour::White);
  CHECK_EQ(x.present(), 2);  // (T1, White)
  CHECK(!x.submit_possible());
  CHECK(!x.submit_legal());
  // 0's rook moves: the present shifts to (T1, Black) - Black's colour - but
  // the action is still White's, and +1's rook may still move (R8).
  t5d_test::play(s, "(0T1)Ra1(0T1)a2");
  expand(s, ExpandOptions{}, x);
  CHECK(x.acting_player() == Colour::White);
  CHECK_EQ(x.present(), 3);
  CHECK(x.submit_possible());
  CHECK(x.submit_legal());
  CHECK(x.may_move());
  CHECK_EQ(x.move_records().size(), std::size_t{8});  // the rook on (+1T2)a1: a2-a5, b1-e1
  // The per-board features are those of any state: +1's head is playable.
  const int plus_one = x.board_index(1, 4);
  REQUIRE(plus_one >= 0);
  CHECK_EQ(x.board_playable()[static_cast<std::size_t>(plus_one)], std::uint8_t{1});
  CHECK_EQ(x.board_side_to_move()[static_cast<std::size_t>(plus_one)], std::uint8_t{0});
  t5d_test::play(s, "(+1T2)Ra1(+1T2)a2");
  expand(s, ExpandOptions{}, x);
  CHECK(x.submit_possible());
  CHECK(x.submit_legal());
  CHECK(x.move_records().empty());  // every board played: submitting is all that is left
  // Under Q4's Guide-literal reading the action ends at the shift: submitting
  // is possible and legal, and no move is left.
  RuleOptions guide;
  guide.moves_after_shift = false;
  GameState g = t5d_test::game({"[5/5/5/5/R4:0:1:w]", "[5/5/5/5/R4:+1:2:w]"}, BoardSize{5, 5},
                               guide);
  t5d_test::play(g, "(0T1)Ra1(0T1)a2");
  expand(g, ExpandOptions{}, x);
  CHECK(x.acting_player() == Colour::White);
  CHECK(x.submit_possible());
  CHECK(x.submit_legal());
  CHECK(!x.may_move());
  CHECK(x.move_records().empty());

  // Possible but not legal: a black rook on b5 commands the b-file. White's
  // king steps to b1 (pseudo-legal; moving into check is allowed during an
  // action, R1): the present has shifted, but the rook can capture the king
  // (C4) - the submit action is illegal. Stepping to a2 instead, it is legal.
  const GameState k = t5d_test::game({"[1r2k/5/5/5/K4:0:1:w]"}, BoardSize{5, 5});
  GameState bad = k;
  t5d_test::play(bad, "(0T1)Ka1(0T1)b1");
  expand(bad, ExpandOptions{}, x);
  CHECK(x.acting_player() == Colour::White);
  CHECK(x.submit_possible());
  CHECK(!x.submit_legal());
  CHECK(bad.submission_status() == SubmitStatus::Unsafe);
  GameState good = k;
  t5d_test::play(good, "(0T1)Ka1(0T1)a2");
  expand(good, ExpandOptions{}, x);
  CHECK(x.submit_possible());
  CHECK(x.submit_legal());
  // Black's action: the acting player is Black, the present White's once it
  // has shifted.
  good.submit();
  expand(good, ExpandOptions{}, x);
  CHECK(x.acting_player() == Colour::Black);
  CHECK(!x.submit_possible());
  t5d_test::play(good, "(0T1)Ke5(0T1)e4");
  expand(good, ExpandOptions{}, x);
  CHECK(x.acting_player() == Colour::Black);
  CHECK_EQ(x.present() & 1, 0);
  CHECK(x.submit_possible());
  CHECK(x.submit_legal());
}
