// The two views of a decision state (t5d/views.hpp): the pin and check
// relations of the expansion (t5d/expand.hpp, (e); register B11) and the
// per-square labels (register D4, M20, A7), on positions built by hand. Every
// expectation is worked out here from the rules reference (docs/rules/README.md)
// and the readings t5d/labels.hpp names (build-log spec issues 172-176) - board
// by board, square by square - never recorded from a run. Then, on random play,
// the check relation against check detection itself (R1, H36: check edges from
// the opponent exactly when in_check at a turn start) and the labels' own
// invariants.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/capi.h"
#include "t5d/check.hpp"
#include "t5d/expand.hpp"
#include "t5d/labels.hpp"
#include "t5d/views.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/records.h"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"

using namespace t5d;

namespace {

// A board with pieces written "Ra1" (white rook on a1), "pa6" (black pawn).
BoardRef board(std::initializer_list<std::string_view> pieces, BoardSize size = {8, 8}) {
  BoardBuilder b(size);
  for (std::string_view text : pieces) {
    const char letter = text[0];
    const std::string_view square = text.substr(1);
    const Colour colour = std::isupper(static_cast<unsigned char>(letter)) != 0 ? Colour::White
                                                                              : Colour::Black;
    b.put(square[0] - 'a', square[1] - '1', Piece(*kind_from_letter(letter), colour, false));
  }
  return b.publish();
}

BoardRef empty_board() { return BoardBuilder(BoardSize{8, 8}).publish(); }

NewTimeline timeline(int position, int first, std::vector<BoardRef> bs,
                     std::optional<Colour> creator = std::nullopt, int order = 0) {
  NewTimeline t;
  t.position = position;
  t.first_half_turn = first;
  t.boards = std::move(bs);
  if (creator.has_value()) t.creator = *creator;
  t.creation_order = order;
  return t;
}

GameState state_of(std::vector<NewTimeline> timelines, Colour to_move = Colour::White) {
  Multiverse mv(BoardSize{8, 8}, false);
  for (NewTimeline& t : timelines) mv.add_timeline(std::move(t));
  mv.set_to_move(to_move);
  return GameState(std::move(mv), RuleOptions{});
}

// A relation as (from board, from square name, to board, to square name):
// boards are expansion indices.
using Edge = std::tuple<int, std::string, int, std::string>;

std::string name_of(int cell) {
  std::string s;
  s += static_cast<char>('a' + cell % 8);
  s += static_cast<char>('1' + cell / 8);
  return s;
}

std::set<Edge> edges_of(const Expansion& x, int id) {
  std::set<Edge> out;
  const std::span<const std::int32_t> r = x.relations();
  for (std::size_t i = 0; i + 4 < r.size(); i += 5) {
    if (r[i + 4] == id) out.insert({r[i], name_of(r[i + 1]), r[i + 2], name_of(r[i + 3])});
  }
  return out;
}

Expansion expanded(const GameState& s, int moves_made = 0) {
  ExpandOptions o;
  o.square_features = SquareFeatures::None;
  o.attack = o.defence = o.xray = o.pin = o.check = true;
  Expansion x;
  expand(s, o, x, moves_made);
  return x;
}

// The squares (board, name) on which a label is set.
using Square = std::pair<int, std::string>;

std::set<Square> labelled(const SquareLabels& l, int label, std::uint8_t value = 1) {
  std::set<Square> out;
  for (std::size_t b = 0; b < l.board_count; ++b) {
    for (int c = 0; c < l.cells(); ++c) {
      if (l.at(static_cast<int>(b), c, label) == value) out.insert({static_cast<int>(b), name_of(c)});
    }
  }
  return out;
}

SquareLabels labels_of(const GameState& s) {
  SquareLabels l;
  SquareLabelOptions o;
  o.mate_budget = SearchBudget::unlimited();
  square_labels(s, o, l);
  return l;
}

}  // namespace

// ---- the opponent's view -------------------------------------------------------------------

TEST(views_the_opponents_view_is_check_status_pass) {
  // Two timelines in the present (h = 2), both White's: both passed (R1), the
  // copies at h = 3 reported at h = 2.
  const GameState s = state_of({timeline(0, 2, {board({"Ke1"})}),
                                timeline(1, 2, {board({"re8"})}, Colour::White, 1)});
  const PassView v = pass_view(s);
  CHECK(v.pass_defined);
  CHECK(v.passed == std::vector<int>({0, 1}));
  CHECK_EQ(v.present, 2);
  CHECK(v.multiverse.board_at(1, 3) != nullptr);
  CHECK_EQ(v.state_half_turn(1, 3), 2);
  CHECK_EQ(v.state_half_turn(1, 1), 1);
  CHECK_EQ(v.state_half_turn(1, 4), 4);
}

// ---- pins ----------------------------------------------------------------------------------

TEST(views_pin_by_the_acting_player_on_its_board) {
  // White to move: the rook on a1 pins the knight on a4 to the king on a8.
  const GameState s = state_of({timeline(0, 2, {board({"Ra1", "na4", "ka8"})})});
  const Expansion x = expanded(s);
  CHECK(edges_of(x, kRelationPin) == std::set<Edge>({{0, "a1", 0, "a4"}}));
  CHECK(edges_of(x, kRelationXRay) == std::set<Edge>({{0, "a1", 0, "a8"}}));
  CHECK(edges_of(x, kRelationCheck).empty());
  // Behind a royal or the slider's own piece, nothing is pinned.
  const GameState t = state_of({timeline(0, 2, {board({"Ra1", "Na4", "ka8", "Bc1", "ke3"})})});
  CHECK(edges_of(expanded(t), kRelationPin).empty());
}

TEST(views_pin_of_the_acting_players_piece_in_the_opponents_view) {
  // White to move on (0, h2); the rook on e8 - Black's, playable once White's
  // board is passed (R1) - pins the bishop on e2 to the king on e1. Reported on
  // the board passed.
  const GameState s = state_of({timeline(0, 2, {board({"Ke1", "Be2", "re8"})})});
  const Expansion x = expanded(s);
  CHECK(edges_of(x, kRelationPin) == std::set<Edge>({{0, "e8", 0, "e2"}}));
  CHECK(edges_of(x, kRelationCheck).empty());
  CHECK(!in_check(s));
}

TEST(views_pins_across_timelines_and_through_time) {
  // Three timelines in the present (h = 2): the white rook on L+1 e4 pins the
  // black knight on L0 e4 to the black king on L-1 e4 along the timeline axis.
  {
    const GameState s = state_of({timeline(-1, 2, {board({"ke4"})}, Colour::Black, 1),
                                  timeline(0, 2, {board({"ne4"})}),
                                  timeline(1, 2, {board({"Re4"})}, Colour::White, 1)});
    const Expansion x = expanded(s);
    // Boards: L-1 is 0, L0 is 1, L+1 is 2.
    CHECK(edges_of(x, kRelationPin) == std::set<Edge>({{2, "e4", 1, "e4"}}));
  }
  // Through time: the rook on (0, h4) e1, a black knight on (0, h2) e1 and the
  // black king on (0, h0) e1 - White's history boards (a capture would branch).
  {
    const GameState s = state_of({timeline(
        0, 0, {board({"ke1"}), empty_board(), board({"ne1"}), empty_board(), board({"Re1"})})});
    const Expansion x = expanded(s);
    CHECK(edges_of(x, kRelationPin) == std::set<Edge>({{4, "e1", 2, "e1"}}));
  }
  // Through time, in the opponent's view: the black rook on (0, h4) e8 - read on
  // its passed copy at h5 - slides back along the time axis over Black's boards
  // h3 and h1: the white bishop on (0, h3) e8 is pinned to the white king on
  // (0, h1) e8.
  {
    const GameState s = state_of({timeline(
        0, 0, {empty_board(), board({"Ke8"}), empty_board(), board({"Be8"}), board({"re8"})})});
    const Expansion x = expanded(s);
    CHECK(edges_of(x, kRelationPin) == std::set<Edge>({{4, "e8", 3, "e8"}}));
  }
}

// ---- checks --------------------------------------------------------------------------------

TEST(views_check_on_the_board_is_r1s_check) {
  const GameState s = state_of({timeline(0, 2, {board({"Ke1", "re8"})})});
  CHECK(in_check(s));
  CHECK(edges_of(expanded(s), kRelationCheck) == std::set<Edge>({{0, "e8", 0, "e1"}}));
}

TEST(views_check_across_timelines_and_through_time) {
  // The black rook on L+1 e1 captures the white king on L0 e1 along the
  // timeline axis, once both boards are passed (R1).
  {
    const GameState s = state_of({timeline(0, 2, {board({"Ke1"})}),
                                  timeline(1, 2, {board({"re1"})}, Colour::White, 1)});
    CHECK(in_check(s));
    CHECK(edges_of(expanded(s), kRelationCheck) == std::set<Edge>({{1, "e1", 0, "e1"}}));
  }
  // Through time (C2): the black rook on (0, h2) e1, read on its passed copy at
  // h3, captures the white king on the history board (0, h1) e1 - a capture
  // that would branch.
  {
    const GameState s =
        state_of({timeline(0, 0, {empty_board(), board({"Ke1"}), board({"re1", "Kh8"})})});
    CHECK(in_check(s));
    CHECK(edges_of(expanded(s), kRelationCheck) == std::set<Edge>({{2, "e1", 1, "e1"}}));
  }
}

TEST(views_check_mid_action_passes_the_boards_still_to_play) {
  // Two present boards of White's; White plays on L0, L+1 is still mandatory:
  // the black rook on L+1 still checks the white king there (the pass of the
  // boards unplayed - spec issue 174).
  GameState s = state_of({timeline(0, 2, {board({"Ra1"})}),
                          timeline(1, 2, {board({"Ke1", "re8"})}, Colour::White, 1)});
  MoveList moves;
  append_board_moves(s, 0, moves);
  const auto quiet = std::find_if(moves.begin(), moves.end(), [](const Move& m) { return m.physical(); });
  REQUIRE(quiet != moves.end());
  s.apply(*quiet);
  REQUIRE_EQ(s.mandatory_count(), 1);
  const Expansion x = expanded(s, 1);
  // Boards: L0 h2, L0 h3, L+1 h2.
  CHECK(edges_of(x, kRelationCheck) == std::set<Edge>({{2, "e8", 2, "e1"}}));
}

TEST(views_check_by_the_acting_player_is_cfours_scan) {
  // A black king left capturable by White (no legal submission leaves one):
  // the acting player's check edge, from check detection's scan of the state.
  const GameState s = state_of({timeline(0, 2, {board({"Ra1", "ka8"})})});
  CHECK(edges_of(expanded(s), kRelationCheck) == std::set<Edge>({{0, "a1", 0, "a8"}}));
}

TEST(views_relations_append_after_the_generators_and_need_no_other) {
  // Pin and check alone: no attack, defence or x-ray edge comes with them.
  const GameState s = state_of({timeline(0, 2, {board({"Ke1", "Be2", "re8", "Ra1", "na4", "ka8"})})});
  ExpandOptions o;
  o.pin = o.check = true;
  Expansion x;
  expand(s, o, x);
  std::set<int> ids;
  for (std::size_t i = 4; i < x.relations().size(); i += 5) ids.insert(x.relations()[i]);
  CHECK(ids == std::set<int>({kRelationPin}));
  CHECK(edges_of(x, kRelationPin) == std::set<Edge>({{0, "a1", 0, "a4"}, {0, "e8", 0, "e2"}}));
  CHECK(edges_of(x, kRelationCheck).empty());
}

TEST(views_check_edges_are_check_status_on_random_turn_starts) {
  // R1, H36: at a turn start the opponent's check edges exist exactly when the
  // player to move is in check (it has a royal piece in every start used).
  std::mt19937_64 rng(20261023);
  int starts = 0;
  int checks = 0;
  for (const char* name : {"Standard", "Standard - Two Timelines", "Standard - Turn Zero"}) {
    for (int game = 0; game < 6; ++game) {
      const Variant* v = find_variant(name);
      REQUIRE(v != nullptr);
      GameState s = variant_start(*v);
      for (int action = 0; action < 18; ++action) {
        const Expansion x = expanded(s);
        bool from_opponent = false;
        const std::span<const std::int32_t> r = x.relations();
        for (std::size_t i = 0; i + 4 < r.size(); i += 5) {
          if (r[i + 4] != kRelationCheck) continue;
          // The target is one of the acting player's royals.
          const Board* target = s.multiverse().board_at(x.board_timelines()[static_cast<std::size_t>(r[i + 2])],
                                                        x.board_half_turns()[static_cast<std::size_t>(r[i + 2])]);
          const Piece royal = target->at(make_square(r[i + 3] % 8, r[i + 3] / 8));
          CHECK(royal.royal());
          from_opponent = from_opponent || royal.colour() == s.to_move();
        }
        CHECK_EQ(from_opponent, in_check(s));
        ++starts;
        checks += from_opponent ? 1 : 0;
        // A random legal action: random moves until submitting is legal.
        GameState next = s;
        bool done = false;
        for (int tries = 0; tries < 40 && !done; ++tries) {
          GameState t = s;
          for (int m = 0; m < 12; ++m) {
            if (t.submission_status() == SubmitStatus::Safe) break;
            MoveList moves;
            generate_moves(t, moves);
            if (moves.empty()) break;
            t.apply(moves[std::uniform_int_distribution<std::size_t>(0, moves.size() - 1)(rng)]);
          }
          if (t.submission_status() == SubmitStatus::Safe) {
            t.submit();
            next = t;
            done = true;
          }
        }
        if (!done) break;
        s = next;
      }
    }
  }
  std::printf("  [check] %d turn starts, %d in check\n", starts, checks);
  CHECK(starts > 100);
  CHECK(checks > 0);
}

// ---- the per-square labels -----------------------------------------------------------------

TEST(labels_royal_zones_are_one_king_step_across_boards) {
  // White to move; kings on e1 (White) and e8 (Black) of L0, and an empty L+1:
  // the zone reaches the neighbouring timeline's board (one step along the
  // timeline axis), not void.
  const GameState s = state_of({timeline(0, 2, {board({"Ke1", "ke8"})}),
                                timeline(1, 2, {empty_board()}, Colour::White, 1)});
  const SquareLabels l = labels_of(s);
  CHECK_EQ(l.acting_player, Colour::White);
  const std::set<Square> ours = labelled(l, kRoyalZoneOurs);
  const std::set<Square> expect_ours = {{0, "d1"}, {0, "e1"}, {0, "f1"}, {0, "d2"}, {0, "e2"},
                                        {0, "f2"}, {1, "d1"}, {1, "e1"}, {1, "f1"}, {1, "d2"},
                                        {1, "e2"}, {1, "f2"}};
  CHECK(ours == expect_ours);
  const std::set<Square> expect_theirs = {{0, "d8"}, {0, "e8"}, {0, "f8"}, {0, "d7"}, {0, "e7"},
                                          {0, "f7"}, {1, "d8"}, {1, "e8"}, {1, "f8"}, {1, "d7"},
                                          {1, "e7"}, {1, "f7"}};
  CHECK(labelled(l, kRoyalZoneTheirs) == expect_theirs);
}

TEST(labels_reachable_in_two_on_one_board) {
  // A lone white knight on a1: one move to b3 or c2; two (not one, not a1):
  // from b3 c1 d2 d4 c5 a5, from c2 a3 b4 d4 e3 e1.
  const GameState s = state_of({timeline(0, 2, {board({"Na1"})})});
  const SquareLabels l = labels_of(s);
  const std::set<Square> expect = {{0, "c1"}, {0, "d2"}, {0, "d4"}, {0, "c5"}, {0, "a5"},
                                   {0, "a3"}, {0, "b4"}, {0, "e3"}, {0, "e1"}};
  CHECK(labelled(l, kReachableInTwoOurs) == expect);
  CHECK(labelled(l, kReachableInTwoTheirs).empty());
  CHECK_EQ(l.stats.first_moves[0], std::uint64_t{2});
}

TEST(labels_reachable_in_two_through_time) {
  // A white rook on (0, h2) a1 over an empty (0, h0): one move to a2-a8, b1-h1
  // and back in time to (0, h0) a1; from (0, h0) a1 on along that board's file
  // and rank; from (0, h2) b1 back in time to (0, h0) b1.
  const GameState s = state_of({timeline(0, 0, {empty_board(), empty_board(), board({"Ra1"})})});
  const SquareLabels l = labels_of(s);
  const std::set<Square> two = labelled(l, kReachableInTwoOurs);
  CHECK(two.count({0, "a5"}) == 1);   // (h0) a1 -> a5
  CHECK(two.count({0, "b1"}) == 1);   // (h2) b1 -> (h0) b1
  CHECK(two.count({0, "b5"}) == 0);   // three moves
  CHECK(two.count({0, "a1"}) == 0);   // one move
  CHECK(two.count({2, "b5"}) == 1);   // a1 -> b1 -> b5
  CHECK(two.count({2, "a5"}) == 0);   // one move
  CHECK(two.count({2, "a1"}) == 0);   // its own square
  CHECK(two.count({1, "a1"}) == 0);   // Black's board: never a rook's destination
}

TEST(labels_arrival_gives_check_on_the_board_and_through_time) {
  {
    // The rook from a1 checks the king on h8 from a8 and from h1.
    const GameState s = state_of({timeline(0, 2, {board({"Ra1", "kh8"})})});
    const SquareLabels l = labels_of(s);
    CHECK(labelled(l, kArrivalGivesCheckOurs) == std::set<Square>({{0, "a8"}, {0, "h1"}}));
    CHECK(labelled(l, kArrivalGivesCheckTheirs).empty());
  }
  {
    // The black king on White's history board (0, h0) e8: the rook from (0, h2)
    // e1 checks it from (0, h2) e8 (back along the time axis) and from (0, h0)
    // e1 (up the file).
    const GameState s =
        state_of({timeline(0, 0, {board({"ke8"}), empty_board(), board({"Re1"})})});
    const SquareLabels l = labels_of(s);
    CHECK(labelled(l, kArrivalGivesCheckOurs) == std::set<Square>({{2, "e8"}, {0, "e1"}}));
  }
  {
    // The opponent's view: Black's rook on a8, read on the passed copy, checks
    // the white king on h1 from h8 and from a1. Reported on the board passed.
    const GameState s = state_of({timeline(0, 2, {board({"Kh1", "ra8"})})});
    const SquareLabels l = labels_of(s);
    CHECK(labelled(l, kArrivalGivesCheckTheirs) == std::set<Square>({{0, "h8"}, {0, "a1"}}));
    CHECK(labelled(l, kArrivalGivesCheckOurs).empty());  // the king's moves check nothing
  }
}

TEST(labels_exchange_loses_occupant) {
  // White to move. The black knight on a5 is attacked by the rook on a1:
  // undefended it is lost; defended by the pawn on b6 (in Black's view, the
  // pawn on the board passed) the exchange loses the rook (500 for 300).
  {
    const GameState s = state_of({timeline(0, 2, {board({"Ra1", "na5"})})});
    CHECK(labelled(labels_of(s), kExchangeLosesOccupant) == std::set<Square>({{0, "a5"}}));
  }
  {
    const GameState s = state_of({timeline(0, 2, {board({"Ra1", "na5", "pb6"})})});
    CHECK(labelled(labels_of(s), kExchangeLosesOccupant).empty());
  }
  {
    // The white bishop on c3, attacked by the black knight on b5 (Black's view)
    // and undefended, is lost; defended by the pawn on b2 it is not (300 for
    // 300: nothing won).
    const GameState s = state_of({timeline(0, 2, {board({"Bc3", "nb5"})})});
    CHECK(labelled(labels_of(s), kExchangeLosesOccupant) == std::set<Square>({{0, "c3"}}));
    const GameState t = state_of({timeline(0, 2, {board({"Bc3", "nb5", "Pb2"})})});
    CHECK(labelled(labels_of(t), kExchangeLosesOccupant).empty());
  }
}

TEST(labels_arrival_gives_mate_is_e4s_checkmate) {
  // The back-rank mate: Ra8 mates the king on h8 behind its pawns (no board
  // before h2: no escape through time). Rb8, Rc8, ... do not reach h8 - only
  // a8 is a legal arrival that mates, and h1 checks nothing (h7 blocks).
  const GameState s = state_of({timeline(0, 2, {board({"Ra1", "kh8", "pg7", "ph7"})})});
  const SquareLabels l = labels_of(s);
  CHECK(labelled(l, kArrivalGivesMate) == std::set<Square>({{0, "a8"}}));
  CHECK_EQ(l.stats.mate_unknown, std::uint64_t{0});
  CHECK_EQ(l.stats.mate_candidates, std::uint64_t{1});
  // Without the g7 pawn the king escapes to g7: check, no mate.
  const GameState t = state_of({timeline(0, 2, {board({"Ra1", "kh8", "ph7"})})});
  CHECK(labelled(labels_of(t), kArrivalGivesMate).empty());
}

TEST(labels_arrival_gives_mate_left_unknown_by_a_spent_budget) {
  const GameState s = state_of({timeline(0, 2, {board({"Ra1", "kh8", "pg7", "ph7"})})});
  SquareLabels l;
  SquareLabelOptions o;
  o.mate_budget = SearchBudget::nodes(1);
  square_labels(s, o, l);
  CHECK(labelled(l, kArrivalGivesMate, kLabelUnknown) == std::set<Square>({{0, "a8"}}));
  CHECK_EQ(l.stats.mate_unknown, std::uint64_t{1});
}

TEST(labels_names_are_the_networks_in_order) {
  const std::array<std::string_view, kSquareLabelCount> names = {
      "exchange_loses_occupant", "reachable_in_two_ours",      "reachable_in_two_theirs",
      "arrival_gives_check_ours", "arrival_gives_check_theirs", "royal_zone_ours",
      "royal_zone_theirs",        "arrival_gives_mate"};
  for (int i = 0; i < kSquareLabelCount; ++i) {
    CHECK_EQ(std::string_view(square_label_name(i)), names[static_cast<std::size_t>(i)]);
  }
  // The C API's ids are these.
  CHECK_EQ(T5D_SQUARE_LABEL_COUNT, kSquareLabelCount);
  CHECK_EQ(T5D_LABEL_EXCHANGE_LOSES_OCCUPANT, int{kExchangeLosesOccupant});
  CHECK_EQ(T5D_LABEL_REACHABLE_IN_TWO_OURS, int{kReachableInTwoOurs});
  CHECK_EQ(T5D_LABEL_REACHABLE_IN_TWO_THEIRS, int{kReachableInTwoTheirs});
  CHECK_EQ(T5D_LABEL_ARRIVAL_GIVES_CHECK_OURS, int{kArrivalGivesCheckOurs});
  CHECK_EQ(T5D_LABEL_ARRIVAL_GIVES_CHECK_THEIRS, int{kArrivalGivesCheckTheirs});
  CHECK_EQ(T5D_LABEL_ROYAL_ZONE_OURS, int{kRoyalZoneOurs});
  CHECK_EQ(T5D_LABEL_ROYAL_ZONE_THEIRS, int{kRoyalZoneTheirs});
  CHECK_EQ(T5D_LABEL_ARRIVAL_GIVES_MATE, int{kArrivalGivesMate});
  CHECK_EQ(int{T5D_LABEL_UNKNOWN}, int{kLabelUnknown});
}
