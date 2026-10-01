// Pseudo-legal movement across timelines (rules reference J2-J4, J7, J8, 4.4,
// 4.5, 4.9) and the pawn and brawn (4.6, 4.7; Q14, Q15).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using t5d_test::contains;
using t5d_test::game;
using t5d_test::piece_texts;

namespace {

constexpr BoardSize k5x5{5, 5};

// Timeline 0: T1w..T3w (head at T3w, `head`); +1: one board at T4w (so +1 has
// no board at T3 - void); +2: one board at T3w (`plus2`), White's second
// timeline, inactive (Black has created none).
GameState void_position(const std::string& head, const std::string& plus2 = "5/5/5/5/5",
                        const RuleOptions& rules = {}) {
  return game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/5:0:2:w]",
               "[5/5/5/5/5:0:2:b]", "[" + head + ":0:3:w]", "[5/5/5/5/5:+1:4:w]",
               "[" + plus2 + ":+2:3:w]"},
              k5x5, rules);
}

// +1: T1w, T1b, T2w (the playable board); 0: T1w..T3w. Black pawns stand on
// every square a white brawn on c2 of +1 could capture, and block its steps.
GameState capture_position(const std::string& mover, const RuleOptions& r = {}) {
  return game({"[5/5/5/2p2/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/2p2/1ppp1/5:0:2:w]",
               "[5/5/5/5/5:0:2:b]", "[5/5/5/2p2/5:0:3:w]", "[5/5/2p2/5/5:+1:1:w]",
               "[5/5/5/5/5:+1:1:b]", "[5/5/1ppp1/2" + mover + "2/5:+1:2:w]"},
              k5x5, r);
}

// Three timelines -1, 0, +1, each T1w, T1b, T2w; `head` is 0's T2 board.
GameState three_timelines(const std::string& head) {
  std::vector<std::string> boards;
  for (const std::string l : {"-1", "0", "+1"}) {
    boards.push_back("[5/5/5/5/5:" + l + ":1:w]");
    boards.push_back("[5/5/5/5/5:" + l + ":1:b]");
    boards.push_back("[" + (l == "0" ? head : std::string("5/5/5/5/5")) + ":" + l + ":2:w]");
  }
  return game(boards, k5x5);
}

}  // namespace

// ---- across timelines (J2, J3, J4, J7, J8) ---------------------------------------------------

TEST(movegen_rook_slides_across_timelines_and_is_blocked_on_them) {
  // Five timelines, one board each at T1 (white). The rook on 0 slides to the
  // playable boards of -1 and -2 (non-branching jumps, J4); the white knight
  // on c3 of +1 blocks the way to +2 (J7).
  const GameState s = game({"[5/5/5/5/5:-2:1:w]", "[5/5/5/5/5:-1:1:w]", "[5/5/2R2/5/5:0:1:w]",
                            "[5/5/2N2/5/5:+1:1:w]", "[5/5/5/5/5:+2:1:w]"},
                           k5x5);
  const std::vector<std::string> r = piece_texts(s, 0, "c3");
  CHECK_EQ(r.size(), std::size_t{8 + 2});
  CHECK(contains(r, "(0T1)Rc3(-1T1)c3"));
  CHECK(contains(r, "(0T1)Rc3(-2T1)c3"));
  CHECK(!contains(r, "(0T1)Rc3(+1T1)c3"));
  CHECK(!contains(r, "(0T1)Rc3(+2T1)c3"));
  const std::optional<Move> jump = t5d_test::find_move(s, "(0T1)Rc3(-2T1)c3");
  REQUIRE(jump.has_value());
  CHECK(jump->jump());
  CHECK(!jump->branching());
}

TEST(movegen_void_blocks_sliders_but_not_knights) {
  // J8: no board on +1 at T3 - void - blocks the rook's slide to +2; the
  // knight jumps over it, two timelines and one file or rank (4.5).
  const GameState s = void_position("5/5/2R2/5/2N2");
  CHECK(!s.active(2));
  const std::vector<std::string> rook = piece_texts(s, 0, "c3");
  CHECK(!contains(rook, "(0T3)Rc3(+1T3)c3"));
  CHECK(!contains(rook, "(0T3)Rc3(+2T3)c3"));
  const std::vector<std::string> knight = piece_texts(s, 0, "c1");
  CHECK(contains(knight, "(0T3)Nc1(+2T3)b1"));
  CHECK(contains(knight, "(0T3)Nc1(+2T3)d1"));
  CHECK(contains(knight, "(0T3)Nc1(+2T3)c2"));
  // No destination on void: (+1, T3) does not exist, nor (+2, T2) or (+2, T4).
  for (const std::string& t : knight) {
    CHECK(t.find("(+1T3)") == std::string::npos);
    CHECK(t.find("(+2T2)") == std::string::npos);
    CHECK(t.find("(+2T4)") == std::string::npos);
  }
  // Q18's dissent: sliders pass void (but never land on it).
  RuleOptions r;
  r.sliders_cross_void = true;
  const GameState crossing = void_position("5/5/2R2/5/2N2", "5/5/5/5/5", r);
  const std::vector<std::string> crossed = piece_texts(crossing, 0, "c3");
  CHECK(contains(crossed, "(0T3)Rc3(+2T3)c3"));
  CHECK(!contains(crossed, "(0T3)Rc3(+1T3)c3"));
}

TEST(movegen_a_slide_beyond_void_needs_the_far_square_free) {
  // Under Q18's dissent the far board's square still matters: a white piece
  // there is no destination, a black one is captured.
  RuleOptions r;
  r.sliders_cross_void = true;
  CHECK(!contains(piece_texts(void_position("5/5/2R2/5/5", "5/5/2N2/5/5", r), 0, "c3"),
                  "(0T3)Rc3(+2T3)c3"));
  CHECK(contains(piece_texts(void_position("5/5/2R2/5/5", "5/5/2n2/5/5", r), 0, "c3"),
                 "(0T3)Rc3(+2T3)c3"));
}

TEST(movegen_branching_iff_the_destination_is_a_history_board) {
  // J3: a move between boards branches iff it lands on a history board. +1's
  // playable board is at T2, earlier than 0's at T3: a bishop's jump onto it
  // (one timeline up, one turn back) does not branch (J3, medium; Q20 notes
  // that only the criterion supports this) - a knight's jump onto +1's T1
  // board, a history board, does.
  const GameState s = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/5:0:2:w]",
                            "[5/5/5/5/5:0:2:b]", "[5/5/2B2/5/2N2:0:3:w]", "[5/5/5/5/5:+1:1:w]",
                            "[5/5/5/5/5:+1:1:b]", "[5/5/5/5/5:+1:2:w]"},
                           k5x5);
  CHECK_EQ(s.present(), half_turn(2, Colour::White));
  const std::optional<Move> onto_playable = t5d_test::find_move(s, "(0T3)Bc3(+1T2)c3");
  REQUIRE(onto_playable.has_value());
  CHECK(!onto_playable->branching());
  const std::optional<Move> onto_history = t5d_test::find_move(s, "(0T3)Nc1(+1T1)c1");
  REQUIRE(onto_history.has_value());
  CHECK(onto_history->branching());
  // And every move back along its own timeline branches.
  const std::optional<Move> own_past = t5d_test::find_move(s, "(0T3)Bc3(0T2)c2");
  REQUIRE(own_past.has_value());
  CHECK(own_past->branching());
}

TEST(movegen_earlier_playable_jump_branches_under_the_quick_rules_reading) {
  // J3's conflict: read literally, the Quick Rules' "Moving a piece to the
  // past causes a new timeline to be created" makes the bishop's jump onto
  // +1's PLAYABLE board at T2 - earlier than its T3 source - branch. The
  // mechanism is J5's: the destination board is left unchanged (so +1 is not
  // played and stays mandatory), and a new timeline starts with its half-step
  // successor, a copy with the bishop.
  const std::vector<std::string> boards = {
      "[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/5:0:2:w]",
      "[5/5/5/5/5:0:2:b]", "[5/5/2B2/5/2N2:0:3:w]", "[5/5/5/5/5:+1:1:w]",
      "[5/5/5/5/5:+1:1:b]", "[5/5/5/5/5:+1:2:w]"};
  RuleOptions r;
  r.earlier_playable_jump_branches = true;
  const GameState s = game(boards, k5x5, r);
  CHECK(s.mandatory_timelines() == std::vector<int>({1}));
  const std::optional<Move> jump = t5d_test::find_move(s, "(0T3)Bc3(+1T2)c3");
  REQUIRE(jump.has_value());
  CHECK(jump->branching());
  GameState after = s;
  const BoardRef destination = s.multiverse().board_ref(1, half_turn(2, Colour::White));
  after.apply(*jump);
  CHECK_EQ(after.multiverse().timeline_count(), 3);
  CHECK(after.multiverse().timeline(1)->head() == destination);  // D unchanged, still playable
  REQUIRE(after.multiverse().has_timeline(2));
  const Timeline& branch = *after.multiverse().timeline(2);
  CHECK_EQ(branch.first_half_turn(), half_turn(2, Colour::Black));
  CHECK((branch.origin() == BoardCoord{1, half_turn(2, Colour::White)}));
  CHECK(t5d_test::at(branch.head().get(), "c3") == Piece(PieceKind::Bishop, Colour::White));
  // White's second timeline while Black has created none: inactive (T10), so
  // the present stays, and +1 must still be played.
  CHECK(!after.active(2));
  CHECK(after.mandatory_timelines() == std::vector<int>({1}));
  // The default reading refuses the branching version of the move, and the
  // ruleset the non-branching one.
  GameState standard = game(boards, k5x5);
  const GameState standard_before = standard;
  CHECK_THROWS_AS(standard.apply(*jump), std::invalid_argument);
  CHECK(standard == standard_before);
  Move plain = *jump;
  plain.flags = static_cast<std::uint8_t>(plain.flags & ~Move::kBranching);
  GameState literal = s;
  CHECK_THROWS_AS(literal.apply(plain), std::invalid_argument);
  CHECK(literal == s);
  // A jump at the same turn onto a playable board still does not branch.
  RuleOptions same;
  same.earlier_playable_jump_branches = true;
  const GameState level = game({"[5/5/2Q2/5/5:0:1:w]", "[5/5/5/5/5:+1:1:w]"}, k5x5, same);
  const std::optional<Move> across = t5d_test::find_move(level, "(0T1)Qc3(+1T1)c3");
  REQUIRE(across.has_value());
  CHECK(!across->branching());
}

TEST(movegen_inactive_history_as_a_destination_q20) {
  // Q20: a history board of an inactive timeline as a branch destination -
  // presumed (J3 makes no exception), never stated. +2 is White's second
  // timeline while Black has created none: inactive. The knight on 0's T2
  // board leaps two timelines up and one turn back onto +2's T1 board, a
  // history board; the bishop's diagonal onto +1's T1 board (active) is
  // unaffected by the alternative.
  const std::vector<std::string> boards = {
      "[5/5/5/5/5:0:1:w]",  "[5/5/5/5/5:0:1:b]",  "[5/5/2B2/5/N4:0:2:w]",
      "[5/5/5/5/5:+1:1:w]", "[5/5/5/5/5:+1:1:b]", "[5/5/5/5/5:+1:2:w]",
      "[5/5/5/5/5:+2:1:w]", "[5/5/5/5/5:+2:1:b]", "[5/5/5/5/5:+2:2:w]"};
  const GameState s = game(boards, k5x5);
  REQUIRE(!s.active(2));
  REQUIRE(s.active(1));
  CHECK(contains(piece_texts(s, 0, "a1"), "(0T2)Na1(+2T1)a1"));
  CHECK(contains(piece_texts(s, 0, "c3"), "(0T2)Bc3(+1T1)c3"));
  RuleOptions r;
  r.inactive_history_is_destination = false;
  const GameState q = game(boards, k5x5, r);
  CHECK(!contains(piece_texts(q, 0, "a1"), "(0T2)Na1(+2T1)a1"));
  CHECK(contains(piece_texts(q, 0, "a1"), "(0T2)Na1(+2T2)b1"));  // its playable board: fine
  CHECK(contains(piece_texts(q, 0, "c3"), "(0T2)Bc3(+1T1)c3"));
  // apply refuses the move under the alternative.
  const Move leap = *t5d_test::find_move(s, "(0T2)Na1(+2T1)a1");
  GameState refuse = q;
  CHECK_THROWS_AS(refuse.apply(leap), std::invalid_argument);
  CHECK(refuse == q);
  // append_moves_of reads activity from the caller (a Multiverse has none).
  MoveList with_activity;
  append_moves_of(q.multiverse(), q.rules(), Colour::White, 0, with_activity,
                  q.inactive_timelines());
  MoveList without;
  append_moves_of(q.multiverse(), q.rules(), Colour::White, 0, without);
  const auto has = [&](const MoveList& list) {
    for (const Move& m : list) {
      if (m == leap) return true;
    }
    return false;
  };
  CHECK(!has(with_activity));
  CHECK(has(without));
}

TEST(movegen_king_knight_and_dragon_in_four_dimensions) {
  // 4.4: one step along any set of axes. dT = +1 has no board, so
  // 3 x 3 x 3 x 2 - 1 = 53 destinations: 8 within the board, 18 jumps onto
  // the other playable boards, 27 branching steps back to T1.
  const GameState k = three_timelines("5/5/2K2/5/5");
  MoveList moves;
  append_piece_moves(k, 0, make_square(2, 2), moves);
  CHECK_EQ(moves.size(), std::size_t{53});
  int physical = 0;
  int jumps = 0;
  int branches = 0;
  for (const Move& m : moves) {
    if (m.physical()) {
      ++physical;
    } else if (m.branching()) {
      ++branches;
    } else {
      ++jumps;
    }
  }
  CHECK_EQ(physical, 8);
  CHECK_EQ(jumps, 18);
  CHECK_EQ(branches, 27);
  CHECK(contains(piece_texts(k, 0, "c3"), "(0T2)Kc3(0T1)c3"));  // 4.4: back in time
  // 4.5: 8 leaps within the board, 4 along file x timeline, 2 file x turn,
  // 4 rank x timeline, 2 rank x turn = 20.
  CHECK_EQ(piece_texts(three_timelines("5/5/2N2/5/5"), 0, "c3").size(), std::size_t{20});
  // 4.9: along all four axes - one step to T1 on -1 or +1, four ways each.
  const std::vector<std::string> dragon = piece_texts(three_timelines("5/5/2D2/5/5"), 0, "c3");
  CHECK_EQ(dragon.size(), std::size_t{8});
  CHECK(contains(dragon, "(0T2)Dc3(-1T1)b2"));
  CHECK(contains(dragon, "(0T2)Dc3(+1T1)d4"));
}

TEST(movegen_rook_tactics_one_from_the_notation_specification) {
  // The notation specification's worked example "Rook Tactics I" (its README:
  // [Size "5x5"], [4k/5/5/5/K1R2:0:1:w], then 1. Kb2 / Ke4 2. Re1 / Kd3
  // 3. (0T3)Re5# {attacks (0T1)Ke5}; C2). Played here move by move; then,
  // with Black's T3 board passed unchanged (a hypothetical multiverse: the
  // rook now stands on e5 of a T4 board), the rook slides three turns back
  // along its own timeline - over the T3 and T2 white boards, e5 empty on
  // both (J7) - to e5 of the T1 board, capturing the king's earlier self
  // there: the attack the annotation names. Whether that is check is the
  // check ruling's; this is only the movement that C2 rests on.
  GameState s = game({"[4k/5/5/5/K1R2:0:1:w]"}, k5x5);
  t5d_test::act(s, {"(0T1)Ka1(0T1)b2"});
  t5d_test::act(s, {"(0T1)Ke5(0T1)e4"});
  t5d_test::act(s, {"(0T2)Rc1(0T2)e1"});
  t5d_test::act(s, {"(0T2)Ke4(0T2)d3"});
  t5d_test::play(s, "(0T3)Re1(0T3)e5");
  CHECK(s.obligations_met());
  Multiverse passed = s.multiverse();
  passed.append_board(0, passed.board_ref(0, half_turn(3, Colour::Black)));
  MoveList white;
  append_moves_of(passed, s.rules(), Colour::White, 0, white);
  bool found = false;
  for (const Move& m : white) {
    if (to_string(m, false) == "(0T4)Re5(0T1)e5") {
      found = true;
      CHECK(m.branching());
      CHECK(m.captured == Piece(PieceKind::King, Colour::Black));
    }
  }
  CHECK(found);
}

// ---- pawns and brawns (4.6, 4.7) ------------------------------------------------------------

TEST(movegen_pawn_moves_forward_on_rank_and_timeline) {
  // 4.6: White's forward is +rank and -timeline. An unmoved pawn on c2 of +1:
  // c3, c4 within the board; c2 on 0 and on -1 (the timeline double step).
  const auto position = [](const RuleOptions& r) {
    return game({"[5/5/5/5/5:-1:1:w]", "[5/5/5/5/5:0:1:w]", "[5/5/5/2P*2/5:+1:1:w]"}, k5x5, r);
  };
  const std::vector<std::string> r = piece_texts(position({}), 1, "c2");
  CHECK_EQ(r.size(), std::size_t{4});
  for (const char* t : {"(+1T1)Pc2(+1T1)c3", "(+1T1)Pc2(+1T1)c4", "(+1T1)Pc2(0T1)c2",
                        "(+1T1)Pc2(-1T1)c2"}) {
    CHECK(contains(r, t));
  }
  const std::optional<Move> two = t5d_test::find_move(position({}), "(+1T1)Pc2(-1T1)c2");
  REQUIRE(two.has_value());
  CHECK(two->double_step());
  CHECK(!two->branching());
  // Q15: no timeline double step under the 2022 guide's reading.
  RuleOptions single;
  single.timeline_double_step = false;
  CHECK_EQ(piece_texts(position(single), 1, "c2").size(), std::size_t{3});
}

TEST(movegen_black_pawn_moves_down_and_towards_higher_timelines) {
  // 4.6: Black's forward is -rank and +timeline.
  const GameState b =
      game({"[5/2p*2/5/5/5:-1:1:b]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/5:+1:1:b]"}, k5x5);
  CHECK(b.to_move() == Colour::Black);
  const std::vector<std::string> r = piece_texts(b, -1, "c4");
  CHECK_EQ(r.size(), std::size_t{4});
  for (const char* t : {"(-1T1)Pc4(-1T1)c3", "(-1T1)Pc4(-1T1)c2", "(-1T1)Pc4(0T1)c4",
                        "(-1T1)Pc4(+1T1)c4"}) {
    CHECK(contains(r, t));
  }
}

TEST(movegen_pawn_steps_need_empty_squares) {
  // A piece in front blocks the step - and the double step through it; a
  // piece on the in-between timeline's square blocks the timeline steps.
  const GameState s = game({"[5/5/5/2n2/5:-1:1:w]", "[5/5/5/2N2/5:0:1:w]",
                            "[5/5/2n2/2P*2/5:+1:1:w]"},
                           k5x5);
  CHECK(piece_texts(s, 1, "c2").empty());
}

TEST(movegen_timeline_double_step_and_void) {
  // Q15: a timeline double step past a timeline that has no board at that
  // turn. Timeline 0 starts at T2 here, so (0, T1) is void.
  const auto position = [](const RuleOptions& r) {
    return game({"[5/5/5/5/5:-1:1:w]", "[5/5/5/5/5:0:2:w]", "[5/5/5/2P*2/5:+1:1:w]"}, k5x5, r);
  };
  const std::vector<std::string> blocked = piece_texts(position({}), 1, "c2");
  CHECK(!contains(blocked, "(+1T1)Pc2(0T1)c2"));   // no board there
  CHECK(!contains(blocked, "(+1T1)Pc2(-1T1)c2"));  // void blocks (J8): the default
  RuleOptions over;
  over.double_step_over_void = true;
  CHECK(contains(piece_texts(position(over), 1, "c2"), "(+1T1)Pc2(-1T1)c2"));
}

TEST(movegen_double_step_right_flag_or_rank) {
  // Q15: by default the piece's unmoved flag; the alternative, the rank.
  const auto moves = [](const std::string& pieces, const RuleOptions& r) {
    return piece_texts(game({"[" + pieces + ":0:1:w]"}, k5x5, r), 0, "c2");
  };
  const RuleOptions flag;
  RuleOptions rank;
  rank.double_step_right = DoubleStepRight::HomeRank;
  CHECK(contains(moves("5/5/5/2P*2/5", flag), "(0T1)Pc2(0T1)c4"));
  CHECK(!contains(moves("5/5/5/2P2/5", flag), "(0T1)Pc2(0T1)c4"));  // moved: no double step
  CHECK(contains(moves("5/5/5/2P2/5", rank), "(0T1)Pc2(0T1)c4"));   // on the second rank
  // A back-rank pawn with the flag (Focused - Just Pawns' layout) may
  // double-step by default, not under the rank reading.
  const auto back = [](const RuleOptions& r) {
    return piece_texts(game({"[5/5/5/5/2P*2:0:1:w]"}, k5x5, r), 0, "c1");
  };
  CHECK(contains(back(flag), "(0T1)Pc1(0T1)c3"));
  CHECK(!contains(back(rank), "(0T1)Pc1(0T1)c3"));
}

TEST(movegen_pawn_captures) {
  // 4.6: forward rank and +-1 file (b3, d3), or forward timeline and +-1 turn
  // (c2 of 0 at T1, a history board - a branching capture - and at T3, 0's
  // playable board). Never straight ahead: c3 and (0,T2)c2 are occupied.
  const GameState s = capture_position("P");
  const std::vector<std::string> r = piece_texts(s, 1, "c2");
  CHECK_EQ(r.size(), std::size_t{4});
  for (const char* t : {"(+1T2)Pc2(+1T2)b3", "(+1T2)Pc2(+1T2)d3", "(+1T2)Pc2(0T1)c2",
                        "(+1T2)Pc2(0T3)c2"}) {
    CHECK(contains(r, t));
  }
  const std::optional<Move> back = t5d_test::find_move(s, "(+1T2)Pc2(0T1)c2");
  REQUIRE(back.has_value());
  CHECK(back->branching());
  CHECK(back->capture());
}

TEST(movegen_brawn_captures) {
  // 4.7: one forward axis and one sideways axis - b3, d3 (rank, file); c3 of
  // +1 at T1 (rank, turn; T3 has no board); b2, d2 of 0 (timeline, file); c2
  // of 0 at T1 and T3 (timeline, turn) - and, Q14's default, c3 of 0 at T2
  // (rank and timeline together).
  const std::vector<std::string> r = piece_texts(capture_position("W"), 1, "c2");
  CHECK_EQ(r.size(), std::size_t{8});
  for (const char* t : {"(+1T2)Wc2(+1T2)b3", "(+1T2)Wc2(+1T2)d3", "(+1T2)Wc2(+1T1)c3",
                        "(+1T2)Wc2(0T2)b2", "(+1T2)Wc2(0T2)d2", "(+1T2)Wc2(0T1)c2",
                        "(+1T2)Wc2(0T3)c2", "(+1T2)Wc2(0T2)c3"}) {
    CHECK(contains(r, t));
  }
  RuleOptions perseus;
  perseus.brawn_rank_timeline_capture = false;
  const std::vector<std::string> p = piece_texts(capture_position("W", perseus), 1, "c2");
  CHECK_EQ(p.size(), std::size_t{7});
  CHECK(!contains(p, "(+1T2)Wc2(0T2)c3"));
}

TEST(movegen_pawn_crosses_the_plus_and_minus_zero_pair) {
  // L8: in an even start, +0 and -0 are adjacent - a white pawn moves from +0
  // to -0 as one timeline forward.
  const GameState s = game({"[5/5/5/5/5:-0:1:w]", "[5/5/5/1P3/5:+0:1:w]"}, k5x5);
  CHECK(s.multiverse().even());
  CHECK(contains(piece_texts(s, 0, "b2"), "(+0T1)Pb2(-0T1)b2"));
}

// ---- destinations (J2, L4) ---------------------------------------------------------------------

TEST(movegen_never_onto_a_friendly_piece_or_off_the_multiverse) {
  // Every generated destination exists, is of the mover's colour, and holds
  // no piece of the mover's.
  for (const GameState& s : {capture_position("W"), capture_position("P"),
                             three_timelines("5/5/2Q2/5/5"), void_position("5/5/2Y2/5/2N2")}) {
    for (const Move& m : t5d_test::moves_of(s)) {
      const Board* b = s.multiverse().board_at(m.to_timeline, m.to_half_turn);
      REQUIRE(b != nullptr);
      CHECK(to_move_at(m.to_half_turn) == Colour::White);
      const Piece there = b->at(m.to_square);
      CHECK(there.empty() || there.colour() == Colour::Black);
      CHECK(is_pseudo_legal(s, m));
    }
  }
}
