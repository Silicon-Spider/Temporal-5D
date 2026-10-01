// Move application (rules reference T2, T3, J4, J5, L7, J9): new boards,
// immutable old ones, new timelines and their numbering, duplicates, unmoved
// flags (Q15), and the refusal of malformed moves.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/state.hpp"
#include "t5d/timeline.hpp"
#include "t5d/zobrist.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using t5d_test::act;
using t5d_test::at;
using t5d_test::game;
using t5d_test::play;

namespace {

constexpr BoardSize k5x5{5, 5};
constexpr BoardSize k8x8{8, 8};

}  // namespace

TEST(apply_a_physical_move_appends_one_board) {
  // T2: a move within a board creates one board, half a step later, with the
  // opponent to move; the old board never changes (A4, L5).
  GameState s(standard_start());
  const BoardRef before = s.multiverse().board_ref(0, 2);
  const std::string before_text = fen_pieces(*before);
  play(s, "(0T1)Pe2(0T1)e4");
  CHECK_EQ(s.multiverse().board_count(), std::size_t{2});
  CHECK(s.multiverse().board_ref(0, 2) == before);  // the same object, kept
  CHECK_EQ(fen_pieces(*before), before_text);
  const Board* after = s.multiverse().head(0);
  CHECK_EQ(fen_pieces(*after),
           std::string("r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/4P3/8/P*P*P*P*1P*P*P*/R*NBQK*BNR*"));
  CHECK_EQ(s.multiverse().timeline(0)->last_half_turn(), 3);
  CHECK_EQ(s.zobrist(), zobrist::key_from_scratch(s.multiverse()));
}

TEST(apply_a_non_branching_jump_appends_two_boards) {
  // J4: from S on timeline A to D, the playable board of timeline B: A gets S
  // without the piece, B gets D with it (capturing), each half a step later.
  GameState s = game({"[5/5/5/5/5:-2:1:w]", "[5/5/2n2/5/5:-1:1:w]", "[5/5/2R2/5/5:0:1:w]",
                      "[5/5/5/5/5:+1:1:w]", "[5/5/5/5/5:+2:1:w]"},
                     k5x5);
  const Move m = play(s, "(0T1)Rc3(-1T1)c3");
  CHECK(m.capture());
  CHECK_EQ(s.multiverse().board_count(), std::size_t{7});
  CHECK_EQ(s.multiverse().timeline(0)->last_half_turn(), 3);
  CHECK_EQ(s.multiverse().timeline(-1)->last_half_turn(), 3);
  CHECK(at(s.multiverse().head(0), "c3").empty());
  CHECK(at(s.multiverse().head(-1), "c3") == Piece(PieceKind::Rook, Colour::White));
  CHECK_EQ(s.multiverse().head(-1)->piece_count(), 1);  // the knight was captured
  CHECK_EQ(s.multiverse().timeline_count(), 5);
}

TEST(apply_a_branching_jump_creates_a_timeline_with_duplicates) {
  // J5: the destination D (a history board) never changes; the source
  // timeline gets S without the piece; a new timeline's only board is D's
  // half-step successor, a copy of D with the piece - keeping D's pieces,
  // the moving piece's earlier self included: two black kings on the new
  // timeline and none on the old one.
  GameState s = game({"[4k3/8/8/8/8/8/8/4K3:0:1:w]", "[4k3/8/8/8/8/8/8/3K4:0:1:b]",
                      "[8/3k4/8/8/8/8/8/3K4:0:2:w]", "[8/3k4/8/8/8/8/8/2K5:0:2:b]"},
                     k8x8);
  REQUIRE(s.to_move() == Colour::Black);
  const BoardRef destination = s.multiverse().board_ref(0, 3);
  const Move m = play(s, "(0T2)Kd7(0T1)d7");
  CHECK(m.branching());
  REQUIRE(s.multiverse().has_timeline(-1));
  const Timeline& branch = *s.multiverse().timeline(-1);
  CHECK(branch.creator() == Colour::Black);
  CHECK_EQ(branch.creation_order(), 1);
  CHECK((branch.origin() == BoardCoord{0, 3}));
  CHECK_EQ(branch.first_half_turn(), 4);  // (T2, white): a black arrival on (T1, black)
  CHECK_EQ(branch.length(), 1);
  CHECK(s.multiverse().board_ref(0, 3) == destination);  // D unchanged
  CHECK_EQ(std::popcount(branch.head()->pieces(Colour::Black, PieceKind::King)), 2);
  CHECK_EQ(std::popcount(s.multiverse().head(0)->pieces(Colour::Black, PieceKind::King)), 0);
  CHECK_EQ(s.multiverse().timeline(0)->last_half_turn(), 6);
  CHECK_EQ(s.zobrist(), zobrist::key_from_scratch(s.multiverse()));
}

TEST(apply_numbers_new_timelines_for_each_colour) {
  // L7: White's new timelines take the next positive index, Black's the next
  // negative one, outermost on the creator's side. L8: in an even start the
  // central pair is -0/+0, so White's first is "+1" and Black's first "-1".
  GameState s(standard_two_timelines_start());
  act(s, {"(-0T1)Pe2(-0T1)e3", "(+0T1)Pe2(+0T1)e3"});
  act(s, {"(-0T1)Pe7(-0T1)e6", "(+0T1)Pe7(+0T1)e6"});
  play(s, "(+0T2)Ng1(+0T1)g3");
  s.submit();
  CHECK_EQ(s.multiverse().max_timeline(), 1);
  CHECK_EQ(timeline_label_string(1, true), std::string("+1"));
  CHECK(s.multiverse().timeline(1)->creator() == Colour::White);
  // Black answers on +1 by branching back onto +0's history: Black's first.
  play(s, "(+1T1)Nb8(+0T1)b6");
  CHECK_EQ(s.multiverse().min_timeline(), -2);
  CHECK_EQ(timeline_label_string(-2, true), std::string("-1"));
  const Timeline& black = *s.multiverse().timeline(-2);
  CHECK(black.creator() == Colour::Black);
  CHECK_EQ(black.creation_order(), 1);
  CHECK(s.active(-2));
  CHECK_EQ(s.multiverse().created_by(Colour::White), 1);
  CHECK_EQ(s.multiverse().created_by(Colour::Black), 1);
  // The canonical labels in the fingerprint string: -1, -0, +0, +1.
  const std::string canonical = canonical_state_string(s.multiverse());
  CHECK(canonical.find(":-1:2:w]") != std::string::npos);
  CHECK(canonical.find(":+1:1:b]") != std::string::npos);
}

TEST(apply_several_branches_in_one_action_take_indices_in_order_j9) {
  // J9 (low; Q28): branches made in one action take their indices in the
  // order they are made. White has created +1 already; both new branches are
  // inactive (Black has created none), so neither moves the present (T8).
  GameState s = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/2R2/5/5:0:2:w]",
                      "[5/5/5/5/5:+1:1:w]", "[5/5/5/5/5:+1:1:b]", "[5/5/2R2/5/5:+1:2:w]"},
                     k5x5);
  CHECK(s.mandatory_timelines() == std::vector<int>({0, 1}));
  play(s, "(+1T2)Rc3(+1T1)c3");
  play(s, "(0T2)Rc3(0T1)c3");
  REQUIRE(s.multiverse().max_timeline() == 3);
  CHECK((s.multiverse().timeline(2)->origin() == BoardCoord{1, 2}));
  CHECK((s.multiverse().timeline(3)->origin() == BoardCoord{0, 2}));
  CHECK_EQ(s.multiverse().timeline(2)->creation_order(), 2);
  CHECK_EQ(s.multiverse().timeline(3)->creation_order(), 3);
  CHECK(!s.active(2));
  CHECK(!s.active(3));
  CHECK(s.obligations_met());
}

TEST(apply_clears_the_unmoved_flag_of_what_moves_q15) {
  // The moving piece loses its unmoved flag - here a pawn's timeline step -
  // unless the ruleset keeps it for pawns and brawns that leave their board.
  const std::vector<std::string> boards = {"[5/5/5/5/5:-1:1:w]", "[5/5/5/5/5:0:1:w]",
                                           "[5/5/5/2P*2/5:+1:1:w]"};
  GameState s = game(boards, k5x5);
  play(s, "(+1T1)Pc2(0T1)c2");
  CHECK(at(s.multiverse().head(0), "c2") == Piece(PieceKind::Pawn, Colour::White, false));
  RuleOptions keep;
  keep.jumps_keep_unmoved = true;
  GameState k = game(boards, k5x5, keep);
  play(k, "(+1T1)Pc2(0T1)c2");
  CHECK(at(k.multiverse().head(0), "c2") == Piece(PieceKind::Pawn, Colour::White, true));
  // Within a board the flag always goes.
  GameState w = game(boards, k5x5, keep);
  play(w, "(+1T1)Pc2(+1T1)c3");
  CHECK(at(w.multiverse().head(1), "c3") == Piece(PieceKind::Pawn, Colour::White, false));
}

TEST(apply_branch_copies_keep_unmoved_flags_by_default) {
  // J5: the copy keeps every piece of the destination board - flags too
  // (Q15/Q21's lead); the alternative clears them on the copy.
  const std::vector<std::string> boards = {"[r*3k*3/8/8/8/8/8/8/R*3K*3:0:1:w]",
                                           "[r*3k*3/8/8/8/8/8/8/R*3K*3:0:1:b]",
                                           "[r*3k*3/8/8/8/8/8/8/R*3K*2N:0:2:w]"};
  GameState s = game(boards, k8x8);
  play(s, "(0T2)Nh1(0T1)h3");  // two ranks up, one turn back
  REQUIRE(s.multiverse().has_timeline(1));
  CHECK_EQ(fen_pieces(*s.multiverse().head(1)), std::string("r*3k*3/8/8/8/8/7N/8/R*3K*3"));
  RuleOptions clear;
  clear.branch_copies_keep_unmoved = false;
  GameState c = game(boards, k8x8, clear);
  play(c, "(0T2)Nh1(0T1)h3");
  CHECK_EQ(fen_pieces(*c.multiverse().head(1)), std::string("r3k3/8/8/8/8/7N/8/R3K3"));
}

TEST(apply_refuses_malformed_moves_and_changes_nothing) {
  const GameState start = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]",
                                "[k4/5/2R2/5/K4:0:2:w]"},
                               k5x5);
  const Move good = *t5d_test::find_move(start, "(0T2)Rc3(0T1)c3");
  REQUIRE(good.branching());
  const auto refused = [&](Move m) {
    GameState s = start;
    bool threw = false;
    try {
      s.apply(m);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
    CHECK(s == start);
  };
  Move m = good;
  m.from_half_turn = 2;  // not the playable board
  refused(m);
  m = good;
  m.piece = Piece(PieceKind::Queen, Colour::White);  // not the piece there
  refused(m);
  m = good;
  m.flags = 0;  // the branching flag disagrees
  refused(m);
  m = good;
  m.to_half_turn = 0;  // no board there
  refused(m);
  m = good;
  m.to_half_turn = 3;  // a board of the other colour
  refused(m);
  m = good;
  m.from_square = static_cast<std::int8_t>(make_square(1, 1));  // an empty square
  refused(m);
  m = good;
  m.captured = Piece(PieceKind::Knight, Colour::Black);  // nothing to capture there
  refused(m);
  m = good;
  m.flags = static_cast<std::uint8_t>(m.flags | Move::kPromotion);  // a rook does not promote
  refused(m);
  m = good;
  m.flags = static_cast<std::uint8_t>(m.flags | Move::kCastle);
  refused(m);
  m = *t5d_test::find_move(start, "(0T2)Rc3(0T2)a3");
  m.to_square = static_cast<std::int8_t>(make_square(0, 0));  // onto its own king
  m.captured = Piece(PieceKind::King, Colour::White);
  m.flags = Move::kCapture;
  refused(m);
  // The good move itself applies.
  GameState s = start;
  CHECK_NOTHROW(s.apply(good));
}

TEST(apply_refuses_malformed_double_steps) {
  // A double step creates an en-passant square (S7) on the square it passed,
  // so apply checks it in full (4.6, Q15): forward, two steps along the rank
  // or the timeline axis, with the right, through an empty square onto an
  // empty one, capturing nothing. Hand-built moves that break any of these
  // are refused, and the state is unchanged.
  const auto refused = [](const GameState& start, const Move& m) {
    GameState s = start;
    bool threw = false;
    try {
      s.apply(m);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
    CHECK(s == start);
  };
  const auto step = [](int from, int to, Piece piece, Piece captured = Piece{}) {
    Move m;
    m.from_timeline = 0;
    m.from_half_turn = 2;
    m.to_timeline = 0;
    m.to_half_turn = 2;
    m.from_square = static_cast<std::int8_t>(from);
    m.to_square = static_cast<std::int8_t>(to);
    m.piece = piece;
    m.captured = captured;
    m.flags = static_cast<std::uint8_t>(Move::kDoubleStep |
                                        (captured.empty() ? 0 : Move::kCapture));
    return m;
  };
  const Piece pawn(PieceKind::Pawn, Colour::White);
  const Piece fresh(PieceKind::Pawn, Colour::White, true);
  const int e2 = make_square(4, 1);
  const int e3 = make_square(4, 2);
  const int e4 = make_square(4, 3);
  // Backwards (the reviewers' probe: e3 holds a knight, which would have
  // received the en-passant square).
  const GameState back = game({"[4k3/8/8/8/4P3/4N3/3p4/4K3:0:1:w]"}, k8x8);
  refused(back, step(e4, e2, pawn));
  // Through a piece, without the right, onto a piece, sideways.
  refused(game({"[4k3/8/8/8/8/4n3/4P*3/K7:0:1:w]"}, k8x8), step(e2, e4, fresh));
  refused(game({"[4k3/8/8/8/8/8/4P3/K7:0:1:w]"}, k8x8), step(e2, e4, pawn));
  const GameState onto = game({"[4k3/8/8/8/4n3/8/4P*3/K7:0:1:w]"}, k8x8);
  refused(onto, step(e2, e4, fresh, Piece(PieceKind::Knight, Colour::Black)));
  refused(onto, step(e2, e4, fresh));  // the capture flag would disagree anyway
  refused(game({"[4k3/8/8/8/8/8/4P*3/K7:0:1:w]"}, k8x8), step(e2, make_square(6, 1), fresh));
  // A knight is no pawn.
  refused(game({"[4k3/8/8/8/8/8/4N3/K7:0:1:w]"}, k8x8),
          step(e2, e4, Piece(PieceKind::Knight, Colour::White)));
  // The legal double step applies, and records the passed square only when an
  // enemy pawn could take (here d4's).
  GameState good = game({"[4k3/8/8/8/3p4/8/4P*3/K7:0:1:w]"}, k8x8);
  CHECK_NOTHROW(good.apply(step(e2, e4, fresh)));
  CHECK_EQ(good.multiverse().head(0)->en_passant(), e3);
  // Along the timeline axis: two timelines forward (White: towards -L), the
  // same square and turn, through an empty square on the in-between board.
  const GameState lines = game({"[5/5/5/5/5:-1:1:w]", "[5/5/5/5/5:0:1:w]",
                                "[5/5/5/2P*2/5:+1:1:w]"},
                               k5x5);
  const Move two = *t5d_test::find_move(lines, "(+1T1)Pc2(-1T1)c2");
  REQUIRE(two.double_step());
  REQUIRE(!two.branching());
  // Backwards along the timeline axis: from -1 to +1 is towards White's own
  // timelines.
  const GameState behind = game({"[5/5/5/2P*2/5:-1:1:w]", "[5/5/5/5/5:0:1:w]",
                                 "[5/5/5/5/5:+1:1:w]"},
                                k5x5);
  Move wrong_way = two;
  wrong_way.from_timeline = -1;
  wrong_way.to_timeline = 1;
  refused(behind, wrong_way);
  const GameState blocked = game({"[5/5/5/5/5:-1:1:w]", "[5/5/5/2n2/5:0:1:w]",
                                  "[5/5/5/2P*2/5:+1:1:w]"},
                                 k5x5);
  refused(blocked, two);  // c2 of 0 is occupied
  RuleOptions single;
  single.timeline_double_step = false;
  refused(game({"[5/5/5/5/5:-1:1:w]", "[5/5/5/5/5:0:1:w]", "[5/5/5/2P*2/5:+1:1:w]"}, k5x5,
               single),
          two);
  GameState fine = lines;
  CHECK_NOTHROW(fine.apply(two));
}

TEST(apply_keeps_keys_and_fingerprints_consistent) {
  // After every move: the incremental key equals the key from scratch, and
  // the canonical string reads back to the same multiverse.
  GameState s(standard_two_timelines_start());
  const std::vector<std::vector<const char*>> actions = {
      {"(-0T1)Pe2(-0T1)e3", "(+0T1)Pe2(+0T1)e3"},
      {"(-0T1)Pe7(-0T1)e6", "(+0T1)Pe7(+0T1)e6"},
      {"(+0T2)Ng1(+0T1)g3"},
      {"(+1T1)Nb8(+0T1)b6"},
  };
  for (const auto& action : actions) {
    for (const char* text : action) {
      play(s, text);
      CHECK_EQ(s.multiverse().zobrist(), zobrist::key_from_scratch(s.multiverse()));
      const std::string canonical = canonical_state_string(s.multiverse());
      std::vector<std::string> strings;
      for (std::size_t i = 0; i < canonical.size();) {
        const std::size_t end = canonical.find(']', i);
        strings.push_back(canonical.substr(i, end - i + 1));
        i = end + 1;
      }
      const Multiverse back =
          multiverse_from_fen(strings, k8x8, Parity::Even, s.multiverse().ruleset(), s.to_move());
      CHECK_EQ(canonical_state_string(back), canonical);
      CHECK_EQ(fingerprint(back).fingerprint, fingerprint(s.multiverse()).fingerprint);
      // The labels carry the creators, so the key reads back too.
      CHECK_EQ(back.zobrist(), s.multiverse().zobrist());
    }
    s.submit();
  }
}
