// The notation specification's worked examples, replayed to the end (or to
// where the record itself is wrong), with board counts and selected boards
// derived by hand from the rules reference (docs/rules/README.md: T2, J4, J5,
// L7) - each derivation is in the comments. The record texts are the
// specification's own (its README, "Full examples", "5DFEN and custom
// variants"); only what the specification leaves out - a starting position -
// is supplied here, and said so.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/state.hpp"
#include "t5d/timeline.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

// A move in 5dpgn, written here for the round trip in full: the board's
// coordinates, the piece letter, the source square in full, 'x' for a capture,
// '>' or '>>' and the target board for a jump, the target square, "=X" for a
// promotion ("Moves", "Jumps", "Promotions"). Castling is the king's move
// ("K<from><to>", "Castling").
std::string pgn_text(const Move& m, bool even) {
  const auto board = [&](int p, int h) {
    return "(" + timeline_label_string(p, even) + "T" + std::to_string(turn_of(h)) + ")";
  };
  std::string out = board(m.from_timeline, m.from_half_turn);
  out += letter_of(m.piece.kind());
  out += square_name(m.from_square);
  if (m.jump()) out += m.branching() ? ">>" : ">";
  if (m.capture()) out += 'x';
  if (m.jump()) out += board(m.to_timeline, m.to_half_turn);
  out += square_name(m.to_square);
  if (m.promotes()) {
    out += '=';
    out += letter_of(m.promotion);
  }
  return out;
}

// Every pseudo-legal move of every position the replay reaches, written in
// full and read back: the reader must name exactly that move. Returns the
// number of positions visited.
std::size_t round_trip_every_move(std::string_view record, const PgnOptions& options = {}) {
  PgnReplay r(read_pgn(record, options), options);
  std::size_t positions = 0;
  MoveList moves;
  for (;;) {
    ++positions;
    generate_moves(r.state(), moves);
    CHECK(!moves.empty());
    for (const Move& m : moves) {
      const std::string text = pgn_text(m, r.state().multiverse().even());
      const Move back = r.resolve(r.state(), read_pgn_move(text));
      if (!(back == m)) {
        CHECK_EQ(to_string(back, r.state().multiverse().even()), text);
      }
    }
    if (r.finished()) break;
    try {
      r.play_next();
    } catch (const RecordError&) {
      break;
    }
  }
  return positions;
}

// Every board of a state as its canonical 5DFEN board string, canonical order.
std::vector<std::string> board_strings(const GameState& s) {
  std::vector<std::string> out;
  const Multiverse& mv = s.multiverse();
  mv.for_each_board([&](int p, int h, const Board& b) {
    out.push_back(fen_board_string(b, p, h, mv.even()));
  });
  return out;
}

// The board (label, turn, colour) as a 5DFEN pieces field, or "(none)".
std::string pieces(const GameState& s, std::string_view label, int turn, Colour c) {
  const Multiverse& mv = s.multiverse();
  const int p = timeline_position(parse_timeline_label(label), mv.even());
  const Board* b = mv.board_at(p, half_turn(turn, c));
  return b != nullptr ? fen_pieces(*b) : std::string("(none)");
}

// Plays every action, recording the board count after each; returns the
// fault that stopped it, if any.
std::optional<RecordError> play(PgnReplay& r, std::vector<std::size_t>& counts) {
  counts.push_back(r.state().multiverse().board_count());
  try {
    while (!r.finished()) {
      r.play_next();
      counts.push_back(r.state().multiverse().board_count());
    }
  } catch (const RecordError& e) {
    return e;
  }
  return std::nullopt;
}

// ---- the specification's texts -----------------------------------------------------------------

// "Full examples", Rook Tactics I.
constexpr std::string_view kRookTacticsFull = R"([Size "5x5"]
[Mode "5D"]

1. (0T1)Kb2 / (0T1)Ke4
2. (0T2)Re1 / (0T2)Kd3
3. (0T3)Re5# {attacks (0T1)Ke5}
)";

// "5DFEN and custom variants", the same puzzle with its position.
constexpr std::string_view kRookTactics5dfen = R"([Size "5x5"]
[Puzzle "mate-in-1"]
[Board "custom"]
[4k/5/5/5/K1R2:0:1:w]

1. Kb2 / Ke4
2. Re1 / Kd3
)";

// A position Knight tactics III's moves fit, made for the test - NOT the
// puzzle's, which the specification does not give (Q27); see the test below.
constexpr std::string_view kKnightTacticsSetUp =
    "[Size \"5x5\"]\n[Board \"custom\"]\n[3k*1/2p*2/5/5/K*N3:0:1:w]\n\n";

// V3's Simple - No Queens row (7x7, rnbknbr/ppppppp/7/7/7/PPPPPPP/RNBKNBR,
// kings d1 and d7, every pawn, rook and king unmoved) as a custom position:
// the engine does not set that variant up by name.
constexpr std::string_view kNoQueensSetUp =
    "[Board \"custom\"]\n[Size \"7x7\"]\n[Mode \"5D\"]\n"
    "[r*nbk*nbr*/p*p*p*p*p*p*p*/7/7/7/P*P*P*P*P*P*P*/R*NBK*NBR*:0:1:w]\n";

// "Full examples", Knight tactics III: moves only.
constexpr std::string_view kKnightTacticsMoves = R"(1. (0T1)Nd2 / (0T1)c3
2. (0T2)Nb3 / (0T2)c2
3. (0T3)Nb3>>(0T1)a3 / (1T1)Ke4
4. (1T2)Nb5# {attacks (0T3)Kd5}
)";

// "Full examples", "Actual game": the headers, then the move text.
constexpr std::string_view kActualGameHeaders = R"([White "Teln0"]
[Black "Shad Amethyst"]
[Board "Simple - No Queens"]
[Size "7x7"]
[Date "2020.08.14"]
[Result "1-0"]
[Mode "5D"]
)";
constexpr std::string_view kActualGameMoves = R"(
1. (0T1)Nd3 / (0T1)Nc5
2. (0T2)Nxc5+ / (0T2)bxc5
3. (0T3)g3 / (0T3)g4
4. (0T4)Bg2 / (0T4)d5
5. (0T5)O-O {to the right} / (0T5)Be5
6. (0T6)Nb1>>(0T5)b3 / (1T5)Bb6
7. (1T6)Nc3 / (0T6)Ne7>(1T6)e5
8. (0T7)a4 (1T7)O-O {to the right} / (0T7)Be5>>(0T6)e4
9. (-1T7)Bxe4 / (-1T7)dxe4 (1T7)c4
10. (1T8)Nb3>(0T8)d3 (-1T8)a4 / (-1T8)Bd4 (0T8)Kd7>(1T8)d6
11. (-1T9)Ra3 (0T9)Ra3 (1T9)b3 / (-1T9)Bd4>x(0T9)d3 (1T9)cxb3
12. (-1T10)d3 (0T10)cxd3 (1T10)Ba3 / (-1T10)Nd5 (0T10)Rb7 (1T10)c5?
13. (-1T11)dxe4 (0T11)Rc3 (1T11)axb3 / (0T11)Rg5 (1T11)Ne5>x(-1T11)e4
14. (-1T12)Rd3 (0T12)d4 (1T12)b4 / (-1T12)c4 (0T12)c4 (1T12)cxb4
15. (-1T13)Rd4 (0T13)d3 (1T13)Ba4+ / (-1T13)O-O-O {to the left} (0T13)cxd3 (1T13)Bc5
16. (-1T14)Rxe4 (0T14)exd3 (1T14)Bxc5+ / (0T14)Bg2 (-1T14)Nd5>x(1T14)c5
17. (-1T15)Rc4 (0T15)Rc6 (1T15)Ra5 / (-1T15)e5 (0T15)Bd3+ (1T15)Ne4
18. (-1T16)Rb4+ (0T16)Kg1 (1T16)Nxe4 / (-1T16)Kb7>x(0T16)c6 (1T16)dxe4
19. (-1T17)Rb4>(1T17)b4 (0T17)Bxg5 / (-1T17)Bc4 (0T17)Bc4 (1T17)Nd5
20. (-1T18)Be3 (0T18)Rc1 (1T18)Rd4 / (-1T18)Rd7 (0T18)Kc6>(1T18)b6?
21. (-1T19)Bb6+ {checks (1T19)d6} (0T19)Rb1 (1T19)Raxd5+ / (-1T19)c6>(0T19)c6 {blocks (-1T20)Bb6>>x(1T20)d6 1-0} (1T19)exd5
22. (-1T20)Bb6>x(0T20)b7+ {checks (1T20)d7} (1T20)Rxd5+ / (-1T20)Bc4>>x(-1T18)a4
23. (-2T19)Bb6 / (-2T19)Rb7
24. (-2T20)Bb6>>x(-1T20)a6+ {checks (1T20)b6, (2T20)b6 also checks (1T19)b6} 1-0 {Black forfeits}
)";

// "Full examples", the game between Shad and PseudoAbstractMeta.
constexpr std::string_view kHalfReflectedGame = R"([Mode "5D"]
[Board "Standard - Half Reflected"]
[Size "8x8"]
[White "Shad Amethyst"]
[Black "PseudoAbstractMeta"]
[Date "2021.01.22"]
[Result "1-0"]

1. f4 / e5
2. f5 / f6
3. g3 / Bd6
4. b3? {This turned out later to be a weak move} / Qh5
5. Bh3 / Qg5
6. Nf3 / (0T6)Qg5>>x(0T4)g3+~
7. (-1T5)hxg3 / (-1T5)e4
8. (-1T6)e3 / (-1T6)g6?
9. (0T7)Bh3>>x(0T5)h5~ / (1T5)Nc6
10. (1T6)Be8+ / (1T6)Kxe8 {Forced}
11. (-1T7)Ne2 (1T7)e3 / (-1T7)gxf5 (0T7)Ne7 (1T7)h5 {Forced}
12. (0T8)Nf3>x(-1T8)f5 (1T8)Nh3 / (-1T8)Qg6 (0T8)Bd6>(1T8)d5
13. (1T9)Nh3>>x(1T8)h5~ / (2T8)Kd8!
14. (0T9)Rf1 (-1T9)Nxd6 (2T9)Qg4 / (-1T9)cxd6 (0T9)Ng8! {Protects (L+2)g7} (1T9)Nd4? (2T9)b6
15. (-1T10)Ba3 (0T10)Ba3 {Threatening mate on f8} (1T10)exd4 (2T10)Ne2 {Protecting (L+1)e4} / (-1T10)f5! (0T10)d6! (1T10)b6 (2T10)Nge7
16. (-1T11)Rg1 (0T11)Qd1>(2T11)f3 (1T11)Nc3 / (-1T11)Nc6 (0T11)Nc6 (1T11)Bxh1? (2T11)Bb7
17. (-1T12)Nf4 {Threatening the queen} (0T12)Nc3 (2T12)Rh1>x(1T12)h1 / (-1T12)Qg6>>x(-1T9)d6~ (>L-2)
18. (-2T10)Rg1 / (-2T10)h5
19. (-2T11)Nf4 / (-2T11)Qg5
20. (-2T12)Qe2 / (-2T12)h4 (0T12)Nd4 (1T12)Ne7 (2T12)Kc8 {Time trouble for black}
21. (2T13)Qg4>>x(2T10)g7~! (>L+3) / (3T10)Bd6>>x(1T10)d4~ (>L-3)
22. (3T11)Qg7g6*! (-3T11)Qh5+ / (-3T11)Rxh5 (2T13)Bd6>>x(2T9)d2~ {One of the two legal moves, black is losing} (>L-4)
23. (-4T10)Bxd2 / 1-0 {White wins by timeout; possible continuation would have been (-4T10)d4; 24. (-4T10)Qxd4#}
)";

}  // namespace

// ---- Rook Tactics I --------------------------------------------------------------------------------

TEST(example_rook_tactics_one_as_written_has_no_position) {
  // The "Full examples" record names no Board (so Standard, "Headers") yet a
  // 5x5 Size, and gives no 5DFEN: it does not say what to set up.
  try {
    PgnReplay r(read_pgn(kRookTacticsFull));
    CHECK(false);
  } catch (const RecordError& e) {
    CHECK(e.fault() == RecordFault::Unreadable);
    CHECK(std::string(e.what()).find("Size header says 5x5") != std::string::npos);
  }
}

TEST(example_rook_tactics_one_replays) {
  // The 5DFEN record, as written: one timeline, so no board coordinates
  // ("Moves"). White K a1, R c1; Black k e5 (no unmoved markers in the
  // specification's string). By hand (T2: each move one board):
  //   1. Kb2 a1-b2 / Ke4 e5-e4;  2. Re1 c1-e1 / Kd3 e4-d3.
  PgnReplay a(read_pgn(kRookTactics5dfen));
  a.play_all();
  CHECK_EQ(a.actions_played(), std::size_t{4});
  CHECK_EQ(board_strings(a.state()), (std::vector<std::string>{
                                         "[4k/5/5/5/K1R2:0:1:w]",
                                         "[4k/5/5/1K3/2R2:0:1:b]",
                                         "[5/4k/5/1K3/2R2:0:2:w]",
                                         "[5/4k/5/1K3/4R:0:2:b]",
                                         "[5/5/3k1/1K3/4R:0:3:w]",
                                     }));
  // The full record's moves on that position, to the mate: 3. Re5, e1-e5.
  const std::string full = "[Size \"5x5\"]\n[Board \"custom\"]\n[4k/5/5/5/K1R2:0:1:w]\n" +
                           std::string(kRookTacticsFull.substr(kRookTacticsFull.find("1.")));
  PgnReplay b(read_pgn(full));
  b.play_all();
  CHECK_EQ(b.actions_played(), std::size_t{5});
  CHECK_EQ(b.state().multiverse().board_count(), std::size_t{6});
  CHECK_EQ(pieces(b.state(), "0", 3, Colour::Black), "4R/5/3k1/1K3/5");
  CHECK(b.state().to_move() == Colour::Black);
  CHECK_EQ(b.record().actions.back().moves.back().mark, '#');  // read, never a verdict
}

// ---- Knight tactics III ------------------------------------------------------------------------------

TEST(example_knight_tactics_three_as_written_has_no_position) {
  const std::string written = "[Size \"5x5\"]\n[Mode \"5D\"]\n\n" + std::string(kKnightTacticsMoves);
  try {
    PgnReplay r(read_pgn(written));
    CHECK(false);
  } catch (const RecordError& e) {
    CHECK(e.fault() == RecordFault::Unreadable);
  }
}

TEST(example_knight_tactics_three_replays_on_a_position_that_fits_it) {
  // The specification gives this puzzle's moves only (Q27: its position is
  // still missing). The position here is NOT the puzzle's: it is the smallest
  // the moves fit, made for this test - White K a1, N b1; Black k d5 (the
  // king the annotation names on d5, and Ke4 on +1 is a step from it), p c4.
  // By hand:
  //   1. Nd2 b1-d2 / c3 c4-c3;  2. Nb3 d2-b3 / c2 c3-c2;
  //   3. (0T3)Nb3>>(0T1)a3: a knight's leap two turns back and one file over
  //      (4.5) onto the history board (0T1) - a branch (J3, J5): timeline 0
  //      gets (0T3, black) without the knight; White's first timeline is +1
  //      (L7), its one board (+1T1, black) a copy of (0T1, white) with the
  //      knight added on a3 - its earlier self still on b1 (J5) - the present
  //      back on it (T8);
  //   3... (1T1)Ke4 d5-e4 on +1;  4. (1T2)Nb5 a3-b5 (the b1 knight cannot
  //      reach b5: no ambiguity).
  //   Boards: 1 + 4 (moves 1-2) + 2 (the branch) + 1 + 1 = 9.
  const std::string record = std::string(kKnightTacticsSetUp) + std::string(kKnightTacticsMoves);
  PgnReplay r(read_pgn(record));
  r.play_all();
  CHECK_EQ(r.actions_played(), std::size_t{7});
  CHECK_EQ(board_strings(r.state()), (std::vector<std::string>{
                                         "[3k*1/2p*2/5/5/K*N3:0:1:w]",
                                         "[3k*1/2p*2/5/3N1/K*4:0:1:b]",
                                         "[3k*1/5/2p2/3N1/K*4:0:2:w]",
                                         "[3k*1/5/1Np2/5/K*4:0:2:b]",
                                         "[3k*1/5/1N3/2p2/K*4:0:3:w]",
                                         "[3k*1/5/5/2p2/K*4:0:3:b]",
                                         "[3k*1/2p*2/N4/5/K*N3:+1:1:b]",
                                         "[5/2p*1k/N4/5/K*N3:+1:2:w]",
                                         "[1N3/2p*1k/5/5/K*N3:+1:2:b]",
                                     }));
  // T4: after White's 4th action the present is on +1 (T2, black).
  CHECK_EQ(r.state().present(), half_turn(2, Colour::Black));
}

// ---- the "Actual game" (2020) ---------------------------------------------------------------

TEST(example_actual_game_as_written_is_an_unsupported_variant) {
  // Simple - No Queens is not among the variants the engine sets up.
  const std::string written = std::string(kActualGameHeaders) + std::string(kActualGameMoves);
  try {
    PgnReplay r(read_pgn(written));
    CHECK(false);
  } catch (const RecordError& e) {
    CHECK(e.fault() == RecordFault::Unsupported);
  }
}

TEST(example_actual_game_from_its_set_up_stops_where_the_record_is_wrong) {
  // The record's moves from V3's Simple - No Queens set-up (kNoQueensSetUp).
  const std::string record = std::string(kNoQueensSetUp) + std::string(kActualGameMoves);

  // Under the default reading (S6) its "O-O" from d1 is unreadable (Q26).
  {
    PgnReplay r(read_pgn(record));
    std::vector<std::size_t> counts;
    const std::optional<RecordError> e = play(r, counts);
    REQUIRE(e.has_value());
    CHECK(e->fault() == RecordFault::Unreadable);
    CHECK_EQ(r.actions_played(), std::size_t{8});
    CHECK(std::string(e->what()).find("(0T5)O-O") != std::string::npos);
  }

  // Under Q26's alternative (O-O castles towards file h) it replays 28
  // actions, then "15. ... (1T13)Ba4+" is impossible: White's bishops on +1
  // are on a3 (12. (1T10)Ba3) and g2 (4. Bg2, copied into +1 by the branch of
  // move 6), and neither reaches a4.
  PgnOptions any;
  any.castling_shortcuts = CastlingShortcuts::AnyKing;
  PgnReplay r(read_pgn(record, any), any);
  std::vector<std::size_t> counts;
  const std::optional<RecordError> e = play(r, counts);
  REQUIRE(e.has_value());
  CHECK(e->fault() == RecordFault::IllegalMove);
  CHECK(std::string(e->what()).find("move 3 \"(1T13)Ba4+\"") != std::string::npos);
  CHECK_EQ(r.actions_played(), std::size_t{28});
  // Board counts by hand (T2: one board per move within a board, two per
  // jump - J4, J5): moves 1-4 one board each; 5. O-O / Be5; 6. the branch
  // (2) / Bb6; 7. Nc3 / Ne7>(1T6)e5 (2); 8. a4, O-O / Be5>>(0T6)e4 (2);
  // 9. Bxe4 / dxe4, c4; 10. Nb3>(0T8)d3 (2), a4 / Bd4, Kd7>(1T8)d6 (2);
  // 11. three / Bd4>x(0T9)d3 (2), cxb3; 12. three / three; 13. three /
  // Rg5, Ne5>x(-1T11)e4 (2); 14. three / three.
  CHECK_EQ(counts, (std::vector<std::size_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                             11, 13, 14, 15, 17, 19, 21, 22, 24, 27,
                                             30, 33, 36, 39, 42, 45, 48, 51, 54}));
  // Three boards at position 28, derived move by move:
  //  +1 (white's, from (0T5, white) by 6. Nb1>>(0T5)b3): Bb6; Nc3; the black
  //   knight arrives on e5 (7...); O-O (Kd1-f1, Rg1-e1); c4; the b3 knight
  //   leaves (10.); Black's king arrives on d6 beside its copy on d7 (10...);
  //   b3; cxb3; Ba3; c5; axb3; the e5 knight leaves (13...); b4; cxb4.
  CHECK_EQ(pieces(r.state(), "+1", 13, Colour::White),
           "r*2k*nbr*/p*b1kp*p*1/3p3/1p4p/B1N3P/2P*P*P*P*B/R*3RK1");
  //  0: O-O; Be5; the b1 knight leaves (6.); the e7 knight leaves (7...); a4;
  //   the e5 bishop leaves (8...); the knight arrives on d3 (10.); the black
  //   king leaves (10...); Ra3; the black bishop takes d3 (11...); cxd3;
  //   Rb7; Rc3; Rg5; d4 (the d3 pawn: d2-d4 is blocked); c4.
  CHECK_EQ(pieces(r.state(), "0", 13, Colour::White),
           "1r3b1/p*1p*1p*p*1/3p2r/P1pP2p/2R3P/1P*1P*P*P*B/2B1RK1");
  //  -1 (black's, from (0T6, black) by 8... Be5>>(0T6)e4 - three black
  //   bishops: e4, its earlier self on e5, f7): Bxe4; dxe4; a4; Bd4 (e5-d4);
  //   Ra3; the d4 bishop leaves (11...); d3; Nd5; dxe4; the knight from +1
  //   takes e4 (13...); Rd3; c4.
  CHECK_EQ(pieces(r.state(), "-1", 13, Colour::White),
           "r*2k*1br*/p*1p*1p*p*1/3n3/P1p1n1p/3R2P/1P*P*1P*P*1/2B1RK1");
  CHECK_EQ(r.state().multiverse().timeline_count(), 3);
}

// ---- Shad vs PseudoAbstractMeta (2021) ------------------------------------------------------

TEST(example_half_reflected_game_replays_to_the_end) {
  // Standard - Half Reflected (V3: Black's king d8, queen e8). Every move,
  // every '~' and every (>Lx) agrees with the state; the record ends "23.
  // (-4T10)Bxd2 / 1-0": Black's action is empty - never played.
  PgnReplay r(read_pgn(kHalfReflectedGame));
  std::vector<std::size_t> counts;
  const std::optional<RecordError> e = play(r, counts);
  if (e) CHECK_EQ(std::string(e->what()), std::string("no error"));
  // Everything below reads the replay's end state: stop here, cleanly, if it
  // did not get there (a timeline looked up below might not exist).
  REQUIRE(!e.has_value());
  REQUIRE_EQ(r.actions_played(), std::size_t{45});
  CHECK(!r.record().unreadable.has_value());
  CHECK_EQ(r.record().result, std::string("1-0"));

  // Board counts by hand: one per move within a board, two per jump (T2):
  // 1.-5. one each; 6. Nf3 / the queen's branch (2); 7.-8. one each;
  // 9. the bishop's branch (2) / Nc6; 10. one each; 11. two / three;
  // 12. Nf3>x(-1T8)f5 (2) and Nh3 / Qg6 and Bd6>(1T8)d5 (2); 13. the
  // knight's branch (2) / Kd8; 14. three / four; 15. four / four;
  // 16. Rg1, Qd1>(2T11)f3 (2), Nc3 / four; 17. Nf4, Nc3, Rh1>x(1T12)h1 (2) /
  // the queen's branch (2); 18.-19. one each; 20. one / four;
  // 21. a branch each (2, 2); 22. two / Rxh5 and a branch (2); 23. one.
  CHECK_EQ(counts, (std::vector<std::size_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                             14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 28, 31,
                                             34, 36, 37, 40, 44, 48, 52, 56, 60, 64, 66, 67,
                                             68, 69, 70, 71, 75, 77, 79, 81, 84, 85}));

  // Eight timelines (L7): Black's -1..-4 in the order made, White's +1..+3,
  // each starting on the half-step successor of the board it branched onto
  // (J5).
  const Multiverse& mv = r.state().multiverse();
  REQUIRE_EQ(mv.min_timeline(), -4);
  REQUIRE_EQ(mv.max_timeline(), 3);
  const std::vector<std::pair<int, int>> first = {
      {-4, half_turn(10, Colour::White)},  // 22... onto (2T9, black)
      {-3, half_turn(11, Colour::White)},  // 21... onto (1T10, black)
      {-2, half_turn(10, Colour::White)},  // 17... onto (-1T9, black)
      {-1, half_turn(5, Colour::White)},   // 6... onto (0T4, black)
      {0, half_turn(1, Colour::White)},
      {1, half_turn(5, Colour::Black)},    // 9. onto (0T5, white)
      {2, half_turn(8, Colour::Black)},    // 13. onto (1T8, white)
      {3, half_turn(10, Colour::Black)},   // 21. onto (2T10, white)
  };
  for (const auto& [p, h] : first) {
    REQUIRE(mv.timeline(p) != nullptr);
    CHECK_EQ(mv.timeline(p)->first_half_turn(), h);
  }

  // 6... (0T6)Qg5>>x(0T4)g3: the queen slides two ranks down and two turns
  // back (4.3) onto (0T4, black) - White's pawn on g3 (3. g3) - and -1's
  // first board is that board's copy with the queen on g3 and its earlier
  // self still on e8 (J5): two black queens.
  CHECK_EQ(pieces(r.state(), "-1", 5, Colour::White),
           "r*nbk*q1nr*/p*p*p*p*2p*p*/3b1p2/4pP2/8/1P4q1/P*1P*P*P*2P*/R*NBQK*BNR*");
  // 21. (2T13)Qg4>>x(2T10)g7: +3 starts from (2T10, white) - +2 copied from
  // (1T8, white) with the h5 knight (13.), then Kd8, Qg4 (14.), b6 (14...) -
  // with the queen on g7; Black's d6 bishop leaves it at once (21...) and
  // 22. (3T11)Qg7g6 follows.
  CHECK_EQ(pieces(r.state(), "+3", 11, Colour::Black),
           "r*1bk2nr*/p*1p*p*4/1pn2pQ1/4pP1N/6Q1/1P2P1P1/P*1P*P*3P*/R*NB1K*BNR*");
  // 22... (2T13)Bd6>>x(2T9)d2: -4 starts from (2T9, black) - before b6 -
  // with the bishop on d2 (White's unmoved d-pawn taken) and its earlier self
  // still on d6; then 23. (-4T10)Bxd2, c1-d2.
  CHECK_EQ(pieces(r.state(), "-4", 10, Colour::White),
           "r*1bk2nr*/p*p*p*p*2p*1/2nb1p2/4pP1N/6Q1/1P2P1P1/P*1P*b3P*/R*NB1K*BNR*");
  CHECK_EQ(pieces(r.state(), "-4", 10, Colour::Black),
           "r*1bk2nr*/p*p*p*p*2p*1/2nb1p2/4pP1N/6Q1/1P2P1P1/P*1P*B3P*/R*N2K*BNR*");
  CHECK(r.state().to_move() == Colour::Black);

  // The final state round-trips through 5DFEN: its board strings read back as
  // the same boards, fingerprint and key. (Not as an equal GameState: 5DFEN
  // does not record which board a timeline branched from, Timeline::origin,
  // which no rule reads and the key leaves out.)
  const GameState again = game_from_fen(board_strings(r.state()), BoardSize{8, 8}, Parity::Infer,
                                        r.state().rules(), Colour::Black);
  CHECK_EQ(board_strings(again), board_strings(r.state()));
  CHECK(fingerprint(again.multiverse()) == fingerprint(r.state().multiverse()));
  CHECK_EQ(again.zobrist(), r.state().zobrist());
  CHECK_EQ(again.present(), r.state().present());
  CHECK(again.inactive_timelines() == r.state().inactive_timelines());
}

TEST(example_records_round_trip_every_move) {
  // At every position the replays reach, every pseudo-legal move written in
  // full 5dpgn reads back as exactly that move - the reader and the generator
  // agree on every move of the examples' positions, jumps, branches,
  // castling, promotions and captures included. (Counts: actions replayed,
  // plus the set-up.)
  CHECK_EQ(round_trip_every_move(kHalfReflectedGame), std::size_t{46});
  PgnOptions any;
  any.castling_shortcuts = CastlingShortcuts::AnyKing;
  CHECK_EQ(round_trip_every_move(std::string(kNoQueensSetUp) + std::string(kActualGameMoves), any),
           std::size_t{29});
  CHECK_EQ(round_trip_every_move(std::string(kKnightTacticsSetUp) +
                                 std::string(kKnightTacticsMoves)),
           std::size_t{8});
  CHECK_EQ(round_trip_every_move(kRookTactics5dfen), std::size_t{5});
  // And at every named variant's set-up: unicorns, brawns, royal queens and
  // common kings, jumps between initial timelines.
  for (const Variant& v : variants()) {
    CHECK_EQ(round_trip_every_move("[Board \"" + std::string(v.name) + "\"]"), std::size_t{1});
  }
}

TEST(example_random_games_round_trip_through_5dpgn) {
  // Random pseudo-legal games from every variant, written as 5dpgn records
  // (turn markers, every move in full) and read back: the replay reaches the
  // same state, action by action.
  for (const Variant& v : variants()) {
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      t5d_test::Rng rng(seed * 0x9E3779B97F4A7C15ull + v.name.size());
      GameState s = variant_start(v);
      const bool even = s.multiverse().even();
      std::string text = "[Board \"" + std::string(v.name) + "\"]\n";
      std::vector<GameState> states{s};
      int turn = 1;
      for (int action = 0; action < 14; ++action) {
        GameState t = s;
        std::string moves;
        bool stuck = false;
        MoveList list;
        while (!t.obligations_met()) {
          generate_moves(t, list);
          if (list.empty()) {
            stuck = true;
            break;
          }
          const Move m = list[static_cast<std::size_t>(rng.below(static_cast<int>(list.size())))];
          moves += ' ' + pgn_text(m, even);
          t.apply(m);
        }
        if (stuck) break;
        t.submit();
        text += s.to_move() == Colour::White ? "\n" + std::to_string(turn) + "." : " /";
        if (s.to_move() == Colour::Black) ++turn;
        text += moves;
        s = t;
        states.push_back(s);
      }
      PgnReplay r(read_pgn(text));
      REQUIRE_EQ(r.record().actions.size(), states.size() - 1);
      for (std::size_t k = 1; k < states.size(); ++k) {
        r.play_next();
        if (!(r.state() == states[k])) {
          CHECK_EQ(std::string(v.name) + " action " + std::to_string(k), std::string("equal"));
          break;
        }
      }
      CHECK(r.finished());
    }
  }
}
