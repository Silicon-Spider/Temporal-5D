// The 5dpgn reader (t5d/pgn.hpp), stage 1: text -> headers, board strings and
// moves as written, no state. Every record here is written for the test from
// the notation specification's text (its README's sections "Coordinates",
// "Moves", "Castling", "Jumps", "Promotions", "Inactive timeline
// reactivation", "Complex scenarios", "Turns", "Headers", "Comments", "5DFEN
// and custom variants") - its quoted examples are its own.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/pgn.hpp"
#include "test_framework.hpp"

using namespace t5d;

namespace {

PgnMove read_move(std::string_view text) { return read_pgn_move(text); }

TimelineLabel label(std::string_view text) { return parse_timeline_label(text); }

std::vector<std::string> texts(const PgnAction& a) {
  std::vector<std::string> out;
  for (const PgnMove& m : a.moves) out.push_back(m.text);
  return out;
}

}  // namespace

// ---- moves ---------------------------------------------------------------------------------

TEST(pgn_move_physical_san_forms) {
  // "Physical moves are written the same as traditional chess standard
  // algebraic notation."
  PgnMove m = read_move("e4");
  CHECK(m.kind == PgnMove::Kind::Ordinary);
  CHECK(!m.board.has_value());
  CHECK(!m.piece.has_value());  // no letter: a pawn
  CHECK_EQ(m.to_file, 4);
  CHECK_EQ(m.to_rank, 3);
  CHECK_EQ(m.from_file, -1);
  CHECK_EQ(m.from_rank, -1);
  CHECK(!m.capture);
  CHECK(m.jump == PgnMove::Jump::None);

  m = read_move("exd5");
  CHECK_EQ(m.from_file, 4);
  CHECK(m.capture);
  CHECK_EQ(m.to_file, 3);
  CHECK_EQ(m.to_rank, 4);

  m = read_move("Nbd7");
  CHECK(m.piece == PieceKind::Knight);
  CHECK_EQ(m.from_file, 1);
  CHECK_EQ(m.from_rank, -1);

  m = read_move("R1a3");
  CHECK(m.piece == PieceKind::Rook);
  CHECK_EQ(m.from_rank, 0);
  CHECK_EQ(m.from_file, -1);

  m = read_move("Qh4xe1");
  CHECK_EQ(m.from_file, 7);
  CHECK_EQ(m.from_rank, 3);
  CHECK(m.capture);
  CHECK_EQ(m.to_file, 4);
  CHECK_EQ(m.to_rank, 0);

  // The specification's own "(3T11)Qg7g6*!": a full source square, the
  // softmate mark and an assessment.
  m = read_move("(3T11)Qg7g6*!");
  REQUIRE(m.board.has_value());
  CHECK(m.board->timeline == label("3"));
  CHECK_EQ(m.board->turn, 11);
  CHECK_EQ(m.from_file, 6);
  CHECK_EQ(m.from_rank, 6);
  CHECK_EQ(m.to_rank, 5);
  CHECK_EQ(m.mark, '*');

  // A pawn may be written with its letter.
  m = read_move("Pe4");
  CHECK(m.piece == PieceKind::Pawn);
  // "e2e4" - the pawn's full square.
  m = read_move("e2e4");
  CHECK_EQ(m.from_file, 4);
  CHECK_EQ(m.from_rank, 1);
}

TEST(pgn_move_every_piece_letter) {
  // "Moves": P W K C Q Y/RQ S/PR N R B U D.
  const std::vector<std::pair<std::string, PieceKind>> letters = {
      {"P", PieceKind::Pawn},     {"W", PieceKind::Brawn},      {"K", PieceKind::King},
      {"C", PieceKind::CommonKing}, {"Q", PieceKind::Queen},    {"Y", PieceKind::RoyalQueen},
      {"RQ", PieceKind::RoyalQueen}, {"S", PieceKind::Princess}, {"PR", PieceKind::Princess},
      {"N", PieceKind::Knight},   {"R", PieceKind::Rook},       {"B", PieceKind::Bishop},
      {"U", PieceKind::Unicorn},  {"D", PieceKind::Dragon}};
  for (const auto& [text, kind] : letters) {
    const PgnMove m = read_move(text + "c3");
    CHECK(m.piece == kind);
    CHECK_EQ(m.to_file, 2);
    CHECK_EQ(m.to_rank, 2);
  }
  // Lower-case letters are files, never pieces: "bxc3" is a pawn.
  const PgnMove b = read_move("bxc3");
  CHECK(!b.piece.has_value());
  CHECK_EQ(b.from_file, 1);
}

TEST(pgn_move_board_coordinates_both_forms) {
  // "Coordinates": "(L<a> T<b>), which can be shortened to (<a>T<b>). Both
  // should be considered valid by parsers".
  const PgnMove a = read_move("(0T1)Nf3");
  const PgnMove b = read_move("(L0 T1)Nf3");
  REQUIRE(a.board.has_value());
  REQUIRE(b.board.has_value());
  CHECK(*a.board == *b.board);
  CHECK_EQ(a.board->turn, 1);
  CHECK(read_move("(-1T6)e4").board->timeline == label("-1"));
  CHECK(read_move("(L-1 T6)e4").board->timeline == label("-1"));
  CHECK(read_move("(+1T3)e4").board->timeline == label("+1"));
  CHECK(read_move("(1T3)e4").board->timeline == label("1"));  // the sign may be left out
  CHECK(read_move("(-0T1)e4").board->timeline == label("-0"));
  CHECK(read_move("(+0T1)e4").board->timeline == label("+0"));
  const PgnMove zero = read_move("(0T0)e4");  // "Turn zero"
  CHECK_EQ(zero.board->turn, 0);
  const PgnMove far = read_move("(-12T345)e4");
  CHECK_EQ(far.board->turn, 345);
  CHECK(far.board->timeline == label("-12"));
}

TEST(pgn_move_jumps) {
  // "Jumps", the specification's examples.
  PgnMove m = read_move("(-1T4)Nc3>>x(0T2)c3+~");
  CHECK(m.jump == PgnMove::Jump::Branching);
  CHECK(m.capture);
  CHECK(m.piece == PieceKind::Knight);
  CHECK(m.board->timeline == label("-1"));
  CHECK_EQ(m.board->turn, 4);
  CHECK_EQ(m.from_file, 2);
  CHECK_EQ(m.from_rank, 2);
  REQUIRE(m.target_board.has_value());
  CHECK(m.target_board->timeline == label("0"));
  CHECK_EQ(m.target_board->turn, 2);
  CHECK_EQ(m.to_file, 2);
  CHECK_EQ(m.to_rank, 2);
  CHECK_EQ(m.mark, '+');
  CHECK(m.present_moved);

  m = read_move("(0T6)Pd5>(1T6)d5");
  CHECK(m.jump == PgnMove::Jump::NonBranching);
  CHECK(!m.capture);
  CHECK(m.piece == PieceKind::Pawn);

  // A pawn's jump without its letter (the 2020 example game's "(-1T19)c6>(0T19)c6").
  m = read_move("(-1T19)c6>(0T19)c6");
  CHECK(!m.piece.has_value());
  CHECK(m.jump == PgnMove::Jump::NonBranching);

  // The fully detailed form of "Complex scenarios", annotation tokens attached.
  m = read_move("(-1T4)Nc3>>x(0T2)c3+(>L2)(~T1)");
  CHECK(m.created == label("2"));
  CHECK(m.present_turn == 1);

  // A promotion on a jump (a brawn's capture onto another board, S8).
  m = read_move("(0T5)Wd7>x(1T5)d8=Q#");
  CHECK(m.promotion == PieceKind::Queen);
  CHECK_EQ(m.mark, '#');
  // Long-form target board.
  m = read_move("(L0 T5)Qd1>(L2 T5)f3");
  CHECK(m.target_board->timeline == label("2"));
}

TEST(pgn_move_promotions_and_marks) {
  PgnMove m = read_move("e8=Q");
  CHECK(m.promotion == PieceKind::Queen);
  m = read_move("exd8=N+");
  CHECK(m.promotion == PieceKind::Knight);
  CHECK_EQ(m.mark, '+');
  CHECK(read_move("a1=RQ").promotion == PieceKind::RoyalQueen);
  CHECK(read_move("a1=PR").promotion == PieceKind::Princess);
  CHECK(read_move("a1=S").promotion == PieceKind::Princess);
  // '+', '#', '*' and the assessments are read and kept, never checked.
  CHECK_EQ(read_move("Nf3#").mark, '#');
  CHECK_EQ(read_move("Nf3*").mark, '*');
  CHECK_EQ(read_move("c5?").mark, '\0');
  CHECK_EQ(read_move("Ng8!").mark, '\0');
  CHECK_EQ(read_move("Nd4!?").to_rank, 3);
  CHECK(read_move("(2T13)Qg4>>x(2T10)g7~!").present_moved);
}

TEST(pgn_move_castling) {
  PgnMove m = read_move("O-O");
  CHECK(m.kind == PgnMove::Kind::CastleKingside);
  CHECK(m.piece == PieceKind::King);
  m = read_move("(0T5)O-O-O+");
  CHECK(m.kind == PgnMove::Kind::CastleQueenside);
  CHECK(m.board.has_value());
  CHECK_EQ(m.mark, '+');
  // "K<file_from><rank_from><file_to><rank_to>": an ordinary king move.
  m = read_move("Ke1g1");
  CHECK(m.kind == PgnMove::Kind::Ordinary);
  CHECK(m.piece == PieceKind::King);
  CHECK_EQ(m.from_file, 4);
  CHECK_EQ(m.to_file, 6);
}

TEST(pgn_move_malformed_is_unreadable) {
  for (const std::string_view bad :
       {"", "e9", "i4", "Ke", "K", "e4x", "Nbbd7", "N1bd7", "e4=", "e4=X", "e8Q", "O-O-O-O",
        "O-0", "o-o", "(0T1)", "(0T1) e4", "(0T)e4", "(T1)e4", "(0T01)e4", "(00T1)e4", "(0t1)e4",
        "(0T1e4", "(1. e4 e5)", "Nc>(0T2)c3", "Nc3>>", "Nc3>>x", "Nc3>>(0T2)", "(0T1)Nc>(0T2)c3",
        "Nf3++", "Nf3!!!", "e4~~", "nf3", "O-O=Q", "(>L2)e4", "e4(>L)", "e4(~T)", "e4(~Tx)",
        "e4(>L2)(>L3)", "e4(x)", "e4 e5", "Nf3/"}) {
    CHECK_THROWS_WITH(read_pgn_move(bad), RecordError, "cannot read the move");
  }
  // The message names the move and what is wrong.
  CHECK_THROWS_WITH(read_pgn_move("(1. e4 e5)"), RecordError, "variation");
  CHECK_THROWS_WITH(read_pgn_move("Nc>(0T2)c3"), RecordError, "in full");
  CHECK_THROWS_WITH(read_pgn_move("e\xC3\xA9" "5"), RecordError, "byte 0xC3");
  // A jump without its source board is well formed (one timeline: "Moves").
  CHECK_NOTHROW(read_pgn_move("Nc3>(0T2)c3"));
}

// ---- records ---------------------------------------------------------------------------------

TEST(pgn_headers_and_board_strings) {
  const PgnRecord r = read_pgn(
      "[Board \"custom\"]\n"
      "[Size\"5x5\"]  [Puzzle \"mate-in-1\"]\n"  // no space before the value; two on a line
      "[Event \"A \\\"quoted\\\" name \\\\ [x]\"]\n"
      "[4k/5/5/5/K1R2:0:1:w]\n"
      "\n"
      "1. Kb2 / Ke4\n");
  REQUIRE_EQ(r.headers.size(), std::size_t{4});
  CHECK_EQ(r.headers[0].name, std::string("Board"));
  CHECK_EQ(*r.header("board"), std::string("custom"));  // names without regard to case
  CHECK_EQ(*r.header("SIZE"), std::string("5x5"));
  CHECK_EQ(*r.header("Puzzle"), std::string("mate-in-1"));
  CHECK_EQ(*r.header("Event"), std::string("A \"quoted\" name \\ [x]"));
  CHECK(r.header("Mode") == nullptr);
  REQUIRE_EQ(r.boards.size(), std::size_t{1});
  CHECK_EQ(r.boards[0], std::string("[4k/5/5/5/K1R2:0:1:w]"));
  CHECK_EQ(r.actions.size(), std::size_t{2});
  CHECK(!r.unreadable.has_value());
}

TEST(pgn_header_faults_are_kept_for_the_set_up) {
  // A fault in the headers or board strings is recorded - the first one, with
  // where it is - and never thrown: the set-up raises it only after deciding
  // whether the record's game is one the engine plays (t5d/pgn.hpp).
  const auto fault_of = [](std::string_view text) {
    const PgnRecord r = read_pgn(text);
    return r.headers_unreadable.value_or(std::string("(none)"));
  };
  const auto says = [&](std::string_view text, std::string_view part) {
    const std::string message = fault_of(text);
    if (message.find(part) != std::string::npos) return true;
    CHECK_EQ(message, std::string(part));
    return false;
  };
  CHECK(says("[Board \"custom]\n1. e4", "unterminated value"));
  CHECK(says("[Board \"x\" junk]", "expected ']'"));
  CHECK(says("[Board custom]", "as a header"));
  CHECK(says("[4k/5/5/5/K1R2:0:1:w", "board string"));
  CHECK(says("[Board \"A\"]\n[board \"B\"]", "two values"));
  CHECK(says("{ unterminated", "unterminated comment"));
  CHECK(says("\n\n  [Board \"A\"][Board \"B\"]", "line 3, column 14"));  // where
  CHECK_EQ(fault_of("[Board \"Standard\"]\n1. e4"), std::string("(none)"));

  // Reading goes on after a fault - past the next ']' on its line, or from
  // the next line - so the headers after it are read, and so is the move text.
  PgnRecord r = read_pgn("[Event \"x\" junk][Board \"Small\"]\n"
                         "[Date \"2020.10.07]\n"
                         "[White Player \"y\"]\n"
                         "[Mode \"5D\"]\n"
                         "1. e3");
  REQUIRE(r.headers_unreadable.has_value());
  CHECK(r.headers_unreadable->find("line 1, column 12: header \"Event\": expected ']'") !=
        std::string::npos);  // the first fault
  REQUIRE(r.header("Board") != nullptr);
  CHECK_EQ(*r.header("Board"), std::string("Small"));
  REQUIRE(r.header("Mode") != nullptr);
  CHECK_EQ(*r.header("Event"), std::string("x"));  // its value was read in full: kept
  CHECK(r.header("Date") == nullptr);              // an unterminated value is not
  CHECK_EQ(r.actions.size(), std::size_t{1});
  CHECK(!r.unreadable.has_value());
  // A value read in full is kept even when its ']' is missing.
  r = read_pgn("[Board \"Simple - No Queens\"\n1. e3");
  REQUIRE(r.header("Board") != nullptr);
  CHECK_EQ(*r.header("Board"), std::string("Simple - No Queens"));
  CHECK(r.headers_unreadable.has_value());

  // A repeated header with the same value changes nothing - also written
  // another way, as the engine reads it; other headers may repeat.
  r = read_pgn("[Board \"Standard\"][Board \"Standard\"][Note \"a\"][Note \"b\"]");
  CHECK_EQ(r.headers.size(), std::size_t{3});
  r = read_pgn("[Size \"7x7\"][Size \"7X7\"][Board \"Very  Small\"][board \"very small \"]"
               "[Mode \"5D\"][Mode \"5d\"][Promotions \"Q,R\"][Promotions \"r q\"]");
  CHECK(!r.headers_unreadable.has_value());
  CHECK_EQ(r.headers.size(), std::size_t{4});
  CHECK_EQ(*r.header("Size"), std::string("7x7"));  // the first value is kept
  CHECK(read_pgn("[Size \"7x7\"][Size \"8x8\"]").headers_unreadable.has_value());
  CHECK(read_pgn("[Promotions \"Q\"][Promotions \"Q,N\"]").headers_unreadable.has_value());
}

TEST(pgn_header_name_and_value_may_be_split_by_a_line_break) {
  // fen.ebnf: HEADER = '[', {key_character}, {space}, '"', ... with
  // space = ' ' | '\n'. CR (a CR-LF file) and tabs read the same way.
  for (const std::string_view text :
       {"[Board\n\"Standard\"]\n1. e4", "[Board\r\n\"Standard\"]\r\n1. e4",
        "[Board \n\t \"Standard\"]\n1. e4", "[Board\"Standard\"]\n1. e4"}) {
    const PgnRecord r = read_pgn(text);
    CHECK(!r.headers_unreadable.has_value());
    REQUIRE(r.header("Board") != nullptr);
    CHECK_EQ(*r.header("Board"), std::string("Standard"));
    CHECK_EQ(r.actions.size(), std::size_t{1});
  }
}

TEST(pgn_turns_and_actions) {
  const PgnRecord r = read_pgn(
      "1. (0T1)d4 / (0T1)d6\n"
      "2w. (0T2)Nf3 2b. (0T2)Nf6\n"
      "3. (0T3)e3 (0T3)e3 / 4.c4\n");
  // "1." White, "/" Black of the same turn, "2w." and "2b.", then "3." White
  // with two moves, Black's "/"... and "4." - which follows Black's empty
  // action: unreadable there.
  REQUIRE(r.unreadable.has_value());
  CHECK(r.unreadable->find("Black's action of turn 3 has no moves") != std::string::npos);
  REQUIRE_EQ(r.actions.size(), std::size_t{5});
  CHECK(r.actions[0].colour == Colour::White);
  CHECK_EQ(r.actions[0].turn_number, 1);
  CHECK(r.actions[1].colour == Colour::Black);
  CHECK_EQ(r.actions[1].turn_number, 1);
  CHECK(r.actions[2].colour == Colour::White);
  CHECK_EQ(r.actions[2].turn_number, 2);
  CHECK(r.actions[3].colour == Colour::Black);
  CHECK_EQ(r.actions[3].turn_number, 2);
  CHECK_EQ(texts(r.actions[4]), (std::vector<std::string>{"(0T3)e3", "(0T3)e3"}));
}

TEST(pgn_black_first_records) {
  // "Turns": "In some rare cases, especially in puzzles, black's moves need to
  // be annotated first" - the specification's example.
  const PgnRecord r = read_pgn(
      "1b. (0T10)Qg6+\n"
      "2w. (0T11)Kh1 / (0T11)Bb7+\n"
      "3. (0T12)Nf3 / (0T12)Qg6>>(0T8)g2#~\n");
  CHECK(!r.unreadable.has_value());
  REQUIRE_EQ(r.actions.size(), std::size_t{5});
  CHECK(r.actions[0].colour == Colour::Black);
  CHECK_EQ(r.actions[0].turn_number, 1);
  CHECK(r.actions[1].colour == Colour::White);
  CHECK_EQ(r.actions[1].turn_number, 2);
  CHECK(r.actions[4].colour == Colour::Black);
  CHECK_EQ(r.actions[4].turn_number, 3);
  CHECK(r.actions[4].moves[0].present_moved);
  CHECK_EQ(r.actions[4].moves[0].mark, '#');
}

TEST(pgn_comments_results_and_annotation_tokens) {
  const PgnRecord r = read_pgn(
      "{before} 1. e4 {after a move} / e5{touching}\n"
      "2. (0T2)Nf3 / (0T2)Bd6>>x(0T1)d2~ {a comment} (>L-1) (~T1)\n"
      "3. (-1T2)Kxd2 / 1-0 {Black forfeits; 4. e4 is no move}");
  CHECK(!r.unreadable.has_value());
  CHECK_EQ(r.result, std::string("1-0"));
  // "3. ... / 1-0": Black's action has no moves - it was never played.
  REQUIRE_EQ(r.actions.size(), std::size_t{5});
  const PgnMove& branch = r.actions[3].moves[0];
  CHECK(branch.created == label("-1"));
  CHECK(branch.present_turn == 1);
  CHECK(branch.text.find("(>L-1)") != std::string::npos);
  CHECK(r.actions[4].colour == Colour::White);

  for (const std::string_view result : {"0-1", "1/2-1/2", "*"}) {
    const PgnRecord q = read_pgn("1. e4 / e5 " + std::string(result));
    CHECK(!q.unreadable.has_value());
    CHECK_EQ(q.result, std::string(result));
    CHECK_EQ(q.actions.size(), std::size_t{2});
  }
  // A '*' attached to a move is the softmate mark, not the result.
  const PgnRecord s = read_pgn("1. e4* / e5");
  CHECK(s.result.empty());
  CHECK_EQ(s.actions[0].moves[0].mark, '*');
}

TEST(pgn_move_text_errors_stop_reading_after_the_complete_actions) {
  struct Case {
    std::string_view text;
    std::size_t actions;    // complete actions kept
    std::string_view says;  // part of the message
  };
  const std::vector<Case> cases = {
      {"1. e4 / e5 2. d4 / ??", 3, "unexpected '?'"},
      {"1. e4 / e5 3. d4", 2, "turn numbers count the turns"},
      {"1. e4 2. d4", 1, "Black's action of turn 1 is missing"},
      {"1. e4 / e5 1b. d5", 2, "White's action is missing"},
      {"1. e4 1b. e5 2. d4 3b. d5", 3, "a turn is White's action, then Black's"},
      {"1. / e5", 0, "has no moves"},
      {"/ e5", 0, "must follow White's moves"},
      {"e4 e5", 0, "before the first turn number"},
      {"1... e5", 0, "introduced by '/'"},
      {"0. e4", 0, "positive number"},
      {"01. e4", 0, "positive number"},
      {"1. e4 / e5 1-0 2. d4", 2, "text after the result"},
      // An error inside an action drops it: its end was never seen.
      {"1. e4 / e5 [Board \"Standard\"]", 1, "after the move text began"},
      {"1. e4 / e5 }", 1, "'}' without '{'"},
      {"1. e4 / e5 {open", 1, "unterminated comment"},
      {"1. 0-0 / e5", 0, "letter O"},
      {"1. e4 (1. d4) / e5", 0, "variation"},
      {"1. e4 (0T1 / e5", 0, "unterminated '('"},
      {"1. (>L1) e4", 0, "follows no move"},
      {"1. e4 / e5 2. 1-0x", 2, "cannot read \"1-0x\""},
      {"1. e4 / e\xC3\xA9" "5", 1, "byte 0xC3"},
      {"1. e4 / e5 2. Nf3 Nxx3", 2, "cannot read the move"},
  };
  for (const Case& c : cases) {
    const PgnRecord r = read_pgn(c.text);
    REQUIRE(r.unreadable.has_value());
    CHECK_EQ(r.actions.size(), c.actions);
    if (r.unreadable->find(c.says) == std::string::npos) {
      CHECK_EQ(*r.unreadable, std::string(c.says));
    }
  }
}

TEST(pgn_turn_numbers_may_be_ignored) {
  PgnOptions loose;
  loose.check_turn_numbers = false;
  const PgnRecord r = read_pgn("5. e4 / e5 9. d4 / d5", loose);
  CHECK(!r.unreadable.has_value());
  CHECK_EQ(r.actions.size(), std::size_t{4});
  // Colours must still alternate.
  CHECK(read_pgn("1. e4 2. d4", loose).unreadable.has_value());
  // A first turn number other than 1 is accepted anyway (a record may start late).
  CHECK(!read_pgn("5. e4 / e5 6. d4").unreadable.has_value());
}

TEST(pgn_reader_takes_any_line_ending_and_a_byte_order_mark) {
  const std::string text =
      "\xEF\xBB\xBF[Board \"Standard\"]\r\n\r\n1. e4 / e5\r\n2. Nf3\t/\tNc6\r\n";
  const PgnRecord r = read_pgn(text);
  CHECK(!r.unreadable.has_value());
  CHECK_EQ(*r.header("Board"), std::string("Standard"));
  CHECK_EQ(r.actions.size(), std::size_t{4});
  // Lines and columns are counted for messages.
  const PgnRecord q = read_pgn("[Board \"Standard\"]\n\n1. e4 / e5\n2. Nf3 / ?");
  REQUIRE(q.unreadable.has_value());
  CHECK(q.unreadable->find("line 4, column 10") != std::string::npos);
  CHECK_EQ(q.actions[1].moves[0].line, std::size_t{3});
  CHECK_EQ(q.actions[1].moves[0].column, std::size_t{9});
}

TEST(pgn_turn_marker_may_touch_its_move) {
  const PgnRecord r = read_pgn("1.e4 /e5 2.(0T2)Nf3/(0T2)Nc6");
  CHECK(!r.unreadable.has_value());
  CHECK_EQ(r.actions.size(), std::size_t{4});
  CHECK(r.actions[2].moves[0].board.has_value());
}
