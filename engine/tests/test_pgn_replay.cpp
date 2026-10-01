// The 5dpgn replay (t5d/pgn.hpp), stage 2: the set-up a record's headers
// declare, and every written move matched against the pseudo-legal moves of
// the board it names. Positions and records are written for the test from the
// rules reference (docs/rules/README.md) and the notation specification's
// text.

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/state.hpp"
#include "t5d/timeline.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using t5d_test::at;

namespace {

PgnReplay replay(std::string_view text, const PgnOptions& options = {}) {
  return PgnReplay(read_pgn(text, options), options);
}

// Plays every action; returns the fault of the first that fails (nullopt if
// none), leaving `r` after the actions before it.
std::optional<RecordFault> play(PgnReplay& r, std::string* message = nullptr) {
  try {
    r.play_all();
  } catch (const RecordError& e) {
    if (message != nullptr) *message = e.what();
    return e.fault();
  }
  return std::nullopt;
}

// The fault (and message) of setting up and playing a whole record.
std::optional<RecordFault> outcome(std::string_view text, const PgnOptions& options = {},
                                   std::string* message = nullptr,
                                   std::size_t* played = nullptr) {
  try {
    PgnReplay r = replay(text, options);
    const std::optional<RecordFault> f = play(r, message);
    if (played != nullptr) *played = r.actions_played();
    return f;
  } catch (const RecordError& e) {
    if (message != nullptr) *message = e.what();
    if (played != nullptr) *played = 0;
    return e.fault();
  }
}

bool fails_with(std::string_view text, RecordFault fault, std::string_view says,
                const PgnOptions& options = {}) {
  std::string message;
  const std::optional<RecordFault> f = outcome(text, options, &message);
  if (f != fault || message.find(says) == std::string::npos) {
    std::fprintf(stderr, "  record %s\n  gave %s: %s\n", std::string(text).c_str(),
                 f ? fault_name(*f) : "no fault", message.c_str());
    return false;
  }
  return true;
}

bool plays(std::string_view text, const PgnOptions& options = {}) {
  std::string message;
  const std::optional<RecordFault> f = outcome(text, options, &message);
  if (f) std::fprintf(stderr, "  record %s\n  gave %s: %s\n", std::string(text).c_str(),
                      fault_name(*f), message.c_str());
  return !f.has_value();
}

const Board* head(const PgnReplay& r, int p) { return r.state().multiverse().head(p); }

std::string pieces_at(const PgnReplay& r, int p, int turn, Colour c) {
  const Board* b = r.state().multiverse().board_at(p, half_turn(turn, c));
  return b != nullptr ? fen_pieces(*b) : std::string("(none)");
}

}  // namespace

// ---- the set-up ---------------------------------------------------------------------------------

TEST(pgn_setup_the_game_comes_from_board_then_variant_then_standard) {
  // "Headers": Board "defaults to Standard"; Variant read when Board is absent
  // (spec issue 2g).
  const auto game_of = [](std::string_view text) { return replay(text).game(); };  // a copy
  CHECK_EQ(game_of("1. e4"), std::string("Standard"));
  CHECK_EQ(game_of("[Board \"Standard - Two Timelines\"]"),
           std::string("Standard - Two Timelines"));
  CHECK_EQ(game_of("[Variant \"Very Small\"]"), std::string("Very Small"));
  CHECK_EQ(game_of("[Variant \"Very Small\"][Board \"Small\"]"), std::string("Small"));
  CHECK_EQ(game_of("[Board \"misc - small\"]"), std::string("Small"));
  const PgnReplay two = replay("[Board \"Standard - Two Timelines\"]");
  CHECK(two.state().multiverse().even());
  CHECK_EQ(two.state().multiverse().timeline_count(), 2);
  CHECK(two.finished());
}

TEST(pgn_setup_custom_positions) {
  // "5DFEN and custom variants": Board "custom", Size, board strings.
  PgnReplay r = replay("[Board \"custom\"][Size \"5x5\"][4k/5/5/5/K1R2:0:1:w]");
  CHECK_EQ(r.game(), std::string("custom"));
  CHECK(r.state().multiverse().board_size() == (BoardSize{5, 5}));
  CHECK(r.state().to_move() == Colour::White);
  // The Size header is taken from the board strings when it is missing, and
  // board strings without a Board header set up a custom position too.
  CHECK(replay("[Board \"Custom\"][2k/3/3:0:1:w]").state().multiverse().board_size() ==
        (BoardSize{3, 3}));
  const PgnReplay no_board_header = replay("[2k/3/3:0:1:w]");
  CHECK_EQ(no_board_header.game(), std::string("custom"));
  // The player to move is the colour to move on the present (T4, T5; Q9).
  CHECK(replay("[Board \"custom\"][2k/3/K2:0:7:b]").state().to_move() == Colour::Black);
  CHECK(plays("[Board \"custom\"][2k/3/K2:0:7:b]\n1b. (0T7)Kc2 2. (0T8)Kb2"));
  // A first action by the other player does not stop the set-up - position 0
  // exists - and is refused as action 1 (Unreadable), the state unchanged.
  for (const std::string_view text :
       {"[Board \"custom\"][2k/3/K2:0:7:b]\n1. (0T7)Kb2",
        "[Board \"custom\"][Size \"5x5\"][4k/5/5/5/K1R2:0:2:w][4k/5/5/5/K1R2:1:1:b]\n"
        "1. (0T2)Kb1"}) {
    PgnReplay wrong = replay(text);
    CHECK(wrong.state().to_move() == Colour::Black);
    std::string message;
    CHECK(play(wrong, &message) == RecordFault::Unreadable);
    CHECK(message.find("action 1 (White's, turn 1): it is Black's turn (the custom position's "
                       "present") != std::string::npos);
    CHECK_EQ(wrong.actions_played(), std::size_t{0});
  }
  // Malformed or inconsistent set-ups.
  CHECK(fails_with("[Board \"custom\"][Size \"5x5\"]", RecordFault::Unreadable,
                   "has none"));
  CHECK(fails_with("[Board \"custom\"][Size \"4x4\"][4k/5/5/5/K1R2:0:1:w]",
                   RecordFault::Unreadable, "row"));
  CHECK(fails_with("[Board \"custom\"][4k/5/5/5/K1R2:0:1:x]", RecordFault::Unreadable,
                   "colour"));
  CHECK(fails_with("[Board \"custom\"][4k/5/5/5/K1R2:+1:1:w]", RecordFault::Unreadable,
                   "central"));
  CHECK(fails_with("[Board \"Standard\"][4k/5/5/5/K1R2:0:1:w]", RecordFault::Unreadable,
                   "needs Board \"custom\""));
}

TEST(pgn_setup_custom_board_names) {
  // "custom" ("5DFEN and custom variants"), and "Even", "Odd" and the names
  // beginning "Custom", which the gate's engine contract files with it under
  // its custom positions: set up from the board strings, whose labels give the
  // parity - which a name that states one must agree with.
  const std::string even = "[4k/5/5/5/K1R2:-0:1:w][4k/5/5/5/K1R2:+0:1:w]";
  const std::string odd = "[4k/5/5/5/K1R2:0:1:w]";
  CHECK_EQ(std::string(replay("[Board \"Even\"][Size \"5x5\"]" + even).game()),
           std::string("custom"));  // a copy: the replay is a temporary
  CHECK(replay("[Board \"even\"]" + even).state().multiverse().even());
  CHECK_EQ(replay("[Board \"Even\"]" + even).state().multiverse().timeline_count(), 2);
  CHECK(!replay("[Board \"Odd\"]" + odd).state().multiverse().even());
  CHECK(replay("[Board \"Custom - Even\"]" + even).state().multiverse().even());
  CHECK(!replay("[Board \"Custom - Odd\"]" + odd).state().multiverse().even());
  CHECK(replay("[Board \"Custom\"]" + even).state().multiverse().even());  // states none
  CHECK(replay("[Variant \"custom-even\"]" + even).state().multiverse().even());
  CHECK(plays("[Board \"Even\"]" + even + "\n1. (-0T1)Kb1 (+0T1)Kb2"));
  CHECK(fails_with("[Board \"Even\"]" + odd, RecordFault::Unreadable, "says the game is even"));
  CHECK(fails_with("[Board \"Custom - Odd\"]" + even, RecordFault::Unreadable,
                   "says the game is odd"));
  // Without board strings there is nothing to set up.
  CHECK(fails_with("[Board \"Odd\"]", RecordFault::Unreadable, "has none"));
  CHECK(fails_with("[Board \"Custom - Even\"]", RecordFault::Unreadable, "has none"));
  // A name that merely starts with the same letters is a variant's name.
  CHECK(fails_with("[Board \"Customs\"]" + odd, RecordFault::Unsupported, "\"Customs\""));
}

TEST(pgn_setup_size_and_mode_headers) {
  CHECK(plays("[Board \"Standard\"][Size \"8x8\"][Mode \"5D\"]\n1. e4"));
  CHECK(plays("[Mode \"5d\"]\n1. e4"));
  CHECK(plays("[Mode \" 5D \"][Size \" 8X8\"]\n1. e4"));
  CHECK(fails_with("[Size \"5x5\"]\n1. e4", RecordFault::Unreadable,
                   "the game when no Board header is given"));
  CHECK(fails_with("[Board \"Small\"][Size \"8x8\"]", RecordFault::Unreadable, "5x5"));
  CHECK(fails_with("[Mode \"4D\"]", RecordFault::Unsupported, "Mode"));
  // A named variant's Size must be its own, whatever the numbers: such a record
  // contradicts itself (Unreadable) - the engine plays the variant it names.
  CHECK(fails_with("[Size \"12x8\"]", RecordFault::Unreadable, "is played on 8x8 boards"));
  CHECK(fails_with("[Size \"9x9\"]\n1. e4", RecordFault::Unreadable, "is played on 8x8 boards"));
  CHECK(fails_with("[Board \"Standard\"][Size \"10x10\"]", RecordFault::Unreadable,
                   "says 10x10, but \"Standard\" is played on 8x8"));
  CHECK(fails_with("[Board \"Standard\"][Size \"5x5\"]", RecordFault::Unreadable,
                   "says 5x5, but \"Standard\" is played on 8x8"));
  // A custom position over 8x8 is Unsupported - by its Size header, or by its
  // board strings when there is none (both the same position).
  CHECK(fails_with("[Board \"custom\"][Size \"9x9\"][4k/5/5/5/K1R2:0:1:w]",
                   RecordFault::Unsupported, "9x9 (the Size header)"));
  CHECK(fails_with("[Board \"custom\"][Size \"9x9\"][9/9/9/9/9/9/9/9/K7k:0:1:w]",
                   RecordFault::Unsupported, "at most 8x8"));
  CHECK(fails_with("[Board \"custom\"][9/9/9/9/9/9/9/9/K7k:0:1:w]", RecordFault::Unsupported,
                   "board string #1 describes boards of 9x9: the engine's boards are at most 8x8"));
  CHECK(fails_with("[Board \"custom\"][8k/9/9/9/9/9/9/9/K8:0:1:w]", RecordFault::Unsupported,
                   "boards of 9x9"));
  CHECK(fails_with("[Board \"custom\"][k/1/1/1/1/1/1/1/K:0:1:w]", RecordFault::Unsupported,
                   "boards of 1x9"));
  CHECK(fails_with("[4k/5/5/5/K1R2:0:1:w][10/5/5/5/K4:+1:1:b]", RecordFault::Unsupported,
                   "board string #2 describes boards of 10x5"));
  CHECK(fails_with("[Board \"custom\"][Size \"8x8\"][+K*8/9/9/9/9/9/9/9/k8:0:1:w]",
                   RecordFault::Unsupported, "boards of 9x9"));  // '+' and '*' are no squares
  // ... whatever else is wrong with the record.
  CHECK(fails_with("[Board \"custom\"][Size \"junk\"][Promotions \"X\"][Event \"x\" y]"
                   "[9/9/9/9/9/9/9/9/K7k:0:1:w]",
                   RecordFault::Unsupported, "9x9"));
  // A custom position within 8x8 is then read in full.
  CHECK(fails_with("[Board \"custom\"][7k/8/8/8/8/8/8/K6:0:1:w]", RecordFault::Unreadable,
                   "rows differ in width"));
  for (const std::string_view size : {"8by8", "0x5", "x8", "8x", "08x8", "8x8x8", ""}) {
    CHECK(fails_with("[Size \"" + std::string(size) + "\"]", RecordFault::Unreadable,
                     "is not \"WxH\""));
  }
}

TEST(pgn_setup_unsupported_variants) {
  // Only the variants of t5d/variants.hpp are set up; anything else is
  // refused before the first position, never guessed.
  CHECK(fails_with("[Board \"Simple - No Queens\"]\n1. (0T1)Nd3", RecordFault::Unsupported,
                   "Simple - No Queens"));
  CHECK(fails_with("[Variant \"Standard - Princess\"]", RecordFault::Unsupported,
                   "the Variant header"));
  // Decided before anything else: a game the engine does not play is
  // unsupported whatever else is wrong with the record - the move text, the
  // other headers, and faults in the headers themselves (kept by the reader).
  CHECK(fails_with("[Board \"Simple - No Queens\"][Size \"junk\"][Promotions \"X\"]\n1b. e5",
                   RecordFault::Unsupported, "Simple - No Queens"));
  CHECK(fails_with("[Mode \"4D\"][Board \"Simple - No Queens\"]", RecordFault::Unsupported,
                   "Mode"));
  for (const std::string_view text : {
           "[Board \"Simple - No Queens\"]\n[Event \"x\" junk]\n1. Nd3",
           "[Event \"x\" junk]\n[Board \"Simple - No Queens\"]\n1. Nd3",
           "[Board \"Simple - No Queens\"][Size \"7x7\"][Size \"8x8\"]\n1. Nd3",
           "[Board \"Simple - No Queens\"][Size \"7x7\"][Size \"7X7\"] 1. (0T1)Nd3",
           "[Date \"2020.10.07]\n[Board \"Simple - No Queens\"]\n1. Nd3",
           "[White Player \"x\"][Board \"Simple - No Queens\"]\n1. Nd3",
           "[Board \"Simple - No Queens\"\n1. Nd3",  // its ']' missing: the value is read
           "[Board \"Simple - No Queens\"][4k/5/5/5/K1R2:0:1:w]\n1. Nd3",
           "[Board \"Simple - No Queens\"]{ unterminated",
       }) {
    CHECK(fails_with(text, RecordFault::Unsupported, "\"Simple - No Queens\" is not one"));
  }
  CHECK(fails_with("[Event \"x\" junk][Mode \"4D\"]", RecordFault::Unsupported, "Mode"));
  // For a game the engine plays, the same faults are Unreadable.
  CHECK(fails_with("[Board \"Standard\"]\n[Event \"x\" junk]\n1. e4", RecordFault::Unreadable,
                   "header \"Event\": expected ']'"));
  CHECK(fails_with("[Board \"Standard\"][Size \"8x8\"][Size \"7x7\"]", RecordFault::Unreadable,
                   "two values for the header \"Size\""));
  CHECK(fails_with("[Board \"Standard\"\n1. e4", RecordFault::Unreadable, "expected ']'"));
  CHECK(plays("[Board \"Standard\"][Size \"8x8\"][Size \"8X8\"][board \"standard\"]\n1. e4"));
}

TEST(pgn_setup_promotions_header) {
  // S8: the header (a client extension) sets the promotion set; commas or
  // spaces (spec issue 2f), any case, RQ and PR.
  const PgnReplay a = replay("[Promotions \"Q,R,B,N\"]");
  const PgnReplay b = replay("[Promotions \"q r  b, n\"]");
  CHECK(a.state().rules().promotions == b.state().rules().promotions);
  CHECK(a.state().rules().promotions.contains(PieceKind::Knight));
  CHECK_NE(a.state().multiverse().ruleset(), 0u);  // not the base game's set
  CHECK(replay("[Promotions \"RQ,PR\"]").state().rules().promotions ==
        PromotionSet::of({PieceKind::RoyalQueen, PieceKind::Princess}));
  // The base game's own set reads as the standard ruleset.
  CHECK_EQ(replay("[Promotions \"Q\"]").state().multiverse().ruleset(), 0u);
  CHECK(fails_with("[Promotions \"Q,X\"]", RecordFault::Unreadable, "not a piece letter"));
  CHECK(fails_with("[Promotions \" , \"]", RecordFault::Unreadable, "names no piece"));
  // Piece letters the engine reads, naming a promotion the rules refuse (to a
  // pawn or a brawn): a ruleset the engine does not implement - Unsupported,
  // as the gate's contract defines it - and so whatever else is wrong.
  CHECK(fails_with("[Promotions \"Q,P\"]", RecordFault::Unsupported, "promote"));
  CHECK(fails_with("[Promotions \"W\"][Size \"junk\"]", RecordFault::Unsupported, "promote"));
}

TEST(pgn_setup_named_variants_start_with_white) {
  CHECK(fails_with("[Board \"Standard\"]\n1b. e5", RecordFault::Unreadable,
                   "White moves first"));
  CHECK(plays("[Board \"Standard - Reversed Royalty\"]\n1. e3 / e6\n2. Yh5 / Ce7"));
  const PgnReplay rr = replay("[Board \"Standard - Reversed Royalty\"]\n1. e3 / e6\n2. Yh5 / Ce7");
  CHECK(rr.state().rules().family == RuleFamily::ReversedRoyalty);
  CHECK_NE(rr.state().multiverse().ruleset(), 0u);
}

// ---- matching moves --------------------------------------------------------------------------

TEST(pgn_resolve_disambiguation_and_ambiguity) {
  // Two white knights can reach c3: "Nc3" is ambiguous, and the reader never
  // guesses ("Omission": a pinned piece may be left out, and pins are the
  // pending check ruling's).
  const std::string set_up = "[Board \"custom\"][k4/5/5/5/KN1N1:0:1:w]\n";
  CHECK(fails_with(set_up + "1. Nc3", RecordFault::Unreadable, "ambiguous"));
  CHECK(fails_with(set_up + "1. N1c3", RecordFault::Unreadable, "ambiguous"));
  CHECK(plays(set_up + "1. Nbc3"));
  CHECK(plays(set_up + "1. Ndc3"));
  CHECK(plays(set_up + "1. Nb1c3"));
  CHECK(plays(set_up + "1. (0T1)Nd1c3"));
  PgnReplay r = replay(set_up + "1. Nbc3");
  r.play_all();
  CHECK(at(head(r, 0), "c3") == Piece(PieceKind::Knight, Colour::White));
  CHECK(at(head(r, 0), "d1") == Piece(PieceKind::Knight, Colour::White));
  CHECK(at(head(r, 0), "b1").empty());
}

TEST(pgn_resolve_the_board_of_a_move) {
  // "Moves": the board's coordinates may be left out while there is one
  // timeline, and must be written otherwise.
  CHECK(plays("1. e4 / e5 2. (0T2)Nf3"));
  CHECK(fails_with("[Board \"Standard - Two Timelines\"]\n1. e3", RecordFault::Unreadable,
                   "must then be preceded by its board's coordinates"));
  CHECK(fails_with("1. e4 / e5 2. (0T1)Nf3", RecordFault::IllegalMove,
                   "is not a playable board of White's"));
  CHECK(fails_with("1. e4 / e5 2. (1T2)Nf3", RecordFault::IllegalMove, "no timeline 1"));
  CHECK(fails_with("1. e4 / e5 2. (-0T2)Nf3", RecordFault::Unreadable,
                   "exists only in a game that starts with an even number"));
  CHECK(fails_with("[Board \"Standard - Two Timelines\"]\n1. (0T1)e3", RecordFault::Unreadable,
                   "does not exist in a game that starts with an even number"));
  // T3: a board takes one move - the second is refused.
  CHECK(fails_with("[Board \"Standard - Two Timelines\"]\n1. (-0T1)e3 (-0T1)d3",
                   RecordFault::IllegalMove, "is not a playable board of White's"));
  // The long form names the same boards.
  CHECK(plays("[Board \"Standard - Two Timelines\"]\n"
              "1. (L-0 T1)e3 (L+0 T1)e3 / (-0T1)e6 (+0T1)e6"));
}

TEST(pgn_resolve_omitted_boards_option) {
  // infer_omitted_boards: a move without its board is matched on every
  // playable board of the mover's, and must fit only one.
  const std::string two = "[Board \"custom\"][4k/5/5/1P3/K4:-0:1:w][4k/5/5/5/K4:+0:1:w]\n";
  PgnOptions infer;
  infer.infer_omitted_boards = true;
  CHECK(fails_with(two + "1. b3 Kb1", RecordFault::Unreadable, "must then be preceded"));
  CHECK(plays(two + "1. b3 Kb1", infer));  // b3 on -0; then only +0 is left to White
  CHECK(fails_with(two + "1. Kb1 b3", RecordFault::Unreadable, "ambiguous", infer));
}

TEST(pgn_resolve_no_matching_move_is_illegal) {
  CHECK(fails_with("1. Nc4", RecordFault::IllegalMove, "no knight of White's can move to c4"));
  CHECK(fails_with("1. e5", RecordFault::IllegalMove, "no pawn of White's"));
  CHECK(fails_with("1. Ke2", RecordFault::IllegalMove, "no king"));
  CHECK(fails_with("1. Bb5", RecordFault::IllegalMove, "no bishop"));
  // A file beyond the board: no move goes there.
  CHECK(fails_with("[Board \"Small\"]\n1. Nh3", RecordFault::IllegalMove, "h3"));
  // The action's piece of the other colour.
  CHECK(fails_with("1. e4 / Nf3", RecordFault::IllegalMove, "no knight of Black's"));
}

TEST(pgn_resolve_capture_marks) {
  // A missing 'x' is read (default); a written 'x' must capture.
  CHECK(plays("1. e4 / d5 2. exd5"));
  CHECK(plays("1. e4 / d5 2. ed5"));
  CHECK(plays("1. e4 / d5 2. Pexd5"));
  CHECK(fails_with("1. Nxf3", RecordFault::Unreadable, "captures nothing"));
  PgnOptions strict;
  strict.require_capture_marks = true;
  CHECK(plays("1. e4 / d5 2. exd5", strict));
  CHECK(fails_with("1. e4 / d5 2. ed5", RecordFault::Unreadable, "without 'x'", strict));
}

TEST(pgn_resolve_jump_markers_must_agree_with_branching_j3) {
  // J3/J4: onto another timeline's playable board, no branch ('>'); onto a
  // history board, a branch ('>>').
  const std::string two = "[Board \"Standard - Two Timelines\"]\n"
                          "1. (-0T1)Nf3 (+0T1)Nf3 / (-0T1)Nf6 (+0T1)Nf6\n";
  CHECK(plays(two + "2. (-0T2)Nf3>(+0T2)d3"));
  CHECK(fails_with(two + "2. (-0T2)Nf3>>(+0T2)d3", RecordFault::Unreadable,
                   "does not branch"));
  CHECK(plays(two + "2. (-0T2)Nf3>>(-0T1)f5"));
  CHECK(fails_with(two + "2. (-0T2)Nf3>(-0T1)f5", RecordFault::Unreadable, "branches"));
  CHECK(fails_with(two + "2. (-0T2)Nf3>(-0T2)d4", RecordFault::Unreadable,
                   "its source board"));
  CHECK(fails_with("1. Nb1>(0T1)c3", RecordFault::Unreadable, "its source board"));
  // The non-branching jump appends a board to both timelines (J4).
  PgnReplay r = replay(two + "2. (-0T2)Nf3>(+0T2)d3");
  r.play_all();
  CHECK_EQ(r.state().multiverse().board_count(), std::size_t{2 + 4 + 2});
  CHECK(at(head(r, 0), "d3") == Piece(PieceKind::Knight, Colour::White));
  CHECK(at(head(r, -1), "f3").empty());
}

TEST(pgn_resolve_promotions_s8) {
  const std::string set_up = "[Board \"custom\"][k4/3P1/5/5/K4:0:1:w]\n";
  // The base game promotes to a queen; "=Q" may be written or not.
  PgnReplay r = replay(set_up + "1. d5");
  r.play_all();
  CHECK(at(head(r, 0), "d5") == Piece(PieceKind::Queen, Colour::White));
  CHECK(plays(set_up + "1. d5=Q+"));
  CHECK(fails_with(set_up + "1. d5=N", RecordFault::IllegalMove, "promoting to a knight"));
  // With a Promotions header offering several, the piece must be written.
  const std::string several = "[Promotions \"Q,N\"]" + set_up;
  CHECK(fails_with(several + "1. d5", RecordFault::Unreadable, "is not written (=X)"));
  PgnReplay n = replay(several + "1. d5=N");
  n.play_all();
  CHECK(at(head(n, 0), "d5") == Piece(PieceKind::Knight, Colour::White));
}

TEST(pgn_resolve_castling_shortcuts_s6_q26) {
  // Kings on e1 and e8: O-O is Ke1g1 / Ke8g8, O-O-O Ke1c1 / Ke8c8 ("Castling").
  const std::string e_file = "[Board \"custom\"][r*3k*2r*/8/8/8/8/8/8/R*3K*2R*:0:1:w]\n";
  PgnReplay r = replay(e_file + "1. O-O / O-O-O");
  r.play_all();
  CHECK(at(head(r, 0), "g1") == Piece(PieceKind::King, Colour::White));
  CHECK(at(head(r, 0), "f1") == Piece(PieceKind::Rook, Colour::White));
  CHECK(at(head(r, 0), "c8") == Piece(PieceKind::King, Colour::Black));
  CHECK(at(head(r, 0), "d8") == Piece(PieceKind::Rook, Colour::Black));
  // The king's move names the same castling ("K<from><to>", "Kg1").
  CHECK(plays(e_file + "1. Ke1g1 / Kc8"));
  CHECK(plays(e_file + "1. (0T1)O-O-O+ / (0T1)O-O"));
  // Kings elsewhere (d1, d8): the shortcut is not defined (S6) - unless the
  // ruleset of readings takes Q26's alternative, castling towards file h.
  const std::string d_file = "[Board \"custom\"][r*2k*3r*/8/8/8/8/8/8/R*2K*3R*:0:1:w]\n";
  CHECK(fails_with(d_file + "1. O-O", RecordFault::Unreadable, "Q26"));
  PgnOptions any;
  any.castling_shortcuts = CastlingShortcuts::AnyKing;
  PgnReplay q = replay(d_file + "1. O-O / O-O-O", any);
  q.play_all();
  CHECK(at(head(q, 0), "f1") == Piece(PieceKind::King, Colour::White));
  CHECK(at(head(q, 0), "e1") == Piece(PieceKind::Rook, Colour::White));
  CHECK(at(head(q, 0), "b8") == Piece(PieceKind::King, Colour::Black));
  CHECK(at(head(q, 0), "c8") == Piece(PieceKind::Rook, Colour::Black));
  CHECK(plays(d_file + "1. Kd1f1 / Kd8b8"));
  // No castling move: illegal.
  CHECK(fails_with("1. O-O", RecordFault::IllegalMove, "no castling O-O of White's"));
  // The shortcuts name the king's move from e1 / e8 only: a king that has
  // walked off its square is not castling when it steps to g1 or c8.
  CHECK(fails_with(e_file + "1. Kf1 / Kf8 2. O-O", RecordFault::IllegalMove,
                   "no castling O-O of White's"));
  CHECK(fails_with(e_file + "1. Kf1 / Kd8 2. Kg1 / O-O-O", RecordFault::IllegalMove,
                   "no castling O-O-O of Black's"));
  CHECK(plays(e_file + "1. Kf1 / Kd8 2. Kg1 / Kc8"));  // the same steps, written as steps
  // Reversed Royalty has no royal king: the shortcut is undefined there.
  CHECK(fails_with("[Board \"Standard - Reversed Royalty\"]\n1. O-O", RecordFault::Unreadable,
                   "S6"));
}

TEST(pgn_resolve_castling_shortcuts_with_several_kings_s6) {
  // "On starting positions with more than one king for either player ... the
  // notation K<file_from><rank_from><file_to><rank_to> must be used." Read per
  // board by default (EFileOnly: one king a board, on e1 / e8, and the board
  // named) - so Two Timelines and Turn Zero use the shortcuts; read over the
  // whole starting position by EFileOneKing, the specification's letter -
  // so they do not, and the king's move is written instead.
  const std::string two =
      "[Board \"Standard - Two Timelines\"]\n"
      "1. (-0T1)Nf3 (+0T1)Nf3 / (-0T1)Nf6 (+0T1)Nf6 2. (-0T2)g3 (+0T2)g3 / (-0T2)g6 (+0T2)g6\n"
      "3. (-0T3)Bg2 (+0T3)Bg2 / (-0T3)Bg7 (+0T3)Bg7\n";
  const std::string shortcuts = "4. (-0T4)O-O (+0T4)O-O / (-0T4)O-O (+0T4)O-O";
  const std::string king_moves = "4. (-0T4)Ke1g1 (+0T4)Ke1g1 / (-0T4)Ke8g8 (+0T4)Kg8";
  PgnOptions letter;
  letter.castling_shortcuts = CastlingShortcuts::EFileOneKing;
  CHECK(plays(two + shortcuts));
  CHECK(plays(two + king_moves));
  CHECK(plays(two + king_moves, letter));
  std::string message;
  std::size_t played = 0;
  CHECK(outcome(two + shortcuts, letter, &message, &played) == RecordFault::Unreadable);
  CHECK_EQ(played, std::size_t{6});
  CHECK(message.find("one king a side") != std::string::npos);
  // Both readings give the same castling where they allow the shortcut.
  PgnReplay a = replay(two + shortcuts);
  PgnReplay b(read_pgn(two + king_moves, letter), letter);
  a.play_all();
  b.play_all();
  CHECK(a.state() == b.state());
  CHECK(at(head(a, -1), "g1") == Piece(PieceKind::King, Colour::White));
  CHECK(at(head(a, 0), "f8") == Piece(PieceKind::Rook, Colour::Black));
  // Turn Zero: two kings a side (its T0 and T1 boards).
  const std::string zero = "[Board \"Standard - Turn Zero\"]\n"
                           "1. Nf3 / Nf6 2. g3 / g6 3. Bg2 / Bg7 4. O-O / O-O";
  CHECK(plays(zero));
  CHECK(fails_with(zero, RecordFault::Unreadable, "one king a side", letter));
  // One king a side, on e1 and e8: the shortcuts under every e-file reading.
  const std::string standard = "1. Nf3 / Nf6 2. g3 / g6 3. Bg2 / Bg7 4. O-O / O-O";
  CHECK(plays(standard));
  CHECK(plays(standard, letter));
  // Kings elsewhere: not under EFileOneKing either.
  const std::string d_file = "[Board \"custom\"][r*2k*3r*/8/8/8/8/8/8/R*2K*3R*:0:1:w]\n";
  CHECK(fails_with(d_file + "1. O-O", RecordFault::Unreadable, "one king a side", letter));
  CHECK(fails_with("[Board \"Standard - Half Reflected\"]\n1. O-O", RecordFault::Unreadable,
                   "Q26"));
}

TEST(pgn_resolve_en_passant_s7) {
  PgnReplay r = replay("1. e4 / a6 2. e5 / d5 3. exd6");
  r.play_all();
  CHECK(at(head(r, 0), "d6") == Piece(PieceKind::Pawn, Colour::White));
  CHECK(at(head(r, 0), "d5").empty());
}

TEST(pgn_turn_zero_branch_onto_the_t0_board_q8) {
  // L5 (Q8 default): the T0 board is a history board; a black knight jumps
  // two turns back onto it and branches.
  PgnReplay r = replay("[Board \"Standard - Turn Zero\"]\n"
                       "1. e3 / Nf6 2. Bc4 / (0T2)Nf6>>(0T0)g6 3. (-1T1)e4");
  r.play_all();
  REQUIRE_EQ(r.state().multiverse().timeline_count(), 2);  // timeline(-1) below must exist
  REQUIRE(r.state().multiverse().timeline(-1) != nullptr);
  CHECK_EQ(r.state().multiverse().timeline(-1)->first_half_turn(), half_turn(1, Colour::White));
  CHECK_EQ(pieces_at(r, -1, 1, Colour::White),
           "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/6n1/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*");
}

// ---- the turn rules ---------------------------------------------------------------------

TEST(pgn_actions_must_shift_the_present_t5) {
  // T5: an action that leaves a mandatory board unplayed cannot be submitted.
  CHECK(fails_with("[Board \"Standard - Two Timelines\"]\n1. (-0T1)e3 / (-0T1)e6",
                   RecordFault::CannotSubmit, "does not move on (+0T1, white to move)"));
  // Q4 (default): no move once the present has shifted.
  CHECK(fails_with("1. e3 e4", RecordFault::IllegalMove, "no move may be made after that (Q4)"));
  CHECK(fails_with("[Board \"Standard - Two Timelines\"]\n1. (-0T1)e3 (+0T1)e3 (+0T1)d3",
                   RecordFault::IllegalMove, "Q4"));
}

TEST(pgn_failed_actions_leave_the_state_as_it_was) {
  PgnReplay r = replay("[Board \"Standard - Two Timelines\"]\n"
                       "1. (-0T1)e3 (+0T1)e3 / (-0T1)e6 (+0T1)Nc4\n");
  const GameState before = r.state();
  r.play_next();
  const GameState after_white = r.state();
  CHECK_EQ(r.actions_played(), std::size_t{1});
  CHECK_THROWS_AS(r.play_next(), RecordError);  // Black's Nc4: illegal
  CHECK_EQ(r.actions_played(), std::size_t{1});
  CHECK(r.state() == after_white);
  CHECK(!(r.state() == before));
  CHECK(r.state().to_move() == Colour::Black);
}

TEST(pgn_play_all_reports_the_unread_tail) {
  PgnReplay r = replay("1. e4 / e5 2. Nf3 / ???");
  std::string message;
  CHECK(play(r, &message) == RecordFault::Unreadable);
  CHECK_EQ(r.actions_played(), std::size_t{3});
  CHECK(message.find("unexpected '?'") != std::string::npos);
}

// ---- annotations (T15) ------------------------------------------------------------------

TEST(pgn_annotations_are_checked_against_the_state_t15) {
  // White's knight jumps a turn back: an active branch (T10), +1, whose first
  // board, (+1T1, black), lies before the present (T2, white) - the present
  // moves there (T8).
  const std::string start = "1. Nf3 / Nf6\n2. (0T2)Nf3>>(0T1)f5";
  CHECK(plays(start + "~"));
  CHECK(plays(start + "~ (>L1)"));
  CHECK(plays(start + "~ (>L+1) (~T1)"));
  CHECK(plays(start));  // tokens left out are never an error
  CHECK(fails_with(start + "~ (>L2)", RecordFault::Unreadable, "it created +1"));
  CHECK(fails_with(start + " (>L-1)", RecordFault::Unreadable, "it created +1"));
  CHECK(fails_with(start + " (~T2)", RecordFault::Unreadable, "it is at turn 1"));
  CHECK(fails_with("1. e4~", RecordFault::Unreadable, "does not branch"));
  CHECK(fails_with("1. e4 (>L1)", RecordFault::Unreadable, "creates none"));
  // White's second branch while Black has made none is inactive (T10): it
  // never moves the present (T8), so '~' is wrong.
  const std::string second = start + "~ / (1T1)e6\n3. (1T2)Nf5>>x(0T2)f7";
  CHECK(plays(second));
  CHECK(fails_with(second + "~", RecordFault::Unreadable, "inactive"));
  PgnReplay r = replay(second);
  r.play_all();
  CHECK(!r.state().active(2));
  // With verify_annotations off, the tokens are not checked.
  PgnOptions off;
  off.verify_annotations = false;
  CHECK(plays(start + "~ (>L5) (~T9)", off));
  CHECK(plays("1. e4~", off));
}

TEST(pgn_annotation_present_moved_needs_a_branch_before_the_present_t15) {
  // An ACTIVE branch whose first board lies after the present does not move
  // the present (T4, T8): '~' is wrong there. Timeline 0 runs from (0T1) to
  // (0T5, white); Black's -1, active (T10: White has created none, so Black's
  // first is active), has one board at (-1T2, white) - the present (T4).
  // White's knight on (0T5) leaps one turn back and two ranks up onto (0T4),
  // a history board: a branch (J3), White's first timeline, +1 - active (T10)
  // - whose first board is (0T4)'s half-step successor, (+1T4, black) (J5),
  // after the present. Then White plays the mandatory (-1T2) board (T5).
  std::string set_up = "[Board \"custom\"][Size \"5x5\"]";
  for (int turn = 1; turn <= 5; ++turn) {
    set_up += "[4k/5/5/5/KN3:0:" + std::to_string(turn) + ":w]";
    if (turn < 5) set_up += "[4k/5/5/5/KN3:0:" + std::to_string(turn) + ":b]";
  }
  set_up += "[4k/5/5/5/KN3:-1:2:w]\n";
  CHECK(fails_with(set_up + "1. (0T5)Nb1>>(0T4)b3~ (-1T2)Kb2", RecordFault::Unreadable,
                   "its first board (+1T4, black to move) lies after the present (T4, T8)"));
  CHECK(plays(set_up + "1. (0T5)Nb1>>(0T4)b3 (-1T2)Kb2"));
  CHECK(plays(set_up + "1. (0T5)Nb1>>(0T4)b3 (>L1) (~T2) (-1T2)Kb2"));
  PgnReplay r = replay(set_up + "1. (0T5)Nb1>>(0T4)b3 (-1T2)Kb2");
  r.play_all();
  REQUIRE_EQ(r.state().multiverse().max_timeline(), 1);
  CHECK(r.state().active(1));
  CHECK_EQ(r.state().multiverse().timeline(1)->first_half_turn(), half_turn(4, Colour::Black));
  CHECK_EQ(r.state().present(), half_turn(2, Colour::Black));
}

TEST(pgn_annotation_created_timelines_in_the_order_made_j9) {
  // Two branches in one action take their indices in the order they are made
  // (J9, L7), and (>Lx) is checked against each in turn. Three timelines, -1,
  // 0, +1, each at (T1, white) with a white knight on b1. White moves on 0
  // first, so (0T1) is a history board (J6); the knights of -1 and +1 then
  // leap onto it (one timeline over, two ranks up: 4.5) - two branches,
  // White's second and third timelines, +2 then +3.
  const std::string set_up = "[Board \"custom\"][Size \"5x5\"]"
                             "[4k/5/5/5/KN3:-1:1:w][4k/5/5/5/KN3:0:1:w][4k/5/5/5/KN3:1:1:w]\n";
  const std::string first = "1. (0T1)Kb2 (-1T1)Nb1>>(0T1)b3";
  const std::string second = " (1T1)Nb1>>(0T1)b3";
  CHECK(plays(set_up + first + second));
  CHECK(plays(set_up + first + " (>L2)" + second + " (>L3)"));
  CHECK(plays(set_up + first + " (>L+2)" + second + "(>L+3)"));
  CHECK(fails_with(set_up + first + " (>L3)" + second + " (>L2)", RecordFault::Unreadable,
                   "move 2 \"(-1T1)Nb1>>(0T1)b3 (>L3)\""));
  CHECK(fails_with(set_up + first + second + " (>L2)", RecordFault::Unreadable,
                   "it created +3 (L7, J9)"));
  PgnReplay r = replay(set_up + first + " (>L2)" + second + " (>L3)");
  r.play_all();
  REQUIRE_EQ(r.state().multiverse().max_timeline(), 3);
  // T10: White's second timeline is active (Black has created one), its third
  // is not (Black has not created two).
  CHECK(r.state().active(2));
  CHECK(!r.state().active(3));
}

TEST(pgn_check_marks_are_never_verdicts) {
  // '+', '#', '*' are the writer's claims, read and never checked: a quiet
  // first move marked as mate replays like any other.
  CHECK(plays("1. e4# / e5+ 2. Nf3*"));
}
