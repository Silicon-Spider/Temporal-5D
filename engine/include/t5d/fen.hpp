// t5d/fen.hpp - 5DFEN board strings, and the standard starting positions.
//
// Written from the pinned notation specification (adri326/5dchess-notation:
// its README, sections "5DFEN and custom variants", "Hashing" and "5DFEN
// Strictness", and its grammar fen.ebnf) - never from another implementation's
// parser (decision H28).
//
// A board string is "[pieces:timeline:turn:colour]":
//
//  * pieces: the rows from the top (highest rank) down, separated by '/'. Each
//    row is pieces and blanks: a piece is a letter, optionally preceded by '+'
//    and optionally followed by '*' (not yet moved); a blank is a run of empty
//    squares written as a base-10 number. White pieces are upper case, black
//    lower case - as the README's prose and every example say. (The grammar's
//    piece_white and piece_black productions have the cases the other way
//    round; that is a defect of the grammar, and this parser follows the prose.)
//  * timeline: a label, t5d/timeline.hpp ("0", "-0", "+0", "+1", "-1", ...).
//  * turn: the turn as displayed, 0 or more.
//  * colour: the player to move on the board, 'w' or 'b'.
//
// Reading is strict where the text is ambiguous or unsupported and lenient
// only where the specification allows a variant - with one exception, a
// project decision rather than the specification's reading:
//
//  * accepted: multi-digit blank counts (read as one number: "10" is ten);
//    '*' after a piece that does not track it, such as a queen (ignored, as the
//    grammar allows); an unsigned timeline label such as "1" (the README's
//    note on even-numbered starting boards allows it).
//  * accepted by decision: '+' before a piece letter, read as the same piece
//    ("+K" is "K"). The specification says the '+' prefix exists "to extend
//    the number of pieces that can be encoded", which makes "+K" a different
//    piece - one it never defines. Reading it as "K" is the project's ruling,
//    pending one upstream (the engine README lists it); the strict reading
//    would reject it as an unknown piece, and is a change to lex_pieces
//    alone. The canonical writer never writes '+', so "+K" and "K" fingerprint
//    alike.
//  * rejected: missing brackets, any whitespace, a field count other than
//    four, a row count or row width that does not match the board size, an
//    unknown letter, '*' anywhere but after a piece letter, a blank of 0,
//    leading zeros, a label that does not exist in the game's kind (even/odd),
//    a colour other than 'w' or 'b'.
//
// Writing is canonical (the notation specification's "5DFEN Strictness"): no
// '+', '*' exactly on unmoved pawns, rooks, kings and brawns, maximal blank
// runs, canonical labels, no whitespace. Reading a canonical string and
// writing it back gives the same string.

#ifndef T5D_FEN_HPP
#define T5D_FEN_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/timeline.hpp"
#include "t5d/types.hpp"

namespace t5d {

/// One parsed board string.
struct FenBoard {
  BoardRef board;
  TimelineLabel timeline;  // as written; its position depends on the game's kind
  int turn = 0;            // as displayed
  Colour to_move = Colour::White;

  int half_turn() const noexcept { return t5d::half_turn(turn, to_move); }
};

/// Parses a board string for boards of `size`. Throws NotationError if the
/// string is malformed and std::invalid_argument if the size is invalid.
FenBoard parse_fen_board(std::string_view text, BoardSize size);

/// The board size a board string's piece field implies: its row count and row
/// width. Throws NotationError if the rows differ in width or the string is
/// malformed.
BoardSize infer_fen_board_size(std::string_view text);

/// The canonical piece field of a board ("r*nbqk*bnr*/.../R*NBQK*BNR*").
std::string fen_pieces(const Board& board);

/// The canonical board string of `board` at timeline position p and half-turn h
/// in an odd (even = false) or even game.
std::string fen_board_string(const Board& board, int p, int h, bool even);
/// The same, appended to `out`.
void append_fen_board_string(std::string& out, const Board& board, int p, int h, bool even);

/// Whether a game starts with an even or an odd number of timelines - or, for
/// parsing, "decide from the labels": "-0"/"+0" mean even, "0" means odd.
enum class Parity : std::uint8_t { Odd, Even, Infer };

/// Builds a multiverse from board strings (the initial boards of a 5dpgn game
/// with Board "custom", or any set of boards). Throws NotationError if a string
/// is malformed or if the boards do not form a multiverse:
///
///  * the labels must agree with `parity` (and with each other, under Infer);
///  * no two boards in one place;
///  * each timeline's boards one per half-turn, without gaps;
///  * the timelines a contiguous range of positions.
///
/// Creators, which 5DFEN does not record, follow the notation's numbering
/// ("Coordinates": white's new timelines take the next positive integer,
/// black's the next negative one): the central timeline(s) - "0", or "-0" and
/// "+0" - are initial; a positive label was created by white, a negative one by
/// black, and the label's magnitude is the creation order. No origins.
///
/// The player to move, which 5DFEN does not record either, is `to_move` when
/// the caller gives it (the 5dpgn movetext says whose turn it is), taken as
/// given. Without it, it is inferred only where no rule of stage E2 can change
/// the answer. The player to move is the colour to move on the present - the
/// earliest head board of the ACTIVE timelines - and which created timelines
/// are active is E2's rule. But initial timelines are always active, so the
/// present is one of the heads no later than the earliest initial head (any
/// head, if there is no initial timeline); when all of those have the same
/// colour to move, that colour is the player to move whatever E2 decides.
/// Otherwise this throws std::invalid_argument, naming two heads that could
/// be the present, rather than guess. That covers every starting position and
/// every position whose only past-branching timelines left the same player to
/// move; a position with, say, an inactive timeline whose head is earlier than
/// the present and of the other colour needs the caller's answer until E2.
/// (The board strings themselves are fine; the fingerprint never depends on
/// the player to move - only the Zobrist key does.)
///
/// `all_initial`: every timeline the strings hold is an initial timeline - a
/// named variant's set-up, such as Timeline Battleground's -1, 0 and +1 - with
/// no creator and creation order 0 (ruling R10, register H46: initial
/// timelines count as created by neither player). Otherwise only the central
/// timeline(s) are initial and the rest are read as created, as above.
Multiverse multiverse_from_fen(std::span<const std::string_view> boards, BoardSize size,
                               Parity parity, std::uint32_t ruleset = 0,
                               std::optional<Colour> to_move = std::nullopt,
                               bool all_initial = false);
Multiverse multiverse_from_fen(const std::vector<std::string>& boards, BoardSize size,
                               Parity parity, std::uint32_t ruleset = 0,
                               std::optional<Colour> to_move = std::nullopt,
                               bool all_initial = false);

/// The Standard board: the chess array, with the unmoved marker on every pawn,
/// rook and king. One shared immutable instance.
BoardRef standard_board();

/// "Standard": the standard board on timeline 0 at T1, white to move.
Multiverse standard_start();

/// "Standard - Two Timelines": the standard board on -0 and on +0 at T1, white
/// to move (an even game). Both timelines share one board object.
Multiverse standard_two_timelines_start();

}  // namespace t5d

#endif  // T5D_FEN_HPP
