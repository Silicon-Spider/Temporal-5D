// t5d/pgn.hpp - the 5dpgn reader, and the replay of a record on a game state
// (decision H28).
//
// Written from the notation specification's text alone - adri326/5dchess-
// notation, its README ("Coordinates", "Moves" with "Castling", "Jumps",
// "Promotions", "Inactive timeline reactivation" and "Complex scenarios",
// "Turns", "Headers", "Comments", "5DFEN and custom variants", "Notes") and its
// grammar fen.ebnf - never from another implementation's parser (that
// repository ships one; it is a parse-only oracle and is not read). Rule IDs
// (L*, T*, J*, S*, Q*) are the rules reference's, docs/rules/README.md.
//
// Reading happens in two stages, as the notation intends ("there should be no
// need to hold the state of the game while initially parsing the moves"):
//
//  1. read_pgn: text -> headers, 5DFEN board strings, and the actions, each a
//     list of moves AS WRITTEN (PgnMove). No state is needed.
//  2. PgnReplay: sets up the starting position the headers declare - a named
//     variant (t5d/variants.hpp) or "custom" 5DFEN boards - and plays the
//     actions one at a time. Each written move is matched against the
//     pseudo-legal moves of the board it names (t5d/movegen.hpp) and applied;
//     the action is submitted once its moves are played. Submission's
//     royal-safety test (C4) is the check ruling's, pending: it is not made
//     (GameState::submit with no hook), so a replayed action is one whose
//     every move is pseudo-legal and allowed by the turn rules and after which
//     the present has shifted (T5).
//
// What a record can fail on (RecordFault):
//
//  * Unreadable - text the reader cannot read; a move that fits two or more
//    pseudo-legal moves (the reader never guesses: the notation lets a writer
//    leave out a pinned piece when disambiguating, "Omission", and pins are the
//    pending check ruling's); a token that disagrees with the state where the
//    notation says it should agree (">"/">>", "x", "~", "(>Lx)", "(~Tx)");
//    headers that contradict each other; an action by the player who is not
//    to move.
//  * Unsupported - a game the engine does not play: a Mode other than "5D", a
//    variant the engine does not set up, a custom position whose boards are
//    wider or higher than 8 (by its Size header or by its board strings), a
//    promotion set the engine's rules do not implement. Decided from the
//    headers that declare the game before any other fault is raised, so it
//    holds whatever else is wrong with the record; only ever before the first
//    position.
//  * IllegalMove - a move no pseudo-legal move fits: a board that is not a
//    playable board of the mover's (L5, T3), no piece that can make the move
//    (sections 3-5), or any move once the present has shifted (Q4).
//  * CannotSubmit - the action ends with the present still on the mover's
//    colour (T5).
//
// The reader's choices where the specification is ambiguous or silent are
// PgnOptions (defaults first); the build log's "Spec issues" item 2 lists the
// 5DFEN and header readings shared with the gate.

#ifndef T5D_PGN_HPP
#define T5D_PGN_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "t5d/timeline.hpp"
#include "t5d/types.hpp"

namespace t5d {

// ---------------------------------------------------------------------------
// Faults
// ---------------------------------------------------------------------------

/// Why a record could not be read or replayed (see the header comment).
enum class RecordFault : std::uint8_t { Unreadable = 0, Unsupported, IllegalMove, CannotSubmit };

/// "unreadable", "unsupported", "illegal_move", "cannot_submit".
const char* fault_name(RecordFault fault) noexcept;

/// Thrown by the reader and the replay. what() says what was wrong and where
/// (line and column, the offending token as written).
class RecordError : public std::runtime_error {
 public:
  RecordError(RecordFault fault, const std::string& message)
      : std::runtime_error(message), fault_(fault) {}
  RecordFault fault() const noexcept { return fault_; }

 private:
  RecordFault fault_;
};

// ---------------------------------------------------------------------------
// Options: the reader's readings of what the specification leaves open
// ---------------------------------------------------------------------------

/// S6 / Q26: what "O-O" and "O-O-O" mean.
enum class CastlingShortcuts : std::uint8_t {
  /// Default (S6, high; the notation's "Castling"): shortcuts for Ke1g1/Ke8g8
  /// and Ke1c1/Ke8c8, usable only in a game whose kings start on e1 and e8 -
  /// every white king of the starting position on e1, every black one on e8,
  /// and both colours with one. ("More than one king for either player" is
  /// read per board: a multi-timeline start has one king a board, and the
  /// board's coordinates say which castles - Standard - Two Timelines and
  /// Turn Zero use the shortcuts; Half Reflected, Reversed Royalty and the
  /// small variants do not.) Elsewhere a record must write the king's move,
  /// K<from><to>.
  EFileOnly = 0,
  /// Q26's alternative - the notation's own 2020 "Actual game" castles with
  /// "O-O {to the right}" from d1 in Simple - No Queens: "O-O" is the mover's
  /// castling towards the higher files (the king moves two squares towards
  /// file h), "O-O-O" towards the lower files, wherever the king stands. In a
  /// game whose kings start on e1 and e8 the two readings agree.
  AnyKing = 1,
  /// The notation's letter ("Castling": "On starting positions with more than
  /// one king for either player or kings in other places than e1 and e8, the
  /// notation K<file_from><rank_from><file_to><rank_to> must be used"), "more
  /// than one king" read over the whole starting position: the shortcuts as
  /// EFileOnly, but only when each colour has exactly one king in the starting
  /// position - so not in Standard - Two Timelines or Standard - Turn Zero
  /// (two kings a side), where a record must write the king's move.
  EFileOneKing = 2,
};

struct PgnOptions {
  /// S6 / Q26.
  CastlingShortcuts castling_shortcuts = CastlingShortcuts::EFileOnly;
  /// Must a capture be written with "x"? Default false: a missing "x" is
  /// accepted - the move's squares and boards determine it, so leaving it out
  /// never makes a move ambiguous (PGN's import reading). A written "x" must
  /// always be a capture. True is the specification's letter (SAN; "Jumps":
  /// "x if a piece is being taken").
  bool require_capture_marks = false;
  /// T15: check "~", "(~Tx)" and "(>Lx)" against the state - the
  /// specification says they "should match with what can be induced from the
  /// described state of the game". A written token that disagrees makes the
  /// move unreadable; a token left out is never an error (records older than
  /// the token omit it). Q26's open case - "~" on a branch whose first board
  /// lies in the column right after the present's, where the present would
  /// arrive anyway - is accepted either way. False ignores the three tokens.
  /// ("+", "#" and "*" are never checked: they are the writer's claims about
  /// check, and check is the pending ruling's - never a check verdict.)
  bool verify_annotations = true;
  /// "Moves": a physical move "must be preceded by their board's coordinates
  /// if there is more than one timeline". Default false: a move without its
  /// board is unreadable then. True: it is matched against every playable
  /// board of the mover's (and is unreadable if that is ambiguous).
  bool infer_omitted_boards = false;
  /// "Turns": "<x>." / "<x>w." / "/" / "<x>b." with x the turn number. Default
  /// true: the numbers must count the turns - a white action numbers one more
  /// than the black action before it, a black action the same as the white
  /// action before it (the first action may have any number from 1). False
  /// ignores the numbers (the colours must alternate all the same).
  bool check_turn_numbers = true;
};

// ---------------------------------------------------------------------------
// The record as written
// ---------------------------------------------------------------------------

/// One header, [Name "Value"].
struct PgnHeader {
  std::string name;
  std::string value;  // escapes (\" and \\) resolved
  std::size_t offset = 0;
};

/// A board as the notation writes it: "(<a>T<b>)" or "(L<a> T<b>)" - the
/// timeline label as written and the turn as displayed. The colour is never
/// written ("Coordinates"): a move starts and ends on boards of the mover's
/// colour (L4, L5).
struct PgnBoard {
  TimelineLabel timeline;
  int turn = 0;

  friend bool operator==(const PgnBoard&, const PgnBoard&) noexcept = default;
};

/// One move as written, before it is matched against a state.
struct PgnMove {
  enum class Kind : std::uint8_t {
    Ordinary,          // a piece's move: physical (SAN) or a jump
    CastleKingside,    // "O-O"
    CastleQueenside,   // "O-O-O"
  };
  enum class Jump : std::uint8_t {
    None,           // a physical move: the target board is the source board
    NonBranching,   // ">"
    Branching,      // ">>"
  };

  Kind kind = Kind::Ordinary;
  /// The source board's coordinate; nullopt when left out (allowed only while
  /// there is one timeline, "Moves").
  std::optional<PgnBoard> board;
  /// The piece letter written ("RQ" and "PR" read as Y and S); nullopt when
  /// none was written - a pawn ("Moves": SAN), or castling.
  std::optional<PieceKind> piece;
  /// The source square's file and rank as far as written (-1: not written).
  /// A jump writes both ("Jumps": "The piece's physical coordinate").
  int from_file = -1;
  int from_rank = -1;
  /// "x".
  bool capture = false;
  Jump jump = Jump::None;
  /// A jump's target board.
  std::optional<PgnBoard> target_board;
  /// The target square (castling: -1).
  int to_file = -1;
  int to_rank = -1;
  /// "=X" ("Promotions").
  std::optional<PieceKind> promotion;
  /// '+', '#' or '*' as written, or 0: the writer's claim of check, mate or
  /// softmate. Never checked, never a verdict.
  char mark = 0;
  /// "~": the present moved to the new branch (T15).
  bool present_moved = false;
  /// "(>Lx)": the timeline the move created (T15).
  std::optional<TimelineLabel> created;
  /// "(~Tx)": the present's turn after the move (T15).
  std::optional<int> present_turn;
  /// The move as written (its board prefix included), for messages.
  std::string text;
  /// Where it starts in the record: byte offset, and line and column (from 1;
  /// 0 for a move not read from a record).
  std::size_t offset = 0;
  std::size_t line = 0;
  std::size_t column = 0;
};

/// One action: the moves one player made in one turn ("Turns"), in order.
struct PgnAction {
  Colour colour = Colour::White;
  /// The record's turn number ("<x>.", "<x>w.", "<x>b.", or the white action's
  /// number for one introduced by "/").
  int turn_number = 1;
  std::vector<PgnMove> moves;
  /// Where its turn marker (or first move) starts: byte offset, line, column.
  std::size_t offset = 0;
  std::size_t line = 0;
  std::size_t column = 0;
};

/// A record as written.
struct PgnRecord {
  std::vector<PgnHeader> headers;
  /// The 5DFEN board strings ("5DFEN and custom variants"), as written.
  std::vector<std::string> boards;
  /// Every action read in full, in order.
  std::vector<PgnAction> actions;
  /// The result token ("1-0", "0-1", "1/2-1/2", "*"), or empty.
  std::string result;
  /// Set when a header or board string could not be read, or two values of a
  /// header the engine reads disagree (compared as the engine reads them:
  /// "7x7" and "7X7" are one Size): what, and where - the first such fault.
  /// Reading goes on after it (past the next ']' on its line, or from the
  /// next line), so the headers that follow are read too: the set-up raises
  /// it (Unreadable) only once it has decided whether the record's game is
  /// one the engine plays - a game it does not play is Unsupported whatever
  /// else is wrong (PgnReplay). Of a header whose value was read in full but
  /// whose closing ']' is missing, the value is kept.
  std::optional<std::string> headers_unreadable;
  /// Set when the move text could not be read to its end: what could not be
  /// read, and where. `actions` then holds the actions before the one in which
  /// reading stopped; replaying the record ends there, Unreadable.
  std::optional<std::string> unreadable;

  /// The value of the header `name` (compared without regard to case: the
  /// specification writes "Board" and "Size" in its README and "variant" and
  /// "size" in its grammar), or nullptr.
  const std::string* header(std::string_view name) const noexcept;
};

/// Reads a record: headers and board strings, then the move text. Throws no
/// RecordError: a fault in the headers or board strings (or two values for one
/// header the engine reads) is recorded in PgnRecord::headers_unreadable, and
/// one in the move text in PgnRecord::unreadable, after the actions before it.
/// A header's name and value may be separated by spaces, tabs or line breaks
/// (fen.ebnf: `space = ' ' | '\n'`), or by nothing. The text is UTF-8; a
/// byte-order mark is skipped, CR is whitespace.
PgnRecord read_pgn(std::string_view text, const PgnOptions& options = {});

/// Reads one move as written, such as "(0T6)Qg5>>x(0T4)g3+~" or "Nbd7". Throws
/// RecordError (Unreadable).
PgnMove read_pgn_move(std::string_view text);

// ---------------------------------------------------------------------------
// The replay
// ---------------------------------------------------------------------------

/// A record's starting position and the replay of its actions.
class PgnReplay {
 public:
  /// Sets up the position the record's headers declare, under `rules` (the
  /// default: the standard rules) with, on top, the variant's rule family and
  /// the record's Promotions header (S8: a client extension; "Q,R" and "Q R"
  /// both read). Throws RecordError:
  ///
  ///  * Unsupported - first, from the headers that declare the game (the
  ///    first value of each): a Mode other than "5D", a variant the engine
  ///    does not set up, a custom position with boards over 8x8 (its Size
  ///    header, or its board strings' own rows and squares, counted without
  ///    the limit), a promotion set the rules refuse (a pawn or a brawn: a
  ///    ruleset the engine does not implement). So a record of a game the
  ///    engine does not play is Unsupported whatever else is wrong with it.
  ///  * Unreadable - then: PgnRecord::headers_unreadable, a Promotions or
  ///    Size header that cannot be read, a Size that is not the named
  ///    variant's (whatever its numbers), board strings with a named variant,
  ///    a custom position without board strings or with malformed ones.
  ///
  /// It does not look at the move text: whose turn the first action is, is
  /// play_next's to check, after position 0.
  ///
  /// The game: its Board header, else its Variant header (spec issue 2g),
  /// else "custom" if it has 5DFEN board strings, else Standard ("Headers").
  /// A custom position - Board "custom" ("5DFEN and custom variants"), or
  /// "Even", "Odd" or a name beginning "Custom", which the gate's engine
  /// contract also files under its custom positions - is set up from the
  /// board strings, whose labels say whether the game is even or odd ("Even"
  /// and "Odd", also as the last word of a "Custom ..." name, must agree);
  /// its size is the Size header's, or the board strings' own when there is
  /// none, and the player to move is the colour to move on its present (T4,
  /// T5; Q9). A named variant takes no board strings, and its Size header, if
  /// any, must be the variant's size; White moves first (V2).
  explicit PgnReplay(PgnRecord record, const PgnOptions& options = {},
                     const RuleOptions& rules = {});

  /// The state after actions_played() actions.
  const GameState& state() const noexcept { return state_; }
  const PgnRecord& record() const noexcept { return record_; }
  /// The variant's name as the engine knows it ("Standard", ...), or "custom".
  const std::string& game() const noexcept { return game_; }
  std::size_t actions_played() const noexcept { return played_; }
  /// Whether every action read has been played.
  bool finished() const noexcept { return played_ == record_.actions.size(); }

  /// Plays the next action: its moves in order, then the submission. Throws
  /// RecordError (Unreadable, IllegalMove or CannotSubmit), leaving the state
  /// as it was before the action - Unreadable also for an action by the
  /// player who is not to move (the first action included: a named variant
  /// starts with White, V2; a custom position with the colour to move on its
  /// present, T4, T5). Precondition: !finished().
  void play_next();

  /// Plays every remaining action, then - if the move text stopped early -
  /// throws RecordError (Unreadable) with PgnRecord::unreadable's message.
  void play_all();

  /// The move `move` names in `state`, for the player to move, under this
  /// replay's options. Throws RecordError (Unreadable or IllegalMove).
  Move resolve(const GameState& state, const PgnMove& move) const;

 private:
  GameState set_up(const RuleOptions& rules);
  void check_annotations(const GameState& before, const GameState& after, const Move& move,
                         const PgnMove& written) const;

  // Declared in initialisation order: set_up() fills game_, e_file_kings_ and
  // one_king_each_ while it builds state_.
  PgnRecord record_;
  PgnOptions options_;
  std::string game_;
  /// Whether "O-O" and "O-O-O" may be used under CastlingShortcuts::EFileOnly:
  /// every king of the starting position on e1 (White) or e8 (Black), and
  /// both colours with one.
  bool e_file_kings_ = false;
  /// The same under CastlingShortcuts::EFileOneKing: also exactly one king a
  /// colour in the whole starting position.
  bool one_king_each_ = false;
  std::size_t played_ = 0;
  GameState state_;
};

}  // namespace t5d

#endif  // T5D_PGN_HPP
