// 5DFEN board strings: reading, canonical writing, multiverses from board
// strings, and the standard starting positions. Written from the notation
// specification's text (README "5DFEN and custom variants", "Hashing", "5DFEN
// Strictness"; grammar fen.ebnf). See t5d/fen.hpp for what is accepted.

#include "t5d/fen.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <climits>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "t5d/error.hpp"

namespace t5d {

namespace {

constexpr std::size_t kNoOffset = static_cast<std::size_t>(-1);

constexpr bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }
constexpr bool is_letter(char c) noexcept {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
constexpr bool is_upper(char c) noexcept { return c >= 'A' && c <= 'Z'; }
constexpr bool is_space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::string describe_char(char c) {
  if (c == ' ') return "a space";
  if (c > ' ' && c < 127) return std::string("'") + c + "'";
  constexpr char kHex[] = "0123456789ABCDEF";
  const auto byte = static_cast<unsigned char>(c);
  return std::string("byte 0x") + kHex[byte >> 4] + kHex[byte & 15];
}

std::string excerpt(std::string_view text) {
  constexpr std::size_t kMax = 160;
  if (text.size() <= kMax) return std::string(text);
  return std::string(text.substr(0, kMax)) + "...";
}

[[noreturn]] void fail(std::string_view text, const std::string& what,
                       std::size_t offset = kNoOffset) {
  std::string message = "5DFEN board string \"" + excerpt(text) + "\": " + what;
  if (offset != kNoOffset) message += " (at offset " + std::to_string(offset) + ")";
  throw NotationError(message);
}

std::string plural(std::size_t n, const char* singular, const char* plural_form) {
  return std::to_string(n) + " " + (n == 1 ? singular : plural_form);
}

// ---- splitting ---------------------------------------------------------------

struct Field {
  std::string_view text;
  std::size_t at = 0;  // offset in the board string
};

struct Fields {
  Field pieces, timeline, turn, colour;
};

Fields split_fields(std::string_view text) {
  if (text.empty() || text.front() != '[') {
    fail(text, "a board string starts with '['", 0);
  }
  if (text.size() < 2 || text.back() != ']') {
    fail(text, "a board string ends with ']'", text.size() - 1);
  }
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (is_space(text[i])) fail(text, "whitespace is not allowed in a board string", i);
  }
  // Split the text between the brackets at each ':'.
  std::array<Field, 4> fields;
  std::size_t count = 0;
  std::size_t start = 1;
  for (std::size_t i = 1; i < text.size() - 1; ++i) {
    if (text[i] == ':') {
      if (count < 3) fields[count] = Field{text.substr(start, i - start), start};
      ++count;
      start = i + 1;
    }
  }
  if (count != 3) {
    fail(text, "expected 4 fields separated by ':' (pieces:timeline:turn:colour), found " +
                   std::to_string(count + 1));
  }
  fields[3] = Field{text.substr(start, text.size() - 1 - start), start};
  return Fields{fields[0], fields[1], fields[2], fields[3]};
}

// ---- the piece field -----------------------------------------------------------

// The piece field read without a board size: rows top to bottom, the squares
// each row describes, and the pieces with their row (from the top) and file.
// Rows and row widths are capped at 8, the largest board side.
struct LexedPieces {
  int rows = 0;
  std::array<int, BoardSize::kMax> width{};
  std::array<std::size_t, BoardSize::kMax> row_at{};
  int count = 0;
  std::array<int, kSquareCount> row_of{};
  std::array<int, kSquareCount> file_of{};
  std::array<Piece, kSquareCount> piece{};
};

std::string row_name(int row) { return "row " + std::to_string(row + 1) + " (from the top)"; }

LexedPieces lex_pieces(std::string_view text, const Field& field) {
  LexedPieces out;
  out.rows = 1;
  out.row_at[0] = field.at;
  int file = 0;
  const std::string_view f = field.text;
  for (std::size_t i = 0; i < f.size();) {
    const char c = f[i];
    const std::size_t at = field.at + i;
    if (c == '/') {
      out.width[static_cast<std::size_t>(out.rows - 1)] = file;
      if (out.rows == BoardSize::kMax) {
        fail(text, "more than " + std::to_string(BoardSize::kMax) +
                       " rows (boards are at most 8 high)", at);
      }
      out.row_at[static_cast<std::size_t>(out.rows)] = at + 1;
      ++out.rows;
      file = 0;
      ++i;
      continue;
    }
    if (is_digit(c)) {
      std::size_t j = i;
      int n = 0;
      while (j < f.size() && is_digit(f[j])) {
        n = std::min(n * 10 + (f[j] - '0'), 100);
        ++j;
      }
      const std::string number(f.substr(i, j - i));
      if (c == '0') {
        fail(text, "blank count \"" + number +
                       "\": a blank is 1 or more squares, written without leading zeros",
             at);
      }
      if (file + n > BoardSize::kMax) {
        fail(text, row_name(out.rows - 1) + " describes more than " +
                       std::to_string(BoardSize::kMax) + " squares at blank \"" + number +
                       "\" (boards are at most 8 wide)", at);
      }
      file += n;
      i = j;
      continue;
    }
    // A piece: ['+'] letter ['*']. A '+' is skipped, so "+K" reads as "K": a
    // project ruling, not the specification's reading (which reserves '+' for
    // an extended piece alphabet it does not define) - see t5d/fen.hpp.
    if (c == '+') {
      if (i + 1 >= f.size() || !is_letter(f[i + 1])) {
        fail(text, "'+' must be followed by a piece letter", at);
      }
      ++i;
    }
    const char letter = f[i];
    const std::size_t letter_at = field.at + i;
    if (!is_letter(letter)) {
      if (letter == '*') fail(text, "'*' (not yet moved) must follow a piece letter", letter_at);
      fail(text, "unexpected " + describe_char(letter) + " in the piece field", letter_at);
    }
    const std::optional<PieceKind> kind = kind_from_letter(letter);
    if (!kind) {
      fail(text, "unknown piece letter " + describe_char(letter) +
                     " (known: P N B R Q K U D S W C Y, upper case white, lower case black)",
           letter_at);
    }
    ++i;
    bool unmoved = false;
    if (i < f.size() && f[i] == '*') {
      unmoved = true;
      ++i;
    }
    if (file + 1 > BoardSize::kMax) {
      fail(text, row_name(out.rows - 1) + " describes more than " +
                     std::to_string(BoardSize::kMax) + " squares (boards are at most 8 wide)",
           at);
    }
    const auto k = static_cast<std::size_t>(out.count);
    out.row_of[k] = out.rows - 1;
    out.file_of[k] = file;
    out.piece[k] = Piece(*kind, is_upper(letter) ? Colour::White : Colour::Black, unmoved);
    ++out.count;
    ++file;
  }
  out.width[static_cast<std::size_t>(out.rows - 1)] = file;
  return out;
}

// ---- the other fields ------------------------------------------------------------

TimelineLabel parse_label_field(std::string_view text, const Field& field) {
  try {
    return parse_timeline_label(field.text);
  } catch (const NotationError& e) {
    fail(text, e.what(), field.at);
  }
}

int parse_turn_field(std::string_view text, const Field& field) {
  const std::string_view f = field.text;
  if (f.empty()) fail(text, "the turn field is empty", field.at);
  for (std::size_t i = 0; i < f.size(); ++i) {
    if (!is_digit(f[i])) {
      fail(text, "the turn is a number, 0 or more; found " + describe_char(f[i]), field.at + i);
    }
  }
  if (f.size() > 1 && f.front() == '0') fail(text, "the turn has a leading zero", field.at);
  long long value = 0;
  for (const char c : f) {
    value = value * 10 + (c - '0');
    if (value > kMaxTurn) {
      fail(text, "the turn is out of range (at most " + std::to_string(kMaxTurn) + ")",
           field.at);
    }
  }
  return static_cast<int>(value);
}

Colour parse_colour_field(std::string_view text, const Field& field) {
  if (field.text == "w") return Colour::White;
  if (field.text == "b") return Colour::Black;
  fail(text, "the colour is 'w' or 'b', found \"" + std::string(field.text) + "\"", field.at);
}

// ---- writing -----------------------------------------------------------------

void append_int(std::string& out, long long value) {
  std::array<char, 24> buffer{};
  const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  out.append(buffer.data(), result.ptr);
}

void append_label(std::string& out, int position, bool even) {
  const TimelineLabel label = timeline_label(position, even);
  if (label.sign == TimelineLabel::Sign::Plus) out += '+';
  if (label.sign == TimelineLabel::Sign::Minus) out += '-';
  append_int(out, label.magnitude);
}

void append_pieces(std::string& out, const Board& board) {
  for (int rank = board.height() - 1; rank >= 0; --rank) {
    int blanks = 0;
    for (int file = 0; file < board.width(); ++file) {
      const Piece piece = board.at(file, rank);
      if (piece.empty()) {
        ++blanks;
        continue;
      }
      if (blanks > 0) {
        append_int(out, blanks);
        blanks = 0;
      }
      out += piece.letter();
      if (piece.unmoved()) out += '*';
    }
    if (blanks > 0) append_int(out, blanks);
    if (rank > 0) out += '/';
  }
}

std::string describe_place(const FenBoard& b) {
  return "timeline " + to_string(b.timeline) + ", turn " + std::to_string(b.turn) + ", " +
         (b.to_move == Colour::White ? "white" : "black") + " to move";
}

std::string board_number(std::size_t index) { return "board string #" + std::to_string(index + 1); }

}  // namespace

// ---- public: one board ------------------------------------------------------------

FenBoard parse_fen_board(std::string_view text, BoardSize size) {
  if (!size.valid()) {
    throw std::invalid_argument("board size " + to_string(size) + " is outside 1x1..8x8");
  }
  const Fields fields = split_fields(text);
  const LexedPieces lexed = lex_pieces(text, fields.pieces);
  if (lexed.rows != size.height) {
    fail(text, "the piece field has " + plural(static_cast<std::size_t>(lexed.rows), "row", "rows") +
                   "; a " + to_string(size) + " board has " +
                   std::to_string(size.height),
         fields.pieces.at);
  }
  for (int r = 0; r < lexed.rows; ++r) {
    const int w = lexed.width[static_cast<std::size_t>(r)];
    if (w != size.width) {
      fail(text, row_name(r) + " describes " +
                     plural(static_cast<std::size_t>(w), "square", "squares") + "; a " +
                     to_string(size) + " board is " + std::to_string(size.width) + " wide",
           lexed.row_at[static_cast<std::size_t>(r)]);
    }
  }
  BoardBuilder builder(size);
  for (int k = 0; k < lexed.count; ++k) {
    const auto i = static_cast<std::size_t>(k);
    builder.put(lexed.file_of[i], size.height - 1 - lexed.row_of[i], lexed.piece[i]);
  }
  FenBoard out;
  out.board = builder.publish();
  out.timeline = parse_label_field(text, fields.timeline);
  out.turn = parse_turn_field(text, fields.turn);
  out.to_move = parse_colour_field(text, fields.colour);
  return out;
}

BoardSize infer_fen_board_size(std::string_view text) {
  const Fields fields = split_fields(text);
  const LexedPieces lexed = lex_pieces(text, fields.pieces);
  const int width = lexed.width[0];
  for (int r = 1; r < lexed.rows; ++r) {
    if (lexed.width[static_cast<std::size_t>(r)] != width) {
      fail(text, "rows differ in width: " + row_name(0) + " describes " +
                     std::to_string(width) + " squares, " + row_name(r) + " " +
                     std::to_string(lexed.width[static_cast<std::size_t>(r)]),
           lexed.row_at[static_cast<std::size_t>(r)]);
    }
  }
  const BoardSize size{width, lexed.rows};
  if (!size.valid()) fail(text, "the piece field describes no board (an empty row)");
  return size;
}

std::string fen_pieces(const Board& board) {
  std::string out;
  out.reserve(72);
  append_pieces(out, board);
  return out;
}

void append_fen_board_string(std::string& out, const Board& board, int p, int h, bool even) {
  out += '[';
  append_pieces(out, board);
  out += ':';
  append_label(out, p, even);
  out += ':';
  append_int(out, turn_of(h));
  out += ':';
  out += colour_letter(to_move_at(h));
  out += ']';
}

std::string fen_board_string(const Board& board, int p, int h, bool even) {
  std::string out;
  out.reserve(84);
  append_fen_board_string(out, board, p, h, even);
  return out;
}

// ---- public: a multiverse from board strings ------------------------------------------

Multiverse multiverse_from_fen(std::span<const std::string_view> boards, BoardSize size,
                               Parity parity, std::uint32_t ruleset,
                               std::optional<Colour> to_move, bool all_initial) {
  if (!size.valid()) {
    throw std::invalid_argument("board size " + to_string(size) + " is outside 1x1..8x8");
  }
  if (boards.empty()) throw NotationError("no board strings: a multiverse needs a board");

  struct Item {
    FenBoard fen;
    std::size_t index = 0;
    int position = 0;
  };
  std::vector<Item> items;
  items.reserve(boards.size());
  for (std::size_t i = 0; i < boards.size(); ++i) {
    try {
      items.push_back(Item{parse_fen_board(boards[i], size), i, 0});
    } catch (const NotationError& e) {
      throw NotationError(board_number(i) + ": " + e.what());
    }
  }

  // Even or odd: "-0"/"+0" occur only in even games, "0" only in odd ones.
  std::optional<std::size_t> even_witness;
  std::optional<std::size_t> odd_witness;
  for (const Item& item : items) {
    const std::optional<bool> implied = implied_even(item.fen.timeline);
    if (!implied) continue;
    std::optional<std::size_t>& witness = *implied ? even_witness : odd_witness;
    if (!witness) witness = item.index;
  }
  const auto label_of = [&](std::size_t index) {
    return "\"" + to_string(items[index].fen.timeline) + "\"";
  };
  if (even_witness && odd_witness) {
    throw NotationError(board_number(*odd_witness) + " is on timeline " + label_of(*odd_witness) +
                        " (a game with an odd number of starting timelines) and " +
                        board_number(*even_witness) + " on " + label_of(*even_witness) +
                        " (a game with an even number)");
  }
  bool even = false;
  switch (parity) {
    case Parity::Odd:
      if (even_witness) {
        throw NotationError(board_number(*even_witness) + ": timeline " +
                            label_of(*even_witness) +
                            " exists only in a game that starts with an even number of "
                            "timelines, and this game was declared odd");
      }
      even = false;
      break;
    case Parity::Even:
      if (odd_witness) {
        throw NotationError(board_number(*odd_witness) +
                            ": timeline \"0\" does not exist in a game that starts with an even "
                            "number of timelines (its central timelines are \"-0\" and \"+0\")");
      }
      even = true;
      break;
    case Parity::Infer:
      if (even_witness) {
        even = true;
      } else if (odd_witness) {
        even = false;
      } else {
        throw NotationError(
            "cannot tell whether the game starts with an even or an odd number of timelines: "
            "no board is on a central timeline (\"0\", \"-0\" or \"+0\")");
      }
      break;
  }
  for (Item& item : items) item.position = timeline_position(item.fen.timeline, even);

  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    return std::make_tuple(a.position, a.fen.half_turn(), a.index) <
           std::make_tuple(b.position, b.fen.half_turn(), b.index);
  });
  for (std::size_t k = 1; k < items.size(); ++k) {
    const Item& prev = items[k - 1];
    const Item& cur = items[k];
    if (cur.position == prev.position) {
      if (cur.fen.half_turn() == prev.fen.half_turn()) {
        throw NotationError(board_number(prev.index) + " and " + board_number(cur.index) +
                            " are both for " + describe_place(cur.fen));
      }
      if (cur.fen.half_turn() != prev.fen.half_turn() + 1) {
        throw NotationError("timeline " + to_string(cur.fen.timeline) + " has boards for " +
                            describe_place(prev.fen) + " and for " + describe_place(cur.fen) +
                            " but none between: a timeline has one board per half-turn, "
                            "without gaps");
      }
    } else if (cur.position != prev.position + 1) {
      throw NotationError(
          "timelines " + timeline_label_string(prev.position, even) + " and " +
          timeline_label_string(cur.position, even) + " have boards but timeline " +
          timeline_label_string(prev.position + 1, even) +
          " has none: the timelines of a multiverse form a contiguous range");
    }
  }

  Multiverse multiverse(size, even, ruleset);
  for (std::size_t k = 0; k < items.size();) {
    std::size_t end = k;
    NewTimeline nt;
    nt.position = items[k].position;
    nt.first_half_turn = items[k].fen.half_turn();
    while (end < items.size() && items[end].position == nt.position) {
      nt.boards.push_back(items[end].fen.board);
      ++end;
    }
    if (!all_initial && !is_central_timeline(nt.position, even)) {
      // 5DFEN records no creator: the notation's numbering implies it (white's
      // timelines take positive labels, black's negative ones, in order). A
      // variant's set-up is all initial timelines instead (R10, H46).
      nt.creator = nt.position > 0 ? Colour::White : Colour::Black;
      nt.creation_order = timeline_label(nt.position, even).magnitude;
    }
    multiverse.add_timeline(std::move(nt));
    k = end;
  }

  if (to_move) {
    multiverse.set_to_move(*to_move);
    return multiverse;
  }
  // Inferred only where timeline activity (stage E2) cannot change it. The
  // present is the earliest head board of the active timelines, and the
  // player to move is the colour to move on it. Initial timelines are always
  // active, so the present is no later than the earliest initial head: it is
  // one of the heads at or before that half-turn (any head at all if there is
  // no initial timeline). Which one depends on activity - but if all of them
  // have the same colour to move, the player to move does not.
  int initial_earliest = INT_MAX;
  for (const TimelineRef& t : multiverse.timelines()) {
    if (!t->creator().has_value()) initial_earliest = std::min(initial_earliest, t->last_half_turn());
  }
  std::array<const Timeline*, 2> candidate{};  // per colour, the earliest such head
  for (const TimelineRef& t : multiverse.timelines()) {
    const int h = t->last_half_turn();
    if (h > initial_earliest) continue;
    const Timeline*& c = candidate[static_cast<std::size_t>(index_of(to_move_at(h)))];
    if (c == nullptr || h < c->last_half_turn()) c = t.get();
  }
  if (candidate[0] != nullptr && candidate[1] != nullptr) {
    const auto head = [&](const Timeline* t) {
      return "timeline " + timeline_label_string(t->position(), even) + " at turn " +
             std::to_string(turn_of(t->last_half_turn())) + ", " +
             (to_move_at(t->last_half_turn()) == Colour::White ? "white" : "black") + " to move";
    };
    const Timeline* first = candidate[0]->last_half_turn() < candidate[1]->last_half_turn()
                                ? candidate[0]
                                : candidate[1];
    const Timeline* second = first == candidate[0] ? candidate[1] : candidate[0];
    throw std::invalid_argument(
        "cannot tell whose turn it is: the present is the earliest head board of the active "
        "timelines, and it could be the head on " + head(first) + " or the one on " +
        head(second) + " - which depends on which timelines are active, a rule of stage E2 "
        "not yet implemented; give the player to move");
  }
  multiverse.set_to_move(candidate[0] != nullptr ? Colour::White : Colour::Black);
  return multiverse;
}

Multiverse multiverse_from_fen(const std::vector<std::string>& boards, BoardSize size,
                               Parity parity, std::uint32_t ruleset,
                               std::optional<Colour> to_move, bool all_initial) {
  std::vector<std::string_view> views(boards.begin(), boards.end());
  return multiverse_from_fen(std::span<const std::string_view>(views), size, parity, ruleset,
                             to_move, all_initial);
}

// ---- public: the standard positions ---------------------------------------------------

namespace {

BoardRef make_standard_board() {
  constexpr std::array<PieceKind, 8> kBackRank = {
      PieceKind::Rook, PieceKind::Knight, PieceKind::Bishop, PieceKind::Queen,
      PieceKind::King, PieceKind::Bishop, PieceKind::Knight, PieceKind::Rook};
  BoardBuilder builder(BoardSize{8, 8});
  for (int file = 0; file < 8; ++file) {
    const PieceKind back = kBackRank[static_cast<std::size_t>(file)];
    // The unmoved flag sticks only to pawns, rooks and kings (Piece drops it
    // from the others).
    builder.put(file, 0, Piece(back, Colour::White, true));
    builder.put(file, 1, Piece(PieceKind::Pawn, Colour::White, true));
    builder.put(file, 6, Piece(PieceKind::Pawn, Colour::Black, true));
    builder.put(file, 7, Piece(back, Colour::Black, true));
  }
  return builder.publish();
}

}  // namespace

BoardRef standard_board() {
  static const BoardRef board = make_standard_board();
  return board;
}

Multiverse standard_start() {
  Multiverse multiverse(BoardSize{8, 8}, /*even=*/false);
  NewTimeline nt;
  nt.position = 0;
  nt.first_half_turn = half_turn(1, Colour::White);
  nt.boards.push_back(standard_board());
  multiverse.add_timeline(std::move(nt));
  multiverse.set_to_move(Colour::White);
  return multiverse;
}

Multiverse standard_two_timelines_start() {
  Multiverse multiverse(BoardSize{8, 8}, /*even=*/true);
  for (const int position : {-1, 0}) {  // "-0", then "+0"
    NewTimeline nt;
    nt.position = position;
    nt.first_half_turn = half_turn(1, Colour::White);
    nt.boards.push_back(standard_board());
    multiverse.add_timeline(std::move(nt));
  }
  multiverse.set_to_move(Colour::White);
  return multiverse;
}

}  // namespace t5d
