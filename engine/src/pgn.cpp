// The 5dpgn reader and the replay of a record (see t5d/pgn.hpp). Written from
// the notation specification's text (its README and fen.ebnf) and the rules
// reference (docs/rules/README.md); no other implementation's parser was read.

#include "t5d/pgn.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/error.hpp"
#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/variants.hpp"

namespace t5d {

// ---- faults ----------------------------------------------------------------------

const char* fault_name(RecordFault fault) noexcept {
  switch (fault) {
    case RecordFault::Unreadable: return "unreadable";
    case RecordFault::Unsupported: return "unsupported";
    case RecordFault::IllegalMove: return "illegal_move";
    case RecordFault::CannotSubmit: return "cannot_submit";
  }
  return "unreadable";
}

namespace {

// ---- characters ------------------------------------------------------------------------

constexpr bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }
constexpr bool is_upper(char c) noexcept { return c >= 'A' && c <= 'Z'; }
constexpr bool is_lower(char c) noexcept { return c >= 'a' && c <= 'z'; }
constexpr bool is_alpha(char c) noexcept { return is_upper(c) || is_lower(c); }
constexpr bool is_space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}
// Files a-h and ranks 1-8: boards are at most 8x8 (t5d/types.hpp).
constexpr bool is_file(char c) noexcept { return c >= 'a' && c <= 'h'; }
constexpr bool is_rank(char c) noexcept { return c >= '1' && c <= '8'; }
constexpr char lower(char c) noexcept { return is_upper(c) ? static_cast<char>(c - 'A' + 'a') : c; }

bool iequals(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (lower(a[i]) != lower(b[i])) return false;
  }
  return true;
}

std::string_view trim(std::string_view s) noexcept {
  while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
  return s;
}

std::string describe_char(char c) {
  if (c == ' ') return "a space";
  if (c == '\n' || c == '\r') return "a line break";
  if (c > ' ' && c < 127) return std::string("'") + c + "'";
  constexpr char kHex[] = "0123456789ABCDEF";
  const auto byte = static_cast<unsigned char>(c);
  return std::string("byte 0x") + kHex[byte >> 4] + kHex[byte & 15];
}

// A piece of the record, quoted, for messages: at most 60 characters, and
// nothing but printable ASCII.
std::string quoted(std::string_view text) {
  constexpr std::size_t kMax = 60;
  std::string out = "\"";
  for (std::size_t i = 0; i < text.size() && i < kMax; ++i) {
    const char c = text[i];
    out += (c >= ' ' && c < 127) ? c : '?';
  }
  if (text.size() > kMax) out += "...";
  out += '"';
  return out;
}

std::string colour_name(Colour c) { return c == Colour::White ? "White" : "Black"; }

[[noreturn]] void fault(RecordFault kind, const std::string& message) {
  throw RecordError(kind, message);
}
[[noreturn]] void unreadable(const std::string& message) {
  fault(RecordFault::Unreadable, message);
}

// The piece kind a move's letters name, and how many characters they take:
// "RQ" (royal queen) and "PR" (princess) are the specification's two-letter
// names ("Moves"); otherwise one upper-case letter. {nullopt, 0} if none.
std::pair<std::optional<PieceKind>, std::size_t> piece_letters(std::string_view s) noexcept {
  if (s.size() >= 2 && s[0] == 'R' && s[1] == 'Q') return {PieceKind::RoyalQueen, 2};
  if (s.size() >= 2 && s[0] == 'P' && s[1] == 'R') return {PieceKind::Princess, 2};
  if (!s.empty() && is_upper(s[0])) {
    if (const std::optional<PieceKind> k = kind_from_letter(s[0])) return {k, 1};
  }
  return {std::nullopt, 0};
}

// A turn as written in a coordinate or "(~Tx)": 0 or more, no leading zeros.
std::optional<int> read_turn(std::string_view digits) noexcept {
  if (digits.empty() || digits.size() > 9) return std::nullopt;
  if (digits.size() > 1 && digits.front() == '0') return std::nullopt;
  long long value = 0;
  for (const char c : digits) {
    if (!is_digit(c)) return std::nullopt;
    value = value * 10 + (c - '0');
  }
  if (value > kMaxTurn) return std::nullopt;
  return static_cast<int>(value);
}

// ---- header values ---------------------------------------------------------------------------

// A name as the engine compares names (Board, Variant, Mode; t5d/variants.hpp
// compares variant names the same way): lower case, trimmed, every run of
// whitespace one space.
std::string normalised_name(std::string_view text) {
  std::string out;
  bool space = false;
  for (const char c : trim(text)) {
    if (is_space(c)) {
      space = true;
      continue;
    }
    if (space) out += ' ';
    space = false;
    out += lower(c);
  }
  return out;
}

// The Size header's "WxH" ("Headers": "8x8", "7x7"; 'X' read as 'x'), each
// number 1-9999 without leading zeros, surrounding whitespace ignored - with
// no limit on the numbers (the engine's own, 8x8, is the set-up's to apply).
// nullopt if the value is not that.
std::optional<std::pair<int, int>> parse_size(std::string_view value) noexcept {
  const std::string_view v = trim(value);
  const std::size_t x = v.find_first_of("xX");
  if (x == std::string_view::npos) return std::nullopt;
  const auto number = [](std::string_view d) -> std::optional<int> {
    if (d.empty() || d.size() > 4 || d.front() == '0') return std::nullopt;
    int n = 0;
    for (const char c : d) {
      if (!is_digit(c)) return std::nullopt;
      n = n * 10 + (c - '0');
    }
    return n;
  };
  const std::optional<int> w = number(v.substr(0, x));
  const std::optional<int> h = number(v.substr(x + 1));
  if (!w || !h) return std::nullopt;
  return std::pair<int, int>{*w, *h};
}

std::string size_text(const std::pair<int, int>& size) {
  return std::to_string(size.first) + "x" + std::to_string(size.second);
}

// The Promotions header: piece letters separated by commas ("5DFEN and custom
// variants") or spaces ("Headers") - both read (build log spec issue 2f).
// Throws RecordError (Unreadable) if it names no piece or something that is
// not a piece letter.
PromotionSet read_promotions(std::string_view value) {
  PromotionSet set = PromotionSet::from_mask(0);
  std::size_t i = 0;
  bool any = false;
  while (i < value.size()) {
    const char c = value[i];
    if (is_space(c) || c == ',') {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < value.size() && !is_space(value[j]) && value[j] != ',') ++j;
    std::string item(value.substr(i, j - i));
    for (char& ch : item) {
      if (is_lower(ch)) ch = static_cast<char>(ch - 'a' + 'A');
    }
    const auto [kind, length] = piece_letters(item);
    if (!kind || length != item.size()) {
      unreadable("the Promotions header " + quoted(value) + ": " + quoted(item) +
                 " is not a piece letter (\"Moves\": P N B R Q K U D S W C Y, RQ, PR)");
    }
    set.insert(*kind);
    any = true;
    i = j;
  }
  if (!any) unreadable("the Promotions header " + quoted(value) + " names no piece");
  return set;
}

// The headers the engine reads; two different values for one of them make the
// record unreadable. (Every other header is kept and ignored.)
bool interpreted_header(std::string_view name) noexcept {
  return iequals(name, "Board") || iequals(name, "Variant") || iequals(name, "Size") ||
         iequals(name, "Mode") || iequals(name, "Promotions");
}

// The value of a header the engine reads, as the engine reads it: two
// spellings of one value - "7x7" and "7X7", "Q,R" and "q r", "Standard" and
// " standard" - are one value, a repeat rather than a contradiction.
std::string interpreted_value(std::string_view name, std::string_view value) {
  if (iequals(name, "Size")) {
    if (const std::optional<std::pair<int, int>> s = parse_size(value)) return size_text(*s);
  } else if (iequals(name, "Promotions")) {
    try {
      return "promotions " + std::to_string(read_promotions(value).mask());
    } catch (const RecordError&) {
      // Not a promotion set: compared as text below.
    }
  }
  return normalised_name(value);
}

// ---- one move as written ----------------------------------------------------------------

// Reads the text of one move (what read_pgn_move takes), or the annotation
// tokens that follow a move ("(>Lx)", "(~Tx)").
class MoveReader {
 public:
  explicit MoveReader(std::string_view text) noexcept : t_(text) {}

  PgnMove read() {
    PgnMove m;
    m.text = std::string(t_);
    if (t_.empty()) fail("an empty move");
    if (peek() == '(') {
      if (peek(1) == '>' || peek(1) == '~') {
        fail("an annotation token, (>Lx) or (~Tx), follows its move; it does not start one");
      }
      m.board = board_group();
      if (at_end()) {
        fail("a board coordinate must be followed directly by its move, as in (0T1)e4");
      }
    }
    if (starts_with("O-O")) {
      castling(m);
    } else {
      ordinary(m);
    }
    suffixes(m);
    if (!at_end()) fail("unexpected " + describe_char(peek()));
    return m;
  }

  // Reads one or more "(>Lx)" / "(~Tx)" tokens into `m`; the whole text.
  void annotations(PgnMove& m) {
    while (peek() == '(') annotation_group(m);
    if (!at_end()) fail("unexpected " + describe_char(peek()));
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    unreadable("cannot read the move " + quoted(t_) + ": " + what);
  }

  bool at_end() const noexcept { return i_ >= t_.size(); }
  char peek(std::size_t k = 0) const noexcept {
    return i_ + k < t_.size() ? t_[i_ + k] : '\0';
  }
  bool starts_with(std::string_view s) const noexcept {
    return t_.substr(i_).substr(0, s.size()) == s;
  }
  void expect(char c, const std::string& what) {
    if (peek() != c) fail(what);
    ++i_;
  }

  [[noreturn]] void bad_board(std::size_t start) const {
    std::size_t end = t_.find(')', start);
    end = end == std::string_view::npos ? t_.size() : end + 1;
    fail("cannot read " + quoted(t_.substr(start, end - start)) +
         " as a board coordinate, such as (0T1), (-1T6) or (L0 T1) (\"Coordinates\"); a "
         "parenthesised variation is not supported by the notation (\"Branching\")");
  }

  // "(<a>T<b>)" or "(L<a> T<b>)" ("Coordinates": both are valid).
  PgnBoard board_group() {
    const std::size_t start = i_;
    ++i_;  // '('
    if (peek() == 'L') ++i_;
    const std::size_t label_start = i_;
    if (peek() == '+' || peek() == '-') ++i_;
    while (is_digit(peek())) ++i_;
    TimelineLabel label;
    try {
      label = parse_timeline_label(t_.substr(label_start, i_ - label_start));
    } catch (const NotationError&) {
      bad_board(start);
    }
    while (peek() == ' ' || peek() == '\t') ++i_;
    if (peek() != 'T') bad_board(start);
    ++i_;
    const std::size_t turn_start = i_;
    while (is_digit(peek())) ++i_;
    const std::optional<int> turn = read_turn(t_.substr(turn_start, i_ - turn_start));
    if (!turn || peek() != ')') bad_board(start);
    ++i_;
    return PgnBoard{label, *turn};
  }

  // "(>L<label>)" (the timeline created) or "(~T<turn>)" (where the present
  // went) - "Complex scenarios", "Inactive timeline reactivation".
  void annotation_group(PgnMove& m) {
    const std::size_t start = i_;
    ++i_;  // '('
    if (peek() == '>') {
      ++i_;
      expect('L', "a created-timeline token is (>L<timeline>), such as (>L-2)");
      const std::size_t label_start = i_;
      while (!at_end() && peek() != ')') ++i_;
      if (at_end()) fail("unterminated " + quoted(t_.substr(start)));
      try {
        const TimelineLabel label = parse_timeline_label(t_.substr(label_start, i_ - label_start));
        if (m.created) fail("two created-timeline tokens (>Lx) for one move");
        m.created = label;
      } catch (const NotationError& e) {
        fail(e.what());
      }
      ++i_;  // ')'
      return;
    }
    if (peek() == '~') {
      ++i_;
      expect('T', "a present token is (~T<turn>), such as (~T3)");
      const std::size_t turn_start = i_;
      while (is_digit(peek())) ++i_;
      const std::optional<int> turn = read_turn(t_.substr(turn_start, i_ - turn_start));
      if (!turn || peek() != ')') fail("a present token is (~T<turn>), such as (~T3)");
      if (m.present_turn) fail("two present tokens (~Tx) for one move");
      m.present_turn = turn;
      ++i_;  // ')'
      return;
    }
    std::size_t end = t_.find(')', start);
    end = end == std::string_view::npos ? t_.size() : end + 1;
    fail("cannot read " + quoted(t_.substr(start, end - start)) +
         " after a move: expected (>Lx) or (~Tx)");
  }

  void castling(PgnMove& m) {
    if (starts_with("O-O-O")) {
      m.kind = PgnMove::Kind::CastleQueenside;
      i_ += 5;
    } else {
      m.kind = PgnMove::Kind::CastleKingside;
      i_ += 3;
    }
    if (peek() == '-' || peek() == 'O' || peek() == '0') fail("castling is O-O or O-O-O");
    m.piece = PieceKind::King;
  }

  [[noreturn]] void bad_squares(std::string_view chunk) const {
    fail("cannot read the squares " + quoted(chunk) +
         ": a move is written [piece][file][rank][x]<square>, such as e4, exd5, Nbd7, R1a3 or "
         "Qh4e1 (\"Moves\": SAN)");
  }

  void ordinary(PgnMove& m) {
    const auto [kind, length] = piece_letters(t_.substr(i_));
    m.piece = kind;
    i_ += length;
    const std::size_t start = i_;
    while (is_file(peek()) || is_rank(peek()) || peek() == 'x') ++i_;
    const std::string_view chunk = t_.substr(start, i_ - start);
    if (chunk.empty()) {
      fail(at_end() ? std::string("a piece letter without its square")
                    : "expected an upper-case piece letter or a square, found " +
                          describe_char(peek()));
    }
    // What may follow the squares: a jump, a promotion, the marks, or nothing.
    if (!at_end() && std::string_view(">=+#*~!?(").find(peek()) == std::string_view::npos) {
      fail("unexpected " + describe_char(peek()));
    }
    if (peek() == '>') {
      // "Jumps": origin board, piece letter, the piece's square, > or >>, x if
      // a piece is taken, the target board, the target square.
      if (chunk.size() != 2 || !is_file(chunk[0]) || !is_rank(chunk[1])) {
        fail("a jump writes its piece's square in full before > or >> (\"Jumps\", \"Omission\")");
      }
      m.from_file = chunk[0] - 'a';
      m.from_rank = chunk[1] - '1';
      ++i_;
      if (peek() == '>') {
        m.jump = PgnMove::Jump::Branching;
        ++i_;
      } else {
        m.jump = PgnMove::Jump::NonBranching;
      }
      if (peek() == 'x') {
        m.capture = true;
        ++i_;
      }
      if (peek() != '(' || peek(1) == '>' || peek(1) == '~') {
        fail("a jump names its target board, such as (0T1), after > or >> (\"Jumps\")");
      }
      m.target_board = board_group();
      if (!is_file(peek()) || !is_rank(peek(1))) {
        fail("a jump ends with its target square, such as e4, after the target board");
      }
      m.to_file = peek() - 'a';
      m.to_rank = peek(1) - '1';
      i_ += 2;
      return;
    }
    // A physical move in SAN: [file][rank][x]<file><rank>.
    std::size_t k = chunk.size();
    if (k < 2 || !is_file(chunk[k - 2]) || !is_rank(chunk[k - 1])) bad_squares(chunk);
    m.to_file = chunk[k - 2] - 'a';
    m.to_rank = chunk[k - 1] - '1';
    k -= 2;
    if (k > 0 && chunk[k - 1] == 'x') {
      m.capture = true;
      --k;
    }
    if (k > 0 && is_rank(chunk[k - 1])) {
      m.from_rank = chunk[k - 1] - '1';
      --k;
    }
    if (k > 0 && is_file(chunk[k - 1])) {
      m.from_file = chunk[k - 1] - 'a';
      --k;
    }
    if (k != 0) bad_squares(chunk);
  }

  // Promotion ("=X", before the check mark), then '+', '#' or '*', then '~'
  // ("Jumps"), then the PGN assessments '!' and '?' that the specification's
  // example games use ("c5?", "Ng8!", "Qg7g6*!"), then (>Lx) / (~Tx).
  void suffixes(PgnMove& m) {
    if (peek() == '=') {
      if (m.kind != PgnMove::Kind::Ordinary) fail("castling does not promote");
      ++i_;
      const auto [kind, length] = piece_letters(t_.substr(i_));
      if (!kind) fail("'=' must be followed by the piece promoted to, such as =Q (\"Promotions\")");
      m.promotion = kind;
      i_ += length;
    }
    if (peek() == '+' || peek() == '#' || peek() == '*') {
      m.mark = peek();
      ++i_;
    }
    if (peek() == '~') {
      m.present_moved = true;
      ++i_;
    }
    int assessments = 0;
    while (peek() == '!' || peek() == '?') {
      if (++assessments > 2) fail("more than two assessment marks (! ?)");
      ++i_;
    }
    while (peek() == '(') annotation_group(m);
  }

  std::string_view t_;
  std::size_t i_ = 0;
};

// ---- the record ----------------------------------------------------------------------------

constexpr bool is_name_char(char c) noexcept {
  // fen.ebnf's key_character (letters, '.', '_', '-'), and digits (PGN).
  return is_alpha(c) || is_digit(c) || c == '.' || c == '_' || c == '-';
}

class RecordReader {
 public:
  RecordReader(std::string_view text, const PgnOptions& options) : t_(text), options_(options) {
    if (t_.size() >= 3 && static_cast<unsigned char>(t_[0]) == 0xEF &&
        static_cast<unsigned char>(t_[1]) == 0xBB && static_cast<unsigned char>(t_[2]) == 0xBF) {
      i_ = 3;  // a UTF-8 byte-order mark
    }
    line_starts_.push_back(0);
    for (std::size_t k = 0; k < t_.size(); ++k) {
      if (t_[k] == '\n') line_starts_.push_back(k + 1);
    }
  }

  PgnRecord read() {
    PgnRecord r;
    preamble(r);
    try {
      move_text(r);
    } catch (const RecordError& e) {
      r.unreadable = e.what();
    }
    return r;
  }

 private:
  // -- positions and messages
  std::size_t line_of(std::size_t offset) const noexcept {
    const auto it = std::upper_bound(line_starts_.begin(), line_starts_.end(), offset);
    return static_cast<std::size_t>(it - line_starts_.begin());
  }
  std::size_t column_of(std::size_t offset) const noexcept {
    return offset - line_starts_[line_of(offset) - 1] + 1;
  }
  std::string where(std::size_t offset) const {
    return "line " + std::to_string(line_of(offset)) + ", column " +
           std::to_string(column_of(offset));
  }
  [[noreturn]] void fail_at(std::size_t offset, const std::string& what) const {
    unreadable(where(offset) + ": " + what);
  }

  bool at_end() const noexcept { return i_ >= t_.size(); }
  char peek(std::size_t k = 0) const noexcept {
    return i_ + k < t_.size() ? t_[i_ + k] : '\0';
  }
  // Whether a token ends at `pos`: the end, whitespace or a comment.
  bool ends_token(std::size_t pos) const noexcept {
    return pos >= t_.size() || is_space(t_[pos]) || t_[pos] == '{';
  }

  // Whitespace and comments ("Comments": {...}, not nested).
  void skip_blank() {
    for (;;) {
      while (!at_end() && is_space(peek())) ++i_;
      if (peek() != '{' || at_end()) return;
      const std::size_t close = t_.find('}', i_);
      if (close == std::string_view::npos) fail_at(i_, "unterminated comment: '{' without '}'");
      i_ = close + 1;
    }
  }

  // -- headers and board strings
  //
  // A fault here does not stop the reading: it is noted (the first one, in
  // PgnRecord::headers_unreadable) and reading goes on past the next ']' on
  // its line, or from the next line, so that the headers after it are read
  // too - the set-up decides whether the record's game is one the engine
  // plays before it raises the fault (PgnReplay).
  void preamble(PgnRecord& r) {
    for (;;) {
      try {
        skip_blank();
      } catch (const RecordError& e) {
        note(r, e.what());  // an unterminated comment: the rest of the record is in it
        i_ = t_.size();
        return;
      }
      if (at_end() || peek() != '[') return;
      try {
        bracket(r);
      } catch (const RecordError& e) {
        note(r, e.what());
        resynchronise();
      }
    }
  }

  static void note(PgnRecord& r, const std::string& what) {
    if (!r.headers_unreadable) r.headers_unreadable = what;
  }

  // After a fault in a header or board string: on past the next ']' on the
  // line, or to the end of the line.
  void resynchronise() noexcept {
    while (!at_end() && peek() != ']' && peek() != '\n') ++i_;
    if (peek() == ']') ++i_;
  }

  void bracket(PgnRecord& r) {
    const std::size_t at = i_;
    ++i_;  // '['
    const std::size_t name_start = i_;
    while (!at_end() && is_name_char(peek())) ++i_;
    const std::string_view name = t_.substr(name_start, i_ - name_start);
    // fen.ebnf's HEADER: the name, {space}, the quoted value, with space = ' '
    // | '\n' - so a line break may come between them (CR and tab read too).
    std::size_t k = i_;
    while (k < t_.size() && (t_[k] == ' ' || t_[k] == '\t' || t_[k] == '\n' || t_[k] == '\r')) ++k;
    if (!name.empty() && k < t_.size() && t_[k] == '"') {
      // [Name "Value"] ("Headers"; the space may be left out, build log spec
      // issue 2e). The value's escapes are \" and \\.
      i_ = k + 1;
      std::string value;
      for (;;) {
        if (at_end() || peek() == '\n' || peek() == '\r') {
          fail_at(at, "header " + quoted(name) + ": unterminated value (no closing '\"')");
        }
        const char c = t_[i_++];
        if (c == '\\' && (peek() == '"' || peek() == '\\')) {
          value += t_[i_++];
          continue;
        }
        if (c == '"') break;
        value += c;
      }
      // The value is read in full: it is kept even if what follows is wrong,
      // so that a game the record declares is known whatever else is wrong.
      add_header(r, std::string(name), std::move(value), at);
      while (peek() == ' ' || peek() == '\t') ++i_;
      if (peek() != ']') fail_at(i_, "header " + quoted(name) + ": expected ']' after the value");
      ++i_;
      return;
    }
    // A 5DFEN board string, [pieces:timeline:turn:colour], with no
    // whitespace ("5DFEN and custom variants"); read in full at set-up.
    std::size_t end = at + 1;
    while (end < t_.size() && t_[end] != ']' && t_[end] != '[' && !is_space(t_[end])) ++end;
    const std::string_view text =
        t_.substr(at, end < t_.size() && t_[end] == ']' ? end - at + 1 : end - at);
    if (end >= t_.size() || t_[end] != ']' || text.find(':') == std::string_view::npos) {
      fail_at(at, "cannot read " + quoted(text) +
                      " as a header, [Name \"Value\"], or a 5DFEN board string, "
                      "[pieces:timeline:turn:colour] (board strings hold no spaces)");
    }
    r.boards.emplace_back(text);
    i_ = end + 1;
  }

  // Keeps a header. A header the engine reads is kept once: a repeat of its
  // value (as the engine reads it) changes nothing, and a different value is
  // a contradiction - noted, and the first value kept.
  void add_header(PgnRecord& r, std::string name, std::string value, std::size_t at) {
    if (interpreted_header(name)) {
      for (const PgnHeader& h : r.headers) {
        if (!iequals(h.name, name)) continue;
        if (interpreted_value(name, h.value) != interpreted_value(name, value)) {
          note(r, where(at) + ": two values for the header " + quoted(name) + ": " +
                      quoted(h.value) + " and " + quoted(value));
        }
        return;
      }
    }
    r.headers.push_back(PgnHeader{std::move(name), std::move(value), at});
  }

  // -- the move text
  void move_text(PgnRecord& r) {
    for (;;) {
      skip_blank();
      if (at_end()) break;
      const std::size_t at = i_;
      const char c = peek();
      if (!r.result.empty()) {
        fail_at(at, "text after the result " + quoted(r.result) +
                        " (the move text ends with its result)");
      }
      if (c == '[') {
        fail_at(at, "a header or board string after the move text began (they come first)");
      }
      if (c == '}') fail_at(at, "'}' without '{'");
      if (c == '/') {
        ++i_;
        slash(r, at);
      } else if (is_digit(c)) {
        number(r, at);
      } else if (c == '*' && ends_token(i_ + 1)) {
        ++i_;
        result(r, "*");
      } else if (c == '(' && (peek(1) == '>' || peek(1) == '~')) {
        annotation_token(at);
      } else if (c == '(' || is_alpha(c)) {
        move_token(at);
      } else {
        fail_at(at, "unexpected " + describe_char(c));
      }
    }
    close(r);
  }

  // The action being read is complete: keep it if it has moves (an action
  // with none, at the end, was never played: "23. ... / 1-0").
  void close(PgnRecord& r) {
    if (current_ && !current_->moves.empty()) r.actions.push_back(std::move(*current_));
    current_.reset();
  }

  void begin(Colour colour, int number, std::size_t at) {
    PgnAction a;
    a.colour = colour;
    a.turn_number = number;
    a.offset = at;
    a.line = line_of(at);
    a.column = column_of(at);
    current_ = std::move(a);
  }

  std::string action_name(const PgnAction& a) const {
    return colour_name(a.colour) + "'s action of turn " + std::to_string(a.turn_number);
  }

  // The action before a turn marker is complete: it must have moves.
  std::optional<PgnAction> finish_for_marker(PgnRecord& r, std::size_t at) {
    if (!current_) return std::nullopt;
    if (current_->moves.empty()) fail_at(at, action_name(*current_) + " has no moves");
    PgnAction done = *current_;
    close(r);
    return done;
  }

  // "<x>." or "<x>w.": White's moves of turn x.
  void white_marker(PgnRecord& r, int n, std::size_t at) {
    const std::optional<PgnAction> before = finish_for_marker(r, at);
    if (before && before->colour == Colour::White) {
      fail_at(at, "turn " + std::to_string(n) + " begins with White's moves, but Black's action of "
                  "turn " + std::to_string(before->turn_number) +
                  " is missing (\"Turns\": White's moves, then '/' or <x>b. and Black's)");
    }
    if (before && options_.check_turn_numbers && n != before->turn_number + 1) {
      fail_at(at, "turn " + std::to_string(n) + " follows turn " +
                      std::to_string(before->turn_number) + ": turn numbers count the turns");
    }
    begin(Colour::White, n, at);
  }

  // "<x>b.": Black's moves of turn x.
  void black_marker(PgnRecord& r, int n, std::size_t at) {
    const std::optional<PgnAction> before = finish_for_marker(r, at);
    if (before && before->colour == Colour::Black) {
      fail_at(at, "Black's moves of turn " + std::to_string(n) + " follow Black's action of turn " +
                      std::to_string(before->turn_number) + ": White's action is missing");
    }
    if (before && options_.check_turn_numbers && n != before->turn_number) {
      fail_at(at, "Black's moves of turn " + std::to_string(n) + " follow White's of turn " +
                      std::to_string(before->turn_number) +
                      ": a turn is White's action, then Black's (\"Turns\")");
    }
    begin(Colour::Black, n, at);
  }

  // "/": Black's moves of the turn whose White moves precede it.
  void slash(PgnRecord& r, std::size_t at) {
    if (!current_ || current_->colour != Colour::White) {
      fail_at(at, "'/' separates White's moves from Black's (\"Turns\"): it must follow White's "
                  "moves");
    }
    const std::optional<PgnAction> before = finish_for_marker(r, at);
    begin(Colour::Black, before->turn_number, at);
  }

  void result(PgnRecord& r, std::string token) {
    close(r);
    r.result = std::move(token);
  }

  // A token that starts with a digit: a turn marker or a result.
  void number(PgnRecord& r, std::size_t at) {
    std::size_t k = i_;
    while (k < t_.size() && is_digit(t_[k])) ++k;
    const std::string_view digits = t_.substr(i_, k - i_);
    const auto rest_is = [&](std::string_view s) {
      return t_.substr(at).substr(0, s.size()) == s && ends_token(at + s.size());
    };
    for (const std::string_view token : {"1-0", "0-1", "1/2-1/2"}) {
      if (rest_is(token)) {
        i_ = at + token.size();
        result(r, std::string(token));
        return;
      }
    }
    if (t_.substr(at).substr(0, 3) == "0-0") {
      fail_at(at, "castling is written with the letter O, O-O or O-O-O, not the digit zero "
                  "(\"Castling\")");
    }
    const char after = k < t_.size() ? t_[k] : '\0';
    Colour colour = Colour::White;
    std::size_t end = k;
    if (after == '.') {
      end = k + 1;
      if (end < t_.size() && t_[end] == '.') {
        const std::size_t shown = std::min<std::size_t>(8, t_.size() - at);
        fail_at(at, "cannot read " + quoted(t_.substr(at, shown)) +
                        ": Black's moves are introduced by '/' or <x>b. (\"Turns\")");
      }
    } else if ((after == 'w' || after == 'b') && k + 1 < t_.size() && t_[k + 1] == '.') {
      colour = after == 'w' ? Colour::White : Colour::Black;
      end = k + 2;
    } else {
      std::size_t stop = k;
      while (stop < t_.size() && !is_space(t_[stop]) && t_[stop] != '{') ++stop;
      fail_at(at, "cannot read " + quoted(t_.substr(at, stop - at)) +
                      ": expected a turn number (<x>., <x>w. or <x>b.) or a result (1-0, 0-1, "
                      "1/2-1/2, *)");
    }
    const std::optional<int> n = read_turn(digits);
    if (!n || *n < 1) {
      fail_at(at, "turn number " + quoted(digits) + ": a positive number without leading zeros");
    }
    i_ = end;
    if (colour == Colour::White) {
      white_marker(r, *n, at);
    } else {
      black_marker(r, *n, at);
    }
  }

  // One move: a run of characters up to whitespace, a comment, '/' or a
  // bracket - where a parenthesised coordinate may hold a space, "(L0 T1)".
  std::string_view take_move_text(std::size_t at) {
    while (!at_end()) {
      const char c = peek();
      if (c == '(') {
        std::size_t j = i_ + 1;
        while (j < t_.size() && t_[j] != ')' && t_[j] != '(' && t_[j] != '\n' && t_[j] != '\r' &&
               t_[j] != '{' && t_[j] != '}') {
          ++j;
        }
        if (j >= t_.size() || t_[j] != ')') {
          fail_at(i_, "unterminated '(' (a board coordinate is (0T1) or (L0 T1); a parenthesised "
                      "variation is not supported by the notation, \"Branching\")");
        }
        i_ = j + 1;
        continue;
      }
      if (is_space(c) || c == '{' || c == '}' || c == '/' || c == '[' || c == ']') break;
      ++i_;
    }
    return t_.substr(at, i_ - at);
  }

  void move_token(std::size_t at) {
    const std::string_view text = take_move_text(at);
    if (!current_) {
      fail_at(at, "the move " + quoted(text) +
                      " comes before the first turn number (\"Turns\": <x>. or <x>b.)");
    }
    PgnMove m;
    try {
      m = MoveReader(text).read();
    } catch (const RecordError& e) {
      fail_at(at, e.what());
    }
    m.offset = at;
    m.line = line_of(at);
    m.column = column_of(at);
    current_->moves.push_back(std::move(m));
  }

  // "(>Lx)" or "(~Tx)" after a move, separated from it by whitespace or a
  // comment (as in the specification's "... (2T9)d2~ {...} (>L-4)").
  void annotation_token(std::size_t at) {
    const std::string_view text = take_move_text(at);
    if (!current_ || current_->moves.empty()) {
      fail_at(at, quoted(text) + " follows no move");
    }
    PgnMove& m = current_->moves.back();
    try {
      MoveReader(text).annotations(m);
    } catch (const RecordError& e) {
      fail_at(at, e.what());
    }
    m.text += ' ';
    m.text += text;
  }

  std::string_view t_;
  const PgnOptions& options_;
  std::size_t i_ = 0;
  std::vector<std::size_t> line_starts_;
  std::optional<PgnAction> current_;
};

// ---- the set-up's headers --------------------------------------------------------------------

bool over_the_limit(const std::pair<int, int>& size) noexcept {
  return size.first > BoardSize::kMax || size.second > BoardSize::kMax;
}

// The width and height a 5DFEN board string's piece field claims, counted
// without the engine's 8x8 limit and without judging anything else: rows split
// at '/', a blank's number counted in full, every other character but '+' and
// '*' one square (numbers capped at 1000). For the Unsupported decision only:
// the board string is read in full afterwards (t5d/fen.hpp).
std::pair<int, int> claimed_dimensions(std::string_view text) noexcept {
  constexpr int kCap = 1000;
  std::string_view f = text;
  if (!f.empty() && f.front() == '[') f.remove_prefix(1);
  if (const std::size_t colon = f.find(':'); colon != std::string_view::npos) {
    f = f.substr(0, colon);
  } else if (!f.empty() && f.back() == ']') {
    f.remove_suffix(1);
  }
  int width = 0;
  int height = 1;
  int row = 0;
  for (std::size_t i = 0; i < f.size();) {
    const char c = f[i];
    if (c == '/') {
      width = std::max(width, row);
      row = 0;
      height = std::min(height + 1, kCap);
      ++i;
      continue;
    }
    if (is_digit(c)) {
      int n = 0;
      while (i < f.size() && is_digit(f[i])) {
        n = std::min(n * 10 + (f[i] - '0'), kCap);
        ++i;
      }
      row = std::min(row + n, kCap);
      continue;
    }
    if (c != '+' && c != '*') row = std::min(row + 1, kCap);
    ++i;
  }
  return {std::max(width, row), height};
}

// A Board (or Variant) value that sets up the record's own position from its
// 5DFEN board strings: "custom" ("5DFEN and custom variants") - and "Even",
// "Odd" or a name beginning "Custom", which the gate's engine contract files
// with it under "(custom positions)". The parity it states, if any: "Even" or
// "Odd", alone or as the last word of a "Custom ..." name.
struct CustomBoardName {
  bool custom = false;
  Parity parity = Parity::Infer;
};

CustomBoardName custom_board_name(std::string_view value) {
  const std::string n = normalised_name(value);
  if (n == "even") return {true, Parity::Even};
  if (n == "odd") return {true, Parity::Odd};
  constexpr std::string_view kCustom = "custom";
  if (n.compare(0, kCustom.size(), kCustom) != 0) return {};
  if (n.size() > kCustom.size() && is_alpha(n[kCustom.size()])) return {};  // "customs"
  const std::size_t cut = n.find_last_of(" -");
  const std::string_view last =
      cut == std::string::npos ? std::string_view(n) : std::string_view(n).substr(cut + 1);
  if (last == "even") return {true, Parity::Even};
  if (last == "odd") return {true, Parity::Odd};
  return {true, Parity::Infer};
}

// S6 ("Castling"): "On starting positions with the white king on e1 and the
// black king on e8" the shortcuts may be used; "with more than one king for
// either player or kings in other places than e1 and e8" the king's move must
// be written. Read per board (CastlingShortcuts::EFileOnly): every king of the
// starting position on e1 (White) or e8 (Black), each side with at least one;
// and over the whole starting position (EFileOneKing): also exactly one king a
// side.
struct CastlingShortcutUse {
  bool per_board = false;
  bool one_king_each = false;
};

CastlingShortcutUse castling_shortcut_use(const Multiverse& mv) {
  const BoardSize size = mv.board_size();
  if (size.width < 5 || size.height < 8) return {};
  const int e1 = make_square(4, 0);
  const int e8 = make_square(4, 7);
  int white = 0;
  int black = 0;
  bool elsewhere = false;
  mv.for_each_board([&](int, int, const Board& b) {
    for (const Colour c : {Colour::White, Colour::Black}) {
      const std::uint64_t kings = b.pieces(c, PieceKind::King);
      if (kings == 0) continue;
      const int home = c == Colour::White ? e1 : e8;
      if (kings != square_bit(home)) elsewhere = true;
      (c == Colour::White ? white : black) += std::popcount(kings);
    }
  });
  CastlingShortcutUse use;
  use.per_board = white > 0 && black > 0 && !elsewhere;
  use.one_king_each = use.per_board && white == 1 && black == 1;
  return use;
}

// ---- messages about boards and moves ---------------------------------------------------------

std::string board_text(const Multiverse& mv, int p, int h) {
  return "(" + timeline_label_string(p, mv.even()) + "T" + std::to_string(turn_of(h)) + ", " +
         (to_move_at(h) == Colour::White ? "white" : "black") + " to move)";
}

std::string board_text(const PgnBoard& b) {
  return "(" + to_string(b.timeline) + "T" + std::to_string(b.turn) + ")";
}

std::string piece_name(PieceKind k) {
  switch (k) {
    case PieceKind::Pawn: return "pawn";
    case PieceKind::Knight: return "knight";
    case PieceKind::Bishop: return "bishop";
    case PieceKind::Rook: return "rook";
    case PieceKind::Queen: return "queen";
    case PieceKind::King: return "king";
    case PieceKind::Unicorn: return "unicorn";
    case PieceKind::Dragon: return "dragon";
    case PieceKind::Princess: return "princess";
    case PieceKind::Brawn: return "brawn";
    case PieceKind::CommonKing: return "common king";
    case PieceKind::RoyalQueen: return "royal queen";
  }
  return "piece";
}

std::string square_text(int file, int rank) {
  std::string s;
  s += static_cast<char>('a' + file);
  s += static_cast<char>('1' + rank);
  return s;
}

std::string move_list(const std::vector<Move>& moves, bool even) {
  std::string out;
  for (const Move& m : moves) {
    if (!out.empty()) out += ", ";
    out += to_string(m, even);
  }
  return out;
}

int position_of(const TimelineLabel& label, bool even) {
  try {
    return timeline_position(label, even);
  } catch (const NotationError& e) {
    unreadable(e.what());
  }
}

}  // namespace

// ---- public: reading --------------------------------------------------------------------------

const std::string* PgnRecord::header(std::string_view name) const noexcept {
  for (const PgnHeader& h : headers) {
    if (iequals(h.name, name)) return &h.value;
  }
  return nullptr;
}

PgnRecord read_pgn(std::string_view text, const PgnOptions& options) {
  return RecordReader(text, options).read();
}

PgnMove read_pgn_move(std::string_view text) { return MoveReader(text).read(); }

// ---- public: the replay -------------------------------------------------------------------------

PgnReplay::PgnReplay(PgnRecord record, const PgnOptions& options, const RuleOptions& rules)
    : record_(std::move(record)), options_(options), state_(set_up(rules)) {}

GameState PgnReplay::set_up(const RuleOptions& base) {
  const PgnRecord& r = record_;
  RuleOptions rules = base;

  // 1. The game the record declares - from the first value of each header
  // that declares it - before anything else: a record of a game the engine
  // does not play is Unsupported whatever else is wrong with it (the gate's
  // engine contract allows "unsupported" before the first position; a
  // refusal of such a record as unreadable would count against the engine).
  const std::string* mode = r.header("Mode");
  if (mode != nullptr && normalised_name(*mode) != "5d") {
    fault(RecordFault::Unsupported,
          "the Mode header is " + quoted(*mode) + ": the engine plays 5D (\"Headers\": "
          "\"should always be set to 5D\")");
  }
  const std::string* board = r.header("Board");
  const bool from_variant_header = board == nullptr && r.header("Variant") != nullptr;
  if (board == nullptr) board = r.header("Variant");
  const std::string game_header = from_variant_header ? "the Variant header" : "the Board header";
  // No Board or Variant header: custom when there are board strings.
  const CustomBoardName custom_name = board != nullptr
                                          ? custom_board_name(*board)
                                          : CustomBoardName{!r.boards.empty(), Parity::Infer};
  const bool custom = custom_name.custom;
  const Variant* variant = nullptr;
  if (!custom) {
    const std::string name = board != nullptr ? std::string(trim(*board)) : "Standard";
    variant = find_variant(name);
    if (variant == nullptr) {
      fault(RecordFault::Unsupported,
            "the variant " + quoted(name) + (from_variant_header ? " (the Variant header)" : "") +
                " is not one the engine sets up (rules reference V3; see the engine README)");
    }
  }
  const std::string* size_header = r.header("Size");
  const std::optional<std::pair<int, int>> size =
      size_header != nullptr ? parse_size(*size_header) : std::nullopt;
  if (custom) {
    // Boards larger than the engine's: by the Size header, or by the board
    // strings themselves (which also give the size when the header is missing).
    if (size && over_the_limit(*size)) {
      fault(RecordFault::Unsupported,
            "boards of " + size_text(*size) +
                " (the Size header): the engine's boards are at most 8x8");
    }
    for (std::size_t k = 0; k < r.boards.size(); ++k) {
      const std::pair<int, int> claimed = claimed_dimensions(r.boards[k]);
      if (over_the_limit(claimed)) {
        fault(RecordFault::Unsupported,
              "board string #" + std::to_string(k + 1) + " describes boards of " +
                  size_text(claimed) + ": the engine's boards are at most 8x8");
      }
    }
  }
  // A promotion set the rules refuse (a pawn or a brawn, RuleOptions::
  // validate) is a ruleset the engine does not implement - Unsupported; a
  // Promotions header that names no piece letters at all cannot be read -
  // Unreadable, below.
  std::optional<std::string> promotions_unreadable;
  if (const std::string* promotions = r.header("Promotions")) {
    try {
      rules.promotions = read_promotions(*promotions);
    } catch (const RecordError& e) {
      promotions_unreadable = e.what();
    }
  }
  if (variant != nullptr) rules.family = variant->family;
  try {
    rules.validate();
  } catch (const std::invalid_argument& e) {
    fault(RecordFault::Unsupported, std::string("the rules this record asks for: ") + e.what());
  }

  // 2. The game is one the engine plays: now the record's own defects.
  if (r.headers_unreadable) unreadable(*r.headers_unreadable);
  if (promotions_unreadable) unreadable(*promotions_unreadable);
  if (size_header != nullptr && !size) {
    unreadable("the Size header " + quoted(*size_header) +
               " is not \"WxH\", such as \"8x8\" (\"Headers\")");
  }

  if (custom) {
    // "5DFEN and custom variants": the board strings, and the Size header -
    // required by the specification, taken from the board strings when it is
    // missing. The player to move is the colour to move on the present (T4,
    // T5; Q9) - the first action, if by the other player, is play_next's to
    // refuse, after position 0.
    if (r.boards.empty()) {  // (board is set: without board strings, custom needs its name)
      unreadable(game_header + " " + quoted(*board) +
                 " sets up its position from 5DFEN board strings, and this record has none "
                 "(\"5DFEN and custom variants\")");
    }
    BoardSize s{};
    try {
      s = size ? BoardSize{size->first, size->second} : infer_fen_board_size(r.boards.front());
    } catch (const NotationError& e) {
      unreadable(e.what());
    }
    std::optional<GameState> state;
    try {
      state.emplace(game_from_fen(r.boards, s, custom_name.parity, rules));
    } catch (const NotationError& e) {
      std::string message = e.what();
      if (custom_name.parity != Parity::Infer) {
        message += " (" + game_header + " " + quoted(*board) + " says the game is " +
                   (custom_name.parity == Parity::Even ? "even" : "odd") + ")";
      }
      unreadable(message);
    } catch (const std::invalid_argument& e) {
      unreadable(std::string("the custom position: ") + e.what());
    }
    game_ = "custom";
    const CastlingShortcutUse use = castling_shortcut_use(state->multiverse());
    e_file_kings_ = use.per_board;
    one_king_each_ = use.one_king_each;
    return std::move(*state);
  }

  if (!r.boards.empty()) {
    unreadable("5DFEN board strings set up a custom position, which needs Board \"custom\"; this "
               "record's game is " + quoted(variant->name) + " (\"5DFEN and custom variants\")");
  }
  if (size && (size->first != variant->size.width || size->second != variant->size.height)) {
    unreadable("the Size header says " + size_text(*size) + ", but " + quoted(variant->name) +
               (board == nullptr ? " (the game when no Board header is given, \"Headers\")"
                                 : std::string()) +
               " is played on " + to_string(variant->size) + " boards");
  }
  GameState state = variant_start(*variant, rules);
  game_ = variant->name;
  const CastlingShortcutUse use = castling_shortcut_use(state.multiverse());
  e_file_kings_ = use.per_board;
  one_king_each_ = use.one_king_each;
  return state;
}

Move PgnReplay::resolve(const GameState& state, const PgnMove& w) const {
  const Multiverse& mv = state.multiverse();
  const bool even = mv.even();
  const Colour mover = state.to_move();
  if (!state.may_move()) {
    fault(RecordFault::IllegalMove,
          colour_name(mover) + "'s action has met its obligations - the present has shifted to " +
              colour_name(opposite(mover)) +
              "'s colour (T5) - and no move may be made after that (Q4)");
  }

  // The source board(s).
  std::vector<int> sources;
  if (w.board) {
    const int p = position_of(w.board->timeline, even);
    const Timeline* t = mv.timeline(p);
    if (t == nullptr) {
      fault(RecordFault::IllegalMove,
            "there is no timeline " + to_string(w.board->timeline) + " (L6, L7)");
    }
    const int h = half_turn(w.board->turn, mover);
    if (t->last_half_turn() != h) {
      fault(RecordFault::IllegalMove,
            board_text(*w.board) + " is not a playable board of " + colour_name(mover) +
                "'s (L5, T3): timeline " + to_string(w.board->timeline) +
                "'s playable board is " + board_text(mv, p, t->last_half_turn()));
    }
    sources.push_back(p);
  } else if (mv.timeline_count() == 1) {
    const int p = mv.min_timeline();
    if (!state.playable_by_mover(p)) {
      fault(RecordFault::IllegalMove, colour_name(mover) + " has no playable board to move on: "
                                          "the timeline's playable board is " +
                                          board_text(mv, p, mv.timeline(p)->last_half_turn()));
    }
    sources.push_back(p);
  } else if (options_.infer_omitted_boards) {
    sources = state.playable_timelines();
  } else {
    unreadable("the move does not name its board, and there are " +
               std::to_string(mv.timeline_count()) +
               " timelines: a move must then be preceded by its board's coordinates, such as "
               "(0T5) (\"Moves\")");
  }

  // The target board of a jump: the mover's colour (L4).
  int target_p = 0;
  int target_h = 0;
  if (w.jump != PgnMove::Jump::None) {
    target_p = position_of(w.target_board->timeline, even);
    target_h = half_turn(w.target_board->turn, mover);
    if (sources.size() == 1 && target_p == sources.front() &&
        target_h == mv.timeline(target_p)->last_half_turn()) {
      unreadable("a jump's target board is its source board; a move within one board is written "
                 "without > or >> (\"Jumps\")");
    }
  }

  // Castling shortcuts (S6 / Q26).
  const bool castling = w.kind != PgnMove::Kind::Ordinary;
  const bool kingside = w.kind == PgnMove::Kind::CastleKingside;
  const bool e_file_reading = options_.castling_shortcuts != CastlingShortcuts::AnyKing;
  if (castling && options_.castling_shortcuts == CastlingShortcuts::EFileOnly && !e_file_kings_) {
    unreadable(std::string(kingside ? "O-O is a shortcut for Ke1g1/Ke8g8"
                                    : "O-O-O is a shortcut for Ke1c1/Ke8c8") +
               " in games whose kings start on e1 and e8 (S6, \"Castling\"); this game's do not, "
               "so the king's move, K<from><to>, must be written (Q26)");
  }
  if (castling && options_.castling_shortcuts == CastlingShortcuts::EFileOneKing &&
      !one_king_each_) {
    unreadable(std::string(kingside ? "O-O is a shortcut for Ke1g1/Ke8g8"
                                    : "O-O-O is a shortcut for Ke1c1/Ke8c8") +
               " in games that start with one king a side, on e1 and e8 (S6; \"Castling\": "
               "\"more than one king for either player\", read over the whole starting position - "
               "EFileOneKing); this game's do not, so the king's move, K<from><to>, must be "
               "written");
  }

  MoveList generated;
  for (const int p : sources) append_board_moves(state, p, generated);

  const int home_rank = mover == Colour::White ? 0 : 7;
  std::vector<Move> fits;        // what is written, every mark included
  std::vector<Move> near_fits;   // the same but for the capture mark
  for (const Move& m : generated) {
    if (castling) {
      if (e_file_reading) {
        // Ke1g1 / Ke8g8, Ke1c1 / Ke8c8 - the king's two-square move.
        if (m.piece.kind() != PieceKind::King || !m.physical() ||
            m.from_square != make_square(4, home_rank) ||
            m.to_square != make_square(kingside ? 6 : 2, home_rank)) {
          continue;
        }
      } else {
        if (!m.castle() || (file_of(m.to_square) > file_of(m.from_square)) != kingside) continue;
      }
      fits.push_back(m);
      continue;
    }
    const PieceKind kind = w.piece ? *w.piece : PieceKind::Pawn;
    if (m.piece.kind() != kind) continue;
    if (w.from_file >= 0 && file_of(m.from_square) != w.from_file) continue;
    if (w.from_rank >= 0 && rank_of(m.from_square) != w.from_rank) continue;
    if (m.to_square != make_square(w.to_file, w.to_rank)) continue;
    if (w.jump == PgnMove::Jump::None) {
      if (!m.physical()) continue;
    } else if (m.physical() || m.to_timeline != target_p || m.to_half_turn != target_h) {
      continue;
    }
    if (w.promotion && (!m.promotes() || m.promotion != *w.promotion)) continue;
    const bool marks_agree = w.capture ? m.capture()
                                       : (!options_.require_capture_marks || !m.capture());
    (marks_agree ? fits : near_fits).push_back(m);
  }

  if (fits.empty()) {
    if (!near_fits.empty()) {
      unreadable(w.capture ? "it is written as a capture ('x'), but " +
                                 to_string(near_fits.front(), even) + " captures nothing"
                           : to_string(near_fits.front(), even) +
                                 " is a capture, written without 'x' (require_capture_marks)");
    }
    std::string what;
    if (castling) {
      what = std::string("no castling ") + (kingside ? "O-O" : "O-O-O") + " of " +
             colour_name(mover) + "'s";
    } else {
      const PieceKind kind = w.piece ? *w.piece : PieceKind::Pawn;
      what = "no " + piece_name(kind) + " of " + colour_name(mover) + "'s";
      if (w.from_file >= 0 || w.from_rank >= 0) {
        what += " on ";
        if (w.from_file >= 0) what += static_cast<char>('a' + w.from_file);
        if (w.from_rank >= 0) what += static_cast<char>('1' + w.from_rank);
      }
      what += " can move to " + square_text(w.to_file, w.to_rank);
      if (w.jump != PgnMove::Jump::None) what += " on " + board_text(*w.target_board);
      if (w.promotion) what += " promoting to a " + piece_name(*w.promotion);
    }
    if (w.board) {
      what += " from " + board_text(*w.board);
    } else if (sources.size() == 1) {
      const int p = sources.front();
      what += " from " + board_text(mv, p, mv.timeline(p)->last_half_turn());
    }
    fault(RecordFault::IllegalMove, what + " (no pseudo-legal move fits: sections 3-5)");
  }
  if (fits.size() > 1) {
    bool promotions_only = !w.promotion;
    for (const Move& m : fits) {
      Move a = m;
      Move b = fits.front();
      a.promotion = b.promotion;
      if (!m.promotes() || !(a == b)) promotions_only = false;
    }
    unreadable(promotions_only
                   ? "the piece promoted to is not written (=X), and this ruleset offers " +
                         std::to_string(fits.size()) + " (S8): " + move_list(fits, even)
                   : "ambiguous: it fits " + std::to_string(fits.size()) +
                         " pseudo-legal moves - " + move_list(fits, even) +
                         " - and the engine does not guess (\"Omission\" lets a writer leave out "
                         "a piece a pin rules out, but legality is judged per action, at "
                         "submission - C4 - not per move)");
  }
  const Move& m = fits.front();
  if (w.jump == PgnMove::Jump::Branching && !m.branching()) {
    unreadable("written with '>>', a branching jump, but " + to_string(m, even) +
               " does not branch: its target board is that timeline's playable board (J3, J4)");
  }
  if (w.jump == PgnMove::Jump::NonBranching && m.branching()) {
    unreadable("written with '>', a jump that does not branch, but " + to_string(m, even) +
               " branches: its target board is a history board (J3, J5)");
  }
  return m;
}

void PgnReplay::check_annotations(const GameState& before, const GameState& after,
                                  const Move& move, const PgnMove& w) const {
  const Multiverse& mv = after.multiverse();
  const bool even = mv.even();
  const Colour mover = before.to_move();
  // A branching move's new timeline: outermost on the mover's side (L7).
  const int created = mover == Colour::White ? mv.max_timeline() : mv.min_timeline();
  if (w.created) {
    if (!move.branching()) {
      unreadable("(>L" + to_string(*w.created) + ") names the timeline the move created, but it "
                 "creates none: it does not branch (J3)");
    }
    if (position_of(*w.created, even) != created) {
      unreadable("(>L" + to_string(*w.created) + ") says the move created timeline " +
                 to_string(*w.created) + ", but it created " +
                 timeline_label_string(created, even) + " (L7, J9)");
    }
  }
  if (w.present_moved) {
    if (!move.branching()) {
      unreadable("'~' marks a branching jump that moves the present (T15), and this move does not "
                 "branch");
    }
    if (!after.active(created)) {
      unreadable("'~' says the present moved to the new branch, but the branch, timeline " +
                 timeline_label_string(created, even) +
                 ", is inactive, and an inactive timeline never moves the present (T8, T10)");
    }
    const int first = mv.timeline(created)->first_half_turn();
    if (first > before.present() + 1) {
      unreadable("'~' says the present moved to the new branch, but its first board " +
                 board_text(mv, created, first) + " lies after the present (T4, T8)");
    }
  }
  if (w.present_turn && turn_of(after.present()) != *w.present_turn) {
    unreadable("(~T" + std::to_string(*w.present_turn) + ") says the present is at turn " +
               std::to_string(*w.present_turn) + " after the move, but it is at turn " +
               std::to_string(turn_of(after.present())) + " (T4, T12)");
  }
}

void PgnReplay::play_next() {
  if (finished()) throw std::logic_error("PgnReplay::play_next: every action has been played");
  const PgnAction& action = record_.actions[played_];
  const std::string name = "action " + std::to_string(played_ + 1) + " (" +
                           colour_name(action.colour) + "'s, turn " +
                           std::to_string(action.turn_number) + ")";
  GameState s = state_;
  if (action.colour != s.to_move()) {
    std::string why;
    if (played_ == 0) {
      // The set-up does not look at the move text (so that position 0 is
      // reported whenever a state can be built): this is where a record that
      // starts with the wrong player is refused.
      why = game_ == "custom"
                ? " (the custom position's present - the column of the earliest playable board "
                  "of its active timelines - is turn " + std::to_string(turn_of(s.present())) +
                      ", " + colour_name(s.to_move()) + "'s colour: T4, T5, Q9)"
                : " (White moves first in " + quoted(game_) + ", V2)";
    }
    unreadable(name + ": it is " + colour_name(s.to_move()) + "'s turn" + why);
  }
  for (std::size_t j = 0; j < action.moves.size(); ++j) {
    const PgnMove& w = action.moves[j];
    try {
      const Move m = resolve(s, w);
      const bool annotated =
          options_.verify_annotations &&
          (w.present_moved || w.created.has_value() || w.present_turn.has_value());
      if (annotated) {
        const GameState before = s;
        s.apply(m);
        check_annotations(before, s, m, w);
      } else {
        s.apply(m);
      }
    } catch (const RecordError& e) {
      std::string where = name + ", move " + std::to_string(j + 1) + " " + quoted(w.text);
      if (w.line != 0) {
        where += " (line " + std::to_string(w.line) + ", column " + std::to_string(w.column) + ")";
      }
      throw RecordError(e.fault(), where + ": " + e.what());
    }
  }
  const SubmitStatus status = s.submission_status(ruling_);
  if (status == SubmitStatus::ObligationsUnmet) {
    std::string boards;
    for (const int p : s.mandatory_timelines()) {
      if (!boards.empty()) boards += ", ";
      boards += board_text(s.multiverse(), p, s.multiverse().timeline(p)->last_half_turn());
    }
    fault(RecordFault::CannotSubmit,
          name + " ends with the present still on " + colour_name(s.to_move()) +
              "'s colour (T5): it does not move on " + boards);
  }
  if (status == SubmitStatus::Unsafe) {
    const std::optional<RoyalCapture> c = submission_witness(s);
    fault(RecordFault::CannotSubmit,
          name + " leaves a royal piece of " + colour_name(s.to_move()) +
              "'s capturable, so it cannot be submitted (C4, R1)" +
              (c ? ": " + describe(*c, s.multiverse().even()) : std::string()));
  }
  s.submit(CheckRuling::none());  // tested above
  state_ = std::move(s);
  ++played_;
}

void PgnReplay::play_all() {
  while (!finished()) play_next();
  if (record_.unreadable) unreadable(*record_.unreadable);
}

}  // namespace t5d
