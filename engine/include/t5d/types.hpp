// t5d/types.hpp - colours, piece kinds, pieces, squares, board sizes, half-turns.
//
// The vocabulary of the rules engine (design spec §11.0). Conventions used by
// every other header:
//
//  * Squares. index = rank * 8 + file on every board size, a1 = 0. A board of
//    width W and height H uses files 0..W-1 and ranks 0..H-1, 1 <= W, H <= 8
//    (the variants use 5x5 to 8x8). One 64-bit word therefore holds the
//    bitboard of any board, and a square's index never depends on the size.
//  * Half-turns. h = 2*T + (0 if white is to move on the board, 1 if black),
//    with T the turn as displayed (0 on turn-zero boards). h orders the boards
//    of a timeline; its parity is the colour to move on each board.
//  * Timelines. An integer position p; see t5d/timeline.hpp for the
//    notation's labels ("0", "+1", "-0", ...).
//  * Piece letters are the notation specification's (5dpgn, "Moves" and
//    "5DFEN and custom variants"): white upper case, black lower case.

#ifndef T5D_TYPES_HPP
#define T5D_TYPES_HPP

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace t5d {

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

enum class Colour : std::uint8_t { White = 0, Black = 1 };

constexpr Colour opposite(Colour c) noexcept {
  return c == Colour::White ? Colour::Black : Colour::White;
}

constexpr int index_of(Colour c) noexcept { return static_cast<int>(c); }

/// 'w' or 'b', as in a 5DFEN board string.
constexpr char colour_letter(Colour c) noexcept { return c == Colour::White ? 'w' : 'b'; }

// ---------------------------------------------------------------------------
// Piece kinds
// ---------------------------------------------------------------------------

/// The twelve kinds the notation specification names, in its letters:
/// P pawn, N knight, B bishop, R rook, Q queen, K king, U unicorn, D dragon,
/// S princess, W brawn, C common king (not royal), Y royal queen.
enum class PieceKind : std::uint8_t {
  Pawn = 0,
  Knight,
  Bishop,
  Rook,
  Queen,
  King,
  Unicorn,
  Dragon,
  Princess,
  Brawn,
  CommonKing,
  RoyalQueen,
};

inline constexpr int kPieceKindCount = 12;

constexpr int index_of(PieceKind k) noexcept { return static_cast<int>(k); }

/// Royal pieces: a check is a move that captures one (spec §11.0 item 1).
/// The king, and the royal queen of Reversed Royalty (§12.6).
constexpr bool is_royal(PieceKind k) noexcept {
  return k == PieceKind::King || k == PieceKind::RoyalQueen;
}

/// Kinds whose moves depend on whether they have moved yet (castling, the
/// double step), and so carry the "unmoved" flag. This is the notation
/// specification's list ("5DFEN Strictness"): pawn, rook, king, brawn. Every
/// other kind never carries the flag (see Piece).
constexpr bool tracks_unmoved(PieceKind k) noexcept {
  return k == PieceKind::Pawn || k == PieceKind::Rook || k == PieceKind::King ||
         k == PieceKind::Brawn;
}

/// The notation letter of a kind, upper case.
constexpr char letter_of(PieceKind k) noexcept {
  constexpr char kLetters[] = "PNBRQKUDSWCY";
  return kLetters[index_of(k)];
}

/// The kind a letter names, in either case; nullopt for any other character.
constexpr std::optional<PieceKind> kind_from_letter(char c) noexcept {
  switch (c) {
    case 'P': case 'p': return PieceKind::Pawn;
    case 'N': case 'n': return PieceKind::Knight;
    case 'B': case 'b': return PieceKind::Bishop;
    case 'R': case 'r': return PieceKind::Rook;
    case 'Q': case 'q': return PieceKind::Queen;
    case 'K': case 'k': return PieceKind::King;
    case 'U': case 'u': return PieceKind::Unicorn;
    case 'D': case 'd': return PieceKind::Dragon;
    case 'S': case 's': return PieceKind::Princess;
    case 'W': case 'w': return PieceKind::Brawn;
    case 'C': case 'c': return PieceKind::CommonKing;
    case 'Y': case 'y': return PieceKind::RoyalQueen;
    default: return std::nullopt;
  }
}

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

/// A piece on a square, or no piece: kind, colour and the unmoved flag packed
/// into one byte. Code 0 is the empty square; otherwise
///   code = 1 + (kind << 2 | colour << 1 | unmoved),   1 <= code <= 48,
/// so code - 1 indexes per-piece tables directly (the Zobrist keys use it).
///
/// The unmoved flag is kept only on kinds that track it (tracks_unmoved):
/// constructing an "unmoved queen" yields a plain queen. A flag on any other
/// kind has no meaning (the notation specification's 5DFEN grammar says it
/// "can safely be ignored"), and dropping it at construction keeps equal
/// positions equal - in equality, in hashes and in the canonical fingerprint.
class Piece {
 public:
  constexpr Piece() noexcept = default;

  constexpr Piece(PieceKind kind, Colour colour, bool unmoved = false) noexcept
      : code_(static_cast<std::uint8_t>(
            1 + ((index_of(kind) << 2) | (index_of(colour) << 1) |
                 ((unmoved && tracks_unmoved(kind)) ? 1 : 0)))) {}

  /// The piece a code describes, normalised like any other piece (the code of
  /// an "unmoved knight" gives a knight); codes outside 1..48 give the empty
  /// square.
  static constexpr Piece from_code(int code) noexcept {
    if (code < 1 || code > kMaxCode) return Piece{};
    const int v = code - 1;
    return Piece(static_cast<PieceKind>(v >> 2), static_cast<Colour>((v >> 1) & 1),
                 (v & 1) != 0);
  }

  static constexpr int kMaxCode = 4 * kPieceKindCount;  // 48

  constexpr bool empty() const noexcept { return code_ == 0; }

  /// Precondition for kind(), colour(): !empty().
  constexpr PieceKind kind() const noexcept {
    return static_cast<PieceKind>((code_ - 1) >> 2);
  }
  constexpr Colour colour() const noexcept {
    return static_cast<Colour>(((code_ - 1) >> 1) & 1);
  }
  constexpr bool unmoved() const noexcept { return code_ != 0 && ((code_ - 1) & 1) != 0; }
  constexpr bool royal() const noexcept { return code_ != 0 && is_royal(kind()); }

  /// The same piece with the unmoved flag cleared.
  constexpr Piece as_moved() const noexcept {
    return empty() ? Piece{} : Piece(kind(), colour(), false);
  }

  /// The packed code, 0 for the empty square.
  constexpr int code() const noexcept { return code_; }

  /// The notation letter: upper case for white, lower case for black. '.' for
  /// the empty square. The unmoved marker is not part of the letter.
  constexpr char letter() const noexcept {
    if (empty()) return '.';
    const char upper = letter_of(kind());
    return colour() == Colour::White ? upper : static_cast<char>(upper - 'A' + 'a');
  }

  friend constexpr bool operator==(Piece, Piece) noexcept = default;

 private:
  std::uint8_t code_ = 0;
};

// ---------------------------------------------------------------------------
// Squares and board sizes
// ---------------------------------------------------------------------------

inline constexpr int kNoSquare = -1;
inline constexpr int kSquareCount = 64;

constexpr int make_square(int file, int rank) noexcept { return rank * 8 + file; }
constexpr int file_of(int square) noexcept { return square & 7; }
constexpr int rank_of(int square) noexcept { return square >> 3; }
constexpr std::uint64_t square_bit(int square) noexcept {
  return std::uint64_t{1} << square;
}

/// "e4"-style name of a square (files a.., ranks 1..).
std::string square_name(int square);

/// Width and height of every board in a game (the 5dpgn "Size" header, "WxH").
struct BoardSize {
  static constexpr int kMin = 1;
  static constexpr int kMax = 8;

  int width = 8;
  int height = 8;

  constexpr bool valid() const noexcept {
    return width >= kMin && width <= kMax && height >= kMin && height <= kMax;
  }
  constexpr bool contains(int file, int rank) const noexcept {
    return file >= 0 && file < width && rank >= 0 && rank < height;
  }
  constexpr bool contains_square(int square) const noexcept {
    return square >= 0 && square < kSquareCount && contains(file_of(square), rank_of(square));
  }
  /// The bitboard of every square on a board of this size.
  constexpr std::uint64_t mask() const noexcept {
    if (!valid()) return 0;
    const std::uint64_t row = (std::uint64_t{1} << width) - 1;
    std::uint64_t m = 0;
    for (int r = 0; r < height; ++r) m |= row << (8 * r);
    return m;
  }
  constexpr int square_count() const noexcept { return width * height; }

  friend constexpr bool operator==(const BoardSize&, const BoardSize&) noexcept = default;
};

/// "WxH", as in the Size header.
std::string to_string(const BoardSize& size);

/// Parses "WxH" (the Size header's value). nullopt if malformed or out of range.
std::optional<BoardSize> parse_board_size(std::string_view text) noexcept;

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

/// The largest turn number accepted from notation. Far above any real game;
/// it keeps every half-turn index well inside an int.
inline constexpr int kMaxTurn = 1 << 20;

/// h = 2*T + (0 for white to move, 1 for black to move).
constexpr int half_turn(int turn, Colour to_move) noexcept {
  return 2 * turn + index_of(to_move);
}
/// The turn T, as displayed, of the board at half-turn h (h >= 0).
constexpr int turn_of(int h) noexcept { return h >> 1; }
/// The colour to move on the board at half-turn h (h >= 0).
constexpr Colour to_move_at(int h) noexcept {
  return (h & 1) != 0 ? Colour::Black : Colour::White;
}

/// A board's place in the multiverse: timeline position p, half-turn h.
/// Ordered by timeline, then half-turn: the canonical board order.
struct BoardCoord {
  int timeline = 0;
  int half_turn = 0;

  friend constexpr auto operator<=>(const BoardCoord&, const BoardCoord&) noexcept = default;
};

}  // namespace t5d

#endif  // T5D_TYPES_HPP
