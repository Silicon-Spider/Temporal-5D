// t5d/zobrist.hpp - the 5D Zobrist key (decision H18; design spec §11.3, "The
// trap: XOR-Zobrist cancels in 5D").
//
// The standard multiverse hash - the XOR of per-board Zobrist hashes - aliases
// whenever two boards have the same contents, because a ^ a = 0, and in 5D that
// is guaranteed: every travel move makes a near-copy of a board on a new
// timeline, and early boards are identical across timelines by symmetry. The
// scheme here, as §11.3 prescribes:
//
//  1. Each board's CONTENT is hashed coordinate-free: the XOR of one key per
//     (piece, square) - piece meaning kind, colour and unmoved flag - plus an
//     en-passant key when the board has an en-passant square. Squares on one
//     board are distinct, so XOR is safe inside a board, and a move updates the
//     content hash incrementally (BoardBuilder does). In the multiverse key a
//     board's en-passant term counts only while the board is its timeline's
//     head (keyed_content_hash): en passant is taken from the board the double
//     step created (S7), and only a playable board is ever moved from (L5), so
//     on a history board the right has lapsed and no rule can read it.
//  2. Each board's content hash is MIXED with its (timeline, half-turn) key by
//     multiply-rotate rounds and a final avalanche (board_term). Not XOR: the
//     mix is a bijection in the content for a fixed place and in the place for
//     a fixed content, so two copies of one board on two timelines give two
//     unrelated values.
//  3. The mixed values are COMBINED by 64-bit addition - commutative (no board
//     order to fix) and never cancelling (x + x = 2x, not 0). Adding, removing
//     or replacing a board is one addition or subtraction (Accumulator).
//  4. The FINAL key folds in, by more multiply-rotate rounds, the ruleset ID
//     (§11.3 and I9: one more ingredient of every exact-store key), the colour
//     to move, the even/odd start and the board size. Each round is a
//     bijection in the accumulated sum, so finalisation adds no collisions of
//     its own; and for a fixed sum each round is a bijection in its
//     ingredient, so two states that differ only in one ingredient never
//     share a key.
//
// Also in the sum, one term per timeline: its position and creator (none for
// an initial timeline). The rules of stage E2 read the creators (timeline
// activity), so two states that differ only there must not share a key. The
// spec is silent on this ingredient; see the engine README.
//
// Why the board size: a content hash is placement only (square indices are
// rank * 8 + file on every size), so one placement on a 5x5 and on an 8x8
// board has one content hash - yet the two are different games, since the
// size bounds every move. Small boards are structure variants under the
// standard rules (§12.6), so they share the standard ruleset ID, and without
// this ingredient an exact store keyed by this hash would serve a 5x5
// puzzle's proof for an 8x8 position - the cross-variant contamination §11.3
// adds the ruleset ID to prevent. (A board's content hash alone therefore does
// not identify a board across sizes; the multiverse key does.)
//
// Every key comes from one fixed-seed SplitMix64 stream, computed at compile
// time. The key schedule is a FORMAT: a key stored anywhere (a solved store,
// an asset map) is only meaningful under the schedule that made it, so changing
// the seed or the order of the stream changes every key (a test pins one). New
// ingredients take values from the END of the stream, so that the keys already
// drawn keep their values.

#ifndef T5D_ZOBRIST_HPP
#define T5D_ZOBRIST_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "t5d/types.hpp"

namespace t5d {

class Multiverse;

namespace zobrist {

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// SplitMix64: the next value of the stream whose state is `state`.
constexpr std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
  std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/// A full 64-bit avalanche (the MurmurHash3 finaliser). A bijection.
constexpr std::uint64_t avalanche(std::uint64_t x) noexcept {
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDull;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ull;
  x ^= x >> 33;
  return x;
}

/// One multiply-rotate round: rotl(acc + input * P2, 31) * P1, with P1 and P2
/// odd. For a fixed input it is a bijection in acc, and for a fixed acc a
/// bijection in input.
constexpr std::uint64_t mr_round(std::uint64_t acc, std::uint64_t input) noexcept {
  constexpr std::uint64_t kP1 = 0x9E3779B185EBCA87ull;
  constexpr std::uint64_t kP2 = 0xC2B2AE3D27D4EB4Full;
  return std::rotl(acc + input * kP2, 31) * kP1;
}

// ---------------------------------------------------------------------------
// The key schedule
// ---------------------------------------------------------------------------

/// The stream's seed: the ASCII bytes of "t5d-zob1".
inline constexpr std::uint64_t kKeySeed = 0x7435642D7A6F6231ull;

namespace detail {

inline constexpr int kPieceKeyCount = Piece::kMaxCode * kSquareCount;  // 48 * 64

struct KeySchedule {
  std::array<std::uint64_t, kPieceKeyCount> piece{};  // [(code - 1) * 64 + square]
  std::array<std::uint64_t, kSquareCount> en_passant{};
  std::array<std::uint64_t, 4> side{};     // [even * 2 + colour to move]
  std::array<std::uint64_t, 3> creator{};  // [0 none, 1 white, 2 black]
  std::uint64_t coord_salt = 0;
  std::uint64_t board_seed = 0;
  std::uint64_t timeline_seed = 0;
  std::uint64_t ruleset_salt = 0;
  std::uint64_t final_seed = 0;
  std::uint64_t size_salt = 0;
  // Appended for t5d/state.hpp: activity under rules where it depends on the
  // history of play (GameState::zobrist).
  std::uint64_t activity_seed = 0;
  std::uint64_t inactive_salt = 0;
  std::uint64_t deferred_salt = 0;
  std::uint64_t last_creator_salt = 0;
};

constexpr KeySchedule make_key_schedule() noexcept {
  KeySchedule k;
  std::uint64_t state = kKeySeed;
  for (auto& key : k.piece) key = splitmix64_next(state);
  for (auto& key : k.en_passant) key = splitmix64_next(state);
  for (auto& key : k.side) key = splitmix64_next(state);
  for (auto& key : k.creator) key = splitmix64_next(state);
  k.coord_salt = splitmix64_next(state);
  k.board_seed = splitmix64_next(state);
  k.timeline_seed = splitmix64_next(state);
  k.ruleset_salt = splitmix64_next(state);
  k.final_seed = splitmix64_next(state);
  k.size_salt = splitmix64_next(state);  // appended: the keys above are unchanged
  k.activity_seed = splitmix64_next(state);      // appended (activity): the keys above
  k.inactive_salt = splitmix64_next(state);      // are unchanged
  k.deferred_salt = splitmix64_next(state);
  k.last_creator_salt = splitmix64_next(state);
  return k;
}

inline constexpr KeySchedule kKeys = make_key_schedule();

}  // namespace detail

// ---------------------------------------------------------------------------
// Board content (coordinate-free)
// ---------------------------------------------------------------------------

/// The key of a (non-empty) piece on a square, 0 <= square < 64. A board's
/// content hash is the XOR of these over its pieces.
constexpr std::uint64_t piece_key(Piece piece, int square) noexcept {
  return detail::kKeys.piece[static_cast<std::size_t>((piece.code() - 1) * kSquareCount + square)];
}

/// The key of an en-passant square (stage E2 sets the field; a board without
/// one contributes nothing).
constexpr std::uint64_t en_passant_key(int square) noexcept {
  return detail::kKeys.en_passant[static_cast<std::size_t>(square)];
}

/// The content hash a board contributes to the multiverse key at its place:
/// the board's own content hash while it is its timeline's head, and the same
/// without the en-passant term once it is a history board. The right is used
/// only by a move FROM the board the double step created (S7), and moves start
/// only from playable boards (L5), so a history board's en-passant square can
/// never matter again; leaving it out keeps the key of a state independent of
/// lapsed rights - as the 5DFEN round trip, which cannot record them, needs.
constexpr std::uint64_t keyed_content_hash(std::uint64_t content_hash, int en_passant,
                                           bool head) noexcept {
  return (head || en_passant == kNoSquare) ? content_hash
                                           : content_hash ^ en_passant_key(en_passant);
}

// ---------------------------------------------------------------------------
// Placing boards and timelines
// ---------------------------------------------------------------------------

/// The key of a place (timeline, half-turn). A bijection of the pair (as two
/// 32-bit words), so two places never share a key.
constexpr std::uint64_t coord_key(int timeline, int half_turn) noexcept {
  const std::uint64_t packed =
      (std::uint64_t{static_cast<std::uint32_t>(timeline)} << 32) |
      std::uint64_t{static_cast<std::uint32_t>(half_turn)};
  return avalanche(packed ^ detail::kKeys.coord_salt);
}

/// A board's term in the multiverse sum: its content hash mixed with its place
/// by multiply-rotate rounds and an avalanche.
constexpr std::uint64_t board_term(std::uint64_t content_hash, int timeline,
                                   int half_turn) noexcept {
  const std::uint64_t x = mr_round(detail::kKeys.board_seed, content_hash);
  return avalanche(mr_round(x, coord_key(timeline, half_turn)));
}

/// A timeline's term in the multiverse sum: its position and creator.
constexpr std::uint64_t timeline_term(int timeline, std::optional<Colour> creator) noexcept {
  const std::size_t c = creator ? 1 + static_cast<std::size_t>(index_of(*creator)) : 0;
  const std::uint64_t x = mr_round(detail::kKeys.timeline_seed,
                                   std::uint64_t{static_cast<std::uint32_t>(timeline)});
  return avalanche(mr_round(x, detail::kKeys.creator[c]));
}

/// The ruleset's ingredient. A bijection of the ID.
constexpr std::uint64_t ruleset_key(std::uint32_t ruleset) noexcept {
  return avalanche(std::uint64_t{ruleset} ^ detail::kKeys.ruleset_salt);
}

/// The board size's ingredient: a bijection of (width, height) packed as
/// width << 8 | height, so no two sizes share a key.
constexpr std::uint64_t size_key(BoardSize size) noexcept {
  const std::uint64_t packed = (std::uint64_t{static_cast<std::uint8_t>(size.width)} << 8) |
                               std::uint64_t{static_cast<std::uint8_t>(size.height)};
  return avalanche(packed ^ detail::kKeys.size_salt);
}

/// The key of a multiverse whose terms sum to `sum`.
constexpr std::uint64_t finalize(std::uint64_t sum, Colour to_move, std::uint32_t ruleset,
                                 bool even, BoardSize size) noexcept {
  std::uint64_t x = mr_round(detail::kKeys.final_seed, sum);
  x = mr_round(x, ruleset_key(ruleset));
  x = mr_round(x, detail::kKeys.side[static_cast<std::size_t>((even ? 2 : 0) + index_of(to_move))]);
  x = mr_round(x, size_key(size));
  return avalanche(x);
}

// ---------------------------------------------------------------------------
// Incremental maintenance
// ---------------------------------------------------------------------------

/// The running sum of a multiverse's board and timeline terms. Every update is
/// O(1); wrap-around is intended (arithmetic modulo 2^64).
class Accumulator {
 public:
  constexpr void add_board(std::uint64_t content_hash, int timeline, int half_turn) noexcept {
    sum_ += board_term(content_hash, timeline, half_turn);
  }
  constexpr void remove_board(std::uint64_t content_hash, int timeline, int half_turn) noexcept {
    sum_ -= board_term(content_hash, timeline, half_turn);
  }
  /// The board at (timeline, half_turn) replaced by one with another content
  /// (a head replaced), or the same board counted with another hash (a head
  /// becoming a history board: keyed_content_hash).
  constexpr void replace_board(std::uint64_t old_content_hash, std::uint64_t new_content_hash,
                               int timeline, int half_turn) noexcept {
    sum_ += board_term(new_content_hash, timeline, half_turn) -
            board_term(old_content_hash, timeline, half_turn);
  }
  constexpr void add_timeline(int timeline, std::optional<Colour> creator) noexcept {
    sum_ += timeline_term(timeline, creator);
  }
  constexpr void remove_timeline(int timeline, std::optional<Colour> creator) noexcept {
    sum_ -= timeline_term(timeline, creator);
  }

  constexpr std::uint64_t sum() const noexcept { return sum_; }

  constexpr std::uint64_t key(Colour to_move, std::uint32_t ruleset, bool even,
                              BoardSize size) const noexcept {
    return finalize(sum_, to_move, ruleset, even, size);
  }

  friend constexpr bool operator==(const Accumulator&, const Accumulator&) noexcept = default;

 private:
  std::uint64_t sum_ = 0;
};

/// The key of a multiverse computed from scratch: every board's content hash
/// recomputed from its squares (keyed_content_hash: the en-passant term on
/// head boards only), every term summed afresh. Tests hold the incremental key
/// (Multiverse::zobrist) to it.
std::uint64_t key_from_scratch(const Multiverse& multiverse);

}  // namespace zobrist
}  // namespace t5d

#endif  // T5D_ZOBRIST_HPP
