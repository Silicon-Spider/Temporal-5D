// The canonical fingerprint (H30): the notation specification's full-state
// hash, its two published examples, and the board order.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/md5.hpp"
#include "test_framework.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

const std::string kStandardPieces =
    "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*";
constexpr BoardSize k5x5{5, 5};

std::vector<std::string> shuffled(std::vector<std::string> v, std::uint64_t seed) {
  t5d_test::Rng rng(seed);
  for (std::size_t i = v.size(); i > 1; --i) {
    std::swap(v[i - 1], v[static_cast<std::size_t>(rng.below(static_cast<int>(i)))]);
  }
  return v;
}

std::string concatenated(const std::vector<std::string>& parts) {
  std::string out;
  for (const std::string& part : parts) out += part;
  return out;
}

}  // namespace

TEST(fingerprint_standard_matches_the_published_example) {
  // Notation specification, "Hashing", "Examples": the Standard variant.
  const Multiverse mv = standard_start();
  const std::string expected_string = "[" + kStandardPieces + ":0:1:w]";
  CHECK_EQ(canonical_state_string(mv), expected_string);
  const Fingerprint f = fingerprint(mv);
  CHECK_EQ(f.fingerprint, std::string("d574889fd9da3f2bc65249ff27249b00"));
  CHECK_EQ(f.placement, md5_hex("[rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR:0:1:w]"));
  CHECK_EQ(f.placement, std::string("2b8fa7fec7e4d7f2ecade5e01de08d0a"));
  CHECK_EQ(f.boards, static_cast<std::size_t>(1));
  CHECK_EQ(f.pieces, static_cast<std::size_t>(32));
  // "both the single board and full state have the same hash"
  CHECK_EQ(md5_hex(fen_board_string(*standard_board(), 0, 2, false)), f.fingerprint);
}

TEST(fingerprint_two_timelines_matches_the_published_example) {
  // Notation specification, "Hashing", "Examples": Standard - Two Timeline.
  const Multiverse mv = standard_two_timelines_start();
  const std::string expected_string =
      "[" + kStandardPieces + ":-0:1:w][" + kStandardPieces + ":+0:1:w]";
  CHECK_EQ(canonical_state_string(mv), expected_string);
  const Fingerprint f = fingerprint(mv);
  CHECK_EQ(f.fingerprint, std::string("3672761404ffcd15ae644c75401812be"));
  CHECK_EQ(f.placement, std::string("bf8a1d35c48ba3b8aed6dd69f39c46cc"));
  CHECK_EQ(f.boards, static_cast<std::size_t>(2));
  CHECK_EQ(f.pieces, static_cast<std::size_t>(64));
}

TEST(fingerprint_orders_by_timeline_then_turn_then_colour_even_game) {
  // An even game: "-1" (p -2), "-0" (p -1), "+0" (p 0), "+1" (p 1). Within a
  // timeline: by turn, white's board before black's.
  const std::vector<std::string> in_order = {
      "[4k/5/5/5/K4:-1:2:w]", "[4k/5/5/5/1K3:-1:2:b]",                         // p -2
      "[4k/5/5/5/2K2:-0:1:w]", "[4k/5/5/5/3K1:-0:1:b]", "[4k/5/5/5/4K:-0:2:w]",  // p -1
      "[3k1/5/5/5/K4:+0:1:w]", "[3k1/5/5/5/1K3:+0:1:b]",                       // p 0
      "[2k2/5/5/5/K4:+1:1:b]", "[2k2/5/5/5/1K3:+1:2:w]",                       // p 1
  };
  for (std::uint64_t seed = 1; seed <= 5; ++seed) {
    const Multiverse mv = multiverse_from_fen(shuffled(in_order, seed), k5x5, Parity::Infer);
    CHECK_EQ(canonical_state_string(mv), concatenated(in_order));
    const Fingerprint f = fingerprint(mv);
    CHECK_EQ(f.fingerprint, md5_hex(concatenated(in_order)));
    CHECK_EQ(f.boards, static_cast<std::size_t>(9));
    CHECK_EQ(f.pieces, static_cast<std::size_t>(18));
  }
}

TEST(fingerprint_orders_by_timeline_then_turn_then_colour_odd_game) {
  const std::vector<std::string> in_order = {
      "[4k/5/5/5/K4:-2:3:w]",
      "[4k/5/5/5/1K3:-1:2:b]", "[4k/5/5/5/2K2:-1:3:w]",
      "[4k/5/5/5/3K1:0:1:w]", "[4k/5/5/5/4K:0:1:b]", "[3k1/5/5/5/4K:0:2:w]",
      "[2k2/5/5/5/K4:+1:2:b]",
      "[1k3/5/5/5/K4:+2:3:w]",
  };
  const Multiverse mv = multiverse_from_fen(shuffled(in_order, 11), k5x5, Parity::Infer);
  CHECK_EQ(canonical_state_string(mv), concatenated(in_order));
  CHECK_EQ(fingerprint(mv).boards, static_cast<std::size_t>(8));
}

TEST(fingerprint_counts_superseded_boards) {
  Multiverse mv = standard_start();
  BoardBuilder b(*standard_board());
  b.put(make_square(4, 1), Piece{});
  b.put(make_square(4, 3), Piece(PieceKind::Pawn, Colour::White));
  mv.append_board(0, b.publish());
  const Fingerprint f = fingerprint(mv);
  CHECK_EQ(f.boards, static_cast<std::size_t>(2));
  CHECK_EQ(f.pieces, static_cast<std::size_t>(64));
  CHECK_EQ(canonical_state_string(mv),
           "[" + kStandardPieces + ":0:1:w][r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/4P3/8/"
           "P*P*P*P*1P*P*P*/R*NBQK*BNR*:0:1:b]");
}

TEST(fingerprint_placement_ignores_only_unmoved_markers) {
  // Same placement, different markers: the fingerprints differ, the
  // placements agree.
  const Multiverse marked = multiverse_from_fen(
      std::vector<std::string>{"[4k*/5/5/5/K*1R*2:0:1:w]"}, k5x5, Parity::Infer);
  const Multiverse unmarked = multiverse_from_fen(
      std::vector<std::string>{"[4k/5/5/5/K1R2:0:1:w]"}, k5x5, Parity::Infer);
  const Fingerprint a = fingerprint(marked);
  const Fingerprint b = fingerprint(unmarked);
  CHECK_NE(a.fingerprint, b.fingerprint);
  CHECK_EQ(a.placement, b.placement);
  // Without any marker the two hashes coincide.
  CHECK_EQ(b.placement, b.fingerprint);
}

TEST(fingerprint_has_no_whitespace_and_only_canonical_fields) {
  t5d_test::Rng rng(30);
  for (int round = 0; round < 50; ++round) {
    const BoardSize size = t5d_test::random_size(rng);
    Multiverse mv(size, rng.chance(50));
    NewTimeline nt;
    nt.position = 0;
    nt.first_half_turn = rng.between(0, 9);
    for (int k = rng.between(1, 4); k > 0; --k) nt.boards.push_back(t5d_test::random_board(rng, size));
    mv.add_timeline(std::move(nt));
    const std::string canonical = canonical_state_string(mv);
    CHECK(canonical.find_first_of(" \t\r\n\v\f") == std::string::npos);
    for (std::size_t i = 0; i < canonical.size(); ++i) {
      // '+' only as a label's sign; '*' only after a pawn, rook, king or brawn.
      if (canonical[i] == '+') CHECK(i > 0 && canonical[i - 1] == ':');
      if (canonical[i] == '*') {
        CHECK(i > 0 && std::string("PRKWprkw").find(canonical[i - 1]) != std::string::npos);
      }
    }
    // Round trip: the boards read back from the canonical string give the
    // same canonical string.
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start < canonical.size()) {
      const std::size_t end = canonical.find(']', start);
      parts.push_back(canonical.substr(start, end - start + 1));
      start = end + 1;
    }
    const Multiverse back = multiverse_from_fen(parts, size, mv.even() ? Parity::Even : Parity::Odd);
    CHECK_EQ(canonical_state_string(back), canonical);
    CHECK((fingerprint(back) == fingerprint(mv)));
  }
}
