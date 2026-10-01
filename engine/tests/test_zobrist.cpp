// The 5D Zobrist key (H18, spec §11.3): incremental equals from scratch,
// duplicated boards never cancel, places matter, the ruleset matters, and
// random states full of duplicates do not collide. Spec §15.2 lists
// "Incremental Zobrist = from-scratch, especially on duplicated boards" in the
// equivalence suite: these tests are that row for the engine core.

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/zobrist.hpp"
#include "test_framework.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

constexpr BoardSize k8x8{8, 8};

NewTimeline timeline_at(int position, int first_half_turn, std::vector<BoardRef> boards,
                        std::optional<Colour> creator = std::nullopt, int order = 0) {
  NewTimeline nt;
  nt.position = position;
  nt.first_half_turn = first_half_turn;
  nt.boards = std::move(boards);
  nt.creator = creator;
  nt.creation_order = order;
  return nt;
}

BoardRef board_from(const char* pieces) {
  return parse_fen_board(std::string("[") + pieces + ":0:1:w]", k8x8).board;
}

// The trap, for contrast: XOR of per-board content hashes, and XOR of
// content hash XOR place key. Both cancel duplicated boards.
std::uint64_t naive_xor(const Multiverse& mv) {
  std::uint64_t h = 0;
  mv.for_each_board([&](int, int, const Board& b) { h ^= b.content_hash(); });
  return h;
}
std::uint64_t naive_xor_with_places(const Multiverse& mv) {
  std::uint64_t h = 0;
  mv.for_each_board([&](int p, int t, const Board& b) {
    h ^= b.content_hash() ^ zobrist::coord_key(p, t);
  });
  return h;
}

}  // namespace

TEST(zobrist_key_schedule_has_distinct_nonzero_keys) {
  // Every entry of the tables (including those of codes no normalised piece
  // uses, such as an "unmoved knight").
  std::set<std::uint64_t> keys(zobrist::detail::kKeys.piece.begin(),
                               zobrist::detail::kKeys.piece.end());
  keys.insert(zobrist::detail::kKeys.en_passant.begin(), zobrist::detail::kKeys.en_passant.end());
  CHECK_EQ(keys.size(), static_cast<std::size_t>(Piece::kMaxCode * kSquareCount + kSquareCount));
  CHECK(keys.count(0) == 0);
  // Every piece a board can hold - 12 kinds x 2 colours, plus the unmoved
  // pawns, rooks, kings and brawns - has its own key on every square.
  std::set<std::uint64_t> used;
  std::set<int> codes;
  for (int code = 1; code <= Piece::kMaxCode; ++code) {
    const Piece piece = Piece::from_code(code);
    if (!codes.insert(piece.code()).second) continue;
    for (int sq = 0; sq < kSquareCount; ++sq) used.insert(zobrist::piece_key(piece, sq));
  }
  CHECK_EQ(codes.size(), static_cast<std::size_t>(32));
  CHECK_EQ(used.size(), static_cast<std::size_t>(32 * kSquareCount));
}

TEST(zobrist_key_schedule_is_pinned) {
  // The schedule is a format: a stored key means something only under the
  // schedule that made it. If this test fails, every stored key is invalid -
  // change these values only together with a format version bump. (They were
  // cross-checked against an independent model of the formulas in
  // t5d/zobrist.hpp when they were pinned. They were re-pinned once, before
  // any key was stored anywhere, when the board size became an ingredient;
  // the piece keys kept their values, since the size's stream value was
  // appended.)
  CHECK_EQ(zobrist::piece_key(Piece(PieceKind::Pawn, Colour::White), 0),
           std::uint64_t{0xb53411f2840c4156});
  CHECK_EQ(standard_start().zobrist(), std::uint64_t{0xb844b95ab1661498});
  CHECK_EQ(standard_two_timelines_start().zobrist(), std::uint64_t{0xb1adfa2912a494fa});
  // One placement on two board sizes (the notation specification's "Rook
  // Tactics I" and the same pieces on 8x8): one content hash, two keys.
  const Multiverse small = multiverse_from_fen(std::vector<std::string>{"[4k/5/5/5/K1R2:0:1:w]"},
                                               BoardSize{5, 5}, Parity::Odd);
  const Multiverse large = multiverse_from_fen(
      std::vector<std::string>{"[8/8/8/4k3/8/8/8/K1R5:0:1:w]"}, k8x8, Parity::Odd);
  CHECK_EQ(small.head(0)->content_hash(), std::uint64_t{0x8e095e4edea2aa8d});
  CHECK_EQ(large.head(0)->content_hash(), std::uint64_t{0x8e095e4edea2aa8d});
  CHECK_EQ(small.zobrist(), std::uint64_t{0x76c2a9dac75a2306});
  CHECK_EQ(large.zobrist(), std::uint64_t{0x2923642c39c889e4});
}

TEST(zobrist_board_size_is_in_the_key) {
  // A content hash is placement only, and small boards share the standard
  // ruleset (spec §12.6: structure variants), so the size must enter the key
  // itself - or a 5x5 puzzle's stored proof would be served for an 8x8
  // position (§11.3, I9).
  std::set<std::uint64_t> size_keys;
  std::set<std::uint64_t> empty_states;
  std::set<std::uint64_t> placed_states;
  for (int width = BoardSize::kMin; width <= BoardSize::kMax; ++width) {
    for (int height = BoardSize::kMin; height <= BoardSize::kMax; ++height) {
      const BoardSize size{width, height};
      size_keys.insert(zobrist::size_key(size));
      // An empty board at one place.
      Multiverse empty(size, false);
      empty.add_timeline(timeline_at(0, 2, {BoardBuilder(size).publish()}));
      empty_states.insert(empty.zobrist());
      CHECK_EQ(empty.zobrist(), zobrist::key_from_scratch(empty));
      // A king on a1, which fits every size.
      BoardBuilder king(size);
      king.put(0, 0, Piece(PieceKind::King, Colour::White, true));
      Multiverse placed(size, true, 3);
      placed.add_timeline(timeline_at(0, 2, {king.publish()}));
      placed_states.insert(placed.zobrist());
      CHECK_EQ(placed.zobrist(), zobrist::key_from_scratch(placed));
    }
  }
  CHECK_EQ(size_keys.size(), static_cast<std::size_t>(64));
  CHECK_EQ(empty_states.size(), static_cast<std::size_t>(64));
  CHECK_EQ(placed_states.size(), static_cast<std::size_t>(64));

  // The reviewers' probe: the same pieces on 5x5 and 8x8, and an empty 3x3
  // against an empty 8x8, at the same places - equal content hashes, so only
  // the size ingredient tells them apart.
  const Multiverse small = multiverse_from_fen(std::vector<std::string>{"[4k/5/5/5/K1R2:0:1:w]"},
                                               BoardSize{5, 5}, Parity::Odd);
  const Multiverse large = multiverse_from_fen(
      std::vector<std::string>{"[8/8/8/4k3/8/8/8/K1R5:0:1:w]"}, k8x8, Parity::Odd);
  CHECK_EQ(small.head(0)->content_hash(), large.head(0)->content_hash());
  CHECK(!(small == large));
  CHECK_NE(small.zobrist(), large.zobrist());
  CHECK_EQ(small.zobrist(), zobrist::key_from_scratch(small));
  CHECK_EQ(large.zobrist(), zobrist::key_from_scratch(large));
  Multiverse three(BoardSize{3, 3}, false);
  three.add_timeline(timeline_at(0, 2, {BoardBuilder(BoardSize{3, 3}).publish()}));
  Multiverse eight(k8x8, false);
  eight.add_timeline(timeline_at(0, 2, {BoardBuilder(k8x8).publish()}));
  CHECK_NE(three.zobrist(), eight.zobrist());
}

TEST(zobrist_place_keys_are_distinct) {
  std::set<std::uint64_t> keys;
  int count = 0;
  for (int p = -30; p <= 30; ++p) {
    for (int h = 0; h <= 60; ++h) {
      keys.insert(zobrist::coord_key(p, h));
      ++count;
    }
  }
  CHECK_EQ(keys.size(), static_cast<std::size_t>(count));
}

TEST(zobrist_incremental_equals_from_scratch) {
  t5d_test::Rng rng(18);
  for (int game = 0; game < 60; ++game) {
    const BoardSize size{rng.between(2, 8), rng.between(2, 8)};
    // A small pool, so that boards repeat - the XOR trap's condition.
    std::vector<BoardRef> pool;
    for (int i = 0; i < 4; ++i) pool.push_back(t5d_test::random_board(rng, size));
    const auto any = [&] { return pool[static_cast<std::size_t>(rng.below(4))]; };
    Multiverse mv(size, rng.chance(50), static_cast<std::uint32_t>(rng.below(3)));
    mv.add_timeline(timeline_at(0, rng.between(0, 5), {any()}));
    CHECK_EQ(mv.zobrist(), zobrist::key_from_scratch(mv));
    int by_white = 0;
    int by_black = 0;
    for (int step = 0; step < 80; ++step) {
      const int p = rng.between(mv.min_timeline(), mv.max_timeline());
      const int action = rng.below(10);
      if (action < 5) {
        mv.append_board(p, any());
      } else if (action < 7) {
        mv.replace_head(p, any());
      } else if (action < 9) {
        // A new timeline at either end, emanating from some existing board.
        const Timeline& from = *mv.timeline(p);
        const int h = rng.between(from.first_half_turn(), from.last_half_turn());
        const bool white = rng.chance(50);
        NewTimeline nt = timeline_at(white ? mv.max_timeline() + 1 : mv.min_timeline() - 1, h + 1,
                                     {any()}, white ? Colour::White : Colour::Black,
                                     white ? ++by_white : ++by_black);
        nt.origin = BoardCoord{p, h};
        mv.add_timeline(std::move(nt));
      } else {
        mv.set_to_move(opposite(mv.to_move()));
      }
      CHECK_EQ(mv.zobrist(), zobrist::key_from_scratch(mv));
    }
  }
}

TEST(zobrist_duplicated_boards_do_not_cancel) {
  const BoardRef x = standard_board();
  const BoardRef b = board_from("4k3/8/8/8/8/8/8/4K3");
  const BoardRef c = board_from("3k4/8/8/8/8/8/8/3K4");
  // Two timelines that both start with the same board x (the two-timeline
  // start's symmetry), then copies of b or c on top.
  const auto base = [&] {
    Multiverse mv(k8x8, true);
    mv.add_timeline(timeline_at(-1, 2, {x}));
    mv.add_timeline(timeline_at(0, 2, {x}));
    return mv;
  };
  const Multiverse none = base();
  Multiverse one = base();
  one.append_board(-1, b);
  Multiverse two = base();
  two.append_board(-1, b);
  two.append_board(0, b);
  Multiverse two_other = base();
  two_other.append_board(-1, c);
  two_other.append_board(0, c);

  // The trap is real for these states: both naive schemes alias them.
  CHECK_EQ(naive_xor(two), naive_xor(none));
  CHECK_EQ(naive_xor(two), naive_xor(two_other));
  CHECK_EQ(naive_xor_with_places(two), naive_xor_with_places(two_other));

  // The 5D key does not.
  CHECK_NE(two.zobrist(), none.zobrist());
  CHECK_NE(two.zobrist(), one.zobrist());
  CHECK_NE(one.zobrist(), none.zobrist());
  CHECK_NE(two.zobrist(), two_other.zobrist());
  // The duplicated start itself differs from duplicated empty boards.
  Multiverse empties(k8x8, true);
  const BoardRef empty = BoardBuilder(k8x8).publish();
  empties.add_timeline(timeline_at(-1, 2, {empty}));
  empties.add_timeline(timeline_at(0, 2, {empty}));
  CHECK_EQ(naive_xor(none), naive_xor(empties));
  CHECK_NE(none.zobrist(), empties.zobrist());
}

TEST(zobrist_board_terms_never_cancel_for_copies) {
  t5d_test::Rng rng(3);
  for (int i = 0; i < 3000; ++i) {
    const std::uint64_t c = rng.next();
    const std::uint64_t d = rng.next();
    const int p1 = rng.between(-20, 20);
    const int p2 = p1 + rng.between(1, 5);
    const int h = rng.between(0, 200);
    const std::uint64_t copies_of_c = zobrist::board_term(c, p1, h) + zobrist::board_term(c, p2, h);
    const std::uint64_t copies_of_d = zobrist::board_term(d, p1, h) + zobrist::board_term(d, p2, h);
    CHECK_NE(copies_of_c, std::uint64_t{0});
    CHECK_NE(copies_of_c, copies_of_d);
    CHECK_NE(zobrist::board_term(c, p1, h), zobrist::board_term(c, p2, h));
    CHECK_NE(zobrist::board_term(c, p1, h), zobrist::board_term(c, p1, h + 1));
    // Swapping the two boards' places changes the sum.
    CHECK_NE(zobrist::board_term(c, p1, h) + zobrist::board_term(d, p2, h),
             zobrist::board_term(d, p1, h) + zobrist::board_term(c, p2, h));
  }
}

TEST(zobrist_swapping_boards_changes_the_key) {
  const BoardRef x = standard_board();
  const BoardRef b = board_from("4k3/8/8/8/8/8/8/4K3");
  const BoardRef c = board_from("3k4/8/8/8/8/8/8/3K4");
  // Across timelines.
  Multiverse across(k8x8, false);
  across.add_timeline(timeline_at(0, 2, {x, b}));
  across.add_timeline(timeline_at(1, 2, {x, c}, Colour::White, 1));
  Multiverse across_swapped(k8x8, false);
  across_swapped.add_timeline(timeline_at(0, 2, {x, c}));
  across_swapped.add_timeline(timeline_at(1, 2, {x, b}, Colour::White, 1));
  CHECK_NE(across.zobrist(), across_swapped.zobrist());
  CHECK_EQ(naive_xor(across), naive_xor(across_swapped));  // what XOR would say
  // Along a timeline.
  Multiverse along(k8x8, false);
  along.add_timeline(timeline_at(0, 2, {x, b, c}));
  Multiverse along_swapped(k8x8, false);
  along_swapped.add_timeline(timeline_at(0, 2, {x, c, b}));
  CHECK_NE(along.zobrist(), along_swapped.zobrist());
  // The same boards one half-turn later.
  Multiverse later(k8x8, false);
  later.add_timeline(timeline_at(0, 3, {x, b, c}));
  CHECK_NE(along.zobrist(), later.zobrist());
}

TEST(zobrist_ruleset_changes_the_key) {
  std::set<std::uint64_t> keys;
  for (std::uint32_t ruleset = 0; ruleset < 2000; ++ruleset) {
    const Multiverse mv = multiverse_from_fen(
        std::vector<std::string>{"[" + std::string("r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/") +
                                 "P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]"},
        k8x8, Parity::Odd, ruleset);
    keys.insert(mv.zobrist());
  }
  CHECK_EQ(keys.size(), static_cast<std::size_t>(2000));
  CHECK(keys.count(0) == 0);
}

TEST(zobrist_player_parity_and_creators_are_in_the_key) {
  const Multiverse white = standard_start();
  Multiverse black = white;
  black.set_to_move(Colour::Black);
  CHECK_NE(white.zobrist(), black.zobrist());

  // The same board at the same (p, h) in an odd and in an even game.
  Multiverse odd(k8x8, false);
  odd.add_timeline(timeline_at(0, 2, {standard_board()}));
  Multiverse even(k8x8, true);
  even.add_timeline(timeline_at(0, 2, {standard_board()}));
  CHECK_NE(odd.zobrist(), even.zobrist());

  // The same boards, timeline +1 initial or created by white or by black.
  const auto with_creator = [](std::optional<Colour> creator) {
    Multiverse mv(k8x8, false);
    mv.add_timeline(timeline_at(0, 2, {standard_board()}));
    mv.add_timeline(timeline_at(1, 2, {standard_board()}, creator, creator ? 1 : 0));
    return mv.zobrist();
  };
  CHECK_NE(with_creator(std::nullopt), with_creator(Colour::White));
  CHECK_NE(with_creator(Colour::White), with_creator(Colour::Black));
  CHECK_NE(with_creator(std::nullopt), with_creator(Colour::Black));
}

TEST(zobrist_no_collisions_among_random_states_with_duplicates) {
  t5d_test::Rng rng(2026);
  const BoardSize size{4, 4};
  // Three boards only: almost every state repeats boards many times.
  std::vector<BoardRef> pool = {BoardBuilder(size).publish(), t5d_test::random_board(rng, size),
                                t5d_test::random_board(rng, size)};
  std::map<std::uint64_t, std::string> seen;
  int distinct = 0;
  for (int sample = 0; sample < 5000; ++sample) {
    Multiverse mv(size, false, static_cast<std::uint32_t>(rng.below(2)));
    const int lo = -rng.between(0, 2);
    const int hi = rng.between(0, 2);
    const auto add = [&](int p) {
      std::vector<BoardRef> boards;
      for (int k = rng.between(1, 3); k > 0; --k) {
        boards.push_back(pool[static_cast<std::size_t>(rng.below(3))]);
      }
      const std::optional<Colour> creator =
          p == 0 ? std::nullopt : std::optional<Colour>(p > 0 ? Colour::White : Colour::Black);
      mv.add_timeline(timeline_at(p, rng.between(0, 2), std::move(boards), creator,
                                  p == 0 ? 0 : (p > 0 ? p : -p)));
    };
    for (int p = 0; p <= hi; ++p) add(p);
    for (int p = -1; p >= lo; --p) add(p);
    mv.set_to_move(rng.chance(50) ? Colour::White : Colour::Black);
    const std::string identity = canonical_state_string(mv) + "|" +
                                 colour_letter(mv.to_move()) + "|" +
                                 std::to_string(mv.ruleset());
    const auto [it, inserted] = seen.emplace(mv.zobrist(), identity);
    if (inserted) {
      ++distinct;
    } else if (it->second != identity) {
      t5d_test::record_failure(__FILE__, __LINE__,
                               "key collision between " + it->second + " and " + identity);
    }
  }
  CHECK(distinct > 2000);  // the sample is really that large
}

TEST(zobrist_accumulator_updates_are_exact) {
  const std::uint64_t c1 = 0x0123456789ABCDEFull;
  const std::uint64_t c2 = 0xFEDCBA9876543210ull;
  zobrist::Accumulator a;
  a.add_board(c1, 0, 2);
  a.add_board(c1, 1, 2);
  a.add_timeline(1, Colour::White);
  CHECK_NE(a.sum(), std::uint64_t{0});
  a.remove_timeline(1, Colour::White);
  a.remove_board(c1, 0, 2);
  a.remove_board(c1, 1, 2);
  CHECK_EQ(a.sum(), std::uint64_t{0});

  zobrist::Accumulator replaced;
  replaced.add_board(c1, 3, 5);
  replaced.replace_board(c1, c2, 3, 5);
  zobrist::Accumulator direct;
  direct.add_board(c2, 3, 5);
  CHECK(replaced == direct);
  CHECK_EQ(replaced.key(Colour::White, 0, false, k8x8), direct.key(Colour::White, 0, false, k8x8));
  CHECK_NE(direct.key(Colour::White, 0, false, k8x8), direct.key(Colour::Black, 0, false, k8x8));
  CHECK_NE(direct.key(Colour::White, 0, false, k8x8), direct.key(Colour::White, 1, false, k8x8));
  CHECK_NE(direct.key(Colour::White, 0, false, k8x8), direct.key(Colour::White, 0, true, k8x8));
  CHECK_NE(direct.key(Colour::White, 0, false, k8x8),
           direct.key(Colour::White, 0, false, BoardSize{8, 7}));
  CHECK_NE(direct.key(Colour::White, 0, false, BoardSize{7, 8}),
           direct.key(Colour::White, 0, false, BoardSize{8, 7}));
}

TEST(zobrist_content_hash_is_coordinate_free) {
  // One board object on two timelines: one content hash, two board terms.
  const Multiverse mv = standard_two_timelines_start();
  CHECK(mv.head(-1) == mv.head(0));
  CHECK_EQ(mv.head(-1)->content_hash(), content_hash_from_scratch(*standard_board()));
  CHECK_NE(zobrist::board_term(mv.head(-1)->content_hash(), -1, 2),
           zobrist::board_term(mv.head(0)->content_hash(), 0, 2));
}
