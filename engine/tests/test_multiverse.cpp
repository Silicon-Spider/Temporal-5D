// The persistent multiverse: contiguous timelines, contiguous boards,
// copy-on-write sharing, queries, incremental counts.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/timeline.hpp"
#include "t5d/zobrist.hpp"
#include "test_framework.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

constexpr BoardSize k5x5{5, 5};

BoardRef board_with_king_on(int file, int rank) {
  BoardBuilder b(k5x5);
  b.put(file, rank, Piece(PieceKind::King, Colour::White));
  return b.publish();
}

NewTimeline initial(int position, int first_half_turn, std::vector<BoardRef> boards) {
  NewTimeline nt;
  nt.position = position;
  nt.first_half_turn = first_half_turn;
  nt.boards = std::move(boards);
  return nt;
}

NewTimeline created(int position, int first_half_turn, BoardRef board, Colour by, int order,
                    std::optional<BoardCoord> origin = std::nullopt) {
  NewTimeline nt = initial(position, first_half_turn, {std::move(board)});
  nt.creator = by;
  nt.creation_order = order;
  nt.origin = origin;
  return nt;
}

void move_assign(Multiverse& to, Multiverse& from) { to = std::move(from); }

}  // namespace

TEST(multiverse_starts_empty) {
  const Multiverse mv(k5x5, false, 3);
  CHECK(mv.empty());
  CHECK_EQ(mv.timeline_count(), 0);
  CHECK_EQ(mv.board_count(), static_cast<std::size_t>(0));
  CHECK_EQ(mv.piece_count(), static_cast<std::size_t>(0));
  CHECK(mv.timeline(0) == nullptr);
  CHECK(mv.board_at(0, 0) == nullptr);
  CHECK(mv.head(0) == nullptr);
  CHECK_EQ(mv.ruleset(), 3u);
  CHECK((mv.board_size() == k5x5));
  CHECK_THROWS_AS(Multiverse(BoardSize{9, 9}, false), std::invalid_argument);
}

TEST(multiverse_timelines_extend_the_range_at_either_end) {
  Multiverse mv(k5x5, false);
  mv.add_timeline(initial(3, 2, {board_with_king_on(0, 0)}));
  CHECK_EQ(mv.min_timeline(), 3);
  CHECK_EQ(mv.max_timeline(), 3);
  mv.add_timeline(initial(4, 2, {board_with_king_on(1, 0)}));
  mv.add_timeline(initial(2, 2, {board_with_king_on(2, 0)}));
  CHECK_EQ(mv.min_timeline(), 2);
  CHECK_EQ(mv.max_timeline(), 4);
  CHECK_EQ(mv.timeline_count(), 3);
  for (int p = 2; p <= 4; ++p) CHECK_EQ(mv.timeline(p)->position(), p);
  CHECK(mv.head(2)->at(2, 0).kind() == PieceKind::King);
  const Multiverse before = mv;
  CHECK_THROWS_AS(mv.add_timeline(initial(6, 2, {board_with_king_on(0, 0)})), std::invalid_argument);
  CHECK_THROWS_AS(mv.add_timeline(initial(3, 2, {board_with_king_on(0, 0)})), std::invalid_argument);
  CHECK_THROWS_AS(mv.add_timeline(initial(0, 2, {board_with_king_on(0, 0)})), std::invalid_argument);
  CHECK(mv == before);
}

TEST(multiverse_rejects_bad_timelines_and_stays_unchanged) {
  Multiverse mv(k5x5, false);
  mv.add_timeline(initial(0, 2, {board_with_king_on(0, 0)}));
  const Multiverse before = mv;
  CHECK_THROWS_AS(mv.add_timeline(initial(1, 2, {})), std::invalid_argument);
  CHECK_THROWS_AS(mv.add_timeline(initial(1, 2, {BoardRef{}})), std::invalid_argument);
  CHECK_THROWS_AS(mv.add_timeline(initial(1, 2, {standard_board()})), std::invalid_argument);
  CHECK_THROWS_AS(mv.add_timeline(initial(1, -1, {board_with_king_on(0, 0)})),
                  std::invalid_argument);
  NewTimeline no_order = initial(1, 2, {board_with_king_on(0, 0)});
  no_order.creator = Colour::White;
  CHECK_THROWS_AS(mv.add_timeline(no_order), std::invalid_argument);
  NewTimeline order_without_creator = initial(1, 2, {board_with_king_on(0, 0)});
  order_without_creator.creation_order = 1;
  CHECK_THROWS_AS(mv.add_timeline(order_without_creator), std::invalid_argument);
  CHECK_THROWS_AS(mv.add_timeline(created(1, 3, board_with_king_on(0, 0), Colour::White, 1,
                                          BoardCoord{0, 9})),
                  std::invalid_argument);
  CHECK_THROWS_AS(mv.append_board(0, BoardRef{}), std::invalid_argument);
  CHECK_THROWS_AS(mv.append_board(0, standard_board()), std::invalid_argument);
  CHECK_THROWS_AS(mv.append_board(5, board_with_king_on(0, 0)), std::invalid_argument);
  CHECK_THROWS_AS(mv.replace_head(5, board_with_king_on(0, 0)), std::invalid_argument);
  CHECK_THROWS_AS(mv.replace_head(0, BoardRef{}), std::invalid_argument);
  CHECK(mv == before);
  CHECK_EQ(mv.zobrist(), before.zobrist());
  CHECK_EQ(mv.board_count(), before.board_count());
}

TEST(multiverse_boards_are_one_per_half_turn) {
  Multiverse mv(k5x5, false);
  const BoardRef a = board_with_king_on(0, 0);
  const BoardRef b = board_with_king_on(1, 1);
  const BoardRef c = board_with_king_on(2, 2);
  mv.add_timeline(initial(0, 2, {a, b}));
  mv.append_board(0, c);
  const Timeline& t = *mv.timeline(0);
  CHECK_EQ(t.first_half_turn(), 2);
  CHECK_EQ(t.last_half_turn(), 4);
  CHECK_EQ(t.length(), 3);
  CHECK(t.board_at(2) == a.get());
  CHECK(t.board_at(3) == b.get());
  CHECK(t.board_at(4) == c.get());
  CHECK(t.board_at(1) == nullptr);
  CHECK(t.board_at(5) == nullptr);
  CHECK(t.head() == c);
  CHECK(mv.board_at(0, 3) == b.get());
  CHECK((mv.board_at(BoardCoord{0, 4}) == c.get()));
  CHECK(mv.board_ref(0, 2) == a);
  CHECK(mv.board_ref(0, 7) == nullptr);
  CHECK(mv.board_ref(1, 2) == nullptr);
  CHECK(mv.head(0) == c.get());
  CHECK_EQ(mv.board_count(), static_cast<std::size_t>(3));
  CHECK_EQ(mv.piece_count(), static_cast<std::size_t>(3));
}

TEST(multiverse_replace_head_touches_only_the_head) {
  Multiverse mv(k5x5, false);
  const BoardRef a = board_with_king_on(0, 0);
  const BoardRef b = board_with_king_on(1, 1);
  mv.add_timeline(initial(0, 2, {a, b}));
  BoardBuilder two_kings(*b);
  two_kings.put(4, 4, Piece(PieceKind::King, Colour::Black));
  const BoardRef replacement = two_kings.publish();
  mv.replace_head(0, replacement);
  CHECK(mv.board_at(0, 2) == a.get());
  CHECK(mv.board_at(0, 3) == replacement.get());
  CHECK_EQ(mv.timeline(0)->length(), 2);
  CHECK_EQ(mv.board_count(), static_cast<std::size_t>(2));
  CHECK_EQ(mv.piece_count(), static_cast<std::size_t>(3));
}

TEST(multiverse_copies_share_untouched_timelines) {
  Multiverse original(k5x5, false);
  original.add_timeline(initial(0, 2, {board_with_king_on(0, 0)}));
  original.add_timeline(created(1, 3, board_with_king_on(1, 0), Colour::White, 1, BoardCoord{0, 2}));
  original.add_timeline(created(-1, 3, board_with_king_on(2, 0), Colour::Black, 1));
  const std::uint64_t key = original.zobrist();

  Multiverse copy = original;
  for (int p = -1; p <= 1; ++p) CHECK(copy.timeline(p) == original.timeline(p));  // shared
  CHECK(copy.timeline(0)->head() == original.timeline(0)->head());

  copy.append_board(1, board_with_king_on(3, 3));
  // The copy has a new timeline object at +1 only; the original is untouched.
  CHECK(copy.timeline(1) != original.timeline(1));
  CHECK(copy.timeline(0) == original.timeline(0));
  CHECK(copy.timeline(-1) == original.timeline(-1));
  CHECK_EQ(original.timeline(1)->length(), 1);
  CHECK_EQ(copy.timeline(1)->length(), 2);
  CHECK_EQ(original.board_count(), static_cast<std::size_t>(3));
  CHECK_EQ(copy.board_count(), static_cast<std::size_t>(4));
  CHECK_EQ(original.zobrist(), key);
  CHECK_NE(copy.zobrist(), key);
  // Superseded boards are shared, not copied.
  CHECK(copy.timeline(1)->boards()[0] == original.timeline(1)->boards()[0]);

  // Editing the original afterwards does not reach the copy either.
  Multiverse second = copy;
  original.replace_head(0, board_with_king_on(4, 4));
  original.add_timeline(created(2, 3, board_with_king_on(0, 1), Colour::White, 2));
  CHECK(copy.timeline(0)->head()->at(0, 0).kind() == PieceKind::King);
  CHECK(!copy.has_timeline(2));
  CHECK(copy == second);
  CHECK_EQ(copy.zobrist(), second.zobrist());
}

TEST(multiverse_long_timelines_share_their_history) {
  // A timeline many times longer than a chunk of its persistent board sequence
  // (BoardSequence::kChunkSize): every board where it was put; snapshots taken
  // along the way keep their own boards; copies that diverge (as a search's
  // children do) share every earlier board and keep their own heads.
  t5d_test::Rng rng(5);
  constexpr int kLength = 7 * static_cast<int>(BoardSequence::kChunkSize) + 3;
  std::vector<BoardRef> made;
  made.push_back(t5d_test::random_board(rng, k5x5));
  Multiverse mv(k5x5, false);
  mv.add_timeline(initial(0, 2, {made[0]}));
  std::vector<Multiverse> snapshots;
  for (int i = 1; i < kLength; ++i) {
    if (i % 4 == 0) snapshots.push_back(mv);  // 4, 8, ... boards
    made.push_back(t5d_test::random_board(rng, k5x5));
    mv.append_board(0, made.back());
  }
  const Timeline& t = *mv.timeline(0);
  REQUIRE_EQ(t.length(), kLength);
  CHECK_EQ(t.boards().size(), static_cast<std::size_t>(kLength));
  for (int i = 0; i < kLength; ++i) {
    CHECK(t.board_at(2 + i) == made[static_cast<std::size_t>(i)].get());
    CHECK(t.boards()[static_cast<std::size_t>(i)] == made[static_cast<std::size_t>(i)]);
  }
  std::size_t k = 0;
  for (const BoardRef& b : t.boards()) CHECK(b == made[k++]);
  CHECK_EQ(k, static_cast<std::size_t>(kLength));
  CHECK(t.head() == made.back());
  CHECK_EQ(mv.board_count(), static_cast<std::size_t>(kLength));
  CHECK_EQ(mv.latest_half_turn(), 2 + kLength - 1);
  CHECK_EQ(mv.zobrist(), zobrist::key_from_scratch(mv));
  for (const Multiverse& snap : snapshots) {
    const Timeline& u = *snap.timeline(0);
    for (int i = 0; i < u.length(); ++i) {
      CHECK(u.board_at(2 + i) == made[static_cast<std::size_t>(i)].get());
    }
    CHECK_EQ(snap.latest_half_turn(), u.last_half_turn());
    CHECK_EQ(snap.zobrist(), zobrist::key_from_scratch(snap));
  }
  // Siblings: two copies of one snapshot - with a full tail (16 and 32
  // boards: the append makes a chunk) or not (20) - each appending its own
  // board, one then replacing its head.
  for (const std::size_t at : {std::size_t{3}, std::size_t{4}, std::size_t{7}}) {
    const Multiverse& parent = snapshots[at];
    Multiverse a = parent;
    Multiverse b = parent;
    const BoardRef x = t5d_test::random_board(rng, k5x5);
    const BoardRef y = t5d_test::random_board(rng, k5x5);
    a.append_board(0, x);
    b.append_board(0, y);
    CHECK(a.timeline(0)->head() == x);
    CHECK(b.timeline(0)->head() == y);
    const int len = parent.timeline(0)->length();
    CHECK_EQ(a.timeline(0)->length(), len + 1);
    for (int i = 0; i < len; ++i) {
      CHECK(a.board_at(0, 2 + i) == parent.board_at(0, 2 + i));
      CHECK(b.board_at(0, 2 + i) == parent.board_at(0, 2 + i));
    }
    b.replace_head(0, x);
    CHECK(b.timeline(0)->head() == x);
    CHECK(b == a);
    CHECK_EQ(b.zobrist(), a.zobrist());
    CHECK_EQ(a.zobrist(), zobrist::key_from_scratch(a));
    CHECK_EQ(parent.timeline(0)->length(), len);  // the parent never changes
    CHECK_EQ(parent.zobrist(), zobrist::key_from_scratch(parent));
  }
}

TEST(multiverse_latest_half_turn_is_kept) {
  Multiverse mv(k5x5, false);
  CHECK_EQ(mv.latest_half_turn(), -1);
  mv.add_timeline(initial(0, 2, {board_with_king_on(0, 0), board_with_king_on(1, 0)}));
  CHECK_EQ(mv.latest_half_turn(), 3);
  mv.add_timeline(created(1, 6, board_with_king_on(2, 0), Colour::White, 1));
  CHECK_EQ(mv.latest_half_turn(), 6);
  mv.append_board(0, board_with_king_on(3, 0));
  CHECK_EQ(mv.latest_half_turn(), 6);  // 0 reaches 4 only
  mv.append_board(1, board_with_king_on(3, 0));
  CHECK_EQ(mv.latest_half_turn(), 7);
  mv.replace_head(1, board_with_king_on(4, 0));
  CHECK_EQ(mv.latest_half_turn(), 7);
  const Multiverse copy = mv;
  CHECK_EQ(copy.latest_half_turn(), 7);
  Multiverse moved = std::move(mv);
  CHECK_EQ(moved.latest_half_turn(), 7);
  CHECK_EQ(mv.latest_half_turn(), -1);  // NOLINT(bugprone-use-after-move): specified
}

TEST(multiverse_records_creators_and_origins) {
  Multiverse mv(k5x5, false);
  mv.add_timeline(initial(0, 2, {board_with_king_on(0, 0), board_with_king_on(0, 1)}));
  mv.add_timeline(created(1, 3, board_with_king_on(1, 0), Colour::White, 1, BoardCoord{0, 2}));
  mv.add_timeline(created(-1, 4, board_with_king_on(1, 0), Colour::Black, 1, BoardCoord{0, 3}));
  mv.add_timeline(created(2, 4, board_with_king_on(1, 1), Colour::White, 2, BoardCoord{1, 3}));
  CHECK_EQ(mv.created_by(Colour::White), 2);
  CHECK_EQ(mv.created_by(Colour::Black), 1);
  CHECK(!mv.timeline(0)->creator().has_value());
  CHECK(!mv.timeline(0)->origin().has_value());
  CHECK(mv.timeline(2)->creator() == Colour::White);
  CHECK_EQ(mv.timeline(2)->creation_order(), 2);
  CHECK((mv.timeline(2)->origin() == BoardCoord{1, 3}));
  CHECK((mv.timeline(-1)->origin() == BoardCoord{0, 3}));
}

TEST(multiverse_visits_boards_in_canonical_order) {
  t5d_test::Rng rng(99);
  Multiverse mv(k5x5, true);
  mv.add_timeline(initial(0, 2, {t5d_test::random_board(rng, k5x5), t5d_test::random_board(rng, k5x5)}));
  mv.add_timeline(initial(-1, 2, {t5d_test::random_board(rng, k5x5)}));
  mv.add_timeline(created(1, 5, t5d_test::random_board(rng, k5x5), Colour::White, 1));
  mv.add_timeline(created(-2, 3, t5d_test::random_board(rng, k5x5), Colour::Black, 1));
  mv.append_board(-2, t5d_test::random_board(rng, k5x5));
  std::vector<BoardCoord> seen;
  std::size_t pieces = 0;
  mv.for_each_board([&](int p, int h, const Board& board) {
    seen.push_back(BoardCoord{p, h});
    CHECK(mv.board_at(p, h) == &board);
    pieces += static_cast<std::size_t>(board.piece_count());
  });
  const std::vector<BoardCoord> expected = {{-2, 3}, {-2, 4}, {-1, 2}, {0, 2}, {0, 3}, {1, 5}};
  CHECK(seen == expected);
  CHECK_EQ(seen.size(), mv.board_count());
  CHECK_EQ(pieces, mv.piece_count());
}

TEST(multiverse_moved_from_is_a_valid_empty_state) {
  // Move construction: the source keeps its game (size, even/odd, ruleset,
  // player to move) and loses its timelines - with counts and key to match.
  Multiverse a = standard_start();
  a.set_to_move(Colour::Black);
  const Multiverse expected = a;
  Multiverse b = std::move(a);
  CHECK(b == expected);
  CHECK_EQ(b.zobrist(), expected.zobrist());
  CHECK(a.empty());  // NOLINT(bugprone-use-after-move): the moved-from state is specified
  CHECK_EQ(a.timeline_count(), 0);
  CHECK_EQ(a.board_count(), static_cast<std::size_t>(0));
  CHECK_EQ(a.piece_count(), static_cast<std::size_t>(0));
  CHECK_EQ(a.created_by(Colour::White), 0);
  CHECK_EQ(a.zobrist(), zobrist::key_from_scratch(a));
  CHECK((a.board_size() == BoardSize{8, 8}));
  CHECK(a.to_move() == Colour::Black);
  Multiverse fresh(BoardSize{8, 8}, false);
  fresh.set_to_move(Colour::Black);
  CHECK(a == fresh);
  CHECK_EQ(a.zobrist(), fresh.zobrist());
  // Reusing it builds a state like any other.
  NewTimeline nt;
  nt.position = 0;
  nt.first_half_turn = 2;
  nt.boards = {standard_board()};
  a.add_timeline(nt);
  a.set_to_move(Colour::White);
  CHECK(a == standard_start());
  CHECK_EQ(a.board_count(), static_cast<std::size_t>(1));
  CHECK_EQ(a.piece_count(), static_cast<std::size_t>(32));
  CHECK_EQ(a.zobrist(), standard_start().zobrist());
  CHECK_EQ(a.zobrist(), zobrist::key_from_scratch(a));

  // Move assignment, onto a non-empty state of another game.
  Multiverse c(k5x5, true, 9);
  c.add_timeline(initial(-1, 2, {board_with_king_on(0, 0)}));
  Multiverse d = standard_two_timelines_start();
  d = std::move(c);
  CHECK((d.board_size() == k5x5));
  CHECK(d.even());
  CHECK_EQ(d.ruleset(), 9u);
  CHECK_EQ(d.board_count(), static_cast<std::size_t>(1));
  CHECK_EQ(d.zobrist(), zobrist::key_from_scratch(d));
  CHECK(c.empty());  // NOLINT(bugprone-use-after-move)
  CHECK_EQ(c.board_count(), static_cast<std::size_t>(0));
  CHECK_EQ(c.piece_count(), static_cast<std::size_t>(0));
  CHECK_EQ(c.zobrist(), zobrist::key_from_scratch(c));
  CHECK(c == Multiverse(k5x5, true, 9));
  c.add_timeline(initial(4, 2, {board_with_king_on(1, 1)}));  // any position: it is empty
  CHECK_EQ(c.min_timeline(), 4);
  CHECK_EQ(c.board_count(), static_cast<std::size_t>(1));
  CHECK_EQ(c.zobrist(), zobrist::key_from_scratch(c));

  // Self-move-assignment leaves the state as it was. (Through a helper, so that
  // no compiler's -Wself-move sees it.)
  Multiverse e = standard_two_timelines_start();
  move_assign(e, e);
  CHECK(e == standard_two_timelines_start());
  CHECK_EQ(e.board_count(), static_cast<std::size_t>(2));
  CHECK_EQ(e.zobrist(), zobrist::key_from_scratch(e));
}

TEST(multiverse_timeline_positions_stay_writable) {
  // Positions are bounded by the label they are written with (at most
  // kMaxTimelineMagnitude), so every multiverse's canonical string reads back:
  // -M..M in an odd game, -M-1..M in an even one (where "-M" is p = -M-1).
  constexpr int M = kMaxTimelineMagnitude;
  const BoardSize tiny{1, 1};
  const auto lone = [&](bool even, int position) {
    Multiverse mv(tiny, even);
    NewTimeline nt;
    nt.position = position;
    nt.first_half_turn = 2;
    nt.boards = {BoardBuilder(tiny).publish()};
    nt.creator = position >= 0 ? Colour::White : Colour::Black;
    nt.creation_order = 1;
    mv.add_timeline(nt);
    return mv;
  };
  CHECK_EQ(min_timeline_position(false), -M);
  CHECK_EQ(min_timeline_position(true), -M - 1);
  CHECK_EQ(kMaxTimelinePosition, M);
  CHECK_THROWS_AS(lone(false, -M - 1), std::invalid_argument);
  CHECK_THROWS_AS(lone(true, -M - 2), std::invalid_argument);
  CHECK_THROWS_AS(lone(false, M + 1), std::invalid_argument);
  CHECK_THROWS_AS(lone(true, M + 1), std::invalid_argument);
  for (const bool even : {false, true}) {
    for (const int position : {min_timeline_position(even), kMaxTimelinePosition}) {
      const Multiverse mv = lone(even, position);
      const std::string text = canonical_state_string(mv);
      const Multiverse back = multiverse_from_fen(std::vector<std::string>{text}, tiny,
                                                  even ? Parity::Even : Parity::Odd);
      CHECK_EQ(canonical_state_string(back), text);
      CHECK_EQ(back.min_timeline(), position);
    }
  }
  CHECK_EQ(canonical_state_string(lone(false, -M)), std::string("[1:-1048576:1:w]"));
  CHECK_EQ(canonical_state_string(lone(true, -M - 1)), std::string("[1:-1048576:1:w]"));
  // The range also binds a timeline added next to an outermost one.
  Multiverse edge = lone(false, -M);
  NewTimeline beyond;
  beyond.position = -M - 1;
  beyond.first_half_turn = 2;
  beyond.boards = {BoardBuilder(tiny).publish()};
  beyond.creator = Colour::Black;
  beyond.creation_order = 2;
  CHECK_THROWS_AS(edge.add_timeline(beyond), std::invalid_argument);
  CHECK_EQ(edge.timeline_count(), 1);
}

TEST(multiverse_equality_is_by_value) {
  Multiverse a(k5x5, false);
  Multiverse b(k5x5, false);
  a.add_timeline(initial(0, 2, {board_with_king_on(0, 0)}));
  b.add_timeline(initial(0, 2, {board_with_king_on(0, 0)}));  // equal content, other object
  CHECK(a == b);
  b.set_to_move(Colour::Black);
  CHECK(!(a == b));
  b.set_to_move(Colour::White);
  b.append_board(0, board_with_king_on(1, 0));
  CHECK(!(a == b));
  CHECK(!(a == Multiverse(k5x5, true)));
  CHECK(!(Multiverse(k5x5, false, 1) == Multiverse(k5x5, false, 2)));
}
