// The C API: handles, fingerprints, enumeration, errors, ownership.

#include <cstdint>
#include <string>
#include <string_view>

#include "t5d/capi.h"
#include "t5d/fen.hpp"
#include "test_framework.hpp"

namespace {

const std::string kStandard =
    "[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]";

// Owns a state; destroys it at scope exit.
struct State {
  t5d_state* p = nullptr;
  State() = default;
  State(const State&) = delete;
  State& operator=(const State&) = delete;
  ~State() { t5d_state_destroy(p); }
};

// Owns a library string; frees it at scope exit.
struct Text {
  char* p = nullptr;
  Text() = default;
  Text(const Text&) = delete;
  Text& operator=(const Text&) = delete;
  ~Text() { t5d_free(p); }
  std::string str() const { return p != nullptr ? std::string(p) : std::string("<null>"); }
};

bool contains(const Text& text, std::string_view part) {
  return text.p != nullptr && std::string_view(text.p).find(part) != std::string_view::npos;
}

}  // namespace

TEST(capi_version_is_set) {
  CHECK(t5d_version() != nullptr);
  CHECK(!std::string(t5d_version()).empty());
}

TEST(capi_standard_state) {
  State s;
  REQUIRE_EQ(t5d_state_new_standard(&s.p), T5D_OK);
  REQUIRE(s.p != nullptr);
  t5d_fingerprint f{};
  CHECK_EQ(t5d_state_fingerprint(s.p, &f), T5D_OK);
  CHECK_EQ(std::string(f.fingerprint), std::string("d574889fd9da3f2bc65249ff27249b00"));
  CHECK_EQ(std::string(f.placement), std::string("2b8fa7fec7e4d7f2ecade5e01de08d0a"));
  CHECK_EQ(f.boards, std::uint64_t{1});
  CHECK_EQ(f.pieces, std::uint64_t{32});
  Text canonical;
  CHECK_EQ(t5d_state_canonical(s.p, &canonical.p), T5D_OK);
  CHECK_EQ(canonical.str(), kStandard);
  std::uint64_t key = 0;
  CHECK_EQ(t5d_state_zobrist(s.p, &key), T5D_OK);
  CHECK_EQ(key, t5d::standard_start().zobrist());
  t5d_info info{};
  CHECK_EQ(t5d_state_info(s.p, &info), T5D_OK);
  CHECK_EQ(info.boards, std::uint64_t{1});
  CHECK_EQ(info.pieces, std::uint64_t{32});
  CHECK_EQ(info.width, 8);
  CHECK_EQ(info.height, 8);
  CHECK_EQ(info.even, 0);
  CHECK_EQ(info.to_move, 0);
  CHECK_EQ(info.min_timeline, 0);
  CHECK_EQ(info.max_timeline, 0);
  CHECK_EQ(info.created_by_white, 0);
  CHECK_EQ(info.created_by_black, 0);
  CHECK_EQ(info.ruleset, 0u);
}

TEST(capi_two_timelines_and_enumeration) {
  State s;
  REQUIRE_EQ(t5d_state_new_standard_two_timelines(&s.p), T5D_OK);
  t5d_fingerprint f{};
  CHECK_EQ(t5d_state_fingerprint(s.p, &f), T5D_OK);
  CHECK_EQ(std::string(f.fingerprint), std::string("3672761404ffcd15ae644c75401812be"));
  size_t count = 0;
  CHECK_EQ(t5d_state_board_count(s.p, &count), T5D_OK);
  CHECK_EQ(count, static_cast<size_t>(2));
  const char* labels[] = {"-0", "+0"};
  for (size_t i = 0; i < count; ++i) {
    int32_t timeline = 99;
    int32_t half_turn = 99;
    Text fen;
    CHECK_EQ(t5d_state_board(s.p, i, &timeline, &half_turn, &fen.p), T5D_OK);
    CHECK_EQ(timeline, static_cast<int32_t>(i) - 1);
    CHECK_EQ(half_turn, 2);
    CHECK(contains(fen, std::string(":") + labels[i] + ":1:w]"));
  }
  Text none;
  CHECK_EQ(t5d_state_board(s.p, 2, nullptr, nullptr, &none.p), T5D_ERR_ARGUMENT);
  CHECK(none.p == nullptr);
  CHECK_EQ(t5d_state_board(s.p, 1, nullptr, nullptr, nullptr), T5D_OK);
  t5d_info info{};
  CHECK_EQ(t5d_state_info(s.p, &info), T5D_OK);
  CHECK_EQ(info.even, 1);
  CHECK_EQ(info.min_timeline, -1);
  CHECK_EQ(info.max_timeline, 0);
}

TEST(capi_enumerates_boards_in_canonical_order) {
  // Nine boards on four timelines of an even game, given out of order.
  const char* boards[] = {
      "[2k2/5/5/5/1K3:+1:2:w]", "[4k/5/5/5/K4:-1:2:w]",   "[3k1/5/5/5/1K3:+0:1:b]",
      "[4k/5/5/5/2K2:-0:1:w]",  "[4k/5/5/5/4K:-0:2:w]",   "[2k2/5/5/5/K4:+1:1:b]",
      "[4k/5/5/5/1K3:-1:2:b]",  "[3k1/5/5/5/K4:+0:1:w]",  "[4k/5/5/5/3K1:-0:1:b]",
  };
  State s;
  Text error;
  REQUIRE_EQ(t5d_state_new_from_fen(boards, 9, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, 0, &s.p,
                                    &error.p),
             T5D_OK);
  size_t count = 0;
  CHECK_EQ(t5d_state_board_count(s.p, &count), T5D_OK);
  REQUIRE_EQ(count, static_cast<size_t>(9));
  const int32_t expected[9][2] = {{-2, 4}, {-2, 5}, {-1, 2}, {-1, 3}, {-1, 4},
                                  {0, 2},  {0, 3},  {1, 3},  {1, 4}};
  std::string joined;
  for (size_t i = 0; i < count; ++i) {
    int32_t timeline = 0;
    int32_t half_turn = 0;
    Text fen;
    CHECK_EQ(t5d_state_board(s.p, i, &timeline, &half_turn, &fen.p), T5D_OK);
    CHECK_EQ(timeline, expected[i][0]);
    CHECK_EQ(half_turn, expected[i][1]);
    joined += fen.str();
  }
  Text canonical;
  CHECK_EQ(t5d_state_canonical(s.p, &canonical.p), T5D_OK);
  CHECK_EQ(joined, canonical.str());
  t5d_info info{};
  CHECK_EQ(t5d_state_info(s.p, &info), T5D_OK);
  CHECK_EQ(info.min_timeline, -2);
  CHECK_EQ(info.max_timeline, 1);
  CHECK_EQ(info.created_by_white, 1);
  CHECK_EQ(info.created_by_black, 1);
  // Heads at h = 5, 4, 3, 4: the earliest is "+0"'s T1 board, on an initial
  // timeline, black to move.
  CHECK_EQ(info.to_move, 1);
}

TEST(capi_player_to_move_given_or_inferred) {
  // White has created +1 and +2 and black none; +2's head (T3, black to move)
  // is the earliest, but whether it is the present depends on timeline
  // activity (stage E2), so INFER refuses and a given player is taken.
  const char* boards[] = {
      "[4k/5/5/5/K1R2:0:4:w]",  "[4k/5/5/5/K1R2:0:4:b]",  "[4k/5/5/5/K1R2:0:5:w]",
      "[4k/5/5/5/K1R2:+1:4:b]", "[4k/5/5/5/K1R2:+1:5:w]", "[4k/5/5/5/K1R2:+2:3:b]",
  };
  {
    int sentinel = 0;
    t5d_state* out = reinterpret_cast<t5d_state*>(&sentinel);
    Text error;
    CHECK_EQ(t5d_state_new_from_fen(boards, 6, 5, 5, T5D_PARITY_ODD, T5D_TO_MOVE_INFER, 0, &out,
                                    &error.p),
             T5D_ERR_ARGUMENT);
    CHECK(out == nullptr);
    CHECK(contains(error, "cannot tell whose turn it is"));
    CHECK(contains(error, "give the player to move"));
  }
  State white;
  REQUIRE_EQ(t5d_state_new_from_fen(boards, 6, 5, 5, T5D_PARITY_ODD, T5D_TO_MOVE_WHITE, 0, &white.p,
                                    nullptr),
             T5D_OK);
  t5d_info info{};
  CHECK_EQ(t5d_state_info(white.p, &info), T5D_OK);
  CHECK_EQ(info.to_move, 0);
  CHECK_EQ(info.created_by_white, 2);
  CHECK_EQ(info.created_by_black, 0);
  State black;
  REQUIRE_EQ(t5d_state_new_from_fen(boards, 6, 5, 5, T5D_PARITY_ODD, T5D_TO_MOVE_BLACK, 0, &black.p,
                                    nullptr),
             T5D_OK);
  CHECK_EQ(t5d_state_info(black.p, &info), T5D_OK);
  CHECK_EQ(info.to_move, 1);
  // The player to move feeds the key, not the fingerprint.
  std::uint64_t white_key = 0;
  std::uint64_t black_key = 0;
  CHECK_EQ(t5d_state_zobrist(white.p, &white_key), T5D_OK);
  CHECK_EQ(t5d_state_zobrist(black.p, &black_key), T5D_OK);
  CHECK_NE(white_key, black_key);
  t5d_fingerprint fw{};
  t5d_fingerprint fb{};
  CHECK_EQ(t5d_state_fingerprint(white.p, &fw), T5D_OK);
  CHECK_EQ(t5d_state_fingerprint(black.p, &fb), T5D_OK);
  CHECK_EQ(std::string(fw.fingerprint), std::string(fb.fingerprint));
}

TEST(capi_from_fen_matches_the_built_in_start) {
  const char* boards[] = {kStandard.c_str()};
  State parsed;
  Text error;
  REQUIRE_EQ(t5d_state_new_from_fen(boards, 1, 8, 8, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, 0,
                                    &parsed.p, &error.p),
             T5D_OK);
  CHECK(error.p == nullptr);
  State inferred_size;
  REQUIRE_EQ(t5d_state_new_from_fen(boards, 1, 0, 0, T5D_PARITY_ODD, T5D_TO_MOVE_WHITE, 0,
                                    &inferred_size.p, nullptr),
             T5D_OK);
  State built;
  REQUIRE_EQ(t5d_state_new_standard(&built.p), T5D_OK);
  std::uint64_t k1 = 1;
  std::uint64_t k2 = 2;
  std::uint64_t k3 = 3;
  CHECK_EQ(t5d_state_zobrist(parsed.p, &k1), T5D_OK);
  CHECK_EQ(t5d_state_zobrist(inferred_size.p, &k2), T5D_OK);
  CHECK_EQ(t5d_state_zobrist(built.p, &k3), T5D_OK);
  CHECK_EQ(k1, k3);
  CHECK_EQ(k2, k3);
  State other_rules;
  REQUIRE_EQ(t5d_state_new_from_fen(boards, 1, 8, 8, T5D_PARITY_ODD, T5D_TO_MOVE_INFER, 5,
                                    &other_rules.p, nullptr),
             T5D_OK);
  std::uint64_t k4 = 0;
  CHECK_EQ(t5d_state_zobrist(other_rules.p, &k4), T5D_OK);
  CHECK_NE(k4, k3);
  t5d_info info{};
  CHECK_EQ(t5d_state_info(other_rules.p, &info), T5D_OK);
  CHECK_EQ(info.ruleset, 5u);
}

TEST(capi_reports_errors_without_throwing) {
  int sentinel = 0;
  t5d_state* const not_null = reinterpret_cast<t5d_state*>(&sentinel);
  const char* bad[] = {"[4x/5/5/5/K1R2:0:1:w]"};
  {
    t5d_state* out = not_null;
    Text error;
    CHECK_EQ(t5d_state_new_from_fen(bad, 1, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, 0, &out,
                                    &error.p),
             T5D_ERR_NOTATION);
    CHECK(out == nullptr);
    CHECK(contains(error, "unknown piece letter 'x'"));
  }
  {
    t5d_state* out = not_null;
    CHECK_EQ(t5d_state_new_from_fen(bad, 1, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, 0, &out,
                                    nullptr),
             T5D_ERR_NOTATION);
    CHECK(out == nullptr);
  }
  const char* good[] = {"[4k/5/5/5/K1R2:0:1:w]"};
  const char* with_null[] = {"[4k/5/5/5/K1R2:0:1:w]", nullptr};
  const struct {
    const char* const* boards;
    size_t count;
    int32_t width, height, parity, to_move;
    t5d_status expected;
    const char* message;
  } cases[] = {
      {nullptr, 1, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, T5D_ERR_ARGUMENT, "boards is NULL"},
      {with_null, 2, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, T5D_ERR_ARGUMENT, "#2 is NULL"},
      {good, 1, 5, 5, 7, T5D_TO_MOVE_INFER, T5D_ERR_ARGUMENT, "parity"},
      {good, 1, 5, 5, T5D_PARITY_INFER, 2, T5D_ERR_ARGUMENT, "to_move"},
      {good, 1, 5, 5, T5D_PARITY_INFER, -2, T5D_ERR_ARGUMENT, "to_move"},
      {good, 1, 9, 9, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, T5D_ERR_ARGUMENT, "outside"},
      {good, 1, 5, 0, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, T5D_ERR_ARGUMENT, "outside"},
      {good, 1, 8, 8, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, T5D_ERR_NOTATION, "rows"},
      {good, 1, 5, 5, T5D_PARITY_EVEN, T5D_TO_MOVE_INFER, T5D_ERR_NOTATION, "\"0\""},
      {good, 0, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, T5D_ERR_NOTATION, "no board strings"},
      {good, 0, 0, 0, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, T5D_ERR_NOTATION, "no board strings"},
  };
  for (const auto& c : cases) {
    t5d_state* out = not_null;
    Text error;
    CHECK_EQ(t5d_state_new_from_fen(c.boards, c.count, c.width, c.height, c.parity, c.to_move, 0,
                                    &out, &error.p),
             c.expected);
    CHECK(out == nullptr);
    if (!contains(error, c.message)) {
      t5d_test::record_failure(__FILE__, __LINE__,
                               "message \"" + error.str() + "\" lacks \"" + c.message + "\"");
    }
  }
  Text error;
  CHECK_EQ(t5d_state_new_from_fen(good, 1, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, 0, nullptr,
                                  &error.p),
           T5D_ERR_ARGUMENT);
  CHECK(contains(error, "out is NULL"));
}

TEST(capi_copies_are_independent) {
  State original;
  REQUIRE_EQ(t5d_state_new_standard_two_timelines(&original.p), T5D_OK);
  State copy;
  REQUIRE_EQ(t5d_state_copy(original.p, &copy.p), T5D_OK);
  std::uint64_t before = 0;
  CHECK_EQ(t5d_state_zobrist(original.p, &before), T5D_OK);
  t5d_state_destroy(original.p);
  original.p = nullptr;
  t5d_fingerprint f{};
  CHECK_EQ(t5d_state_fingerprint(copy.p, &f), T5D_OK);
  CHECK_EQ(std::string(f.fingerprint), std::string("3672761404ffcd15ae644c75401812be"));
  std::uint64_t after = 0;
  CHECK_EQ(t5d_state_zobrist(copy.p, &after), T5D_OK);
  CHECK_EQ(after, before);
}

TEST(capi_null_arguments) {
  t5d_free(nullptr);
  t5d_state_destroy(nullptr);
  CHECK_EQ(t5d_state_new_standard(nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_state_new_standard_two_timelines(nullptr), T5D_ERR_ARGUMENT);
  t5d_state* out = nullptr;
  CHECK_EQ(t5d_state_copy(nullptr, &out), T5D_ERR_ARGUMENT);
  CHECK(out == nullptr);
  State s;
  REQUIRE_EQ(t5d_state_new_standard(&s.p), T5D_OK);
  CHECK_EQ(t5d_state_copy(s.p, nullptr), T5D_ERR_ARGUMENT);
  t5d_info info{};
  CHECK_EQ(t5d_state_info(nullptr, &info), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_state_info(s.p, nullptr), T5D_ERR_ARGUMENT);
  t5d_fingerprint f{};
  CHECK_EQ(t5d_state_fingerprint(nullptr, &f), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_state_fingerprint(s.p, nullptr), T5D_ERR_ARGUMENT);
  char* text = nullptr;
  CHECK_EQ(t5d_state_canonical(nullptr, &text), T5D_ERR_ARGUMENT);
  CHECK(text == nullptr);
  CHECK_EQ(t5d_state_canonical(s.p, nullptr), T5D_ERR_ARGUMENT);
  std::uint64_t key = 0;
  CHECK_EQ(t5d_state_zobrist(nullptr, &key), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_state_zobrist(s.p, nullptr), T5D_ERR_ARGUMENT);
  size_t count = 0;
  CHECK_EQ(t5d_state_board_count(nullptr, &count), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_state_board_count(s.p, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_state_board(nullptr, 0, nullptr, nullptr, nullptr), T5D_ERR_ARGUMENT);
}
