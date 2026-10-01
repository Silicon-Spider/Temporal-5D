// Timeline labels and positions, in odd and even games.

#include <string>

#include "t5d/error.hpp"
#include "t5d/timeline.hpp"
#include "test_framework.hpp"

using namespace t5d;
using Sign = TimelineLabel::Sign;

TEST(labels_parse) {
  CHECK(parse_timeline_label("0") == (TimelineLabel{Sign::None, 0}));
  CHECK(parse_timeline_label("-0") == (TimelineLabel{Sign::Minus, 0}));
  CHECK(parse_timeline_label("+0") == (TimelineLabel{Sign::Plus, 0}));
  CHECK(parse_timeline_label("+12") == (TimelineLabel{Sign::Plus, 12}));
  CHECK(parse_timeline_label("-3") == (TimelineLabel{Sign::Minus, 3}));
  CHECK(parse_timeline_label("7") == (TimelineLabel{Sign::None, 7}));
  for (const char* text : {"0", "-0", "+0", "+12", "-3", "7", "+1048576"}) {
    CHECK_EQ(to_string(parse_timeline_label(text)), std::string(text));
  }
}

TEST(malformed_labels_are_rejected) {
  for (const char* bad : {"", "+", "-", "++1", "+-1", "01", "+01", "-00", "1a", " 1", "1 ",
                          "L1", "+1x", "1048577", "-99999999999999999999", "0x1", "１"}) {
    CHECK_THROWS_AS(parse_timeline_label(bad), NotationError);
  }
  CHECK_THROWS_WITH(parse_timeline_label("+01"), NotationError, "leading zero");
  CHECK_THROWS_WITH(parse_timeline_label("+x"), NotationError, "\"+x\"");
}

TEST(labels_imply_the_kind_of_game) {
  CHECK(implied_even(parse_timeline_label("-0")) == std::optional<bool>(true));
  CHECK(implied_even(parse_timeline_label("+0")) == std::optional<bool>(true));
  CHECK(implied_even(parse_timeline_label("0")) == std::optional<bool>(false));
  CHECK(!implied_even(parse_timeline_label("+1")).has_value());
  CHECK(!implied_even(parse_timeline_label("-1")).has_value());
  CHECK(!implied_even(parse_timeline_label("4")).has_value());
}

TEST(odd_game_positions) {
  CHECK_EQ(timeline_position(parse_timeline_label("0"), false), 0);
  CHECK_EQ(timeline_position(parse_timeline_label("+1"), false), 1);
  CHECK_EQ(timeline_position(parse_timeline_label("-1"), false), -1);
  CHECK_EQ(timeline_position(parse_timeline_label("+5"), false), 5);
  CHECK_EQ(timeline_position(parse_timeline_label("-5"), false), -5);
  CHECK_EQ(timeline_position(parse_timeline_label("3"), false), 3);
  CHECK_THROWS_AS(timeline_position(parse_timeline_label("+0"), false), NotationError);
  CHECK_THROWS_AS(timeline_position(parse_timeline_label("-0"), false), NotationError);
}

TEST(even_game_positions) {
  CHECK_EQ(timeline_position(parse_timeline_label("+0"), true), 0);
  CHECK_EQ(timeline_position(parse_timeline_label("-0"), true), -1);
  CHECK_EQ(timeline_position(parse_timeline_label("+1"), true), 1);
  CHECK_EQ(timeline_position(parse_timeline_label("-1"), true), -2);
  CHECK_EQ(timeline_position(parse_timeline_label("+5"), true), 5);
  CHECK_EQ(timeline_position(parse_timeline_label("-5"), true), -6);
  CHECK_EQ(timeline_position(parse_timeline_label("1"), true), 1);
  CHECK_THROWS_AS(timeline_position(parse_timeline_label("0"), true), NotationError);
}

TEST(canonical_labels) {
  CHECK_EQ(timeline_label_string(0, false), std::string("0"));
  CHECK_EQ(timeline_label_string(1, false), std::string("+1"));
  CHECK_EQ(timeline_label_string(-1, false), std::string("-1"));
  CHECK_EQ(timeline_label_string(17, false), std::string("+17"));
  CHECK_EQ(timeline_label_string(0, true), std::string("+0"));
  CHECK_EQ(timeline_label_string(-1, true), std::string("-0"));
  CHECK_EQ(timeline_label_string(-2, true), std::string("-1"));
  CHECK_EQ(timeline_label_string(1, true), std::string("+1"));
  CHECK_EQ(timeline_label_string(-17, true), std::string("-16"));
}

TEST(labels_and_positions_round_trip) {
  for (const bool even : {false, true}) {
    for (int p = -60; p <= 60; ++p) {
      const TimelineLabel label = timeline_label(p, even);
      CHECK_EQ(timeline_position(label, even), p);
      CHECK(parse_timeline_label(to_string(label)) == label);
      // Every canonical label but the odd game's "0" carries a sign.
      CHECK_EQ(label.sign == Sign::None, !even && p == 0);
    }
  }
}

TEST(even_game_labels_are_adjacent_positions) {
  const char* labels[] = {"-2", "-1", "-0", "+0", "+1", "+2"};
  for (int i = 0; i < 6; ++i) {
    CHECK_EQ(timeline_position(parse_timeline_label(labels[i]), true), i - 3);
  }
}

TEST(central_timelines) {
  CHECK(is_central_timeline(0, false));
  CHECK(!is_central_timeline(-1, false));
  CHECK(!is_central_timeline(1, false));
  CHECK(is_central_timeline(0, true));
  CHECK(is_central_timeline(-1, true));
  CHECK(!is_central_timeline(1, true));
  CHECK(!is_central_timeline(-2, true));
}
