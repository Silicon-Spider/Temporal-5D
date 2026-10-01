// Timeline labels <-> positions (see t5d/timeline.hpp for the mapping).

#include "t5d/timeline.hpp"

#include <string>

#include "t5d/error.hpp"

namespace t5d {

namespace {

constexpr bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

std::string quoted(std::string_view text) { return "\"" + std::string(text) + "\""; }

}  // namespace

TimelineLabel parse_timeline_label(std::string_view text) {
  TimelineLabel label;
  std::string_view digits = text;
  if (!digits.empty() && (digits.front() == '+' || digits.front() == '-')) {
    label.sign = digits.front() == '+' ? TimelineLabel::Sign::Plus : TimelineLabel::Sign::Minus;
    digits.remove_prefix(1);
  }
  if (digits.empty()) {
    throw NotationError("timeline label " + quoted(text) +
                        ": expected an optional sign and a number, such as 0, +1, -0");
  }
  for (const char c : digits) {
    if (!is_digit(c)) {
      throw NotationError("timeline label " + quoted(text) +
                          ": expected an optional sign and a number, such as 0, +1, -0");
    }
  }
  if (digits.size() > 1 && digits.front() == '0') {
    throw NotationError("timeline label " + quoted(text) + ": leading zero");
  }
  long long value = 0;
  for (const char c : digits) {
    value = value * 10 + (c - '0');
    if (value > kMaxTimelineMagnitude) {
      throw NotationError("timeline label " + quoted(text) + ": out of range (at most " +
                          std::to_string(kMaxTimelineMagnitude) + ")");
    }
  }
  label.magnitude = static_cast<int>(value);
  return label;
}

std::string to_string(const TimelineLabel& label) {
  std::string text;
  if (label.sign == TimelineLabel::Sign::Plus) text += '+';
  if (label.sign == TimelineLabel::Sign::Minus) text += '-';
  text += std::to_string(label.magnitude);
  return text;
}

std::optional<bool> implied_even(const TimelineLabel& label) noexcept {
  if (label.magnitude != 0) return std::nullopt;
  return label.sign != TimelineLabel::Sign::None;
}

int timeline_position(const TimelineLabel& label, bool even) {
  if (label.magnitude == 0) {
    if (label.sign == TimelineLabel::Sign::None) {
      if (even) {
        throw NotationError(
            "timeline \"0\" does not exist in a game that starts with an even number of "
            "timelines (its central timelines are \"-0\" and \"+0\")");
      }
      return 0;
    }
    if (!even) {
      throw NotationError("timeline " + quoted(to_string(label)) +
                          " exists only in a game that starts with an even number of timelines");
    }
    return label.sign == TimelineLabel::Sign::Plus ? 0 : -1;
  }
  if (label.sign == TimelineLabel::Sign::Minus) {
    return even ? -label.magnitude - 1 : -label.magnitude;
  }
  return label.magnitude;
}

TimelineLabel timeline_label(int position, bool even) noexcept {
  using Sign = TimelineLabel::Sign;
  const long long p = position;  // no overflow negating INT_MIN
  if (!even) {
    if (p == 0) return TimelineLabel{Sign::None, 0};
    if (p > 0) return TimelineLabel{Sign::Plus, static_cast<int>(p)};
    return TimelineLabel{Sign::Minus, static_cast<int>(-p)};
  }
  if (p >= 0) return TimelineLabel{Sign::Plus, static_cast<int>(p)};
  return TimelineLabel{Sign::Minus, static_cast<int>(-p - 1)};
}

std::string timeline_label_string(int position, bool even) {
  return to_string(timeline_label(position, even));
}

}  // namespace t5d
