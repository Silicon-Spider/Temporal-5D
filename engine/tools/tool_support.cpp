// What t5d-measure and t5d-bench share (see tool_support.hpp).

#include "tool_support.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace t5d::tools {

void parse_mode(PgnReplay& replay) { replay.set_check_ruling(CheckRuling::none()); }

PgnOptions sixb_pgn_options() {
  PgnOptions o;
  o.capture_marks = CaptureMarks::MarkedMustCapture;  // the reading before H51
  return o;
}

std::string read_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) throw std::runtime_error("cannot open " + path);
  std::string text;
  char buffer[1 << 16];
  for (;;) {
    const std::size_t n = std::fread(buffer, 1, sizeof buffer, f);
    text.append(buffer, n);
    if (n < sizeof buffer) break;
  }
  const bool failed = std::ferror(f) != 0;
  std::fclose(f);
  if (failed) throw std::runtime_error("cannot read " + path);
  return text;
}

std::string read_stdin() {
#if defined(_WIN32)
  _setmode(_fileno(stdin), _O_BINARY);
#endif
  std::string text;
  char buffer[1 << 16];
  for (;;) {
    const std::size_t n = std::fread(buffer, 1, sizeof buffer, stdin);
    text.append(buffer, n);
    if (n < sizeof buffer) break;
  }
  if (std::ferror(stdin) != 0) throw std::runtime_error("cannot read standard input");
  return text;
}

std::optional<std::uint64_t> parse_number(std::string_view text) noexcept {
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    base = 16;
    text.remove_prefix(2);
  }
  if (text.empty() || text.size() > 20) return std::nullopt;
  std::uint64_t value = 0;
  for (const char c : text) {
    int digit = 0;
    if (c >= '0' && c <= '9') {
      digit = c - '0';
    } else if (base == 16 && c >= 'a' && c <= 'f') {
      digit = c - 'a' + 10;
    } else if (base == 16 && c >= 'A' && c <= 'F') {
      digit = c - 'A' + 10;
    } else {
      return std::nullopt;
    }
    if (digit >= base) return std::nullopt;
    const auto b = static_cast<std::uint64_t>(base);
    const auto d = static_cast<std::uint64_t>(digit);
    if (value > (UINT64_MAX - d) / b) return std::nullopt;
    value = value * b + d;
  }
  return value;
}

int ruleset_option(int argc, char** argv, int i, RuleOptions& rules) {
  const char* arg = argv[i];
  if (std::strcmp(arg, "--no-moves-after-shift") == 0) {
    rules.moves_after_shift = false;
    return 1;
  }
  if (std::strcmp(arg, "--moves-after-shift") == 0) {
    return 1;  // the default since R8: accepted, so that old command lines still run
  }
  if (std::strcmp(arg, "--rules") == 0) {
    if (i + 1 >= argc) throw std::invalid_argument("--rules needs an encoding");
    const std::optional<std::uint64_t> e = parse_number(argv[i + 1]);
    if (!e) throw std::invalid_argument(std::string("--rules: not a number: ") + argv[i + 1]);
    // Combined with a --no-moves-after-shift given before it (the default is
    // true, so either one asking for the Guide-literal reading decides).
    const RuleOptions given = RuleOptions::decode(*e);
    const bool after_shift = rules.moves_after_shift && given.moves_after_shift;
    rules = given;
    rules.moves_after_shift = after_shift;
    rules.validate();
    return 2;
  }
  return 0;
}

std::string ruleset_text(const RuleOptions& rules) { return describe(rules); }

std::string absolute_rules_text(const RuleOptions& r) {
  std::string out;
  const auto add = [&out](const char* name, const std::string& value) {
    if (!out.empty()) out += ' ';
    out += name;
    out += '=';
    out += value;
  };
  const auto b = [](bool v) { return std::string(v ? "true" : "false"); };
  add("activation", r.activation == ActivationRule::Count ? "Count" : "Consecutive");
  add("initial_credit", r.initial_credit == InitialTimelineCredit::None ? "None" : "MinusZeroToBlack");
  add("play_activates", b(r.play_activates));
  add("moves_after_shift", b(r.moves_after_shift));
  add("mandatory", r.mandatory == MandatoryBoards::InPresent ? "InPresent" : "AllActive");
  add("reactivation", r.reactivation == Reactivation::Rewinds ? "Rewinds" : "Deferred");
  add("turn_zero_reachable", b(r.turn_zero_reachable));
  add("sliders_cross_void", b(r.sliders_cross_void));
  add("past_self_blocks", b(r.past_self_blocks));
  add("earlier_playable_jump_branches", b(r.earlier_playable_jump_branches));
  add("inactive_history_is_destination", b(r.inactive_history_is_destination));
  add("brawn_rank_timeline_capture", b(r.brawn_rank_timeline_capture));
  add("double_step_right", r.double_step_right == DoubleStepRight::UnmovedFlag ? "UnmovedFlag" : "HomeRank");
  add("timeline_double_step", b(r.timeline_double_step));
  add("double_step_over_void", b(r.double_step_over_void));
  add("branch_copies_keep_unmoved", b(r.branch_copies_keep_unmoved));
  add("jumps_keep_unmoved", b(r.jumps_keep_unmoved));
  add("en_passant_brawns", b(r.en_passant_brawns));
  add("brawns_promote", b(r.brawns_promote));
  add("brawn_cross_board_promotion", b(r.brawn_cross_board_promotion));
  char mask[16];
  std::snprintf(mask, sizeof mask, "0x%04x", static_cast<unsigned>(r.promotions.mask()));
  add("promotions", mask);
  add("castling_empty_between", b(r.castling_empty_between));
  add("castling_destination_test", b(r.castling_destination_test));
  add("castling_rook_path_test", b(r.castling_rook_path_test));
  add("castling_any_rank", b(r.castling_any_rank));
  add("castling_rook_two_away", b(r.castling_rook_two_away));
  add("castling_on_optional_boards", b(r.castling_on_optional_boards));
  add("family", r.family == RuleFamily::Standard ? "Standard" : "ReversedRoyalty");
  return out;
}

bool is_synthetic(const PgnRecord& record) { return record.header("Synthetic") != nullptr; }

}  // namespace t5d::tools
