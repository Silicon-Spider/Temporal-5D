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

namespace {

// PgnReplay::set_check_ruling exists from stage E4 on: call it where it does.
template <class Replay>
auto set_no_ruling(Replay& replay, int) -> decltype(replay.set_check_ruling(CheckRuling{}), void()) {
  replay.set_check_ruling(CheckRuling{});
}
template <class Replay>
void set_no_ruling(Replay&, long) {}

}  // namespace

void parse_mode(PgnReplay& replay) { set_no_ruling(replay, 0); }

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
  if (std::strcmp(arg, "--moves-after-shift") == 0) {
    rules.moves_after_shift = true;
    return 1;
  }
  if (std::strcmp(arg, "--rules") == 0) {
    if (i + 1 >= argc) throw std::invalid_argument("--rules needs an encoding");
    const std::optional<std::uint64_t> e = parse_number(argv[i + 1]);
    if (!e) throw std::invalid_argument(std::string("--rules: not a number: ") + argv[i + 1]);
    // Combined with any option given before it.
    const RuleOptions given = RuleOptions::decode(*e);
    const bool after_shift = rules.moves_after_shift || given.moves_after_shift;
    rules = given;
    rules.moves_after_shift = after_shift;
    rules.validate();
    return 2;
  }
  return 0;
}

std::string ruleset_text(const RuleOptions& rules) { return describe(rules); }

bool is_synthetic(const PgnRecord& record) { return record.header("Synthetic") != nullptr; }

}  // namespace t5d::tools
