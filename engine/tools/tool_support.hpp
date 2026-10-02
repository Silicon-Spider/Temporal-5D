// tools/tool_support.hpp - what t5d-measure and t5d-bench share: reading
// records, the ruleset options of their command lines, record replay into
// positions.

#ifndef T5D_TOOLS_TOOL_SUPPORT_HPP
#define T5D_TOOLS_TOOL_SUPPORT_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "t5d/pgn.hpp"
#include "t5d/rules.hpp"

namespace t5d::tools {

/// The whole of a file, in binary (CR-LF kept: the reader treats CR as
/// whitespace). Throws std::runtime_error if it cannot be read.
std::string read_file(const std::string& path);

/// The whole of standard input, in binary.
std::string read_stdin();

/// A count or encoding: decimal, or hexadecimal after "0x".
std::optional<std::uint64_t> parse_number(std::string_view text) noexcept;

/// The ruleset options both tools take. Returns how many arguments it used at
/// argv[i] (0: not a ruleset option); throws std::invalid_argument for a
/// malformed one.
///
///   --moves-after-shift   rules reference Q4's other reading: moves may be
///                         made after the present has shifted, until the
///                         action is submitted (RuleOptions::moves_after_shift)
///   --rules ENCODING      any ruleset, as RuleOptions' canonical encoding
///                         (RuleOptions::encode; bit 3 is --moves-after-shift)
int ruleset_option(int argc, char** argv, int i, RuleOptions& rules);

/// "standard", or the non-default choices (t5d::describe).
std::string ruleset_text(const RuleOptions& rules);

/// Whether a record was made by the synthetic generator (its "Synthetic"
/// header): such positions are for throughput only, never the |E| verdict.
bool is_synthetic(const PgnRecord& record);

/// Makes `replay` replay as the gate adapter's parse mode does: every move
/// pseudo-legal and allowed by the turn rules, the present shifted at each
/// submission (T5), royal safety (C4) not tested - which is how synthetic
/// records were played. PgnReplay tests royal safety unless told otherwise
/// (set_check_ruling(CheckRuling::none())).
void parse_mode(PgnReplay& replay);

}  // namespace t5d::tools

#endif  // T5D_TOOLS_TOOL_SUPPORT_HPP
