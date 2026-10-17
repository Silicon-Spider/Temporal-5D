// 6b's roots, shared by the 6b'' tool and Hunt's ground truth (stage E9):
// the turn starts of the 5dpgn records given (replayed as the gate adapter's
// parse mode does, under the ruleset given), every K-th, distinct by position
// - t5d-scan6b's construction, unchanged - with each root's NEXT ACTION, the
// action its record plays from it (the chosen move 6b'' guards). The records
// are read as 6b read them (tool_support.hpp, sixb_pgn_options: a written 'x'
// must capture - the reading before H51), not with the reader's defaults. The
// records come on the command line: no code here names a corpus.

#ifndef T5D_TOOLS_SIXB_ROOTS_HPP
#define T5D_TOOLS_SIXB_ROOTS_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"

namespace t5d::tools {

/// A record to read: its id (the path as listed) and where it is.
struct RecordRef {
  std::string id;
  std::string path;
};

/// The records of `--record-list` files (one path per line, '#' lines and
/// blank lines skipped; a relative path is taken from `root` when given, its
/// id staying as listed), after those given directly (id = path). Throws
/// std::runtime_error when a list cannot be read.
std::vector<RecordRef> record_refs(const std::vector<std::string>& direct, const std::vector<std::string>& lists,
                                   const std::string& root);

/// A record's content id: "sha256:" and the SHA-256 of `text` with CRLF read
/// as LF (a checkout's line endings do not change it).
std::string record_content_id(const std::string& text);

/// Each record's content id, in order; a record that cannot be read gives
/// "unreadable: <why>".
std::vector<std::string> record_contents(const std::vector<RecordRef>& records);

struct SixbRoot {
  std::string record;  // the record's id (as listed: for messages)
  /// The record's content id, "sha256:" and the SHA-256 of its bytes with
  /// CRLF read as LF (a checkout's line endings do not change it): how
  /// recorded data names a record without naming a path (Hunt's truth).
  std::string content;
  int index = 0;       // the actions played before it
  bool synthetic = false;
  GameState state;
  /// The record's next action from it, its moves in the record's order;
  /// nullopt where the record ends here or the action cannot be read.
  std::optional<std::vector<Move>> next_action;
  /// Why there is no next action (empty when there is one): "its record ends
  /// here", or what the engine could not read or play - the record goes on
  /// with an action the engine cannot read (PgnRecord::unreadable), cannot
  /// resolve, or cannot play.
  std::string no_next;
};

struct SixbRoots {
  std::vector<SixbRoot> roots;
  /// Per record: how far it was read - "N actions read, ok" only where the
  /// record was read and played to its end.
  std::vector<std::string> notes;
  std::size_t duplicates = 0;
};

/// The roots of `records` under `rules`: every `every`-th turn start, the
/// first of equal positions kept, at most `max_roots` (0: all).
SixbRoots sixb_roots(const std::vector<RecordRef>& records, const RuleOptions& rules, int every,
                     std::size_t max_roots);

}  // namespace t5d::tools

#endif  // T5D_TOOLS_SIXB_ROOTS_HPP
