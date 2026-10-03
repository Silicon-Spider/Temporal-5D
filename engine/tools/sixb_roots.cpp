// 6b's roots with their next actions (see sixb_roots.hpp).

#include "sixb_roots.hpp"

#include <exception>
#include <unordered_set>
#include <utility>

#include "t5d/pgn.hpp"
#include "tool_support.hpp"

namespace t5d::tools {

namespace {

std::string trimmed(std::string s) {
  const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  while (!s.empty() && space(s.back())) s.pop_back();
  std::size_t i = 0;
  while (i < s.size() && space(s[i])) ++i;
  return s.substr(i);
}

// The action the record plays next from `replay`'s state: its moves resolved
// one by one on a copy (as play_next does); nullopt if one cannot be.
std::optional<std::vector<Move>> next_action_of(const PgnReplay& replay) {
  if (replay.finished()) return std::nullopt;
  try {
    GameState s = replay.state();
    std::vector<Move> moves;
    for (const PgnMove& pm : replay.record().actions[replay.actions_played()].moves) {
      const Move m = replay.resolve(s, pm);
      s.apply(m);
      moves.push_back(m);
    }
    return moves;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace

std::vector<RecordRef> record_refs(const std::vector<std::string>& direct, const std::vector<std::string>& lists,
                                   const std::string& root) {
  std::vector<RecordRef> out;
  for (const std::string& p : direct) out.push_back(RecordRef{p, p});
  for (const std::string& file : lists) {
    const std::string text = read_file(file);
    std::size_t start = 0;
    while (start < text.size()) {
      std::size_t end = text.find('\n', start);
      if (end == std::string::npos) end = text.size();
      std::string id = trimmed(text.substr(start, end - start));
      start = end + 1;
      if (id.empty() || id[0] == '#') continue;
      const bool absolute = id[0] == '/' || id[0] == '\\' || (id.size() > 1 && id[1] == ':');
      out.push_back(RecordRef{id, (!root.empty() && !absolute) ? root + "/" + id : id});
    }
  }
  return out;
}

SixbRoots sixb_roots(const std::vector<RecordRef>& records, const RuleOptions& rules, int every,
                     std::size_t max_roots) {
  SixbRoots out;
  std::unordered_set<std::uint64_t> seen;
  if (every < 1) every = 1;
  for (const RecordRef& ref : records) {
    try {
      const std::string text = read_file(ref.path);
      PgnRecord record = read_pgn(text);
      const bool synthetic = is_synthetic(record);
      PgnReplay replay(std::move(record), PgnOptions{}, rules);
      parse_mode(replay);
      std::string end = "ok";
      for (;;) {
        const int k = static_cast<int>(replay.actions_played());
        if (k % every == 0) {
          if (!seen.insert(replay.state().zobrist()).second) {
            ++out.duplicates;
          } else if (max_roots == 0 || out.roots.size() < max_roots) {
            out.roots.push_back(SixbRoot{ref.id, k, synthetic, replay.state(), next_action_of(replay)});
          }
        }
        if (replay.finished()) break;
        try {
          replay.play_next();
        } catch (const RecordError& e) {
          end = std::string(fault_name(e.fault())) + " at action " + std::to_string(replay.actions_played() + 1);
          // The root's next action could not be played: none.
          if (!out.roots.empty() && out.roots.back().record == ref.id &&
              out.roots.back().index == static_cast<int>(replay.actions_played())) {
            out.roots.back().next_action.reset();
          }
          break;
        }
      }
      out.notes.push_back(ref.id + ": " + std::to_string(replay.actions_played()) + " actions read, " + end);
    } catch (const std::exception& e) {
      out.notes.push_back(ref.id + ": not read (" + e.what() + ")");
    }
  }
  return out;
}

}  // namespace t5d::tools
