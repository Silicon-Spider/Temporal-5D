// 6b's roots with their next actions (see sixb_roots.hpp).

#include "sixb_roots.hpp"

#include <exception>
#include <unordered_set>
#include <utility>

#include "t5d/pgn.hpp"
#include "sha256.hpp"
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

struct Next {
  std::optional<std::vector<Move>> action;
  std::string why;  // when there is none
};

// The action the record plays next from `replay`'s state: its moves resolved
// one by one on a copy (as play_next does); none, and why, if there is none.
Next next_action_of(const PgnReplay& replay) {
  if (replay.finished()) {
    if (replay.record().unreadable) {
      return {std::nullopt, "its record goes on with an action the engine cannot read (" +
                                *replay.record().unreadable + ")"};
    }
    return {std::nullopt, "its record ends here"};
  }
  try {
    GameState s = replay.state();
    std::vector<Move> moves;
    for (const PgnMove& pm : replay.record().actions[replay.actions_played()].moves) {
      const Move m = replay.resolve(s, pm);
      s.apply(m);
      moves.push_back(m);
    }
    return {moves, ""};
  } catch (const std::exception& e) {
    return {std::nullopt, std::string("its record's next action cannot be resolved (") + e.what() + ")"};
  }
}

}  // namespace

std::string record_content_id(const std::string& text) {
  std::string lf;
  lf.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\r' && i + 1 < text.size() && text[i + 1] == '\n') continue;
    lf += text[i];
  }
  return "sha256:" + sha256_hex(lf);
}

std::vector<std::string> record_contents(const std::vector<RecordRef>& records) {
  std::vector<std::string> out;
  out.reserve(records.size());
  for (const RecordRef& ref : records) {
    try {
      out.push_back(record_content_id(read_file(ref.path)));
    } catch (const std::exception& e) {
      out.push_back(std::string("unreadable: ") + e.what());
    }
  }
  return out;
}

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
      const std::string content = record_content_id(text);
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
            Next next = next_action_of(replay);
            out.roots.push_back(
                SixbRoot{ref.id, content, k, synthetic, replay.state(), std::move(next.action), std::move(next.why)});
          }
        }
        if (replay.finished()) {
          // Read to its end - or to an action the engine cannot read.
          if (replay.record().unreadable) {
            end = "then an action the engine cannot read (" + *replay.record().unreadable + ")";
          }
          break;
        }
        try {
          replay.play_next();
        } catch (const RecordError& e) {
          end = std::string(fault_name(e.fault())) + " at action " + std::to_string(replay.actions_played() + 1);
          // The root's next action could not be played: none.
          if (!out.roots.empty() && out.roots.back().record == ref.id &&
              out.roots.back().index == static_cast<int>(replay.actions_played())) {
            out.roots.back().next_action.reset();
            out.roots.back().no_next = std::string("its record's next action cannot be played (") +
                                       fault_name(e.fault()) + ": " + e.what() + ")";
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
