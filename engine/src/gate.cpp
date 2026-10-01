// The gate adapter's protocol (see t5d/gate.hpp): parse mode replays a 5dpgn
// record (t5d/pgn.hpp) and prints what the engine set up at every position.

#include "t5d/gate.hpp"

#include <cstddef>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "t5d/fen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/pgn.hpp"
#include "t5d/timeline.hpp"

namespace t5d::gate {

namespace {

constexpr std::string_view kFullModeReason = "full mode waits on the check ruling";

// Every end event carries actions_accepted; all but "ok" carry a reason.
std::string end_event(std::string_view status, std::size_t accepted,
                      std::optional<std::string_view> reason) {
  std::string out = "{\"event\":\"end\",\"status\":";
  out += json_string(status);
  out += ",\"actions_accepted\":";
  out += std::to_string(accepted);
  if (reason) {
    out += ",\"reason\":";
    out += json_string(*reason);
  }
  out += '}';
  return out;
}

}  // namespace

std::string json_string(std::string_view text) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(text.size() + 2);
  out += '"';
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\t') {
      out += "\\t";
    } else if (byte < 0x20 || byte == 0x7F) {
      out += "\\u00";
      out += kHex[byte >> 4];
      out += kHex[byte & 15];
    } else if (byte >= 0x80) {
      out += '?';
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

std::string state_event(const GameState& state, std::size_t index) {
  const Multiverse& mv = state.multiverse();
  std::string out;
  out.reserve(64 + mv.board_count() * 80);
  out += "{\"event\":\"state\",\"index\":";
  out += std::to_string(index);
  out += ",\"size\":\"";
  out += to_string(mv.board_size());
  out += "\",\"markers\":true,\"boards\":[";
  bool first = true;
  mv.for_each_board([&](int p, int h, const Board& board) {
    if (!first) out += ',';
    first = false;
    out += '"';
    out += timeline_label_string(p, mv.even());
    out += ':';
    out += std::to_string(turn_of(h));
    out += ':';
    out += colour_letter(to_move_at(h));
    out += ':';
    out += fen_pieces(board);  // letters, digits, '/' and '*': nothing to escape
    out += '"';
  });
  out += "]}";
  return out;
}

std::string position_event(const GameState& state, std::size_t index) {
  std::string out = "{\"event\":\"position\",\"index\":";
  out += std::to_string(index);
  out += ",\"side\":\"";
  out += state.to_move() == Colour::White ? "white" : "black";
  out += "\",\"in_check\":null,\"legal_actions\":null,\"terminal\":null}";
  return out;
}

int run(std::string_view record, const Options& options, const Emit& emit, const Emit& diagnose) {
  const auto note = [&](const std::string& text) {
    if (diagnose) diagnose(text);
  };
  if (!options.parse_only) {
    // Full mode answers in_check, legal_actions and terminal: it waits on the
    // check ruling (Q10, C3, C6, C7). The same end-event fields as every
    // other end: actions_accepted (0) and the reason.
    emit(end_event("unsupported", 0, kFullModeReason));
    return 0;
  }
  std::size_t accepted = 0;
  bool positions = false;  // whether position 0 was reported
  try {
    PgnReplay replay(read_pgn(record));
    // Parse mode replays pseudo-legally: royal safety at submission (C4) is
    // full mode's, with the verdicts.
    replay.set_check_ruling(CheckRuling::none());
    emit(state_event(replay.state(), 0));
    emit(position_event(replay.state(), 0));
    positions = true;
    while (!replay.finished()) {
      replay.play_next();
      accepted = replay.actions_played();
      emit(state_event(replay.state(), accepted));
      emit(position_event(replay.state(), accepted));
    }
    if (replay.record().unreadable) {
      note("t5d-gate: rejected after " + std::to_string(accepted) + " actions: " +
           *replay.record().unreadable);
      emit(end_event("rejected", accepted, fault_name(RecordFault::Unreadable)));
      return 0;
    }
    emit(end_event("ok", accepted, std::nullopt));
    return 0;
  } catch (const RecordError& e) {
    if (e.fault() == RecordFault::Unsupported && !positions) {
      note(std::string("t5d-gate: unsupported: ") + e.what());
      emit(end_event("unsupported", accepted, e.what()));
      return 0;
    }
    note("t5d-gate: rejected after " + std::to_string(accepted) + " actions: " + e.what());
    emit(end_event("rejected", accepted, fault_name(e.fault())));
    return 0;
  } catch (const std::bad_alloc&) {
    note("t5d-gate: error: out of memory");
    emit(end_event("error", accepted, "out of memory"));
    return 1;
  } catch (const std::exception& e) {
    note(std::string("t5d-gate: error: ") + e.what());
    emit(end_event("error", accepted, e.what()));
    return 1;
  }
}

}  // namespace t5d::gate
