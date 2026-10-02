// The gate adapter's protocol (see t5d/gate.hpp): replays a 5dpgn record
// (t5d/pgn.hpp) and prints what the engine set up at every position - and, in
// full mode, what it says of it (t5d/check.hpp, t5d/search.hpp).

#include "t5d/gate.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "t5d/check.hpp"
#include "t5d/fen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/pgn.hpp"
#include "t5d/search.hpp"
#include "t5d/timeline.hpp"

namespace t5d::gate {

namespace {

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

std::string position_prefix(const GameState& state, std::size_t index) {
  std::string out = "{\"event\":\"position\",\"index\":";
  out += std::to_string(index);
  out += ",\"side\":\"";
  out += state.to_move() == Colour::White ? "white" : "black";
  out += "\",";
  return out;
}

const char* terminal_name(Outcome o) {
  switch (o) {
    case Outcome::Checkmate: return "checkmate";
    case Outcome::Stalemate: return "stalemate";
    case Outcome::Ongoing: return "none";
    case Outcome::Unknown: break;
  }
  throw std::logic_error("t5d-gate: a terminal verdict is never unknown (H37: exhaustive)");
}

// "NAME=VALUE" split; false without '='.
bool split(std::string_view text, std::string_view& name, std::string_view& value) {
  const std::size_t eq = text.find('=');
  if (eq == std::string_view::npos) return false;
  name = text.substr(0, eq);
  value = text.substr(eq + 1);
  return true;
}

ActionCount count_with(const Options& options, const GameState& s, std::size_t cap, Detector d,
                       const SearchBudget& budget) {
  return options.counter ? options.counter(s, cap, d, budget)
                         : count_legal_actions(s, cap, d, budget);
}

bool read_bool(std::string_view value, bool& out) {
  if (value == "true") {
    out = true;
    return true;
  }
  if (value == "false") {
    out = false;
    return true;
  }
  return false;
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
  return position_prefix(state, index) +
         "\"in_check\":null,\"legal_actions\":null,\"terminal\":null}";
}

std::string position_event(const GameState& state, std::size_t index, const Verdicts& v) {
  std::string out = position_prefix(state, index);
  out += "\"in_check\":";
  out += v.in_check ? "true" : "false";
  out += ",\"legal_actions\":";
  out += std::to_string(v.legal_actions);
  out += ",\"terminal\":\"";
  out += terminal_name(v.terminal);
  out += "\"}";
  return out;
}

Verdicts verdicts(const GameState& s, const Options& options) {
  Verdicts v;
  v.in_check = in_check(s);  // R1
  // At least one action is sought, so that terminal is known whatever the cap.
  const std::size_t cap = std::max<std::size_t>(options.cap, 1);
  // No overall budget: the count is exhaustive (or reaches the cap) - H37.
  Detector answered = Detector::Hypercuboid;
  ActionCount count;
  // Every search keeps the memory ceiling (SearchBudget::max_bytes): one that
  // reaches it answers nothing - the run ends "error", saying so (H37).
  const auto ceiling = [&]() {
    return std::runtime_error("the search reached its memory ceiling (" +
                              std::to_string(options.max_bytes >> 20) +
                              " MiB) before it completed: no verdict (H37)");
  };
  if (options.detector) {
    answered = *options.detector;
    SearchBudget unbounded = SearchBudget::unlimited();
    unbounded.max_bytes = options.max_bytes;
    count = count_with(options, s, cap, answered, unbounded);
    if (count.stats.stopped_by_memory) throw ceiling();
  } else {
    // The portfolio: each detector in turn on a node budget that doubles
    // every round; the first to complete answers. Each round's search starts
    // afresh, so the cost is a small multiple of the faster detector's. A
    // detector that reaches the memory ceiling is not run again.
    bool out_of_memory[2] = {false, false};
    for (std::uint64_t nodes = kPortfolioFirstBudget; !count.known(); nodes *= 2) {
      const bool last = nodes > (std::uint64_t{1} << 62);  // no doubling past it: no node limit
      for (const Detector d : {Detector::Hypercuboid, Detector::Naive}) {
        bool& full = out_of_memory[d == Detector::Naive ? 1 : 0];
        if (full) continue;
        SearchBudget b = last ? SearchBudget::unlimited() : SearchBudget::nodes(nodes);
        b.max_bytes = options.max_bytes;
        count = count_with(options, s, cap, d, b);
        if (count.known()) {
          answered = d;
          break;
        }
        full = count.stats.stopped_by_memory;
      }
      if (!count.known() && out_of_memory[0] && out_of_memory[1]) throw ceiling();
    }
  }
  if (!count.known()) throw std::logic_error("t5d-gate: an unbudgeted count did not complete");
  v.legal_actions = std::min(count.count, options.cap);
  v.terminal = count.count > 0 ? Outcome::Ongoing
                               : (v.in_check ? Outcome::Checkmate : Outcome::Stalemate);
  if (options.cross_check_nodes > 0) {
    // Spec §11.0 item 3: the two detectors cross-checked. The other one runs
    // on a node budget; where it does not finish it confirms nothing.
    const Detector other = answered == Detector::Naive ? Detector::Hypercuboid : Detector::Naive;
    SearchBudget b = SearchBudget::nodes(options.cross_check_nodes);
    b.max_bytes = options.max_bytes;
    const ActionCount check = count_with(options, s, cap, other, b);
    if (check.known() && check.count != count.count) {
      throw std::runtime_error(std::string("the detectors disagree: ") +
                               detector_name(answered) + " counts " +
                               std::to_string(count.count) + ", " + detector_name(other) +
                               " counts " + std::to_string(check.count) + " (cap " +
                               std::to_string(cap) + ")");
    }
  }
  return v;
}

int run(std::string_view record, const Options& options, const Emit& emit, const Emit& diagnose) {
  const auto note = [&](const std::string& text) {
    if (diagnose) diagnose(text);
  };
  std::size_t accepted = 0;
  bool positions = false;  // whether position 0 was reported
  try {
    PgnReplay replay(read_pgn(record, options.pgn), options.pgn, options.rules);
    // Parse mode replays pseudo-legally: royal safety at submission (C4) is
    // full mode's, with the verdicts.
    if (options.parse_only) replay.set_check_ruling(CheckRuling::none());
    const auto report = [&](std::size_t index) {
      const GameState& s = replay.state();
      emit(state_event(s, index));
      if (options.parse_only) {
        emit(position_event(s, index));
      } else {
        Verdicts v;
        try {
          v = verdicts(s, options);
        } catch (const std::runtime_error& e) {
          throw std::runtime_error("position " + std::to_string(index) + ": " + e.what());
        }
        emit(position_event(s, index, v));
      }
      if (options.list_actions && *options.list_actions == index) {
        // Diagnostic: the position's legal actions, canonical texts sorted.
        const std::size_t cap = std::max<std::size_t>(options.cap, 1);
        const ActionList list =
            list_legal_actions(s, cap, options.detector.value_or(Detector::Hypercuboid));
        note("t5d-gate: position " + std::to_string(index) + ", " +
             (s.to_move() == Colour::White ? "White" : "Black") + " to move: " +
             std::to_string(list.actions.size()) + (list.reached_cap ? " or more" : "") +
             " legal actions");
        const bool even = s.multiverse().even();
        for (std::size_t i = 0; i < list.actions.size(); ++i) {
          note("  " + std::to_string(i + 1) + ". " + action_text(list.actions[i].moves, even));
        }
      }
    };
    report(0);
    positions = true;
    std::size_t warned = 0;
    const auto warn = [&] {
      for (; warned < replay.warnings().size(); ++warned) {
        note("t5d-gate: warning: " + replay.warnings()[warned]);
      }
    };
    while (!replay.finished()) {
      replay.play_next();
      warn();
      accepted = replay.actions_played();
      report(accepted);
    }
    if (options.list_actions && *options.list_actions > accepted) {
      note("t5d-gate: --actions " + std::to_string(*options.list_actions) +
           ": the record has no such position (positions 0.." + std::to_string(accepted) + ")");
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

// ---- the diagnostics' option readers ------------------------------------------------------

bool set_rule_option(RuleOptions& r, std::string_view assignment, std::string* error) {
  const auto fail = [&](const std::string& why) {
    if (error != nullptr) *error = why;
    return false;
  };
  std::string_view name;
  std::string_view value;
  if (!split(assignment, name, value)) {
    return fail("expected NAME=VALUE, got \"" + std::string(assignment) + "\"");
  }
  struct Flag {
    std::string_view name;
    bool RuleOptions::*field;
  };
  static constexpr Flag kFlags[] = {
      {"play_activates", &RuleOptions::play_activates},
      {"moves_after_shift", &RuleOptions::moves_after_shift},
      {"turn_zero_reachable", &RuleOptions::turn_zero_reachable},
      {"sliders_cross_void", &RuleOptions::sliders_cross_void},
      {"past_self_blocks", &RuleOptions::past_self_blocks},
      {"earlier_playable_jump_branches", &RuleOptions::earlier_playable_jump_branches},
      {"inactive_history_is_destination", &RuleOptions::inactive_history_is_destination},
      {"brawn_rank_timeline_capture", &RuleOptions::brawn_rank_timeline_capture},
      {"timeline_double_step", &RuleOptions::timeline_double_step},
      {"double_step_over_void", &RuleOptions::double_step_over_void},
      {"branch_copies_keep_unmoved", &RuleOptions::branch_copies_keep_unmoved},
      {"jumps_keep_unmoved", &RuleOptions::jumps_keep_unmoved},
      {"en_passant_brawns", &RuleOptions::en_passant_brawns},
      {"brawns_promote", &RuleOptions::brawns_promote},
      {"brawn_cross_board_promotion", &RuleOptions::brawn_cross_board_promotion},
      {"castling_empty_between", &RuleOptions::castling_empty_between},
      {"castling_destination_test", &RuleOptions::castling_destination_test},
      {"castling_rook_path_test", &RuleOptions::castling_rook_path_test},
      {"castling_any_rank", &RuleOptions::castling_any_rank},
      {"castling_rook_two_away", &RuleOptions::castling_rook_two_away},
      {"castling_on_optional_boards", &RuleOptions::castling_on_optional_boards},
  };
  for (const Flag& f : kFlags) {
    if (name != f.name) continue;
    bool b = false;
    if (!read_bool(value, b)) return fail(std::string(name) + " takes true or false");
    r.*(f.field) = b;
    return true;
  }
  const auto choice = [&](auto& field, std::string_view first, auto first_value,
                          std::string_view second, auto second_value) {
    if (value == first) {
      field = first_value;
      return true;
    }
    if (value == second) {
      field = second_value;
      return true;
    }
    return fail(std::string(name) + " takes " + std::string(first) + " or " +
                std::string(second));
  };
  if (name == "activation") {
    return choice(r.activation, "count", ActivationRule::Count, "consecutive",
                  ActivationRule::Consecutive);
  }
  if (name == "initial_credit") {
    return choice(r.initial_credit, "none", InitialTimelineCredit::None, "minus_zero_to_black",
                  InitialTimelineCredit::MinusZeroToBlack);
  }
  if (name == "mandatory") {
    return choice(r.mandatory, "in_present", MandatoryBoards::InPresent, "all_active",
                  MandatoryBoards::AllActive);
  }
  if (name == "reactivation") {
    return choice(r.reactivation, "rewinds", Reactivation::Rewinds, "deferred",
                  Reactivation::Deferred);
  }
  if (name == "double_step_right") {
    return choice(r.double_step_right, "unmoved_flag", DoubleStepRight::UnmovedFlag, "home_rank",
                  DoubleStepRight::HomeRank);
  }
  if (name == "family") {
    return choice(r.family, "standard", RuleFamily::Standard, "reversed_royalty",
                  RuleFamily::ReversedRoyalty);
  }
  if (name == "promotions") {
    PromotionSet set = PromotionSet::from_mask(0);
    for (const char c : value) {
      const std::optional<PieceKind> k = kind_from_letter(c);
      if (!k) return fail("promotions takes piece letters, such as QRBN");
      set.insert(*k);
    }
    r.promotions = set;
    return true;
  }
  return fail("no rule option \"" + std::string(name) + "\" (see RuleOptions, t5d/rules.hpp)");
}

bool set_pgn_option(PgnOptions& o, std::string_view assignment, std::string* error) {
  const auto fail = [&](const std::string& why) {
    if (error != nullptr) *error = why;
    return false;
  };
  std::string_view name;
  std::string_view value;
  if (!split(assignment, name, value)) {
    return fail("expected NAME=VALUE, got \"" + std::string(assignment) + "\"");
  }
  bool* flag = nullptr;
  if (name == "verify_annotations") flag = &o.verify_annotations;
  if (name == "annotation_mismatch_is_error") flag = &o.annotation_mismatch_is_error;
  if (name == "require_capture_marks") flag = &o.require_capture_marks;
  if (name == "infer_omitted_boards") flag = &o.infer_omitted_boards;
  if (name == "check_turn_numbers") flag = &o.check_turn_numbers;
  if (flag != nullptr) {
    if (!read_bool(value, *flag)) return fail(std::string(name) + " takes true or false");
    return true;
  }
  if (name == "castling_shortcuts") {
    if (value == "e_file_only") {
      o.castling_shortcuts = CastlingShortcuts::EFileOnly;
    } else if (value == "any_king") {
      o.castling_shortcuts = CastlingShortcuts::AnyKing;
    } else if (value == "e_file_one_king") {
      o.castling_shortcuts = CastlingShortcuts::EFileOneKing;
    } else {
      return fail("castling_shortcuts takes e_file_only, any_king or e_file_one_king");
    }
    return true;
  }
  return fail("no reader option \"" + std::string(name) + "\" (see PgnOptions, t5d/pgn.hpp)");
}

}  // namespace t5d::gate
