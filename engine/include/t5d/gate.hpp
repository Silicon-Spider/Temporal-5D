// t5d/gate.hpp - the engine's side of the differential-testing gate (design
// spec §11.0, "Running the gate"; decisions H23, H30, H36, H37): what t5d-gate,
// the engine's adapter, prints for one 5dpgn record.
//
// The protocol is the gate's engine contract, in the harness's manual ("The
// engine contract"); the engine README describes it as implemented. In brief:
// the adapter reads one record and prints one JSON object per line - for each
// position K (0 = the set-up, K = after K submitted actions) a "state" event
// listing every board, then a "position" event; then one "end" event. The
// adapter answers from the engine alone (t5d/pgn.hpp replays the record).
//
//  * FULL MODE (the default): the replay is legal - every action's
//    submission tested under the engine's check ruling (C4, R1: an action
//    that leaves a royal capturable is "cannot_submit") - and every position
//    event answers every field: in_check (R1's check status), legal_actions
//    (the legal actions as the harness's manual defines them - sets of moves,
//    their timeline-creating moves in order - t5d/search.hpp, min(count,
//    cap)), terminal ("none", "checkmate" or "stalemate", C5). The count and
//    the verdict come from a search run to completion, with no overall budget
//    (H37: a mate verdict is a proof, and the gate's answer is never
//    "unknown"; the harness's own timeout bounds the run): by default a
//    PORTFOLIO of the two detectors, each in turn on a node budget that
//    doubles, until one completes. The other detector then cross-checks the
//    count under a node budget (spec §11.0 item 3): if both know it and they
//    differ, the run ends "error".
//  * PARSE MODE (--parse-only): the replay is pseudo-legal (no royal-safety
//    test at submission) and the position events carry null in_check,
//    legal_actions and terminal.
//
// The options beyond --parse-only and --cap are DIAGNOSTICS, for people
// investigating a record - never for the gate's configuration (gate.toml):
// another reading of an open rule (RuleOptions), lenient annotations
// (PgnOptions), which detector counts, and a listing of a position's legal
// actions on standard error.

#ifndef T5D_GATE_HPP
#define T5D_GATE_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "t5d/pgn.hpp"
#include "t5d/rules.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"

namespace t5d::gate {

/// The adapter's command line.
struct Options {
  /// --parse-only: replay the record pseudo-legally and report the states; no
  /// verdicts.
  bool parse_only = false;
  /// --cap N (the gate appends it): legal_actions is min(count, cap). 1000,
  /// the manual's cap, when not given.
  std::size_t cap = 1000;

  // ---- diagnostics (never for gate.toml) ----------------------------------------
  /// --rule NAME=VALUE, --no-moves-after-shift (Q4's Guide-literal reading,
  /// the same as --rule moves_after_shift=false): the rules (default: the
  /// standard rules, R8's moves after the shift among them; a record's
  /// variant and Promotions header still apply).
  RuleOptions rules{};
  /// --pgn NAME=VALUE, --lenient-annotations: the reader's readings.
  PgnOptions pgn{};
  /// --detector naive|hypercuboid|portfolio: which detector answers, run
  /// without a budget; none (the default, "portfolio"): both in turn on
  /// doubling node budgets, the first to complete answering. The other
  /// cross-checks it.
  std::optional<Detector> detector;
  /// --cross-check-nodes N: the cross-checking detector's node budget per
  /// position (0: no cross-check, --no-cross-check).
  std::uint64_t cross_check_nodes = 50000;
  /// --max-memory-mib N: every search's memory ceiling (SearchBudget::max_bytes;
  /// default kDefaultSearchBytes). A count that reaches it ends the run
  /// "error", saying so - never a verdict (H37).
  std::size_t max_bytes = kDefaultSearchBytes;
  /// --actions K: list position K's legal actions (sorted canonical texts) on
  /// standard error, through `diagnose`.
  std::optional<std::size_t> list_actions;

  /// A test seam, no command-line option: what counts a position's legal
  /// actions (count_legal_actions when empty). The tests use it to make the
  /// detectors disagree.
  std::function<ActionCount(const GameState&, std::size_t, Detector, const SearchBudget&)> counter;
};

/// The portfolio's first node budget (it doubles each round).
inline constexpr std::uint64_t kPortfolioFirstBudget = 1u << 14;

/// Called with one line of output: a JSON object, without the newline.
using Emit = std::function<void(const std::string&)>;

/// Runs the adapter on one record (the bytes read from standard input).
/// `emit` receives each output line in order (the caller writes it, a newline
/// and flushes); `diagnose`, if set, receives human-readable notes for
/// standard error - why a record was rejected, with the offending token, and
/// the diagnostics' listings. Returns the process's exit status: 0 for a run
/// that ended "ok", "rejected" or "unsupported", 1 for "error".
int run(std::string_view record, const Options& options, const Emit& emit,
        const Emit& diagnose = {});

/// The state event of position `index`: {"event":"state","index":K,
/// "size":"WxH","markers":true,"boards":["<L>:<T>:<c>:<placement>",...]} -
/// every board, superseded and head, in the canonical order (ascending
/// timeline, then half-turn); L the canonical label ("0", "-0", "+0", "+1",
/// "-1"), T the turn as displayed, c the colour to move on the board, and the
/// placement the canonical 5DFEN pieces field ('*' on exactly the unmoved
/// pawns, rooks, kings and brawns).
std::string state_event(const GameState& state, std::size_t index);

/// The parse-mode position event of position `index`: {"event":"position",
/// "index":K,"side":"white"|"black","in_check":null,"legal_actions":null,
/// "terminal":null}.
std::string position_event(const GameState& state, std::size_t index);

/// What full mode says of a position.
struct Verdicts {
  bool in_check = false;
  /// min(count, cap).
  std::size_t legal_actions = 0;
  /// Ongoing ("none"), Checkmate or Stalemate - never Unknown.
  Outcome terminal = Outcome::Ongoing;
};

/// Full mode's verdicts at a turn's start: R1's check status, and a count of
/// the legal actions run to completion (to max(cap, 1), so that terminal is
/// known even with cap 0) by options.detector (or the portfolio),
/// cross-checked by the other detector under options.cross_check_nodes.
/// Throws std::runtime_error if the two detectors disagree.
Verdicts verdicts(const GameState& turn_start, const Options& options);

/// The full-mode position event: {"event":"position","index":K,"side":...,
/// "in_check":true|false,"legal_actions":N,"terminal":"none"|"checkmate"|
/// "stalemate"} - exactly six fields.
std::string position_event(const GameState& state, std::size_t index, const Verdicts& verdicts);

/// Sets one rule option from "NAME=VALUE" (a RuleOptions field's name; true or
/// false for a flag, the enumerator's name for a choice - activation=consecutive,
/// mandatory=all_active, reactivation=deferred, initial_credit=minus_zero_to_black,
/// double_step_right=home_rank, family=reversed_royalty). False, with a
/// message, for an unknown name or value.
bool set_rule_option(RuleOptions& rules, std::string_view assignment, std::string* error);

/// The same for the reader's PgnOptions (verify_annotations,
/// annotation_mismatch_is_error, turn_number_mismatch_is_error,
/// infer_omitted_boards, check_turn_numbers,
/// castling_shortcuts=e_file_only|any_king|e_file_one_king,
/// capture_marks=lenient|marked_must_capture|strict - and
/// require_capture_marks=true|false, its spelling before H51: strict, and the
/// reading that was then the default, marked_must_capture).
bool set_pgn_option(PgnOptions& options, std::string_view assignment, std::string* error);

/// `text` as a JSON string literal, quotes included. Control characters are
/// escaped; a byte outside ASCII becomes '?', so the output is ASCII.
std::string json_string(std::string_view text);

}  // namespace t5d::gate

#endif  // T5D_GATE_HPP
