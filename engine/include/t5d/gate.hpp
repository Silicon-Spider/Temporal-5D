// t5d/gate.hpp - the engine's side of the differential-testing gate (design
// spec §11.0, "Running the gate"; decisions H23, H30): what t5d-gate, the
// engine's adapter, prints for one 5dpgn record.
//
// The protocol is the gate's engine contract, in the harness's manual ("The
// engine contract"); the engine README describes it as implemented. In brief:
// the adapter reads one record and prints one JSON object per line - for each
// position K (0 = the set-up, K = after K submitted actions) a "state" event
// listing every board, then a "position" event; then one "end" event. The
// adapter answers from the engine alone (t5d/pgn.hpp replays the record).
//
// Parse mode (--parse-only) is all there is until the check ruling: the
// position events carry null in_check, legal_actions and terminal (no verdict
// is given), and an action counts as accepted when its moves are pseudo-legal,
// the turn rules allow them and the present has shifted (T5) - the royal-
// safety test at submission (C4) is the pending ruling's. Full mode answers
// "unsupported" before any position.

#ifndef T5D_GATE_HPP
#define T5D_GATE_HPP

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

#include "t5d/state.hpp"

namespace t5d::gate {

/// The adapter's command line.
struct Options {
  /// --parse-only: replay the record and report the states; no verdicts.
  bool parse_only = false;
};

/// Called with one line of output: a JSON object, without the newline.
using Emit = std::function<void(const std::string&)>;

/// Runs the adapter on one record (the bytes read from standard input).
/// `emit` receives each output line in order (the caller writes it, a newline
/// and flushes); `diagnose`, if set, receives human-readable notes for
/// standard error - why a record was rejected, with the offending token.
/// Returns the process's exit status: 0 for a run that ended "ok", "rejected"
/// or "unsupported", 1 for "error".
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

/// `text` as a JSON string literal, quotes included. Control characters are
/// escaped; a byte outside ASCII becomes '?', so the output is ASCII.
std::string json_string(std::string_view text);

}  // namespace t5d::gate

#endif  // T5D_GATE_HPP
