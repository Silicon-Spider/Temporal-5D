// tools/play_record.hpp - the 5dpgn record of a game the engine played against
// itself (t5d/selfplay.hpp; engine stage E6a): what t5d-play prints, and what
// its tests replay through the engine's reader and the gate adapter.

#ifndef T5D_TOOLS_PLAY_RECORD_HPP
#define T5D_TOOLS_PLAY_RECORD_HPP

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "t5d/selfplay.hpp"

namespace t5d::tools {

using Headers = std::vector<std::pair<std::string, std::string>>;

/// The record: headers - Mode, Board (`board`: a variant's name, or "custom",
/// which also writes the Size and the start's board strings), Result,
/// Termination, then `extra` - and the actions in the reader's long form
/// (synthetic::pgn_move: source board, piece letter, squares, ">"/">>" for
/// jumps), numbered by turn, then the result ("*" for a game the rules did not
/// end - the turn cap included, which self-play scores as a draw; the
/// Termination header says which). The engine's reader replays it to the
/// game's final state, under the rules the game was played with.
std::string play_record(const mcts::GameRecord& game, std::string_view board,
                        const Headers& extra = {});

}  // namespace t5d::tools

#endif  // T5D_TOOLS_PLAY_RECORD_HPP
