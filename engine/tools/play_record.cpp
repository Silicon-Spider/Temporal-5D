// The 5dpgn record of a game the engine played (play_record.hpp).

#include "play_record.hpp"

#include <string>

#include "synthetic.hpp"
#include "t5d/fen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/types.hpp"

namespace t5d::tools {

namespace {

std::string escaped(std::string_view value) {
  std::string out;
  for (const char c : value) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

void header(std::string& out, std::string_view name, std::string_view value) {
  out += '[';
  out += name;
  out += " \"";
  out += escaped(value);
  out += "\"]\n";
}

}  // namespace

std::string play_record(const mcts::GameRecord& game, std::string_view board, const Headers& extra) {
  const Multiverse& start = game.start.multiverse();
  const bool even = start.even();
  std::string out;
  header(out, "Mode", "5D");
  header(out, "Board", board);
  if (board == "custom") {
    header(out, "Size", to_string(start.board_size()));
  }
  header(out, "Result", game.result());
  std::string termination = mcts::game_end_name(game.end);
  if (game.end == mcts::GameEnd::TurnCap) {
    termination += " (self-play's convention, I11: scored as a draw; not a game rule)";
  }
  header(out, "Termination", termination);
  for (const auto& [name, value] : extra) header(out, name, value);
  if (board == "custom") {
    start.for_each_board([&](int p, int h, const Board& b) {
      out += fen_board_string(b, p, h, even);
      out += '\n';
    });
  }
  out += '\n';
  Colour colour = game.start.to_move();
  int turn = turn_of(game.start.present());
  if (turn < 1) turn = 1;
  for (std::size_t a = 0; a < game.actions.size(); ++a) {
    if (colour == Colour::White) {
      if (a != 0) out += '\n';
      out += std::to_string(turn) + ".";
    } else {
      out += a == 0 ? std::to_string(turn) + "b." : std::string(" /");
    }
    for (const Move& m : game.actions[a]) {
      out += ' ';
      out += synthetic::pgn_move(m, even);
    }
    if (colour == Colour::Black) ++turn;
    colour = opposite(colour);
  }
  if (!game.actions.empty()) out += ' ';
  out += game.result();
  out += '\n';
  return out;
}

}  // namespace t5d::tools
