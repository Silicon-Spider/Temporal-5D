// Helpers for the rules tests: states from 5DFEN board strings, moves by their
// export-notation text ("(0T1)Pe2(0T1)e4", t5d/move.hpp), playing actions.
//
// Every position in the tests is written for the test from the rules
// reference (docs/rules/README.md) and the notation specification's text -
// never taken from a game corpus or another implementation's test data.

#ifndef T5D_TEST_POSITIONS_HPP
#define T5D_TEST_POSITIONS_HPP

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/move.hpp"
#include "t5d/movegen.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"

namespace t5d_test {

/// A state from board strings; the player to move inferred from the present
/// unless given.
inline t5d::GameState game(const std::vector<std::string>& boards, t5d::BoardSize size,
                           const t5d::RuleOptions& rules = {},
                           std::optional<t5d::Colour> to_move = std::nullopt) {
  return t5d::game_from_fen(boards, size, t5d::Parity::Infer, rules, to_move);
}

inline t5d::MoveList moves_of(const t5d::GameState& state) {
  t5d::MoveList moves;
  t5d::generate_moves(state, moves);
  return moves;
}

/// Every move's export text, sorted.
inline std::vector<std::string> texts(const t5d::GameState& state, const t5d::MoveList& moves) {
  std::vector<std::string> out;
  out.reserve(moves.size());
  for (const t5d::Move& m : moves) out.push_back(t5d::to_string(m, state.multiverse().even()));
  std::sort(out.begin(), out.end());
  return out;
}

inline std::vector<std::string> texts(const t5d::GameState& state) {
  return texts(state, moves_of(state));
}

/// The moves of the piece on `square` ("e4") of timeline p's playable board,
/// as texts, sorted.
inline std::vector<std::string> piece_texts(const t5d::GameState& state, int timeline,
                                            std::string_view square) {
  const int sq = (square[1] - '1') * 8 + (square[0] - 'a');
  t5d::MoveList moves;
  t5d::append_piece_moves(state, timeline, sq, moves);
  return texts(state, moves);
}

/// The generated move whose text is `text`, if any.
inline std::optional<t5d::Move> find_move(const t5d::GameState& state, std::string_view text) {
  for (const t5d::Move& m : moves_of(state)) {
    if (t5d::to_string(m, state.multiverse().even()) == text) return m;
  }
  return std::nullopt;
}

inline bool has_move(const t5d::GameState& state, std::string_view text) {
  return find_move(state, text).has_value();
}

/// Applies the generated move whose text is `text`; throws (failing the test)
/// with the available moves if there is none.
inline t5d::Move play(t5d::GameState& state, std::string_view text) {
  const std::optional<t5d::Move> m = find_move(state, text);
  if (!m) {
    std::string all;
    for (const std::string& t : texts(state)) all += " " + t;
    throw std::runtime_error("no generated move " + std::string(text) + "; moves:" + all);
  }
  state.apply(*m);
  return *m;
}

/// Plays a whole action - the moves in order - and submits it (without a
/// royal-safety test: CheckRuling::none(), pseudo-legal play).
inline void act(t5d::GameState& state, std::initializer_list<std::string_view> moves) {
  for (const std::string_view text : moves) play(state, text);
  state.submit(t5d::CheckRuling::none());
}

/// The same, submitted under the engine's check ruling (C4, R1): throws
/// std::logic_error if the action leaves a royal piece capturable.
inline void act_legally(t5d::GameState& state, std::initializer_list<std::string_view> moves) {
  for (const std::string_view text : moves) play(state, text);
  state.submit();
}

/// The board at timeline p, turn T, colour c.
inline const t5d::Board* board(const t5d::GameState& state, int p, int turn, t5d::Colour c) {
  return state.multiverse().board_at(p, t5d::half_turn(turn, c));
}

/// The piece at "e4" on a board.
inline t5d::Piece at(const t5d::Board* board, std::string_view square) {
  // No board (a timeline or turn that does not exist): fail this test, not
  // the whole runner - the runner catches the exception per test.
  if (board == nullptr) throw std::logic_error("at(): no such board");
  return board->at(square[0] - 'a', square[1] - '1');
}

inline bool contains(const std::vector<std::string>& v, std::string_view s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

}  // namespace t5d_test

#endif  // T5D_TEST_POSITIONS_HPP
