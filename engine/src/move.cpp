// Moves as text (the 5dpgn export form, with the piece letter).

#include "t5d/move.hpp"

#include <string>

#include "t5d/timeline.hpp"

namespace t5d {

namespace {

void append_board(std::string& out, int timeline, int half_turn, bool even) {
  out += '(';
  out += timeline_label_string(timeline, even);
  out += 'T';
  out += std::to_string(turn_of(half_turn));
  out += ')';
}

}  // namespace

std::string to_string(const Move& move, bool even) {
  std::string out;
  out.reserve(24);
  append_board(out, move.from_timeline, move.from_half_turn, even);
  out += move.piece.empty() ? '?' : letter_of(move.piece.kind());
  out += square_name(move.from_square);
  append_board(out, move.to_timeline, move.to_half_turn, even);
  out += square_name(move.to_square);
  if (move.promotes()) {
    out += '=';
    out += letter_of(move.promotion);
  }
  return out;
}

}  // namespace t5d
