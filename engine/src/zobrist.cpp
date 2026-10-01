// The 5D Zobrist key from scratch (the incremental path is in t5d/zobrist.hpp
// and Multiverse's edits).

#include "t5d/zobrist.hpp"

#include <cstddef>

#include "t5d/board.hpp"
#include "t5d/multiverse.hpp"

namespace t5d::zobrist {

std::uint64_t key_from_scratch(const Multiverse& multiverse) {
  Accumulator sum;
  for (const TimelineRef& t : multiverse.timelines()) {
    sum.add_timeline(t->position(), t->creator());
    const BoardSequence& boards = t->boards();
    for (std::size_t k = 0; k < boards.size(); ++k) {
      const Board& board = *boards[k];
      const bool head = k + 1 == boards.size();
      sum.add_board(keyed_content_hash(content_hash_from_scratch(board), board.en_passant(), head),
                    t->position(), t->first_half_turn() + static_cast<int>(k));
    }
  }
  return sum.key(multiverse.to_move(), multiverse.ruleset(), multiverse.even(),
                 multiverse.board_size());
}

}  // namespace t5d::zobrist
