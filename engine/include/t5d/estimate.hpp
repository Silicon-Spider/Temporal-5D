// t5d/estimate.hpp - the estimate channel (design spec §11.1, "The redesign
// after 6b (CHECKPOINT 3)", part 1; register M14, H2, H6; rung 1 of §11.2,
// H10). Rule IDs are the rulebook's (docs/rules/README.md).
//
// A cheap, INEXACT local search over capture and threat sequences, WITHOUT
// TURN COMPLETION: the side to move makes one move and the opponent replies
// at once, as if that move were the whole turn - no obligations (T5), no
// submission test (C4), no activity or present. Each side moves from the
// boards it is to move on (L5): its untouched playable boards and the boards
// the other side's moves created. A royal left capturable is simply
// captured (the capturer scores royal_value). The eval is the v1 material
// count (H6): the side to move's material on every timeline's latest board
// minus the opponent's.
//
// The boards a move creates must be answered on: those the reply does not
// play are PASSED (advanced unchanged, the mover to move on them again), so a
// threat made there carries. A royal of the side to move attacked, on a board
// the opponent's last move created, by a piece on that board is a LOCAL
// CHECK.
//
// The search: negamax alpha-beta; stand-pat at every node not in local
// check; captures at every ply, MVV-LVA first; in the first `threat_plies`
// plies also THREATS - moves within a board after which a piece on the board
// they create attacks an enemy royal there; in local check every move of the
// checked board (its local evasions), and no answer loses the royal. Delta pruning
// (a capture that cannot lift the score to alpha even with `delta_margin`
// to spare is not searched) - speculative pruning is estimate-channel only
// (§11.1). A ply limit and a node cap bound it. The captures are each side's
// all-capture map (t5d/attacks.hpp), advanced move by move - exact for the
// local position, and only the boards a move makes rescanned; the threats
// and evasions are the generator's moves (append_moves_of). En passant is
// not searched.
//
// ITS OUTPUT IS A SOFT PRIOR ONLY (H2): a score, its swing against the static
// eval, and whether the position is tactically live. It proves nothing,
// exports nothing to the solved store, and no move it makes reaches the
// tree - moves reach the tree only through the generator, which is exact.
// Its errors cost search efficiency, never correctness.

#ifndef T5D_ESTIMATE_HPP
#define T5D_ESTIMATE_HPP

#include <cstdint>
#include <optional>

#include "t5d/move.hpp"
#include "t5d/scanner.hpp"
#include "t5d/state.hpp"

namespace t5d {

/// Every parameter a PLACEHOLDER (spec issue: §11.1 gives the channel's
/// shape, not its depth, move set or margins).
struct EstimateOptions {
  PieceValues values;
  /// Plies (single moves) searched below the position: 3, anchored on the
  /// depth §15.6's rung-1 test asked of a leaf scan (3 forcing half-turns).
  int max_plies = 3;
  /// Plies in which threats (local checks) are searched besides captures.
  int threat_plies = 1;
  /// Delta pruning's margin, centipawns.
  int delta_margin = 200;
  /// The score of capturing a royal.
  int royal_value = 20000;
  /// A swing of at least this much (either way) makes the position
  /// tactically live.
  int live_margin = 100;
  /// The node cap (a node: a position searched, or a board scanned - a
  /// capture map's source, a board's moves generated).
  std::uint64_t max_nodes = 100'000;
};

struct EstimateResult {
  /// The material eval, the side to move's view.
  int static_eval = 0;
  /// The search's score, the side to move's view.
  int score = 0;
  int swing() const noexcept { return score - static_eval; }
  /// |swing| >= live_margin.
  bool live = false;
  /// The first move of the best line, where it beats standing pat.
  std::optional<Move> best;
  /// The work: positions searched plus boards scanned.
  std::uint64_t nodes = 0;
  std::uint64_t positions = 0;
  /// The node cap stopped it.
  bool capped = false;
  /// The deepest ply reached.
  int depth = 0;
  /// Boards whose captures were generated.
  std::uint64_t boards_scanned = 0;
};

/// The estimate at `position` (a turn start, or any state: the side to move
/// is the multiverse's).
EstimateResult estimate(const GameState& position, const EstimateOptions& options = {});

}  // namespace t5d

#endif  // T5D_ESTIMATE_HPP
