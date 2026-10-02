// t5d/legal.hpp - which moves are LEGAL: the pseudo-legal moves after which
// the action under way can still be finished legally (rulebook C4, R1:
// legality is judged per action, so a single move is legal when a legal
// action begins with it - or, during an action, continues with it; build-log
// spec issue 28). Fast, and exact unless a budget is given.
//
// A search per move (t5d/search.hpp, can_finish_legally) answers the question
// exactly but slowly. classify_moves answers it in TIERS, each sound - a move
// is called legal only with a legal finish in hand or by an argument that one
// exists, and illegal only with a proof - and only what no tier settles goes
// to a search:
//
//  0. A royal capture already standing - from a board on which the opponent
//     is to move, onto a royal on a board of the opponent's colour - persists
//     to the end of the action: the mover moves only on boards of its own
//     colour, so it can neither take the attacker, nor move the royal, nor
//     change any board the capture rests on (boards never change; a move
//     makes a new one). Then no move is legal. (Under Q18's alternative a
//     crossed void could later hold a board, so only captures that cross no
//     void are taken as standing.)
//  1. THE STRIPPED MULTIVERSE: the position with, on every playable board of
//     the mover's, a successor holding the board's opponent pieces and the
//     mover's royals but none of the mover's other pieces. Any action whose
//     moves create no timeline and move no royal ("quiet") ends in a position
//     whose royal captures are among the stripped multiverse's: it has the
//     same attackers or fewer, the same royals, and more blockers - and, where
//     a board was not played, void where the stripped one has a board (a
//     slide cannot cross void, unless Q18's alternative, where this tier is
//     off). So when the stripped multiverse holds no capture, every quiet
//     action is safe, and a quiet move is legal iff the boards the action
//     must still play can be played by quiet moves: decided per move in O(1).
//     A capture found is first removed by keeping a blocker (the mover's
//     piece it crosses stays, and that piece's moves leave the tier), or by
//     dropping an optional board from the multiverse (its moves leave the
//     tier). A branching move adds a new timeline - outermost, one board, a
//     copy of its destination: one scan per destination board decides every
//     quiet branch onto it, the move's own effect on the present (T8, T10,
//     T11, T12) taken from the state machine.
//     Where the stripped multiverse keeps a capture, two more arguments:
//     REFUTATION BY A PASSED BOARD - the state plus board p played as it is
//     (a successor of the opponent's colour, its pieces unmoved): a capture
//     found there stands, by tier 0's lemma, after any move on p that changes
//     none of the squares it rests on, so such moves are illegal; and
//     AROUND A FINISH - given one legal finish A0 (made greedily - a move per
//     board, replayed and tested - or found by one search), each board A0
//     plays within itself is stripped alone in A0's end position: if that
//     holds no capture, a quiet move on that board, put in place of A0's move
//     there, finishes legally too (the same argument as the stripped
//     multiverse's, on one board, the others fixed as A0 leaves them; the
//     present taken from one replay per board).
//  2. Per move: after it, a capture standing as in tier 0 refutes it; a
//     finish by quiet moves on the boards still to play, replayed through the
//     state machine and tested (C4), proves it.
//  3. The rest: one search from the state (if the action has no legal finish
//     at all, nothing is legal), then a search per move, within the budget.
//     A legal finish found also proves every move that can lead it.
//
// Exact unless a budget stops a search (or the memory ceiling, SearchBudget):
// then the move is UNKNOWN - never called illegal without proof.

#ifndef T5D_LEGAL_HPP
#define T5D_LEGAL_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"

namespace t5d {

enum class MoveLegality : std::uint8_t { Legal = 0, Illegal = 1, Unknown = 2 };

/// What classify_moves may spend, and with what. Beyond a budget, the moves it
/// would have settled are left to the next tier, or Unknown.
struct LegalityOptions {
  /// Checks: tier 2's per-move checks (a state copy, a replay and a few
  /// scans each), the greedy finish's tries, and tier 1's scans of a branch's
  /// destination board. The default: no limit. 0 with search_nodes 0: tier 0,
  /// tier 1's quiet moves and its refutations by a passed board only - the
  /// FAST classification, every other move Unknown.
  std::uint64_t checks = std::numeric_limits<std::uint64_t>::max();
  /// Search nodes in all (SearchBudget's nodes), over every search the
  /// classification makes (tier 3). The default: no limit - exact.
  std::uint64_t search_nodes = std::numeric_limits<std::uint64_t>::max();
  /// Each search's memory ceiling.
  std::size_t max_bytes = kDefaultSearchBytes;
  /// The detector tier 3 searches with.
  Detector detector = Detector::Hypercuboid;

  static LegalityOptions exact() noexcept { return LegalityOptions{}; }
  static LegalityOptions nodes(std::uint64_t n) noexcept {
    LegalityOptions b;
    b.search_nodes = n;
    return b;
  }
  static LegalityOptions budgeted(std::uint64_t checks, std::uint64_t nodes) noexcept {
    LegalityOptions b;
    b.checks = checks;
    b.search_nodes = nodes;
    return b;
  }
};

/// How the moves were settled, by tier.
struct LegalityStats {
  std::size_t moves = 0;
  std::size_t standing_at_start = 0;  // tier 0: none legal
  std::size_t bulk_quiet = 0;         // tier 1: quiet moves
  std::size_t bulk_branches = 0;      // tier 1: branching moves
  std::size_t refuted_in_bulk = 0;    // tier 1: a capture standing once the move's board is played
  std::size_t modified = 0;           // tier 2: the move's own boards stripped (counted in finished too)
  std::size_t finished = 0;           // tier 2: proved legal (a finish replayed, the move ends a legal
                                      // action, or its own boards stripped)
  std::size_t standing_after = 0;     // tier 2: a standing capture after the move
  std::size_t learned = 0;            // tiers 1-3: proved by a finish found for another move
  std::size_t greedy = 0;             // states where a finish was made greedily (no search)
  std::size_t no_finish_at_all = 0;   // tier 3: the search from the state found none
  std::size_t searched_legal = 0;     // tier 3
  std::size_t searched_illegal = 0;   // tier 3
  std::size_t unknown = 0;            // tier 3, the budget (or memory ceiling) spent
  std::uint64_t search_nodes = 0;     // spent by tier 3
  std::size_t scans = 0;              // royal-capture scans of tier 1 (stripped and destination)
  bool bulk = false;                  // tier 1's stripped multiverse was capture-free
  std::size_t kept_blockers = 0;      // pieces tier 1 kept
  std::size_t dropped_boards = 0;     // optional boards tier 1 dropped
};

/// The legality of each of `moves` - pseudo-legal moves of the player to move
/// at `state` (generate_moves) - into `out` (resized; in the same order).
/// `state` may be a turn's start or a state during an action.
void classify_moves(const GameState& state, std::span<const Move> moves,
                    std::vector<MoveLegality>& out,
                    const LegalityOptions& budget = LegalityOptions::exact(),
                    LegalityStats* stats = nullptr);

}  // namespace t5d

#endif  // T5D_LEGAL_HPP
