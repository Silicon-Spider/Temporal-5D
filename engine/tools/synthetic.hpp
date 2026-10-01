// tools/synthetic.hpp - SYNTHETIC positions: seeded random pseudo-legal play,
// for throughput and stress tests only (engine stage E3; t5d-bench, the tests).
//
// No game record in the repository reaches the 100-600 boards of design spec
// §1 and §15.6, so the movegen benchmark (register M9) runs on positions made
// here. They are random play - not chess anyone plays - and they are labelled
// as such wherever they go (a "Synthetic" header in every record written, a
// "synthetic" field in every tool's output). They are NEVER evidence for the
// |E| verdict (M7), which needs real games.
//
// The player: at each move of an action it picks, uniformly at random, one of
// the pseudo-legal moves of the player to move (t5d/movegen.hpp) - never one
// that captures a royal piece (rules reference 4.13; a common king, 4.12, is
// not royal) - and plays until the present has shifted (T5), then submits
// without the royal-safety test (C4: a ruling with no hook, CheckRuling::none()
// in stage E4's terms), so its games are pseudo-legal; the turn state machine
// decides when the present has shifted. If it gets stuck - no move, and the
// action cannot be submitted - the game stops there. One knob steers the
// number of ACTIVE timelines (T10) towards a target: while there are fewer,
// each move is, with probability branch_percent, a branching move that creates
// an active timeline (when one exists); at the target, no move branches unless
// every move does. So the active-timeline count can be set to the spec's
// assumed T at each N (6, 7, 8, 10, 12 at 100, 160, 240, 400, 600 boards) and
// held while N grows (§15.6's flatness test).

#ifndef T5D_TOOLS_SYNTHETIC_HPP
#define T5D_TOOLS_SYNTHETIC_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"

namespace t5d::synthetic {

/// The generator's random numbers: xorshift64*, the same on every platform and
/// compiler (the standard library's distributions are not).
class Rng {
 public:
  explicit Rng(std::uint64_t seed) noexcept : state_(seed == 0 ? 0x9E3779B97F4A7C15ull : seed) {
    for (int i = 0; i < 4; ++i) next();  // spread nearby seeds apart
  }
  std::uint64_t next() noexcept {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return state_ * 0x2545F4914F6CDD1Dull;
  }
  /// In [0, n), n >= 1.
  std::size_t below(std::size_t n) noexcept { return static_cast<std::size_t>((next() >> 11) % n); }
  bool chance(int percent) noexcept { return static_cast<int>(below(100)) < percent; }

 private:
  std::uint64_t state_;
};

struct Spec {
  /// The starting position: a variant the engine sets up (t5d/variants.hpp).
  std::string variant = "Standard";
  RuleOptions rules{};
  std::uint64_t seed = 1;
  /// The number of active timelines to steer towards.
  int target_active = 8;
  /// While below target_active: the chance (percent) that a move is a branching
  /// move creating an active timeline, where one exists.
  int branch_percent = 25;
  /// False: a branching move may create an inactive timeline too (T10) - for
  /// tests that want inactive timelines; the active count is then not steered.
  bool active_branches_only = true;
  /// Play until an action starts with at least this many boards.
  int target_boards = 240;
  /// Also keep the position at the first action start with at least each of
  /// these many boards (the flatness series: one game, longer and longer).
  std::vector<int> snapshots;
  /// Give up after this many actions.
  int max_actions = 4000;
};

/// A position the game passed through, at the start of an action.
struct Snapshot {
  int target = 0;            // the board count asked for
  std::size_t actions = 0;   // actions played to reach it
  GameState state;
};

enum class Stop : std::uint8_t {
  Target = 0,  // reached target_boards
  Stuck,       // no move and the action cannot be submitted
  MaxActions,  // max_actions played
};

const char* stop_name(Stop stop) noexcept;

struct Game {
  Spec spec;
  /// The seed of this game (Spec::seed, or a later one: play_until_reached).
  std::uint64_t seed = 0;
  /// How many games were played to get this one.
  int attempts = 1;
  GameState start;
  /// Every action played, in order: the moves as generated.
  std::vector<std::vector<Move>> actions;
  /// The position reached: the start of the action after the last.
  GameState state;
  std::vector<Snapshot> snapshots;
  Stop stop = Stop::Target;
};

/// Plays one game from spec.seed.
Game play(const Spec& spec);

/// Plays games from spec.seed, spec.seed + 1, ... until one reaches
/// target_boards (with every snapshot), at most `attempts` of them; returns the
/// first that does, or the last one played.
Game play_until_reached(const Spec& spec, int attempts = 32);

/// The 5dpgn record of the game's first `actions` actions (all of them by
/// default), in the reader's long form (pgn_move), with headers naming the
/// variant and labelling the record synthetic, with its seed and spec. The
/// engine's own reader (t5d/pgn.hpp) replays it to the same state - branch
/// origins included, which 5DFEN would lose.
std::string record(const Game& game, std::size_t actions = static_cast<std::size_t>(-1));

/// One move in 5dpgn, written in full so that it never needs disambiguating:
/// the source board, the piece letter (P for a pawn), the source square, then
/// "x" and the target square for a move within a board ("(0T1)Pe2e4",
/// "(0T4)Nf3xe5"), or ">" / ">>", "x", the target board and square for a jump
/// ("(0T5)Qd1>>x(0T3)d6"); "=Q" after a promotion. Castling is the king's
/// move ("(0T6)Ke1g1"). `even`: the labels of an even game ("-0", "+0").
std::string pgn_move(const Move& move, bool even);

/// The state after the first `count` actions, replayed from `start` (submitted
/// pseudo-legally, as they were played): the same
/// position as the game's, on boards of its own (what a benchmark thread
/// needs to expand its own copy).
GameState replay(const GameState& start, const std::vector<std::vector<Move>>& actions,
                 std::size_t count);

}  // namespace t5d::synthetic

#endif  // T5D_TOOLS_SYNTHETIC_HPP
