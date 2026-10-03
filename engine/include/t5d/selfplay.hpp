// t5d/selfplay.hpp - games of the engine against itself (engine stage E6a;
// design spec §8.7, §12.3, §12.5; register C12-C14, I11, E5).
//
// Two ways to play a decision:
//
//  * SELF-PLAY (the default): a FRESH search per decision (C14: no statistics
//    carried across moves - §12.2's forced-visit provenance and §12.5's
//    playout-cap accounting assume it), the move sampled from the root's
//    visit distribution at a temperature for the first decisions, then the
//    most visited (§12.3). State-keyed evaluations are shared between the
//    fresh searches (an EvalCache): they are facts about states, not tree
//    statistics.
//  * MATCH PLAY (match_play): one search carried from decision to decision -
//    advanced onto the chosen child, its statistics kept and everything the
//    new root cannot reach deleted (C12) - and the move chosen by the search's
//    final rule (E5, E9). Root noise is off (§12.3's match configuration).
//
// A game is a sequence of ACTIONS. Every turn start is first judged by E4's
// exhaustive detector (H37): checkmate (the side to move loses) or stalemate
// (a draw, rulebook C10) ends the game; an Unknown verdict (the detector's
// memory ceiling) ends it unjudged. Then SELF-PLAY'S TURN CAP (I11): a
// convention defined from the state (TurnCap), scored as a draw and counted -
// never a game rule. Every move the search chooses is checked pseudo-legal
// before it is played and every submission checked legal (T5, C4/R1): a
// failure is the engine's bug and throws std::logic_error.
//
// Under the board-move split an action is a sequence of DECISIONS - its moves,
// then submit, a decision of its own (register D8, rulebook R8): nothing is
// submitted automatically, and the action ends only when submit is chosen. A
// game's actions are counted at submissions (GameRecord::actions); the turn
// cap reads the state at each turn start, after the submission.

#ifndef T5D_SELFPLAY_HPP
#define T5D_SELFPLAY_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "t5d/mcts.hpp"
#include "t5d/move.hpp"
#include "t5d/state.hpp"

namespace t5d::mcts {

enum class GameEnd : std::uint8_t {
  Checkmate = 0,  ///< E4's detector: the side to move is checkmated
  Stalemate,      ///< E4's detector: a draw (C10)
  TurnCap,        ///< I11's convention: scored as a draw
  Unknown,        ///< the detector could not decide (H37): the game stops unjudged
  ActionLimit,    ///< PlayParams::max_actions: a safety stop, not a rule
  TimeLimit,      ///< PlayParams::max_seconds: a safety stop, not a rule
};
const char* game_end_name(GameEnd end) noexcept;

struct PlayParams {
  SearchParams search{};
  /// Playouts per decision (the node budget).
  std::uint64_t playouts = 64;
  /// I11 - handed to the search too, so that it scores capped positions as the
  /// game will.
  TurnCap turn_cap{};
  /// false: self-play (fresh searches, visit sampling); true: match play.
  bool match_play = false;
  /// Self-play: the temperature for the first temperature_decisions decisions,
  /// then the most visited child.
  double temperature = 1.0;
  std::uint32_t temperature_decisions = 16;
  std::uint64_t seed = 1;
  /// A safety stop: the most actions played (ActionLimit).
  std::uint32_t max_actions = 400;
  /// A safety stop: wall-clock seconds, checked at each turn start (TimeLimit);
  /// 0: none. Not deterministic - for tools, never for training games.
  double max_seconds = 0;
};

struct GameRecord {
  explicit GameRecord(const GameState& start_state) : start(start_state), final_state(start_state) {}

  GameState start;
  /// Every action, its moves in the order played.
  std::vector<std::vector<Move>> actions;
  GameState final_state;
  GameEnd end = GameEnd::Unknown;
  /// 0 white, 1 black, -1 none (no optional: g++ 13/14 warn on copying one).
  int winner = -1;
  /// Searches run: under the board-move split one per move and one per
  /// submission; under whole turns one per action.
  std::uint64_t decisions = 0;
  std::uint64_t playouts = 0;
  std::uint64_t evaluations = 0;
  std::uint64_t nodes = 0;            ///< nodes created by the searches
  std::uint64_t search_turn_caps = 0; ///< capped positions the searches met (I11)
  /// Visits the decisions' roots already held when their searches began: the
  /// tree carried across decisions (C12) under match play; 0 in self-play,
  /// where every decision's search is fresh (C14).
  std::uint64_t reused_visits = 0;
  /// Chosen children the legality budget had left unknown, classified exactly
  /// before they were played (a move is never played unverified).
  std::uint64_t verified_choices = 0;
  double seconds = 0;
  // Where the time went (the searches' SearchStats, summed; and the game's own
  // turn-start verdicts and exact verifications of chosen moves).
  double seconds_search_mate = 0;
  double seconds_search_expand = 0;
  double seconds_search_listing = 0;
  double seconds_search_verify = 0;
  double seconds_verdicts = 0;

  bool cap_bound() const noexcept { return end == GameEnd::TurnCap; }
  /// The 5dpgn result: "1-0", "0-1", "1/2-1/2" (stalemate), or "*" (not ended
  /// by the rules - the turn cap included, which self-play SCORES as a draw).
  std::string result() const;
  /// Self-play's score from White's view: 1, 0.5 or 0 (the cap a draw); -1 if unscored.
  double white_score() const noexcept;
};

/// Plays one game from `start` (a turn start).
GameRecord play_game(const GameState& start, Evaluator& evaluator, const PlayParams& params);

}  // namespace t5d::mcts

#endif  // T5D_SELFPLAY_HPP
