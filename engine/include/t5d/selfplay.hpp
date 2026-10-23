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
// cap reads the state at each turn start, after the submission. Under D2's
// flat form a decision is a whole turn, played and submitted.
//
// THE VERIFICATION LADDER (stage E10, t5d/ladder.hpp): one per game, shared by
// its searches as the evaluation cache is - the asset map's assessments and the
// settling prover's table have the game's lifetime (8.7, H17), and a mate Hunt
// proves at an action's turn start is carried through the action's decisions.
// Its parts are the searches' (SearchParams' E10 block: Hunt under 100 boards,
// the settling scans under 50, the asset map - all on by default). GUARD (H42)
// is adopted in no range: its hook at submission is OFF (guard_final); on - for
// experiments, and only with the proof channel on, since a refutation is a
// proof - it rejects the chosen action and searches again. No reserve is held
// for it while Guard is offline (build-log spec issue 306).

#ifndef T5D_SELFPLAY_HPP
#define T5D_SELFPLAY_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "t5d/guard.hpp"
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
  /// The searches' parameters - root Dirichlet noise ON by default (E1,
  /// §8.4: noise at the root in self-play); match play turns it off (§12.3).
  SearchParams search = [] {
    SearchParams p;
    p.root_noise = true;
    return p;
  }();
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
  /// A safety stop: wall-clock seconds, checked before every decision and
  /// given to each search as its deadline (Limits::seconds); when it passes
  /// the game ends TimeLimit at the last turn start (an action under way is
  /// not recorded). 0: none. Not deterministic - for tools, never for
  /// training games.
  double max_seconds = 0;
  /// Stage E10: GUARD at submission (H42, spec 11.4: "before the chosen move
  /// is played, can the opponent force mate against it?") - ADOPTED IN NO
  /// RANGE (6b''): OFF. On (experiments), it needs the proof channel
  /// (search.proof_channel; play_game refuses it otherwise): the action that
  /// completes the turn is asked Guard's question under guard_ms (and
  /// guard_nodes; synchronous: guard_nodes, or the search's sync_job_nodes),
  /// and a refutation whose certificate E4 confirms is offered to the search as
  /// a proof - the action proven lost - which then searches again
  /// (guard_research_playouts; 0: playouts) and chooses again, at most
  /// guard_max_rejections times. Under the board-move split the rejection
  /// reaches the submission's decision only: the action's moves are played
  /// (a reading - build-log spec issue 185). Placeholders.
  bool guard_final = false;
  double guard_ms = 50;
  std::uint64_t guard_nodes = 0;
  GuardOptions guard{};
  std::uint64_t guard_research_playouts = 0;
  std::uint32_t guard_max_rejections = 4;
  /// Stage E10, H44: record each decision for the offline audit lane
  /// (GameRecord::decision_log). Off by default (memory).
  bool decision_log = false;
};

/// Stage E10, H44: one decision as the offline audit lane reads it - its
/// root's children (each one's decision, visits, backed-up value, whether it
/// was the most visited and its largest share of the root's visits), the PV
/// changes, the child chosen; the moves made in its action before it.
struct DecisionLog {
  std::uint32_t action = 0;    ///< the action's index in the game
  std::uint32_t decision = 0;  ///< the game's decision index
  std::vector<Move> prefix;    ///< the action's moves before the decision
  std::size_t boards = 0;      ///< the root's board count
  std::uint32_t root_visits = 0;
  double root_q = 0;           ///< the root's backed-up value, its player's view
  struct Child {
    PathStep step;
    std::uint32_t visits = 0;
    double q = 0;
    bool was_pv = false;
    double peak_share = 0;
  };
  std::vector<Child> children;
  std::vector<PvChange> pv_changes;
  std::size_t chosen = 0;  ///< into children
};

struct GameRecord {
  explicit GameRecord(const GameState& start_state) : start(start_state), final_state(start_state) {}

  GameState start;
  /// Every action, its moves in the order played.
  std::vector<std::vector<Move>> actions;
  GameState final_state;
  /// D2's arm the game was searched under (the audit lane's log names it).
  TurnSplit split = TurnSplit::BoardMove;
  GameEnd end = GameEnd::Unknown;
  /// 0 white, 1 black, -1 none (no optional: g++ 13/14 warn on copying one).
  int winner = -1;
  /// Searches run: under the board-move split one per move and one per
  /// submission (the tree form too); under the flat form one per turn.
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
  double seconds_search_generate = 0;  ///< D2's flat form: turns generated
  double seconds_search_verify = 0;
  double seconds_verdicts = 0;
  /// Stage E10: the verification ladder's work over the game (the searches'
  /// SearchStats, summed) and Guard's at submission (guard_final).
  std::uint64_t hunt_runs = 0;
  std::uint64_t hunt_mates = 0;
  std::uint64_t settling_runs = 0;
  std::uint64_t settling_proven = 0;
  std::uint64_t settling_disproven = 0;
  std::uint64_t refutations = 0;
  std::uint64_t proofs_withheld = 0;
  std::uint64_t proofs_injected = 0;
  std::uint64_t pv_changes = 0;
  std::uint64_t asset_boosted_edges = 0;
  std::uint64_t steered_visits = 0;
  std::uint64_t guard_runs = 0;
  std::uint64_t guard_refutations = 0;
  std::uint64_t guard_rejections = 0;  ///< refuted actions rejected and searched again
  /// Stage E10, H44: each decision as the offline audit lane reads it
  /// (PlayParams::decision_log; empty otherwise) - tools/audit.hpp's events.
  std::vector<DecisionLog> decision_log;

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
