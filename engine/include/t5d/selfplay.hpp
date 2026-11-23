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
// detector within the game loop's node budget (H22, PlayParams::terminal_nodes;
// H37: a verdict only from an exhaustive search): checkmate (the side to move
// loses) or stalemate (a draw, rulebook C10) ends the game; a verdict the
// budget or the memory ceiling stops ends it Unknown - unscored, never a mate.
// Then SELF-PLAY'S TURN CAP (I11): a
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
//
// MATCH PLAY'S CLOCK (stage E11, t5d/timeman.hpp; PlayParams::clock, off): each
// side a clock of base seconds plus an increment after each of its actions
// (Fischer's, a reading - the 5D match clock is unspecified: build-log spec
// issue 456), each side's decisions run by its own TimeManager (run_timed)
// instead of fixed playouts. A decision is charged its whole wall time - the
// search, the choice and the exact verification of a chosen move - or, on the
// model clock, its work at the cost model's price (deterministic). A side whose
// clock runs out has lost on time (GameEnd::Flag: scored, not a rule - "*" in
// the record). Self-play never runs a clock: its budgets are PCR's (§12.5).
//
// SELF-PLAY RUNS WITHOUT THE ASSET MAP (the E10 review; build-log spec issues
// 179, 189): PlayParams' default search has asset_map OFF - a named placeholder
// - until I2's steered-visit definition is validated (the review measured the
// counterfactual count biasing the policy target against the boosted moves).
// The search's own default (SearchParams) and match play (t5d-play, Python's
// netplay) keep 6b''s table, the map on; a caller of play_game that wants the
// map in self-play sets search.asset_map itself.

#ifndef T5D_SELFPLAY_HPP
#define T5D_SELFPLAY_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "t5d/certificate.hpp"
#include "t5d/guard.hpp"
#include "t5d/mcts.hpp"
#include "t5d/move.hpp"
#include "t5d/state.hpp"
#include "t5d/timeman.hpp"

namespace t5d::mcts {

enum class GameEnd : std::uint8_t {
  Checkmate = 0,  ///< E4's detector: the side to move is checkmated
  Stalemate,      ///< E4's detector: a draw (C10)
  TurnCap,        ///< I11's convention: scored as a draw
  Unknown,        ///< unscored (H22): a question the game loop's budget left unsettled (GameRecord::unsettled)
  ActionLimit,    ///< PlayParams::max_actions: a safety stop, not a rule
  TimeLimit,      ///< PlayParams::max_seconds: a safety stop, not a rule
  Flag,           ///< stage E11: the side to move's clock ran out (PlayParams::clock) - a loss, not a rule
};
const char* game_end_name(GameEnd end) noexcept;

/// Register H22 (§11.0; review decision 13, 2026-11-15): which of the game
/// loop's two exact questions an Unknown game ended on - the turn start's
/// verdict, or the choice: the chosen move was not shown legal within the
/// budget, no other is known legal, and the moves played have left the
/// verdict's witness. None for every other end.
enum class Unsettled : std::uint8_t { None = 0, Verdict, Choice };
const char* unsettled_name(Unsettled u) noexcept;  ///< "", "verdict", "choice"

/// H22: the game loop's node budget for its exact questions, in search nodes
/// (SearchBudget's - deterministic, never a time limit). A PLACEHOLDER (build-log
/// issue 413): the spec gives no figure; it is SearchParams::mate_nodes' default, the budget
/// each fresh search's own root check runs under, so that the loop and its
/// searches cannot disagree (a test holds the two equal); Python's
/// netplay.TERMINAL_NODES is the same.
inline constexpr std::uint64_t kTerminalNodes = 2'000'000;

/// Self-play's search parameters: the search's defaults with root Dirichlet
/// noise ON (E1, §8.4) and the asset map OFF (a named placeholder: I2's steered
/// visits are unvalidated - issues 179, 189; the header comment).
SearchParams self_play_search_defaults();

/// Stage E11: match play's clock (the header comment). Every number a named
/// placeholder.
struct MatchClock {
  bool enabled = false;
  double base_seconds = 60;      ///< each side's time at the start (a placeholder no driver reads: issue 456)
  double increment_seconds = 0;  ///< added after each of a side's actions (issue 456)
  TimeParams time{};             ///< each side's time manager (6o's arms: allocator, signals)
  /// The model clock (run_timed with no clock): deterministic - for tests and
  /// reproducible runs; the wall clock otherwise.
  bool model_clock = false;
  /// A clock in place of the steady clock (tests: a virtual one; not owned,
  /// unused with model_clock): each decision is run (run_timed), charged and
  /// its Guard re-search bounded by it; the search is given no in-batch
  /// deadline (that is the steady clock's).
  Clock* clock = nullptr;
};

struct PlayParams {
  /// The searches' parameters - root Dirichlet noise ON by default (E1,
  /// §8.4: noise at the root in self-play); match play turns it off (§12.3).
  SearchParams search = self_play_search_defaults();
  /// Playouts per decision (the node budget).
  std::uint64_t playouts = 64;
  /// H22 (§11.0): the node budget of the game loop's two exact questions -
  /// each turn start's verdict (detect_mate) and the verification of a chosen
  /// move the expansion's budget left unknown (LegalityOptions' search nodes;
  /// tier 2's checks unlimited) - under the memory ceiling search.mate_bytes. A
  /// verdict it does not reach ends the game Unknown (never a mate, H37); a
  /// verification it leaves unknown gives way to the most visited child known
  /// legal - with none, to the next step of the turn-start verdict's witness
  /// while the action's moves are its prefix (review decision 13's X2;
  /// GameRecord::witness_moves) - and otherwise the game ends Unknown at the
  /// action's turn start (the action under way is not recorded). UINT64_MAX:
  /// exact (the behaviour before the ruling); 0 refused. Deterministic, unlike
  /// max_seconds. A PLACEHOLDER (kTerminalNodes; build-log issue 413).
  std::uint64_t terminal_nodes = kTerminalNodes;
  /// I11 - handed to the search too, which scores a capped position as a draw.
  /// The game ends a capped turn start as the rules come first: a verdict the
  /// game loop's budget does not settle ends it Unknown, not TurnCap - where
  /// the search, valuing that position, reads the cap as a draw (values only:
  /// the game's targets are masked; the review of decision 13, L9).
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
  /// given to each search as its deadline (Limits::seconds: the time left,
  /// recomputed before every run), and to the game's own rules-engine calls
  /// as theirs - each turn start's verdict and the exact verification of a
  /// chosen move (build-log issue 406); when it passes the game ends
  /// TimeLimit at the last turn start (an action under way is not recorded; a
  /// verdict it stopped ends the game at that turn start). After it no search
  /// is run and no choice made: a decision whose search it stopped, or with no
  /// child left to choose, ends the game there; a choice made before it whose
  /// exact verification it stops gives way to the most visited child known
  /// legal - or, with none, ends the game there. 0, or 1e9 or more (+infinity
  /// included; t5d/mcts.hpp's THE DEADLINE): none - and then the verdicts and
  /// verifications have no time limit, only terminal_nodes (H22). NaN or a
  /// negative value: refused. Not deterministic - for tools, never for
  /// training games.
  /// What it does not reach bounds its overshoot: the flat form's turn
  /// generator (its node budget, generator_nodes - about 1.3 s past a 0.5 s
  /// stop at issue 406's 18-board Q3 node, and 1.5-1.9 s at a 155-board turn
  /// start, as measured in the review of 2026-11-09); a fresh search's root
  /// check in its constructor (the turn-start detector under mate_nodes,
  /// repeating the verdict the loop has just made - 0.02-0.05 s at that Q3
  /// node; counted in the game's totals); the proof channel's paths (place()'s
  /// verification, check_certificate in offer and in Guard - off, H41); and
  /// the ladder's synchronous jobs (node-bounded, H15). It is the C++ play
  /// loop's alone: the C API's t5d_search_verify and its step calls take no
  /// deadline, and Python's play loops (netplay, self-play) set none.
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
  /// Stage E11: match play's clock - off (fixed playouts per decision). On,
  /// match play only (refused in self-play); `playouts` is then unread.
  MatchClock clock{};
  /// TESTS ONLY: applied to each refutation's certificate Guard's hook is about
  /// to have E4 check (guard_final) - the seam that feeds it a certificate E4
  /// rejects, so the "only confirmed refutations count" gate is exercised.
  /// Null (the default): none.
  std::function<void(std::shared_ptr<const ProofCertificate>&)> guard_certificate_hook_for_tests;
  /// TESTS ONLY: asked before each chosen move's verification with the number
  /// the game has made so far (GameRecord::verified_choices); true: none is
  /// made and the move stays Unknown, as if the budget had run out - the seam
  /// that leaves a choice unsettled after a given number of verifications (H22;
  /// the review of decision 13, F5). Null (the default): none.
  std::function<bool(std::uint64_t)> unsettle_verification_for_tests;
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
  /// An Unknown game's question (H22); None otherwise.
  Unsettled unsettled = Unsettled::None;
  /// 0 white, 1 black, -1 none (no optional: g++ 13/14 warn on copying one).
  int winner = -1;
  /// Moves (or submissions) played: under the board-move split one per move
  /// and one per submission (the tree form too); under the flat form one per
  /// turn. A decision whose choice stayed unsettled (H22) is not counted -
  /// its search's work is, in playouts and the rest; the decisions of an
  /// action dropped with it stay counted here and in decision_log (the audit
  /// lane's readers skip a decision past the last recorded action).
  std::uint64_t decisions = 0;
  std::uint64_t playouts = 0;
  std::uint64_t evaluations = 0;
  std::uint64_t nodes = 0;            ///< nodes created by the searches
  std::uint64_t search_turn_caps = 0; ///< capped positions the searches met (I11)
  /// Visits the decisions' roots already held when their searches began: the
  /// tree carried across decisions (C12) under match play; 0 in self-play,
  /// where every decision's search is fresh (C14).
  std::uint64_t reused_visits = 0;
  /// Chosen children the legality budget had left unknown, classified within
  /// terminal_nodes before they were played (a move is never played
  /// unverified: H22's budget, review decision 13).
  std::uint64_t verified_choices = 0;
  /// Moves played as steps of the turn-start verdict's witness, where no root
  /// child was known legal (H22; review decision 13's X2).
  std::uint64_t witness_moves = 0;
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
  /// Stage E11, match play's clock (PlayParams::clock): each side's time left
  /// at the end (White, Black; NaN: no clock), the timed decisions and why
  /// each stopped (by TimeStop), the extensions they took, and the decisions
  /// charged more than their hard bound, with the largest excess (seconds).
  std::array<double, 2> clock_remaining{std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t timed_decisions = 0;
  std::array<std::uint64_t, kTimeStops> time_stops{};
  std::uint64_t extensions = 0;
  std::uint64_t hard_overruns = 0;
  double max_overrun_seconds = 0;
  /// Stage E10, H44: each decision as the offline audit lane reads it
  /// (PlayParams::decision_log; empty otherwise) - tools/audit.hpp's events.
  std::vector<DecisionLog> decision_log;

  bool cap_bound() const noexcept { return end == GameEnd::TurnCap; }
  /// The 5dpgn result: "1-0", "0-1", "1/2-1/2" (stalemate), or "*" (not ended
  /// by the rules - the turn cap included, which self-play SCORES as a draw,
  /// and a flag, scored as the other side's win).
  std::string result() const;
  /// Self-play's score from White's view: 1, 0.5 or 0 (the cap a draw); -1 if unscored.
  double white_score() const noexcept;
};

/// Plays one game from `start` (a turn start).
GameRecord play_game(const GameState& start, Evaluator& evaluator, const PlayParams& params);

}  // namespace t5d::mcts

#endif  // T5D_SELFPLAY_HPP
