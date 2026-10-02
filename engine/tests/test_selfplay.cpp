// Self-play end to end (t5d/selfplay.hpp; engine stage E6a; spec §15.5: "an
// engine that plays legal 5D chess end to end with random weights"): games of
// the engine against itself under the uniform (and material) evaluator, from
// the Standard start and other variants, under both D2 arms, in self-play
// (fresh searches, C14) and match play (the carried tree, C12). Every action
// is checked legal by E4's tests (each move pseudo-legal, each submission
// safe, C4/R1), every ending by E4's exhaustive detector (H37) or the turn
// cap's convention (I11), and every record replayed by the engine's 5dpgn
// reader to the same final state.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "play_record.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/search.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;
using namespace t5d::mcts;
using t5d_test::game;

namespace {

GameState variant(const char* name, const RuleOptions& rules = {}) {
  const Variant* v = find_variant(name);
  if (v == nullptr) throw std::logic_error(std::string("no variant ") + name);
  return variant_start(*v, rules);
}

PlayParams quick(TurnSplit split, std::uint64_t seed) {
  PlayParams p;
  p.search.expand.square_features = SquareFeatures::None;
  p.search.split = split;
  p.search.max_actions = 128;
  p.playouts = 12;
  p.turn_cap.turns = 8;
  p.turn_cap.boards = 300;  // branching can hold the present back
  p.seed = seed;
  return p;
}

// Replays a game through the state machine with E4's tests, and checks how it
// ended. Returns the final state.
GameState check_game(const GameRecord& g, const TurnCap& cap) {
  GameState s = g.start;
  MoveList scratch;
  for (const std::vector<Move>& action : g.actions) {
    REQUIRE(!action.empty());
    // Each action begins at a turn start the detector judged ongoing.
    REQUIRE(detect_mate(s, Detector::Hypercuboid).verdict == Outcome::Ongoing);
    for (const Move& m : action) {
      REQUIRE(s.may_move());
      REQUIRE(is_pseudo_legal(s, m, scratch));
      s.apply(m);
    }
    REQUIRE(s.submission_status() == SubmitStatus::Safe);  // T5 and C4 (R1)
    s.submit();
  }
  CHECK(s == g.final_state);
  const Outcome verdict = detect_mate(s, Detector::Hypercuboid).verdict;
  switch (g.end) {
    case GameEnd::Checkmate:
      CHECK(verdict == Outcome::Checkmate);
      CHECK_EQ(g.winner, index_of(opposite(s.to_move())));
      break;
    case GameEnd::Stalemate:
      CHECK(verdict == Outcome::Stalemate);
      CHECK_EQ(g.winner, -1);
      break;
    case GameEnd::TurnCap:
      CHECK(verdict == Outcome::Ongoing);  // the rules had not ended it
      CHECK(cap.reached(s));
      CHECK_EQ(g.winner, -1);
      CHECK_EQ(g.white_score(), 0.5);
      break;
    case GameEnd::Unknown:
    case GameEnd::ActionLimit:
      CHECK(false);  // neither may end these games
      break;
  }
  return s;
}

// The record, replayed by the engine's reader under the engine's check ruling
// (R1), reaches the same state.
void check_record(const GameRecord& g, const char* board, const RuleOptions& rules = {}) {
  const std::string text = tools::play_record(g, board);
  PgnReplay replay(read_pgn(text), PgnOptions{}, rules);
  CHECK_NOTHROW(replay.play_all());
  CHECK_EQ(replay.actions_played(), g.actions.size());
  CHECK(replay.state() == g.final_state);
  CHECK_EQ(fingerprint(replay.state().multiverse()).fingerprint,
           fingerprint(g.final_state.multiverse()).fingerprint);
}

}  // namespace

TEST(selfplay_games_are_legal_end_to_end_under_both_splits) {
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  int games = 0;
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const char* name : {"Standard", "Standard - Two Timelines", "Very Small - Open",
                             "Misc - Timeline Tactician"}) {
      const PlayParams p = quick(split, 17 + static_cast<std::uint64_t>(games));
      const GameRecord g = play_game(variant(name), *uniform, p);
      ++games;
      check_game(g, p.turn_cap);
      check_record(g, name);
      CHECK(g.decisions >= g.actions.size());
      CHECK(g.playouts > 0);
    }
  }
  CHECK_EQ(games, 8);
}

TEST(selfplay_games_end_by_checkmate_stalemate_or_the_cap) {
  // Small boards and a material evaluator make the rules end games; positions
  // one action from a mate and from a stalemate make sure both endings occur.
  int checkmates = 0;
  int stalemates = 0;
  int capped = 0;
  const auto tally = [&](const GameRecord& g) {
    checkmates += g.end == GameEnd::Checkmate ? 1 : 0;
    stalemates += g.end == GameEnd::Stalemate ? 1 : 0;
    capped += g.end == GameEnd::TurnCap ? 1 : 0;
  };
  const std::unique_ptr<Evaluator> material = make_evaluator(BuiltinEvaluator::Material);
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      PlayParams p = quick(split, seed);
      p.turn_cap.turns = 12;
      const GameRecord g = play_game(variant("Very Small - Open"), *material, p);
      check_game(g, p.turn_cap);
      check_record(g, "Very Small - Open");
      tally(g);
    }
    // White mates in one (Qb4#, Qa4#, Qe5#): the game ends by checkmate.
    {
      PlayParams p = quick(split, 5);
      p.playouts = 64;
      const GameRecord g = play_game(game({"[k4/4Q/1K3/5/5:0:1:w]"}, BoardSize{5, 5}), *uniform, p);
      check_game(g, p.turn_cap);
      check_record(g, "custom");
      CHECK(g.end == GameEnd::Checkmate);
      CHECK_EQ(g.winner, 0);
      CHECK_EQ(g.result(), std::string("1-0"));
      tally(g);
    }
    // White's only move, Kxc2, stalemates Black.
    {
      const PlayParams p = quick(split, 6);
      const GameRecord g = play_game(game({"[3k/4/B1qK/4:0:1:w]"}, BoardSize{4, 4}), *uniform, p);
      check_game(g, p.turn_cap);
      check_record(g, "custom");
      CHECK(g.end == GameEnd::Stalemate);
      CHECK_EQ(g.result(), std::string("1/2-1/2"));
      tally(g);
    }
  }
  CHECK(checkmates >= 2);
  CHECK(stalemates >= 2);
  CHECK(checkmates + stalemates + capped == 10);
}

TEST(selfplay_match_play_carries_the_tree) {
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Random, 4);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    for (const bool graph : {false, true}) {
      PlayParams p = quick(split, 9);
      p.match_play = true;
      p.search.graph = graph;
      p.turn_cap.turns = 5;
      const GameRecord g = play_game(variant("Standard - Two Timelines"), *uniform, p);
      check_game(g, p.turn_cap);
      check_record(g, "Standard - Two Timelines");
    }
  }
}

TEST(selfplay_under_q4s_other_reading_submits_by_choice) {
  // moves_after_shift (Q4's other reading): the board-move split offers
  // "submit" as a child; the record replays under the same rules.
  RuleOptions rules;
  rules.moves_after_shift = true;
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::WholeTurn}) {
    PlayParams p = quick(split, 3);
    p.turn_cap.turns = 4;
    const GameRecord g = play_game(variant("Standard - Two Timelines", rules), *uniform, p);
    check_game(g, p.turn_cap);
    check_record(g, "Standard - Two Timelines", rules);
  }
}

TEST(selfplay_plays_only_moves_known_legal_whatever_the_budget) {
  // The fast classification with no verification in search: many children stay
  // unknown, and the driver classifies a chosen one exactly before playing it.
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 6);
  for (const bool match : {false, true}) {
    PlayParams p = quick(TurnSplit::BoardMove, 12);
    p.search.legality = LegalityOptions::budgeted(0, 0);
    p.search.verify_unknown = false;
    p.match_play = match;
    p.turn_cap.turns = 6;
    for (const char* name : {"Standard - Two Timelines", "Very Small - Open"}) {
      const GameRecord g = play_game(variant(name), *ev, p);
      check_game(g, p.turn_cap);
      check_record(g, name);
    }
  }
}

TEST(selfplay_is_reproducible_and_counts_the_cap) {
  const std::unique_ptr<Evaluator> ev = make_evaluator(BuiltinEvaluator::Random, 2);
  PlayParams p = quick(TurnSplit::BoardMove, 44);
  p.search.root_noise = true;
  p.turn_cap.turns = 3;
  const GameRecord a = play_game(variant("Standard"), *ev, p);
  const GameRecord b = play_game(variant("Standard"), *ev, p);
  CHECK(a.actions == b.actions);
  CHECK(a.end == GameEnd::TurnCap);
  CHECK(a.cap_bound());
  CHECK_EQ(a.result(), std::string("*"));  // the rules did not end it
  CHECK(a.search_turn_caps > 0);           // the searches scored capped positions as draws
  check_game(a, p.turn_cap);
}
