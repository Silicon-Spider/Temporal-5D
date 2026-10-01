// The synthetic positions' generator (tools/synthetic.hpp): deterministic for a
// seed, steering the active timelines, never taking a royal piece, and writing
// records the engine's own reader replays to the same state, branch origins
// included.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "synthetic.hpp"
#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "tool_support.hpp"

using namespace t5d;

namespace {

synthetic::Spec spec_of(std::uint64_t seed, const char* variant = "Standard") {
  synthetic::Spec spec;
  spec.variant = variant;
  spec.seed = seed;
  spec.target_boards = 90;
  spec.target_active = 4;
  spec.branch_percent = 40;
  spec.snapshots = {30, 60};
  return spec;
}

int active_count(const GameState& s) {
  return s.multiverse().timeline_count() - static_cast<int>(s.inactive_timelines().size());
}

}  // namespace

TEST(synthetic_games_are_deterministic_for_a_seed) {
  const synthetic::Game a = synthetic::play(spec_of(7));
  const synthetic::Game b = synthetic::play(spec_of(7));
  REQUIRE(a.stop == synthetic::Stop::Target);
  CHECK(a.actions == b.actions);
  CHECK(a.state == b.state);
  CHECK_EQ(a.state.zobrist(), b.state.zobrist());
  CHECK_EQ(synthetic::record(a), synthetic::record(b));
  const synthetic::Game c = synthetic::play(spec_of(8));
  CHECK(a.actions != c.actions);
  // The generator's numbers are its own: the same on every platform.
  synthetic::Rng rng(1);
  const std::uint64_t first = rng.next();
  synthetic::Rng again(1);
  CHECK_EQ(again.next(), first);
}

TEST(synthetic_games_reach_their_targets_and_keep_snapshots) {
  const synthetic::Game g = synthetic::play_until_reached(spec_of(3));
  REQUIRE(g.stop == synthetic::Stop::Target);
  CHECK(g.state.multiverse().board_count() >= 90);
  REQUIRE_EQ(g.snapshots.size(), std::size_t{2});
  CHECK(g.snapshots[0].state.multiverse().board_count() >= 30);
  CHECK(g.snapshots[1].state.multiverse().board_count() >= 60);
  CHECK(g.snapshots[0].actions < g.snapshots[1].actions);
  // Snapshots are positions of the game, at the start of an action.
  for (const synthetic::Snapshot& s : g.snapshots) {
    CHECK(synthetic::replay(g.start, g.actions, s.actions) == s.state);
    CHECK(s.state.present_colour() == s.state.to_move());
  }
  // Steered: the active timelines reach the target and stay there.
  CHECK_EQ(active_count(g.state), 4);
  CHECK(g.state.inactive_timelines().empty());
  // No move took a royal piece (4.13).
  for (const std::vector<Move>& action : g.actions) {
    for (const Move& m : action) CHECK(!m.captured.royal());
  }
}

TEST(synthetic_records_replay_to_the_same_state) {
  for (const char* variant : {"Standard", "Standard - Two Timelines",
                              "Misc - Timeline Battleground", "Focused - Just Brawns"}) {
    synthetic::Spec spec = spec_of(11, variant);
    spec.target_boards = 60;
    spec.snapshots.clear();
    const synthetic::Game g = synthetic::play(spec);
    REQUIRE(!g.actions.empty());
    const std::string text = synthetic::record(g);
    const PgnRecord record = read_pgn(text);
    CHECK(!record.unreadable.has_value());
    CHECK(record.header("Synthetic") != nullptr);
    CHECK_EQ(record.actions.size(), g.actions.size());
    PgnReplay replay(record);
    tools::parse_mode(replay);  // the games are pseudo-legal
    replay.play_all();
    // Equal states: every board, every creator and every branch origin.
    CHECK(replay.state() == g.state);
    CHECK_EQ(fingerprint(replay.state().multiverse()).fingerprint,
             fingerprint(g.state.multiverse()).fingerprint);
    // A prefix of the record is the game's prefix.
    const std::string half = synthetic::record(g, g.actions.size() / 2);
    PgnReplay part(read_pgn(half));
    tools::parse_mode(part);
    part.play_all();
    CHECK(part.state() == synthetic::replay(g.start, g.actions, g.actions.size() / 2));
  }
}

TEST(synthetic_moves_are_written_in_full) {
  GameState s(standard_start());
  MoveList moves;
  generate_moves(s, moves);
  std::vector<std::string> texts;
  for (const Move& m : moves) texts.push_back(synthetic::pgn_move(m, false));
  CHECK(std::find(texts.begin(), texts.end(), "(0T1)Pe2e4") != texts.end());
  CHECK(std::find(texts.begin(), texts.end(), "(0T1)Ng1f3") != texts.end());
  GameState two(standard_two_timelines_start());
  generate_moves(two, moves);
  bool jump = false;
  for (const Move& m : moves) {
    const std::string t = synthetic::pgn_move(m, true);
    if (t == "(+0T1)Nb1>(-0T1)b3") jump = true;  // a knight's non-branching jump
  }
  CHECK(jump);
}

TEST(synthetic_games_are_played_under_their_ruleset) {
  // Standard - Turn Zero has a turn-zero history board (h 1, Black's): under
  // Q8's alternative it is no destination, so Black's first move list is
  // shorter, and the same seed plays another game.
  synthetic::Spec spec = spec_of(21, "Standard - Turn Zero");
  spec.target_boards = 30;
  spec.snapshots.clear();
  const synthetic::Game plain = synthetic::play(spec);
  spec.rules.turn_zero_reachable = false;
  const synthetic::Game q8 = synthetic::play(spec);
  CHECK(q8.start.rules() == spec.rules);
  CHECK(q8.state.rules() == spec.rules);
  REQUIRE(plain.actions.size() > 2);
  REQUIRE(q8.actions.size() > 2);
  CHECK(plain.actions != q8.actions);
  // Black's first position: fewer moves under Q8's alternative.
  GameState a = plain.start;
  for (const Move& m : plain.actions[0]) a.apply(m);
  a.submit(CheckRuling{});
  GameState b = q8.start;
  for (const Move& m : q8.actions[0]) b.apply(m);
  b.submit(CheckRuling{});
  if (a.multiverse() == b.multiverse()) {
    MoveList ma;
    MoveList mb;
    generate_moves(a, ma);
    generate_moves(b, mb);
    CHECK(mb.size() < ma.size());
  }
  // Its record names the ruleset (informational: the reader takes rules from
  // the tools' --rules, never from a header).
  CHECK(synthetic::record(q8).find("[Rules \"0x") != std::string::npos);
}
