// The synthetic positions' generator (tools/synthetic.hpp): deterministic for a
// seed, steering the active timelines, never taking a royal piece, and writing
// records the engine's own reader replays to the same state, branch origins
// included - with the capture mark written strictly (H51).

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "synthetic.hpp"
#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/pgn.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_rulesets.hpp"
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

// ---- the capture mark, written strictly (H51) ---------------------------------------------

namespace {

// Moves by kind, as the boards show them (check_mark).
struct MarkTally {
  int physical_captures = 0;
  int en_passant = 0;  // physical captures onto an empty square
  int jump_captures = 0;  // non-branching jumps
  int branching_captures = 0;
  int physical_quiet = 0;
  int jump_quiet = 0;
  int branching_quiet = 0;
};

// `m`, a pseudo-legal move in `s`, is written with "x" exactly when it takes a
// piece - judged from the boards, never from the move's flags: the opponent's
// pieces on the move's target board against those on the board the move makes
// there (the target's successor on its timeline, or the first board of the
// timeline a branching move creates).
void check_mark(const GameState& s, const Move& m, bool even, MarkTally& tally) {
  const Colour mover = s.to_move();
  const Board* target = s.multiverse().board_at(m.to_timeline, m.to_half_turn);
  REQUIRE(target != nullptr);
  const bool onto_empty = target->at(m.to_square).empty();
  const int before = std::popcount(target->pieces(opposite(mover)));
  GameState next = s;
  next.apply(m);
  const Multiverse& mv = next.multiverse();
  const int arrived = !m.branching()            ? m.to_timeline
                      : mover == Colour::White ? mv.max_timeline()
                                               : mv.min_timeline();
  const Board* made = mv.board_at(arrived, m.to_half_turn + 1);
  REQUIRE(made != nullptr);
  const int taken = before - std::popcount(made->pieces(opposite(mover)));
  CHECK(taken == 0 || taken == 1);
  const std::string text = synthetic::pgn_move(m, even);
  const auto marks = std::count(text.begin(), text.end(), 'x');
  if (marks != (taken == 1 ? 1 : 0)) {
    std::fprintf(stderr, "  %s: %d piece(s) taken, %d 'x' written\n", text.c_str(), taken,
                 static_cast<int>(marks));
  }
  CHECK_EQ(marks, static_cast<std::ptrdiff_t>(taken == 1 ? 1 : 0));
  CHECK_EQ(read_pgn_move(text).capture, taken == 1);  // the mark as the reader reads it
  if (m.physical()) {
    (taken == 1 ? tally.physical_captures : tally.physical_quiet)++;
    if (taken == 1 && onto_empty) {
      ++tally.en_passant;
      CHECK(m.en_passant());  // the only capture onto an empty square (S7)
    }
  } else if (!m.branching()) {
    (taken == 1 ? tally.jump_captures : tally.jump_quiet)++;
  } else {
    (taken == 1 ? tally.branching_captures : tally.branching_quiet)++;
  }
}

// Every move the game played, and every other pseudo-legal move it could have
// played where it played one, is written as check_mark says; the record
// replays with no capture-mark warning under the reader's default (H51) and to
// its end under the strict reading.
void check_capture_marks(const synthetic::Game& g, MarkTally& tally) {
  const bool even = g.start.multiverse().even();
  GameState s = g.start;
  MoveList offered;
  for (const std::vector<Move>& action : g.actions) {
    for (const Move& m : action) {
      generate_moves(s, offered);
      REQUIRE(std::find(offered.begin(), offered.end(), m) != offered.end());
      for (const Move& o : offered) check_mark(s, o, even, tally);
      s.apply(m);
    }
    s.submit(CheckRuling::none());  // as the game was played: pseudo-legal
  }
  CHECK(s == g.state);
  const std::string record = synthetic::record(g);
  PgnReplay lenient(read_pgn(record), PgnOptions{}, g.spec.rules);
  tools::parse_mode(lenient);
  lenient.play_all();
  CHECK(lenient.state() == g.state);
  if (!lenient.warnings().empty()) {
    std::fprintf(stderr, "  %s, seed %llu: %s\n", g.spec.variant.c_str(),
                 static_cast<unsigned long long>(g.seed), lenient.warnings().front().c_str());
  }
  CHECK(lenient.warnings().empty());
  PgnOptions strict;
  strict.capture_marks = CaptureMarks::Strict;
  PgnReplay letter(read_pgn(record, strict), strict, g.spec.rules);
  tools::parse_mode(letter);
  CHECK_NOTHROW(letter.play_all());
  CHECK(letter.state() == g.state);
}

}  // namespace

TEST(synthetic_moves_mark_exactly_the_captures_h51) {
  // H51 (rulebook R11): the engine writes the capture mark strictly. Random
  // games under several rulesets - the variants' set-ups under the default
  // rules, and every alternative rule option (t5d_test::alternative_rulesets)
  // in a two-timeline start - every move offered on the way checked, and the
  // games must between them offer every kind of move the mark is written on:
  // captures (en passant among them) and quiet moves, physical, by
  // non-branching jumps and by branching jumps.
  MarkTally tally;
  int games = 0;
  const auto play_and_check = [&](const char* variant, const RuleOptions& rules,
                                  std::uint64_t seed) {
    synthetic::Spec spec;
    spec.variant = variant;
    spec.rules = rules;
    spec.seed = seed;
    spec.target_boards = 70;
    spec.target_active = 4;
    spec.branch_percent = 30;
    spec.active_branches_only = seed % 2 == 0;  // inactive timelines in every other game
    spec.max_actions = 400;
    const synthetic::Game g = synthetic::play(spec);
    check_capture_marks(g, tally);
    ++games;
  };
  for (const char* variant :
       {"Standard", "Standard - Turn Zero", "Standard - Two Timelines",
        "Standard - Reversed Royalty", "Standard - Half Reflected", "Very Small - Open",
        "Focused - Just Pawns", "Focused - Just Brawns", "Misc - Timeline Battleground"}) {
    for (std::uint64_t seed = 1; seed <= 3; ++seed) play_and_check(variant, RuleOptions{}, seed);
  }
  std::uint64_t seed = 101;
  for (const RuleOptions& rules : t5d_test::alternative_rulesets()) {
    play_and_check("Standard - Two Timelines", rules, seed++);
  }
  std::fprintf(stderr,
               "  %d games, every move offered: captures %d physical (%d en passant), %d by "
               "jumps, %d by branches; quiet %d physical, %d jumps, %d branches\n",
               games, tally.physical_captures, tally.en_passant, tally.jump_captures,
               tally.branching_captures, tally.physical_quiet, tally.jump_quiet,
               tally.branching_quiet);
  CHECK(tally.physical_captures > 0);
  CHECK(tally.en_passant > 0);
  CHECK(tally.jump_captures > 0);
  CHECK(tally.branching_captures > 0);
  CHECK(tally.physical_quiet > 0);
  CHECK(tally.jump_quiet > 0);
  CHECK(tally.branching_quiet > 0);
}
