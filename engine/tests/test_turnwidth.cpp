// The turn width of an action (register M22; tools/turn_width.hpp): its
// counts on constructed actions with known answers - Standard's first action,
// Two Timelines' (T13), a jump between two present boards (J4, A6's "two
// fewer"), a rewinding branch that releases a mandatory board (T8), moves
// after the shift (R8), moves on boards ahead of the present and on an
// inactive timeline (T6), a jump from an optional board onto a mandatory one
// (T6), a reactivation that makes a board mandatory mid-action, with a rewind
// (T12) and without one (T11) - and
// t5d-measure --turns / --turn-summary run end to end on the hand-written
// records of engine/testdata/turn-width (the build passes the executable's
// path, T5D_MEASURE_EXE, the engine directory, T5D_ENGINE_DIR, and a scratch
// directory, T5D_TEST_SCRATCH).
//
// Every position and record here is written for the test from the rules
// reference (docs/rules/README.md) and the notation specification's text -
// never taken from t0database, a game corpus or another implementation.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "t5d/fen.hpp"
#include "t5d/move.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "turn_width.hpp"

using namespace t5d;
using t5d::tools::classify_action;
using t5d::tools::TurnCounter;
using t5d::tools::TurnWidth;
using t5d_test::act;
using t5d_test::game;

namespace {

constexpr BoardSize k5x5{5, 5};

Move move_of(const GameState& state, const char* text) {
  const std::optional<Move> m = t5d_test::find_move(state, text);
  if (!m) throw std::runtime_error(std::string("no generated move ") + text);
  return *m;
}

// Counts an action given by its moves' export texts, each found among the
// moves generated for the state the earlier ones reach.
TurnWidth count(const GameState& start, std::initializer_list<const char*> texts) {
  TurnCounter counter(start);
  std::vector<Move> moves;
  for (const char* text : texts) {
    moves.push_back(move_of(counter.state(), text));
    counter.apply(moves.back());
  }
  // The convenience function counts the same.
  CHECK(classify_action(start, moves) == counter.width());
  return counter.width();
}

// 0's playable board at T2 (the present: mandatory); +1, White's first
// timeline (active), playable at T3 - ahead of the present; +2, White's
// second (inactive: Black has created none), playable at T1 - behind the
// present. Optional both (T6).
GameState optional_boards() {
  return game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/R4:0:2:w]",
               "[5/5/5/5/R4:+1:3:w]", "[5/5/5/5/R4:+2:1:w]"},
              k5x5);
}

// Two Timelines after 1. e3 e3 / e6 e6: White to move on both boards of T2.
GameState two_timelines_at_t2() {
  GameState s(standard_two_timelines_start());
  act(s, {"(-0T1)Pe2(-0T1)e3", "(+0T1)Pe2(+0T1)e3"});
  act(s, {"(-0T1)Pe7(-0T1)e6", "(+0T1)Pe7(+0T1)e6"});
  return s;
}

}  // namespace

TEST(turn_width_standard_first_action) {
  const GameState s(standard_start());
  const TurnWidth w = count(s, {"(0T1)Pe2(0T1)e4"});
  CHECK_EQ(w.boards, std::size_t{1});
  CHECK_EQ(w.timelines, 1);
  CHECK_EQ(w.active, 1);
  CHECK_EQ(w.mandatory, 1);
  CHECK_EQ(w.playable, 1);
  CHECK_EQ(w.optional, 0);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 0);
  CHECK_EQ(w.optional_when_made, 0);
  CHECK_EQ(w.boards_played, 1);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.jumps, 0);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.rewinds, 0);
  CHECK_EQ(w.mandatory_released, 0);
  CHECK_EQ(w.mandatory_gained, 0);
}

TEST(turn_width_two_timelines_first_action_t13) {
  // T13: both initial boards are in the present and mandatory. The second
  // move is made before the shift - one obligation is still open.
  const GameState s(standard_two_timelines_start());
  const TurnWidth w = count(s, {"(-0T1)Pe2(-0T1)e3", "(+0T1)Pe2(+0T1)e3"});
  CHECK_EQ(w.boards, std::size_t{2});
  CHECK_EQ(w.timelines, 2);
  CHECK_EQ(w.active, 2);
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.playable, 2);
  CHECK_EQ(w.optional, 0);
  CHECK_EQ(w.moves, 2);
  CHECK_EQ(w.moves_from_mandatory, 2);
  CHECK_EQ(w.optional_moves, 0);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 2);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.jumps_between_mandatory, 0);
}

TEST(turn_width_a_jump_between_two_present_boards_plays_both_j4) {
  // J4: a knight from +0 onto -0's playable board in the same column makes
  // a move on both boards - one move, two mandatory boards played (A6's
  // "two fewer"); the obligations are met by it alone.
  const GameState s(standard_two_timelines_start());
  TurnCounter c(s);
  c.apply(move_of(s, "(+0T1)Nb1(-0T1)b3"));
  CHECK(c.state().obligations_met());
  const TurnWidth w = c.width();
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 0);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 2);
  CHECK_EQ(w.optional_played, 0);
  CHECK_EQ(w.jumps, 1);
  CHECK_EQ(w.branches, 0);
  CHECK_EQ(w.jumps_between_mandatory, 1);
  CHECK_EQ(w.optional_onto_mandatory, 0);
  CHECK_EQ(w.mandatory_released, 0);
}

TEST(turn_width_a_rewinding_branch_releases_a_mandatory_board_t8) {
  // T8: the knight's branch from +0 back to T1 creates White's first
  // timeline, active, behind the present: the present moves back to it,
  // Black's colour, and -0's board - mandatory at the start - is released.
  const GameState s = two_timelines_at_t2();
  const TurnWidth w = count(s, {"(+0T2)Ng1(+0T1)g3"});
  CHECK_EQ(w.boards, std::size_t{6});
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.jumps, 1);
  CHECK_EQ(w.branches, 1);
  CHECK_EQ(w.rewinds, 1);
  CHECK_EQ(w.boards_played, 1);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.mandatory_released, 1);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.optional_moves, 0);
  // R8: the released board may still be moved on, after the shift. By its
  // status at the start the move is not optional; by its status when made it
  // is - counted apart, and the board is no longer released.
  const TurnWidth v = count(s, {"(+0T2)Ng1(+0T1)g3", "(-0T2)Pd2(-0T2)d3"});
  CHECK_EQ(v.moves, 2);
  CHECK_EQ(v.moves_from_mandatory, 2);
  CHECK_EQ(v.optional_moves, 0);
  CHECK_EQ(v.optional_when_made, 1);
  CHECK_EQ(v.moves_on_released, 1);
  CHECK_EQ(v.moves_after_shift, 1);
  CHECK_EQ(v.rewinds, 1);
  CHECK_EQ(v.mandatory_released, 0);
  CHECK_EQ(v.boards_played, 2);
}

TEST(turn_width_moves_after_the_shift_r8) {
  // R8: once 0's board is played the obligations are met; the move on +1,
  // ahead of the present, comes after the shift and is optional.
  const TurnWidth w = count(optional_boards(), {"(0T2)Ra1(0T2)a2", "(+1T3)Ra1(+1T3)a2"});
  CHECK_EQ(w.boards, std::size_t{5});
  CHECK_EQ(w.timelines, 3);
  CHECK_EQ(w.active, 2);
  CHECK_EQ(w.mandatory, 1);
  CHECK_EQ(w.playable, 3);
  CHECK_EQ(w.optional, 2);
  CHECK_EQ(w.moves, 2);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_when_made, 1);
  CHECK_EQ(w.moves_after_shift, 1);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.optional_played, 1);
}

TEST(turn_width_optional_boards_ahead_and_inactive_t6) {
  // T6: +1 (ahead of the present) and +2 (an inactive timeline, behind it)
  // may be played before the mandatory board - optional moves, not after
  // the shift.
  const TurnWidth w =
      count(optional_boards(), {"(+1T3)Ra1(+1T3)a2", "(+2T1)Ra1(+2T1)a2", "(0T2)Ra1(0T2)a2"});
  CHECK_EQ(w.optional, 2);
  CHECK_EQ(w.moves, 3);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 2);
  CHECK_EQ(w.optional_when_made, 2);
  CHECK_EQ(w.moves_after_shift, 0);
  CHECK_EQ(w.boards_played, 3);
  CHECK_EQ(w.optional_played, 2);
  CHECK_EQ(w.mandatory_released, 0);
}

TEST(turn_width_a_jump_from_an_optional_board_onto_a_mandatory_one_t6) {
  // T6: a mandatory board is also satisfied when a piece arrives on it. A
  // bishop on +1 (ahead of the present) jumps back one timeline and one turn
  // onto 0's playable board (J3: a playable board, so no branch): by its
  // source the move is optional, and it plays the mandatory board.
  const GameState s = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[5/5/5/5/5:0:2:w]",
                            "[5/5/5/5/B4:+1:3:w]"},
                           k5x5);
  REQUIRE(s.mandatory_timelines() == std::vector<int>({0}));
  TurnCounter c(s);
  c.apply(move_of(s, "(+1T3)Ba1(0T2)a1"));
  CHECK(c.state().obligations_met());
  const TurnWidth w = c.width();
  CHECK_EQ(w.mandatory, 1);
  CHECK_EQ(w.optional, 1);
  CHECK_EQ(w.moves, 1);
  CHECK_EQ(w.moves_from_mandatory, 0);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.jumps, 1);
  CHECK_EQ(w.branches, 0);
  CHECK_EQ(w.optional_onto_mandatory, 1);
  CHECK_EQ(w.jumps_between_mandatory, 0);
  CHECK_EQ(w.boards_played, 2);
  CHECK_EQ(w.mandatory_played, 1);
  CHECK_EQ(w.optional_played, 1);
  CHECK_EQ(w.mandatory_released, 0);
}

TEST(turn_width_a_reactivation_gains_a_mandatory_board_t12) {
  // T10-T12: White's second branch is inactive; Black's first activates it,
  // and its playable board - Black's, behind the present - becomes the
  // present: mandatory mid-action (gained), while +1's board, mandatory at
  // the start, now lies ahead and is released.
  GameState s = game({"[5/5/5/5/5:0:1:w]", "[5/5/5/5/5:0:1:b]", "[n4/5/5/5/5:0:2:w]",
                      "[5/5/5/5/K4:0:2:b]", "[4r/5/2R2/5/5:0:3:w]", "[5/5/5/5/R4:+1:3:w]"},
                     k5x5);
  // White's action: the inactive branch, then +1 (both from mandatory boards).
  const TurnWidth white = count(s, {"(0T3)Rc3(0T2)c3", "(+1T3)Ra1(+1T3)a2"});
  CHECK_EQ(white.mandatory, 2);
  CHECK_EQ(white.moves_from_mandatory, 2);
  CHECK_EQ(white.branches, 1);
  CHECK_EQ(white.rewinds, 0);  // T8: an inactive branch never moves the present
  act(s, {"(0T3)Rc3(0T2)c3", "(+1T3)Ra1(+1T3)a2"});
  REQUIRE(s.mandatory_timelines() == std::vector<int>({0, 1}));
  REQUIRE(!s.active(2));
  const TurnWidth w = count(s, {"(0T3)Re5(0T2)e5", "(+2T2)Na5(+2T2)b3"});
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.playable, 3);
  CHECK_EQ(w.optional, 1);  // +2: inactive at the start
  CHECK_EQ(w.moves, 2);
  CHECK_EQ(w.moves_from_mandatory, 1);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_when_made, 0);  // +2 was mandatory when its move was made
  CHECK_EQ(w.moves_on_gained, 1);
  CHECK_EQ(w.mandatory_gained, 1);
  CHECK_EQ(w.rewinds, 1);
  CHECK_EQ(w.mandatory_released, 1);  // +1
  CHECK_EQ(w.branches, 1);
  CHECK_EQ(w.moves_after_shift, 0);
}

TEST(turn_width_a_reactivation_in_the_present_gains_a_board_without_a_rewind_t11) {
  // Black to move on 0 and +1 at T3 (the present); +2, White's second timeline,
  // inactive (Black has created none), also has Black's board at T3. Black
  // plays 0, then jumps from +1 onto 0's T3 board - history now, so the jump
  // branches (J6) - creating Black's first timeline, active, whose first board
  // (T4, White's) lies after the present: no rewind. The creation activates
  // +2 (T11), whose board is in the present: mandatory mid-action, without the
  // present moving.
  const GameState s = game({"[4r/5/5/5/5:0:3:b]", "[r4/5/5/5/5:+1:3:b]", "[1r3/5/5/5/5:+2:3:b]"},
                           k5x5);
  REQUIRE(s.to_move() == Colour::Black);
  REQUIRE(!s.active(2));
  REQUIRE(s.mandatory_timelines() == std::vector<int>({0, 1}));
  TurnCounter c(s);
  c.apply(move_of(c.state(), "(0T3)Re5(0T3)e4"));
  const Move branch = move_of(c.state(), "(+1T3)Ra5(0T3)a5");
  CHECK(branch.branching());
  c.apply(branch);
  CHECK(c.state().active(2));
  CHECK(c.state().mandatory_timelines() == std::vector<int>({2}));
  CHECK(!c.state().obligations_met());
  c.apply(move_of(c.state(), "(+2T3)Rb5(+2T3)b4"));
  CHECK(c.state().obligations_met());
  const TurnWidth w = c.width();
  CHECK_EQ(w.mandatory, 2);
  CHECK_EQ(w.playable, 3);
  CHECK_EQ(w.optional, 1);
  CHECK_EQ(w.moves, 3);
  CHECK_EQ(w.moves_from_mandatory, 2);
  CHECK_EQ(w.optional_moves, 1);
  CHECK_EQ(w.optional_when_made, 0);
  CHECK_EQ(w.moves_on_gained, 1);
  CHECK_EQ(w.mandatory_gained, 1);
  CHECK_EQ(w.rewinds, 0);
  CHECK_EQ(w.mandatory_released, 0);
  CHECK_EQ(w.branches, 1);
  CHECK_EQ(w.boards_played, 3);
}

TEST(turn_width_a_move_that_does_not_fit_changes_nothing) {
  GameState s(standard_start());
  const Move second = move_of(s, "(0T1)Pd2(0T1)d4");
  TurnCounter c(s);
  c.apply(move_of(s, "(0T1)Pe2(0T1)e4"));
  const TurnWidth before = c.width();
  const GameState state_before = c.state();
  // T3: the board has been played - its move no longer fits.
  CHECK_THROWS_AS(c.apply(second), std::invalid_argument);
  CHECK(c.width() == before);
  CHECK(c.state() == state_before);
}

// ---- t5d-measure --turns / --turn-summary, end to end -----------------------------------------

namespace {

std::string read_all(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> v;
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t nl = text.find('\n', at);
    if (nl == std::string::npos) nl = text.size();
    if (nl > at) v.push_back(text.substr(at, nl - at));
    at = nl + 1;
  }
  return v;
}

// Runs t5d-measure with `args`; standard output's lines, and the exit status.
std::vector<std::string> run_measure(const std::vector<std::string>& args, int& status) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "measure-exe";
  fs::create_directories(dir);
  const fs::path out = dir / "stdout.txt";
  const fs::path err = dir / "stderr.txt";
  fs::remove(out);
  fs::remove(err);
  std::string command = quoted(fs::path(T5D_MEASURE_EXE).make_preferred().string());
  for (const std::string& a : args) command += " " + quoted(a);
  command += " > " + quoted(out.string()) + " 2> " + quoted(err.string());
#if defined(_WIN32)
  command = "\"" + command + "\"";  // cmd /c strips one pair of quotes around the whole
#endif
  std::fflush(nullptr);
  status = std::system(command.c_str());
#if !defined(_WIN32)
  status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
  return lines_of(read_all(out));
}

// The integer value of "name": in a JSON line (-1 if absent or not one).
long long int_field(const std::string& line, const std::string& name) {
  for (const char* before : {"{", ","}) {
    const std::string key = before + ("\"" + name + "\":");
    const std::size_t at = line.find(key);
    if (at == std::string::npos) continue;
    std::size_t p = at + key.size();
    std::size_t end = p;
    while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) != 0 ||
                                 (end == p && line[end] == '-'))) {
      ++end;
    }
    if (end == p) return -1;
    return std::stoll(line.substr(p, end - p));
  }
  return -1;
}

bool has(const std::string& line, const std::string& text) {
  return line.find(text) != std::string::npos;
}

std::vector<std::string> with(const std::vector<std::string>& lines, const std::string& text) {
  std::vector<std::string> out;
  for (const std::string& l : lines) {
    if (has(l, text)) out.push_back(l);
  }
  return out;
}

}  // namespace

TEST(measure_exe_turns_on_hand_written_records) {
  // two-timelines.5dpgn (Standard - Two Timelines), four actions:
  //   1. (-0T1)e3 (+0T1)e3       two mandatory boards (T13), two moves
  //    / (+0T1)Nb8>(-0T1)b6       a jump between the two present boards (J4)
  //   2. (+0T2)Ng1>>(+0T1)g3      a rewinding branch (T8), then a move on the
  //      (-0T2)d3                 board it released, after the shift (R8)
  //    / (+1T1)e6 (-0T2)d6        the new timeline's board is the present's;
  //                               -0's is ahead: an optional move, after the shift
  // prefix.5dpgn: its first action pair - a record that repeats the first.
  // refused.5dpgn (Standard): 1. e4 / e5 2. Ke3 - refused at its third action.
  namespace fs = std::filesystem;
  const fs::path data = fs::path(T5D_ENGINE_DIR) / "testdata" / "turn-width";
  const std::string two = (data / "two-timelines.5dpgn").string();
  const std::string prefix = (data / "prefix.5dpgn").string();
  const std::string refused = (data / "refused.5dpgn").string();
  int status = -1;
  const std::vector<std::string> lines =
      run_measure({"--turns", "--turn-summary", two, prefix, refused}, status);
  CHECK_EQ(status, 0);

  // Every action is played twice and checked; a mismatch would end the
  // record in "error" (exit status 1).
  const std::vector<std::string> ends = with(lines, "\"event\":\"end\"");
  REQUIRE_EQ(ends.size(), std::size_t{3});
  CHECK(has(ends[0], "\"status\":\"ok\""));
  CHECK(has(ends[1], "\"status\":\"ok\""));
  CHECK(has(ends[2], "\"status\":\"rejected\""));
  CHECK(has(ends[2], "\"reason\":\"illegal_move\""));
  CHECK_EQ(int_field(ends[2], "actions_accepted"), 2);

  std::vector<std::string> turns;
  for (const std::string& l : with(lines, "\"event\":\"turn\"")) {
    if (has(l, "two-timelines.5dpgn")) turns.push_back(l);
  }
  REQUIRE_EQ(turns.size(), std::size_t{4});
  struct Expected {
    const char* side;
    long long boards, timelines, mandatory, playable, optional, moves, from_mandatory,
        optional_moves, when_made, played, jumps, branches, between, after_shift, rewinds,
        released, on_released;
  };
  const Expected expected[4] = {
      {"white", 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0},
      {"black", 4, 2, 2, 2, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0},
      {"white", 6, 2, 2, 2, 0, 2, 2, 0, 1, 2, 1, 1, 0, 1, 1, 0, 1},
      {"black", 9, 3, 1, 3, 2, 2, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0},
  };
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string& t = turns[i];
    const Expected& e = expected[i];
    CHECK_EQ(int_field(t, "index"), static_cast<long long>(i));
    CHECK(has(t, std::string("\"side\":\"") + e.side + "\""));
    CHECK_EQ(int_field(t, "boards"), e.boards);
    CHECK_EQ(int_field(t, "timelines"), e.timelines);
    CHECK_EQ(int_field(t, "mandatory"), e.mandatory);
    CHECK_EQ(int_field(t, "playable"), e.playable);
    CHECK_EQ(int_field(t, "optional"), e.optional);
    CHECK_EQ(int_field(t, "moves"), e.moves);
    CHECK_EQ(int_field(t, "moves_from_mandatory"), e.from_mandatory);
    CHECK_EQ(int_field(t, "optional_moves"), e.optional_moves);
    CHECK_EQ(int_field(t, "optional_when_made"), e.when_made);
    CHECK_EQ(int_field(t, "boards_played"), e.played);
    CHECK_EQ(int_field(t, "jumps"), e.jumps);
    CHECK_EQ(int_field(t, "branches"), e.branches);
    CHECK_EQ(int_field(t, "jumps_between_mandatory"), e.between);
    CHECK_EQ(int_field(t, "moves_after_shift"), e.after_shift);
    CHECK_EQ(int_field(t, "rewinds"), e.rewinds);
    CHECK_EQ(int_field(t, "mandatory_released"), e.released);
    CHECK_EQ(int_field(t, "moves_on_released"), e.on_released);
  }
  // Each action's position after it is the next action's start.
  for (std::size_t i = 0; i + 1 < 4; ++i) {
    const std::size_t a = turns[i].find("\"fingerprint_after\":\"");
    const std::size_t b = turns[i + 1].find("\"fingerprint\":\"");
    REQUIRE(a != std::string::npos && b != std::string::npos);
    CHECK_EQ(turns[i].substr(a + 21, 32), turns[i + 1].substr(b + 15, 32));
  }

  // The summary: prefix.5dpgn repeats two-timelines.5dpgn, so the strict
  // reading counts one game of four actions; the refused record's two
  // actions are apart.
  const std::vector<std::string> head = with(lines, "\"event\":\"turn_summary_records\"");
  REQUIRE_EQ(head.size(), std::size_t{1});
  CHECK_EQ(int_field(head[0], "records"), 3);
  CHECK_EQ(int_field(head[0], "replayed"), 2);
  CHECK_EQ(int_field(head[0], "refused"), 1);
  CHECK_EQ(int_field(head[0], "actions_checked"), 4 + 2 + 2);
  CHECK_EQ(int_field(head[0], "strict_games"), 1);
  CHECK_EQ(int_field(head[0], "grouped_games"), 1);
  CHECK(has(head[0], "prefix.5dpgn\",\"repeats\":"));
  std::string strict, grouped, apart, by_two;
  for (const std::string& l : lines) {
    if (has(l, "\"event\":\"turn_summary\",\"reading\":\"strict\",\"range\":\"<50\"")) strict = l;
    if (has(l, "\"event\":\"turn_summary\",\"reading\":\"grouped\",\"range\":\"<50\"")) grouped = l;
    if (has(l, "\"event\":\"turn_summary\",\"reading\":\"refused\",\"range\":\"<50\"")) apart = l;
    if (has(l, "\"reading\":\"strict\",\"range\":\"<50\",\"mandatory_class\":\"2\"")) by_two = l;
  }
  REQUIRE(!strict.empty() && !grouped.empty() && !apart.empty() && !by_two.empty());
  CHECK_EQ(int_field(strict, "actions"), 4);
  CHECK_EQ(int_field(strict, "games"), 1);
  CHECK(has(strict, "\"mandatory\":{\"1\":1,\"2\":3,\"3\":0,\"4+\":0}"));
  CHECK(has(strict, "\"mandatory_over_1\":0.75,"));
  CHECK(has(strict, "\"optional_moves\":{\"0\":3,\"1\":1,\"2\":0,\"3+\":0}"));
  CHECK(has(strict, "\"moves\":{\"1\":1,\"2\":3,\"3\":0,\"4+\":0}"));
  CHECK(has(strict, "\"moves_mean\":1.75,"));
  CHECK(has(strict, "\"with_branch\":0.25,"));
  CHECK_EQ(int_field(strict, "jumps_between_mandatory"), 1);
  CHECK_EQ(int_field(grouped, "actions"), 4);
  CHECK_EQ(int_field(apart, "actions"), 2);
  CHECK_EQ(int_field(apart, "records"), 1);
  CHECK_EQ(int_field(by_two, "actions"), 3);
  // Nothing at 50 boards or more.
  for (const std::string& l : with(lines, "\"reading\":\"strict\",\"range\":\"50-99\"")) {
    CHECK_EQ(int_field(l, "actions"), 0);
  }

  // Without --turns and --turn-summary the tool prints no turn line.
  const std::vector<std::string> plain = run_measure({two}, status);
  CHECK_EQ(status, 0);
  CHECK(with(plain, "\"event\":\"turn").empty());
  CHECK_EQ(with(plain, "\"event\":\"position\"").size(), std::size_t{5});
}
