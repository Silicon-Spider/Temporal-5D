// The gate adapter's protocol in parse mode (t5d/gate.hpp): the exact lines
// t5d-gate prints. The expected output is written here by hand from the rules
// (T2: one new board per move within a board, with the opponent to move) and
// the notation's 5DFEN ("5DFEN and custom variants", "5DFEN Strictness") -
// never recorded from a run.

#include <string>
#include <string_view>
#include <vector>

#include "t5d/gate.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"

using namespace t5d;

namespace {

struct Run {
  int status = -1;
  std::vector<std::string> lines;
  std::vector<std::string> notes;
};

Run run(std::string_view record, bool parse_only = true) {
  Run r;
  gate::Options options;
  options.parse_only = parse_only;
  r.status = gate::run(
      record, options, [&](const std::string& line) { r.lines.push_back(line); },
      [&](const std::string& note) { r.notes.push_back(note); });
  return r;
}

const std::string kStandard = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*";

std::string position(int index, std::string_view side) {
  return "{\"event\":\"position\",\"index\":" + std::to_string(index) + ",\"side\":\"" +
         std::string(side) + "\",\"in_check\":null,\"legal_actions\":null,\"terminal\":null}";
}

}  // namespace

TEST(gate_parse_mode_prints_every_board_at_every_position) {
  // Two Timelines (V3): the standard board on -0 and +0. White: e3 on -0,
  // Nf3 (g1-f3) on +0; Black: d5 on both.
  const Run r = run("[Board \"Standard - Two Timelines\"]\n"
                    "1. (-0T1)e3 (+0T1)Nf3 / (-0T1)d5 (+0T1)d5\n");
  const std::string e3 = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/4P3/P*P*P*P*1P*P*P*/R*NBQK*BNR*";
  const std::string nf3 = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/5N2/P*P*P*P*P*P*P*P*/R*NBQK*B1R*";
  const std::string e3_d5 = "r*nbqk*bnr*/p*p*p*1p*p*p*p*/8/3p4/8/4P3/P*P*P*P*1P*P*P*/R*NBQK*BNR*";
  const std::string nf3_d5 = "r*nbqk*bnr*/p*p*p*1p*p*p*p*/8/3p4/8/5N2/P*P*P*P*P*P*P*P*/R*NBQK*B1R*";
  const std::vector<std::string> expected = {
      "{\"event\":\"state\",\"index\":0,\"size\":\"8x8\",\"markers\":true,\"boards\":[\"-0:1:w:" +
          kStandard + "\",\"+0:1:w:" + kStandard + "\"]}",
      position(0, "white"),
      "{\"event\":\"state\",\"index\":1,\"size\":\"8x8\",\"markers\":true,\"boards\":[\"-0:1:w:" +
          kStandard + "\",\"-0:1:b:" + e3 + "\",\"+0:1:w:" + kStandard + "\",\"+0:1:b:" + nf3 +
          "\"]}",
      position(1, "black"),
      "{\"event\":\"state\",\"index\":2,\"size\":\"8x8\",\"markers\":true,\"boards\":[\"-0:1:w:" +
          kStandard + "\",\"-0:1:b:" + e3 + "\",\"-0:2:w:" + e3_d5 + "\",\"+0:1:w:" + kStandard +
          "\",\"+0:1:b:" + nf3 + "\",\"+0:2:w:" + nf3_d5 + "\"]}",
      position(2, "white"),
      "{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":2}",
  };
  CHECK_EQ(r.lines, expected);
  CHECK_EQ(r.status, 0);
  CHECK(r.notes.empty());
}

TEST(gate_parse_mode_matches_the_contracts_own_example) {
  // The engine contract's example (the harness's manual): the Standard start
  // and "1. e3" - its state events, line for line.
  const Run r = run("1. e3\n");
  REQUIRE_EQ(r.lines.size(), std::size_t{5});
  CHECK_EQ(r.lines[0],
           std::string("{\"event\":\"state\",\"index\":0,\"size\":\"8x8\",\"markers\":true,"
                       "\"boards\":[\"0:1:w:r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/"
                       "R*NBQK*BNR*\"]}"));
  CHECK_EQ(r.lines[2],
           std::string("{\"event\":\"state\",\"index\":1,\"size\":\"8x8\",\"markers\":true,"
                       "\"boards\":[\"0:1:w:r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/"
                       "R*NBQK*BNR*\",\"0:1:b:r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/4P3/"
                       "P*P*P*P*1P*P*P*/R*NBQK*BNR*\"]}"));
  CHECK_EQ(r.lines[1], position(0, "white"));
  CHECK_EQ(r.lines[3], position(1, "black"));
  CHECK_EQ(r.lines[4], std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":1}"));
}

TEST(gate_full_mode_waits_on_the_check_ruling) {
  // The same end-event fields as parse mode's "unsupported": actions_accepted
  // (0) and the reason.
  const Run r = run("1. e3\n", /*parse_only=*/false);
  CHECK_EQ(r.lines, (std::vector<std::string>{
                        "{\"event\":\"end\",\"status\":\"unsupported\",\"actions_accepted\":0,"
                        "\"reason\":\"full mode waits on the check ruling\"}"}));
  CHECK_EQ(r.status, 0);
}

TEST(gate_rejections_name_their_reason) {
  // An illegal move after one accepted action: positions 0 and 1, then the end.
  Run r = run("1. e4 / Nf3\n");
  REQUIRE_EQ(r.lines.size(), std::size_t{5});
  CHECK_EQ(r.lines[3], position(1, "black"));
  CHECK_EQ(r.lines[4], std::string("{\"event\":\"end\",\"status\":\"rejected\","
                                   "\"actions_accepted\":1,\"reason\":\"illegal_move\"}"));
  REQUIRE_EQ(r.notes.size(), std::size_t{1});
  CHECK(r.notes[0].find("(no pseudo-legal move fits") != std::string::npos);
  CHECK_EQ(r.status, 0);

  // An action that does not shift the present (T5).
  r = run("[Board \"Standard - Two Timelines\"]\n1. (-0T1)e3\n");
  REQUIRE_EQ(r.lines.size(), std::size_t{3});
  CHECK_EQ(r.lines[2], std::string("{\"event\":\"end\",\"status\":\"rejected\","
                                   "\"actions_accepted\":0,\"reason\":\"cannot_submit\"}"));

  // Text the reader cannot read, after two actions.
  r = run("1. e4 / e5 2. ???\n");
  REQUIRE_EQ(r.lines.size(), std::size_t{7});
  CHECK_EQ(r.lines[6], std::string("{\"event\":\"end\",\"status\":\"rejected\","
                                   "\"actions_accepted\":2,\"reason\":\"unreadable\"}"));

  // A set-up that cannot be built - the headers contradict each other (the
  // way the specification's own Rook Tactics I is written: no Board, so
  // Standard, and a 5x5 Size), or cannot be read: no position at all.
  r = run("[Size \"5x5\"]\n1. e4\n");
  CHECK_EQ(r.lines, (std::vector<std::string>{"{\"event\":\"end\",\"status\":\"rejected\","
                                              "\"actions_accepted\":0,\"reason\":"
                                              "\"unreadable\"}"}));
  r = run("[Board \"Standard\"\n1. e4\n");
  CHECK_EQ(r.lines, (std::vector<std::string>{"{\"event\":\"end\",\"status\":\"rejected\","
                                              "\"actions_accepted\":0,\"reason\":"
                                              "\"unreadable\"}"}));
  REQUIRE_EQ(r.notes.size(), std::size_t{1});
  CHECK(r.notes[0].find("expected ']'") != std::string::npos);
}

TEST(gate_position_zero_is_reported_whenever_a_state_can_be_built) {
  // The set-up does not read the move text: a record whose first action is
  // by the player not to move still reports position 0, then is rejected at
  // action 1 - one position, actions_accepted 0 (the contract's n - 1).
  Run r = run("1b. e5\n");
  REQUIRE_EQ(r.lines.size(), std::size_t{3});
  CHECK(r.lines[0].rfind("{\"event\":\"state\",\"index\":0,", 0) == 0);
  CHECK_EQ(r.lines[1], position(0, "white"));
  CHECK_EQ(r.lines[2], std::string("{\"event\":\"end\",\"status\":\"rejected\","
                                   "\"actions_accepted\":0,\"reason\":\"unreadable\"}"));
  REQUIRE_EQ(r.notes.size(), std::size_t{1});
  CHECK(r.notes[0].find("it is White's turn (White moves first in \"Standard\", V2)") !=
        std::string::npos);
  // A custom position: the player to move is the present's colour (T4, T5).
  r = run("[Board \"custom\"][Size \"5x5\"][4k/5/5/5/K1R2:0:2:w][4k/5/5/5/K1R2:1:1:b]\n"
          "1. (0T2)Kb1\n");
  REQUIRE_EQ(r.lines.size(), std::size_t{3});
  CHECK_EQ(r.lines[0], std::string("{\"event\":\"state\",\"index\":0,\"size\":\"5x5\","
                                   "\"markers\":true,\"boards\":[\"0:2:w:4k/5/5/5/K1R2\","
                                   "\"+1:1:b:4k/5/5/5/K1R2\"]}"));
  CHECK_EQ(r.lines[1], position(0, "black"));
  CHECK_EQ(r.lines[2], std::string("{\"event\":\"end\",\"status\":\"rejected\","
                                   "\"actions_accepted\":0,\"reason\":\"unreadable\"}"));
  // The same record with Black's action first plays it: +1's board (T1,
  // black) is the present; then the present is (0T2, white).
  r = run("[Board \"custom\"][Size \"5x5\"][4k/5/5/5/K1R2:0:2:w][4k/5/5/5/K1R2:1:1:b]\n"
          "1b. (1T1)Kd5\n");
  REQUIRE_EQ(r.lines.size(), std::size_t{5});
  CHECK_EQ(r.lines[3], position(1, "white"));
  CHECK_EQ(r.lines[4], std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":1}"));
}

TEST(gate_unsupported_is_decided_whatever_else_is_wrong) {
  // A game the engine does not play is "unsupported" before any position,
  // whatever else is wrong with the record - faults in its headers included.
  for (const std::string_view text : {
           "[Board \"Simple - No Queens\"]\n[Event \"x\" junk]\n1. Nd3\n",
           "[Board \"Simple - No Queens\"][Size \"7x7\"][Size \"8x8\"]\n1. Nd3\n",
           "[Board \"Simple - No Queens\"][Size \"7x7\"][Size \"7X7\"] 1. (0T1)Nd3\n",
           "[Board \"x\"\n",
           "[Mode \"4D\"][Event \"x\" junk]\n",
           "[Board \"custom\"][8k/9/9/9/9/9/9/9/K8:0:1:w]\n",
           "[Board \"custom\"][9/9/9/9/9/9/9/9/K7k:0:1:w]\n",
       }) {
    const Run r = run(text);
    REQUIRE_EQ(r.lines.size(), std::size_t{1});
    CHECK(r.lines[0].rfind("{\"event\":\"end\",\"status\":\"unsupported\",\"actions_accepted\":0,"
                           "\"reason\":",
                           0) == 0);
    CHECK_EQ(r.status, 0);
  }
  // A Size that is not the named variant's is the record's contradiction,
  // whatever the numbers: the engine plays Standard.
  for (const std::string_view text :
       {"[Board \"Standard\"][Size \"10x10\"]\n", "[Size \"9x9\"]\n"}) {
    const Run r = run(text);
    REQUIRE_EQ(r.lines.size(), std::size_t{1});
    CHECK(r.lines[0].find("\"status\":\"rejected\"") != std::string::npos);
  }
}

TEST(gate_unsupported_variants_are_said_before_any_position) {
  const Run r = run("[Board \"Simple - No Queens\"]\n1. (0T1)Nd3\n");
  REQUIRE_EQ(r.lines.size(), std::size_t{1});
  CHECK_EQ(r.lines[0],
           std::string("{\"event\":\"end\",\"status\":\"unsupported\",\"actions_accepted\":0,"
                       "\"reason\":\"the variant \\\"Simple - No Queens\\\" is not one the engine "
                       "sets up (rules reference V3; see the engine README)\"}"));
  CHECK_EQ(r.status, 0);
  CHECK(run("[Mode \"4D\"]").lines[0].find("\"status\":\"unsupported\"") != std::string::npos);
}

TEST(gate_every_named_variant_reports_its_set_up) {
  // Position 0 of every variant: one state line listing the set-up's boards,
  // then the position line with White to move, then "ok".
  for (const Variant& v : variants()) {
    const Run r = run("[Board \"" + std::string(v.name) + "\"]\n");
    REQUIRE_EQ(r.lines.size(), std::size_t{3});
    const std::string size = std::to_string(v.size.width) + "x" + std::to_string(v.size.height);
    CHECK(r.lines[0].rfind("{\"event\":\"state\",\"index\":0,\"size\":\"" + size +
                               "\",\"markers\":true,\"boards\":[",
                           0) == 0);
    std::size_t boards = 0;
    for (std::size_t at = r.lines[0].find(":1:w:"); at != std::string::npos;
         at = r.lines[0].find(":1:w:", at + 1)) {
      ++boards;
    }
    const bool turn_zero = v.name == std::string_view("Standard - Turn Zero");
    CHECK_EQ(boards, v.boards.size() - (turn_zero ? 1 : 0));  // its T0 board is "0:0:b:..."
    CHECK_EQ(r.lines[1], position(0, "white"));
    CHECK_EQ(r.lines[2],
             std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":0}"));
  }
}

TEST(gate_json_strings_are_escaped_ascii) {
  CHECK_EQ(gate::json_string("plain"), std::string("\"plain\""));
  CHECK_EQ(gate::json_string("a \"b\" \\ c"), std::string("\"a \\\"b\\\" \\\\ c\""));
  CHECK_EQ(gate::json_string("line\nbreak\ttab"), std::string("\"line\\nbreak\\ttab\""));
  CHECK_EQ(gate::json_string(std::string_view("\x01\x1f\x7f", 3)),
           std::string("\"\\u0001\\u001f\\u007f\""));
  CHECK_EQ(gate::json_string("caf\xC3\xA9"), std::string("\"caf??\""));
  CHECK_EQ(gate::json_string(std::string_view("nul\0", 4)), std::string("\"nul\\u0000\""));
}
