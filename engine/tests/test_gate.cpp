// The gate adapter's protocol, parse mode and full mode (t5d/gate.hpp): the
// exact lines t5d-gate prints. The expected output is written here by hand
// from the rules (T2: one new board per move within a board, with the
// opponent to move; C4, C5, R1 for full mode's verdicts) and the notation's
// 5DFEN ("5DFEN and custom variants", "5DFEN Strictness") - never recorded
// from a run.

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/gate.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_reference.hpp"

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

TEST(gate_full_mode_matches_the_contracts_own_example) {
  // The engine contract's example, line for line: the Standard start and
  // "1. e3", both positions with 20 legal actions, no check, no end.
  const Run r = run("1. e3\n", /*parse_only=*/false);
  REQUIRE_EQ(r.lines.size(), std::size_t{5});
  CHECK_EQ(r.lines[0],
           std::string("{\"event\":\"state\",\"index\":0,\"size\":\"8x8\",\"markers\":true,"
                       "\"boards\":[\"0:1:w:" + kStandard + "\"]}"));
  CHECK_EQ(r.lines[1], std::string("{\"event\":\"position\",\"index\":0,\"side\":\"white\","
                                   "\"in_check\":false,\"legal_actions\":20,\"terminal\":\"none\"}"));
  CHECK_EQ(r.lines[3], std::string("{\"event\":\"position\",\"index\":1,\"side\":\"black\","
                                   "\"in_check\":false,\"legal_actions\":20,\"terminal\":\"none\"}"));
  CHECK_EQ(r.lines[4], std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":1}"));
  CHECK_EQ(r.status, 0);
}

namespace {

Run run_with(std::string_view record, const gate::Options& options) {
  Run r;
  r.status = gate::run(
      record, options, [&](const std::string& line) { r.lines.push_back(line); },
      [&](const std::string& note) { r.notes.push_back(note); });
  return r;
}

std::string full_position(int index, std::string_view side, bool check, int count,
                          std::string_view terminal) {
  return "{\"event\":\"position\",\"index\":" + std::to_string(index) + ",\"side\":\"" +
         std::string(side) + "\",\"in_check\":" + (check ? "true" : "false") +
         ",\"legal_actions\":" + std::to_string(count) + ",\"terminal\":\"" +
         std::string(terminal) + "\"}";
}

}  // namespace

TEST(gate_full_mode_reports_checkmate_and_stalemate) {
  // One board, Black to move (rulebook C5, R1): king a5 against queen b4
  // guarded by king b3 - checkmate; against a queen on c4 - stalemate.
  Run r = run("[Board \"custom\"][k4/1Q3/1K3/5/5:0:1:b]\n", false);
  REQUIRE_EQ(r.lines.size(), std::size_t{3});
  CHECK_EQ(r.lines[1], full_position(0, "black", true, 0, "checkmate"));
  CHECK_EQ(r.lines[2], std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":0}"));
  r = run("[Board \"custom\"][k4/2Q2/5/5/4K:0:1:b]\n", false);
  REQUIRE_EQ(r.lines.size(), std::size_t{3});
  CHECK_EQ(r.lines[1], full_position(0, "black", false, 0, "stalemate"));
  // A cross-board stalemate (C6): Black's king a5 on (0T1); White's rooks on
  // +1's board (T2, white) cover a4, b4, b5 of timeline 0 down the timelines.
  r = run("[Board \"custom\"][k4/5/5/5/5:0:1:b][1R3/RR3/5/5/5:+1:2:w]\n", false);
  REQUIRE_EQ(r.lines.size(), std::size_t{3});
  CHECK_EQ(r.lines[1], full_position(0, "black", false, 0, "stalemate"));
}

TEST(gate_full_mode_refuses_an_action_that_leaves_a_king_capturable) {
  // 1. e4 f6 2. Qh5+ a6: Black's last action leaves the king to the queen
  // (C4, R1): full mode refuses it - cannot_submit, after positions 0-3; parse
  // mode, pseudo-legal, plays it.
  const std::string text = "1. e4 / f6 2. Qh5 / a6\n";
  const Run full = run(text, false);
  REQUIRE_EQ(full.lines.size(), std::size_t{9});
  // Position 3: Black in check, with legal answers (g6, the king's f7, ...).
  CHECK(full.lines[7].find("\"index\":3,\"side\":\"black\",\"in_check\":true") !=
        std::string::npos);
  CHECK(full.lines[7].find("\"terminal\":\"none\"") != std::string::npos);
  CHECK_EQ(full.lines[8], std::string("{\"event\":\"end\",\"status\":\"rejected\","
                                      "\"actions_accepted\":3,\"reason\":\"cannot_submit\"}"));
  REQUIRE_EQ(full.notes.size(), std::size_t{1});
  CHECK(full.notes[0].find("(C4, R1)") != std::string::npos);
  const Run parse = run(text, true);
  CHECK_EQ(parse.lines.back(),
           std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":4}"));
}

TEST(gate_full_mode_caps_the_count) {
  gate::Options o;
  o.cap = 5;
  Run r = run_with("", o);
  REQUIRE_EQ(r.lines.size(), std::size_t{3});
  CHECK_EQ(r.lines[1], full_position(0, "white", false, 5, "none"));
  // Cap 0 counts nothing - but "terminal" still says whether an action exists.
  o.cap = 0;
  r = run_with("", o);
  CHECK_EQ(r.lines[1], full_position(0, "white", false, 0, "none"));
  r = run_with("[Board \"custom\"][k4/1Q3/1K3/5/5:0:1:b]\n", o);
  CHECK_EQ(r.lines[1], full_position(0, "black", true, 0, "checkmate"));
}

TEST(gate_full_mode_answers_with_either_detector) {
  // Two kings a side on -0 and +0 (33 legal actions: rulebook J4, J6; see
  // test_search.cpp), and a checkmate: the same answers from both detectors
  // and from the portfolio (the default), each cross-checked by the other.
  for (const std::optional<Detector> d :
       {std::optional<Detector>(Detector::Naive), std::optional<Detector>(Detector::Hypercuboid),
        std::optional<Detector>()}) {
    gate::Options o;
    o.detector = d;
    Run r = run_with("[Board \"custom\"][k4/5/5/5/K4:-0:1:w][k4/5/5/5/K4:+0:1:w]\n", o);
    REQUIRE_EQ(r.lines.size(), std::size_t{3});
    CHECK_EQ(r.lines[1], full_position(0, "white", false, 33, "none"));
    r = run_with("[Board \"custom\"][k4/5/5/5/R4:0:1:b][5/5/5/5/RR3:+1:1:b]\n", o);
    CHECK_EQ(r.lines[1], full_position(0, "black", true, 0, "checkmate"));  // C8
  }
}

TEST(gate_full_mode_counts_two_creation_orders_as_two_actions) {
  // The harness's definition of an action (t5d/search.hpp): two orders of two
  // branching moves are two actions, even where they reach one state. The
  // position of test_search.cpp's search_two_creation_orders_are_two_actions_j6
  // (1x1 boards; two rooks that must both branch back, White's creations
  // inactive): 2 legal actions, and --actions lists both orders.
  gate::Options o;
  o.list_actions = 0;
  const Run r = run_with("[Size \"1x1\"]\n[Board \"custom\"]\n"
                         "[1:-0:1:w][1:-0:1:b][R:-0:2:w][1:+0:1:w][1:+0:1:b][R:+0:2:w]"
                         "[1:+1:1:b][R:+1:2:w][1:+1:2:b]\n",
                         o);
  REQUIRE_EQ(r.lines.size(), std::size_t{3});
  CHECK_EQ(r.lines[1], full_position(0, "white", false, 2, "none"));
  const std::vector<std::string> notes = {
      "t5d-gate: position 0, White to move: 2 legal actions",
      "  1. (+0T2)Ra1(+0T1)a1 (-0T2)Ra1(-0T1)a1",
      "  2. (-0T2)Ra1(-0T1)a1 (+0T2)Ra1(+0T1)a1",
  };
  CHECK_EQ(r.notes, notes);
}

TEST(gate_full_mode_counts_thousands_of_actions_exactly) {
  // Misc - Timeline Battleground (V3): three timelines, each its own board of
  // White's to play - thousands of legal actions, counted to a cap above them
  // (many thousands of search nodes; no budget cuts the gate's count, H37)
  // and compared with the brute-force reference's count (test_reference.hpp).
  const std::string name = "Misc - Timeline Battleground";
  const Variant* v = find_variant(name);
  REQUIRE(v != nullptr);
  const t5d_test::Reference ref = t5d_test::reference_actions(variant_start(*v), 10000000);
  REQUIRE(!ref.overflow);
  REQUIRE(ref.actions.size() > 1000);
  for (const std::optional<Detector> d :
       {std::optional<Detector>(Detector::Naive), std::optional<Detector>(Detector::Hypercuboid),
        std::optional<Detector>()}) {
    gate::Options o;
    o.cap = 100000;
    o.detector = d;
    const Run r = run_with("[Board \"" + name + "\"]\n", o);
    REQUIRE_EQ(r.lines.size(), std::size_t{3});
    CHECK_EQ(r.lines[1], full_position(0, "white", false, static_cast<int>(ref.actions.size()), "none"));
  }
  // And the search behind it took many thousands of nodes.
  const ActionCount c = count_legal_actions(variant_start(*v), 100000, Detector::Hypercuboid);
  CHECK(c.stats.nodes > 5000);
}

TEST(gate_full_mode_ends_in_error_when_the_detectors_disagree) {
  // Spec §11.0 item 3: the detectors cross-check each other. Through the
  // counting seam, the cross-checking detector counts one more: the run ends
  // "error" (exit status 1) at position 0, after its state event, and says why.
  gate::Options o;
  o.detector = Detector::Hypercuboid;
  o.counter = [](const GameState& s, std::size_t cap, Detector d, const SearchBudget& b) {
    ActionCount c = count_legal_actions(s, cap, d, b);
    if (d == Detector::Naive && c.known()) ++c.count;
    return c;
  };
  const Run r = run_with("1. e3\n", o);
  CHECK_EQ(r.status, 1);
  REQUIRE_EQ(r.lines.size(), std::size_t{2});
  CHECK(r.lines[0].find("\"event\":\"state\",\"index\":0") != std::string::npos);
  CHECK_EQ(r.lines[1], std::string("{\"event\":\"end\",\"status\":\"error\",\"actions_accepted\":0,"
                                   "\"reason\":\"position 0: the detectors disagree: hypercuboid "
                                   "counts 20, naive counts 21 (cap 1000)\"}"));
  REQUIRE_EQ(r.notes.size(), std::size_t{1});
  CHECK(r.notes[0].find("the detectors disagree") != std::string::npos);
  // With the cross-check off, nothing disagrees.
  o.cross_check_nodes = 0;
  CHECK_EQ(run_with("1. e3\n", o).status, 0);
  // The portfolio is cross-checked the same way (the hypercuboid answers
  // first: it completes within the first budget).
  o.cross_check_nodes = 50000;
  o.detector.reset();
  CHECK_EQ(run_with("1. e3\n", o).status, 1);
}

TEST(gate_full_mode_ends_in_error_at_the_memory_ceiling) {
  // Every search keeps a memory ceiling (Options::max_bytes, --max-memory-mib;
  // SearchBudget::max_bytes): a count that reaches it is no verdict (H37) -
  // the run ends "error", saying so, with exit status 1. The portfolio stops
  // when both detectors have reached it; a named detector at once.
  for (const std::optional<Detector> d :
       {std::optional<Detector>(Detector::Naive), std::optional<Detector>(Detector::Hypercuboid),
        std::optional<Detector>()}) {
    gate::Options o;
    o.detector = d;
    o.max_bytes = 1;
    const Run r = run_with("1. e3\n", o);
    CHECK_EQ(r.status, 1);
    REQUIRE_EQ(r.lines.size(), std::size_t{2});
    CHECK_EQ(r.lines[1], std::string("{\"event\":\"end\",\"status\":\"error\",\"actions_accepted\":0,"
                                     "\"reason\":\"position 0: the search reached its memory ceiling "
                                     "(0 MiB) before it completed: no verdict (H37)\"}"));
  }
  // The default ceiling is far above what a position needs.
  CHECK_EQ(gate::Options{}.max_bytes, kDefaultSearchBytes);
  CHECK_EQ(run("1. e3\n", false).status, 0);
}

TEST(gate_diagnostics_list_actions_and_read_options) {
  // --actions K lists position K's legal actions on standard error (sorted
  // canonical texts); stdout is unchanged.
  gate::Options o;
  o.list_actions = 0;
  const Run r = run_with("[Board \"custom\"][k4/5/5/5/K4:-0:1:w][k4/5/5/5/K4:+0:1:w]\n", o);
  REQUIRE_EQ(r.notes.size(), std::size_t{34});
  CHECK(r.notes[0].find("position 0, White to move: 33 legal actions") != std::string::npos);
  CHECK(std::find(r.notes.begin(), r.notes.end(),
                  "  9. (+0T1)Ka1(+0T1)b2 (-0T1)Ka1(+0T1)b2") != r.notes.end());
  CHECK_EQ(r.lines.size(), std::size_t{3});
  // The rule and reader options, by name.
  RuleOptions rules;
  std::string error;
  CHECK(gate::set_rule_option(rules, "moves_after_shift=false", &error));
  CHECK(!rules.moves_after_shift);
  CHECK(gate::set_rule_option(rules, "moves_after_shift=true", &error));
  CHECK(rules.moves_after_shift);
  CHECK(gate::set_rule_option(rules, "activation=consecutive", &error));
  CHECK(rules.activation == ActivationRule::Consecutive);
  CHECK(gate::set_rule_option(rules, "promotions=QN", &error));
  CHECK(rules.promotions.contains(PieceKind::Knight));
  CHECK(!gate::set_rule_option(rules, "moves_after_shift=yes", &error));
  CHECK(!gate::set_rule_option(rules, "no_such_rule=true", &error));
  CHECK(error.find("no_such_rule") != std::string::npos);
  CHECK(!gate::set_rule_option(rules, "activation", &error));
  PgnOptions pgn;
  CHECK(gate::set_pgn_option(pgn, "verify_annotations=false", &error));
  CHECK(!pgn.verify_annotations);
  CHECK(gate::set_pgn_option(pgn, "castling_shortcuts=any_king", &error));
  CHECK(pgn.castling_shortcuts == CastlingShortcuts::AnyKing);
  CHECK(!gate::set_pgn_option(pgn, "castling_shortcuts=d_file", &error));
  // H51's capture-mark readings, and the option's spelling before H51.
  CHECK(PgnOptions{}.capture_marks == CaptureMarks::Lenient);
  CHECK(gate::set_pgn_option(pgn, "capture_marks=strict", &error));
  CHECK(pgn.capture_marks == CaptureMarks::Strict);
  CHECK(gate::set_pgn_option(pgn, "capture_marks=marked_must_capture", &error));
  CHECK(pgn.capture_marks == CaptureMarks::MarkedMustCapture);
  CHECK(gate::set_pgn_option(pgn, "capture_marks=lenient", &error));
  CHECK(pgn.capture_marks == CaptureMarks::Lenient);
  CHECK(!gate::set_pgn_option(pgn, "capture_marks=loose", &error));
  CHECK(error.find("lenient, marked_must_capture or strict") != std::string::npos);
  CHECK(gate::set_pgn_option(pgn, "require_capture_marks=true", &error));
  CHECK(pgn.capture_marks == CaptureMarks::Strict);
  CHECK(gate::set_pgn_option(pgn, "require_capture_marks=false", &error));
  CHECK(pgn.capture_marks == CaptureMarks::MarkedMustCapture);
  CHECK(!gate::set_pgn_option(pgn, "require_capture_marks=yes", &error));
  // Q4: an optional move after the shift - accepted by default (R8), refused
  // under the Guide-literal reading given through the options.
  gate::Options guide;
  guide.rules.moves_after_shift = false;
  const std::string late = "[Board \"custom\"][5/5/5/5/R4:0:1:w][5/5/5/5/R4:+1:2:w]\n"
                            "1. (0T1)Ra2 (+1T2)Ra2\n";
  CHECK(run(late, false).lines.back().find("\"status\":\"ok\"") != std::string::npos);
  CHECK(run_with(late, guide).lines.back().find("\"reason\":\"illegal_move\"") !=
        std::string::npos);
}

TEST(gate_capture_marks_are_read_leniently_h51) {
  // H51 (rulebook R11): a move written with 'x' that takes nothing is read as
  // its squares define it - the record is replayed to its end - and the
  // warning goes to the diagnostic channel (stderr), never into the protocol:
  // the events are the record's without the mark.
  const Run r = run("1. Nxf3 / Nc6\n");
  REQUIRE(!r.lines.empty());
  CHECK_EQ(r.lines.back(), std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":2}"));
  CHECK_EQ(r.lines, run("1. Nf3 / Nc6\n").lines);
  REQUIRE_EQ(r.notes.size(), std::size_t{1});
  CHECK(r.notes[0].rfind("t5d-gate: warning: action 1 (White's, turn 1), move 1 \"Nxf3\"", 0) == 0);
  CHECK(r.notes[0].find("takes no piece") != std::string::npos);
  // Full mode, likewise.
  const Run full = run("1. Nxf3 / Nc6\n", false);
  CHECK_EQ(full.lines, run("1. Nf3 / Nc6\n", false).lines);
  // The reading before H51 refuses it as unreadable (a diagnostic now).
  gate::Options marked;
  marked.pgn.capture_marks = CaptureMarks::MarkedMustCapture;
  const Run refused = run_with("1. Nxf3 / Nc6\n", marked);
  REQUIRE(!refused.lines.empty());
  CHECK_EQ(refused.lines.back(),
           std::string("{\"event\":\"end\",\"status\":\"rejected\",\"actions_accepted\":0,"
                       "\"reason\":\"unreadable\"}"));
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
