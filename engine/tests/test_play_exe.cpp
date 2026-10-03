// The t5d-play executable itself (tools/t5d_play.cpp): its record, piped into
// the gate adapter t5d-gate in full mode, replays to the end - every action
// legal by the engine's check ruling (R1), every position answered - and its
// command line. The build passes both executables' paths (T5D_PLAY_EXE,
// T5D_GATE_EXE) and a scratch directory (T5D_TEST_SCRATCH).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "t5d/pgn.hpp"
#include "test_framework.hpp"

namespace {

std::string read_all(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

// Runs `exe` with `args`, standard input from `in` (or none), standard output
// and error to files in the scratch directory; returns the exit status.
int run(const std::string& exe, const std::vector<std::string>& args, const std::filesystem::path& in,
        const std::filesystem::path& out, const std::filesystem::path& err) {
  namespace fs = std::filesystem;
  std::string command = quoted(fs::path(exe).make_preferred().string());
  for (const std::string& a : args) command += " " + quoted(a);
  if (!in.empty()) command += " < " + quoted(in.string());
  command += " > " + quoted(out.string()) + " 2> " + quoted(err.string());
#if defined(_WIN32)
  command = "\"" + command + "\"";
#endif
  std::fflush(nullptr);
  const int status = std::system(command.c_str());
#if !defined(_WIN32)
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#else
  return status;
#endif
}

std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> v;
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t nl = text.find('\n', at);
    if (nl == std::string::npos) nl = text.size();
    std::string line = text.substr(at, nl - at);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) v.push_back(line);
    at = nl + 1;
  }
  return v;
}

// Plays a game with t5d-play and replays its record with t5d-gate; returns
// the record as read.
t5d::PgnRecord play_and_gate(const std::vector<std::string>& play_args,
                             const std::vector<std::string>& gate_args) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "play-exe";
  fs::create_directories(dir);
  const fs::path record = dir / "game.5dpgn";
  const fs::path play_err = dir / "play-stderr.txt";
  const fs::path gate_out = dir / "gate-stdout.txt";
  const fs::path gate_err = dir / "gate-stderr.txt";
  if (run(T5D_PLAY_EXE, play_args, {}, record, play_err) != 0) {
    throw std::runtime_error("t5d-play failed: " + read_all(play_err));
  }
  const std::string text = read_all(record);
  const t5d::PgnRecord parsed = t5d::read_pgn(text);
  CHECK(parsed.header("Termination") != nullptr);
  CHECK(read_all(play_err).find("t5d-play: ") != std::string::npos);
  if (run(T5D_GATE_EXE, gate_args, record, gate_out, gate_err) != 0) {
    throw std::runtime_error("t5d-gate failed: " + read_all(gate_err));
  }
  const std::vector<std::string> lines = lines_of(read_all(gate_out));
  if (lines.empty()) throw std::runtime_error("t5d-gate printed nothing");
  const std::string expected_end = "{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":" +
                                   std::to_string(parsed.actions.size());
  CHECK(lines.back().rfind(expected_end, 0) == 0);
  // One state and one position event per position, the last one a game end
  // where the record says so.
  std::size_t positions = 0;
  for (const std::string& l : lines) {
    if (l.find("\"event\":\"position\"") != std::string::npos) ++positions;
  }
  CHECK_EQ(positions, parsed.actions.size() + 1);
  const std::string termination = *parsed.header("Termination");
  if (termination == "checkmate") CHECK(lines[lines.size() - 2].find("\"terminal\":\"checkmate\"") != std::string::npos);
  if (termination == "stalemate") CHECK(lines[lines.size() - 2].find("\"terminal\":\"stalemate\"") != std::string::npos);
  if (termination.rfind("turn_cap", 0) == 0) CHECK(lines[lines.size() - 2].find("\"terminal\":\"none\"") != std::string::npos);
  return parsed;
}

}  // namespace

TEST(play_exe_records_replay_through_the_gate_adapter) {
  play_and_gate({"--playouts", "8", "--cap-turns", "6", "--seed", "3"}, {});
  play_and_gate({"--split", "turn", "--playouts", "8", "--cap-turns", "5", "--seed", "4",
                 "--listing-cap", "128"},
                {});
  play_and_gate({"--variant", "Standard - Two Timelines", "--graph", "--match", "--playouts", "10",
                 "--cap-turns", "4", "--evaluator", "random"},
                {});
  play_and_gate({"--variant", "Very Small - Open", "--evaluator", "material", "--playouts", "16",
                 "--cap-turns", "12", "--seed", "2"},
                {});
  // The in-tree solver is off unless asked for (H23); the Engine header says so.
  const t5d::PgnRecord plain = play_and_gate({"--variant", "Very Small - Open", "--playouts", "16",
                                              "--cap-turns", "6", "--seed", "2"},
                                             {});
  REQUIRE(plain.header("Engine") != nullptr);
  CHECK(plain.header("Engine")->find("solver") == std::string::npos);
  const t5d::PgnRecord solved = play_and_gate({"--variant", "Very Small - Open", "--solver", "--playouts", "16",
                                               "--cap-turns", "6", "--seed", "2"},
                                              {});
  REQUIRE(solved.header("Engine") != nullptr);
  CHECK(solved.header("Engine")->find(", solver") != std::string::npos);
}

TEST(play_exe_takes_its_rules_through_the_shared_option_reader) {
  // The default is the engine's (Q4 as ruled, R8): no Rules header, and the
  // gate replays the record under its own default.
  const t5d::PgnRecord r8 = play_and_gate(
      {"--variant", "Standard - Two Timelines", "--playouts", "8", "--cap-turns", "4"}, {});
  CHECK(r8.header("Rules") == nullptr);
  // "--moves-after-shift" is accepted and changes nothing (tools::ruleset_option).
  const t5d::PgnRecord same = play_and_gate(
      {"--moves-after-shift", "--playouts", "8", "--cap-turns", "3"}, {});
  CHECK(same.header("Rules") == nullptr);
  // Q4's Guide-literal reading, by its option or by the encoding's bit 3: the
  // record names the rules, and replays under the gate's same option.
  for (const std::vector<std::string>& rules :
       {std::vector<std::string>{"--no-moves-after-shift"}, std::vector<std::string>{"--rules", "0x8"}}) {
    std::vector<std::string> args = rules;
    for (const char* a : {"--variant", "Standard - Two Timelines", "--playouts", "8", "--cap-turns", "4"}) {
      args.emplace_back(a);
    }
    const t5d::PgnRecord literal = play_and_gate(args, {"--no-moves-after-shift"});
    REQUIRE(literal.header("Rules") != nullptr);
    CHECK(literal.header("Rules")->find("moves_after_shift") != std::string::npos);
  }
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "play-exe";
  CHECK_EQ(run(T5D_PLAY_EXE, {"--rules", "nonsense"}, {}, dir / "o.txt", dir / "e.txt"), 2);
}

TEST(play_exe_command_line) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "play-exe";
  fs::create_directories(dir);
  CHECK_EQ(run(T5D_PLAY_EXE, {"--bogus"}, {}, dir / "o.txt", dir / "e.txt"), 2);
  CHECK(read_all(dir / "o.txt").empty());
  CHECK(read_all(dir / "e.txt").find("usage") != std::string::npos);
  CHECK_EQ(run(T5D_PLAY_EXE, {"--split", "sideways"}, {}, dir / "o.txt", dir / "e.txt"), 2);
  CHECK_EQ(run(T5D_PLAY_EXE, {"--variant", "No Such Variant"}, {}, dir / "o.txt", dir / "e.txt"), 2);
  // Values out of range are refused, never wrapped into a narrower type.
  for (const std::vector<std::string>& bad :
       {std::vector<std::string>{"--max-actions", "4294967296"}, std::vector<std::string>{"--cap-turns", "2147483648"},
        std::vector<std::string>{"--cap-boards", "18446744073709551615"}, std::vector<std::string>{"--max-seconds", "nan"},
        std::vector<std::string>{"--max-seconds", "-1"}, std::vector<std::string>{"--playouts", "0"},
        std::vector<std::string>{"--temperature", "inf"}, std::vector<std::string>{"--batch", "0"},
        std::vector<std::string>{"--listing-cap", "0"}}) {
    CHECK_EQ(run(T5D_PLAY_EXE, bad, {}, dir / "o.txt", dir / "e.txt"), 2);
    CHECK(read_all(dir / "e.txt").find("range") != std::string::npos ||
          read_all(dir / "e.txt").find("finite") != std::string::npos);
  }
  CHECK_EQ(run(T5D_PLAY_EXE, {"--help"}, {}, dir / "o.txt", dir / "e.txt"), 0);
  CHECK(read_all(dir / "o.txt").find("usage") != std::string::npos);
}
