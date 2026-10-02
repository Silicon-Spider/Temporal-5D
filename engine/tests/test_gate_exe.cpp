// The t5d-gate executable itself (tools/t5d_gate.cpp), run as the harness runs
// it: a record on standard input, JSON lines on standard output, notes on
// standard error, and the exit status - its command line (--cap, the
// diagnostics, exit status 2 for a bad one), its input bytes (CR LF line
// ends, a byte-order mark: the reader's, in binary mode) and its exit status
// for each kind of end. ("error", exit status 1, cannot be reached from the
// command line - only through the counting seam: test_gate.cpp's
// gate_full_mode_ends_in_error_when_the_detectors_disagree.)
//
// The build passes the executable's path (T5D_GATE_EXE) and a scratch
// directory for the runs' files (T5D_TEST_SCRATCH).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "test_framework.hpp"

namespace {

struct ExeRun {
  int status = -1;
  std::string out;
  std::string err;
  std::vector<std::string> lines;  // standard output, split at LF
};

std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> v;
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t nl = text.find('\n', at);
    if (nl == std::string::npos) {
      v.push_back(text.substr(at));
      break;
    }
    v.push_back(text.substr(at, nl - at));
    at = nl + 1;
  }
  return v;
}

std::string read_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

// Runs t5d-gate with `args`, `input` as its standard input (bytes as given).
ExeRun run_gate(const std::vector<std::string>& args, const std::string& input) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "gate-exe";
  fs::create_directories(dir);
  const fs::path in = dir / "record.5dpgn";
  const fs::path out = dir / "stdout.txt";
  const fs::path err = dir / "stderr.txt";
  {
    std::ofstream f(in, std::ios::binary | std::ios::trunc);
    f.write(input.data(), static_cast<std::streamsize>(input.size()));
  }
  fs::remove(out);
  fs::remove(err);
  std::string command = quoted(fs::path(T5D_GATE_EXE).make_preferred().string());
  for (const std::string& a : args) command += " " + quoted(a);
  command += " < " + quoted(in.string()) + " > " + quoted(out.string()) + " 2> " + quoted(err.string());
#if defined(_WIN32)
  command = "\"" + command + "\"";  // cmd /c strips one pair of quotes around the whole
#endif
  std::fflush(nullptr);
  int status = std::system(command.c_str());
#if !defined(_WIN32)
  status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
  ExeRun r;
  r.status = status;
  r.out = read_file(out);
  r.err = read_file(err);
  r.lines = split_lines(r.out);
  return r;
}

const char* const kE3Position1 =
    "{\"event\":\"position\",\"index\":1,\"side\":\"black\",\"in_check\":false,\"legal_actions\":20,"
    "\"terminal\":\"none\"}";

}  // namespace

TEST(gate_exe_full_mode_and_its_cap) {
  // The contract's example: "1. e3", 20 legal actions at both positions.
  ExeRun r = run_gate({}, "1. e3\n");
  CHECK_EQ(r.status, 0);
  REQUIRE_EQ(r.lines.size(), std::size_t{5});
  CHECK_EQ(r.lines[3], std::string(kE3Position1));
  CHECK_EQ(r.lines[4], std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":1}"));
  CHECK(r.out.find('\r') == std::string::npos);  // binary: LF only
  CHECK(r.err.empty());
  // --cap N: min(count, N); 0 counts nothing.
  r = run_gate({"--cap", "5"}, "1. e3\n");
  CHECK_EQ(r.status, 0);
  CHECK(r.lines[3].find("\"legal_actions\":5,") != std::string::npos);
  r = run_gate({"--cap", "0"}, "");
  CHECK_EQ(r.status, 0);
  CHECK(r.lines[1].find("\"legal_actions\":0,\"terminal\":\"none\"") != std::string::npos);
  r = run_gate({"--cap", "18446744073709551615"}, "");  // the largest it reads
  CHECK_EQ(r.status, 0);
  CHECK(r.lines[1].find("\"legal_actions\":20,") != std::string::npos);
  // Parse mode: null verdicts.
  r = run_gate({"--parse-only", "--cap", "1000"}, "1. e3\n");
  CHECK_EQ(r.status, 0);
  CHECK(r.lines[3].find("\"legal_actions\":null") != std::string::npos);
}

TEST(gate_exe_refuses_a_bad_command_line_with_status_2) {
  const std::vector<std::vector<std::string>> bad = {
      {"--cap"},                                // no value
      {"--cap", "abc"},                         // not a number
      {"--cap", "-3"},                          // negative
      {"--cap", "18446744073709551616"},        // beyond 64 bits
      {"--cap", ""},                            // empty
      {"--no-such-option"},
      {"--detector", "random"},
      {"--rule", "no_such_rule=true"},
      {"--rule", "moves_after_shift=yes"},
      {"--pgn", "verify_annotations"},
      {"--actions", "x"},
  };
  for (const std::vector<std::string>& args : bad) {
    const ExeRun r = run_gate(args, "1. e3\n");
    CHECK_EQ(r.status, 2);
    CHECK(r.out.empty());                                   // nothing on standard output
    CHECK(r.err.find("usage: t5d-gate") != std::string::npos);  // why, and the usage
  }
  // --help and --version are not errors.
  ExeRun r = run_gate({"--help"}, "");
  CHECK_EQ(r.status, 0);
  CHECK(r.out.find("usage: t5d-gate") != std::string::npos);
  r = run_gate({"--version"}, "");
  CHECK_EQ(r.status, 0);
  CHECK(!r.out.empty());
}

TEST(gate_exe_reads_crlf_and_a_byte_order_mark) {
  // CR is white space to the reader, and a leading byte-order mark is
  // skipped (engine README, "Reading 5dpgn"): the same lines as plain LF.
  const ExeRun lf = run_gate({}, "[Board \"Standard\"]\n1. e3 / e6\n2. Qf3\n");
  REQUIRE_EQ(lf.status, 0);
  REQUIRE_EQ(lf.lines.size(), std::size_t{9});
  CHECK_EQ(lf.lines.back(), std::string("{\"event\":\"end\",\"status\":\"ok\",\"actions_accepted\":3}"));
  const ExeRun crlf = run_gate({}, "[Board \"Standard\"]\r\n1. e3 / e6\r\n2. Qf3\r\n");
  CHECK_EQ(crlf.status, 0);
  CHECK_EQ(crlf.out, lf.out);
  const ExeRun bom = run_gate({}, "\xEF\xBB\xBF[Board \"Standard\"]\r\n1. e3 / e6\r\n2. Qf3\r\n");
  CHECK_EQ(bom.status, 0);
  CHECK_EQ(bom.out, lf.out);
}

TEST(gate_exe_exit_status_for_each_end) {
  // ok, rejected and unsupported are answers: exit status 0, the end event
  // saying which (and a note on standard error for the last two).
  ExeRun r = run_gate({}, "1. e3\n");
  CHECK_EQ(r.status, 0);
  CHECK(r.lines.back().find("\"status\":\"ok\"") != std::string::npos);
  r = run_gate({}, "1. e4 / Nf3\n");  // Nf3 is White's move: illegal for Black
  CHECK_EQ(r.status, 0);
  CHECK_EQ(r.lines.back(), std::string("{\"event\":\"end\",\"status\":\"rejected\","
                                         "\"actions_accepted\":1,\"reason\":\"illegal_move\"}"));
  CHECK(r.err.find("rejected after 1 actions") != std::string::npos);
  r = run_gate({}, "1. e4 / f6 2. Qh5 / a6\n");  // leaves the king to the queen (C4, R1)
  CHECK_EQ(r.status, 0);
  CHECK(r.lines.back().find("\"reason\":\"cannot_submit\"") != std::string::npos);
  r = run_gate({}, "[Board \"Misc - Timeline Invasion\"]\n1. e3\n");
  CHECK_EQ(r.status, 0);
  REQUIRE_EQ(r.lines.size(), std::size_t{1});
  CHECK(r.lines[0].find("\"status\":\"unsupported\"") != std::string::npos);
  r = run_gate({}, "1. e3 ]]]\n");  // text the reader cannot read
  CHECK_EQ(r.status, 0);
  CHECK(r.lines.back().find("\"reason\":\"unreadable\"") != std::string::npos);
}
