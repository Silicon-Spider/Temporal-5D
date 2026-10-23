// The H41 cross-check tools as executables (register H41; spec §11.1,
// §15.4): t5d-solver-check - the search's solver against df-pn at the turn
// starts of records - run on records the test writes itself (random legal
// play from the sparse starts, test_play.hpp, through play_record's custom
// set-up). The build passes the executables' paths and a scratch directory.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "play_record.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_play.hpp"
#include "test_support.hpp"

namespace {

namespace fs = std::filesystem;

std::string read_all(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_all(const fs::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  out << text;
}

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

// Runs `exe` with `args`, standard input from `in` (or none), standard output
// and error to files; returns the exit status.
int run(const std::string& exe, const std::vector<std::string>& args, const fs::path& in, const fs::path& out,
        const fs::path& err) {
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

fs::path scratch(const char* name) {
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

// A record of random legal play from sparse start `which` (test_play.hpp):
// a custom set-up, `turns` actions.
std::string random_record(std::size_t which, int turns, std::uint64_t seed) {
  const t5d_test::SparseStart start = t5d_test::sparse_starts().at(which);
  t5d::GameState s = t5d_test::game(start.boards, start.size);
  t5d::mcts::GameRecord game(s);
  t5d_test::Rng rng(seed);
  for (int t = 0; t < turns; ++t) {
    const std::optional<std::vector<t5d::Move>> a = t5d_test::random_legal_action(s, rng);
    if (!a) break;
    for (const t5d::Move& m : *a) s.apply(m);
    s.submit();
    game.actions.push_back(*a);
  }
  game.final_state = s;
  return t5d::tools::play_record(game, "custom");
}

}  // namespace

TEST(solver_check_exe_finds_no_contradiction_on_records_of_random_play) {
  const fs::path dir = scratch("solver-check");
  std::vector<std::string> args{"--playouts", "200", "--dfpn-nodes", "50000", "--per-position"};
  for (std::size_t which : {0u, 4u, 7u, 9u}) {
    const fs::path record = dir / ("game" + std::to_string(which) + ".5dpgn");
    write_all(record, random_record(which, 3, 40 + which));
    args.push_back(record.string());
  }
  const fs::path out = dir / "stdout.txt";
  const fs::path err = dir / "stderr.txt";
  const int status = run(T5D_SOLVER_CHECK_EXE, args, {}, out, err);
  const std::string text = read_all(out);
  if (status != 0) std::fprintf(stderr, "%s%s", text.c_str(), read_all(err).c_str());
  CHECK_EQ(status, 0);
  CHECK(text.find("CONDITIONS ") == 0);
  CHECK(text.find("splits=ply,tree,flat") != std::string::npos);
  CHECK(text.find("proof_channel=on(offline, no store)") != std::string::npos);
  CHECK(text.find("RANGE <50: ") != std::string::npos);
  CHECK(text.find("0 contradictions, 0 certificate failures - the solver agrees") != std::string::npos);
  CHECK(text.find("CONTRADICTION") == std::string::npos);
  // Roots searched, and solved turn starts compared and decided.
  const std::size_t at = text.find("RANGE <50: ");
  REQUIRE(at != std::string::npos);
  const std::string line = text.substr(at, text.find('\n', at) - at);
  std::fprintf(stderr, "  %s\n", line.c_str());
  CHECK(line.find("RANGE <50: 0 roots") == std::string::npos);
  const std::size_t d = line.find(" decided");
  REQUIRE(d != std::string::npos);
  const std::size_t from = line.rfind(' ', d - 1) + 1;
  CHECK(std::stoul(line.substr(from, d - from)) >= 20);
  // A bad command line: status 2.
  CHECK_EQ(run(T5D_SOLVER_CHECK_EXE, {"--split", "sideways", "x.5dpgn"}, {}, out, err), 2);
  CHECK_EQ(run(T5D_SOLVER_CHECK_EXE, {}, {}, out, err), 2);
}
