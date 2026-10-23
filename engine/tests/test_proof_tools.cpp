// The H41 cross-check tools as executables (register H41; spec §11.1,
// §15.4): t5d-solver-check - the search's solver against df-pn at the turn
// starts of records - and t5d-proof-records - the solver's proof lines as
// 5dpgn records, replayed here through the gate adapter - run on records the
// test writes itself (the constructed positions and random legal play from
// the sparse starts, test_play.hpp, through play_record's custom set-up).
// The build passes the executables' paths and a scratch directory.

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
#include "sha256.hpp"
#include "t5d/pgn.hpp"
#include "t5d/search.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_play.hpp"
#include "test_solver_refs.hpp"
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

// ---- t5d-proof-records (MT-02's engine side) ----------------------------------------------------

namespace {

std::string constructed_record(const t5d::GameState& s) {
  t5d::mcts::GameRecord game(s);
  return t5d::tools::play_record(game, "custom");
}

// The number before `key` in a JSON line ("\"legal_actions\":12" -> 12).
long json_number(const std::string& line, const std::string& key) {
  const std::size_t at = line.find("\"" + key + "\":");
  if (at == std::string::npos) return -1;
  return std::stol(line.substr(at + key.size() + 3));
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

}  // namespace

TEST(proof_records_exe_lines_replay_through_the_gate_to_their_mates) {
  // The constructed mates as custom set-ups (records of no action), a record
  // of random play and a game t5d-play played from a named variant (their turn
  // starts mid-game roots), through the tool under each D2 arm; every record it writes replayed by t5d-gate (full mode): the last
  // position E4's checkmate, no position before it terminal, and at each
  // defender turn start the record names, the proof tree's defences E4's
  // distinct successors and the gate's legal_actions E4's count of actions.
  const fs::path dir = scratch("proof-records");
  std::vector<std::string> inputs;
  const t5d::RuleOptions rules;
  const std::vector<std::pair<const char*, t5d::GameState>> positions{
      {"mate_in_one", t5d_test::solver::mate_in_one(rules)},
      {"cross_board_mate", t5d_test::solver::cross_board_mate(rules)},
      {"mate_in_two", t5d_test::solver::mate_in_two(rules)},
      {"history_royal_mate", t5d_test::solver::history_royal_mate(rules)}};
  for (const auto& [name, s] : positions) {
    const fs::path record = dir / (std::string(name) + ".5dpgn");
    write_all(record, constructed_record(s));
    inputs.push_back(record.string());
  }
  const fs::path game = dir / "game.5dpgn";
  write_all(game, random_record(7, 4, 70));
  inputs.push_back(game.string());
  const fs::path out = dir / "stdout.txt";
  const fs::path err = dir / "stderr.txt";
  // A game the engine played from a named variant's start (t5d-play): its
  // mid-game roots' records are the whole game from the start, then the line.
  const fs::path played = dir / "played.5dpgn";
  REQUIRE(run(T5D_PLAY_EXE,
              {"--variant", "Very Small - Open", "--evaluator", "material", "--playouts", "32", "--cap-turns", "14",
               "--seed", "2"},
              {}, played, err) == 0);
  std::size_t named = 0;
  std::size_t records = 0;
  std::size_t defender_turns = 0;
  std::size_t mates_in_two = 0;
  // The runs: every D2 arm - the ply split, the tree form, the flat form - on
  // the constructed and random records; the played game under the ply split
  // alone, every other turn start (each root is a search). Move legality is
  // exact within a node budget (a move left unknown proves nothing): exact
  // legality without one can take minutes a search on the played game's
  // roots (build-log issue 406).
  struct Run {
    std::string name;
    std::vector<std::string> args;
  };
  std::vector<Run> runs;
  for (const std::string split : {"ply", "tree", "flat"}) {
    Run r{split, {"--split", split, "--playouts", "2000", "--max-lines", "16", "--legality-nodes", "200000"}};
    r.args.insert(r.args.end(), inputs.begin(), inputs.end());
    runs.push_back(std::move(r));
  }
  runs.push_back(Run{"played",
                     {"--split", "ply", "--playouts", "1000", "--max-lines", "16", "--legality-nodes", "200000",
                      "--every", "2", played.string()}});
  for (Run& run_spec : runs) {
  const fs::path out_dir = dir / ("out-" + run_spec.name);
  std::vector<std::string> args{"--out", out_dir.string()};
  args.insert(args.end(), run_spec.args.begin(), run_spec.args.end());
  const int status = run(T5D_PROOF_RECORDS_EXE, args, {}, out, err);
  const std::string text = read_all(out);
  std::fprintf(stderr, "%s", text.c_str());
  if (status != 0) std::fprintf(stderr, "%s", read_all(err).c_str());
  REQUIRE(status == 0);
  CHECK(text.find("proof_channel=on(offline, no store)") != std::string::npos);
  CHECK(text.find("ERROR") == std::string::npos);
  // FILES.tsv: every record with its SHA-256 and size, and nothing else.
  const std::vector<std::string> listed = lines_of(read_all(out_dir / "FILES.tsv"));
  REQUIRE(!listed.empty());
  for (const std::string& row : listed) {
    if (row[0] == '#') continue;
    const std::size_t t1 = row.find('\t');
    const std::size_t t2 = row.find('\t', t1 + 1);
    REQUIRE(t1 != std::string::npos && t2 != std::string::npos);
    const std::string sha = row.substr(0, t1);
    const std::size_t size = std::stoul(row.substr(t1 + 1, t2 - t1 - 1));
    const fs::path file = out_dir / row.substr(t2 + 1);
    const std::string body = read_all(file);
    CHECK_EQ(body.size(), size);
    CHECK_EQ(t5d::tools::sha256_hex(body), sha);
    ++records;
    // The record as the engine reads it.
    const t5d::PgnRecord rec = t5d::read_pgn(body);
    REQUIRE(rec.header("ProofLine") != nullptr);
    REQUIRE(rec.header("ProofDefences") != nullptr);
    CHECK(*rec.header("Termination") == "checkmate");
    if (*rec.header("ProofClaim") == "win 3") ++mates_in_two;
    if (*rec.header("Board") == "Very Small - Open") ++named;
    // The gate adapter's replay.
    const fs::path gate_out = dir / "gate-stdout.txt";
    REQUIRE(run(T5D_GATE_EXE, {"--cap", "1000"}, file, gate_out, err) == 0);  // counts above 1,000 compare as "at least"
    std::vector<std::string> pos;
    for (const std::string& l : lines_of(read_all(gate_out))) {
      if (l.find("\"event\":\"position\"") != std::string::npos) pos.push_back(l);
    }
    REQUIRE_EQ(pos.size(), rec.actions.size() + 1);
    CHECK(pos.back().find("\"terminal\":\"checkmate\"") != std::string::npos);
    for (std::size_t i = 0; i + 1 < pos.size(); ++i) CHECK(pos[i].find("\"terminal\":\"none\"") != std::string::npos);
    // The states along the record, to ask E4 at the defender turn starts.
    t5d::PgnReplay replay(rec);
    std::vector<t5d::GameState> states{replay.state()};
    while (!replay.finished()) {
      replay.play_next();
      states.push_back(replay.state());
    }
    std::string defences = *rec.header("ProofDefences");
    std::size_t at = 0;
    while (at < defences.size()) {
      std::size_t sp = defences.find(' ', at);
      if (sp == std::string::npos) sp = defences.size();
      const std::string item = defences.substr(at, sp - at);
      at = sp + 1;
      const std::size_t colon = item.find(':');
      const std::size_t index = std::stoul(item.substr(0, colon));
      const std::size_t n = std::stoul(item.substr(colon + 1));
      REQUIRE(index < states.size());
      const t5d::GameState& s = states[index];
      const t5d::ActionCount actions = t5d::count_legal_actions(s, 1000, t5d::Detector::Hypercuboid);
      const t5d::ActionCount successors = t5d::count_legal_successors(s, 100000, t5d::Detector::Hypercuboid);
      REQUIRE(actions.known() && successors.stats.exhaustive);
      CHECK_EQ(n, successors.count);
      CHECK_EQ(json_number(pos[index], "legal_actions"), static_cast<long>(actions.count));
      ++defender_turns;
    }
  }
  }
  std::fprintf(stderr,
               "  %zu records (%zu from the named variant's game), %zu defender turn starts checked, %zu lines of a "
               "mate in two\n",
               records, named, defender_turns, mates_in_two);
  CHECK(records >= 15);
  CHECK(defender_turns >= 6);
  CHECK(mates_in_two >= 3);
  CHECK(named >= 3);
  // A bad command line: status 2; no --out: status 2.
  CHECK_EQ(run(T5D_PROOF_RECORDS_EXE, {"--out", (dir / "bad").string(), "--split", "sideways", "x.5dpgn"}, {}, out, err),
           2);
  CHECK_EQ(run(T5D_PROOF_RECORDS_EXE, {"x.5dpgn"}, {}, out, err), 2);
}
