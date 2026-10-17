// Experiment 6b'''s counting and roots (stage E9's review; register M17):
// the board-count ranges and the pre-registered thresholds to the root (an
// off-by-one fails here); only proofs E4 confirmed count, a refuted one
// withholds every verdict; the option lists name every option; and 6b's
// roots label each missing next action truthfully - the record ends, or goes
// on with an action the engine cannot read or play - from records written
// here (the engine's tests read no corpus). 6b's and 6b''s own tools
// (t5d-scan6b, t5d-scan6bp: the build passes their paths) are run here too,
// for the roots they read.

#include <cctype>
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

#include "sixb_roots.hpp"
#include "sixbpp.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/rules.hpp"
#include "test_framework.hpp"
#include "tool_support.hpp"

#ifndef T5D_TEST_SCRATCH
#error "T5D_TEST_SCRATCH: a directory the tests may write"
#endif
#if !defined(T5D_SCAN6B_EXE) || !defined(T5D_SCAN6BP_EXE)
#error "T5D_SCAN6B_EXE, T5D_SCAN6BP_EXE: the paths of 6b's and 6b''s tools"
#endif

namespace pp = t5d::tools::sixbpp;
using pp::Check;
using pp::RootOutcome;
using pp::Verdict;

namespace {

RootOutcome guard_root(int boards, int completed, int refuted_at, Check check) {
  RootOutcome r;
  r.boards = boards;
  r.guard = true;
  r.completed_depth = completed;
  r.refuted = refuted_at > 0;
  r.refutation_depth = refuted_at;
  r.guard_check = r.refuted ? check : Check::None;
  return r;
}

RootOutcome hunt_root(int boards, bool mate, Check check, bool truth_mate) {
  RootOutcome r;
  r.boards = boards;
  r.hunt = true;
  r.hunt_mate = mate;
  r.hunt_check = mate ? check : Check::None;
  r.truth_mate = truth_mate;
  return r;
}

void write(const std::string& path, const std::string& bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("cannot write " + path);
  std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

std::string read_all(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

// Runs `exe` with `args`, standard output and error to files; the exit status.
int run_tool(const std::string& exe, const std::vector<std::string>& args, const std::filesystem::path& out,
             const std::filesystem::path& err) {
  std::string command = quoted(std::filesystem::path(exe).make_preferred().string());
  for (const std::string& a : args) command += " " + quoted(a);
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

// The positions a tool's --per-position JSON lists, in order, as
// "index:fingerprint" - each fingerprint with the index its object gives
// before it.
std::string listed_positions(const std::string& json) {
  const std::string index_key = "\"index\":";
  const std::string fingerprint_key = "\"fingerprint\":\"";
  const std::size_t start = json.find("\"positions\":[");
  if (start == std::string::npos) return "(no positions listed)";
  std::string out;
  for (std::size_t at = start;;) {
    const std::size_t f = json.find(fingerprint_key, at);
    if (f == std::string::npos) break;
    const std::size_t i = json.rfind(index_key, f);
    if (i == std::string::npos || i < start) return out + " (a fingerprint without its index)";
    std::size_t end = i + index_key.size();
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end])) != 0) ++end;
    const std::size_t value = f + fingerprint_key.size();
    out += (out.empty() ? "" : ",") + json.substr(i + index_key.size(), end - i - index_key.size()) + ":" +
           json.substr(value, 32);
    at = value;
  }
  return out;
}

}  // namespace

TEST(sixbpp_ranges_and_thresholds_are_the_pre_registered_ones) {
  CHECK_EQ(pp::range_of(1), 0);
  CHECK_EQ(pp::range_of(49), 0);
  CHECK_EQ(pp::range_of(50), 1);
  CHECK_EQ(pp::range_of(99), 1);
  CHECK_EQ(pp::range_of(100), 2);
  CHECK_EQ(pp::range_of(199), 2);
  CHECK_EQ(pp::range_of(200), 3);
  CHECK_EQ(pp::range_of(5000), 3);
  // Guard: >= 95% of a range's roots.
  CHECK(pp::guard_passes(95, 100));
  CHECK(!pp::guard_passes(94, 100));
  CHECK(pp::guard_passes(19, 20));
  CHECK(!pp::guard_passes(18, 20));
  CHECK(pp::guard_passes(1, 1));
  CHECK(!pp::guard_passes(0, 1));
  CHECK(!pp::guard_passes(0, 0));
  CHECK(pp::guard_passes(513, 540));   // 95.0%
  CHECK(!pp::guard_passes(512, 540));  // 94.8%
  // Hunt: >= 50% of the truth's mates.
  CHECK(pp::hunt_passes(1, 2));
  CHECK(!pp::hunt_passes(0, 2));
  CHECK(pp::hunt_passes(3, 5));
  CHECK(!pp::hunt_passes(2, 5));
  CHECK(pp::hunt_passes(36, 71));
  CHECK(!pp::hunt_passes(35, 71));
  CHECK(!pp::hunt_passes(0, 0));
  // "Too few to read": fewer than 10 truth mates, at least one.
  std::vector<RootOutcome> roots;
  for (int i = 0; i < 9; ++i) roots.push_back(hunt_root(250, false, Check::None, true));    // 200+: 9
  for (int i = 0; i < 10; ++i) roots.push_back(hunt_root(150, false, Check::None, true));   // 100-199: 10
  roots.push_back(hunt_root(60, false, Check::None, false));                                // 50-99: none
  const pp::Tallies t = pp::tally(roots);
  CHECK(pp::too_few_to_read(t, 3));
  CHECK(!pp::too_few_to_read(t, 2));
  CHECK(!pp::too_few_to_read(t, 1));
  CHECK_EQ(pp::kFewTruthMates, std::size_t{10});
}

TEST(sixbpp_only_proofs_e4_confirmed_count) {
  // Guard's depth 1: "safe to depth 1" counts; a refutation at depth 1 only
  // when E4 confirmed it.
  CHECK(pp::depth_one_done(guard_root(10, 1, 0, Check::None)));
  CHECK(pp::depth_one_done(guard_root(10, 3, 0, Check::None)));
  CHECK(pp::depth_one_done(guard_root(10, 0, 1, Check::Confirmed)));
  CHECK(!pp::depth_one_done(guard_root(10, 0, 1, Check::Unconfirmed)));
  CHECK(pp::depth_one_unconfirmed(guard_root(10, 0, 1, Check::Unconfirmed)));
  CHECK(!pp::depth_one_done(guard_root(10, 0, 1, Check::Refuted)));
  CHECK(!pp::depth_one_done(guard_root(10, 0, 0, Check::None)));  // stopped in depth 1
  CHECK(pp::depth_one_done(guard_root(10, 1, 2, Check::Unconfirmed)));  // depth 1 safe; depth 2 refuted
  RootOutcome absent = guard_root(10, 1, 0, Check::None);
  absent.guard = false;  // not counted (no next action)
  CHECK(!pp::depth_one_done(absent));

  std::vector<RootOutcome> roots{
      guard_root(10, 1, 0, Check::None),         // < 50: done
      guard_root(10, 0, 1, Check::Confirmed),    // < 50: done
      guard_root(10, 0, 1, Check::Unconfirmed),  // < 50: not counted, reported
      guard_root(10, 0, 0, Check::None),         // < 50: not done
      guard_root(70, 1, 2, Check::Confirmed),    // 50-99: done, refuted at 2
      hunt_root(10, true, Check::Confirmed, true),     // found
      hunt_root(10, true, Check::Unconfirmed, true),   // found, not counted
      hunt_root(10, false, Check::None, true),         // missed
      hunt_root(10, true, Check::Confirmed, false),    // beyond the truth
      hunt_root(10, true, Check::Unconfirmed, false),  // beyond, not confirmed: not "extra"
      hunt_root(120, false, Check::None, false),
  };
  const pp::Tallies t = pp::tally(roots);
  const pp::GuardTally& g = t.guard[0];
  CHECK_EQ(g.roots, std::size_t{4});
  CHECK_EQ(g.depth_one, std::size_t{2});
  CHECK_EQ(g.unconfirmed, std::size_t{1});
  CHECK_EQ(g.refuted, std::size_t{2});
  CHECK_EQ(g.refuted_one, std::size_t{2});
  CHECK_EQ(g.completed.at(0), std::size_t{3});
  CHECK_EQ(g.completed.at(1), std::size_t{1});
  CHECK_EQ(t.guard[1].depth_one, std::size_t{1});
  CHECK_EQ(t.guard[1].refuted_one, std::size_t{0});
  CHECK_EQ(t.guard[pp::kRanges].roots, std::size_t{5});  // pooled
  CHECK_EQ(t.guard[pp::kRanges].depth_one, std::size_t{3});
  const pp::HuntTally& h = t.hunt[0];
  CHECK_EQ(h.roots, std::size_t{5});
  CHECK_EQ(h.truth, std::size_t{3});
  CHECK_EQ(h.found, std::size_t{1});
  CHECK_EQ(h.found_unconfirmed, std::size_t{1});
  CHECK_EQ(h.mates, std::size_t{4});
  CHECK_EQ(h.unconfirmed, std::size_t{2});
  CHECK_EQ(h.extra, std::size_t{1});
  CHECK_EQ(t.hunt[2].roots, std::size_t{1});
  CHECK_EQ(t.refuted_proofs, std::size_t{0});
  // 2 of 4 counted: Guard fails < 50; 1 of 1: passes 50-99. Hunt: 1 of 3
  // confirmed (the unconfirmed one would have made it 2 of 3): fails.
  CHECK(pp::guard_verdict(t, 0, true) == Verdict::NotAdopted);
  CHECK(pp::guard_verdict(t, 1, true) == Verdict::Adopt);
  CHECK(pp::guard_verdict(t, 2, true) == Verdict::NoRoots);
  CHECK(pp::hunt_verdict(t, 0, true, true) == Verdict::NotAdopted);
  CHECK(pp::hunt_verdict(t, 2, true, true) == Verdict::NoTruthMates);
  CHECK(pp::hunt_verdict(t, 3, true, true) == Verdict::NoRoots);
  CHECK(pp::hunt_verdict(t, 0, false, true) == Verdict::NoTruth);
  // No cross-check: nothing confirmed, no verdict.
  CHECK(pp::guard_verdict(t, 1, false) == Verdict::Withheld);
  CHECK(pp::hunt_verdict(t, 0, true, false) == Verdict::Withheld);
}

TEST(sixbpp_a_refuted_proof_withholds_every_verdict) {
  std::vector<RootOutcome> roots;
  for (int i = 0; i < 20; ++i) roots.push_back(guard_root(10, 1, 0, Check::None));  // < 50: 100%
  for (int i = 0; i < 4; ++i) roots.push_back(hunt_root(10, true, Check::Confirmed, true));
  const pp::Tallies clean = pp::tally(roots);
  CHECK(pp::guard_verdict(clean, 0, true) == Verdict::Adopt);
  CHECK(pp::hunt_verdict(clean, 0, true, true) == Verdict::Adopt);
  for (const bool hunt : {false, true}) {
    std::vector<RootOutcome> bad = roots;
    bad.push_back(hunt ? hunt_root(150, true, Check::Refuted, false) : guard_root(150, 0, 1, Check::Refuted));
    const pp::Tallies t = pp::tally(bad);
    CHECK_EQ(t.refuted_proofs, std::size_t{1});
    for (int k = 0; k < pp::kRanges; ++k) {
      CHECK(pp::guard_verdict(t, k, true) == Verdict::Withheld);
      CHECK(pp::hunt_verdict(t, k, true, true) == Verdict::Withheld);
    }
  }
}

TEST(sixbpp_the_option_lists_name_every_option) {
  const auto has = [](const std::vector<std::pair<std::string, std::string>>& options, const std::string& key,
                      const std::string& value) {
    for (const auto& [k, v] : options) {
      if (k == key) return v == value;
    }
    return false;
  };
  CHECK_EQ(pp::prover_options(t5d::ProverOptions{}).size(), std::size_t{21});
  t5d::GuardOptions g;
  g.max_depth = 60;
  g.threads = 5;
  const auto go = pp::guard_options(g);
  CHECK(has(go, "max_depth", "60"));
  CHECK(has(go, "prover.depth", "99"));  // 50 depths at most: 99 half-turns
  CHECK(has(go, "prover.seeds", t5d::seed_class_name(t5d::SeedClass::All)));
  CHECK(has(go, "prover.completion", t5d::completion_class_name(t5d::CompletionClass::Every)));
  CHECK(has(go, "prover.threads", "5"));
  CHECK(has(go, "prover.prove_only", "false"));
  CHECK(has(go, "prover.check_filter", "false"));  // as Guard ships
  CHECK(has(go, "prover.attacker_probe", "true"));
  CHECK(has(go, "prover.listing_cap", "65536"));
  const auto ho = pp::hunt_options(t5d::HuntOptions{});
  CHECK(has(ho, "export_proofs", "false"));
  CHECK(has(ho, "suggest", "false"));
  CHECK(has(ho, "prover.prove_only", "true"));
  CHECK(has(ho, "prover.depth", "5"));
  CHECK(has(ho, "prover.check_filter", "false"));
  CHECK_EQ(go.size(), std::size_t{2 + 21});
  CHECK_EQ(ho.size(), std::size_t{4 + 21});
}

TEST(sixb_roots_label_their_next_actions_truthfully) {
  const std::string dir = (std::filesystem::path(T5D_TEST_SCRATCH) / "sixb-roots").string();
  std::filesystem::create_directories(dir);
  const std::string head = "[Board \"Custom\"]\n[Size \"5x5\"]\n[k4/5/5/5/K3Q:0:1:w]\n\n";
  const std::string ends = head + "1. (0T1)Qe1e2 / (0T1)Ka5a4\n";
  const std::string unreadable = head + "1. (0T1)Qe1e2 / (0T1)Ka5a4\n2. (0T2)@@@\n";
  const std::string illegal = head + "1. (0T1)Qe1e2 / (0T1)Ka5a2\n";
  std::string crlf;
  for (const char c : ends) {
    if (c == '\n') crlf += '\r';
    crlf += c;
  }
  write(dir + "/ends.5dpgn", ends);
  write(dir + "/unreadable.5dpgn", unreadable);
  write(dir + "/illegal.5dpgn", illegal);
  write(dir + "/crlf.5dpgn", crlf);
  write(dir + "/list.txt", "# the records\n\nends.5dpgn\n  unreadable.5dpgn  \n" + dir + "/illegal.5dpgn\ncrlf.5dpgn\n");

  const std::vector<t5d::tools::RecordRef> refs = t5d::tools::record_refs({}, {dir + "/list.txt"}, dir);
  REQUIRE_EQ(refs.size(), std::size_t{4});
  CHECK_EQ(refs[0].id, std::string("ends.5dpgn"));
  CHECK_EQ(refs[0].path, dir + "/ends.5dpgn");
  CHECK_EQ(refs[1].id, std::string("unreadable.5dpgn"));
  CHECK_EQ(refs[2].path, dir + "/illegal.5dpgn");  // absolute: as listed
  CHECK_THROWS_AS(t5d::tools::record_refs({}, {dir + "/no-such-list.txt"}, ""), std::runtime_error);

  // Content ids: CRLF read as LF.
  const std::vector<std::string> contents = t5d::tools::record_contents(refs);
  CHECK_EQ(contents[0], contents[3]);
  CHECK(contents[0] != contents[1]);
  CHECK_EQ(contents[0], t5d::tools::record_content_id(ends));
  CHECK(contents[0].rfind("sha256:", 0) == 0);
  const std::vector<std::string> missing = t5d::tools::record_contents({{"x", dir + "/no-such.5dpgn"}});
  CHECK(missing[0].rfind("unreadable: ", 0) == 0);

  const t5d::tools::SixbRoots roots = t5d::tools::sixb_roots(refs, t5d::RuleOptions{}, 1, 0);
  // ends: 3 roots (0, 1, 2); unreadable, illegal and crlf: the same
  // positions - duplicates (the first of equal positions kept).
  REQUIRE_EQ(roots.roots.size(), std::size_t{3});
  CHECK_EQ(roots.duplicates, std::size_t{3 + 2 + 3});
  for (const t5d::tools::SixbRoot& r : roots.roots) CHECK_EQ(r.record, std::string("ends.5dpgn"));
  CHECK(roots.roots[0].next_action.has_value() && roots.roots[0].no_next.empty());
  CHECK(roots.roots[1].next_action.has_value());
  CHECK(!roots.roots[2].next_action.has_value());
  CHECK_EQ(roots.roots[2].no_next, std::string("its record ends here"));
  REQUIRE_EQ(roots.notes.size(), std::size_t{4});
  CHECK(contains(roots.notes[0], "2 actions read, ok"));
  CHECK(contains(roots.notes[1], "cannot read"));
  CHECK(!contains(roots.notes[1], ", ok"));
  CHECK(!contains(roots.notes[2], ", ok"));

  // Each record alone: its own last root's label.
  const t5d::tools::SixbRoots u = t5d::tools::sixb_roots({refs[1]}, t5d::RuleOptions{}, 1, 0);
  REQUIRE_EQ(u.roots.size(), std::size_t{3});
  CHECK(!u.roots[2].next_action.has_value());
  CHECK(contains(u.roots[2].no_next, "goes on with an action the engine cannot read"));
  const t5d::tools::SixbRoots il = t5d::tools::sixb_roots({refs[2]}, t5d::RuleOptions{}, 1, 0);
  REQUIRE_EQ(il.roots.size(), std::size_t{2});
  CHECK(il.roots[0].next_action.has_value());
  CHECK(!il.roots[1].next_action.has_value());
  CHECK(contains(il.roots[1].no_next, "cannot be"));
  CHECK(!contains(il.roots[1].no_next, "ends here"));
}

TEST(sixb_roots_read_records_as_6b_read_them) {
  // 6b's roots are rebuilt with the reading 6b, 6b', 6b'' and Hunt's ground
  // truth were measured under: the reader's defaults but for the capture mark,
  // which must capture where it is written - the reading before H51 made the
  // mark lenient (tools/tool_support.hpp). A record with an 'x' on a jump that
  // takes nothing stops there, as it did when the truth was recorded, where
  // the reader's default (H51) reads on - the truth's readers would refuse it.
  const t5d::PgnOptions six = t5d::tools::sixb_pgn_options();
  const t5d::PgnOptions defaults;
  CHECK(defaults.capture_marks == t5d::CaptureMarks::Lenient);
  CHECK(six.capture_marks == t5d::CaptureMarks::MarkedMustCapture);
  CHECK(six.castling_shortcuts == defaults.castling_shortcuts);
  CHECK_EQ(six.verify_annotations, defaults.verify_annotations);
  CHECK_EQ(six.annotation_mismatch_is_error, defaults.annotation_mismatch_is_error);
  CHECK_EQ(six.infer_omitted_boards, defaults.infer_omitted_boards);
  CHECK_EQ(six.check_turn_numbers, defaults.check_turn_numbers);
  CHECK_EQ(six.turn_number_mismatch_is_error, defaults.turn_number_mismatch_is_error);

  const std::string dir = (std::filesystem::path(T5D_TEST_SCRATCH) / "sixb-roots-h51").string();
  std::filesystem::create_directories(dir);
  const std::string text = "1. Nf3 / Nf6\n2. (0T2)Nf3>>x(0T1)f5~ / (1T1)e6\n";
  write(dir + "/superfluous.5dpgn", text);
  const t5d::tools::SixbRoots roots =
      t5d::tools::sixb_roots({{"superfluous.5dpgn", dir + "/superfluous.5dpgn"}}, t5d::RuleOptions{}, 1, 0);
  REQUIRE_EQ(roots.roots.size(), std::size_t{3});  // positions 0, 1 and 2: action 3 is not read
  CHECK(!roots.roots[2].next_action.has_value());
  CHECK(contains(roots.roots[2].no_next, "captures nothing"));
  REQUIRE_EQ(roots.notes.size(), std::size_t{1});
  CHECK(contains(roots.notes[0], "2 actions read, unreadable at action 3"));
  // The reader's default reads the record to its end.
  t5d::PgnReplay lenient(t5d::read_pgn(text));
  t5d::tools::parse_mode(lenient);
  lenient.play_all();
  CHECK_EQ(lenient.actions_played(), std::size_t{4});
  CHECK_EQ(lenient.warnings().size(), std::size_t{1});
}

TEST(sixb_tools_read_records_as_6b_read_them) {
  // t5d-scan6b and t5d-scan6bp read their records in loops of their own, not
  // through sixb_roots: each is run here on a record with an 'x' on a jump
  // that takes nothing, and must list the roots sixb_roots builds - index and
  // fingerprint, in order - and stop where it stops. A tool that read with the
  // reader's default (H51: on, with a warning) would list positions 3 and 4
  // too; on the corpus, 6b's or 6b''s roots would be renumbered from 169
  // (build-log spec issue 129), which no other check compares.
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(T5D_TEST_SCRATCH) / "sixb-tools-h51";
  fs::create_directories(dir);
  const std::string record = (dir / "superfluous.5dpgn").string();
  write(record, "1. Nf3 / Nf6\n2. (0T2)Nf3>>x(0T1)f5~ / (1T1)e6\n");
  const t5d::tools::SixbRoots six = t5d::tools::sixb_roots({{record, record}}, t5d::RuleOptions{}, 1, 0);
  REQUIRE_EQ(six.roots.size(), std::size_t{3});
  std::string roots;
  for (const t5d::tools::SixbRoot& r : six.roots) {
    roots += (roots.empty() ? "" : ",") + std::to_string(r.index) + ":" +
             t5d::fingerprint(r.state.multiverse()).fingerprint;
  }

  // 6b: the roots only (no leaves), a one-node curve, no wall-clock scans, no
  // stand-pat check - its positions, not its figures.
  const fs::path json_6b = dir / "6b.json";
  REQUIRE_EQ(run_tool(T5D_SCAN6B_EXE,
                      {"--leaves", "0", "--max-nodes", "1", "--budgets", "1", "--nps", "1000", "--no-root-wall",
                       "--no-spv", "--per-position", "--json", json_6b.string(), record},
                      dir / "6b-stdout.txt", dir / "6b-stderr.txt"),
             0);
  const std::string out_6b = read_all(json_6b);
  CHECK_EQ(listed_positions(out_6b), roots);
  CHECK(contains(out_6b, "2 actions read, unreadable at action 3"));

  // 6b': every position outside its board window - listed, not measured.
  const fs::path json_6bp = dir / "6bp.json";
  REQUIRE_EQ(run_tool(T5D_SCAN6BP_EXE, {"--min-boards", "1000", "--per-position", "--json", json_6bp.string(), record},
                      dir / "6bp-stdout.txt", dir / "6bp-stderr.txt"),
             0);
  CHECK_EQ(listed_positions(read_all(json_6bp)), roots);
  CHECK(contains(read_all(dir / "6bp-stdout.txt"), "2 actions read, unreadable at action 3"));
}
