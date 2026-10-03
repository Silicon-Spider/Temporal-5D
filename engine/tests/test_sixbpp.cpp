// Experiment 6b'''s counting and roots (stage E9's review; register M17):
// the board-count ranges and the pre-registered thresholds to the root (an
// off-by-one fails here); only proofs E4 confirmed count, a refuted one
// withholds every verdict; the option lists name every option; and 6b's
// roots label each missing next action truthfully - the record ends, or goes
// on with an action the engine cannot read or play - from records written
// here (the engine's tests read no corpus).

#include <cstdio>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sixb_roots.hpp"
#include "sixbpp.hpp"
#include "t5d/rules.hpp"
#include "test_framework.hpp"
#include "tool_support.hpp"

#ifndef T5D_TEST_SCRATCH
#error "T5D_TEST_SCRATCH: a directory the tests may write"
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
