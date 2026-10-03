// Hunt's ground truth, pinned and hashed (stage E9; the user's ruling 5;
// register M17): SHA-256 against the standard's vectors; the file and its
// provenance round-trip; files whose bytes do not match their MANIFEST are
// refused (a valid file under a wrong MANIFEST too), and so is a provenance
// that names another truth or other records; the rules are recorded by every
// option's value; and the committed truth (engine/testdata/hunt-truth/) is
// the one pinned here by its SHA-256, matches its MANIFEST, and is well
// formed. That its roots are 6b's roots - record, index and fingerprint -
// the 6b'' tool and the recorder check against the records they are given
// (the engine's tests read no corpus).

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "hunt_truth.hpp"
#include "sha256.hpp"
#include "t5d/rules.hpp"
#include "test_framework.hpp"
#include "tool_support.hpp"

#ifndef T5D_ENGINE_DIR
#error "T5D_ENGINE_DIR: the engine's source directory (CMakeLists.txt defines it)"
#endif
#ifndef T5D_TEST_SCRATCH
#error "T5D_TEST_SCRATCH: a directory the tests may write"
#endif

using t5d::tools::HuntTruth;
using t5d::tools::HuntTruthRoot;

namespace {

void write(const std::string& path, const std::string& bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("cannot write " + path);
  std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
}

// The committed truth's SHA-256 (also in its MANIFEST, README and build
// log): a re-recording changes it here, in a reviewed change.
constexpr const char* kCommittedTruthSha256 = "ce62e5f8c52e0d53b1519ef145aae00760d84a4b271b9e9643e5fcab02cf1a73";

// Every rule option's value under the defaults (absolute_rules_text): what
// the committed truth's provenance records. A change of the defaults changes
// this line - and the truth's rules then need reading again.
constexpr const char* kDefaultRules =
    "activation=Count initial_credit=None play_activates=false moves_after_shift=true mandatory=InPresent "
    "reactivation=Rewinds turn_zero_reachable=true sliders_cross_void=false past_self_blocks=true "
    "earlier_playable_jump_branches=false inactive_history_is_destination=true brawn_rank_timeline_capture=true "
    "double_step_right=UnmovedFlag timeline_double_step=true double_step_over_void=false "
    "branch_copies_keep_unmoved=true jumps_keep_unmoved=false en_passant_brawns=true brawns_promote=true "
    "brawn_cross_board_promotion=true promotions=0x0010 castling_empty_between=true "
    "castling_destination_test=false castling_rook_path_test=false castling_any_rank=true "
    "castling_rook_two_away=true castling_on_optional_boards=true family=Standard";

HuntTruth small_truth() {
  HuntTruth t;
  t.header = {{"t5d-hunt-truth", "1"}, {"engine", "test"},   {"rules", "standard\t0x0"},
              {"search", "a test"},   {"depth", "5"},        {"crosscheck", "a test"},
              {"roots", "2"}};
  HuntTruthRoot a;
  a.id = 0;
  a.record = "a.5dpgn";
  a.index = 0;
  a.fingerprint = "d574889fd9da3f2bc65249ff27249b00";
  a.boards = 1;
  a.verdict = "none";
  a.literal = "unknown";
  a.literal_nodes = 10;
  a.deflt = "exhausted";
  a.default_nodes = 3;
  HuntTruthRoot b = a;
  b.id = 1;
  b.index = 3;
  b.verdict = "mate";
  b.distance = 1;
  b.found_by = "both";
  b.literal = "proven";
  b.deflt = "proven";
  b.crosscheck = "confirmed";
  b.line = "(0T2)Qd1xf3";
  t.roots = {a, b};
  return t;
}

}  // namespace

TEST(hunt_truth_sha256_matches_the_standard) {
  CHECK_EQ(t5d::tools::sha256_hex(""), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  CHECK_EQ(t5d::tools::sha256_hex("abc"),
           std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  CHECK_EQ(t5d::tools::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
           std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
  // A million 'a': padding across several blocks.
  CHECK_EQ(t5d::tools::sha256_hex(std::string(1000000, 'a')),
           std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
}

TEST(hunt_truth_round_trips_and_refuses_a_changed_byte) {
  const HuntTruth t = small_truth();
  const std::string text = t5d::tools::format_hunt_truth(t);
  const HuntTruth back = t5d::tools::parse_hunt_truth(text);
  CHECK_EQ(t5d::tools::format_hunt_truth(back), text);
  CHECK_EQ(back.roots.size(), std::size_t{2});
  CHECK(back.roots[1].mate());
  CHECK_EQ(back.roots[1].distance, 1);
  t5d::tools::HuntTruthProvenance prov;
  prov.truth = "sha256:" + t5d::tools::sha256_hex(text);
  prov.rules = t5d::tools::absolute_rules_text(t5d::RuleOptions{});
  prov.code = "sha256:test";
  prov.records = {"b.5dpgn", "a.5dpgn"};
  const std::string ptext = t5d::tools::format_provenance(prov);
  CHECK_EQ(t5d::tools::format_provenance(t5d::tools::parse_provenance(ptext)), ptext);
  // Written with its provenance and MANIFEST: loaded.
  const std::string dir = (std::filesystem::path(T5D_TEST_SCRATCH) / "hunt-truth").string();
  std::filesystem::create_directories(dir);
  const auto put = [&](const std::string& truth, const std::string& provenance, const std::string& manifest) {
    write(dir + "/" + t5d::tools::kHuntTruthFile, truth);
    write(dir + "/" + t5d::tools::kHuntTruthProvenance, provenance);
    write(dir + "/" + t5d::tools::kHuntTruthManifest, manifest);
  };
  put(text, ptext, t5d::tools::hunt_truth_manifest(text, ptext));
  const HuntTruth loaded = t5d::tools::load_hunt_truth(dir);
  CHECK_EQ(loaded.sha256, t5d::tools::sha256_hex(text));
  CHECK_EQ(loaded.provenance.records.size(), std::size_t{2});
  // A VALID file under a wrong MANIFEST: refused (the pin, not the parse).
  const std::string zeros(64, '0');
  put(text, ptext, zeros + "  hunt-truth.tsv\n" + t5d::tools::sha256_hex(ptext) + "  provenance.tsv\n");
  CHECK_THROWS_WITH(t5d::tools::load_hunt_truth(dir), std::runtime_error, "MANIFEST");
  put(text, ptext, t5d::tools::sha256_hex(text) + "  hunt-truth.tsv\n" + zeros + "  provenance.tsv\n");
  CHECK_THROWS_WITH(t5d::tools::load_hunt_truth(dir), std::runtime_error, "MANIFEST");
  put(text, ptext, t5d::tools::sha256_hex(text) + "  hunt-truth.tsv\n");  // the provenance unpinned
  CHECK_THROWS_WITH(t5d::tools::load_hunt_truth(dir), std::runtime_error, "MANIFEST");
  // One byte changed that still parses (a node count), the MANIFEST stale:
  // refused.
  std::string changed = text;
  changed.replace(changed.find("\t10\t"), 4, "\t11\t");
  CHECK_NOTHROW((void)t5d::tools::parse_hunt_truth(changed));
  put(changed, ptext, t5d::tools::hunt_truth_manifest(text, ptext));
  CHECK_THROWS_WITH(t5d::tools::load_hunt_truth(dir), std::runtime_error, "MANIFEST");
  // A provenance naming another truth, or other records, or the roots out of
  // its order: refused even under a MANIFEST that pins them.
  t5d::tools::HuntTruthProvenance other = prov;
  other.truth = "sha256:" + zeros;
  std::string otext = t5d::tools::format_provenance(other);
  put(text, otext, t5d::tools::hunt_truth_manifest(text, otext));
  CHECK_THROWS_WITH(t5d::tools::load_hunt_truth(dir), std::runtime_error, "another truth");
  other = prov;
  other.records = {"b.5dpgn"};
  otext = t5d::tools::format_provenance(other);
  put(text, otext, t5d::tools::hunt_truth_manifest(text, otext));
  CHECK_THROWS_WITH(t5d::tools::load_hunt_truth(dir), std::runtime_error, "not in its provenance");
  HuntTruth two = small_truth();
  two.roots[1].record = "b.5dpgn";
  two.roots[0].record = "c.5dpgn";
  const std::string ttext = t5d::tools::format_hunt_truth(two);
  other = prov;
  other.truth = "sha256:" + t5d::tools::sha256_hex(ttext);
  other.records = {"b.5dpgn", "c.5dpgn"};  // root 0's record after root 1's: out of order
  otext = t5d::tools::format_provenance(other);
  put(ttext, otext, t5d::tools::hunt_truth_manifest(ttext, otext));
  CHECK_THROWS_WITH(t5d::tools::load_hunt_truth(dir), std::runtime_error, "order");
  // The records given to a tool: the provenance's, in its order.
  CHECK_NOTHROW(t5d::tools::check_records(prov, {"b.5dpgn", "a.5dpgn"}));
  CHECK_THROWS_WITH(t5d::tools::check_records(prov, {"a.5dpgn", "b.5dpgn"}), std::runtime_error, "record 0");
  CHECK_THROWS_WITH(t5d::tools::check_records(prov, {"b.5dpgn"}), std::runtime_error, "record 1");
  CHECK_THROWS_WITH(t5d::tools::check_records(prov, {"b.5dpgn", "a.5dpgn", "c.5dpgn"}), std::runtime_error,
                    "record 2");
  // Malformed lines are refused too, whatever the MANIFEST says.
  CHECK_THROWS_AS(t5d::tools::parse_hunt_truth(text.substr(0, text.size() - 5)), std::runtime_error);
  std::string wrong_count = text;
  wrong_count.replace(wrong_count.find("roots\t2"), 7, "roots\t3");
  CHECK_THROWS_AS(t5d::tools::parse_hunt_truth(wrong_count), std::runtime_error);
  CHECK_THROWS_AS(t5d::tools::parse_provenance(ptext.substr(0, ptext.size() - 3)), std::runtime_error);
  std::string bad_count = ptext;
  bad_count.replace(bad_count.find("records\t2"), 9, "records\t3");
  CHECK_THROWS_AS(t5d::tools::parse_provenance(bad_count), std::runtime_error);
}

TEST(hunt_truth_records_the_rules_by_every_options_value) {
  CHECK_EQ(t5d::tools::absolute_rules_text(t5d::RuleOptions{}), std::string(kDefaultRules));
  // Every option is in it: each choice changed alone changes the text.
  const auto changed = [](void (*f)(t5d::RuleOptions&)) {
    t5d::RuleOptions r;
    f(r);
    return t5d::tools::absolute_rules_text(r) != kDefaultRules;
  };
  CHECK(changed([](t5d::RuleOptions& r) { r.activation = t5d::ActivationRule::Consecutive; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.initial_credit = t5d::InitialTimelineCredit::MinusZeroToBlack; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.play_activates = true; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.moves_after_shift = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.mandatory = t5d::MandatoryBoards::AllActive; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.reactivation = t5d::Reactivation::Deferred; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.turn_zero_reachable = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.sliders_cross_void = true; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.past_self_blocks = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.earlier_playable_jump_branches = true; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.inactive_history_is_destination = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.brawn_rank_timeline_capture = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.double_step_right = t5d::DoubleStepRight::HomeRank; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.timeline_double_step = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.double_step_over_void = true; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.branch_copies_keep_unmoved = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.jumps_keep_unmoved = true; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.en_passant_brawns = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.brawns_promote = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.brawn_cross_board_promotion = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.promotions.insert(t5d::PieceKind::Knight); }));
  CHECK(changed([](t5d::RuleOptions& r) { r.castling_empty_between = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.castling_destination_test = true; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.castling_rook_path_test = true; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.castling_any_rank = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.castling_rook_two_away = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.castling_on_optional_boards = false; }));
  CHECK(changed([](t5d::RuleOptions& r) { r.family = t5d::RuleFamily::ReversedRoyalty; }));
}

TEST(hunt_truth_the_committed_file_matches_its_manifest) {
  const std::string dir = std::string(T5D_ENGINE_DIR) + "/testdata/hunt-truth";
  const HuntTruth t = t5d::tools::load_hunt_truth(dir);  // throws if refused
  CHECK_EQ(t.sha256, std::string(kCommittedTruthSha256));
  CHECK_EQ(t.value("rules"), std::string("standard\t0x0"));
  CHECK_EQ(t.provenance.rules, std::string(kDefaultRules));
  CHECK_EQ(t.provenance.records.size(), std::size_t{41});
  CHECK(!t.provenance.code.empty());
  CHECK_EQ(t.value("depth"), std::string("5"));
  CHECK(!t.value("engine").empty());
  CHECK(!t.roots.empty());
  std::size_t mates = 0;
  for (const HuntTruthRoot& r : t.roots) {
    if (!r.mate()) continue;
    ++mates;
    CHECK(r.distance >= 1 && r.distance <= 5);
    CHECK(r.crosscheck == "confirmed" || r.crosscheck == "undecided");
    CHECK(r.found_by == "literal" || r.found_by == "default" || r.found_by == "both");
  }
  std::fprintf(stderr, "  %zu roots, %zu mates, sha256 %s\n", t.roots.size(), mates, t.sha256.c_str());
  CHECK(mates > 0);
}
