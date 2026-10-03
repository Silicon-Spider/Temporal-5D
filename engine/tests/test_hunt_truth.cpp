// Hunt's ground truth, pinned and hashed (stage E9; the user's ruling 5;
// register M17): SHA-256 against the standard's vectors; the file's format
// round-trips; a copy whose bytes do not match its MANIFEST is refused; and
// the committed truth (engine/testdata/hunt-truth/) matches its MANIFEST and
// is well formed. That its roots are 6b's roots - record, index and
// fingerprint - the 6b'' tool checks against the records it is given (the
// engine's tests read no corpus).

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "hunt_truth.hpp"
#include "sha256.hpp"
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
  // Written with its MANIFEST: loaded; one byte changed: refused.
  const std::string dir = (std::filesystem::path(T5D_TEST_SCRATCH) / "hunt-truth").string();
  std::filesystem::create_directories(dir);
  write(dir + "/" + t5d::tools::kHuntTruthFile, text);
  write(dir + "/" + t5d::tools::kHuntTruthManifest, t5d::tools::hunt_truth_manifest(text));
  const HuntTruth loaded = t5d::tools::load_hunt_truth(dir);
  CHECK_EQ(loaded.sha256, t5d::tools::sha256_hex(text));
  std::string changed = text;
  changed[changed.find("mate")] = 'n';  // "nate": one byte
  write(dir + "/" + t5d::tools::kHuntTruthFile, changed);
  CHECK_THROWS_AS(t5d::tools::load_hunt_truth(dir), std::runtime_error);
  // Malformed lines are refused too, whatever the MANIFEST says.
  CHECK_THROWS_AS(t5d::tools::parse_hunt_truth(text.substr(0, text.size() - 5)), std::runtime_error);
  std::string wrong_count = text;
  wrong_count.replace(wrong_count.find("roots\t2"), 7, "roots\t3");
  CHECK_THROWS_AS(t5d::tools::parse_hunt_truth(wrong_count), std::runtime_error);
}

TEST(hunt_truth_the_committed_file_matches_its_manifest) {
  const std::string dir = std::string(T5D_ENGINE_DIR) + "/testdata/hunt-truth";
  const HuntTruth t = t5d::tools::load_hunt_truth(dir);  // throws if refused
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
