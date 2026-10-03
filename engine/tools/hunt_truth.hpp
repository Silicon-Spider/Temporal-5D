// Hunt's ground truth (stage E9; design spec 15.6, 6b''; register M17, H43;
// the user's ruling 5: "one long offline search per 6b position, recorded
// once, pinned and hashed like the recorded answers").
//
// THE FILE, engine/testdata/hunt-truth/hunt-truth.tsv - text, one record per
// line, fields separated by a tab:
//   t5d-hunt-truth  1                        the format and its version
//   engine          <commit>                 the engine that recorded it
//   rules           <describe>  <encoding>   the ruleset
//   search          <text>                   the search, its class and budgets
//   depth           5                        half-turns
//   crosscheck      <text>                   how every mate was cross-checked
//   roots           <n>
//   columns         id record index fingerprint boards verdict distance
//                   found_by literal literal_nodes default default_nodes
//                   crosscheck line
//   root            <one root's fields, in the columns' order>
// record: "sha256:" and the SHA-256 of the record's bytes, CRLF read as LF
// (engine data names no path under the oracle harness; its boundary check
// refuses one);
// verdict: "mate" (proven), "none" (no mate found within the budget), or
// "skipped: <why>" (a root 6b measures nothing at); distance: half-turns, or
// "-"; found_by: "literal", "default", "both" or "-"; literal, default: each
// class's status - "proven", "exhausted" (no mate in the class) or "unknown"
// (its budget ran out); crosscheck: the mate's certificate checked against E4
// alone (t5d/certificate.hpp) - "confirmed", "undecided", or "-"; line: the
// mate's line, actions separated by " / ".
//
// THE PIN, engine/testdata/hunt-truth/MANIFEST: "<sha256>  hunt-truth.tsv" -
// the SHA-256 of the file's bytes. Readers (the 6b'' tool, the engine's
// test) refuse a file whose bytes do not match it. Never edited by hand;
// re-recording is a reviewed change (t5d-hunt-truth).

#ifndef T5D_TOOLS_HUNT_TRUTH_HPP
#define T5D_TOOLS_HUNT_TRUTH_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace t5d::tools {

struct HuntTruthRoot {
  int id = 0;
  std::string record;
  int index = 0;
  std::string fingerprint;
  int boards = 0;
  std::string verdict;
  int distance = -1;
  std::string found_by = "-";
  std::string literal = "-";
  std::uint64_t literal_nodes = 0;
  std::string deflt = "-";
  std::uint64_t default_nodes = 0;
  std::string crosscheck = "-";
  std::string line;
  bool mate() const { return verdict == "mate"; }
  bool skipped() const { return verdict.rfind("skipped", 0) == 0; }
};

struct HuntTruth {
  std::vector<std::pair<std::string, std::string>> header;  // key, value (tabs kept)
  std::vector<HuntTruthRoot> roots;
  std::string sha256;  // the file's, as loaded
  /// A header value ("" if absent).
  std::string value(std::string_view key) const;
};

inline constexpr const char* kHuntTruthFile = "hunt-truth.tsv";
inline constexpr const char* kHuntTruthManifest = "MANIFEST";

/// The file's text.
std::string format_hunt_truth(const HuntTruth& truth);
/// Parses the file's text; throws std::runtime_error on any malformed line.
HuntTruth parse_hunt_truth(std::string_view text);
/// The MANIFEST's text for a file of these bytes.
std::string hunt_truth_manifest(std::string_view file_bytes);
/// Loads `dir`'s file after checking its bytes against `dir`'s MANIFEST;
/// throws std::runtime_error (refuses) on a mismatch, a missing file or a
/// malformed one.
HuntTruth load_hunt_truth(const std::string& dir);

}  // namespace t5d::tools

#endif  // T5D_TOOLS_HUNT_TRUTH_HPP
