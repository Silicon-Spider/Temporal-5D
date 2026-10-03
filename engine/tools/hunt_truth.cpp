// Hunt's ground truth: the file and its pin (see hunt_truth.hpp).

#include "hunt_truth.hpp"

#include <cstdlib>
#include <stdexcept>

#include "sha256.hpp"
#include "tool_support.hpp"

namespace t5d::tools {

namespace {

std::vector<std::string> split_tabs(std::string_view line) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (;;) {
    const std::size_t tab = line.find('\t', start);
    out.emplace_back(line.substr(start, tab == std::string_view::npos ? std::string_view::npos : tab - start));
    if (tab == std::string_view::npos) break;
    start = tab + 1;
  }
  return out;
}

long long number(const std::string& text, const std::string& what) {
  if (text.empty()) throw std::runtime_error("hunt truth: an empty " + what);
  char* end = nullptr;
  const long long v = std::strtoll(text.c_str(), &end, 10);
  if (end == nullptr || *end != '\0') throw std::runtime_error("hunt truth: not a number: " + what + " " + text);
  return v;
}

constexpr const char* kColumns =
    "id\trecord\tindex\tfingerprint\tboards\tverdict\tdistance\tfound_by\tliteral\tliteral_nodes\tdefault\t"
    "default_nodes\tcrosscheck\tline";
constexpr std::size_t kColumnCount = 14;

}  // namespace

std::string HuntTruth::value(std::string_view key) const {
  for (const auto& [k, v] : header) {
    if (k == key) return v;
  }
  return "";
}

std::string format_hunt_truth(const HuntTruth& truth) {
  std::string out;
  for (const auto& [k, v] : truth.header) out += k + "\t" + v + "\n";
  out += std::string("columns\t") + kColumns + "\n";
  for (const HuntTruthRoot& r : truth.roots) {
    out += "root\t" + std::to_string(r.id) + "\t" + r.record + "\t" + std::to_string(r.index) + "\t" +
           r.fingerprint + "\t" + std::to_string(r.boards) + "\t" + r.verdict + "\t" +
           (r.distance < 0 ? std::string("-") : std::to_string(r.distance)) + "\t" + r.found_by + "\t" + r.literal +
           "\t" + std::to_string(r.literal_nodes) + "\t" + r.deflt + "\t" + std::to_string(r.default_nodes) + "\t" +
           r.crosscheck + "\t" + (r.line.empty() ? std::string("-") : r.line) + "\n";
  }
  return out;
}

HuntTruth parse_hunt_truth(std::string_view text) {
  // Every line ends in a newline: a file cut short is malformed.
  if (text.empty() || text.back() != '\n') throw std::runtime_error("hunt truth: does not end in a newline");
  HuntTruth t;
  bool columns = false;
  std::size_t start = 0;
  int line_no = 0;
  while (start < text.size()) {
    std::size_t end = text.find('\n', start);
    if (end == std::string_view::npos) end = text.size();
    std::string_view line = text.substr(start, end - start);
    start = end + 1;
    ++line_no;
    if (!line.empty() && line.back() == '\r') throw std::runtime_error("hunt truth: a CR at line " + std::to_string(line_no));
    if (line.empty()) throw std::runtime_error("hunt truth: an empty line at " + std::to_string(line_no));
    const std::vector<std::string> f = split_tabs(line);
    if (f[0] == "columns") {
      if (line.substr(8) != kColumns) throw std::runtime_error("hunt truth: unexpected columns");
      columns = true;
      continue;
    }
    if (f[0] == "root") {
      if (!columns) throw std::runtime_error("hunt truth: a root before the columns");
      if (f.size() != kColumnCount + 1) {
        throw std::runtime_error("hunt truth: " + std::to_string(f.size() - 1) + " fields at line " +
                                 std::to_string(line_no));
      }
      HuntTruthRoot r;
      r.id = static_cast<int>(number(f[1], "id"));
      r.record = f[2];
      r.index = static_cast<int>(number(f[3], "index"));
      r.fingerprint = f[4];
      r.boards = static_cast<int>(number(f[5], "boards"));
      r.verdict = f[6];
      r.distance = f[7] == "-" ? -1 : static_cast<int>(number(f[7], "distance"));
      r.found_by = f[8];
      r.literal = f[9];
      r.literal_nodes = static_cast<std::uint64_t>(number(f[10], "literal_nodes"));
      r.deflt = f[11];
      r.default_nodes = static_cast<std::uint64_t>(number(f[12], "default_nodes"));
      r.crosscheck = f[13];
      r.line = f[14] == "-" ? std::string() : f[14];
      if (r.verdict != "mate" && r.verdict != "none" && !r.skipped()) {
        throw std::runtime_error("hunt truth: a verdict it does not know: " + r.verdict);
      }
      if (r.mate() != (r.distance > 0)) throw std::runtime_error("hunt truth: a mate's distance");
      if (r.id != static_cast<int>(t.roots.size())) throw std::runtime_error("hunt truth: roots out of order");
      t.roots.push_back(std::move(r));
      continue;
    }
    if (columns) throw std::runtime_error("hunt truth: a header line after the columns: " + f[0]);
    std::string value;
    for (std::size_t i = 1; i < f.size(); ++i) value += (i > 1 ? "\t" : "") + f[i];
    t.header.emplace_back(f[0], value);
  }
  if (t.value("t5d-hunt-truth") != "1") throw std::runtime_error("hunt truth: not a version-1 truth file");
  for (const char* key : {"engine", "rules", "search", "depth", "crosscheck", "roots"}) {
    if (t.value(key).empty()) throw std::runtime_error(std::string("hunt truth: no ") + key);
  }
  if (number(t.value("roots"), "roots") != static_cast<long long>(t.roots.size())) {
    throw std::runtime_error("hunt truth: the root count does not match its rows");
  }
  return t;
}

std::string hunt_truth_manifest(std::string_view file_bytes) {
  return sha256_hex(file_bytes) + "  " + kHuntTruthFile + "\n";
}

HuntTruth load_hunt_truth(const std::string& dir) {
  const std::string bytes = read_file(dir + "/" + kHuntTruthFile);
  const std::string manifest = read_file(dir + "/" + kHuntTruthManifest);
  const std::string hash = sha256_hex(bytes);
  if (manifest != hunt_truth_manifest(bytes)) {
    throw std::runtime_error("hunt truth: " + dir + "/" + kHuntTruthFile + " does not match its MANIFEST (sha256 " +
                             hash + ") - refused: never edit it by hand; re-record it (t5d-hunt-truth)");
  }
  HuntTruth t = parse_hunt_truth(bytes);
  t.sha256 = hash;
  return t;
}

}  // namespace t5d::tools
