// t5d-hunt-truth: Hunt's ground truth (stage E9; design spec 15.6, 6b'';
// register M17, H43; the user's ruling 5: "one long offline search per 6b
// position, recorded once, pinned and hashed like the recorded answers").
//
//   t5d-hunt-truth --out DIR --engine COMMIT [options] [RECORD ...]
//   t5d-hunt-truth --curve K [options] [RECORD ...]
//   t5d-hunt-truth --provenance-only --out DIR [options] [RECORD ...]
//
// THE POSITIONS: 6b's roots (sixb_roots.hpp: the records' turn starts, every
// one, distinct by position), the records given on the command line or in
// --record-list files. A root whose side to move could capture a royal (its
// record broke C4) or that is terminal is recorded "skipped", as 6b skips it.
//
// THE SEARCH, per root: the prover (t5d/prover.hpp) with node budgets, one
// thread, fresh stores - deterministic (H15) - twice: the LITERAL question
// (seeds All, completion Every: every first move, every completion) and the
// DEFAULT class (checking seeds, canonical completions: Hunt's own), each
// within 5 half-turns. The truth is the UNION of their proofs - the literal
// class is the broadest question there is, and the default class's long
// search makes sure Hunt's own mates are in it - so Hunt's 50 ms run is
// judged against mates it did not have to find in its own class. Each class's
// status is recorded: proven, exhausted (no mate in the class) or unknown
// (its budget ran out). Roots run in parallel (--workers), each on one thread.
//
// THE CROSS-CHECK: every mate's certificate is checked against E4 alone
// (t5d/certificate.hpp: the attacker's moves replayed, the defender's every
// action from E4's listing, every mate by both detectors) - confirmed or
// undecided (its budget); a refuted certificate is an error.
//
// THE PROVENANCE (hunt_truth.hpp): beside the truth, what it was recorded
// from - the records' content ids in their order, the rules by every
// option's value, the code signature of this build (cmake/code_signature.cmake)
// - and the MANIFEST pinning both files. --provenance-only writes the
// provenance of a truth already recorded (whose MANIFEST pins the truth
// alone, as the first recorder wrote it), after checking that the records
// given rebuild its roots exactly (record, index, fingerprint) and that its
// rules are these; its code signature is "unrecorded". Nothing is searched.
//
// THE CURVE (--curve K): every K-th measured root at the budgets given, the
// mates found against the node budget (a df-pn with a node budget is
// deterministic: a mate found at n nodes is found by any budget >= n), per
// class and their union - to choose a budget past which more search stops
// finding mates. Nothing is written.
//
// The tool never names a corpus or anything under the oracle harness.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hunt_truth.hpp"
#include "sixb_roots.hpp"
#include "t5d/certificate.hpp"
#include "t5d/check.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/prover.hpp"
#include "sha256.hpp"
#include "t5d/search.hpp"
#include "t5d_code_signature.hpp"
#include "tool_support.hpp"

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  t5d::RuleOptions rules;
  std::vector<std::string> direct;
  std::vector<std::string> lists;
  std::string record_root;
  std::string out;
  std::string engine;
  std::uint64_t literal_nodes = 4'000'000;
  std::uint64_t default_nodes = 4'000'000;
  int depth = 5;
  int workers = 4;
  std::size_t max_roots = 0;
  int curve = 0;
  std::size_t max_tree_nodes = std::size_t{2} << 20;
  std::size_t check_positions = 200000;
  bool provenance_only = false;
};

[[noreturn]] void usage(int status) {
  std::FILE* f = status == 0 ? stdout : stderr;
  std::fputs(
      "t5d-hunt-truth " T5D_VERSION_STRING " - Hunt's ground truth for 6b'' (design spec 15.6; M17, H43)\n"
      "\n"
      "usage: t5d-hunt-truth --out DIR --engine COMMIT [options] [RECORD ...]\n"
      "       t5d-hunt-truth --curve K [options] [RECORD ...]\n"
      "       t5d-hunt-truth --provenance-only --out DIR [options] [RECORD ...]\n"
      "  --record-list FILE   records listed one per line ('#' lines skipped); ids as listed\n"
      "  --record-root DIR    ... their relative paths taken from DIR\n"
      "  --rules ENCODING  --no-moves-after-shift   the ruleset (default 0, the engine's)\n"
      "  --literal-nodes N    the literal question's node budget per root (4000000)\n"
      "  --default-nodes N    the default class's (4000000)\n"
      "  --depth D            half-turns (5)\n"
      "  --workers W          roots searched at once, one thread each (4)\n"
      "  --max-tree-nodes N   each search's tree (2097152)\n"
      "  --check-positions N  the cross-check's defender positions per mate (200000)\n"
      "  --max-roots N        the first N roots only (information)\n"
      "  --out DIR            write DIR/hunt-truth.tsv, DIR/provenance.tsv and DIR/MANIFEST\n"
      "  --engine COMMIT      the engine commit the header names\n"
      "  --curve K            every K-th measured root: mates found against the budget; nothing written\n"
      "  --provenance-only    DIR's truth recorded already: check it against the records and rules given,\n"
      "                       write its provenance and the MANIFEST pinning both; nothing searched\n"
      "  --version            the version and this build's code signature\n",
      f);
  std::exit(status);
}

std::uint64_t number(const char* text) {
  const std::optional<std::uint64_t> n = t5d::tools::parse_number(text);
  if (!n) throw std::invalid_argument(std::string("not a number: ") + text);
  return *n;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    if (const int used = t5d::tools::ruleset_option(argc, argv, i, o.rules)) {
      i += used - 1;
    } else if (a == "--help" || a == "-h") {
      usage(0);
    } else if (a == "--version") {
      std::puts("t5d-hunt-truth " T5D_VERSION_STRING ", code " T5D_CODE_SIGNATURE);
      std::exit(0);
    } else if (a == "--provenance-only") {
      o.provenance_only = true;
    } else if (a == "--record-list") {
      o.lists.push_back(value());
    } else if (a == "--record-root") {
      o.record_root = value();
    } else if (a == "--literal-nodes") {
      o.literal_nodes = number(value());
    } else if (a == "--default-nodes") {
      o.default_nodes = number(value());
    } else if (a == "--depth") {
      o.depth = static_cast<int>(number(value()));
    } else if (a == "--workers") {
      o.workers = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--max-tree-nodes") {
      o.max_tree_nodes = static_cast<std::size_t>(number(value()));
    } else if (a == "--check-positions") {
      o.check_positions = static_cast<std::size_t>(number(value()));
    } else if (a == "--max-roots") {
      o.max_roots = static_cast<std::size_t>(number(value()));
    } else if (a == "--out") {
      o.out = value();
    } else if (a == "--engine") {
      o.engine = value();
    } else if (a == "--curve") {
      o.curve = static_cast<int>(number(value()));
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("unknown option " + a);
    } else {
      o.direct.push_back(a);
    }
  }
  if (o.provenance_only) {
    if (o.out.empty()) throw std::invalid_argument("--provenance-only needs --out DIR");
  } else if (o.curve == 0 && (o.out.empty() || o.engine.empty())) {
    throw std::invalid_argument("--out DIR and --engine COMMIT are needed (or --curve K)");
  }
  return o;
}

void write_file(const std::string& path, const std::string& bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("cannot write " + path);
  const std::size_t n = std::fwrite(bytes.data(), 1, bytes.size(), f);
  if (std::fclose(f) != 0 || n != bytes.size()) throw std::runtime_error("cannot write " + path);
}

std::string encoding_text(const t5d::RuleOptions& rules) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "0x%llx", static_cast<unsigned long long>(rules.encode()));
  return buf;
}

// --provenance-only: the provenance of DIR's truth, recorded already (its
// MANIFEST pinning the truth alone), after checking it against the records
// and the rules given.
int provenance_only(const Options& o, const std::vector<t5d::tools::RecordRef>& refs) {
  const std::string bytes = t5d::tools::read_file(o.out + "/" + t5d::tools::kHuntTruthFile);
  const std::string manifest = t5d::tools::read_file(o.out + "/" + t5d::tools::kHuntTruthManifest);
  const std::string hash = t5d::tools::sha256_hex(bytes);
  if (manifest != hash + "  " + t5d::tools::kHuntTruthFile + "\n") {
    throw std::runtime_error("the truth does not match its MANIFEST (a MANIFEST pinning the truth alone, as the "
                             "first recorder wrote it, is expected)");
  }
  t5d::tools::HuntTruth truth = t5d::tools::parse_hunt_truth(bytes);
  truth.sha256 = hash;
  // Its rules are these: the header's encoding read back is the ruleset given.
  const std::string rules = truth.value("rules");
  const std::size_t tab = rules.rfind('\t');
  if (tab == std::string::npos || rules.substr(tab + 1) != encoding_text(o.rules) ||
      rules.substr(0, tab) != t5d::tools::ruleset_text(o.rules)) {
    throw std::runtime_error("the truth's rules (" + rules + ") are not the ones given (" +
                             t5d::tools::ruleset_text(o.rules) + "\t" + encoding_text(o.rules) + ")");
  }
  if (std::stoi(truth.value("depth")) != o.depth) throw std::runtime_error("the truth's depth is not --depth");
  // The records rebuild its roots exactly.
  const t5d::tools::SixbRoots roots = t5d::tools::sixb_roots(refs, o.rules, 1, 0);
  if (roots.roots.size() != truth.roots.size()) {
    throw std::runtime_error("the records give " + std::to_string(roots.roots.size()) + " roots, the truth holds " +
                             std::to_string(truth.roots.size()));
  }
  for (std::size_t i = 0; i < roots.roots.size(); ++i) {
    const t5d::tools::SixbRoot& r = roots.roots[i];
    const t5d::tools::HuntTruthRoot& row = truth.roots[i];
    if (row.record != r.content || row.index != r.index ||
        row.fingerprint != t5d::fingerprint(r.state.multiverse()).fingerprint) {
      throw std::runtime_error("root " + std::to_string(i) + " (" + r.record + " #" + std::to_string(r.index) +
                               ") is not the truth's");
    }
  }
  t5d::tools::HuntTruthProvenance p;
  p.truth = "sha256:" + hash;
  p.rules = t5d::tools::absolute_rules_text(o.rules);
  p.code = "unrecorded: recorded at engine " + truth.value("engine") +
           " by the first recorder, before code signatures; provenance written by code " T5D_CODE_SIGNATURE;
  p.records = t5d::tools::record_contents(refs);
  for (const std::string& c : p.records) {
    if (c.rfind("sha256:", 0) != 0) throw std::runtime_error("a record cannot be read: " + c);
  }
  t5d::tools::check_provenance(truth, p);
  const std::string text = t5d::tools::format_provenance(p);
  (void)t5d::tools::parse_provenance(text);
  write_file(o.out + "/" + t5d::tools::kHuntTruthProvenance, text);
  write_file(o.out + "/" + t5d::tools::kHuntTruthManifest, t5d::tools::hunt_truth_manifest(bytes, text));
  (void)t5d::tools::load_hunt_truth(o.out);  // the round trip: it loads as a reader loads it
  std::fprintf(stderr, "t5d-hunt-truth: the provenance of %s (sha256 %s): %zu records, %zu roots - written\n",
               o.out.c_str(), hash.c_str(), p.records.size(), truth.roots.size());
  return 0;
}

std::string status_word(const t5d::ProveResult& r) {
  switch (r.status) {
    case t5d::ProofStatus::Proven: return "proven";
    case t5d::ProofStatus::Disproven: return "exhausted";
    case t5d::ProofStatus::Unknown: return "unknown";
  }
  return "?";
}

std::string line_text(const std::vector<std::vector<t5d::Move>>& line, bool even) {
  std::string out;
  for (const std::vector<t5d::Move>& action : line) {
    if (!out.empty()) out += " / ";
    out += t5d::action_text(action, even);
  }
  return out;
}

// Why 6b measures nothing at `s`, or "".
std::string skipped(const t5d::GameState& s) {
  if (t5d::royal_capture_exists(s.multiverse(), s.rules(), s.to_move(), s.inactive_timelines())) {
    return "skipped: a royal capturable by the side to move";
  }
  t5d::SearchBudget b = t5d::SearchBudget::nodes(2'000'000);
  b.max_bytes = std::size_t{256} << 20;
  const t5d::MateResult m = t5d::detect_mate(s, t5d::Detector::Hypercuboid, b);
  if (m.verdict == t5d::Outcome::Checkmate) return "skipped: terminal (checkmate)";
  if (m.verdict == t5d::Outcome::Stalemate) return "skipped: terminal (stalemate)";
  return "";
}

struct Run {
  t5d::ProveResult literal;
  t5d::ProveResult deflt;
  std::string skip;
  t5d::CertificateCheck check;
  bool checked = false;
};

t5d::ProverOptions class_options(const Options& o, bool literal) {
  t5d::ProverOptions p;
  p.depth = o.depth;
  p.threads = 1;
  p.max_tree_nodes = o.max_tree_nodes;
  p.record_certificate = true;
  if (literal) {
    p.seeds = t5d::SeedClass::All;
    p.completion = t5d::CompletionClass::Every;
  }
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = parse(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-hunt-truth: %s\n", e.what());
    usage(2);
  }
  std::vector<t5d::tools::RecordRef> refs;
  try {
    refs = t5d::tools::record_refs(o.direct, o.lists, o.record_root);
    if (o.provenance_only) return provenance_only(o, refs);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-hunt-truth: %s\n", e.what());
    return 1;
  }
  const t5d::tools::SixbRoots roots = t5d::tools::sixb_roots(refs, o.rules, 1, o.max_roots);
  for (const std::string& n : roots.notes) std::fprintf(stderr, "  %s\n", n.c_str());
  const std::size_t n = roots.roots.size();
  std::vector<Run> runs(n);
  std::vector<std::size_t> todo;
  for (std::size_t i = 0; i < n; ++i) {
    runs[i].skip = skipped(roots.roots[i].state);
    if (runs[i].skip.empty()) todo.push_back(i);
  }
  if (o.curve > 0) {
    std::vector<std::size_t> sample;
    for (std::size_t k = 0; k < todo.size(); k += static_cast<std::size_t>(o.curve)) sample.push_back(todo[k]);
    todo = sample;
  }
  std::fprintf(stderr, "t5d-hunt-truth: %zu roots, %zu searched (%d workers)\n", n, todo.size(), o.workers);

  const Clock::time_point t0 = Clock::now();
  std::atomic<std::size_t> next{0};
  std::atomic<std::size_t> done{0};
  std::mutex out_mutex;
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(o.workers));
  const auto work = [&](std::size_t w) {
    try {
      for (;;) {
        const std::size_t k = next.fetch_add(1);
        if (k >= todo.size()) return;
        const std::size_t i = todo[k];
        const t5d::GameState& s = roots.roots[i].state;
        Run& r = runs[i];
        {
          t5d::Prover lit(class_options(o, true));
          r.literal = lit.prove(s, t5d::ProveBudget::nodes(o.literal_nodes));
        }
        {
          t5d::Prover def(class_options(o, false));
          r.deflt = def.prove(s, t5d::ProveBudget::nodes(o.default_nodes));
        }
        // The cross-check: the mate's certificate (the shorter proof's; the
        // literal one's on a tie), against E4 alone.
        const t5d::ProveResult* m = nullptr;
        if (r.literal.status == t5d::ProofStatus::Proven) m = &r.literal;
        if (r.deflt.status == t5d::ProofStatus::Proven &&
            (m == nullptr || r.deflt.proof.distance < m->proof.distance)) {
          m = &r.deflt;
        }
        if (m != nullptr && o.curve == 0 && m->certificate) {
          t5d::CertificateCheckOptions co;
          co.max_positions = o.check_positions;
          r.check = t5d::check_certificate(*m->certificate, s, m->proof.distance, co);
          r.checked = true;
        }
        // The trees and certificates are not kept.
        r.literal.certificate.reset();
        r.deflt.certificate.reset();
        const std::size_t d = done.fetch_add(1) + 1;
        if (d % 25 == 0) {
          std::lock_guard<std::mutex> lock(out_mutex);
          std::fprintf(stderr, "t5d-hunt-truth: %zu of %zu roots, %.0f s\n", d, todo.size(),
                       std::chrono::duration<double>(Clock::now() - t0).count());
        }
      }
    } catch (...) {
      errors[w] = std::current_exception();
    }
  };
  {
    std::vector<std::thread> pool;
    for (int w = 0; w < o.workers; ++w) pool.emplace_back(work, static_cast<std::size_t>(w));
    for (std::thread& t : pool) t.join();
  }
  for (const std::exception_ptr& e : errors) {
    if (e) {
      try {
        std::rethrow_exception(e);
      } catch (const std::exception& ex) {
        std::fprintf(stderr, "t5d-hunt-truth: %s\n", ex.what());
        return 1;
      }
    }
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();

  if (o.curve > 0) {
    // Mates found against the node budget, per class and their union.
    std::vector<std::uint64_t> budgets;
    const std::uint64_t top = std::max(o.literal_nodes, o.default_nodes);
    for (std::uint64_t b = top; b >= 1000 && budgets.size() < 14; b /= 2) budgets.push_back(b);
    std::reverse(budgets.begin(), budgets.end());
    std::printf("t5d-hunt-truth curve: %zu roots (every %d-th measured), depth %d, %.0f s\n", todo.size(), o.curve,
                o.depth, seconds);
    std::printf("%12s %8s %8s %8s\n", "budget", "literal", "default", "union");
    for (const std::uint64_t b : budgets) {
      std::size_t lit = 0, def = 0, uni = 0;
      for (const std::size_t i : todo) {
        const bool l = runs[i].literal.status == t5d::ProofStatus::Proven && runs[i].literal.stats.nodes <= b &&
                       b <= o.literal_nodes;
        const bool d = runs[i].deflt.status == t5d::ProofStatus::Proven && runs[i].deflt.stats.nodes <= b &&
                       b <= o.default_nodes;
        lit += l ? std::size_t{1} : std::size_t{0};
        def += d ? std::size_t{1} : std::size_t{0};
        uni += (l || d) ? std::size_t{1} : std::size_t{0};
      }
      std::printf("%12llu %8zu %8zu %8zu\n", static_cast<unsigned long long>(b), lit, def, uni);
    }
    std::size_t lit_ex = 0, def_ex = 0, lit_unk = 0, def_unk = 0, mem = 0;
    for (const std::size_t i : todo) {
      lit_ex += runs[i].literal.status == t5d::ProofStatus::Disproven ? std::size_t{1} : std::size_t{0};
      def_ex += runs[i].deflt.status == t5d::ProofStatus::Disproven ? std::size_t{1} : std::size_t{0};
      lit_unk += runs[i].literal.status == t5d::ProofStatus::Unknown ? std::size_t{1} : std::size_t{0};
      def_unk += runs[i].deflt.status == t5d::ProofStatus::Unknown ? std::size_t{1} : std::size_t{0};
      mem += (runs[i].literal.stats.stopped_by_memory || runs[i].deflt.stats.stopped_by_memory) ? std::size_t{1} : std::size_t{0};
    }
    std::printf("exhausted: literal %zu, default %zu; unknown at the top budget: literal %zu, default %zu; "
                "stopped by the tree's memory: %zu\n",
                lit_ex, def_ex, lit_unk, def_unk, mem);
    for (const std::size_t i : todo) {
      const t5d::tools::SixbRoot& r = roots.roots[i];
      std::printf("  %s #%d (%zu boards): literal %s %llu, default %s %llu\n", r.record.c_str(), r.index,
                  r.state.multiverse().board_count(), status_word(runs[i].literal).c_str(),
                  static_cast<unsigned long long>(runs[i].literal.stats.nodes), status_word(runs[i].deflt).c_str(),
                  static_cast<unsigned long long>(runs[i].deflt.stats.nodes));
    }
    return 0;
  }

  // The file.
  t5d::tools::HuntTruth truth;
  truth.header = {
      {"t5d-hunt-truth", "1"},
      {"engine", o.engine},
      {"rules", t5d::tools::ruleset_text(o.rules) + "\t" + encoding_text(o.rules)},
      {"search", "df-pn (t5d/prover.hpp), one thread per root, fresh stores, node budgets (deterministic); "
                 "the union of two classes' proofs: literal (seeds all, completion every) " +
                     std::to_string(o.literal_nodes) + " nodes, default (seeds checks, completion canonical) " +
                     std::to_string(o.default_nodes) + " nodes; tree cap " + std::to_string(o.max_tree_nodes) +
                     " nodes"},
      {"depth", std::to_string(o.depth)},
      {"crosscheck", "every mate's certificate against E4 alone (t5d/certificate.hpp: the defender's every action "
                     "by E4's listing, every mate by both detectors), at most " +
                         std::to_string(o.check_positions) + " defender positions each"},
      {"roots", std::to_string(n)},
  };
  std::size_t mates = 0, refuted = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const t5d::tools::SixbRoot& root = roots.roots[i];
    const Run& r = runs[i];
    t5d::tools::HuntTruthRoot row;
    row.id = static_cast<int>(i);
    row.record = root.content;  // the record by its content: no path under the oracle harness
    row.index = root.index;
    row.fingerprint = t5d::fingerprint(root.state.multiverse()).fingerprint;
    row.boards = static_cast<int>(root.state.multiverse().board_count());
    if (!r.skip.empty()) {
      row.verdict = r.skip;
      truth.roots.push_back(row);
      continue;
    }
    row.literal = status_word(r.literal);
    row.literal_nodes = r.literal.stats.nodes;
    row.deflt = status_word(r.deflt);
    row.default_nodes = r.deflt.stats.nodes;
    const bool lit = r.literal.status == t5d::ProofStatus::Proven;
    const bool def = r.deflt.status == t5d::ProofStatus::Proven;
    if (!lit && !def) {
      row.verdict = "none";
      truth.roots.push_back(row);
      continue;
    }
    ++mates;
    row.verdict = "mate";
    const t5d::ProveResult& m =
        (lit && (!def || r.literal.proof.distance <= r.deflt.proof.distance)) ? r.literal : r.deflt;
    row.distance = m.proof.distance;
    row.found_by = lit && def ? "both" : lit ? "literal" : "default";
    row.line = line_text(m.line, root.state.multiverse().even());
    if (!m.line_complete) row.line += " (incomplete)";
    if (r.checked) {
      switch (r.check.verdict) {
        case t5d::CertificateVerdict::Confirmed: row.crosscheck = "confirmed"; break;
        case t5d::CertificateVerdict::Undecided: row.crosscheck = "undecided"; break;
        default:
          row.crosscheck = std::string(t5d::certificate_verdict_name(r.check.verdict)) + ": " + r.check.why;
          ++refuted;
          break;
      }
    }
    truth.roots.push_back(row);
  }
  if (refuted > 0) {
    std::fprintf(stderr, "t5d-hunt-truth: %zu mate(s) NOT confirmed by E4 - not written\n", refuted);
    for (const t5d::tools::HuntTruthRoot& row : truth.roots) {
      if (row.mate() && row.crosscheck != "confirmed" && row.crosscheck != "undecided") {
        std::fprintf(stderr, "  %s #%d: %s\n", row.record.c_str(), row.index, row.crosscheck.c_str());
      }
    }
    return 1;
  }
  const std::string text = t5d::tools::format_hunt_truth(truth);
  // A round trip before writing.
  (void)t5d::tools::parse_hunt_truth(text);
  try {
    t5d::tools::HuntTruthProvenance p;
    p.truth = "sha256:" + t5d::tools::sha256_hex(text);
    p.rules = t5d::tools::absolute_rules_text(o.rules);
    p.code = T5D_CODE_SIGNATURE;
    p.records = t5d::tools::record_contents(refs);
    const std::string provenance = t5d::tools::format_provenance(p);
    write_file(o.out + "/" + t5d::tools::kHuntTruthFile, text);
    write_file(o.out + "/" + t5d::tools::kHuntTruthProvenance, provenance);
    write_file(o.out + "/" + t5d::tools::kHuntTruthManifest, t5d::tools::hunt_truth_manifest(text, provenance));
    (void)t5d::tools::load_hunt_truth(o.out);  // the round trip: it loads as a reader loads it
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-hunt-truth: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "t5d-hunt-truth: %zu roots, %zu mates; %.0f s; written to %s\n", n, mates, seconds,
               o.out.c_str());
  return 0;
}
