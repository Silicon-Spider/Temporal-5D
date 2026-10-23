// t5d-solver-check: the search's solver against df-pn, offline, at the turn
// starts of the records given (register H41: the solver's proofs "are
// cross-checked against df-pn"; design spec §11.1, §15.4; §11.4's "above 100
// boards, proofs come only from the search's solver" - this tool says how far
// df-pn can follow it there).
//
//   t5d-solver-check [options] [RECORD ...]
//
// POSITIONS - 6b's roots construction (sixb_roots.hpp): the turn starts of
// the 5dpgn records given (parse mode, the ruleset given; read as 6b read
// them), every K-th, distinct by position. A root whose side to move could
// capture a royal (its record broke C4), or that is terminal, is counted and
// not searched - as 6b, 6b' and 6b'' do.
//
// AT EACH ROOT, for each D2 arm asked (--split, default all three: the ply
// split, the tree form, the flat form), the search runs with its proof
// channel ON - offline: nothing is exported, no store exists (H23, H41) -
// under exact legality, the uniform evaluator, a playout budget; then every
// solved turn start's claim (Win(k), Loss(k), Draw, derived from the solved
// subtree) is compared with df-pn in its literal class (one thread, a node
// budget: deterministic), and every df-pn proof certified by E4 alone
// (tools/solver_proofs.hpp has the rules). Read PER BOARD-COUNT RANGE - < 50,
// 50-99, 100-199, 200+ - never pooled.
//
// Exit status: 0 no contradiction and no certificate failure; 3 a
// contradiction or a failure (each printed); 1 the records could not be read;
// 2 a bad command line.
//
// The tool never names a corpus or anything under the oracle harness: the
// records come on the command line.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sixb_roots.hpp"
#include "solver_proofs.hpp"
#include "t5d/check.hpp"
#include "t5d/mcts.hpp"
#include "t5d/search.hpp"
#include "tool_support.hpp"

namespace {

using t5d::mcts::TurnSplit;

struct Options {
  std::vector<std::string> direct;
  std::vector<std::string> lists;
  std::string record_root;
  t5d::RuleOptions rules;
  int every = 1;
  std::size_t max_roots = 0;
  int min_boards = 0;
  int max_boards = 1 << 30;
  std::vector<TurnSplit> splits;
  std::uint64_t playouts = 400;
  std::uint64_t mate_nodes = 2'000'000;
  /// 0: exact move legality; else exact within this many nodes per
  /// classification (LegalityOptions::nodes; build-log issue 406).
  std::uint64_t legality_nodes = 0;
  t5d::tools::DfpnCheckOptions dfpn;
  bool per_position = false;
};

[[noreturn]] void usage(int status) {
  std::FILE* out = status == 0 ? stdout : stderr;
  std::fputs(
      "t5d-solver-check " T5D_VERSION_STRING
      " - the search's solver against df-pn at the turn starts of the records given (register H41,\n"
      "design spec 11.1, 15.4), per board-count range. Offline: the proof channel is on in the tool's\n"
      "own searches; nothing is exported.\n"
      "\n"
      "usage: t5d-solver-check [options] [RECORD ...]\n"
      " positions (6b's roots construction)\n"
      "  --record-list FILE         records listed one per line ('#' lines skipped); ids as listed\n"
      "  --record-root DIR          ... their relative paths taken from DIR\n"
      "  --rules ENCODING           the ruleset (RuleOptions::encode; default 0, the engine's)\n"
      "  --no-moves-after-shift     a diagnostic: Q4's Guide-literal reading\n"
      "  --every K (1)  --max-roots N  --min-boards N  --max-boards N\n"
      " the search (the proof channel on, exact legality, the uniform evaluator)\n"
      "  --split ply|tree|flat      a D2 arm (repeatable; default all three)\n"
      "  --playouts N (400)         each search's playouts\n"
      "  --mate-nodes N (2000000)   E4's budget at each turn start the search meets\n"
      "  --legality-nodes N (0)     move legality exact within N nodes per classification (0: no\n"
      "                             budget); a move left unknown proves nothing\n"
      " df-pn (the literal class, one thread)\n"
      "  --dfpn-nodes N (200000)    each question's node budget\n"
      "  --max-depth D (9)          deeper claims are counted, not asked\n"
      "  --draw-depth D (3)         the depth a Draw claim is asked at\n"
      "  --no-certify               skip E4's check of df-pn's proofs\n"
      "  --check-positions N        the certificate check's defender positions each (20000)\n"
      " output\n"
      "  --per-position  --help  --version\n",
      out);
  std::exit(status);
}

std::uint64_t number(const char* text) {
  const std::optional<std::uint64_t> n = t5d::tools::parse_number(text);
  if (!n) throw std::invalid_argument(std::string("not a number: ") + text);
  return *n;
}

Options parse(int argc, char** argv) {
  Options o;
  o.dfpn.certificate.max_positions = 20000;
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
      std::puts("t5d-solver-check " T5D_VERSION_STRING);
      std::exit(0);
    } else if (a == "--record-list") {
      o.lists.push_back(value());
    } else if (a == "--record-root") {
      o.record_root = value();
    } else if (a == "--every") {
      o.every = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--max-roots") {
      o.max_roots = static_cast<std::size_t>(number(value()));
    } else if (a == "--min-boards") {
      o.min_boards = static_cast<int>(number(value()));
    } else if (a == "--max-boards") {
      o.max_boards = static_cast<int>(number(value()));
    } else if (a == "--split") {
      const std::string v = value();
      if (v == "ply") {
        o.splits.push_back(TurnSplit::BoardMove);
      } else if (v == "tree") {
        o.splits.push_back(TurnSplit::Tree);
      } else if (v == "flat") {
        o.splits.push_back(TurnSplit::Flat);
      } else {
        throw std::invalid_argument("--split takes ply, tree or flat, not " + v);
      }
    } else if (a == "--playouts") {
      o.playouts = std::max<std::uint64_t>(1, number(value()));
    } else if (a == "--mate-nodes") {
      o.mate_nodes = number(value());
    } else if (a == "--legality-nodes") {
      o.legality_nodes = number(value());
    } else if (a == "--dfpn-nodes") {
      o.dfpn.nodes = number(value());
    } else if (a == "--max-depth") {
      o.dfpn.max_depth = static_cast<int>(number(value()));
    } else if (a == "--draw-depth") {
      o.dfpn.draw_depth = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--no-certify") {
      o.dfpn.certify = false;
    } else if (a == "--check-positions") {
      o.dfpn.certificate.max_positions = static_cast<std::size_t>(number(value()));
    } else if (a == "--per-position") {
      o.per_position = true;
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("unknown option " + a);
    } else {
      o.direct.push_back(a);
    }
  }
  if (o.direct.empty() && o.lists.empty()) throw std::invalid_argument("no records given");
  if (o.splits.empty()) o.splits = {TurnSplit::BoardMove, TurnSplit::Tree, TurnSplit::Flat};
  return o;
}

const char* split_name(TurnSplit s) {
  switch (s) {
    case TurnSplit::BoardMove: return "ply";
    case TurnSplit::Tree: return "tree";
    case TurnSplit::Flat: return "flat";
  }
  return "?";
}

constexpr std::array<const char*, 4> kRanges{"<50", "50-99", "100-199", "200+"};
int range_of(int boards) { return boards < 50 ? 0 : boards < 100 ? 1 : boards < 200 ? 2 : 3; }

struct Range {
  std::size_t roots = 0;
  std::size_t searches = 0;
  std::size_t roots_solved = 0;
  t5d::tools::DfpnTally tally;
  double seconds = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = parse(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-solver-check: %s\n", e.what());
    usage(2);
  }
  std::vector<t5d::tools::RecordRef> refs;
  try {
    refs = t5d::tools::record_refs(o.direct, o.lists, o.record_root);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-solver-check: %s\n", e.what());
    return 1;
  }
  const t5d::tools::SixbRoots roots = t5d::tools::sixb_roots(refs, o.rules, o.every, o.max_roots);
  std::printf("CONDITIONS rules=%s records=%zu roots=%zu (duplicates %zu) every=%d splits=",
              t5d::tools::ruleset_text(o.rules).c_str(), refs.size(), roots.roots.size(), roots.duplicates, o.every);
  for (std::size_t i = 0; i < o.splits.size(); ++i) std::printf("%s%s", i ? "," : "", split_name(o.splits[i]));
  const std::string legality =
      o.legality_nodes == 0 ? std::string("exact") : "exact within " + std::to_string(o.legality_nodes) + " nodes";
  std::printf(" playouts=%llu mate_nodes=%llu evaluator=uniform legality=%s proof_channel=on(offline, no store) "
              "dfpn=literal,1 thread,nodes=%llu,max_depth=%d,draw_depth=%d certify=%s\n",
              static_cast<unsigned long long>(o.playouts), static_cast<unsigned long long>(o.mate_nodes), legality.c_str(),
              static_cast<unsigned long long>(o.dfpn.nodes), o.dfpn.max_depth, o.dfpn.draw_depth,
              o.dfpn.certify ? "E4" : "off");
  for (const std::string& note : roots.notes) std::printf("RECORD %s\n", note.c_str());

  std::array<Range, 4> ranges{};
  std::size_t skipped = 0;
  t5d::tools::DfpnChecker checker(o.dfpn);
  for (const t5d::tools::SixbRoot& root : roots.roots) {
    const t5d::GameState& s = root.state;
    const int boards = static_cast<int>(s.multiverse().board_count());
    std::string skip;
    if (boards < o.min_boards || boards > o.max_boards) {
      skip = "outside --min-boards/--max-boards";
    } else if (t5d::royal_capture_exists(s.multiverse(), s.rules(), s.to_move(), s.inactive_timelines())) {
      skip = "a royal capturable by the side to move (the last action broke C4)";
    } else {
      t5d::SearchBudget b = t5d::SearchBudget::nodes(o.mate_nodes);
      b.max_bytes = std::size_t{256} << 20;
      const t5d::MateResult m = t5d::detect_mate(s, t5d::Detector::Hypercuboid, b);
      if (m.verdict != t5d::Outcome::Ongoing) {
        skip = m.verdict == t5d::Outcome::Checkmate   ? "terminal (checkmate)"
               : m.verdict == t5d::Outcome::Stalemate ? "terminal (stalemate)"
                                                      : "E4 undecided within --mate-nodes";
      }
    }
    if (!skip.empty()) {
      ++skipped;
      if (o.per_position) std::printf("ROOT %s #%d boards=%d skipped: %s\n", root.record.c_str(), root.index, boards, skip.c_str());
      continue;
    }
    Range& r = ranges[static_cast<std::size_t>(range_of(boards))];
    ++r.roots;
    for (const TurnSplit split : o.splits) {
      const auto t0 = std::chrono::steady_clock::now();
      t5d::mcts::SearchParams p;
      p.split = split;
      p.proof_channel = true;  // offline, the tool's own search: no store (H23, H41)
      p.legality = o.legality_nodes == 0 ? t5d::LegalityOptions::exact() : t5d::LegalityOptions::nodes(o.legality_nodes);
      p.verify_legality = p.legality;
      p.mate_nodes = o.mate_nodes;
      p.expand.square_features = t5d::SquareFeatures::None;  // the uniform evaluator reads none
      t5d::tools::DfpnTally tally;
      bool solved = false;
      std::string error;
      try {
        const std::unique_ptr<t5d::mcts::Evaluator> ev = t5d::mcts::make_evaluator(t5d::mcts::BuiltinEvaluator::Uniform);
        t5d::mcts::Search search(s, p);
        search.run(*ev, t5d::mcts::Limits{o.playouts});
        search.check_invariants();
        solved = search.root_solved();
        checker.check_search(search, tally);
      } catch (const std::exception& e) {
        error = e.what();
      }
      ++r.searches;
      if (solved) ++r.roots_solved;
      if (!error.empty()) tally.contradictions.push_back("the search or its claims failed: " + error);
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      r.seconds += sec;
      if (o.per_position) {
        std::printf("ROOT %s #%d boards=%d split=%s solved=%s %.2fs: %s\n", root.record.c_str(), root.index, boards,
                    split_name(split), solved ? "yes" : "no", sec, tally.summary().c_str());
      }
      for (const std::string& c : tally.contradictions) {
        std::printf("CONTRADICTION %s #%d split=%s: %s\n", root.record.c_str(), root.index, split_name(split), c.c_str());
      }
      for (const std::string& c : tally.certificate_failures) {
        std::printf("CERTIFICATE %s #%d split=%s: %s\n", root.record.c_str(), root.index, split_name(split), c.c_str());
      }
      r.tally.add(tally);
    }
  }
  std::size_t contradictions = 0;
  std::size_t failures = 0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const Range& r = ranges[i];
    contradictions += r.tally.contradictions.size();
    failures += r.tally.certificate_failures.size();
    std::printf("RANGE %s: %zu roots, %zu searches, %zu roots solved, %.1f s; %s\n", kRanges[i], r.roots, r.searches,
                r.roots_solved, r.seconds, r.tally.summary().c_str());
  }
  std::printf("t5d-solver-check: %zu roots searched, %zu skipped; %zu contradictions, %zu certificate failures - %s\n",
              roots.roots.size() - skipped, skipped, contradictions, failures,
              contradictions + failures == 0 ? "the solver agrees with df-pn wherever df-pn decides"
                                             : "FINDINGS (above)");
  return contradictions + failures == 0 ? 0 : 3;
}
