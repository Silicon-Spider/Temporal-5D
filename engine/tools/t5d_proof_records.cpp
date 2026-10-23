// t5d-proof-records: the search solver's proof lines as 5dpgn records - the
// engine side of register H41's cross-check against the oracles' mate
// verdicts (design spec §11.1: "against the oracles' mate verdicts where they
// exist"; §15.4). The verdicts reach the solver only through the gate: engine
// -> these files -> the harness, never back (AGENTS.md).
//
//   t5d-proof-records --out DIR [options] RECORD ...
//
// THE ROOTS: the turn starts of the records given - the engine's own: a
// constructed position as a record of no action (a custom 5DFEN set-up), or
// a game the engine played (t5d-play, from a named variant's start). Never a
// third party's record text: the records come on the command line, and the
// tool reads nothing else.
//
// AT EACH ROOT the search runs offline with its proof channel ON and no store
// (the tool's own search: nothing is exported; H23, H41), exact legality, the
// uniform evaluator. Where it proves the root won, the proof tree is read
// (tools/solver_proofs.hpp, proof_tree): the attacker's winning action at
// each of its turns, every legal defence at each of the defender's - the
// defences checked against E4's exhaustive listing (the same successors) -
// and every line replayed to E4's checkmate.
//
// THE RECORDS: one per root-to-mate line, written by play_record from the
// input record's own start - its custom 5DFEN set-up, or its named variant -
// through the actions before the root, then the line (5DFEN records no
// creators or origins, so a mid-game root is never set up from its boards).
// Headers: ProofLine (what it is), ProofRoot (the actions before the root),
// ProofClaim ("win k"), ProofSplit, ProofDefences ("i:n ..." - after i
// actions from the record's start the defender is to move with n distinct
// defences in the proof tree, each E4's), ProofRules (non-default rules),
// ProofFrom (the input record's content id, never its path). Then FILES.tsv:
// each record's SHA-256, size and name.
//
// The records go to --out, never into the gate's corpus: adding them there,
// with the corpus manifest's fields, and re-recording the golden files is the
// harness side - a separate reviewed change (build-log issue 405).
//
// Exit status: 0 every proof tree read and checked; 3 an error (a defence set
// that is not E4's, a line that does not replay to E4's checkmate; printed,
// and that root's records not written); 1 a record could not be read or the
// output not written; 2 a bad command line.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "play_record.hpp"
#include "sha256.hpp"
#include "sixb_roots.hpp"
#include "solver_proofs.hpp"
#include "t5d/check.hpp"
#include "t5d/mcts.hpp"
#include "t5d/pgn.hpp"
#include "t5d/search.hpp"
#include "t5d/selfplay.hpp"
#include "tool_support.hpp"

namespace {

namespace fs = std::filesystem;
using t5d::mcts::TurnSplit;

struct Options {
  std::vector<std::string> records;
  std::string out;
  t5d::RuleOptions rules;
  TurnSplit split = TurnSplit::BoardMove;
  std::uint64_t playouts = 3000;
  std::uint64_t mate_nodes = 2'000'000;
  /// 0: exact move legality; else exact within this many nodes per
  /// classification (LegalityOptions::nodes; build-log issue 406).
  std::uint64_t legality_nodes = 0;
  int every = 1;
  std::size_t max_lines = 64;
  std::size_t max_records = 10000;
};

[[noreturn]] void usage(int status) {
  std::FILE* out = status == 0 ? stdout : stderr;
  std::fputs(
      "t5d-proof-records " T5D_VERSION_STRING
      " - the search solver's proof lines as 5dpgn records, one per root-to-mate line (register H41,\n"
      "design spec 11.1). Offline: the proof channel is on in the tool's own searches; nothing is exported.\n"
      "\n"
      "usage: t5d-proof-records --out DIR [options] RECORD ...\n"
      "  --out DIR                  where the records and FILES.tsv go (created; never the gate's corpus)\n"
      "  --rules ENCODING           the ruleset (RuleOptions::encode; default 0, the engine's)\n"
      "  --no-moves-after-shift     a diagnostic: Q4's Guide-literal reading\n"
      "  --every K (1)              every K-th turn start of each record is a root\n"
      "  --split ply|tree|flat      the D2 arm (ply)\n"
      "  --playouts N (3000)        each search's playouts\n"
      "  --mate-nodes N (2000000)   E4's budget at each turn start the search meets\n"
      "  --legality-nodes N (0)     move legality exact within N nodes per classification (0: no\n"
      "                             budget); a move left unknown proves nothing\n"
      "  --max-lines N (64)         lines per root at most\n"
      "  --max-records N (10000)    records in all at most\n"
      "  --help  --version\n",
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
      std::puts("t5d-proof-records " T5D_VERSION_STRING);
      std::exit(0);
    } else if (a == "--out") {
      o.out = value();
    } else if (a == "--every") {
      o.every = static_cast<int>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--split") {
      const std::string v = value();
      if (v == "ply") {
        o.split = TurnSplit::BoardMove;
      } else if (v == "tree") {
        o.split = TurnSplit::Tree;
      } else if (v == "flat") {
        o.split = TurnSplit::Flat;
      } else {
        throw std::invalid_argument("--split takes ply, tree or flat, not " + v);
      }
    } else if (a == "--playouts") {
      o.playouts = std::max<std::uint64_t>(1, number(value()));
    } else if (a == "--mate-nodes") {
      o.mate_nodes = number(value());
    } else if (a == "--legality-nodes") {
      o.legality_nodes = number(value());
    } else if (a == "--max-lines") {
      o.max_lines = static_cast<std::size_t>(std::max<std::uint64_t>(1, number(value())));
    } else if (a == "--max-records") {
      o.max_records = static_cast<std::size_t>(number(value()));
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("unknown option " + a);
    } else {
      o.records.push_back(a);
    }
  }
  if (o.out.empty()) throw std::invalid_argument("--out is required");
  if (o.records.empty()) throw std::invalid_argument("no records given");
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

// A record read and replayed: its start, its game's name, every action's
// moves, and the state at each turn start.
struct Game {
  std::string board;  // the variant's name, or "custom"
  std::vector<std::vector<t5d::Move>> actions;
  std::vector<t5d::GameState> states;  // states[i]: after i actions
};

Game read_game(const std::string& text, const t5d::RuleOptions& rules) {
  t5d::PgnReplay replay(t5d::read_pgn(text), {}, rules);
  Game g;
  g.board = replay.game();
  g.states.push_back(replay.state());
  while (!replay.finished()) {
    const t5d::PgnAction& written = replay.record().actions[replay.actions_played()];
    t5d::GameState t = replay.state();
    std::vector<t5d::Move> moves;
    for (const t5d::PgnMove& pm : written.moves) {
      const t5d::Move m = replay.resolve(t, pm);
      t.apply(m);
      moves.push_back(m);
    }
    replay.play_next();
    g.actions.push_back(std::move(moves));
    g.states.push_back(replay.state());
  }
  if (replay.record().unreadable) throw std::runtime_error("the move text stops early: " + *replay.record().unreadable);
  return g;
}

std::string defences_text(const t5d::tools::ProofLine& line, std::size_t offset) {
  std::string out;
  for (const auto& [index, n] : line.defences) {
    if (!out.empty()) out += ' ';
    out += std::to_string(offset + index) + ":" + std::to_string(n);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = parse(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-proof-records: %s\n", e.what());
    usage(2);
  }
  try {
    fs::create_directories(o.out);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-proof-records: cannot make %s: %s\n", o.out.c_str(), e.what());
    return 1;
  }
  const std::string legality =
      o.legality_nodes == 0 ? std::string("exact") : "exact within " + std::to_string(o.legality_nodes) + " nodes";
  std::printf("CONDITIONS rules=%s split=%s playouts=%llu mate_nodes=%llu evaluator=uniform legality=%s "
              "proof_channel=on(offline, no store) every=%d max_lines=%zu\n",
              t5d::tools::ruleset_text(o.rules).c_str(), split_name(o.split),
              static_cast<unsigned long long>(o.playouts), static_cast<unsigned long long>(o.mate_nodes),
              legality.c_str(), o.every, o.max_lines);
  struct Written {
    std::string name;
    std::string sha;
    std::size_t size;
  };
  std::vector<Written> written;
  std::size_t roots = 0;
  std::size_t proven = 0;
  std::size_t errors = 0;
  std::size_t truncated = 0;
  for (std::size_t r = 0; r < o.records.size(); ++r) {
    Game g;
    std::string content;
    try {
      const std::string text = t5d::tools::read_file(o.records[r]);
      content = t5d::tools::record_content_id(text);
      g = read_game(text, o.rules);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "t5d-proof-records: %s: %s\n", o.records[r].c_str(), e.what());
      return 1;
    }
    for (std::size_t i = 0; i < g.states.size(); i += static_cast<std::size_t>(o.every)) {
      const t5d::GameState& s = g.states[i];
      if (t5d::royal_capture_exists(s.multiverse(), s.rules(), s.to_move(), s.inactive_timelines())) continue;
      t5d::SearchBudget b = t5d::SearchBudget::nodes(o.mate_nodes);
      if (t5d::detect_mate(s, t5d::Detector::Hypercuboid, b).verdict != t5d::Outcome::Ongoing) continue;
      ++roots;
      t5d::mcts::SearchParams p;
      p.split = o.split;
      p.proof_channel = true;  // offline, the tool's own search: no store (H23, H41)
      p.legality = o.legality_nodes == 0 ? t5d::LegalityOptions::exact() : t5d::LegalityOptions::nodes(o.legality_nodes);
      p.verify_legality = p.legality;
      p.mate_nodes = o.mate_nodes;
      p.expand.square_features = t5d::SquareFeatures::None;  // the uniform evaluator reads none
      const std::unique_ptr<t5d::mcts::Evaluator> ev = t5d::mcts::make_evaluator(t5d::mcts::BuiltinEvaluator::Uniform);
      t5d::mcts::Search search(s, p);
      search.run(*ev, t5d::mcts::Limits{o.playouts});
      search.check_invariants();
      if (!search.root_solved()) continue;
      const t5d::tools::SolverClaim claim = t5d::tools::solver_claim(search, search.root());
      if (claim.kind != t5d::tools::SolverClaim::Kind::Win) continue;
      ++proven;
      const t5d::tools::ProofTree tree = t5d::tools::proof_tree(search, search.root(), o.max_lines);
      std::printf("ROOT %zu #%zu: %s, %zu lines%s, %zu defender turns (%zu defences)\n", r, i,
                  t5d::tools::to_string(claim).c_str(), tree.lines.size(), tree.truncated ? " (truncated)" : "",
                  tree.defender_turns, tree.defences);
      if (tree.truncated) ++truncated;
      if (!tree.errors.empty()) {
        errors += tree.errors.size();
        for (const std::string& e : tree.errors) std::printf("ERROR %zu #%zu: %s\n", r, i, e.c_str());
        continue;
      }
      for (std::size_t l = 0; l < tree.lines.size() && written.size() < o.max_records; ++l) {
        const t5d::tools::ProofLine& line = tree.lines[l];
        t5d::mcts::GameRecord rec(g.states.front());
        rec.actions.assign(g.actions.begin(), g.actions.begin() + static_cast<std::ptrdiff_t>(i));
        t5d::GameState t = s;
        for (const std::vector<t5d::Move>& a : line.actions) {
          for (const t5d::Move& m : a) t.apply(m);
          t.submit();
          rec.actions.push_back(a);
        }
        rec.final_state = t;
        rec.end = t5d::mcts::GameEnd::Checkmate;
        rec.winner = s.to_move() == t5d::Colour::White ? 0 : 1;
        t5d::tools::Headers headers{{"ProofLine", "H41 cross-check: the search solver's proof line"},
                                    {"ProofRoot", std::to_string(i)},
                                    {"ProofClaim", t5d::tools::to_string(claim)},
                                    {"ProofSplit", split_name(o.split)},
                                    {"ProofDefences", defences_text(line, i)},
                                    {"ProofFrom", content}};
        if (t5d::tools::ruleset_text(o.rules) != "standard") {
          headers.emplace_back("ProofRules", t5d::tools::ruleset_text(o.rules));
        }
        const std::string text = t5d::tools::play_record(rec, g.board, headers);
        const std::string name = "proof-" + std::to_string(r) + "-" + std::to_string(i) + "-" + std::to_string(l) + ".5dpgn";
        std::ofstream out(fs::path(o.out) / name, std::ios::binary);
        out << text;
        if (!out) {
          std::fprintf(stderr, "t5d-proof-records: cannot write %s\n", name.c_str());
          return 1;
        }
        written.push_back(Written{name, t5d::tools::sha256_hex(text), text.size()});
      }
    }
  }
  std::ofstream list(fs::path(o.out) / "FILES.tsv", std::ios::binary);
  list << "# sha256\tbytes\tfile - t5d-proof-records " T5D_VERSION_STRING "\n";
  for (const Written& w : written) list << w.sha << '\t' << w.size << '\t' << w.name << '\n';
  if (!list) {
    std::fprintf(stderr, "t5d-proof-records: cannot write FILES.tsv\n");
    return 1;
  }
  std::printf("t5d-proof-records: %zu roots searched, %zu proven won, %zu records written%s; %zu errors - %s\n", roots,
              proven, written.size(), truncated ? " (some proof trees truncated)" : "", errors,
              errors == 0 ? "every proof tree read and checked against E4" : "FINDINGS (above)");
  return errors == 0 ? 0 : 3;
}
