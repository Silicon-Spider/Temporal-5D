// t5d-audit: the offline audit lane (engine stage E10; design spec 11.4;
// register H44, H42) - tools/audit.hpp holds the logic.
//
//   t5d-audit sample [--seed S] [--rates F,P,D] [--cap N] [--demotion-share X]
//                    [--out FILE] LOG...
//   t5d-audit run [--guard-depth D] [--guard-nodes N] [--mate-depth D]
//                 [--mate-nodes N] [--tree-cap N] [--threads T]
//                 [--contradictions FILE] [--out FILE] JOBS
//   t5d-audit report [--shallow D] [--json FILE] RESULTS...
//
// sample: the decision logs ("t5d-decisions" v1, written by t5d-play
// --decision-log or Python's netplay --decision-log) in, a seeded job list
// out (the games it needs, then the jobs) - per category (final, pv_change,
// demotion) a rate, per (category, board range) a cap: placeholders.
// run: each job's exact questions with large budgets (placeholders) - Guard's
// offline on its action where complete, our mate at its turn start - results
// out ("t5d-audit-results" v1), contradictions to --contradictions
// ("t5d-contradictions" v1: the curriculum's hook, an interface only). Nothing
// is exported to any solved store (H19, H23).
// report: per board range, the shallow-mate rate (11.4: "how often the
// network's chosen move allows a shallow forced mate"), Guard's completed
// depths, the undecided and the contradictions - text on standard output,
// --json FILE beside it.
// Output goes to standard output unless --out. Exit status: 0, 2 for a bad
// command line, 1 for an error. The tool reads self-play logs only - never a
// path under oracles/ (AGENTS.md).

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "audit.hpp"
#include "tool_support.hpp"

namespace {

using namespace t5d;

void usage(std::FILE* to) {
  std::fputs(
      "usage: t5d-audit sample [--seed S] [--rates F,P,D] [--cap N] [--out FILE] LOG...\n"
      "       t5d-audit run [--guard-depth D] [--guard-nodes N] [--mate-depth D] [--mate-nodes N]\n"
      "                     [--tree-cap N] [--threads T] [--contradictions FILE] [--out FILE] JOBS\n"
      "       t5d-audit report [--shallow D] [--json FILE] RESULTS...\n",
      to);
}

std::uint64_t number(const std::string& option, const char* text) {
  const std::optional<std::uint64_t> n = tools::parse_number(text);
  if (!n) throw std::invalid_argument(option + ": not a number: " + text);
  return *n;
}

std::uint64_t number_in(const std::string& option, const char* text, std::uint64_t low, std::uint64_t high) {
  const std::uint64_t n = number(option, text);
  if (n < low || n > high) throw std::invalid_argument(option + ": " + text + " is out of range");
  return n;
}

double share(const std::string& option, std::string_view text) {
  const std::string t(text);
  char* end = nullptr;
  const double v = std::strtod(t.c_str(), &end);
  if (end == t.c_str() || *end != '\0' || !(v >= 0 && v <= 1)) {
    throw std::invalid_argument(option + ": a number from 0 to 1, not " + t);
  }
  return v;
}

void write(const std::string& path, const std::string& text) {
  if (path.empty()) {
    std::fputs(text.c_str(), stdout);
    return;
  }
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << text;
}

int sample_main(const std::vector<std::string>& args) {
  audit::SampleOptions o;
  std::string out;
  std::vector<std::string> inputs;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
      return args[++i].c_str();
    };
    if (a == "--seed") {
      o.seed = number(a, value());
    } else if (a == "--rates") {
      const std::string v = value();
      std::size_t at = 0;
      for (int k = 0; k < audit::kCategories; ++k) {
        const std::size_t comma = k + 1 < audit::kCategories ? v.find(',', at) : v.size();
        if (comma == std::string::npos) throw std::invalid_argument("--rates: three shares, F,P,D");
        o.rate[static_cast<std::size_t>(k)] = share(a, std::string_view(v).substr(at, comma - at));
        at = comma + 1;
      }
    } else if (a == "--cap") {
      o.cap = static_cast<std::size_t>(number_in(a, value(), 1, std::uint64_t{1} << 40));
    } else if (a == "--out") {
      out = value();
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("unknown option " + a);
    } else {
      inputs.push_back(a);
    }
  }
  if (inputs.empty()) throw std::invalid_argument("sample: no decision log given");
  audit::Log all;
  for (const std::string& path : inputs) {
    const audit::Log log = audit::read_log(tools::read_file(path));
    for (const audit::GameLog& g : log.games) {
      if (all.game(g.record) == nullptr) all.games.push_back(g);
    }
    all.events.insert(all.events.end(), log.events.begin(), log.events.end());
  }
  const audit::Log jobs = audit::sample(all, o);
  write(out, audit::log_text(jobs));
  std::fprintf(stderr, "t5d-audit: %zu events, %zu games; %zu jobs sampled (seed %llu)\n", all.events.size(),
               all.games.size(), jobs.jobs.size(), static_cast<unsigned long long>(o.seed));
  return 0;
}

int run_main(const std::vector<std::string>& args) {
  audit::RunOptions o;
  std::string out;
  std::string contradictions;
  std::string input;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
      return args[++i].c_str();
    };
    if (a == "--guard-depth") {
      o.guard_max_depth = static_cast<int>(number_in(a, value(), 1, 50));
    } else if (a == "--guard-nodes") {
      o.guard_nodes = number_in(a, value(), 1, ~std::uint64_t{0});
    } else if (a == "--mate-depth") {
      o.mate_depth = static_cast<int>(number_in(a, value(), 1, 99));
    } else if (a == "--mate-nodes") {
      o.mate_nodes = number_in(a, value(), 1, ~std::uint64_t{0});
    } else if (a == "--tree-cap") {
      o.tree_cap = static_cast<std::size_t>(number_in(a, value(), 1024, std::uint64_t{1} << 40));
    } else if (a == "--threads") {
      o.threads = static_cast<int>(number_in(a, value(), 1, 256));
    } else if (a == "--contradictions") {
      contradictions = value();
    } else if (a == "--out") {
      out = value();
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("unknown option " + a);
    } else if (input.empty()) {
      input = a;
    } else {
      throw std::invalid_argument("run: one job list");
    }
  }
  if (input.empty()) throw std::invalid_argument("run: no job list given");
  const audit::Log jobs = audit::read_log(tools::read_file(input));
  std::ofstream sink_file;
  std::optional<audit::JsonlCurriculumSink> sink;
  if (!contradictions.empty()) {
    sink_file.open(contradictions, std::ios::binary);
    if (!sink_file) throw std::runtime_error("cannot write " + contradictions);
    sink.emplace(sink_file);
  }
  const std::vector<audit::JobResult> results = audit::run(jobs, o, sink ? &*sink : nullptr);
  std::string text;
  for (const audit::JobResult& r : results) text += audit::result_json(r) + "\n";
  write(out, text);
  std::fprintf(stderr, "t5d-audit: %zu jobs run, %zu contradictions\n", results.size(), sink ? sink->emitted() : 0);
  return 0;
}

int report_main(const std::vector<std::string>& args) {
  int shallow = 1;
  std::string json;
  std::vector<std::string> inputs;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
      return args[++i].c_str();
    };
    if (a == "--shallow") {
      shallow = static_cast<int>(number_in(a, value(), 1, 50));
    } else if (a == "--json") {
      json = value();
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("unknown option " + a);
    } else {
      inputs.push_back(a);
    }
  }
  if (inputs.empty()) throw std::invalid_argument("report: no results given");
  std::vector<audit::JobResult> results;
  for (const std::string& path : inputs) {
    const std::vector<audit::JobResult> r = audit::read_results(tools::read_file(path));
    results.insert(results.end(), r.begin(), r.end());
  }
  const audit::Report rep = audit::make_report(results, shallow);
  std::fputs(audit::report_text(rep).c_str(), stdout);
  if (!json.empty()) write(json, audit::report_json(rep) + "\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
    usage(argc < 2 ? stderr : stdout);
    return argc < 2 ? 2 : 0;
  }
  const std::string command = argv[1];
  std::vector<std::string> args(argv + 2, argv + argc);
  int (*body)(const std::vector<std::string>&) = nullptr;
  if (command == "sample") {
    body = sample_main;
  } else if (command == "run") {
    body = run_main;
  } else if (command == "report") {
    body = report_main;
  } else {
    std::fprintf(stderr, "t5d-audit: unknown command %s\n", command.c_str());
    usage(stderr);
    return 2;
  }
  try {
    return body(args);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "t5d-audit: %s\n", e.what());
    usage(stderr);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-audit: error: %s\n", e.what());
    return 1;
  }
}
