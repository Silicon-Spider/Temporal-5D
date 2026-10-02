// t5d-gate: the rules engine's adapter for the differential-testing gate
// (design spec §11.0, "Running the gate"; decisions H23, H30, H36, H37). It
// reads one 5dpgn record on standard input and prints, one JSON object per
// line, what the engine set up at every position and what it says of it - the
// protocol of t5d/gate.hpp, described in the engine README. It answers from
// the engine alone: it never runs, loads or reads the harness or an oracle.
//
//   t5d-gate [--cap N]                full mode: states and every verdict
//   t5d-gate --parse-only [--cap N]   replay pseudo-legally; states, no verdicts
//   t5d-gate --help | --version
//
// --cap N (the gate appends it) caps the count of legal actions. Every other
// option is a DIAGNOSTIC, for investigating a record by hand - never for the
// gate's configuration (gate.toml): see print_usage.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "t5d/gate.hpp"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif

namespace {

void print_usage(std::FILE* stream) {
  std::fputs(
      "t5d-gate " T5D_VERSION_STRING
      " - the 5D chess rules engine's adapter for the differential-testing gate.\n"
      "\n"
      "Reads one 5dpgn record on standard input; prints one JSON object per line on\n"
      "standard output (see the engine README, \"The gate adapter\").\n"
      "\n"
      "usage: t5d-gate [--cap N]                full mode: states, check, legal actions, terminal\n"
      "       t5d-gate --parse-only [--cap N]   replay pseudo-legally: states, no verdicts\n"
      "       t5d-gate --help | --version\n"
      "\n"
      "Diagnostics - for investigating a record by hand, never for gate.toml:\n"
      "  --moves-after-shift          rules Q4's other reading (moves after the present shifts)\n"
      "  --rule NAME=VALUE            any other rule option (RuleOptions, t5d/rules.hpp)\n"
      "  --lenient-annotations        do not check ~, (>Lx), (~Tx) against the state\n"
      "  --pgn NAME=VALUE             any other reader option (PgnOptions, t5d/pgn.hpp)\n"
      "  --actions K                  list position K's legal actions on standard error\n"
      "  --detector NAME              naive, hypercuboid or portfolio (the default): which\n"
      "                               detector answers (the other cross-checks it)\n"
      "  --cross-check-nodes N        the cross-check's node budget per position (default 50000)\n"
      "  --no-cross-check             no cross-check\n"
      "  --max-memory-mib N           every search's memory ceiling (default 2048); a count\n"
      "                               that reaches it ends the run \"error\" (no verdict, H37)\n",
      stream);
}

bool read_count(const char* text, std::uint64_t& out) {
  if (*text == '\0') return false;
  std::uint64_t v = 0;
  for (const char* c = text; *c != '\0'; ++c) {
    if (*c < '0' || *c > '9') return false;
    const auto digit = static_cast<std::uint64_t>(*c - '0');
    if (v > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) return false;
    v = v * 10 + digit;
  }
  out = v;
  return true;
}

int bad_argument(const std::string& why) {
  std::fprintf(stderr, "t5d-gate: %s\n\n", why.c_str());
  print_usage(stderr);
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  t5d::gate::Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    std::uint64_t n = 0;
    std::string error;
    if (arg == "--help" || arg == "-h") {
      print_usage(stdout);
      return 0;
    }
    if (arg == "--version") {
      std::puts(T5D_VERSION_STRING);
      return 0;
    }
    if (arg == "--parse-only") {
      options.parse_only = true;
    } else if (arg == "--cap" && has_value && read_count(argv[i + 1], n)) {
      options.cap = static_cast<std::size_t>(n);
      ++i;
    } else if (arg == "--moves-after-shift") {
      options.rules.moves_after_shift = true;
    } else if (arg == "--lenient-annotations") {
      options.pgn.verify_annotations = false;
    } else if (arg == "--rule" && has_value) {
      if (!t5d::gate::set_rule_option(options.rules, argv[++i], &error)) {
        return bad_argument("--rule: " + error);
      }
    } else if (arg == "--pgn" && has_value) {
      if (!t5d::gate::set_pgn_option(options.pgn, argv[++i], &error)) {
        return bad_argument("--pgn: " + error);
      }
    } else if (arg == "--actions" && has_value && read_count(argv[i + 1], n)) {
      options.list_actions = static_cast<std::size_t>(n);
      ++i;
    } else if (arg == "--detector" && has_value) {
      const std::string which = argv[++i];
      if (which == "naive") {
        options.detector = t5d::Detector::Naive;
      } else if (which == "hypercuboid") {
        options.detector = t5d::Detector::Hypercuboid;
      } else if (which == "portfolio") {
        options.detector.reset();
      } else {
        return bad_argument("--detector takes naive, hypercuboid or portfolio");
      }
    } else if (arg == "--cross-check-nodes" && has_value && read_count(argv[i + 1], n)) {
      options.cross_check_nodes = n;
      ++i;
    } else if (arg == "--no-cross-check") {
      options.cross_check_nodes = 0;
    } else if (arg == "--max-memory-mib" && has_value && read_count(argv[i + 1], n) && n > 0 &&
               n <= (std::numeric_limits<std::size_t>::max() >> 20)) {
      options.max_bytes = static_cast<std::size_t>(n) << 20;
      ++i;
    } else {
      return bad_argument("unknown or incomplete argument \"" + arg + "\"");
    }
  }

#if defined(_WIN32)
  // Bytes in, bytes out: no CR-LF translation of the record or of the JSON lines.
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  // The whole record first: the harness writes it and closes the pipe.
  std::string record;
  char buffer[1 << 16];
  for (;;) {
    const std::size_t got = std::fread(buffer, 1, sizeof buffer, stdin);
    record.append(buffer, got);
    if (got < sizeof buffer) break;
  }
  if (std::ferror(stdin)) {
    std::fputs("t5d-gate: cannot read standard input\n", stderr);
    return 2;
  }

  const auto emit = [](const std::string& line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  };
  const auto diagnose = [](const std::string& line) {
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
  };
  return t5d::gate::run(record, options, emit, diagnose);
}
