// t5d-gate: the rules engine's adapter for the differential-testing gate
// (design spec §11.0, "Running the gate"; decisions H23, H30). It reads one
// 5dpgn record on standard input and prints, one JSON object per line, what
// the engine set up at every position and what it says of it - the protocol of
// t5d/gate.hpp, described in the engine README. It answers from the engine
// alone: it never runs, loads or reads the harness or an oracle.
//
//   t5d-gate --parse-only [--cap N]   replay the record; states, no verdicts
//   t5d-gate [--cap N]                full mode: "unsupported" until the check ruling
//   t5d-gate --help | --version
//
// --cap N (the gate appends it: the cap on counted legal actions) is accepted
// and ignored: parse mode counts nothing.

#include <cstdio>
#include <cstring>
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
      "usage: t5d-gate --parse-only [--cap N]   replay the record: states, no verdicts\n"
      "       t5d-gate [--cap N]                full mode: unsupported until the check ruling\n"
      "       t5d-gate --help | --version\n",
      stream);
}

bool is_count(const char* text) {
  if (*text == '\0') return false;
  for (const char* c = text; *c != '\0'; ++c) {
    if (*c < '0' || *c > '9') return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  t5d::gate::Options options;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      print_usage(stdout);
      return 0;
    }
    if (std::strcmp(arg, "--version") == 0) {
      std::puts(T5D_VERSION_STRING);
      return 0;
    }
    if (std::strcmp(arg, "--parse-only") == 0) {
      options.parse_only = true;
      continue;
    }
    if (std::strcmp(arg, "--cap") == 0 && i + 1 < argc && is_count(argv[i + 1])) {
      ++i;  // accepted and ignored
      continue;
    }
    std::fprintf(stderr, "t5d-gate: unknown or incomplete argument \"%s\"\n\n", arg);
    print_usage(stderr);
    return 2;
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
    const std::size_t n = std::fread(buffer, 1, sizeof buffer, stdin);
    record.append(buffer, n);
    if (n < sizeof buffer) break;
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
