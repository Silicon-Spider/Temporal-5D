// The test runner: t5d-tests [--list] [--long] [name-fragment ...]
//
// Runs every registered test, or only those whose names contain one of the
// fragments. Tests whose names begin with "long_" (long cross-checks) run only
// with --long or when a fragment names them. Prints one line per test and a
// summary; exits 1 if any test failed (or if the fragments matched nothing), 0
// otherwise.

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "test_framework.hpp"

namespace t5d_test {

std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

namespace {
int g_failures = 0;  // in the running test
}

void record_failure(const char* file, int line, const std::string& message) {
  ++g_failures;
  std::fprintf(stderr, "%s:%d: %s\n", file, line, message.c_str());
  std::fflush(stderr);
}

}  // namespace t5d_test

int main(int argc, char** argv) {
  using namespace t5d_test;
  bool list = false;
  bool long_tests = false;
  std::vector<std::string> filters;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--list") == 0) {
      list = true;
    } else if (std::strcmp(argv[i], "--long") == 0) {
      long_tests = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::puts("usage: t5d-tests [--list] [--long] [name-fragment ...]");
      return 0;
    } else {
      filters.emplace_back(argv[i]);
    }
  }
  const auto selected = [&](const TestCase& test) {
    const std::string name(test.name);
    const bool is_long = name.rfind("long_", 0) == 0;
    if (filters.empty()) return !is_long || long_tests;
    for (const std::string& f : filters) {
      if (name.find(f) == std::string::npos) continue;
      // A fragment selects a long test only if it names it as long.
      if (!is_long || long_tests || f.rfind("long_", 0) == 0) return true;
    }
    return false;
  };

  int run = 0;
  int failed = 0;
  for (const TestCase& test : registry()) {
    if (!selected(test)) continue;
    if (list) {
      std::printf("%s\n", test.name);
      continue;
    }
    ++run;
    g_failures = 0;
    try {
      test.function();
    } catch (const RequireFailed&) {
      // already recorded
    } catch (const std::exception& e) {
      record_failure(test.file, test.line, std::string("uncaught exception: ") + e.what());
    } catch (...) {
      record_failure(test.file, test.line, "uncaught non-standard exception");
    }
    if (g_failures > 0) {
      ++failed;
      std::printf("[FAIL] %s (%d failure%s)\n", test.name, g_failures, g_failures == 1 ? "" : "s");
    } else {
      std::printf("[ ok ] %s\n", test.name);
    }
    std::fflush(stdout);
  }
  if (list) return 0;
  if (run == 0) {
    std::printf("t5d-tests: no test matched\n");
    return 1;
  }
  std::printf("t5d-tests: %d passed, %d failed, %d run\n", run - failed, failed, run);
  return failed == 0 ? 0 : 1;
}
