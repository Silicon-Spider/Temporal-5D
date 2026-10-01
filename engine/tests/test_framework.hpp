// A small in-house test framework (no third-party test library).
//
//   TEST(name) { ... }           registers a test (name: an identifier)
//   CHECK(cond)                   records a failure and carries on
//   CHECK_EQ(a, b), CHECK_NE      the same, printing both values
//   REQUIRE(cond), REQUIRE_EQ     records a failure and ends the test
//   CHECK_THROWS_AS(expr, type)   expr must throw `type` (or a subclass)
//   CHECK_THROWS_WITH(expr, type, text)  ... with `text` in what()
//   CHECK_NOTHROW(expr)
//
// Every failure prints "file:line: message". The runner (test_main.cpp) runs
// every test - or those whose names contain one of its arguments - and exits
// non-zero if any failed.

#ifndef T5D_TEST_FRAMEWORK_HPP
#define T5D_TEST_FRAMEWORK_HPP

#include <exception>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace t5d_test {

using TestFunction = void (*)();

struct TestCase {
  const char* name;
  TestFunction function;
  const char* file;
  int line;
};

std::vector<TestCase>& registry();

struct Registrar {
  Registrar(const char* name, TestFunction function, const char* file, int line) {
    registry().push_back(TestCase{name, function, file, line});
  }
};

/// Thrown by REQUIRE to end the current test.
struct RequireFailed {};

void record_failure(const char* file, int line, const std::string& message);

template <class T>
std::string show(const T& value) {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<U, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_same_v<U, char>) {
    return std::string("'") + value + "'";
  } else if constexpr (std::is_same_v<U, signed char> || std::is_same_v<U, unsigned char>) {
    return std::to_string(static_cast<int>(value));
  } else if constexpr (std::is_convertible_v<const U&, std::string_view>) {
    return "\"" + std::string(std::string_view(value)) + "\"";
  } else if constexpr (std::is_enum_v<U>) {
    return std::to_string(static_cast<long long>(value));
  } else if constexpr (requires(std::ostream& os, const U& v) { os << v; }) {
    std::ostringstream os;
    os << value;
    return os.str();
  } else {
    return "<value>";
  }
}

template <class E>
bool what_contains(const E& e, std::string_view text) {
  return std::string_view(e.what()).find(text) != std::string_view::npos;
}

}  // namespace t5d_test

#define T5D_TEST_CAT2(a, b) a##b
#define T5D_TEST_CAT(a, b) T5D_TEST_CAT2(a, b)

#define TEST(name)                                                                       \
  static void T5D_TEST_CAT(t5d_test_body_, name)();                                     \
  static const ::t5d_test::Registrar T5D_TEST_CAT(t5d_test_registrar_, name){            \
      #name, &T5D_TEST_CAT(t5d_test_body_, name), __FILE__, __LINE__};                   \
  static void T5D_TEST_CAT(t5d_test_body_, name)()

#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) ::t5d_test::record_failure(__FILE__, __LINE__, "CHECK(" #cond ") failed"); \
  } while (false)

#define T5D_TEST_COMPARE(a, b, op, name, on_fail)                                        \
  do {                                                                                   \
    const auto& t5d_lhs_ = (a);                                                          \
    const auto& t5d_rhs_ = (b);                                                          \
    if (!(t5d_lhs_ op t5d_rhs_)) {                                                       \
      ::t5d_test::record_failure(__FILE__, __LINE__,                                     \
                                 name "(" #a ", " #b ") failed: " +                      \
                                     ::t5d_test::show(t5d_lhs_) + " vs " +               \
                                     ::t5d_test::show(t5d_rhs_));                        \
      on_fail;                                                                           \
    }                                                                                    \
  } while (false)

#define CHECK_EQ(a, b) T5D_TEST_COMPARE(a, b, ==, "CHECK_EQ", (void)0)
#define CHECK_NE(a, b) T5D_TEST_COMPARE(a, b, !=, "CHECK_NE", (void)0)
#define REQUIRE_EQ(a, b) T5D_TEST_COMPARE(a, b, ==, "REQUIRE_EQ", throw ::t5d_test::RequireFailed{})

#define REQUIRE(cond)                                                                    \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      ::t5d_test::record_failure(__FILE__, __LINE__, "REQUIRE(" #cond ") failed");      \
      throw ::t5d_test::RequireFailed{};                                                 \
    }                                                                                    \
  } while (false)

#define CHECK_THROWS_WITH(expr, type, text)                                              \
  do {                                                                                   \
    bool t5d_caught_ = false;                                                            \
    try {                                                                                \
      (void)(expr);                                                                      \
    } catch (const type& t5d_e_) {                                                       \
      t5d_caught_ = true;                                                                \
      if (!::t5d_test::what_contains(t5d_e_, text)) {                                    \
        ::t5d_test::record_failure(__FILE__, __LINE__,                                   \
                                   "CHECK_THROWS_WITH(" #expr "): message \"" +          \
                                       std::string(t5d_e_.what()) + "\" lacks \"" +      \
                                       std::string(text) + "\"");                        \
      }                                                                                  \
    } catch (const std::exception& t5d_e_) {                                             \
      t5d_caught_ = true;                                                                \
      ::t5d_test::record_failure(__FILE__, __LINE__,                                     \
                                 "CHECK_THROWS(" #expr "): threw another exception: " +  \
                                     std::string(t5d_e_.what()));                        \
    } catch (...) {                                                                      \
      t5d_caught_ = true;                                                                \
      ::t5d_test::record_failure(__FILE__, __LINE__,                                     \
                                 "CHECK_THROWS(" #expr "): threw a non-standard exception"); \
    }                                                                                    \
    if (!t5d_caught_) {                                                                  \
      ::t5d_test::record_failure(__FILE__, __LINE__,                                     \
                                 "CHECK_THROWS(" #expr ", " #type "): nothing thrown");   \
    }                                                                                    \
  } while (false)

#define CHECK_THROWS_AS(expr, type) CHECK_THROWS_WITH(expr, type, "")

#define CHECK_NOTHROW(expr)                                                              \
  do {                                                                                   \
    try {                                                                                \
      (void)(expr);                                                                      \
    } catch (const std::exception& t5d_e_) {                                             \
      ::t5d_test::record_failure(__FILE__, __LINE__,                                     \
                                 "CHECK_NOTHROW(" #expr "): threw: " +                   \
                                     std::string(t5d_e_.what()));                        \
    } catch (...) {                                                                      \
      ::t5d_test::record_failure(__FILE__, __LINE__,                                     \
                                 "CHECK_NOTHROW(" #expr "): threw a non-standard exception"); \
    }                                                                                    \
  } while (false)

#endif  // T5D_TEST_FRAMEWORK_HPP
