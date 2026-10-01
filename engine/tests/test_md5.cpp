// MD5 against RFC 1321's test suite (appendix A.5) and against reference
// digests at the padding boundaries (computed with an independent MD5, Python's
// hashlib, and recorded here).

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "t5d/md5.hpp"
#include "test_framework.hpp"
#include "test_support.hpp"

using namespace t5d;

TEST(md5_rfc1321_test_suite) {
  CHECK_EQ(md5_hex(""), std::string("d41d8cd98f00b204e9800998ecf8427e"));
  CHECK_EQ(md5_hex("a"), std::string("0cc175b9c0f1b6a831c399e269772661"));
  CHECK_EQ(md5_hex("abc"), std::string("900150983cd24fb0d6963f7d28e17f72"));
  CHECK_EQ(md5_hex("message digest"), std::string("f96b697d7cb7938d525a2f31aaf161d0"));
  CHECK_EQ(md5_hex("abcdefghijklmnopqrstuvwxyz"),
           std::string("c3fcd3d76192e4007dfb496cca67e13b"));
  CHECK_EQ(md5_hex("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
           std::string("d174ab98d277d9f5a5611c2c9f419d9f"));
  CHECK_EQ(md5_hex("12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
           std::string("57edf4a22be3c955ac49da2e2107b67a"));
}

namespace {

std::string alphabet_run(std::size_t n) {
  std::string s;
  for (std::size_t i = 0; i < n; ++i) s += static_cast<char>('a' + i % 26);
  return s;
}

}  // namespace

TEST(md5_padding_boundaries) {
  const struct {
    std::size_t length;
    const char* digest;
  } cases[] = {
      {55, "0d7ae056b2f015cd7dc67494efd658f1"},  {56, "31fcfb5165169eb55898e7e4cf34d19a"},
      {57, "fd62afaf3aa1e2a52882cb464f5ccc4d"},  {63, "1b30c0670c15e7da3c2ba7bce77ebe99"},
      {64, "a2eaf6295c32adc403865fd96a2f182b"},  {65, "eba2cce0ca8df47e62414a736b3105a2"},
      {119, "b05187e08da41fa3ef16bd56afaafd99"}, {120, "62af9b597a9f55e16ab2b897387fc052"},
      {127, "f41f37f8b5a7782d5cfeafc43e3d3c1c"}, {128, "3e8c1ccbd71838ef3df4b72e57fb9bf6"},
      {1000, "303fb697b589019cb3edba04b794e575"},
  };
  for (const auto& c : cases) {
    CHECK_EQ(md5_hex(alphabet_run(c.length)), std::string(c.digest));
  }
}

TEST(md5_long_and_binary_input) {
  CHECK_EQ(md5_hex(std::string(1000000, 'a')), std::string("7707d6ae4e027c70eea2a935c2296f21"));
  std::string bytes;
  for (int i = 0; i < 256; ++i) bytes += static_cast<char>(i);
  CHECK_EQ(md5_hex(bytes), std::string("e2c865db4162bed963bfaa9ef6ac18f0"));
}

TEST(md5_incremental_updates_match_one_shot) {
  t5d_test::Rng rng(1321);
  for (int round = 0; round < 50; ++round) {
    const std::string text = alphabet_run(static_cast<std::size_t>(rng.below(700)));
    const std::string expected = md5_hex(text);
    Md5 md5;
    std::size_t at = 0;
    while (at < text.size()) {
      const auto chunk = std::min(text.size() - at, static_cast<std::size_t>(rng.below(130)));
      md5.update(text.data() + at, chunk);
      at += chunk;
    }
    CHECK_EQ(Md5::hex(md5.finish()), expected);
  }
}

TEST(md5_hex_is_lower_case) {
  const std::string hex = md5_hex("t5d");
  CHECK_EQ(hex.size(), static_cast<std::size_t>(32));
  for (const char c : hex) CHECK((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
}
