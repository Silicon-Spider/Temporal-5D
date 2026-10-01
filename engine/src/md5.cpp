// MD5, from RFC 1321 ("The MD5 Message-Digest Algorithm", R. Rivest, 1992),
// section 3: append padding bits (3.1) and the length (3.2), initialise the
// buffer (3.3), process the message in 16-word blocks (3.4), output (3.5).
// Words are assembled from bytes explicitly, so the code does not depend on
// the machine's byte order.

#include "t5d/md5.hpp"

#include <algorithm>
#include <cstring>

namespace t5d {

namespace {

// T[i] = floor(2^32 * abs(sin(i + 1))), i = 0..63 (RFC 1321, 3.4).
constexpr std::array<std::uint32_t, 64> kT = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
    0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
    0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
    0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
    0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
    0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
    0xeb86d391};

// The per-step rotation amounts of the four rounds.
constexpr std::array<int, 64> kShift = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,  // round 1
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,  // round 2
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,  // round 3
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,  // round 4
};

constexpr std::uint32_t rotl32(std::uint32_t x, int n) noexcept {
  return (x << n) | (x >> (32 - n));
}

}  // namespace

Md5::Md5() noexcept : state_{0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u} {}

void Md5::process_block(const std::uint8_t* block) noexcept {
  std::array<std::uint32_t, 16> x{};
  for (std::size_t i = 0; i < 16; ++i) {
    x[i] = std::uint32_t{block[4 * i]} | (std::uint32_t{block[4 * i + 1]} << 8) |
           (std::uint32_t{block[4 * i + 2]} << 16) | (std::uint32_t{block[4 * i + 3]} << 24);
  }
  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  // The 64 steps "a = b + ((a + F(b,c,d) + X[k] + T[i]) <<< s)", with the
  // roles of a, b, c, d rotating after each step.
  for (std::size_t i = 0; i < 64; ++i) {
    std::uint32_t f = 0;
    std::size_t k = 0;
    if (i < 16) {
      f = (b & c) | (~b & d);  // F
      k = i;
    } else if (i < 32) {
      f = (b & d) | (c & ~d);  // G
      k = (5 * i + 1) % 16;
    } else if (i < 48) {
      f = b ^ c ^ d;  // H
      k = (3 * i + 5) % 16;
    } else {
      f = c ^ (b | ~d);  // I
      k = (7 * i) % 16;
    }
    const std::uint32_t next = b + rotl32(a + f + x[k] + kT[i], kShift[i]);
    a = d;
    d = c;
    c = b;
    b = next;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
}

void Md5::update(const void* data, std::size_t size) noexcept {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  length_ += size;
  if (buffered_ > 0) {
    const std::size_t take = std::min(size, buffer_.size() - buffered_);
    std::memcpy(buffer_.data() + buffered_, bytes, take);
    buffered_ += take;
    bytes += take;
    size -= take;
    if (buffered_ < buffer_.size()) return;
    process_block(buffer_.data());
    buffered_ = 0;
  }
  while (size >= 64) {
    process_block(bytes);
    bytes += 64;
    size -= 64;
  }
  if (size > 0) {
    std::memcpy(buffer_.data(), bytes, size);
    buffered_ = size;
  }
}

Md5::Digest Md5::finish() noexcept {
  const std::uint64_t bit_length = length_ * 8;  // modulo 2^64, as 3.2 says
  // 3.1: a single 1 bit, then 0 bits until the length is 448 modulo 512.
  const std::uint8_t one = 0x80;
  update(&one, 1);
  const std::uint8_t zero = 0;
  while (buffered_ != 56) update(&zero, 1);
  // 3.2: the 64-bit length, low-order word first, each word low byte first.
  std::array<std::uint8_t, 8> length_bytes{};
  for (std::size_t i = 0; i < 8; ++i) {
    length_bytes[i] = static_cast<std::uint8_t>(bit_length >> (8 * i));
  }
  update(length_bytes.data(), length_bytes.size());
  // 3.5: A, B, C, D, each low byte first.
  Digest digest{};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      digest[4 * i + j] = static_cast<std::uint8_t>(state_[i] >> (8 * j));
    }
  }
  return digest;
}

std::string Md5::hex(const Digest& digest) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (const std::uint8_t byte : digest) {
    out += kHex[byte >> 4];
    out += kHex[byte & 15];
  }
  return out;
}

std::string md5_hex(std::string_view text) {
  Md5 md5;
  md5.update(text);
  return Md5::hex(md5.finish());
}

}  // namespace t5d
