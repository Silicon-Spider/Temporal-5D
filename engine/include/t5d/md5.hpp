// t5d/md5.hpp - MD5, implemented from RFC 1321.
//
// The notation specification's hash of a state ("Hashing") is the MD5 of its
// boards' 5DFEN strings; the canonical fingerprint (H30) is that hash. MD5 is
// used here as a checksum for matching states between implementations, not
// for security.

#ifndef T5D_MD5_HPP
#define T5D_MD5_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace t5d {

class Md5 {
 public:
  using Digest = std::array<std::uint8_t, 16>;

  Md5() noexcept;

  /// Feeds bytes. Must not be called after finish().
  void update(const void* data, std::size_t size) noexcept;
  void update(std::string_view text) noexcept { update(text.data(), text.size()); }

  /// Pads, and returns the digest. Call once.
  Digest finish() noexcept;

  /// Lower-case hexadecimal, 32 characters.
  static std::string hex(const Digest& digest);

 private:
  void process_block(const std::uint8_t* block) noexcept;

  std::array<std::uint32_t, 4> state_;
  std::uint64_t length_ = 0;  // bytes fed so far
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffered_ = 0;
};

/// The lower-case hex MD5 of a string's bytes.
std::string md5_hex(std::string_view text);

}  // namespace t5d

#endif  // T5D_MD5_HPP
