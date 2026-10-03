// SHA-256 (FIPS 180-4), for pinning recorded files by their bytes: Hunt's
// ground truth and its MANIFEST (stage E9), as the oracle harness pins its
// recorded answers. Written from the standard.

#ifndef T5D_TOOLS_SHA256_HPP
#define T5D_TOOLS_SHA256_HPP

#include <string>
#include <string_view>

namespace t5d::tools {

/// The lower-case hex SHA-256 of `bytes` (64 digits).
std::string sha256_hex(std::string_view bytes);

}  // namespace t5d::tools

#endif  // T5D_TOOLS_SHA256_HPP
