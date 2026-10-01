// t5d/error.hpp - the engine's exception types.
//
// Inside the C++ core, malformed notation raises NotationError and misuse of an
// API (a null board, an off-board square, a timeline that would leave a gap)
// raises std::invalid_argument or std::out_of_range. The C API (t5d/capi.h)
// catches everything: no exception crosses that boundary.

#ifndef T5D_ERROR_HPP
#define T5D_ERROR_HPP

#include <stdexcept>
#include <string>

namespace t5d {

/// Input that does not follow the pinned notation specification (5dpgn /
/// 5DFEN), or that describes no possible multiverse (two boards in one place,
/// a gap in a timeline). The message says what was wrong and where.
class NotationError : public std::runtime_error {
 public:
  explicit NotationError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace t5d

#endif  // T5D_ERROR_HPP
