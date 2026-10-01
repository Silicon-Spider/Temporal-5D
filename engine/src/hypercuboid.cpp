// The hypercuboid detector (see t5d/search.hpp and the engine README). Built
// in the next step of stage E4; until then it refuses to answer.

#include <stdexcept>

#include "search_internal.hpp"

namespace t5d::detail {

void run_hypercuboid(SearchRun&) {
  throw std::logic_error("the hypercuboid detector is not built yet");
}

}  // namespace t5d::detail
