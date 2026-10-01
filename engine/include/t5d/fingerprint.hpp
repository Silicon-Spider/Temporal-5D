// t5d/fingerprint.hpp - the canonical fingerprint of a state (decision H30).
//
// "Match states, not move counts" (spec §11.0, "Running the gate"): every
// implementation exports a canonical fingerprint of the position it parsed, at
// every ply, and a mismatch is a parse divergence, triaged before any rules
// comparison at that ply.
//
// The fingerprint is the notation specification's full-state hash ("Hashing",
// with "5DFEN Strictness"):
//
//  * the canonical state string is the canonical 5DFEN board string of EVERY
//    board of the multiverse - superseded and head boards alike - concatenated
//    with no separator and no whitespace, in ascending order of timeline, then
//    of turn, then white's board before black's within a turn ("-0" sorts
//    before "+0");
//  * fingerprint = the lower-case hex MD5 of that string;
//  * placement  = the MD5 of the same string with every '*' removed (the
//    placement without the unmoved markers, for telling a marker disagreement
//    from a placement one);
//  * boards = the number of boards; pieces = the number of pieces on them all.
//
// The specification sorts by timeline then turn and is silent on the colour
// order within a turn; white-before-black is this engine's resolution (it is
// ascending half-turn order). The Standard start hashes to
// d574889fd9da3f2bc65249ff27249b00 and the two-timeline start to
// 3672761404ffcd15ae644c75401812be, the specification's two published examples.

#ifndef T5D_FINGERPRINT_HPP
#define T5D_FINGERPRINT_HPP

#include <cstddef>
#include <string>

#include "t5d/multiverse.hpp"

namespace t5d {

/// The canonical state string.
std::string canonical_state_string(const Multiverse& multiverse);

struct Fingerprint {
  std::string fingerprint;  // 32 lower-case hex digits
  std::string placement;    // 32 lower-case hex digits
  std::size_t boards = 0;
  std::size_t pieces = 0;

  friend bool operator==(const Fingerprint&, const Fingerprint&) = default;
};

Fingerprint fingerprint(const Multiverse& multiverse);

}  // namespace t5d

#endif  // T5D_FINGERPRINT_HPP
