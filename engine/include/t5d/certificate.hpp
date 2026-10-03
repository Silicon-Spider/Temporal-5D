// t5d/certificate.hpp - checking a proof's certificate against E4 alone
// (stage E9; verification v3, spec 11.4; register H37, H41, H42, H43).
//
// A mate the prover proves (t5d/prover.hpp; Hunt, Guard and Hunt's ground
// truth) is checked here with none of the prover's machinery - no semimove
// tree, no attack map, no pruning:
//  * the ATTACKER's moves come from the certificate (ProofCertificate: a
//    proving action per position), each replayed through the state machine
//    - pseudo-legal where it is made (is_pseudo_legal), the obligations met
//    (T5), no royal left capturable (C4) - and submitted;
//  * the DEFENDER's every legal action comes from E4's exhaustive listing
//    (list_legal_actions, the hypercuboid detector);
//  * a CHECKMATE is E4's verdict: the hypercuboid detector's, and - where it
//    finishes within its budget - the naive detector's, which must agree
//    (spec 11.0's two detectors, cross-checked).
// A stalemate, a defender not mated at the horizon, an illegal move or a
// disagreement of the detectors REFUTES the proof; a position the
// certificate lacks makes it INCOMPLETE; a listing or a detector that runs
// out of its budget leaves it UNDECIDED. Never "confirmed" on a budget stop
// (H37's rule).

#ifndef T5D_CERTIFICATE_HPP
#define T5D_CERTIFICATE_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include "t5d/prover.hpp"
#include "t5d/state.hpp"

namespace t5d {

struct CertificateCheckOptions {
  /// A defender's turn with more legal actions than this: undecided.
  std::size_t listing_cap = 100000;
  std::uint64_t listing_nodes = 5'000'000;
  /// The hypercuboid detector's budget per defender position.
  std::uint64_t detector_nodes = 2'000'000;
  /// The naive detector as well, where it finishes within naive_nodes.
  bool naive = true;
  std::uint64_t naive_nodes = 500'000;
  /// Defender positions visited at most (then undecided).
  std::size_t max_positions = 200000;
  std::size_t max_bytes = std::size_t{256} << 20;
};

enum class CertificateVerdict : std::uint8_t { Confirmed = 0, Refuted, Incomplete, Undecided };
const char* certificate_verdict_name(CertificateVerdict v) noexcept;

struct CertificateCheck {
  CertificateVerdict verdict = CertificateVerdict::Undecided;
  /// Refuted, Incomplete, Undecided: where and why.
  std::string why;
  std::size_t attacker_positions = 0;
  std::size_t defender_positions = 0;
  std::size_t defender_actions = 0;  // E4's listed actions followed
  std::size_t mates = 0;             // checkmates by the hypercuboid detector
  std::size_t mates_naive = 0;       // ... and the naive one agreeing
  std::size_t naive_undecided = 0;   // ... the naive one out of its budget
};

/// Checks that `certificate` proves mate within `depth` half-turns for the
/// side to move at `root`.
CertificateCheck check_certificate(const ProofCertificate& certificate, const GameState& root, int depth,
                                   const CertificateCheckOptions& options = {});

}  // namespace t5d

#endif  // T5D_CERTIFICATE_HPP
