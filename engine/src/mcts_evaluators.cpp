// The built-in evaluators (t5d/mcts.hpp): uniform, seeded random and material
// count - for tests, the §15.4 invariants and the tools. None is a network;
// each is a pure function of the state it is asked about (and, for the random
// one, its seed and the refresh epoch), as the network's evaluations must be
// (§8.5, C19-C21).

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>

#include "t5d/mcts.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/types.hpp"

namespace t5d::mcts {

namespace {

constexpr std::uint64_t mix(std::uint64_t z) noexcept {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// In [0, 1), from a 64-bit value.
double unit(std::uint64_t x) noexcept { return static_cast<double>(x >> 11) * 0x1.0p-53; }

class UniformEvaluator final : public Evaluator {
 public:
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      out[i].logits.assign(batch[i].expansion->moves().size(), 0.0f);
      out[i].value = two_hot(0.0);
    }
  }
};

class RandomEvaluator final : public Evaluator {
 public:
  explicit RandomEvaluator(std::uint64_t seed) : seed_(seed) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const std::uint64_t base = mix(seed_ ^ mix(batch[i].key ^ mix(batch[i].epoch + 0x51ull)));
      const std::size_t n = batch[i].expansion->moves().size();
      out[i].logits.resize(n);
      for (std::size_t m = 0; m < n; ++m) {
        // An approximately normal logit: the sum of three uniforms.
        const std::uint64_t r = mix(base + 0x9E3779B97F4A7C15ull * (m + 1));
        const double g = unit(r) + unit(mix(r)) + unit(mix(r + 1)) - 1.5;
        out[i].logits[m] = static_cast<float>(2.0 * g);
      }
      // A bump at a random mean with a random width: values and variances vary.
      const double mean = 1.6 * unit(mix(base ^ 0xA5A5ull)) - 0.8;
      const double sd = 0.05 + 0.35 * unit(mix(base ^ 0x5A5Aull));
      double sum = 0;
      std::array<double, kValueBins> p{};
      for (int k = 0; k < kValueBins; ++k) {
        const double d = (value_bin_centre(k) - mean) / sd;
        p[static_cast<std::size_t>(k)] = std::exp(-0.5 * d * d);
        sum += p[static_cast<std::size_t>(k)];
      }
      for (std::size_t k = 0; k < p.size(); ++k) out[i].value[k] = static_cast<float>(p[k] / sum);
    }
  }

 private:
  std::uint64_t seed_;
};

class MaterialEvaluator final : public Evaluator {
 public:
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const GameState& s = *batch[i].state;
      const Colour mover = s.to_move();
      double balance = 0;
      for (const TimelineRef& t : s.multiverse().timelines()) {
        const Board& b = *t->head();
        std::uint64_t occupied = b.occupied();
        while (occupied != 0) {
          const int sq = std::countr_zero(occupied);
          occupied &= occupied - 1;
          const Piece p = b.at(sq);
          balance += (p.colour() == mover ? 1.0 : -1.0) * material_value(p.kind());
        }
      }
      out[i].value = two_hot(std::tanh(balance / 10.0));
      const MoveList& moves = batch[i].expansion->moves();
      out[i].logits.resize(moves.size());
      for (std::size_t m = 0; m < moves.size(); ++m) {
        const Piece taken = moves[m].captured;
        out[i].logits[m] = static_cast<float>(taken.empty() ? 0.0 : 0.5 * material_value(taken.kind()));
      }
    }
  }
};

}  // namespace

double material_value(PieceKind kind) noexcept {
  switch (kind) {
    case PieceKind::Pawn: return 1;
    case PieceKind::Knight: return 3;
    case PieceKind::Bishop: return 3;
    case PieceKind::Rook: return 5;
    case PieceKind::Queen: return 9;
    case PieceKind::King: return 0;        // royal
    // The fairy pieces: the scanner's placeholders (build-log spec issue 56),
    // one set for the engine until the spec gives values.
    case PieceKind::Unicorn: return 2.5;
    case PieceKind::Dragon: return 2.0;
    case PieceKind::Princess: return 7.0;
    case PieceKind::Brawn: return 1.25;
    case PieceKind::CommonKing: return 3.0;  // not royal (4.12)
    case PieceKind::RoyalQueen: return 0;  // royal
  }
  return 0;
}

std::unique_ptr<Evaluator> make_evaluator(BuiltinEvaluator kind, std::uint64_t seed) {
  switch (kind) {
    case BuiltinEvaluator::Uniform: return std::make_unique<UniformEvaluator>();
    case BuiltinEvaluator::Random: return std::make_unique<RandomEvaluator>(seed);
    case BuiltinEvaluator::Material: return std::make_unique<MaterialEvaluator>();
  }
  return std::make_unique<UniformEvaluator>();
}

const char* evaluator_name(BuiltinEvaluator kind) noexcept {
  switch (kind) {
    case BuiltinEvaluator::Uniform: return "uniform";
    case BuiltinEvaluator::Random: return "random";
    case BuiltinEvaluator::Material: return "material";
  }
  return "?";
}

}  // namespace t5d::mcts
