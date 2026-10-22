// t5d-play: the engine plays a game against itself (engine stage E6a) and
// prints its 5dpgn record on standard output - a record the engine's reader,
// and so the gate adapter (t5d-gate), replays. A summary goes to standard
// error.
//
//   t5d-play [--variant NAME] [--evaluator uniform|random|material]
//            [--playouts N] [--split ply|flat|tree] [--tree-gates all|none|always|LIST]
//            [--graph] [--node-key state|origins]
//            [--no-widening] [--e2] [--e3] [--proof-channel] [--batch N] [--synchronous]
//            [--match] [--final visits|lcb] [--temperature T] [--temperature-decisions N]
//            [--noise | --no-noise] [--cap-turns N] [--cap-boards N] [--max-actions N]
//            [--mate-nodes N] [--max-seconds S] [--seed S]
//            [--no-moves-after-shift] [--rules ENCODING]
//
// Defaults: Standard, the uniform evaluator, 64 playouts per decision, the
// board-move split (D2 is undecided: --split flat and --split tree are the
// other arms - the flat form generates whole turns best first, and replaced
// the enumerating whole-turn arm, retired with its --split turn and its
// --listing-cap; the tree form's gates, --tree-gates, are §8.2's four by default -
// a comma-separated list of between, capture, check and present, or always:
// every mid-turn node evaluated, the ply split's search; refused with another
// split, which reads no gate; the Engine header names them), a tree
// (--graph: MCGS, C11), self-play (a fresh search per decision, C14; visits
// sampled at temperature 1 for the first 16 decisions, Dirichlet noise at the
// root: --no-noise for none) - --match plays the
// final rule (E5, LCB by default) with the tree carried (C12) - and a turn cap
// of 20 turns or 300 boards (I11: a convention, scored as a draw; --cap-turns 0
// --cap-boards 0 lift it: with root noise, time travel can hold the present
// back for good).
// --proof-channel turns on the in-tree solver's propagation of proven bounds
// (H2, E9) - the proof channel's switch, off by rule (H41) until the gate
// passes and the channel is signed off: for experiments only.
// --max-seconds stops a long game (a safety stop, "*": checked before every
// decision, and each search's deadline; the record ends at the last turn
// start), its record and summary still printed.
// Exit status: 0 for a game played, 2 for a bad command line (a number out of
// range included), 1 for an error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "play_record.hpp"
#include "t5d/mcts.hpp"
#include "t5d/rules.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/variants.hpp"
#include "tool_support.hpp"

namespace {

using namespace t5d;
using namespace t5d::mcts;

void usage(std::FILE* to) {
  std::fputs(
      "usage: t5d-play [--variant NAME] [--evaluator uniform|random|material]\n"
      "                [--playouts N] [--split ply|flat|tree]\n"
      "                [--tree-gates all|none|always|LIST] [--graph] [--node-key state|origins]\n"
      "                [--no-widening] [--e2] [--e3] [--proof-channel] [--batch N]\n"
      "                [--synchronous]\n"
      "                [--match] [--final visits|lcb] [--temperature T]\n"
      "                [--temperature-decisions N] [--noise | --no-noise] [--cap-turns N]\n"
      "                [--cap-boards N]\n"
      "                [--max-actions N] [--mate-nodes N]\n"
      "                [--max-seconds S] [--seed S]\n"
      "                [--no-moves-after-shift] [--rules ENCODING]\n",
      to);
}

std::uint64_t number(const char* option, const char* text) {
  const std::optional<std::uint64_t> n = tools::parse_number(text);
  if (!n) throw std::invalid_argument(std::string(option) + ": not a number: " + text);
  return *n;
}

// A whole number in [low, high] - never wrapped into a narrower type.
std::uint64_t number_in(const char* option, const char* text, std::uint64_t low, std::uint64_t high) {
  const std::uint64_t n = number(option, text);
  if (n < low || n > high) {
    throw std::invalid_argument(std::string(option) + ": " + text + " is out of range (" + std::to_string(low) +
                                " to " + std::to_string(high) + ")");
  }
  return n;
}

// A finite real number, 0 or more.
double real(const char* option, const char* text) {
  char* end = nullptr;
  const double v = std::strtod(text, &end);
  if (end == text || *end != '\0') throw std::invalid_argument(std::string(option) + ": not a number: " + text);
  if (!std::isfinite(v) || v < 0) {
    throw std::invalid_argument(std::string(option) + ": " + text + " is not a finite number of 0 or more");
  }
  return v;
}

// The tree form's gates by name, as --tree-gates takes them.
std::string gate_names(std::uint32_t gates) {
  if (gates == 0) return "none";
  if ((gates & kGateAlways) != 0) return "always";
  std::string out;
  for (const auto& [bit, name] : {std::pair<std::uint32_t, const char*>{kGateBetween, "between"},
                                  {kGateCapture, "capture"}, {kGateCheck, "check"}, {kGatePresent, "present"}}) {
    if ((gates & bit) == 0) continue;
    if (!out.empty()) out += ",";
    out += name;
  }
  return out;
}

constexpr std::uint64_t kU32 = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t kInt = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
constexpr std::uint64_t kAny = std::numeric_limits<std::uint64_t>::max();

}  // namespace

int main(int argc, char** argv) {
  std::string variant_name = "Standard";
  BuiltinEvaluator evaluator = BuiltinEvaluator::Uniform;
  PlayParams params;
  params.search.expand.square_features = SquareFeatures::None;  // the built-ins read none
  params.turn_cap.turns = 20;
  params.turn_cap.boards = 300;
  RuleOptions rules;
  bool tree_gates_given = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const char* a = argv[i];
      const auto value = [&]() -> const char* {
        if (i + 1 >= argc) throw std::invalid_argument(std::string(a) + " needs a value");
        return argv[++i];
      };
      if (const int used = tools::ruleset_option(argc, argv, i, rules); used > 0) {
        i += used - 1;
      } else if (std::strcmp(a, "--variant") == 0) {
        variant_name = value();
      } else if (std::strcmp(a, "--evaluator") == 0) {
        const std::string v = value();
        if (v == "uniform") {
          evaluator = BuiltinEvaluator::Uniform;
        } else if (v == "random") {
          evaluator = BuiltinEvaluator::Random;
        } else if (v == "material") {
          evaluator = BuiltinEvaluator::Material;
        } else {
          throw std::invalid_argument("--evaluator: uniform, random or material, not " + v);
        }
      } else if (std::strcmp(a, "--playouts") == 0) {
        params.playouts = number_in(a, value(), 1, kAny);
      } else if (std::strcmp(a, "--split") == 0) {
        const std::string v = value();
        if (v == "ply") {
          params.search.split = TurnSplit::BoardMove;
        } else if (v == "flat") {
          params.search.split = TurnSplit::Flat;
        } else if (v == "tree") {
          params.search.split = TurnSplit::Tree;
        } else if (v == "turn") {
          throw std::invalid_argument(
              "--split turn: the enumerating whole-turn arm was retired - D2's flat form (--split flat) "
              "replaced it");
        } else {
          throw std::invalid_argument("--split: ply, flat or tree, not " + v);
        }
      } else if (std::strcmp(a, "--tree-gates") == 0) {
        const std::string v = value();
        std::uint32_t gates = 0;
        if (v == "all") {
          gates = kGatesInteraction;
        } else if (v == "always") {
          gates = kGateAlways;
        } else if (v != "none") {
          std::size_t at = 0;
          while (at <= v.size()) {
            const std::size_t comma = std::min(v.find(',', at), v.size());
            const std::string g = v.substr(at, comma - at);
            if (g == "between") {
              gates |= kGateBetween;
            } else if (g == "capture") {
              gates |= kGateCapture;
            } else if (g == "check") {
              gates |= kGateCheck;
            } else if (g == "present") {
              gates |= kGatePresent;
            } else {
              throw std::invalid_argument("--tree-gates: all, none, always or a list of between, capture, check "
                                          "and present, not " + v);
            }
            at = comma + 1;
          }
        }
        params.search.tree_gates = gates;
        tree_gates_given = true;
      } else if (std::strcmp(a, "--graph") == 0) {
        params.search.graph = true;
      } else if (std::strcmp(a, "--node-key") == 0) {
        const std::string v = value();
        if (v == "state") {
          params.search.node_key = NodeKey::State;
        } else if (v == "origins") {
          params.search.node_key = NodeKey::StateAndOrigins;
        } else {
          throw std::invalid_argument("--node-key: state or origins, not " + v);
        }
      } else if (std::strcmp(a, "--no-widening") == 0) {
        params.search.widening = false;
      } else if (std::strcmp(a, "--e2") == 0) {
        params.search.variance_scaled_cpuct = true;
        params.search.uncertainty_weighted_backup = true;
      } else if (std::strcmp(a, "--e3") == 0) {
        params.search.variance_bonus = true;
      } else if (std::strcmp(a, "--proof-channel") == 0) {
        params.search.proof_channel = true;  // H41: off by rule; experiments only
      } else if (std::strcmp(a, "--batch") == 0) {
        params.search.batch_size = static_cast<std::size_t>(number_in(a, value(), 1, kU32));
      } else if (std::strcmp(a, "--synchronous") == 0) {
        params.search.synchronous = true;
      } else if (std::strcmp(a, "--match") == 0) {
        params.match_play = true;
      } else if (std::strcmp(a, "--final") == 0) {
        const std::string v = value();
        if (v == "visits") {
          params.search.final_rule = FinalRule::Visits;
        } else if (v == "lcb") {
          params.search.final_rule = FinalRule::Lcb;
        } else {
          throw std::invalid_argument("--final: visits or lcb, not " + v);
        }
      } else if (std::strcmp(a, "--temperature") == 0) {
        params.temperature = real(a, value());
      } else if (std::strcmp(a, "--temperature-decisions") == 0) {
        params.temperature_decisions = static_cast<std::uint32_t>(number_in(a, value(), 0, kU32));
      } else if (std::strcmp(a, "--noise") == 0) {
        params.search.root_noise = true;  // the self-play default (match play has none)
      } else if (std::strcmp(a, "--no-noise") == 0) {
        params.search.root_noise = false;
      } else if (std::strcmp(a, "--cap-turns") == 0) {
        params.turn_cap.turns = static_cast<int>(number_in(a, value(), 0, kInt));
      } else if (std::strcmp(a, "--cap-boards") == 0) {
        params.turn_cap.boards = static_cast<int>(number_in(a, value(), 0, kInt));
      } else if (std::strcmp(a, "--max-actions") == 0) {
        params.max_actions = static_cast<std::uint32_t>(number_in(a, value(), 1, kU32));
      } else if (std::strcmp(a, "--mate-nodes") == 0) {
        params.search.mate_nodes = number_in(a, value(), 1, kAny);
      } else if (std::strcmp(a, "--max-seconds") == 0) {
        params.max_seconds = real(a, value());
      } else if (std::strcmp(a, "--seed") == 0) {
        params.seed = number(a, value());
        params.search.seed = params.seed;
      } else if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
        usage(stdout);
        return 0;
      } else {
        throw std::invalid_argument(std::string("unknown option ") + a);
      }
    }
    // The gates are the tree form's alone: given with another split they
    // would be read by nothing, and the record would not say so.
    if (tree_gates_given && params.search.split != TurnSplit::Tree) {
      throw std::invalid_argument("--tree-gates: the tree form's interaction gates, read only with --split tree");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-play: %s\n", e.what());
    usage(stderr);
    return 2;
  }
  try {
    const Variant* v = find_variant(variant_name);
    if (v == nullptr) {
      std::fprintf(stderr, "t5d-play: the engine sets up no variant named \"%s\"\n", variant_name.c_str());
      return 2;
    }
    const GameState start = variant_start(*v, rules);
    const std::unique_ptr<Evaluator> ev = make_evaluator(evaluator, params.seed);
    const GameRecord g = play_game(start, *ev, params);
    tools::Headers extra;
    extra.emplace_back("White", "t5d-play");
    extra.emplace_back("Black", "t5d-play");
    std::string engine = std::string("t5d search: ") + evaluator_name(evaluator) + " evaluator, " +
                         std::to_string(params.playouts) + " playouts per decision, " +
                         (params.search.split == TurnSplit::BoardMove ? "board-move split"
                          : params.search.split == TurnSplit::Flat    ? "flat form"
                                                                       : "tree form, gates " +
                                                                             gate_names(params.search.tree_gates)) +
                         (params.search.graph ? ", graph" : ", tree") +
                         (params.search.proof_channel ? ", proof channel ON (H41: experiments only)" : "") +
                         (params.search.root_noise && !params.match_play ? ", root noise" : "") +
                         (params.match_play ? ", match play" : ", self-play") + ", seed " +
                         std::to_string(params.seed);
    extra.emplace_back("Engine", engine);
    if (params.turn_cap.enabled()) {
      extra.emplace_back("TurnCap", "turns " + std::to_string(params.turn_cap.turns) + ", boards " +
                                        std::to_string(params.turn_cap.boards));
    }
    if (rules.encode() != 0) extra.emplace_back("Rules", tools::ruleset_text(rules));
    std::fputs(tools::play_record(g, v->name, extra).c_str(), stdout);
    std::fflush(stdout);
    std::fprintf(stderr,
                 "t5d-play: %zu actions, %s, result %s, %llu decisions, %llu playouts, %llu "
                 "evaluations, %llu nodes, %.2f s, turn cap %s (capped positions met in search: %llu)\n",
                 g.actions.size(), game_end_name(g.end), g.result().c_str(),
                 static_cast<unsigned long long>(g.decisions),
                 static_cast<unsigned long long>(g.playouts),
                 static_cast<unsigned long long>(g.evaluations),
                 static_cast<unsigned long long>(g.nodes), g.seconds,
                 g.cap_bound() ? "bound" : "not bound",
                 static_cast<unsigned long long>(g.search_turn_caps));
    std::fprintf(stderr,
                 "t5d-play: seconds in the rules engine - search's terminal checks %.2f, leaf "
                 "expansions %.2f, turn generation %.2f, verifications %.2f; the game's turn-start "
                 "verdicts %.2f; final position %zu boards, %d timelines\n",
                 g.seconds_search_mate, g.seconds_search_expand, g.seconds_search_generate,
                 g.seconds_search_verify, g.seconds_verdicts, g.final_state.multiverse().board_count(),
                 g.final_state.multiverse().timeline_count());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t5d-play: error: %s\n", e.what());
    return 1;
  }
}
