// The offline audit lane's tooling (tools/audit.hpp; engine stage E10).

#include "audit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <utility>

#include "t5d/fen.hpp"
#include "t5d/gate.hpp"
#include "t5d/guard.hpp"
#include "t5d/movegen.hpp"
#include "t5d/variants.hpp"

namespace t5d::audit {

namespace {

using gate::json_string;
using tools::sixbpp::Check;

// ---- a small JSON reader (the logs' own lines) ------------------------------------------

struct Json {
  enum class Type : std::uint8_t { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool b = false;
  std::string text;  // String: the value; Number: its literal
  std::vector<Json> items;
  std::vector<std::pair<std::string, Json>> fields;

  const Json* get(std::string_view key) const {
    for (const auto& [k, v] : fields) {
      if (k == key) return &v;
    }
    return nullptr;
  }
  const Json& at(std::string_view key) const {
    const Json* v = get(key);
    if (v == nullptr) throw std::invalid_argument("a log line lacks \"" + std::string(key) + "\"");
    return *v;
  }
  double number() const {
    if (type != Type::Number) throw std::invalid_argument("a log field is not a number");
    return std::strtod(text.c_str(), nullptr);
  }
  std::uint64_t u64() const {
    if (type != Type::Number) throw std::invalid_argument("a log field is not a number");
    return std::strtoull(text.c_str(), nullptr, 10);
  }
  const std::string& str() const {
    if (type != Type::String) throw std::invalid_argument("a log field is not a string");
    return text;
  }
  bool boolean() const {
    if (type != Type::Bool) throw std::invalid_argument("a log field is not true or false");
    return b;
  }
  std::vector<std::string> strings() const {
    if (type != Type::Array) throw std::invalid_argument("a log field is not an array");
    std::vector<std::string> out;
    for (const Json& v : items) out.push_back(v.str());
    return out;
  }
};

class Parser {
 public:
  explicit Parser(std::string_view s) : s_(s) {}
  Json document() {
    Json v = value();
    space();
    if (i_ != s_.size()) fail("trailing text");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    throw std::invalid_argument(std::string("a log line is not JSON (") + what + " at " + std::to_string(i_) + ")");
  }
  void space() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\r' || s_[i_] == '\n')) ++i_;
  }
  char peek() {
    space();
    if (i_ >= s_.size()) fail("unexpected end");
    return s_[i_];
  }
  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++i_;
  }
  bool literal(std::string_view w) {
    if (s_.substr(i_, w.size()) != w) return false;
    i_ += w.size();
    return true;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) fail("a bad escape");
        const char e = s_[i_++];
        switch (e) {
          case '"': c = '"'; break;
          case '\\': c = '\\'; break;
          case '/': c = '/'; break;
          case 'b': c = '\b'; break;
          case 'f': c = '\f'; break;
          case 'n': c = '\n'; break;
          case 'r': c = '\r'; break;
          case 't': c = '\t'; break;
          case 'u': {
            if (i_ + 4 > s_.size()) fail("a bad escape");
            const unsigned long code = std::strtoul(std::string(s_.substr(i_, 4)).c_str(), nullptr, 16);
            i_ += 4;
            if (code < 0x80) {
              c = static_cast<char>(code);
            } else if (code < 0x800) {
              out += static_cast<char>(0xC0 | (code >> 6));
              c = static_cast<char>(0x80 | (code & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (code >> 12));
              out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
              c = static_cast<char>(0x80 | (code & 0x3F));
            }
            break;
          }
          default: fail("a bad escape");
        }
      }
      out += c;
    }
    if (i_ >= s_.size()) fail("an unterminated string");
    ++i_;
    return out;
  }
  Json value() {
    Json v;
    const char c = peek();
    if (c == '{') {
      ++i_;
      v.type = Json::Type::Object;
      if (peek() == '}') {
        ++i_;
        return v;
      }
      for (;;) {
        std::string k = string();
        expect(':');
        v.fields.emplace_back(std::move(k), value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      ++i_;
      v.type = Json::Type::Array;
      if (peek() == ']') {
        ++i_;
        return v;
      }
      for (;;) {
        v.items.push_back(value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.type = Json::Type::String;
      v.text = string();
      return v;
    }
    if (literal("true")) {
      v.type = Json::Type::Bool;
      v.b = true;
      return v;
    }
    if (literal("false")) {
      v.type = Json::Type::Bool;
      return v;
    }
    if (literal("null")) return v;
    const std::size_t start = i_;
    while (i_ < s_.size() && (std::strchr("+-0123456789.eE", s_[i_]) != nullptr)) ++i_;
    if (i_ == start) fail("a value expected");
    v.type = Json::Type::Number;
    v.text = std::string(s_.substr(start, i_ - start));
    return v;
  }

  std::string_view s_;
  std::size_t i_ = 0;
};

std::string number_text(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

std::string strings_json(const std::vector<std::string>& v) {
  std::string out = "[";
  for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + json_string(v[i]);
  return out + "]";
}

std::string step_json(const Step& s) {
  return std::string("{\"moves\":") + strings_json(s.moves) + ",\"submit\":" + (s.submit ? "true" : "false") + "}";
}

Step step_of(const Json& j) {
  Step s;
  s.moves = j.at("moves").strings();
  s.submit = j.at("submit").boolean();
  return s;
}

std::string head(const char* kind) {
  return std::string("{\"format\":") + json_string(kDecisionsFormat) + ",\"version\":" + std::to_string(kFormatVersion) +
         ",\"kind\":" + json_string(kind);
}

std::string event_fields(const Event& e) {
  std::string out = ",\"record\":" + json_string(e.record) + ",\"action\":" + std::to_string(e.action) +
                    ",\"decision\":" + std::to_string(e.decision) + ",\"category\":" +
                    json_string(category_name(e.category)) + ",\"prefix\":" + strings_json(e.prefix) +
                    ",\"step\":" + step_json(e.step) + ",\"from\":" + step_json(e.from) +
                    ",\"boards\":" + std::to_string(e.boards) + ",\"visits\":" + std::to_string(e.visits) +
                    ",\"q\":" + number_text(e.q) + ",\"from_visits\":" + std::to_string(e.from_visits) +
                    ",\"from_q\":" + number_text(e.from_q) + ",\"root_visits\":" + std::to_string(e.root_visits) +
                    ",\"root_q\":" + number_text(e.root_q) + ",\"peak_share\":" + number_text(e.peak_share);
  return out;
}

double number_or_zero(const Json& j) { return j.type == Json::Type::Null ? 0.0 : j.number(); }

Event event_of(const Json& j) {
  Event e;
  e.record = j.at("record").str();
  e.action = static_cast<std::uint32_t>(j.at("action").u64());
  e.decision = static_cast<std::uint32_t>(j.at("decision").u64());
  const std::optional<Category> c = category_of(j.at("category").str());
  if (!c) throw std::invalid_argument("a log event's category is not final, pv_change or demotion");
  e.category = *c;
  e.prefix = j.at("prefix").strings();
  e.step = step_of(j.at("step"));
  e.from = step_of(j.at("from"));
  e.boards = static_cast<int>(j.at("boards").u64());
  e.visits = static_cast<std::uint32_t>(j.at("visits").u64());
  e.q = number_or_zero(j.at("q"));
  e.from_visits = static_cast<std::uint32_t>(j.at("from_visits").u64());
  e.from_q = number_or_zero(j.at("from_q"));
  e.root_visits = static_cast<std::uint32_t>(j.at("root_visits").u64());
  e.root_q = number_or_zero(j.at("root_q"));
  e.peak_share = number_or_zero(j.at("peak_share"));
  return e;
}

void check_head(const Json& j, const char* format) {
  if (j.type != Json::Type::Object) throw std::invalid_argument("a log line is not a JSON object");
  if (j.at("format").str() != format) {
    throw std::invalid_argument("a log line of format \"" + j.at("format").str() + "\", not \"" + format + "\"");
  }
  if (j.at("version").u64() != static_cast<std::uint64_t>(kFormatVersion)) {
    throw std::invalid_argument(std::string("a log line of another version of ") + format);
  }
}

template <class F>
void for_each_line(std::string_view text, F&& f) {
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t end = text.find('\n', at);
    if (end == std::string_view::npos) end = text.size();
    std::string_view line = text.substr(at, end - at);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1);
    if (!line.empty()) f(line);
    at = end + 1;
  }
}

std::uint64_t mix(std::uint64_t z) noexcept {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::uint64_t hash_text(std::string_view s, std::uint64_t h) noexcept {
  for (const char c : s) h = mix(h ^ static_cast<unsigned char>(c)) + 0x9E3779B97F4A7C15ull;
  return h;
}

const char* check_text(Check c) noexcept { return tools::sixbpp::check_name(c); }

Check check_from(std::string_view name) {
  for (const Check c : {Check::None, Check::Confirmed, Check::Unconfirmed, Check::Refuted}) {
    if (name == tools::sixbpp::check_name(c)) return c;
  }
  throw std::invalid_argument("an audit result's check is not one of the counting's");
}

GameState after_action(const GameState& s, const std::vector<std::string>& action) {
  GameState t = s;
  for (const std::string& m : action) t.apply(move_of_text(t, m));
  if (t.submission_status() != SubmitStatus::Safe) {
    throw std::invalid_argument("an audited action cannot be submitted legally");
  }
  t.submit();
  return t;
}

}  // namespace

const char* category_name(Category c) noexcept {
  switch (c) {
    case Category::Final: return "final";
    case Category::PvChange: return "pv_change";
    case Category::Demotion: return "demotion";
  }
  return "?";
}

std::optional<Category> category_of(std::string_view name) noexcept {
  for (const Category c : {Category::Final, Category::PvChange, Category::Demotion}) {
    if (name == category_name(c)) return c;
  }
  return std::nullopt;
}

const GameLog* Log::game(std::string_view record) const {
  for (const GameLog& g : games) {
    if (g.record == record) return &g;
  }
  return nullptr;
}

std::string game_json(const GameLog& g) {
  std::string out = head("game") + ",\"record\":" + json_string(g.record) + ",\"variant\":" + json_string(g.variant) +
                    ",\"start\":" + strings_json(g.start) + ",\"width\":" + std::to_string(g.width) +
                    ",\"height\":" + std::to_string(g.height) + ",\"rules\":" + std::to_string(g.rules) +
                    ",\"split\":" + json_string(g.split) + ",\"actions\":[";
  for (std::size_t i = 0; i < g.actions.size(); ++i) out += (i ? "," : "") + strings_json(g.actions[i]);
  return out + "]}";
}

std::string event_json(const Event& e) { return head("event") + event_fields(e) + "}"; }

std::string job_json(const Job& j) {
  return head("job") + ",\"job\":" + std::to_string(j.id) + ",\"range\":" + std::to_string(j.range) +
         event_fields(j.event) + "}";
}

Log read_log(std::string_view text) {
  Log log;
  for_each_line(text, [&](std::string_view line) {
    const Json j = Parser(line).document();
    check_head(j, kDecisionsFormat);
    const std::string& kind = j.at("kind").str();
    if (kind == "game") {
      GameLog g;
      g.record = j.at("record").str();
      g.variant = j.at("variant").str();
      g.start = j.at("start").strings();
      g.width = static_cast<int>(j.at("width").u64());
      g.height = static_cast<int>(j.at("height").u64());
      g.rules = j.at("rules").u64();
      g.split = j.at("split").str();
      for (const Json& a : j.at("actions").items) g.actions.push_back(a.strings());
      if (log.game(g.record) == nullptr) log.games.push_back(std::move(g));
    } else if (kind == "event") {
      log.events.push_back(event_of(j));
    } else if (kind == "job") {
      Job job;
      job.id = j.at("job").u64();
      job.range = static_cast<int>(j.at("range").u64());
      job.event = event_of(j);
      log.jobs.push_back(std::move(job));
    } else {
      throw std::invalid_argument("a log line of kind \"" + kind + "\" (game, event or job)");
    }
  });
  for (const Event& e : log.events) {
    if (log.game(e.record) == nullptr) throw std::invalid_argument("an event's game \"" + e.record + "\" is not in the log");
  }
  for (const Job& job : log.jobs) {
    if (log.game(job.event.record) == nullptr) {
      throw std::invalid_argument("a job's game \"" + job.event.record + "\" is not in the log");
    }
  }
  return log;
}

std::string log_text(const Log& log) {
  std::string out;
  for (const GameLog& g : log.games) out += game_json(g) + "\n";
  for (const Event& e : log.events) out += event_json(e) + "\n";
  for (const Job& j : log.jobs) out += job_json(j) + "\n";
  return out;
}

Move move_of_text(const GameState& s, std::string_view text) {
  MoveList moves;
  generate_moves(s, moves);
  for (const Move& m : moves) {
    if (to_string(m, s.multiverse().even()) == text) return m;
  }
  throw std::invalid_argument("no move \"" + std::string(text) + "\" here");
}

GameState start_of(const GameLog& g) {
  const RuleOptions rules = RuleOptions::decode(g.rules);
  if (g.variant != "custom") {
    const Variant* v = find_variant(g.variant);
    if (v == nullptr) throw std::invalid_argument("no variant \"" + g.variant + "\"");
    return variant_start(*v, rules);
  }
  return game_from_fen(g.start, BoardSize{g.width, g.height}, Parity::Infer, rules);
}

GameState turn_start_of(const GameLog& g, std::uint32_t action) {
  if (action > g.actions.size()) throw std::invalid_argument("an event's action is past its game's");
  GameState s = start_of(g);
  for (std::uint32_t a = 0; a < action; ++a) s = after_action(s, g.actions[a]);
  return s;
}

Log log_of_game(const mcts::GameRecord& game, std::string_view variant, std::string_view record_id,
                const EventOptions& options) {
  if (game.decision_log.empty() && !game.actions.empty()) {
    throw std::invalid_argument("the game has no decision log (PlayParams::decision_log)");
  }
  Log log;
  GameLog g;
  g.record = std::string(record_id);
  g.variant = std::string(variant);
  const Multiverse& mv = game.start.multiverse();
  const bool even = mv.even();
  if (g.variant == "custom") {
    mv.for_each_board([&](int p, int h, const Board& board) { g.start.push_back(fen_board_string(board, p, h, even)); });
    g.width = mv.board_size().width;
    g.height = mv.board_size().height;
  }
  g.rules = game.start.rules().encode();
  g.split = game.split == mcts::TurnSplit::Flat ? "flat" : game.split == mcts::TurnSplit::Tree ? "tree" : "ply";
  GameState s = game.start;
  std::vector<GameState> starts;
  for (const std::vector<Move>& action : game.actions) {
    starts.push_back(s);
    std::vector<std::string> texts;
    for (const Move& m : action) {
      texts.push_back(to_string(m, even));
      s.apply(m);
    }
    s.submit();
    g.actions.push_back(std::move(texts));
  }
  const auto text_of = [even](const mcts::PathStep& p) {
    Step out;
    out.submit = p.submit;
    for (const Move& m : p.moves) out.moves.push_back(to_string(m, even));
    return out;
  };
  for (const mcts::DecisionLog& d : game.decision_log) {
    if (d.action >= game.actions.size()) continue;  // the game stopped during the action
    Event base;
    base.record = g.record;
    base.action = d.action;
    base.decision = d.decision;
    for (const Move& m : d.prefix) base.prefix.push_back(to_string(m, even));
    base.boards = static_cast<int>(d.boards);
    base.root_visits = d.root_visits;
    base.root_q = d.root_q;
    // A PV change: the new most visited child (and the old one).
    for (const mcts::PvChange& c : d.pv_changes) {
      if (c.from == mcts::PvChange::kNoChild && !options.first_pv) continue;
      Event e = base;
      e.category = Category::PvChange;
      e.step = text_of(c.to_step);
      e.visits = c.to_visits;
      e.q = c.to_q;
      e.from = text_of(c.from_step);
      e.from_visits = c.from_visits;
      e.from_q = c.from_q;
      log.events.push_back(std::move(e));
    }
    // Demotions: a child that led, or held a heavy share, and was not chosen.
    for (std::size_t i = 0; i < d.children.size(); ++i) {
      const mcts::DecisionLog::Child& c = d.children[i];
      if (i == d.chosen || !(c.was_pv || c.peak_share >= options.demotion_share)) continue;
      Event e = base;
      e.category = Category::Demotion;
      e.step = text_of(c.step);
      e.visits = c.visits;
      e.q = c.q;
      e.peak_share = c.peak_share;
      log.events.push_back(std::move(e));
    }
    // The final move: the whole action, at the decision that completed it.
    const bool completes = d.chosen < d.children.size() &&
                           (game.split == mcts::TurnSplit::Flat || d.children[d.chosen].step.submit);
    if (completes) {
      Event e = base;
      e.category = Category::Final;
      e.step.moves = g.actions[d.action];
      e.visits = d.children[d.chosen].visits;
      e.q = d.children[d.chosen].q;
      e.peak_share = d.children[d.chosen].peak_share;
      log.events.push_back(std::move(e));
    }
  }
  log.games.push_back(std::move(g));
  return log;
}

// ---- sampling ---------------------------------------------------------------------------

Log sample(const Log& events, const SampleOptions& o) {
  for (const double r : o.rate) {
    if (!(r >= 0 && r <= 1)) throw std::invalid_argument("sample: each rate in [0, 1]");
  }
  struct Candidate {
    std::uint64_t key;
    std::size_t index;
  };
  std::map<std::pair<int, int>, std::vector<Candidate>> strata;
  for (std::size_t i = 0; i < events.events.size(); ++i) {
    const Event& e = events.events[i];
    std::uint64_t h = mix(o.seed + 0x6175646974ull);
    h = hash_text(e.record, h);
    h = mix(h ^ (static_cast<std::uint64_t>(e.action) << 32) ^ e.decision);
    h = mix(h ^ static_cast<std::uint64_t>(e.category));
    for (const std::string& m : e.step.moves) h = hash_text(m, h);
    h = mix(h ^ (e.step.submit ? 1u : 0u));
    const double u = static_cast<double>(h >> 11) * 0x1.0p-53;
    if (!(u < o.rate[static_cast<std::size_t>(e.category)])) continue;
    strata[{static_cast<int>(e.category), tools::sixbpp::range_of(e.boards)}].push_back(Candidate{h, i});
  }
  std::vector<std::pair<std::size_t, std::uint64_t>> kept;
  for (auto& [stratum, cs] : strata) {
    std::stable_sort(cs.begin(), cs.end(), [](const Candidate& a, const Candidate& b) { return a.key < b.key; });
    if (cs.size() > o.cap) cs.resize(o.cap);
    for (const Candidate& c : cs) kept.emplace_back(c.index, c.key);
  }
  std::sort(kept.begin(), kept.end());
  Log out;
  for (const auto& [i, key] : kept) {
    Job j;
    j.id = key;
    j.event = events.events[i];
    j.range = tools::sixbpp::range_of(j.event.boards);
    if (out.game(j.event.record) == nullptr) {
      const GameLog* g = events.game(j.event.record);
      if (g == nullptr) throw std::invalid_argument("sample: an event's game is not in the log");
      out.games.push_back(*g);
    }
    out.jobs.push_back(std::move(j));
  }
  return out;
}

// ---- running ----------------------------------------------------------------------------

std::string contradiction_json(const Contradiction& c) {
  return std::string("{\"format\":") + json_string(kContradictionsFormat) +
         ",\"version\":" + std::to_string(kFormatVersion) + ",\"kind\":" + json_string(c.kind) +
         ",\"distance\":" + std::to_string(c.distance) + ",\"certificate\":" + json_string(c.certificate) +
         ",\"search_q\":" + number_text(c.search_q) + event_fields(c.event) + "}";
}

void JsonlCurriculumSink::emit(const Contradiction& c) {
  out_ << contradiction_json(c) << '\n';
  out_.flush();
  ++emitted_;
}

std::string result_json(const JobResult& r) {
  const GuardOutcome& g = r.guard;
  const MateOutcome& m = r.mate;
  std::string out = std::string("{\"format\":") + json_string(kResultsFormat) +
                    ",\"version\":" + std::to_string(kFormatVersion) + ",\"job\":" + std::to_string(r.job.id) +
                    ",\"range\":" + std::to_string(r.job.range) + ",\"guard\":{\"asked\":" +
                    (g.asked ? "true" : "false") + ",\"why\":" + json_string(g.why) +
                    ",\"completed_depth\":" + std::to_string(g.completed_depth) +
                    ",\"refuted\":" + (g.refuted ? "true" : "false") +
                    ",\"refutation_depth\":" + std::to_string(g.refutation_depth) + ",\"check\":" +
                    json_string(check_text(g.check)) + ",\"nodes\":" + std::to_string(g.nodes) +
                    ",\"seconds\":" + number_text(g.seconds) + "},\"mate\":{\"asked\":" + (m.asked ? "true" : "false") +
                    ",\"status\":" + json_string(proof_status_name(m.status)) +
                    ",\"distance\":" + std::to_string(m.distance) + ",\"first_action\":" + strings_json(m.first_action) +
                    ",\"check\":" + json_string(check_text(m.check)) + ",\"nodes\":" + std::to_string(m.nodes) +
                    ",\"seconds\":" + number_text(m.seconds) + "},\"contradictions\":" + strings_json(r.contradictions) +
                    event_fields(r.job.event) + "}";
  return out;
}

std::vector<JobResult> read_results(std::string_view text) {
  std::vector<JobResult> out;
  for_each_line(text, [&](std::string_view line) {
    const Json j = Parser(line).document();
    check_head(j, kResultsFormat);
    JobResult r;
    r.job.id = j.at("job").u64();
    r.job.range = static_cast<int>(j.at("range").u64());
    r.job.event = event_of(j);
    const Json& g = j.at("guard");
    r.guard.asked = g.at("asked").boolean();
    r.guard.why = g.at("why").str();
    r.guard.completed_depth = static_cast<int>(g.at("completed_depth").u64());
    r.guard.refuted = g.at("refuted").boolean();
    r.guard.refutation_depth = static_cast<int>(g.at("refutation_depth").u64());
    r.guard.check = check_from(g.at("check").str());
    r.guard.nodes = g.at("nodes").u64();
    r.guard.seconds = number_or_zero(g.at("seconds"));
    const Json& m = j.at("mate");
    r.mate.asked = m.at("asked").boolean();
    const std::string& st = m.at("status").str();
    r.mate.status = st == proof_status_name(ProofStatus::Proven)      ? ProofStatus::Proven
                    : st == proof_status_name(ProofStatus::Disproven) ? ProofStatus::Disproven
                                                                      : ProofStatus::Unknown;
    r.mate.distance = static_cast<int>(m.at("distance").u64());
    r.mate.first_action = m.at("first_action").strings();
    r.mate.check = check_from(m.at("check").str());
    r.mate.nodes = m.at("nodes").u64();
    r.mate.seconds = number_or_zero(m.at("seconds"));
    r.contradictions = j.at("contradictions").strings();
    out.push_back(std::move(r));
  });
  return out;
}

namespace {

// Guard's question offline on `action` from `turn_start` (the literal class,
// iterative deepening; Guard's own options), its refutation's certificate
// checked by E4 alone.
GuardOutcome ask_guard(const GameState& turn_start, const std::vector<std::string>& action, const RunOptions& o) {
  GuardOutcome out;
  std::vector<Move> moves;
  GameState t = turn_start;
  for (const std::string& m : action) {
    moves.push_back(move_of_text(t, m));
    t.apply(moves.back());
  }
  GuardOptions go;
  go.max_depth = o.guard_max_depth;
  go.threads = o.threads;
  go.prover.max_tree_nodes = o.tree_cap;
  go.prover.record_certificate = true;
  const GuardResult r = guard(turn_start, moves, ProveBudget::nodes(o.guard_nodes), go);
  out.asked = true;
  out.completed_depth = r.completed_depth;
  out.refuted = r.refuted;
  out.refutation_depth = r.refutation_depth;
  out.nodes = r.nodes;
  out.seconds = r.seconds;
  if (r.refuted) {
    if (!r.certificate) {
      out.check = Check::Unconfirmed;
    } else {
      GameState after = t;
      after.submit();
      out.check = tools::sixbpp::check_of(check_certificate(*r.certificate, after, 2 * r.refutation_depth - 1, o.check));
    }
  }
  return out;
}

// Our forced mate at `turn_start`: the truth recorder's method - the literal
// class and the default class, each with fresh stores and a node budget, the
// shorter proof's certificate checked by E4 alone.
MateOutcome ask_mate(const GameState& turn_start, const RunOptions& o) {
  MateOutcome out;
  out.asked = true;
  const auto t0 = std::chrono::steady_clock::now();
  const auto options = [&](bool literal) {
    ProverOptions p;
    p.depth = o.mate_depth;
    p.threads = o.threads;
    p.max_tree_nodes = o.tree_cap;
    p.record_certificate = true;
    if (literal) {
      p.seeds = SeedClass::All;
      p.completion = CompletionClass::Every;
    }
    return p;
  };
  Prover lit(options(true));
  const ProveResult a = lit.prove(turn_start, ProveBudget::nodes(o.mate_nodes));
  Prover def(options(false));
  const ProveResult b = def.prove(turn_start, ProveBudget::nodes(o.mate_nodes));
  out.nodes = a.stats.nodes + b.stats.nodes;
  const ProveResult* m = nullptr;
  if (a.status == ProofStatus::Proven) m = &a;
  if (b.status == ProofStatus::Proven && (m == nullptr || b.proof.distance < m->proof.distance)) m = &b;
  if (m != nullptr) {
    out.status = ProofStatus::Proven;
    out.distance = m->proof.distance;
    for (const Move& mv : m->first_action) out.first_action.push_back(to_string(mv, turn_start.multiverse().even()));
    out.check = m->certificate ? tools::sixbpp::check_of(check_certificate(*m->certificate, turn_start,
                                                                           m->proof.distance, o.check))
                               : Check::Unconfirmed;
  } else if (a.status == ProofStatus::Disproven) {
    out.status = ProofStatus::Disproven;  // the literal class exhausted: no mate within the depth
  }
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

}  // namespace

std::vector<JobResult> run(const Log& jobs, const RunOptions& o, CurriculumSink* sink) {
  std::vector<JobResult> out;
  std::map<std::pair<std::string, std::uint32_t>, MateOutcome> mates;  // one per turn start
  for (const Job& job : jobs.jobs) {
    const Event& e = job.event;
    const GameLog* g = jobs.game(e.record);
    if (g == nullptr) throw std::invalid_argument("run: a job's game is not in the log");
    const GameState ts = turn_start_of(*g, e.action);
    JobResult r;
    r.job = job;
    // The action asked about: whole for a final move; for a PV change or a
    // demotion, whole only where its step completes it (a flat-form turn, or a
    // submission); a move of the ply split is not an action (issue 192).
    std::optional<std::vector<std::string>> action;
    if (e.category == Category::Final || g->split == "flat") {
      action = e.step.moves;
    } else if (e.step.submit) {
      action = e.prefix;
    }
    if (action && e.category != Category::Demotion) {
      r.guard = ask_guard(ts, *action, o);
    } else {
      r.guard.why = action ? "a demoted action: the search left it" : "a move of the ply split, not an action";
    }
    const auto key = std::make_pair(e.record, e.action);
    auto it = mates.find(key);
    if (it == mates.end()) it = mates.emplace(key, ask_mate(ts, o)).first;
    r.mate = it->second;
    const auto contradiction = [&](const char* kind, int distance) {
      r.contradictions.emplace_back(kind);
      if (sink != nullptr) {
        Contradiction c;
        c.kind = kind;
        c.event = e;
        c.distance = distance;
        c.certificate = check_text(Check::Confirmed);
        c.search_q = e.q;
        sink->emit(c);
      }
    };
    if (r.guard.refuted && r.guard.check == Check::Confirmed) {
      contradiction("allowed_mate", 2 * r.guard.refutation_depth - 1);
    }
    if (r.mate.status == ProofStatus::Proven && r.mate.check == Check::Confirmed) {
      if (e.category == Category::Final && after_action(ts, e.step.moves) != after_action(ts, r.mate.first_action)) {
        contradiction("missed_mate", r.mate.distance);
      }
      if (e.category == Category::Demotion) {
        bool begins = false;
        if (g->split == "flat" || e.step.submit) {
          begins = action && after_action(ts, *action) == after_action(ts, r.mate.first_action);
        } else {
          // The demoted move and the moves before it all in the mate's first action.
          std::vector<std::string> made = e.prefix;
          made.insert(made.end(), e.step.moves.begin(), e.step.moves.end());
          begins = std::all_of(made.begin(), made.end(), [&](const std::string& m) {
            return std::find(r.mate.first_action.begin(), r.mate.first_action.end(), m) != r.mate.first_action.end();
          });
        }
        if (begins) contradiction("abandoned_mate", r.mate.distance);
      }
    }
    out.push_back(std::move(r));
  }
  return out;
}

// ---- the report -------------------------------------------------------------------------

Report make_report(const std::vector<JobResult>& results, int shallow_depth) {
  if (shallow_depth < 1) throw std::invalid_argument("report: the shallow depth is 1 or more");
  Report rep;
  rep.shallow_depth = shallow_depth;
  rep.jobs = results.size();
  for (const JobResult& r : results) {
    const int range = std::clamp(r.job.range, 0, tools::sixbpp::kRanges - 1);
    RangeReport& rr = rep.ranges[static_cast<std::size_t>(range)];
    for (const std::string& k : r.contradictions) {
      if (k == "allowed_mate") ++rr.allowed_mates;
      if (k == "missed_mate") ++rr.missed_mates;
      if (k == "abandoned_mate") ++rr.abandoned_mates;
    }
    if (r.job.event.category != Category::Final) continue;
    ++rr.finals;
    if (!r.guard.asked) continue;
    ++rr.guard_asked;
    ++rr.completed[r.guard.completed_depth];
    // 6b''s counting: confirmed only; a refuted certificate withholds the verdict.
    if (r.guard.refuted && r.guard.check == Check::Refuted) {
      ++rr.withheld;
      continue;
    }
    if (r.guard.refuted && r.guard.check == Check::Unconfirmed) {
      ++rr.unconfirmed;
      continue;
    }
    const bool shallow_mate = r.guard.refuted && r.guard.check == Check::Confirmed &&
                              r.guard.refutation_depth <= shallow_depth;
    if (shallow_mate) {
      ++rr.decided;
      ++rr.shallow;
    } else if (r.guard.completed_depth >= shallow_depth) {
      ++rr.decided;
    } else {
      ++rr.undecided;
    }
  }
  for (RangeReport& rr : rep.ranges) rr.too_few = rr.finals < tools::sixbpp::kFewTruthMates;
  return rep;
}

std::string report_text(const Report& r) {
  std::string out = "The offline audit lane: " + std::to_string(r.jobs) + " jobs; shallow = Guard's depth <= " +
                    std::to_string(r.shallow_depth) + " (the opponent's actions)\n";
  char buf[512];
  for (int k = 0; k < tools::sixbpp::kRanges; ++k) {
    const RangeReport& rr = r.ranges[static_cast<std::size_t>(k)];
    std::snprintf(buf, sizeof buf,
                  "  %-12s finals %zu, Guard asked %zu, decided %zu, shallow mates %zu (rate %.3f), undecided %zu, "
                  "withheld %zu, unconfirmed %zu; contradictions: allowed %zu, missed %zu, abandoned %zu%s%s\n",
                  tools::sixbpp::range_name(k), rr.finals, rr.guard_asked, rr.decided, rr.shallow, rr.shallow_rate(),
                  rr.undecided, rr.withheld, rr.unconfirmed, rr.allowed_mates, rr.missed_mates, rr.abandoned_mates,
                  rr.too_few ? " - too few to read" : "", k == tools::sixbpp::kRanges - 1 ? " (weak at 200+, 11.4)" : "");
    out += buf;
    if (!rr.completed.empty()) {
      out += "               Guard's completed depth:";
      for (const auto& [d, n] : rr.completed) out += " " + std::to_string(d) + ":" + std::to_string(n);
      out += "\n";
    }
  }
  return out;
}

std::string report_json(const Report& r) {
  std::string out = "{\"format\":\"t5d-audit-report\",\"version\":" + std::to_string(kFormatVersion) +
                    ",\"jobs\":" + std::to_string(r.jobs) + ",\"shallow_depth\":" + std::to_string(r.shallow_depth) +
                    ",\"ranges\":[";
  for (int k = 0; k < tools::sixbpp::kRanges; ++k) {
    const RangeReport& rr = r.ranges[static_cast<std::size_t>(k)];
    out += std::string(k ? "," : "") + "{\"range\":" + json_string(tools::sixbpp::range_name(k)) +
           ",\"finals\":" + std::to_string(rr.finals) + ",\"guard_asked\":" + std::to_string(rr.guard_asked) +
           ",\"decided\":" + std::to_string(rr.decided) + ",\"shallow\":" + std::to_string(rr.shallow) +
           ",\"shallow_rate\":" + number_text(rr.shallow_rate()) + ",\"undecided\":" + std::to_string(rr.undecided) +
           ",\"withheld\":" + std::to_string(rr.withheld) + ",\"unconfirmed\":" + std::to_string(rr.unconfirmed) +
           ",\"allowed_mates\":" + std::to_string(rr.allowed_mates) +
           ",\"missed_mates\":" + std::to_string(rr.missed_mates) +
           ",\"abandoned_mates\":" + std::to_string(rr.abandoned_mates) +
           ",\"too_few\":" + (rr.too_few ? "true" : "false") + ",\"completed\":{";
    bool first = true;
    for (const auto& [d, n] : rr.completed) {
      out += std::string(first ? "" : ",") + "\"" + std::to_string(d) + "\":" + std::to_string(n);
      first = false;
    }
    out += "}}";
  }
  return out + "]}";
}

}  // namespace t5d::audit
