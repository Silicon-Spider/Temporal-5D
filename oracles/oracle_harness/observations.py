"""What a rules implementation says about one game record, and how two such
reports are compared.

The gate compares three observables at every position of a record's main line
(position k is the state after k submitted actions):

  in_check       check status (decision H36; the rulebook's R1): passing on all
                 active boards in the present, could the opponent capture a royal
                 from any of their playable boards? Optional boards ahead of the
                 present and inactive timelines are not passed, and a capture
                 counts even if it would branch into an inactive timeline. This is
                 the check that separates checkmate from stalemate.
  legal_actions  number of legal actions (whole turns), counted up to a cap. An
                 action is the set of its moves with its timeline-creating moves
                 in creation order: two orders of two branching moves are two
                 actions, even where they reach the same state (rulebook J6). It
                 is what ftxi and cwmtt count - the protocol's definition, chosen
                 so that counts compare, not a rules ruling, and not the number
                 of distinct successor states
  terminal       "none" while a legal action exists, else "checkmate" or "stalemate" -
                 a proof, from a search that terminated: an exhausted budget is
                 unknown (a null), never a verdict (decision H37)

plus legality itself: whether each recorded action is accepted at all.

Each oracle answers with its own reading of these definitions - which boards its
check test passes, for one. Where the readings differ the reports disagree, and
the rulebook (docs/rules/README.md, H38), not the harness, says who is right.

Before any of that, the two reports must be about the same position. Each source
says what it set up at every position (its state event, fingerprint.py), and the
harness compares those first (decision H30): where two sources hold different
boards - different pieces, or the same pieces on boards at other coordinates -
they read the record's moves as different positions. That is a parse divergence,
"placement", and none of their rules answers from that position on is compared,
because they would be answers about different positions. Where they hold the
same boards and pieces but disagree about which pieces have not moved, that is
the other parse divergence, "unmoved": one is reported where each run of such
positions begins, and it suppresses no comparison, so a rules disagreement it
causes shows beside it. A placement divergence is settled against the notation
specification; a marker-only one against the notation's "5DFEN Strictness" or
the rulebook (what having moved means); a rules divergence against the rulebook.

Every field may be None: an oracle that cannot answer a question, or ran out of
time, says nothing rather than something. Nothing is ever inferred on an
oracle's behalf, and an unknown never disagrees with anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fingerprint import is_hash

SIDES = ("white", "black")
TERMINALS = ("none", "checkmate", "stalemate", "no_action")
STATUSES = ("ok", "rejected", "unsupported", "error", "timeout")
# The three things the gate exists to cross-check (design spec 11.0): move
# counts, check verdicts, and mate/stalemate verdicts.
OBSERVABLES = ("in_check", "legal_actions", "terminal")
# What a source answers at a position: the side to move, and the three above.
ANSWERS = ("side", *OBSERVABLES)
# What a source says it set up at a position (its state event): the full-state
# hash with unmoved markers, the same without them, and the board and piece counts.
FINGERPRINT_FIELDS = ("fingerprint", "placement", "boards", "pieces")
# The two parse divergences (decision H30), and the field each is read from.
PARSE_KINDS = ("placement", "unmoved")
PARSE_FIELD = {"placement": "placement", "unmoved": "fingerprint"}
# How a source replays the record, as opposed to what it says about a position.
REPLAY_KINDS = ("legality", "length")
# The columns of a golden row, in order. A column is added by appending it here
# (and raising the golden schema); everything that reads or writes rows follows this.
ROW_FIELDS = (*ANSWERS, *FINGERPRINT_FIELDS)

# Why a source stopped replaying a record. The soft reasons cannot tell a rules
# disagreement from a notation the source does not read: both surface as "no
# move I can generate matches this text". The first four are an oracle's.
ORACLE_SOFT_REJECTIONS = ("no_matching_move", "ambiguous_move", "notation_mismatch", "unrecognised")
# "unreadable" is the engine's alone: its own parser could not read the move
# text - or, refusing the record before its first position, the record's headers
# or board strings (a set-up refusal: refused_setup()).
UNREADABLE = "unreadable"
SOFT_REJECTIONS = (*ORACLE_SOFT_REJECTIONS, UNREADABLE)
HARD_REJECTIONS = ("illegal_move", "cannot_submit")
# The only reason a run may give for refusing a record's set-up - ending
# "rejected" before it reports even its starting position.
SETUP_REFUSALS = (UNREADABLE,)
# Why a run that was cut short stopped: its time ran out ("timeout"), or the gate
# stopped it once everything its tier requires had arrived ("stopped", H32).
CUT_REASONS = ("timeout", "stopped")


@dataclass(frozen=True)
class Position:
    index: int
    side: str | None = None
    in_check: bool | None = None
    legal_actions: int | None = None
    terminal: str | None = None
    # What the source says it set up here (fingerprint.py): the full-state hash
    # with unmoved markers - None from a source that does not track them - the
    # same without them, and how many boards and pieces the multiverse holds.
    fingerprint: str | None = None
    placement: str | None = None
    boards: int | None = None
    pieces: int | None = None

    def known(self) -> int:
        """How many of the answers this position has. What the source set up is not an answer."""
        return sum(getattr(self, name) is not None for name in ANSWERS)


@dataclass
class Observation:
    source: str
    record: str
    status: str
    positions: list[Position] = field(default_factory=list)
    actions_accepted: int | None = None
    reason: str = ""
    detail: str = ""
    last_action: bool | None = None  # a rejected action was the record's final one
    cap: int | None = None
    elapsed_s: float = 0.0
    # Fields a source printed in a position event beside the protocol's - a
    # driver's timing ("ms"), the limit a question it left unanswered ran into
    # ("limit") - by position index. Kept for budgeting; never compared, never
    # recorded in a golden file.
    extras: dict[int, dict] = field(default_factory=dict)
    # When what this run says about each position had arrived: seconds from the
    # start of the run to the line that completed it - its position event, or,
    # for the position a run cut short was searching, its state event. None
    # where it was not measured. The cost of an answer (H32): never compared,
    # never recorded in a golden file; golden --write keeps it in the cost files.
    arrival_s: list[float | None] = field(default_factory=list)
    # When the run's outcome arrived (its end event, or the end of the process), if it did.
    ended_s: float | None = None
    # Diagnostic events ("explain", "actions") a source printed because
    # `observe` asked for them, in order. Never compared, never recorded.
    diagnostics: list[dict] = field(default_factory=list)

    def known(self) -> int:
        return sum(position.known() for position in self.positions)


@dataclass(frozen=True)
class Disagreement:
    record: str
    a: str
    b: str
    kind: str  # side | in_check | legal_actions | terminal | legality | length | placement | unmoved
    index: int
    a_value: str
    b_value: str
    soft: bool = False

    def key(self) -> tuple:
        return (self.record, self.a, self.b, self.kind, self.index)

    def describe(self) -> str:
        where = f"action {self.index}" if self.kind in REPLAY_KINDS else f"position {self.index}"
        if self.kind == "legality" and self.index == 0:
            where = "action 0 (the record's set-up)"
        note = " (soft: may be notation, not rules)" if self.soft else ""
        if self.kind == "placement":
            note = (" (parse divergence: different boards - their pieces or their coordinates; nothing from here on "
                    "is compared)")
        elif self.kind == "unmoved":
            note = (" (parse divergence: the same boards and pieces, marked unmoved differently from here until "
                    "the markers agree again)")
        return f"{self.record} {where}: {self.kind}: {self.a}={self.a_value} vs {self.b}={self.b_value}{note}"


def validate(observation: Observation) -> list[str]:
    """Internal contradictions in one report. A source that contradicts itself is broken."""
    problems: list[str] = []
    positions = observation.positions
    if observation.status not in STATUSES:
        problems.append(f"unknown status '{observation.status}'")
    for expected, position in enumerate(positions):
        where = f"position {position.index}"
        if position.index != expected:
            problems.append(f"positions are not numbered 0..n in order (saw {position.index} at slot {expected})")
            break
        if position.side is not None and position.side not in SIDES:
            problems.append(f"{where}: side '{position.side}'")
        if position.terminal is not None and position.terminal not in TERMINALS:
            problems.append(f"{where}: terminal '{position.terminal}'")
        count = position.legal_actions
        if count is not None:
            if count < 0 or (observation.cap is not None and count > observation.cap):
                problems.append(f"{where}: legal_actions {count} outside 0..cap")
            if position.terminal is not None and (count == 0) != (position.terminal != "none"):
                problems.append(f"{where}: legal_actions {count} contradicts terminal '{position.terminal}'")
        if position.in_check is not None:
            if position.terminal == "checkmate" and not position.in_check:
                problems.append(f"{where}: checkmate without check")
            if position.terminal == "stalemate" and position.in_check:
                problems.append(f"{where}: stalemate while in check")
        dead_end = count == 0 or position.terminal not in (None, "none")
        if dead_end and expected + 1 < len(positions):
            problems.append(f"{where}: no legal action exists, yet action {expected + 1} was accepted")
        for name in ("fingerprint", "placement"):
            value = getattr(position, name)
            if value is not None and not is_hash(value):
                problems.append(f"{where}: {name} {value!r} is not 32 lower-case hex digits")
        for name in ("boards", "pieces"):
            value = getattr(position, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                problems.append(f"{where}: {name} {value!r} is not a whole number")
        if position.fingerprint is not None and position.placement is None:
            problems.append(f"{where}: a fingerprint without the placement it includes")
        if position.placement is not None and None in (position.boards, position.pieces):
            problems.append(f"{where}: a placement without its board and piece counts (one state event gives all three)")
        if expected and position.placement is not None and positions[expected - 1].placement is None:
            problems.append(f"{where} says what was set up there, and position {expected - 1} does not: a source says it "
                            "at every position from the start or at none (only the last ones may be missing, where "
                            "it stopped saying it)")
    marked = {position.fingerprint is not None for position in positions if position.placement is not None}
    if len(marked) > 1:
        problems.append("the unmoved markers are given at some positions and not at others: a source tracks them "
                        "throughout a run, or not at all")
    # Position k exists exactly when k actions were accepted, whatever ended the run.
    if observation.actions_accepted != max(len(positions) - 1, 0):
        problems.append(f"actions_accepted is {observation.actions_accepted}, but {len(positions)} position(s) were reported")
    if observation.status in ("ok", "rejected") and not positions and not refused_setup(observation):
        problems.append(f"status {observation.status} without even the starting position")
    return problems


def refused_setup(observation: Observation) -> bool:
    """Did the run refuse the record's set-up - end "rejected", as unreadable, before even its starting position?

    Only the engine may (adapters/engine.py), in either mode: its parser could
    not read the record's headers or board strings, so there is no position 0.
    That is refusing "action 0", the set-up, and it is compared like any refusal.
    """
    return observation.status == "rejected" and not observation.positions and observation.reason in SETUP_REFUSALS


def refused_action(observation: Observation) -> int | None:
    """The action a run refused - counted from 1, and 0 for the record's set-up - or None when it refused none."""
    if observation.status != "rejected" or observation.actions_accepted is None:
        return None
    return 0 if refused_setup(observation) else observation.actions_accepted + 1


def _counts_compatible(a: int, a_cap: int | None, b: int, b_cap: int | None) -> bool:
    """A count equal to its cap means "at least this many"."""
    a_saturated = a_cap is not None and a >= a_cap
    b_saturated = b_cap is not None and b >= b_cap
    if a_saturated and b_saturated:
        return True
    if a_saturated:
        return b >= a_cap
    if b_saturated:
        return a >= b_cap
    return a == b


def _terminals_compatible(a: str, b: str) -> bool:
    if (a == "none") != (b == "none"):
        return False
    # "no_action" is a source that knows nothing is playable but not why.
    return a == b or "no_action" in (a, b)


def _count_text(count: int, cap: int | None) -> str:
    return f">={count}" if cap is not None and count >= cap else str(count)


def _actions_seen(observation: Observation) -> int:
    """How many of the record's actions a source is known to have read.

    A source that refused an action read that action too; one that refused the
    record's set-up read none.
    """
    if refused_setup(observation):
        return 0
    accepted = max(len(observation.positions) - 1, 0)
    return accepted + 1 if observation.status == "rejected" else accepted


def _length_clash(a: Observation, b: Observation) -> tuple[Observation, Observation] | None:
    """(the report that says the record ends, the one that found another action), where two reports disagree so."""
    for first, second in ((a, b), (b, a)):
        if first.status == "ok" and _actions_seen(second) > _actions_seen(first):
            return first, second
    return None


def _length_text(observation: Observation) -> str:
    """How many actions a source found in the record: exactly, if it read it to the end."""
    seen = _actions_seen(observation)
    return f"{seen} actions" if observation.status == "ok" else f"at least {seen} actions"


def decisive(name: str, value: object) -> bool:
    """Does this answer settle the question it answers?

    An unknown does not, and neither does "no_action" as a terminal verdict: it
    says nothing is playable without saying whether that is checkmate or stalemate.
    """
    return value is not None and not (name == "terminal" and value == "no_action")


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def counts_text(position: Position) -> str:
    """"2 boards, 63 pieces" where the position's state is known; "" where it is not."""
    if position.boards is None or position.pieces is None:
        return ""
    return f"{_plural(position.boards, 'board')}, {_plural(position.pieces, 'piece')}"


def _state_text(position: Position, name: str) -> str:
    """A hash as compare() prints it: with the board and piece counts it was computed from."""
    counts = counts_text(position)
    return f"{getattr(position, name)} ({counts})" if counts else str(getattr(position, name))


def judges_actions(source: str) -> bool:
    """Is this source's playing an action a statement that the action is legal?

    A parse-only source (adapters.PARSE_ONLY_SOURCES) applies any move it can
    place, legal or not: that it played on says only that it read the move. So
    it contradicts a refusal only where the refusal was a failure to read the
    move text (a soft one), never where another source held the action illegal.
    """
    from .adapters import PARSE_ONLY_SOURCES, oracle_of  # the adapters import this module: not at the top

    return oracle_of(source) not in PARSE_ONLY_SOURCES


def first_placement_divergence(a: Observation, b: Observation) -> int | None:
    """The first position at which both reports say what they set up, and say different things.

    From there on the two are answering about different positions, so nothing
    they say is compared; see compare().
    """
    for pa, pb in zip(a.positions, b.positions, strict=False):
        if pa.placement is not None and pb.placement is not None and pa.placement != pb.placement:
            return pa.index
    return None


def aligned(pa: Position, pb: Position) -> bool | None:
    """Do two reports' positions hold the same boards and pieces? None when one says nothing about it.

    That is what shows the two read the record's moves alike, so that counting
    actions lined them up. The unmoved markers are not compared: see same_state().
    """
    if pa.placement is None or pb.placement is None:
        return None
    return pa.placement == pb.placement


def same_state(pa: Position, pb: Position) -> bool | None:
    """Do two positions hold the same state, unmoved markers included?

    None where either has no fingerprint: it said nothing of what it set up, or
    it does not track which pieces have not moved.
    """
    if pa.fingerprint is None or pb.fingerprint is None:
        return None
    return pa.fingerprint == pb.fingerprint


def value_at(observation: Observation, kind: str, index: int) -> str | None:
    """What a report says about one observation, in the words compare() prints; None if it says nothing.

    For legality and length, ``index`` is an action number, counted from 1; for
    legality, action 0 is the record's set-up (refused_setup()).
    """
    if kind in PARSE_KINDS:
        if index < 0 or index >= len(observation.positions):
            return None
        position = observation.positions[index]
        return None if getattr(position, PARSE_FIELD[kind]) is None else _state_text(position, PARSE_FIELD[kind])
    if kind == "legality":
        if index < 0:
            return None
        if refused_action(observation) == index:
            return f"rejected:{observation.reason}"
        return "accepted" if len(observation.positions) > index else None
    if kind == "length":
        # A length disagreement sits at the first action one of the two did not
        # find: at most one past the actions this report read.
        seen = _actions_seen(observation)
        limit = seen + 1 if observation.status == "ok" else seen
        return _length_text(observation) if observation.positions and 1 <= index <= limit else None
    if index < 0 or index >= len(observation.positions):
        return None
    value = getattr(observation.positions[index], kind, None)
    if value is None:
        return None
    return _count_text(value, observation.cap) if kind == "legal_actions" else str(value)


def compare(a: Observation, b: Observation, *, replay: bool = True) -> list[Disagreement]:
    """Every point at which two reports on the same record contradict each other.

    What the two set up is compared first (decision H30). At the first position
    where their placements differ there is one "placement" disagreement, and
    nothing from that position on is compared - no observable at it or after
    it, no legality or length of an action after it - because the two are no
    longer answering about the same position. Before that, positions where the
    boards and pieces agree and the unmoved markers do not come in runs: each
    run - from a position whose markers differ, after one whose markers agreed
    or at the first whose markers both know, to the next whose markers agree -
    is one "unmoved" disagreement, at its first position, and it suppresses
    nothing. (So one known-divergence entry excuses one run, and markers that
    part again after agreeing are a new disagreement.) Both kinds are compared
    between any two reports, two searches of one implementation included: they
    must parse alike.

    ``replay=False`` leaves out legality and length. Those describe how an
    implementation replays the record, which does not depend on which of its
    searches counted the actions - so they are compared once per implementation,
    not once per search.
    """
    if a.record != b.record:
        raise ValueError("compare() needs two observations of the same record")
    found: list[Disagreement] = []

    def add(kind: str, index: int, a_value: object, b_value: object, soft: bool = False) -> None:
        found.append(Disagreement(a.record, a.source, b.source, kind, index, str(a_value), str(b_value), soft))

    diverged = first_placement_divergence(a, b)
    # Compared up to the shorter report - what one source did not reach is
    # unknown - and never past the point where the two parted.
    shared = list(zip(a.positions, b.positions, strict=False))[:diverged]
    differing = False  # inside a run of positions whose markers differ
    for pa, pb in shared:
        same = same_state(pa, pb)  # before the parting, the boards and pieces agree wherever both are known
        if same is None:
            continue  # a position whose markers one side does not give neither starts nor ends a run
        if not same and not differing:
            add("unmoved", pa.index, _state_text(pa, "fingerprint"), _state_text(pb, "fingerprint"))
        differing = not same
    if diverged is not None:
        add("placement", diverged, _state_text(a.positions[diverged], "placement"),
            _state_text(b.positions[diverged], "placement"))

    for pa, pb in shared:
        if pa.side is not None and pb.side is not None and pa.side != pb.side:
            add("side", pa.index, pa.side, pb.side)
        if pa.in_check is not None and pb.in_check is not None and pa.in_check != pb.in_check:
            add("in_check", pa.index, pa.in_check, pb.in_check)
        if (pa.legal_actions is not None and pb.legal_actions is not None
                and not _counts_compatible(pa.legal_actions, a.cap, pb.legal_actions, b.cap)):
            add("legal_actions", pa.index, _count_text(pa.legal_actions, a.cap), _count_text(pb.legal_actions, b.cap))
        if pa.terminal is not None and pb.terminal is not None and not _terminals_compatible(pa.terminal, pb.terminal):
            add("terminal", pa.index, pa.terminal, pb.terminal)
    if not replay:
        return found

    def parted_before(action: int) -> bool:
        """Is this action played from a position the two no longer share?"""
        return diverged is not None and action > diverged

    # Legality: one source refused an action the other went on to play - where
    # the other's playing it is a verdict (see judges_actions). A refusal of the
    # record's set-up (action 0) is contradicted by any source that set it up.
    for first, second, flipped in ((a, b, False), (b, a, True)):
        refused = refused_action(first)
        if refused is None:
            continue
        soft = first.reason in SOFT_REJECTIONS
        if len(second.positions) > refused and not parted_before(refused) and (soft or judges_actions(second.source)):
            values = (f"rejected:{first.reason}", "accepted")
            add("legality", refused, *(values[::-1] if flipped else values), soft=soft)

    # Length: one source says the record ends where the other found another
    # action - one it played, or one it refused. (A source that silently stops
    # reading early would otherwise look like one that agrees about everything
    # it saw, and would never have to say whether the next action is legal.)
    clash = _length_clash(a, b)
    if clash is not None and not parted_before(_actions_seen(clash[0]) + 1):
        first, second = clash
        values = (_length_text(first), _length_text(second))
        soft = second.status == "rejected" and second.reason in SOFT_REJECTIONS
        add("length", _actions_seen(first) + 1, *(values if first is a else values[::-1]), soft=soft)
    return found


def agrees(a: Observation, b: Observation, kind: str, index: int) -> bool | None:
    """Do two reports agree about one observation, by compare()'s own rule? None where either says nothing of it.

    Agreeing is not saying the same words: a count at the cap agrees with any
    count at least as large, "no_action" with checkmate and with stalemate, two
    refusals of one action whatever reason each gives, and two reports about
    the length of a record wherever neither says it ends before an action the
    other found. ``kind`` and ``index`` are as in value_at().
    """
    if kind in REPLAY_KINDS:
        if value_at(a, kind, index) is None or value_at(b, kind, index) is None:
            return None
        if kind == "legality":
            return (value_at(a, kind, index) == "accepted") == (value_at(b, kind, index) == "accepted")
        return _length_clash(a, b) is None
    if not (0 <= index < len(a.positions) and 0 <= index < len(b.positions)):
        return None
    name = PARSE_FIELD.get(kind, kind)
    va, vb = getattr(a.positions[index], name, None), getattr(b.positions[index], name, None)
    if va is None or vb is None:
        return None
    if kind == "legal_actions":
        return _counts_compatible(va, a.cap, vb, b.cap)
    if kind == "terminal":
        return _terminals_compatible(va, vb)
    return va == vb


def _printed_count(text: str) -> tuple[int, int | None] | None:
    """A count as the gate prints it - "20", or ">=1000" at the cap - as (count, the cap it stopped at, if it did)."""
    capped = text.startswith(">=")
    digits = text[2:] if capped else text
    if not digits.isdigit():
        return None
    return int(digits), int(digits) if capped else None


def texts_agree(kind: str, a: str, b: str) -> bool:
    """agrees(), for two values as the gate prints them (value_at), where the reports are not at hand."""
    if kind == "legality":
        return (a == "accepted") == (b == "accepted")
    if kind == "terminal" and a in TERMINALS and b in TERMINALS:
        return _terminals_compatible(a, b)
    if kind == "legal_actions":
        first, second = _printed_count(a), _printed_count(b)
        if first is not None and second is not None:
            return _counts_compatible(first[0], first[1], second[0], second[1])
    return a == b


def cut_short(observation: Observation) -> bool:
    """Did the run stop because it ran out of time - or because the gate stopped it, everything it needed in hand?

    How far a slow run gets depends on the machine, so it is never pinned. A
    process that died is not in this class: that is an outcome like any other,
    and a change in it is reported.
    """
    return observation.status == "timeout"


def status_changed(golden: Observation, fresh: Observation) -> str | None:
    """An outcome that no longer matches its recorded value."""
    def outcome(report: Observation) -> str:
        return f"{report.status}{'/' + report.reason if report.reason else ''} after {report.actions_accepted} actions"

    if cut_short(fresh):
        return None
    if cut_short(golden):
        # The recorded run got this far before its time ran out; a run that
        # finished cannot have stopped earlier - and a run that failed has
        # failed, however far it got first.
        if fresh.status == "error":
            return f"the recorded run timed out, now {outcome(fresh)}"
        if len(fresh.positions) < len(golden.positions):
            return f"the recorded run reached position {len(golden.positions) - 1} before timing out, now {outcome(fresh)}"
        return None
    before = (golden.status, golden.reason, golden.actions_accepted)
    after = (fresh.status, fresh.reason, fresh.actions_accepted)
    return f"was {outcome(golden)}, now {outcome(fresh)}" if before != after else None


def values(observation: Observation) -> list[tuple[int, str]]:
    """(position index, field) of every value a report holds: every answer, and everything it said it set up.

    These are what a golden file records, and what the gate's required sets are
    made of (H32): a source that stops exporting its state loses values like one
    that stops answering.
    """
    return [(position.index, name) for position in observation.positions for name in ROW_FIELDS
            if getattr(position, name) is not None]


def value(observation: Observation, index: int, name: str) -> object:
    """One field of one position of a report; None where the report says nothing about it."""
    return getattr(observation.positions[index], name) if 0 <= index < len(observation.positions) else None


# --- serialisation (golden files and run reports) ---------------------------

_SIDE_LETTERS = {"w": "white", "b": "black", None: None}


def to_row(position: Position) -> list:
    """One position as a golden row: the values of ROW_FIELDS, the side as "w" or "b"."""
    return [(position.side[0] if position.side else None) if name == "side" else getattr(position, name)
            for name in ROW_FIELDS]


def from_row(index: int, row: object) -> Position:
    if not isinstance(row, list) or len(row) != len(ROW_FIELDS):
        raise ValueError(f"position {index}: a row has {len(ROW_FIELDS)} columns ({', '.join(ROW_FIELDS)})")
    values = dict(zip(ROW_FIELDS, row, strict=True))
    values["side"] = _SIDE_LETTERS[values["side"]]
    return Position(index, **values)


def to_json(observation: Observation) -> dict:
    return {
        "status": observation.status,
        "reason": observation.reason,
        "detail": observation.detail[:300],
        "actions_accepted": observation.actions_accepted,
        "last_action": observation.last_action,
        "positions": [to_row(position) for position in observation.positions],
    }


def from_json(source: str, record: str, cap: int | None, data: dict) -> Observation:
    positions = [from_row(index, row) for index, row in enumerate(data.get("positions", []))]
    return Observation(
        source=source, record=record, status=data["status"], positions=positions,
        actions_accepted=data.get("actions_accepted"), reason=data.get("reason", ""),
        detail=data.get("detail", ""), last_action=data.get("last_action"), cap=cap,
    )
