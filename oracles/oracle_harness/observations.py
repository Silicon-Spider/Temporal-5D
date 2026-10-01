"""What a rules implementation says about one game record, and how two such
reports are compared.

The gate compares three observables at every position of a record's main line
(position k is the state after k submitted actions):

  in_check       the "phantom" check of design spec 11.0: can the opponent capture
                 a royal once every eligible timeline has been advanced to its
                 phantom board (its next board, with no move played on it)? This
                 is the check that separates checkmate from stalemate.
  legal_actions  number of distinct legal actions (whole turns), counted up to a cap
  terminal       "none" while a legal action exists, else "checkmate" or "stalemate"

plus legality itself: whether each recorded action is accepted at all.

Each oracle answers with its own reading of these definitions - which timelines
count as eligible, for one. Where the readings differ the reports disagree, and
the rulebook, not the harness, says who is right.

Every field may be None: an oracle that cannot answer a question, or ran out of
time, says nothing rather than something. Nothing is ever inferred on an
oracle's behalf, and an unknown never disagrees with anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIDES = ("white", "black")
TERMINALS = ("none", "checkmate", "stalemate", "no_action")
STATUSES = ("ok", "rejected", "unsupported", "error", "timeout")
# The three things the gate exists to cross-check (design spec 11.0): move
# counts, check verdicts, and mate/stalemate verdicts.
OBSERVABLES = ("in_check", "legal_actions", "terminal")

# Why a source stopped replaying a record. The soft reasons cannot tell a rules
# disagreement from a notation the source does not read: both surface as "no
# move I can generate matches this text".
SOFT_REJECTIONS = ("no_matching_move", "ambiguous_move", "notation_mismatch", "unrecognised")
HARD_REJECTIONS = ("illegal_move", "cannot_submit")


@dataclass(frozen=True)
class Position:
    index: int
    side: str | None = None
    in_check: bool | None = None
    legal_actions: int | None = None
    terminal: str | None = None

    def known(self) -> int:
        return sum(value is not None for value in (self.side, self.in_check, self.legal_actions, self.terminal))


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

    def known(self) -> int:
        return sum(position.known() for position in self.positions)


@dataclass(frozen=True)
class Disagreement:
    record: str
    a: str
    b: str
    kind: str  # side | in_check | legal_actions | terminal | legality | length
    index: int
    a_value: str
    b_value: str
    soft: bool = False

    def key(self) -> tuple:
        return (self.record, self.a, self.b, self.kind, self.index)

    def describe(self) -> str:
        where = f"action {self.index}" if self.kind in ("legality", "length") else f"position {self.index}"
        soft = " (soft: may be notation, not rules)" if self.soft else ""
        return f"{self.record} {where}: {self.kind}: {self.a}={self.a_value} vs {self.b}={self.b_value}{soft}"


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
    # Position k exists exactly when k actions were accepted, whatever ended the run.
    if observation.actions_accepted != max(len(positions) - 1, 0):
        problems.append(f"actions_accepted is {observation.actions_accepted}, but {len(positions)} position(s) were reported")
    if observation.status in ("ok", "rejected") and not positions:
        problems.append(f"status {observation.status} without even the starting position")
    return problems


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

    A source that refused an action read that action too.
    """
    accepted = max(len(observation.positions) - 1, 0)
    return accepted + 1 if observation.status == "rejected" else accepted


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


def value_at(observation: Observation, kind: str, index: int) -> str | None:
    """What a report says about one observation, in the words compare() prints; None if it says nothing.

    For legality and length, ``index`` is an action number, counted from 1.
    """
    if kind == "legality":
        if index < 1:
            return None
        if observation.status == "rejected" and observation.actions_accepted is not None and observation.actions_accepted + 1 == index:
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

    # Compared up to the shorter report: what one source did not reach is unknown.
    for pa, pb in zip(a.positions, b.positions, strict=False):
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

    # Legality: one source refused an action the other went on to play.
    for first, second, flipped in ((a, b, False), (b, a, True)):
        if first.status != "rejected" or first.actions_accepted is None:
            continue
        refused = first.actions_accepted + 1
        if len(second.positions) > refused:
            soft = first.reason in SOFT_REJECTIONS
            values = (f"rejected:{first.reason}", "accepted")
            add("legality", refused, *(values[::-1] if flipped else values), soft=soft)

    # Length: one source says the record ends where the other found another
    # action - one it played, or one it refused. (A source that silently stops
    # reading early would otherwise look like one that agrees about everything
    # it saw, and would never have to say whether the next action is legal.)
    for first, second, flipped in ((a, b, False), (b, a, True)):
        if first.status == "ok" and _actions_seen(second) > _actions_seen(first):
            values = (_length_text(first), _length_text(second))
            soft = second.status == "rejected" and second.reason in SOFT_REJECTIONS
            add("length", _actions_seen(first) + 1, *(values[::-1] if flipped else values), soft=soft)
    return found


def cut_short(observation: Observation) -> bool:
    """Did the run stop because it ran out of time?

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


def coverage(golden: Observation, fresh: Observation) -> tuple[int, int]:
    """(golden values the fresh run also produced, golden values) - guards against a vacuous pass."""
    total = reproduced = 0
    for index, position in enumerate(golden.positions):
        other = fresh.positions[index] if index < len(fresh.positions) else Position(index)
        for name in ("side", "in_check", "legal_actions", "terminal"):
            if getattr(position, name) is not None:
                total += 1
                reproduced += getattr(other, name) is not None
    return reproduced, total


# --- serialisation (golden files and run reports) ---------------------------

def to_json(observation: Observation) -> dict:
    return {
        "status": observation.status,
        "reason": observation.reason,
        "detail": observation.detail[:300],
        "actions_accepted": observation.actions_accepted,
        "last_action": observation.last_action,
        "positions": [[p.side[0] if p.side else None, p.in_check, p.legal_actions, p.terminal] for p in observation.positions],
    }


def from_json(source: str, record: str, cap: int | None, data: dict) -> Observation:
    sides = {"w": "white", "b": "black", None: None}
    positions = [
        Position(index, sides[row[0]], row[1], row[2], row[3]) for index, row in enumerate(data.get("positions", []))
    ]
    return Observation(
        source=source, record=record, status=data["status"], positions=positions,
        actions_accepted=data.get("actions_accepted"), reason=data.get("reason", ""),
        detail=data.get("detail", ""), last_action=data.get("last_action"), cap=cap,
    )
