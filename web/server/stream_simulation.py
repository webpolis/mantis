"""
Parse MANTIS protocol text into structured tick data for the frontend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedAgent:
    aid: int
    x: float
    y: float
    energy: float
    age: int = 0
    state: str = "rest"
    target_aid: int | None = None
    dead: bool = False
    species_sid: int = 0


@dataclass
class ParsedSpecies:
    sid: int
    plan: str = ""
    population: int = 0
    locations: list[str] = field(default_factory=list)
    traits: dict[str, float] = field(default_factory=dict)
    energy_in: float = 0
    energy_out: float = 0
    energy_store: float = 0


@dataclass
class ParsedTick:
    number: int
    epoch: int = 1
    species: list[ParsedSpecies] = field(default_factory=list)
    agents: list[ParsedAgent] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)


_AGENT_RE = re.compile(r"A(\d+):\(([^)]+)\)")
_AGENT_DEAD_RE = re.compile(r"A(\d+):\u2020")


def parse_agent_line(line: str) -> ParsedAgent | None:
    """Parse a single agent line from @AGENT block."""
    line = line.strip()

    dead_match = _AGENT_DEAD_RE.match(line)
    if dead_match:
        return ParsedAgent(aid=int(dead_match.group(1)), x=0, y=0, energy=0, dead=True)

    match = _AGENT_RE.match(line)
    if not match:
        return None

    aid = int(match.group(1))
    fields = match.group(2).split(",")
    if len(fields) < 3:
        return None

    x = float(fields[0])
    y = float(fields[1])

    energy = 0.0
    age = 0
    state = "rest"
    target_aid = None

    for f in fields[2:]:
        f = f.strip()
        if f.startswith("E="):
            energy = float(f[2:])
        elif f.startswith("age="):
            age = int(f[4:])
        elif f.startswith("hunt->A"):
            state = "hunt"
            target_aid = int(f[7:])
        elif f.startswith("flee"):
            state = "flee"
        elif f in ("forage", "rest", "mate", "flock"):
            state = f

    return ParsedAgent(
        aid=aid, x=x, y=y, energy=energy, age=age,
        state=state, target_aid=target_aid,
    )


def parse_agent_block(lines: list[str]) -> list[ParsedAgent]:
    """Parse an @AGENT block into agent list."""
    agents = []
    for line in lines:
        agent = parse_agent_line(line)
        if agent is not None:
            agents.append(agent)
    return agents


def parse_protocol_to_ticks(text: str) -> list[ParsedTick]:
    """Parse full protocol text into list of tick snapshots."""
    ticks: list[ParsedTick] = []
    current_tick = ParsedTick(number=0)
    tick_num = 0
    current_epoch = 1
    current_species: ParsedSpecies | None = None
    in_agent_block = False
    agent_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Epoch header
        if stripped.startswith("=EPOCH:"):
            parts = stripped.split("|")
            for p in parts:
                if p.startswith("=EPOCH:"):
                    try:
                        current_epoch = int(p.split(":")[1])
                    except (ValueError, IndexError):
                        pass
            current_tick.epoch = current_epoch
            continue

        # Separator = end of tick
        if stripped == "---":
            # Flush agent block
            if in_agent_block and current_species is not None:
                parsed = parse_agent_block(agent_lines)
                current_tick.agents.extend(
                    _agent_to_dict_agent(a, current_species.sid) for a in parsed
                )
                in_agent_block = False
                agent_lines = []

            current_tick.epoch = current_epoch
            ticks.append(current_tick)
            tick_num += 1
            current_tick = ParsedTick(number=tick_num)
            current_species = None
            continue

        # Agent block header
        if "@AGENT" in stripped:
            in_agent_block = True
            agent_lines = []
            continue

        # Agent lines (indented, start with A)
        if in_agent_block:
            if stripped.startswith("A"):
                agent_lines.append(stripped)
                continue
            else:
                # End of agent block
                if current_species is not None:
                    parsed = parse_agent_block(agent_lines)
                    current_tick.agents.extend(
                        _agent_to_dict_agent(a, current_species.sid) for a in parsed
                    )
                in_agent_block = False
                agent_lines = []

        # Species header
        if stripped.startswith("@SP|"):
            sp = _parse_species_header(stripped)
            if sp:
                current_species = sp
                current_tick.species.append(sp)
            continue

        # Interaction
        if stripped.startswith("@INT|"):
            current_tick.interactions.append({"raw": stripped})
            continue

    return ticks


def _agent_to_dict_agent(a: ParsedAgent, species_sid: int) -> ParsedAgent:
    """Attach species SID to parsed agent."""
    a.species_sid = species_sid
    return a


def _parse_species_header(line: str) -> ParsedSpecies | None:
    """Parse @SP|S{sid}|... header."""
    parts = line.split("|")
    if len(parts) < 2:
        return None

    sid_match = re.match(r"S(\d+)", parts[1])
    if not sid_match:
        return None
    sid = int(sid_match.group(1))
    sp = ParsedSpecies(sid=sid)

    for part in parts[2:]:
        if part.startswith("plan="):
            sp.plan = part[5:]
        elif part.startswith("pop="):
            pop_str = part[4:].split("±")[0].split("(")[0]
            try:
                sp.population = int(pop_str)
            except ValueError:
                pass
        elif part.startswith("L") or "," in part:
            sp.locations = [p.strip() for p in part.split(",")]

    return sp


def split_worlds(text: str) -> list[str]:
    """Split a multi-world dataset into individual world texts.

    Worlds are separated by one or more blank lines.
    """
    worlds = re.split(r"\n\s*\n(?=\s*=EPOCH:)", text)
    return [w.strip() for w in worlds if w.strip()]


def serialize_agent_for_frontend(agent: ParsedAgent, species_sid: int | None = None) -> dict:
    """Convert ParsedAgent to JSON-serializable dict for Socket.IO."""
    return {
        "aid": agent.aid,
        "species_sid": species_sid if species_sid is not None else agent.species_sid,
        "x": agent.x,
        "y": agent.y,
        "energy": agent.energy,
        "age": agent.age,
        "state": agent.state,
        "target_aid": agent.target_aid,
        "dead": agent.dead,
    }


def serialize_species_for_frontend(sp: ParsedSpecies) -> dict:
    """Convert ParsedSpecies to JSON-serializable dict."""
    return {
        "sid": sp.sid,
        "plan": sp.plan,
        "population": sp.population,
        "locations": sp.locations,
    }
