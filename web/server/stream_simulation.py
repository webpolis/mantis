"""
Parse MANTIS protocol text into structured tick data for the frontend.

Supports both v1 (pipe-delimited) and v2 (compact space-separated) formats.
Auto-detects format from the first =EPOCH line.
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
    count: int = 1


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
class ParsedBiome:
    lid: int
    name: str
    vegetation: float
    detritus: float
    nitrogen: float = 0.5
    phosphorus: float = 0.3


@dataclass
class ParsedEvent:
    target: str       # "WORLD" or "S3"
    event_type: str   # "catastrophe", "catastrophe_end", "disease", "symbiogenesis", etc.
    detail: str       # remaining payload


@dataclass
class ParsedTick:
    number: int
    epoch: int = 1
    species: list[ParsedSpecies] = field(default_factory=list)
    agents: list[ParsedAgent] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    biomes: list[ParsedBiome] = field(default_factory=list)
    events: list[ParsedEvent] = field(default_factory=list)


# ------------------------------------------------------------------
# v1 agent parsing
# ------------------------------------------------------------------

_AGENT_RE = re.compile(r"A(\d+):\(([^)]+)\)")
_AGENT_DEAD_RE = re.compile(r"A(\d+):\u2020")


def parse_agent_line(line: str) -> ParsedAgent | None:
    """Parse a single v1 agent line from @AGENT block."""
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


# ------------------------------------------------------------------
# v2 compact agent parsing
# ------------------------------------------------------------------

_COMPACT_AGENT_RE = re.compile(r"A(\d+)\s+(.+)")
_COMPACT_AGENT_DEAD_RE = re.compile(r"A(\d+)\s+\u2020")


def _parse_compact_agent_line(line: str) -> ParsedAgent | None:
    """Parse a single v2 compact agent line: A0 120 340 45 8 forage."""
    line = line.strip()

    dead_match = _COMPACT_AGENT_DEAD_RE.match(line)
    if dead_match:
        return ParsedAgent(aid=int(dead_match.group(1)), x=0, y=0, energy=0, dead=True)

    match = _COMPACT_AGENT_RE.match(line)
    if not match:
        return None

    aid = int(match.group(1))
    tokens = match.group(2).split()
    if len(tokens) < 3:
        return None

    x = float(tokens[0])
    y = float(tokens[1])
    energy = float(tokens[2])
    age = 0
    state = "rest"
    target_aid = None

    if len(tokens) >= 4:
        try:
            age = int(tokens[3])
        except ValueError:
            pass
    if len(tokens) >= 5:
        state_str = tokens[4]
        if state_str.startswith("hunt->A"):
            state = "hunt"
            try:
                target_aid = int(state_str[7:])
            except ValueError:
                pass
        elif state_str in ("forage", "rest", "mate", "flock", "flee", "hunt"):
            state = state_str

    return ParsedAgent(
        aid=aid, x=x, y=y, energy=energy, age=age,
        state=state, target_aid=target_aid,
    )


# ------------------------------------------------------------------
# Grid+notable agent parsing
# ------------------------------------------------------------------

_ABBREV_TO_BEHAVIOR = {
    "r": "rest",
    "f": "forage",
    "h": "hunt",
    "m": "mate",
    "fl": "flee",
    "fk": "flock",
}


def _parse_grid_line(line: str, compact: bool, cell_size: int = 100) -> list[ParsedAgent]:
    """Parse a grid cell line into pseudo-agents at the cell center.

    v2: G col,row count avg_energy beh:count ...
    v1: G(col,row):n=N,E=avg|beh:count,...
    """
    line = line.strip()
    if compact:
        # v2: G 1,3 17 45 f:12 h:3 fl:2
        parts = line.split()
        if len(parts) < 4:
            return []
        coords = parts[1].split(",")
        if len(coords) != 2:
            return []
        try:
            col, row = int(coords[0]), int(coords[1])
            count = int(parts[2])
            avg_energy = float(parts[3])
        except (ValueError, IndexError):
            return []

        # Find dominant behavior from remaining tokens
        dominant = "rest"
        max_beh_count = 0
        for tok in parts[4:]:
            if ":" in tok:
                beh_abbrev, beh_count_str = tok.split(":", 1)
                try:
                    beh_count = int(beh_count_str)
                    if beh_count > max_beh_count:
                        max_beh_count = beh_count
                        dominant = _ABBREV_TO_BEHAVIOR.get(beh_abbrev, beh_abbrev)
                except ValueError:
                    pass
    else:
        # v1: G(col,row):n=N,E=avg|beh:count,...
        m = re.match(r"G\((\d+),(\d+)\):n=(\d+),E=(\d+)\|(.+)", line)
        if not m:
            return []
        col, row = int(m.group(1)), int(m.group(2))
        count = int(m.group(3))
        avg_energy = float(m.group(4))
        beh_str = m.group(5)
        dominant = "rest"
        max_beh_count = 0
        for pair in beh_str.split(","):
            if ":" in pair:
                beh_abbrev, beh_count_str = pair.split(":", 1)
                try:
                    beh_count = int(beh_count_str)
                    if beh_count > max_beh_count:
                        max_beh_count = beh_count
                        dominant = _ABBREV_TO_BEHAVIOR.get(beh_abbrev, beh_abbrev)
                except ValueError:
                    pass

    # Create pseudo-agent at cell center with stable negative ID
    center_x = col * cell_size + cell_size // 2
    center_y = row * cell_size + cell_size // 2
    aid = -(col * 1000 + row + 1)

    return [ParsedAgent(
        aid=aid,
        x=float(center_x),
        y=float(center_y),
        energy=avg_energy,
        state=dominant,
        count=count,
    )]


def _parse_notable_line(line: str, compact: bool) -> ParsedAgent | None:
    """Parse a notable agent line.

    v2: N A1 130 350 52 10 hunt->A3
    v1: N:A1:(130,350,E=52,age=10,hunt->A3)
    """
    line = line.strip()
    if compact:
        # Strip "N " prefix, delegate to compact parser
        if line.startswith("N "):
            return _parse_compact_agent_line(line[2:])
    else:
        # Strip "N:" prefix, delegate to v1 parser
        if line.startswith("N:"):
            return parse_agent_line(line[2:])
    return None


def parse_agent_block(lines: list[str], compact: bool = False) -> list[ParsedAgent]:
    """Parse an @AGENT block into agent list.

    Dispatches lines by prefix: G (grid), N (notable), A (death/legacy).
    """
    agents = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("G"):
            agents.extend(_parse_grid_line(stripped, compact))
        elif stripped.startswith("N"):
            agent = _parse_notable_line(stripped, compact)
            if agent is not None:
                agents.append(agent)
        elif stripped.startswith("A"):
            # Death markers or legacy per-agent lines
            parser = _parse_compact_agent_line if compact else parse_agent_line
            agent = parser(stripped)
            if agent is not None:
                agents.append(agent)
    return agents


# ------------------------------------------------------------------
# Format detection
# ------------------------------------------------------------------

def _detect_format(text: str) -> str:
    """Detect protocol format from first =EPOCH line. Returns 'v1' or 'v2'."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("=EPOCH"):
            if ":" in stripped[:10]:
                return "v1"
            return "v2"
    return "v1"


# ------------------------------------------------------------------
# v1 species header parsing
# ------------------------------------------------------------------

def _parse_species_header(line: str) -> ParsedSpecies | None:
    """Parse v1 @SP|S{sid}|... header."""
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
            pop_str = part[4:].split("\u00b1")[0].split("(")[0]
            try:
                sp.population = int(pop_str)
            except ValueError:
                pass
        elif part.startswith("L") or "," in part:
            sp.locations = [p.strip() for p in part.split(",")]

    return sp


# ------------------------------------------------------------------
# v2 compact species header parsing
# ------------------------------------------------------------------

def _parse_compact_species_header(line: str) -> ParsedSpecies | None:
    """Parse v2 @SP S{sid} L0 scavenger 6565 D det 51 plt 35 ..."""
    tokens = line.split()
    if len(tokens) < 2:
        return None

    # tokens[0] = "@SP", tokens[1] = "S0", ...
    sid_match = re.match(r"S(\d+)", tokens[1])
    if not sid_match:
        return None
    sid = int(sid_match.group(1))
    sp = ParsedSpecies(sid=sid)

    # Gather location tokens (L-prefixed) until we hit a non-L token
    idx = 2
    locs = []
    while idx < len(tokens) and tokens[idx].startswith("L"):
        locs.append(tokens[idx])
        idx += 1
    sp.locations = locs

    # Body plan name (not a keyword or number)
    if idx < len(tokens) and tokens[idx] not in ("D", "pop", "locs") and not tokens[idx][0].isdigit():
        sp.plan = tokens[idx]
        idx += 1

    # Population (direct int for keyframe, "pop N" for delta)
    if idx < len(tokens):
        tok = tokens[idx]
        if tok == "pop":
            idx += 1
            if idx < len(tokens):
                try:
                    sp.population = int(tokens[idx])
                    idx += 1
                except ValueError:
                    pass
        else:
            try:
                sp.population = int(tok)
                idx += 1
            except ValueError:
                pass

    # Delta may also have "locs L0 L1 ..."
    if idx < len(tokens) and tokens[idx] == "locs":
        idx += 1
        while idx < len(tokens) and tokens[idx].startswith("L"):
            sp.locations.append(tokens[idx])
            idx += 1

    return sp


# ------------------------------------------------------------------
# Biome parsing
# ------------------------------------------------------------------

_BIO_V1_RE = re.compile(r"L(\d+):(\w+)\(veg=([\d.]+),det=([\d.]+)(?:,N=([\d.]+),P=([\d.]+))?\)")


def _parse_bio_line(line: str, compact: bool) -> list[ParsedBiome]:
    """Parse a @BIO line into biome list.

    v2: @BIO L0 shallows 80 120 L1 forest 30 0
    v1: @BIO|L0:shallows(veg=0.8,det=120)|L1:forest(veg=0.3,det=0)
    """
    biomes: list[ParsedBiome] = []
    if compact:
        # Strip "@BIO " prefix, then groups of 4+ tokens per biome
        tokens = line.split()
        idx = 1  # skip "@BIO"
        while idx + 3 < len(tokens):
            lid_tok = tokens[idx]
            if not lid_tok.startswith("L"):
                idx += 1
                continue
            try:
                lid = int(lid_tok[1:])
                name = tokens[idx + 1]
                veg = float(tokens[idx + 2]) / 100.0
                det = float(tokens[idx + 3])
                idx += 4
                # Peek for optional N/P tokens (e.g. N7 P3 → 0.7, 0.3)
                nitrogen = 0.5
                phosphorus = 0.3
                if idx < len(tokens) and tokens[idx].startswith("N") and tokens[idx][1:].isdigit():
                    nitrogen = int(tokens[idx][1:]) / 10.0
                    idx += 1
                    if idx < len(tokens) and tokens[idx].startswith("P") and tokens[idx][1:].isdigit():
                        phosphorus = int(tokens[idx][1:]) / 10.0
                        idx += 1
                biomes.append(ParsedBiome(lid=lid, name=name, vegetation=veg, detritus=det,
                                          nitrogen=nitrogen, phosphorus=phosphorus))
            except (ValueError, IndexError):
                idx += 4
    else:
        # v1: pipe-delimited
        parts = line.split("|")
        for part in parts[1:]:  # skip "@BIO"
            m = _BIO_V1_RE.match(part.strip())
            if m:
                nitrogen = float(m.group(5)) if m.group(5) else 0.5
                phosphorus = float(m.group(6)) if m.group(6) else 0.3
                biomes.append(ParsedBiome(
                    lid=int(m.group(1)),
                    name=m.group(2),
                    vegetation=float(m.group(3)),
                    detritus=float(m.group(4)),
                    nitrogen=nitrogen,
                    phosphorus=phosphorus,
                ))
    return biomes


# ------------------------------------------------------------------
# Event parsing
# ------------------------------------------------------------------

def _parse_event_line(line: str, compact: bool) -> ParsedEvent | None:
    """Parse an @EVT line into a ParsedEvent.

    v2: @EVT WORLD catastrophe meteor_impact dur=5
    v1: @EVT|WORLD|catastrophe|meteor_impact dur=5
    """
    if compact:
        tokens = line.split()
        if len(tokens) < 3:
            return None
        target = tokens[1]
        event_type = tokens[2]
        detail = " ".join(tokens[3:]) if len(tokens) > 3 else ""
    else:
        parts = line.split("|")
        if len(parts) < 3:
            return None
        target = parts[1].strip()
        event_type = parts[2].strip()
        detail = "|".join(parts[3:]).strip() if len(parts) > 3 else ""
    return ParsedEvent(target=target, event_type=event_type, detail=detail)


# ------------------------------------------------------------------
# Main parser
# ------------------------------------------------------------------

def parse_protocol_to_ticks(text: str) -> list[ParsedTick]:
    """Parse full protocol text into list of tick snapshots.

    Auto-detects v1 (pipe-delimited) and v2 (compact space-separated) formats.
    """
    fmt = _detect_format(text)
    compact = fmt == "v2"

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
        if stripped.startswith("=EPOCH"):
            if compact:
                # v2: =EPOCH 1 1000 W0
                tokens = stripped.split()
                if len(tokens) >= 2:
                    try:
                        current_epoch = int(tokens[1])
                    except (ValueError, IndexError):
                        pass
            else:
                # v1: =EPOCH:1|TICK_SCALE:1000gen|W0
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
                parsed = parse_agent_block(agent_lines, compact=compact)
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

        # Agent lines (indented, start with G, N, or A)
        if in_agent_block:
            if stripped[0] in ("G", "N", "A"):
                agent_lines.append(stripped)
                continue
            else:
                # End of agent block
                if current_species is not None:
                    parsed = parse_agent_block(agent_lines, compact=compact)
                    current_tick.agents.extend(
                        _agent_to_dict_agent(a, current_species.sid) for a in parsed
                    )
                in_agent_block = False
                agent_lines = []

        # Species header
        if compact:
            if stripped.startswith("@SP "):
                sp = _parse_compact_species_header(stripped)
                if sp:
                    current_species = sp
                    current_tick.species.append(sp)
                continue
        else:
            if stripped.startswith("@SP|"):
                sp = _parse_species_header(stripped)
                if sp:
                    current_species = sp
                    current_tick.species.append(sp)
                continue

        # Events
        if stripped.startswith("@EVT"):
            evt = _parse_event_line(stripped, compact)
            if evt:
                current_tick.events.append(evt)
            continue

        # Biome data
        if stripped.startswith("@BIO"):
            current_tick.biomes = _parse_bio_line(stripped, compact)
            continue

        # Interaction
        if stripped.startswith("@INT"):
            current_tick.interactions.append({"raw": stripped})
            continue

    return ticks


def _agent_to_dict_agent(a: ParsedAgent, species_sid: int) -> ParsedAgent:
    """Attach species SID to parsed agent."""
    a.species_sid = species_sid
    return a


def split_worlds(text: str) -> list[str]:
    """Split a multi-world dataset into individual world texts.

    Worlds are separated by one or more blank lines.
    """
    worlds = re.split(r"\n\s*\n(?=\s*=EPOCH)", text)
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
        "count": agent.count,
    }


def serialize_species_for_frontend(sp: ParsedSpecies) -> dict:
    """Convert ParsedSpecies to JSON-serializable dict."""
    return {
        "sid": sp.sid,
        "plan": sp.plan,
        "population": sp.population,
        "locations": sp.locations,
    }


def serialize_biome_for_frontend(biome: ParsedBiome, patches: list[dict] | None = None) -> dict:
    """Convert ParsedBiome to JSON-serializable dict."""
    return {
        "lid": biome.lid,
        "name": biome.name,
        "vegetation": biome.vegetation,
        "detritus": biome.detritus,
        "nitrogen": biome.nitrogen,
        "phosphorus": biome.phosphorus,
        "patches": patches or [],
    }


def serialize_event_for_frontend(evt: ParsedEvent) -> dict:
    """Convert ParsedEvent to JSON-serializable dict."""
    return {
        "target": evt.target,
        "event_type": evt.event_type,
        "detail": evt.detail,
    }
