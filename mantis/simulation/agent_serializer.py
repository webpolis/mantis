"""
Serialization for agent-based simulation blocks.

Produces @AGENT protocol blocks with per-agent positions:
- Each agent gets an individual line with 10-unit quantized position
- Delta encoding: only changed/new/dead agents emitted between keyframes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent, AgentManager

QUANTIZE_GRID = 10

BEHAVIOR_ABBREV = {
    "rest": "r",
    "forage": "f",
    "hunt": "h",
    "mate": "m",
    "flee": "fl",
    "flock": "fk",
}

ABBREV_TO_BEHAVIOR = {v: k for k, v in BEHAVIOR_ABBREV.items()}


def _quantize(x: float) -> int:
    """Round to nearest 10 units for positions."""
    return int(round(x / QUANTIZE_GRID)) * QUANTIZE_GRID


def _format_agent_line(a: Agent, compact: bool) -> str:
    """Format a single agent as a protocol line."""
    qx = _quantize(a.x)
    qy = _quantize(a.y)
    e = int(round(a.energy))
    state_str = a.state
    if a.state == "hunt" and a.target_aid is not None:
        state_str = f"hunt->A{a.target_aid}"

    if compact:
        return f"   N A{a.aid} {qx} {qy} {e} {a.age} {state_str}"
    else:
        return f"    N:A{a.aid}:({qx},{qy},E={e},age={a.age},{state_str})"


def _snapshot_agent(a: Agent) -> tuple[int, int, int, int, str, int | None]:
    """Snapshot an agent's state for delta comparison."""
    return (_quantize(a.x), _quantize(a.y), int(round(a.energy)),
            a.age, a.state, a.target_aid)


def serialize_agents_keyframe(
    manager: AgentManager,
    compact: bool = False,
) -> tuple[list[str], dict]:
    """Full agent state dump — every agent with its quantized position.

    Returns (lines, snapshot).
    """
    agents = manager.agents
    if not agents:
        return [], {}

    total = len(agents)
    lines = []
    if compact:
        lines.append(f"  @AGENT {total}")
    else:
        lines.append(f"  @AGENT|count={total}")

    snapshot: dict[int, tuple] = {}
    for a in agents:
        lines.append(_format_agent_line(a, compact))
        snapshot[a.aid] = _snapshot_agent(a)

    return lines, {"agents": snapshot}


def serialize_agents_delta(
    manager: AgentManager,
    prev_snapshot: dict,
    compact: bool = False,
) -> tuple[list[str], dict]:
    """Delta encoding: only changed, new, or dead agents.

    Returns (lines, new_snapshot).
    Agent emitted if position moved >5 units, energy changed >2, or age changed.
    """
    agents = manager.agents
    if not agents and not manager.event_log.deaths:
        return [], prev_snapshot

    prev_agents = prev_snapshot.get("agents", {})
    changed_lines: list[str] = []

    new_agents: dict[int, tuple] = {}
    for a in agents:
        snap = _snapshot_agent(a)
        new_agents[a.aid] = snap
        qx, qy, e, age = snap[0], snap[1], snap[2], snap[3]

        prev = prev_agents.get(a.aid)
        if prev is None:
            # New agent
            changed_lines.append(_format_agent_line(a, compact))
        else:
            px, py, pe, p_age = prev[0], prev[1], prev[2], prev[3]
            moved = abs(qx - px) > 5 or abs(qy - py) > 5
            energy_changed = abs(e - pe) > 2
            age_changed = age != p_age
            if moved or energy_changed or age_changed:
                changed_lines.append(_format_agent_line(a, compact))

    # Dead agents
    for dead_aid in manager.event_log.deaths:
        if dead_aid in prev_agents:
            if compact:
                changed_lines.append(f"   A{dead_aid} \u2020")
            else:
                changed_lines.append(f"    A{dead_aid}:\u2020")

    if not changed_lines:
        return [], {"agents": new_agents}

    if compact:
        lines = [f"  @AGENT \u0394"]
    else:
        lines = [f"  @AGENT|\u0394pos"]
    lines.extend(changed_lines)

    return lines, {"agents": new_agents}


class AgentSerializerState:
    """Tracks per-species agent snapshots for delta encoding."""

    def __init__(self):
        self._snapshots: dict[int, dict] = {}

    def serialize(
        self,
        species_sid: int,
        manager: AgentManager,
        is_keyframe: bool,
        compact: bool = False,
    ) -> list[str]:
        """Serialize agent block for a species. Returns protocol lines."""
        if not manager.agents and not manager.event_log.deaths:
            return []

        if is_keyframe:
            lines, snapshot = serialize_agents_keyframe(manager, compact=compact)
            self._snapshots[species_sid] = snapshot
            return lines
        else:
            prev = self._snapshots.get(species_sid, {})
            lines, new_snap = serialize_agents_delta(manager, prev, compact=compact)
            self._snapshots[species_sid] = new_snap
            return lines
