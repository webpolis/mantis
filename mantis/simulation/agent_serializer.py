"""
Serialization for agent-based simulation blocks.

Produces @AGENT protocol blocks with representative agent sampling:
- MAX_REPRESENTATIVES (20) agents per species, selected for behavioral interest
- Persistent tracking across ticks for trajectory continuity
- Delta encoding: only changed/dead tracked agents emitted between keyframes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent, AgentManager

QUANTIZE_GRID = 10
MAX_REPRESENTATIVES = 20

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


def _select_representatives(
    agents: list[Agent],
    prev_tracked: set[int],
    max_reps: int = MAX_REPRESENTATIVES,
) -> set[int]:
    """Select up to max_reps representative agents by behavioral interest.

    Priority:
    1. Previously tracked agents still alive (trajectory continuity)
    2. Agents in active interactions (hunting, fleeing, mating)
    3. Energy extremes (2 lowest, 2 highest)
    4. Spatially diverse fill (farthest-from-selected greedy)
    """
    if len(agents) <= max_reps:
        return {a.aid for a in agents}

    alive_by_aid = {a.aid: a for a in agents}
    selected: set[int] = set()

    # 1. Continuity: previously tracked agents still alive
    for aid in prev_tracked:
        if aid in alive_by_aid:
            selected.add(aid)
            if len(selected) >= max_reps:
                return selected

    # 2. Interacting agents (hunt, flee, mate — not rest/forage/flock)
    for a in agents:
        if a.aid in selected:
            continue
        if a.state in ("hunt", "flee", "mate"):
            selected.add(a.aid)
            # Also include hunt targets if they're in this species
            if a.state == "hunt" and a.target_aid is not None and a.target_aid in alive_by_aid:
                selected.add(a.target_aid)
            if len(selected) >= max_reps:
                return selected

    # 3. Energy extremes (2 lowest, 2 highest)
    remaining = [a for a in agents if a.aid not in selected]
    if remaining:
        by_energy = sorted(remaining, key=lambda a: a.energy)
        for a in by_energy[:2]:
            selected.add(a.aid)
            if len(selected) >= max_reps:
                return selected
        for a in by_energy[-2:]:
            selected.add(a.aid)
            if len(selected) >= max_reps:
                return selected

    # 4. Spatial diversity: greedily pick agent farthest from already-selected
    remaining = [a for a in agents if a.aid not in selected]
    if not remaining:
        return selected

    sel_positions = [(alive_by_aid[aid].x, alive_by_aid[aid].y) for aid in selected]

    while len(selected) < max_reps and remaining:
        best_agent = None
        best_min_dist = -1.0

        for a in remaining:
            if not sel_positions:
                min_dist = float('inf')
            else:
                min_dist = min(
                    (a.x - sx) ** 2 + (a.y - sy) ** 2
                    for sx, sy in sel_positions
                )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_agent = a

        if best_agent is None:
            break
        selected.add(best_agent.aid)
        sel_positions.append((best_agent.x, best_agent.y))
        remaining.remove(best_agent)

    return selected


def serialize_agents_keyframe(
    manager: AgentManager,
    tracked_aids: set[int],
    compact: bool = False,
) -> tuple[list[str], dict]:
    """Keyframe for tracked representative agents only.

    Header shows total population (informational). Lines emitted only for
    agents in tracked_aids.

    Returns (lines, snapshot_of_tracked).
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
        if a.aid in tracked_aids:
            lines.append(_format_agent_line(a, compact))
            snapshot[a.aid] = _snapshot_agent(a)

    return lines, {"agents": snapshot}


def serialize_agents_delta(
    manager: AgentManager,
    tracked_aids: set[int],
    prev_snapshot: dict,
    compact: bool = False,
) -> tuple[list[str], dict]:
    """Delta encoding for tracked representative agents only.

    Agent emitted if position moved >5 units, energy changed >2,
    or behavioral state changed.

    Returns (lines, new_snapshot).
    """
    agents = manager.agents
    if not agents and not manager.event_log.deaths:
        return [], prev_snapshot

    prev_agents = prev_snapshot.get("agents", {})
    changed_lines: list[str] = []

    new_agents: dict[int, tuple] = {}
    for a in agents:
        if a.aid not in tracked_aids:
            continue
        snap = _snapshot_agent(a)
        new_agents[a.aid] = snap
        qx, qy, e, age, state = snap[0], snap[1], snap[2], snap[3], snap[4]

        prev = prev_agents.get(a.aid)
        if prev is None:
            # Newly tracked agent
            changed_lines.append(_format_agent_line(a, compact))
        else:
            px, py, pe, p_state = prev[0], prev[1], prev[2], prev[4]
            moved = abs(qx - px) > 5 or abs(qy - py) > 5
            energy_changed = abs(e - pe) > 2
            state_changed = state != p_state
            if moved or energy_changed or state_changed:
                changed_lines.append(_format_agent_line(a, compact))

    # Dead tracked agents
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
    """Tracks per-species representative agent sets and snapshots for delta encoding."""

    def __init__(self):
        self._snapshots: dict[int, dict] = {}
        self._tracked: dict[int, set[int]] = {}

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

        # Refresh representative set each tick
        prev_tracked = self._tracked.get(species_sid, set())
        tracked = _select_representatives(manager.agents, prev_tracked)
        self._tracked[species_sid] = tracked

        if is_keyframe:
            lines, snapshot = serialize_agents_keyframe(
                manager, tracked, compact=compact,
            )
            self._snapshots[species_sid] = snapshot
            return lines
        else:
            prev = self._snapshots.get(species_sid, {})
            lines, new_snap = serialize_agents_delta(
                manager, tracked, prev, compact=compact,
            )
            self._snapshots[species_sid] = new_snap
            return lines
