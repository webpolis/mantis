"""
Serialization for agent-based simulation blocks.

Produces @AGENT protocol blocks using grid+notable hybrid format:
- Grid cells: 100-unit spatial aggregation with count, avg energy, behavior distribution
- Notable agents: top-N by energy, individually tracked with 10-unit quantized positions
- Delta encoding: only changed grid cells and changed/dead notables
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent, AgentManager

QUANTIZE_GRID = 10
NOTABLE_COUNT = 5

BEHAVIOR_ABBREV = {
    "rest": "r",
    "forage": "f",
    "hunt": "h",
    "mate": "m",
    "flee": "fl",
    "flock": "fk",
}

ABBREV_TO_BEHAVIOR = {v: k for k, v in BEHAVIOR_ABBREV.items()}


def _get_cell(x: float, y: float) -> tuple[int, int]:
    """Bin position to 100-unit grid cell (col, row)."""
    from .agent import CELL_SIZE
    return (int(x // CELL_SIZE), int(y // CELL_SIZE))


def _bin_agents_to_cells(agents: list[Agent]) -> dict[tuple[int, int], list[Agent]]:
    """Group agents by grid cell."""
    cells: dict[tuple[int, int], list[Agent]] = {}
    for a in agents:
        cell = _get_cell(a.x, a.y)
        cells.setdefault(cell, []).append(a)
    return cells


def _aggregate_cell(agents: list[Agent]) -> tuple[int, int, dict[str, int]]:
    """Aggregate agents in a cell: (count, avg_energy_int, {behavior_abbrev: count})."""
    count = len(agents)
    avg_energy = int(round(sum(a.energy for a in agents) / count))
    behaviors: dict[str, int] = {}
    for a in agents:
        abbrev = BEHAVIOR_ABBREV.get(a.state, a.state)
        behaviors[abbrev] = behaviors.get(abbrev, 0) + 1
    return count, avg_energy, behaviors


def _dominant_behavior(behaviors: dict[str, int]) -> str:
    """Return the behavior abbreviation with highest count."""
    return max(behaviors, key=behaviors.__getitem__)


def _select_notables(agents: list[Agent], n: int = NOTABLE_COUNT) -> list[Agent]:
    """Top N agents by energy, deterministic (sorted by energy desc, then aid asc)."""
    sorted_agents = sorted(agents, key=lambda a: (-a.energy, a.aid))
    return sorted_agents[:n]


def _quantize(x: float) -> int:
    """Round to nearest 10 units for notable positions."""
    return int(round(x / QUANTIZE_GRID)) * QUANTIZE_GRID


def serialize_agents_keyframe(
    manager: AgentManager,
    compact: bool = False,
) -> tuple[list[str], dict]:
    """Full agent state dump for a keyframe tick.

    Returns (lines, snapshot).
    """
    agents = manager.agents
    if not agents:
        return [], {}

    from .agent import CELL_SIZE

    total = len(agents)
    cells = _bin_agents_to_cells(agents)
    notables = _select_notables(agents)

    lines = []
    if compact:
        lines.append(f"  @AGENT {total} grid+{len(notables)} {CELL_SIZE}")
    else:
        lines.append(
            f"  @AGENT|count={total}"
            f"|mode=grid+{len(notables)}"
            f"|cell={CELL_SIZE}"
        )

    # Grid lines sorted by cell coord
    grid_snapshot = {}
    for (col, row) in sorted(cells.keys()):
        count, avg_e, behaviors = _aggregate_cell(cells[(col, row)])
        dominant = _dominant_behavior(behaviors)
        grid_snapshot[(col, row)] = (count, avg_e, dominant)

        # Sort behaviors by count desc
        sorted_beh = sorted(behaviors.items(), key=lambda x: -x[1])
        if compact:
            beh_str = " ".join(f"{b}:{c}" for b, c in sorted_beh)
            lines.append(f"   G {col},{row} {count} {avg_e} {beh_str}")
        else:
            beh_str = ",".join(f"{b}:{c}" for b, c in sorted_beh)
            lines.append(f"    G({col},{row}):n={count},E={avg_e}|{beh_str}")

    # Notable lines with 10-unit quantized positions
    notable_snapshot = {}
    notable_ids = set()
    for a in notables:
        qx = _quantize(a.x)
        qy = _quantize(a.y)
        e = int(round(a.energy))
        notable_snapshot[a.aid] = (qx, qy, e)
        notable_ids.add(a.aid)

        state_str = a.state
        if a.state == "hunt" and a.target_aid is not None:
            state_str = f"hunt->A{a.target_aid}"

        if compact:
            lines.append(f"   N A{a.aid} {qx} {qy} {e} {a.age} {state_str}")
        else:
            lines.append(f"    N:A{a.aid}:({qx},{qy},E={e},age={a.age},{state_str})")

    snapshot = {
        "grid": grid_snapshot,
        "notables": notable_snapshot,
        "notable_ids": notable_ids,
    }
    return lines, snapshot


def serialize_agents_delta(
    manager: AgentManager,
    prev_snapshot: dict,
    compact: bool = False,
) -> tuple[list[str], dict]:
    """Delta encoding: only changed grid cells and changed/dead notables.

    Returns (lines, new_snapshot).
    Grid cell emitted if count changed >2 OR avg energy changed >5 OR dominant behavior changed.
    Notable emitted if position moved >5 units or energy changed >2.
    """
    agents = manager.agents
    if not agents and not manager.event_log.deaths:
        return [], prev_snapshot

    from .agent import CELL_SIZE

    cells = _bin_agents_to_cells(agents)
    prev_grid = prev_snapshot.get("grid", {})
    prev_notables = prev_snapshot.get("notables", {})
    prev_notable_ids = prev_snapshot.get("notable_ids", set())

    changed_lines: list[str] = []

    # Build new grid snapshot and emit changed cells
    new_grid = {}
    all_cells = set(cells.keys()) | set(prev_grid.keys())
    for (col, row) in sorted(all_cells):
        cell_agents = cells.get((col, row), [])
        if not cell_agents:
            # Cell went empty — skip (don't emit empty cells)
            continue

        count, avg_e, behaviors = _aggregate_cell(cell_agents)
        dominant = _dominant_behavior(behaviors)
        new_grid[(col, row)] = (count, avg_e, dominant)

        prev = prev_grid.get((col, row))
        if prev is None:
            # New cell
            sorted_beh = sorted(behaviors.items(), key=lambda x: -x[1])
            if compact:
                beh_str = " ".join(f"{b}:{c}" for b, c in sorted_beh)
                changed_lines.append(f"   G {col},{row} {count} {avg_e} {beh_str}")
            else:
                beh_str = ",".join(f"{b}:{c}" for b, c in sorted_beh)
                changed_lines.append(f"    G({col},{row}):n={count},E={avg_e}|{beh_str}")
        else:
            p_count, p_avg_e, p_dominant = prev
            if abs(count - p_count) > 2 or abs(avg_e - p_avg_e) > 5 or dominant != p_dominant:
                sorted_beh = sorted(behaviors.items(), key=lambda x: -x[1])
                if compact:
                    beh_str = " ".join(f"{b}:{c}" for b, c in sorted_beh)
                    changed_lines.append(f"   G {col},{row} {count} {avg_e} {beh_str}")
                else:
                    beh_str = ",".join(f"{b}:{c}" for b, c in sorted_beh)
                    changed_lines.append(f"    G({col},{row}):n={count},E={avg_e}|{beh_str}")

    # Notable agents — same IDs from last keyframe, no replacements
    new_notables = {}
    agent_by_aid = {a.aid: a for a in agents}
    for aid in prev_notable_ids:
        a = agent_by_aid.get(aid)
        if a is not None:
            qx = _quantize(a.x)
            qy = _quantize(a.y)
            e = int(round(a.energy))
            new_notables[aid] = (qx, qy, e)

            prev_n = prev_notables.get(aid)
            if prev_n is not None:
                px, py, pe = prev_n
                moved = abs(qx - px) > 5 or abs(qy - py) > 5
                energy_changed = abs(e - pe) > 2
                if moved or energy_changed:
                    if compact:
                        changed_lines.append(f"   N A{aid} {qx} {qy} {e}")
                    else:
                        changed_lines.append(f"    N:A{aid}:({qx},{qy},E={e})")
            else:
                # Was notable but no prev data — emit full
                if compact:
                    changed_lines.append(f"   N A{aid} {qx} {qy} {e}")
                else:
                    changed_lines.append(f"    N:A{aid}:({qx},{qy},E={e})")

    # Dead notables
    for dead_aid in manager.event_log.deaths:
        if dead_aid in prev_notable_ids:
            if compact:
                changed_lines.append(f"   A{dead_aid} \u2020")
            else:
                changed_lines.append(f"    A{dead_aid}:\u2020")

    if not changed_lines:
        return [], {
            "grid": new_grid,
            "notables": new_notables,
            "notable_ids": prev_notable_ids,
        }

    if compact:
        lines = [f"  @AGENT \u0394 {CELL_SIZE}"]
    else:
        lines = [f"  @AGENT|\u0394pos|cell={CELL_SIZE}"]
    lines.extend(changed_lines)

    new_snapshot = {
        "grid": new_grid,
        "notables": new_notables,
        "notable_ids": prev_notable_ids,
    }
    return lines, new_snapshot


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
