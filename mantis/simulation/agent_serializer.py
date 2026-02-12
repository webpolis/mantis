"""
Serialization for agent-based simulation blocks.

Produces @AGENT protocol blocks with grid-cell aggregation:
- 500-unit cells on 1000×1000 world = 2×2 grid = 4 cells max per species
- Delta encoding: only changed/empty cells emitted between keyframes
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent, AgentManager

GRID_CELL_SIZE = 500

BEHAVIOR_ABBREV = {
    "rest": "r",
    "forage": "f",
    "hunt": "h",
    "mate": "m",
    "flee": "fl",
    "flock": "fk",
}

ABBREV_TO_BEHAVIOR = {v: k for k, v in BEHAVIOR_ABBREV.items()}


def _aggregate_grid(
    agents: list[Agent], cell_size: int = GRID_CELL_SIZE,
) -> dict[tuple[int, int], dict]:
    """Bucket agents into grid cells and compute per-cell summaries.

    Returns dict mapping (col, row) to:
        {count, avg_energy, avg_age, behaviors: Counter}
    """
    cells: dict[tuple[int, int], dict] = {}
    for a in agents:
        col = int(a.x) // cell_size
        row = int(a.y) // cell_size
        key = (col, row)
        if key not in cells:
            cells[key] = {
                "count": 0,
                "total_energy": 0.0,
                "total_age": 0,
                "behaviors": Counter(),
            }
        c = cells[key]
        c["count"] += 1
        c["total_energy"] += a.energy
        c["total_age"] += a.age
        c["behaviors"][a.state] += 1

    result: dict[tuple[int, int], dict] = {}
    for key, c in cells.items():
        n = c["count"]
        result[key] = {
            "count": n,
            "avg_energy": int(round(c["total_energy"] / n)),
            "avg_age": int(round(c["total_age"] / n)),
            "behaviors": c["behaviors"],
        }
    return result


def _format_grid_line(
    col: int, row: int, cell: dict, compact: bool,
) -> str:
    """Format a single grid cell as a protocol line."""
    count = cell["count"]
    avg_e = cell["avg_energy"]
    avg_age = cell["avg_age"]
    beh = cell["behaviors"]

    # Sort behaviors by count descending
    beh_sorted = sorted(beh.items(), key=lambda x: -x[1])

    if compact:
        beh_parts = []
        for state, cnt in beh_sorted:
            abbrev = BEHAVIOR_ABBREV.get(state, state)
            beh_parts.append(f"{abbrev}:{cnt}")
        beh_str = " ".join(beh_parts)
        return f"   G {col},{row} {count} {avg_e} {avg_age} {beh_str}"
    else:
        beh_parts = []
        for state, cnt in beh_sorted:
            abbrev = BEHAVIOR_ABBREV.get(state, state)
            beh_parts.append(f"{abbrev}:{cnt}")
        beh_str = ",".join(beh_parts)
        return f"    G:({col},{row})|n={count}|E={avg_e}|age={avg_age}|{beh_str}"


def _snapshot_grid(
    cells: dict[tuple[int, int], dict],
) -> dict[tuple[int, int], tuple[int, int, int, Counter]]:
    """Create a snapshot for delta comparison."""
    return {
        key: (c["count"], c["avg_energy"], c["avg_age"], Counter(c["behaviors"]))
        for key, c in cells.items()
    }


def serialize_agents_keyframe(
    manager: AgentManager,
    compact: bool = False,
) -> tuple[list[str], dict]:
    """Full agent state dump — grid-cell aggregated.

    Returns (lines, snapshot).
    """
    agents = manager.agents
    if not agents:
        return [], {}

    total = len(agents)
    cells = _aggregate_grid(agents)

    lines = []
    if compact:
        lines.append(f"  @AGENT {total}")
    else:
        lines.append(f"  @AGENT|count={total}")

    for (col, row), cell in sorted(cells.items()):
        lines.append(_format_grid_line(col, row, cell, compact))

    snapshot = _snapshot_grid(cells)
    return lines, {"cells": snapshot}


def serialize_agents_delta(
    manager: AgentManager,
    prev_snapshot: dict,
    compact: bool = False,
) -> tuple[list[str], dict]:
    """Delta encoding: only changed or emptied grid cells.

    Emits cell if count changed, energy diff >5, or age diff >2.
    Emits death marker for cells that existed previously but are now empty.
    """
    agents = manager.agents
    if not agents and not manager.event_log.deaths:
        return [], prev_snapshot

    cells = _aggregate_grid(agents) if agents else {}
    new_snapshot = _snapshot_grid(cells)
    prev_cells = prev_snapshot.get("cells", {})

    changed_lines: list[str] = []

    # Check current cells against previous
    for key in sorted(cells.keys()):
        cell = cells[key]
        col, row = key
        prev = prev_cells.get(key)
        if prev is None:
            # New cell
            changed_lines.append(_format_grid_line(col, row, cell, compact))
        else:
            p_count, p_energy, p_age, _ = prev
            count_changed = cell["count"] != p_count
            energy_changed = abs(cell["avg_energy"] - p_energy) > 5
            age_changed = abs(cell["avg_age"] - p_age) > 2
            if count_changed or energy_changed or age_changed:
                changed_lines.append(_format_grid_line(col, row, cell, compact))

    # Cells that disappeared (existed in prev but not in current)
    for key in sorted(prev_cells.keys()):
        if key not in cells:
            col, row = key
            if compact:
                changed_lines.append(f"   G {col},{row} \u2020")
            else:
                changed_lines.append(f"    G:({col},{row})|\u2020")

    if not changed_lines:
        return [], {"cells": new_snapshot}

    if compact:
        lines = [f"  @AGENT \u0394"]
    else:
        lines = [f"  @AGENT|\u0394pos"]
    lines.extend(changed_lines)

    return lines, {"cells": new_snapshot}


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
