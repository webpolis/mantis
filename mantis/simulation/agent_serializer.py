"""
Serialization for agent-based simulation blocks.

Produces @AGENT protocol blocks with:
- Quantized coordinates (10-unit grid)
- Keyframe (full state) and delta (changes only) encoding
- Sampling: top-200 by energy + random-50 for diversity
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .agent import Agent, AgentManager


QUANTIZE_GRID = 10


def quantize_position(x: float) -> int:
    """Round to nearest 10 units for serialization."""
    return int(round(x / QUANTIZE_GRID)) * QUANTIZE_GRID


def serialize_agents_keyframe(
    manager: AgentManager,
    rng: Optional[np.random.Generator] = None,
    max_sample: int = 250,
    compact: bool = False,
) -> list[str]:
    """Full agent state dump for a keyframe tick."""
    agents = manager.sample_for_serialization(max_sample, rng)
    if not agents:
        return []

    total = len(manager.agents)
    top_n = int(max_sample * 0.8)
    rand_n = max_sample - top_n

    lines = []
    sample_str = f"top{min(top_n, total)}+rand{min(rand_n, max(0, total - top_n))}"

    if compact:
        lines.append(f"  @AGENT {total} {sample_str} {QUANTIZE_GRID}")
    else:
        lines.append(
            f"  @AGENT|count={total}"
            f"|sample={sample_str}"
            f"|quantize={QUANTIZE_GRID}"
        )

    for a in agents:
        qx = quantize_position(a.x)
        qy = quantize_position(a.y)
        e = int(round(a.energy))

        state_str = a.state
        if a.state == "hunt" and a.target_aid is not None:
            state_str = f"hunt->A{a.target_aid}"
        elif a.state == "flee":
            state_str = "flee"

        if compact:
            lines.append(f"   A{a.aid} {qx} {qy} {e} {a.age} {state_str}")
        else:
            lines.append(f"    A{a.aid}:({qx},{qy},E={e},age={a.age},{state_str})")

    return lines


def serialize_agents_delta(
    manager: AgentManager,
    prev_snapshot: dict[int, tuple[int, int, int]],
    rng: Optional[np.random.Generator] = None,
    max_sample: int = 250,
    compact: bool = False,
) -> tuple[list[str], dict[int, tuple[int, int, int]]]:
    """Delta encoding: only changed agents.

    Returns (lines, new_snapshot).
    An agent is included if position moved >5 units or energy changed >2.
    Dead agents marked with dagger.
    """
    agents = manager.sample_for_serialization(max_sample, rng)
    if not agents and not manager.event_log.deaths:
        return [], prev_snapshot

    new_snapshot: dict[int, tuple[int, int, int]] = {}
    changed_lines: list[str] = []

    for a in agents:
        qx = quantize_position(a.x)
        qy = quantize_position(a.y)
        e = int(round(a.energy))
        new_snapshot[a.aid] = (qx, qy, e)

        prev = prev_snapshot.get(a.aid)
        if prev is None:
            # New agent
            state_str = a.state
            if a.state == "hunt" and a.target_aid is not None:
                state_str = f"hunt->A{a.target_aid}"
            if compact:
                changed_lines.append(f"   A{a.aid} {qx} {qy} {e} {a.age} {state_str}")
            else:
                changed_lines.append(f"    A{a.aid}:({qx},{qy},E={e},age={a.age},{state_str})")
        else:
            px, py, pe = prev
            moved = abs(qx - px) > 5 or abs(qy - py) > 5
            energy_changed = abs(e - pe) > 2
            if moved or energy_changed:
                if compact:
                    changed_lines.append(f"   A{a.aid} {qx} {qy} {e}")
                else:
                    changed_lines.append(f"    A{a.aid}:({qx},{qy},E={e})")

    # Dead agents
    for dead_aid in manager.event_log.deaths:
        if dead_aid in prev_snapshot:
            if compact:
                changed_lines.append(f"   A{dead_aid} \u2020")
            else:
                changed_lines.append(f"    A{dead_aid}:\u2020")  # dagger

    if not changed_lines:
        return [], new_snapshot

    if compact:
        lines = [f"  @AGENT \u0394 {QUANTIZE_GRID}"]
    else:
        lines = [f"  @AGENT|\u0394pos|quantize={QUANTIZE_GRID}"]
    lines.extend(changed_lines)
    return lines, new_snapshot


class AgentSerializerState:
    """Tracks per-species agent snapshots for delta encoding."""

    def __init__(self):
        self._snapshots: dict[int, dict[int, tuple[int, int, int]]] = {}

    def serialize(
        self,
        species_sid: int,
        manager: AgentManager,
        is_keyframe: bool,
        rng: Optional[np.random.Generator] = None,
        compact: bool = False,
    ) -> list[str]:
        """Serialize agent block for a species. Returns protocol lines."""
        if not manager.agents and not manager.event_log.deaths:
            return []

        if is_keyframe:
            lines = serialize_agents_keyframe(manager, rng, compact=compact)
            # Build snapshot
            snapshot = {}
            for a in manager.agents:
                snapshot[a.aid] = (
                    quantize_position(a.x),
                    quantize_position(a.y),
                    int(round(a.energy)),
                )
            self._snapshots[species_sid] = snapshot
            return lines
        else:
            prev = self._snapshots.get(species_sid, {})
            lines, new_snap = serialize_agents_delta(manager, prev, rng, compact=compact)
            self._snapshots[species_sid] = new_snap
            return lines
