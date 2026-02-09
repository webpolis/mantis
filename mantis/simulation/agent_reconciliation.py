"""
Dual-layer accounting: reconcile agent-level events with macro population.

Discrete events (births, deaths, hunts) scale 1:1.
Continuous processes (foraging, metabolism) scale by population ratio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgentManager
    from .species import Species


class PopulationReconciler:
    """Manages bidirectional sync between agents and macro population."""

    def __init__(self, species: Species, agent_count: int):
        self.species = species
        self.scaling = species.population / max(1, agent_count)

    def reconcile_tick(self, manager: AgentManager) -> None:
        """Called after each agent tick to update macro stats."""
        events = manager.event_log
        agents = manager.agents

        if not agents and not events.deaths:
            return

        # 1. DISCRETE EVENTS (do not scale)
        deaths = len(events.deaths)
        births = len(events.births)
        self.species.population += (births - deaths)
        self.species.population = max(0, self.species.population)

        if self.species.population == 0:
            self.species.alive = False
            return

        # 2. CONTINUOUS ENERGY (scale aggregate change)
        if agents and events.energy_deltas:
            total_delta = sum(events.energy_deltas.values())
            avg_delta = total_delta / max(1, len(agents))
            self.species.energy_store += avg_delta * self.scaling
            self.species.energy_store = max(0.0, self.species.energy_store)

        # Update scaling factor for next tick
        self.scaling = self.species.population / max(1, len(agents))
