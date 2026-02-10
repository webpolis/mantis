"""
Agent-based simulation layer for individual organisms.

Agent — individual organism with position, energy, traits, behavior state.
AgentManager — manages agents for a single species (spawning, stepping, pruning).
EventLog — tracks discrete events (births, deaths) and energy deltas for reconciliation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from .constants import BRAIN_TAX, TRAIT_TO_TIER
from .spatial import SpatialHash, VegetationPatch

if TYPE_CHECKING:
    from .species import Species


# ---------------------------------------------------------------------------
# Event Log (for reconciliation with macro population)
# ---------------------------------------------------------------------------

@dataclass
class EventLog:
    """Collects per-tick discrete events and energy deltas."""
    deaths: list[int] = field(default_factory=list)       # agent IDs that died
    births: list[int] = field(default_factory=list)        # agent IDs born
    energy_deltas: dict[int, float] = field(default_factory=dict)  # aid -> delta

    def clear(self) -> None:
        self.deaths.clear()
        self.births.clear()
        self.energy_deltas.clear()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Individual organism in agent-based simulation."""

    # Identity
    aid: int
    species_sid: int
    biome_lid: int

    # Spatial state
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

    # Biological state
    energy: float = 50.0
    age: int = 0
    traits: dict[str, float] = field(default_factory=dict)
    alive: bool = True

    # Behavior state
    state: str = "rest"
    state_commitment: int = 0
    target_aid: Optional[int] = None
    target_sid: Optional[int] = None
    prev_energy: float = 50.0


# ---------------------------------------------------------------------------
# Agent Manager
# ---------------------------------------------------------------------------

# Agent configuration constants
AGENT_MAX_PER_SPECIES = 250
AGENT_MIN_PER_SPECIES = 20
AGENT_TOTAL_MAX = 2000
WORLD_SIZE_DEFAULT = 1000
CELL_SIZE = 100           # matches max sense range


class AgentManager:
    """Manages agents for a single species."""

    def __init__(
        self,
        species_sid: int,
        biome_lid: int,
        world_size: int = WORLD_SIZE_DEFAULT,
        base_metabolism: float = 1.0,
    ):
        self.species_sid = species_sid
        self.biome_lid = biome_lid
        self.world_size = world_size
        self.base_metabolism = base_metabolism  # from body plan
        self.agents: list[Agent] = []
        self.spatial_hash = SpatialHash(world_size, cell_size=CELL_SIZE)
        self.next_aid = 0
        self.event_log = EventLog()
        self.simple_mode = False  # True = ECOSYSTEM (basic flocking/grazing only)

    def spawn_agents(
        self,
        count: int,
        species: Species,
        rng: np.random.Generator,
    ) -> None:
        """Create *count* agents, sampling traits from species distributions."""
        all_traits = species.get_all_traits()
        for _ in range(count):
            traits = {}
            for name, dist in all_traits.items():
                traits[name] = float(dist.sample(n=1, rng=rng))

            agent = Agent(
                aid=self.next_aid,
                species_sid=self.species_sid,
                biome_lid=self.biome_lid,
                x=float(rng.uniform(0, self.world_size)),
                y=float(rng.uniform(0, self.world_size)),
                energy=50.0,
                age=0,
                traits=traits,
                state="rest",
            )
            agent.prev_energy = agent.energy
            self.agents.append(agent)
            self.spatial_hash.insert(agent)
            self.next_aid += 1

    def step(
        self,
        all_managers: dict[int, AgentManager],
        vegetation_patches: list[VegetationPatch],
        prey_sids: set[int],
        predator_sids: set[int],
        rng: np.random.Generator,
        dt: float = 1.0,
    ) -> None:
        """Execute one simulation tick for this species' agents.

        1. Record prev_energy for delta tracking
        2. Update behaviors (utility system)
        3. Move agents (steering)
        4. Resolve local interactions (forage, hunt, flee)
        5. Apply metabolism cost
        6. Prune dead agents
        """
        from .behavior import update_behaviors, compute_velocity

        self.event_log.clear()

        if not self.agents:
            return

        # Record prev energy for reconciliation
        for a in self.agents:
            a.prev_energy = a.energy

        # Rebuild spatial hash
        self.spatial_hash.rebuild(self.agents)

        # Gather all agents from other species for cross-species queries
        all_agents_by_sid: dict[int, list[Agent]] = {}
        for sid, mgr in all_managers.items():
            all_agents_by_sid[sid] = mgr.agents

        # 1. Update behaviors
        update_behaviors(
            self.agents,
            self.spatial_hash,
            all_managers,
            vegetation_patches,
            prey_sids,
            predator_sids,
            rng,
            simple_mode=self.simple_mode,
        )

        # 2. Move agents
        for a in self.agents:
            if not a.alive:
                continue
            a.x += a.vx * dt
            a.y += a.vy * dt
            # Clamp to world bounds
            a.x = max(0.0, min(float(self.world_size - 1), a.x))
            a.y = max(0.0, min(float(self.world_size - 1), a.y))

        # 3. Resolve local interactions
        self._resolve_interactions(all_managers, vegetation_patches, prey_sids, rng)

        # 4. Metabolism cost (matches population-level formula in engine.py)
        for a in self.agents:
            if not a.alive:
                continue
            size = a.traits.get("size", 1.0)
            # Basal: body_plan base_metabolism * Kleiber scaling
            basal = self.base_metabolism * (size ** 0.75) * 0.1
            # Brain tax: cognitive traits cost energy (same as population-level)
            brain = 0.0
            for trait_name, val in a.traits.items():
                tier = TRAIT_TO_TIER.get(trait_name, 0)
                if tier > 0:
                    brain += val * BRAIN_TAX.get(tier, 0.0)
            # Movement: velocity-based
            movement = math.sqrt(a.vx * a.vx + a.vy * a.vy) * 0.02
            a.energy -= basal + brain + movement
            a.age += 1

        # 5. Prune dead
        self._prune_dead()

        # 6. Track energy deltas
        for a in self.agents:
            self.event_log.energy_deltas[a.aid] = a.energy - a.prev_energy

    def _resolve_interactions(
        self,
        all_managers: dict[int, AgentManager],
        vegetation_patches: list[VegetationPatch],
        prey_sids: set[int],
        rng: np.random.Generator,
    ) -> None:
        """Resolve foraging, hunting, and mating for this species' agents."""
        for a in self.agents:
            if not a.alive:
                continue

            if a.state == "forage":
                self._resolve_forage(a, vegetation_patches)

            elif a.state == "hunt" and a.target_aid is not None:
                self._resolve_hunt(a, all_managers, prey_sids, rng)

            elif a.state == "mate":
                self._resolve_mate(a, rng)

    def _resolve_forage(self, a: Agent, patches: list[VegetationPatch]) -> None:
        """Agent forages from nearest vegetation patch."""
        best_density = 0.0
        best_patch = None
        for p in patches:
            d = p.get_density_at(a.x, a.y)
            if d > best_density:
                best_density = d
                best_patch = p

        if best_patch is not None and best_density > 0.01:
            amount = best_density * a.traits.get("size", 1.0) * 0.5
            gained = best_patch.try_forage(amount)
            a.energy += gained

    def _resolve_hunt(
        self,
        a: Agent,
        all_managers: dict[int, AgentManager],
        prey_sids: set[int],
        rng: np.random.Generator,
    ) -> None:
        """Predator tries to catch and kill target prey."""
        # Find target agent in the specific prey species
        target = None
        target_mgr = None
        if a.target_sid is not None:
            mgr = all_managers.get(a.target_sid)
            if mgr is not None:
                for prey_a in mgr.agents:
                    if prey_a.aid == a.target_aid and prey_a.alive:
                        target = prey_a
                        target_mgr = mgr
                        break

        if target is None or not target.alive:
            a.target_aid = None
            a.target_sid = None
            return

        # Check if close enough to strike
        dx = target.x - a.x
        dy = target.y - a.y
        dist = math.sqrt(dx * dx + dy * dy)

        strike_range = 15.0  # units
        if dist > strike_range:
            return

        # Success probability based on traits
        pred_speed = a.traits.get("speed", 0.0)
        prey_speed = target.traits.get("speed", 0.0)
        pred_sense = a.traits.get("sense", 0.0)
        prey_camo = target.traits.get("camo", 0.0)
        advantage = (pred_speed - prey_speed * 0.8) + (pred_sense - prey_camo * 0.7)
        success_prob = min(0.9, max(0.1, 0.3 + advantage * 0.05))

        if rng.random() < success_prob:
            # Kill prey
            prey_size = target.traits.get("size", 1.0)
            energy_gain = prey_size * 0.12 * 100.0  # trophic efficiency
            a.energy += energy_gain
            target.alive = False
            target_mgr.event_log.deaths.append(target.aid)
            a.target_aid = None
            a.target_sid = None

    def _resolve_mate(self, a: Agent, rng: np.random.Generator) -> None:
        """Mating interaction — find nearby conspecific and reproduce."""
        if a.energy < 70:
            return

        # Find nearby mate
        neighbors = self.spatial_hash.query_radius(a.x, a.y, 30.0)
        mates = [
            n for n in neighbors
            if n.aid != a.aid and n.alive and n.energy > 60
            and n.species_sid == a.species_sid
        ]
        if not mates:
            return

        mate = mates[int(rng.integers(len(mates)))]
        # Both lose energy
        a.energy -= 20
        mate.energy -= 20

        # Spawn offspring
        child_traits = {}
        for t in a.traits:
            if t in mate.traits:
                child_traits[t] = (a.traits[t] + mate.traits[t]) / 2.0 + float(rng.normal(0, 0.3))
                child_traits[t] = max(0.0, min(10.0, child_traits[t]))
            else:
                child_traits[t] = a.traits[t]

        child = Agent(
            aid=self.next_aid,
            species_sid=self.species_sid,
            biome_lid=self.biome_lid,
            x=a.x + float(rng.normal(0, 5)),
            y=a.y + float(rng.normal(0, 5)),
            energy=30.0,
            age=0,
            traits=child_traits,
            state="rest",
        )
        child.x = max(0.0, min(float(self.world_size - 1), child.x))
        child.y = max(0.0, min(float(self.world_size - 1), child.y))
        child.prev_energy = child.energy
        self.agents.append(child)
        self.event_log.births.append(child.aid)
        self.next_aid += 1

    def _prune_dead(self) -> None:
        """Remove dead agents (energy <= 0 or flagged)."""
        survivors = []
        for a in self.agents:
            if a.energy <= 0:
                a.alive = False
            if a.alive:
                survivors.append(a)
            else:
                if a.aid not in self.event_log.deaths:
                    self.event_log.deaths.append(a.aid)
        self.agents = survivors

