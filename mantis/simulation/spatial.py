"""
Spatial indexing and vegetation patches for agent-based simulation.

SpatialHash — O(k) neighbor queries via grid-based spatial hashing.
VegetationPatch — localized food sources with depletion and regrowth.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .agent import Agent


# ---------------------------------------------------------------------------
# Spatial Hash
# ---------------------------------------------------------------------------

class SpatialHash:
    """Grid-based spatial index for efficient neighbor queries.

    Divides a square world into cells. Queries check the 3x3 neighborhood
    around the target position, reducing O(n^2) to O(k) where k ~ agents
    in nearby cells.

    Cell size is set to match the maximum sense range so that a 3x3
    neighborhood is always sufficient — no multi-layer lookups needed.
    """

    def __init__(self, world_size: int, cell_size: int = 100):
        self.world_size = world_size
        self.cell_size = cell_size
        self.grid: dict[tuple[int, int], list[Agent]] = defaultdict(list)

    def clear(self) -> None:
        self.grid.clear()

    def insert(self, agent: Agent) -> None:
        cell = self._get_cell(agent.x, agent.y)
        self.grid[cell].append(agent)

    def rebuild(self, agents: list[Agent]) -> None:
        """Clear and reinsert all agents."""
        self.clear()
        for a in agents:
            self.insert(a)

    def query_radius(self, x: float, y: float, radius: float) -> list[Agent]:
        """Return agents within *radius* of (x, y)."""
        r2 = radius * radius
        result = []
        for cell in self._get_cells_in_radius(x, y, radius):
            for a in self.grid.get(cell, ()):
                dx = a.x - x
                dy = a.y - y
                if dx * dx + dy * dy <= r2:
                    result.append(a)
        return result

    def _get_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _get_cells_in_radius(self, x: float, y: float, radius: float) -> list[tuple[int, int]]:
        cx, cy = self._get_cell(x, y)
        # Number of cells to extend in each direction
        span = max(1, int(math.ceil(radius / self.cell_size)))
        cells = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                cells.append((cx + dx, cy + dy))
        return cells


# ---------------------------------------------------------------------------
# Vegetation Patch
# ---------------------------------------------------------------------------

@dataclass
class VegetationPatch:
    """Spatial distribution of vegetation within a biome.

    Large patches with slow depletion prevent instant-collapse oscillations.
    A patch can sustain ~50 grazers for 10+ ticks before noticeable
    depletion, creating spatial strategy: good patches attract agents
    and cause competition.
    """

    x: float
    y: float
    density: float          # 0.0 (depleted) to 1.0 (lush)
    radius: float           # area of influence (typically 80-120 units)
    capacity: float         # total food units available (e.g., 500)
    max_capacity: float     # original max for normalization
    regen_rate: float       # regrowth speed (slow: 0.05-0.1/tick)

    def get_density_at(self, px: float, py: float) -> float:
        """Gaussian falloff from center."""
        dx = px - self.x
        dy = py - self.y
        dist2 = dx * dx + dy * dy
        r2 = self.radius * self.radius
        if dist2 > r2:
            return 0.0
        falloff = math.exp(-dist2 / r2)
        return self.density * falloff

    def try_forage(self, amount: float) -> float:
        """Agent attempts to forage. Returns actual amount consumed."""
        actual = min(amount, self.capacity)
        self.capacity = max(0.0, self.capacity - actual)
        self.density = self.capacity / self.max_capacity if self.max_capacity > 0 else 0.0
        return actual

    def regenerate(self, dt: float = 1.0) -> None:
        """Logistic regrowth. Slow to prevent instant respawn oscillation."""
        if self.capacity < self.max_capacity:
            growth = self.regen_rate * dt * (1.0 - self.capacity / self.max_capacity)
            self.capacity = min(self.max_capacity, self.capacity + growth * self.max_capacity * 0.2)
            self.density = self.capacity / self.max_capacity if self.max_capacity > 0 else 0.0

    @staticmethod
    def create_random(
        rng: np.random.Generator,
        world_size: int,
        biome_veg: float = 0.5,
    ) -> VegetationPatch:
        """Create a random vegetation patch."""
        max_cap = float(rng.uniform(300, 700))
        return VegetationPatch(
            x=float(rng.uniform(0, world_size)),
            y=float(rng.uniform(0, world_size)),
            density=float(rng.uniform(0.4, 1.0)) * biome_veg,
            radius=float(rng.uniform(80, 120)),
            capacity=max_cap * biome_veg,
            max_capacity=max_cap,
            regen_rate=float(rng.uniform(0.05, 0.1)),
        )
