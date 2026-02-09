"""
Biome definitions for the ecological evolution simulator.

Each biome is a distinct location with vegetation (logistic regrowth),
detritus accumulation from dead organisms, solar energy, and
environmental axes that drift over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import ENV_AXES
from .spatial import VegetationPatch


BIOME_NAMES = [
    "shallows", "reef", "deep_ocean", "tidal_pools", "mangrove",
    "savanna", "forest", "rainforest", "desert", "tundra",
    "cave", "volcanic_vent", "meadow", "swamp", "alpine",
]


@dataclass
class Biome:
    """A distinct location in the simulated world."""

    lid: int
    name: str
    vegetation: float = 0.5       # [0, 1] — regrows logistically
    detritus: float = 0.0         # accumulates from dead organisms
    solar: float = 1.0            # base solar energy availability
    env: dict[str, float] = field(default_factory=dict)

    # Vegetation patches for agent-based simulation
    vegetation_patches: list[VegetationPatch] = field(default_factory=list)

    # Logistic regrowth parameters
    veg_growth_rate: float = 0.1
    veg_capacity: float = 1.0

    def regenerate(self, rng: np.random.Generator) -> None:
        """Logistic vegetation regrowth + detritus decay."""
        # Logistic regrowth: dV/dt = r * V * (1 - V/K)
        growth = self.veg_growth_rate * self.vegetation * (1.0 - self.vegetation / self.veg_capacity)
        self.vegetation = float(np.clip(self.vegetation + growth + rng.normal(0, 0.01), 0.0, self.veg_capacity))

        # Detritus decays slowly
        self.detritus = max(0.0, self.detritus * 0.95 + rng.normal(0, 0.5))

    def drift_env(self, rng: np.random.Generator, sigma: float = 0.02) -> None:
        """Small random perturbation of environmental axes."""
        for ax in self.env:
            self.env[ax] = float(np.clip(self.env[ax] + rng.normal(0, sigma), 0.0, 1.0))

    def add_detritus(self, amount: float) -> None:
        """Add detritus from dead organisms."""
        self.detritus += max(0.0, amount)

    def init_vegetation_patches(self, rng: np.random.Generator, world_size: int, n_patches: int = 8) -> None:
        """Create spatial vegetation patches for agent-based simulation."""
        self.vegetation_patches = [
            VegetationPatch.create_random(rng, world_size, biome_veg=self.vegetation)
            for _ in range(n_patches)
        ]

    def get_vegetation_at(self, x: float, y: float) -> float:
        """Get vegetation density at a specific position (max across patches)."""
        if not self.vegetation_patches:
            return self.vegetation
        best = 0.0
        for p in self.vegetation_patches:
            d = p.get_density_at(x, y)
            if d > best:
                best = d
        return best

    def regenerate_patches(self, dt: float = 1.0) -> None:
        """Regenerate all vegetation patches."""
        for p in self.vegetation_patches:
            p.regenerate(dt)

    def serialize_header(self) -> str:
        """Serialize biome state for protocol output."""
        return f"L{self.lid}:{self.name}(veg={self.vegetation:.1f},det={self.detritus:.0f})"

    @staticmethod
    def create_random(lid: int, rng: np.random.Generator, n_env_axes: int = 4) -> Biome:
        """Create a biome with random initial conditions."""
        name = BIOME_NAMES[lid % len(BIOME_NAMES)]
        axes = list(rng.choice(ENV_AXES, size=min(n_env_axes, len(ENV_AXES)), replace=False))
        env = {ax: float(rng.uniform(0.1, 0.9)) for ax in axes}

        # Biome-specific defaults
        solar = float(rng.uniform(0.3, 1.0))
        if name in ("cave", "deep_ocean"):
            solar = float(rng.uniform(0.0, 0.2))
        elif name == "volcanic_vent":
            solar = float(rng.uniform(0.0, 0.1))
            env["temp"] = max(env.get("temp", 0.5), 0.7)

        vegetation = float(rng.uniform(0.2, 0.9))
        if name in ("desert", "tundra", "deep_ocean", "volcanic_vent"):
            vegetation = float(rng.uniform(0.05, 0.3))
        elif name in ("rainforest", "swamp", "meadow"):
            vegetation = float(rng.uniform(0.6, 1.0))

        return Biome(
            lid=lid,
            name=name,
            vegetation=vegetation,
            detritus=float(rng.uniform(0, 50)),
            solar=solar,
            env=env,
        )
