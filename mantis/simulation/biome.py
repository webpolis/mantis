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

from .constants import ENV_AXES, NUTRIENT_RELEASE_N, NUTRIENT_RELEASE_P, NUTRIENT_UPTAKE_RATE
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
    nitrogen: float = 0.5         # [0, 1] — released from detritus by decomposers
    phosphorus: float = 0.3       # [0, 1] — released more slowly from detritus
    env: dict[str, float] = field(default_factory=dict)

    # Vegetation patches for agent-based simulation
    vegetation_patches: list[VegetationPatch] = field(default_factory=list)

    # Logistic regrowth parameters
    veg_growth_rate: float = 0.1
    veg_capacity: float = 1.0

    # Minimum vegetation floor: seed bank / spore dispersal allows recovery
    VEG_SEED_BANK = 0.005

    def regenerate(self, rng: np.random.Generator, veg_growth_mult: float = 1.0) -> None:
        """Logistic vegetation regrowth + detritus decay + nutrient cycling."""
        # Seed bank: even fully stripped biomes recover slowly via spores/seeds
        if self.vegetation < self.VEG_SEED_BANK:
            self.vegetation = self.VEG_SEED_BANK

        # Nutrient-limited logistic regrowth: dV/dt = r * V * (1 - V/K) * min(N, P, 1)
        nutrient_factor = min(self.nitrogen, self.phosphorus, 1.0)
        effective_rate = self.veg_growth_rate * veg_growth_mult
        growth = effective_rate * self.vegetation * (1.0 - self.vegetation / self.veg_capacity) * nutrient_factor
        growth = max(0.0, growth)  # no negative growth from nutrients
        self.vegetation = float(np.clip(self.vegetation + growth + rng.normal(0, 0.01), 0.0, self.veg_capacity))

        # Consume nutrients proportional to growth
        self.nitrogen = float(np.clip(self.nitrogen - growth * NUTRIENT_UPTAKE_RATE, 0.0, 1.0))
        self.phosphorus = float(np.clip(self.phosphorus - growth * NUTRIENT_UPTAKE_RATE, 0.0, 1.0))

        # Detritus decays slowly — releases nutrients back
        detritus_decay = self.detritus * 0.05
        self.detritus = max(0.0, self.detritus * 0.95 + rng.normal(0, 0.5))
        self.nitrogen = float(np.clip(self.nitrogen + detritus_decay * NUTRIENT_RELEASE_N, 0.0, 1.0))
        self.phosphorus = float(np.clip(self.phosphorus + detritus_decay * NUTRIENT_RELEASE_P, 0.0, 1.0))

        # Geological buffer: slow baseline nutrient release prevents permanent
        # nutrient lockup if all decomposers die
        self.nitrogen = float(np.clip(self.nitrogen + 0.002, 0.0, 1.0))
        self.phosphorus = float(np.clip(self.phosphorus + 0.001, 0.0, 1.0))

    def drift_env(self, rng: np.random.Generator, sigma: float = 0.02) -> None:
        """Small random perturbation of environmental axes."""
        for ax in self.env:
            self.env[ax] = float(np.clip(self.env[ax] + rng.normal(0, sigma), 0.0, 1.0))

    def add_detritus(self, amount: float) -> None:
        """Add detritus from dead organisms."""
        self.detritus += max(0.0, amount)

    def consume_vegetation(self, amount: float) -> float:
        """Consume vegetation and return actual amount taken."""
        taken = min(amount, self.vegetation)
        self.vegetation -= taken
        return taken

    def consume_detritus(self, amount: float) -> float:
        """Consume detritus and return actual amount taken."""
        taken = min(amount, self.detritus)
        self.detritus -= taken
        return taken

    def release_nutrients(self, detritus_consumed: float) -> None:
        """Release nutrients from decomposer activity on detritus."""
        self.nitrogen = float(np.clip(self.nitrogen + detritus_consumed * NUTRIENT_RELEASE_N, 0.0, 1.0))
        self.phosphorus = float(np.clip(self.phosphorus + detritus_consumed * NUTRIENT_RELEASE_P, 0.0, 1.0))

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
        return f"L{self.lid}:{self.name}(veg={self.vegetation:.1f},det={self.detritus:.0f},N={self.nitrogen:.1f},P={self.phosphorus:.1f})"

    def serialize_header_compact(self) -> str:
        """Serialize biome state in compact v2 format: L{lid} {name} {veg×100} {det} N{n×10} P{p×10}."""
        return (f"L{self.lid} {self.name} {int(round(self.vegetation * 100))} {int(round(self.detritus))} "
                f"N{int(round(self.nitrogen * 10))} P{int(round(self.phosphorus * 10))}")

    @staticmethod
    def create_random(lid: int, rng: np.random.Generator, n_env_axes: int = 4,
                      name: str | None = None) -> Biome:
        """Create a biome with random initial conditions."""
        name = name or BIOME_NAMES[lid % len(BIOME_NAMES)]
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

        # Nutrient initialization based on biome fertility
        if name in ("rainforest", "swamp", "meadow", "mangrove"):
            nitrogen = float(rng.uniform(0.6, 0.9))
            phosphorus = float(rng.uniform(0.4, 0.7))
        elif name in ("desert", "tundra", "deep_ocean", "volcanic_vent"):
            nitrogen = float(rng.uniform(0.1, 0.3))
            phosphorus = float(rng.uniform(0.05, 0.2))
        else:
            nitrogen = float(rng.uniform(0.3, 0.7))
            phosphorus = float(rng.uniform(0.2, 0.5))

        return Biome(
            lid=lid,
            name=name,
            vegetation=vegetation,
            detritus=float(rng.uniform(0, 50)),
            solar=solar,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            env=env,
        )
