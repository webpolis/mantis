"""
Species foundation classes for the ecological evolution simulator.

Classes:
    TraitDistribution — mean ± variance for a single trait
    DietVector — normalized feeding preference distribution
    BodyPlan — morphological constraint checker with transition logic
    Species — full species state
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import (
    BODY_PLANS,
    BODY_PLAN_TRANSITIONS,
    FUSION_RULES,
    PREREQUISITES,
    TRAIT_TO_TIER,
)

# TYPE_CHECKING avoids circular import (agent.py imports species types)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import AgentManager
    from .agent_reconciliation import PopulationReconciler


# ---------------------------------------------------------------------------
# TraitDistribution
# ---------------------------------------------------------------------------

@dataclass
class TraitDistribution:
    """Population-level distribution for a single trait (0-10 scale)."""

    mean: float
    variance: float

    def __post_init__(self):
        self.mean = float(np.clip(self.mean, 0.0, 10.0))
        self.variance = float(max(self.variance, 0.01))

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> float | np.ndarray:
        """Draw from normal(mean, sqrt(variance)), clamped to [0, 10]."""
        gen = rng or np.random.default_rng()
        values = gen.normal(self.mean, math.sqrt(self.variance), size=n)
        values = np.clip(values, 0.0, 10.0)
        if n == 1:
            return float(values[0])
        return values

    def shift(self, delta: float) -> None:
        """Move mean by *delta*, clamped to [0, 10]."""
        self.mean = float(np.clip(self.mean + delta, 0.0, 10.0))

    def narrow(self, factor: float) -> None:
        """Stabilizing selection — reduce variance."""
        self.variance = float(max(self.variance * (1.0 - factor), 0.01))

    def widen(self, factor: float) -> None:
        """Disruptive selection — increase variance."""
        self.variance = float(self.variance * (1.0 + factor))

    def __repr__(self) -> str:
        return f"{self.mean:.1f}±{math.sqrt(self.variance):.2f}"


# ---------------------------------------------------------------------------
# DietVector
# ---------------------------------------------------------------------------

class DietVector:
    """Normalized distribution over food sources.

    Sources are arbitrary strings: ``"solar"``, ``"plant"``, ``"detritus"``,
    ``"chemical"``, or species IDs like ``"S3"``.
    """

    # Category mapping for body-plan transition checks
    _MEAT_PREFIXES = ("S",)  # species IDs start with S

    def __init__(self, sources: Optional[dict[str, float]] = None):
        self._d: dict[str, float] = dict(sources) if sources else {}
        if self._d:
            self.normalize()

    # -- core operations ---------------------------------------------------

    def normalize(self) -> None:
        total = sum(self._d.values())
        if total > 0:
            for k in self._d:
                self._d[k] /= total

    def mutate(self, rng: np.random.Generator, magnitude: float = 0.05) -> None:
        """Small random perturbation, re-normalized."""
        if not self._d:
            return
        for k in list(self._d):
            self._d[k] = max(0.0, self._d[k] + rng.normal(0, magnitude))
        # Drop near-zero entries
        self._d = {k: v for k, v in self._d.items() if v > 1e-4}
        if self._d:
            self.normalize()

    def add_source(self, source: str, initial_proportion: float = 0.1) -> None:
        if source in self._d:
            return
        # Scale existing proportions down to make room
        scale = 1.0 - initial_proportion
        for k in self._d:
            self._d[k] *= scale
        self._d[source] = initial_proportion
        self.normalize()

    def remove_source(self, source: str) -> None:
        if source not in self._d:
            return
        del self._d[source]
        if self._d:
            self.normalize()

    # -- queries -----------------------------------------------------------

    def get(self, source: str, default: float = 0.0) -> float:
        return self._d.get(source, default)

    def sources(self) -> dict[str, float]:
        return dict(self._d)

    def get_category_totals(self) -> dict[str, float]:
        """Aggregate into high-level categories for body plan checks."""
        cats: dict[str, float] = {"solar": 0.0, "plant": 0.0, "meat": 0.0, "detritus": 0.0, "chemical": 0.0}
        for src, prop in self._d.items():
            if src == "solar":
                cats["solar"] += prop
            elif src == "plant":
                cats["plant"] += prop
            elif src == "detritus":
                cats["detritus"] += prop
            elif src == "chemical":
                cats["chemical"] += prop
            elif src.startswith(self._MEAT_PREFIXES):
                cats["meat"] += prop
            else:
                # Unknown sources count as plant by default
                cats["plant"] += prop
        return cats

    def dominant_source(self) -> str:
        if not self._d:
            return ""
        return max(self._d, key=self._d.get)

    def specialization(self) -> float:
        """Herfindahl index: 0 = uniform, 1 = monophagous."""
        if not self._d:
            return 0.0
        return sum(v * v for v in self._d.values())

    def has_parasite_source(self) -> bool:
        """True if any meat source contributes > 0 but species also has camo/deception style."""
        # Simple heuristic: any species-targeted source flagged with 'parasite_' prefix
        return any(k.startswith("parasite_") for k in self._d)

    def __repr__(self) -> str:
        parts = [f"{k}:{v:.2f}" for k, v in sorted(self._d.items(), key=lambda x: -x[1])]
        return "{" + ",".join(parts) + "}"

    def __len__(self) -> int:
        return len(self._d)


# ---------------------------------------------------------------------------
# BodyPlan
# ---------------------------------------------------------------------------

class BodyPlan:
    """Morphological constraint checker.

    Wraps a body plan name and provides methods to check trait legality,
    caps, and transitions.
    """

    VALID_NAMES = tuple(BODY_PLANS.keys())

    def __init__(self, name: str):
        if name not in BODY_PLANS:
            raise ValueError(f"Unknown body plan: {name!r}. Valid: {self.VALID_NAMES}")
        self.name = name
        self._cfg = BODY_PLANS[name]

    @property
    def base_metabolism(self) -> float:
        return self._cfg["base_metabolism"]

    def can_evolve(self, trait_name: str) -> bool:
        return trait_name not in self._cfg["blocked"]

    def get_cap(self, trait_name: str) -> Optional[int]:
        return self._cfg["caps"].get(trait_name)

    def check_transition(
        self,
        diet: DietVector,
        traits: dict[str, TraitDistribution],
    ) -> Optional[BodyPlan]:
        """Return a new BodyPlan if a transition is triggered, else None.

        Evaluates transitions in order; first match wins.
        """
        cats = diet.get_category_totals()
        for from_plan, to_plan, conds in BODY_PLAN_TRANSITIONS:
            if from_plan != self.name:
                continue
            if _check_transition_conds(conds, cats, traits, diet):
                return BodyPlan(to_plan)
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BodyPlan):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"BodyPlan({self.name!r})"


def _check_transition_conds(
    conds: dict,
    cats: dict[str, float],
    traits: dict[str, TraitDistribution],
    diet: DietVector,
) -> bool:
    """Evaluate a single transition's condition dict."""
    for key, threshold in conds.items():
        if key == "has_speed":
            if "speed" not in traits or traits["speed"].mean <= 0:
                return False
        elif key == "parasite_feeding":
            if not diet.has_parasite_source():
                return False
        elif key == "detritus_min":
            if cats.get("detritus", 0) < threshold:
                return False
        elif key == "meat_min":
            if cats.get("meat", 0) < threshold:
                return False
        elif key == "plant_min":
            if cats.get("plant", 0) < threshold:
                return False
        elif key == "solar_max":
            if cats.get("solar", 0) > threshold:
                return False
        elif key == "meat_plus_detritus_min":
            if cats.get("meat", 0) + cats.get("detritus", 0) < threshold:
                return False
    return True


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

@dataclass
class Species:
    """Full species state for the simulation."""

    sid: int
    traits: dict[str, TraitDistribution] = field(default_factory=dict)
    fused_traits: dict[str, TraitDistribution] = field(default_factory=dict)
    body_plan: BodyPlan = field(default_factory=lambda: BodyPlan("grazer"))
    diet: DietVector = field(default_factory=DietVector)
    population: int = 100
    energy_store: float = 0.0
    locations: set[str] = field(default_factory=set)
    age: int = 0
    alive: bool = True

    # Agent-based simulation (None = population-level only)
    agent_manager: Optional[AgentManager] = field(default=None, repr=False)
    reconciler: Optional[PopulationReconciler] = field(default=None, repr=False)

    # -- derived properties ------------------------------------------------

    def reproduction_strategy(self) -> str:
        """'r' (many offspring, low investment) or 'K' (few, high investment)."""
        repro_mean = self.traits["repro"].mean if "repro" in self.traits else 0.0
        social_mean = self.traits["social"].mean if "social" in self.traits else 0.0
        # r-strategy: high repro, low social.  K-strategy: low repro, high social.
        if repro_mean > social_mean + 2:
            return "r"
        return "K"

    def spotlight_score(self) -> float:
        """Score that determines whether a species triggers spotlight events."""
        intel = self.traits["intel"].mean if "intel" in self.traits else 0.0
        social = self.traits["social"].mean if "social" in self.traits else 0.0
        lang = self.traits["language"].mean if "language" in self.traits else 0.0
        return intel * social * (1.0 + lang * 0.5)

    def has_prerequisites(self, trait: str) -> bool:
        """Check if this species meets all prerequisites for *trait*."""
        if trait not in PREREQUISITES:
            return True
        all_t = dict(self.traits)
        all_t.update(self.fused_traits)
        for req_trait, req_val in PREREQUISITES[trait]:
            td = all_t.get(req_trait)
            if td is None or td.mean < req_val:
                return False
        return True

    def apply_body_plan_constraints(self) -> None:
        """Enforce blocked traits and caps from the current body plan."""
        blocked = BODY_PLANS[self.body_plan.name]["blocked"]
        caps = BODY_PLANS[self.body_plan.name]["caps"]

        # Remove blocked base traits
        for t in list(self.traits):
            if t in blocked:
                del self.traits[t]

        # Remove fused traits whose component traits are both blocked
        for tA, tB, fused_name, _ in FUSION_RULES:
            if fused_name in self.fused_traits:
                if tA in blocked or tB in blocked:
                    del self.fused_traits[fused_name]

        # Enforce caps
        for t, cap in caps.items():
            for container in (self.traits, self.fused_traits):
                if t in container and container[t].mean > cap:
                    container[t].mean = float(cap)

    def check_body_plan_transition(self) -> Optional[BodyPlan]:
        """Check if diet/traits trigger a body plan transition."""
        return self.body_plan.check_transition(self.diet, self.traits)

    def get_all_traits(self) -> dict[str, TraitDistribution]:
        """Merged view of base + fused traits."""
        merged = dict(self.traits)
        merged.update(self.fused_traits)
        return merged

    def max_tier(self) -> int:
        """Highest trait tier present with mean > 0."""
        best = 0
        for t, td in self.get_all_traits().items():
            if td.mean > 0:
                tier = TRAIT_TO_TIER.get(t, 0)
                if tier > best:
                    best = tier
        return best
