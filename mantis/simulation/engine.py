"""
Core simulation engine for the ecological evolution simulator.

Manages a World containing biomes and species, advancing through
epochs with energy-based population dynamics, trait mutations,
interactions, body plan transitions, and intelligence spotlights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .agent import AgentManager, AGENT_MAX_PER_SPECIES, AGENT_TOTAL_MAX
from .agent_reconciliation import PopulationReconciler
from .biome import Biome, BIOME_NAMES
from .constants import (
    BODY_PLAN_DIET_AFFINITY,
    BODY_PLANS,
    BRAIN_TAX,
    CATASTROPHE_PROB,
    CATASTROPHE_TYPES,
    DIET_MISMATCH_PENALTY,
    DISEASE_BASE_PROB,
    DISEASE_TYPES,
    EPOCH_CONFIG,
    EXTINCTION_CAUSES,
    FAT_STORE_MAX_FACTOR,
    FUSION_RULES,
    GOALS,
    HIGH_VARIANCE_THRESHOLD,
    KLEIBER_EXPONENT,
    LOW_VARIANCE_THRESHOLD,
    PREREQUISITES,
    REPRODUCTION_ENERGY_FRACTION,
    STARVATION_RATE,
    STARVATION_THRESHOLD,
    SYMBIOGENESIS_MIN_TICKS,
    SYMBIOGENESIS_PROB,
    TRAIT_TIERS,
    TRAIT_TO_TIER,
    TROPHIC_EFFICIENCY,
    Epoch,
)
from .species import BodyPlan, DietVector, Species, TraitDistribution


# ---------------------------------------------------------------------------
# Data classes for simulation events
# ---------------------------------------------------------------------------

@dataclass
class Interaction:
    verb: str           # hunt, graze, compete, scavenge, parasitize, pollinate, symbiosis
    actor_sid: int
    target_sid: int
    success: float
    effects: dict = field(default_factory=dict)  # {sid: {key: delta}}


@dataclass
class Hero:
    hid: int
    role: str           # Elder, Warrior, Scout, Youth, Mother
    influence: float


@dataclass
class CulturalMemory:
    name: str
    mtype: str          # taboo, legend, tradition, sacred
    strength: float
    origin_gen: int


@dataclass
class SpotlightEvent:
    world_id: int
    gen: int
    biome_lid: int
    species_sid: int
    heroes: list[Hero]
    intent: str
    react: str
    resolve: str
    effect: str
    cultural_memories: list[CulturalMemory]


# ---------------------------------------------------------------------------
# Primordial body plans (used for initial species)
# ---------------------------------------------------------------------------

PRIMORDIAL_PLANS = [
    "sessile_autotroph", "mobile_autotroph", "filter_feeder", "decomposer",
    "grazer", "scavenger",
]

PRIMORDIAL_DIETS = {
    "sessile_autotroph": {"solar": 1.0},
    "mobile_autotroph": {"solar": 0.7, "plant": 0.3},
    "filter_feeder": {"detritus": 0.6, "plant": 0.4},
    "decomposer": {"detritus": 1.0},
    "grazer": {"plant": 0.8, "detritus": 0.2},
    "scavenger": {"detritus": 0.5, "plant": 0.3, "solar": 0.2},
}

SPOTLIGHT_ROLES = ["Elder", "Warrior", "Scout", "Youth", "Mother"]

SPOTLIGHT_ACTIONS = [
    "report", "warn", "propose", "challenge", "request", "observe",
]

SPOTLIGHT_REACTIONS = [
    "endorse", "reject", "debate", "defer", "counter", "accept",
]

SPOTLIGHT_RESOLUTIONS = [
    "council", "vote", "elder_decree", "trial_by_combat", "consensus", "ritual",
]

SPOTLIGHT_OUTCOMES = [
    "split_colony", "migrate", "build_shelter", "form_alliance",
    "declare_territory", "begin_trade", "exile_member", "adopt_ritual",
    "develop_tool", "share_knowledge", "punish_violator", "honor_hero",
]


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    """Self-contained ecological simulation."""

    def __init__(self, wid: int, seed: int, enable_agents: bool = False,
                 agent_epoch: str = "INTELLIGENCE", agent_threshold: float = 15.0):
        self.wid = wid
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.epoch = Epoch.PRIMORDIAL

        # Agent configuration
        self.enable_agents = enable_agents
        self.agent_epoch = Epoch[agent_epoch] if isinstance(agent_epoch, str) else agent_epoch
        self.agent_threshold = agent_threshold
        self.agents_active = False

        # Create biomes (2-5)
        n_biomes = int(self.rng.integers(2, 6))
        biome_indices = self.rng.choice(len(BIOME_NAMES), size=n_biomes, replace=False)
        self.biomes: list[Biome] = [
            Biome.create_random(i, self.rng, name=BIOME_NAMES[idx])
            for i, idx in enumerate(biome_indices)
        ]

        # Create initial species (3-8, primordial body plans)
        n_species = int(self.rng.integers(3, 9))
        self.species: list[Species] = []
        self.next_sid = 0
        for _ in range(n_species):
            sp = self._create_primordial_species()
            self.species.append(sp)

        # Goals tracked per-species on World (Species dataclass has no goal field)
        self.goals: dict[int, str] = {}
        for sp in self.species:
            self.goals[sp.sid] = str(self.rng.choice(GOALS[:6]))

        # Per-tick energy tracking for serializer
        self.energy_log: dict[int, tuple[float, float]] = {}  # sid -> (e_income, e_cost)

        # Interaction log for current tick
        self.interactions: list[Interaction] = []

        # Spotlight events for current tick
        self.spotlights: list[SpotlightEvent] = []

        # Mutation/event logs for current tick (sid -> list of strings)
        self.mutations: dict[int, list[str]] = {}
        self.events: dict[int, list[str]] = {}

        # Cultural memories per species
        self.cultural_memories: dict[int, list[CulturalMemory]] = {}

        # Epoch transition flag for serializer
        self.epoch_just_changed = False

        # Symbiosis history for symbiogenesis: (sid_a, sid_b) -> co-location ticks
        self.symbiosis_history: dict[tuple[int, int], int] = {}

        # Active catastrophe state
        self.catastrophe: dict | None = None

        # World-level events for current tick (serialized before species blocks)
        self.world_events: list[str] = []

        # Biome adjacency graph: lid -> {lid: permeability}
        self.biome_adjacency: dict[int, dict[int, float]] = {}
        self._build_biome_adjacency()

        # Per-tick caches
        self._biome_cache: dict[int, list[Biome]] = {}
        self._species_by_sid: dict[int, Species] = {sp.sid: sp for sp in self.species}

    def _build_biome_adjacency(self) -> None:
        """Build random biome adjacency graph (chain + extra edges)."""
        for b in self.biomes:
            self.biome_adjacency[b.lid] = {}
        # Chain all biomes in order for guaranteed connectivity
        for i in range(len(self.biomes) - 1):
            a, b = self.biomes[i], self.biomes[i + 1]
            perm = float(self.rng.uniform(0.3, 0.8))
            self.biome_adjacency[a.lid][b.lid] = perm
            self.biome_adjacency[b.lid][a.lid] = perm
        # Add 0-2 random extra edges
        n_extra = int(self.rng.integers(0, 3))
        for _ in range(n_extra):
            if len(self.biomes) < 2:
                break
            a, b = self.rng.choice(self.biomes, size=2, replace=False)
            if b.lid not in self.biome_adjacency[a.lid]:
                perm = float(self.rng.uniform(0.3, 0.8))
                self.biome_adjacency[a.lid][b.lid] = perm
                self.biome_adjacency[b.lid][a.lid] = perm

    def _create_primordial_species(self) -> Species:
        """Create a species with a primordial body plan."""
        plan_name = str(self.rng.choice(PRIMORDIAL_PLANS))
        plan = BodyPlan(plan_name)
        diet = DietVector(dict(PRIMORDIAL_DIETS[plan_name]))

        # Start with 2-4 T0 traits allowed by body plan
        allowed_t0 = [t for t in TRAIT_TIERS[0] if plan.can_evolve(t)]
        n_traits = min(int(self.rng.integers(2, 5)), len(allowed_t0))
        chosen = list(self.rng.choice(allowed_t0, size=n_traits, replace=False))

        traits = {}
        for t in chosen:
            mean = float(self.rng.uniform(1.0, 5.0))
            var = float(self.rng.uniform(0.1, 1.5))
            cap = plan.get_cap(t)
            if cap is not None:
                mean = min(mean, float(cap))
            traits[t] = TraitDistribution(mean, var)

        # Assign to random biome(s)
        n_locs = int(self.rng.integers(1, min(3, len(self.biomes)) + 1))
        locs = set(f"L{b.lid}" for b in self.rng.choice(self.biomes, size=n_locs, replace=False))

        pop = int(self.rng.integers(200, 10000))
        sid = self.next_sid
        self.next_sid += 1

        return Species(
            sid=sid,
            traits=traits,
            body_plan=plan,
            diet=diet,
            population=pop,
            energy_store=float(pop * 0.5),
            locations=locs,
        )

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the world by one tick."""
        self.tick += 1
        self.interactions.clear()
        self.spotlights.clear()
        self.mutations.clear()
        self.events.clear()
        self.energy_log.clear()
        self.world_events.clear()
        self.epoch_just_changed = False

        # Per-tick caches (cleared each tick)
        self._biome_cache: dict[int, list[Biome]] = {}
        self._species_by_sid: dict[int, Species] = {sp.sid: sp for sp in self.species}

        alive = [sp for sp in self.species if sp.alive]
        if not alive:
            return

        # 1. Catastrophe check (before biome regen — modifies biome params)
        self._check_catastrophes()

        # 2. Biome regeneration + env drift (nutrient-limited)
        veg_growth_mult = 1.0
        if self.catastrophe and "veg_growth_mult" in self.catastrophe:
            veg_growth_mult = self.catastrophe["veg_growth_mult"]
        sigma = 0.04 if self.epoch == Epoch.PRIMORDIAL else 0.02
        for biome in self.biomes:
            biome.regenerate(self.rng, veg_growth_mult=veg_growth_mult)
            biome.drift_env(self.rng, sigma)

        # 3. Energy computation (with digestive penalty + nutrient factor)
        self._compute_energy(alive)

        # 4. Interaction resolution (tracks symbiosis history)
        self._resolve_interactions(alive)

        # 5. Population dynamics (low-variance starvation penalty)
        self._update_populations(alive)

        # 6. Disease check
        self._check_disease(alive)

        # 7. Evolution: mutations, trait acquisition, fusion, body plan transitions
        #    (stress-boosted Mleap via variance-as-evolvability)
        self._evolve(alive)

        # 8. Symbiogenesis check
        self._check_symbiogenesis(alive)

        # 9. Speciation (migration friction boost)
        self._try_speciation(alive)

        # 10. Spotlights (INTELLIGENCE epoch only)
        if self.epoch.value >= Epoch.INTELLIGENCE.value:
            self._generate_spotlights(alive)

        # 11. Agent activation and stepping
        if self.enable_agents:
            self._activate_agents(alive)
            if self.agents_active:
                self._step_agents(alive)

        # 12. Epoch transition checks
        self._check_epoch_transition()

        # Age all species
        for sp in alive:
            sp.age += 1

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def _compute_energy(self, alive: list[Species]) -> None:
        """Compute E_income and E_cost for each species."""
        # Shuffle feeding order so no species consistently drains biomes first
        feeding_order = list(alive)
        self.rng.shuffle(feeding_order)
        for sp in feeding_order:
            e_income = self._compute_income(sp)
            e_cost = self._compute_cost(sp)
            self.energy_log[sp.sid] = (e_income, e_cost)

    def _compute_income(self, sp: Species) -> float:
        """Energy income from non-predation feeding.

        Predation income is excluded — it comes solely from successful
        hunts in _resolve_predation to avoid double-counting.
        Per-biome contributions are divided by biome count (population split).
        Consumption depletes biome vegetation and detritus as a side effect.
        Digestive penalty applied based on body plan / diet affinity.
        """
        income = 0.0
        diet = sp.diet._d  # read-only access, avoid dict copy
        size_mean = sp.traits["size"].mean if "size" in sp.traits else 1.0
        biomes = self._species_biomes(sp)
        n_biomes = max(1, len(biomes))
        pop_per_biome = sp.population / n_biomes

        # Digestive affinity lookup
        affinity = BODY_PLAN_DIET_AFFINITY.get(sp.body_plan.name, {})

        # Effective solar multiplier from catastrophe
        solar_mult = 1.0
        if self.catastrophe and "solar_mult" in self.catastrophe:
            solar_mult = self.catastrophe["solar_mult"]

        for source, proportion in diet.items():
            if source == "solar":
                eff = affinity.get("solar", DIET_MISMATCH_PENALTY)
                photo = sp.traits["photosynth"].mean if "photosynth" in sp.traits else 0.0
                for biome in biomes:
                    nutrient_factor = min(biome.nitrogen, biome.phosphorus, 1.0)
                    income += proportion * biome.solar * solar_mult * photo * 100.0 * eff * nutrient_factor / n_biomes
            elif source == "plant":
                eff = affinity.get("plant", DIET_MISMATCH_PENALTY)
                for biome in biomes:
                    nutrient_factor = min(biome.nitrogen, biome.phosphorus, 1.0)
                    income += proportion * biome.vegetation * size_mean * 80.0 * eff * nutrient_factor / n_biomes
                    grazing = proportion * size_mean * pop_per_biome * 0.00005
                    biome.vegetation = max(0.0, biome.vegetation - grazing)
            elif source == "detritus":
                eff = affinity.get("detritus", DIET_MISMATCH_PENALTY)
                is_decomposer = sp.body_plan.name == "decomposer"
                for biome in biomes:
                    consumable = min(biome.detritus, pop_per_biome * 0.1)
                    income += proportion * consumable * 50.0 * eff
                    consumed = consumable * proportion
                    biome.detritus = max(0.0, biome.detritus - consumed)
                    # Decomposers release nutrients when consuming detritus
                    if is_decomposer and consumed > 0:
                        biome.release_nutrients(consumed)
            elif source.startswith("S"):
                pass  # Predation: energy comes from _resolve_predation only
            else:
                # Unknown source — apply mismatch penalty
                eff = affinity.get(source, DIET_MISMATCH_PENALTY)
                income += proportion * 30.0 * eff

        return income

    def _compute_cost(self, sp: Species) -> float:
        """Energy cost: basal metabolism * size^0.75 + brain_tax + movement."""
        size_mean = sp.traits["size"].mean if "size" in sp.traits else 1.0
        basal = sp.body_plan.base_metabolism * (size_mean ** KLEIBER_EXPONENT) * sp.population

        # Brain tax from cognitive traits
        brain = 0.0
        for trait_name, td in sp.traits.items():
            tier = TRAIT_TO_TIER.get(trait_name, 0)
            if tier > 0:
                brain += td.mean * BRAIN_TAX.get(tier, 0.0) * sp.population

        for trait_name, td in sp.fused_traits.items():
            brain += td.mean * 0.05 * sp.population

        # Movement cost
        speed_mean = sp.traits["speed"].mean if "speed" in sp.traits else 0.0
        movement = speed_mean * 0.1 * sp.population

        return basal + brain + movement

    def _species_biomes(self, sp: Species) -> list[Biome]:
        """Get biomes where a species is present (cached per tick)."""
        cached = self._biome_cache.get(sp.sid)
        if cached is not None:
            return cached
        result = [b for b in self.biomes if f"L{b.lid}" in sp.locations]
        self._biome_cache[sp.sid] = result
        return result

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def _resolve_interactions(self, alive: list[Species]) -> None:
        """Resolve predation, competition, and symbiosis between species."""
        # Predation
        for sp in alive:
            for source, proportion in sp.diet._d.items():
                if source.startswith("S") and proportion > 0.1:
                    try:
                        prey_sid = int(source[1:])
                        prey = self._species_by_sid.get(prey_sid)
                        if prey and prey.alive and self._share_biome(sp, prey):
                            self._resolve_predation(sp, prey, proportion)
                    except ValueError:
                        pass

        # Competition (species in same biome with overlapping diets)
        for i, sp_a in enumerate(alive):
            for sp_b in alive[i + 1:]:
                if self._share_biome(sp_a, sp_b):
                    overlap = self._diet_overlap(sp_a, sp_b)
                    if overlap > 0.3 and self.rng.random() < 0.3:
                        self._resolve_competition(sp_a, sp_b, overlap)

        # Symbiosis (rare) — also track for symbiogenesis
        if len(alive) >= 2 and self.rng.random() < 0.03:
            pair = self.rng.choice(alive, size=2, replace=False)
            if self._share_biome(pair[0], pair[1]):
                a_traits = set(pair[0].traits.keys())
                b_traits = set(pair[1].traits.keys())
                if len(a_traits.symmetric_difference(b_traits)) >= 3:
                    self.interactions.append(Interaction(
                        verb="symbiosis",
                        actor_sid=pair[0].sid,
                        target_sid=pair[1].sid,
                        success=1.0,
                        effects={
                            pair[0].sid: {"E": 200},
                            pair[1].sid: {"E": 200},
                        },
                    ))
                    pair[0].energy_store += 200
                    pair[1].energy_store += 200
                    # Track symbiosis history for symbiogenesis
                    key = (min(pair[0].sid, pair[1].sid), max(pair[0].sid, pair[1].sid))
                    self.symbiosis_history[key] = self.symbiosis_history.get(key, 0) + 1

    def _resolve_predation(self, predator: Species, prey: Species, diet_prop: float) -> None:
        """Resolve a hunting interaction."""
        # Success based on trait overlap (speed, sense, camo)
        pred_speed = predator.traits["speed"].mean if "speed" in predator.traits else 0.0
        prey_speed = prey.traits["speed"].mean if "speed" in prey.traits else 0.0
        pred_sense = predator.traits["sense"].mean if "sense" in predator.traits else 0.0
        prey_camo = prey.traits["camo"].mean if "camo" in prey.traits else 0.0

        advantage = (pred_speed - prey_speed * 0.8) + (pred_sense - prey_camo * 0.7)
        success = float(np.clip(0.3 + advantage * 0.05 + self.rng.normal(0, 0.05), 0.05, 0.95))

        kills = max(1, int(prey.population * success * diet_prop * 0.1))
        kills = min(kills, prey.population // 2)  # safety rail: can't kill more than half

        prey_size = prey.traits["size"].mean if "size" in prey.traits else 1.0
        energy_gain = kills * prey_size * TROPHIC_EFFICIENCY * 100.0

        prey.population = max(1, prey.population - kills)
        predator.energy_store += energy_gain

        # Detritus from kills
        for biome in self._species_biomes(prey):
            biome.add_detritus(kills * prey_size * 0.3)

        self.interactions.append(Interaction(
            verb="hunt",
            actor_sid=predator.sid,
            target_sid=prey.sid,
            success=success,
            effects={
                prey.sid: {"pop": -kills},
                predator.sid: {"E": energy_gain},
            },
        ))

    def _resolve_competition(self, a: Species, b: Species, overlap: float) -> None:
        """Resolve competition between two species."""
        # Larger/more armored species wins
        a_power = (a.traits.get("size", TraitDistribution(1, 0.1)).mean +
                   a.traits.get("armor", TraitDistribution(0, 0.1)).mean * 0.5)
        b_power = (b.traits.get("size", TraitDistribution(1, 0.1)).mean +
                   b.traits.get("armor", TraitDistribution(0, 0.1)).mean * 0.5)

        advantage = a_power - b_power + float(self.rng.normal(0, 0.5))
        loser, winner = (b, a) if advantage > 0 else (a, b)
        success = float(np.clip(0.5 + abs(advantage) * 0.1, 0.1, 0.9))

        pop_loss = max(1, int(loser.population * overlap * 0.05))
        loser.population = max(1, loser.population - pop_loss)

        self.interactions.append(Interaction(
            verb="compete",
            actor_sid=winner.sid,
            target_sid=loser.sid,
            success=success,
            effects={loser.sid: {"pop": -pop_loss}},
        ))

    def _share_biome(self, a: Species, b: Species) -> bool:
        return bool(a.locations & b.locations)

    def _diet_overlap(self, a: Species, b: Species) -> float:
        """Compute diet overlap (Pianka index simplified)."""
        a_cats = a.diet.get_category_totals()
        b_cats = b.diet.get_category_totals()
        overlap = 0.0
        for cat in a_cats:
            overlap += min(a_cats[cat], b_cats.get(cat, 0.0))
        return overlap

    # ------------------------------------------------------------------
    # Population dynamics
    # ------------------------------------------------------------------

    def _update_populations(self, alive: list[Species]) -> None:
        """Update populations based on energy balance."""
        for sp in alive:
            e_income, e_cost = self.energy_log.get(sp.sid, (0.0, 0.0))
            balance = e_income - e_cost

            if balance > 0:
                # Surplus: store fat, then reproduce
                max_store = e_cost * FAT_STORE_MAX_FACTOR
                store_room = max(0, max_store - sp.energy_store)
                stored = min(balance, store_room)
                sp.energy_store += stored
                surplus = balance - stored

                if surplus > 0:
                    repro_energy = surplus * REPRODUCTION_ENERGY_FRACTION
                    repro_rate = sp.traits["repro"].mean if "repro" in sp.traits else 1.0
                    offspring = max(0, int(repro_energy * repro_rate * 0.01))
                    # Growth cap: max 20% per tick
                    offspring = min(offspring, int(sp.population * 0.2))
                    sp.population += offspring
            else:
                # Deficit: burn fat stores
                deficit = abs(balance)
                if sp.energy_store >= deficit:
                    sp.energy_store -= deficit
                else:
                    # Stores exhausted — starvation
                    remaining_deficit = deficit - sp.energy_store
                    sp.energy_store = 0.0
                    severity = min(1.0, remaining_deficit / (deficit + 1.0))
                    # Low-variance species are more vulnerable to starvation
                    mean_var = self._mean_variance(sp)
                    if mean_var < LOW_VARIANCE_THRESHOLD:
                        severity = min(1.0, severity * 1.5)
                    loss = max(1, int(sp.population * STARVATION_RATE * severity))
                    sp.population = max(0, sp.population - loss)

                    if sp.population == 0:
                        sp.alive = False
                        self.events.setdefault(sp.sid, []).append(
                            f"extinction:{self.rng.choice(EXTINCTION_CAUSES)}"
                        )
                        # Add detritus from extinction
                        size = sp.traits["size"].mean if "size" in sp.traits else 1.0
                        for biome in self._species_biomes(sp):
                            biome.add_detritus(loss * size * 0.5)

            # Population floor — only for viable species with energy reserves
            if sp.alive and sp.population < 2 and sp.energy_store > 0:
                sp.population = 2

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def _evolve(self, alive: list[Species]) -> None:
        """Run evolutionary operations on all living species."""
        epoch_cfg = EPOCH_CONFIG[self.epoch]
        mut_mult = epoch_cfg["mutation_rate_mult"]

        for sp in alive:
            self.mutations.setdefault(sp.sid, [])
            self.events.setdefault(sp.sid, [])

            # Evolutionary trap: stressed + low variance = over-specialized
            e_income, e_cost = self.energy_log.get(sp.sid, (0.0, 0.0))
            stressed = sp.energy_store == 0 or (sp.energy_store < e_cost and sp.energy_store > 0)
            if stressed and self._mean_variance(sp) < LOW_VARIANCE_THRESHOLD:
                self.events[sp.sid].append("evo_trap:low_variance")

            # Mutate existing traits
            self._mutate_traits(sp, mut_mult)

            # Try acquiring new traits (prerequisite-gated)
            self._try_acquire_trait(sp, mut_mult)

            # Trait fusion
            self._try_fuse_traits(sp)

            # Body plan transition
            new_plan = sp.check_body_plan_transition()
            if new_plan is not None:
                old_name = sp.body_plan.name
                sp.body_plan = new_plan
                sp.apply_body_plan_constraints()
                self.events[sp.sid].append(f"body_plan:{old_name}->{new_plan.name}")

            # Diet mutation
            sp.diet.mutate(self.rng, magnitude=0.03 * mut_mult)

            # Diet evolution: predators may acquire new prey
            self._evolve_diet(sp, alive)

            # Goal update
            if self.rng.random() < 0.04:
                max_tier = sp.max_tier()
                available = GOALS[:6]
                if max_tier >= 2:
                    available = GOALS[:10]
                if max_tier >= 3:
                    available = GOALS[:14]
                if max_tier >= 4:
                    available = GOALS
                self.goals[sp.sid] = str(self.rng.choice(available))

    def _mutate_traits(self, sp: Species, mut_mult: float) -> None:
        """Point/drift/leap/loss mutations on existing traits."""
        all_mutable = list(sp.traits.keys()) + list(sp.fused_traits.keys())
        if not all_mutable:
            return

        intel = sp.traits["intel"].mean if "intel" in sp.traits else 0.0
        base_rate = max(0.1, 0.35 - intel * 0.01) * mut_mult

        if self.rng.random() > base_rate:
            return

        trait = str(self.rng.choice(all_mutable))
        is_fused = trait in sp.fused_traits
        container = sp.fused_traits if is_fused else sp.traits
        td = container[trait]

        # Stress-boosted Mleap (variance-as-evolvability)
        e_income, e_cost = self.energy_log.get(sp.sid, (0.0, 0.0))
        stress = 1.0 if sp.energy_store == 0 else (0.3 if sp.energy_store < e_cost else 0.0)
        mean_var = self._mean_variance(sp)

        roll = float(self.rng.random())
        # Under stress + high variance: Mleap range expands from [0.75, 0.90) to [0.50, 0.90)
        mleap_floor = 0.75
        if stress > 0 and mean_var >= HIGH_VARIANCE_THRESHOLD:
            mleap_floor = 0.50

        if roll < 0.50:
            mut_type = "Mpoint"
            delta = float(self.rng.choice([-0.5, -0.3, 0.3, 0.5]))
        elif roll < mleap_floor:
            mut_type = "Mdrift"
            delta = float(self.rng.normal(0, 0.3))
            # Also widen/narrow variance
            if delta > 0:
                td.widen(0.05)
            else:
                td.narrow(0.05)
        elif roll < 0.90:
            mut_type = "Mleap"
            delta = float(self.rng.choice([-2.0, -1.5, 1.5, 2.0]))
        else:
            mut_type = "M-"
            delta = -float(self.rng.uniform(0.5, 1.5))

        old_mean = td.mean
        td.shift(delta)

        # Enforce body plan cap
        cap = sp.body_plan.get_cap(trait)
        if cap is not None and td.mean > cap:
            td.mean = float(cap)

        actual_delta = td.mean - old_mean
        if abs(actual_delta) < 0.01:
            return

        sign = "+" if actual_delta > 0 else ""
        self.mutations[sp.sid].append(f"{mut_type}|{trait}={sign}{actual_delta:.1f}")

    def _try_acquire_trait(self, sp: Species, mut_mult: float) -> None:
        """Attempt to acquire a new trait from the next tier up."""
        max_tier = sp.max_tier()
        target_tier = min(max_tier + 1, 4)

        candidates = [
            t for t in TRAIT_TIERS.get(target_tier, [])
            if t not in sp.traits
            and t not in sp.fused_traits
            and sp.body_plan.can_evolve(t)
            and sp.has_prerequisites(t)
        ]
        # Also check same tier for unfilled slots
        same_tier = [
            t for t in TRAIT_TIERS.get(max_tier, [])
            if t not in sp.traits
            and t not in sp.fused_traits
            and sp.body_plan.can_evolve(t)
            and sp.has_prerequisites(t)
        ]
        candidates.extend(same_tier)

        if not candidates:
            return

        prob = {0: 0.12, 1: 0.06, 2: 0.04, 3: 0.025, 4: 0.012}
        trait = str(self.rng.choice(candidates))
        tier = TRAIT_TO_TIER[trait]
        if self.rng.random() < prob.get(tier, 0.01) * mut_mult:
            init_mean = float(self.rng.uniform(0.5, 2.5))
            init_var = float(self.rng.uniform(0.2, 1.0))
            cap = sp.body_plan.get_cap(trait)
            if cap is not None:
                init_mean = min(init_mean, float(cap))
            sp.traits[trait] = TraitDistribution(init_mean, init_var)
            self.mutations[sp.sid].append(f"M+|{trait}={init_mean:.1f}±{math.sqrt(init_var):.1f}")

    def _try_fuse_traits(self, sp: Species) -> None:
        """Check fusion rules and create composite traits."""
        all_t = sp.get_all_traits()
        for tA, tB, fused_name, threshold in FUSION_RULES:
            if fused_name in sp.fused_traits:
                continue
            td_a = all_t.get(tA)
            td_b = all_t.get(tB)
            if td_a is None or td_b is None:
                continue
            if td_a.mean >= threshold and td_b.mean >= threshold and self.rng.random() < 0.04:
                init_mean = (td_a.mean + td_b.mean) / 4.0
                init_var = (td_a.variance + td_b.variance) / 2.0
                sp.fused_traits[fused_name] = TraitDistribution(init_mean, init_var)
                self.mutations[sp.sid].append(
                    f"Mfuse|{tA}+{tB}->{fused_name}={init_mean:.1f}"
                )
                break  # max one fusion per tick

    def _evolve_diet(self, sp: Species, alive: list[Species]) -> None:
        """Evolve diet: species may acquire new food sources opportunistically."""
        if self.rng.random() > 0.08:
            return

        cats = sp.diet.get_category_totals()
        speed = sp.traits["speed"].mean if "speed" in sp.traits else 0.0
        sense = sp.traits["sense"].mean if "sense" in sp.traits else 0.0

        # Any mobile species with speed+sense can opportunistically hunt smaller species
        if sp.body_plan.name in ("predator", "omnivore", "grazer", "scavenger") and speed > 1.0:
            potential_prey = [
                s for s in alive
                if s.sid != sp.sid
                and self._share_biome(sp, s)
                and f"S{s.sid}" not in sp.diet._d
                and s.population > 50
            ]
            # Prefer smaller prey
            potential_prey = [
                s for s in potential_prey
                if (s.traits.get("size", TraitDistribution(1, 0.1)).mean <
                    sp.traits.get("size", TraitDistribution(1, 0.1)).mean + 2)
            ]
            if potential_prey:
                # Chance scales with speed and sense
                hunt_chance = 0.15 if sp.body_plan.name in ("predator", "omnivore") else 0.06
                if self.rng.random() < hunt_chance:
                    prey = self.rng.choice(potential_prey)
                    init_prop = 0.1 if sp.body_plan.name in ("predator", "omnivore") else 0.05
                    sp.diet.add_source(f"S{prey.sid}", init_prop)

        # Grazers/scavengers might start eating detritus
        if sp.body_plan.name == "grazer" and "detritus" not in sp.diet._d:
            if self.rng.random() < 0.1:
                sp.diet.add_source("detritus", 0.05)

        # Filter feeders/decomposers might discover plant matter
        if sp.body_plan.name in ("filter_feeder", "decomposer") and "plant" not in sp.diet._d:
            if self.rng.random() < 0.05:
                sp.diet.add_source("plant", 0.05)

        # Clean up dead prey from diet
        for source in list(sp.diet._d.keys()):
            if source.startswith("S"):
                try:
                    prey_sid = int(source[1:])
                    prey = self._species_by_sid.get(prey_sid)
                    if prey is None or not prey.alive:
                        sp.diet.remove_source(source)
                except ValueError:
                    sp.diet.remove_source(source)

    # ------------------------------------------------------------------
    # Speciation
    # ------------------------------------------------------------------

    def _try_speciation(self, alive: list[Species]) -> None:
        """Branch new species from thriving ones."""
        if len(alive) >= 20:
            return

        spec_prob = EPOCH_CONFIG[self.epoch]["speciation_prob"]

        for sp in alive:
            if sp.population <= 500 or sp.energy_store <= 0:
                continue

            # Migration friction: disconnected biomes boost speciation
            adjusted_prob = spec_prob * self._isolation_factor(sp)

            if self.rng.random() < adjusted_prob:
                child = self._branch_species(sp)
                # Subtract child population from parent (conservation of individuals)
                sp.population -= child.population
                sp.energy_store = max(0.0, sp.energy_store - child.energy_store)
                self.species.append(child)
                self._species_by_sid[child.sid] = child
                self.goals[child.sid] = str(self.rng.choice(GOALS[:8]))
                self.events.setdefault(sp.sid, []).append(f"speciation:S{child.sid}")
                break  # max one speciation per tick

    def _branch_species(self, parent: Species) -> Species:
        """Create a child species diverging from parent."""
        # Copy and mutate traits
        child_traits = {}
        for t, td in parent.traits.items():
            new_mean = float(np.clip(td.mean + self.rng.normal(0, 0.5), 0, 10))
            new_var = float(max(0.01, td.variance * (1.0 + self.rng.normal(0, 0.2))))
            child_traits[t] = TraitDistribution(new_mean, new_var)

        # Maybe lose a higher-tier trait
        higher = [t for t in child_traits if TRAIT_TO_TIER.get(t, 0) >= 2]
        if higher and self.rng.random() < 0.3:
            lost = str(self.rng.choice(higher))
            del child_traits[lost]

        # Copy some fused traits
        child_fused = {}
        for ft, ftd in parent.fused_traits.items():
            if self.rng.random() < 0.4:
                child_fused[ft] = TraitDistribution(
                    max(0.5, ftd.mean - float(self.rng.uniform(0, 1))),
                    ftd.variance,
                )

        child_diet = DietVector(parent.diet.sources())
        child_diet.mutate(self.rng, magnitude=0.1)

        sid = self.next_sid
        self.next_sid += 1

        child_pop = max(10, parent.population // int(self.rng.integers(5, 15)))

        # Child gets a random subset of parent's locations (migration friction)
        parent_locs = sorted(parent.locations)
        if len(parent_locs) > 1:
            n_child_locs = int(self.rng.integers(1, len(parent_locs) + 1))
            child_locs = set(self.rng.choice(parent_locs, size=n_child_locs, replace=False))
        else:
            child_locs = set(parent_locs)

        return Species(
            sid=sid,
            traits=child_traits,
            fused_traits=child_fused,
            body_plan=BodyPlan(parent.body_plan.name),
            diet=child_diet,
            population=child_pop,
            energy_store=float(child_pop * 0.3),
            locations=child_locs,
        )

    # ------------------------------------------------------------------
    # Catastrophes
    # ------------------------------------------------------------------

    def _check_catastrophes(self) -> None:
        """Check for new catastrophe or apply ongoing one."""
        if self.catastrophe is not None:
            # Decrement duration
            self.catastrophe["remaining"] -= 1
            if self.catastrophe["remaining"] <= 0:
                ctype = self.catastrophe["type"]
                self.world_events.append(f"catastrophe_end:{ctype}")
                self.catastrophe = None
            return

        if self.rng.random() >= CATASTROPHE_PROB:
            return

        ctype = str(self.rng.choice(list(CATASTROPHE_TYPES.keys())))
        cfg = CATASTROPHE_TYPES[ctype]
        lo, hi = cfg["duration_range"]
        duration = int(self.rng.integers(lo, hi + 1))

        self.catastrophe = {"type": ctype, "remaining": duration, **cfg}
        self.world_events.append(f"catastrophe:{ctype}|dur={duration}")

        # Immediate effects
        if ctype == "meteor_impact":
            # Pick a random biome for impact
            target = self.rng.choice(self.biomes)
            target.vegetation = cfg.get("veg_set", 0.0)
            target.detritus += cfg.get("detritus_spike", 0)
            pop_lo, pop_hi = cfg["pop_loss"]
            for sp in self.species:
                if sp.alive and f"L{target.lid}" in sp.locations:
                    loss_frac = float(self.rng.uniform(pop_lo, pop_hi))
                    pop_loss = int(sp.population * loss_frac)
                    sp.population = max(1, sp.population - pop_loss)
                    size = sp.traits["size"].mean if "size" in sp.traits else 1.0
                    target.add_detritus(pop_loss * size * 0.5)

    # ------------------------------------------------------------------
    # Disease
    # ------------------------------------------------------------------

    def _check_disease(self, alive: list[Species]) -> None:
        """Disease outbreak check for each species."""
        for sp in alive:
            mean_var = self._mean_variance(sp)
            density_factor = sp.population / 5000.0
            vulnerability = density_factor * max(0.1, 1.0 / (mean_var + 0.1))

            if self.rng.random() >= vulnerability * DISEASE_BASE_PROB:
                continue

            # Outbreak — mortality depends on variance
            if mean_var >= HIGH_VARIANCE_THRESHOLD:
                mortality = float(self.rng.uniform(0.1, 0.2))
            elif mean_var < LOW_VARIANCE_THRESHOLD:
                mortality = float(self.rng.uniform(0.4, 0.8))
            else:
                mortality = float(self.rng.uniform(0.15, 0.4))

            pop_loss = max(1, int(sp.population * mortality))
            pop_loss = min(pop_loss, sp.population - 1)  # leave at least 1
            sp.population -= pop_loss

            # Detritus from dead
            size = sp.traits["size"].mean if "size" in sp.traits else 1.0
            biomes = self._species_biomes(sp)
            for biome in biomes:
                biome.add_detritus(pop_loss * size * 0.3 / max(1, len(biomes)))

            disease_type = str(self.rng.choice(DISEASE_TYPES))
            self.events.setdefault(sp.sid, []).append(
                f"disease:{disease_type}|pop-={pop_loss}"
            )

    # ------------------------------------------------------------------
    # Symbiogenesis
    # ------------------------------------------------------------------

    def _check_symbiogenesis(self, alive: list[Species]) -> None:
        """Check for symbiogenesis events from sustained symbiotic pairs."""
        if len(alive) >= 20:
            return

        # Decay absent pairs (remove if both species not sharing a biome)
        for key in list(self.symbiosis_history.keys()):
            sid_a, sid_b = key
            sp_a = self._species_by_sid.get(sid_a)
            sp_b = self._species_by_sid.get(sid_b)
            if sp_a is None or sp_b is None or not sp_a.alive or not sp_b.alive:
                del self.symbiosis_history[key]
                continue
            if not self._share_biome(sp_a, sp_b):
                self.symbiosis_history[key] = max(0, self.symbiosis_history[key] - 1)
                if self.symbiosis_history[key] <= 0:
                    del self.symbiosis_history[key]

        # Check for symbiogenesis
        for key, ticks in list(self.symbiosis_history.items()):
            if ticks < SYMBIOGENESIS_MIN_TICKS:
                continue
            if self.rng.random() >= SYMBIOGENESIS_PROB:
                continue

            sid_a, sid_b = key
            sp_a = self._species_by_sid.get(sid_a)
            sp_b = self._species_by_sid.get(sid_b)
            if sp_a is None or sp_b is None or not sp_a.alive or not sp_b.alive:
                continue

            # Major partner = larger population; minor = smaller
            if sp_a.population >= sp_b.population:
                major, minor = sp_a, sp_b
            else:
                major, minor = sp_b, sp_a

            # Pick traits from minor that major lacks
            gained_traits = [t for t in minor.traits if t not in major.traits and major.body_plan.can_evolve(t)]
            if not gained_traits:
                continue
            # Take up to 3 traits
            n_gain = min(3, len(gained_traits))
            gained = list(self.rng.choice(gained_traits, size=n_gain, replace=False))

            # Create child species with major's base + gained traits
            child_traits = {t: TraitDistribution(td.mean, td.variance) for t, td in major.traits.items()}
            for t in gained:
                td = minor.traits[t]
                child_traits[t] = TraitDistribution(td.mean * 0.5, td.variance)

            child_sid = self.next_sid
            self.next_sid += 1
            child_pop = max(10, min(major.population, minor.population) // 10)

            child = Species(
                sid=child_sid,
                traits=child_traits,
                body_plan=BodyPlan(major.body_plan.name),
                diet=DietVector(major.diet.sources()),
                population=child_pop,
                energy_store=float(child_pop * 0.3),
                locations=set(major.locations & minor.locations) or set(major.locations),
            )

            # Deduct from parents
            major.population = max(1, major.population - child_pop // 2)
            minor.population = max(1, minor.population - child_pop // 2)

            self.species.append(child)
            self._species_by_sid[child.sid] = child
            self.goals[child.sid] = str(self.rng.choice(GOALS[:8]))

            gained_str = ",".join(gained)
            self.events.setdefault(major.sid, []).append(
                f"symbiogenesis:S{major.sid}+S{minor.sid}->S{child_sid}|gained={gained_str}"
            )

            # Remove from history so it doesn't trigger again
            del self.symbiosis_history[key]
            break  # max one per tick

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mean_variance(self, sp: Species) -> float:
        """Mean trait variance across all traits."""
        all_vars = [td.variance for td in sp.traits.values()]
        if sp.fused_traits:
            all_vars.extend(td.variance for td in sp.fused_traits.values())
        return sum(all_vars) / max(1, len(all_vars))

    def _isolation_factor(self, sp: Species) -> float:
        """Compute isolation factor from disconnected biome pairs."""
        lids = [int(loc[1:]) for loc in sp.locations if loc.startswith("L")]
        if len(lids) < 2:
            return 1.0
        total_pairs = 0
        disconnected = 0
        for i in range(len(lids)):
            for j in range(i + 1, len(lids)):
                total_pairs += 1
                adj = self.biome_adjacency.get(lids[i], {})
                if lids[j] not in adj:
                    disconnected += 1
        return 1.0 + 0.5 * disconnected / max(1, total_pairs)

    # ------------------------------------------------------------------
    # Spotlights
    # ------------------------------------------------------------------

    def _generate_spotlights(self, alive: list[Species]) -> None:
        """Generate spotlight events for intelligent species."""
        for sp in alive:
            score = sp.spotlight_score()
            if score < 20.0:
                continue
            if self.rng.random() > 0.15:
                continue

            biomes = self._species_biomes(sp)
            if not biomes:
                continue
            biome = self.rng.choice(biomes)

            # Generate heroes
            n_heroes = int(self.rng.integers(2, 5))
            heroes = []
            for h in range(n_heroes):
                heroes.append(Hero(
                    hid=h + 1,
                    role=str(self.rng.choice(SPOTLIGHT_ROLES)),
                    influence=float(self.rng.uniform(1.0, 9.0)),
                ))

            # Cultural memories
            memories = self.cultural_memories.get(sp.sid, [])
            # Maybe create a new memory
            if self.rng.random() < 0.2:
                mem_types = ["taboo", "legend", "tradition", "sacred"]
                event_names = [
                    "great_hunt", "reef_collapse", "drought", "first_fire",
                    "elder_teaching", "migration", "battle_won", "plague",
                ]
                mem = CulturalMemory(
                    name=str(self.rng.choice(event_names)),
                    mtype=str(self.rng.choice(mem_types)),
                    strength=float(self.rng.uniform(0.2, 0.9)),
                    origin_gen=self.tick,
                )
                memories.append(mem)
                self.cultural_memories[sp.sid] = memories

            # Generate spotlight scene
            actor = heroes[int(self.rng.integers(len(heroes)))]
            others = [h for h in heroes if h.hid != actor.hid] or heroes
            reactor = others[int(self.rng.integers(len(others)))]

            action = str(self.rng.choice(SPOTLIGHT_ACTIONS))
            reaction = str(self.rng.choice(SPOTLIGHT_REACTIONS))
            resolution = str(self.rng.choice(SPOTLIGHT_RESOLUTIONS))
            outcome = str(self.rng.choice(SPOTLIGHT_OUTCOMES))

            # Build reason strings referencing cultural memory if available
            intent_reason = "environmental_pressure"
            react_reason = "survival_instinct"
            if memories:
                mem = memories[int(self.rng.integers(len(memories)))]
                react_reason = f"Cmem:{mem.name}"

            intent_str = f"H{actor.hid}->S{sp.sid}:{action}({outcome})|reason={intent_reason}"
            react_str = f"H{reactor.hid}->H{actor.hid}:{reaction}({outcome})|reason={react_reason}"
            resolve_str = f"{resolution}(H{reactor.hid}.inf+H{actor.hid}.{action})|outcome={outcome}"

            # Effect
            effect_parts = [f"S{sp.sid}"]
            if "migrate" in outcome or "colony" in outcome:
                if len(self.biomes) > 1:
                    target_biome = self.rng.choice([b for b in self.biomes if f"L{b.lid}" not in sp.locations] or self.biomes)
                    sp.locations.add(f"L{target_biome.lid}")
                    self._biome_cache.pop(sp.sid, None)  # invalidate cache
                    effect_parts.append(f"loc+={{L{target_biome.lid}:60}}")
            effect_parts.append(f"H{actor.hid}:inf+={float(self.rng.uniform(-0.5, 1.0)):.1f}")
            effect_str = "|".join(effect_parts)

            self.spotlights.append(SpotlightEvent(
                world_id=self.wid,
                gen=self.tick,
                biome_lid=biome.lid,
                species_sid=sp.sid,
                heroes=heroes,
                intent=intent_str,
                react=react_str,
                resolve=resolve_str,
                effect=effect_str,
                cultural_memories=memories[-3:],  # last 3
            ))

    # ------------------------------------------------------------------
    # Agent-based simulation
    # ------------------------------------------------------------------

    def _activate_agents(self, alive: list[Species]) -> None:
        """Spawn agents for species meeting activation criteria."""
        if self.epoch.value < self.agent_epoch.value:
            return

        simple_mode = (self.epoch == Epoch.ECOSYSTEM)

        for sp in alive:
            if sp.agent_manager is not None:
                continue

            # Check activation threshold
            if self.agent_epoch == Epoch.INTELLIGENCE:
                if sp.spotlight_score() < self.agent_threshold:
                    continue
            # ECOSYSTEM: activate all species with population > 50
            elif self.agent_epoch == Epoch.ECOSYSTEM:
                if sp.population < 50:
                    continue

            # Count total agents across all species
            total_agents = sum(
                len(s.agent_manager.agents)
                for s in self.species
                if s.agent_manager is not None
            )
            if total_agents >= AGENT_TOTAL_MAX:
                continue

            # Determine agent count
            n_agents = min(
                AGENT_MAX_PER_SPECIES,
                sp.population,
                AGENT_TOTAL_MAX - total_agents,
            )
            n_agents = max(20, n_agents)

            # Pick primary biome
            biomes = self._species_biomes(sp)
            biome = biomes[0] if biomes else self.biomes[0]

            # Initialize vegetation patches if needed
            if not biome.vegetation_patches:
                biome.init_vegetation_patches(self.rng, world_size=1000)

            mgr = AgentManager(
                species_sid=sp.sid,
                biome_lid=biome.lid,
                world_size=1000,
            )
            mgr.simple_mode = simple_mode
            mgr.spawn_agents(n_agents, sp, self.rng)
            sp.agent_manager = mgr
            sp.reconciler = PopulationReconciler(sp, n_agents)
            self.agents_active = True

    def _step_agents(self, alive: list[Species]) -> None:
        """Step all active agent managers and reconcile with macro population."""
        # Build manager lookup and prey/predator maps
        all_managers: dict[int, AgentManager] = {}
        for sp in alive:
            if sp.agent_manager is not None:
                all_managers[sp.sid] = sp.agent_manager

        if not all_managers:
            return

        # Regenerate vegetation patches
        for biome in self.biomes:
            biome.regenerate_patches()

        for sp in alive:
            mgr = sp.agent_manager
            if mgr is None:
                continue

            # Determine prey and predator SIDs from diet
            prey_sids: set[int] = set()
            for source in sp.diet.sources():
                if source.startswith("S"):
                    try:
                        prey_sids.add(int(source[1:]))
                    except ValueError:
                        pass

            predator_sids: set[int] = set()
            for other_sp in alive:
                if other_sp.sid == sp.sid:
                    continue
                for source in other_sp.diet.sources():
                    if source == f"S{sp.sid}":
                        predator_sids.add(other_sp.sid)

            # Get vegetation patches from species' biome
            biomes = self._species_biomes(sp)
            patches = []
            for b in biomes:
                patches.extend(b.vegetation_patches)

            mgr.step(all_managers, patches, prey_sids, predator_sids, self.rng)

            # Reconcile with macro population
            if sp.reconciler is not None:
                sp.reconciler.reconcile_tick(mgr)

    # ------------------------------------------------------------------
    # Epoch transitions
    # ------------------------------------------------------------------

    def _check_epoch_transition(self) -> None:
        """Check if the world should advance to the next epoch."""
        alive = [sp for sp in self.species if sp.alive]

        if self.epoch == Epoch.PRIMORDIAL:
            # Transition to CAMBRIAN: any species has a T1 trait
            for sp in alive:
                if sp.max_tier() >= 1:
                    self.epoch = Epoch.CAMBRIAN
                    self.epoch_just_changed = True
                    return

        elif self.epoch == Epoch.CAMBRIAN:
            # Transition to ECOSYSTEM: 3+ trophic levels exist
            trophic_levels = set()
            for sp in alive:
                cats = sp.diet.get_category_totals()
                if cats.get("solar", 0) > 0.5:
                    trophic_levels.add("producer")
                elif cats.get("plant", 0) > 0.5:
                    trophic_levels.add("herbivore")
                elif cats.get("meat", 0) > 0.3:
                    trophic_levels.add("predator")
                elif cats.get("detritus", 0) > 0.5:
                    trophic_levels.add("decomposer")
                else:
                    trophic_levels.add("omnivore")
            if len(trophic_levels) >= 3:
                self.epoch = Epoch.ECOSYSTEM
                self.epoch_just_changed = True

        elif self.epoch == Epoch.ECOSYSTEM:
            # Transition to INTELLIGENCE: any species has spotlight_score > 15
            for sp in alive:
                if sp.spotlight_score() > 15.0:
                    self.epoch = Epoch.INTELLIGENCE
                    self.epoch_just_changed = True
                    return
