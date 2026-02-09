#!/usr/bin/env python3
"""
Generate evolutionary simulation traces for MANTIS training.

Produces a .txt file of serialized world-state traces compatible with
TextDataset (train.py) or preprocess_data.py for pre-tokenization.

The simulation models diverse species evolving across tiered trait systems,
with dynamic trait acquisition, pruning, fusion, and emergent higher-order
attributes (language, subconsciousness, theory of mind, etc.).

Encoding Protocol
=================

World header:
    W<id>G<gen>|E<axis>:<val>,...

Species block:
    S<id>|P<pop>|F<fitness>|T<trait>=<val>,...|G<goal>
      A<key>:<value>                       # attribute (non-numeric state)
      M+<trait>=<val>                      # trait acquired
      M-<trait>                            # trait pruned
      Mpoint:<trait>=<delta>               # point mutation
      Mdrift:<trait>=<delta>               # genetic drift
      Mleap:<trait>=<delta>               # macro-mutation
      Mfuse:<traitA>+<traitB>-><new>       # trait fusion
      X<event>:<description>               # event

Generation separator:
    ---

Trait Tiers
===========
    T0 (Physical)  : size speed armor metab sense camo repro regen venom
    T1 (Behavioral): social aggression curiosity patience nocturnal
    T2 (Cognitive) : intel memory learning planning deception
    T3 (Cultural)  : language tooluse ritual teaching trade
    T4 (Abstract)  : subconscious theory_of_mind creativity abstraction ethics

Higher tiers emerge when prerequisite lower-tier traits reach thresholds.
Traits can be pruned under environmental pressure or fitness collapse.
"""

import random
import math
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import Optional

# ---------------------------------------------------------------------------
# Trait taxonomy with emergence prerequisites
# ---------------------------------------------------------------------------

TRAIT_TIERS = {
    0: ["size", "speed", "armor", "metab", "sense", "camo", "repro", "regen", "venom"],
    1: ["social", "aggression", "curiosity", "patience", "nocturnal"],
    2: ["intel", "memory", "learning", "planning", "deception"],
    3: ["language", "tooluse", "ritual", "teaching", "trade"],
    4: ["subconscious", "theory_of_mind", "creativity", "abstraction", "ethics"],
}

ALL_TRAITS = []
TRAIT_TO_TIER = {}
for tier, traits in TRAIT_TIERS.items():
    for t in traits:
        ALL_TRAITS.append(t)
        TRAIT_TO_TIER[t] = tier

# Prerequisites: {trait: [(required_trait, min_value), ...]}
# A species must meet ALL prerequisites to acquire a trait.
PREREQUISITES = {
    # T1 <- T0
    "social":      [("sense", 4), ("repro", 3)],
    "aggression":  [("speed", 4), ("venom", 2)],
    "curiosity":   [("sense", 5), ("speed", 3)],
    "patience":    [("camo", 4), ("sense", 3)],
    "nocturnal":   [("sense", 6)],
    # T2 <- T1
    "intel":       [("curiosity", 3), ("social", 3)],
    "memory":      [("sense", 5), ("curiosity", 4)],
    "learning":    [("intel", 3), ("curiosity", 4)],
    "planning":    [("intel", 4), ("memory", 3)],
    "deception":   [("intel", 3), ("social", 3)],
    # T3 <- T2
    "language":    [("intel", 5), ("social", 5), ("memory", 4)],
    "tooluse":     [("intel", 4), ("planning", 3)],
    "ritual":      [("social", 6), ("memory", 4)],
    "teaching":    [("language", 3), ("social", 5)],
    "trade":       [("language", 3), ("tooluse", 3), ("social", 5)],
    # T4 <- T3
    "subconscious":   [("memory", 6), ("intel", 6), ("learning", 5)],
    "theory_of_mind": [("intel", 6), ("social", 6), ("language", 4)],
    "creativity":     [("intel", 5), ("curiosity", 6), ("learning", 5)],
    "abstraction":    [("intel", 7), ("language", 5), ("planning", 5)],
    "ethics":         [("theory_of_mind", 4), ("social", 7), ("abstraction", 3)],
}

# Fusion rules: (traitA, traitB) -> new_trait when both >= threshold
FUSION_RULES = [
    ("social", "aggression", "dominance", 5),
    ("memory", "learning", "wisdom", 5),
    ("language", "teaching", "oral_tradition", 4),
    ("tooluse", "planning", "engineering", 4),
    ("creativity", "abstraction", "philosophy", 4),
    ("ethics", "theory_of_mind", "empathy", 4),
    ("ritual", "language", "mythology", 4),
    ("deception", "intel", "strategy", 5),
    ("trade", "tooluse", "economy", 4),
    ("curiosity", "planning", "science", 5),
]

# Attribute schemas: traits that trigger non-numeric state attributes
# {trait: [(threshold, attr_key, attr_value), ...]}
ATTRIBUTE_TRIGGERS = {
    "language": [
        (2, "lang", "proto_gestural"),
        (4, "lang", "vocal_simple"),
        (6, "lang", "vocal_grammar"),
        (8, "lang", "symbolic_written"),
    ],
    "subconscious": [
        (2, "consciousness", "reactive"),
        (4, "consciousness", "aware"),
        (6, "consciousness", "reflective"),
        (8, "consciousness", "introspective"),
    ],
    "theory_of_mind": [
        (2, "tom", "self_other"),
        (5, "tom", "perspective_taking"),
        (8, "tom", "recursive_modeling"),
    ],
    "social": [
        (3, "social_structure", "pair_bond"),
        (5, "social_structure", "pack"),
        (7, "social_structure", "tribe"),
        (9, "social_structure", "civilization"),
    ],
    "intel": [
        (3, "cognition", "associative"),
        (5, "cognition", "causal"),
        (7, "cognition", "abstract"),
        (9, "cognition", "meta_cognitive"),
    ],
    "tooluse": [
        (2, "tech", "found_objects"),
        (4, "tech", "shaped_tools"),
        (6, "tech", "composite_tools"),
        (8, "tech", "machines"),
    ],
    "creativity": [
        (3, "expression", "play"),
        (5, "expression", "art"),
        (7, "expression", "narrative"),
    ],
    "ethics": [
        (3, "moral_framework", "kin_altruism"),
        (5, "moral_framework", "reciprocal"),
        (7, "moral_framework", "principled"),
    ],
}

GOALS = [
    "forage", "hunt", "evade", "territory", "parasite", "symbiote",
    "migrate", "hoard", "nurture", "swarm", "explore", "cultivate",
    "build", "dominate", "cooperate", "innovate",
]

ENV_AXES = ["temp", "water", "light", "toxin", "oxygen", "radiation",
            "fertility", "density", "altitude", "salinity"]

EXTINCTION_CAUSES = [
    "drought", "predation", "disease", "competition", "habitat_loss",
    "volcanic", "ice_age", "radiation_spike", "overpopulation", "famine",
]

DECLINE_CAUSES = [
    "food_scarcity", "water_scarcity", "disease_outbreak",
    "inbreeding", "territorial_loss", "climate_shift",
]


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class Species:
    __slots__ = (
        "sid", "traits", "fused_traits", "goal", "pop", "fitness",
        "alive", "mutations", "events", "attributes", "age",
    )

    def __init__(
        self,
        sid: int,
        traits: Optional[dict] = None,
        fused_traits: Optional[dict] = None,
        goal: Optional[str] = None,
        pop: Optional[int] = None,
    ):
        self.sid = sid
        if traits is not None:
            self.traits = OrderedDict(traits)
        else:
            # Start with 3-6 random T0 traits
            n = random.randint(3, 6)
            chosen = random.sample(TRAIT_TIERS[0], n)
            self.traits = OrderedDict(
                (t, random.randint(1, 7)) for t in chosen
            )
        self.fused_traits = OrderedDict(fused_traits or {})
        self.goal = goal or random.choice(GOALS[:10])  # start with basic goals
        self.pop = pop or random.randint(50, 2000)
        self.fitness = 0.5
        self.alive = True
        self.mutations = []
        self.events = []
        self.attributes = OrderedDict()
        self.age = 0

    def _has_prerequisites(self, trait: str) -> bool:
        """Check if this species meets all prerequisites for a trait."""
        if trait not in PREREQUISITES:
            return True  # T0 traits have no prerequisites
        for req_trait, req_val in PREREQUISITES[trait]:
            val = self.traits.get(req_trait, 0)
            if val < req_val:
                # Also check fused traits
                fval = self.fused_traits.get(req_trait, 0)
                if fval < req_val:
                    return False
        return True

    def _resolve_attributes(self):
        """Derive non-numeric attributes from current trait levels."""
        self.attributes.clear()
        all_traits = dict(self.traits)
        all_traits.update(self.fused_traits)
        for trait, thresholds in ATTRIBUTE_TRIGGERS.items():
            val = all_traits.get(trait, 0)
            if val <= 0:
                continue
            # Pick the highest threshold met
            best = None
            for thresh, akey, aval in thresholds:
                if val >= thresh:
                    best = (akey, aval)
            if best:
                self.attributes[best[0]] = best[1]

    def compute_fitness(self, env: dict) -> float:
        """Compute fitness from trait-environment interaction."""
        score = 0.5
        all_t = dict(self.traits)
        all_t.update(self.fused_traits)

        # Physical fitness from environment
        if "temp" in env and "metab" in all_t:
            optimal_temp = all_t["metab"] / 10
            score -= abs(optimal_temp - env["temp"]) * 0.08

        if "toxin" in env:
            resist = all_t.get("regen", 0) / 10
            score -= max(0, env["toxin"] - resist * 0.5) * 0.12

        if "water" in env:
            if "aquatic" in all_t:
                score += all_t["aquatic"] / 10 * env["water"] * 0.06
            elif env["water"] < 0.3:
                score -= 0.05

        if "oxygen" in env and env["oxygen"] < 0.4:
            score -= (0.4 - env["oxygen"]) * 0.15

        if "radiation" in env and env["radiation"] > 0.5:
            resist = all_t.get("regen", 0) / 10
            score -= max(0, env["radiation"] - 0.5 - resist * 0.3) * 0.1

        # Behavioral / cognitive bonuses
        if "intel" in all_t:
            score += all_t["intel"] / 10 * 0.04
        if "planning" in all_t:
            score += all_t["planning"] / 10 * 0.03
        if "tooluse" in all_t:
            score += all_t["tooluse"] / 10 * 0.03
        if "social" in all_t and self.pop > 100:
            score += all_t["social"] / 10 * 0.03

        # Goal alignment bonus
        goal_trait_map = {
            "hunt": "speed", "evade": "camo", "forage": "sense",
            "territory": "aggression", "build": "tooluse",
            "cooperate": "social", "innovate": "creativity",
            "explore": "curiosity", "cultivate": "planning",
            "dominate": "aggression",
        }
        if self.goal in goal_trait_map:
            gt = goal_trait_map[self.goal]
            if gt in all_t:
                score += all_t[gt] / 10 * 0.04

        # Tier bonus: higher-tier species are more adaptable
        max_tier = max(
            (TRAIT_TO_TIER.get(t, 0) for t in all_t if all_t[t] > 0),
            default=0,
        )
        score += max_tier * 0.015

        self.fitness = round(max(0.0, min(1.0, score + random.gauss(0, 0.025))), 2)
        return self.fitness

    def _try_acquire_trait(self):
        """Attempt to acquire a new trait from the next tier up."""
        current_tiers = set(TRAIT_TO_TIER.get(t, 0) for t in self.traits if self.traits[t] > 0)
        max_tier = max(current_tiers, default=0)
        target_tier = min(max_tier + 1, 4)

        candidates = [
            t for t in TRAIT_TIERS.get(target_tier, [])
            if t not in self.traits
            and t not in self.fused_traits
            and self._has_prerequisites(t)
        ]
        # Also check same tier for unfilled slots
        same_tier = [
            t for t in TRAIT_TIERS.get(max_tier, [])
            if t not in self.traits
            and t not in self.fused_traits
            and self._has_prerequisites(t)
        ]
        candidates.extend(same_tier)

        if not candidates:
            return

        # Probability decreases with tier
        prob = {0: 0.15, 1: 0.08, 2: 0.05, 3: 0.03, 4: 0.015}
        trait = random.choice(candidates)
        tier = TRAIT_TO_TIER[trait]
        if random.random() < prob.get(tier, 0.01):
            init_val = random.randint(1, 3)
            self.traits[trait] = init_val
            self.mutations.append(("acquire", f"T{trait}", f"={init_val}"))

    def _try_prune_trait(self):
        """Remove a trait that is no longer useful (low value + low fitness)."""
        if self.fitness > 0.55:
            return  # no pressure to prune when doing well
        if len(self.traits) <= 2:
            return  # minimum viable trait count

        # Candidates: low-value traits in tiers that are energetically costly
        candidates = [
            (t, v) for t, v in self.traits.items()
            if v <= 2 and TRAIT_TO_TIER.get(t, 0) >= 1
        ]
        if not candidates:
            # Under severe pressure, even T0 traits can be lost
            if self.fitness < 0.3:
                candidates = [
                    (t, v) for t, v in self.traits.items() if v <= 1
                ]
        if not candidates:
            return

        if random.random() < 0.08:
            trait, _ = random.choice(candidates)
            # Check nothing depends on this trait
            dependents = [
                t for t in self.traits
                if t != trait
                and any(req == trait for req, _ in PREREQUISITES.get(t, []))
            ]
            if dependents:
                return  # can't prune if higher traits depend on it

            del self.traits[trait]
            self.mutations.append(("prune", f"T{trait}", ""))
            self.events.append(("trait_loss", f"{trait}_vestigial"))

    def _try_fuse_traits(self):
        """Check fusion rules and create composite traits."""
        all_t = dict(self.traits)
        all_t.update(self.fused_traits)

        for tA, tB, fused_name, threshold in FUSION_RULES:
            if fused_name in self.fused_traits:
                continue  # already fused
            vA = all_t.get(tA, 0)
            vB = all_t.get(tB, 0)
            if vA >= threshold and vB >= threshold and random.random() < 0.04:
                init_val = min(vA, vB) // 2 + 1
                self.fused_traits[fused_name] = min(10, init_val)
                self.mutations.append((
                    "fuse", f"T{tA}+T{tB}", f"->T{fused_name}={init_val}"
                ))
                break  # max one fusion per generation

    def _mutate_existing(self):
        """Point/drift/leap mutations on existing traits."""
        all_mutable = list(self.traits.keys()) + list(self.fused_traits.keys())
        if not all_mutable:
            return

        # More intelligent species have lower harmful mutation rates
        intel = self.traits.get("intel", 0) + self.fused_traits.get("intel", 0)
        mut_rate = max(0.1, 0.35 - intel * 0.01)

        if random.random() > mut_rate:
            return

        trait = random.choice(all_mutable)
        is_fused = trait in self.fused_traits
        container = self.fused_traits if is_fused else self.traits

        roll = random.random()
        if roll < 0.55:
            mut_type = "point"
            delta = random.choice([-1, 1])
        elif roll < 0.80:
            mut_type = "drift"
            delta = random.choice([-1, 0, 1])
        elif roll < 0.90:
            mut_type = "leap"
            delta = random.choice([-3, -2, 2, 3])
        else:
            mut_type = "loss"
            delta = -random.randint(1, 2)

        old_val = container[trait]
        new_val = max(0, min(10, old_val + delta))
        actual_delta = new_val - old_val
        if actual_delta == 0:
            return

        container[trait] = new_val
        sign = "+" if actual_delta > 0 else ""
        self.mutations.append((mut_type, f"T{trait}", f"{sign}{actual_delta}"))

        # If a trait hits 0, it might be pruned next generation
        if new_val == 0 and not is_fused:
            # Immediate removal for fused traits at 0
            if is_fused:
                del container[trait]
                self.events.append(("trait_loss", f"{trait}_collapsed"))

    def _update_goal(self):
        """Goal can shift based on available trait portfolio."""
        if random.random() > 0.04:
            return

        # Higher-tier species unlock advanced goals
        all_t = dict(self.traits)
        all_t.update(self.fused_traits)
        max_tier = max(
            (TRAIT_TO_TIER.get(t, 0) for t in all_t if all_t.get(t, 0) > 0),
            default=0,
        )

        available_goals = GOALS[:10]  # basic goals always available
        if max_tier >= 2:
            available_goals = GOALS[:13]  # explore, cultivate, build
        if max_tier >= 3:
            available_goals = GOALS[:15]  # + dominate, cooperate
        if max_tier >= 4:
            available_goals = GOALS  # + innovate

        old_goal = self.goal
        self.goal = random.choice(available_goals)
        if self.goal != old_goal:
            self.events.append(("adaptation", f"goal_{old_goal}_to_{self.goal}"))

    def step(self, env: dict):
        """One generation of evolution."""
        self.mutations.clear()
        self.events.clear()
        self.age += 1

        self.compute_fitness(env)

        # Population dynamics
        growth_rate = (self.fitness - 0.5) * 0.2 + random.gauss(0, 0.04)
        # Social species grow faster when population is healthy
        if "social" in self.traits and self.traits["social"] >= 4 and self.pop > 50:
            growth_rate += 0.02
        self.pop = max(0, int(self.pop * (1 + growth_rate)))

        if self.pop == 0:
            self.alive = False
            self.events.append(("extinction", random.choice(EXTINCTION_CAUSES)))
            return

        # --- Evolutionary operations ---

        # 1. Mutate existing traits
        self._mutate_existing()

        # 2. Try acquiring new traits (emergence)
        self._try_acquire_trait()

        # 3. Try pruning vestigial traits
        self._try_prune_trait()

        # 4. Try fusing traits into composites
        self._try_fuse_traits()

        # 5. Update goal based on capabilities
        self._update_goal()

        # 6. Resolve derived attributes
        self._resolve_attributes()

        # 7. Environmental stress events
        if self.fitness < 0.30 and random.random() < 0.5:
            self.events.append(("crisis", "population_collapse"))
            self.pop = max(1, self.pop // 2)
        elif self.fitness < 0.40 and random.random() < 0.3:
            self.events.append(("decline", random.choice(DECLINE_CAUSES)))
        elif self.fitness > 0.70 and self.pop > 300 and random.random() < 0.1:
            self.events.append(("boom", "thriving"))


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    def __init__(self, wid: int, n_species: Optional[int] = None,
                 n_env_axes: Optional[int] = None, scenario: Optional[str] = None):
        self.wid = wid
        n_env = n_env_axes or random.randint(3, 7)
        self.env_axes = random.sample(ENV_AXES, min(n_env, len(ENV_AXES)))
        self.env = OrderedDict(
            (ax, round(random.uniform(0.1, 0.9), 2)) for ax in self.env_axes
        )

        # Optional scenario modifiers
        self.scenario = scenario or random.choice([
            None, None, None,  # mostly no scenario
            "harsh", "paradise", "volatile", "toxic", "dark",
            "aquatic", "extreme_radiation", "resource_scarce",
        ])
        if self.scenario:
            self._apply_scenario()

        n_sp = n_species or random.randint(2, 8)
        self.species = [Species(i) for i in range(n_sp)]
        self.next_sid = n_sp
        self.gen = 0

    def _apply_scenario(self):
        """Modify environment based on scenario."""
        if self.scenario == "harsh":
            for ax in self.env:
                self.env[ax] = round(max(0.0, self.env[ax] - 0.2), 2)
        elif self.scenario == "paradise":
            for ax in ["water", "oxygen", "fertility"]:
                if ax in self.env:
                    self.env[ax] = round(min(1.0, self.env[ax] + 0.3), 2)
            if "toxin" in self.env:
                self.env["toxin"] = round(max(0.0, self.env["toxin"] - 0.3), 2)
        elif self.scenario == "volatile":
            pass  # handled by larger drift in drift_environment
        elif self.scenario == "toxic":
            if "toxin" in self.env:
                self.env["toxin"] = round(min(1.0, self.env["toxin"] + 0.4), 2)
            else:
                self.env_axes.append("toxin")
                self.env["toxin"] = round(random.uniform(0.5, 0.9), 2)
        elif self.scenario == "dark":
            if "light" in self.env:
                self.env["light"] = round(max(0.0, self.env["light"] - 0.4), 2)
        elif self.scenario == "aquatic":
            if "water" not in self.env:
                self.env_axes.append("water")
            self.env["water"] = round(random.uniform(0.8, 1.0), 2)
        elif self.scenario == "extreme_radiation":
            if "radiation" not in self.env:
                self.env_axes.append("radiation")
            self.env["radiation"] = round(random.uniform(0.6, 0.95), 2)
        elif self.scenario == "resource_scarce":
            if "fertility" in self.env:
                self.env["fertility"] = round(max(0.05, self.env["fertility"] - 0.4), 2)

    def drift_environment(self):
        """Gradual environmental change each generation."""
        sigma = 0.04 if self.scenario == "volatile" else 0.02
        for ax in self.env:
            self.env[ax] = round(
                max(0.0, min(1.0, self.env[ax] + random.gauss(0, sigma))), 2
            )

    def _try_speciation(self):
        """Branch a new species from an existing, thriving one."""
        alive = [s for s in self.species if s.alive]
        if len(alive) >= 15:
            return

        for sp in alive:
            if sp.pop > 400 and sp.fitness > 0.5 and random.random() < 0.04:
                child_traits = OrderedDict(sp.traits)
                # Mutate 1-4 traits for divergence
                n_muts = random.randint(1, min(4, len(child_traits)))
                for _ in range(n_muts):
                    t = random.choice(list(child_traits.keys()))
                    child_traits[t] = max(0, min(10,
                        child_traits[t] + random.choice([-2, -1, 1, 2])))

                # Child might lose a higher-tier trait in the split
                higher = [t for t in child_traits if TRAIT_TO_TIER.get(t, 0) >= 2]
                if higher and random.random() < 0.3:
                    lost = random.choice(higher)
                    del child_traits[lost]

                child_fused = OrderedDict()
                # Small chance to inherit fused traits
                for ft, fv in sp.fused_traits.items():
                    if random.random() < 0.4:
                        child_fused[ft] = max(1, fv - random.randint(0, 2))

                child = Species(
                    self.next_sid,
                    traits=child_traits,
                    fused_traits=child_fused,
                    goal=random.choice(GOALS[:10]),
                    pop=max(10, sp.pop // random.randint(5, 15)),
                )
                child.events.append(("speciation", f"from_S{sp.sid}"))
                self.species.append(child)
                self.next_sid += 1
                break

    def _try_symbiosis(self):
        """Two compatible species may form a symbiotic bond."""
        alive = [s for s in self.species if s.alive and s.pop > 50]
        if len(alive) < 2:
            return

        if random.random() > 0.02:
            return

        a, b = random.sample(alive, 2)
        # Symbiosis more likely between species with complementary traits
        a_traits = set(a.traits.keys())
        b_traits = set(b.traits.keys())
        complementary = len(a_traits.symmetric_difference(b_traits))
        if complementary >= 3 and random.random() < 0.5:
            a.events.append(("symbiosis", f"with_S{b.sid}"))
            b.events.append(("symbiosis", f"with_S{a.sid}"))
            # Both get a small fitness bump
            a.fitness = min(1.0, a.fitness + 0.05)
            b.fitness = min(1.0, b.fitness + 0.05)

    def step(self):
        """Advance the world by one generation."""
        self.drift_environment()
        for sp in self.species:
            if sp.alive:
                sp.step(self.env)
        self._try_speciation()
        self._try_symbiosis()
        self.gen += 1

    def serialize_generation(self) -> str:
        """Serialize current world state to protocol format."""
        lines = []

        # World header
        env_str = ",".join(f"E{ax}:{v:.2f}" for ax, v in self.env.items())
        header = f"W{self.wid}G{self.gen}|{env_str}"
        if self.scenario:
            header += f"|SC{self.scenario}"
        lines.append(header)

        # Species blocks
        for sp in self.species:
            if not sp.alive and not sp.events:
                continue

            if not sp.alive:
                lines.append(f"S{sp.sid}|P0|F0.00|Textinct")
                for etype, edesc in sp.events:
                    lines.append(f"  X{etype}:{edesc}")
                continue

            # Core traits
            trait_parts = []
            for t, v in sp.traits.items():
                trait_parts.append(f"T{t}={v}")
            # Fused traits (prefix with *)
            for t, v in sp.fused_traits.items():
                trait_parts.append(f"T*{t}={v}")

            trait_str = ",".join(trait_parts) if trait_parts else "Tnone"
            lines.append(
                f"S{sp.sid}|P{sp.pop}|F{sp.fitness:.2f}"
                f"|{trait_str}|G{sp.goal}"
            )

            # Attributes (non-numeric derived state)
            for akey, aval in sp.attributes.items():
                lines.append(f"  A{akey}:{aval}")

            # Mutations
            for mtype, mtarget, mdelta in sp.mutations:
                if mtype == "acquire":
                    lines.append(f"  M+{mtarget}{mdelta}")
                elif mtype == "prune":
                    lines.append(f"  M-{mtarget}")
                elif mtype == "fuse":
                    lines.append(f"  Mfuse:{mtarget}{mdelta}")
                else:
                    lines.append(f"  M{mtype}:{mtarget}={mdelta}")

            # Events
            for etype, edesc in sp.events:
                lines.append(f"  X{etype}:{edesc}")

        lines.append("---")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_worlds: int, gens_per_world: int, output_path: str,
                     verbose: bool = False):
    """Generate simulation traces and write to text file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    total_speciations = 0
    total_extinctions = 0
    max_tier_reached = 0

    with open(path, "w") as f:
        for wid in range(n_worlds):
            world = World(wid)
            for g in range(gens_per_world):
                world.step()
                block = world.serialize_generation()
                f.write(block + "\n")
                total_lines += block.count("\n") + 1

                # Stats tracking
                for sp in world.species:
                    for etype, _ in sp.events:
                        if etype == "speciation":
                            total_speciations += 1
                        elif etype == "extinction":
                            total_extinctions += 1
                    if sp.alive:
                        for t in list(sp.traits.keys()) + list(sp.fused_traits.keys()):
                            tier = TRAIT_TO_TIER.get(t, 0)
                            max_tier_reached = max(max_tier_reached, tier)

            f.write("\n")  # blank line between worlds = EOS boundary

            if verbose and (wid + 1) % max(1, n_worlds // 20) == 0:
                pct = (wid + 1) / n_worlds * 100
                alive_count = sum(1 for s in world.species if s.alive)
                print(
                    f"  [{pct:5.1f}%] W{wid}: {alive_count} alive species, "
                    f"gen {world.gen}, scenario={world.scenario}"
                )

    print(f"\nDataset generated: {path}")
    print(f"  Worlds: {n_worlds}")
    print(f"  Generations per world: {gens_per_world}")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Total speciations: {total_speciations:,}")
    print(f"  Total extinctions: {total_extinctions:,}")
    print(f"  Max trait tier reached: T{max_tier_reached} "
          f"({['Physical','Behavioral','Cognitive','Cultural','Abstract'][max_tier_reached]})")
    print(f"  File size: {path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evolutionary simulation traces for MANTIS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (small)
  python scripts/gen_evo_dataset.py --worlds 100 --generations 50

  # Pilot run
  python scripts/gen_evo_dataset.py --worlds 1000 --generations 100

  # Production dataset
  python scripts/gen_evo_dataset.py --worlds 10000 --generations 200 --verbose

  # Then train with:
  python train.py --stage 1 data/evo_train.txt --val-split 0.1 --model-size tiny
        """,
    )
    parser.add_argument("--worlds", type=int, default=10000,
                        help="Number of independent worlds (default: 10000)")
    parser.add_argument("--generations", type=int, default=100,
                        help="Generations per world (default: 100)")
    parser.add_argument("--output", type=str, default="data/evo_train.txt",
                        help="Output file path (default: data/evo_train.txt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress during generation")
    args = parser.parse_args()

    random.seed(args.seed)
    generate_dataset(args.worlds, args.generations, args.output, args.verbose)


if __name__ == "__main__":
    main()
