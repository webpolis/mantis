"""
Constants for the ecological evolution simulator.

Defines trait taxonomies, body plan constraints, transition rules,
prerequisite maps, fusion rules, attribute triggers, energy parameters,
and epoch definitions.
"""

from enum import IntEnum

# ---------------------------------------------------------------------------
# Trait taxonomy
# ---------------------------------------------------------------------------

TRAIT_TIERS = {
    0: [
        "size", "speed", "armor", "metab", "sense", "camo", "repro", "regen",
        "venom", "photosynth", "mouth", "endurance", "chem_digest",
        "toxin_resist", "toxin",
    ],
    1: ["social", "aggression", "curiosity", "patience", "nocturnal"],
    2: ["intel", "memory", "learning", "planning", "deception"],
    3: ["language", "tooluse", "ritual", "teaching", "trade"],
    4: ["subconscious", "theory_of_mind", "creativity", "abstraction", "ethics"],
}

ALL_TRAITS: list[str] = []
TRAIT_TO_TIER: dict[str, int] = {}
for _tier, _traits in TRAIT_TIERS.items():
    for _t in _traits:
        ALL_TRAITS.append(_t)
        TRAIT_TO_TIER[_t] = _tier

# ---------------------------------------------------------------------------
# Body plan definitions
# ---------------------------------------------------------------------------
# A trait is allowed if it is NOT in the blocked set.
# "caps" limits specific traits to a maximum tier-value.

BODY_PLANS: dict[str, dict] = {
    "sessile_autotroph": {
        "blocked": {
            "speed", "sense", "mouth", "endurance", "camo",
            # All T1+
            "social", "aggression", "curiosity", "patience", "nocturnal",
            "intel", "memory", "learning", "planning", "deception",
            "language", "tooluse", "ritual", "teaching", "trade",
            "subconscious", "theory_of_mind", "creativity", "abstraction", "ethics",
        },
        "caps": {},
        "base_metabolism": 0.5,
    },
    "mobile_autotroph": {
        "blocked": {
            "mouth", "venom", "endurance", "chem_digest", "toxin_resist",
            # T1+ cognitive blocked
            "social", "aggression", "curiosity", "patience", "nocturnal",
            "intel", "memory", "learning", "planning", "deception",
            "language", "tooluse", "ritual", "teaching", "trade",
            "subconscious", "theory_of_mind", "creativity", "abstraction", "ethics",
        },
        "caps": {"speed": 4},
        "base_metabolism": 0.8,
    },
    "filter_feeder": {
        "blocked": {
            "venom", "photosynth", "endurance", "chem_digest", "toxin_resist",
            "aggression", "curiosity", "nocturnal",
            "intel", "memory", "learning", "planning", "deception",
            "language", "tooluse", "ritual", "teaching", "trade",
            "subconscious", "theory_of_mind", "creativity", "abstraction", "ethics",
        },
        "caps": {"speed": 3},
        "base_metabolism": 0.7,
    },
    "grazer": {
        "blocked": {
            "photosynth", "chem_digest", "toxin_resist", "toxin",
        },
        "caps": {},
        "base_metabolism": 1.2,
    },
    "predator": {
        "blocked": {
            "photosynth", "chem_digest", "toxin_resist", "toxin",
        },
        "caps": {"armor": 4},
        "base_metabolism": 1.8,
    },
    "scavenger": {
        "blocked": {
            "photosynth", "venom", "chem_digest", "toxin_resist", "toxin",
        },
        "caps": {},
        "base_metabolism": 1.0,
    },
    "omnivore": {
        "blocked": {
            "photosynth", "chem_digest", "toxin_resist", "toxin",
        },
        "caps": {},
        "base_metabolism": 1.5,
    },
    "parasite": {
        "blocked": {
            "photosynth", "endurance", "mouth", "chem_digest", "toxin_resist",
            # Cognitive gated hard
            "intel", "memory", "learning", "planning",
            "language", "tooluse", "ritual", "teaching", "trade",
            "subconscious", "theory_of_mind", "creativity", "abstraction", "ethics",
        },
        "caps": {"size": 3, "speed": 3},
        "base_metabolism": 0.6,
    },
    "decomposer": {
        "blocked": {
            "speed", "photosynth", "venom", "mouth", "endurance",
            "social", "aggression", "curiosity", "patience", "nocturnal",
            "intel", "memory", "learning", "planning", "deception",
            "language", "tooluse", "ritual", "teaching", "trade",
            "subconscious", "theory_of_mind", "creativity", "abstraction", "ethics",
        },
        "caps": {},
        "base_metabolism": 0.4,
    },
}

# ---------------------------------------------------------------------------
# Body plan transition rules
# ---------------------------------------------------------------------------
# Each entry: (from_plan, to_plan, condition_dict)
# Condition keys:
#   detritus_min / meat_min / plant_min / solar_max — diet category proportions
#   meat_plus_detritus_min — combined threshold
#   has_speed — species must have speed trait > 0
#   parasite_feeding — species has a parasitic diet source

BODY_PLAN_TRANSITIONS: list[tuple[str, str, dict]] = [
    # Autotroph transitions
    ("sessile_autotroph", "mobile_autotroph", {"has_speed": True}),
    ("mobile_autotroph", "filter_feeder", {"solar_max": 0.3, "detritus_min": 0.3}),
    # Scavenger gateway
    ("grazer", "scavenger", {"detritus_min": 0.3}),
    ("grazer", "omnivore", {"meat_plus_detritus_min": 0.3}),
    ("scavenger", "omnivore", {"meat_min": 0.2, "plant_min": 0.15}),
    ("scavenger", "predator", {"meat_min": 0.5}),
    ("omnivore", "predator", {"meat_min": 0.6}),
    # Decomposer
    ("filter_feeder", "decomposer", {"detritus_min": 0.7}),
    # Parasite
    ("predator", "parasite", {"parasite_feeding": True}),
]

# ---------------------------------------------------------------------------
# Prerequisites: {trait: [(required_trait, min_value), ...]}
# ---------------------------------------------------------------------------

PREREQUISITES: dict[str, list[tuple[str, int]]] = {
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

# ---------------------------------------------------------------------------
# Fusion rules: (traitA, traitB, fused_name, threshold)
# ---------------------------------------------------------------------------

FUSION_RULES: list[tuple[str, str, str, int]] = [
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

# Collect fused trait names for tier lookups
FUSED_TRAIT_NAMES: list[str] = [r[2] for r in FUSION_RULES]

# ---------------------------------------------------------------------------
# Attribute triggers: {trait: [(threshold, attr_key, attr_value), ...]}
# ---------------------------------------------------------------------------

ATTRIBUTE_TRIGGERS: dict[str, list[tuple[int, str, str]]] = {
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

# ---------------------------------------------------------------------------
# Goals and environment
# ---------------------------------------------------------------------------

GOALS = [
    "forage", "hunt", "evade", "territory", "parasite", "symbiote",
    "migrate", "hoard", "nurture", "swarm", "explore", "cultivate",
    "build", "dominate", "cooperate", "innovate",
]

ENV_AXES = [
    "temp", "water", "light", "toxin", "oxygen", "radiation",
    "fertility", "density", "altitude", "salinity",
]

EXTINCTION_CAUSES = [
    "drought", "predation", "disease", "competition", "habitat_loss",
    "volcanic", "ice_age", "radiation_spike", "overpopulation", "famine",
]

DECLINE_CAUSES = [
    "food_scarcity", "water_scarcity", "disease_outbreak",
    "inbreeding", "territorial_loss", "climate_shift",
]

# ---------------------------------------------------------------------------
# Energy constants
# ---------------------------------------------------------------------------

TROPHIC_EFFICIENCY = 0.12          # 10-15% range, canonical ~12%
KLEIBER_EXPONENT = 0.75            # mass^0.75 metabolic scaling

# Brain tax: energy cost multiplier per cognitive tier
# Applied as sum(trait_mean * multiplier) for all traits in that tier
BRAIN_TAX: dict[int, float] = {
    0: 0.0,    # Physical traits — no extra cognitive cost
    1: 0.02,   # Behavioral — slight overhead
    2: 0.08,   # Cognitive — significant
    3: 0.15,   # Cultural — expensive
    4: 0.25,   # Abstract — very expensive
}

STARVATION_THRESHOLD = 0.0         # energy_store at which pop loss begins
STARVATION_RATE = 0.10             # fraction of pop lost per tick when starving
FAT_STORE_MAX_FACTOR = 5.0         # max energy_store = factor * per-tick cost
REPRODUCTION_ENERGY_FRACTION = 0.3 # fraction of surplus allocated to reproduction

# ---------------------------------------------------------------------------
# Nutrient cycling
# ---------------------------------------------------------------------------

NUTRIENT_RELEASE_N = 0.08    # N released per unit detritus decay
NUTRIENT_RELEASE_P = 0.03    # P released per unit detritus decay (slower)
NUTRIENT_UPTAKE_RATE = 0.04  # N/P consumed per unit vegetation growth

# ---------------------------------------------------------------------------
# Disease
# ---------------------------------------------------------------------------

DISEASE_BASE_PROB = 0.015    # per-species per-tick base probability
DISEASE_TYPES = ["plague", "blight", "parasitic_worm", "viral_outbreak", "fungal_rot"]

# ---------------------------------------------------------------------------
# Catastrophe
# ---------------------------------------------------------------------------

CATASTROPHE_PROB = 0.005     # per-tick (expect ~1 per 200 ticks)
CATASTROPHE_TYPES: dict[str, dict] = {
    "volcanic_winter": {"solar_mult": 0.1, "duration_range": (5, 10)},
    "meteor_impact":   {"veg_set": 0.0, "detritus_spike": 500, "pop_loss": (0.5, 0.9), "duration_range": (1, 1)},
    "ice_age":         {"veg_growth_mult": 0.3, "duration_range": (15, 30)},
}

# ---------------------------------------------------------------------------
# Digestive efficiency
# ---------------------------------------------------------------------------

BODY_PLAN_DIET_AFFINITY: dict[str, dict[str, float]] = {
    "sessile_autotroph": {"solar": 1.0},
    "mobile_autotroph":  {"solar": 1.0, "plant": 0.7},
    "filter_feeder":     {"detritus": 1.0, "plant": 0.8},
    "grazer":            {"plant": 1.0, "detritus": 0.7},
    "scavenger":         {"detritus": 1.0, "plant": 0.8, "meat": 0.5},
    "omnivore":          {"plant": 0.9, "detritus": 0.8, "meat": 0.9},
    "predator":          {"meat": 1.0, "detritus": 0.6},
    "parasite":          {"meat": 1.0},
    "decomposer":        {"detritus": 1.0},
}
DIET_MISMATCH_PENALTY = 0.4  # default efficiency for unlisted diet categories

# ---------------------------------------------------------------------------
# Symbiogenesis
# ---------------------------------------------------------------------------

SYMBIOGENESIS_MIN_TICKS = 5   # sustained co-location ticks required
SYMBIOGENESIS_PROB = 0.02     # per-tick probability once threshold met

# ---------------------------------------------------------------------------
# Variance-as-evolvability
# ---------------------------------------------------------------------------

STRESS_MLEAP_MULT = 5.0      # Mleap probability multiplier under stress + high variance
LOW_VARIANCE_THRESHOLD = 0.15 # below this, species is "over-specialized"
HIGH_VARIANCE_THRESHOLD = 0.5 # above this, species can do evolutionary rescue

# ---------------------------------------------------------------------------
# Epoch definitions
# ---------------------------------------------------------------------------


class Epoch(IntEnum):
    PRIMORDIAL = 1    # ~1000 microbial generations per tick
    CAMBRIAN = 2      # ~10 generations per tick
    ECOSYSTEM = 3     # ~1 generation per tick
    INTELLIGENCE = 4  # Sub-generational (spotlight-level)


EPOCH_CONFIG: dict[Epoch, dict] = {
    Epoch.PRIMORDIAL: {
        "tick_scale": 1000,
        "mutation_rate_mult": 3.0,
        "speciation_prob": 0.08,
        "description": "Chemical/solar energy, simple producers + consumers",
    },
    Epoch.CAMBRIAN: {
        "tick_scale": 10,
        "mutation_rate_mult": 1.5,
        "speciation_prob": 0.06,
        "description": "Body plan diversification, predation emerges",
    },
    Epoch.ECOSYSTEM: {
        "tick_scale": 1,
        "mutation_rate_mult": 1.0,
        "speciation_prob": 0.04,
        "description": "Niche specialization, migration, stable food webs",
    },
    Epoch.INTELLIGENCE: {
        "tick_scale": 0.1,
        "mutation_rate_mult": 0.5,
        "speciation_prob": 0.02,
        "description": "Individual-level events, culture, language",
    },
}
