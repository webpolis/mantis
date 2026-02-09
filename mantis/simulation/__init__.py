# mantis/simulation — Ecological evolution simulator
# No CUDA-dependent imports here.

from .constants import (
    ALL_TRAITS,
    ATTRIBUTE_TRIGGERS,
    BODY_PLANS,
    BODY_PLAN_TRANSITIONS,
    BRAIN_TAX,
    EPOCH_CONFIG,
    EXTINCTION_CAUSES,
    FUSION_RULES,
    GOALS,
    PREREQUISITES,
    TRAIT_TIERS,
    TRAIT_TO_TIER,
    Epoch,
)
from .species import BodyPlan, DietVector, Species, TraitDistribution
from .biome import Biome, BIOME_NAMES
from .engine import World, Interaction, SpotlightEvent, Hero, CulturalMemory
from .serializer import Serializer
