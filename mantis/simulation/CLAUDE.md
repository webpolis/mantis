# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Module Does

`mantis/simulation` is an ecological evolution simulator (~3,250 lines) that generates synthetic training data for the MANTIS LLM. It models populations of species competing for energy through food webs, evolving traits across 5 tiers, transitioning body plans, and (in the INTELLIGENCE epoch) running individual-based agents with spatial behaviors and cultural memories.

The output is protocol-formatted text consumed by `mantis/tokenizer.py` (283 domain tokens in a custom trie-based tokenizer, 512 total with reserved slots) for next-token prediction training.

## Generating Data

```bash
# Basic (sequential)
python scripts/gen_evo_dataset.py --worlds 100 --max-generations 50 --verbose

# Parallel with agents enabled
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 \
    --workers 8 --seed 42 --enable-agents --agent-epoch INTELLIGENCE

# Agents at earlier epoch (simpler behaviors)
python scripts/gen_evo_dataset.py --worlds 1000 --enable-agents --agent-epoch ECOSYSTEM

# Partitioned generation for curriculum training (cap by epoch)
python scripts/gen_evo_dataset.py --worlds 5000 --max-epoch CAMBRIAN  --output data/evo_bio.txt --compact --workers 8
python scripts/gen_evo_dataset.py --worlds 5000 --max-epoch ECOSYSTEM --output data/evo_eco.txt --compact --workers 8 --enable-agents
python scripts/gen_evo_dataset.py --worlds 5000                       --output data/evo_intel.txt --compact --workers 8 --enable-agents
```

`--max-epoch` caps worlds at the named epoch (uses `>` comparison: `--max-epoch ECOSYSTEM` allows entry into ECOSYSTEM but stops before INTELLIGENCE).

`gen_evo_dataset.py` uses `importlib` to import this module directly, bypassing `mantis/__init__.py` to avoid pulling in CUDA dependencies. This module has **no CUDA deps** — only numpy and stdlib.

## Programmatic Usage

```python
from mantis.simulation import World, Serializer

world = World(wid=0, seed=42, enable_agents=True, agent_epoch="INTELLIGENCE")
serializer = Serializer(keyframe_interval=20)

for _ in range(100):
    world.step()
    text = serializer.serialize_tick(world)
    if not any(sp.alive for sp in world.species):
        break
```

## Module Architecture

Data flows in one direction: **constants → species/biome → agent/behavior/spatial → engine → serializer**

- **`constants.py`** — All rules: trait taxonomy (5 tiers, 35 traits), body plans (9 types), prerequisites, fusion rules, epoch configs (mutation_rate_mult is raw — engine divides by tick_scale), energy constants. This is the single source of truth for simulation parameters.
- **`species.py`** — `Species` (population + traits as `TraitDistribution` mean/variance + `DietVector` + `BodyPlan`). Population-level state.
- **`biome.py`** — `Biome` (location with vegetation/detritus/solar/environmental axes).
- **`spatial.py`** — `SpatialHash` (100-unit grid cells, O(k) neighbor queries) + `VegetationPatch` (Gaussian density falloff, logistic regrowth).
- **`agent.py`** — `Agent` (individual organism with sampled traits, position, energy, behavioral state) + `AgentManager` (per-species agent pool, max 250/species).
- **`behavior.py`** — Utility-based AI: softmax action selection with hysteresis commitment periods. Actions: forage, hunt, mate, flee, flock. Emergency override at energy < 10.
- **`agent_reconciliation.py`** — `PopulationReconciler`: dual-layer accounting. Discrete events (birth/death) map 1:1. Continuous energy scales by population/agent_count ratio.
- **`engine.py`** — `World` orchestrator. Per-tick pipeline: energy flows → interactions → mutations → body plan transitions → speciation → extinction → epoch check → agent stepping → spotlights.
- **`serializer.py`** — Converts `World` state to protocol text (`=EPOCH`, `@BIO`, `@SP`, `@INT`, `@EVT`, `@SPOT` blocks). Keyframe every 20 ticks, delta encoding between.
- **`agent_serializer.py`** — `@AGENT` blocks using per-agent format: every agent listed individually with 10-unit quantized positions, energy, age, and behavioral state.

## Simulation Pipeline (per tick)

1. Energy accounting — food web (producer → herbivore → predator). Agent-managed species are **skipped** (agents handle their own energy via foraging/metabolism, reconciled back via `PopulationReconciler`).
2. Interactions — hunts, grazes, parasitism (population-level probability). Agent-managed species are **skipped** for predation to prevent double-counting.
3. Population update — surplus → reproduction, deficit → starvation. Minimum viable population floor (10) only applies when species has energy surplus (not being starved/hunted out).
4. Mutations — trait drift per generation (rate normalized by `mutation_rate_mult / tick_scale`), respecting body plan caps.
5. Body plan transitions — diet-driven morphological changes
6. Speciation — split when genetic divergence is high (SD=0.2 trait divergence, gradual not saltational)
7. Extinction — population ≤ 0 or starvation
8. Epoch transitions — triggered by complexity milestones (max tier reached)
9. Agent stepping — if active: behavior selection → movement → local interactions → metabolism → death/birth
10. Spotlights — INTELLIGENCE epoch cultural reasoning scenes for high-cognition species

## Protocol Format

Output tokens processed by `mantis/tokenizer.py` with per-block loss weights:

| Block | Weight | Content |
|-------|--------|---------|
| `=EPOCH` | 0.5 | Epoch header, tick scale |
| `@BIO` | 0.5 | Biome vegetation/detritus/solar state |
| `@SP` | 1.0 | Species traits, population, diet, energy |
| `@INT` | 1.5 | Interactions (hunt, graze, compete) |
| `@EVT` | 1.5 | Mutations, extinctions, body plan changes |
| `@SPOT` | 2.0 | Intelligence spotlight scenes (CTX/ACTORS/INTENT/REACT/RESOLVE/EFFECT) |
| `@AGENT` | 0.8 | Agent positions, states (keyframe or delta) |
| `---` | 0.1 | Tick separator |

## Key Design Decisions

- **Energy-based, not fitness-based**: Population dynamics driven by actual energy flows through food web (Kleiber scaling, trophic efficiency). Species starve or thrive based on energy balance, not abstract fitness scores. Plant/solar income scales with population (per-capita harvest × head count, capped by available resources).
- **Body plans constrain evolution**: Each of 9 body plans blocks/caps certain traits. A `sessile_autotroph` can't evolve `speed`. Transitions happen when diet changes enough (e.g., grazer → omnivore when meat intake exceeds threshold).
- **Dual-layer simulation**: Population-level dynamics (always on) + optional agent-level individuals (activated per-epoch). `PopulationReconciler` prevents the two layers from diverging. Agent-managed species are skipped in population-level energy computation and predation to prevent double-counting.
- **Per-generation mutation normalization**: Raw `mutation_rate_mult` is divided by `tick_scale` so PRIMORDIAL (1000 gen/tick) and INTELLIGENCE (0.1 gen/tick) have comparable per-generation mutation rates.
- **Vegetation resilience**: Seed bank floor (0.005) prevents permanent vegetation death. Geological nutrient buffer slowly replenishes N/P even without decomposers.
- **Symbiogenesis restricted**: Only occurs in PRIMORDIAL/CAMBRIAN epochs (real endosymbiosis is an ancient event). Requires 20 ticks of co-location, 0.3% per-tick probability.
- **Hysteresis in behavior**: Agents commit to actions for multiple ticks (flee: 10, hunt: 8, forage: 3) to prevent oscillation. Emergency energy override breaks commitment.
- **Agent metabolism matches population-level**: Agent basal cost uses `body_plan.base_metabolism × size^0.75`, plus brain tax from cognitive traits (same formula as `_compute_cost` in engine.py).
- **Keyframe + delta serialization**: Full state every 20 ticks, only changes between. Agent blocks list every agent individually with 10-unit quantized positions. Delta encoding emits only agents whose position moved >5 units, energy changed >2, or age changed, plus dead agents marked with `†`.

## Gotchas

- **Import path**: Always import via `from mantis.simulation import ...` or use the `importlib` trick in `gen_evo_dataset.py`. Never import `mantis` top-level in contexts without CUDA (it pulls in `mamba_ssm`).
- **All domain words are atomic tokens**: Trait names ("speed", "size", "armor"), body plans, biome names, and protocol markers are all single tokens in the trie-based tokenizer. Numbers are digit-by-digit.
- **`constants.py` is the single source of truth**: All trait lists, body plan rules, epoch thresholds, and energy constants live here. Don't scatter magic numbers into other modules.
- **Species cap**: Hard limit of 20 alive species per world (speciation returns early if cap reached).
- **Agent max limits**: 250 agents/species (`AGENT_MAX_PER_SPECIES`), enforced in `AgentManager`. `SpatialHash` cell size (100 units) matches max sense range.
- **RNG discipline**: `World` takes a seed and creates `self.rng = np.random.default_rng(seed)`. All randomness must flow through `self.rng` for reproducibility. Never use `np.random` module-level.
- **Diet preservation**: `DietVector.mutate()` always keeps the dominant source to prevent degeneracy where a species loses all feeding ability.
- **Population floor is conditional**: Minimum viable population (10) only kicks in when `e_in >= e_out` (energy surplus). Species being starved or hunted into deficit decline naturally to extinction.
