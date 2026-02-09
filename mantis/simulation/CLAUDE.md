# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Module Does

`mantis/simulation` is an ecological evolution simulator (~3,250 lines) that generates synthetic training data for the MANTIS LLM. It models populations of species competing for energy through food webs, evolving traits across 5 tiers, transitioning body plans, and (in the INTELLIGENCE epoch) running individual-based agents with spatial behaviors and cultural memories.

The output is protocol-formatted text consumed by `mantis/tokenizer.py` (87 custom tokens on top of GPT-2 BPE) for next-token prediction training.

## Generating Data

```bash
# Basic (sequential)
python scripts/gen_evo_dataset.py --worlds 100 --max-generations 50 --verbose

# Parallel with agents enabled
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 \
    --workers 8 --seed 42 --enable-agents --agent-epoch INTELLIGENCE

# Agents at earlier epoch (simpler behaviors)
python scripts/gen_evo_dataset.py --worlds 1000 --enable-agents --agent-epoch ECOSYSTEM
```

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

- **`constants.py`** — All rules: trait taxonomy (5 tiers, 35 traits), body plans (9 types), prerequisites, fusion rules, epoch configs, energy constants. This is the single source of truth for simulation parameters.
- **`species.py`** — `Species` (population + traits as `TraitDistribution` mean/variance + `DietVector` + `BodyPlan`). Population-level state.
- **`biome.py`** — `Biome` (location with vegetation/detritus/solar/environmental axes).
- **`spatial.py`** — `SpatialHash` (100-unit grid cells, O(k) neighbor queries) + `VegetationPatch` (Gaussian density falloff, logistic regrowth).
- **`agent.py`** — `Agent` (individual organism with sampled traits, position, energy, behavioral state) + `AgentManager` (per-species agent pool, max 250/species).
- **`behavior.py`** — Utility-based AI: softmax action selection with hysteresis commitment periods. Actions: forage, hunt, mate, flee, flock. Emergency override at energy < 10.
- **`agent_reconciliation.py`** — `PopulationReconciler`: dual-layer accounting. Discrete events (birth/death) map 1:1. Continuous energy scales by population/agent_count ratio.
- **`engine.py`** — `World` orchestrator. Per-tick pipeline: energy flows → interactions → mutations → body plan transitions → speciation → extinction → epoch check → agent stepping → spotlights.
- **`serializer.py`** — Converts `World` state to protocol text (`=EPOCH`, `@BIO`, `@SP`, `@INT`, `@EVT`, `@SPOT` blocks). Keyframe every 20 ticks, delta encoding between.
- **`agent_serializer.py`** — `@AGENT` blocks with quantized coordinates (10-unit grid), keyframe/delta encoding.

## Simulation Pipeline (per tick)

1. Energy accounting — food web (producer → herbivore → predator)
2. Interactions — hunts, grazes, parasitism (population-level probability)
3. Mutations — trait drift (directional or random, respecting body plan caps)
4. Body plan transitions — diet-driven morphological changes
5. Speciation — split when genetic divergence is high
6. Extinction — population ≤ 0 or starvation
7. Epoch transitions — triggered by complexity milestones (max tier reached)
8. Agent stepping — if active: behavior selection → movement → local interactions → metabolism → death/birth
9. Spotlights — INTELLIGENCE epoch cultural reasoning scenes for high-cognition species

## Protocol Format

Output tokens processed by `mantis/tokenizer.py` with per-block loss weights:

| Block | Weight | Content |
|-------|--------|---------|
| `=EPOCH` | 0.3 | Epoch header, tick scale |
| `@BIO` | 0.5 | Biome vegetation/detritus/solar state |
| `@SP` | 1.0 | Species traits, population, diet, energy |
| `@INT` | 1.5 | Interactions (hunt, graze, compete) |
| `@EVT` | 1.5 | Mutations, extinctions, body plan changes |
| `@SPOT` | 2.0 | Intelligence spotlight scenes (CTX/ACTORS/INTENT/REACT/RESOLVE/EFFECT) |
| `@AGENT` | 0.8 | Agent positions, states (keyframe or delta) |
| `---` | 0.1 | Tick separator |

## Key Design Decisions

- **Energy-based, not fitness-based**: Population dynamics driven by actual energy flows through food web (Kleiber scaling, trophic efficiency). Species starve or thrive based on energy balance, not abstract fitness scores.
- **Body plans constrain evolution**: Each of 9 body plans blocks/caps certain traits. A `sessile_autotroph` can't evolve `speed`. Transitions happen when diet changes enough (e.g., grazer → omnivore when meat intake exceeds threshold).
- **Dual-layer simulation**: Population-level dynamics (always on) + optional agent-level individuals (activated per-epoch). `PopulationReconciler` prevents the two layers from diverging.
- **Hysteresis in behavior**: Agents commit to actions for multiple ticks (flee: 10, hunt: 8, forage: 3) to prevent oscillation. Emergency energy override breaks commitment.
- **Keyframe + delta serialization**: Full state every 20 ticks, only changes between. Agent coordinates quantized to 10-unit grid to reduce token entropy.

## Gotchas

- **Import path**: Always import via `from mantis.simulation import ...` or use the `importlib` trick in `gen_evo_dataset.py`. Never import `mantis` top-level in contexts without CUDA (it pulls in `mamba_ssm`).
- **Trait names are GPT-2 tokens**: Common English words like "speed", "size", "armor" are already single GPT-2 tokens. The tokenizer does NOT re-add them — it only adds protocol markers and rare compound terms.
- **`constants.py` is the single source of truth**: All trait lists, body plan rules, epoch thresholds, and energy constants live here. Don't scatter magic numbers into other modules.
- **Agent max limits**: 250 agents/species (`AGENT_MAX_PER_SPECIES`), enforced in `AgentManager`. `SpatialHash` cell size (100 units) matches max sense range.
- **RNG discipline**: `World` takes a seed and creates `self.rng = np.random.default_rng(seed)`. All randomness must flow through `self.rng` for reproducibility. Never use `np.random` module-level.
