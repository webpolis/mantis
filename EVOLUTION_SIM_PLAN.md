# Ecological Evolution Simulator — Implementation Plan

## Goal

Rewrite `scripts/gen_evo_dataset.py` into a full ecological simulation engine that
generates training data for MANTIS. The trained model should be able to simulate an
evolving universe for a video game where:

- Species survive by acquiring energy from actual sources (sun, chemical, other species)
- Food webs emerge from trait evolution, not hardcoded roles
- Predation, mutualism, parasitism, and competition arise naturally
- Intelligent species develop language, culture, individual-level interactions
- A player can zoom into any species and see individual agents with unique traits

The output remains a `.txt` file consumable by `TextDataset` in `train.py`.

---

## Architecture Overview

```
SimulationEngine
├── TimeManager (epoch transitions, tick scaling)
├── World
│   ├── BiomeGraph (spatial structure)
│   │   └── BiomeNode[] (resources, conditions, detritus)
│   ├── Species[]
│   │   ├── TraitDistribution (mean ± variance per trait)
│   │   ├── DietVector (feeding preferences, evolves)
│   │   ├── BodyPlan (morphological constraints)
│   │   ├── ReproductionStrategy (r/K spectrum)
│   │   └── CulturalMemory (memes, taboos — T2+ only)
│   └── InteractionMatrix (who eats whom, competition overlaps)
├── Systems (per-tick logic)
│   ├── ProductionSystem (autotroph energy capture)
│   ├── ConsumptionSystem (grazing, predation, scavenging)
│   ├── MetabolismSystem (energy costs, starvation)
│   ├── ReproductionSystem (population growth, r/K logic)
│   ├── DecompositionSystem (death → detritus → nutrients)
│   ├── EvolutionSystem (mutation, selection, speciation)
│   ├── MigrationSystem (movement between biomes)
│   └── SpotlightSystem (individual-level scenes for T2+)
├── NarrativeDirector (prune boring worlds, fork promising ones)
└── LayeredSerializer (encode state to training protocol)
```

---

## File Structure

All simulation code lives under `mantis/simulation/`. The entry-point script
remains at `scripts/gen_evo_dataset.py` but delegates to the package.

```
mantis/simulation/
├── __init__.py
├── engine.py           # SimulationEngine, TimeManager, EpochConfig
├── environment.py      # BiomeGraph, BiomeNode
├── species.py          # Species, TraitDistribution, BodyPlan, DietVector
├── culture.py          # CulturalMemory, Meme
├── systems/
│   ├── __init__.py
│   ├── production.py   # Autotroph energy capture
│   ├── consumption.py  # Grazing, predation, scavenging
│   ├── metabolism.py   # Energy costs, starvation
│   ├── reproduction.py # Population dynamics, r/K
│   ├── decomposition.py# Nutrient cycling
│   ├── evolution.py    # Mutation, selection, speciation
│   ├── migration.py    # Inter-biome movement
│   └── spotlight.py    # Individual-level scenes
├── director.py         # NarrativeDirector (quality control)
├── serializer.py       # LayeredSerializer (protocol encoding)
└── constants.py        # All trait/body-plan/threshold definitions
```

Updated entry point:

```
scripts/gen_evo_dataset.py  # CLI wrapper: parses args, calls engine.run()
```

---

## Core Concepts

### 1. Energy as Foundation

Everything derives from energy flow. No abstract "fitness" — species live or die
based on their energy budget.

```
E_income  = energy acquired from feeding (photosynthesis, grazing, hunting, scavenging)
E_cost    = basal_metabolism + movement + trait_maintenance + reproduction
E_balance = E_income - E_cost

E_balance > 0  →  surplus  →  fat storage or population growth
E_balance < 0  →  deficit  →  burn fat stores, then population decline
```

**Trait maintenance cost** is critical: higher-tier traits (intel, planning, language)
are expensive. Intelligence cannot evolve without a reliable, calorie-dense food source.
This naturally gates cognition to omnivores/predators with efficient foraging.

### 2. Epochs (Timescale Solution)

A single simulation passes through up to 4 epochs. Transitions are triggered by
complexity milestones, not elapsed time.

| Epoch            | Tick Meaning                | Trigger                                | Dynamics                                                                            |
| ---------------- | --------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------- |
| **PRIMORDIAL**   | ~1000 microbial generations | Start                                  | Chemical/solar energy. Simple producers + consumers. No predation. Rapid evolution. |
| **CAMBRIAN**     | ~10 generations             | First multicellular (size > threshold) | Body plan diversification. Predation emerges via scavenger gateway. Arms races.     |
| **ECOSYSTEM**    | ~1 generation               | Stable food web (3+ trophic levels)    | Niche specialization. Migration. Speciation. Slower evolution.                      |
| **INTELLIGENCE** | Sub-generational            | Spotlight score > threshold            | Individual-level events. Culture. Heroes. Language.                                 |

Within each epoch the mutation rates, speciation probabilities, and serialization
detail level adjust automatically.

### 3. Body Plans and Trait Constraints

A body plan is a set of morphological constraints that determines which traits a
species CAN evolve. Body plans are not fixed at creation — they shift gradually as
the diet vector and traits change (the "scavenger gateway").

| Body Plan           | Can Evolve                                        | Cannot Evolve                  | Base Metabolism |
| ------------------- | ------------------------------------------------- | ------------------------------ | --------------- |
| `sessile_autotroph` | size, armor, toxin, repro, regen, photosynth      | speed, sense, social, intel    | 0.5x            |
| `mobile_autotroph`  | size, speed (limited), photosynth, sense, camo    | mouth, intel, social           | 0.8x            |
| `filter_feeder`     | size, armor, repro                                | speed (much), aggression       | 0.7x            |
| `grazer`            | speed, sense, armor, social, intel (gated), mouth | photosynth                     | 1.2x            |
| `predator`          | speed, sense, aggression, venom, intel (gated)    | photosynth, armor (limited)    | 1.8x            |
| `scavenger`         | sense, endurance, social, intel (gated), mouth    | photosynth, venom              | 1.0x            |
| `omnivore`          | broad: most T0-T1, intel path open                | photosynth                     | 1.5x            |
| `parasite`          | camo, repro, sense, venom, deception              | size (much), speed, photosynth | 0.6x            |
| `decomposer`        | chem_digest, repro, toxin_resist                  | speed, intel, social           | 0.4x            |

**Body plan transitions**: derived from diet vector shifts. If a grazer's diet
vector drifts to `{plant: 0.5, meat: 0.2, detritus: 0.3}`, it reclassifies as
`omnivore`. Transitions must cross an energy-viability threshold (EROI check).

### 4. Trait Distributions (Not Single Values)

Each trait on a species is stored as `(mean, variance)` rather than a single int.

```python
class TraitDistribution:
    mean: float    # population average (0-10 scale)
    variance: float  # spread within population
```

**Why this matters**:

- Predation success = probability that `sample(predator.speed) > sample(prey.speed)`
  based on distribution overlap, not just `speed_a > speed_b`.
- Selection pressure: if slow individuals get eaten, mean shifts up, variance
  narrows (stabilizing selection).
- Sexual reproduction maintains variance; asexual reproduction collapses it.
- When the player "zooms in", individuals are sampled from these distributions.

### 5. Food Web (Emergent, Not Hardcoded)

Species have a `diet_vector`: a probability distribution over food sources.

```python
# Example: an omnivore species
diet = {
    "solar": 0.0,       # can't photosynthesize
    "S0_algae": 0.4,    # 40% of calories from algae
    "S2_insects": 0.3,  # 30% from insects
    "detritus": 0.2,    # 20% scavenging
    "S5_fish": 0.1,     # 10% opportunistic fishing
}
```

Diet vectors evolve via mutation and selection pressure:

- If primary food source declines → diet shifts toward alternatives
- Discovering a new food source (e.g., scavenging meat) → diet vector gains a
  new entry
- Specialization (narrowing diet) increases efficiency but decreases resilience

**The Scavenger Gateway** (how predation emerges):

1. Pure herbivore: `{plant: 1.0}`
2. Opportunist: develops protein digestion, scavenges: `{plant: 0.8, detritus: 0.2}`
3. Facultative predator: kills weak/young: `{plant: 0.5, detritus: 0.1, meat: 0.4}`
4. Obligate carnivore: specializes: `{meat: 0.9, detritus: 0.1}`

Gated by EROI: a species cannot invest in hunting adaptations (speed, weapons)
unless the caloric return from available prey exceeds the metabolic cost of those
adaptations.

### 6. Spatial Model (Biome Graph)

World geography is a weighted graph of discrete locations.

```python
class BiomeNode:
    id: str
    type: str              # forest, plains, ocean, cave, volcanic_vent, ...
    resources: dict        # {vegetation: float, minerals: float, water: float}
    detritus_pool: float   # dead biomass available for scavenging/decomposition
    conditions: dict       # {temp, humidity, light, toxin, altitude}
    species_present: set   # which species have populations here
```

Edges between nodes have migration costs (energy to traverse). High cost = barrier
= drives allopatric speciation. Costs can change (ice bridge forms, river dries up).

Resources regenerate logistically, deplete under exploitation, and flow along edges
(water, nutrients downstream).

### 7. Reproduction Strategies (r/K Spectrum)

The `repro` trait and `social` trait together determine position on the r/K spectrum.

| Strategy                           | Offspring | Investment | Pop Recovery | Intel Path?             |
| ---------------------------------- | --------- | ---------- | ------------ | ----------------------- |
| **r-max** (repro=9, social=0)      | Many      | None       | Fast         | No                      |
| **r-moderate** (repro=6, social=2) | Several   | Low        | Medium       | Unlikely                |
| **K-moderate** (repro=3, social=5) | Few       | Medium     | Slow         | Possible                |
| **K-max** (repro=1, social=8)      | 1-2       | Very high  | Very slow    | Yes (teaching, culture) |

K-strategy species invest energy per offspring → offspring survive better →
individual offspring matter → this is where Spotlights and Heroes emerge.

Only K-strategy species with sufficient social + intel develop culture.

### 8. Cultural Memory (T2+ Species Only)

When a species crosses the intelligence threshold, it gains a `CulturalMemory`
object that stores learned, non-genetic information.

```python
class CulturalMemory:
    memes: dict       # {concept: Meme(strength, sentiment, origin_gen)}
    oral_history: list  # [(gen, event_type, description)] — bounded buffer
    taboos: dict      # {concept: avoidance_strength}
    traditions: dict   # {activity: reinforcement_strength}
```

**How memes form**:

- Traumatic event (pop loss > 30%): `event → taboo` (e.g., flood → avoid_river)
- Euphoric event (pop boom): `event → sacred` (e.g., fire_discovery → fire_ritual)
- Hero death (influence > 7): `hero → legend` (e.g., Elder_Kah → wisdom_meme)

**Meme decay**: strength decreases by ~5% per generation unless reinforced by a
similar event. Memes at strength < 0.1 are forgotten.

**Meme influence**: memes bias Spotlight decisions. A species with
`taboo:river(0.8)` is unlikely to choose `migrate_to_river` in a Spotlight.

### 9. Spotlights (Individual-Level Scenes)

Activated when a species' spotlight score exceeds threshold:

```
spotlight_score = intel_mean * social_mean * (1 + language_mean * 0.5)
```

A spotlight instantiates 2-5 temporary `IndividualActor` objects sampled from
the species' trait distributions, simulates a structured interaction, and feeds
the outcome back to macro state.

**IndividualActor**:

```python
class IndividualActor:
    id: str                 # H1, H2, ...
    role: str               # Elder, Warrior, Scout, Youth, Mother, ...
    traits: dict            # sampled from species TraitDistribution
    influence: float        # social standing (0-10)
    personal_memory: list   # recent events this individual experienced
    age: int                # generations since "birth"
```

**Persistent heroes**: When a species enters INTELLIGENCE epoch, the top ~5-10
individuals (by influence) become persistent — they appear across multiple
spotlights, age, learn, and eventually die (becoming legends in CulturalMemory).

**Spotlight protocol** (the `@SPOT` block):

```
@SPOT|W<id>G<gen>|L<loc>|S<sid>
  CTX|<species_state>|<cultural_context>
  ACTORS|<actor1_state>|<actor2_state>|...
  INTENT|<actor>-><target>:<action>(<topic>)|reason=<motivation>
  REACT|<target>-><actor>:<reaction>(<topic>)|reason=<motivation>
  RESOLVE|<mechanism>(<details>)|outcome=<result>
  EFFECT|<state_changes>|<cultural_updates>
```

This is synthetic chain-of-thought: the model learns
`context → intent → reasoning → resolution → state change`.

### 10. Nutrient Cycling

Closed-loop system prevents infinite energy:

```
Solar Energy → Producers (photosynthesis) → Herbivores → Predators → ...
                                                    ↓
                                              Death (any cause)
                                                    ↓
                                              Detritus Pool
                                                    ↓
                                              Decomposers
                                                    ↓
                                              Nutrient Pool → Producers
```

Without decomposers, nutrients accumulate as detritus and producers starve.
A world that loses its decomposers will collapse within ~20 generations.

### 11. Interspecies Relationships (Beyond Predation)

Emerge from trait overlap and interaction outcomes:

| Relationship     | Condition                                      | Effect                                          |
| ---------------- | ---------------------------------------------- | ----------------------------------------------- |
| **Predation**    | A feeds on B                                   | A gains energy, B loses population              |
| **Competition**  | A and B share food source                      | Both get reduced intake proportional to overlap |
| **Mutualism**    | A pollinates B (mobile + producer interaction) | Both get fitness bonus                          |
| **Parasitism**   | A has parasite body plan, B is host            | A gains, B loses slowly                         |
| **Commensalism** | A occupies B's territory without competing     | A benefits, B unaffected                        |

These are computed from diet vector overlap, body plan compatibility, and spatial
co-location — not hardcoded relationship types.

---

## Encoding Protocol

### Layer Types

Each generation's output consists of one or more layers. Not all layers appear
every tick — only layers with changes.

| Layer                | Prefix   | When                                      | Content                             |
| -------------------- | -------- | ----------------------------------------- | ----------------------------------- |
| Epoch header         | `=EPOCH` | On epoch transition                       | Timescale context                   |
| Biome state          | `@BIO`   | Keyframes (every N gens) + deltas         | Spatial structure, resources        |
| Species state        | `@SP`    | Every tick (delta when possible)          | Population, energy, traits, diet    |
| Interaction log      | `@INT`   | When interactions occur                   | Predation/competition outcomes      |
| Event                | `@EVT`   | On significant state changes              | Speciation, extinction, trait shift |
| Spotlight            | `@SPOT`  | For T2+ species with high spotlight score | Individual-level interaction        |
| Generation separator | `---`    | Every tick                                | Delimiter                           |

### Keyframe vs Delta

- **Keyframe** (every 10-25 generations): Full state dump for all layers.
- **Delta** (intermediate): Only changed state. Species with no changes omitted.
  Trait deltas encoded as `Δspeed=+0.3` instead of full value.

This keeps data dense — a quiet generation might be 3 lines, a dramatic one 50+.

### Example Output

```
=EPOCH:2|TICK_SCALE:10gen|W42
@BIO|L0:shallows(veg=0.8,mineral=0.5,det=120)|L1:reef(veg=0.3,mineral=0.9)|L0<->L1:cost=1
@SP|S0|L0|plan=sessile_auto|pop=8200±900|diet={solar:1.0}
  T:size=2.1±0.4,photosynth=6.8±0.9,armor=1.2±0.5
  E:in=4200,out=1800,store=6400|repro=r(rate=0.3)
@SP|S1|L0,L1|plan=grazer|pop=600±80|diet={S0:0.8,det:0.2}
  T:speed=4.1±1.2,sense=3.8±0.8,mouth=3.5±0.6,size=3.0±0.7
  E:in=1800,out=1500,store=900|repro=r(rate=0.15)
@SP|S3|L1|plan=predator|pop=40±12|diet={S1:0.7,S4:0.2,det:0.1}
  T:speed=6.2±0.9,sense=5.1±0.7,aggr=4.5±1.1,size=4.8±0.8,intel=2.1±0.6
  E:in=800,out=750,store=200|repro=K(rate=0.04)
@INT|S3 hunt S1|success=0.62|S1:pop-=35|S3:E+=420
@INT|S1 graze S0|eff=0.71|S0:pop-=600|S1:E+=1200
---
@SP|S3|L1|Δpop=+2|Δintel=+0.1±0.0
@EVT|S3|trait_emerge|social=1.5±0.8|prereq:intel>2,sense>4
@INT|S3 hunt S1|success=0.58|S1:pop-=30|S3:E+=380
---
```

Later, when S3 reaches intelligence:

```
=EPOCH:4|TICK_SCALE:sub_gen|W42
@SP|S3|L1|plan=omnivore|pop=180±35|diet={S1:0.4,S7:0.3,det:0.2,S0:0.1}
  T:speed=5.8±0.7,intel=5.5±0.9,social=6.2±1.1,lang=2.8±0.7
  E:in=2200,out=2050,store=1500|repro=K(rate=0.02)
  Cmem:reef_collapse=taboo(0.6)|great_hunt=sacred(0.4)
  Heroes:[H1:Elder(inf=7.2,age=12),H3:Scout(inf=3.1,age=4)]
@SPOT|W42G310|L1|S3
  CTX|S3:intel=5.5,social=6.2,pop=180|Cmem:reef_collapse=taboo(0.6)|res:L1:veg=0.2(low)
  ACTORS|H1:Elder(inf=7.2,int=6.8,soc=7.1,mem=[reef_collapse_g280,famine_g300])|H3:Scout(inf=3.1,int=5.2,soc=4.8,mem=[found_L2_g308])
  INTENT|H3->S3:report(new_territory,L2)|reason=scouted_L2+resource_high
  REACT|H1->H3:endorse(migrate_partial,L1+L2)|reason=Cmem:reef_collapse+risk_spread
  RESOLVE|council(H1.inf+H3.report)|outcome=split_colony(L1:120,L2:60)
  EFFECT|S3:loc+={L2:60}|H3:inf+=0.5|H1:inf+=0.2|Cmem+:expansion(pride,0.3)
---
```

---

## Implementation Phases

### Phase 1: Constants and Species Foundation

**Files**: `mantis/simulation/constants.py`, `mantis/simulation/species.py`

**Deliverables**:

- All trait definitions with tier assignments
- Body plan definitions with trait constraints and base metabolism multipliers
- Body plan transition rules (diet thresholds for reclassification)
- Prerequisite map for higher-tier traits
- Fusion rules
- `TraitDistribution` class (mean + variance, with sample/shift/narrow methods)
- `DietVector` class (normalized distribution over food sources, evolvable)
- `BodyPlan` class (constraint checker, transition logic)
- `Species` class with: traits, diet, body plan, population, energy store,
  reproduction strategy, location set, age, alive flag

**Validation**: Unit test that creates species of each body plan, verifies trait
constraints, and confirms body plan transitions work correctly.

### Phase 2: Environment and Spatial Model

**Files**: `mantis/simulation/environment.py`

**Deliverables**:

- `BiomeNode` class: type, resources (vegetation, minerals, water), detritus pool,
  conditions (temp, humidity, light, toxin), resident species
- `BiomeGraph` class: collection of nodes + weighted edges (migration cost)
- Resource regeneration (logistic growth per tick)
- Resource depletion (based on consumer pressure)
- Environmental drift (gradual condition changes)
- World scenario templates: `temperate_island`, `volcanic_vents`, `ice_world`,
  `ocean_world`, `desert`, `lush_forest`, etc.
- Graph generation: random connected graphs with 3-8 nodes, biome type assignment

**Validation**: Create a world, run 50 ticks of resource regen/depletion without
species, verify resources oscillate within bounds and don't explode or collapse.

### Phase 3: Energy Systems (Production, Consumption, Metabolism, Decomposition)

**Files**: `mantis/simulation/systems/production.py`, `consumption.py`,
`metabolism.py`, `decomposition.py`

**Deliverables**:

**ProductionSystem**:

- Autotrophs capture solar/chemical energy proportional to:
  `photosynth_trait * light_condition * nutrient_availability`
- Energy capped by biome carrying capacity
- Competition among co-located producers for light/nutrients

**ConsumptionSystem**:

- For each consumer, iterate potential food sources (from diet vector)
- Grazing: success based on `mouth * sense` vs producer `armor + toxin`
- Predation: success probability from speed/sense distribution overlap
- Scavenging: proportional to detritus pool and `sense` trait
- Energy gained = biomass_consumed \* trophic_efficiency (0.10-0.15)
- Competition: species sharing food sources at same location split intake
  proportional to relative efficiency

**MetabolismSystem**:

- Per-species cost = `population * mass * metabolic_rate`
- `metabolic_rate` = body_plan_base \* (1 + kleiber(size) + brain_tax(intel, planning, language))`
- Kleiber scaling: mass^0.75 — bigger animals more efficient per unit but need more total
- Brain tax: sum of T2+ trait means \* cost_per_tier
- Energy deficit → burn fat (energy_store), then starve (pop loss)

**DecompositionSystem**:

- Dead biomass (from starvation, predation, old age) enters biome's detritus pool
- Decomposer species convert detritus → nutrients at rate based on their traits
- Background decay: detritus slowly converts to nutrients even without decomposers
  (but much slower)
- Nutrients feed back to producers

**Validation**: Seed a world with 1 producer + 1 herbivore + 1 decomposer.
Run 200 ticks. Verify:

- Producer population oscillates (grows when ungrazed, shrinks when grazed)
- Herbivore population tracks producer with ~lag
- Nutrients cycle (don't accumulate infinitely or deplete to zero)
- Removing decomposer causes nutrient crash within ~30 ticks

### Phase 4: Evolution and Speciation

**Files**: `mantis/simulation/systems/evolution.py`

**Deliverables**:

**Mutation**:

- Point mutation: shift trait mean by small delta, rate scales with epoch
- Drift: random walk on variance
- Leap mutation: rare large shift (useful in PRIMORDIAL/CAMBRIAN epochs)
- Diet mutation: small shift in diet vector (enables food web evolution)
- New trait acquisition: when prerequisites met, trait appears at low mean

**Selection**:

- After consumption/metabolism: compute which trait ranges survived
- Narrow variance for traits under stabilizing selection
- Widen variance for traits under disruptive selection
- Mean shifts toward values that survived better

**Speciation**:

- **Allopatric**: population split across biomes with high migration cost.
  After N generations of isolation, divergence exceeds threshold → new Species
- **Sympatric**: trait variance exceeds threshold (bimodal distribution) →
  split into two species with different means
- **Population threshold**: only species with pop > minimum can speciate

**Body Plan Transition**:

- Check diet vector each tick
- If diet composition crosses threshold (e.g., meat > 0.3 for a grazer),
  reclassify body plan
- EROI gate: transition only if new body plan is energy-viable given current
  trait levels and available food sources

**Trait Pruning**:

- Under energy pressure, expensive traits with low utility get pruned
- Prevents runaway trait accumulation

**Validation**: Run 500 ticks. Verify speciation events occur. Verify a grazer
can transition to omnivore given sufficient scavenging opportunity. Verify
T2+ traits only emerge when prerequisites are met.

### Phase 5: Migration

**Files**: `mantis/simulation/systems/migration.py`

**Deliverables**:

- Migration pressure: computed from resource scarcity at current biome vs
  adjacent biomes (pull factor) and population density (push factor)
- Migration cost: energy expenditure to traverse edge, scales with body size
- Population split: portion of population moves, rest stays
- Colonization: if species reaches empty biome, establishes new population
- Barrier formation: edges can become impassable (ice, flood), triggering
  allopatric speciation setup

**Validation**: Create 3-node world. Deplete resources at node A. Verify species
migrates to node B. Verify population splits across nodes.

### Phase 6: Reproduction System

**Files**: `mantis/simulation/systems/reproduction.py`

**Deliverables**:

**r/K Spectrum**:

- `repro` trait determines base offspring count
- `social` trait modifies investment per offspring
- r-strategy (high repro, low social): many offspring, low survival modifier,
  fast population recovery
- K-strategy (low repro, high social): few offspring, high survival modifier,
  each offspring "inherits" more of parent traits (lower variance in offspring)

**Sexual vs Asexual**:

- Asexual: offspring = parent mean + mutation. Variance stays low.
- Sexual (emerges when `repro_mode` trait crosses threshold): offspring traits
  sampled from parent distribution. Maintains variance. More adaptive but
  higher cost (needs mate-finding).

**Reproduction Cost**:

- Energy per offspring = `offspring_mass * (1 + investment_factor)`
- K-strategy species pay more per offspring but offspring survive better
- Species can only reproduce when `energy_store > reproduction_threshold`

**Validation**: Compare two populations — one r-strategy, one K-strategy — after
environmental shock. r-recovers faster but K-is more stable long-term.

### Phase 7: Cultural Memory and Intelligence

**Files**: `mantis/simulation/culture.py`

**Deliverables**:

**CulturalMemory**:

- Attached to species when `spotlight_score > threshold`
- Stores memes: `{concept: Meme(strength, sentiment, origin_gen)}`
- Meme formation rules:
  - Population loss > 30% in one tick → trauma meme (taboo)
  - Population gain > 20% in one tick → euphoric meme (sacred)
  - Hero death with influence > 7 → legend meme
  - Successful innovation → tradition meme
- Meme decay: strength \* 0.95 per generation (unless reinforced)
- Meme influence: biases decisions in spotlight scenes

**Intelligence Threshold**:

- `spotlight_score = intel_mean * social_mean * (1 + language_mean * 0.5)`
- Score > 15: basic spotlights (simple interactions)
- Score > 30: persistent heroes, cultural memory
- Score > 60: complex scenes (multi-actor, nested intent)

**Validation**: Create a species at T2+ threshold. Trigger a disaster. Verify
meme forms. Advance 20 generations. Verify meme decays. Trigger similar event.
Verify meme reinforces.

### Phase 8: Spotlight System

**Files**: `mantis/simulation/systems/spotlight.py`

**Deliverables**:

**IndividualActor**:

- Sampled from species trait distributions
- Role assignment based on trait profile (highest intel → Elder, highest aggr →
  Warrior, highest sense → Scout, etc.)
- Personal memory buffer (5 most recent events)
- Influence score (determines social weight in resolutions)
- Age tracking, trait evolution from experience

**Persistent Heroes**:

- When species enters INTELLIGENCE epoch: top 5-10 by influence become persistent
- They appear across multiple spotlights
- Traits shift based on experience vectors:
  - Win conflict → aggr+, inf+
  - Lose conflict → aggr-, intel+ (learns caution)
  - Teach successfully → teaching+, inf+
  - Survive disaster → regen+, memory reinforced
- Death → become legend in CulturalMemory if influence > 7
- Replaced by next highest-influence individual

**SceneGenerator**:

- Trigger conditions: resource crisis, territorial conflict, discovery,
  population milestone, internal schism, external threat
- Cast selection: 2-5 actors relevant to the trigger
- Intent generation: based on actor traits, memories, and cultural context
- Reaction generation: based on target traits and relationship to initiator
- Resolution: mechanism depends on social structure
  (hierarchy → influence wins, egalitarian → vote, anarchic → strength wins)
- Effect application: update species state, cultural memory, hero stats

**Protocol Output**: The `@SPOT` block as specified in the encoding protocol
section above.

**Validation**: Create a T2+ species with 5 heroes. Run 10 spotlights. Verify
heroes age and evolve. Kill a high-influence hero. Verify legend meme forms.

### Phase 9: Serializer and Narrative Director

**Files**: `mantis/simulation/serializer.py`, `mantis/simulation/director.py`

**LayeredSerializer**:

- Keyframe serialization: full state dump in protocol format
- Delta serialization: diff against previous keyframe
- Layer-specific formatters: `@BIO`, `@SP`, `@INT`, `@EVT`, `@SPOT`
- Configurable keyframe interval (default: every 20 generations)
- Compression: omit unchanged species, use delta notation for small changes

**NarrativeDirector**:

- Monitors simulation quality per-world
- Kill rules:
  - All species extinct → stop
  - Stuck at epoch 1 after 50 ticks → stop (boring)
  - Only producers alive after 100 ticks → stop (no food web)
- Fork rules:
  - Species reaches T2 → mark world as "interesting", extend generation count
  - Species reaches T3 → fork 2-3 variations (Monte Carlo) to maximize
    chance of reaching T4
- Save policy:
  - Always save worlds that reach epoch 3+
  - Save 10% of worlds that stall at epoch 1-2 (biological foundation data)
  - Discard the rest
- Stats tracking: total worlds generated, saved, per-epoch distribution,
  max tier reached, speciation/extinction counts

### Phase 10: Engine and CLI Integration

**Files**: `mantis/simulation/engine.py`, `scripts/gen_evo_dataset.py`

**SimulationEngine**:

- Owns World, TimeManager, all Systems, Director, Serializer
- Main loop: `for tick in range(max_ticks): run_all_systems(); serialize(); check_director()`
- System execution order per tick:
  1. Environment drift
  2. Production (autotrophs capture energy)
  3. Consumption (grazing, predation, scavenging)
  4. Metabolism (energy costs applied)
  5. Decomposition (dead biomass → nutrients)
  6. Reproduction (population growth if energy surplus)
  7. Evolution (mutation, selection)
  8. Migration (population movement)
  9. Spotlight (if any T2+ species qualify)
  10. Serialize current state

**TimeManager**:

- Tracks current epoch
- Adjusts tick scale and system parameters per epoch
- Detects epoch transitions from world state

**CLI** (`scripts/gen_evo_dataset.py`):

- Thin wrapper: parse args, instantiate engine, call `engine.run()`
- Arguments: `--worlds`, `--max-generations`, `--output`, `--seed`, `--verbose`,
  `--workers` (multiprocessing), `--min-epoch` (minimum epoch to save)

**Multiprocessing**:

- Each world is independent → embarrassingly parallel
- Use `multiprocessing.Pool` with `--workers` flag
- Each worker runs one world, returns serialized text
- Main process concatenates outputs with EOS boundaries

**Output format**: Plain `.txt` compatible with `TextDataset` in `train.py`.
Worlds separated by blank lines (EOS boundary, matching existing convention).

---

## Performance Targets

- 10,000 worlds x 200 generations: < 2 hours on 8-core CPU
- Output size: ~500MB - 2GB depending on intelligence frequency
- Memory: < 4GB RAM per worker process

**Optimization strategies**:

- NumPy vectorization for energy/population math within a single world
- Avoid Python loops over individuals (distribution math instead)
- Lazy spotlight instantiation (only when threshold crossed)
- String builder for serialization (avoid repeated concatenation)
- Multiprocessing across worlds

---

## Dependencies

No new dependencies beyond what's already in `requirements.txt`:

- `numpy` (already listed) — vectorized math
- `random` / `math` (stdlib) — stochastic processes

Optional for testing:

- `matplotlib` (already listed) — visualize population dynamics during dev

---

## Integration with Training Pipeline

The output `.txt` file slots directly into the existing training flow:

```bash
# Generate dataset
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 --output data/evo_train.txt --workers 8 --verbose

# Train (existing pipeline, no changes needed)
python train.py --stage 1 data/evo_train.txt --val-split 0.1 --model-size tiny

# Or pre-tokenize first for speed
python scripts/preprocess_data.py --input data/evo_train.txt --output data/tokenized/evo
python train.py --stage 1 data/tokenized/evo --pretokenized --val-split 0.1
```

The `TextDataset` class already handles:

- Line-based tokenization
- EOS token insertion at blank lines (world boundaries)
- Chunking into fixed-length sequences

However, to fully exploit the protocol format, several training-side changes are
needed (detailed below).

---

## Training Configuration

Training on this dataset differs fundamentally from natural language pre-training.
The data is a dense structured protocol with hierarchical nesting, numeric values,
causal chains, and long-range dependencies. The following parameters and
modifications are required.

### Sequence Length: 4096 Tokens

The default `--seq-len 512` is completely insufficient. Cultural memories reference
events 50-150 generations ago. At 4096 tokens:

- ~7 spotlight generations or ~80 biological generations fit per window
- Combined with self-contained framing (below), 4096 is sufficient for the model
  to see cause and effect within the same sequence
- 8192 is ideal but quadratic attention cost makes it impractical for initial runs

### Self-Contained Framing

Rather than relying on ultra-long sequences to capture causal chains, each
serialized frame must inline enough context to be a complete logical unit.

When a `Cmem` is referenced, the serializer includes its origin:

```
Cmem:flood(origin=G50,pop_loss=40%,strength=0.7)
```

When a hero references a personal memory, the memory content is inlined:

```
H1:Elder(inf=7.2,mem=[flood_g50:pop_loss_40%,drought_g80:migration_L1_L2])
```

This makes each frame semantically self-contained. The model learns "this memory
caused this decision" from a single frame rather than pattern-matching across
distant frames. The tradeoff is fewer generations per window, but each generation
carries full causal context.

### Tokenizer Extension: ~65 Protocol Tokens

GPT-2 BPE fractures protocol syntax (`@SPOT` → 3+ tokens, trait names split
unpredictably). We extend the vocabulary with atomic protocol tokens:

| Category | Count | Examples |
|----------|-------|---------|
| Layer markers | 7 | `=EPOCH`, `@BIO`, `@SP`, `@INT`, `@EVT`, `@SPOT`, `---` |
| Spotlight logic | 6 | `CTX`, `ACTORS`, `INTENT`, `REACT`, `RESOLVE`, `EFFECT` |
| Mutation types | 6 | `M+`, `M-`, `Mpoint`, `Mdrift`, `Mleap`, `Mfuse` |
| Body plans | 9 | `sessile_autotroph`, `mobile_autotroph`, `filter_feeder`, `grazer`, `predator`, `scavenger`, `omnivore`, `parasite`, `decomposer` |
| Trait names | ~29 | `speed`, `intel`, `social`, `theory_of_mind`, ... |
| Other | ~8 | `±`, `Δ`, `solar`, `plant`, `meat`, `detritus`, `r(`, `K(` |

Implementation: use `tokenizer.add_special_tokens({'additional_special_tokens': [...]})`.
These tokens are never split by BPE. Vocab goes from 50304 → ~50432 (negligible
model size impact, <0.1%). The embedding layer must be resized after adding tokens.

This reduces token count per frame by ~30-40%, effectively expanding the usable
context window.

### Per-Line Loss Weighting

Not all tokens carry equal simulation value. Protocol delimiters are trivial;
spotlight reasoning chains are the core intelligence. Per-token weighting is
impractical (structural and numeric content interleave within lines), so we weight
per line based on layer prefix:

| Line Prefix | Weight | Rationale |
|------------|--------|-----------|
| `---` separators | 0.1 | Trivial to predict |
| `@BIO` lines | 0.5 | Slow-changing, low information density |
| `@SP` lines | 1.0 | Core simulation state |
| `@INT` lines | 1.5 | Interaction dynamics (the "physics") |
| `@EVT` lines | 1.5 | Rare, high-importance events |
| `@SPOT` blocks | 2.0 | Intelligence reasoning chains |

Implementation: during tokenization, compute a `loss_weight` mask parallel to
`input_ids` and `labels`. Store alongside them in the dataset. The training loop
uses `reduction='none'` on the cross-entropy loss and multiplies by the weight
mask before averaging:

```python
raw_loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction='none')
weighted_loss = (raw_loss * loss_weights.view(-1)).sum() / loss_weights.sum()
```

This requires a modified `WeightedTextDataset` class that tokenizes line-by-line
and builds the weight mask during data loading.

### Training Curriculum: Gradual Mixing

Sequential phase training (biology only → add spotlights) causes catastrophic
forgetting of earlier dynamics. Instead, use a multi-source sampler that shifts
mixing ratios based on training progress:

| Training Progress | Bio Traces | Ecosystem | Intelligence/Spotlights |
|-------------------|-----------|-----------|------------------------|
| 0-20% of steps | 100% | 0% | 0% |
| 20-40% | 50% | 50% | 0% |
| 40-60% | 25% | 35% | 40% |
| 60-100% | 20% | 30% | 50% |

Implementation: the dataset generator outputs three separate files
(`evo_bio.txt`, `evo_eco.txt`, `evo_full.txt`) partitioned by maximum epoch
reached. A custom `CurriculumSampler` draws from these pools according to the
schedule above, controlled by `global_step / total_steps`.

The biological data is always present throughout training, reinforcing the
physical foundations even as the model learns higher-order dynamics.

### World Boundary Handling: Pad, Never Pack Cross-World

Worlds are independent simulations with unrelated species, environments, and
histories. Attending across a world boundary teaches nothing useful and risks
spurious correlations.

**Rule**: At each world boundary (blank line / EOS), pad the current sequence to
`seq_len`. Never pack tokens from different worlds into the same sequence.

Waste is <10% for worlds with 10K+ tokens (typical). The last sequence of each
world wastes at most `seq_len - 1` tokens.

### Stride: 2048 (50% Overlap)

With `--stride 2048` and `--seq-len 4096`, every event appears in at least two
sequence contexts. This ensures no causal chain is missed due to sequence boundary
alignment. Combined with world-boundary padding, no sequence ever spans two
different worlds.

### Validation Split Strategy

The Narrative Director biases the training set toward interesting worlds. The
validation set must match this distribution:

- **Train**: Director-guided worlds, seeds 0-9000
- **Val**: Director-guided worlds, seeds 9001-9500
- **Test**: Director-guided worlds, seeds 9501-10000
- **OOD set** (optional): unfiltered random worlds for robustness sanity check

Use the same Director parameters for val/test but different random seeds, ensuring
the distribution matches but the specific worlds are unseen.

### Optimizer and Schedule

- **Optimizer**: AdamW, betas (0.9, 0.95), weight decay 0.01
- **Schedule**: Cosine decay with warmup
- **Peak LR**: 5e-4 for tiny/small, 1e-4 for base
- **Min LR**: 1e-5 (never decay to zero; late-stage data is the most complex)
- **Warmup**: 2000 steps (essential for MoE router stabilization)
- **Gradient clipping**: 1.0

### Evaluation Metrics

Standard perplexity is necessary but insufficient. Domain-specific metrics
must be tracked during validation:

| Metric | Target | What It Measures |
|--------|--------|-----------------|
| **Perplexity** | Decreasing | Overall token prediction quality |
| **Parse Rate** | >99.9% | Can the model output valid protocol syntax? |
| **Energy Conservation** | <5% deviation | Does E_in - E_out ≈ ΔE_store + Δpop * mass? |
| **Population Dynamics** | Correlation >0.8 | Realistic predator-prey oscillation patterns? |
| **Causal Consistency** | >95% | Extinct species stay extinct? Taboos referenced correctly? |
| **Trait Validity** | >99% | Body plan constraints respected? No speed on sessile autotrophs? |
| **Spotlight Coherence** | >90% | Does EFFECT logically follow from INTENT + RESOLVE? |

Implementation: a validation script (`scripts/eval_sim_quality.py`) that:
1. Generates N simulation traces from the model (temperature=0.7)
2. Attempts to parse each trace back into simulation objects (parse rate)
3. Validates conservation laws, causal consistency, and trait constraints
4. Reports all metrics

This should run every `--eval-every` steps during training.

### Convergence Signals

Training should stop when:

1. Validation loss plateaus for 3 consecutive evaluation intervals, AND
2. Parse Rate has saturated at >99.9%, AND
3. Causal Consistency has stopped improving

Stop on whichever condition is met **last**. Syntax is typically learned first,
then semantics, then subtle causality. The model may plateau on loss while still
improving on domain-specific metrics.

### Concrete Training Recipes

**Sanity check** (verify the pipeline works, overfit on small data):

```bash
python scripts/gen_evo_dataset.py --worlds 50 --max-generations 50 \
    --output data/evo_sanity.txt --seed 42

python train.py --stage 1 data/evo_sanity.txt \
    --model-size micro --seq-len 512 --batch-size 4 \
    --epochs 1 --steps-per-epoch 50 --val-split 0.1
```

**Development run** (tiny model, full dataset, real training):

```bash
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 \
    --output data/evo_train.txt --workers 8 --seed 42 --verbose

python train.py --stage 1 data/evo_train.txt \
    --model-size tiny --seq-len 4096 --stride 2048 \
    --batch-size 8 --gradient-accumulation-steps 4 \
    --learning-rate 5e-4 --warmup-steps 2000 --epochs 10 \
    --mixed-precision --val-split 0.1 --eval-every 500
```

**Production run** (small model, multi-GPU):

```bash
python train.py --stage 1 data/evo_train.txt \
    --model-size small --seq-len 4096 --stride 2048 \
    --batch-size 8 --gradient-accumulation-steps 8 \
    --learning-rate 3e-4 --warmup-steps 2000 --epochs 3 \
    --mixed-precision --gradient-checkpointing \
    --val-split 0.1 --eval-every 1000
```

**Full-scale run** (base model, multi-GPU cluster):

```bash
python train.py --stage 1 data/evo_train.txt \
    --model-size base --seq-len 4096 --stride 2048 \
    --batch-size 8 --gradient-accumulation-steps 8 \
    --learning-rate 1e-4 --warmup-steps 2000 --epochs 1 \
    --mixed-precision --gradient-checkpointing \
    --deepspeed --val-split 0.1 --eval-every 2000
```

### Multi-GPU Considerations

| Model Size | Strategy | Notes |
|-----------|----------|-------|
| Tiny (100M) | Single GPU | Fits easily on any modern GPU |
| Small (1B) | DDP or ZeRO-1 | Gradient checkpointing recommended |
| Base (12B) | ZeRO-2 + gradient checkpointing | ZeRO-3 if VRAM < 80GB per card |

The per-line loss weighting works with gradient accumulation and DDP: the weighted
loss is a standard scalar, gradients accumulate and average normally. Ensure the
loss function handles empty-target sequences (all padding) by returning zero loss
and zero weight to avoid NaN gradients.

### Required Code Changes for Training

These training features require modifications beyond the dataset generator:

1. **`mantis/tokenizer.py`**: Add `PROTOCOL_TOKENS` list, extend vocabulary
   via `add_special_tokens`, resize embeddings after loading model
2. **`mantis/training/weighted_dataset.py`** (new): `WeightedTextDataset` that
   tokenizes line-by-line and builds per-token loss weight masks
3. **`mantis/training/curriculum_sampler.py`** (new): `CurriculumSampler` that
   mixes data from multiple source files based on training progress
4. **`train.py`**: Support `--stride`, weighted loss, curriculum mixing flags,
   tokenizer extension, embedding resize on model init
5. **`scripts/eval_sim_quality.py`** (new): Simulation-specific validation
   metrics (parse rate, conservation, causal consistency)

---

## Open Questions (To Resolve During Implementation)

1. **Keyframe interval tuning**: 20 generations between keyframes is a guess.
   Too frequent = redundant data. Too sparse = model can't reconstruct state
   from deltas alone. Needs empirical testing with actual token counts.

2. **Spotlight frequency cap**: If multiple T2+ species exist simultaneously,
   do we generate spotlights for all of them each tick? Probably need a budget
   (max 2-3 spotlights per tick) to avoid data imbalance toward intelligence
   at the expense of ecological dynamics.

3. **Epoch transition smoothness**: Hard transitions between epochs could create
   jarring data discontinuities. May need a "transition zone" where both
   timescales coexist briefly.

4. **Diet vector dimensionality**: As species count grows (speciation), diet
   vectors grow too. Need a pruning mechanism for extinct food sources to keep
   serialization compact.

5. **Loss weight tuning**: The per-line weights (0.1 for separators, 2.0 for
   spotlights) are initial guesses. May need tuning based on gradient magnitude
   analysis during early training runs.

6. **Curriculum transition sharpness**: The mixing ratio schedule (100% bio →
   gradual introduction of spotlights) may need smoother transitions to avoid
   loss spikes when new data types are introduced.
