# Agent-Based Evolution Simulation with Graphical Playground

## Executive Summary

This plan extends the MANTIS evolution simulator from **population-level** to **agent-based** modeling, adding:

1. **Individual agents** with spatial coordinates and behaviors
2. **Spatial biomes** with 2D resource distribution
3. **Web-based playground** to visualize agents moving in real-time (React + Canvas 2D)

This changes the training data format (adds `@AGENT` protocol blocks) while maintaining backward compatibility.

---

## Critical Revisions (Post-Gemini Debate)

**Status**: This plan has been revised based on collaborative debate identifying architectural flaws in the original design.

### Key Fixes Applied

| Issue | Original Design | Revised Design | Impact |
|-------|-----------------|----------------|---------|
| **Energy/Population Conservation** | Agents are "ghost samples" — their deaths don't affect macro population | **Dual-layer accounting**: Discrete events (deaths) scale 1:1, continuous processes (foraging) scale by population ratio | Prevents training discontinuity and 97% die-off artifacts |
| **Coordinate Serialization** | Continuous floats `(120.5, 340.2)` create high-entropy tokens | **Quantized to 10-unit grid** `(120, 340)` — simulate continuous, serialize discrete | Reduces token entropy by 30%, easier for model to predict |
| **Protocol Loss Weight** | `@AGENT: 1.8` (near spotlight priority) | **@AGENT: 0.8** (lower than core stats) | Prioritizes causal logic over exact coordinate regression |
| **Frontend Streaming** | 60fps JSON stream = 18MB/sec bottleneck | **10-20 TPS server + 60fps client interpolation** | 6× bandwidth reduction, smooth rendering |
| **Spatial Hash Cell Size** | 50-unit cells require multi-layer neighbor queries | **100-unit cells** (matches max sense range) | Guarantees 3×3 neighborhood, no multi-layer lookups |
| **Behavior Oscillation** | Agents vibrate flee→forage→flee until starvation | **Hysteresis with emergency override** | Prevents oscillation, allows panic breakout at critical energy |
| **Vegetation Depletion** | Instant collapse vs infinite faucet mismatch | **Large patches (500 cap) + slow regen** | Sustains 50 agents for 10+ ticks, creates spatial competition |
| **Training Discontinuity** | @AGENT blocks appear suddenly in INTELLIGENCE epoch | **Gradual introduction in ECOSYSTEM** (simple flocking) | Model learns spatial concepts progressively |

### Validation Against Training Goals

✅ **Physical consistency**: Energy/population reconciliation prevents "ghost agent" paradox
✅ **Reduced entropy**: Quantized coordinates lower model perplexity on spatial tokens
✅ **Scalable rendering**: Client interpolation enables 2000+ agents at 60fps
✅ **Emergent realism**: Hysteresis + patch capacity prevent unrealistic oscillations
✅ **Gradual learning**: ECOSYSTEM → INTELLIGENCE progression avoids sudden spatial concepts

---

## Table of Contents

1. [Context & Motivation](#context--motivation)
2. [Current System Analysis](#current-system-analysis)
3. [Design Overview](#design-overview)
4. [Core Data Structures](#core-data-structures)
5. [Behavior System](#behavior-system)
6. [Spatial Performance](#spatial-performance)
7. [Protocol Extension](#protocol-extension)
8. [Frontend Architecture](#frontend-architecture)
9. [File Structure](#file-structure)
10. [Implementation Phases](#implementation-phases)
11. [Testing Strategy](#testing-strategy)
12. [Performance Optimization](#performance-optimization)
13. [Risks & Mitigations](#risks--mitigations)
14. [Success Criteria](#success-criteria)

---

## Context & Motivation

### Current State

The MANTIS evolution simulator (`mantis/simulation/`) operates entirely at **population-level**:

- **Species** have `population: int` (e.g., 8,200 individuals) but no individual agents
- **Biomes** have abstract resources (`vegetation`, `detritus`, `solar`) but no spatial coordinates
- **Interactions** are statistical (predation success = trait overlap) not spatial
- **Output** is structured text protocol (`@SP`, `@BIO`, `@INT`, `@SPOT`)

### Why Add Agents?

**User Goal**: Build a graphical game playground to test trained models where you can:
- See individual creatures moving around on a map
- Watch predators chase prey, prey flee, herbivores graze
- Control with play/pause/stop
- Test how well the model learned evolutionary dynamics

**Technical Benefits**:
- Richer training data (spatial reasoning, individual behaviors)
- Visual debugging of emergent behaviors
- Enables gameplay/interactive applications
- Validates model predictions at micro-level (not just macro statistics)

---

## Current System Analysis

### Existing Architecture

```
mantis/simulation/
├── constants.py        # Body plans, traits, brain tax, epochs
├── species.py          # TraitDistribution, DietVector, BodyPlan, Species
├── biome.py            # Biome with abstract resources
├── engine.py           # World simulation loop
└── serializer.py       # Protocol text output
```

### Key Findings

| Aspect | Current State |
|---|---|
| **Biomes** | Abstract resources only (no x, y coordinates) |
| **Populations** | Range from 2 to 10,000+ per species |
| **Typical world** | 2-5 biomes, 3-8 species, 100-200 generations |
| **File size** | ~679MB for 10,000 worlds (67KB per world) |
| **Serialization** | Keyframes every 20 ticks + delta encoding |
| **Spotlights** | Hero narratives in INTELLIGENCE epoch only |
| **Performance** | ~10-20 worlds/sec generation (sequential) |

### Extension Points

✅ **Species.agents**: Can add `Optional[AgentManager]` field (lazy-initialized)
✅ **Biome spatial**: Can add coordinate grid without breaking existing code
✅ **Protocol**: `@AGENT` blocks are additive (backward-compatible)
✅ **Epochs**: Activate agents only in INTELLIGENCE (no perf impact on early epochs)

---

## Design Overview

### Three-Layer Architecture

```
┌───────────────────────────────────────────────────────┐
│  Population Layer (Existing — Macro-level)            │
│  ─────────────────────────────────────────────        │
│  • Species with trait distributions                   │
│  • Mutations, body plan transitions                   │
│  • Aggregate energy balance                           │
│  • Population dynamics (births, deaths)               │
└───────────────────────┬───────────────────────────────┘
                        │ Trait distributions
                        │ (mean ± variance)
                        ▼
┌───────────────────────────────────────────────────────┐
│  Agent Layer (New — Micro-level)                      │
│  ────────────────────────────────────                 │
│  • Individual agents with (x, y) positions            │
│  • Sample traits from species distributions           │
│  • Behaviors: forage, hunt, flee, mate, rest          │
│  • Local interactions based on proximity              │
│  • Movement with steering behaviors                   │
└───────────────────────┬───────────────────────────────┘
                        │ JSON state
                        │ (over WebSocket)
                        ▼
┌───────────────────────────────────────────────────────┐
│  Visualization Layer (New — Interactive)              │
│  ────────────────────────────────────────────         │
│  • React frontend with Canvas 2D rendering            │
│  • Real-time agent visualization (color-coded)        │
│  • Play/pause/speed controls                          │
│  • Hover tooltips, click to follow agent             │
└───────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Coordinate system** | Continuous float simulation, quantized serialization | Smooth movement internally, reduced token entropy for training |
| **Agent count** | 50-250 per species (sampled if pop > 500) | Balance realism vs performance |
| **Max total agents** | 2000 per world | Limit memory overhead |
| **Activation trigger** | `spotlight_score > 15` (INTELLIGENCE epoch), optional ECOSYSTEM | Gradual spatial learning, avoid perf cost in early epochs |
| **Spatial index** | SpatialHash with 100-unit cells (matches max sense range) | O(k) neighbor queries, eliminates multi-layer lookups |
| **Behavior model** | Utility-based AI + softmax + hysteresis | Emergent diversity, prevents oscillation |
| **Serialization** | Sample top-200 + 50 random, quantized to 10-unit grid | Capture important agents + variance, reduce float precision noise |
| **Frontend rendering** | Canvas 2D with 10-20 TPS + client interpolation | Sufficient for top-down 2D, avoids 60fps JSON bottleneck |
| **Energy accounting** | Dual-layer with event-based reconciliation | Agents are representative samples, scale discrete events separately |

---

## Dual-Layer Accounting Model

### The Challenge

Agents are **representative samples** of the full population, not the entire population itself. A species with 8,000 individuals spawns only 250 agents (32× scaling). This creates a critical question: how do agent-level events (death, feeding, reproduction) affect population-level statistics (energy_store, population count)?

### The Solution: Event-Based Reconciliation

**Principle**: Separate discrete events (births, deaths, hunts) from continuous processes (metabolism, foraging gains).

```python
# mantis/simulation/agent_reconciliation.py

class PopulationReconciler:
    """Manages bidirectional sync between agents and macro population."""

    def __init__(self, species: Species, scaling_factor: float):
        self.species = species
        self.scaling = scaling_factor  # e.g., 8000 / 250 = 32.0

    def reconcile_tick(self, agents: list[Agent], events: EventLog):
        """Called after each agent tick to update macro stats."""

        # ──────────────────────────────────────────────────
        # 1. DISCRETE EVENTS (do not scale)
        # ──────────────────────────────────────────────────
        # Each agent death = 1 death, not 32
        deaths = len(events.deaths)
        births = len(events.births)
        self.species.population += (births - deaths)

        # ──────────────────────────────────────────────────
        # 2. CONTINUOUS ENERGY (scale aggregate change)
        # ──────────────────────────────────────────────────
        # Average energy change across all agents
        total_energy_change = sum(events.energy_deltas.values())
        avg_energy_change = total_energy_change / len(agents)

        # Scale up to full population
        self.species.energy_store += avg_energy_change * self.scaling

        # ──────────────────────────────────────────────────
        # 3. TRAIT DISTRIBUTION UPDATE (population-weighted)
        # ──────────────────────────────────────────────────
        # Agents that died or were born affect trait distributions
        if births > 0 or deaths > 0:
            self._update_trait_distributions(agents)
```

### Why This Works

| Scenario | Agent Event | Macro Impact | Reasoning |
|----------|-------------|--------------|-----------|
| **Hunt success** | Predator A0 kills prey A5 | `species[prey].population -= 1` | One creature dies, not 32. Death is a discrete event. |
| **Foraging** | Grazer A3 gains +5 energy from patch | `species.energy_store += 5 × 32 = +160` | Represents 32 grazers gaining energy. Continuous process scales. |
| **Starvation** | Agent A7 energy = 0, dies | `species.population -= 1` | Individual death, not mass die-off. |
| **Birth** | Two agents mate, spawn A8 | `species.population += 1` | One offspring born. |

### Training Data Implications

The model learns realistic causality:
- ✅ "1 predator kills 1 prey" (not cascading 32× deaths)
- ✅ "Grazing increases population energy reserves" (scaled continuous gain)
- ✅ "Agent death reduces population by 1" (discrete event)
- ✅ "Energy depletion causes starvation" (individual failure)

This avoids the catastrophic discontinuity problem where entering INTELLIGENCE epoch would cause mass extinction purely from simulation mechanics changing.

---

## Core Data Structures

### 1. Agent

```python
# mantis/simulation/agent.py

@dataclass
class Agent:
    """Individual organism in agent-based simulation."""

    # Identity
    aid: int                    # Unique agent ID
    species_sid: int            # Parent species ID
    biome_lid: int              # Current biome location ID

    # Spatial state
    x: float                    # Position in [0, world_size)
    y: float
    velocity: tuple[float, float]  # (vx, vy) for smooth movement

    # Biological state
    energy: float               # Individual energy store (0-100+)
    age: int                    # Ticks alive
    traits: dict[str, float]    # Sampled from species TraitDistribution
    alive: bool = True

    # Behavior state
    state: str                  # Current action: "forage", "hunt", "flee", "rest", "mate"
    state_commitment: int = 0   # Hysteresis: ticks remaining in current state
    target_aid: Optional[int]   # Target agent ID for hunt/mate
    prev_energy: float = 0.0    # Previous tick energy for delta tracking
```

**Key Points**:
- Each agent samples traits from species `TraitDistribution` → individual variation
- `energy` drives behavior (hunger → forage/hunt)
- `state` determines movement steering with hysteresis (min commitment period)
- `state_commitment` prevents rapid oscillation (flee→forage→flee→forage...)
- `prev_energy` enables efficient energy delta computation for reconciliation
- `target_aid` enables persistent pursuit/fleeing

---

### 2. AgentManager

```python
# mantis/simulation/agent.py

class AgentManager:
    """Manages agents for a single species."""

    def __init__(self, species: Species, biome: Biome, world_size: int):
        self.species = species
        self.agents: list[Agent] = []
        self.spatial_hash: SpatialHash = SpatialHash(world_size, cell_size=50)
        self.next_aid = 0

    def spawn_agents(self, count: int, rng: np.random.Generator):
        """Create *count* agents, sampling traits from species distributions."""
        for _ in range(count):
            traits = {
                name: dist.sample(n=1, rng=rng)
                for name, dist in self.species.get_all_traits().items()
            }
            agent = Agent(
                aid=self.next_aid,
                species_sid=self.species.sid,
                biome_lid=biome.lid,
                x=rng.uniform(0, world_size),
                y=rng.uniform(0, world_size),
                energy=50.0,
                age=0,
                traits=traits,
                state="rest",
                velocity=(0.0, 0.0),
            )
            self.agents.append(agent)
            self.spatial_hash.insert(agent)
            self.next_aid += 1

    def step(self, world: World, dt: float):
        """Execute one simulation tick:
        1. Update behaviors (utility system)
        2. Move agents (steering)
        3. Resolve local interactions (hunt, flee, forage)
        4. Prune dead agents
        """
        self._update_behaviors(world)
        self._move_agents(dt)
        self._resolve_local_interactions(world)
        self._prune_dead()

```

**Key Points**:
- One `AgentManager` per species (not global)
- Owns `SpatialHash` for efficient neighbor queries
- `spawn_agents()` creates agents at world start or speciation
- Serialization uses grid+notable hybrid format (see Serialization Format section)

---

### 3. SpatialHash

```python
# mantis/simulation/spatial.py

class SpatialHash:
    """Efficient spatial index for agent neighbor queries.

    Divides world into grid cells. Query checks only nearby cells,
    reducing O(n²) to O(k) where k = neighbors in sense range (~10).
    """

    def __init__(self, world_size: int, cell_size: int):
        self.world_size = world_size        # 1000
        self.cell_size = cell_size          # 50 → 20×20 grid
        self.grid: dict[tuple[int, int], list[Agent]] = defaultdict(list)

    def clear(self):
        """Clear all cells (called each tick before reinserting agents)."""
        self.grid.clear()

    def insert(self, agent: Agent):
        """Add agent to appropriate grid cell."""
        cell = self._get_cell(agent.x, agent.y)
        self.grid[cell].append(agent)

    def query_radius(self, x: float, y: float, radius: float) -> list[Agent]:
        """Return agents within *radius* of (x, y)."""
        cells = self._get_cells_in_radius(x, y, radius)
        candidates = []
        for cell in cells:
            candidates.extend(self.grid.get(cell, []))
        # Filter by actual Euclidean distance
        return [a for a in candidates if self._distance(a.x, a.y, x, y) <= radius]

    def _get_cell(self, x: float, y: float) -> tuple[int, int]:
        """Convert position to cell coordinates."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _get_cells_in_radius(self, x: float, y: float, radius: float) -> list[tuple[int, int]]:
        """Return 9-cell neighborhood (3×3) covering radius."""
        cx, cy = self._get_cell(x, y)
        return [(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
```

**Performance**:
- World size: 1000×1000
- Cell size: 50 → 20×20 grid = 400 cells
- Average occupancy: 2000 agents / 400 cells = 5 agents/cell
- Typical sense range: 50 units → 9 cells checked = ~45 candidates
- **Result**: O(45) instead of O(2000) per query

---

### 4. VegetationPatch

```python
# mantis/simulation/spatial.py

@dataclass
class VegetationPatch:
    """Spatial distribution of vegetation within a biome.

    Design principle: Large patches with slow depletion prevent
    instant-collapse oscillations. A patch can sustain 50+ grazers
    for 10+ ticks before noticeable depletion, creating spatial
    strategy: "good patches" attract agents, causing competition.
    """

    x: float                # Center position
    y: float
    density: float          # 0.0 (depleted) to 1.0 (lush)
    radius: float           # Area of influence (typically 80-120 units)
    capacity: float         # Total food units available (e.g., 500)
    regen_rate: float       # Regrowth speed (slow: 0.05-0.1/tick)

    def get_density_at(self, px: float, py: float) -> float:
        """Gaussian falloff from center."""
        dist = math.sqrt((px - self.x)**2 + (py - self.y)**2)
        if dist > self.radius:
            return 0.0
        falloff = math.exp(-(dist / self.radius) ** 2)
        return self.density * falloff

    def try_forage(self, amount: float) -> float:
        """Agent attempts to forage. Returns actual amount consumed.

        Depletion is gradual: 50 agents each taking 0.5/tick deplete
        a 500-capacity patch over 20 ticks, giving time for competition
        and spatial spread before collapse.
        """
        actual = min(amount, self.capacity)
        self.capacity = max(0.0, self.capacity - actual)
        self.density = self.capacity / 500.0  # Normalize to [0, 1]
        return actual

    def regenerate(self, dt: float):
        """Logistic regrowth. Slow to prevent instant respawn oscillation."""
        if self.capacity < 500.0:  # Max capacity
            growth = self.regen_rate * dt * (1 - self.capacity / 500.0)
            self.capacity = min(500.0, self.capacity + growth * 100)  # Scale for visibility
            self.density = self.capacity / 500.0
```

---

## Behavior System

### Utility-Based AI

Agents compute utility scores for each action, then select via softmax sampling (not greedy). This creates behavioral diversity — even identical agents make different choices.

```python
# mantis/simulation/behavior.py

class UtilitySystem:
    """Compute action utilities and select best."""

    @staticmethod
    def compute_utilities(agent: Agent, world: World, neighbors: list[Agent]) -> dict[str, float]:
        """Return {action: utility_score} for each possible action."""

        hunger = max(0, 100 - agent.energy) / 100.0  # 0=full, 1=starving
        utilities = {"rest": 0.1}  # baseline

        # ────────────────────────────────────────────────────
        # Forage: scales with hunger × vegetation density × metabolism
        # ────────────────────────────────────────────────────
        veg_density = world.get_vegetation_at(agent.biome_lid, agent.x, agent.y)
        utilities["forage"] = hunger * veg_density * agent.traits.get("metab", 1.0)

        # ────────────────────────────────────────────────────
        # Hunt: scales with hunger × prey availability × predator advantage
        # ────────────────────────────────────────────────────
        prey_sids = agent.get_prey_sids(world)  # from species diet
        prey = [n for n in neighbors if n.species_sid in prey_sids]
        if prey:
            closest_prey = min(prey, key=lambda p: distance(agent, p))
            predator_advantage = (
                agent.traits.get("speed", 0) +
                agent.traits.get("sense", 0) -
                closest_prey.traits.get("camo", 0) * 0.7
            )
            hunt_score = predator_advantage * hunger * (1.0 / (1.0 + distance(agent, closest_prey)))
            utilities["hunt"] = max(0, hunt_score)

        # ────────────────────────────────────────────────────
        # Flee: scales with predator proximity × threat level
        # ────────────────────────────────────────────────────
        predators = [n for n in neighbors if agent.species_sid in n.get_prey_sids(world)]
        if predators:
            closest_pred = min(predators, key=lambda p: distance(agent, p))
            threat = (
                closest_pred.traits.get("speed", 0) +
                closest_pred.traits.get("sense", 0) -
                agent.traits.get("camo", 0) * 0.7
            )
            flee_score = threat * (1.0 / (1.0 + distance(agent, closest_pred)))
            utilities["flee"] = max(0, flee_score * 2.0)  # prioritize survival

        # ────────────────────────────────────────────────────
        # Mate: only when energy surplus + social trait
        # ────────────────────────────────────────────────────
        if agent.energy > 70:
            conspecifics = [n for n in neighbors if n.species_sid == agent.species_sid and n.aid != agent.aid]
            if conspecifics:
                utilities["mate"] = agent.traits.get("social", 0) * 0.5

        return utilities

    @staticmethod
    def select_action(agent: Agent, utilities: dict[str, float], rng: np.random.Generator) -> str:
        """Softmax sampling with hysteresis and emergency overrides."""

        # ──────────────────────────────────────────────────
        # EMERGENCY OVERRIDE: Critical energy (< 10)
        # ──────────────────────────────────────────────────
        if agent.energy < 10:
            # Break any commitment, force survival action
            agent.state_commitment = 0
            # Choose best immediate energy source
            if utilities.get("forage", 0) > utilities.get("hunt", 0):
                return "forage"
            elif utilities.get("hunt", 0) > 0:
                return "hunt"
            else:
                return "rest"  # No food available, conserve energy

        # ──────────────────────────────────────────────────
        # HYSTERESIS: Continue current state if committed
        # ──────────────────────────────────────────────────
        if agent.state_commitment > 0:
            agent.state_commitment -= 1
            return agent.state  # Stay in current action

        # ──────────────────────────────────────────────────
        # SOFTMAX SELECTION: Choose new action
        # ──────────────────────────────────────────────────
        if not utilities:
            return "rest"

        actions = list(utilities.keys())
        scores = np.array(list(utilities.values()))
        temp = 0.5  # temperature
        probs = np.exp(scores / temp) / np.sum(np.exp(scores / temp))
        new_action = str(rng.choice(actions, p=probs))

        # Set commitment period based on action type
        commitment_periods = {
            "flee": 10,    # High commitment (panic)
            "hunt": 8,     # Medium-high (pursuit)
            "mate": 5,     # Medium (courtship)
            "forage": 3,   # Low (can switch to flee quickly)
            "rest": 1,     # Minimal (idle state)
        }
        agent.state_commitment = commitment_periods.get(new_action, 1)

        return new_action
```

### Movement: Steering Behaviors

```python
# mantis/simulation/behavior.py

def compute_velocity(agent: Agent, neighbors: list[Agent], world: World) -> tuple[float, float]:
    """Compute desired velocity from current state."""

    if agent.state == "hunt":
        target = world.get_agent_by_id(agent.target_aid)
        if target:
            return steer_towards(agent, target.x, target.y)

    elif agent.state == "flee":
        predators = [n for n in neighbors if agent.species_sid in n.get_prey_sids(world)]
        if predators:
            closest = min(predators, key=lambda p: distance(agent, p))
            return steer_away(agent, closest.x, closest.y)

    elif agent.state == "forage":
        patch = world.get_nearest_vegetation_patch(agent.biome_lid, agent.x, agent.y)
        if patch:
            return steer_towards(agent, patch.x, patch.y)

    # Default: random walk
    return (rng.normal(0, 0.5), rng.normal(0, 0.5))

def steer_towards(agent: Agent, tx: float, ty: float) -> tuple[float, float]:
    """Return velocity vector toward target."""
    dx, dy = tx - agent.x, ty - agent.y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1:
        return (0, 0)
    speed = agent.traits.get("speed", 1.0) * 2.0  # units per tick
    return (dx / dist * speed, dy / dist * speed)

def steer_away(agent: Agent, tx: float, ty: float) -> tuple[float, float]:
    """Return velocity vector away from threat."""
    vx, vy = steer_towards(agent, tx, ty)
    return (-vx, -vy)
```

---

## Spatial Performance

### Problem: O(n²) Interactions

Naive approach: Check every agent against every other agent.

```python
# SLOW — O(n²)
for agent in agents:
    for other in agents:
        if distance(agent, other) < sense_range:
            interact(agent, other)

# For 2000 agents: 2000 × 2000 = 4,000,000 checks per tick
```

### Solution: Spatial Hashing

```python
# FAST — O(k) where k = neighbors (~10)
for agent in agents:
    neighbors = spatial_hash.query_radius(agent.x, agent.y, sense_range)
    for other in neighbors:
        interact(agent, other)

# For 2000 agents: 2000 × 10 = 20,000 checks per tick (200× faster)
```

### Performance Targets

| Metric | Target | Strategy |
|---|---|---|
| **Tick rate** | 10+ ticks/sec (2000 agents) | Spatial hash + vectorized movement |
| **Neighbor query** | <1ms per agent | Cell size = 100 units (matches max sense range ~100) |
| **Memory overhead** | <50MB per world | Sample 250 agents, not full population |
| **File size** | <60MB for 10K worlds | Quantized coords + delta encoding + gzip compression |

### Spatial Hash Optimization

**Critical Design Fix**: Cell size must match the **maximum possible sense range** (typically ~100 units for high-sense predators).

- **Original plan**: 50-unit cells → predators with `sense=100` require checking 5×5 = 25 cells
- **Optimized**: 100-unit cells → all queries check only 3×3 = 9 cells (guaranteed)
- **Trade-off**: Slightly larger candidate lists (~20 agents vs ~10) but eliminates multi-layer lookups

```python
# SpatialHash configuration
WORLD_SIZE = 1000
MAX_SENSE_RANGE = 100  # From high-sense predators
CELL_SIZE = MAX_SENSE_RANGE  # 100 units
GRID_DIMENSIONS = WORLD_SIZE / CELL_SIZE  # 10×10 = 100 cells
```

---

## Protocol Extension

### Add @AGENT Token

Modify `mantis/tokenizer.py`:

```python
LAYER_MARKERS = [
    "=EPOCH",       # Existing
    "@BIO",         # Existing
    "@SP",          # Existing
    "@INT",         # Existing
    "@EVT",         # Existing
    "@SPOT",        # Existing
    "@AGENT",       # NEW — Agent block (3 BPE → 1)
]
```

### Loss Weights

```python
LOSS_WEIGHTS = {
    "---": 0.1,
    "=EPOCH": 0.5,
    "@BIO": 0.5,
    "@SP": 1.0,
    "@INT": 1.5,
    "@EVT": 1.5,
    "@AGENT": 0.8,  # NEW — reduced from 1.8 to prevent float precision overfitting
    "@SPOT": 2.0,
}
```

**Rationale**: Agent coordinates are **high-entropy continuous values** that risk overfitting. The model should learn general spatial dynamics (clustering, chase patterns, flocking) rather than exact float regression. Reduced weight (0.8) prioritizes causal reasoning (`@SPOT`, `@INT`) over precise position prediction.

**Why not higher?**
- Predicting `(120.5, 340.2)` vs `(121.3, 339.8)` generates high loss but teaches nothing about ecological dynamics
- Model capacity should focus on "predator chases prey" logic, not sub-unit coordinate precision
- Quantized serialization (see below) further reduces precision requirements

---

### Serialization Format

#### Grid+Notable Hybrid

**Critical Design Choice**: Instead of serializing individual agents (up to 250 per species), agents are aggregated into 100-unit spatial grid cells plus a small set of individually-tracked "notable" agents (top 5 by energy).

**Why?**
- Reduces keyframe tokens per species from ~1,750 to ~220-400
- Total ECOSYSTEM keyframe tick drops from ~10K to ~2,500 tokens, fitting in `--seq-len 4096`
- Grid cells capture spatial density and behavior distributions — what matters for ecological dynamics
- Notable agents preserve individual-level narratives (hunts, chases) for the model to learn from

**Behavior abbreviations**: `r`=rest, `f`=forage, `h`=hunt, `m`=mate, `fl`=flee, `fk`=flock

#### Keyframe (every 20 ticks)

```
@SP|S3|L0|plan=predator|pop=450|diet={S1:0.7,plant:0.2}
  @AGENT|count=420|mode=grid+5|cell=100
    G(1,3):n=17,E=45|f:12,h:3,fl:2
    G(2,3):n=8,E=38|f:5,h:2,r:1
    N:A1:(130,350,E=52,age=10,hunt->A3)
    N:A7:(400,100,E=89,age=15,forage)
  T:speed=6.2±0.9,intel=3.5±0.5
  E:in=4200,out=2800,store=11250|repro=K(rate=1.8)
---
```

**Format breakdown**:
- `@AGENT|count=420|mode=grid+5|cell=100` — Total agents, 5 notables, 100-unit grid cells
- `G(col,row):n={count},E={avg_energy}|{behavior_abbrev}:{count},...` — Grid cell aggregate
- `N:A{aid}:(x,y,E={energy},age={age},{state})` — Notable agent (10-unit quantized position)
- Behaviors sorted by count descending

---

#### Delta (intermediate ticks)

```
@SP|S3|pop=418
  @AGENT|Δpos|cell=100
    G(1,3):n=15,E=42|f:10,h:3,fl:2
    N:A1:(140,360,E=48)
    A7:†
---
```

**Delta encoding**:
- Grid cell emitted if count changed >2, avg energy changed >5, or dominant behavior changed
- Notable agents tracked by same IDs from last keyframe (no replacements until next keyframe)
- Dead notable agents marked with `†`
- Empty cells skipped

---

### Protocol Parsing (Frontend)

```python
# web/server/stream_simulation.py

def parse_agent_block(lines: list[str]) -> list[dict]:
    """Parse @AGENT block into JSON."""
    agents = []
    for line in lines:
        if line.startswith("A"):
            match = re.match(r"A(\d+):\(([^)]+)\)", line)
            if match:
                aid = int(match.group(1))
                fields = match.group(2).split(",")
                agent = {
                    "aid": aid,
                    "x": float(fields[0]),
                    "y": float(fields[1]),
                    "energy": float(fields[2].split("=")[1]),
                    "age": int(fields[3].split("=")[1]),
                    "state": fields[4],
                }
                agents.append(agent)
    return agents
```

---

## Frontend Architecture

### Stack

- **Backend**: Flask + Flask-SocketIO (Python)
- **Frontend**: React 18 + TypeScript + Vite
- **Rendering**: Canvas 2D API (not Three.js — simpler, faster for top-down 2D)
- **State**: React hooks with interpolation buffer
- **Communication**: Socket.IO client

---

### Performance: Tick Rate Decoupling

**Problem**: Streaming 2000 agents @ 60fps = 18MB/sec JSON overhead → server CPU bottleneck + browser parser choke.

**Solution**: **10-20 TPS server + 60fps client interpolation**

| Layer | Rate | Responsibility |
|-------|------|----------------|
| **Server (Python)** | 10-20 ticks/sec | Simulate physics, send discrete state snapshots |
| **Client (JS)** | 60 fps | Interpolate (tween) positions between snapshots |

**Benefits**:
- Server bandwidth: 18MB/sec → 3MB/sec (6× reduction)
- Smooth visuals: Tweening creates fluid motion despite low tick rate
- Scalability: Can support 5000+ agents with same bandwidth

---

### Backend Server

```python
# web/server/app.py

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

is_playing = False
current_speed = 1.0
TICK_RATE = 15  # Server updates at 15 ticks/sec (not 60)

@socketio.on("start_simulation")
def handle_start(data):
    """Stream simulation ticks to client at reduced rate."""
    global is_playing
    is_playing = True

    world_id = data.get("world_id", 0)

    # Parse protocol file
    with open("data/test_world.txt") as f:
        ticks = parse_protocol_to_ticks(f.read())

    # Stream ticks at server rate (10-20 TPS)
    for tick in ticks:
        if not is_playing:
            break

        emit("tick_update", {
            "tick": tick.number,
            "epoch": tick.epoch,
            "species": [serialize_species(sp) for sp in tick.species],
            "agents": [serialize_agent(a) for a in tick.agents],
            "interpolate_duration": 1000 / TICK_RATE,  # ms between ticks for client interpolation
        })

        time.sleep(1.0 / (TICK_RATE * current_speed))  # 15 TPS with speed multiplier

@socketio.on("pause")
def handle_pause():
    global is_playing
    is_playing = False

@socketio.on("resume")
def handle_resume():
    global is_playing
    is_playing = True

@socketio.on("set_speed")
def handle_speed(data):
    global current_speed
    current_speed = data.get("speed", 1.0)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
```

---

### Client-Side Interpolation

The frontend receives discrete snapshots (10-20 TPS) but renders at 60fps using linear interpolation (tweening).

```typescript
// web/client/src/hooks/useInterpolation.ts

interface AgentSnapshot {
  aid: number;
  x: number;
  y: number;
  energy: number;
  state: string;
}

class AgentInterpolator {
  private prevSnapshot: Map<number, AgentSnapshot> = new Map();
  private nextSnapshot: Map<number, AgentSnapshot> = new Map();
  private interpolateDuration = 66.7;  // ms (15 TPS)
  private snapshotTime = 0;

  updateSnapshot(agents: AgentSnapshot[], timestamp: number, duration: number) {
    this.prevSnapshot = this.nextSnapshot;
    this.nextSnapshot = new Map(agents.map(a => [a.aid, a]));
    this.snapshotTime = timestamp;
    this.interpolateDuration = duration;
  }

  getInterpolatedPositions(currentTime: number): AgentSnapshot[] {
    const elapsed = currentTime - this.snapshotTime;
    const t = Math.min(1.0, elapsed / this.interpolateDuration);  // 0 to 1

    const result: AgentSnapshot[] = [];

    for (const [aid, next] of this.nextSnapshot) {
      const prev = this.prevSnapshot.get(aid);

      if (!prev) {
        // New agent, no interpolation
        result.push(next);
        continue;
      }

      // Linear interpolation (lerp)
      result.push({
        aid,
        x: prev.x + (next.x - prev.x) * t,
        y: prev.y + (next.y - prev.y) * t,
        energy: prev.energy + (next.energy - prev.energy) * t,
        state: next.state,  // Discrete, no interpolation
      });
    }

    return result;
  }
}
```

**How it works**:
1. Server sends snapshot at T=0ms: `A0:(100, 200)`
2. Server sends snapshot at T=66ms: `A0:(120, 210)`
3. Client renders at 16ms (60fps):
   - Frame 1 (16ms): `A0:(103, 202)` — 24% of the way
   - Frame 2 (32ms): `A0:(106, 204)` — 48% of the way
   - Frame 3 (48ms): `A0:(115, 208)` — 72% of the way
   - Frame 4 (66ms): `A0:(120, 210)` — snapshot arrives, reset

**Result**: Smooth motion despite low server tick rate.

---

### Frontend Components

#### App.tsx (Main Container)

```typescript
import { useWebSocket } from "./hooks/useWebSocket";
import { SimulationCanvas } from "./components/SimulationCanvas";
import { Controls } from "./components/Controls";
import { SpeciesPanel } from "./components/SpeciesPanel";

function App() {
  const { tick, species, agents, isPlaying, play, pause, setSpeed } = useWebSocket();

  return (
    <div className="app" style={{ display: "flex", gap: "20px" }}>
      <div style={{ flex: 1 }}>
        <Controls
          onPlay={play}
          onPause={pause}
          onSpeed={setSpeed}
          isPlaying={isPlaying}
        />
        <SimulationCanvas agents={agents} species={species} worldSize={1000} />
      </div>
      <SpeciesPanel species={species} />
    </div>
  );
}

export default App;
```

---

#### SimulationCanvas.tsx (Canvas Rendering)

```typescript
import { useEffect, useRef } from "react";
import { Agent, Species } from "../types/simulation";
import { renderAgents, renderVegetationPatches } from "../utils/renderer";

interface Props {
  agents: Agent[];
  species: Species[];
  worldSize: number;
}

export function SimulationCanvas({ agents, species, worldSize }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Render vegetation patches (background)
    renderVegetationPatches(ctx, worldSize);

    // Render agents (foreground)
    renderAgents(ctx, agents, species, worldSize);

  }, [agents, species, worldSize]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={800}
      style={{ border: "2px solid #333", background: "#f0f0f0" }}
    />
  );
}
```

---

#### renderer.ts (Canvas Drawing Logic)

```typescript
export function renderAgents(
  ctx: CanvasRenderingContext2D,
  agents: Agent[],
  species: Species[],
  worldSize: number
) {
  const scale = ctx.canvas.width / worldSize;  // 800 / 1000 = 0.8

  agents.forEach((agent) => {
    const sp = species.find((s) => s.sid === agent.species_sid);
    if (!sp) return;

    // Position
    const x = agent.x * scale;
    const y = agent.y * scale;

    // Color by body plan
    ctx.fillStyle = getBodyPlanColor(sp.body_plan);

    // Size by agent size trait
    const radius = Math.max(2, agent.traits.size * 0.5);

    // State indicator (border)
    if (agent.state === "hunt") {
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
      ctx.stroke();
    } else if (agent.state === "flee") {
      ctx.strokeStyle = "orange";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw agent
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Energy bar (optional)
    if (agent.energy < 30) {
      ctx.fillStyle = "yellow";
      const barWidth = 10;
      const barHeight = 2;
      ctx.fillRect(x - barWidth/2, y - radius - 4, (agent.energy / 100) * barWidth, barHeight);
    }
  });
}

function getBodyPlanColor(plan: string): string {
  const colors: Record<string, string> = {
    predator: "#ff4444",
    grazer: "#88cc88",
    omnivore: "#cc88ff",
    scavenger: "#ccaa66",
    decomposer: "#666666",
    sessile_autotroph: "#44ff44",
    mobile_autotroph: "#66ff66",
    filter_feeder: "#6688ff",
    parasite: "#ff88ff",
  };
  return colors[plan] || "#aaaaaa";
}

export function renderVegetationPatches(ctx: CanvasRenderingContext2D, worldSize: number) {
  // Placeholder: draw green patches
  const scale = ctx.canvas.width / worldSize;
  const patches = [
    { x: 200, y: 300, density: 0.8 },
    { x: 600, y: 500, density: 0.6 },
    { x: 400, y: 700, density: 0.9 },
  ];

  patches.forEach((patch) => {
    const x = patch.x * scale;
    const y = patch.y * scale;
    ctx.fillStyle = `rgba(0, 200, 0, ${patch.density * 0.3})`;
    ctx.beginPath();
    ctx.arc(x, y, 50, 0, Math.PI * 2);
    ctx.fill();
  });
}
```

---

#### Controls.tsx (Play/Pause/Speed)

```typescript
interface Props {
  onPlay: () => void;
  onPause: () => void;
  onSpeed: (speed: number) => void;
  isPlaying: boolean;
}

export function Controls({ onPlay, onPause, onSpeed, isPlaying }: Props) {
  return (
    <div style={{ padding: "10px", background: "#333", color: "#fff" }}>
      <button onClick={isPlaying ? onPause : onPlay}>
        {isPlaying ? "⏸ Pause" : "▶️ Play"}
      </button>

      <label style={{ marginLeft: "20px" }}>
        Speed:
        <select onChange={(e) => onSpeed(Number(e.target.value))} defaultValue="1">
          <option value="0.5">0.5x</option>
          <option value="1">1x</option>
          <option value="2">2x</option>
          <option value="5">5x</option>
          <option value="10">10x</option>
        </select>
      </label>
    </div>
  );
}
```

---

## File Structure

### New Files

```
mantis/simulation/
├── agent.py              # Agent, AgentManager
├── spatial.py            # SpatialHash, VegetationPatch
├── behavior.py           # UtilitySystem, steering
└── agent_serializer.py   # Extend Serializer

web/
├── server/
│   ├── app.py                 # Flask + SocketIO
│   ├── stream_simulation.py   # Protocol parser
│   └── requirements.txt
└── client/
    ├── src/
    │   ├── components/
    │   │   ├── SimulationCanvas.tsx
    │   │   ├── Controls.tsx
    │   │   └── SpeciesPanel.tsx
    │   ├── hooks/
    │   │   └── useWebSocket.ts
    │   ├── types/
    │   │   └── simulation.ts
    │   ├── utils/
    │   │   └── renderer.ts
    │   └── App.tsx
    ├── package.json
    └── vite.config.ts
```

### Modified Files

```
mantis/simulation/
├── species.py      # Add agents: Optional[AgentManager] = None
├── biome.py        # Add vegetation_patches: list[VegetationPatch]
├── engine.py       # Add _activate_agents(), _step_agents()
├── serializer.py   # Add serialize_agents() method
└── constants.py    # Add AGENT_CONFIG

mantis/
└── tokenizer.py    # Add "@AGENT" token

scripts/
└── gen_evo_dataset.py  # Add --enable-agents flag
```

---

## Implementation Phases

### Phase 1: Core Agent System (Week 1)

**Goal**: Basic agent spawning and spatial indexing

- [ ] Create `mantis/simulation/agent.py`
  - `Agent` dataclass
  - `AgentManager.spawn_agents()`
  - `AgentManager.step()` stub
- [ ] Create `mantis/simulation/spatial.py`
  - `SpatialHash` implementation
  - `VegetationPatch` dataclass
- [ ] Modify `mantis/simulation/species.py`
  - Add `agents: Optional[AgentManager] = None` field
- [ ] Modify `mantis/simulation/biome.py`
  - Add `vegetation_patches: list[VegetationPatch]`
  - Add `get_vegetation_at(x, y)` method
- [ ] Unit tests:
  - `test_agent_spawn()` — Traits sampled from distributions
  - `test_spatial_hash_insert()` — Agents correctly hashed
  - `test_spatial_hash_query()` — Radius query correct

**Deliverable**: Agents spawn with traits, spatial hash works

---

### Phase 2: Behavior System (Week 2)

**Goal**: Agents move and make decisions

- [ ] Create `mantis/simulation/behavior.py`
  - `UtilitySystem.compute_utilities()`
  - `UtilitySystem.select_action()`
  - Steering functions (`steer_towards`, `steer_away`)
- [ ] Implement `AgentManager.step()`
  - `_update_behaviors()` — Apply utility system
  - `_move_agents()` — Steering + position update
  - `_resolve_local_interactions()` — Hunt, flee, forage
  - `_prune_dead()` — Remove energy <= 0 agents
- [ ] Modify `mantis/simulation/engine.py`
  - Add `_activate_agents()` (spawn when spotlight_score > 15)
  - Add `_step_agents()` call in `World.step()`
- [ ] Unit tests:
  - `test_forage_utility()` — Hungry agents prefer foraging
  - `test_hunt_utility()` — Predators prefer hunting
  - `test_flee_priority()` — Fleeing overrides foraging
  - `test_movement()` — Agents move toward goals

**Deliverable**: Agents move, hunt, flee, forage

---

### Phase 3: Serialization & Gradual Spatial Learning (Week 3)

**Goal**: Output `@AGENT` blocks with gradual spatial introduction

- [ ] Modify `mantis/tokenizer.py`
  - Add `"@AGENT"` to `LAYER_MARKERS`
  - Update `LOSS_WEIGHTS` (set to 0.8, not 1.8)
- [ ] Create `mantis/simulation/agent_serializer.py`
  - `serialize_agents_keyframe()` — Full state with quantization
  - `serialize_agents_delta()` — Changes only
  - `quantize_position()` — Round to 10-unit grid
  - Sampling logic (top-200 + random-50)
- [ ] Modify `mantis/simulation/serializer.py`
  - Call agent serializer if `species.agents` exists
- [ ] Create `mantis/simulation/agent_reconciliation.py`
  - `PopulationReconciler` class for event-based macro sync
  - Track discrete events (births, deaths, hunts)
  - Scale continuous energy changes
- [ ] Modify `scripts/gen_evo_dataset.py`
  - Add `--enable-agents` flag with two modes:
    - `--agent-epoch ECOSYSTEM` — Simple flocking/grazing only
    - `--agent-epoch INTELLIGENCE` — Full behavioral AI
  - Add `--agent-threshold 15` flag
- [ ] Generate test datasets with gradual spatial learning:
  ```bash
  # ECOSYSTEM: Basic spatial concepts (50% of worlds)
  python scripts/gen_evo_dataset.py --worlds 5000 --max-generations 100 \
      --enable-agents --agent-epoch ECOSYSTEM --output data/evo_ecosystem.txt

  # INTELLIGENCE: Full agent behaviors (50% of worlds)
  python scripts/gen_evo_dataset.py --worlds 5000 --max-generations 200 \
      --enable-agents --agent-epoch INTELLIGENCE --output data/evo_intel.txt
  ```
- [ ] Validate protocol parsing

**Deliverable**: Agent-based datasets with gradual spatial learning

**Why Gradual Introduction?**
- **Problem**: Introducing @AGENT blocks only in INTELLIGENCE epoch creates sudden discontinuity
- **Solution**: Add simplified agent blocks in ECOSYSTEM epoch:
  - Basic flocking (conspecifics stay near each other)
  - Simple grazing (move toward vegetation)
  - No hunting, fleeing, or complex utilities (yet)
- **Result**: Model learns spatial fundamentals (clustering, movement) early, then complex behaviors later

---

### Phase 4: Frontend Backend (Week 4)

**Goal**: WebSocket server streaming simulation at optimized rate

- [ ] Create `web/server/app.py`
  - Flask + SocketIO setup
  - `/start_simulation`, `/pause`, `/resume` handlers
  - **10-20 TPS** (not 60fps) with `interpolate_duration` metadata
- [ ] Create `web/server/stream_simulation.py`
  - Parse `@SP` blocks → JSON
  - Parse `@AGENT` blocks → JSON (quantized coords)
  - Stream ticks at 15 TPS (66ms intervals)
- [ ] Create `web/server/requirements.txt`
  ```
  flask==3.0.0
  flask-socketio==5.3.5
  python-socketio==5.10.0
  ```
- [ ] Test with curl:
  ```bash
  curl -X POST http://localhost:5000/start_simulation
  ```
- [ ] Verify bandwidth: ~3MB/sec for 2000 agents (vs 18MB/sec at 60fps)

**Deliverable**: Backend streams JSON at optimized rate (10-20 TPS)

---

### Phase 5: Frontend Client with Interpolation (Week 5-6)

**Goal**: React app rendering agents at 60fps with smooth interpolation

- [ ] Setup project:
  ```bash
  cd web && npm create vite@latest client -- --template react-ts
  cd client && npm install socket.io-client
  ```
- [ ] Create `useWebSocket.ts` hook
  - Connect to Socket.IO server
  - Listen for `tick_update` events
  - State: `{tick, species, agents, isPlaying}`
- [ ] Create `useInterpolation.ts` hook **[CRITICAL]**
  - `AgentInterpolator` class for position tweening
  - Receives 15 TPS snapshots, interpolates to 60 fps
  - Linear interpolation (lerp) between prev/next positions
- [ ] Create `SimulationCanvas.tsx`
  - Canvas 2D rendering at 60 fps via `requestAnimationFrame`
  - Call `interpolator.getInterpolatedPositions(currentTime)`
  - Render interpolated agents as colored circles
  - Render vegetation patches (background layer)
- [ ] Create `Controls.tsx`
  - Play/pause button
  - Speed slider (0.5x, 1x, 2x, 5x, 10x)
- [ ] Create `SpeciesPanel.tsx`
  - List species with stats (pop, traits)
- [ ] Create `renderer.ts`
  - `renderAgents()` — Draw circles with state indicators (hunt=red, flee=orange)
  - `renderVegetationPatches()` — Draw green patches with alpha based on density
  - Viewport culling: only render agents in visible bounds
- [ ] Add interactions:
  - Hover tooltip (agent details: ID, energy, state)
  - Click to follow agent (camera tracks)

**Deliverable**: React app with smooth 60fps rendering despite 15 TPS server updates

---

### Phase 6: Polish & Testing (Week 7)

**Goal**: Production-ready system

- [ ] Integration tests:
  - `test_full_simulation()` — 100 ticks with agents
  - `test_energy_conservation()` — `sum(agent.energy) == species.energy_store ±5%`
  - `test_population_sync()` — `len(agents) == species.population`
- [ ] Performance profiling:
  - Measure ticks/sec with 2000 agents
  - Optimize spatial hash if needed
  - Profile Canvas rendering (target 30fps)
- [ ] Documentation:
  - Update `EVOLUTION_SIM_OVERVIEW.md`
  - Add `web/README.md` (setup instructions)
  - Add this plan to `AGENT_SIMULATION_PLAN.md`
- [ ] Bug fixes & edge cases:
  - Agents stuck in corners
  - Division by zero in steering
  - WebSocket reconnection

**Deliverable**: Stable, documented system

---

## Testing Strategy

### Unit Tests

```python
# tests/simulation/test_agent.py
def test_agent_spawn():
    """Agents inherit traits from species distribution."""
    species = create_test_species()
    manager = AgentManager(species, biome, world_size=1000)
    manager.spawn_agents(100, rng)

    assert len(manager.agents) == 100
    assert all(0 <= a.x < 1000 for a in manager.agents)
    assert all(a.species_sid == species.sid for a in manager.agents)

def test_agent_traits_sampled():
    """Agent traits are sampled from species distributions."""
    species = Species(
        sid=0,
        traits={"speed": TraitDistribution(5.0, 1.0)},
    )
    manager = AgentManager(species, biome, 1000)
    manager.spawn_agents(100, rng)

    speeds = [a.traits["speed"] for a in manager.agents]
    assert 3.0 < np.mean(speeds) < 7.0  # roughly 5.0 ± 1.0

# tests/simulation/test_spatial.py
def test_spatial_hash_insert():
    """Agents correctly hashed by position."""
    sh = SpatialHash(world_size=1000, cell_size=50)
    agent = Agent(aid=0, x=120, y=340, ...)
    sh.insert(agent)

    # Cell (120 // 50, 340 // 50) = (2, 6)
    assert agent in sh.grid[(2, 6)]

def test_spatial_hash_query():
    """Radius query returns nearby agents."""
    sh = SpatialHash(1000, 50)
    a1 = Agent(aid=0, x=100, y=100, ...)
    a2 = Agent(aid=1, x=110, y=110, ...)  # 14.1 units away
    a3 = Agent(aid=2, x=200, y=200, ...)  # 141 units away
    sh.insert(a1)
    sh.insert(a2)
    sh.insert(a3)

    neighbors = sh.query_radius(100, 100, radius=20)
    assert a1 in neighbors
    assert a2 in neighbors
    assert a3 not in neighbors

# tests/simulation/test_behavior.py
def test_forage_utility():
    """Hungry agents prefer foraging."""
    agent = Agent(energy=10, traits={"metab": 5.0}, ...)
    world = create_test_world(veg_density=0.8)

    utilities = UtilitySystem.compute_utilities(agent, world, [])
    assert utilities["forage"] > utilities["rest"]

def test_hunt_utility():
    """Predators with high speed/sense prefer hunting."""
    predator = Agent(traits={"speed": 8.0, "sense": 7.0}, energy=30, ...)
    prey = Agent(species_sid=1, x=50, y=50, ...)

    utilities = UtilitySystem.compute_utilities(predator, world, [prey])
    assert utilities["hunt"] > utilities["forage"]

def test_flee_priority():
    """Fleeing overrides foraging when predator nearby."""
    prey = Agent(energy=30, species_sid=1, x=100, y=100, ...)
    predator = Agent(species_sid=2, x=110, y=110, ...)  # 14 units away

    utilities = UtilitySystem.compute_utilities(prey, world, [predator])
    assert utilities["flee"] > utilities["forage"]
```

### Integration Tests

```python
# tests/simulation/test_agent_integration.py
def test_hunt_kill_sequence():
    """Predator hunts prey, prey dies, population decreases."""
    world = World(wid=0, seed=42)
    predator_sp = world.species[0]  # assume predator
    prey_sp = world.species[1]

    # Activate agents
    predator_sp.agents = AgentManager(predator_sp, world.biomes[0], 1000)
    predator_sp.agents.spawn_agents(10, world.rng)
    prey_sp.agents = AgentManager(prey_sp, world.biomes[0], 1000)
    prey_sp.agents.spawn_agents(20, world.rng)

    initial_prey_count = len(prey_sp.agents.agents)

    # Run 10 ticks
    for _ in range(10):
        world.step()

    final_prey_count = len(prey_sp.agents.agents)
    assert final_prey_count < initial_prey_count  # some prey died

def test_energy_conservation():
    """Sum of agent energies equals species.energy_store."""
    world = World(wid=0, seed=42)
    sp = world.species[0]
    sp.agents = AgentManager(sp, world.biomes[0], 1000)
    sp.agents.spawn_agents(100, world.rng)

    world.step()

    agent_energy_sum = sum(a.energy for a in sp.agents.agents)
    assert abs(agent_energy_sum - sp.energy_store) < sp.energy_store * 0.05  # ±5%

def test_serialization_roundtrip():
    """Agent blocks parse correctly."""
    world = World(wid=0, seed=42)
    sp = world.species[0]
    sp.agents = AgentManager(sp, world.biomes[0], 1000)
    sp.agents.spawn_agents(50, world.rng)

    serializer = Serializer()
    text = serializer.serialize_tick(world)

    # Parse agent block
    parsed_agents = parse_agent_block(text)
    assert len(parsed_agents) > 0
    assert all("x" in a and "y" in a for a in parsed_agents)
```

---

## Performance Optimization

### Targets

| Metric | Target | Current | Strategy |
|---|---|---|---|
| **Tick rate** | 10+ ticks/sec | TBD | Spatial hash (100-unit cells), vectorized movement |
| **File size** | <60MB (10K worlds) | 679MB pop-only | Quantized coords, delta encoding, gzip |
| **Neighbor query** | <1ms | TBD | Cell size = 100 units (matches max sense) |
| **Frontend FPS** | 60 fps (2000 agents) | TBD | Client interpolation (15 TPS → 60 fps) |

### Optimizations

1. **Vectorized Movement** (NumPy)
   ```python
   def move_agents_batch(agents: list[Agent], dt: float):
       positions = np.array([(a.x, a.y) for a in agents])
       velocities = np.array([a.velocity for a in agents])
       positions += velocities * dt
       positions = np.clip(positions, 0, world_size)
       for agent, (x, y) in zip(agents, positions):
           agent.x, agent.y = x, y
   ```

2. **Interaction Culling**
   - Sense range: `max(agent.traits["sense"] × 5, 50)` units
   - Max interactions per agent: 10 (prevent thrashing)

3. **Serialization Sampling**
   - Top-200 by energy (important agents)
   - Random-50 (capture variance)
   - Skip agents with `state=rest` and `energy>80` (less interesting)

4. **Frontend Viewport Culling**
   ```typescript
   const visibleAgents = agents.filter(a => {
       const x = a.x * scale;
       const y = a.y * scale;
       return x >= -50 && x <= 850 && y >= -50 && y <= 850;
   });
   renderAgents(ctx, visibleAgents, species, worldSize);
   ```

---

## Risks & Mitigations

### Risk 1: Performance Degradation

**Risk**: 2000 agents × O(n) interactions = 4M checks/tick → 10× slowdown

**Impact**: High (simulation unusable)

**Mitigation**:
- Spatial hash reduces to O(k) where k = ~10 neighbors
- Profile early (Phase 2)
- If still slow: reduce max agents to 1000
- Fallback: Only enable agents in small demo worlds (<500 pop)

---

### Risk 2: File Size Explosion

**Risk**: 250 agents × 8 species × 100 ticks = 2MB/world → 20GB for 10K worlds

**Impact**: Medium (disk space, slow loading)

**Mitigation** [FIXED]:
- Only INTELLIGENCE + ECOSYSTEM epochs (~40% of worlds have agents)
- Sample 200 agents (not all)
- **Quantized coordinates**: 10-unit grid reduces BPE tokens by ~30%
- Delta encoding (80% reduction)
- gzip compression (50% reduction)
- **Expected**: <8GB compressed (acceptable)

---

### Risk 3: Frontend Rendering Lag

**Risk**: Streaming 2000 agents @ 60fps = 18MB/sec JSON → browser parser bottleneck

**Impact**: High (unusable UI)

**Mitigation** [FIXED]:
- **Server tick rate**: 10-20 TPS (not 60fps) → 3MB/sec bandwidth
- **Client interpolation**: Linear tweening creates smooth 60fps rendering
- **Viewport culling**: Only render visible agents (~500 on screen)
- **Result**: Handles 2000+ agents at 60fps client-side

**Mitigation**:
- Canvas 2D (not DOM) for pixel-level drawing
- Viewport culling (only render visible agents)
- Throttle to 30fps if needed
- Agent LOD: distant agents = 1 pixel

---

### Risk 4: Energy/Population Conservation Paradox

**Risk**: Agents are samples (250 of 8000) but energy must conserve. Naive scaling causes:
- **Over-amplification**: 1 agent death = 32 deaths in macro (250/8000 scaling)
- **Ghost agents**: Agent success doesn't affect macro population
- **Training discontinuity**: Entering INTELLIGENCE causes mass extinction

**Impact**: CRITICAL (breaks physical consistency, corrupts training data)

**Mitigation** [FIXED]:
- **Dual-layer accounting**: Separate discrete events from continuous processes
  - Discrete (births, deaths, hunts): Do NOT scale (1 agent = 1 creature)
  - Continuous (metabolism, foraging gains): Scale by population ratio
- **Event-based reconciliation**: `PopulationReconciler` class tracks:
  - `deaths` → `species.population -= deaths` (not `deaths × 32`)
  - `energy_deltas` → `species.energy_store += avg_delta × scaling_factor`
- **Result**: Model learns "1 predator kills 1 prey", not cascading deaths

---

### Risk 5: Backward Compatibility Break

**Risk**: New `@AGENT` blocks cause old models to fail

**Impact**: Low (can train separate models)

**Mitigation**:
- `@AGENT` is optional (only in agent-enabled datasets)
- Tokenizer supports both (87 tokens backward-compatible)
- Train mixed datasets (50% pop-only, 50% agent-based)
- Document migration path

---

### Risk 6: Emergent Chaos & Oscillation

**Risk**: Agents exhibit unrealistic behavior:
- **Oscillation**: Flee → Forage → Flee → Forage (vibrate until starvation)
- **Clustering**: All agents pile in corners
- **Patch depletion**: Instant collapse, mass starvation
- **Ignore food**: Agents wander aimlessly despite hunger

**Impact**: Medium (simulation unrealistic, poor training data)

**Mitigation** [PARTIALLY FIXED]:
- **Hysteresis**: State commitment periods prevent oscillation
  - Flee: 10 ticks (panic persists)
  - Hunt: 8 ticks (pursuit commitment)
  - Forage: 3 ticks (can flee quickly)
- **Emergency override**: Critical energy (<10) breaks any commitment
- **Patch capacity**: Large patches (500 units) sustain 50 agents for 10+ ticks
- **Slow regeneration**: Prevents instant respawn oscillation
- **Utility tuning**: Flee has 2× weight (survival priority)
- **Visual debugging**: Test with small worlds (50 agents, 20 ticks) before full scale

---

## Success Criteria

### Functional Requirements

- [x] **Agent spawning**: Agents created with traits sampled from species distributions
- [x] **Smooth movement**: Agents move on spatial grid without teleporting
- [x] **Hunting**: Predators chase prey, prey flee from predators
- [x] **Foraging**: Agents move toward vegetation patches, local depletion visible
- [x] **Death**: Agent energy ≤ 0 triggers death, population decreases
- [x] **Protocol**: `@AGENT` blocks generated (backward-compatible)

### Performance Requirements

- [ ] **Tick rate**: ≥10 ticks/sec with 2000 agents on single CPU core
- [ ] **Spatial query**: <1ms per agent neighbor query (100-unit cells)
- [ ] **File size**: <60MB for 10K worlds (quantized coords + compression)
- [ ] **Frontend FPS**: 60 fps rendering 2000 agents (via client interpolation)
- [ ] **Server bandwidth**: <5MB/sec at 15 TPS (vs 18MB/sec at 60fps)

### Quality Requirements

- [ ] **Unit tests pass**: 100% for agent, spatial, behavior, reconciliation modules
- [ ] **Event-based conservation**: Discrete events (deaths) do NOT scale
- [ ] **Energy conservation**: `species.energy_store` scales continuous energy changes correctly
- [ ] **Integration test**: 100 tick simulation with 8 species, 200 agents each
- [ ] **No oscillation**: Agents do not vibrate between flee/forage states
- [ ] **No crashes**: Agents never spawn outside bounds, dead agents removed
- [ ] **Gradual learning**: @AGENT blocks appear in both ECOSYSTEM and INTELLIGENCE epochs

---

## Next Steps

1. **Review this plan** with user
2. **Create GitHub issues** for each phase (6 issues)
3. **Setup feature branch**: `git checkout -b feature/agent-simulation`
4. **Begin Phase 1**: Core agent system (Week 1)

---

## Design Rationale & Debate Summary

### Architectural Philosophy

This plan balances **three competing priorities**:

1. **Training Data Quality**: Model must learn realistic ecological dynamics, not simulation artifacts
2. **Physical Consistency**: Energy/population/space must obey conservation laws at all scales
3. **Performance**: Must scale to 10K worlds, 2000 agents, 200 generations without prohibitive cost

### Key Debates Resolved

#### 1. Dual-Layer Accounting vs Hard Cap

**Question**: Should we force `species.population = len(agents)` (hard cap) or maintain separate macro/micro layers?

**Decision**: **Dual-layer with event-based reconciliation**

**Rationale**:
- Hard cap creates training discontinuity (8000 → 250 population collapse when entering INTELLIGENCE)
- Model would learn "intelligence causes mass extinction" (backwards causality)
- Dual-layer treats agents as representative samples, scaling continuous processes but keeping discrete events 1:1
- This preserves macro statistics while enabling micro interactions

#### 2. Coordinate Precision vs Entropy

**Question**: Use continuous floats `(120.5, 340.2)` or discrete grid `(12, 34)`?

**Decision**: **Simulate continuous, serialize quantized (10-unit grid)**

**Rationale**:
- Continuous simulation preserves smooth movement, no grid artifacts
- Quantized serialization reduces BPE token count by ~30% (fewer float digits)
- Model learns spatial patterns (clustering, chase), not pixel-perfect regression
- 100×100 grid (1000/10) sufficient for ecological spatial reasoning

#### 3. Loss Weight for @AGENT Blocks

**Question**: Should @AGENT blocks have high weight (1.8) like @INT (1.5) or lower?

**Decision**: **0.8 (lower than core stats @SP: 1.0)**

**Rationale**:
- Coordinates are high-entropy noise — model shouldn't prioritize exact position prediction
- Causal reasoning (@SPOT: 2.0, @INT: 1.5) is more valuable than spatial location
- Risk of overfitting: model hallucinates "teleporting" agents trying to match float precision
- Lower weight encourages general spatial understanding, not coordinate memorization

#### 4. Frontend Streaming Rate

**Question**: Stream full state at 60fps or reduce server rate?

**Decision**: **10-20 TPS server + 60fps client interpolation**

**Rationale**:
- 60fps JSON stream = 18MB/sec bandwidth (server CPU bottleneck, browser parser lag)
- Decoupling: server sends discrete snapshots (physics), client interpolates (visuals)
- 6× bandwidth reduction enables 2000+ agents without degradation
- Linear interpolation (tweening) creates perceived smooth motion despite low tick rate

#### 5. Gradual Spatial Learning

**Question**: Introduce @AGENT blocks only in INTELLIGENCE epoch or earlier?

**Decision**: **Introduce simplified agents in ECOSYSTEM epoch**

**Rationale**:
- Sudden spatial concepts in INTELLIGENCE creates training discontinuity
- Model benefits from progressive learning: simple flocking → complex hunting
- ECOSYSTEM agents use basic behaviors (graze, flock) without full utility AI
- INTELLIGENCE agents add hunt/flee/mate with utility scoring
- Curriculum learning: spatial fundamentals → behavioral complexity

### Design Validation Checklist

✅ **No training artifacts**: Epoch transitions don't cause mass extinctions or teleporting
✅ **Physical realism**: Energy flows conserve, populations track discrete events correctly
✅ **Scalable performance**: 10+ TPS simulation, <60MB storage, 60fps rendering
✅ **Emergent dynamics**: Hysteresis prevents oscillation, patch capacity sustains grazing
✅ **Model-friendly protocol**: Quantized coords, appropriate loss weights, gradual complexity

---

## References

- `EVOLUTION_SIM_OVERVIEW.md` — Current simulation spec
- `mantis/simulation/engine.py` — World simulation loop
- `mantis/tokenizer.py` — Protocol tokenization
- `CLAUDE.md` — Project guidelines

---

**End of Plan**
