# Emergent Communication System — Design Plan

## Summary

Extend the evolution simulator to allow species to develop communication, ranging from primitive alarm calls (T1) to coordinated pack tactics (T3+). Communication is modeled as **sparse, event-driven signals** folded into the existing `@INT` interaction protocol — no new top-level markers, no continuous fields, no free-text content.

**Token cost**: 9 new vocabulary entries, ~3% token increase per world (~1,200 tokens on 42K median).

---

## Motivation

### The Missing Evolutionary Bridge

The trait system jumps from T1 `social` (passive stat boost) to T3 `language` (complex grammar) with no intermediate communication mechanism. In real biology, communication evolves gradually:

- T0-T1: Chemical signals, alarm calls, color displays (ants, frogs, birds)
- T2: Referential signaling, directed intent (wolves, dolphins)
- T3+: Grammar, abstract reference, teaching (primates, humans)

Without signals, the model sees "telepathic" coordination — 50 agents suddenly flee with no visible cause. With signals, the causal chain is explicit: `SIG_ALERT` emitted → nearby agents shift to flee. Even a 100M parameter model learns `Token A → Action B` faster than `Hidden State → Action B`.

### Signals as Causal Bridges

Without communication:
```
@AGENT Δ 100
  N A1 130 350 48 hunt->A3
  N A2 135 348 50 hunt->A3
```
The model must infer coordination from coincident motion. Ambiguous — planned or accidental?

With communication:
```
@INT A1 A2 SIG SIG_HUNT
@INT A1+A2 hunt A3 82
```
Explicit causal link. The model learns that `SIG_HUNT` increases coordinated hunt success.

### Entropy Reduction

Signals cost ~3% more tokens but make subsequent motion predictions much easier. After `SIG_RALLY` at position (100,100), the model knows nearby agents will converge — their positions become low-entropy. The signal "pays for itself" in perplexity reduction on the ~50 subsequent coordinate tokens.

### Deception Becomes Meaningful

Currently the `deception` trait is just a dodge roll modifier. With signals, deception = emitting `SIG_ALERT` when there's no predator, causing rivals to flee a food source. This creates training data about Theory of Mind before T4 `theory_of_mind` even evolves.

---

## Design Decisions

### What We Include

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Signal format | Folded into `@INT` with `SIG` verb | Reuses existing infrastructure, loss weight (1.5), and parsing |
| Signal content | Fixed vocabulary of 8 content tokens | No BPE fragmentation, sufficient for all key behaviors |
| Signal triggers | Event-driven with cooldowns | Prevents token flooding; signals fire on state transitions only |
| Signal visibility | Notable agents only emit @INT SIG lines | Grid cells show behavioral effect (shift to flee), not individual signals |
| Deception marking | None — model infers reliability from context | More interesting training data than explicit true/false labels |
| Loss weight | 1.5 (same as @INT) | Causal communication events deserve high attention |

### What We Drop

| Dropped Feature | Why |
|-----------------|-----|
| T0 continuous signals (pheromone trails, bioluminescence) | Persistent gradient fields are expensive to serialize and not event-driven |
| New top-level `@SIG` marker | Reuses existing `@INT` — avoids new token, keeps protocol simple |
| `SIG_FALSE` / `SIG_TRUE` truth markers | Model should learn trust from context (sender traits, environmental mismatch) |
| Free-text signal content | Fixed vocabulary prevents token explosion and ensures learnability |
| Continuous visual displays | Same as pheromone trails — serialize as events, not fields |

---

## Token Definitions

9 new atomic tokens added to `MANTISTokenizer` (50,345 → 50,354 total vocabulary):

| Token | Type | BPE Savings | Description |
|-------|------|-------------|-------------|
| `SIG` | Verb | 1 BPE → 1 | Universal action verb for communication |
| `SIG_ALERT` | Content | 4 BPE → 1 | Danger/predator nearby — high urgency |
| `SIG_FOOD` | Content | 4 BPE → 1 | Resource location found |
| `SIG_MATE` | Content | 4 BPE → 1 | Mating call / fitness display |
| `SIG_HUNT` | Content | 4 BPE → 1 | Coordinate pack attack |
| `SIG_HELP` | Content | 4 BPE → 1 | Distress call |
| `SIG_RALLY` | Content | 4 BPE → 1 | Follow me / regroup / migrate |
| `SIG_CLAIM` | Content | 4 BPE → 1 | Territory marking / resource ownership |
| `SIG_WARN` | Content | 4 BPE → 1 | Stay away / threat display to rivals |

---

## Protocol Format

Signals are `@INT` interactions with the `SIG` verb. Supports both v1 and v2 notation.

### v1 (Verbose, pipe-delimited)

```
@INT|S3:A10 *|SIG|SIG_ALERT|rx=4
@INT|S3:A5 S3:A8|SIG|SIG_HUNT
```

### v2 (Compact, space-separated)

```
@INT A10 * SIG SIG_ALERT rx=4
@INT A5 A8 SIG SIG_HUNT
```

**Fields:**
- Source: Agent ID (e.g., `A10`) — always a notable agent
- Target: Agent ID (e.g., `A8`) for directed, or `*` for broadcast
- Verb: `SIG`
- Content: One of the 8 `SIG_*` content tokens
- Outcome (optional): `rx=N` — number of agents that received/reacted

### Integration With @AGENT Blocks

Signals only appear as explicit `@INT` lines for individually-tracked notable agents (top 5 by energy). Grid cells show the **effect** of signals through aggregate behavioral shifts:

```
@INT A10 * SIG SIG_ALERT rx=4
@AGENT Δ 100
  G 1,3 15 42 f:3 h:0 fl:12      ← cell shifted from forage to flee
  N A10 130 350 45 forage          ← sender (may be deceptive)
  N A3 140 360 38 flee             ← receiver reacted
```

The model learns: "SIG_ALERT causes grid cells to shift toward flee behavior" — a causal pattern linking micro-level signals to macro-level population dynamics.

### Signal-to-Spotlight Bridge

`@INT SIG` events are **tactical** (real-time, agent-level). `@SPOT INTENT/REACT` are **strategic** (narrative, species-level). Many `@INT SIG` events can culminate in a `@SPOT` decision:

```
# Tactical signals accumulate...
@INT A10 * SIG SIG_ALERT rx=8
@INT A3 * SIG SIG_ALERT rx=6
@INT A7 A1 SIG SIG_RALLY

# ...leading to a strategic spotlight decision
@SPOT|W42G180|L0|S3
  CTX|S3:intel=5.5,social=6.2|Cmem:predator_L0=taboo(0.7)
  ACTORS|H1:Elder(inf=7.2)|H7:Scout(inf=4.1)
  INTENT|H7->S3:report(repeated_attacks)|reason=3_alerts_in_10_ticks
  REACT|H1->H7:endorse(migrate)|reason=Cmem:predator_L0
  RESOLVE|council(H1.inf+H7.report)|outcome=migrate_L1
  EFFECT|S3:loc+={L1:80}|H7:inf+=0.5
```

No format changes needed for Spotlight blocks — the existing INTENT/REACT system already handles communication at the narrative level.

---

## Trait Gating

Signals are gated by existing traits. Higher-complexity signals require more evolved social and cognitive capabilities.

| Signal | Unlock Requirement | Biological Analog |
|--------|-------------------|-------------------|
| `SIG_MATE` | Base (all agents) | Universal mating calls |
| `SIG_WARN` | `aggression ≥ 2` | Threat displays, posturing |
| `SIG_ALERT` | `sense ≥ 2` | Alarm calls (vervet monkeys, prairie dogs) |
| `SIG_FOOD` | `social ≥ 2` | Altruistic food sharing (honeybees waggle dance) |
| `SIG_CLAIM` | `social ≥ 3` + `aggression ≥ 3` | Territorial scent marking (wolves, big cats) |
| `SIG_HELP` | `social ≥ 4` | Distress calls (dolphins, elephants) |
| `SIG_HUNT` | `social ≥ 5` + `planning ≥ 1` | Pack hunting coordination (wolves, orcas) |
| `SIG_RALLY` | `social ≥ 5` + `intel ≥ 3` | Flock/herd leadership, migration coordination |

### Trait Interactions

- **`sense`**: Determines reception range = `sense × 10` grid units
- **`deception ≥ 3`**: Allows emitting signals that don't match environmental state (false `SIG_ALERT` to scare rivals from food)
- **`intel`**: Reduces signal cooldown by `intel × 10%` (smarter agents communicate more efficiently)
- **`language ≥ 4`**: Enables directed signals (specific target agent ID instead of broadcast `*`)

### Evolution Path

```
T0 Physical:   sense ≥ 2 → SIG_ALERT (detect and warn)
T1 Behavioral:  social ≥ 2 → SIG_FOOD (share information)
                social ≥ 3 + aggr ≥ 3 → SIG_CLAIM (territory)
                social ≥ 4 → SIG_HELP (altruism)
T2 Cognitive:   social ≥ 5 + planning ≥ 1 → SIG_HUNT (coordinate)
                social ≥ 5 + intel ≥ 3 → SIG_RALLY (lead)
                deception ≥ 3 → false signals (exploit trust)
T3 Cultural:    language ≥ 4 → directed signals (specific recipients)
```

---

## Emission Rules

### Trigger Conditions

Signals fire on **state transitions** (rising edge), not continuously:

| Signal | Trigger |
|--------|---------|
| `SIG_ALERT` | Agent detects predator within sense range AND was not fleeing last tick |
| `SIG_FOOD` | Agent finds vegetation patch with density > 0.5 AND was searching |
| `SIG_MATE` | Agent energy > 70 AND no conspecific within 20 units AND `repro` season |
| `SIG_HUNT` | Agent selects hunt state AND has valid prey target AND nearby pack member |
| `SIG_HELP` | Agent energy < 15 AND was recently attacked |
| `SIG_RALLY` | Agent changes biome or moves > 50 units in one tick (migration) |
| `SIG_CLAIM` | Agent enters contested territory (2+ species present in grid cell) |
| `SIG_WARN` | Non-conspecific agent enters within `aggression × 5` units |

### Cooldowns

- **Global cooldown**: 10 ticks per agent between any signal emissions
- **`SIG_ALERT` override**: 5 tick cooldown (urgency)
- **`intel` reduction**: Cooldown reduced by `intel × 10%` (minimum 3 ticks)

### Energy Cost

- **Broadcast (`*`)**: 0.5 energy (metabolically expensive — shouting)
- **Directed (specific target)**: 0.1 energy (cheap — whispering/signaling)

### Emission Rate Estimate

For a typical ECOSYSTEM world (5 species, 1 with social ≥ 2, ~50 signaling agents, 200 ticks):
- An agent signals once every ~50 ticks on average (cooldown + trigger rarity)
- 50 agents × (200 / 50) = **~200 signal events per world**
- Token cost: 200 signals × 6 tokens/signal = **~1,200 tokens**
- Impact on 42K median world: **+2.8%**

---

## Reception Mechanics

### Range

- Reception radius = `sense × 10` grid units
- Example: `sense = 5` → 50 unit range (covers 3×3 grid cells at cell_size=100)

### Distance Decay (Broadcast Only)

Broadcast signals have probability of failure increasing linearly with distance:

```
P(receive) = 1.0 - (distance / max_range)
```

Agents at the edge of range have ~0% chance; agents nearby have ~100%.

### Biome Blocking

- Acoustic signals (`SIG_ALERT`, `SIG_MATE`, `SIG_RALLY`, `SIG_HELP`) do not cross biome boundaries
- Visual signals (`SIG_WARN`, `SIG_CLAIM`) cross biomes if within range
- All signals propagate within same biome

### Receiver Reaction

Receiving a signal triggers a behavior utility modifier in the receiver's next tick:

| Received Signal | Behavior Modifier |
|----------------|-------------------|
| `SIG_ALERT` | `flee` utility += 3.0 (strong override) |
| `SIG_FOOD` | `forage` utility += 1.5, steer toward sender position |
| `SIG_MATE` | `mate` utility += 2.0, steer toward sender |
| `SIG_HUNT` | `hunt` utility += 2.0, adopt sender's target |
| `SIG_HELP` | Move toward sender (if `social ≥ 3`) |
| `SIG_RALLY` | `flock` utility += 2.5, steer toward sender |
| `SIG_CLAIM` | `flee` utility += 1.0 if weaker, `aggression` check if stronger |
| `SIG_WARN` | `flee` utility += 1.5 if same species, ignored if different |

---

## Deception

### Mechanism

There is **no** `SIG_FALSE` or `SIG_TRUE` token. Deception is emergent.

An agent with `deception ≥ 3` can emit signals that don't match environmental state. The simulation engine checks:

```python
if agent.traits['deception'] >= 3:
    # Can emit any signal regardless of actual state
    # e.g., SIG_ALERT when no predator nearby
    # e.g., SIG_FOOD when no food nearby
    pass
else:
    # Can only emit signals matching actual state
    # SIG_ALERT requires actual predator in sense range
    # SIG_FOOD requires actual vegetation patch nearby
    pass
```

### Training Data Pattern

Deceptive signals appear identical to honest signals in the protocol:

```
# Honest signal — predator S1 is actually nearby
@INT A10 * SIG SIG_ALERT rx=4
@SP|S1|L0|plan=predator|pop=35      ← predator exists in same biome

# Deceptive signal — no predator present
@INT A12 * SIG SIG_ALERT rx=3
                                      ← no predator @SP in this biome
```

The model must learn to correlate signal reliability with:
- Sender's species traits (high `deception` = less reliable)
- Environmental context (is there actually a predator in the biome?)
- Historical patterns (does this agent frequently emit false alerts?)

This creates training data about **Theory of Mind** — inferring the reliability of another agent's claims from indirect evidence.

---

## Loss Weight

Signal interactions inherit the standard `@INT` weight of **1.5**.

| Token Context | Weight | Rationale |
|--------------|--------|-----------|
| `SIG` verb | 1.5 | Causal event — model must learn signal → reaction chains |
| `SIG_*` content | 1.5 | Discriminative feature for predicting population outcomes |
| `rx=N` outcome | 1.5 | Signal reach affects magnitude of behavioral shift |
| Grid behavioral shift caused by signal | 0.8 | Inherited from `@AGENT` weight — spatial effect |

---

## Token Impact Analysis

### Per-World Estimates (v2 compact, agent-enabled)

| Metric | Without Signals | With Signals | Delta |
|--------|----------------|--------------|-------|
| Median tokens/world | 42,000 | 43,200 | +2.8% |
| p95 tokens/world | 180,000 | 185,400 | +3.0% |
| Signal events/world | 0 | ~200 | — |
| New vocab entries | 0 | 9 | 50,345 → 50,354 |

### Dataset-Level Impact (10K worlds)

- Additional tokens: ~12M (200 signals × 6 tokens × 10K worlds)
- Additional storage (v2 compact): ~48MB uncompressed, ~20MB gzipped
- Negligible impact on training time (< 3% more tokens to process)

### Signal Distribution Across Epochs

| Epoch | Signals/World | Notes |
|-------|--------------|-------|
| PRIMORDIAL | 0 | No agents, no signals |
| CAMBRIAN | 0 | No agents, no signals |
| ECOSYSTEM | ~50-100 | Basic signals only (ALERT, FOOD, WARN) — social ≥ 2 species |
| INTELLIGENCE | ~100-300 | Full signal vocabulary — T2+ species with pack tactics |

---

## Emergent Behaviors Enabled

### 1. Coordinated Pack Hunting
```
@INT A5 A8 SIG SIG_HUNT          ← alpha signals pack member
@INT A5 A9 SIG SIG_HUNT          ← signals another
@INT A5+A8+A9 hunt A3 92         ← coordinated hunt, higher success
```
Prediction: species with `social ≥ 5 + planning` achieve 30-50% higher hunt success rates vs solo hunters.

### 2. Alarm Cascades
```
@INT A10 * SIG SIG_ALERT rx=4    ← scout spots predator
@AGENT Δ 100
  G 1,3 15 42 f:3 h:0 fl:12     ← entire grid cell shifts to flee
```
Prediction: species with alarm calling suffer 40-60% fewer predation losses.

### 3. Deceptive Exploitation
```
@INT A12 * SIG SIG_ALERT rx=3    ← deceptive alarm (no real predator)
@AGENT Δ 100
  G 2,4 8 38 f:1 h:0 fl:7       ← rivals flee food source
  N A12 210 440 52 forage         ← decepter takes the food
```
Prediction: high-deception species gain energy advantage but lose group trust over time.

### 4. Territory Establishment
```
@INT A7 * SIG SIG_CLAIM           ← mark territory
@INT A15 * SIG SIG_WARN           ← rival contests
@INT A7 A15 compete 65            ← resolved by dominance contest
```
Prediction: territorial signaling reduces physical combat frequency by 20-30%, saving energy for both parties.

### 5. Migration Leadership
```
@INT A1 * SIG SIG_RALLY rx=12    ← elder leads migration
@AGENT Δ 100
  G 3,3 0 0                       ← origin cell empties
  G 7,7 14 55 fk:14               ← destination cell fills with flocking
```
Prediction: species with rally capability migrate cohesively instead of fragmenting.

### 6. Mating Displays (The Lek Effect)
```
@INT A20 * SIG SIG_MATE           ← costly broadcast (0.5 energy)
@INT A21 * SIG SIG_MATE           ← competing male
@INT A35 A20 mate                  ← female selects stronger signal
```
Prediction: `SIG_MATE` energy cost acts as honest signal of fitness — higher energy agents can signal more, attracting more mates.

---

## Implementation Scope

### Files to Modify

```
mantis/tokenizer.py                  # Add 9 new tokens + update PROTOCOL_TOKENS
mantis/simulation/behavior.py        # Add signal emission logic to utility system
mantis/simulation/agent.py           # Add signal_cooldown field to Agent
mantis/simulation/engine.py          # Add _resolve_signals() step in tick loop
mantis/simulation/serializer.py      # Serialize @INT SIG lines for notable agents
mantis/simulation/constants.py       # Add SIGNAL_CONFIG (cooldowns, costs, gating)
scripts/gen_evo_dataset.py           # No changes needed (signals auto-generate when agents active)
```

### New Data Structures

```python
# In agent.py — extend Agent dataclass
@dataclass
class Agent:
    # ... existing fields ...
    signal_cooldown: int = 0        # Ticks until next signal allowed
    received_signals: list = None   # Buffer of signals heard this tick

# In constants.py — signal configuration
SIGNAL_CONFIG = {
    'SIG_ALERT':  {'gate': {'sense': 2}, 'cooldown': 5, 'energy': 0.5},
    'SIG_FOOD':   {'gate': {'social': 2}, 'cooldown': 10, 'energy': 0.5},
    'SIG_MATE':   {'gate': {}, 'cooldown': 10, 'energy': 0.5},
    'SIG_HUNT':   {'gate': {'social': 5, 'planning': 1}, 'cooldown': 10, 'energy': 0.1},
    'SIG_HELP':   {'gate': {'social': 4}, 'cooldown': 10, 'energy': 0.5},
    'SIG_RALLY':  {'gate': {'social': 5, 'intel': 3}, 'cooldown': 10, 'energy': 0.5},
    'SIG_CLAIM':  {'gate': {'social': 3, 'aggression': 3}, 'cooldown': 10, 'energy': 0.5},
    'SIG_WARN':   {'gate': {'aggression': 2}, 'cooldown': 10, 'energy': 0.1},
}
```

### Simulation Loop Integration

The signal resolution step fits between behavior selection and movement:

```
AgentManager.step():
  1. _update_behaviors()       # Utility scoring + action selection
  2. _resolve_signals()        # NEW — emit signals, propagate to receivers
  3. _apply_signal_effects()   # NEW — modify receiver utilities based on heard signals
  4. _move_agents()            # Steering + position update
  5. _resolve_interactions()   # Hunt, flee, forage outcomes
  6. _prune_dead()             # Remove energy ≤ 0 agents
```

### Tokenizer Changes

```python
# In tokenizer.py — add to existing token lists
SIGNAL_TOKENS = [
    "SIG",          # Verb (1 BPE → 1)
    "SIG_ALERT",    # (4 BPE → 1)
    "SIG_FOOD",     # (4 BPE → 1)
    "SIG_MATE",     # (4 BPE → 1)
    "SIG_HUNT",     # (4 BPE → 1)
    "SIG_HELP",     # (4 BPE → 1)
    "SIG_RALLY",    # (4 BPE → 1)
    "SIG_CLAIM",    # (4 BPE → 1)
    "SIG_WARN",     # (4 BPE → 1)
]

PROTOCOL_TOKENS = (
    LAYER_MARKERS + SPOTLIGHT_TOKENS + MUTATION_TOKENS +
    BODY_PLAN_TOKENS + TRAIT_TOKENS + MEME_TOKENS +
    ROLE_TOKENS + INTERACTION_TOKENS + SYMBOL_TOKENS +
    GLUE_TOKENS + SIGNAL_TOKENS  # NEW
)
```

---

## Validation Criteria

### Functional

- [ ] Signals only emitted by agents meeting trait gate requirements
- [ ] Cooldowns prevent signal flooding (max 1 signal per 5-10 ticks per agent)
- [ ] Broadcast signals respect range = `sense × 10` with distance decay
- [ ] Acoustic signals blocked across biome boundaries
- [ ] Deceptive signals only emitted by agents with `deception ≥ 3`
- [ ] Receivers modify behavior utilities based on signal content
- [ ] Grid cells show behavioral shifts caused by signals
- [ ] Signal events appear as @INT lines in serialized output

### Token Efficiency

- [ ] Total signal tokens per world < 5% of base token count
- [ ] No signal content token appears < 50 times in 10K world dataset (sufficient for learning)
- [ ] Signal-heavy worlds (INTELLIGENCE epoch) stay within p95 token budget

### Emergent Behavior

- [ ] Species with alarm calling show measurably lower predation losses
- [ ] Pack hunting species show higher hunt success than solo hunters
- [ ] Deceptive species gain short-term energy advantage
- [ ] Territorial signaling reduces physical combat frequency
- [ ] Rally signals produce cohesive group movement (grid cell convergence)

### Model Training

- [ ] Signal tokens correctly tokenized as single atoms (not BPE-fractured)
- [ ] Loss weights propagate correctly (SIG lines inherit @INT weight 1.5)
- [ ] Trained model can generate valid @INT SIG lines in correct contexts
- [ ] Model predicts behavioral shifts in grid cells following signal events

---

## Debate Record

This design was produced through structured adversarial debate (2026-02-10):

- **Pro (Gemini)**: Argued signals create richer causal training data, fill the T1→T3 evolutionary gap, enable meaningful deception, and reduce prediction entropy despite small token cost.
- **Skeptic (Claude)**: Argued against token bloat, model capacity limits, vocabulary explosion, duplication with Spotlight system, and the compression claim being backwards.

**Key concessions:**
- Pro conceded: Drop continuous pheromone fields, fold into @INT, use fixed vocabulary
- Skeptic conceded: Signals as causal bridges simplify learning, entropy reduction argument is valid, event-driven emission keeps token cost at ~3%
