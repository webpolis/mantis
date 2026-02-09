# MANTIS Evolution Simulator — System Overview

## What This Is

An ecological simulation engine that generates synthetic training data for MANTIS. Instead of natural language, it produces structured traces of evolving ecosystems — species competing for energy, developing food webs, and eventually evolving intelligence. A trained model learns to simulate these universes autoregressively.

---

## Core Idea: Energy, Not Fitness

Everything derives from energy flow. Species don't have abstract "fitness" — they live or die based on calories in vs. calories out.

```
E_income  = energy from feeding (photosynthesis, grazing, hunting, scavenging)
E_cost    = basal_metabolism + movement + trait_maintenance + brain_tax
E_balance = E_income - E_cost

surplus  → fat storage or reproduction
deficit  → burn stores, then population decline
```

Intelligence is expensive (brain tax scales with cognitive tier). A species can't evolve `intel` without a calorie-dense food source, which naturally gates cognition to efficient omnivores/predators.

---

## Species Architecture

Each species has:

| Component      | Type                        | Purpose                                              |
| -------------- | --------------------------- | ---------------------------------------------------- |
| `traits`       | `{name: TraitDistribution}` | Population-level mean ± variance (not single values) |
| `fused_traits` | `{name: TraitDistribution}` | Composite traits from fusion rules                   |
| `body_plan`    | `BodyPlan`                  | Morphological constraints (what can/can't evolve)    |
| `diet`         | `DietVector`                | Normalized distribution over food sources            |
| `population`   | `int`                       | Head count                                           |
| `energy_store` | `float`                     | Fat reserves                                         |
| `locations`    | `set[str]`                  | Biomes where present                                 |

**Trait tiers** (35 traits across 5 tiers):

- T0 Physical (15): `size speed armor metab sense camo repro regen venom photosynth mouth endurance chem_digest toxin_resist toxin`
- T1 Behavioral (5): `social aggression curiosity patience nocturnal`
- T2 Cognitive (5): `intel memory learning planning deception`
- T3 Cultural (5): `language tooluse ritual teaching trade`
- T4 Abstract (5): `subconscious theory_of_mind creativity abstraction ethics`

Higher tiers require prerequisites from lower tiers (e.g., `intel` needs `curiosity≥3` + `social≥3`).

**Trait distributions** store `(mean, variance)` — predation success is computed from distribution overlap, not simple value comparison. Selection pressure narrows or widens variance.

**9 body plans** constrain what a species can evolve:

| Body Plan           | Metabolism | Key Constraints                            |
| ------------------- | ---------- | ------------------------------------------ |
| `sessile_autotroph` | 0.5x       | No speed/sense/intel. Photosynthesis only. |
| `mobile_autotroph`  | 0.8x       | Limited speed. No mouth/intel.             |
| `filter_feeder`     | 0.7x       | Limited speed. No aggression/intel.        |
| `grazer`            | 1.2x       | No photosynth. Intel gated by prereqs.     |
| `scavenger`         | 1.0x       | No photosynth/venom. Intel gated.          |
| `omnivore`          | 1.5x       | Broad access. No photosynth.               |
| `predator`          | 1.8x       | No photosynth. Armor capped at 4.          |
| `parasite`          | 0.6x       | Size/speed capped. No intel.               |
| `decomposer`        | 0.4x       | No speed/intel/social.                     |

Body plans shift when diet composition crosses thresholds (the "scavenger gateway"):

```
herbivore {plant:1.0}
  → scavenger {plant:0.7, detritus:0.3}
    → omnivore {plant:0.5, detritus:0.2, meat:0.3}
      → predator {meat:0.6, detritus:0.1, ...}
```

---

## Epochs

Simulations pass through 4 timescale epochs, triggered by complexity milestones:

| Epoch        | Tick =            | Trigger                     | Dynamics                               |
| ------------ | ----------------- | --------------------------- | -------------------------------------- |
| PRIMORDIAL   | ~1000 generations | Start                       | Chemical/solar energy, rapid evolution |
| CAMBRIAN     | ~10 generations   | First multicellular         | Body plan diversification, arms races  |
| ECOSYSTEM    | ~1 generation     | Stable food web (3+ levels) | Niche specialization, migration        |
| INTELLIGENCE | Sub-generational  | Spotlight score > threshold | Individual-level events, culture       |

Mutation rates, speciation probabilities, and serialization detail adjust per epoch.

---

## Output Protocol

The simulator outputs structured text consumed by `TextDataset` in `train.py`. Each generation produces layered blocks:

```
=EPOCH:2|TICK_SCALE:10gen|W42
@BIO|L0:shallows(veg=0.8,det=120)|L1:reef(veg=0.3)
@SP|S0|L0|plan=sessile_auto|pop=8200±900|diet={solar:1.0}
  T:size=2.1±0.4,photosynth=6.8±0.9
  E:in=4200,out=1800,store=6400|repro=r(rate=0.3)
@SP|S3|L1|plan=predator|pop=40±12|diet={S1:0.7,det:0.1}
  T:speed=6.2±0.9,intel=2.1±0.6
@INT|S3 hunt S1|success=0.62|S1:pop-=35|S3:E+=420
@EVT|S3|trait_emerge|social=1.5±0.8
---
```

For intelligent species, spotlight blocks add chain-of-thought:

```
@SPOT|W42G310|L1|S3
  CTX|S3:intel=5.5,social=6.2|Cmem:reef_collapse=taboo(0.6)
  ACTORS|H1:Elder(inf=7.2)|H3:Scout(inf=3.1)
  INTENT|H3->S3:report(new_territory)|reason=scouted_L2
  REACT|H1->H3:endorse(migrate)|reason=Cmem:reef_collapse
  RESOLVE|council(H1.inf+H3.report)|outcome=split_colony
  EFFECT|S3:loc+={L2:60}|H3:inf+=0.5
---
```

Keyframes dump full state every ~20 generations; intermediate ticks use delta encoding (`Δspeed=+0.3`).

When agent-based simulation is active, species blocks include `@AGENT` sub-blocks with individual organism positions and states:

```
@SP|S3|L1|plan=omnivore|pop=420±42(200 agents)|diet={S1:0.5,plant:0.3}
  T:speed=5.1±0.8,intel=4.2±0.6,social=5.8±0.7
  E:in=8400,out=6200,store=12000|repro=r(rate=0.4)
  @AGENT|count=420|sample=top200+rand50|quantize=10
    A0:(120,340,E=45,age=8,forage)
    A1:(130,350,E=52,age=10,hunt->A3)
    A5:(90,280,E=38,age=3,flee)
```

Delta ticks encode only changed agents (moved >5 units or energy changed >2), with `†` for deaths:

```
  @AGENT|Δpos|quantize=10
    A0:(125,345,E=43)
    A2:(140,360,E=48)
    A5:†
```

---

## Tokenization

`MANTISTokenizer` extends GPT-2 BPE (50,257 tokens) with 87 atomic protocol tokens → 50,345 total.

**Why**: GPT-2 BPE fractures protocol syntax. `theory_of_mind` → 6 BPE tokens. `@SPOT` → 3 tokens. Adding them as atomic tokens reduces per-frame token count by ~30-40%.

**What's added** (87 tokens):

- Layer markers: `=EPOCH @BIO @SP @INT @EVT @SPOT @AGENT`
- Spotlight logic: `CTX ACTORS INTENT REACT RESOLVE EFFECT`
- Mutations: `M+ M- Mpoint Mdrift Mleap Mfuse`
- Body plans: all 9 names
- Fractured traits: `venom camo metab repro theory_of_mind` etc. (32 total)
- Glue prefixes: `pop= plan= diet={ rate=` etc.

**What's NOT added**: tokens already single in GPT-2 (`---`, `±`, `->`, `speed`, `size`, `armor`, `intel`, `memory`, `language`, `trade`, `science`, `engineering`, `hunt`, `Mother`).

**Longest-match disambiguation**: `@SP` vs `@SPOT` — HuggingFace's trie tokenizer matches longest first, so `@SPOT` always wins when present.

**Per-token loss weights** use a state machine: layer markers (`@SP`, `@INT`, etc.) set the weight, subsequent tokens inherit it until the next marker.

| Marker   | Weight | Rationale              |
| -------- | ------ | ---------------------- |
| `---`    | 0.1    | Trivial separators     |
| `=EPOCH` | 0.5    | Metadata               |
| `@BIO`   | 0.5    | Slow-changing state    |
| `@SP`    | 1.0    | Core simulation        |
| `@INT`   | 1.5    | Interaction dynamics   |
| `@EVT`   | 1.5    | Rare important events  |
| `@SPOT`  | 2.0    | Intelligence reasoning |
| `@AGENT` | 0.8    | Agent spatial data     |

Pad tokens get weight 0.0.

---

## Training an Evolution Model

### Recommended approach (population-only)

```bash
# 1. Generate dataset (no agents)
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 \
    --output data/evo_train.txt --workers 8 --seed 42

# 2. Train (tiny model, single GPU)
python train.py --stage 1 data/evo_train.txt \
    --model-size tiny --seq-len 4096 --batch-size 8 \
    --gradient-accumulation-steps 4 --learning-rate 5e-4 \
    --warmup-steps 2000 --epochs 10 --mixed-precision --val-split 0.1
```

### Recommended approach (with agent simulation)

Agent blocks dramatically increase per-tick token volume. A single agent keyframe tick with 3 species can produce 6,000-14,000 tokens — exceeding `seq_len=4096` entirely. Longer sequences, smaller batches, and adjusted keyframe intervals are required.

```bash
# 1. Generate dataset (agents enabled, longer keyframe interval)
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 \
    --output data/evo_agents_train.txt --workers 8 --seed 42 \
    --enable-agents --agent-epoch INTELLIGENCE --keyframe-interval 40

# 2. Train (tiny model, 24GB GPU)
python train.py --stage 1 data/evo_agents_train.txt \
    --model-size tiny --seq-len 8192 --batch-size 4 \
    --gradient-accumulation-steps 8 --learning-rate 5e-4 \
    --warmup-steps 3000 --epochs 10 --mixed-precision \
    --gradient-checkpointing --val-split 0.1
```

### Critical training parameters

Parameters differ based on whether agent simulation is enabled:

**Population-only (no agents):**

| Parameter     | Value                          | Why                                                                  |
| ------------- | ------------------------------ | -------------------------------------------------------------------- |
| `--seq-len`   | 4096                           | Cultural memories reference events 50-150 gens back. 512 is useless. |
| `--stride`    | 2048                           | 50% overlap ensures no causal chain falls on a boundary.             |
| Warmup        | 2000 steps                     | MoE router needs stabilization time.                                 |
| Peak LR       | 5e-4 (tiny/small), 1e-4 (base) | Standard for MoE.                                                    |
| Min LR        | 1e-5                           | Never decay to zero — late data is the most complex.                 |
| Gradient clip | 1.0                            | MoE can produce gradient spikes.                                     |

**With agent simulation:**

| Parameter     | Value                          | Why                                                                       |
| ------------- | ------------------------------ | ------------------------------------------------------------------------- |
| `--seq-len`   | 8192 (min), 16384 (ideal)      | Agent keyframe ticks produce 6K-14K tokens. 4096 truncates mid-block.     |
| `--stride`    | 4096-6144                      | Agent coordinates have weak inter-tick causality; overlap less valuable.   |
| Warmup        | 3000 steps                     | Agent tokens increase vocabulary diversity; router needs more time.        |
| Peak LR       | 5e-4 (tiny/small), 1e-4 (base) | Same as population-only.                                                  |
| Min LR        | 1e-5                           | Same as population-only.                                                  |
| Gradient clip | 1.0                            | Same as population-only.                                                  |
| Keyframe interval | 40                         | Halves agent keyframe frequency — reduces token spikes by 2x.             |

**Batch size scaling for agent-enabled training (24GB GPU):**

| Model | `--seq-len` | `--batch-size` | `--gradient-accumulation-steps` | Effective batch |
| ----- | ----------- | -------------- | ------------------------------- | --------------- |
| Tiny  | 8192        | 4              | 8                               | 32              |
| Tiny  | 16384       | 2              | 16                              | 32              |
| Small | 8192        | 2              | 16                              | 32              |
| Small | 16384       | 1              | 32                              | 32              |

Use `--gradient-checkpointing` unconditionally with agent-enabled data.

### Token volume comparison

Approximate tokens per tick by epoch (agent-enabled):

| Epoch        | Delta tick   | Keyframe tick   | Notes                                    |
| ------------ | ------------ | --------------- | ---------------------------------------- |
| PRIMORDIAL   | 50-100       | 200-400         | No agents, same as population-only.      |
| CAMBRIAN     | 50-100       | 200-400         | No agents, same as population-only.      |
| ECOSYSTEM    | 800-2,500    | 8,000-14,000    | 3-5 species with agents (simple mode).   |
| INTELLIGENCE | 1,000-3,000  | 10,000-14,000   | 1-3 species with agents + spotlight.     |

A 100-gen agent-enabled world produces ~80,000-260,000 tokens (10-20x more than population-only). This has direct implications for curriculum mixing — see below.

### World boundary handling

Worlds are independent simulations. **Never pack tokens from different worlds into the same sequence.** Pad to `seq_len` at each world boundary (EOS).

Padding waste varies by mode:
- **Population-only**: <10% waste at `seq_len=4096` (worlds produce ~8K-15K tokens).
- **Agent-enabled**: <1% waste at `seq_len=8192` (worlds produce ~80K-260K tokens). The bottleneck shifts from padding to compute-per-world.

### Training curriculum

Don't train sequentially (biology → ecosystems → intelligence) — causes catastrophic forgetting. Instead, mix gradually.

**Important**: Mix by **token count**, not world count. Agent-enabled INTELLIGENCE worlds produce 10-20x more tokens per world than PRIMORDIAL worlds. Mixing by world count causes INTELLIGENCE tokens to dominate (~90% of training), overfitting the model to coordinate prediction at the expense of core ecological dynamics.

| Progress | Bio tokens | Ecosystem tokens | Intelligence tokens |
| -------- | ---------- | ---------------- | ------------------- |
| 0-20%    | 100%       | 0%               | 0%                  |
| 20-40%   | 50%        | 50%              | 0%                  |
| 40-60%   | 25%        | 35%              | 40%                 |
| 60-100%  | 20%        | 30%              | 50%                 |

Generate three separate files partitioned by max epoch reached and mix with a curriculum sampler that samples **by token budget per batch**, not by file line count.

### Evaluation metrics

Perplexity alone is insufficient. Track:

**Core metrics (all modes):**

| Metric              | Target    | Measures                                                  |
| ------------------- | --------- | --------------------------------------------------------- |
| Parse rate          | >99.9%    | Valid protocol syntax                                     |
| Energy conservation | <5% error | E_in - E_out ≈ ΔE_store                                   |
| Population dynamics | r>0.8     | Realistic predator-prey oscillations                      |
| Causal consistency  | >95%      | Extinct species stay extinct, taboos referenced correctly |
| Trait validity      | >99%      | Body plan constraints respected                           |
| Spotlight coherence | >90%      | EFFECT follows from INTENT + RESOLVE                      |

**Agent-specific metrics (when `--enable-agents`):**

| Metric                  | Target       | Measures                                                           |
| ----------------------- | ------------ | ------------------------------------------------------------------ |
| Spatial coherence       | <20 units/tick | Agents don't teleport between delta frames                       |
| Behavioral consistency  | >85%         | Foragers near vegetation, fleers move away from predators          |
| Agent-population sync   | <10% drift   | Agent births/deaths track macro population changes                 |
| Dead-agent permanence   | 100%         | `†`-marked agents never reappear in subsequent ticks               |
| State commitment        | >80%         | Hysteresis periods respected (no forage→hunt→forage in 3 ticks)   |

### Model size guide

| Size        | Params    | Strategy      | Use case              |
| ----------- | --------- | ------------- | --------------------- |
| Micro (10M) | Dense     | Single GPU    | Pipeline sanity check |
| Tiny (100M) | 4 experts | Single GPU    | Development iteration |
| Small (1B)  | 4 experts | DDP or ZeRO-1 | Experimentation       |
| Base (12B)  | 8 experts | ZeRO-2/3      | Production target     |

### What the trained model can do

Given a partial simulation trace, the model autoregressively continues it — predicting population changes, trait mutations, energy flows, species interactions, body plan transitions, individual agent behaviors and positions, and (for intelligent species) spotlight scenes with chain-of-thought reasoning. A game engine can sample from this model to run an evolving universe in real time.
