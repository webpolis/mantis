# MANTIS Evolution Simulator — System Overview

## What This Is

An ecological simulation engine that generates synthetic training data for MANTIS. Instead of natural language, it produces structured traces of evolving ecosystems — species competing for energy, developing food webs, and eventually evolving intelligence. A trained model learns to simulate these universes autoregressively.

---

## Core Idea: Energy, Not Fitness

Everything derives from energy flow. Species don't have abstract "fitness" — they live or die based on calories in vs. calories out.

```
E_income  = energy from feeding (photosynthesis, grazing, hunting, scavenging)
            × digestive_affinity (body plan / diet match)
            × nutrient_factor (min(N, P) for producers)
E_cost    = basal_metabolism + movement + trait_maintenance + brain_tax
E_balance = E_income - E_cost

surplus  → fat storage or reproduction
deficit  → burn stores, then population decline
```

Energy income is modulated by two factors: **digestive affinity** (a grazer eating meat gets only 40% efficiency) and **nutrient availability** (producers limited by `min(nitrogen, phosphorus)` in their biome). Nutrients cycle back through decomposer activity on detritus.

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
grazer {plant:1.0}
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

Mutation rates are normalized **per generation** (raw `mutation_rate_mult / tick_scale`), so PRIMORDIAL (1000 gen/tick) and INTELLIGENCE (0.1 gen/tick) evolve at comparable per-generation rates despite vastly different tick scales. Speciation probabilities and serialization detail also adjust per epoch.

---

## Output Protocol

The simulator outputs structured text consumed by `TextDataset` in `train.py`. Two formats are supported:

### v1 format (pipe-delimited, default)

```
=EPOCH:2|TICK_SCALE:10gen|W42
@EVT|WORLD|catastrophe|volcanic_winter|dur=8
@BIO|L0:shallows(veg=0.8,det=120,N=0.6,P=0.4)|L1:reef(veg=0.3,det=5,N=0.3,P=0.2)
@SP|S0|L0|plan=sessile_auto|pop=8200±900|diet={solar:1.0}
  T:size=2.1±0.4,photosynth=6.8±0.9
  E:in=4200,out=1800,store=6400|repro=r(rate=0.3)
@SP|S3|L1|plan=predator|pop=40±12|diet={S1:0.7,det:0.1}
  T:speed=6.2±0.9,intel=2.1±0.6
@INT|S3 hunt S1|success=0.62|S1:pop-=35|S3:E+=420
@EVT|S3|trait_emerge|social=1.5±0.8
@EVT|S1|disease|plague|pop-=200
---
```

### v2 format (compact, `--compact` flag)

Int-scaled values, space-separated, ~40% fewer tokens:

```
=EPOCH 2 10 W42
@BIO L0 shallows 80 120 N6 P4 L1 reef 30 0 N3 P2
@SP S0 L0 sessile_auto 8200 D sol 100
  T size 21±4 photosynth 68±9
  E 4200 1800 6400 r 3
@SP S3 L1 predator 40 D S1 70 det 10
  T speed 62±9 intel 21±6
@INT S3 hunt S1 62 S1 p-35
@EVT WORLD catastrophe volcanic_winter dur 8
@EVT S3 M+ social 15±8
@EVT S1 disease plague p-200
---
```

v2 scaling rules: traits ×10, diet proportions ×100, vegetation ×100, success ×100, repro rate ×10, nutrients (N/P) ×10. Diet abbreviations: `detritus`→`det`, `plant`→`plt`, `solar`→`sol`.

### Spotlight blocks

For intelligent species (v1 shown; v2 compresses headers but keeps narrative lines unchanged):

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

Keyframes dump full state every ~20 generations; intermediate ticks use delta encoding (`Δspeed=+0.3` in v1, `Δspeed +3` in v2).

### Agent blocks

When agent-based simulation is active, species blocks include `@AGENT` sub-blocks using grid+notable hybrid format. Agents are aggregated into 100-unit spatial grid cells with behavior distributions, plus a small set of individually-tracked "notable" agents (top 5 by energy).

v1 keyframe:
```
  @AGENT|count=420|mode=grid+5|cell=100
    G(1,3):n=17,E=45|f:12,h:3,fl:2
    G(2,3):n=8,E=38|f:5,h:2,r:1
    N:A1:(130,350,E=52,age=10,hunt->A3)
    N:A7:(400,100,E=89,age=15,forage)
```

v2 keyframe:
```
  @AGENT 420 grid+5 100
   G 1,3 17 45 f:12 h:3 fl:2
   G 2,3 8 38 f:5 h:2 r:1
   N A1 130 350 52 10 hunt->A3
   N A7 400 100 89 15 forage
```

Delta ticks encode only changed grid cells (count changed >2, avg energy changed >5, or dominant behavior changed) and changed/dead notables:

v1 delta:
```
  @AGENT|Δpos|cell=100
    G(1,3):n=15,E=42|f:10,h:3,fl:2
    N:A1:(140,360,E=48)
    A7:†
```

v2 delta:
```
  @AGENT Δ 100
   G 1,3 15 42 f:10 h:3 fl:2
   N A1 140 360 48
   A7 †
```

Behavior abbreviations: `r`=rest, `f`=forage, `h`=hunt, `m`=mate, `fl`=flee, `fk`=flock.

---

## Tokenization

`MANTISTokenizer` is a custom trie-based longest-match tokenizer with 282 domain tokens padded to 512 for tensor core alignment. No GPT-2 / BPE / `transformers` dependency.

**Why custom over GPT-2 BPE**: The simulation format is ~98% structured protocol — not English. GPT-2's 50,257-token vocabulary wastes 1.5+ GB VRAM on dead embedding weights (at d=2048), runs softmax over 50K logits when only ~300 matter, and splits numbers inconsistently (`"2429"` → `["24","29"]` but `"2430"` differently). The custom tokenizer fixes all three:

| Metric              | GPT-2 BPE (old) | Custom trie (new) |
| ------------------- | ---------------- | ----------------- |
| Vocab size          | 50,345           | 512               |
| Embedding params    | 103M (d=2048)    | 1.05M             |
| Embedding VRAM      | 1.65 GB          | 16.8 MB           |
| Softmax width       | 50,345           | 512               |
| `"2429"` encoding   | 2 tokens (varies) | 4 tokens (always `2` `4` `2` `9`) |
| `"@SPOT"` encoding  | 3 tokens          | 1 token           |
| Vocab utilization   | ~2-5%            | ~60-90%           |

**Vocabulary (282 real + 230 reserved = 512)**:

- Special (4): `<pad>` `<eos>` `<bos>` `<unk>`
- Digits (10): `0`–`9` (numbers always digit-by-digit)
- Whitespace (3): space, newline, 2-space indent
- Protocol markers (8): `=EPOCH` `@BIO` `@SP` `@INT` `@EVT` `@SPOT` `@AGENT` `---`
- Spotlight logic (6): `CTX` `ACTORS` `INTENT` `REACT` `RESOLVE` `EFFECT`
- Mutations (6): `M+` `M-` `Mpoint` `Mdrift` `Mleap` `Mfuse`
- Body plans (9): all 9 names (`sessile_autotroph` .. `decomposer`)
- Traits (45): 35 base + 10 fused
- Biomes (15): all 15 names
- Interactions (7): `hunt` `graze` `compete` `scavenge` `parasitize` `pollinate` `symbiosis`
- Spotlight narrative (~50): roles, actions, reactions, resolutions, outcomes, reasons, meme types, cultural events
- Events/diseases/catastrophes (~25): `speciation` `extinction` `plague` `volcanic_winter` etc.
- Diet (5): `det` `plt` `sol` `chemical` `none`
- Agent behaviors (7): `forage` `rest` `flock` `flee` `mate` `fl` `fk`
- Glue prefixes (15): `pop` `plan=` `inf+=` `loc+=` `locs` `outcome=` `reason=` `Cmem` `grid+` etc.
- Symbols (18): `±` `Δ` `+` `-` `=` `|` `:` `(` `)` `{` `}` `*` `.` `,` `/` `->` `†` `_`
- ID prefixes (12): `S` `L` `H` `W` `G` `A` `N` `T` `E` `K` `D` `P`
- Letters (52): `a`–`z` `A`–`Z` (character fallback for rare/unknown text)

**Trie-based longest-match**: Multi-character tokens (`@SP`, `sessile_autotroph`, `inf+=`) are matched greedily before falling through to single characters. Handles `@SP` vs `@SPOT` and `inf` vs `inf+=` disambiguation automatically — the trie always matches the longest candidate.

**v1 (verbose) format compatibility**: All v1-only keywords (`TICK_SCALE`, `success=`, `rate=`, `diet=`, etc.) are tokenized via character-level fallback through the LETTERS list. 0% UNK, but ~1.8x more tokens than v2 compact. Always use `--compact` for training data.

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
# 1. Generate dataset (compact v2 format)
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 \
    --output data/evo_train.txt --workers 8 --seed 42 --compact

# 2. Train (tiny model, single GPU, 12GB VRAM)
python train.py --stage 1 data/evo_train.txt \
    --model-size tiny --seq-len 2048 --stride 1024 --batch-size 8 \
    --gradient-accumulation-steps 4 --learning-rate 5e-4 \
    --warmup-steps 2000 --epochs 10 --mixed-precision --val-split 0.1
```

### Recommended approach (with agent simulation)

Agent blocks use grid+notable hybrid format to keep token volume manageable. A single ECOSYSTEM agent keyframe tick has a median of ~721 tokens (p95 = 6,139) with the 512-token trie tokenizer. Standard sequence lengths work with adjusted keyframe intervals.

```bash
# 1. Generate dataset (agents enabled, compact v2 format)
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 \
    --output data/evo_agents_train.txt --workers 8 --seed 42 \
    --enable-agents --agent-epoch ECOSYSTEM --keyframe-interval 40 --compact

# 2. Train (tiny model, 24GB GPU)
python train.py --stage 1 data/evo_agents_train.txt \
    --model-size tiny --seq-len 4096 --batch-size 8 \
    --gradient-accumulation-steps 4 --learning-rate 5e-4 \
    --warmup-steps 3000 --epochs 10 --mixed-precision \
    --gradient-checkpointing --val-split 0.1
```

### Critical training parameters

All values below assume **v2 compact format** (`--compact`) and the **custom 512-token trie tokenizer**. v1 (verbose) format requires ~1.8x the seq-len for equivalent coverage due to character-level fallback on verbose keywords.

The 512-token vocabulary saves ~1.6 GB VRAM on embedding/projection weights compared to GPT-2's 50K vocab (at d=2048). This headroom can be traded for larger batch sizes or longer sequences.

**Population-only (no agents):**

| Parameter     | Value                          | Why                                                                              |
| ------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| `--seq-len`   | 2048                           | INTELLIGENCE delta p95 = 1,604 fits within 2048. Keyframes (median 3K–21K) are too large for any single window — learned via overlapping views. |
| `--stride`    | 1024                           | 50% overlap ensures every token appears in ~2 windows. Keyframes span ~3–21 windows each, giving full coverage.                                  |
| Warmup        | 2000 steps                     | MoE router needs stabilization time.                                             |
| Peak LR       | 5e-4 (tiny/small), 1e-4 (base) | Standard for MoE.                                                                |
| Min LR        | 1e-5                           | Never decay to zero — late data is the most complex.                             |
| Gradient clip | 1.0                            | MoE can produce gradient spikes.                                                 |

**With agent simulation (grid+notable format):**

| Parameter     | Value                          | Why                                                                              |
| ------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| `--seq-len`   | 4096 (24GB), 8192 (48GB+)     | ECOSYSTEM agent keyframe p95 = 6,139. INTELLIGENCE p95 = 4,813.                 |
| `--stride`    | 2048 (24GB), 4096 (48GB+)     | 50% overlap. Grid cells have weak inter-tick causality.                          |
| Warmup        | 3000 steps                     | Agent tokens increase vocabulary diversity; router needs more time.               |
| Peak LR       | 5e-4 (tiny/small), 1e-4 (base) | Same as population-only.                                                         |
| Min LR        | 1e-5                           | Same as population-only.                                                         |
| Gradient clip | 1.0                            | Same as population-only.                                                         |
| Keyframe interval | 40                         | Halves agent keyframe frequency — reduces token spikes by 2x.                    |

**Batch size scaling (24GB GPU):**

The 512-token vocab frees ~1.6 GB VRAM vs GPT-2, allowing +1-2 batch size headroom at equivalent seq-len.

| Model | `--seq-len` | `--batch-size` | `--gradient-accumulation-steps` | Effective batch |
| ----- | ----------- | -------------- | ------------------------------- | --------------- |
| Tiny  | 4096        | 10             | 3                               | 30              |
| Tiny  | 8192        | 4              | 8                               | 32              |
| Small | 4096        | 4              | 8                               | 32              |
| Small | 8192        | 2              | 16                              | 32              |

Use `--gradient-checkpointing` unconditionally with agent-enabled data.

### Token volume comparison

Empirically measured tokens per tick with the **512-token trie tokenizer** (v2 compact format, kf=20, n=10 worlds, ~28.5M total tokens):

**Population-only (no agents):**

| Epoch        | Delta (median) | Delta (p95) | Keyframe (median) | Keyframe (p95) | Notes                                              |
| ------------ | -------------- | ----------- | ------------------ | -------------- | -------------------------------------------------- |
| PRIMORDIAL   | 42             | 252         | 1,114              | 2,095          | Compact; few species, fast ticks.                  |
| CAMBRIAN     | 911            | 1,160       | 913                | 1,162          | Brief transitional epoch (n=5 in sample).          |
| ECOSYSTEM    | 162            | 1,302       | 2,981              | 23,288         | Many species; keyframes grow with species count.   |
| INTELLIGENCE | 773            | 1,604       | 21,029             | 23,044         | Spotlight blocks dominate keyframes.               |

Delta blocks (between `---` markers) across all epochs: median=164, P90=429, P95=559, P99=1,609.

Keyframes in ECOSYSTEM/INTELLIGENCE are 3K–23K tokens — far too large for any single training window. The model learns keyframe structure across multiple overlapping windows via stride-based sliding. Delta blocks (the vast majority of training data) fit comfortably within seq-len=2048.

**With agent simulation (estimated):**

Agent `@AGENT` sub-blocks add ~50-75% more tokens to ECOSYSTEM/INTELLIGENCE keyframes. Use `--seq-len 4096` minimum with `--gradient-checkpointing`.

### World boundary handling

Worlds are independent simulations. **Never pack tokens from different worlds into the same sequence.** Pad to `seq_len` at each world boundary (EOS).

Note: `TextDataset` currently concatenates all worlds into a single flat token stream and applies a sliding window — it does **not** enforce world boundaries. Cross-world sequences are rare (one per world boundary, ~10 in a 10-world dataset vs ~27K total sequences), so the impact is negligible for training. A future improvement could mask cross-world positions in the loss.

### Training curriculum

Don't train sequentially (biology → ecosystems → intelligence) — causes catastrophic forgetting. Instead, mix gradually.

**Important**: Mix by **token count**, not world count. Agent-enabled ECOSYSTEM worlds produce 2-5x more tokens per world than population-only worlds (v2 compact). Mixing by world count causes agent tokens to dominate training, overfitting the model to coordinate prediction at the expense of core ecological dynamics.

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
