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

Higher tiers require prerequisites from lower tiers (e.g., `intel` needs `curiosity≥2` + `social≥2`).

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
| CAMBRIAN     | ~10 generations   | Any species evolves T1 trait | Body plan diversification, arms races  |
| ECOSYSTEM    | ~1 generation     | 3+ trophic levels exist     | Niche specialization, migration        |
| INTELLIGENCE | Sub-generational  | Spotlight score > 3.0       | Individual-level events, culture       |

Mutation rates are normalized **per generation** (raw `mutation_rate_mult / tick_scale`), so PRIMORDIAL (1000 gen/tick) and INTELLIGENCE (0.1 gen/tick) evolve at comparable per-generation rates despite vastly different tick scales. Speciation probabilities and serialization detail also adjust per epoch. Speciation is capped at 20 alive species per world.

---

## Output Protocol

The simulator outputs structured text for `train_evo.py` (or `train.py`). It supports two formats:

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

Keyframes dump full state every 20 ticks (`--keyframe-interval`); intermediate ticks use delta encoding (`Δspeed=+0.3` in v1, `Δspeed +3` in v2).

### Agent blocks

When agent-based simulation is active, each species block includes an `@AGENT` sub-block. The header gives the species' total agent count, and one line follows for each of up to 20 representative agents with its 10-unit quantized position, energy, age, and behavioral state.

The serializer picks representatives by behavioral interest and keeps them across ticks, so trajectories stay continuous. Priority goes to agents it already tracks, then agents hunting, fleeing or mating, then the two lowest- and two highest-energy agents, and finally agents spread across the map.

v1 keyframe:
```
  @AGENT|count=420
    N:A1:(130,350,E=52,age=10,hunt->A3)
    N:A7:(400,100,E=89,age=15,forage)
    N:A12:(220,510,E=35,age=5,flee)
```

v2 keyframe:
```
  @AGENT 420
   N A1 130 350 52 10 hunt->A3
   N A7 400 100 89 15 forage
   N A12 220 510 35 5 flee
```

Delta ticks list only tracked agents that are newly tracked, moved more than 5 units, changed energy by more than 2, or changed behavioral state, plus tracked agents that died (`†`):

v1 delta:
```
  @AGENT|Δpos
    N:A1:(140,360,E=48,age=11,forage)
    A7:†
```

v2 delta:
```
  @AGENT Δ
   N A1 140 360 48 11 forage
   A7 †
```

---

## Tokenization

`MANTISTokenizer` is a custom trie-based longest-match tokenizer with 512 tokens, a size that suits tensor cores. It has no GPT-2, BPE or `transformers` dependency, and a UTF-8 byte fallback lets any text round-trip losslessly.

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

**Vocabulary (283 domain + 18 extra ASCII + 161 byte fallback + 50 reserved = 512)**:

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
- Glue prefixes (16): `pop` `plan=` `D` `inf+=` `loc+=` `locs` `outcome=` `reason=` `Cmem` `dur` `low_var` `low_variance` `grid+` `gen` `age` `inf`
- Symbols (18): `±` `Δ` `+` `-` `=` `|` `:` `(` `)` `{` `}` `*` `.` `,` `/` `->` `†` `_`
- ID prefixes (12): `S` `L` `H` `W` `G` `A` `N` `T` `E` `K` `D` `P`
- Letters (52): `a`–`z` `A`–`Z` (character fallback for rare/unknown text)
- Extra ASCII (18): the printable characters the lists above miss (`!` `"` `#` `$` `%` `&` `'` `;` `<` `>` `?` `@` `[` `\` `]` `^` `` ` `` `~`)
- Byte fallback (161): `<0x00>`–`<0x1F>` and `<0x7F>`–`<0xFF>`; any other character encodes as its UTF-8 bytes
- Reserved (50): IDs 462–511, free for new protocol tokens

`grid+`, `fl` and `fk` belong to the retired grid-cell agent format. They stay in the vocabulary so that token IDs remain stable.

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

### Recommended approach: Curriculum training with `train_evo.py`

Generate 3 partitioned datasets by complexity tier, then train with `train_evo.py` which mixes them with shifting proportions via a curriculum schedule.

```bash
# 1. Generate partitioned datasets (compact v2 format)
python scripts/gen_evo_dataset.py --worlds 5000 --max-epoch CAMBRIAN  --output data/evo_bio.txt --compact --workers 8
python scripts/gen_evo_dataset.py --worlds 5000 --max-epoch ECOSYSTEM --output data/evo_eco.txt --compact --workers 8 --enable-agents
python scripts/gen_evo_dataset.py --worlds 5000  --output data/evo_intel.txt --compact --workers 8 --enable-agents

# 2. Train with curriculum (tiny model, single GPU, 12GB VRAM)
python train_evo.py \
    --bio data/evo_bio.txt --eco data/evo_eco.txt --intel data/evo_intel.txt \
    --model-size tiny --seq-len 2048 --batch-size 8 \
    --steps-per-epoch 1000 --epochs 20 \
    --learning-rate 5e-4 --warmup-steps 2000 \
    --mixed-precision --val-split 0.1

# 3. Generate from trained model
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --new-world --seed 42 --max-ticks 100
```

`--max-epoch CAMBRIAN` caps worlds at the CAMBRIAN epoch (they can enter CAMBRIAN but stop before ECOSYSTEM). This creates a "bio" partition with simpler ecological dynamics. The `--max-epoch` flag uses `>` comparison, so the named epoch is included.

`train_evo.py` key features:
- **`EvoWorldDataset`**: World-boundary-aware chunking (splits on `\n\n`, never crosses worlds)
- **`CurriculumDataset`**: IterableDataset mixing partitions with token-budget proportions that shift across training
- **Weighted cross-entropy**: Per-token loss weights from `tokenizer.compute_loss_weights()` (protocol markers set weight)
- **`--steps-per-epoch`** is required for training (IterableDataset has no `__len__`)
- Schedule presets: `default` (gradual shift), `linear`, `bio-only`
- **Token cache**: each partition is tokenized and weighted once into `--cache-dir` (default `data/.evo_cache`), keyed by file path, size, mtime and tokenizer fingerprint, then memory-mapped so DDP ranks share it. `--prepare-data-only` builds the caches and exits, so a CPU box can prepare data before a GPU run

### Alternative: Single-file training with `train.py`

For quick iteration without curriculum mixing, `train.py` works with a single evolution dataset:

```bash
# Generate single dataset
python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 \
    --output data/evo_train.txt --workers 8 --seed 42 --compact

# Train (no per-token weighting, no curriculum)
python train.py --stage 1 data/evo_train.txt \
    --model-size tiny --seq-len 2048 --stride 1024 --batch-size 8 \
    --gradient-accumulation-steps 4 --learning-rate 5e-4 \
    --warmup-steps 2000 --epochs 10 --mixed-precision --val-split 0.1
```

Note: `train.py` uses plain cross-entropy (no per-token weighting) and `TextDataset` (which does not respect world boundaries). For production evolution training, prefer `train_evo.py`.

### Agent-enabled training

Agent blocks carry one line per representative agent (up to 20 per species) with quantized position, energy, age, and behavioral state. Use `--enable-agents` on the eco and intel partitions (agents only activate at `--agent-epoch`, default ECOSYSTEM — the bio partition never reaches that epoch).

```bash
# Agent-enabled with longer sequences (24GB GPU)
python train_evo.py \
    --bio data/evo_bio.txt --eco data/evo_eco.txt --intel data/evo_intel.txt \
    --model-size tiny --seq-len 4096 --batch-size 4 \
    --steps-per-epoch 1000 --epochs 20 \
    --gradient-accumulation-steps 4 --learning-rate 5e-4 \
    --warmup-steps 3000 --mixed-precision \
    --gradient-checkpointing --val-split 0.1
```

### Critical training parameters

All values below assume **v2 compact format** (`--compact`) and the **custom 512-token trie tokenizer**. v1 (verbose) format requires ~1.8x the seq-len for equivalent coverage due to character-level fallback on verbose keywords.

The 512-token vocabulary saves ~1.6 GB VRAM on embedding/projection weights compared to GPT-2's 50K vocab (at d=2048). This headroom can be traded for larger batch sizes or longer sequences.

**Population-only (no agents):**

| Parameter     | Value                          | Why                                                                              |
| ------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| `--seq-len`   | 2048                           | Measured keyframe p95 is about 1,050 tokens and delta p95 about 330, so a whole keyframe fits in one window. |
| `--stride`    | 1024 (`train.py` only)         | 50% overlap puts every token in about 2 windows. `train_evo.py` cuts each world into back-to-back windows and pads the last one. |
| Warmup        | 2000 steps                     | MoE router needs stabilization time.                                             |
| Peak LR       | 5e-4 (tiny/small), 1e-4 (base) | Standard for MoE.                                                                |
| Min LR        | 1e-5                           | Never decay to zero — late data is the most complex.                             |
| Gradient clip | 1.0                            | MoE can produce gradient spikes.                                                 |

**With agent simulation:**

| Parameter     | Value                          | Why                                                                              |
| ------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| `--seq-len`   | 4096 (24GB), 8192 (48GB+)     | Measured agent keyframe p95 is about 4,950 tokens (max 5,331). 8192 fits every keyframe; 4096 splits about a quarter of them. |
| `--stride`    | 2048 (24GB), 4096 (48GB+), `train.py` only | 50% overlap.                                                          |
| Warmup        | 3000 steps                     | Agent tokens increase vocabulary diversity; router needs more time.               |
| Peak LR       | 5e-4 (tiny/small), 1e-4 (base) | Same as population-only.                                                         |
| Min LR        | 1e-5                           | Same as population-only.                                                         |
| Gradient clip | 1.0                            | Same as population-only.                                                         |
| Keyframe interval | 40                         | `--keyframe-interval 40` halves how often agent keyframes, the largest ticks, appear. |

**Batch size scaling (24GB GPU):**

The 512-token vocab frees ~1.6 GB VRAM vs GPT-2, allowing +1-2 batch size headroom at equivalent seq-len.

| Model | `--seq-len` | `--batch-size` | `--gradient-accumulation-steps` | Effective batch |
| ----- | ----------- | -------------- | ------------------------------- | --------------- |
| Tiny  | 4096        | 10             | 3                               | 30              |
| Tiny  | 8192        | 4              | 8                               | 32              |
| Small | 4096        | 4              | 8                               | 32              |
| Small | 8192        | 2              | 16                              | 32              |

Use `--gradient-checkpointing` unconditionally with agent-enabled data.

### Token volume

Tokens per tick, measured on 2026-09-24 with `python scripts/calc_seq_len.py --worlds 60` (v2 compact format, a keyframe every 20 ticks, up to 200 generations per world):

| Partition                        | Worlds | Keyframe median | Keyframe p95 | Delta median | Delta p95 |
| -------------------------------- | ------ | --------------- | ------------ | ------------ | --------- |
| bio (no agents)                  | 60     | 500             | 1,056        | 82           | 332       |
| eco (agents from ECOSYSTEM)      | 10     | 874             | 4,949        | 262          | 2,142     |
| intel (agents from INTELLIGENCE) | 10     | 567             | 1,034        | 68           | 264       |

Agent lines make up 73% of the eco partition's tokens, about 1,130 per tick while agents are active. None of the 10 intel worlds activated agents within 200 generations, so that row shows population-level ticks only. The samples are small: rerun the script with your own generation settings before you settle on `--seq-len`.

### World boundary handling

Worlds are independent simulations. **Never pack tokens from different worlds into the same sequence.** Pad to `seq_len` at each world boundary (EOS).

`train_evo.py` does this: `EvoWorldDataset` ends each world with EOS, cuts it into its own windows, and pads the last one. `train.py` does not. Its data pipeline also ends each world with EOS, but then packs all worlds into one stream, so the windows that straddle a boundary (about two per boundary at 50% stride) mix two worlds.

### Training curriculum

Don't train sequentially (biology → ecosystems → intelligence) — causes catastrophic forgetting. Instead, mix gradually.

**Important**: Mix by **token count**, not world count. Agent-enabled ECOSYSTEM worlds produce 2-5x more tokens per world than population-only worlds (v2 compact). Mixing by world count causes agent tokens to dominate training, overfitting the model to coordinate prediction at the expense of core ecological dynamics.

`train_evo.py` implements this via `CurriculumDataset`, which tracks global training progress and shifts proportions automatically. The default schedule:

| Progress | Bio tokens | Ecosystem tokens | Intelligence tokens |
| -------- | ---------- | ---------------- | ------------------- |
| 0-20%    | 100%       | 0%               | 0%                  |
| 20-40%   | 50%        | 50%              | 0%                  |
| 40-60%   | 25%        | 35%              | 40%                 |
| 60-100%  | 20%        | 30%              | 50%                 |

Progress is the fraction of planned micro-steps completed (`--epochs` × `--steps-per-epoch`), counted across the whole run rather than per epoch. `--schedule linear` shifts in three steps: bio only, then equal thirds from 33%, then 20/30/50 from 66%. `--schedule bio-only` trains on the bio partition alone.

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

| Size          | Experts   | Strategy                  | Use case              |
| ------------- | --------- | ------------------------- | --------------------- |
| Micro (3M)    | Dense     | Single GPU                | Pipeline sanity check |
| Tiny (55M)    | 4 experts | Single GPU                | Development iteration |
| Small (435M)  | 4 experts | Single GPU or DDP         | Experimentation       |
| Base (6.7B)   | 8 experts | DDP + ZeRO-2 (`--deepspeed`) | Production target  |

### What the trained model can do

Given a partial simulation trace, the model autoregressively continues it — predicting population changes, trait mutations, energy flows, species interactions, body plan transitions, individual agent behaviors and positions, and (for intelligent species) spotlight scenes with chain-of-thought reasoning. A game engine can sample from this model to run an evolving universe in real time.

### Inference with `inference_evo.py`

Tick-by-tick generation with `---` separator detection. Designed as an importable module for web apps.

```bash
# CLI: generate a new world
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --new-world --seed 42 --max-ticks 100

# CLI: continue from partial trace
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --continue trace.txt --max-ticks 50

# CLI: save to file
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --new-world --seed 0 --output generated_world.txt
```

```python
# Python API (for web app / WebSocket streaming)
from inference_evo import EvoInferenceEngine

engine = EvoInferenceEngine("checkpoints/evo_train/best_model.pt")

# Stream ticks to client
for tick in engine.generate_world(seed=42, temperature=0.7):
    send_to_client(tick)

# Continue from existing state
for tick in engine.continue_trace(existing_trace, max_ticks=10):
    send_to_client(tick)
```

`EvoInferenceEngine` decodes through the shared loop in `mantis/inference/generation.py`. The KV cache carries across ticks, and when it fills, the loop re-encodes the latest half-window.

### Full-engine inference

With a Stage 3 policy trained on the same evolution backbone, every tick runs through `MANTISInferenceEngine.generate()`: the meta-controller picks the gates, retrieved history is prepended as trace text, and the critic can reject the tick. The policy checkpoint records the Stage 2 memory, semantic store and Stage 4 critic paths; the explicit flags override them.

```bash
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --new-world --seed 42 --max-ticks 100 \
    --policy-checkpoint checkpoints/evo_policy/meta_controller_rl.pt \
    --memory-checkpoint checkpoints/evo_memory/memory_system_final.pt \
    --semantic-store checkpoints/evo_memory/semantic_memory \
    --critic-checkpoint checkpoints/evo_critic/critic_best.pt \
    --memory-dir runtime/evo --namespace world-42
```

The Python API takes the same arguments. `engine.last_result` holds the route, evidence, confidence and cost of the last yielded tick; `engine.close()` saves the runtime memory. A critic rejection ends the run instead of writing a refusal into the trace. `--route-policy always` opens every available gate for an ablation. The README's Evolution Simulation section describes the training data each stage needs.
