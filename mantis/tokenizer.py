"""
MANTIS Tokenizer — Trie-Based Domain-Specific Tokenizer

Custom tokenizer for the MANTIS ecological evolution simulation format.
Uses trie-based longest-match tokenization with ~300 domain tokens padded
to 512 for tensor core alignment. No GPT-2 / BPE dependency.

Numbers are always tokenized digit-by-digit for consistent encoding.
All protocol markers, body plans, traits, and domain keywords are
matched atomically via the trie before falling through to single
character tokens.

Vocabulary (~300 tokens, padded to 512):
    Special (4):        <pad> <eos> <bos> <unk>
    Digits (10):        0-9
    Protocol markers (8): =EPOCH @BIO @SP @INT @EVT @SPOT @AGENT ---
    Spotlight logic (6): CTX ACTORS INTENT REACT RESOLVE EFFECT
    Mutations (6):      M+ M- Mpoint Mdrift Mleap Mfuse
    Body plans (9):     sessile_autotroph .. decomposer
    Traits (45):        base + fused traits
    Biomes (15):        deep_ocean .. alpine
    Interactions (7):   hunt graze compete scavenge parasitize pollinate symbiosis
    Roles (5):          Elder Warrior Scout Youth Mother
    Actions (6):        report warn propose challenge request observe
    Reactions (6):      endorse reject debate defer counter accept
    Resolutions (6):    council vote elder_decree trial_by_combat consensus ritual
    Outcomes (12):      migrate build_shelter form_alliance ...
    Events/diseases (8+5+3+10): speciation extinction body_plan disease ...
    Meme types (4):     taboo legend tradition sacred
    Cultural events (8): great_hunt reef_collapse drought ...
    Glue prefixes (16): pop= plan= diet= inf+= outcome= ...
    Symbols (20):       ± Δ + - = | : ( ) { } * . , / -> † ...
    Whitespace (3):     space newline 2-space-indent
    Letters (52):       a-z A-Z (character fallback)
    Agent tokens (12):  grid+ forage rest flock flee mate fl fk ...
"""

from typing import List, Union, Optional, Dict, Set
import json
import os
import torch


# ---------------------------------------------------------------------------
# Vocabulary definitions — every token that can appear in serialized output
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]

DIGITS = [str(d) for d in range(10)]

LAYER_MARKERS = ["=EPOCH", "@BIO", "@SP", "@INT", "@EVT", "@SPOT", "@AGENT", "---"]

SPOTLIGHT_TOKENS = ["CTX", "ACTORS", "INTENT", "REACT", "RESOLVE", "EFFECT"]

MUTATION_TOKENS = ["M+", "M-", "Mpoint", "Mdrift", "Mleap", "Mfuse"]

BODY_PLAN_TOKENS = [
    "sessile_autotroph", "mobile_autotroph", "filter_feeder",
    "grazer", "predator", "scavenger", "omnivore", "parasite", "decomposer",
]

TRAIT_TOKENS = [
    # T0 (Physical)
    "size", "speed", "armor", "metab", "sense", "camo", "repro", "regen",
    "venom", "photosynth", "mouth", "endurance", "chem_digest",
    "toxin_resist", "toxin",
    # T1 (Behavioral)
    "social", "aggression", "curiosity", "patience", "nocturnal",
    # T2 (Cognitive)
    "intel", "memory", "learning", "planning", "deception",
    # T3 (Cultural)
    "language", "tooluse", "ritual", "teaching", "trade",
    # T4 (Abstract)
    "subconscious", "theory_of_mind", "creativity", "abstraction", "ethics",
    # Fused
    "dominance", "wisdom", "oral_tradition", "engineering",
    "philosophy", "empathy", "mythology", "strategy", "economy", "science",
]

BIOME_TOKENS = [
    "shallows", "reef", "deep_ocean", "tidal_pools", "mangrove",
    "savanna", "forest", "rainforest", "desert", "tundra",
    "cave", "volcanic_vent", "meadow", "swamp", "alpine",
]

INTERACTION_TOKENS = [
    "hunt", "graze", "compete", "scavenge", "parasitize", "pollinate", "symbiosis",
]

ROLE_TOKENS = ["Elder", "Warrior", "Scout", "Youth", "Mother"]

ACTION_TOKENS = ["report", "warn", "propose", "challenge", "request", "observe"]

REACTION_TOKENS = ["endorse", "reject", "debate", "defer", "counter", "accept"]

RESOLUTION_TOKENS = [
    "council", "vote", "elder_decree", "trial_by_combat", "consensus",
]
# NOTE: "ritual" is already in TRAIT_TOKENS — shared token, not duplicated

OUTCOME_TOKENS = [
    "split_colony", "migrate", "build_shelter", "form_alliance",
    "declare_territory", "begin_trade", "exile_member", "adopt_ritual",
    "develop_tool", "share_knowledge", "punish_violator", "honor_hero",
]

EVENT_KEYWORDS = [
    "speciation", "extinction", "body_plan", "disease",
    "symbiogenesis", "evo_trap", "catastrophe", "catastrophe_end",
    "WORLD",
]

DISEASE_TOKENS = ["plague", "blight", "parasitic_worm", "viral_outbreak", "fungal_rot"]

CATASTROPHE_TOKENS = ["volcanic_winter", "meteor_impact", "ice_age"]

EXTINCTION_CAUSE_TOKENS = [
    "drought", "predation", "competition", "habitat_loss",
    "volcanic", "radiation_spike", "overpopulation", "famine",
]
# NOTE: "disease", "ice_age", "plague" already in other lists — shared tokens

MEME_TOKENS = ["taboo", "legend", "tradition", "sacred"]

CULTURAL_EVENT_TOKENS = [
    "great_hunt", "reef_collapse", "elder_teaching", "first_fire",
    "migration", "battle_won",
]
# NOTE: "drought", "plague" already in other lists — shared tokens

REASON_TOKENS = ["environmental_pressure", "survival_instinct"]

DIET_TOKENS = ["det", "plt", "sol", "chemical", "none"]

REPRO_TOKENS = ["sexual", "asexual"]

# Glue prefixes — multi-char tokens that appear as field prefixes
GLUE_TOKENS = [
    "pop", "plan=", "D", "inf+=",
    "outcome=", "reason=", "Cmem", "dur",
    "low_var", "low_variance",
    "grid+",
    "gen", "age",
    "loc+=", "locs", "inf",
]

# Agent behavior tokens
AGENT_BEHAVIOR_TOKENS = ["forage", "rest", "flock", "flee", "mate", "fl", "fk"]
# NOTE: "hunt" already in INTERACTION_TOKENS — shared token
# NOTE: "fl"/"fk" are abbreviations for flee/flock used in agent grid behavior counts

SYMBOLS = [
    "±", "Δ", "+", "-", "=", "|", ":", "(", ")", "{", "}", "*",
    ".", ",", "/", "->", "†", "_",
]

# ID prefixes — single letters used as entity ID prefixes
ID_PREFIXES = ["S", "L", "H", "W", "G", "A", "N", "T", "E", "K", "D", "P"]

# Whitespace tokens
WHITESPACE = [" ", "\n", "  "]  # space, newline, 2-space indent

# Character fallback — lowercase and uppercase letters not covered above
# We include all 26+26 letters for rare/unknown text fallback
LETTERS = [chr(c) for c in range(ord('a'), ord('z') + 1)] + \
          [chr(c) for c in range(ord('A'), ord('Z') + 1)]

# Combined protocol tokens list (for analysis, excluding specials/digits/letters/whitespace)
PROTOCOL_TOKENS = (
    LAYER_MARKERS + SPOTLIGHT_TOKENS + MUTATION_TOKENS +
    BODY_PLAN_TOKENS + TRAIT_TOKENS + BIOME_TOKENS +
    INTERACTION_TOKENS + ROLE_TOKENS + ACTION_TOKENS +
    REACTION_TOKENS + RESOLUTION_TOKENS + OUTCOME_TOKENS +
    EVENT_KEYWORDS + DISEASE_TOKENS + CATASTROPHE_TOKENS +
    EXTINCTION_CAUSE_TOKENS + MEME_TOKENS + CULTURAL_EVENT_TOKENS +
    REASON_TOKENS + DIET_TOKENS + REPRO_TOKENS + GLUE_TOKENS +
    AGENT_BEHAVIOR_TOKENS + SYMBOLS + ID_PREFIXES
)

# Per-layer loss weights (from EVOLUTION_SIM_PLAN.md §Training Configuration)
LAYER_LOSS_WEIGHTS = {
    "---":    0.1,      # Trivial separators
    "=EPOCH": 0.5,      # Metadata headers
    "@BIO":   0.5,      # Slow-changing environment state
    "@SP":    1.0,      # Core simulation state
    "@INT":   1.5,      # Interaction dynamics ("the physics")
    "@EVT":   1.5,      # Rare, high-importance events
    "@AGENT": 0.8,      # Agent spatial data
    "@SPOT":  2.0,      # Intelligence reasoning chains
}

DEFAULT_LOSS_WEIGHT = 1.0

# Target vocab size — padded to 512 for tensor core alignment
TARGET_VOCAB_SIZE = 512


# ---------------------------------------------------------------------------
# Trie for longest-match tokenization
# ---------------------------------------------------------------------------

class _TrieNode:
    __slots__ = ('children', 'token_id')

    def __init__(self):
        self.children: Dict[str, '_TrieNode'] = {}
        self.token_id: Optional[int] = None


class _Trie:
    def __init__(self):
        self.root = _TrieNode()

    def insert(self, token: str, token_id: int):
        node = self.root
        for ch in token:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.token_id = token_id

    def longest_match(self, text: str, start: int) -> tuple:
        """Find longest matching token starting at position `start`.
        Returns (token_id, length) or (None, 0) if no match."""
        node = self.root
        best_id = None
        best_len = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch not in node.children:
                break
            node = node.children[ch]
            if node.token_id is not None:
                best_id = node.token_id
                best_len = i - start + 1
        return best_id, best_len


# ---------------------------------------------------------------------------
# Build the canonical vocabulary
# ---------------------------------------------------------------------------

def _build_vocab() -> Dict[str, int]:
    """Build the complete token→ID mapping. Order matters for determinism."""
    vocab: Dict[str, int] = {}
    idx = 0

    def add(token: str):
        nonlocal idx
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    # Special tokens first (IDs 0-3)
    for t in SPECIAL_TOKENS:
        add(t)

    # Digits (IDs 4-13)
    for t in DIGITS:
        add(t)

    # Whitespace (IDs 14-16)
    for t in WHITESPACE:
        add(t)

    # Protocol markers
    for t in LAYER_MARKERS:
        add(t)

    # Spotlight logic
    for t in SPOTLIGHT_TOKENS:
        add(t)

    # Mutations
    for t in MUTATION_TOKENS:
        add(t)

    # Body plans
    for t in BODY_PLAN_TOKENS:
        add(t)

    # Traits
    for t in TRAIT_TOKENS:
        add(t)

    # Biomes
    for t in BIOME_TOKENS:
        add(t)

    # Interactions
    for t in INTERACTION_TOKENS:
        add(t)

    # Roles
    for t in ROLE_TOKENS:
        add(t)

    # Actions
    for t in ACTION_TOKENS:
        add(t)

    # Reactions
    for t in REACTION_TOKENS:
        add(t)

    # Resolutions
    for t in RESOLUTION_TOKENS:
        add(t)

    # Outcomes
    for t in OUTCOME_TOKENS:
        add(t)

    # Event keywords
    for t in EVENT_KEYWORDS:
        add(t)

    # Disease types
    for t in DISEASE_TOKENS:
        add(t)

    # Catastrophe types
    for t in CATASTROPHE_TOKENS:
        add(t)

    # Extinction causes
    for t in EXTINCTION_CAUSE_TOKENS:
        add(t)

    # Meme types
    for t in MEME_TOKENS:
        add(t)

    # Cultural events
    for t in CULTURAL_EVENT_TOKENS:
        add(t)

    # Reason tokens
    for t in REASON_TOKENS:
        add(t)

    # Diet tokens
    for t in DIET_TOKENS:
        add(t)

    # Reproduction strategies
    for t in REPRO_TOKENS:
        add(t)

    # Glue prefixes
    for t in GLUE_TOKENS:
        add(t)

    # Agent behaviors
    for t in AGENT_BEHAVIOR_TOKENS:
        add(t)

    # Symbols
    for t in SYMBOLS:
        add(t)

    # ID prefixes
    for t in ID_PREFIXES:
        add(t)

    # Letters (character fallback)
    for t in LETTERS:
        add(t)

    # Pad to TARGET_VOCAB_SIZE with placeholder tokens
    while idx < TARGET_VOCAB_SIZE:
        placeholder = f"<reserved_{idx}>"
        vocab[placeholder] = idx
        idx += 1

    return vocab


# Canonical vocabulary — built once at module load
VOCAB = _build_vocab()
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}


# ---------------------------------------------------------------------------
# MANTISTokenizer
# ---------------------------------------------------------------------------

class MANTISTokenizer:
    """
    Trie-based domain-specific tokenizer for MANTIS evolution simulation format.

    ~300 tokens padded to 512 for tensor core alignment. Numbers are tokenized
    digit-by-digit. All domain keywords matched atomically via longest-match trie.
    """

    def __init__(self):
        self.vocab = dict(VOCAB)
        self.id_to_token = dict(ID_TO_TOKEN)
        self.vocab_size = len(self.vocab)

        self._trie = _Trie()
        for token, tid in self.vocab.items():
            if not token.startswith("<"):  # skip special tokens in trie
                self._trie.insert(token, tid)

        self._build_caches()

    def _build_caches(self):
        """Build fast lookup structures for protocol token IDs and loss weights."""
        self.pad_token_id = self.vocab["<pad>"]
        self.eos_token_id = self.vocab["<eos>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.unk_token_id = self.vocab["<unk>"]

        # Map layer marker strings → token IDs
        self._marker_to_id: Dict[str, int] = {}
        for marker in LAYER_MARKERS:
            if marker in self.vocab:
                self._marker_to_id[marker] = self.vocab[marker]

        # Token ID → loss weight
        self._id_to_weight: Dict[int, float] = {}
        for marker, weight in LAYER_LOSS_WEIGHTS.items():
            if marker in self._marker_to_id:
                self._id_to_weight[self._marker_to_id[marker]] = weight

        # Reverse lookup: token ID → marker string
        self._id_to_marker: Dict[int, str] = {
            v: k for k, v in self._marker_to_id.items()
        }

        # All protocol token IDs (for analysis)
        self._protocol_ids: Set[int] = set()
        for token in PROTOCOL_TOKENS:
            if token in self.vocab:
                self._protocol_ids.add(self.vocab[token])

    def _encode_single(self, text: str) -> List[int]:
        """Encode a single string using trie longest-match."""
        ids = []
        i = 0
        n = len(text)
        unk_id = self.unk_token_id
        while i < n:
            token_id, length = self._trie.longest_match(text, i)
            if token_id is not None:
                ids.append(token_id)
                i += length
            else:
                ids.append(unk_id)
                i += 1
        return ids

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: int = None,
        truncation: bool = True,
        padding: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, list):
            return [
                self.encode(t, add_special_tokens=add_special_tokens,
                            max_length=max_length, truncation=truncation,
                            padding=padding)
                for t in text
            ]

        ids = self._encode_single(text)

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        if padding and max_length is not None:
            while len(ids) < max_length:
                ids.append(self.pad_token_id)

        return ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        special_ids = {self.pad_token_id, self.eos_token_id, self.bos_token_id}
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(self.id_to_token.get(tid, "<unk>"))
        return "".join(tokens)

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [self.decode(ids, skip_special_tokens=skip_special_tokens)
                for ids in token_ids]

    def compute_loss_weights(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token loss weights from protocol layer markers.

        State machine: scans for layer marker tokens (=EPOCH, @BIO, @SP,
        @INT, @EVT, @SPOT, @AGENT, ---). When found, all subsequent tokens
        receive that layer's weight until the next marker.

        Weights (from EVOLUTION_SIM_PLAN.md):
            --- separators:  0.1
            =EPOCH headers:  0.5
            @BIO lines:      0.5
            @SP lines:       1.0
            @INT lines:      1.5
            @EVT lines:      1.5
            @AGENT lines:    0.8
            @SPOT blocks:    2.0

        Pad tokens always get weight 0.0.
        """
        squeeze = False
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze = True

        batch_size, seq_len = input_ids.shape
        weights = torch.full(
            (batch_size, seq_len), DEFAULT_LOSS_WEIGHT, dtype=torch.float32
        )

        ids = input_ids.tolist()
        pad = self.pad_token_id
        for b in range(batch_size):
            w = DEFAULT_LOSS_WEIGHT
            for i, tid in enumerate(ids[b]):
                if tid == pad:
                    weights[b, i] = 0.0
                    continue
                if tid in self._id_to_weight:
                    w = self._id_to_weight[tid]
                weights[b, i] = w

        if squeeze:
            weights = weights.squeeze(0)
        return weights

    def get_line_type(self, token_ids: List[int]) -> Optional[str]:
        """Return the protocol layer marker for a token sequence, or None."""
        for tid in token_ids:
            if tid in self._id_to_marker:
                return self._id_to_marker[tid]
        return None

    def protocol_token_ratio(self, input_ids: Union[List[int], torch.Tensor]) -> float:
        """Fraction of tokens that are protocol-specific (vs general)."""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if not input_ids:
            return 0.0
        count = sum(1 for tid in input_ids if tid in self._protocol_ids)
        return count / len(input_ids)

    @property
    def has_protocol_tokens(self) -> bool:
        """Whether protocol tokens are present in the vocabulary."""
        return len(self._marker_to_id) > 0

    def __len__(self) -> int:
        return self.vocab_size

    def save(self, path: str):
        """Save tokenizer to directory as vocab.json + config.json."""
        os.makedirs(path, exist_ok=True)
        vocab_path = os.path.join(path, "vocab.json")
        config_path = os.path.join(path, "config.json")

        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        config = {
            "tokenizer_class": "MANTISTokenizer",
            "vocab_size": self.vocab_size,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MANTISTokenizer':
        """Load tokenizer from a directory containing vocab.json."""
        vocab_path = os.path.join(path, "vocab.json")

        instance = cls.__new__(cls)

        with open(vocab_path, "r") as f:
            instance.vocab = json.load(f)

        instance.id_to_token = {v: k for k, v in instance.vocab.items()}
        instance.vocab_size = len(instance.vocab)

        instance._trie = _Trie()
        for token, tid in instance.vocab.items():
            if not token.startswith("<"):
                instance._trie.insert(token, tid)

        instance._build_caches()
        return instance
