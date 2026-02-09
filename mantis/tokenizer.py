"""
MANTIS Tokenizer — Evolution Simulation Protocol Extension

Extends GPT-2 BPE (50,257 tokens) with ~87 atomic protocol tokens for
the MANTIS ecological evolution simulation training data format.

Protocol tokens are matched atomically before BPE splitting, ensuring
structural markers, body plans, compound traits, and symbolic notation
are never fractured into subword pieces. Reduces token count per frame
by ~30-40%, effectively expanding the usable context window.

Token Categories (87 added):
    Layer Markers (6):      =EPOCH @BIO @SP @INT @EVT @SPOT
    Spotlight Logic (6):    CTX ACTORS INTENT REACT RESOLVE EFFECT
    Mutations (6):          M+ M- Mpoint Mdrift Mleap Mfuse
    Body Plans (9):         sessile_autotroph .. decomposer
    Traits (32):            Compound/abbreviated terms that BPE fractures
    Meme Types (4):         taboo legend tradition sacred
    Spotlight Roles (4):    Elder Warrior Scout Youth
    Interaction Verbs (5):  graze scavenge compete parasitize pollinate
    Symbols (3):            Δ r( K(
    Glue Prefixes (12):     pop= plan= diet={ rate= success= eff= ...

Tokens already single in GPT-2 (not re-added): ---, ±, ->,
    speed, size, armor, sense, social, memory, learning, language,
    trade, intel, engineering, science, hunt, Mother
"""

from transformers import GPT2Tokenizer, AddedToken
from typing import List, Union, Optional, Dict, Set
import torch


# ---------------------------------------------------------------------------
# Protocol Token Definitions — only tokens that GPT-2 BPE fractures
# ---------------------------------------------------------------------------

LAYER_MARKERS = [
    "=EPOCH",       # Epoch transition header (4 BPE → 1)
    "@BIO",         # Biome state block (3 BPE → 1)
    "@SP",          # Species state block (2 BPE → 1)
    "@INT",         # Interaction log (2 BPE → 1)
    "@EVT",         # Event line (3 BPE → 1)
    "@SPOT",        # Spotlight block (3 BPE → 1)
    "@AGENT",       # Agent block for spatial simulation (3 BPE → 1)
    # NOTE: "---" is already a single GPT-2 token (ID 6329). Not re-added.
]

SPOTLIGHT_TOKENS = [
    "CTX",          # Context section (2 BPE → 1)
    "ACTORS",       # Actor descriptions (2 BPE → 1)
    "INTENT",       # Action intent / chain-of-thought (2 BPE → 1)
    "REACT",        # Reaction to intent (2 BPE → 1)
    "RESOLVE",      # Resolution mechanism (3 BPE → 1)
    "EFFECT",       # State changes (2 BPE → 1)
]

MUTATION_TOKENS = [
    "M+",           # Trait acquired (2 BPE → 1)
    "M-",           # Trait pruned (2 BPE → 1)
    "Mpoint",       # Point mutation (2 BPE → 1)
    "Mdrift",       # Genetic drift (3 BPE → 1)
    "Mleap",        # Macro-mutation (3 BPE → 1)
    "Mfuse",        # Trait fusion (3 BPE → 1)
]

BODY_PLAN_TOKENS = [
    "sessile_autotroph",   # (7 BPE → 1)
    "mobile_autotroph",    # (5 BPE → 1)
    "filter_feeder",       # (4 BPE → 1)
    "grazer",              # (3 BPE → 1)
    "predator",            # (2 BPE → 1)
    "scavenger",           # (3 BPE → 1)
    "omnivore",            # (4 BPE → 1)
    "parasite",            # (3 BPE → 1)
    "decomposer",          # (4 BPE → 1)
]

# Only traits that GPT-2 BPE fractures into 2+ tokens.
# Single-token traits are: speed, size, armor, sense, social, memory,
# learning, language, trade, intel, engineering, science
TRAIT_TOKENS = [
    # T0 (Physical) — fractured
    "venom",            # ven+om
    "camo",             # c+amo
    "metab",            # met+ab (abbreviated metabolism)
    "repro",            # re+pro (abbreviated reproduction)
    "regen",            # reg+en (abbreviated regeneration)
    "photosynth",       # photos+yn+th (abbreviated photosynthesis)
    # T1 (Behavioral) — fractured
    "aggression",       # agg+ression
    "curiosity",        # cur+iosity
    "patience",         # pat+ience
    "nocturnal",        # no+ct+urnal
    "aggr",             # ag+gr (abbreviated, used in compact trait blocks)
    # T2 (Cognitive) — fractured
    "planning",         # plan+ning
    "deception",        # de+ception
    # T3 (Cultural) — fractured
    "ritual",           # rit+ual
    "teaching",         # te+aching
    "tooluse",          # tool+use (compound)
    # T4 (Abstract) — fractured
    "subconscious",     # sub+conscious
    "theory_of_mind",   # the+ory+_+of+_+mind (6 BPE!)
    "creativity",       # creat+ivity
    "abstraction",      # ab+st+raction
    "ethics",           # eth+ics
    # Fused traits — fractured
    "dominance",        # dom+inance
    "wisdom",           # w+isdom
    "oral_tradition",   # oral+_+tr+ad+ition (5 BPE!)
    "philosophy",       # phil+os+ophy
    "empathy",          # em+pathy
    "mythology",        # my+th+ology
    "strategy",         # str+ategy
    "economy",          # econom+y
    # Body-plan specific traits
    "chem_digest",      # chem+_+dig+est
    "toxin_resist",     # t+oxin+_+resist
    # Diet component
    "detritus",         # det+rit+us
]

MEME_TOKENS = [
    "taboo",            # tab+oo
    "legend",           # leg+end
    "tradition",        # tr+ad+ition
    "sacred",           # sac+red
]

ROLE_TOKENS = [
    "Elder",            # E+lder
    "Warrior",          # War+rior
    "Scout",            # Sc+out
    "Youth",            # Y+outh
    # Mother is already a single GPT-2 token
]

INTERACTION_TOKENS = [
    "graze",            # gra+ze
    "scavenge",         # sc+aven+ge
    "compete",          # comp+ete
    "parasitize",       # par+as+it+ize
    "pollinate",        # poll+inate
    # hunt is already a single GPT-2 token
]

SYMBOL_TOKENS = [
    "Δ",                # Mangled bytes in GPT-2 (2 BPE → 1)
    "r(",               # r-strategy marker (2 BPE → 1)
    "K(",               # K-strategy marker (2 BPE → 1)
    # ± is already a single GPT-2 token (ID 22519)
    # -> is already a single GPT-2 token (ID 3784)
]

GLUE_TOKENS = [
    "pop=",             # Population prefix (pop + =)
    "plan=",            # Body plan assignment (plan + =)
    "diet={",           # Diet vector opener (d+iet+={)
    "rate=",            # Reproduction rate (rate + =)
    "success=",         # Interaction success (success + =)
    "eff=",             # Efficiency (eff + =)
    "Cmem",             # Cultural memory prefix (C+mem)
    "reason=",          # Spotlight motivation (reason + =)
    "outcome=",         # Resolution outcome (out+come+=)
    "prereq:",          # Prerequisite notation (pre+req+:)
    "origin=",          # Meme origin (origin + =)
    "inf=",             # Hero influence (inf + =)
]

PROTOCOL_TOKENS = (
    LAYER_MARKERS + SPOTLIGHT_TOKENS + MUTATION_TOKENS +
    BODY_PLAN_TOKENS + TRAIT_TOKENS + MEME_TOKENS +
    ROLE_TOKENS + INTERACTION_TOKENS + SYMBOL_TOKENS + GLUE_TOKENS
)

# Per-layer loss weights (from EVOLUTION_SIM_PLAN.md §Training Configuration)
LAYER_LOSS_WEIGHTS = {
    "---":    0.1,      # Trivial separators
    "=EPOCH": 0.5,      # Metadata headers
    "@BIO":   0.5,      # Slow-changing environment state
    "@SP":    1.0,      # Core simulation state
    "@INT":   1.5,      # Interaction dynamics ("the physics")
    "@EVT":   1.5,      # Rare, high-importance events
    "@AGENT": 0.8,      # Agent spatial data (lower to avoid float precision overfitting)
    "@SPOT":  2.0,      # Intelligence reasoning chains
}

DEFAULT_LOSS_WEIGHT = 1.0


class MANTISTokenizer:
    """
    GPT-2 BPE tokenizer extended with atomic protocol tokens
    for the MANTIS ecological evolution simulation format.

    Adds 87 tokens to the base vocabulary (50,257 → 50,345).
    All protocol tokens are matched atomically before BPE splitting.
    Tokens that are already single in GPT-2 (---, ±, ->, common words)
    are referenced by their existing IDs, not re-added.
    """

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Inject protocol tokens — atomic, never split by BPE, decode naturally
        protocol_added = [
            AddedToken(t, lstrip=False, rstrip=False,
                       single_word=False, normalized=False)
            for t in PROTOCOL_TOKENS
        ]
        self.tokenizer.add_tokens(protocol_added)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        self.vocab_size = len(self.tokenizer)
        self._build_caches()

    def _build_caches(self):
        """Build fast lookup structures for protocol token IDs and loss weights."""
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

        unk = self.tokenizer.unk_token_id

        # Map layer marker strings → token IDs (for added markers)
        self._marker_to_id: Dict[str, int] = {}
        for marker in LAYER_MARKERS:
            tid = self.tokenizer.convert_tokens_to_ids(marker)
            if tid != unk:
                self._marker_to_id[marker] = tid

        # "---" is already a single GPT-2 token — get its native ID
        dash_ids = self.tokenizer.encode('---', add_special_tokens=False)
        if len(dash_ids) == 1:
            self._marker_to_id['---'] = dash_ids[0]

        # Token ID → loss weight (for state-machine scanner)
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
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if tid != unk:
                self._protocol_ids.add(tid)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: int = None,
        truncation: bool = True,
        padding: bool = False
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding='max_length' if padding else False
            )
        encodings = self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            padding='max_length' if padding else False
        )
        return encodings['input_ids']

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> List[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def compute_loss_weights(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token loss weights from protocol layer markers.

        State machine: scans for layer marker tokens (=EPOCH, @BIO, @SP,
        @INT, @EVT, @SPOT, ---). When found, all subsequent tokens receive
        that layer's weight until the next marker. Indented sub-lines
        (T: trait blocks, E: energy, spotlight sections) inherit from
        their parent marker.

        Weights (from EVOLUTION_SIM_PLAN.md):
            --- separators:  0.1
            =EPOCH headers:  0.5
            @BIO lines:      0.5
            @SP lines:       1.0
            @INT lines:      1.5
            @EVT lines:      1.5
            @SPOT blocks:    2.0

        Args:
            input_ids: (batch_size, seq_len) or (seq_len,) tensor

        Returns:
            weights: same shape, dtype float32
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
        """Fraction of tokens that are protocol-specific (vs general BPE)."""
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
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str):
        instance = cls.__new__(cls)
        instance.tokenizer = GPT2Tokenizer.from_pretrained(path)
        instance.vocab_size = len(instance.tokenizer)
        instance._build_caches()
        return instance
