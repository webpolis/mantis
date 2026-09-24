"""
Shared data pipeline for every training mode.

Conventions (identical for raw text, pre-tokenized and HuggingFace data):
- Documents in text files are separated by blank lines.
- Each document is encoded without special tokens and followed by one EOS.
- Documents are concatenated and cut into windows of seq_len + 1 tokens every
  `stride` tokens; input_ids = window[:-1], labels = window[1:].
- Train/validation splits happen at document boundaries (or with a gap of
  windows for already-packed data), so no text appears in both.
"""

import math
from array import array
from typing import Iterable, Iterator, List, Tuple

import numpy as np


def iter_documents(path: str) -> Iterator[str]:
    """Yield blank-line-separated documents from a UTF-8 text file."""
    lines: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line)
            elif lines:
                yield ''.join(lines).rstrip('\n')
                lines = []
    if lines:
        yield ''.join(lines).rstrip('\n')


def encode_documents(documents: Iterable[str], tokenizer) -> Iterator[List[int]]:
    """Encode each document and append EOS."""
    eos = tokenizer.eos_token_id
    for doc in documents:
        yield tokenizer.encode(doc) + [eos]


def pack_windows(token_lists: Iterable[List[int]], seq_len: int, stride: int) -> Iterator[List[int]]:
    """Concatenate token lists and yield windows of seq_len + 1 tokens every `stride` tokens."""
    if not 0 < stride <= seq_len:
        raise ValueError(f"stride must be in [1, seq_len], got {stride}")
    buffer: List[int] = []
    for tokens in token_lists:
        buffer.extend(tokens)
        while len(buffer) >= seq_len + 1:
            yield buffer[:seq_len + 1]
            del buffer[:stride]


def tokenize_file(path: str, tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tokenize a text file into one flat token array.

    Returns:
        tokens: uint16 array of all document tokens (EOS after each document)
        doc_ends: int64 array with the end offset of each document
    """
    tokens = array('H')
    doc_ends = array('q')
    for doc_tokens in encode_documents(iter_documents(path), tokenizer):
        tokens.extend(doc_tokens)
        doc_ends.append(len(tokens))
    return np.frombuffer(tokens, dtype=np.uint16), np.frombuffer(doc_ends, dtype=np.int64)


def num_windows(n_tokens: int, seq_len: int, stride: int) -> int:
    """Number of windows pack_windows produces from n_tokens contiguous tokens."""
    if n_tokens < seq_len + 1:
        return 0
    return (n_tokens - seq_len - 1) // stride + 1


def split_at_document(tokens: np.ndarray, doc_ends: np.ndarray, val_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split a token array so the last ~val_fraction of tokens (whole documents) is validation."""
    target = len(tokens) * (1.0 - val_fraction)
    candidates = doc_ends[(doc_ends >= target) & (doc_ends < len(tokens))]
    if len(candidates) == 0:
        raise ValueError(
            f"Cannot split {len(doc_ends)} document(s) for --val-split {val_fraction}: "
            "need at least one whole document on each side of the split"
        )
    cut = int(candidates[0])
    return tokens[:cut], tokens[cut:]


def split_packed(dataset, val_fraction: float, seq_len: int, stride: int):
    """
    Split an already-packed, sequentially ordered window dataset into (train, val).

    Validation is the tail; the windows that would overlap it are dropped from
    train. Returns HuggingFace `Dataset` objects (anything with len/select).
    """
    n = len(dataset)
    n_val = int(n * val_fraction)
    gap = math.ceil((seq_len + 1) / stride) - 1
    n_train = n - n_val - gap
    if n_val < 1 or n_train < 1:
        raise ValueError(
            f"Cannot split {n} windows with val fraction {val_fraction}: "
            f"need at least 1 validation and 1 training window (gap of {gap})"
        )
    return dataset.select(range(n_train)), dataset.select(range(n - n_val, n))
