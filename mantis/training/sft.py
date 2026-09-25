"""
Stage 5: Generator Adaptation

Instruction / evidence fine-tuning of the base model on the prompt format
the inference engine uses at runtime (chat roles plus a delimited evidence
block with source identifiers). Loss is computed on response tokens only.

Data: JSONL with {"query": str, "response": str, "evidence": [str] (optional)}
per line. Distractor, contradiction and unanswerable cases are ordinary
records whose response says so (for example an abstention).
"""

import json
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from mantis.inference.prompting import build_prompt_ids, format_query


def load_sft_records(path: str) -> List[Dict]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records.append({
                    'query': record['query'],
                    'response': record['response'],
                    'evidence': list(record.get('evidence') or []),
                })
    return records


def split_records(records: List[Dict], val_split: float, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle once and hold out the last `val_split` fraction (at least one record)."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]


class SFTDataset(Dataset):
    """
    Windows of seq_len + 1 tokens: evidence block + formatted query, then
    " response" + EOS. Prompt and pad positions are labelled -100. A record
    longer than the window keeps its response and the tail of the prompt.
    """

    def __init__(self, records: List[Dict], tokenizer, seq_len: int):
        if not records:
            raise ValueError("SFTDataset needs at least one record")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = [self._encode(r) for r in records]

    def _encode(self, record: Dict) -> Tuple[List[int], List[int]]:
        tokenizer = self.tokenizer
        items = [{'id': f"D{i}", 'tier': 'document', 'text': text, 'source': 'external'}
                 for i, text in enumerate(record['evidence'])]
        prompt = build_prompt_ids(items, tokenizer.encode(format_query(record['query'], 'chat')), tokenizer, 'chat')
        response = tokenizer.encode(" " + record['response']) + [tokenizer.eos_token_id]

        window = self.seq_len + 1
        response = response[-window:]
        prompt = prompt[-(window - len(response)):] if window > len(response) else []
        ids = prompt + response
        labels = [-100] * len(prompt) + response
        return ids, labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, labels = self.examples[idx]
        window = self.seq_len + 1
        pad = window - len(ids)
        ids = torch.tensor(ids + [self.tokenizer.pad_token_id] * pad, dtype=torch.long)
        labels = torch.tensor(labels + [-100] * pad, dtype=torch.long)
        return {'input_ids': ids[:-1], 'labels': labels[1:]}
