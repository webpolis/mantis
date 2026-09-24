"""
Memory Consolidation Process

Transfers high-importance memories from episodic to semantic storage.
Mimics human sleep consolidation.
"""

import threading
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .episodic import EpisodicMemory, group_similar
from .semantic import SemanticMemory


class MemoryConsolidator:
    """
    Background process for consolidating episodic to semantic memory.

    Runs periodically to:
    1. Identify high-importance episodic memories
    2. Group similar memories
    3. Store each group's representative text in semantic memory
    4. Remove from the episodic buffer only the entries that were stored
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        base_model: nn.Module,
        tokenizer,
        consolidation_interval: int = 3600,  # 1 hour
        importance_threshold: float = 0.7,
        similarity_threshold: float = 0.8,
        min_consolidation_size: int = 5,
    ):
        if tokenizer is None:
            raise ValueError("MemoryConsolidator needs the tokenizer to decode and encode facts")
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.model = base_model
        self.tokenizer = tokenizer
        self.interval = consolidation_interval
        self.importance_threshold = importance_threshold
        self.similarity_threshold = similarity_threshold
        self.min_consolidation_size = min_consolidation_size

        self._stop = threading.Event()
        self._run_lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        self.stats = {
            'total_consolidated': 0,
            'last_consolidation': None,
            'consolidations_count': 0
        }

    def start(self):
        """Start background consolidation thread."""
        if self.thread is not None and self.thread.is_alive():
            print("Consolidator already running")
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._consolidate_loop, daemon=True)
        self.thread.start()
        print(f"Memory consolidation started (interval: {self.interval}s)")

    def stop(self):
        """Stop background consolidation, waiting for an in-progress cycle to finish."""
        self._stop.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        print("Memory consolidation stopped")

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _consolidate_loop(self):
        """Main consolidation loop; wakes immediately when stopped."""
        while not self._stop.wait(self.interval):
            try:
                self.consolidate()
            except Exception as e:
                print(f"Consolidation error: {e}")

    def consolidate(self) -> Dict:
        """
        Run one consolidation cycle (serialized with any other cycle).

        Returns:
            Dict with consolidation statistics
        """
        with self._run_lock:
            if self.episodic.size() < self.min_consolidation_size:
                return {'status': 'skipped', 'reason': 'insufficient_memories'}

            candidates = self.episodic.get_high_importance(self.importance_threshold)
            if not candidates:
                return {'status': 'skipped', 'reason': 'no_important_memories'}

            groups = group_similar(candidates, self.similarity_threshold)
            stored: List[Dict] = []
            failed = 0

            for group in groups:
                try:
                    fact = self._summarize_group(group)
                    if not fact:
                        continue
                    self.semantic.add(
                        self._encode_fact(fact),
                        fact,
                        metadata={
                            'consolidated_from': len(group),
                            'importance': sum(e['importance'] for e in group) / len(group),
                            'timestamp': time.time(),
                        }
                    )
                    stored.extend(group)
                except Exception as e:
                    failed += 1
                    print(f"Error consolidating group: {e}")

            self.episodic.remove(stored)

            self.stats['total_consolidated'] += len(stored)
            self.stats['last_consolidation'] = time.time()
            self.stats['consolidations_count'] += 1

            return {
                'status': 'success',
                'candidates': len(candidates),
                'groups': len(groups),
                'consolidated': len(stored),
                'failed_groups': failed,
                'episodic_remaining': self.episodic.size(),
                'semantic_total': self.semantic.size()
            }

    def _summarize_group(self, group: List[Dict]) -> str:
        """
        Represent a group of similar memories by its longest interaction's text.
        """
        longest = max(group, key=lambda e: len(e['tokens']))
        return self.tokenizer.decode(longest['tokens']).strip()

    @torch.no_grad()
    def _encode_fact(self, fact: str) -> torch.Tensor:
        """
        Encode a fact exactly like the inference engine encodes queries
        (mean-pooled base-model hidden state).

        Returns:
            (d_model,) embedding
        """
        token_ids = self.tokenizer.encode(fact)[-self.model.max_seq_len:]
        device = next(self.model.parameters()).device
        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
        return self.model.encode(tokens).squeeze(0).cpu()

    def get_stats(self) -> Dict:
        """Get consolidation statistics."""
        return self.stats.copy()

    def trigger_consolidation(self) -> Dict:
        """Run one consolidation cycle now (waits for a background cycle in progress)."""
        return self.consolidate()
