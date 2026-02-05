"""
Memory Consolidation Process

Transfers high-importance memories from episodic to semantic storage.
Mimics human sleep consolidation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import threading
import time
from .episodic import EpisodicMemory, group_similar
from .semantic import SemanticMemory


class MemoryConsolidator:
    """
    Background process for consolidating episodic to semantic memory.

    Runs periodically to:
    1. Identify high-importance episodic memories
    2. Group similar memories
    3. Summarize and store in semantic memory
    4. Remove from episodic buffer
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        base_model: nn.Module,
        consolidation_interval: int = 3600,  # 1 hour
        importance_threshold: float = 0.7,
        similarity_threshold: float = 0.8,
        min_consolidation_size: int = 5
    ):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.model = base_model
        self.interval = consolidation_interval
        self.importance_threshold = importance_threshold
        self.similarity_threshold = similarity_threshold
        self.min_consolidation_size = min_consolidation_size

        # Threading
        self.running = False
        self.thread = None

        # Statistics
        self.stats = {
            'total_consolidated': 0,
            'last_consolidation': None,
            'consolidations_count': 0
        }

    def start(self):
        """Start background consolidation thread."""
        if self.running:
            print("Consolidator already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._consolidate_loop, daemon=True)
        self.thread.start()
        print(f"Memory consolidation started (interval: {self.interval}s)")

    def stop(self):
        """Stop background consolidation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Memory consolidation stopped")

    def _consolidate_loop(self):
        """Main consolidation loop."""
        while self.running:
            try:
                # Wait for interval
                time.sleep(self.interval)

                # Run consolidation
                self.consolidate()

            except Exception as e:
                print(f"Consolidation error: {e}")

    def consolidate(self) -> Dict:
        """
        Run one consolidation cycle.

        Returns:
            Dict with consolidation statistics
        """
        if self.episodic.size() < self.min_consolidation_size:
            return {'status': 'skipped', 'reason': 'insufficient_memories'}

        print(f"Starting consolidation ({self.episodic.size()} episodic memories)")

        # Step 1: Get high-importance memories
        candidates = self.episodic.get_high_importance(self.importance_threshold)

        if len(candidates) == 0:
            return {'status': 'skipped', 'reason': 'no_important_memories'}

        print(f"Found {len(candidates)} high-importance memories")

        # Step 2: Group similar memories
        groups = group_similar(candidates, self.similarity_threshold)

        print(f"Grouped into {len(groups)} clusters")

        # Step 3: Consolidate each group
        consolidated_count = 0

        for group in groups:
            try:
                fact = self._summarize_group(group)

                if fact:
                    # Encode and store in semantic memory
                    embedding = self._encode_fact(fact)
                    self.semantic.add(
                        embedding,
                        fact,
                        metadata={
                            'consolidated_from': len(group),
                            'importance': sum(
                                self.episodic.importance_scores[i]
                                for i, e in enumerate(self.episodic.entries)
                                if e in group
                            ) / len(group),
                            'timestamp': time.time()
                        }
                    )
                    consolidated_count += 1

            except Exception as e:
                print(f"Error consolidating group: {e}")
                continue

        # Step 4: Remove consolidated memories from episodic
        self.episodic.remove(candidates)

        # Update statistics
        self.stats['total_consolidated'] += consolidated_count
        self.stats['last_consolidation'] = time.time()
        self.stats['consolidations_count'] += 1

        result = {
            'status': 'success',
            'candidates': len(candidates),
            'groups': len(groups),
            'consolidated': consolidated_count,
            'episodic_remaining': self.episodic.size(),
            'semantic_total': self.semantic.size()
        }

        print(f"Consolidation complete: {result}")

        return result

    def _summarize_group(self, group: List[Dict]) -> Optional[str]:
        """
        Summarize a group of similar memories into a factual statement.

        Args:
            group: List of memory entries

        Returns:
            Summarized fact string or None
        """
        if len(group) == 0:
            return None

        # Simple heuristic: use longest memory as summary
        # In production, use model to generate summary
        longest = max(group, key=lambda e: len(e['tokens']))
        fact = f"Memory cluster (n={len(group)})"  # Placeholder

        return fact

    def _encode_fact(self, fact: str) -> torch.Tensor:
        """
        Encode fact into embedding vector.

        Args:
            fact: Text string

        Returns:
            (dimension,) embedding vector
        """
        # Placeholder: In production, use model's embedding layer
        # For now, return random vector
        embedding = torch.randn(self.semantic.dimension)
        return embedding

    def get_stats(self) -> Dict:
        """Get consolidation statistics."""
        return self.stats.copy()

    def trigger_consolidation(self):
        """Manually trigger consolidation cycle."""
        if not self.running:
            print("Starting one-time consolidation")
            return self.consolidate()
        else:
            print("Consolidation already running in background")
            return {'status': 'already_running'}
