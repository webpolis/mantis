"""
Memory Consolidation Lifecycle

Moves episodic entries into semantic memory:

- An entry that would overflow the episodic buffer is written to semantic
  memory before the buffer releases it. Failed writes leave it in the buffer.
- A periodic cycle promotes entries that retrieval has already found useful
  (hit count >= `min_hits`) or that carry an `important` flag.
- Stopping flushes the queue; `persist_dir` checkpoints both stores after
  every cycle that wrote something.
"""

import hashlib
import queue
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from ..inference.prompting import render_entry
from .episodic import EpisodicMemory
from .semantic import SemanticMemory


class MemoryConsolidator:
    """Periodic episodic -> semantic transfer with synchronous overflow writes."""

    RECENT_TEXTS = 10_000

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        tokenizer,
        consolidation_interval: int = 600,
        min_hits: int = 1,
        queue_size: int = 256,
        persist_dir: Optional[str] = None,
    ):
        if tokenizer is None:
            raise ValueError("MemoryConsolidator needs the tokenizer to render entries")
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.tokenizer = tokenizer
        self.interval = consolidation_interval
        self.min_hits = min_hits
        self.persist_dir = persist_dir

        self._queue: 'queue.Queue[List[Dict]]' = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._run_lock = threading.Lock()
        self._store_lock = threading.Lock()
        self._recent: 'OrderedDict[str, None]' = OrderedDict()
        for entry in semantic_memory.entries.values():
            meta = entry['metadata']
            origin = meta.get('origin', '')
            if origin.startswith('episodic:'):
                digest = self._entry_key(meta.get('namespace', 'default'), origin[9:])
                self._recent[digest] = None
                if len(self._recent) > self.RECENT_TEXTS:
                    self._recent.popitem(last=False)
        self.thread: Optional[threading.Thread] = None

        self.stats = {
            'stored': 0, 'duplicates': 0, 'failures': 0, 'overflow_stored': 0,
            'cycles': 0, 'last_cycle': None, 'queue_pending': 0,
        }
        episodic_memory.on_overflow = self.enqueue

    # ---------------------------------------------------------------- thread

    def start(self):
        """Start the background thread (queue drain + periodic cycles)."""
        if self.running:
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the thread and flush everything still queued."""
        self._stop.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.flush()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def pause(self):
        """Suspend periodic cycles (queued evictions are still written)."""
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def _loop(self):
        next_cycle = time.time() + self.interval
        while not self._stop.is_set():
            try:
                entries = self._queue.get(timeout=1.0)
            except queue.Empty:
                entries = None
            if entries is not None:
                self._store_evicted(entries)
            if time.time() >= next_cycle:
                next_cycle = time.time() + self.interval
                if not self._paused.is_set():
                    try:
                        self.consolidate()
                    except Exception as e:
                        print(f"Consolidation error: {e}")

    # ----------------------------------------------------------------- queue

    def enqueue(self, entries: List[Dict]) -> None:
        """Preserve overflow entries before the episodic buffer drops them."""
        self._store_evicted(entries)

    def flush(self) -> None:
        """Write every queued eviction now."""
        while True:
            try:
                entries = self._queue.get_nowait()
            except queue.Empty:
                break
            self._store_evicted(entries)

    def _store_evicted(self, entries: List[Dict]) -> None:
        stored, failed = self.store_entries(entries)
        self.stats['overflow_stored'] += len(stored)
        self.stats['queue_pending'] = self._queue.qsize()
        if stored:
            self.persist()
        if failed:
            raise RuntimeError(f"Could not preserve {len(failed)} evicted episodic entries")

    # ----------------------------------------------------------------- store

    @staticmethod
    def _entry_key(namespace: str, entry_id) -> str:
        return hashlib.sha1(f"{namespace}\0{entry_id}".encode('utf-8')).hexdigest()

    def store_entries(self, entries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Write each entry as its own semantic record, provenance attached.

        Returns:
            (stored, failed) entry lists. A retried write of the same episodic
            entry counts as stored but does not make a second semantic copy.
        """
        stored, failed = [], []
        with self._store_lock:
            for entry in entries:
                try:
                    text = render_entry(entry, self.tokenizer)
                    if not text:
                        failed.append(entry)
                        continue
                    namespace = entry['metadata'].get('namespace', 'default')
                    digest = self._entry_key(namespace, entry['id'])
                    if digest in self._recent:
                        self.stats['duplicates'] += 1
                        stored.append(entry)
                        continue
                    self.semantic.add(entry['embedding'], text, {
                        **entry['metadata'],
                        'origin': f"episodic:{entry['id']}",
                        'hits': entry['hits'],
                        'consolidated_at': time.time(),
                    })
                    self._recent[digest] = None
                    if len(self._recent) > self.RECENT_TEXTS:
                        self._recent.popitem(last=False)
                    stored.append(entry)
                    self.stats['stored'] += 1
                except Exception as e:
                    failed.append(entry)
                    self.stats['failures'] += 1
                    print(f"Error consolidating entry {entry.get('id')}: {e}")
        return stored, failed

    def consolidate(self) -> Dict:
        """
        Promote useful entries now (serialized with any other cycle).

        Only entries whose own write succeeded leave the episodic buffer.
        """
        with self._run_lock:
            candidates = self.episodic.candidates(self.min_hits)
            stored, failed = self.store_entries(candidates)
            if stored and self.persist_dir:
                self._persist_semantic()
            self.episodic.remove(stored)
            self.stats['cycles'] += 1
            self.stats['last_cycle'] = time.time()
            if stored:
                self._persist_episodic()
            return {
                'status': 'success' if candidates else 'skipped',
                'candidates': len(candidates),
                'consolidated': len(stored),
                'failed': len(failed),
                'episodic_remaining': self.episodic.size(),
                'semantic_total': self.semantic.size(),
            }

    def delete_namespace(self, namespace: str) -> None:
        """Delete a temporary namespace without racing a promotion cycle."""
        with self._run_lock:
            self.episodic.delete_namespace(namespace)
            self.semantic.delete_namespace(namespace)
            self.persist()

    def persist(self) -> None:
        """Checkpoint both stores under persist_dir (no-op without one)."""
        self._persist_semantic()
        self._persist_episodic()

    def _persist_semantic(self) -> None:
        if not self.persist_dir:
            return
        import os
        os.makedirs(self.persist_dir, exist_ok=True)
        self.semantic.save(os.path.join(self.persist_dir, 'semantic'))

    def _persist_episodic(self) -> None:
        if not self.persist_dir:
            return
        import os
        self.episodic.save(os.path.join(self.persist_dir, 'episodic.pt'))

    def get_stats(self) -> Dict:
        return {**self.stats, 'queue_pending': self._queue.qsize(),
                'episodic': dict(self.episodic.stats), 'semantic': dict(self.semantic.stats)}
