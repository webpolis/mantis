import os
from types import SimpleNamespace

import pytest
import torch

from tests.conftest import needs_faiss, needs_mamba, seed


def segments(tokenizer, query, response=None):
    segs = [('query', tokenizer.encode(query))]
    if response is not None:
        segs.append(('response', tokenizer.encode(response)))
    return segs


# --------------------------------------------------------------- episodic


@needs_mamba
def test_episodic_entries_keep_keys_not_hidden_tensors(episodic, tokenizer, random_hidden):
    entry_id = episodic.add(random_hidden(12), segments(tokenizer, "hello", "world"), {'namespace': 'a'})
    entry = episodic.snapshot()[0]
    assert entry['id'] == entry_id and 'embeddings' not in entry
    assert entry['state'].shape == (episodic.ssm.d_state,)
    assert entry['embedding'].shape == (random_hidden(1).shape[-1],)
    assert entry['metadata']['namespace'] == 'a' and entry['metadata']['source'] == 'interaction'
    assert entry['tokens'].tolist() == tokenizer.encode("hello") + tokenizer.encode("world")


@needs_mamba
def test_episodic_overflow_hands_evicted_entry_to_hook(episodic, tokenizer, random_hidden):
    seen = []
    episodic.on_overflow = lambda entries: seen.append(([e['id'] for e in entries], episodic.size()))
    for i in range(episodic.max_entries + 2):
        episodic.add(random_hidden(5), segments(tokenizer, f"q{i}"))
    assert seen == [([0], episodic.max_entries), ([1], episodic.max_entries)]
    assert [e['id'] for e in episodic.snapshot()] == [2, 3, 4, 5]
    assert episodic.stats == {'writes': 6, 'evictions': 2}


@needs_mamba
def test_episodic_retrieve_counts_hits_filters_namespace_and_excludes(episodic, tokenizer, random_hidden):
    a = episodic.add(random_hidden(6), segments(tokenizer, "alpha"), {'namespace': 'a'})
    b = episodic.add(random_hidden(6), segments(tokenizer, "beta"), {'namespace': 'b'})
    hits = episodic.retrieve(random_hidden(6), top_k=5, namespace='a')
    assert [h['id'] for h in hits] == [a] and 'score' in hits[0]
    assert episodic.retrieve(random_hidden(6), top_k=5, exclude_ids=[a, b]) == []
    assert [e['hits'] for e in episodic.snapshot()] == [1, 0]
    assert [e['id'] for e in episodic.candidates(min_hits=1)] == [a]
    episodic.snapshot()[1]['metadata']['important'] = True
    assert [e['id'] for e in episodic.candidates(min_hits=1)] == [a, b]


@needs_mamba
def test_episodic_save_load_round_trip(episodic, tokenizer, random_hidden, tmp_path):
    episodic.add(random_hidden(6), segments(tokenizer, "alpha", "one"))
    episodic.add(random_hidden(6), segments(tokenizer, "beta"))
    path = tmp_path / "episodic.pt"
    episodic.save(str(path))
    episodic.clear()
    assert episodic.load(str(path)) == 2
    assert [e['id'] for e in episodic.snapshot()] == [0, 1]
    assert episodic.add(random_hidden(3), segments(tokenizer, "gamma")) == 2


# --------------------------------------------------------------- semantic


def vectors(n, dim=8):
    seed()
    return torch.randn(n, dim)


@needs_faiss
def make_semantic(dim=8, **kwargs):
    from mantis.memory.semantic import SemanticMemory
    return SemanticMemory(dimension=dim, index_type='Flat', use_gpu=False, **kwargs)


@needs_faiss
def test_semantic_namespace_trust_and_supersession_filters():
    memory = make_semantic()
    vecs = vectors(4)
    ids = memory.add_batch(vecs, ["a-user", "a-model", "global-fact", "b-user"], [
        {'namespace': 'a', 'source': 'user'}, {'namespace': 'a', 'source': 'model'},
        {'namespace': 'global', 'source': 'stage2'}, {'namespace': 'b', 'source': 'user'},
    ])
    texts = lambda **f: sorted(h['text'] for h in memory.retrieve_with_metadata(vecs[0], top_k=10, **f))
    assert texts(namespaces={'a', 'global'}) == ['a-model', 'a-user', 'global-fact']
    assert texts(namespaces={'a', 'global'}, min_trust=1) == ['a-user', 'global-fact']
    assert texts(namespaces={'b'}) == ['b-user']
    assert texts(exclude_ids=[ids[0], ids[3]]) == ['a-model', 'global-fact']

    new_id = memory.supersede(ids[0], vecs[0], "a-user v2", {'namespace': 'a', 'source': 'user'})
    assert memory.entries[ids[0]]['metadata']['superseded_by'] == new_id
    assert memory.entries[new_id]['metadata']['supersedes'] == ids[0]
    assert 'a-user' not in texts() and 'a-user v2' in texts()
    assert memory.retrieve(vecs[0], top_k=1, namespaces={'b'}) == ['b-user']


@needs_faiss
def test_semantic_delete_bounded_overfetch_and_rebuild():
    memory = make_semantic()
    memory._rebuild_in_background = lambda: None  # keep tombstones for the over-fetch check
    vecs = vectors(200)
    ids = memory.add_batch(vecs, [f"t{i}" for i in range(200)])
    assert memory.delete(ids[:150]) == 150
    assert memory.size() == 50 and memory.index.ntotal == 200 and memory._stale == 150
    hits = memory.retrieve_with_metadata(vecs[199], top_k=5)
    assert len(hits) == 5 and all(h['id'] >= 150 for h in hits) and hits[0]['id'] == ids[199]
    assert memory.delete_namespace('nope') == 0

    memory.rebuild()
    assert memory.index.ntotal == 50 and memory._stale == 0 and memory.stats['rebuilds'] == 1
    assert memory.retrieve(vecs[199], top_k=1) == ['t199']


@needs_faiss
def test_semantic_rebuild_keeps_entries_added_during_build():
    memory = make_semantic()
    memory._rebuild_in_background = lambda: None  # test only the explicit rebuild
    vecs = vectors(6)
    memory.add_batch(vecs[:4], ["a", "b", "c", "d"])
    memory.delete([0, 1])
    original_build = memory._build
    added = []

    def build_with_concurrent_write(vectors_, ids_):
        index, trained = original_build(vectors_, ids_)
        added.append(memory.add(vecs[4], "late"))
        memory.delete([2])
        return index, trained

    memory._build = build_with_concurrent_write
    memory.rebuild()
    assert memory.index.ntotal == 3  # c, d, late in the new index; c is a fresh tombstone
    assert memory._stale == 1
    assert memory.retrieve(vecs[4], top_k=1) == ['late']
    assert set(memory.retrieve(vecs[0], top_k=10)) == {'d', 'late'}


@needs_faiss
def test_semantic_recall_fingerprint_and_save_load(tmp_path):
    memory = make_semantic(embedding_fingerprint='abc123')
    vecs = vectors(30)
    memory.add_batch(vecs, [f"t{i}" for i in range(30)], [{'namespace': 'global'}] * 30)
    assert memory.recall_at_k(vecs[:5], k=5) == 1.0
    memory.save(str(tmp_path / "store"))
    from mantis.memory.semantic import SemanticMemory
    loaded = SemanticMemory.load(str(tmp_path / "store"), use_gpu=False)
    assert loaded.embedding_fingerprint == 'abc123' and loaded.size() == 30
    assert loaded.retrieve(vecs[3], top_k=1) == ['t3']
    assert loaded.entries[0]['metadata']['trust'] == 0 and loaded.entries[0]['metadata']['namespace'] == 'global'


@needs_faiss
def test_semantic_recall_applies_projection_once():
    projection = torch.nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        projection.weight.zero_()
        for i in range(8):
            projection.weight[(i + 1) % 8, i] = 1.0
    memory = make_semantic(projection=projection)
    basis = torch.eye(8)
    memory.add_batch(basis, [str(i) for i in range(8)])
    assert memory.recall_at_k(basis, k=1) == 1.0


@needs_faiss
def test_semantic_batch_validation_does_not_mutate_index():
    memory = make_semantic(max_entries=2)
    with pytest.raises(ValueError, match='metadata'):
        memory.add_batch(vectors(2), ['a', 'b'], [{'namespace': 'a'}])
    with pytest.raises(ValueError, match='capacity'):
        memory.add_batch(vectors(3), ['a', 'b', 'c'])
    assert memory.size() == 0 and memory.index.ntotal == 0


@needs_faiss
def test_consolidation_recovers_deduplication_after_restart(tokenizer):
    from mantis.memory.consolidation import MemoryConsolidator
    memory = make_semantic()
    entry = {'id': 7, 'segments': segments(tokenizer, 'a fact'),
             'embedding': vectors(1)[0], 'metadata': {'namespace': 'tenant-a', 'source': 'user'}, 'hits': 1}
    first = MemoryConsolidator(SimpleNamespace(on_overflow=None), memory, tokenizer)
    assert len(first.store_entries([entry])[0]) == 1
    restarted = MemoryConsolidator(SimpleNamespace(on_overflow=None), memory, tokenizer)
    assert len(restarted.store_entries([entry])[0]) == 1
    assert memory.size() == 1 and restarted.stats['duplicates'] == 1
    other_tenant = {**entry, 'metadata': {**entry['metadata'], 'namespace': 'tenant-b'}}
    assert len(restarted.store_entries([other_tenant])[0]) == 1
    assert memory.size() == 2


@needs_faiss
def test_consolidation_keeps_source_when_semantic_checkpoint_fails(tokenizer, tmp_path, monkeypatch):
    from mantis.memory.consolidation import MemoryConsolidator

    memory = make_semantic()
    entry = {'id': 7, 'segments': segments(tokenizer, 'a fact'),
             'embedding': vectors(1)[0], 'metadata': {'namespace': 'tenant-a', 'source': 'user'}, 'hits': 1}

    class Source:
        on_overflow = None
        def __init__(self):
            self.entries = [entry]
        def candidates(self, min_hits):
            return list(self.entries)
        def remove(self, stored):
            self.entries = [e for e in self.entries if e not in stored]
        def size(self):
            return len(self.entries)

    source = Source()
    consolidator = MemoryConsolidator(source, memory, tokenizer, persist_dir=str(tmp_path))
    def fail_save(path):
        raise OSError('disk full')
    monkeypatch.setattr(memory, 'save', fail_save)
    with pytest.raises(OSError, match='disk full'):
        consolidator.consolidate()
    assert source.size() == 1


# ----------------------------------------------------------- consolidation


@needs_mamba
@needs_faiss
def test_consolidation_preserves_every_evicted_entry(episodic, semantic, tokenizer, random_hidden, tmp_path):
    from mantis.memory.consolidation import MemoryConsolidator
    episodic.max_entries = 2
    consolidator = MemoryConsolidator(episodic, semantic, tokenizer, persist_dir=str(tmp_path))
    episodic.add(random_hidden(8), segments(tokenizer, "the appointment is Tuesday"), {'namespace': 'a', 'source': 'user'})
    episodic.add(random_hidden(8), segments(tokenizer, "the appointment moved to Wednesday"), {'namespace': 'a', 'source': 'user'})
    episodic.add(random_hidden(8), segments(tokenizer, "unrelated"))
    episodic.add(random_hidden(8), segments(tokenizer, "unrelated"))  # distinct source event with the same text
    stored = sorted(e['text'] for e in semantic.entries.values())
    assert stored == ["the appointment is Tuesday", "the appointment moved to Wednesday"]
    metas = [e['metadata'] for e in semantic.entries.values()]
    assert {m['origin'] for m in metas} == {'episodic:0', 'episodic:1'}
    assert all(m['namespace'] == 'a' and m['source'] == 'user' and m['trust'] == 2 for m in metas)
    assert consolidator.stats['overflow_stored'] == 2
    assert os.path.exists(tmp_path / "episodic.pt") and os.path.exists(tmp_path / "semantic.meta")

    # periodic cycle: only entries retrieval found useful leave the buffer
    episodic.retrieve(random_hidden(8), top_k=1)
    report = consolidator.consolidate()
    assert report['consolidated'] == 1 and episodic.size() == 1 and semantic.size() == 3
    episodic.retrieve(random_hidden(8), top_k=1)
    consolidator.consolidate()
    assert episodic.size() == 0 and semantic.size() == 4 and consolidator.stats['duplicates'] == 0


@needs_mamba
@needs_faiss
def test_consolidation_queue_drains_and_stop_flushes(episodic, semantic, tokenizer, random_hidden):
    import time
    from mantis.memory.consolidation import MemoryConsolidator
    consolidator = MemoryConsolidator(episodic, semantic, tokenizer, consolidation_interval=3600)
    episodic.add(random_hidden(4), segments(tokenizer, "first"))
    consolidator._queue.put_nowait(list(episodic.snapshot()))  # queued while no thread runs
    consolidator.stop()
    assert [e['text'] for e in semantic.entries.values()] == ["first"]

    consolidator.start()
    assert consolidator.running
    episodic.max_entries = 1
    episodic.add(random_hidden(4), segments(tokenizer, "second"))  # evicts "first" (a duplicate text)
    episodic.add(random_hidden(4), segments(tokenizer, "third"))   # evicts "second"
    deadline = time.time() + 5
    while semantic.size() < 2 and time.time() < deadline:
        time.sleep(0.05)
    consolidator.stop()
    assert not consolidator.running
    assert sorted(e['text'] for e in semantic.entries.values()) == ["first", "second"]
    stats = consolidator.get_stats()
    assert stats['queue_pending'] == 0 and stats['duplicates'] == 1 and stats['overflow_stored'] == 3
