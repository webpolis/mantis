from types import SimpleNamespace

import pytest
import torch

import inference_evo


class FakeFullEngine:
    def __init__(self, tokenizer):
        self.base = SimpleNamespace(max_seq_len=120, count_parameters=lambda: {'total': 3})
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(prompt_format='raw', route_policy='learned')
        self.episodic = object()
        self.semantic = object()
        self.critic = object()
        self.calls = []
        self.ingests = []
        self.closed = False

    def ingest(self, *args, **kwargs):
        self.ingests.append((args, kwargs))

    def generate(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if len(self.calls) == 1:
            return {'response': '@SP S2 predator 12\n', 'stop_token_id': self.tokenizer.vocab['---'],
                    'abstained': False, 'verified': True, 'path': 'full'}
        return {'response': "I can't verify this tick", 'stop_token_id': None,
                'abstained': True, 'verified': True, 'path': 'full'}

    def close(self):
        self.closed = True


def test_evolution_uses_full_engine_for_each_tick(tokenizer, monkeypatch):
    full = FakeFullEngine(tokenizer)
    monkeypatch.setattr(inference_evo.MANTISInferenceEngine, 'from_checkpoints',
                        lambda **kwargs: full)
    engine = inference_evo.EvoInferenceEngine('base.pt', device='cpu', policy_checkpoint='policy.pt')
    trace = '=EPOCH 1 1000 W7\n' + '@SP S1 predator 99\n---\n' * 10
    ticks = list(engine.continue_trace(trace, max_ticks=3, max_tokens_per_tick=30,
                                       top_k=11, namespace='world7'))

    assert ticks == ['@SP S2 predator 12\n---']
    assert len(full.ingests) == 1 and full.ingests[0][1]['namespace'] == 'world7'
    assert len(full.calls) == 2
    assert full.calls[0][1]['stop_ids'] == [tokenizer.vocab['---']]
    assert full.calls[0][1]['top_k'] == 11
    assert full.calls[1][1]['namespace'] == 'world7'
    assert full.calls[1][0].endswith(ticks[0])
    assert engine.last_result['abstained']
    engine.close()
    assert full.closed


def test_full_evolution_rejects_incomplete_artifacts(tokenizer, monkeypatch):
    full = FakeFullEngine(tokenizer)
    full.critic = None
    monkeypatch.setattr(inference_evo.MANTISInferenceEngine, 'from_checkpoints',
                        lambda **kwargs: full)
    with pytest.raises(ValueError, match='requires critic checkpoint'):
        inference_evo.EvoInferenceEngine('base.pt', device='cpu', policy_checkpoint='policy.pt')
    assert full.closed


def test_evolution_requires_policy_for_full_engine_options():
    with pytest.raises(ValueError, match='policy checkpoint is required'):
        inference_evo.EvoInferenceEngine('base.pt', device='cpu', critic_checkpoint='critic.pt')


def test_full_evolution_routes_retrieves_verifies_and_remembers(make_engine, critic, tokenizer, monkeypatch):
    import mantis.inference.generation as generation

    class Episodic:
        max_entries = 10

        def __init__(self):
            self.ssm = torch.nn.Linear(1, 1)
            self.writes = []

        def size(self):
            return 1

        def retrieve(self, *args):
            return [{'id': 0, 'segments': [('response', tokenizer.encode('x'))],
                     'score': 0.9, 'metadata': {'source': 'interaction', 'verified': True, 'trust': 1}}]

        def add(self, hidden, segments, metadata):
            self.writes.append((segments, metadata))

    class Semantic:
        def size(self):
            return 1

        def retrieve_with_metadata(self, *args, **kwargs):
            return [{'id': 0, 'text': 'y', 'distance': 0.1,
                     'metadata': {'source': 'document', 'trust': 2}}]

    episodic = Episodic()
    full = make_engine(episodic=episodic, semantic=Semantic(), critic=critic,
                       route_policy='always', verification_confidence_threshold=0.0)
    monkeypatch.setattr(inference_evo.MANTISInferenceEngine, 'from_checkpoints',
                        lambda **kwargs: full)
    tokens = iter([tokenizer.vocab['@SP'], tokenizer.vocab['---']])
    monkeypatch.setattr(generation, 'sample_next_token', lambda *args, **kwargs: next(tokens))

    engine = inference_evo.EvoInferenceEngine('base.pt', device=full.device,
                                               policy_checkpoint='policy.pt')
    tick = engine.generate_tick('W0\n', max_tokens=4, namespace='world')
    result = engine.last_result

    assert tick == '@SP---'
    assert result['path'] == 'full' and result['verified']
    assert result['memory_used'] == {'episodic': True, 'semantic': True}
    assert result['evidence'] and result['cost']['critic_tokens'] > 0
    assert episodic.writes[0][1]['namespace'] == 'world'
    assert full.config.prompt_format == 'trace'
