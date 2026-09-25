import os

import pytest
import torch

from mantis.inference.engine import ABSTENTION, MANTISInferenceEngine
from mantis.inference.prompting import build_prompt_ids, select_evidence
from mantis.models.meta_controller import GATES
from tests.conftest import DEVICE, needs_faiss, needs_mamba, seed


@pytest.fixture
def forward_lengths(base, monkeypatch):
    """Record the input length of every backbone forward call."""
    lengths = []
    original = base.forward

    def recording_forward(input_ids, *args, **kwargs):
        lengths.append(input_ids.shape[1])
        return original(input_ids, *args, **kwargs)

    monkeypatch.setattr(base, 'forward', recording_forward)
    return lengths


QUERY = "hello"


def test_bypass_encodes_query_once(make_engine, tokenizer, forward_lengths):
    engine = make_engine(route_policy='bypass', max_length=2)
    n = len(tokenizer.encode(QUERY))
    result = engine.generate(QUERY)
    assert result['path'] == 'bypass' and result['num_tokens'] == 2
    assert forward_lengths == [n, 1]  # not [n, n]: the query prefill is reused
    assert result['cost']['backbone_tokens'] == n + 2 and result['cost']['query_tokens'] == n


def test_full_path_without_evidence_reuses_prefill(make_engine, tokenizer, forward_lengths):
    engine = make_engine(route_policy='never', max_length=2)
    n = len(tokenizer.encode(QUERY))
    result = engine.generate(QUERY)
    assert result['path'] == 'full' and result['evidence'] == [] and forward_lengths == [n, 1]


def test_oracle_evidence_reprefills_and_is_reported(make_engine, tokenizer, forward_lengths):
    engine = make_engine(route_policy='never', max_length=2)
    n = len(tokenizer.encode(QUERY))
    result = engine.generate(QUERY, evidence=["Paris is the capital of France."])
    assert result['path'] == 'full'
    assert [e['id'] for e in result['evidence']] == ['O0'] and result['evidence'][0]['tier'] == 'oracle'
    assert forward_lengths[0] == n and forward_lengths[1] > n and forward_lengths[2] == 1


def test_query_filling_window_leaves_no_evidence_budget(make_engine, base):
    engine = make_engine(route_policy='never', max_length=2)
    long_query = "a" * (base.max_seq_len + 5)
    result = engine.generate(long_query, evidence=["fact"])
    assert result['evidence'] == [] and 'ingest' not in result['timings']


def test_evidence_budget_includes_prompt_wrapper(make_engine, base, tokenizer, monkeypatch):
    monkeypatch.setattr(base, 'max_seq_len', 128)
    engine = make_engine(prompt_format='chat', max_length=1)
    query_ids = tokenizer.encode('User: question\nAssistant:')
    candidates = [{'id': 'O0', 'tier': 'oracle', 'text': 'fact ' * 100,
                   'source': 'external', 'dense': 1.0}]
    selected = select_evidence(candidates, 'question', engine._budget(len(query_ids), 1), tokenizer)
    assert selected
    assert len(build_prompt_ids(selected, query_ids, tokenizer, 'chat')) <= base.max_seq_len - 1


def test_abstention_semantics(make_engine, critic):
    engine = make_engine(critic=critic, route_policy='always', verification_confidence_threshold=1.01)
    result = engine.generate(QUERY)
    assert result['abstained'] and result['response'] == ABSTENTION
    assert result['confidence'] == 0.0 and result['confidence_source'] == 'abstained'
    assert result['num_tokens'] == 0 and result['verified'] and result['verification_rounds'] == 0
    assert QUERY not in result['response']
    stats = engine.get_stats()
    assert stats['abstention_rate'] == 1.0 and stats['verification_rate'] == 1.0 and 'latency_p95' in stats


def test_verified_answer_reports_critic_score(make_engine, critic):
    engine = make_engine(critic=critic, route_policy='always', verification_confidence_threshold=0.0)
    result = engine.generate(QUERY)
    assert not result['abstained'] and result['confidence_source'] == 'critic'
    assert result['confidence'] == result['critic_score'] and 0.0 <= result['critic_score'] <= 1.0
    assert result['cost']['critic_tokens'] > 0 and result['timings']['verify'] > 0
    plain = make_engine(route_policy='never').generate(QUERY)
    assert plain['confidence_source'] == 'token_likelihood' and plain['confidence'] == plain['token_likelihood']


def test_custom_stop_token_is_reported_outside_response(make_engine, tokenizer, monkeypatch):
    import mantis.inference.generation as generation

    tokens = iter([tokenizer.vocab['@SP'], tokenizer.vocab['---']])
    monkeypatch.setattr(generation, 'sample_next_token', lambda *args, **kwargs: next(tokens))
    engine = make_engine(route_policy='never')
    result = engine.generate('=EPOCH 1 1000 W0\n', max_length=4, top_k=7,
                             stop_ids=[tokenizer.vocab['---']])
    assert result['response'] == '@SP'
    assert result['stop_token_id'] == tokenizer.vocab['---']
    assert result['num_tokens'] == 1


def test_action_mask_and_expert_bias(make_engine, critic):
    engine = make_engine(critic=critic, route_policy='bypass')
    rollout = engine.generate(QUERY, return_details=True)['rollout']
    assert rollout['action_mask'].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]

    result = make_engine(critic=critic, route_policy='never').generate(QUERY, return_details=True)
    assert result['rollout']['action_mask'].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert not result['expert_bias_applied']

    biased = make_engine(critic=critic, route_policy='never', expert_bias=True)
    result = biased.generate(QUERY, return_details=True)
    assert result['expert_bias_applied'] and result['rollout']['action_mask'].tolist() == [1.0, 0.0, 0.0, 1.0, 1.0]
    assert result['rollout']['expert_raw'].shape == (biased.base.n_layers * biased.base.n_experts,)
    assert not biased.generate(QUERY, force_gates={'bypass': True}, return_details=True)['expert_bias_applied']


def test_route_policies_and_force_gates(make_engine, critic):
    always = make_engine(critic=critic, route_policy='always').generate(QUERY, return_details=True)
    assert always['routing_decisions'] == {'bypass': False, 'episodic': False, 'semantic': False, 'verification': True}
    assert always['path'] == 'full'
    never = make_engine(critic=critic, route_policy='never').generate(QUERY, return_details=True)
    assert never['routing_decisions'] == {g: False for g in GATES} and not never['verified']
    learned = make_engine(critic=critic, bypass_uncertainty_threshold=0.0)
    forced = learned.generate(QUERY, force_gates={'bypass': True}, return_details=True)
    assert forced['path'] == 'bypass'
    verified = learned.generate(QUERY, force_gates={'verification': True}, return_details=True)
    assert verified['path'] == 'full' and verified['verified']
    assert verified['routing_decisions'] == {'bypass': False, 'episodic': False, 'semantic': False, 'verification': True}


@needs_mamba
@needs_faiss
def test_namespace_isolation(make_engine, episodic, semantic, base):
    engine = make_engine(episodic=episodic, semantic=semantic, verification_confidence_threshold=0.0)
    engine.ingest("The alpha project ships on Tuesday.", namespace='a', source='user')
    seed()
    semantic.add(torch.randn(base.d_model), "shared global fact", {'namespace': 'global', 'source': 'stage2'})
    semantic.add(torch.randn(base.d_model), "private fact of a", {'namespace': 'a', 'source': 'user'})
    gates = {'episodic': True, 'semantic': True}

    from_b = engine.generate("alpha project?", namespace='b', force_gates=gates, max_length=1)
    assert [e['tier'] for e in from_b['evidence']] == ['semantic']
    assert from_b['evidence'][0]['id'] == 'S0'  # the global fact only
    from_a = engine.generate("alpha project?", namespace='a', force_gates=gates, max_length=1)
    assert {e['tier'] for e in from_a['evidence']} == {'episodic', 'semantic'}
    assert from_a['memory_used'] == {'episodic': True, 'semantic': True}
    facts_for_a = engine.generate("alpha project?", namespace='a', force_gates={'semantic': True}, max_length=1)
    assert {e['id'] for e in facts_for_a['evidence']} == {'S0', 'S1'}


@needs_mamba
def test_untrusted_model_claims_are_not_cited_as_facts(make_engine, episodic, base):
    if not pytest.importorskip('faiss'):
        return
    from mantis.memory.semantic import SemanticMemory
    semantic = SemanticMemory(dimension=base.d_model, index_type='Flat', use_gpu=False)
    seed()
    semantic.add(torch.randn(base.d_model), "unverified model claim", {'namespace': 'default', 'source': 'model'})
    engine = make_engine(episodic=episodic, semantic=semantic)
    result = engine.generate("claim?", force_gates={'semantic': True}, max_length=1)
    assert result['evidence'] == []
    engine.config.min_evidence_trust = 0
    result = engine.generate("claim?", force_gates={'semantic': True}, max_length=1)
    assert [e['source'] for e in result['evidence']] == ['model']


@needs_mamba
def test_memory_writes_provenance_and_frozen_memory(make_engine, episodic, critic):
    engine = make_engine(episodic=episodic, critic=critic, route_policy='never')
    engine.generate("remember this", namespace='u')
    assert episodic.size() == 1
    entry = episodic.snapshot()[0]
    assert entry['metadata']['namespace'] == 'u' and entry['metadata']['source'] == 'interaction'
    assert entry['metadata']['verified'] is False and entry['metadata']['trust'] == 0
    assert [role for role, _ in entry['segments']] == ['query', 'response']

    with engine.frozen_memory():
        engine.generate("not stored")
    assert episodic.size() == 1

    engine.config.route_policy = 'always'
    engine.config.verification_confidence_threshold = 1.01
    engine.generate("rejected draft")
    assert episodic.size() == 1  # abstentions are never remembered
    engine.config.verification_confidence_threshold = 0.0
    engine.generate("verified answer")
    assert episodic.size() == 2 and episodic.snapshot()[1]['metadata']['trust'] == 1


@needs_faiss
def test_long_query_prefix_is_ingested(make_engine, base, tokenizer):
    from mantis.memory.semantic import SemanticMemory
    semantic = SemanticMemory(dimension=base.d_model, index_type='Flat', use_gpu=False)
    engine = make_engine(semantic=semantic, route_policy='never', max_length=1)
    query = "word " * (3 * base.max_seq_len)
    result = engine.generate(query, namespace='doc')
    assert semantic.size() >= 2 and 'ingest' in result['timings']
    assert all(e['metadata']['namespace'] == 'doc' and e['metadata']['source'] == 'user' for e in semantic.entries.values())
    assert engine.ingest("short document", namespace='doc') == 1


@needs_faiss
def test_from_checkpoints_rejects_foreign_semantic_store(config, base, tokenizer, tmp_path):
    from mantis.memory.semantic import SemanticMemory
    from mantis.utils.checkpoints import model_fingerprint, save_training_checkpoint
    ckpt = str(tmp_path / "base.pt")
    save_training_checkpoint(ckpt, model=base, config=config, tokenizer=tokenizer,
                             epoch=0, epoch_step=0, optimizer_step=0, global_step=0)
    seed()
    foreign = SemanticMemory(dimension=base.d_model, index_type='Flat', use_gpu=False, embedding_fingerprint='deadbeef')
    foreign.add(torch.randn(base.d_model), "x")
    foreign.save(str(tmp_path / "foreign"))
    with pytest.raises(ValueError, match="different base model"):
        MANTISInferenceEngine.from_checkpoints(base_checkpoint=ckpt, semantic_store=str(tmp_path / "foreign"), device=DEVICE)

    own = SemanticMemory(dimension=base.d_model, index_type='Flat', use_gpu=False,
                         embedding_fingerprint=model_fingerprint(base, tokenizer))
    own.save(str(tmp_path / "own"))
    engine = MANTISInferenceEngine.from_checkpoints(base_checkpoint=ckpt, semantic_store=str(tmp_path / "own"),
                                                    device=DEVICE, memory_dir=str(tmp_path / "state"))
    assert engine.semantic.size() == 0 and engine.consolidator is None
    engine.close()
    assert os.path.exists(tmp_path / "state" / "semantic.meta")


@needs_faiss
def test_memory_benchmark_does_not_leave_synthetic_facts(make_engine, base, tokenizer, monkeypatch):
    from evaluation import make_memory_dataset
    from evaluation.memory_bench import MemoryRecallRunner
    from mantis.memory.semantic import SemanticMemory
    import evaluation.memory_bench as memory_bench

    monkeypatch.setattr(memory_bench, 'MAX_NEW_TOKENS', 1)
    semantic = SemanticMemory(dimension=base.d_model, index_type='Flat', use_gpu=False)
    engine = make_engine(semantic=semantic, route_policy='always')
    dataset = make_memory_dataset(n_sessions=1, facts_per_session=1, updates=0, distractors=0)
    result = MemoryRecallRunner(engine, tokenizer, device=DEVICE).run(dataset)
    assert result['num_examples'] == len(dataset['questions'])
    assert semantic.size() == 0
    assert engine.config.remember


def test_checkpoint_fingerprint_covers_non_embedding_weights(base, tokenizer):
    from mantis.utils.checkpoints import model_fingerprint
    before = model_fingerprint(base, tokenizer)
    key = next(name for name in base.state_dict() if 'embed' not in name)
    tensor = base.state_dict()[key]
    original = tensor.view(-1)[0].clone()
    with torch.no_grad():
        tensor.view(-1)[0] += 1
    try:
        assert model_fingerprint(base, tokenizer) != before
    finally:
        with torch.no_grad():
            tensor.view(-1)[0].copy_(original)


def test_from_checkpoints_rejects_foreign_stage_artifacts(config, base, tokenizer, tmp_path):
    from mantis.utils.checkpoints import save_training_checkpoint
    ckpt = str(tmp_path / 'base.pt')
    save_training_checkpoint(ckpt, model=base, config=config, tokenizer=tokenizer,
                             epoch=0, epoch_step=0, optimizer_step=0, global_step=0)
    for kind, field, artifact in (
        ('Policy', 'policy_checkpoint', {'meta_controller_state_dict': {}}),
        ('Episodic', 'memory_checkpoint', {'episodic_ssm_state_dict': {}}),
        ('Critic', 'critic_checkpoint', {'critic_state_dict': {}}),
    ):
        path = str(tmp_path / f'{kind.lower()}.pt')
        torch.save({**artifact, 'embedding_fingerprint': 'foreign',
                    'tokenizer_fingerprint': tokenizer.fingerprint()}, path)
        with pytest.raises(ValueError, match=f'{kind} .*different base model'):
            MANTISInferenceEngine.from_checkpoints(base_checkpoint=ckpt, device=DEVICE, **{field: path})
