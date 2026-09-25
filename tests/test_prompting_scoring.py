from mantis.inference.prompting import (EVIDENCE_CLOSE, EVIDENCE_OPEN, MIN_ITEM_TOKENS, build_prompt_ids,
                                        evidence_block, format_query, select_evidence, word_f1)
from mantis.training.scoring import answer_correct, has_negation, normalize_answer, score_answer


def item(id_, tier, text, dense=0.5, source='user'):
    return {'id': id_, 'tier': tier, 'origin': id_, 'text': text, 'dense': dense, 'source': source}


def test_select_evidence_shares_budget_between_tiers(tokenizer):
    long_episodic = item('E1', 'episodic', "recent context " * 200)
    fact = item('S1', 'semantic', "Paris is the capital of France.")
    selected = select_evidence([long_episodic, fact], "capital of France", 120, tokenizer)
    assert {i['tier'] for i in selected} == {'episodic', 'semantic'}
    assert sum(len(i['ids']) for i in selected) <= 120
    assert len(long_episodic['ids']) <= 60  # truncated to its share


def test_select_evidence_zero_budget_and_dedupe(tokenizer):
    assert select_evidence([item('E1', 'episodic', "x")], "x", 0, tokenizer) == []
    dup = [item('S1', 'semantic', "same text"), item('S2', 'semantic', "same text ")]
    selected = select_evidence(dup, "same", 500, tokenizer)
    assert [i['id'] for i in selected] == ['S1']


def test_select_evidence_truncates_or_skips_items_that_do_not_fit(tokenizer):
    big = item('S1', 'semantic', "y" * 400)
    small = item('S2', 'semantic', "short", dense=0.0)
    selected = select_evidence([big, small], "q", 30, tokenizer)
    assert [i['id'] for i in selected] == ['S1'] and len(big['ids']) == 30
    assert select_evidence([item('S1', 'semantic', "y" * 400)], "q", MIN_ITEM_TOKENS - 1, tokenizer) == []


def test_prompt_formats(tokenizer):
    assert format_query("hi", 'raw') == "hi"
    assert format_query("hi", 'chat') == "User: hi\nAssistant:"
    items = [item('S1', 'semantic', "fact", source='model')]
    items[0]['verified'] = False
    block = evidence_block(items, 'chat')
    assert block.startswith(EVIDENCE_OPEN) and EVIDENCE_CLOSE in block and "model, unverified" in block
    query_ids = tokenizer.encode("q")
    assert build_prompt_ids([], query_ids, tokenizer, 'chat') == query_ids
    items[0]['ids'] = tokenizer.encode("x")
    assert build_prompt_ids(items, query_ids, tokenizer, 'raw')[-len(query_ids):] == query_ids


def test_trace_prompt_preserves_evidence_budget(tokenizer):
    items = select_evidence([item('E1', 'episodic', '@SP S1 predator 99\n' * 30)],
                            '@SP S1', 40, tokenizer)
    query_ids = tokenizer.encode('=EPOCH 1 1000 W7\n')
    prompt_ids = build_prompt_ids(items, query_ids, tokenizer, 'trace')
    assert len(prompt_ids) <= 40 + len(query_ids)
    assert prompt_ids[-len(query_ids):] == query_ids
    assert '[E1' not in tokenizer.decode(prompt_ids)
    assert tokenizer.decode(prompt_ids[:-len(query_ids)]).endswith('---\n')


def test_word_f1():
    assert word_f1("a b c", "a b c") == 1.0
    assert word_f1("", "") == 1.0
    assert word_f1("a", "") == 0.0
    assert abs(word_f1("a b", "a c") - 0.5) < 1e-9


def test_scoring_probes_from_review():
    assert answer_correct("Paris is NOT the capital of France", "Paris") is False
    assert answer_correct("Paris not capital", "Paris") is False
    assert answer_correct("The capital of France is Paris.", "Paris") is True
    assert answer_correct("I do not know", "Paris") is False
    assert score_answer("anything about Paris", "Paris", abstained=True) is None
    assert score_answer("Paris", "Paris", abstained=False) is True
    # F1 path: no exact phrase but heavy overlap
    assert answer_correct("plants convert sunlight into energy", "Plants convert sunlight into energy using chlorophyll") is True
    assert has_negation("that isn't right") and not has_negation("Paris")
    assert normalize_answer("The Capital, of France!") == "capital of france"
