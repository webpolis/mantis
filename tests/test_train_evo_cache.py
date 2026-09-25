import numpy as np
import torch

from mantis.data import iter_documents
from mantis.tokenizer import MANTISTokenizer
from train_evo import EvoWorldDataset


def _reference_chunks(path, tokenizer, seq_len):
    chunks = []
    for text in iter_documents(str(path)):
        tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
        weights = tokenizer.compute_loss_weights(torch.tensor(tokens)).numpy()
        for start in range(0, len(tokens) - 1, seq_len):
            real = min(seq_len, len(tokens) - start - 1)
            ids = np.full(seq_len, tokenizer.pad_token_id, dtype=np.int64)
            labels = np.full(seq_len, -100, dtype=np.int64)
            loss_weights = np.zeros(seq_len, dtype=np.float16)
            ids[:real] = tokens[start:start + real]
            labels[:real] = tokens[start + 1:start + real + 1]
            loss_weights[:real] = weights[start:start + real]
            chunks.append((ids, labels, loss_weights.astype(np.float32)))
    return chunks


def test_evo_cache_preserves_world_chunks_and_reuses_tokens(tmp_path, monkeypatch):
    source = tmp_path / 'worlds.txt'
    source.write_text('=EPOCH one\n@BIO first\n\n@SP two\n\n@INT three\n', encoding='utf-8')
    tokenizer = MANTISTokenizer()
    cache_dir = tmp_path / 'cache'

    expected = _reference_chunks(source, tokenizer, seq_len=7)
    dataset = EvoWorldDataset(source, tokenizer, seq_len=7, cache_dir=cache_dir, verbose=False)
    assert len(dataset.worlds) == 3
    assert len(dataset) == len(expected)
    for i, (ids, labels, weights) in enumerate(expected):
        item = dataset[i]
        np.testing.assert_array_equal(item['input_ids'].numpy(), ids)
        np.testing.assert_array_equal(item['labels'].numpy(), labels)
        np.testing.assert_array_equal(item['loss_weights'].numpy(), weights)

    train, val = dataset.split_by_world(0.34)
    assert set(train.worlds).isdisjoint(val.worlds)
    assert len(train) + len(val) == len(dataset)
    assert train.tokens.filename == val.tokens.filename

    def no_encode(_):
        raise AssertionError('A cache hit must not tokenize the source again')

    monkeypatch.setattr(tokenizer, 'encode', no_encode)
    reused = EvoWorldDataset(source, tokenizer, seq_len=5, cache_dir=cache_dir, verbose=False)
    assert len(reused) >= len(dataset)


def test_evo_cache_rebuilds_when_source_changes(tmp_path):
    source = tmp_path / 'worlds.txt'
    source.write_text('@BIO one\n', encoding='utf-8')
    tokenizer = MANTISTokenizer()
    cache_dir = tmp_path / 'cache'
    original = EvoWorldDataset(source, tokenizer, seq_len=8, cache_dir=cache_dir, verbose=False)
    original_tokens = original.real_tokens

    with source.open('a', encoding='utf-8') as f:
        f.write('\n@SP a second, longer world\n')
    changed = EvoWorldDataset(source, tokenizer, seq_len=8, cache_dir=cache_dir, verbose=False)
    assert len(changed.worlds) == 2
    assert changed.real_tokens > original_tokens
    assert changed.tokens.filename != original.tokens.filename
