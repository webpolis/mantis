import pytest
import torch

from mantis.tokenizer import BPETokenizer, MANTISTokenizer, load_tokenizer

CORPUS = ["Once upon a time, there was a little girl named Lily.\nShe had 12 apples.",
          'Tom said: "héllo wörld" 🙂 and left.'] * 20


@pytest.fixture(scope='module')
def bpe():
    return BPETokenizer.train(CORPUS, vocab_size=400)


def test_round_trips_any_text_and_never_emits_unk(bpe):
    for text in CORPUS[:2] + ["unseen ∑ symbols\ttabs\n\n", ""]:
        ids = bpe.encode(text)
        assert bpe.decode(ids) == text
        assert bpe.unk_token_id not in ids
    assert 256 + 4 < len(bpe) <= 400  # bytes and specials plus the merges the corpus supports
    assert bpe.encode(CORPUS[:2]) == [bpe.encode(t) for t in CORPUS[:2]]


def test_special_tokens_are_shared_with_the_trie_tokenizer(bpe):
    trie = MANTISTokenizer()
    assert (bpe.pad_token_id, bpe.eos_token_id, bpe.bos_token_id, bpe.unk_token_id) == \
        (trie.pad_token_id, trie.eos_token_id, trie.bos_token_id, trie.unk_token_id)
    ids = bpe.encode("hi", add_special_tokens=True, max_length=6, padding=True)
    assert ids[0] == bpe.bos_token_id and ids[-1] == bpe.pad_token_id and len(ids) == 6
    assert bpe.decode(torch.tensor(ids)) == "hi"
    assert set(bpe.non_generable_ids) == {bpe.pad_token_id, bpe.bos_token_id, bpe.unk_token_id}


def test_stream_decoder_completes_multibyte_characters(bpe):
    text = "wörld 🙂!"
    decoder = bpe.stream_decoder()
    pieces = [decoder.push(t) for t in bpe.encode(text)]
    assert "".join(pieces) == text
    assert all("�" not in piece for piece in pieces)


def test_save_load_keeps_ids_and_fingerprint(bpe, tmp_path):
    bpe.save(str(tmp_path))
    loaded = load_tokenizer(str(tmp_path))
    assert isinstance(loaded, BPETokenizer)
    assert loaded.fingerprint() == bpe.fingerprint()
    assert loaded.encode(CORPUS[1]) == bpe.encode(CORPUS[1])
    assert BPETokenizer.train(CORPUS, vocab_size=300).fingerprint() != bpe.fingerprint()

    trie = MANTISTokenizer()
    trie.save(str(tmp_path / "trie"))
    assert isinstance(load_tokenizer(str(tmp_path / "trie")), MANTISTokenizer)
