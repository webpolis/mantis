import torch

from mantis.inference.generation import generate_tokens
from tests.conftest import DEVICE, seed


def greedy(base, prompt, n, **kwargs):
    return list(generate_tokens(base, prompt, n, temperature=0.0, **kwargs))


def prompt_ids(n):
    seed()
    return torch.randint(4, 300, (n,)).tolist()


@torch.no_grad()
def test_prefill_reuse_matches_fresh_prefill(base):
    prompt = prompt_ids(10)
    fresh = greedy(base, prompt, 5)
    out = base(torch.tensor([prompt], device=DEVICE), use_cache=True)
    reused = greedy(base, prompt, 5, prefill=(out['past_key_values'], out['logits'][0, -1]))
    assert [t for t, _ in fresh] == [t for t, _ in reused]
    assert len(fresh) >= 1


def expected_rows(prompt, tokens, eos):
    return len(prompt) + len(tokens) - (1 if tokens and tokens[-1] == eos else 0)


@torch.no_grad()
def test_hidden_out_matches_full_forward(base, tokenizer):
    prompt = prompt_ids(10)
    eos = tokenizer.eos_token_id
    for use_prefill in (False, True):
        rows = []
        prefill = None
        if use_prefill:
            out = base(torch.tensor([prompt], device=DEVICE), use_cache=True, return_hidden=True)
            prefill = (out['past_key_values'], out['logits'][0, -1])
            rows = list(out['last_hidden'][0])
        tokens = [t for t, _ in greedy(base, prompt, 4, stop_ids=[eos], prefill=prefill, hidden_out=rows)]
        assert len(rows) == expected_rows(prompt, tokens, eos)
        full = base(torch.tensor([prompt + tokens], device=DEVICE), return_hidden=True)['last_hidden'][0]
        assert (torch.stack(rows) - full[:len(rows)]).abs().max().item() < 1e-4


@torch.no_grad()
def test_window_rollover_keeps_hidden_out_aligned(base, tokenizer):
    window = base.max_seq_len
    prompt = prompt_ids(window - 4)
    rows = []
    tokens = [t for t, _ in greedy(base, prompt, 12, stop_ids=[tokenizer.eos_token_id], hidden_out=rows)]
    assert len(prompt) + len(tokens) > window  # the cache filled and rolled over
    assert len(rows) == expected_rows(prompt[-window:], tokens, tokenizer.eos_token_id)
    assert all(r.shape == (base.d_model,) for r in rows)
