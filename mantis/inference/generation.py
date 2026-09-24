"""
Shared autoregressive decoding for every MANTIS entry point.

One loop handles the KV cache, the attention window, banned token IDs and
sampling filters. When the cache fills the window, the most recent half-window
is re-encoded from scratch, so every prediction sees exactly the fresh context
a training sequence would (no hidden state built from evicted tokens).
"""

from typing import Iterator, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Pick the next token from (vocab,) logits. temperature <= 0 means greedy."""
    if temperature <= 0:
        return int(logits.argmax())

    logits = logits / temperature

    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits = logits.masked_fill(logits < kth, float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.masked_fill(
            torch.zeros_like(remove).scatter(0, sorted_indices, remove), float('-inf')
        )

    return int(torch.multinomial(F.softmax(logits, dim=-1), num_samples=1))


@torch.no_grad()
def generate_tokens(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    banned_ids: Optional[Sequence[int]] = None,
    stop_ids: Sequence[int] = (),
    expert_weights: Optional[torch.Tensor] = None,
) -> Iterator[Tuple[int, float]]:
    """
    Yield (token_id, log_prob) pairs, one per generated token.

    `log_prob` is the token's log-probability under the model's untempered
    distribution (with banned IDs removed), usable as a confidence signal.
    Generation stops after yielding a token in `stop_ids`.
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must contain at least one token")

    device = next(model.parameters()).device
    window = model.max_seq_len
    context = list(prompt_ids)[-window:]
    stop = set(stop_ids)
    banned = torch.tensor(list(banned_ids or []), dtype=torch.long, device=device)

    pending = context
    past = None

    for _ in range(max_new_tokens):
        if past is not None and model.cache_len(past) + len(pending) > window:
            pending = context[-max(1, window // 2):]
            past = None

        output = model(
            torch.tensor([pending], dtype=torch.long, device=device),
            expert_weights=expert_weights,
            past_key_values=past,
            use_cache=True,
        )
        past = output['past_key_values']

        logits = output['logits'][0, -1].float()
        if banned.numel():
            logits[banned] = float('-inf')

        token = sample_next_token(logits, temperature, top_k, top_p)
        yield token, float(F.log_softmax(logits, dim=-1)[token])

        if token in stop:
            return
        context = (context + [token])[-window:]
        pending = [token]
