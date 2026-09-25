"""
Prompt construction shared by the inference engine, Stage 5 fine-tuning and
memory rendering: query formatting, evidence blocks with source identifiers,
and evidence selection under an explicit token budget.

`raw` feeds the query as-is (Stage 1 models); `chat` wraps it in conversation
roles and puts evidence in a delimited block (the format Stage 5 trains the
generator on). `trace` prepends retrieved history as plain protocol text for
the evolution model. Retrieved text is data, never instructions.
"""

from collections import Counter
from typing import Dict, List, Sequence

EVIDENCE_OPEN, EVIDENCE_CLOSE = "<evidence>", "</evidence>"
MIN_ITEM_TOKENS = 16


def format_query(query: str, prompt_format: str) -> str:
    if prompt_format == 'chat':
        return f"User: {query}\nAssistant:"
    return query


def render_entry(entry: Dict, tokenizer) -> str:
    """Text of an episodic entry: its role segments, one per line."""
    return "\n".join(tokenizer.decode(ids).strip() for _, ids in entry['segments']).strip()


def render_evidence(item: Dict) -> str:
    """One evidence line with its source identifier and provenance label."""
    label = item.get('source', 'unknown')
    if item.get('source') in ('model', 'interaction'):
        label = 'model, verified' if item.get('verified') else 'model, unverified'
    return f"[{item['id']} {label}] {item['text']}"


def evidence_block(items: Sequence[Dict], prompt_format: str) -> str:
    if prompt_format == 'trace':
        return "".join(_trace_text(item) for item in items)
    lines = "\n".join(render_evidence(item) for item in items)
    if prompt_format == 'chat':
        return f"{EVIDENCE_OPEN}\n{lines}\n{EVIDENCE_CLOSE}\n"
    return lines + "\n\n"


def item_ids(item: Dict, tokenizer) -> List[int]:
    """Token ids of one evidence line (select_evidence may have truncated them)."""
    if 'ids' not in item:
        item['ids'] = tokenizer.encode(render_evidence(item) + "\n")
    return item['ids']


def _trace_text(item: Dict) -> str:
    text = item['text'].rstrip()
    return text + ("\n" if text.endswith('---') else "\n---\n")


def build_prompt_ids(items: Sequence[Dict], query_ids: List[int], tokenizer, prompt_format: str) -> List[int]:
    """Prompt = budgeted evidence (if any), then the formatted query."""
    if not items:
        return list(query_ids)
    if prompt_format == 'trace':
        delimiter = tokenizer.encode("\n---\n")
        history = []
        for item in items:
            limit = len(item_ids(item, tokenizer))
            ids = tokenizer.encode(_trace_text(item))
            if len(ids) > limit:
                ids = tokenizer.encode(item['text'].rstrip())[:max(0, limit - len(delimiter))] + delimiter
            history.extend(ids)
        return history + list(query_ids)
    lines = [tid for item in items for tid in item_ids(item, tokenizer)]
    if prompt_format == 'chat':
        return tokenizer.encode(f"{EVIDENCE_OPEN}\n") + lines + tokenizer.encode(f"{EVIDENCE_CLOSE}\n") + list(query_ids)
    return lines + tokenizer.encode("\n") + list(query_ids)


def word_f1(a: str, b: str) -> float:
    """Word-level F1 with multiset overlap (repeated words count)."""
    a_tokens, b_tokens = a.lower().split(), b.lower().split()
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    common = sum((Counter(a_tokens) & Counter(b_tokens)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(a_tokens), common / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def select_evidence(items: List[Dict], query_text: str, budget: int, tokenizer) -> List[Dict]:
    """
    Rerank candidates from every tier and fit them into `budget` tokens.

    Each item carries a tier-native dense score in [0, 1] (`dense`); it is
    averaged with word-level F1 against the query for a hybrid score.
    Duplicate texts are dropped. Every tier with candidates first gets an
    equal share of the budget, filled by score; leftover capacity then goes
    to the remaining items across tiers. An item that does not fit whole is
    truncated when at least MIN_ITEM_TOKENS of it fit, else skipped.

    Returns the selected items (with 'score' and 'ids' set), best first.
    """
    seen, unique = set(), []
    for item in items:
        key = item['text'].strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    for item in unique:
        item['score'] = 0.5 * item.get('dense', 0.0) + 0.5 * word_f1(query_text, item['text'])
        item.pop('ids', None)
        item_ids(item, tokenizer)
    unique.sort(key=lambda item: item['score'], reverse=True)

    tiers = sorted({item['tier'] for item in unique})
    if not tiers or budget <= 0:
        return []
    share = budget // len(tiers)
    newline = tokenizer.encode("\n")
    selected, remaining = [], set(map(id, unique))

    def take(pool: List[Dict], limit: int) -> int:
        used = 0
        for item in pool:
            if id(item) not in remaining:
                continue
            room = limit - used
            if room <= 0:
                break
            if len(item['ids']) > room:
                if room < MIN_ITEM_TOKENS:
                    continue
                item['ids'] = item['ids'][:room - len(newline)] + newline
            used += len(item['ids'])
            selected.append(item)
            remaining.discard(id(item))
        return used

    used = sum(take([i for i in unique if i['tier'] == tier], share) for tier in tiers)
    take(unique, budget - used)
    selected.sort(key=lambda item: item['score'], reverse=True)
    return selected


def evidence_ids(items: Sequence[Dict]) -> List[int]:
    """Token ids of the selected evidence lines, in order."""
    return [tid for item in items for tid in item['ids']]
