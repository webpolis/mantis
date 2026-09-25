"""
Synthetic multi-session memory benchmark.

Facts about entities arrive over sessions; some are later updated, and
similar entities supply distractors. Questions ask for the latest value,
so the benchmark measures recall across sessions, correct handling of
updates (the old value is not credited), robustness to distractors and
abstention on unanswerable questions.

Two modes share one dataset:
- `memory`: each session is ingested into a temporary engine namespace, then
  questions run with writes frozen (memory does the work). The namespace is
  deleted when scoring ends.
- `prompt`: every fact seen so far is prepended to the question (the
  "same backbone + recent-text buffer" baseline of the review's ablation
  ladder); works with a bare base model as well.
"""

import contextlib
import random
import uuid
from typing import Dict, List

from tqdm import tqdm

from evaluation.benchmarks import BenchmarkRunner
from mantis.training.scoring import answer_correct

ENTITIES = ["Ada", "Bruno", "Chen", "Dalia", "Ezra", "Farah", "Gus", "Hana", "Ivo", "Jun",
            "Kira", "Leo", "Mara", "Nils", "Orla", "Pia", "Quinn", "Rosa", "Sven", "Tara",
            "Uma", "Vik", "Wren", "Xola", "Yael", "Zane", "Amos", "Bea", "Cato", "Dora"]
ATTRIBUTES = {
    'favorite color': ["red", "blue", "green", "amber", "violet", "teal", "ivory", "coral"],
    'home city': ["Lima", "Oslo", "Perth", "Quito", "Riga", "Turin", "Ulm", "Vigo"],
    'pet': ["cat", "dog", "parrot", "rabbit", "turtle", "ferret", "gecko", "pony"],
    'job': ["baker", "pilot", "nurse", "welder", "tailor", "farmer", "judge", "miner"],
    'lucky number': ["3", "7", "12", "19", "24", "31", "42", "58"],
}
MAX_NEW_TOKENS = 32


def _fact(entity: str, attribute: str, value: str, update: bool = False) -> str:
    verb = "changed to" if update else "is"
    return f"The {entity}'s {attribute} {verb} {value}."


def make_memory_dataset(n_sessions: int = 20, facts_per_session: int = 5, updates: int = 2,
                        distractors: int = 3, seed: int = 0) -> Dict:
    """
    Build sessions of facts and the questions to ask after all of them.

    Returns:
        {'sessions': [[fact, ...], ...], 'questions': [{'question', 'answer',
        'old_answer', 'unanswerable', 'kind', 'session'}, ...]}
    """
    rng = random.Random(seed)
    entities = list(ENTITIES)
    rng.shuffle(entities)
    attributes = list(ATTRIBUTES)
    pairs = [(e, a) for e in entities for a in attributes]
    rng.shuffle(pairs)
    n_reserved = max(1, len(pairs) // 5)
    unknown, free = pairs[:n_reserved], pairs[n_reserved:]  # reserved pairs are never mentioned

    state: Dict[tuple, Dict] = {}      # (entity, attribute) -> {'value', 'old', 'session', 'kind'}
    sessions: List[List[str]] = []
    for s in range(n_sessions):
        facts = []
        for _ in range(facts_per_session):
            if not free:
                break
            entity, attribute = free.pop()
            value = rng.choice(ATTRIBUTES[attribute])
            state[(entity, attribute)] = {'value': value, 'old': None, 'session': s, 'kind': 'recall'}
            facts.append(_fact(entity, attribute, value))
        # Updates of facts from earlier sessions
        earlier = [k for k, v in state.items() if v['session'] < s and v['old'] is None]
        rng.shuffle(earlier)
        for key in earlier[:updates] if s > 0 else []:
            entry = state[key]
            new_value = rng.choice([v for v in ATTRIBUTES[key[1]] if v != entry['value']])
            entry.update({'old': entry['value'], 'value': new_value, 'session': s, 'kind': 'update'})
            facts.append(_fact(key[0], key[1], new_value, update=True))
        # Distractors: the same attribute for a different entity
        targets = [k for k, v in state.items() if v['kind'] != 'distractor_source' and not v.get('distracted')]
        rng.shuffle(targets)
        for entity, attribute in targets[:distractors]:
            candidates = [(e, a) for e, a in free if a == attribute]
            if not candidates:
                continue
            other = rng.choice(candidates)
            free.remove(other)
            value = rng.choice(ATTRIBUTES[attribute])
            state[other] = {'value': value, 'old': None, 'session': s, 'kind': 'distractor_source'}
            state[(entity, attribute)]['distracted'] = True
            facts.append(_fact(other[0], other[1], value))
        rng.shuffle(facts)
        sessions.append(facts)

    questions = []
    for (entity, attribute), entry in state.items():
        if entry['kind'] == 'distractor_source':
            continue
        kind = 'update' if entry['old'] is not None else ('distractor' if entry.get('distracted') else 'recall')
        questions.append({
            'question': f"What is the {entity}'s {attribute}?",
            'answer': entry['value'],
            'old_answer': entry['old'],
            'unanswerable': False,
            'kind': kind,
            'session': entry['session'],
        })
    for entity, attribute in sorted(unknown[:max(1, len(questions) // 5)]):
        questions.append({
            'question': f"What is the {entity}'s {attribute}?",
            'answer': "",
            'old_answer': None,
            'unanswerable': True,
            'kind': 'unanswerable',
            'session': -1,
        })
    rng.shuffle(questions)
    return {'sessions': sessions, 'questions': questions}


class MemoryRecallRunner(BenchmarkRunner):
    """
    Runs make_memory_dataset() output in `memory` or `prompt` mode.

    Results carry per-kind accuracy (recall, update, distractor), the
    abstention rate on unanswerable questions, `old_value_rate` (updates
    answered with the superseded value) and the usual per-example lists.
    """

    def __init__(self, model, tokenizer, device='cuda', mode: str = 'memory'):
        super().__init__(model, tokenizer, device)
        if mode not in ('memory', 'prompt'):
            raise ValueError(f"Unknown memory benchmark mode: {mode}")
        if mode == 'memory' and not (self.is_engine and (self.model.episodic is not None or self.model.semantic is not None)):
            raise ValueError("memory mode needs a MANTISInferenceEngine with episodic or semantic memory")
        self.mode = mode

    def run(self, dataset: Dict) -> Dict:
        """
        Args:
            dataset: {'sessions': [[fact, ...]], 'questions': [...]} from make_memory_dataset()
        """
        sessions, questions = dataset['sessions'], dataset['questions']
        namespace = f"membench-{uuid.uuid4().hex[:8]}"
        try:
            return self._run_in_namespace(sessions, questions, namespace)
        finally:
            if self.mode == 'memory':
                self.model.delete_namespace(namespace)

    def _run_in_namespace(self, sessions: List[List[str]], questions: List[Dict], namespace: str) -> Dict:
        print(f"\nRunning memory benchmark: {len(sessions)} sessions, {len(questions)} questions ({self.mode} mode)...")

        if self.mode == 'memory':
            for facts in tqdm(sessions, desc="Ingest"):
                self.model.ingest(" ".join(facts), namespace=namespace, source='user')
        history = " ".join(fact for facts in sessions for fact in facts)

        records = []
        scope = self.model.frozen_memory() if self.is_engine else contextlib.nullcontext()
        with scope:
            for item in tqdm(questions, desc="Memory QA"):
                if self.mode == 'memory':
                    result = self.model.generate(item['question'], max_length=MAX_NEW_TOKENS, temperature=0.0,
                                                 namespace=namespace)
                    record = {
                        'response': result['response'], 'confidence': result['confidence'],
                        'confidence_source': result['confidence_source'], 'abstained': result['abstained'],
                        'latency': result['latency'], 'path': result['path'],
                        'evidence': [e['id'] for e in result['evidence']], 'num_tokens': result['num_tokens'],
                        'compute_units': result['compute_units'], 'timings': result['timings'],
                    }
                    prompt = item['question']
                else:
                    prompt = f"{history}\nQ: {item['question']}\nA:"
                    record = self.generate_response(prompt, max_length=MAX_NEW_TOKENS)
                response = record['response'].strip()
                answered = not record['abstained']
                if item['unanswerable']:
                    correct = not answered
                else:
                    correct = answered and answer_correct(response, item['answer'])
                    if correct and item['old_answer'] and answer_correct(response, item['old_answer']):
                        correct = False  # credited neither the stale value nor a hedge listing both
                records.append({
                    **record, 'prompt': prompt, 'prediction': response, 'target': item['answer'],
                    'correct': correct, 'kind': item['kind'], 'unanswerable': item['unanswerable'],
                    'old_value': bool(answered and item['old_answer'] and answer_correct(response, item['old_answer'])),
                })

        def rate(flag: str, kind: str) -> float:
            rows = [r for r in records if r['kind'] == kind]
            return sum(r[flag] for r in rows) / len(rows) if rows else 0.0

        results = self._results(records, mode=self.mode, namespace=namespace)
        results['by_kind'] = {kind: rate('correct', kind) for kind in ('recall', 'update', 'distractor')}
        results['unanswerable_abstention_rate'] = rate('abstained', 'unanswerable')
        results['old_value_rate'] = rate('old_value', 'update')
        return results
