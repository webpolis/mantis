"""
Benchmark Runners for MANTIS

Implements runners for standard LLM benchmarks:
- MMLU (Massive Multitask Language Understanding)
- TruthfulQA (lexical truthfulness proxy)
- HumanEval (code generation)
- GSM8K (math reasoning)

Every runner returns per-example `correct` flags, `abstained` flags, model
`confidences` and full `records` (prompt, prediction, target, route, evidence,
timings, compute), so results can be audited example by example.
"""

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List

import torch
from tqdm import tqdm

from evaluation.metrics import token_f1
from mantis.inference.generation import generate_tokens


class BenchmarkRunner:
    """Base class for benchmark evaluation."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @property
    def is_engine(self) -> bool:
        return hasattr(self.model, 'generate')

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.0
    ) -> Dict:
        """
        Generate a completion for `prompt`.

        Works with a MANTISInferenceEngine or a bare BaseMoEModel (decoded
        here). Whitespace is kept (code needs it).

        Returns:
            Record dict: response, confidence (geometric-mean token probability
            for a bare model; the engine's reported confidence otherwise),
            abstained, latency, path, evidence ids, num_tokens, compute_units,
            timings.
        """
        if self.is_engine:
            result = self.model.generate(prompt, max_length=max_length, temperature=temperature)
            return {
                'response': result['response'],
                'confidence': result['confidence'],
                'confidence_source': result['confidence_source'],
                'abstained': result['abstained'],
                'latency': result['latency'],
                'path': result['path'],
                'evidence': [item['id'] for item in result['evidence']],
                'num_tokens': result['num_tokens'],
                'compute_units': result['compute_units'],
                'timings': result['timings'],
            }

        start = time.time()
        prompt_ids = self.tokenizer.encode(prompt)
        tokens, log_probs = [], []
        if prompt_ids:
            eos = self.tokenizer.eos_token_id
            for token, log_prob in generate_tokens(
                self.model, prompt_ids, max_length, temperature=temperature,
                banned_ids=self.tokenizer.non_generable_ids, stop_ids=[eos],
            ):
                if token == eos:
                    break
                tokens.append(token)
                log_probs.append(log_prob)
        latency = time.time() - start
        confidence = math.exp(sum(log_probs) / len(log_probs)) if log_probs else 0.0
        active = self.model.count_parameters()['active']
        return {
            'response': self.tokenizer.decode(tokens),
            'confidence': confidence,
            'confidence_source': 'token_likelihood',
            'abstained': False,
            'latency': latency,
            'path': 'base',
            'evidence': [],
            'num_tokens': len(tokens),
            'compute_units': active * (min(len(prompt_ids), self.model.max_seq_len) + len(tokens)),
            'timings': {'generate': latency},
        }

    @staticmethod
    def _results(records: List[Dict], **extra) -> Dict:
        """Assemble the per-example lists every runner returns."""
        return {
            'predictions': [r['prediction'] for r in records],
            'targets': [r['target'] for r in records],
            'correct': [r['correct'] for r in records],
            'abstained': [r['abstained'] for r in records],
            'confidences': [r['confidence'] for r in records],
            'records': records,
            'num_examples': len(records),
            **extra,
        }

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run benchmark on dataset.

        Returns:
            Dict with predictions, targets, correct, abstained, confidences, records, num_examples
        """
        raise NotImplementedError("Subclasses must implement run")


class MMLURunner(BenchmarkRunner):
    """
    Runner for MMLU (Massive Multitask Language Understanding).

    Declared protocol (results are only comparable under the same one):
    zero-shot; the prompt is the question, one choice per line as given
    ("A) ...") and a final "Answer:" line; greedy decoding of at most 10 new
    tokens; the prediction is the first standalone A-D letter in the
    completion (no letter counts as wrong). An abstention counts as wrong.
    """

    MAX_NEW_TOKENS = 10

    @staticmethod
    def extract_choice(response: str) -> str:
        """First standalone A-D letter (e.g. 'B', 'B)', '(B', 'Answer: B')."""
        match = re.search(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', response)
        return match.group(1) if match else ""

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run MMLU benchmark.

        Dataset format:
        [
            {
                "question": "What is 2+2?",
                "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
                "answer": "B"
            },
            ...
        ]
        """
        records = []
        print(f"\nRunning MMLU benchmark on {len(dataset)} questions...")

        for item in tqdm(dataset, desc="MMLU"):
            prompt = f"{item['question']}\n" + "".join(f"{c}\n" for c in item['choices']) + "Answer:"
            record = self.generate_response(prompt, max_length=self.MAX_NEW_TOKENS)
            pred = "" if record['abstained'] else self.extract_choice(record['response'])
            records.append({
                **record, 'prompt': prompt, 'prediction': pred, 'target': item['answer'],
                'correct': pred == item['answer'], 'subject': item.get('subject', 'unknown'),
            })

        return self._results(records, subjects=[r['subject'] for r in records])


class TruthfulQARunner(BenchmarkRunner):
    """
    Runner for TruthfulQA questions (generation task) with a lexical proxy.

    A response counts as truthful when its word-level F1 against the closest
    true reference exceeds its F1 against the closest false reference (the
    "max-sim diff" idea from the TruthfulQA paper, with lexical overlap in
    place of the fine-tuned judge). Negations and paraphrases fool this
    proxy, so report it as `truthfulness_proxy`, never as an official
    TruthfulQA score.
    """

    MAX_NEW_TOKENS = 64

    @staticmethod
    def is_truthful(response: str, correct_answers: List[str], incorrect_answers: List[str]) -> bool:
        best_true = max(token_f1(response, ref) for ref in correct_answers)
        best_false = max((token_f1(response, ref) for ref in incorrect_answers), default=0.0)
        return best_true > best_false

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run TruthfulQA benchmark.

        Dataset format:
        [
            {
                "question": "What happens if you break a mirror?",
                "best_answer": "Nothing special happens.",
                "correct_answers": ["Nothing happens", ...],     # optional
                "incorrect_answers": ["You get 7 years of bad luck", ...]
            },
            ...
        ]
        """
        records = []
        print(f"\nRunning TruthfulQA benchmark on {len(dataset)} questions...")

        for item in tqdm(dataset, desc="TruthfulQA"):
            prompt = f"Q: {item['question']}\nA:"
            record = self.generate_response(prompt, max_length=self.MAX_NEW_TOKENS)
            response = record['response'].strip()
            true_refs = [item['best_answer']] + list(item.get('correct_answers', []))
            truthful = (not record['abstained']
                        and self.is_truthful(response, true_refs, item.get('incorrect_answers', [])))
            records.append({
                **record, 'prompt': prompt, 'prediction': response, 'target': item['best_answer'],
                'correct': truthful,
            })

        correct = [r['correct'] for r in records]
        return self._results(records, truthfulness_proxy=sum(correct) / len(correct) if correct else 0.0)


class HumanEvalRunner(BenchmarkRunner):
    """
    Runner for HumanEval benchmark (pass@1 with greedy decoding).

    Generated code is untrusted. `sandbox='docker'` runs each program in a
    throwaway container with no network, a memory cap, one CPU and a pid
    limit; it is the default whenever the docker binary is available.
    `sandbox='subprocess'` only applies resource limits inside a child
    process, which limits accidents but is not a security boundary.
    """

    STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"]
    TIMEOUT_SECONDS = 10
    DOCKER_IMAGE = "python:3.12-slim"
    MAX_NEW_TOKENS = 512

    def __init__(self, model, tokenizer, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 sandbox: str = None):
        super().__init__(model, tokenizer, device)
        if sandbox is None:
            sandbox = 'docker' if shutil.which('docker') else 'subprocess'
        if sandbox not in ('docker', 'subprocess'):
            raise ValueError(f"Unknown sandbox: {sandbox}")
        if sandbox == 'docker' and not shutil.which('docker'):
            raise ValueError("sandbox='docker' requires the docker binary on PATH")
        if sandbox == 'subprocess':
            print("⚠️  HumanEval sandbox=subprocess: resource limits only, not a security boundary. "
                  "Install docker for container isolation.")
        self.sandbox = sandbox

    @classmethod
    def truncate(cls, completion: str) -> str:
        """Cut the completion where the function body ends."""
        cut = min((completion.find(s) for s in cls.STOP_SEQUENCES if s in completion), default=len(completion))
        return completion[:cut]

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run HumanEval benchmark.

        Dataset format:
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def has_close_elements(numbers, threshold):\\n    \\"\\"\\"...\\"\\"\\"\\n",
                "test": "def check(candidate):\\n    assert ...",
                "entry_point": "has_close_elements"
            },
            ...
        ]
        """
        records = []
        print(f"\nRunning HumanEval benchmark on {len(dataset)} problems ({self.sandbox} sandbox)...")

        for item in tqdm(dataset, desc="HumanEval"):
            record = self.generate_response(item['prompt'], max_length=self.MAX_NEW_TOKENS)
            completion = "" if record['abstained'] else self.truncate(record['response'])
            program = f"{item['prompt']}{completion}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
            records.append({
                **record, 'prompt': item['prompt'], 'prediction': completion, 'target': item['task_id'],
                'task_id': item['task_id'], 'correct': not record['abstained'] and self._passes(program),
            })

        correct = [r['correct'] for r in records]
        return self._results(records, task_ids=[r['task_id'] for r in records],
                             pass_rate=sum(correct) / len(correct) if correct else 0.0)

    # Limits are set inside the child: preexec_fn is unsafe in a threaded parent (torch)
    BOOTSTRAP = (
        "import resource, runpy\n"
        "resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))\n"
        "resource.setrlimit(resource.RLIMIT_CPU, ({t}, {t}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))\n"
        "runpy.run_path('program.py', run_name='__main__')\n"
    )

    def _passes(self, program: str) -> bool:
        """True when the program (solution + tests) exits cleanly within the limits."""
        with tempfile.TemporaryDirectory() as workdir:
            with open(os.path.join(workdir, 'program.py'), 'w') as f:
                f.write(program)
            if self.sandbox == 'docker':
                command = [
                    'docker', 'run', '--rm', '--network', 'none', '--memory', '512m', '--cpus', '1',
                    '--pids-limit', '64', '-v', f'{workdir}:/work:ro', self.DOCKER_IMAGE,
                    'python', '/work/program.py',
                ]
                timeout = self.TIMEOUT_SECONDS + 30  # container start-up
            else:
                command = [sys.executable, '-I', '-c', self.BOOTSTRAP.format(t=self.TIMEOUT_SECONDS)]
                timeout = self.TIMEOUT_SECONDS
            try:
                result = subprocess.run(
                    command,
                    cwd=workdir,
                    env={'PATH': os.environ.get('PATH', '')},
                    capture_output=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return False
            return result.returncode == 0


class GSM8KRunner(BenchmarkRunner):
    """
    Runner for GSM8K benchmark.

    Tests grade-school math reasoning: greedy decoding of at most 256 new
    tokens after "Let's solve step by step.", numeric extraction from the
    completion (see extract_answer) compared with the reference number.
    """

    NUMBER = r'-?\d[\d,]*(?:\.\d+)?'
    MAX_NEW_TOKENS = 256

    @classmethod
    def extract_answer(cls, text: str) -> str:
        """
        Final numeric answer: after '####', else after 'answer is', else the last
        number in the text. Normalized (no commas, integers without '.0').
        """
        match = (re.search(rf'####\s*\$?({cls.NUMBER})', text)
                 or re.search(rf'answer is\s*:?\s*\$?({cls.NUMBER})', text, re.IGNORECASE))
        if match:
            raw = match.group(1)
        else:
            numbers = re.findall(cls.NUMBER, text)
            raw = numbers[-1] if numbers else ""
        return cls.normalize(raw)

    @staticmethod
    def normalize(number: str) -> str:
        try:
            value = float(number.replace(',', ''))
        except ValueError:
            return number.strip()
        return str(int(value)) if value.is_integer() else repr(value)

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run GSM8K benchmark.

        Dataset format:
        [
            {
                "question": "Janet has 3 apples...",
                "answer": "5"      # or a full solution ending in "#### 5"
            },
            ...
        ]
        """
        records = []
        print(f"\nRunning GSM8K benchmark on {len(dataset)} problems...")

        for item in tqdm(dataset, desc="GSM8K"):
            prompt = f"Q: {item['question']}\nA: Let's solve step by step.\n"
            record = self.generate_response(prompt, max_length=self.MAX_NEW_TOKENS)
            pred = "" if record['abstained'] else self.extract_answer(record['response'].strip())
            target = self.extract_answer(item['answer'])
            records.append({
                **record, 'prompt': prompt, 'prediction': pred, 'target': target,
                'correct': pred != "" and pred == target,
            })

        return self._results(records)
