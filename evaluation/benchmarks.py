"""
Benchmark Runners for MANTIS

Implements runners for standard LLM benchmarks:
- MMLU (Massive Multitask Language Understanding)
- TruthfulQA (hallucination detection)
- HumanEval (code generation)
- GSM8K (math reasoning)

Every runner returns per-example `correct` flags and model `confidences`
(geometric-mean probability of the generated tokens).
"""

import math
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

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

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.0
    ) -> Tuple[str, float]:
        """
        Generate a completion for `prompt`.

        Works with a MANTISInferenceEngine (dict results) or a bare
        BaseMoEModel (decoded here).

        Returns:
            (completion_text, confidence) where confidence is the geometric-mean
            probability of the generated tokens. Whitespace is kept (code needs it).
        """
        if hasattr(self.model, 'generate'):
            result = self.model.generate(prompt, max_length=max_length, temperature=temperature)
            return result['response'], result['confidence']

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            return "", 0.0
        eos = self.tokenizer.eos_token_id
        tokens, log_probs = [], []
        for token, log_prob in generate_tokens(
            self.model, prompt_ids, max_length, temperature=temperature,
            banned_ids=self.tokenizer.non_generable_ids, stop_ids=[eos],
        ):
            if token == eos:
                break
            tokens.append(token)
            log_probs.append(log_prob)
        confidence = math.exp(sum(log_probs) / len(log_probs)) if log_probs else 0.0
        return self.tokenizer.decode(tokens), confidence

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run benchmark on dataset.

        Returns:
            Dict with predictions, targets, correct, confidences, num_examples
        """
        raise NotImplementedError("Subclasses must implement run")


class MMLURunner(BenchmarkRunner):
    """
    Runner for MMLU (Massive Multitask Language Understanding).

    Tests knowledge across 57 subjects (math, science, history, etc.).
    """

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
        predictions, targets, correct, confidences, subjects = [], [], [], [], []
        print(f"\nRunning MMLU benchmark on {len(dataset)} questions...")

        for item in tqdm(dataset, desc="MMLU"):
            prompt = f"{item['question']}\n" + "".join(f"{c}\n" for c in item['choices']) + "Answer:"
            response, conf = self.generate_response(prompt, max_length=10)
            pred = self.extract_choice(response)

            predictions.append(pred)
            targets.append(item['answer'])
            correct.append(pred == item['answer'])
            confidences.append(conf)
            subjects.append(item.get('subject', 'unknown'))

        return {
            'predictions': predictions,
            'targets': targets,
            'correct': correct,
            'confidences': confidences,
            'subjects': subjects,
            'num_examples': len(dataset)
        }


class TruthfulQARunner(BenchmarkRunner):
    """
    Runner for TruthfulQA benchmark (generation task).

    A response counts as truthful when it is more similar to the closest
    true reference than to the closest false reference, the "max-sim diff"
    criterion from the TruthfulQA paper, using token F1 as similarity.
    """

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
        predictions, targets, correct, confidences = [], [], [], []
        print(f"\nRunning TruthfulQA benchmark on {len(dataset)} questions...")

        for item in tqdm(dataset, desc="TruthfulQA"):
            response, conf = self.generate_response(f"Q: {item['question']}\nA:", max_length=64)
            response = response.strip()
            true_refs = [item['best_answer']] + list(item.get('correct_answers', []))

            predictions.append(response)
            targets.append(item['best_answer'])
            correct.append(self.is_truthful(response, true_refs, item.get('incorrect_answers', [])))
            confidences.append(conf)

        return {
            'predictions': predictions,
            'targets': targets,
            'correct': correct,
            'confidences': confidences,
            'truthfulness_rate': sum(correct) / len(correct) if correct else 0.0,
            'num_examples': len(dataset)
        }


class HumanEvalRunner(BenchmarkRunner):
    """
    Runner for HumanEval benchmark (pass@1 with greedy decoding).

    Generated code runs in a subprocess with a timeout and memory, CPU-time
    and file-size limits. This limits accidents; it is not a security
    sandbox, so only evaluate models you trust or run inside a container.
    """

    STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"]
    TIMEOUT_SECONDS = 10

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
        predictions, task_ids, correct, confidences = [], [], [], []
        print(f"\nRunning HumanEval benchmark on {len(dataset)} problems...")

        for item in tqdm(dataset, desc="HumanEval"):
            completion, conf = self.generate_response(item['prompt'], max_length=512)
            completion = self.truncate(completion)
            program = f"{item['prompt']}{completion}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"

            predictions.append(completion)
            task_ids.append(item['task_id'])
            correct.append(self._passes(program))
            confidences.append(conf)

        return {
            'predictions': predictions,
            'task_ids': task_ids,
            'correct': correct,
            'confidences': confidences,
            'pass_rate': sum(correct) / len(correct) if correct else 0.0,
            'num_examples': len(dataset)
        }

    # Limits are set inside the child: preexec_fn is unsafe in a threaded parent (torch)
    BOOTSTRAP = (
        "import resource, runpy\n"
        "resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))\n"
        "resource.setrlimit(resource.RLIMIT_CPU, ({t}, {t}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))\n"
        "runpy.run_path('program.py', run_name='__main__')\n"
    )

    @classmethod
    def _passes(cls, program: str) -> bool:
        """True when the program (solution + tests) exits cleanly within the limits."""
        with tempfile.TemporaryDirectory() as workdir:
            with open(os.path.join(workdir, 'program.py'), 'w') as f:
                f.write(program)
            try:
                result = subprocess.run(
                    [sys.executable, '-I', '-c', cls.BOOTSTRAP.format(t=cls.TIMEOUT_SECONDS)],
                    cwd=workdir,
                    env={'PATH': os.environ.get('PATH', '')},
                    capture_output=True,
                    timeout=cls.TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                return False
            return result.returncode == 0


class GSM8KRunner(BenchmarkRunner):
    """
    Runner for GSM8K benchmark.

    Tests grade-school math reasoning.
    """

    NUMBER = r'-?\d[\d,]*(?:\.\d+)?'

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
        predictions, targets, correct, confidences = [], [], [], []
        print(f"\nRunning GSM8K benchmark on {len(dataset)} problems...")

        for item in tqdm(dataset, desc="GSM8K"):
            prompt = f"Q: {item['question']}\nA: Let's solve step by step.\n"
            response, conf = self.generate_response(prompt, max_length=256)

            pred = self.extract_answer(response.strip())
            target = self.extract_answer(item['answer'])
            predictions.append(pred)
            targets.append(target)
            correct.append(pred != "" and pred == target)
            confidences.append(conf)

        return {
            'predictions': predictions,
            'targets': targets,
            'correct': correct,
            'confidences': confidences,
            'num_examples': len(dataset)
        }
