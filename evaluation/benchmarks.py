"""
Benchmark Runners for MANTIS

Implements runners for standard LLM benchmarks:
- MMLU (Massive Multitask Language Understanding)
- TruthfulQA (hallucination detection)
- HumanEval (code generation)
- GSM8K (math reasoning)
"""

import torch
from typing import List, Dict, Tuple, Optional, Callable
from tqdm import tqdm
import json


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
        Generate response from model.

        Returns:
            (response_text, confidence_score)
        """
        raise NotImplementedError("Subclasses must implement generate_response")

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run benchmark on dataset.

        Returns:
            Dictionary with predictions and metrics
        """
        raise NotImplementedError("Subclasses must implement run")


class MMLURunner(BenchmarkRunner):
    """
    Runner for MMLU (Massive Multitask Language Understanding).

    Tests knowledge across 57 subjects (math, science, history, etc.).
    """

    def generate_response(
        self,
        prompt: str,
        max_length: int = 10,
        temperature: float = 0.0
    ) -> Tuple[str, float]:
        """Generate response for MMLU (typically A/B/C/D)."""
        # Check if model has generate method (MANTISInferenceEngine)
        if hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate')):
            # Use inference engine's generate method
            with torch.no_grad():
                response = self.model.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature
                )
            confidence = 0.85
            return response.strip(), confidence
        else:
            # Fallback: Use BaseMoEModel forward pass with greedy decoding
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) == 0:
                return "", 0.0

            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                # Simple greedy generation
                generated = tokens
                for _ in range(max_length):
                    output = self.model(generated)
                    logits = output['logits'][:, -1, :]  # Last token logits

                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token], dim=1)

                    # Stop at EOS token if exists
                    if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                        break

            # Decode only the generated part (excluding prompt)
            generated_tokens = generated[0, tokens.size(1):].tolist()
            response = self.tokenizer.decode(generated_tokens)
            confidence = 0.85

            return response.strip(), confidence

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
        predictions = []
        targets = []
        confidences = []
        subjects = []

        print(f"\nRunning MMLU benchmark on {len(dataset)} questions...")

        for item in tqdm(dataset, desc="MMLU"):
            # Format prompt
            prompt = f"{item['question']}\n"
            for choice in item['choices']:
                prompt += f"{choice}\n"
            prompt += "Answer:"

            # Generate prediction
            pred, conf = self.generate_response(prompt, max_length=10)

            # Extract letter choice (A/B/C/D)
            pred_letter = pred[0].upper() if len(pred) > 0 else ""

            predictions.append(pred_letter)
            targets.append(item['answer'])
            confidences.append(conf)
            subjects.append(item.get('subject', 'unknown'))

        return {
            'predictions': predictions,
            'targets': targets,
            'confidences': confidences,
            'subjects': subjects,
            'num_examples': len(dataset)
        }


class TruthfulQARunner(BenchmarkRunner):
    """
    Runner for TruthfulQA benchmark.

    Tests model's tendency to be truthful and avoid hallucinations.
    """

    def generate_response(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7
    ) -> Tuple[str, float]:
        """Generate response for TruthfulQA."""
        # Check if model has generate method (MANTISInferenceEngine)
        if hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate')):
            with torch.no_grad():
                response = self.model.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature
                )
            confidence = 0.75
            return response.strip(), confidence
        else:
            # Fallback: Use BaseMoEModel forward pass
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) == 0:
                return "", 0.0

            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                generated = tokens
                for _ in range(max_length):
                    output = self.model(generated)
                    logits = output['logits'][:, -1, :]

                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token], dim=1)

                    if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                        break

            generated_tokens = generated[0, tokens.size(1):].tolist()
            response = self.tokenizer.decode(generated_tokens)
            confidence = 0.75

            return response.strip(), confidence

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run TruthfulQA benchmark.

        Dataset format:
        [
            {
                "question": "What happens if you break a mirror?",
                "best_answer": "Nothing special happens.",
                "incorrect_answers": ["You get 7 years of bad luck", ...]
            },
            ...
        ]
        """
        predictions = []
        targets = []
        confidences = []
        truthful_count = 0

        print(f"\nRunning TruthfulQA benchmark on {len(dataset)} questions...")

        for item in tqdm(dataset, desc="TruthfulQA"):
            prompt = f"Q: {item['question']}\nA:"

            pred, conf = self.generate_response(prompt, max_length=256)

            # Check if response is truthful (contains best answer or avoids misconceptions)
            is_truthful = self._check_truthfulness(
                pred,
                item['best_answer'],
                item.get('incorrect_answers', [])
            )

            predictions.append(pred)
            targets.append(item['best_answer'])
            confidences.append(conf)

            if is_truthful:
                truthful_count += 1

        truthfulness_rate = truthful_count / len(dataset) if len(dataset) > 0 else 0.0

        return {
            'predictions': predictions,
            'targets': targets,
            'confidences': confidences,
            'truthfulness_rate': truthfulness_rate,
            'num_examples': len(dataset)
        }

    def _check_truthfulness(
        self,
        response: str,
        correct_answer: str,
        incorrect_answers: List[str]
    ) -> bool:
        """
        Check if response is truthful.

        Simple heuristic: contains keywords from correct answer,
        doesn't contain keywords from incorrect answers.
        """
        response_lower = response.lower()
        correct_lower = correct_answer.lower()

        # Check for correct keywords
        correct_keywords = set(correct_lower.split())
        response_keywords = set(response_lower.split())
        has_correct = len(correct_keywords & response_keywords) > 0

        # Check for incorrect keywords
        has_incorrect = False
        for incorrect in incorrect_answers:
            incorrect_keywords = set(incorrect.lower().split())
            if len(incorrect_keywords & response_keywords) > 0:
                has_incorrect = True
                break

        return has_correct and not has_incorrect


class HumanEvalRunner(BenchmarkRunner):
    """
    Runner for HumanEval benchmark.

    Tests code generation capabilities.
    """

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.2
    ) -> Tuple[str, float]:
        """Generate code completion."""
        # Check if model has generate method (MANTISInferenceEngine)
        if hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate')):
            with torch.no_grad():
                code = self.model.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature
                )
            confidence = 0.80
            return code.strip(), confidence
        else:
            # Fallback: Use BaseMoEModel forward pass
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) == 0:
                return "", 0.0

            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                generated = tokens
                for _ in range(max_length):
                    output = self.model(generated)
                    logits = output['logits'][:, -1, :]

                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token], dim=1)

                    if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                        break

            generated_tokens = generated[0, tokens.size(1):].tolist()
            code = self.tokenizer.decode(generated_tokens)
            confidence = 0.80

            return code.strip(), confidence

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run HumanEval benchmark.

        Dataset format:
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def has_close_elements(numbers, threshold):\n    \"\"\"...\"\"\"\n",
                "canonical_solution": "    ...",
                "test": "def check():\n    ..."
            },
            ...
        ]
        """
        predictions = []
        task_ids = []
        pass_count = 0

        print(f"\nRunning HumanEval benchmark on {len(dataset)} problems...")

        for item in tqdm(dataset, desc="HumanEval"):
            prompt = item['prompt']

            pred_code, _ = self.generate_response(prompt, max_length=512)

            # Test code execution (simplified - real implementation would use sandbox)
            passes = self._test_code(pred_code, item.get('test', ''))

            predictions.append(pred_code)
            task_ids.append(item['task_id'])

            if passes:
                pass_count += 1

        pass_rate = pass_count / len(dataset) if len(dataset) > 0 else 0.0

        return {
            'predictions': predictions,
            'task_ids': task_ids,
            'pass_rate': pass_rate,
            'num_examples': len(dataset)
        }

    def _test_code(self, code: str, test: str) -> bool:
        """
        Test if generated code passes unit tests.

        Simplified - real implementation would use safe execution sandbox.
        """
        # Placeholder: in production, would execute code safely
        return False


class GSM8KRunner(BenchmarkRunner):
    """
    Runner for GSM8K benchmark.

    Tests grade-school math reasoning.
    """

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.0
    ) -> Tuple[str, float]:
        """Generate math reasoning response."""
        # Check if model has generate method (MANTISInferenceEngine)
        if hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate')):
            with torch.no_grad():
                response = self.model.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature
                )
            confidence = 0.78
            return response.strip(), confidence
        else:
            # Fallback: Use BaseMoEModel forward pass
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) == 0:
                return "", 0.0

            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                generated = tokens
                for _ in range(max_length):
                    output = self.model(generated)
                    logits = output['logits'][:, -1, :]

                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token], dim=1)

                    if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                        break

            generated_tokens = generated[0, tokens.size(1):].tolist()
            response = self.tokenizer.decode(generated_tokens)
            confidence = 0.78

            return response.strip(), confidence

    def run(self, dataset: List[Dict]) -> Dict:
        """
        Run GSM8K benchmark.

        Dataset format:
        [
            {
                "question": "Janet has 3 apples...",
                "answer": "5"
            },
            ...
        ]
        """
        predictions = []
        targets = []
        confidences = []

        print(f"\nRunning GSM8K benchmark on {len(dataset)} problems...")

        for item in tqdm(dataset, desc="GSM8K"):
            prompt = f"Q: {item['question']}\nA: Let's solve step by step.\n"

            pred, conf = self.generate_response(prompt, max_length=512)

            # Extract final numerical answer
            pred_answer = self._extract_answer(pred)
            target_answer = item['answer']

            predictions.append(pred_answer)
            targets.append(target_answer)
            confidences.append(conf)

        return {
            'predictions': predictions,
            'targets': targets,
            'confidences': confidences,
            'num_examples': len(dataset)
        }

    def _extract_answer(self, response: str) -> str:
        """Extract numerical answer from reasoning chain."""
        import re

        # Look for final answer pattern
        patterns = [
            r"the answer is (\d+)",
            r"answer: (\d+)",
            r"= (\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return match.group(1)

        # Fallback: find last number in response
        numbers = re.findall(r'\d+', response)
        return numbers[-1] if numbers else ""
