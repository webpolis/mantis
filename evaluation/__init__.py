"""
MANTIS Evaluation Framework

Provides benchmark runners and evaluation utilities for MANTIS models.
"""

import contextlib
import json

from evaluation.benchmarks import (
    MMLURunner,
    TruthfulQARunner,
    HumanEvalRunner,
    GSM8KRunner
)
from evaluation.memory_bench import MemoryRecallRunner, make_memory_dataset
from evaluation.metrics import (
    compute_accuracy,
    compute_f1_score,
    confident_error_rate,
    error_rate,
    coverage,
    answered_error_rate,
    brier_score,
    risk_coverage_curve,
    compute_calibration_error,
    compute_metrics_summary
)

__all__ = [
    'MMLURunner',
    'TruthfulQARunner',
    'HumanEvalRunner',
    'GSM8KRunner',
    'MemoryRecallRunner',
    'make_memory_dataset',
    'EvaluationHarness',
    'compute_accuracy',
    'compute_f1_score',
    'confident_error_rate',
    'error_rate',
    'coverage',
    'answered_error_rate',
    'brier_score',
    'risk_coverage_curve',
    'compute_calibration_error',
    'compute_metrics_summary'
]


class EvaluationHarness:
    """
    Main evaluation harness for running benchmarks on MANTIS models.

    With a MANTISInferenceEngine and `memory_mode='frozen'` (the default),
    each independent-example benchmark runs inside `engine.frozen_memory()`:
    no interaction is written and no consolidation runs, so results do not
    depend on evaluation order or earlier exposure to evaluation prompts.
    `memory_mode='stateful'` keeps writes on (conversational evaluations).
    The memory benchmark ingests sessions into its own temporary namespace,
    freezes writes while scoring questions, then deletes that namespace.
    """

    def __init__(self, model, tokenizer, device='cuda', memory_mode='frozen', sandbox=None,
                 memory_bench_mode='memory'):
        if memory_mode not in ('frozen', 'stateful'):
            raise ValueError(f"Unknown memory_mode: {memory_mode}")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.memory_mode = memory_mode

        # Initialize benchmark runners
        self.runners = {
            'mmlu': MMLURunner(model, tokenizer, device),
            'truthfulqa': TruthfulQARunner(model, tokenizer, device),
            'humaneval': HumanEvalRunner(model, tokenizer, device, sandbox=sandbox),
            'gsm8k': GSM8KRunner(model, tokenizer, device),
            'memory': MemoryRecallRunner(model, tokenizer, device, mode=memory_bench_mode),
        }

    @property
    def is_engine(self) -> bool:
        return hasattr(self.model, 'frozen_memory')

    def _memory_scope(self, benchmark_name: str):
        if self.is_engine and self.memory_mode == 'frozen' and benchmark_name != 'memory':
            return self.model.frozen_memory()
        return contextlib.nullcontext()

    def run_benchmark(self, benchmark_name: str, dataset) -> dict:
        """
        Run a specific benchmark.

        Args:
            benchmark_name: Name of benchmark ('mmlu', 'truthfulqa', etc.)
            dataset: Dataset to evaluate on

        Returns:
            Results dictionary with predictions, records and metrics
        """
        if benchmark_name not in self.runners:
            raise ValueError(
                f"Unknown benchmark: {benchmark_name}. "
                f"Available: {list(self.runners.keys())}"
            )

        runner = self.runners[benchmark_name]
        with self._memory_scope(benchmark_name):
            results = runner.run(dataset)

        results['metrics'] = compute_metrics_summary(
            results['correct'], results.get('confidences'), results.get('abstained')
        )
        return results

    def run_all_benchmarks(self, datasets: dict) -> dict:
        """
        Run all benchmarks and generate comprehensive report.

        Args:
            datasets: Dict mapping benchmark name -> dataset

        Returns:
            Dictionary with results for all benchmarks
        """
        all_results = {}

        for benchmark_name, dataset in datasets.items():
            if benchmark_name in self.runners:
                print(f"\n{'='*80}")
                print(f"Running {benchmark_name.upper()}")
                print(f"{'='*80}")

                results = self.run_benchmark(benchmark_name, dataset)
                all_results[benchmark_name] = results

                # Print summary
                print("\nResults:")
                for metric_name, score in results['metrics'].items():
                    print(f"  {metric_name}: {score:.4f}")
                for key in ('pass_rate', 'truthfulness_proxy', 'unanswerable_abstention_rate', 'old_value_rate'):
                    if key in results:
                        print(f"  {key}: {results[key]:.4f}")
                if 'by_kind' in results:
                    for kind, score in results['by_kind'].items():
                        print(f"  accuracy[{kind}]: {score:.4f}")

        return all_results

    def generate_report(self, results: dict, output_path: str = None, artifacts: dict = None,
                        records_path: str = None):
        """
        Generate evaluation report.

        Args:
            results: Results from run_all_benchmarks()
            output_path: Optional path to save the report JSON
            artifacts: Checkpoint paths, tokenizer fingerprint, settings (recorded verbatim)
            records_path: Optional JSONL path for per-example records (audits)
        """
        import torch

        report = {
            'model': str(type(self.model).__name__),
            'memory_mode': self.memory_mode,
            'artifacts': dict(artifacts or {}),
            'benchmarks': {}
        }
        if 'tokenizer_fingerprint' not in report['artifacts']:
            report['artifacts']['tokenizer_fingerprint'] = self.tokenizer.fingerprint()

        for benchmark_name, benchmark_results in results.items():
            entry = {
                'num_examples': benchmark_results.get('num_examples', 0),
                'metrics': benchmark_results.get('metrics', {}),
            }
            for key in ('pass_rate', 'truthfulness_proxy', 'by_kind', 'unanswerable_abstention_rate',
                        'old_value_rate', 'mode'):
                if key in benchmark_results:
                    entry[key] = benchmark_results[key]
            records = benchmark_results.get('records', [])
            if records:
                latencies = sorted(r['latency'] for r in records)
                entry['latency_p50'] = latencies[len(latencies) // 2]
                entry['latency_p95'] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
                entry['mean_compute_units'] = sum(r['compute_units'] for r in records) / len(records)
                entry['mean_output_tokens'] = sum(r['num_tokens'] for r in records) / len(records)
            report['benchmarks'][benchmark_name] = entry

        if hasattr(self.model, 'get_stats'):
            report['engine_stats'] = self.model.get_stats()
        if hasattr(self.model, 'memory_stats'):
            report['memory_stats'] = self.model.memory_stats()
        if torch.cuda.is_available():
            report['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()

        # Print report
        print(f"\n{'='*80}")
        print("EVALUATION REPORT")
        print(f"{'='*80}\n")
        print(json.dumps(report, indent=2, default=str))

        # Save to file
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n✓ Report saved to: {output_path}")

        if records_path:
            with open(records_path, 'w') as f:
                for benchmark_name, benchmark_results in results.items():
                    for record in benchmark_results.get('records', []):
                        f.write(json.dumps({'benchmark': benchmark_name, **record}, default=str) + "\n")
            print(f"✓ Per-example records saved to: {records_path}")

        return report
