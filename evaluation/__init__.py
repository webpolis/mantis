"""
HMST Evaluation Framework

Provides benchmark runners and evaluation utilities for HMST models.
"""

from evaluation.benchmarks import (
    MMLURunner,
    TruthfulQARunner,
    HumanEvalRunner,
    GSM8KRunner
)
from evaluation.metrics import (
    compute_accuracy,
    compute_f1_score,
    compute_hallucination_rate,
    compute_calibration_error,
    compute_metrics_summary
)

__all__ = [
    'MMLURunner',
    'TruthfulQARunner',
    'HumanEvalRunner',
    'GSM8KRunner',
    'compute_accuracy',
    'compute_f1_score',
    'compute_hallucination_rate',
    'compute_calibration_error',
    'compute_metrics_summary'
]


class EvaluationHarness:
    """
    Main evaluation harness for running benchmarks on HMST models.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize benchmark runners
        self.runners = {
            'mmlu': MMLURunner(model, tokenizer, device),
            'truthfulqa': TruthfulQARunner(model, tokenizer, device),
            'humaneval': HumanEvalRunner(model, tokenizer, device),
            'gsm8k': GSM8KRunner(model, tokenizer, device)
        }

    def run_benchmark(self, benchmark_name: str, dataset) -> dict:
        """
        Run a specific benchmark.

        Args:
            benchmark_name: Name of benchmark ('mmlu', 'truthfulqa', etc.)
            dataset: Dataset to evaluate on

        Returns:
            Results dictionary with predictions and metrics
        """
        if benchmark_name not in self.runners:
            raise ValueError(
                f"Unknown benchmark: {benchmark_name}. "
                f"Available: {list(self.runners.keys())}"
            )

        runner = self.runners[benchmark_name]
        results = runner.run(dataset)

        # Compute metrics
        if 'predictions' in results and 'targets' in results:
            metrics = compute_metrics_summary(
                results['predictions'],
                results['targets'],
                results.get('confidences')
            )
            results['metrics'] = metrics

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
                if 'metrics' in results:
                    print(f"\nResults:")
                    for metric_name, score in results['metrics'].items():
                        print(f"  {metric_name}: {score:.4f}")
                if 'pass_rate' in results:
                    print(f"  pass_rate: {results['pass_rate']:.4f}")
                if 'truthfulness_rate' in results:
                    print(f"  truthfulness_rate: {results['truthfulness_rate']:.4f}")

        return all_results

    def generate_report(self, results: dict, output_path: str = None):
        """
        Generate evaluation report.

        Args:
            results: Results from run_all_benchmarks()
            output_path: Optional path to save report
        """
        import json

        report = {
            'model': str(type(self.model).__name__),
            'benchmarks': {}
        }

        for benchmark_name, benchmark_results in results.items():
            report['benchmarks'][benchmark_name] = {
                'num_examples': benchmark_results.get('num_examples', 0),
                'metrics': benchmark_results.get('metrics', {}),
                'pass_rate': benchmark_results.get('pass_rate'),
                'truthfulness_rate': benchmark_results.get('truthfulness_rate')
            }

        # Print report
        print(f"\n{'='*80}")
        print("EVALUATION REPORT")
        print(f"{'='*80}\n")
        print(json.dumps(report, indent=2))

        # Save to file
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n✓ Report saved to: {output_path}")

        return report
