"""
Main script for comparing zero-shot LLM vs fine-tuned classifier.
Runs complete evaluation pipeline and generates comparison report.
"""

import os
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from utils.llm_client import LLMClient, get_clause_type_examples
from utils.data_loader import load_cuad_dataset, preprocess_data, get_clause_distribution
from utils.classifier import FineTunedClassifier
from utils.metrics import (
    calculate_metrics, aggregate_metrics, print_comparison_table,
    ClassificationMetrics, InferenceStats
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Complete comparison result."""
    zero_shot_metrics: Dict[str, ClassificationMetrics]
    fine_tuned_metrics: Dict[str, ClassificationMetrics]
    zero_shot_stats: InferenceStats
    fine_tuned_stats: InferenceStats
    training_result: Optional[object]
    cost_per_document: Dict[str, float]


def run_comparison(
    max_samples: Optional[int] = None,
    clause_types: Optional[List[str]] = None,
    quick_test: bool = False
) -> ComparisonResult:
    """Run full comparison between classifiers.
    
    Args:
        max_samples: Maximum number of samples to evaluate
        clause_types: List of clause types to classify
        quick_test: If True, use smaller samples for faster testing
        
    Returns:
        ComparisonResult with all metrics and statistics
    """
    if clause_types is None:
        clause_types = config.data.clause_types
    
    print("=" * 80)
    print("CONTRACT CLAUSE CLASSIFIER COMPARISON")
    print("=" * 80)
    print(f"\nClause types: {', '.join(clause_types)}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    
    # Load dataset
    print("\n" + "-" * 40)
    print("Loading Dataset...")
    print("-" * 40)
    contracts = load_cuad_dataset(
        split="test",
        max_samples=max_samples if not quick_test else 50
    )
    logger.info(f"Loaded {len(contracts)} contracts")
    
    # Get clause distribution
    distribution = get_clause_distribution(contracts)
    print("\nClause Distribution:")
    print(distribution.to_string())
    
    # Preprocess data
    print("\n" + "-" * 40)
    print("Preprocessing Data...")
    print("-" * 40)
    
    texts_by_clause = {ct: [] for ct in clause_types}
    labels_by_clause = {ct: [] for ct in clause_types}
    
    for contract in contracts:
        for clause_type in clause_types:
            text = contract.text[:512]  # Limit for efficiency
            is_present = contract.clauses.get(clause_type, False)
            texts_by_clause[clause_type].append(text)
            labels_by_clause[clause_type].append(1 if is_present else 0)
    
    # Initialize classifiers
    print("\n" + "-" * 40)
    print("Initializing Classifiers...")
    print("-" * 40)
    
    # Zero-shot LLM
    llm_client = LLMClient()
    logger.info(f"Zero-shot model: {config.llm.provider}/{config.llm.model}")
    
    # Fine-tuned model
    fine_tuned = FineTunedClassifier()
    training_result = None
    
    if quick_test:
        # Skip training in quick test mode
        logger.info("Quick test mode: Skipping fine-tuned model training")
    else:
        # Train fine-tuned model
        print("\n" + "-" * 40)
        print("Training Fine-Tuned Model...")
        print("-" * 40)
        
        # Use subset for training (we need labeled data)
        train_contracts = load_cuad_dataset(
            split="train",
            max_samples=1000 if max_samples is None else min(max_samples, 1000)
        )
        
        train_texts, train_labels = preprocess_data(train_contracts, clause_types)
        
        train_dataset, val_dataset, _ = fine_tuned.prepare_data(
            train_texts, train_labels, test_size=0.2, val_size=0.1
        )
        
        training_result = fine_tuned.train(train_dataset, val_dataset)
        print(f"\nTraining completed in {training_result.training_time_seconds:.2f} seconds")
        print(f"Training metrics: {training_result.metrics}")
    
    # Evaluate Zero-Shot LLM
    print("\n" + "-" * 40)
    print("Evaluating Zero-Shot LLM...")
    print("-" * 40)
    
    zero_shot_metrics = {}
    zero_shot_costs = []
    zero_shot_latencies = []
    
    for clause_type in clause_types:
        print(f"\nEvaluating clause: {clause_type}")
        
        texts = texts_by_clause[clause_type]
        labels = labels_by_clause[clause_type]
        
        predictions = []
        start_times = []
        costs = []
        
        # Sample for quick test
        eval_texts = texts[:100] if quick_test else texts
        eval_labels = labels[:100] if quick_test else labels
        
        for i, text in enumerate(eval_texts):
            start_times.append(time.time())
            response = llm_client.classify_single(text, clause_type)
            start_times[-1] = time.time() - start_times[-1]
            
            is_present = response.text.strip().upper().startswith("YES")
            predictions.append(1 if is_present else 0)
            costs.append(response.cost_usd)
            zero_shot_latencies.append(response.latency_ms)
        
        # Calculate metrics
        metrics = calculate_metrics(eval_labels, predictions, clause_type)
        zero_shot_metrics[clause_type] = metrics
        
        # Calculate costs
        total_cost = sum(costs)
        zero_shot_costs.append(total_cost)
        logger.info(f"Zero-shot cost for {clause_type}: ${total_cost:.6f}")
    
    avg_zero_shot_latency = np.mean(zero_shot_latencies) if zero_shot_latencies else 0
    
    # Evaluate Fine-Tuned Model
    print("\n" + "-" * 40)
    print("Evaluating Fine-Tuned Model...")
    print("-" * 40)
    
    fine_tuned_metrics = {}
    fine_tuned_costs = []
    fine_tuned_latencies = []
    
    # Load fine-tuned model if not trained (for comparison)
    if not quick_test and training_result is None:
        model_path = os.path.join(config.paths.models_dir, "fine_tuned")
        if os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            fine_tuned.load(model_path)
    
    for clause_type in clause_types:
        print(f"\nEvaluating clause: {clause_type}")
        
        texts = texts_by_clause[clause_type]
        labels = labels_by_clause[clause_type]
        
        # Sample for quick test
        eval_texts = texts[:100] if quick_test else texts
        eval_labels = labels[:100] if quick_test else labels
        
        # Predict
        predictions, probs, avg_latency = fine_tuned.predict(eval_texts)
        fine_tuned_latencies.append(avg_latency)
        
        # Calculate metrics
        metrics = calculate_metrics(eval_labels, predictions, clause_type)
        fine_tuned_metrics[clause_type] = metrics
        
        # Estimate cost (GPU time + inference)
        # Using approximate cost based on model size
        estimated_cost = (avg_latency / 1000) * 0.001  # $0.001 per second GPU time
        fine_tuned_costs.append(estimated_cost)
        logger.info(f"Fine-tuned cost for {clause_type}: ${estimated_cost:.6f}")
    
    avg_fine_tuned_latency = np.mean(fine_tuned_latencies) if fine_tuned_latencies else 0
    
    # Calculate cost per document
    num_documents = len(contracts)
    zero_shot_cost_per_doc = np.sum(zero_shot_costs) / num_documents if num_documents > 0 else 0
    fine_tuned_cost_per_doc = np.sum(fine_tuned_costs) / num_documents if num_documents > 0 else 0
    
    # Create comparison result
    result = ComparisonResult(
        zero_shot_metrics=zero_shot_metrics,
        fine_tuned_metrics=fine_tuned_metrics,
        zero_shot_stats=InferenceStats(
            total_latency_ms=sum(zero_shot_latencies),
            avg_latency_ms=avg_zero_shot_latency,
            min_latency_ms=min(zero_shot_latencies) if zero_shot_latencies else 0,
            max_latency_ms=max(zero_shot_latencies) if zero_shot_latencies else 0,
            total_cost_usd=sum(zero_shot_costs),
            avg_cost_usd=zero_shot_cost_per_doc,
            input_tokens=0,
            output_tokens=0
        ),
        fine_tuned_stats=InferenceStats(
            total_latency_ms=sum(fine_tuned_latencies),
            avg_latency_ms=avg_fine_tuned_latency,
            min_latency_ms=min(fine_tuned_latencies) if fine_tuned_latencies else 0,
            max_latency_ms=max(fine_tuned_latencies) if fine_tuned_latencies else 0,
            total_cost_usd=sum(fine_tuned_costs),
            avg_cost_usd=fine_tuned_cost_per_doc,
            input_tokens=0,
            output_tokens=0
        ),
        training_result=training_result,
        cost_per_document={
            "zero_shot": zero_shot_cost_per_doc,
            "fine_tuned": fine_tuned_cost_per_doc
        }
    )
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print_comparison_table(zero_shot_metrics, fine_tuned_metrics)
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("SUMMARY STATISTICS")
    print("-" * 40)
    
    zero_shot_agg = aggregate_metrics(zero_shot_metrics)
    fine_tuned_agg = aggregate_metrics(fine_tuned_metrics)
    
    print("\nZero-Shot LLM (Aggregate):")
    for key, value in zero_shot_agg.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nFine-Tuned Model (Aggregate):")
    for key, value in fine_tuned_agg.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "-" * 40)
    print("COST & LATENCY COMPARISON")
    print("-" * 40)
    print(f"\nZero-Shot LLM:")
    print(f"  Cost per document: ${result.cost_per_document['zero_shot']:.6f}")
    print(f"  Avg latency: {result.zero_shot_stats.avg_latency_ms:.2f} ms")
    
    print(f"\nFine-Tuned Model:")
    print(f"  Cost per document: ${result.cost_per_document['fine_tuned']:.6f}")
    print(f"  Avg latency: {result.fine_tuned_stats.avg_latency_ms:.2f} ms")
    
    # Save results
    save_results(result)
    
    return result


def save_results(result: ComparisonResult):
    """Save comparison results to files."""
    outputs_dir = config.paths.outputs_dir
    
    # Save metrics as CSV
    metrics_data = []
    for clause_type in result.zero_shot_metrics.keys():
        zs = result.zero_shot_metrics[clause_type]
        ft = result.fine_tuned_metrics[clause_type]
        
        metrics_data.extend([
            {"clause_type": clause_type, "method": "zero_shot", "metric": "precision", "value": zs.precision},
            {"clause_type": clause_type, "method": "zero_shot", "metric": "recall", "value": zs.recall},
            {"clause_type": clause_type, "method": "zero_shot", "metric": "f1", "value": zs.f1},
            {"clause_type": clause_type, "method": "zero_shot", "metric": "accuracy", "value": zs.accuracy},
            {"clause_type": clause_type, "method": "fine_tuned", "metric": "precision", "value": ft.precision},
            {"clause_type": clause_type, "method": "fine_tuned", "metric": "recall", "value": ft.recall},
            {"clause_type": clause_type, "method": "fine_tuned", "metric": "f1", "value": ft.f1},
            {"clause_type": clause_type, "method": "fine_tuned", "metric": "accuracy", "value": ft.accuracy},
        ])
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(outputs_dir, "comparison_metrics.csv"), index=False)
    logger.info(f"Saved metrics to {outputs_dir}/comparison_metrics.csv")
    
    # Save summary
    summary = f"""# Contract Clause Classifier Comparison Report

## Zero-Shot LLM ({config.llm.provider}/{config.llm.model})

- Cost per document: ${result.cost_per_document['zero_shot']:.6f}
- Average latency: {result.zero_shot_stats.avg_latency_ms:.2f} ms

## Fine-Tuned Model ({config.training.model_name})

- Cost per document: ${result.cost_per_document['fine_tuned']:.6f}
- Average latency: {result.fine_tuned_stats.avg_latency_ms:.2f} ms

## Training Details

- Model: {config.training.model_name}
- Epochs: {config.training.num_epochs}
- Batch size: {config.training.batch_size}
- Learning rate: {config.training.learning_rate}

## Aggregate Metrics

### Zero-Shot LLM
"""
    
    for key, value in aggregate_metrics(result.zero_shot_metrics).items():
        summary += f"- {key}: {value:.4f}\n"
    
    summary += """
### Fine-Tuned Model
"""
    
    for key, value in aggregate_metrics(result.fine_tuned_metrics).items():
        summary += f"- {key}: {value:.4f}\n"
    
    with open(os.path.join(outputs_dir, "summary.md"), "w") as f:
        f.write(summary)
    
    logger.info(f"Saved summary to {outputs_dir}/summary.md")


def plot_results(result: ComparisonResult, save_path: Optional[str] = None):
    """Create visualization of comparison results.
    
    Args:
        result: ComparisonResult from run_comparison
        save_path: Optional path to save figure
    """
    clause_types = list(result.zero_shot_metrics.keys())
    
    # Prepare data for plotting
    metrics = ["precision", "recall", "f1", "accuracy"]
    zs_data = {m: [] for m in metrics}
    ft_data = {m: [] for m in metrics}
    
    for clause_type in clause_types:
        zs = result.zero_shot_metrics[clause_type]
        ft = result.fine_tuned_metrics[clause_type]
        
        for m in metrics:
            zs_data[m].append(getattr(zs, m))
            ft_data[m].append(getattr(ft, m))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    x = np.arange(len(clause_types))
    width = 0.35
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        zs_bars = ax.bar(x - width/2, zs_data[metric], width, 
                        label='Zero-Shot LLM', color='#2ecc71', alpha=0.8)
        ft_bars = ax.bar(x + width/2, ft_data[metric], width,
                        label='Fine-Tuned', color='#3498db', alpha=0.8)
        
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(clause_types, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(zs_bars, zs_data[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(ft_bars, ft_data[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()


def generate_report(result: ComparisonResult, output_path: str):
    """Generate comprehensive comparison report.
    
    Args:
        result: ComparisonResult from run_comparison
        output_path: Path to save report
    """
    report = f"""# Contract Clause Classification - Model Comparison Report

## Executive Summary

This report compares a zero-shot LLM approach against a fine-tuned transformer model
for classifying contract clauses in the CUAD dataset.

## Methodology

### Zero-Shot LLM Approach
- Model: {config.llm.provider}/{config.llm.model}
- Method: Direct prompt-based classification without training
- Cost: Pay-per-token based on {config.llm.provider} pricing

### Fine-Tuned Model Approach
- Base Model: {config.training.model_name}
- Training: Fine-tuned on CUAD training set
- Inference: Local execution on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}

## Dataset Statistics

- Total documents: {len(load_cuad_dataset(split='test', max_samples=None))}
- Clause types analyzed: {len(result.zero_shot_metrics)}
- Evaluation set: Test split

## Performance Comparison

### Aggregate Metrics

| Metric | Zero-Shot LLM | Fine-Tuned |
|--------|---------------|------------|
| Average Precision | {aggregate_metrics(result.zero_shot_metrics).get('avg_precision', 0):.4f} | {aggregate_metrics(result.fine_tuned_metrics).get('avg_precision', 0):.4f} |
| Average Recall | {aggregate_metrics(result.zero_shot_metrics).get('avg_recall', 0):.4f} | {aggregate_metrics(result.fine_tuned_metrics).get('avg_recall', 0):.4f} |
| Average F1 Score | {aggregate_metrics(result.zero_shot_metrics).get('avg_f1', 0):.4f} | {aggregate_metrics(result.fine_tuned_metrics).get('avg_f1', 0):.4f} |
| Average Accuracy | {aggregate_metrics(result.zero_shot_metrics).get('avg_accuracy', 0):.4f} | {aggregate_metrics(result.fine_tuned_metrics).get('avg_accuracy', 0):.4f} |

### Cost Analysis

| Metric | Zero-Shot LLM | Fine-Tuned |
|--------|---------------|------------|
| Cost per Document | ${result.cost_per_document['zero_shot']:.6f} | ${result.cost_per_document['fine_tuned']:.6f} |

### Latency Analysis

| Metric | Zero-Shot LLM | Fine-Tuned |
|--------|---------------|------------|
| Average Latency | {result.zero_shot_stats.avg_latency_ms:.2f} ms | {result.fine_tuned_stats.avg_latency_ms:.2f} ms |

## Clause-Type Performance

"""
    
    for clause_type in result.zero_shot_metrics.keys():
        zs = result.zero_shot_metrics[clause_type]
        ft = result.fine_tuned_metrics[clause_type]
        
        report += f"""
### {clause_type}

| Metric | Zero-Shot LLM | Fine-Tuned | Improvement |
|--------|---------------|------------|-------------|
| Precision | {zs.precision:.4f} | {ft.precision:.4f} | {ft.precision - zs.precision:+.4f} |
| Recall | {zs.recall:.4f} | {ft.recall:.4f} | {ft.recall - zs.recall:+.4f} |
| F1 Score | {zs.f1:.4f} | {ft.f1:.4f} | {ft.f1 - zs.f1:+.4f} |
| Accuracy | {zs.accuracy:.4f} | {ft.accuracy:.4f} | {ft.accuracy - zs.accuracy:+.4f} |

"""
    
    report += """
## Engineering Judgment: When is Fine-Tuning Worth It?

### Cost-Benefit Analysis

1. **High Volume Scenarios** (>1000 documents/day)
   - Fine-tuned model becomes cost-effective
   - Predictable costs vs. variable LLM API costs
   - Lower per-document cost at scale

2. **Low Latency Requirements** (<50ms response)
   - Fine-tuned model runs locally
   - No network overhead
   - Consistent response times

3. **Data Privacy Concerns**
   - Fine-tuned: Data stays in-house
   - Zero-shot: Data sent to LLM provider

4. **Model Control & Customization**
   - Fine-tuned: Full control over behavior
   - Zero-shot: Dependent on LLM provider updates

### Recommendations

- **Use Zero-Shot LLM when:**
  - Low volume (few docs/day)
  - Quick prototype/POC needed
  - Limited ML expertise
  - Data privacy not critical

- **Use Fine-Tuned Model when:**
  - High volume processing
  - Cost optimization important
  - Strict latency requirements
  - Data privacy required
  - Long-term deployment

## Conclusion

The fine-tuned model typically provides better accuracy and lower operational costs
at scale, while zero-shot LLM offers faster prototyping and lower initial complexity.

## Configuration

- LLM Provider: {config.llm.provider}
- LLM Model: {config.llm.model}
- Training Model: {config.training.model_name}
- Training Epochs: {config.training.num_epochs}
"""
    
    with open(output_path, "w") as f:
        f.write(report)
    
    logger.info(f"Generated report: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare contract clause classifiers")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to evaluate")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test with limited samples")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.output:
        config.paths.outputs_dir = args.output
        os.makedirs(args.output, exist_ok=True)
    
    result = run_comparison(
        max_samples=args.max_samples,
        quick_test=args.quick_test
    )
    
    # Generate visualizations
    plot_path = os.path.join(config.paths.outputs_dir, "comparison_plot.png")
    plot_results(result, save_path=plot_path)
    
    # Generate comprehensive report
    report_path = os.path.join(config.paths.outputs_dir, "comparison_report.md")
    generate_report(result, report_path)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.paths.outputs_dir}")
    print(f"  - Metrics CSV: comparison_metrics.csv")
    print(f"  - Summary: summary.md")
    print(f"  - Plot: comparison_plot.png")
    print(f"  - Full Report: comparison_report.md")
