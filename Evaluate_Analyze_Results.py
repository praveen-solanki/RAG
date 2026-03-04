"""
PERFORMANCE ANALYSIS & VISUALIZATION
=====================================
Analyze and visualize RAG system performance
"""

"""
python Evaluate_Analyze_Results.py --results results.json --output-dir analysis_dir
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Plotting disabled.")

import numpy as np


def load_results(filepath: str) -> Dict:
    """Load evaluation results"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_by_query_type(results: Dict) -> Dict:
    """Analyze performance by query type"""
    
    if "by_question_type" in results:
        return results["by_question_type"]
    
    # Group by query characteristics
    query_types = defaultdict(list)
    
    for result in results.get("detailed_results", []):
        query = result["question"].lower()
        found = result["metrics"]["found"]
        mrr = result["metrics"]["mrr"]
        
        # Categorize queries
        if any(word in query for word in ["what", "which", "who"]):
            query_types["factual"].append({"found": found, "mrr": mrr})
        elif any(word in query for word in ["how", "why"]):
            query_types["explanatory"].append({"found": found, "mrr": mrr})
        elif any(word in query for word in ["list", "name", "give"]):
            query_types["listing"].append({"found": found, "mrr": mrr})
        else:
            query_types["other"].append({"found": found, "mrr": mrr})
    
    # Aggregate
    analysis = {}
    for qtype, metrics in query_types.items():
        if metrics:
            analysis[qtype] = {
                "count": len(metrics),
                "success_rate": np.mean([m["found"] for m in metrics]),
                "mean_mrr": np.mean([m["mrr"] for m in metrics]),
            }
    
    return analysis


def analyze_failure_patterns(results: Dict) -> Dict:
    """Analyze common failure patterns"""
    
    failures = []
    
    for result in results.get("detailed_results", []):
        if not result["metrics"]["found"]:
            failures.append({
                "query": result["question"],
                "query_length": len(result["question"].split()),
                "expected_doc": result["ground_truth"],
            })
    
    if not failures:
        return {"total_failures": 0}
    
    # Analyze patterns
    query_lengths = [f["query_length"] for f in failures]
    
    return {
        "total_failures": len(failures),
        "avg_query_length": np.mean(query_lengths),
        "query_length_range": (min(query_lengths), max(query_lengths)),
        "sample_failures": failures[:5],  # Top 5 failures
    }


def analyze_score_distributions(results: Dict):
    """Analyze score distributions"""
    
    dense_scores = []
    sparse_scores = []
    rerank_scores = []
    
    for result in results.get("detailed_results", []):
        for res in result.get("results", []):
            if res.get("dense_score"):
                dense_scores.append(res["dense_score"])
            if res.get("sparse_score"):
                sparse_scores.append(res["sparse_score"])
            if res.get("rerank_score"):
                rerank_scores.append(res["rerank_score"])
    
    analysis = {}
    
    if dense_scores:
        analysis["dense"] = {
            "mean": np.mean(dense_scores),
            "std": np.std(dense_scores),
            "min": np.min(dense_scores),
            "max": np.max(dense_scores),
            "quartiles": [
                np.percentile(dense_scores, 25),
                np.percentile(dense_scores, 50),
                np.percentile(dense_scores, 75),
            ]
        }
    
    if sparse_scores:
        analysis["sparse"] = {
            "mean": np.mean(sparse_scores),
            "std": np.std(sparse_scores),
            "min": np.min(sparse_scores),
            "max": np.max(sparse_scores),
        }
    
    if rerank_scores:
        analysis["rerank"] = {
            "mean": np.mean(rerank_scores),
            "std": np.std(rerank_scores),
            "min": np.min(rerank_scores),
            "max": np.max(rerank_scores),
        }
    
    return analysis


def plot_metrics_comparison(results: Dict, output_path: str = "metrics_comparison.png"):
    """Plot metrics comparison"""
    
    if not HAS_PLOTTING:
        print("Plotting not available (install matplotlib and seaborn)")
        return
    
    metrics = results.get("aggregate_metrics", {})
    
    # Extract precision at different k
    k_values = [1, 3, 5, 10]
    precisions = []
    
    for k in k_values:
        key = f"precision@{k}"
        if key in metrics:
            precisions.append(metrics[key]["mean"])
        else:
            precisions.append(0)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Precision @ K
    axes[0].plot(k_values, precisions, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('K', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision @ K', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Plot 2: Overall metrics
    metric_names = []
    metric_values = []
    
    if "mrr" in metrics:
        metric_names.append("MRR")
        metric_values.append(metrics["mrr"]["mean"])
    
    if "found" in metrics:
        metric_names.append("Success\nRate")
        metric_values.append(metrics["found"]["mean"])
    
    if "precision@5" in metrics:
        metric_names.append("Precision\n@5")
        metric_values.append(metrics["precision@5"]["mean"])
    
    axes[1].bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Key Metrics', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")


def plot_latency_distribution(results: Dict, output_path: str = "latency_distribution.png"):
    """Plot latency distribution"""
    
    if not HAS_PLOTTING:
        return
    
    latencies = [r["latency_ms"] for r in results.get("detailed_results", [])]
    
    if not latencies:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(latencies, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(latencies), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(latencies):.1f}ms')
    axes[0].axvline(np.median(latencies), color='green', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(latencies):.1f}ms')
    axes[0].set_xlabel('Latency (ms)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Latency Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(latencies, vert=True)
    axes[1].set_ylabel('Latency (ms)', fontsize=12)
    axes[1].set_title('Latency Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")


def generate_report(results: Dict, output_path: str = "analysis_report.txt"):
    """Generate text report"""
    
    lines = []
    lines.append("="*80)
    lines.append("PERFORMANCE ANALYSIS REPORT")
    lines.append("="*80)
    lines.append("")
    
    # Summary
    summary = results.get("summary", {})
    lines.append("SUMMARY:")
    lines.append(f"  Total Questions: {summary.get('total_questions', 'N/A')}")
    lines.append(f"  Timestamp: {summary.get('timestamp', 'N/A')}")
    lines.append("")
    
    # Aggregate metrics
    metrics = results.get("aggregate_metrics", {})
    lines.append("AGGREGATE METRICS:")
    
    for metric_name, values in metrics.items():
        if isinstance(values, dict) and "mean" in values:
            lines.append(f"  {metric_name}:")
            lines.append(f"    Mean: {values['mean']:.4f}")
            lines.append(f"    Std:  {values['std']:.4f}")
            lines.append(f"    Min:  {values['min']:.4f}")
            lines.append(f"    Max:  {values['max']:.4f}")
    
    lines.append("")
    
    # Query type analysis
    query_analysis = analyze_by_query_type(results)
    if query_analysis:
        lines.append("PERFORMANCE BY QUERY TYPE:")
        for qtype, stats in query_analysis.items():
            lines.append(f"  {qtype.upper()}:")
            lines.append(f"    Count: {stats.get('count', 0)}")
            lines.append(f"    Success Rate: {stats.get('success_rate', 0):.2%}")
            lines.append(f"    Mean MRR: {stats.get('mean_mrr', 0):.4f}")
    
    lines.append("")
    
    # Failure analysis
    failure_analysis = analyze_failure_patterns(results)
    lines.append("FAILURE ANALYSIS:")
    lines.append(f"  Total Failures: {failure_analysis.get('total_failures', 0)}")
    
    if failure_analysis.get("avg_query_length"):
        lines.append(f"  Avg Query Length: {failure_analysis['avg_query_length']:.1f} words")
    
    if "sample_failures" in failure_analysis:
        lines.append("\n  Sample Failed Queries:")
        for i, failure in enumerate(failure_analysis["sample_failures"][:3], 1):
            lines.append(f"    {i}. {failure['query']}")
            lines.append(f"       Expected: {failure['expected_doc']}")
    
    lines.append("")
    
    # Score distributions
    score_dist = analyze_score_distributions(results)
    if score_dist:
        lines.append("SCORE DISTRIBUTIONS:")
        for score_type, stats in score_dist.items():
            lines.append(f"  {score_type.upper()}:")
            lines.append(f"    Mean: {stats['mean']:.4f}")
            lines.append(f"    Std:  {stats['std']:.4f}")
            lines.append(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    lines.append("")
    lines.append("="*80)
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Report saved to {output_path}")
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG system performance")
    parser.add_argument("--results", required=True, help="Evaluation results JSON")
    parser.add_argument("--output-dir", default="analysis", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    print("✓ Results loaded\n")
    
    # Generate report
    report_path = output_dir / "analysis_report.txt"
    generate_report(results, str(report_path))
    
    # Generate plots
    if not args.no_plots and HAS_PLOTTING:
        print("\nGenerating plots...")
        
        metrics_plot = output_dir / "metrics_comparison.png"
        plot_metrics_comparison(results, str(metrics_plot))
        
        latency_plot = output_dir / "latency_distribution.png"
        plot_latency_distribution(results, str(latency_plot))
    
    print(f"\n✓ Analysis complete! Results in {output_dir}/")


if __name__ == "__main__":
    main()