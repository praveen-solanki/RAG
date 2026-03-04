#!/usr/bin/env python3
"""
DETAILED RESULTS ANALYZER FOR 100-QUESTION EVALUATION
======================================================
Analyzes evaluation results with focus on question types, difficulty, and documents
"""
"""
# Generate detailed analysis and visualizations
python analyze_100q_results.py \
    --results autosar_evaluation_results.json \
    --output-dir autosar_analysis
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    sns.set_style('whitegrid')
except ImportError:
    HAS_PLOTTING = False


class ResultsAnalyzer:
    """Analyze 100-question evaluation results"""
    
    def __init__(self, results: Dict):
        self.results = results
        self.questions = results.get('detailed_results', [])
        
    def generate_markdown_report(self) -> str:
        """Generate detailed Markdown report"""
        
        lines = []
        lines.append("# Comprehensive Evaluation Report")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        
        # Overview
        info = self.results['evaluation_info']
        agg = self.results['aggregate_metrics']
        
        lines.append(f"- **Total Questions**: {info['total_questions']}")
        lines.append(f"- **Evaluation Time**: {info['total_time_seconds']/60:.2f} minutes")
        lines.append(f"- **Success Rate**: {agg['found']['mean']:.2%}")
        lines.append(f"- **Mean MRR**: {agg['mrr']['mean']:.4f}")
        lines.append(f"- **Precision@5**: {agg['precision@5']['mean']:.4f}")
        lines.append("")
        
        # Performance by question type
        lines.append("## Performance by Question Type")
        lines.append("")
        lines.append("| Type | Count | Success Rate | MRR | P@5 |")
        lines.append("|------|-------|--------------|-----|-----|")
        
        for qtype, stats in sorted(self.results['by_question_type'].items()):
            lines.append(f"| {qtype} | {stats['count']} | {stats['success_rate']:.2%} | "
                        f"{stats['mean_mrr']:.4f} | {stats['mean_precision@5']:.4f} |")
        
        lines.append("")
        
        # Performance by difficulty
        lines.append("## Performance by Difficulty")
        lines.append("")
        lines.append("| Difficulty | Count | Success Rate | MRR | P@5 |")
        lines.append("|------------|-------|--------------|-----|-----|")
        
        for diff, stats in sorted(self.results['by_difficulty'].items()):
            lines.append(f"| {diff} | {stats['count']} | {stats['success_rate']:.2%} | "
                        f"{stats['mean_mrr']:.4f} | {stats['mean_precision@5']:.4f} |")
        
        lines.append("")
        
        # Top performing documents
        lines.append("## Top 5 Best Performing Documents")
        lines.append("")
        lines.append("| Document | Questions | Success Rate | MRR |")
        lines.append("|----------|-----------|--------------|-----|")
        
        by_doc = sorted(self.results['by_document'].items(), 
                       key=lambda x: x[1]['success_rate'], 
                       reverse=True)
        
        for doc, stats in by_doc[:5]:
            doc_name = Path(doc).name
            lines.append(f"| {doc_name} | {stats['questions']} | "
                        f"{stats['success_rate']:.2%} | {stats['mean_mrr']:.4f} |")
        
        lines.append("")
        
        # Worst performing documents
        lines.append("## Top 5 Worst Performing Documents")
        lines.append("")
        lines.append("| Document | Questions | Success Rate | MRR |")
        lines.append("|----------|-----------|--------------|-----|")
        
        for doc, stats in by_doc[-5:]:
            doc_name = Path(doc).name
            lines.append(f"| {doc_name} | {stats['questions']} | "
                        f"{stats['success_rate']:.2%} | {stats['mean_mrr']:.4f} |")
        
        lines.append("")
        
        # Failure analysis
        fail = self.results['failure_analysis']
        lines.append("## Failure Analysis")
        lines.append("")
        lines.append(f"**Total Failures**: {fail['total_failures']} ({fail['failure_rate']:.2%})")
        lines.append("")
        
        if fail['sample_failures']:
            lines.append("### Sample Failed Queries")
            lines.append("")
            
            for i, failure in enumerate(fail['sample_failures'][:5], 1):
                lines.append(f"**{i}. {failure['question']}**")
                lines.append(f"- Expected: `{failure['expected']}`")
                lines.append(f"- Got: {', '.join([f'`{d}`' for d in failure['got_top3'][:3]])}")
                lines.append(f"- Type: {failure['type']}, Difficulty: {failure['difficulty']}")
                lines.append("")
        
        # Latency stats
        lat = self.results['latency_stats']
        lines.append("## Latency Statistics")
        lines.append("")
        lines.append(f"- Mean: {lat['mean_ms']:.2f} ms")
        lines.append(f"- Median: {lat['median_ms']:.2f} ms")
        lines.append(f"- P95: {lat['p95_ms']:.2f} ms")
        lines.append(f"- P99: {lat['p99_ms']:.2f} ms")
        lines.append("")
        
        return "\n".join(lines)
    
    def plot_performance_by_type(self, output_path: str):
        """Plot performance by question type"""
        
        if not HAS_PLOTTING:
            return
        
        by_type = self.results['by_question_type']
        
        types = list(by_type.keys())
        success_rates = [by_type[t]['success_rate'] for t in types]
        mrrs = [by_type[t]['mean_mrr'] for t in types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Success rates
        bars1 = ax1.bar(range(len(types)), success_rates, color='skyblue', edgecolor='black')
        ax1.set_xticks(range(len(types)))
        ax1.set_xticklabels(types, rotation=45, ha='right')
        ax1.set_ylabel('Success Rate', fontsize=12)
        ax1.set_title('Success Rate by Question Type', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=10)
        
        # MRR
        bars2 = ax2.bar(range(len(types)), mrrs, color='lightcoral', edgecolor='black')
        ax2.set_xticks(range(len(types)))
        ax2.set_xticklabels(types, rotation=45, ha='right')
        ax2.set_ylabel('Mean Reciprocal Rank', fontsize=12)
        ax2.set_title('MRR by Question Type', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars2, mrrs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {output_path}")
    
    def plot_performance_by_difficulty(self, output_path: str):
        """Plot performance by difficulty"""
        
        if not HAS_PLOTTING:
            return
        
        by_diff = self.results['by_difficulty']
        
        # Sort by typical difficulty order
        order = ['easy', 'medium', 'hard']
        diffs = [d for d in order if d in by_diff]
        success_rates = [by_diff[d]['success_rate'] for d in diffs]
        mrrs = [by_diff[d]['mean_mrr'] for d in diffs]
        counts = [by_diff[d]['count'] for d in diffs]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        # Success rates
        bars1 = axes[0].bar(range(len(diffs)), success_rates, color=colors, edgecolor='black')
        axes[0].set_xticks(range(len(diffs)))
        axes[0].set_xticklabels(diffs)
        axes[0].set_ylabel('Success Rate', fontsize=12)
        axes[0].set_title('Success Rate by Difficulty', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0, 1.05])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, success_rates):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2%}', ha='center', va='bottom', fontsize=11)
        
        # MRR
        bars2 = axes[1].bar(range(len(diffs)), mrrs, color=colors, edgecolor='black')
        axes[1].set_xticks(range(len(diffs)))
        axes[1].set_xticklabels(diffs)
        axes[1].set_ylabel('Mean Reciprocal Rank', fontsize=12)
        axes[1].set_title('MRR by Difficulty', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, mrrs):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=11)
        
        # Count
        bars3 = axes[2].bar(range(len(diffs)), counts, color=colors, edgecolor='black')
        axes[2].set_xticks(range(len(diffs)))
        axes[2].set_xticklabels(diffs)
        axes[2].set_ylabel('Number of Questions', fontsize=12)
        axes[2].set_title('Questions by Difficulty', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars3, counts):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {output_path}")
    
    def plot_document_performance(self, output_path: str):
        """Plot top and bottom performing documents"""
        
        if not HAS_PLOTTING:
            return
        
        by_doc = sorted(self.results['by_document'].items(), 
                       key=lambda x: x[1]['success_rate'])
        
        # Get top 5 and bottom 5
        bottom_5 = by_doc[:5]
        top_5 = by_doc[-5:]
        
        docs = [Path(d[0]).name[:30] for d in bottom_5 + top_5]
        success = [d[1]['success_rate'] for d in bottom_5 + top_5]
        colors_list = ['#e74c3c'] * 5 + ['#2ecc71'] * 5
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(docs)), success, color=colors_list, edgecolor='black')
        ax.set_yticks(range(len(docs)))
        ax.set_yticklabels(docs, fontsize=10)
        ax.set_xlabel('Success Rate', fontsize=12)
        ax.set_title('Top 5 & Bottom 5 Documents by Success Rate', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, success):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.2%}', va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Bottom 5'),
            Patch(facecolor='#2ecc71', label='Top 5')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {output_path}")
    
    def plot_precision_recall_curve(self, output_path: str):
        """Plot Precision@K and Recall@K curves"""
        
        if not HAS_PLOTTING:
            return
        
        agg = self.results['aggregate_metrics']
        
        k_values = [1, 3, 5, 10]
        precisions = [agg[f'precision@{k}']['mean'] for k in k_values]
        recalls = [agg[f'recall@{k}']['mean'] for k in k_values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(k_values, precisions, marker='o', linewidth=2, markersize=10,
               label='Precision@K', color='#3498db')
        ax.plot(k_values, recalls, marker='s', linewidth=2, markersize=10,
               label='Recall@K', color='#e74c3c')
        
        ax.set_xlabel('K (Number of Retrieved Documents)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision and Recall @ K', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add value labels
        for k, p, r in zip(k_values, precisions, recalls):
            ax.text(k, p + 0.03, f'{p:.3f}', ha='center', fontsize=9)
            ax.text(k, r - 0.05, f'{r:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze 100-question evaluation results")
    parser.add_argument('--results', required=True, help='Evaluation results JSON')
    parser.add_argument('--output-dir', default='analysis_output', help='Output directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    with open(args.results, 'r') as f:
        results = json.load(f)
    print("✓ Results loaded\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(results)
    
    # Generate markdown report
    print("Generating markdown report...")
    report = analyzer.generate_markdown_report()
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Report saved: {report_path}\n")
    
    # Generate plots
    if not args.no_plots and HAS_PLOTTING:
        print("Generating plots...")
        
        analyzer.plot_performance_by_type(
            str(output_dir / "performance_by_type.png")
        )
        
        analyzer.plot_performance_by_difficulty(
            str(output_dir / "performance_by_difficulty.png")
        )
        
        analyzer.plot_document_performance(
            str(output_dir / "document_performance.png")
        )
        
        analyzer.plot_precision_recall_curve(
            str(output_dir / "precision_recall_curve.png")
        )
        
        print()
    
    print(f"✓ Analysis complete! Results in {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  • evaluation_report.md")
    if not args.no_plots and HAS_PLOTTING:
        print(f"  • performance_by_type.png")
        print(f"  • performance_by_difficulty.png")
        print(f"  • document_performance.png")
        print(f"  • precision_recall_curve.png")


if __name__ == "__main__":
    main()