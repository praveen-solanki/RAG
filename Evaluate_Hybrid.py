"""
COMPREHENSIVE EVALUATION SYSTEM
================================
Evaluates hybrid retrieval system with detailed analysis
"""
"""
python Evaluate_Hybrid.py \
    --questions evaluation_questions.json \
    --collection rag_hybrid_bge_m3 \
    --top-k 10 \
    --output results.json
"""
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

import numpy as np
from Evaluate_Retrieval_With_Reranker_Template import HybridRetriever, AdvancedEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_evaluation_questions(path: str) -> List[Dict]:
    """Load evaluation questions from JSON"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get("questions", [])


def print_summary(results: Dict[str, Any]):
    """Print evaluation summary"""
    
    print("\n" + "="*80)
    print("HYBRID RETRIEVAL EVALUATION RESULTS")
    print("="*80)
    
    summary = results["summary"]
    print(f"\n📊 Total questions: {summary['total_questions']}")
    print(f"📅 Timestamp: {summary['timestamp']}")
    
    print("\n" + "-"*80)
    print("AGGREGATE METRICS")
    print("-"*80)
    
    metrics = results["aggregate_metrics"]
    
    print("\n📍 PRECISION @ K:")
    for k in [1, 3, 5, 10]:
        if f"precision@{k}" in metrics:
            mean = metrics[f"precision@{k}"]["mean"]
            std = metrics[f"precision@{k}"]["std"]
            print(f"  P@{k:2d}: {mean:.3f} ± {std:.3f}")
    
    if "mrr" in metrics:
        mrr = metrics["mrr"]["mean"]
        print(f"\n🎯 MRR: {mrr:.3f}")
    
    if "found" in metrics:
        found_rate = metrics["found"]["mean"]
        print(f"\n✅ Success Rate: {found_rate:.1%}")
    
    print("\n" + "="*80)


def analyze_failure_cases(results: Dict[str, Any], top_n: int = 10):
    """Analyze queries where retrieval failed"""
    
    failures = []
    
    for result in results["detailed_results"]:
        if not result["metrics"]["found"]:
            failures.append({
                "question": result["question"],
                "ground_truth": result["ground_truth"],
                "top_retrieved": result["retrieved_docs"][:3],
            })
    
    if failures:
        print("\n" + "="*80)
        print(f"FAILURE ANALYSIS - Top {min(top_n, len(failures))} Failed Queries")
        print("="*80)
        
        for i, failure in enumerate(failures[:top_n], 1):
            print(f"\n{i}. Question: {failure['question']}")
            print(f"   Expected: {failure['ground_truth']}")
            print(f"   Got: {', '.join(failure['top_retrieved'])}")


def compare_scores(results: Dict[str, Any]):
    """Compare dense, sparse, and rerank scores"""
    
    print("\n" + "="*80)
    print("SCORE ANALYSIS")
    print("="*80)
    
    dense_scores = []
    sparse_scores = []
    rerank_scores = []
    hybrid_scores = []
    
    for result in results["detailed_results"]:
        for res in result.get("results", [])[:5]:  # Top 5
            if res.get("dense_score"):
                dense_scores.append(res["dense_score"])
            if res.get("sparse_score"):
                sparse_scores.append(res["sparse_score"])
            if res.get("rerank_score"):
                rerank_scores.append(res["rerank_score"])
            hybrid_scores.append(res["score"])
    
    print("\nScore distributions (mean ± std):")
    if dense_scores:
        print(f"  Dense:  {np.mean(dense_scores):.4f} ± {np.std(dense_scores):.4f}")
    if sparse_scores:
        print(f"  Sparse: {np.mean(sparse_scores):.4f} ± {np.std(sparse_scores):.4f}")
    if hybrid_scores:
        print(f"  Hybrid: {np.mean(hybrid_scores):.4f} ± {np.std(hybrid_scores):.4f}")
    if rerank_scores:
        print(f"  Rerank: {np.mean(rerank_scores):.4f} ± {np.std(rerank_scores):.4f}")


def analyze_latency(results: Dict[str, Any]):
    """Analyze query latency"""
    
    latencies = [r["latency_ms"] for r in results["detailed_results"]]
    
    print("\n" + "="*80)
    print("LATENCY ANALYSIS")
    print("="*80)
    
    print(f"\n  Mean:   {np.mean(latencies):.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  Std:    {np.std(latencies):.2f} ms")
    print(f"  Min:    {np.min(latencies):.2f} ms")
    print(f"  Max:    {np.max(latencies):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")


def export_results(results: Dict[str, Any], output_path: str):
    """Export results to JSON"""
    
    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval system")
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to evaluation questions JSON"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="rag_hybrid_bge_m3",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:7333",
        help="Qdrant URL"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hybrid_evaluation_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Disable Ollama BGE-M3"
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable cross-encoder reranking"
    )
    
    args = parser.parse_args()
    
    # Load questions
    logger.info(f"Loading questions from {args.questions}...")
    questions = load_evaluation_questions(args.questions)
    logger.info(f"✓ Loaded {len(questions)} questions")
    
    # Initialize retriever
    logger.info("Initializing hybrid retriever...")
    retriever = HybridRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        use_ollama=not args.no_ollama,
        use_reranker=not args.no_reranker,
    )
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(retriever)
    
    # Run evaluation
    logger.info("\nStarting evaluation...")
    start_time = time.time()
    
    results = evaluator.evaluate_all(questions, top_k=args.top_k)
    
    eval_time = time.time() - start_time
    logger.info(f"\n✓ Evaluation completed in {eval_time:.2f}s")
    
    # Print analysis
    print_summary(results)
    analyze_failure_cases(results, top_n=5)
    compare_scores(results)
    analyze_latency(results)
    
    # Export results
    export_results(results, args.output)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()