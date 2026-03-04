"""
SYSTEM COMPARISON SCRIPT
========================
Compare original vs improved RAG system performance
"""
"""
python Evaluate_Compare_Systems.py \
    --questions evaluation_questions.json \
    --original-collection rag_database_384_10 \
    --improved-collection rag_hybrid_bge_m3 \
    --output comparison_autosar.json
"""
import json
import time
import argparse
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
from Evaluate_Retrieval_With_Reranker import HybridRetriever, AdvancedEvaluator

# For comparison with original system
try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
except ImportError:
    print("Please install: pip install sentence-transformers qdrant-client")
    exit(1)


class OriginalRetriever:
    """Wrapper for original simple retrieval"""
    
    def __init__(self, qdrant_url: str, collection: str, model_name: str):
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.model = SentenceTransformer(model_name)
    
    def search(self, query: str, top_k: int = 10):
        """Simple dense-only search"""
        # Encode query
        query_vector = self.model.encode(query, convert_to_numpy=True)
        
        # Search
        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector.tolist(),
                limit=top_k,
            ).points
        except AttributeError:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
            )
        
        return results


def evaluate_original(
    retriever: OriginalRetriever,
    questions: List[Dict],
    top_k: int = 10
) -> Dict[str, Any]:
    """Evaluate original system"""
    
    results = []
    latencies = []
    
    for question in questions:
        query = question["question"]
        ground_truth = question["source_document"]
        
        # Search
        start = time.time()
        search_results = retriever.search(query, top_k=top_k)
        latency = time.time() - start
        latencies.append(latency * 1000)
        
        # Extract filenames
        retrieved = []
        for point in search_results:
            try:
                if hasattr(point, 'payload'):
                    filename = point.payload.get('filename') or point.payload.get('source_path', '')
                else:
                    filename = point.get('payload', {}).get('filename', '')
                
                if filename:
                    from pathlib import Path
                    retrieved.append(Path(filename).name)
            except:
                pass
        
        # Calculate metrics
        found = ground_truth in retrieved
        mrr = 1.0 / (retrieved.index(ground_truth) + 1) if found else 0.0
        
        results.append({
            "found": found,
            "mrr": mrr,
            "precision@5": 1.0 if ground_truth in retrieved[:5] else 0.0,
            "latency_ms": latency * 1000,
        })
    
    # Aggregate
    return {
        "found_rate": np.mean([r["found"] for r in results]),
        "mrr": np.mean([r["mrr"] for r in results]),
        "precision@5": np.mean([r["precision@5"] for r in results]),
        "latency_ms": {
            "mean": np.mean(latencies),
            "p50": np.median(latencies),
            "p95": np.percentile(latencies, 95),
        }
    }


def evaluate_improved(
    retriever: HybridRetriever,
    questions: List[Dict],
    top_k: int = 10,
    use_reranking: bool = True,
) -> Dict[str, Any]:
    """Evaluate improved system"""
    
    evaluator = AdvancedEvaluator(retriever)
    
    # Temporarily override reranking setting
    original_reranker = retriever.reranker
    if not use_reranking:
        retriever.reranker = None
    
    results = evaluator.evaluate_all(questions, top_k=top_k)
    
    # Restore reranker
    retriever.reranker = original_reranker
    
    # Extract metrics
    metrics = results["aggregate_metrics"]
    
    return {
        "found_rate": metrics["found"]["mean"],
        "mrr": metrics["mrr"]["mean"],
        "precision@5": metrics["precision@5"]["mean"],
        "latency_ms": {
            "mean": np.mean([r["latency_ms"] for r in results["detailed_results"]]),
            "p50": np.median([r["latency_ms"] for r in results["detailed_results"]]),
            "p95": np.percentile([r["latency_ms"] for r in results["detailed_results"]], 95),
        }
    }


def print_comparison(comparisons: Dict[str, Dict]):
    """Print side-by-side comparison"""
    
    print("\n" + "="*100)
    print("SYSTEM COMPARISON RESULTS")
    print("="*100)
    
    # Header
    print(f"\n{'Metric':<30} {'Original':<20} {'Improved':<20} {'Gain':<20}")
    print("-"*100)
    
    # Compare each metric
    for system_name, metrics in comparisons.items():
        if system_name == "original":
            continue
        
        orig = comparisons["original"]
        imp = metrics
        
        # Found rate
        orig_found = orig["found_rate"]
        imp_found = imp["found_rate"]
        gain_found = ((imp_found - orig_found) / orig_found * 100) if orig_found > 0 else 0
        
        print(f"{'Success Rate':<30} {orig_found:.3f} ({orig_found*100:.1f}%)"
              f"{'':>3} {imp_found:.3f} ({imp_found*100:.1f}%)"
              f"{'':>3} {gain_found:+.1f}%")
        
        # MRR
        orig_mrr = orig["mrr"]
        imp_mrr = imp["mrr"]
        gain_mrr = ((imp_mrr - orig_mrr) / orig_mrr * 100) if orig_mrr > 0 else 0
        
        print(f"{'MRR':<30} {orig_mrr:.3f}"
              f"{'':>15} {imp_mrr:.3f}"
              f"{'':>15} {gain_mrr:+.1f}%")
        
        # Precision@5
        orig_p5 = orig["precision@5"]
        imp_p5 = imp["precision@5"]
        gain_p5 = ((imp_p5 - orig_p5) / orig_p5 * 100) if orig_p5 > 0 else 0
        
        print(f"{'Precision@5':<30} {orig_p5:.3f}"
              f"{'':>15} {imp_p5:.3f}"
              f"{'':>15} {gain_p5:+.1f}%")
        
        # Latency
        orig_lat = orig["latency_ms"]["mean"]
        imp_lat = imp["latency_ms"]["mean"]
        lat_diff = imp_lat - orig_lat
        
        print(f"{'Latency (mean)':<30} {orig_lat:.1f} ms"
              f"{'':>12} {imp_lat:.1f} ms"
              f"{'':>12} {lat_diff:+.1f} ms")
        
        print(f"{'Latency (p95)':<30} {orig['latency_ms']['p95']:.1f} ms"
              f"{'':>12} {imp['latency_ms']['p95']:.1f} ms")
        
        print("-"*100)
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    imp = comparisons.get("hybrid_with_rerank") or comparisons.get("hybrid_no_rerank")
    orig = comparisons["original"]
    
    if imp:
        # Calculate overall improvement
        quality_gain = (
            ((imp["found_rate"] - orig["found_rate"]) / orig["found_rate"]) +
            ((imp["mrr"] - orig["mrr"]) / orig["mrr"]) +
            ((imp["precision@5"] - orig["precision@5"]) / orig["precision@5"])
        ) / 3 * 100
        
        latency_cost = ((imp["latency_ms"]["mean"] - orig["latency_ms"]["mean"]) / 
                       orig["latency_ms"]["mean"] * 100)
        
        print(f"\n✓ Average Quality Improvement: {quality_gain:+.1f}%")
        print(f"✓ Latency Cost: {latency_cost:+.1f}%")
        print(f"\n{'Quality/Latency Ratio:':<30} {quality_gain/abs(latency_cost) if latency_cost != 0 else float('inf'):.2f}x")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="Compare RAG system versions")
    parser.add_argument("--questions", required=True, help="Evaluation questions JSON")
    parser.add_argument("--original-collection", default="rag_database_384_10")
    parser.add_argument("--improved-collection", default="rag_hybrid_bge_m3")
    parser.add_argument("--qdrant-url", default="http://localhost:7333")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="comparison_results.json")
    
    args = parser.parse_args()
    
    # Load questions
    print(f"Loading questions from {args.questions}...")
    with open(args.questions, 'r') as f:
        data = json.load(f)
    questions = data["questions"]
    print(f"✓ Loaded {len(questions)} questions\n")
    
    comparisons = {}
    
    # Test 1: Original system
    print("="*100)
    print("TEST 1: Original System (Dense-only, no reranking)")
    print("="*100)
    
    try:
        original = OriginalRetriever(
            qdrant_url=args.qdrant_url,
            collection=args.original_collection,
            model_name="all-MiniLM-L6-v2"
        )
        
        print("Evaluating original system...")
        comparisons["original"] = evaluate_original(original, questions, args.top_k)
        print("✓ Original system evaluated")
        
    except Exception as e:
        print(f"✗ Could not evaluate original system: {e}")
        print("  Make sure the original collection exists")
        return
    
    # Test 2: Improved system without reranking
    print("\n" + "="*100)
    print("TEST 2: Improved System (Hybrid, no reranking)")
    print("="*100)
    
    try:
        improved = HybridRetriever(
            qdrant_url=args.qdrant_url,
            collection_name=args.improved_collection,
            use_ollama=True,
            use_reranker=False,
        )
        
        print("Evaluating hybrid system without reranking...")
        comparisons["hybrid_no_rerank"] = evaluate_improved(
            improved, questions, args.top_k, use_reranking=False
        )
        print("✓ Hybrid system (no rerank) evaluated")
        
    except Exception as e:
        print(f"✗ Could not evaluate hybrid system: {e}")
    
    # Test 3: Improved system with reranking
    print("\n" + "="*100)
    print("TEST 3: Improved System (Hybrid + Reranking)")
    print("="*100)
    
    try:
        improved = HybridRetriever(
            qdrant_url=args.qdrant_url,
            collection_name=args.improved_collection,
            use_ollama=True,
            use_reranker=True,
        )
        
        print("Evaluating hybrid system with reranking...")
        comparisons["hybrid_with_rerank"] = evaluate_improved(
            improved, questions, args.top_k, use_reranking=True
        )
        print("✓ Hybrid system (with rerank) evaluated")
        
    except Exception as e:
        print(f"✗ Could not evaluate with reranking: {e}")
    
    # Print comparison
    if len(comparisons) > 1:
        print_comparison(comparisons)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(comparisons, f, indent=2)
        
        print(f"\n✓ Detailed results saved to {args.output}")
    else:
        print("\n✗ Not enough systems evaluated for comparison")


if __name__ == "__main__":
    main()