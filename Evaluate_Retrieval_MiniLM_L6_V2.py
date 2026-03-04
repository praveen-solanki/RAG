"""
RETRIEVAL EVALUATION SYSTEM - FINAL VERSION
Fixed numpy type serialization for JSON output
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ================= CONFIG =================

EVALUATION_QUESTIONS_PATH = "/home/olj3kor/praveen/RAG_work/evaluation_questions.json"
COLLECTION = "rag_database_384_10"
QDRANT_URL = "http://localhost:7333"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# K values for evaluation
K_VALUES = [1, 3, 5, 10]

# Output paths
RESULTS_OUTPUT = "/home/olj3kor/praveen/RAG_work/retrieval_evaluation_results.json"
DETAILED_OUTPUT = "/home/olj3kor/praveen/RAG_work/retrieval_detailed_results.json"

# =========================================


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class RetrievalEvaluator:
    """Evaluate retrieval performance with comprehensive metrics"""
    
    def __init__(self, qdrant_url: str, collection_name: str, embedding_model: str):
        print("Initializing Retrieval Evaluator...")
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        print(f"  → Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        print("  ✓ Model loaded")
        
        # Storage for results
        self.query_results = []
        self.latencies = []
    
    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List, float]:
        """Retrieve top-k documents for a query"""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search in Qdrant - compatible with multiple API versions
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=top_k,
            ).points
        except AttributeError:
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                )
            except AttributeError:
                from qdrant_client.http import models
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                )
        
        latency = time.time() - start_time
        return results, latency
    
    def calculate_precision_at_k(self, retrieved_docs: List[str], 
                                 ground_truth: str, k: int) -> float:
        """Precision@k: Proportion of relevant documents in top-k"""
        if k > len(retrieved_docs):
            k = len(retrieved_docs)
        
        top_k_docs = retrieved_docs[:k]
        return 1.0 if ground_truth in top_k_docs else 0.0
    
    def calculate_recall_at_k(self, retrieved_docs: List[str], 
                             ground_truth: str, k: int) -> float:
        """Recall@k: Whether ground truth is in top-k"""
        return self.calculate_precision_at_k(retrieved_docs, ground_truth, k)
    
    def calculate_mrr(self, retrieved_docs: List[str], ground_truth: str) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant document"""
        try:
            rank = retrieved_docs.index(ground_truth) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    def calculate_average_precision(self, retrieved_docs: List[str], 
                                   ground_truth: str) -> float:
        """Average Precision for single ground truth"""
        try:
            rank = retrieved_docs.index(ground_truth) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    def calculate_ndcg_at_k(self, retrieved_docs: List[str], 
                           ground_truth: str, k: int) -> float:
        """Normalized Discounted Cumulative Gain@k"""
        if k > len(retrieved_docs):
            k = len(retrieved_docs)
        
        # DCG calculation
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], start=1):
            if doc == ground_truth:
                dcg += 1.0 / np.log2(i + 1)
                break
        
        # IDCG - ideal DCG
        idcg = 1.0 / np.log2(2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def extract_filename(self, source_path: str) -> str:
        """Extract just the filename from full path"""
        return Path(source_path).name
    
    def evaluate_single_query(self, question: Dict[str, Any], 
                             max_k: int = 10) -> Dict[str, Any]:
        """Evaluate a single question"""
        query_text = question["question"]
        ground_truth_doc = question["source_document"]
        
        # Retrieve documents
        results, latency = self.retrieve(query_text, top_k=max_k)
        
        # Extract filenames from retrieved results
        retrieved_filenames = []
        for point in results:
            try:
                if hasattr(point, 'payload'):
                    payload = point.payload
                else:
                    payload = point.get('payload', {})
                
                source_path = payload.get('source_path') or payload.get('filename') or payload.get('file_path')
                if source_path:
                    retrieved_filenames.append(self.extract_filename(source_path))
            except Exception as e:
                continue
        
        # Calculate metrics for different k values
        metrics = {
            "question_id": question["id"],
            "question": query_text,
            "ground_truth": ground_truth_doc,
            "retrieved_docs": retrieved_filenames[:max_k],
            "latency_seconds": float(latency),  # Convert to native float
            "metrics": {}
        }
        
        # Precision and Recall at different k
        for k in K_VALUES:
            if k <= max_k:
                metrics["metrics"][f"precision@{k}"] = float(self.calculate_precision_at_k(
                    retrieved_filenames, ground_truth_doc, k
                ))
                metrics["metrics"][f"recall@{k}"] = float(self.calculate_recall_at_k(
                    retrieved_filenames, ground_truth_doc, k
                ))
                metrics["metrics"][f"ndcg@{k}"] = float(self.calculate_ndcg_at_k(
                    retrieved_filenames, ground_truth_doc, k
                ))
        
        # MRR and MAP
        metrics["metrics"]["mrr"] = float(self.calculate_mrr(retrieved_filenames, ground_truth_doc))
        metrics["metrics"]["average_precision"] = float(self.calculate_average_precision(
            retrieved_filenames, ground_truth_doc
        ))
        
        # Check if ground truth found - convert to native bool
        metrics["metrics"]["found"] = bool(ground_truth_doc in retrieved_filenames)
        
        return metrics
    
    def evaluate_all(self, questions: List[Dict[str, Any]], 
                    max_k: int = 10) -> Dict[str, Any]:
        """Evaluate all questions and compute aggregate statistics"""
        print("\n" + "="*80)
        print("STARTING RETRIEVAL EVALUATION")
        print("="*80)
        print(f"Total questions: {len(questions)}")
        print(f"Max k: {max_k}")
        print(f"Evaluating k values: {K_VALUES}")
        print("="*80 + "\n")
        
        all_results = []
        metric_sums = defaultdict(list)
        
        for idx, question in enumerate(questions, start=1):
            if idx % 10 == 0:
                print(f"Progress: {idx}/{len(questions)} questions evaluated...")
            
            result = self.evaluate_single_query(question, max_k=max_k)
            all_results.append(result)
            
            for metric_name, value in result["metrics"].items():
                metric_sums[metric_name].append(value)
            
            self.latencies.append(result["latency_seconds"])
        
        # Calculate aggregate statistics - convert all numpy types
        aggregate_metrics = {}
        
        for metric_name, values in metric_sums.items():
            aggregate_metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
        
        # Latency statistics
        latency_stats = {
            "mean_ms": float(np.mean(self.latencies) * 1000),
            "std_ms": float(np.std(self.latencies) * 1000),
            "min_ms": float(np.min(self.latencies) * 1000),
            "max_ms": float(np.max(self.latencies) * 1000),
            "median_ms": float(np.median(self.latencies) * 1000),
            "p95_ms": float(np.percentile(self.latencies, 95) * 1000),
            "p99_ms": float(np.percentile(self.latencies, 99) * 1000),
        }
        
        # Quality vs Latency analysis
        quality_latency = self.analyze_quality_latency_tradeoff(all_results)
        
        # Document-level analysis
        doc_analysis = self.analyze_by_document(all_results, questions)
        
        # Question type analysis
        type_analysis = self.analyze_by_question_type(all_results, questions)
        
        return {
            "summary": {
                "total_questions": len(questions),
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "max_k": max_k,
                "k_values": K_VALUES,
            },
            "aggregate_metrics": aggregate_metrics,
            "latency_statistics": latency_stats,
            "quality_vs_latency": quality_latency,
            "by_document": doc_analysis,
            "by_question_type": type_analysis,
            "detailed_results": all_results,
        }
    
    def analyze_quality_latency_tradeoff(self, results: List[Dict]) -> Dict:
        """Analyze relationship between retrieval quality and latency"""
        latencies = [r["latency_seconds"] * 1000 for r in results]
        recall_at_5 = [r["metrics"]["recall@5"] for r in results]
        
        p25 = np.percentile(latencies, 25)
        p50 = np.percentile(latencies, 50)
        p75 = np.percentile(latencies, 75)
        
        bins = {
            "fast_queries_0-25p": [],
            "medium_queries_25-50p": [],
            "slow_queries_50-75p": [],
            "very_slow_queries_75-100p": [],
        }
        
        for lat, recall in zip(latencies, recall_at_5):
            if lat <= p25:
                bins["fast_queries_0-25p"].append(recall)
            elif lat <= p50:
                bins["medium_queries_25-50p"].append(recall)
            elif lat <= p75:
                bins["slow_queries_50-75p"].append(recall)
            else:
                bins["very_slow_queries_75-100p"].append(recall)
        
        return {
            bin_name: {
                "count": len(recalls),
                "mean_recall@5": float(np.mean(recalls)) if recalls else 0.0,
            }
            for bin_name, recalls in bins.items()
        }
    
    def analyze_by_document(self, results: List[Dict], 
                           questions: List[Dict]) -> Dict:
        """Analyze performance grouped by source document"""
        doc_metrics = defaultdict(lambda: {
            "questions": 0,
            "recall@5": [],
            "mrr": [],
            "latency_ms": [],
        })
        
        for result in results:
            doc = result["ground_truth"]
            doc_metrics[doc]["questions"] += 1
            doc_metrics[doc]["recall@5"].append(result["metrics"]["recall@5"])
            doc_metrics[doc]["mrr"].append(result["metrics"]["mrr"])
            doc_metrics[doc]["latency_ms"].append(result["latency_seconds"] * 1000)
        
        doc_summary = {}
        for doc, metrics in doc_metrics.items():
            doc_summary[doc] = {
                "num_questions": metrics["questions"],
                "mean_recall@5": float(np.mean(metrics["recall@5"])),
                "mean_mrr": float(np.mean(metrics["mrr"])),
                "mean_latency_ms": float(np.mean(metrics["latency_ms"])),
            }
        
        return doc_summary
    
    def analyze_by_question_type(self, results: List[Dict], 
                                 questions: List[Dict]) -> Dict:
        """Analyze performance by question type"""
        id_to_type = {q["id"]: q.get("question_type", "unknown") for q in questions}
        
        type_metrics = defaultdict(lambda: {
            "questions": 0,
            "recall@5": [],
            "mrr": [],
        })
        
        for result in results:
            qtype = id_to_type.get(result["question_id"], "unknown")
            type_metrics[qtype]["questions"] += 1
            type_metrics[qtype]["recall@5"].append(result["metrics"]["recall@5"])
            type_metrics[qtype]["mrr"].append(result["metrics"]["mrr"])
        
        type_summary = {}
        for qtype, metrics in type_metrics.items():
            type_summary[qtype] = {
                "num_questions": metrics["questions"],
                "mean_recall@5": float(np.mean(metrics["recall@5"])),
                "mean_mrr": float(np.mean(metrics["mrr"])),
            }
        
        return type_summary


def print_evaluation_summary(results: Dict):
    """Print human-readable summary of evaluation results"""
    print("\n" + "="*80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*80)
    
    summary = results["summary"]
    print(f"\n📊 Dataset: {summary['total_questions']} questions")
    print(f"📅 Date: {summary['evaluation_date']}")
    print(f"🎯 Max K: {summary['max_k']}")
    
    print("\n" + "-"*80)
    print("AGGREGATE METRICS")
    print("-"*80)
    
    agg = results["aggregate_metrics"]
    
    print("\n📍 PRECISION & RECALL:")
    for k in K_VALUES:
        if f"precision@{k}" in agg:
            p = agg[f"precision@{k}"]["mean"]
            r = agg[f"recall@{k}"]["mean"]
            print(f"  K={k:2d} → Precision: {p:.3f} | Recall: {r:.3f}")
    
    print(f"\n🎯 MEAN RECIPROCAL RANK (MRR): {agg['mrr']['mean']:.3f}")
    print(f"   (Higher is better, max=1.0)")
    
    print(f"\n📈 NORMALIZED DCG:")
    for k in K_VALUES:
        if f"ndcg@{k}" in agg:
            ndcg = agg[f"ndcg@{k}"]["mean"]
            print(f"  NDCG@{k}: {ndcg:.3f}")
    
    if "found" in agg:
        success_rate = agg["found"]["mean"]
        print(f"\n✅ SUCCESS RATE: {success_rate:.1%}")
        print(f"   ({int(success_rate * summary['total_questions'])}/{summary['total_questions']} "
              f"queries found ground truth in top-{summary['max_k']})")
    
    print("\n" + "-"*80)
    print("LATENCY STATISTICS")
    print("-"*80)
    lat = results["latency_statistics"]
    print(f"  Mean:   {lat['mean_ms']:.2f} ms")
    print(f"  Median: {lat['median_ms']:.2f} ms")
    print(f"  Std:    {lat['std_ms']:.2f} ms")
    print(f"  Min:    {lat['min_ms']:.2f} ms")
    print(f"  Max:    {lat['max_ms']:.2f} ms")
    print(f"  P95:    {lat['p95_ms']:.2f} ms")
    print(f"  P99:    {lat['p99_ms']:.2f} ms")
    
    print("\n" + "-"*80)
    print("QUALITY vs LATENCY TRADE-OFF")
    print("-"*80)
    for bin_name, stats in results["quality_vs_latency"].items():
        print(f"  {bin_name}: {stats['count']} queries, "
              f"Recall@5 = {stats['mean_recall@5']:.3f}")
    
    print("\n" + "-"*80)
    print("PERFORMANCE BY QUESTION TYPE")
    print("-"*80)
    for qtype, stats in results["by_question_type"].items():
        print(f"  {qtype:12s}: {stats['num_questions']:3d} questions, "
              f"Recall@5={stats['mean_recall@5']:.3f}, MRR={stats['mean_mrr']:.3f}")
    
    print("\n" + "="*80)


def main():
    """Main evaluation pipeline"""
    
    # Load evaluation questions
    print("Loading evaluation questions...")
    with open(EVALUATION_QUESTIONS_PATH, 'r') as f:
        data = json.load(f)
    
    questions = data["questions"]
    print(f"✓ Loaded {len(questions)} questions from {data['dataset_info']['total_documents']} documents")
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION,
        embedding_model=EMBEDDING_MODEL,
    )
    
    # Run evaluation
    results = evaluator.evaluate_all(questions, max_k=max(K_VALUES))
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results with proper type conversion
    print(f"\n💾 Saving results to:")
    
    # Summary results (without detailed per-query results)
    summary_results = {k: v for k, v in results.items() if k != "detailed_results"}
    summary_results = convert_numpy_types(summary_results)  # Convert numpy types
    
    with open(RESULTS_OUTPUT, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"  → {RESULTS_OUTPUT}")
    
    # Detailed results (including per-query)
    results = convert_numpy_types(results)  # Convert numpy types
    with open(DETAILED_OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  → {DETAILED_OUTPUT}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()