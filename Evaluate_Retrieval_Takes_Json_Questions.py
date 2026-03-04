#!/usr/bin/env python3
"""
COMPLETE RETRIEVAL & EVALUATION PIPELINE
=========================================
End-to-end evaluation for your AUTOSAR documentation with 100 questions
"""

"""
python Evaluate_Retrieval_Takes_Json_Questions.py \
  --questions evaluation_questions.json \
  --resume progress_75.json
OR
python Evaluate_Retrieval_Takes_Json_Questions.py --questions evaluation_questions.json
"""
import sys
from io import StringIO
from difflib import SequenceMatcher
import unicodedata
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import logging

import numpy as np
from Evaluate_Retrieval_With_Reranker_Template import HybridRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EVALUATION_MODE = "chunk"  # Options: "document" or "chunk"


def normalize(text: str) -> str:
    """Normalize text for comparison"""
    return (
        unicodedata.normalize("NFKC", text)
        .replace("\n", " ")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
    )


def fuzzy_match(snippet: str, text: str, threshold: float = 0.8) -> bool:
    """Fuzzy match a snippet against text using sliding window"""
    snippet_clean = normalize(snippet.lower())
    text_clean = normalize(text.lower())
    # Check exact first
    if snippet_clean in text_clean:
        return True
    # Fall back to sliding window fuzzy match
    window_size = len(snippet_clean)
    if window_size > len(text_clean):
        # Snippet longer than text — compare full text directly
        ratio = SequenceMatcher(None, snippet_clean, text_clean).ratio()
        return ratio >= threshold
    step = max(1, window_size // 4)
    for i in range(0, len(text_clean) - window_size + 1, step):
        window = text_clean[i:min(i + window_size + 20, len(text_clean))]
        ratio = SequenceMatcher(None, snippet_clean, window).ratio()
        if ratio >= threshold:
            return True
    return False


class ComprehensiveEvaluator:
    """Complete evaluation system for 100-question dataset"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def evaluate_single_question(
        self,
        question_data: Dict[str, Any],
        top_k: int = 10,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a single question with detailed metrics"""
        
        query = question_data["question"]
        ground_truth = question_data["source_document"]
        evidence_snippets = question_data.get("evidence_snippets", [])
        question_id = question_data["id"]
        question_type = question_data.get("question_type", "unknown")
        difficulty = question_data.get("difficulty", "unknown")
        
        if verbose:
            logger.info(f"\nQuestion: {query}")
            logger.info(f"Expected: {ground_truth}")
        
        # Perform search
        start_time = time.time()
        try:
            search_results = self.retriever.search(query, top_k=top_k)
            latency = time.time() - start_time
            
            # Extract retrieved documents
            retrieved_docs = []
            retrieved_chunks = []
            
            for result in search_results:
                filename = result.metadata.get('filename', '')
                if filename:
                    retrieved_docs.append(filename)
                    retrieved_chunks.append({
                        'filename': filename,
                        'content': result.content,
                        'score': result.score,
                        'section': result.metadata.get('section_title', ''),
                        'dense_score': result.dense_score,
                        'sparse_score': result.sparse_score,
                        'rerank_score': result.rerank_score,
                    })
            
            # Calculate metrics based on evaluation mode
            if EVALUATION_MODE == "document":
                # Document level — did the right PDF appear in results?
                hit_list = retrieved_docs
                found = ground_truth in hit_list

            elif EVALUATION_MODE == "chunk":
                # Chunk level — did any retrieved chunk contain the evidence snippet?
                if not evidence_snippets:
                    # No snippets available — fall back to document level with a warning
                    logger.warning(f"No evidence_snippets for {question_id} — falling back to document level")
                    hit_list = retrieved_docs
                    found = ground_truth in hit_list
                else:
                    # Build a hit list based on which chunks contain any evidence snippet
                    hit_list = []
                    for chunk in retrieved_chunks:
                        matched = any(
                            fuzzy_match(snippet, chunk["content"])
                            for snippet in evidence_snippets
                        )
                        if matched:
                            hit_list.append(ground_truth)  # treat as a hit at this position
                        else:
                            hit_list.append("__no_match__")
                    found = ground_truth in hit_list
            
            # Rank of first correct hit (1-based)
            try:
                rank = hit_list.index(ground_truth) + 1
                reciprocal_rank = 1.0 / rank
            except ValueError:
                rank = None
                reciprocal_rank = 0.0
            
            # Precision@K: count of relevant docs in top-K / K
            metrics = {
                'found': found,
                'rank': rank,
                'mrr': reciprocal_rank,
            }
            for k in [1, 3, 5, 10]:
                relevant_count = sum(1 for doc in hit_list[:k] if doc == ground_truth)
                metrics[f'precision@{k}'] = relevant_count / k
                metrics[f'recall@{k}'] = 1.0 if ground_truth in hit_list[:k] else 0.0

            # Calculate NDCG@K
            for k in [1, 3, 5, 10]:
                dcg = 0.0
                num_relevant_in_list = sum(1 for doc in hit_list[:k] if doc == ground_truth)
                for i, doc in enumerate(hit_list[:k], start=1):
                    if doc == ground_truth:
                        dcg += 1.0 / np.log2(i + 1)
                # iDCG: best possible arrangement
                idcg = sum(1.0 / np.log2(j + 1) for j in range(1, min(num_relevant_in_list, k) + 1))
                metrics[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0.0
            
            if verbose:
                logger.info(f"Found: {found}, Rank: {rank}, MRR: {reciprocal_rank:.4f}")
                if retrieved_docs:
                    logger.info(f"Top-3: {retrieved_docs[:3]}")
                        
            # Check evidence snippets against retrieved chunks
            evidence_match = []
            for snippet in evidence_snippets:
                match = {"snippet": snippet, "found_in_rank": None, "found_in_filename": None}
                for chunk_rank, chunk in enumerate(retrieved_chunks[:5], start=1):
                    if fuzzy_match(snippet, chunk["content"]):
                        match["found_in_rank"] = chunk_rank
                        match["found_in_filename"] = chunk["filename"]
                        break
                evidence_match.append(match)

            return {
                'question_id': question_id,
                'evaluation_mode': EVALUATION_MODE,
                'question': query,
                'answer': question_data.get('answer', ''),
                'evidence_snippets': evidence_snippets,
                'evidence_match': evidence_match,
                'question_type': question_type,
                'difficulty': difficulty,
                'page_reference': question_data.get('page_reference', ''),
                'ground_truth': ground_truth,
                'retrieved_documents': retrieved_docs,
                'retrieved_chunks': retrieved_chunks[:5],
                'latency_ms': latency * 1000,
                'metrics': metrics,
            }
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            return {
                'question_id': question_id,
                'evaluation_mode': EVALUATION_MODE,
                'question': query,
                'answer': question_data.get('answer', ''),
                'question_type': question_type,
                'difficulty': difficulty,
                'page_reference': question_data.get('page_reference', ''),
                'ground_truth': ground_truth,
                'retrieved_documents': [],
                'retrieved_chunks': [],
                'latency_ms': 0.0,
                'evidence_snippets': evidence_snippets,
                'evidence_match': [],
                'error': str(e),
                'metrics': {
                    'found': 0.0, 'rank': None, 'mrr': 0.0,
                    'precision@1': 0.0, 'precision@3': 0.0,
                    'precision@5': 0.0, 'precision@10': 0.0,
                    'recall@1': 0.0, 'recall@3': 0.0,
                    'recall@5': 0.0, 'recall@10': 0.0,
                    'ndcg@1': 0.0, 'ndcg@3': 0.0,
                    'ndcg@5': 0.0, 'ndcg@10': 0.0,
                }
            }
    
    def evaluate_all(
        self,
        questions: List[Dict[str, Any]],
        top_k: int = 10,
        verbose: bool = False,
        save_progress_every: int = 10,
        resume_file: str = None
    ) -> Dict[str, Any]:

        """Evaluate all questions with progress tracking"""
        
        logger.info("="*80)
        logger.info(f"EVALUATING {len(questions)} QUESTIONS")
        logger.info("="*80)
        
        all_results = []

        # ------------------ RESUME LOGIC ------------------
        if resume_file and Path(resume_file).exists():
            logger.info(f"Resuming from {resume_file}...")

            with open(resume_file, 'r') as f:
                all_results = json.load(f)

            completed_ids = {r['question_id'] for r in all_results}
            original_count = len(questions)

            questions = [q for q in questions if q['id'] not in completed_ids]

            logger.info(f"✓ Already completed: {len(completed_ids)}")
            logger.info(f"→ Remaining: {len(questions)} / {original_count}")
        # ---------------------------------------------------

        start_time = time.time()
        
        for idx, question in enumerate(questions, start=1):
            # Progress indicator
            if idx % 10 == 0 or verbose:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                eta = avg_time * (len(questions) - idx)
                logger.info(f"\nProgress: {idx}/{len(questions)} ({idx/len(questions)*100:.1f}%) - "
                          f"ETA: {eta/60:.1f} min")
            
            # Evaluate question
            result = self.evaluate_single_question(question, top_k, verbose)
            all_results.append(result)
            
            # Save progress periodically
            if save_progress_every and idx % save_progress_every == 0:
                self._save_progress(all_results, f"progress_{idx}.json")
        
        total_time = time.time() - start_time
        logger.info(f"\n✓ Completed {len(questions)} questions in {total_time/60:.2f} minutes")
        
        # Calculate aggregate statistics
        aggregate_results = self._calculate_aggregates(all_results)
        
        return {
            'evaluation_info': {
                'total_questions': len(all_results),
                'evaluation_mode': EVALUATION_MODE,
                'total_time_seconds': total_time,
                'avg_time_per_question': (
                    total_time / len(questions) if len(questions) > 0 else 0.0
                ),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'aggregate_metrics': aggregate_results['aggregate_metrics'],
            'by_question_type': aggregate_results['by_question_type'],
            'by_difficulty': aggregate_results['by_difficulty'],
            'by_document': aggregate_results['by_document'],
            'latency_stats': aggregate_results['latency_stats'],
            'failure_analysis': aggregate_results['failure_analysis'],
            'detailed_results': all_results,
        }
    
    def _calculate_aggregates(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive aggregate statistics"""
        
        # Overall metrics
        metric_values = defaultdict(list)
        latencies = []
        
        for result in results:
            if 'error' not in result:
                for metric_name, value in result['metrics'].items():
                    if value is not None:
                        metric_values[metric_name].append(value)
                latencies.append(result['latency_ms'])
        
        aggregate_metrics = {}
        for metric_name, values in metric_values.items():
            if len(values) == 0:
                continue
            values = np.array(values, dtype=float)
            aggregate_metrics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }

        # By question type
        by_type = defaultdict(lambda: defaultdict(list))
        for result in results:
            if 'error' not in result:
                qtype = result.get('question_type', 'unknown')
                for metric_name, value in result['metrics'].items():
                    if value is not None:
                        by_type[qtype][metric_name].append(value)
        
        by_question_type = {}
        for qtype, metrics in by_type.items():
            by_question_type[qtype] = {
                'count': len(metrics['found']),
                'success_rate': float(np.mean(metrics['found'])),
                'mean_mrr': float(np.mean(metrics['mrr'])),
                'mean_precision@5': float(np.mean(metrics['precision@5'])),
            }
        
        # By difficulty
        by_diff = defaultdict(lambda: defaultdict(list))
        for result in results:
            if 'error' not in result:
                difficulty = result.get('difficulty', 'unknown')
                for metric_name, value in result['metrics'].items():
                    if value is not None:
                        by_diff[difficulty][metric_name].append(value)
        
        by_difficulty = {}
        for difficulty, metrics in by_diff.items():
            by_difficulty[difficulty] = {
                'count': len(metrics['found']),
                'success_rate': float(np.mean(metrics['found'])),
                'mean_mrr': float(np.mean(metrics['mrr'])),
                'mean_precision@5': float(np.mean(metrics['precision@5'])),
            }
        
        # By document
        by_doc = defaultdict(lambda: defaultdict(list))
        for result in results:
            if 'error' not in result:
                doc = result['ground_truth']
                for metric_name, value in result['metrics'].items():
                    if value is not None:
                        by_doc[doc][metric_name].append(value)
        
        by_document = {}
        for doc, metrics in by_doc.items():
            by_document[doc] = {
                'questions': len(metrics['found']),
                'success_rate': float(np.mean(metrics['found'])),
                'mean_mrr': float(np.mean(metrics['mrr'])),
                'mean_precision@5': float(np.mean(metrics['precision@5'])),
            }
        
        # Latency statistics
        if latencies:
            latency_stats = {
                'mean_ms': float(np.mean(latencies)),
                'median_ms': float(np.median(latencies)),
                'std_ms': float(np.std(latencies)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'p99_ms': float(np.percentile(latencies, 99)),
            }
        else:
            latency_stats = {
                'mean_ms': 0.0, 'median_ms': 0.0, 'std_ms': 0.0,
                'min_ms': 0.0, 'max_ms': 0.0, 'p95_ms': 0.0, 'p99_ms': 0.0,
            }
        
        # Failure analysis
        failures = [r for r in results if not r['metrics'].get('found', False)]
        failure_analysis = {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(results) if results else 0.0,
            'failures_by_type': {},
            'failures_by_difficulty': {},
            'sample_failures': []
        }
        
        # Group failures
        type_failures = defaultdict(int)
        diff_failures = defaultdict(int)

        for failure in failures:
            type_failures[failure.get('question_type', 'unknown')] += 1
            diff_failures[failure.get('difficulty', 'unknown')] += 1
                
        for failure in failures[:10]:  # Sample of 10
            failure_analysis['sample_failures'].append({
                'question': failure['question'],
                'expected': failure['ground_truth'],
                'got_top3': failure.get('retrieved_documents', [])[:3],
                'type': failure.get('question_type'),
                'difficulty': failure.get('difficulty'),
            })
        
        failure_analysis['failures_by_type'] = dict(type_failures)
        failure_analysis['failures_by_difficulty'] = dict(diff_failures)
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'by_question_type': by_question_type,
            'by_difficulty': by_difficulty,
            'by_document': by_document,
            'latency_stats': latency_stats,
            'failure_analysis': failure_analysis,
        }
    
    def _save_progress(self, results: List[Dict], filename: str):
        """Save progress to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"  💾 Progress saved to {filename}")
        except Exception as e:
            logger.error(f"  ✗ Could not save progress: {e}")


def print_detailed_summary(results: Dict):
    """Print comprehensive evaluation summary"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*100)
    
    # Overview
    info = results['evaluation_info']
    print(f"\n📊 OVERVIEW:")
    print(f"   Evaluation Mode: {info.get('evaluation_mode', 'document').upper()}")
    print(f"   Total Questions: {info['total_questions']}")
    print(f"   Total Time: {info['total_time_seconds']/60:.2f} minutes")
    print(f"   Avg Time/Question: {info['avg_time_per_question']:.2f} seconds")
    print(f"   Timestamp: {info['timestamp']}")
    
    # Overall metrics
    print("\n" + "-"*100)
    print("OVERALL PERFORMANCE")
    print("-"*100)
    
    agg = results['aggregate_metrics']
    
    if 'found' in agg:
        success = agg['found']['mean']
        print(f"\n✅ SUCCESS RATE: {success:.2%} ({int(success * info['total_questions'])}/{info['total_questions']})")
    
    if 'mrr' in agg:
        print(f"\n🎯 MEAN RECIPROCAL RANK: {agg['mrr']['mean']:.4f}")
    
    print(f"\n📍 PRECISION @ K:")
    for k in [1, 3, 5, 10]:
        if f'precision@{k}' in agg:
            p = agg[f'precision@{k}']
            print(f"   P@{k:2d}: {p['mean']:.4f} ± {p['std']:.4f} (min: {p['min']:.4f}, max: {p['max']:.4f})")
    
    print(f"\n📊 NDCG @ K:")
    for k in [1, 3, 5, 10]:
        if f'ndcg@{k}' in agg:
            ndcg = agg[f'ndcg@{k}']
            print(f"   NDCG@{k:2d}: {ndcg['mean']:.4f}")
    
    # Latency
    print("\n" + "-"*100)
    print("LATENCY STATISTICS")
    print("-"*100)
    
    lat = results['latency_stats']
    print(f"\n   Mean:   {lat['mean_ms']:.2f} ms")
    print(f"   Median: {lat['median_ms']:.2f} ms")
    print(f"   Std:    {lat['std_ms']:.2f} ms")
    print(f"   Min:    {lat['min_ms']:.2f} ms")
    print(f"   Max:    {lat['max_ms']:.2f} ms")
    print(f"   P95:    {lat['p95_ms']:.2f} ms")
    print(f"   P99:    {lat['p99_ms']:.2f} ms")
    
    # By question type
    print("\n" + "-"*100)
    print("PERFORMANCE BY QUESTION TYPE")
    print("-"*100)
    
    print(f"\n{'Type':<15} {'Count':>8} {'Success':>10} {'MRR':>10} {'P@5':>10}")
    print("-"*100)
    
    for qtype, stats in sorted(results['by_question_type'].items()):
        print(f"{qtype:<15} {stats['count']:>8} {stats['success_rate']:>9.2%} "
              f"{stats['mean_mrr']:>10.4f} {stats['mean_precision@5']:>10.4f}")
    
    # By difficulty
    print("\n" + "-"*100)
    print("PERFORMANCE BY DIFFICULTY")
    print("-"*100)
    
    print(f"\n{'Difficulty':<15} {'Count':>8} {'Success':>10} {'MRR':>10} {'P@5':>10}")
    print("-"*100)
    
    for diff, stats in sorted(results['by_difficulty'].items()):
        print(f"{diff:<15} {stats['count']:>8} {stats['success_rate']:>9.2%} "
              f"{stats['mean_mrr']:>10.4f} {stats['mean_precision@5']:>10.4f}")
    
    # By document
    print("\n" + "-"*100)
    print("PERFORMANCE BY DOCUMENT")
    print("-"*100)
    
    print(f"\n{'Document':<50} {'Qs':>5} {'Success':>10} {'MRR':>10}")
    print("-"*100)
    
    for doc, stats in sorted(results['by_document'].items(), 
                            key=lambda x: x[1]['success_rate'], 
                            reverse=True):
        doc_short = doc[:47] + "..." if len(doc) > 50 else doc
        print(f"{doc_short:<50} {stats['questions']:>5} {stats['success_rate']:>9.2%} "
              f"{stats['mean_mrr']:>10.4f}")
    
    # Failure analysis
    print("\n" + "-"*100)
    print("FAILURE ANALYSIS")
    print("-"*100)
    
    fail = results['failure_analysis']
    print(f"\n   Total Failures: {fail['total_failures']} ({fail['failure_rate']:.2%})")
    
    if fail['failures_by_type']:
        print(f"\n   Failures by Type:")
        for qtype, count in sorted(fail['failures_by_type'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"      {qtype}: {count}")
    
    if fail['failures_by_difficulty']:
        print(f"\n   Failures by Difficulty:")
        for diff, count in sorted(fail['failures_by_difficulty'].items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"      {diff}: {count}")
    
    if fail['sample_failures']:
        print(f"\n   Sample Failed Queries:")
        for i, failure in enumerate(fail['sample_failures'][:5], 1):
            print(f"\n   {i}. Question: {failure['question']}")
            print(f"      Expected: {failure['expected']}")
            print(f"      Got Top-3: {', '.join(failure['got_top3'][:3])}")
            print(f"      Type: {failure['type']}, Difficulty: {failure['difficulty']}")
    
    print("\n" + "="*100)


def save_results(results: Dict, output_path: str):
    """Save results with pretty formatting"""
    
    def convert_types(obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete evaluation for 100-question AUTOSAR dataset"
    )
    parser.add_argument(
        '--questions',
        type=str,
        required=True,
        help='Path to evaluation_questions.json'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to progress JSON file to resume from'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='rag_hybrid_bge_m3',
        help='Qdrant collection name'
    )
    parser.add_argument(
        '--qdrant-url',
        type=str,
        default='http://localhost:7333',
        help='Qdrant URL'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to retrieve per question'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='complete_evaluation_results.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    parser.add_argument(
        '--no-ollama',
        action='store_true',
        help='Disable Ollama BGE-M3'
    )
    parser.add_argument(
        '--no-reranker',
        action='store_true',
        help='Disable cross-encoder reranking'
    )
    parser.add_argument(
        '--save-progress',
        type=int,
        default=25,
        help='Save progress every N questions (0 to disable)'
    )
    parser.add_argument(
        '--eval-mode',
        type=str,
        choices=['document', 'chunk'],
        default=None,
        help='Evaluation mode: "document" or "chunk" (overrides EVALUATION_MODE constant)'
    )

    args = parser.parse_args()

    # Override module-level EVALUATION_MODE if CLI argument provided
    global EVALUATION_MODE
    if args.eval_mode is not None:
        EVALUATION_MODE = args.eval_mode

    # Load questions
    logger.info(f"Loading questions from {args.questions}...")
    with open(args.questions, 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    dataset_info = data.get('dataset_info', {})
    
    logger.info(f"✓ Loaded {len(questions)} questions")
    logger.info(f"   Documents: {dataset_info.get('total_documents', 'N/A')}")
    logger.info(f"   Generated: {dataset_info.get('generation_date', 'N/A')}")
    
    # Initialize retriever
    logger.info("\nInitializing hybrid retriever...")
    retriever = HybridRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        use_ollama=not args.no_ollama,
        use_reranker=not args.no_reranker,
    )
    logger.info("✓ Retriever initialized")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(retriever)
    
    # Run evaluation
    logger.info("\nStarting evaluation...\n")
    results = evaluator.evaluate_all(
        questions,
        top_k=args.top_k,
        verbose=args.verbose,
        save_progress_every=args.save_progress,
        resume_file=args.resume
    )

    # Print summary
    print_detailed_summary(results)
    
    # Save results
    save_results(results, args.output)
    
    # Generate summary file
    summary_path = args.output.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        old_stdout = sys.stdout
        try:
            sys.stdout = StringIO()
            print_detailed_summary(results)
            summary_text = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        f.write(summary_text)
    
    logger.info(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE!")
    print("="*100)
    print(f"\nResults saved to:")
    print(f"  • Detailed JSON: {args.output}")
    print(f"  • Text Summary: {summary_path}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()