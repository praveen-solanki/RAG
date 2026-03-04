"""
ADVANCED RAG RETRIEVAL SYSTEM
==============================
Features:
- Hybrid search (Dense + Sparse BM25)
- Metadata filtering
- Cross-encoder reranking (BGE-M3 via Ollama)
- Query expansion
- Result caching
- Comprehensive evaluation metrics
"""

"""
This only a sample file, Retrival for only some example queries. 
"""
import os
import json
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import logging

import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SparseVector,
    Prefetch,
    Query,
    FusionQuery,
)
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        nltk.download('punkt', quiet=True)

# ================= CONFIG =================

COLLECTION = "rag_hybrid_bge_m3"
QDRANT_URL = "http://localhost:7333"

# Embedding configuration
USE_OLLAMA_BGE_M3 = True
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "bge-m3"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reranker configuration
USE_CROSS_ENCODER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast local reranker
USE_OLLAMA_RERANKER = True  # Use Ollama for reranking if available

# Retrieval parameters
DENSE_TOP_K = 50  # Dense search results
SPARSE_TOP_K = 50  # Sparse search results
HYBRID_TOP_K = 30  # After fusion
FINAL_TOP_K = 10  # After reranking

# Fusion weights
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5

# Query expansion
ENABLE_QUERY_EXPANSION = True
EXPANSION_SYNONYMS = {
    "ai": ["artificial intelligence", "machine learning", "deep learning"],
    "ml": ["machine learning", "ai", "artificial intelligence"],
    "llm": ["large language model", "language model", "transformer"],
}

# Caching
ENABLE_CACHE = True
CACHE_SIZE = 1000

# =========================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    id: str
    content: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QueryCache:
    """LRU cache for query results"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def _hash_query(self, query: str, filters: Optional[Dict] = None) -> str:
        """Generate cache key"""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        return hashlib.md5(f"{query}:{filter_str}".encode()).hexdigest()
    
    def get(self, query: str, filters: Optional[Dict] = None) -> Optional[List[SearchResult]]:
        """Get cached results"""
        key = self._hash_query(query, filters)
        
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        return None
    
    def put(self, query: str, results: List[SearchResult], filters: Optional[Dict] = None):
        """Cache results"""
        key = self._hash_query(query, filters)
        
        # Remove oldest if cache full
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = results
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()


class OllamaBGEM3:
    """BGE-M3 embedder and reranker via Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "bge-m3"):
        self.base_url = base_url
        self.model = model
        self.dimension = 1024
        self.available = self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test Ollama availability"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info(f"✓ Ollama available at {self.base_url}")
                return True
        except Exception:
            logger.warning(f"✗ Ollama not available at {self.base_url}")
        return False
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings"""
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                else:
                    embeddings.append([0.0] * self.dimension)
                    
            except Exception as e:
                logger.warning(f"Encoding error: {e}")
                embeddings.append([0.0] * self.dimension)
        
        return embeddings
    
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Rerank documents using BGE-M3"""
        scores = []
        
        # Encode query once outside the loop
        try:
            query_emb = self.encode([query])[0]
        except Exception as e:
            logger.warning(f"Failed to encode query for reranking: {e}")
            return [0.0] * len(documents)

        for doc in documents:
            try:
                doc_emb = self.encode([doc])[0]
                
                # Cosine similarity
                score = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                scores.append(float(score))
                
            except Exception as e:
                logger.warning(f"Reranking error: {e}")
                scores.append(0.0)
        
        return scores

class BM25Encoder:
    """BM25 query encoder"""
    
    def __init__(
        self,
        vocabulary: Optional[Dict[str, int]] = None,
        token_idf: Optional[Dict[str, float]] = None,
    ):
        self.vocabulary = vocabulary or {}
        self.token_idf = token_idf or {}   # ← ADD: IDF weights from ingestion
    
    def encode_query(self, query: str) -> SparseVector:
        """Encode query to BM25 sparse vector"""
        tokens = [t.lower() for t in word_tokenize(query) if t.isalnum()]
        total = len(tokens)
        
        token_counts = {}
        for token in tokens:
            if token in self.vocabulary:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        indices = []
        values = []
        
        for token, count in token_counts.items():
            tf = count / total if total else 0.0
            idf = self.token_idf.get(token, 1.0)   # ← USE IDF weight
            indices.append(self.vocabulary[token])
            values.append(float(tf * idf))          # ← TF-IDF instead of TF only
        
        return SparseVector(indices=indices, values=values)

class QueryExpander:
    """Expand queries with synonyms and related terms"""
    
    def __init__(self, expansions: Dict[str, List[str]]):
        self.expansions = expansions
    
    def expand(self, query: str) -> str:
        """Expand query with related terms"""
        query_lower = query.lower()
        expanded_terms = [query]
        
        for key, synonyms in self.expansions.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)
        
        return " ".join(expanded_terms)


class MetadataFilterBuilder:
    """Build Qdrant filters from user criteria"""
    
    @staticmethod
    def build_filter(
        file_types: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        folders: Optional[List[str]] = None,
        min_word_count: Optional[int] = None,
        max_word_count: Optional[int] = None,
        section_titles: Optional[List[str]] = None,
        has_tables: Optional[bool] = None,
    ) -> Optional[Filter]:
        """Build complex metadata filter"""
        conditions = []
        
        if file_types:
            conditions.append(
                FieldCondition(key="file_type", match=MatchAny(any=file_types))
            )
        
        if filenames:
            conditions.append(
                FieldCondition(key="filename", match=MatchAny(any=filenames))
            )
        
        if folders:
            conditions.append(
                FieldCondition(key="folder", match=MatchAny(any=folders))
            )
        
        if min_word_count is not None or max_word_count is not None:
            range_filter = {}
            if min_word_count is not None:
                range_filter["gte"] = min_word_count
            if max_word_count is not None:
                range_filter["lte"] = max_word_count
            
            conditions.append(
                FieldCondition(key="word_count", range=Range(**range_filter))
            )
        
        if section_titles:
            conditions.append(
                FieldCondition(key="section_title", match=MatchAny(any=section_titles))
            )
        
        if has_tables is not None:
            conditions.append(
                FieldCondition(key="has_tables", match=MatchValue(value=has_tables))
            )
        
        if not conditions:
            return None
        
        return Filter(must=conditions)


class HybridRetriever:
    """Advanced hybrid retrieval system"""
    
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        use_ollama: bool = True,
        use_reranker: bool = True,
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
        # Initialize embedder
        if use_ollama:
            self.ollama = OllamaBGEM3(OLLAMA_URL, OLLAMA_MODEL)
            if self.ollama.available:
                self.embedder = self.ollama
                self.embedding_dim = 1024
            else:
                logger.warning("Falling back to SentenceTransformer")
                self.embedder = SentenceTransformer(FALLBACK_MODEL)
                self.embedding_dim = 384
        else:
            self.embedder = SentenceTransformer(FALLBACK_MODEL)
            self.embedding_dim = 384
        
        # Initialize reranker
        self.use_reranker = use_reranker
        if use_reranker:
            if USE_OLLAMA_RERANKER and hasattr(self, 'ollama') and self.ollama.available:
                self.reranker = self.ollama
                logger.info("✓ Using Ollama for reranking")
            else:
                try:
                    self.reranker = CrossEncoder(RERANKER_MODEL)
                    logger.info(f"✓ Loaded reranker: {RERANKER_MODEL}")
                except Exception:
                    self.reranker = None
                    logger.warning("✗ No reranker available")
        else:
            self.reranker = None
        
        # Initialize components — load BM25 vocab and IDF from ingestion
        bm25_vocab = {}
        bm25_idf = {}
        bm25_path = "bm25_index.json"
        if os.path.exists(bm25_path):
            with open(bm25_path, "r") as f:
                bm25_data = json.load(f)
            bm25_vocab = bm25_data.get("vocabulary", {})
            bm25_idf = bm25_data.get("token_idf", {})
            logger.info(f"✓ Loaded BM25 index: {len(bm25_vocab)} tokens")
        else:
            logger.warning(f"✗ BM25 index not found at {bm25_path} — sparse search will be empty!")

        self.bm25_encoder = BM25Encoder(vocabulary=bm25_vocab, token_idf=bm25_idf)

        self.query_expander = QueryExpander(EXPANSION_SYNONYMS)
        self.filter_builder = MetadataFilterBuilder()
        self.cache = QueryCache(CACHE_SIZE) if ENABLE_CACHE else None
        
        logger.info(f"✓ Hybrid retriever initialized")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Reranker: {'Enabled' if self.reranker else 'Disabled'}")
    
    def _dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter_: Optional[Filter] = None
    ) -> List[SearchResult]:
        """Dense vector search"""
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                limit=top_k,
                query_filter=filter_,
            ).points
            
            search_results = []
            for point in results:
                search_results.append(SearchResult(
                    id=point.id,
                    content=point.payload.get("content", ""),
                    score=point.score,
                    dense_score=point.score,
                    metadata=point.payload
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []
    
    def _sparse_search(
        self,
        query_sparse: SparseVector,
        top_k: int,
        filter_: Optional[Filter] = None
    ) -> List[SearchResult]:
        """Sparse BM25 search"""
        try:
            # Qdrant sparse search using named vector
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_sparse,
                using="bm25",
                limit=top_k,
                query_filter=filter_,
            ).points
            
            search_results = []
            for point in results:
                search_results.append(SearchResult(
                    id=point.id,
                    content=point.payload.get("content", ""),
                    score=point.score,
                    sparse_score=point.score,
                    metadata=point.payload
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            return []
    
    def _hybrid_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion of dense and sparse results"""
        
        # Build score maps
        dense_scores = {r.id: (rank + 1, r) for rank, r in enumerate(dense_results)}
        sparse_scores = {r.id: (rank + 1, r) for rank, r in enumerate(sparse_results)}
        
        # Combine scores using RRF
        combined_scores = {}
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        for doc_id in all_ids:
            rrf_score = 0.0
            result = None
            
            if doc_id in dense_scores:
                rank, res = dense_scores[doc_id]
                rrf_score += dense_weight / (60 + rank)  # RRF formula
                result = res
            
            if doc_id in sparse_scores:
                rank, res = sparse_scores[doc_id]
                rrf_score += sparse_weight / (60 + rank)
                if result is None:
                    result = res
            
            combined_scores[doc_id] = (rrf_score, result)
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )
        
        # Update scores
        fused_results = []
        for score, result in sorted_results:
            result.score = score
            fused_results.append(result)
        
        return fused_results
    
    def _rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Rerank results using cross-encoder"""
        if not self.reranker or not results:
            return results[:top_k]
        
        try:
            documents = [r.content for r in results]
            
            # Get reranking scores
            if isinstance(self.reranker, OllamaBGEM3):
                scores = self.reranker.rerank(query, documents)
            else:
                # CrossEncoder
                pairs = [[query, doc] for doc in documents]
                scores = self.reranker.predict(pairs, show_progress_bar=False)
                scores = scores.tolist()
            
            # Update results with rerank scores
            for result, score in zip(results, scores):
                result.rerank_score = float(score)
            
            # Sort by rerank score
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return results[:top_k]
    
    def search(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        filters: Optional[Dict] = None,
        use_expansion: bool = True,
        use_reranking: bool = True,
    ) -> List[SearchResult]:
        """
        Hybrid search with metadata filtering and reranking
        
        Args:
            query: Search query
            top_k: Number of final results
            filters: Metadata filters (file_types, filenames, folders, etc.)
            use_expansion: Enable query expansion
            use_reranking: Enable cross-encoder reranking
        
        Returns:
            List of ranked search results
        """
        
        # Check cache
        if self.cache:
            cached = self.cache.get(query, filters)
            if cached:
                logger.info("✓ Cache hit")
                return cached[:top_k]
        
        start_time = time.time()
        
        # Query expansion
        if use_expansion and ENABLE_QUERY_EXPANSION:
            expanded_query = self.query_expander.expand(query)
            logger.info(f"Expanded query: {expanded_query}")
        else:
            expanded_query = query
        
        # Build metadata filter
        filter_ = None
        if filters:
            filter_ = self.filter_builder.build_filter(**filters)
        
        # Generate query embeddings
        if isinstance(self.embedder, OllamaBGEM3):
            query_vector = self.embedder.encode([expanded_query])[0]
            if query_vector is None:
                logger.error("Failed to generate query embedding, aborting search")
                return []
        else:
            query_vector = self.embedder.encode(expanded_query, convert_to_numpy=True).tolist()
        
        # Generate sparse vector
        query_sparse = self.bm25_encoder.encode_query(expanded_query)
        
        # Dense search
        dense_results = self._dense_search(query_vector, DENSE_TOP_K, filter_)
        logger.info(f"Dense search: {len(dense_results)} results")
        
        # Sparse search
        sparse_results = self._sparse_search(query_sparse, SPARSE_TOP_K, filter_)
        logger.info(f"Sparse search: {len(sparse_results)} results")
        
        # Hybrid fusion
        fused_results = self._hybrid_fusion(
            dense_results,
            sparse_results,
            DENSE_WEIGHT,
            SPARSE_WEIGHT
        )
        fused_results = fused_results[:HYBRID_TOP_K]
        logger.info(f"Hybrid fusion: {len(fused_results)} results")
        
        # Reranking
        if use_reranking and self.reranker:
            final_results = self._rerank(query, fused_results, top_k)
            logger.info(f"Reranked to top {len(final_results)}")
        else:
            final_results = fused_results[:top_k]
        
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time*1000:.2f}ms")
        
        # Cache results
        if self.cache:
            self.cache.put(query, final_results, filters)
        
        return final_results
    
    def search_with_metadata(
        self,
        query: str,
        top_k: int = 10,
        file_types: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        folders: Optional[List[str]] = None,
        min_word_count: Optional[int] = None,
        section_titles: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Convenience method with explicit metadata filters"""
        
        filters = {
            "file_types": file_types,
            "filenames": filenames,
            "folders": folders,
            "min_word_count": min_word_count,
            "section_titles": section_titles,
        }
        
        return self.search(query, top_k=top_k, filters=filters)


class AdvancedEvaluator:
    """Enhanced evaluation system"""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.results = []
    
    def evaluate_single(
        self,
        question: Dict[str, Any],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Evaluate single query"""
        
        query = question["question"]
        ground_truth = question["source_document"]
        
        # Retrieve
        start_time = time.time()
        results = self.retriever.search(query, top_k=top_k)
        latency = time.time() - start_time
        
        # Extract filenames
        retrieved_docs = []
        for result in results:
            filename = result.metadata.get("filename", "")
            if filename:
                retrieved_docs.append(filename)
        
        # Calculate metrics
        metrics = {
            "precision@1": 1.0 if ground_truth in retrieved_docs[:1] else 0.0,
            "precision@3": 1.0 if ground_truth in retrieved_docs[:3] else 0.0,
            "precision@5": 1.0 if ground_truth in retrieved_docs[:5] else 0.0,
            "precision@10": 1.0 if ground_truth in retrieved_docs[:10] else 0.0,
            "mrr": 1.0 / (retrieved_docs.index(ground_truth) + 1) if ground_truth in retrieved_docs else 0.0,
            "found": ground_truth in retrieved_docs,
        }
        
        return {
            "question_id": question.get("id", ""),
            "question": query,
            "ground_truth": ground_truth,
            "retrieved_docs": retrieved_docs,
            "latency_ms": latency * 1000,
            "metrics": metrics,
            "results": [
                {
                    "content": r.content[:200],
                    "score": r.score,
                    "dense_score": r.dense_score,
                    "sparse_score": r.sparse_score,
                    "rerank_score": r.rerank_score,
                    "filename": r.metadata.get("filename", ""),
                }
                for r in results
            ]
        }
    
    def evaluate_all(
        self,
        questions: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Evaluate all questions"""
        
        logger.info(f"\nEvaluating {len(questions)} questions...")
        
        all_results = []
        metrics_sum = defaultdict(list)
        
        for i, question in enumerate(questions, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(questions)}")
            
            result = self.evaluate_single(question, top_k)
            all_results.append(result)
            
            for metric, value in result["metrics"].items():
                metrics_sum[metric].append(value)
        
        # Aggregate metrics
        aggregate = {}
        for metric, values in metrics_sum.items():
            aggregate[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        
        return {
            "summary": {
                "total_questions": len(questions),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "aggregate_metrics": aggregate,
            "detailed_results": all_results,
        }


def main():
    """Demo usage"""
    
    logger.info("="*80)
    logger.info("ADVANCED HYBRID RETRIEVAL DEMO")
    logger.info("="*80)
    
    # Initialize retriever
    retriever = HybridRetriever(
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION,
        use_ollama=USE_OLLAMA_BGE_M3,
        use_reranker=USE_CROSS_ENCODER,
    )
    
    # Example search
    query = "What are the key features of machine learning?"
    
    logger.info(f"\nQuery: {query}")
    logger.info("Performing hybrid search...")
    
    results = retriever.search(query, top_k=5)
    
    logger.info(f"\nTop {len(results)} Results:")
    logger.info("="*80)
    
    for i, result in enumerate(results, 1):
        logger.info(f"\n{i}. Score: {result.score:.4f}")
        if result.rerank_score:
            logger.info(f"   Rerank: {result.rerank_score:.4f}")
        logger.info(f"   File: {result.metadata.get('filename', 'N/A')}")
        logger.info(f"   Section: {result.metadata.get('section_title', 'N/A')}")
        logger.info(f"   Content: {result.content[:150]}...")
    
    # Example with metadata filtering
    logger.info("\n" + "="*80)
    logger.info("Search with metadata filters:")
    logger.info("="*80)
    
    filtered_results = retriever.search_with_metadata(
        query=query,
        top_k=3,
        file_types=[".pdf"],
        min_word_count=50,
    )
    
    logger.info(f"\nFiltered to {len(filtered_results)} results")
    for i, result in enumerate(filtered_results, 1):
        logger.info(f"{i}. {result.metadata.get('filename')} - Score: {result.score:.4f}")


if __name__ == "__main__":
    main()