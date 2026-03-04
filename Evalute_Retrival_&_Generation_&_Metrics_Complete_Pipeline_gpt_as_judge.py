"""
UNIFIED RAG EVALUATION SYSTEM
==============================
- Combines retrieval metrics (Precision, Recall, MRR, NDCG) 
- With generation quality metrics (Faithfulness, Completeness, Correctness)
- Uses Ollama models + Gemini 2.0 Flash Lite for generation
- Uses GPT-4o-mini as judge
- Outputs both CSV and JSON formats
"""

import os
import json
import time
import requests
import logging
import csv
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SparseVector
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================================================
# CONFIGURATION
# ==================================================

# Paths
EVALUATION_QUESTIONS_PATH = "/home/olj3kor/praveen/RAG_work/evaluation_questions.json"
COLLECTION = "rag_hybrid_bge_m3"
QDRANT_URL = "http://localhost:7333"

# Embedding Config
EMBEDDING_MODEL = "bge-m3"

# Output paths
OUTPUT_DIR = "/home/olj3kor/praveen/RAG_work/"
RESULTS_JSON_PATH = os.path.join(OUTPUT_DIR, "unified_evaluation_results.json")
RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "unified_evaluation_results.csv")

# Retrieval config
TOP_K = 10
DENSE_TOP_K = 50
SPARSE_TOP_K = 50
HYBRID_TOP_K = 30
FINAL_TOP_K = 10

# Fusion weights
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

# API Keys
GPT4O_MINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GPT4O_MINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# Endpoints
GPT4O_MINI_ENDPOINT = "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-08-01-preview"
OLLAMA_BASE_URL = "http://localhost:11434"

# Ollama Config
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "bge-m3:latest"
FALLBACK_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_OLLAMA_RERANKER = True

# Cache and expansion settings
ENABLE_CACHE = False
CACHE_SIZE = 1000
ENABLE_QUERY_EXPANSION = False
EXPANSION_SYNONYMS = {}

# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass
class SearchResult:
    """Search result with multiple scores"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0


# ==================================================
# HYBRID RETRIEVER COMPONENTS
# ==================================================

class OllamaBGEM3:
    """Ollama BGE-M3 embedder and reranker"""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        self.available = self._check_availability()
        if not self.available:
            logger.error(f"❌ Ollama model '{model_name}' not available!")
            logger.error(f"   Attempted URL: {base_url}/api/tags")
        else:
            logger.info(f"✅ Ollama model '{model_name}' is available")    
    def _check_availability(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                return self.model_name in models
            return False
        except:
            return False
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def _get_embedding(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                logger.error(f"Embedding error: {response.status_code}")
                return [0.0] * 1024
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * 1024
    
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Rerank documents using Ollama"""
        # Simple similarity-based reranking
        query_emb = self._get_embedding(query)
        doc_embs = [self._get_embedding(doc) for doc in documents]
        
        scores = []
        for doc_emb in doc_embs:
            sim = cosine_similarity(
                np.array(query_emb).reshape(1, -1),
                np.array(doc_emb).reshape(1, -1)
            )[0][0]
            scores.append(float(sim))
        
        return scores


class BM25Encoder:
    """Simplified BM25 encoder"""
    
    def encode_query(self, text: str) -> SparseVector:
        """Convert text to sparse vector for BM25"""
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        indices = []
        values = []
        for word, freq in word_freq.items():
            word_hash = hash(word) % 100000
            indices.append(word_hash)
            values.append(float(freq))
        
        return SparseVector(indices=indices, values=values)


class QueryExpander:
    """Simple query expansion"""
    
    def __init__(self, synonyms: Dict[str, List[str]]):
        self.synonyms = synonyms
    
    def expand(self, query: str) -> str:
        return query


class MetadataFilterBuilder:
    """Build Qdrant metadata filters"""
    
    def build_filter(self, **kwargs) -> Optional[Filter]:
        return None


class QueryCache:
    """Simple query cache"""
    
    def __init__(self, max_size: int):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, query: str, filters: Any) -> Optional[List[SearchResult]]:
        return None
    
    def put(self, query: str, results: List[SearchResult], filters: Any):
        pass


# ==================================================
# HYBRID RETRIEVER (Integrated from your class)
# ==================================================

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
                except:
                    self.reranker = None
                    logger.warning("✗ No reranker available")
        else:
            self.reranker = None
        
        # Initialize components
        self.bm25_encoder = BM25Encoder()
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
        
        dense_scores = {r.id: (rank + 1, r) for rank, r in enumerate(dense_results)}
        sparse_scores = {r.id: (rank + 1, r) for rank, r in enumerate(sparse_results)}
        
        combined_scores = {}
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        for doc_id in all_ids:
            rrf_score = 0.0
            result = None
            
            if doc_id in dense_scores:
                rank, res = dense_scores[doc_id]
                rrf_score += dense_weight / (60 + rank)
                result = res
            
            if doc_id in sparse_scores:
                rank, res = sparse_scores[doc_id]
                rrf_score += sparse_weight / (60 + rank)
                if result is None:
                    result = res
            
            combined_scores[doc_id] = (rrf_score, result)
        
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )
        
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
            
            if isinstance(self.reranker, OllamaBGEM3):
                scores = self.reranker.rerank(query, documents)
            else:
                pairs = [[query, doc] for doc in documents]
                scores = self.reranker.predict(pairs, show_progress_bar=False)
                scores = scores.tolist()
            
            for result, score in zip(results, scores):
                result.rerank_score = float(score)
            
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
        """Hybrid search with metadata filtering and reranking"""
        
        if self.cache:
            cached = self.cache.get(query, filters)
            if cached:
                return cached[:top_k]
        
        start_time = time.time()
        
        if use_expansion and ENABLE_QUERY_EXPANSION:
            expanded_query = self.query_expander.expand(query)
        else:
            expanded_query = query
        
        filter_ = None
        if filters:
            filter_ = self.filter_builder.build_filter(**filters)
        
        if isinstance(self.embedder, OllamaBGEM3):
            query_vector = self.embedder.encode([expanded_query])[0]
        else:
            query_vector = self.embedder.encode(expanded_query, convert_to_numpy=True).tolist()
        
        query_sparse = self.bm25_encoder.encode_query(expanded_query)
        
        dense_results = self._dense_search(query_vector, DENSE_TOP_K, filter_)
        sparse_results = self._sparse_search(query_sparse, SPARSE_TOP_K, filter_)
        
        fused_results = self._hybrid_fusion(
            dense_results,
            sparse_results,
            DENSE_WEIGHT,
            SPARSE_WEIGHT
        )
        fused_results = fused_results[:HYBRID_TOP_K]
        
        if use_reranking and self.reranker:
            final_results = self._rerank(query, fused_results, top_k)
        else:
            final_results = fused_results[:top_k]
        
        if self.cache:
            self.cache.put(query, final_results, filters)
        
        return final_results


# ==================================================
# OLLAMA & GEMINI INTEGRATION
# ==================================================

def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        return []
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {e}")
        return []


def get_ollama_embedding(text: str, model_name: str) -> List[float]:
    """Generate embedding using Ollama API"""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": model_name, "prompt": text}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f"Ollama Embedding Error: {response.status_code}")
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise e


def generate_with_ollama(model_name: str, prompt: str, temperature: float = 0.1) -> str:
    """Generate response using Ollama model"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
        "options": {
            "num_predict": 512,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def generate_with_gemini(prompt: str, temperature: float = 0.1) -> str:
    """Generate response using Gemini 2.0 Flash Lite"""
    headers = {
        "genaiplatform-farm-subscription-key": GPT4O_MINI_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(
            GPT4O_MINI_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def evaluate_with_gpt4o_mini(question: str, context: str, answer: str, 
                             ground_truth: str) -> Dict[str, Any]:
    """Use GPT-4o-mini to evaluate answer quality"""
    
    judge_prompt = f"""### SYSTEM ROLE
You are an expert QA Auditor for RAG systems in automotive technical documentation. Evaluate the Generated Answer strictly against the Retrieved Context and the Ground Truth.

### STRICT RULES
1. Return ONLY the required JSON object (no extra text, no markdown code blocks).
2. Base scores solely on what is present in the provided Context and Ground Truth.
3. Penalize any claim in the Generated Answer that is not supported by the Context (hallucination).
4. If the Generated Answer omits safety warnings present in the Context, deduct from Completeness.

### INPUTS

**Question:** {question}

**Retrieved Context:**
{context}

**Generated Answer:**
{answer}

**Ground Truth Answer:**
{ground_truth}

### SCORING CRITERIA (0-10 integers)

1. **Faithfulness (0-10):**
   - Are all factual claims in the Generated Answer directly supported by the Context?
   - 10: All claims supported
   - 7-9: Mostly supported, minor unsupported details
   - 4-6: Mix of supported and unsupported claims
   - 0-3: Major hallucinations or fabricated information

2. **Completeness (0-10):**
   - Does the Generated Answer address all aspects required by the Question using available Context?
   - 10: Fully covers question with all relevant context details
   - 7-9: Answers question but misses minor details
   - 4-6: Partial answer, missing key information
   - 0-3: Barely addresses the question

3. **Correctness (0-10):**
   - Are technical details accurate compared to Ground Truth?
   - 10: All technical details match exactly
   - 7-9: Minor terminology differences, meaning preserved
   - 4-6: Some inaccuracies or imprecision
   - 0-3: Incorrect or misleading information

### REQUIRED OUTPUT (JSON only, no code blocks)

Return exactly this structure:

{{
  "faithfulness": <int 0-10>,
  "completeness": <int 0-10>,
  "correctness": <int 0-10>,
  "explanation": "<1-2 sentence justification mentioning most important support/failure and any missing safety warnings or hallucinations>"
}}

### FEW-SHOT CALIBRATION EXAMPLES

Example 1 (Perfect):
Context: "Section 5.1 — Replace brake fluid every 24 months."
Generated Answer: "Replace brake fluid every 24 months (Section 5.1)."
Ground Truth: "Replace brake fluid every 24 months."

Output:
{{"faithfulness":10,"completeness":10,"correctness":10,"explanation":"Answer exactly matches context and ground truth with proper citation."}}

Example 2 (Hallucination):
Context: "Section 5.1 — Replace brake fluid every 24 months."
Generated Answer: "Replace brake fluid every 24 months or 40,000 km."
Ground Truth: "Replace brake fluid every 24 months."

Output:
{{"faithfulness":2,"completeness":10,"correctness":3,"explanation":"The '40,000 km' clause is not present in the context and is a hallucination; rest matches."}}

Example 3 (Incomplete):
Context: "Section 3.1 — Disconnect battery before starter service. WARNING: Risk of electric shock."
Generated Answer: "Disconnect battery before starter service."
Ground Truth: "Disconnect battery to avoid electric shock."

Output:
{{"faithfulness":10,"completeness":6,"correctness":8,"explanation":"Answer is faithful but omits the safety warning about electric shock present in the context."}}

### YOUR EVALUATION
"""

    headers = {
        "genaiplatform-farm-subscription-key": GPT4O_MINI_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.0,  # Deterministic scoring
        "max_tokens": 300
    }
    
    try:
        response = requests.post(
            GPT4O_MINI_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            
            # Clean JSON
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            scores = json.loads(response_text)
            return scores
        else:
            return {
                "faithfulness": 0,
                "completeness": 0,
                "correctness": 0,
                "explanation": f"Judge API error: {response.status_code}"
            }
    except Exception as e:
        return {
            "faithfulness": 0,
            "completeness": 0,
            "correctness": 0,
            "explanation": f"Judge error: {str(e)}"
        }


def calculate_semantic_similarity(answer1: str, answer2: str, 
                                  embedding_model_name: str) -> float:
    """Calculate semantic similarity using Ollama embeddings"""
    if not answer1.strip() or not answer2.strip():
        return 0.0
    
    emb1_list = get_ollama_embedding(answer1, embedding_model_name)
    emb2_list = get_ollama_embedding(answer2, embedding_model_name)
    
    emb1 = np.array(emb1_list).reshape(1, -1)
    emb2 = np.array(emb2_list).reshape(1, -1)
    
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)


def parse_rag_output(raw_answer: str) -> Dict[str, str]:
    """
    Parse the 5-field RAG output format
    
    Returns:
        Dictionary with keys: answer, evidence, confidence, safety_notes, missing_info
    """
    parsed = {
        "answer": "",
        "evidence": "",
        "confidence": "Unknown",
        "safety_notes": "None",
        "missing_info": "None"
    }
    
    # Extract fields using regex patterns
    import re
    
    # Pattern for **Field:** content
    answer_match = re.search(r'\*\*Answer:\*\*\s*(.+?)(?=\n\*\*|\Z)', raw_answer, re.DOTALL)
    evidence_match = re.search(r'\*\*Evidence:\*\*\s*(.+?)(?=\n\*\*|\Z)', raw_answer, re.DOTALL)
    confidence_match = re.search(r'\*\*Confidence:\*\*\s*(.+?)(?=\n|\Z)', raw_answer, re.DOTALL)
    safety_match = re.search(r'\*\*Safety Notes:\*\*\s*(.+?)(?=\n\*\*|\Z)', raw_answer, re.DOTALL)
    missing_match = re.search(r'\*\*Missing Info:\*\*\s*(.+?)(?=\n\*\*|\Z)', raw_answer, re.DOTALL)
    
    if answer_match:
        parsed["answer"] = answer_match.group(1).strip()
    else:
        # Fallback: use entire text if no fields found
        parsed["answer"] = raw_answer.strip()
    
    if evidence_match:
        parsed["evidence"] = evidence_match.group(1).strip()
    
    if confidence_match:
        conf_text = confidence_match.group(1).strip()
        # Extract just "High", "Medium", or "Low"
        if "High" in conf_text:
            parsed["confidence"] = "High"
        elif "Medium" in conf_text:
            parsed["confidence"] = "Medium"
        elif "Low" in conf_text:
            parsed["confidence"] = "Low"
    
    if safety_match:
        parsed["safety_notes"] = safety_match.group(1).strip()
    
    if missing_match:
        parsed["missing_info"] = missing_match.group(1).strip()
    
    return parsed


# ==================================================
# RETRIEVAL METRICS CALCULATION
# ==================================================

def calculate_retrieval_metrics(
    retrieved_docs: List[str],
    ground_truth: str,
    top_k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Calculate Precision@k, Recall@k, MRR, NDCG@k
    
    Args:
        retrieved_docs: List of retrieved document filenames
        ground_truth: Ground truth document filename
        top_k_values: List of k values to calculate metrics for
    
    Returns:
        Dictionary with all retrieval metrics
    """
    metrics = {}
    
    # Check if ground truth is in retrieved docs
    found = ground_truth in retrieved_docs
    
    # Calculate rank (1-based)
    try:
        rank = retrieved_docs.index(ground_truth) + 1
        reciprocal_rank = 1.0 / rank
    except ValueError:
        rank = None
        reciprocal_rank = 0.0
    
    metrics['found'] = found
    metrics['rank'] = rank
    metrics['mrr'] = reciprocal_rank
    
    # Precision and Recall at different K
    for k in top_k_values:
        # Precision@k (1 if found in top-k, else 0)
        metrics[f'precision@{k}'] = 1.0 if ground_truth in retrieved_docs[:k] else 0.0
        
        # Recall@k (same as precision for single ground truth)
        metrics[f'recall@{k}'] = 1.0 if ground_truth in retrieved_docs[:k] else 0.0
    
    # Calculate NDCG@k
    for k in top_k_values:
        dcg = 0.0
        
        # Search for ground truth in top-k results
        docs_to_check = min(k, len(retrieved_docs))
        for i in range(docs_to_check):
            if retrieved_docs[i] == ground_truth:
                dcg = 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed, formula needs rank (1-indexed)
                break
        
        # IDCG: Best possible score (relevant doc at position 1)
        idcg = 1.0 / np.log2(2)  # = 1.0
        
        # NDCG = DCG / IDCG
        if dcg > 0:
            metrics[f'ndcg@{k}'] = dcg / idcg
        else:
            metrics[f'ndcg@{k}'] = 0.0
    
    return metrics


# ==================================================
# RAG SYSTEM
# ==================================================

class UnifiedRAGSystem:
    """Unified RAG system with integrated retrieval metrics"""
    
    def __init__(self, qdrant_url: str, collection: str):
        logger.info("Initializing Unified RAG System...")
        self.retriever = HybridRetriever(
            qdrant_url=qdrant_url,
            collection_name=collection,
            use_ollama=True,
            use_reranker=True
        )
        logger.info("✓ Unified RAG System initialized")
    
    def retrieve_with_metrics(
        self, 
        query: str, 
        ground_truth_doc: str,
        top_k: int = 10
    ) -> Tuple[List[SearchResult], str, Dict[str, float]]:
        """
        Retrieve documents and calculate retrieval metrics
        
        Returns:
            - List of search results
            - Context string
            - Retrieval metrics dictionary
        """
        # Perform search
        results = self.retriever.search(query, top_k=top_k)
        
        # Extract document filenames
        retrieved_docs = []
        for res in results:
            filename = res.metadata.get('filename', '')
            if filename:
                retrieved_docs.append(filename)
        
        # Calculate retrieval metrics
        retrieval_metrics = calculate_retrieval_metrics(retrieved_docs, ground_truth_doc)
        
        # Build context string
        context = "\n\n".join([
            f"[Document: {res.metadata.get('filename', 'unknown')}]\n{res.content}" 
            for res in results
        ])
        
        return results, context, retrieval_metrics
    
    def build_prompt(self, question: str, context: str, question_type: str = "factoid") -> str:
        """Build RAG prompt with adaptive complexity"""
        
        # Determine if Chain-of-Thought is beneficial
        use_cot = question_type in ["reasoning", "comparison"]
        cot_instruction = ""
        
        if use_cot:
            cot_instruction = """
### REASONING APPROACH (for complex questions)
For complex reasoning or comparison questions, you may use step-by-step thinking:
1. Identify relevant information from context
2. Analyze relationships or differences
3. Synthesize your answer
Include your reasoning briefly in the Evidence section if it helps clarify your answer.
"""
        
        prompt = f"""### SYSTEM ROLE
You are a Senior AUTOSAR technical documentation assistant. Your job is to answer user questions ONLY from the provided context documents. Do NOT use or invent outside knowledge.

### STRICT RULES
1. **Grounding only:** All factual claims must be supported by the provided Context.
2. **No hallucination:** If the Context does not contain the required information, reply exactly:
   "The provided context does not contain information to answer this question."
3. **Citations:** For any claim, include a precise citation in this format: [Document Title — Section or clause]. When quoting, include the exact quoted text in Evidence.
4. **Safety:** If any safety warnings appear in the Context, include them verbatim under Safety Notes.
5. **Ambiguity handling:** If the Context supports multiple plausible answers, explicitly list each interpretation with supporting citations.
{cot_instruction}
### INPUT DATA

Context Documents:
{context}

User Question:
{question}

### OUTPUT FORMAT (Markdown with these exact fields)

If an answer exists in context, produce:

**Answer:** <one-paragraph technical answer — only factual claims supported by the citations below>

**Evidence:** <one or more precise citations; when quoting, include quoted text and the citation: "..." [Document — Section]>

**Confidence:** <High / Medium / Low>
- High: Answer is explicitly stated in context
- Medium: Answer is inferred from multiple lines
- Low: Answer is based on partial data

**Safety Notes:** <exact safety warnings from context; if none, write "None">

**Missing Info:** <If additional document sections or specs would be needed to fully resolve the question, list them; if not needed, write "None">

---

If the answer is NOT present, reply exactly:

**Answer:** The provided context does not contain information to answer this question.

**Evidence:** <state which documents/sections you searched; e.g., "Searched: [Doc A — Sec 2], [Doc B — Appendix]">

**Confidence:** Low

**Safety Notes:** None

**Missing Info:** <list what specific spec/data is needed to answer>

### RESPONSE
"""
        return prompt


# ==================================================
# UNIFIED EVALUATION PIPELINE
# ==================================================

def run_unified_evaluation(
    questions_data: Dict,
    rag_system: UnifiedRAGSystem,
    ollama_models: List[str],
    embedding_model_name: str
) -> List[Dict[str, Any]]:
    """
    Run complete evaluation pipeline with both retrieval and generation metrics
    
    Returns:
        List of flat records (one per question per model)
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING UNIFIED EVALUATION PIPELINE")
    logger.info("="*80)
    
    questions = questions_data["questions"]
    all_models = ollama_models + ["gemini-2.0-flash-lite"]
    
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"Models: {len(all_models)}")
    logger.info(f"Total evaluations: {len(questions) * len(all_models)}")
    logger.info("="*80)
    
    flat_results = []
    
    for q_idx, question in enumerate(questions, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {q_idx}/{len(questions)}: {question['question'][:60]}...")
        logger.info(f"{'='*80}")
        
        # Retrieve with metrics
        logger.info("  → Retrieving and calculating retrieval metrics...")
        search_results, context, retrieval_metrics = rag_system.retrieve_with_metrics(
            query=question["question"],
            ground_truth_doc=question["source_document"],
            top_k=TOP_K
        )
        logger.info(f"  ✓ Retrieved {len(search_results)} documents")
        logger.info(f"  ✓ Retrieval metrics: Precision@5={retrieval_metrics['precision@5']:.2f}, MRR={retrieval_metrics['mrr']:.4f}")
        
        # Build prompt (with question type for adaptive CoT)
        prompt = rag_system.build_prompt(
            question["question"], 
            context,
            question_type=question.get("question_type", "factoid")
        )
        
        # Get Gemini answer first (baseline for semantic similarity)
        logger.info(f"  → Generating with Gemini (baseline)...")
        gemini_start = time.time()
        gemini_raw_answer = generate_with_gemini(prompt)
        gemini_latency = time.time() - gemini_start
        logger.info(f"    ✓ Generated ({gemini_latency:.2f}s)")
        
        # Parse Gemini output
        gemini_parsed = parse_rag_output(gemini_raw_answer)
        
        # Process all models
        for model_name in all_models:
            logger.info(f"  → Processing {model_name}...")
            
            # Generate answer
            if model_name == "gemini-2.0-flash-lite":
                raw_answer = gemini_raw_answer
                parsed_answer = gemini_parsed
                latency = gemini_latency
            else:
                start_time = time.time()
                raw_answer = generate_with_ollama(model_name, prompt)
                parsed_answer = parse_rag_output(raw_answer)
                latency = time.time() - start_time
                logger.info(f"    ✓ Generated ({latency:.2f}s)")
            
            # Evaluate with LLM-as-judge (using parsed answer field)
            logger.info(f"    → Evaluating with GPT-4o-mini...")
            judge_scores = evaluate_with_gpt4o_mini(
                question=question["question"],
                context=context,  # Full context, no truncation
                answer=parsed_answer["answer"],
                ground_truth=question["answer"]
            )
            
            # Calculate semantic similarity with Gemini (using parsed answer)
            if model_name != "gemini-2.0-flash-lite":
                sem_sim = calculate_semantic_similarity(
                    parsed_answer["answer"], 
                    gemini_parsed["answer"], 
                    embedding_model_name
                )
            else:
                sem_sim = 1.0
            
            # Calculate overall score
            overall_score = (
                judge_scores.get("faithfulness", 0) * 0.35 +
                judge_scores.get("completeness", 0) * 0.35 +
                judge_scores.get("correctness", 0) * 0.30
            )
            
            # Create flat record with all parsed fields
            record = {
                # Question info
                "question_id": question["id"],
                "question": question["question"],
                "ground_truth_answer": question["answer"],
                "source_document": question["source_document"],
                "question_type": question.get("question_type", "unknown"),
                "difficulty": question.get("difficulty", "unknown"),
                
                # Model info
                "model_name": model_name,
                "generated_answer_full": raw_answer,  # Full raw output
                "generated_answer": parsed_answer["answer"],  # Parsed answer field
                "evidence": parsed_answer["evidence"],
                "confidence": parsed_answer["confidence"],
                "safety_notes": parsed_answer["safety_notes"],
                "missing_info": parsed_answer["missing_info"],
                
                # Retrieval metrics (same for all models of this question)
                "precision@1": retrieval_metrics["precision@1"],
                "precision@3": retrieval_metrics["precision@3"],
                "precision@5": retrieval_metrics["precision@5"],
                "precision@10": retrieval_metrics["precision@10"],
                "recall@1": retrieval_metrics["recall@1"],
                "recall@3": retrieval_metrics["recall@3"],
                "recall@5": retrieval_metrics["recall@5"],
                "recall@10": retrieval_metrics["recall@10"],
                "mrr": retrieval_metrics["mrr"],
                "ndcg@1": retrieval_metrics["ndcg@1"],
                "ndcg@3": retrieval_metrics["ndcg@3"],
                "ndcg@5": retrieval_metrics["ndcg@5"],
                "ndcg@10": retrieval_metrics["ndcg@10"],
                "retrieval_rank": retrieval_metrics["rank"],
                
                # Generation quality metrics
                "faithfulness": judge_scores.get("faithfulness", 0),
                "completeness": judge_scores.get("completeness", 0),
                "correctness": judge_scores.get("correctness", 0),
                "overall_score": overall_score,
                "semantic_similarity_to_gemini": sem_sim,
                
                # Performance
                "latency_seconds": latency,
                
                # Additional info
                "judge_explanation": judge_scores.get("explanation", ""),
            }
            
            flat_results.append(record)
            logger.info(f"    ✓ Complete: Overall={overall_score:.2f}, Precision@5={retrieval_metrics['precision@5']:.2f}")
    
    return flat_results


# ==================================================
# OUTPUT FUNCTIONS
# ==================================================

def save_results_json(results: List[Dict[str, Any]], filepath: str):
    """Save results as JSON"""
    output = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": len(results),
            "evaluation_type": "unified_retrieval_and_generation"
        },
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"✓ JSON saved to: {filepath}")


def save_results_csv(results: List[Dict[str, Any]], filepath: str):
    """Save results as CSV"""
    if not results:
        logger.warning("No results to save to CSV")
        return
    
    # Define column order
    columns = [
        "question_id",
        "question",
        "ground_truth_answer",
        "source_document",
        "question_type",
        "difficulty",
        "model_name",
        "generated_answer_full",
        "generated_answer",
        "evidence",
        "confidence",
        "safety_notes",
        "missing_info",
        "precision@1",
        "precision@3",
        "precision@5",
        "precision@10",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",
        "retrieval_rank",
        "faithfulness",
        "completeness",
        "correctness",
        "overall_score",
        "semantic_similarity_to_gemini",
        "latency_seconds",
        "judge_explanation"
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"✓ CSV saved to: {filepath}")


def print_summary_statistics(results: List[Dict[str, Any]]):
    """Print summary statistics"""
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Group by model
    model_stats = defaultdict(lambda: {
        'precision@5': [],
        'recall@5': [],
        'mrr': [],
        'ndcg@5': [],
        'faithfulness': [],
        'completeness': [],
        'correctness': [],
        'overall_score': [],
        'latency': []
    })
    
    for record in results:
        model = record['model_name']
        model_stats[model]['precision@5'].append(record['precision@5'])
        model_stats[model]['recall@5'].append(record['recall@5'])
        model_stats[model]['mrr'].append(record['mrr'])
        model_stats[model]['ndcg@5'].append(record['ndcg@5'])
        model_stats[model]['faithfulness'].append(record['faithfulness'])
        model_stats[model]['completeness'].append(record['completeness'])
        model_stats[model]['correctness'].append(record['correctness'])
        model_stats[model]['overall_score'].append(record['overall_score'])
        model_stats[model]['latency'].append(record['latency_seconds'])
    
    # Print stats per model
    for model, stats in sorted(model_stats.items()):
        logger.info(f"\n{model}:")
        logger.info(f"  Retrieval Metrics:")
        logger.info(f"    Precision@5: {np.mean(stats['precision@5']):.4f}")
        logger.info(f"    Recall@5:    {np.mean(stats['recall@5']):.4f}")
        logger.info(f"    MRR:         {np.mean(stats['mrr']):.4f}")
        logger.info(f"    NDCG@5:      {np.mean(stats['ndcg@5']):.4f}")
        logger.info(f"  Generation Quality:")
        logger.info(f"    Faithfulness:  {np.mean(stats['faithfulness']):.2f}/10")
        logger.info(f"    Completeness:  {np.mean(stats['completeness']):.2f}/10")
        logger.info(f"    Correctness:   {np.mean(stats['correctness']):.2f}/10")
        logger.info(f"    Overall Score: {np.mean(stats['overall_score']):.2f}/10")
        logger.info(f"  Performance:")
        logger.info(f"    Avg Latency:   {np.mean(stats['latency']):.2f}s")
    
    logger.info("\n" + "="*80)


# ==================================================
# MAIN
# ==================================================

def main():
    """Main execution pipeline"""
    
    logger.info("="*80)
    logger.info("UNIFIED RAG EVALUATION SYSTEM")
    logger.info("="*80)
    
    # Load questions
    logger.info("\n1. Loading evaluation questions...")
    try:
        with open(EVALUATION_QUESTIONS_PATH, 'r') as f:
            questions_data = json.load(f)
        logger.info(f"   ✓ Loaded {len(questions_data['questions'])} questions")
    except FileNotFoundError:
        logger.error(f"Error: File not found at {EVALUATION_QUESTIONS_PATH}")
        return
    
    # Get Ollama models
    logger.info("\n2. Checking available Ollama models...")
    available_ollama = get_available_ollama_models()
    
    if not available_ollama:
        logger.warning("   ⚠️  No Ollama models found. Using Gemini only.")
        ollama_models = []
    else:
        logger.info(f"   ✓ Found {len(available_ollama)} Ollama models:")
        for i, model in enumerate(available_ollama, 1):
            logger.info(f"      {i}. {model}")
        
        selection = input("\n   Enter model numbers (comma-separated) or 'all': ").strip()
        
        if selection.lower() == 'all':
            ollama_models = available_ollama
        elif selection:
            indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
            ollama_models = [available_ollama[i] for i in indices if 0 <= i < len(available_ollama)]
        else:
            ollama_models = [available_ollama[0]] if available_ollama else []
        
        logger.info(f"   ✓ Selected {len(ollama_models)} models")
    
    # Initialize RAG system
    logger.info("\n3. Initializing Unified RAG system...")
    rag_system = UnifiedRAGSystem(QDRANT_URL, COLLECTION)
    
    # Run evaluation
    logger.info("\n4. Running unified evaluation...")
    results = run_unified_evaluation(
        questions_data=questions_data,
        rag_system=rag_system,
        ollama_models=ollama_models,
        embedding_model_name=EMBEDDING_MODEL
    )
    
    # Save results
    logger.info("\n5. Saving results...")
    save_results_json(results, RESULTS_JSON_PATH)
    save_results_csv(results, RESULTS_CSV_PATH)
    
    # Print summary
    print_summary_statistics(results)
    
    logger.info("\n" + "="*80)
    logger.info("✅ EVALUATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nResults saved to:")
    logger.info(f"  • JSON: {RESULTS_JSON_PATH}")
    logger.info(f"  • CSV:  {RESULTS_CSV_PATH}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()