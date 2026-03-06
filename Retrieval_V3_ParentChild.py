"""
ADVANCED RAG RETRIEVAL SYSTEM V3 — Parent-Child Collection
===========================================================
Companion to Qdrant_Database_Generation_V3.py.

Retrieval strategy
------------------
1. Hybrid search (dense BGE-M3 + sparse BM25) on the *children* collection.
   Small child chunks give high-precision matching.
2. Reciprocal Rank Fusion (RRF) to merge dense and sparse result lists.
3. Expand each matched child to its *parent* chunk (larger context window).
   If multiple children from the same parent are retrieved, the parent is
   included only once (highest-score child wins the score).
4. Optional cross-encoder re-ranking of the expanded parent texts.
5. Return the parent texts to the LLM — they carry the rich context that
   improves generation quality.

The parent-child approach combines the precision of small-chunk retrieval
with the contextual richness of large-chunk generation, without sacrificing
either property.
"""

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SparseVector,
)
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG  (mirrors Qdrant_Database_Generation_V3.py defaults)
# ═══════════════════════════════════════════════════════════════════════════

COLLECTION_BASE  = "rag_v3"
CHILDREN_COL     = f"{COLLECTION_BASE}_children"
PARENTS_COL      = f"{COLLECTION_BASE}_parents"
QDRANT_URL       = "http://localhost:7333"
OLLAMA_URL       = "http://localhost:11434"
OLLAMA_MODEL     = "bge-m3:latest"
FALLBACK_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
BM25_INDEX_PATH  = "bm25_index_v3.json"

# Retrieval knobs
CHILD_DENSE_TOP_K  = 50
CHILD_SPARSE_TOP_K = 50
CHILD_HYBRID_TOP_K = 30   # after RRF fusion
FINAL_TOP_K        = 10   # after parent-expansion + re-ranking

# Fusion weights for RRF
RRF_K          = 60        # RRF constant (k in 1/(k+rank))
DENSE_WEIGHT   = 0.6
SPARSE_WEIGHT  = 0.4

# Cross-encoder re-ranking
USE_OLLAMA_RERANKER  = True
MAX_RERANK_CHARS     = 2000   # truncate parent text for reranking to avoid timeouts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChildResult:
    """A retrieved child chunk with its scores."""
    child_id: str
    parent_id: str
    content: str
    filename: str
    section_title: str
    section_hierarchy: List[str]
    page_number: Optional[int]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentResult:
    """A parent chunk expanded from child hits, ready for LLM consumption."""
    parent_id: str
    content: str              # full parent text (the context fed to the LLM)
    filename: str
    section_title: str
    section_hierarchy: List[str]
    page_number: Optional[int]
    child_score: float        # best hybrid score among its matched children
    rerank_score: float = 0.0
    final_score: float = 0.0
    matched_children: int = 1 # how many child chunks matched for this parent
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Embedder (same interface as V3 ingestion)
# ═══════════════════════════════════════════════════════════════════════════

class OllamaBGEM3:
    """BGE-M3 query embedder + cosine-safe reranker."""

    def __init__(self, base_url: str = OLLAMA_URL,
                 model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.dimension = 1024
        self.available = self._check()

    def _check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                logger.info(f"✓ Ollama at {self.base_url}")
                return True
        except Exception:
            pass
        logger.warning(f"✗ Ollama unavailable at {self.base_url}")
        return False

    def embed_query(self, text: str) -> Optional[List[float]]:
        if not self.available:
            return None
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            if r.status_code == 200:
                vec = r.json().get("embedding")
                if vec:
                    arr = np.array(vec, dtype=np.float64)
                    if not np.isfinite(arr).all():
                        arr[~np.isfinite(arr)] = 0.0
                        vec = arr.tolist()
                    return vec
        except Exception as e:
            logger.warning(f"Query embedding error: {e}")
        return None

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Cosine-similarity reranking using BGE-M3 embeddings."""
        scores: List[float] = []
        try:
            q_vec = self.embed_query(query)
            if q_vec is None:
                return [0.0] * len(documents)
            q_arr = np.array(q_vec, dtype=np.float64)
            q_norm = np.linalg.norm(q_arr)
        except Exception:
            return [0.0] * len(documents)

        for doc in documents:
            try:
                d_vec = self.embed_query(doc[:MAX_RERANK_CHARS])
                if d_vec is None:
                    scores.append(0.0)
                    continue
                d_arr = np.array(d_vec, dtype=np.float64)
                d_norm = np.linalg.norm(d_arr)
                if q_norm < 1e-10 or d_norm < 1e-10:
                    scores.append(0.0)
                else:
                    scores.append(float(np.dot(q_arr, d_arr) / (q_norm * d_norm)))
            except Exception as e:
                logger.debug(f"Rerank error: {e}")
                scores.append(0.0)
        return scores


# ═══════════════════════════════════════════════════════════════════════════
# BM25 index loader (for sparse query vector)
# ═══════════════════════════════════════════════════════════════════════════

class BM25QueryEncoder:
    def __init__(self, index_path: str = BM25_INDEX_PATH):
        self.vocabulary: Dict[str, int] = {}
        self.token_idf: Dict[str, float] = {}
        self._load(index_path)

    def _load(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
            self.vocabulary = data.get("vocabulary", {})
            self.token_idf  = data.get("token_idf",  {})
            logger.info(
                f"✓ BM25 index loaded ({len(self.vocabulary)} tokens) from {path}"
            )
        except Exception as e:
            logger.warning(f"Could not load BM25 index from {path}: {e}")

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in word_tokenize(text) if t.isalnum()]

    def encode(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        total = len(tokens)
        counts: Dict[str, int] = {}
        for t in tokens:
            if t in self.vocabulary:
                counts[t] = counts.get(t, 0) + 1
        indices, values = [], []
        for t, cnt in counts.items():
            tf = cnt / total if total else 0.0
            indices.append(self.vocabulary[t])
            values.append(float(tf * self.token_idf.get(t, 1.0)))
        return SparseVector(indices=indices, values=values)


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid search + RRF
# ═══════════════════════════════════════════════════════════════════════════

def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)


def hybrid_search_children(
    client: QdrantClient,
    query_vec: List[float],
    sparse_vec: SparseVector,
    collection: str = CHILDREN_COL,
    dense_top_k: int = CHILD_DENSE_TOP_K,
    sparse_top_k: int = CHILD_SPARSE_TOP_K,
    hybrid_top_k: int = CHILD_HYBRID_TOP_K,
    metadata_filter: Optional[Filter] = None,
) -> List[ChildResult]:
    """
    Dense + sparse search on the children collection, fused with RRF.
    """

    # Dense search
    dense_hits = client.search(
        collection_name=collection,
        query_vector=("dense", query_vec),
        query_filter=metadata_filter,
        limit=dense_top_k,
        with_payload=True,
    )

    # Sparse search
    sparse_hits = client.search(
        collection_name=collection,
        query_vector=("bm25", sparse_vec),
        query_filter=metadata_filter,
        limit=sparse_top_k,
        with_payload=True,
    )

    # RRF fusion
    scores: Dict[str, float] = defaultdict(float)
    hit_payloads: Dict[str, Any] = {}
    hit_dense: Dict[str, float] = {}
    hit_sparse: Dict[str, float] = {}

    for rank, hit in enumerate(dense_hits):
        pid = str(hit.id)
        scores[pid]  += DENSE_WEIGHT  * _rrf_score(rank)
        hit_dense[pid] = hit.score
        hit_payloads[pid] = hit.payload

    for rank, hit in enumerate(sparse_hits):
        pid = str(hit.id)
        scores[pid]  += SPARSE_WEIGHT * _rrf_score(rank)
        hit_sparse[pid] = hit.score
        if pid not in hit_payloads:
            hit_payloads[pid] = hit.payload

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:hybrid_top_k]

    results: List[ChildResult] = []
    for cid, score in ranked:
        pl = hit_payloads.get(cid, {})
        results.append(ChildResult(
            child_id=cid,
            parent_id=pl.get("parent_id", ""),
            content=pl.get("content", ""),
            filename=pl.get("filename", ""),
            section_title=pl.get("section_title", ""),
            section_hierarchy=pl.get("section_hierarchy", []),
            page_number=pl.get("page_number"),
            dense_score=hit_dense.get(cid, 0.0),
            sparse_score=hit_sparse.get(cid, 0.0),
            hybrid_score=score,
            metadata=pl,
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Parent expansion
# ═══════════════════════════════════════════════════════════════════════════

def expand_to_parents(
    client: QdrantClient,
    child_results: List[ChildResult],
    parents_collection: str = PARENTS_COL,
) -> List[ParentResult]:
    """
    Group child results by parent_id, fetch the parent documents from Qdrant,
    and return one ParentResult per unique parent (keyed by highest child score).
    """
    # Group by parent
    parent_groups: Dict[str, List[ChildResult]] = defaultdict(list)
    for cr in child_results:
        if cr.parent_id:
            parent_groups[cr.parent_id].append(cr)
        else:
            # No parent_id in payload → fall back to using the child text itself
            parent_groups[cr.child_id].append(cr)

    if not parent_groups:
        return []

    # Fetch parent payloads in one batch
    parent_ids = list(parent_groups.keys())
    try:
        parent_points = client.retrieve(
            collection_name=parents_collection,
            ids=parent_ids,
            with_payload=True,
        )
        fetched: Dict[str, Any] = {str(p.id): p.payload for p in parent_points}
    except Exception as e:
        logger.warning(f"Parent fetch error (falling back to child text): {e}")
        fetched = {}

    results: List[ParentResult] = []
    for pid, children in parent_groups.items():
        best_child = max(children, key=lambda c: c.hybrid_score)
        payload = fetched.get(pid, {})

        # Use parent text if available; else fall back to the best child text
        content = payload.get("content") or best_child.content

        results.append(ParentResult(
            parent_id=pid,
            content=content,
            filename=payload.get("filename", best_child.filename),
            section_title=payload.get("section_title", best_child.section_title),
            section_hierarchy=payload.get("section_hierarchy",
                                          best_child.section_hierarchy),
            page_number=payload.get("page_number", best_child.page_number),
            child_score=best_child.hybrid_score,
            matched_children=len(children),
            metadata=payload or best_child.metadata,
        ))

    # Sort by best child hybrid score
    results.sort(key=lambda r: r.child_score, reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main retriever class
# ═══════════════════════════════════════════════════════════════════════════

class ParentChildRetriever:
    """
    High-level retriever using the V3 parent-child index.

    Usage
    -----
    retriever = ParentChildRetriever()
    results = retriever.retrieve("What is the safety requirement for X?")
    for r in results:
        print(r.filename, r.content[:200])
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        children_col: str = CHILDREN_COL,
        parents_col: str = PARENTS_COL,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
        bm25_index_path: str = BM25_INDEX_PATH,
    ):
        self.client       = QdrantClient(url=qdrant_url)
        self.children_col = children_col
        self.parents_col  = parents_col
        self.embedder     = OllamaBGEM3(ollama_url, ollama_model)
        self.bm25         = BM25QueryEncoder(bm25_index_path)

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        metadata_filter: Optional[Filter] = None,
        rerank: bool = USE_OLLAMA_RERANKER,
    ) -> List[ParentResult]:
        """
        Execute the full parent-child retrieval pipeline and return ranked
        ParentResult objects whose `.content` is suitable for LLM context.
        """
        # 1. Embed query
        query_vec = self.embedder.embed_query(query)
        if query_vec is None:
            logger.error("Query embedding failed; cannot retrieve.")
            return []

        sparse_vec = self.bm25.encode(query)

        # 2. Hybrid child search
        children = hybrid_search_children(
            client=self.client,
            query_vec=query_vec,
            sparse_vec=sparse_vec,
            collection=self.children_col,
            metadata_filter=metadata_filter,
        )
        logger.info(f"  Child hits: {len(children)}")

        # 3. Expand to parents
        parents = expand_to_parents(
            client=self.client,
            child_results=children,
            parents_collection=self.parents_col,
        )
        logger.info(f"  Unique parents after expansion: {len(parents)}")

        # 4. Re-rank using BGE-M3 cosine similarity on parent texts
        if rerank and self.embedder.available and parents:
            parent_texts = [p.content[:MAX_RERANK_CHARS] for p in parents]
            rr_scores    = self.embedder.rerank(query, parent_texts)
            for pr, rr in zip(parents, rr_scores):
                pr.rerank_score = rr
                # Blend child hybrid score and rerank score
                pr.final_score = 0.5 * pr.child_score + 0.5 * rr
            parents.sort(key=lambda r: r.final_score, reverse=True)
        else:
            for pr in parents:
                pr.final_score = pr.child_score

        return parents[:top_k]

    def format_context(
        self,
        results: List[ParentResult],
        max_chars_per_result: int = MAX_RERANK_CHARS,
    ) -> str:
        """
        Format retrieved parent chunks as a numbered context block suitable
        for insertion into an LLM prompt.
        """
        parts: List[str] = []
        for i, r in enumerate(results, 1):
            section = " > ".join(r.section_hierarchy) if r.section_hierarchy else ""
            header = f"[{i}] {r.filename}"
            if section:
                header += f" — {section}"
            if r.page_number:
                header += f" (p.{r.page_number})"
            text = r.content[:max_chars_per_result]
            parts.append(f"{header}\n{text}")
        return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Quick CLI smoke-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="V3 Parent-Child Retriever — interactive query test"
    )
    parser.add_argument("--query",     default="What are the safety requirements?")
    parser.add_argument("--top-k",     type=int, default=FINAL_TOP_K)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--collection", default=COLLECTION_BASE)
    args = parser.parse_args()

    retriever = ParentChildRetriever(
        children_col=f"{args.collection}_children",
        parents_col =f"{args.collection}_parents",
    )

    logger.info(f"\nQuery: {args.query}")
    t0 = time.time()
    results = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        rerank=not args.no_rerank,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"Query : {args.query}")
    print(f"Results: {len(results)}  |  Time: {elapsed:.2f}s")
    print("=" * 70)
    for r in results:
        section = " > ".join(r.section_hierarchy) if r.section_hierarchy else "—"
        print(f"\n[{r.filename}]  section={section}  "
              f"child_score={r.child_score:.4f}  "
              f"final_score={r.final_score:.4f}  "
              f"matched_children={r.matched_children}")
        print(r.content[:400])
        print("...")

    print("\n=== Context block (for LLM) ===")
    print(retriever.format_context(results, max_chars_per_result=400))
