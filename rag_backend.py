"""
RAG Backend Module
==================
Provides generate_answer() for the Streamlit dashboard (app.py).

Wraps the Haystack / Qdrant retrieval pipeline that mirrors the approach
used by Retrieval_Advanced.py, but exposes a clean function interface
that app.py can call without running top-level pipeline setup code.
"""

import os
from typing import Any, Dict, List, Optional

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from ollama_models import OllamaGenerator

# ─── Configuration ────────────────────────────────────────────────────────────

QDRANT_URL      = os.environ.get("QDRANT_URL", "http://localhost:7333")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
DEFAULT_TOP_K   = 10

PROMPT_TEMPLATE = """
You are a helpful AI assistant answering questions based on provided context.

Context information:
{% for doc in documents %}
{{ doc.content }}

{% endfor %}

Guidelines:
- Answer based primarily on the context above.
- If the context does not contain the answer, say "I cannot find this information in the provided documents."
- Be concise but thorough.
- Synthesize information from multiple context fragments when needed.

Question: {{ question }}

Answer:
"""

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_document_store(collection: str) -> QdrantDocumentStore:
    """Connect to an existing Qdrant collection (never recreates it)."""
    return QdrantDocumentStore(
        url=QDRANT_URL,
        index=collection,
        embedding_dim=EMBEDDING_DIM,
        similarity="cosine",
        recreate_index=False,
        return_embedding=False,
    )


def _build_pipeline(model_name: str, document_store: QdrantDocumentStore) -> Pipeline:
    """Assemble the Haystack RAG pipeline for one Ollama model."""
    query_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)

    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=DEFAULT_TOP_K,
    )

    prompt_builder = PromptBuilder(
        template=PROMPT_TEMPLATE,
        required_variables=["documents", "question"],
    )

    llm = OllamaGenerator(
        model=model_name,
        generation_kwargs={
            "temperature": 0.1,
            "num_predict": 512,
            "top_p": 0.9,
        },
    )

    pipeline = Pipeline()
    pipeline.add_component("query_embedder", query_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)

    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    return pipeline


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    model_name: str,
    file_objects: Optional[List] = None,
    collection: str = "rag_database_ARAI",
) -> Dict[str, Any]:
    """
    Retrieve relevant documents from Qdrant and generate an answer using Ollama.

    Parameters
    ----------
    question     : The user's question.
    model_name   : Ollama model name (e.g. "mistral:7b").
    file_objects : Streamlit UploadedFile objects (currently unused for retrieval;
                   the collection already contains the pre-indexed documents).
    collection   : Qdrant collection name to query.

    Returns
    -------
    dict with keys:
        answer  (str)  – generated answer text
        model   (str)  – model name used
        sources (list) – list of dicts with filename, score, content, page
    """
    document_store = _build_document_store(collection)
    pipeline = _build_pipeline(model_name, document_store)

    result = pipeline.run(
        {
            "query_embedder": {"text": question},
            "prompt_builder": {"question": question},
        },
        include_outputs_from=["retriever"],
    )

    answer = (result.get("llm", {}).get("replies") or [""])[0]
    docs   = result.get("retriever", {}).get("documents") or []

    sources: List[Dict[str, Any]] = []
    for doc in docs:
        meta = doc.meta or {}
        sources.append({
            "filename": meta.get("filename", meta.get("file_name", "Unknown")),
            "score":    float(doc.score or 0.0),
            "content":  doc.content or "",
            "page":     meta.get("page_number", meta.get("page")),
        })

    return {
        "answer":  answer.strip() if answer else "No answer generated.",
        "model":   model_name,
        "sources": sources,
    }
