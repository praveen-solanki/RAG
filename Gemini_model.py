"""
Gemini Model Backend
====================
Provides generate_answer() for the Streamlit dashboard (app.py) using
the Bosch AI API with Gemini 2.0 Flash Lite.

Used for the "Evaluate with Gemini" comparison feature in app.py.
Requires the GEMINI_API_KEY environment variable to be set.
"""

import os
from typing import Any, Dict, List, Optional

import requests
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

# ─── Configuration ────────────────────────────────────────────────────────────

QDRANT_URL      = os.environ.get("QDRANT_URL", "http://localhost:7333")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
DEFAULT_TOP_K   = 10

# Bosch AI API endpoint for Gemini 2.0 Flash Lite (OpenAI-compatible format)
GEMINI_ENDPOINT   = (
    "https://aoai-farm.bosch-temp.com/api/openai/deployments/"
    "google-gemini-2-0-flash-lite/chat/completions"
)
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"

PROMPT_TEMPLATE = (
    "You are a helpful AI assistant answering questions based on provided context.\n\n"
    "Context information:\n"
    "{context}\n\n"
    "Guidelines:\n"
    "- Answer based primarily on the context above.\n"
    "- If the context does not contain the answer, say "
    '"I cannot find this information in the provided documents."\n'
    "- Be concise but thorough.\n"
    "- Synthesize information from multiple context fragments when needed.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _retrieve_documents(question: str, collection: str) -> list:
    """Embed the query and retrieve top-K documents from Qdrant."""
    document_store = QdrantDocumentStore(
        url=QDRANT_URL,
        index=collection,
        embedding_dim=EMBEDDING_DIM,
        similarity="cosine",
        recreate_index=False,
        return_embedding=False,
    )

    embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
    embedder.warm_up()
    embedding_result = embedder.run(text=question)
    query_embedding  = embedding_result["embedding"]

    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=DEFAULT_TOP_K,
    )
    retrieval_result = retriever.run(query_embedding=query_embedding)
    return retrieval_result.get("documents") or []


def _call_gemini_api(prompt: str, api_key: str) -> str:
    """POST to the Bosch Gemini 2.0 Flash Lite endpoint and return the reply."""
    headers = {
        "genaiplatform-farm-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": GEMINI_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    response = requests.post(
        GEMINI_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    file_objects: Optional[List] = None,
    collection: str = "rag_database_ARAI",
) -> Dict[str, Any]:
    """
    Retrieve documents from Qdrant and generate an answer with Gemini 2.0 Flash Lite.

    Parameters
    ----------
    question     : The user's question.
    file_objects : Streamlit UploadedFile objects (currently unused for retrieval;
                   the collection already contains the pre-indexed documents).
    collection   : Qdrant collection name to query.

    Returns
    -------
    dict with keys:
        answer  (str)  – generated answer text
        model   (str)  – model name used
        sources (list) – list of dicts with filename, score, content, page

    Raises
    ------
    RuntimeError  – if GEMINI_API_KEY is not set.
    requests.HTTPError – if the Gemini API call fails.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    docs    = _retrieve_documents(question, collection)
    context = "\n\n".join(
        f"[{i + 1}] {doc.content}" for i, doc in enumerate(docs)
    )

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = _call_gemini_api(prompt, api_key)

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
        "model":   GEMINI_MODEL_NAME,
        "sources": sources,
    }
