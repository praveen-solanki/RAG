"""
Minimal retrieval test - verify basic functionality
"""

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

print("=" * 80)
print("MINIMAL RETRIEVAL TEST")
print("=" * 80)

# 1. Connect to Qdrant
print("\n1??  Connecting to Qdrant...")
document_store = QdrantDocumentStore(
    url="http://localhost:7333",
    index="rag_database_384_new",
    embedding_dim=384,
    similarity="cosine",
    recreate_index=False,
)

doc_count = document_store.count_documents()
print(f"   ? Connected. Documents in store: {doc_count}")

if doc_count == 0:
    print("   ? No documents found! Something is wrong.")
    exit(1)

# 2. Test embedding
print("\n2??  Testing query embedding...")
query_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

test_query = "What is the content of the document?"
embedding_result = query_embedder.run(text=test_query)
query_embedding = embedding_result["embedding"]

print(f"   Query: {test_query}")
print(f"   Embedding shape: {len(query_embedding)}")
print(f"   ? Embedding generated")

# 3. Test retriever directly
print("\n3??  Testing retriever...")
retriever = QdrantEmbeddingRetriever(
    document_store=document_store,
    top_k=5,
)

# Run retriever with the embedding
retrieval_result = retriever.run(query_embedding=query_embedding)
docs = retrieval_result["documents"]

print(f"   Retrieved documents: {len(docs)}")

if len(docs) == 0:
    print("   ? No documents retrieved!")
    print("   Trying without score threshold...")
    
    # Try again with no threshold
    retriever_no_threshold = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=5,
        # No score_threshold
    )
    
    retrieval_result = retriever_no_threshold.run(query_embedding=query_embedding)
    docs = retrieval_result["documents"]
    print(f"   Retrieved (no threshold): {len(docs)}")

if docs:
    print("\n   ? SUCCESS! Retrieved documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n   Doc {i}:")
        print(f"   Score: {doc.score:.4f}")
        print(f"   Content: {doc.content[:200]}...")
else:
    print("\n   ? Still no documents. Debugging needed.")

# 4. Test full pipeline
print("\n4??  Testing full pipeline...")

query_embedder_pipeline = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

retriever_pipeline = QdrantEmbeddingRetriever(
    document_store=document_store,
    top_k=5,
)

prompt_builder = PromptBuilder(
    template="""
Answer based on the context below.

Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:
""",
    required_variables=["documents", "question"],
)

pipeline = Pipeline()
pipeline.add_component("query_embedder", query_embedder_pipeline)
pipeline.add_component("retriever", retriever_pipeline)
pipeline.add_component("prompt_builder", prompt_builder)

pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")

result = pipeline.run(
    {
        "query_embedder": {"text": test_query},
        "prompt_builder": {"question": test_query},
    },
    include_outputs_from=["retriever", "prompt_builder"]
)

docs = result["retriever"]["documents"]
prompt = result["prompt_builder"]["prompt"]

print(f"   Retrieved {len(docs)} documents")
print(f"\n   Generated prompt length: {len(prompt)}")

if docs:
    print("\n   ? Pipeline working! Sample doc:")
    print(f"   Score: {docs[0].score:.4f}")
    print(f"   Content: {docs[0].content[:200]}...")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)