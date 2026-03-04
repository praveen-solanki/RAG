## -*- coding: utf-8 -*-
#
#import os
#from haystack import Pipeline
#from haystack.components.embedders import SentenceTransformersTextEmbedder
#from haystack.components.builders import PromptBuilder
#
#from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
#from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
#
## ?? IMPORT FROM ollama_models.py
#from ollama_models import get_available_models, OllamaGenerator
#
#
## ==================================================
## 1. CONNECT TO EXISTING QDRANT
## ==================================================
#
#print("Connecting to Qdrant...")
#document_store = QdrantDocumentStore(
#    url="http://localhost:7333",
#    index="rag_database_384_new",
#    embedding_dim=384,
#    similarity="cosine",
#    recreate_index=False,
#    return_embedding=False,
#)
#
#doc_count = document_store.count_documents()
#print(f"? Connected. Found {doc_count} documents in Qdrant")
#
#if doc_count == 0:
#    print("? ERROR: No documents found in Qdrant!")
#    print("Please run the ingestion script first.")
#    exit(1)
#
#
## ==================================================
## 2. BUILD RAG PIPELINE FUNCTION
## ==================================================
#
#def build_rag_pipeline(model_name: str):
#    """Build a fresh RAG pipeline for each model"""
#    
#    query_embedder = SentenceTransformersTextEmbedder(
#        model="sentence-transformers/all-MiniLM-L6-v2"
#    )
#
#    retriever = QdrantEmbeddingRetriever(
#        document_store=document_store,
#        top_k=5,
#    )
#
#    prompt_builder = PromptBuilder(
#        template="""
#You are a strict question-answering system.
#
#RULES:
#- Use ONLY the information inside <context>.
#- If the answer is NOT explicitly stated, reply exactly:
#  I don't know.
#- Do NOT use prior knowledge.
#- Do NOT infer or guess.
#
#<context>
#{% for doc in documents %}
#{{ doc.content }}
#{% endfor %}
#</context>
#
#Question: {{ question }}
#
#Answer:
#""",
#        required_variables=["documents", "question"],
#    )
#
#    llm = OllamaGenerator(
#        model=model_name,
#        generation_kwargs={
#            "temperature": 0.0,
#            "num_predict": 256,
#        },
#    )
#
#    p = Pipeline()
#    p.add_component("query_embedder", query_embedder)
#    p.add_component("retriever", retriever)
#    p.add_component("prompt_builder", prompt_builder)
#    p.add_component("llm", llm)
#
#    p.connect("query_embedder.embedding", "retriever.query_embedding")
#    p.connect("retriever.documents", "prompt_builder.documents")
#    p.connect("prompt_builder.prompt", "llm.prompt")
#
#    return p
#
#
## ==================================================
## 3. GET QUESTIONS FROM USER (MULTIPLE METHODS)
## ==================================================
#
#def get_questions_from_file(filepath):
#    """Read questions from a text file (one question per line)"""
#    if not os.path.exists(filepath):
#        print(f"? File not found: {filepath}")
#        return []
#    
#    with open(filepath, 'r', encoding='utf-8') as f:
#        questions = [line.strip() for line in f if line.strip()]
#    
#    return questions
#
#
#def get_questions_interactive():
#    """Get questions interactively from user"""
#    questions = []
#    print("\nEnter your questions (one per line).")
#    print("Type 'done' when finished, or press Ctrl+C to cancel.\n")
#    
#    while True:
#        try:
#            q = input(f"Question {len(questions) + 1}: ").strip()
#            if q.lower() == 'done':
#                break
#            if q:
#                questions.append(q)
#        except KeyboardInterrupt:
#            print("\n\n??  Cancelled by user")
#            break
#    
#    return questions
#
#
#def get_questions():
#    """Main function to get questions from user"""
#    print("\n" + "=" * 80)
#    print("QUESTION INPUT OPTIONS")
#    print("=" * 80)
#    print("1. Enter a single question")
#    print("2. Enter multiple questions interactively")
#    print("3. Load questions from a text file")
#    print("4. Use default question")
#    print("=" * 80)
#    
#    choice = input("\nSelect option (1-4): ").strip()
#    
#    questions = []
#    
#    if choice == '1':
#        # Single question
#        q = input("\nEnter your question: ").strip()
#        if q:
#            questions.append(q)
#    
#    elif choice == '2':
#        # Multiple questions interactively
#        questions = get_questions_interactive()
#    
#    elif choice == '3':
#        # Load from file
#        filepath = input("\nEnter path to questions file: ").strip()
#        questions = get_questions_from_file(filepath)
#        if questions:
#            print(f"? Loaded {len(questions)} questions from file")
#    
#    elif choice == '4':
#        # Default question
#        questions = ["What is the content of the document?"]
#        print("? Using default question")
#    
#    else:
#        print("? Invalid choice. Using default question.")
#        questions = ["What is the content of the document?"]
#    
#    if not questions:
#        print("??  No questions provided. Using default question.")
#        questions = ["What is the content of the document?"]
#    
#    return questions
#
#
## ==================================================
## 4. SELECT OLLAMA MODELS
## ==================================================
#
#ollama_models = get_available_models()
#
#print(f"\n?? Found {len(ollama_models)} Ollama models")
#for i, m in enumerate(ollama_models, start=1):
#    print(f"  {i}. {m}")
#
#selection = input(
#    "\nEnter model numbers (comma-separated), or press Enter to run all: "
#).strip()
#
#if not selection:
#    selected_models = ollama_models
#else:
#    indices = []
#    for x in selection.split(","):
#        x = x.strip()
#        if x.isdigit():
#            idx = int(x) - 1
#            if 0 <= idx < len(ollama_models):
#                indices.append(idx)
#
#    selected_models = [ollama_models[i] for i in indices]
#
#
## ==================================================
## 5. GET QUESTIONS
## ==================================================
#
#QUESTIONS = get_questions()
#
#print(f"\n?? Configuration:")
#print(f"   Questions: {len(QUESTIONS)}")
#print(f"   Models: {len(selected_models)}")
#print(f"   Total runs: {len(QUESTIONS) * len(selected_models)}")
#
#print("\n?? Questions to process:")
#for i, q in enumerate(QUESTIONS, 1):
#    print(f"   {i}. {q}")
#
#proceed = input("\nProceed with these settings? (y/n): ").strip().lower()
#if proceed != 'y':
#    print("? Cancelled by user")
#    exit(0)
#
#
## ==================================================
## 6. RUN QUERIES
## ==================================================
#
#OUTPUT_FILE = "rag_outputs_multiple_questions.txt"
#
#print(f"\n{'='*80}")
#print("STARTING RAG PIPELINE")
#print(f"{'='*80}")
#
#with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#    # Write header
#    f.write("RAG PIPELINE OUTPUT\n")
#    f.write("=" * 80 + "\n")
#    f.write(f"Total Questions: {len(QUESTIONS)}\n")
#    f.write(f"Total Models: {len(selected_models)}\n")
#    f.write("=" * 80 + "\n\n")
#    
#    for model_idx, model_name in enumerate(selected_models, 1):
#
#        # Skip large models
#        if any(x in model_name for x in ["70b", "34b", "32b", "falcon"]):
#            print(f"\n?? Skipping large model: {model_name}")
#            f.write(f"\n[SKIPPED] {model_name} (too large)\n")
#            continue
#
#        print(f"\n{'='*80}")
#        print(f"?? MODEL {model_idx}/{len(selected_models)}: {model_name}")
#        print(f"{'='*80}")
#
#        # Write model header to file
#        f.write("\n" + "=" * 80 + "\n")
#        f.write(f"MODEL: {model_name}\n")
#        f.write("=" * 80 + "\n")
#
#        # Build pipeline once per model
#        try:
#            rag_pipeline = build_rag_pipeline(model_name)
#        except Exception as e:
#            error_msg = f"Failed to build pipeline: {str(e)}"
#            print(f"? {error_msg}")
#            f.write(f"\nERROR: {error_msg}\n")
#            continue
#
#        # Process each question
#        for q_num, question in enumerate(QUESTIONS, 1):
#            print(f"\n{'-'*80}")
#            print(f"?? Question {q_num}/{len(QUESTIONS)}: {question}")
#            print(f"{'-'*80}")
#
#            try:
#                result = rag_pipeline.run(
#                    {
#                        "query_embedder": {"text": question},
#                        "prompt_builder": {"question": question},
#                    },
#                    include_outputs_from=["retriever"]
#                )
#                
#                # Get retrieved documents
#                docs = result["retriever"]["documents"]
#
#                print(f"   Retrieved: {len(docs)} documents")
#                
#                if len(docs) == 0:
#                    print("   ??  No documents retrieved")
#                else:
#                    print(f"   Top score: {docs[0].score:.4f}")
#                
#                # Show top 2 docs in console (abbreviated)
#                for idx, d in enumerate(docs[:2], 1):
#                    print(f"   Doc {idx}: {d.content[:80]}...")
#
#                # Get answer
#                if not docs:
#                    answer = "I don't know."
#                    print(f"   ? Answer: {answer}")
#                else:
#                    answer = result["llm"]["replies"][0]
#                    print(f"   ? Answer: {answer[:100]}...")
#
#                # Write to file (detailed)
#                f.write(f"\n{'-'*80}\n")
#                f.write(f"QUESTION {q_num}: {question}\n")
#                f.write(f"{'-'*80}\n")
#                f.write(f"Documents Retrieved: {len(docs)}\n")
#                
#                if docs:
#                    f.write(f"\nTop Retrieved Documents:\n")
#                    for idx, d in enumerate(docs, 1):
#                        f.write(f"\n  [{idx}] Score: {d.score:.4f}\n")
#                        f.write(f"      {d.content[:200]}...\n")
#                
#                f.write(f"\nANSWER:\n{answer.strip()}\n")
#                
#            except Exception as e:
#                error_msg = f"Error: {str(e)}"
#                print(f"   ? {error_msg}")
#                f.write(f"\n{'-'*80}\n")
#                f.write(f"QUESTION {q_num}: {question}\n")
#                f.write(f"{'-'*80}\n")
#                f.write(f"ERROR: {error_msg}\n")
#
#        f.write("\n")
#
#print(f"\n{'='*80}")
#print(f"? COMPLETE! All outputs saved to: {OUTPUT_FILE}")
#print(f"{'='*80}")






# -*- coding: utf-8 -*-
"""
IMPROVED RAG RETRIEVAL - PRODUCTION GRADE
- Better prompting
- Higher retrieval limits
- Context synthesis
- Relaxed constraints
"""

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from ollama_models import get_available_models, OllamaGenerator


# ==================================================
# 1. CONNECT TO QDRANT
# ==================================================

print("=" * 80)
print("IMPROVED RAG RETRIEVAL")
print("=" * 80)

# Ask which collection to use
print("\nAvailable collections:")
print("  1. rag_database_384_new (old)")
print("  2. rag_database_improved (new - recommended)")

choice = input("\nSelect collection (1 or 2, default=2): ").strip()
COLLECTION = "rag_database_384_new" if choice == "1" else "rag_database_ARAI"

print(f"\nConnecting to Qdrant collection: {COLLECTION}")
document_store = QdrantDocumentStore(
    url="http://localhost:7333",
    index=COLLECTION,
    embedding_dim=384,
    similarity="cosine",
    recreate_index=False,
    return_embedding=False,
)

doc_count = document_store.count_documents()
print(f"? Connected. Found {doc_count} documents")

if doc_count == 0:
    print("? ERROR: No documents found!")
    print("Please run ingestion_improved.py first.")
    exit(1)


# ==================================================
# 2. IMPROVED PROMPT TEMPLATES
# ==================================================

# Template 1: Relaxed (allows synthesis)
PROMPT_RELAXED = """
You are a helpful AI assistant answering questions based on provided context.

Context information:
{% for doc in documents %}
{{ doc.content }}

{% endfor %}

Guidelines:
- Answer based primarily on the context above
- If the exact answer isn't in the context, use the context to provide the best possible answer
- If you truly cannot answer from the context, say "I cannot find this information in the provided documents"
- Be concise but thorough
- You may synthesize information from multiple parts of the context

Question: {{ question }}

Answer:
"""

# Template 2: Moderate (balanced)
PROMPT_MODERATE = """
You are an AI assistant specialized in analyzing documents.

Context:
{% for doc in documents %}
{{ doc.content }}

{% endfor %}

Instructions:
- Answer the question using the context above
- If the answer requires combining information from multiple parts, do so
- If the context doesn't contain the answer, explain what information IS available
- Be specific and cite relevant parts when possible

Question: {{ question }}

Answer:
"""

# Template 3: Strict (original)
PROMPT_STRICT = """
You are a strict question-answering system.

RULES:
- Use ONLY the information inside <context>.
- If the answer is NOT explicitly stated, reply exactly: I don't know.
- Do NOT use prior knowledge.
- Do NOT infer or guess.

<context>
{% for doc in documents %}
{{ doc.content }}
{% endfor %}
</context>

Question: {{ question }}

Answer:
"""


# ==================================================
# 3. BUILD PIPELINE FUNCTION
# ==================================================

def build_rag_pipeline(model_name: str, prompt_style: str = "moderate"):
    """
    Build RAG pipeline with improved configuration
    
    Args:
        model_name: Name of Ollama model
        prompt_style: 'relaxed', 'moderate', or 'strict'
    """
    
    query_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # IMPROVED: More results, no threshold
    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=40,  # Increased from 5 to get more context
        # No score_threshold - get all top results
    )

    # Select prompt template
    if prompt_style == "relaxed":
        template = PROMPT_RELAXED
    elif prompt_style == "strict":
        template = PROMPT_STRICT
    else:
        template = PROMPT_MODERATE

    prompt_builder = PromptBuilder(
        template=template,
        required_variables=["documents", "question"],
    )

    # IMPROVED: Better generation parameters
    llm = OllamaGenerator(
        model=model_name,
        generation_kwargs={
            "temperature": 0.1,  # Slightly increased for better generation
            "num_predict": 512,  # Increased for longer, more detailed answers
            "top_p": 0.9,
            "top_k": 40,
        },
    )

    p = Pipeline()
    p.add_component("query_embedder", query_embedder)
    p.add_component("retriever", retriever)
    p.add_component("prompt_builder", prompt_builder)
    p.add_component("llm", llm)

    p.connect("query_embedder.embedding", "retriever.query_embedding")
    p.connect("retriever.documents", "prompt_builder.documents")
    p.connect("prompt_builder.prompt", "llm.prompt")

    return p


# ==================================================
# 4. CONFIGURATION
# ==================================================

# Select prompt style
print("\n" + "=" * 80)
print("SELECT PROMPT STYLE")
print("=" * 80)
print("  1. Relaxed - Allows synthesis and inference (recommended for production)")
print("  2. Moderate - Balanced approach")
print("  3. Strict - Only exact matches (may give many 'I don't know')")

style_choice = input("\nSelect style (1-3, default=1): ").strip()
PROMPT_STYLE = {
    "1": "relaxed",
    "2": "moderate",
    "3": "strict",
}.get(style_choice, "relaxed")

print(f"? Using {PROMPT_STYLE} prompt style")

# Get questions
print("\n" + "=" * 80)
print("ENTER QUESTIONS")
print("=" * 80)
print("Enter questions (one per line). Type 'done' when finished.")

questions = []
while True:
    q = input(f"Question {len(questions) + 1}: ").strip()
    if q.lower() == 'done':
        break
    if q:
        questions.append(q)

if not questions:
    questions = [
        "What is the name of the company?",
        "Who are the auditors?",
        "What are the main financial figures mentioned?",
    ]
    print(f"??  No questions entered. Using default questions.")

print(f"\n? Questions to ask: {len(questions)}")

# Select models
ollama_models = get_available_models()

print(f"\n?? Found {len(ollama_models)} Ollama models")
for i, m in enumerate(ollama_models, start=1):
    print(f"  {i}. {m}")

selection = input(
    "\nEnter model numbers (comma-separated), or press Enter for first model: "
).strip()

if not selection:
    selected_models = [ollama_models[0]] if ollama_models else []
else:
    indices = []
    for x in selection.split(","):
        x = x.strip()
        if x.isdigit():
            idx = int(x) - 1
            if 0 <= idx < len(ollama_models):
                indices.append(idx)
    selected_models = [ollama_models[i] for i in indices]

print(f"\n? Will use {len(selected_models)} model(s)")


# ==================================================
# 5. RUN QUERIES
# ==================================================

OUTPUT_FILE = "rag_outputs_10_Q.txt"

print(f"\n{'='*80}")
print("STARTING RAG PIPELINE")
print(f"{'='*80}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    # Write header
    f.write("IMPROVED RAG PIPELINE OUTPUT\n")
    f.write("=" * 80 + "\n")
    f.write(f"Collection: {COLLECTION}\n")
    f.write(f"Prompt Style: {PROMPT_STYLE}\n")
    f.write(f"Top-K: 10\n")
    f.write(f"Questions: {len(questions)}\n")
    f.write(f"Models: {len(selected_models)}\n")
    f.write("=" * 80 + "\n\n")
    
    for model_idx, model_name in enumerate(selected_models, 1):

        # Skip large models
#        if any(x in model_name for x in ["70b", "34b", "32b", "falcon"]):
#            print(f"\n?? Skipping large model: {model_name}")
#            continue

        print(f"\n{'='*80}")
        print(f"?? MODEL {model_idx}/{len(selected_models)}: {model_name}")
        print(f"{'='*80}")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"MODEL: {model_name}\n")
        f.write("=" * 80 + "\n")

        # Build pipeline
        try:
            rag_pipeline = build_rag_pipeline(model_name, PROMPT_STYLE)
        except Exception as e:
            error_msg = f"Failed to build pipeline: {str(e)}"
            print(f"? {error_msg}")
            f.write(f"\nERROR: {error_msg}\n")
            continue

        # Process each question
        for q_num, question in enumerate(questions, 1):
            print(f"\n{'-'*80}")
            print(f"?? Question {q_num}/{len(questions)}: {question}")
            print(f"{'-'*80}")

            try:
                result = rag_pipeline.run(
                    {
                        "query_embedder": {"text": question},
                        "prompt_builder": {"question": question},
                    },
                    include_outputs_from=["retriever"]
                )
                
                # Get retrieved documents
                docs = result["retriever"]["documents"]

                print(f"   ?? Retrieved: {len(docs)} documents")
                
                if docs:
                    print(f"   ?? Score range: {docs[0].score:.4f} - {docs[-1].score:.4f}")
                    
                    # Show snippet of best match
                    print(f"   ?? Best match: {docs[0].content[:100]}...")

                # Get answer
                answer = result["llm"]["replies"][0]
                
                # Check if it's a real answer or "don't know"
                is_dont_know = any(phrase in answer.lower() for phrase in [
                    "i don't know",
                    "i do not know",
                    "cannot find",
                    "not mentioned",
                    "no information"
                ])
                
                if is_dont_know:
                    print(f"   ??  Answer: {answer[:100]}...")
                else:
                    print(f"   ? Answer: {answer[:100]}...")

                # Write to file
                f.write(f"\n{'-'*80}\n")
                f.write(f"QUESTION {q_num}: {question}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Retrieved: {len(docs)} documents\n")
                
                if docs:
                    f.write(f"Score range: {docs[0].score:.4f} - {docs[-1].score:.4f}\n")
                    f.write(f"\nTop 3 Retrieved Chunks:\n")
                    for idx, d in enumerate(docs[:3], 1):
                        f.write(f"\n[{idx}] Score: {d.score:.4f}\n")
                        f.write(f"{d.content[:300]}...\n")
                
                f.write(f"\nANSWER:\n{answer.strip()}\n")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"   ? {error_msg}")
                f.write(f"\n{'-'*80}\n")
                f.write(f"QUESTION {q_num}: {question}\n")
                f.write(f"ERROR: {error_msg}\n")

print(f"\n{'='*80}")
print(f"? COMPLETE! Output saved to: {OUTPUT_FILE}")
print(f"{'='*80}")