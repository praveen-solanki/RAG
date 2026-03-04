import os
import requests
from haystack import Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

# ==================================================
# 0. BOSCH GEMINI 2.0 FLASH LITE (OpenAI-compatible)
# ==================================================
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# Correct Gemini endpoint - uses OpenAI-compatible format!
GEMINI_ENDPOINT = "https://aoai-farm.bosch-temp.com/api/openai/deployments/google-gemini-2-0-flash-lite/chat/completions"

@component
class BoschGeminiFlashLiteGenerator:
    """Generator for Bosch Gemini 2.0 Flash Lite (OpenAI-compatible API)"""
    
    def __init__(self, endpoint_url: str, api_key: str, temperature: float = 0.1, max_tokens: int = 256):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = "gemini-2.0-flash-lite"
    
    @component.output_types(replies=list)
    def run(self, prompt: str):
        headers = {
            "genaiplatform-farm-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # OpenAI-compatible format (same as GPT)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")
            
            result = response.json()
            # OpenAI-compatible response format
            text = result["choices"][0]["message"]["content"]
            return {"replies": [text]}
            
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")

# ==================================================
# 1. CONNECT TO QDRANT
# ==================================================
document_store = QdrantDocumentStore(
    url="http://localhost:7333",
    index="rag_database_384_ARAI",
    embedding_dim=384,
    similarity="cosine",
    recreate_index=False,
)

# ==================================================
# 2. RAG COMPONENTS
# ==================================================
query_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

retriever = QdrantEmbeddingRetriever(
    document_store=document_store,
    top_k=3,
)

prompt_builder = PromptBuilder(
    template="""
Answer the question using ONLY the information provided below.
If the answer is not present, say "I don't know."

Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:
""",
    required_variables=["documents", "question"],
)

llm = BoschGeminiFlashLiteGenerator(
    endpoint_url=GEMINI_ENDPOINT,
    api_key=API_KEY,
    temperature=0.1,
    max_tokens=256
)

# ==================================================
# 3. BUILD RAG PIPELINE
# ==================================================
rag_pipeline = Pipeline()
rag_pipeline.add_component("query_embedder", query_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# ==================================================
# 4. RUN QUERY
# ==================================================
if __name__ == "__main__":
    QUESTION = "What is the total value of 'Current Assets' for the year ending March 31, 2025? Compare this to the total 'Current Liabilities' for the same year. By what specific amount do current assets exceed current liabilities?"
    
    print("=" * 80)
    print("RAG Pipeline with Bosch Gemini 2.0 Flash Lite")
    print("=" * 80)
    print(f"Question: {QUESTION}\n")
    
    try:
        result = rag_pipeline.run(
            {
                "query_embedder": {"text": QUESTION},
                "prompt_builder": {"question": QUESTION},
            }
        )
        
        answer = result["llm"]["replies"][0]
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
    except Exception as e:
        print(f"\n? ERROR: {e}")