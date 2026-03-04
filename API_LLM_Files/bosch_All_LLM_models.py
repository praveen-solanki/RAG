import os
import requests
from haystack import Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

# ==================================================
# CONFIGURATION - SELECT MODELS TO RUN
# ==================================================
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# Available models configuration
MODELS = {
    "gpt-4o-mini": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-08-01-preview",
        "type": "openai",
        "model_name": "gpt-4o-mini"
    },
    "gemini-2.0-flash-lite": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/openai/deployments/google-gemini-2-0-flash-lite/chat/completions",
        "type": "openai",
        "model_name": "gemini-2.0-flash-lite"
    },
    "claude-haiku-4.5": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/google/v1/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
        "type": "anthropic",
        "model_name": "claude-haiku-4-5@20251001"
    },
    "claude-sonnet-4.5": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/google/v1/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict",
        "type": "anthropic",
        "model_name": "claude-sonnet-4-5@20250929"
    },
    "claude-3.5-haiku": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/google/v1/publishers/anthropic/models/claude-3-5-haiku@20241022:rawPredict",
        "type": "anthropic",
        "model_name": "claude-3-5-haiku@20241022"
    },
    "gpt-5": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-2025-08-07/chat/completions?api-version=2025-04-01-preview",
        "type": "openai",
        "model_name": "gpt-5-2025-08-07"
    }
}

# ==================================================
# SELECT MODELS TO RUN
# Choose one or more models from the list above
# Examples:
#   SELECTED_MODELS = ["gpt-4o-mini"]  # Run only one
#   SELECTED_MODELS = ["gpt-4o-mini", "gemini-2.0-flash-lite"]  # Run multiple
#   SELECTED_MODELS = list(MODELS.keys())  # Run all models
# ==================================================
SELECTED_MODELS = ["gpt-4o-mini"]  # ? CHANGE THIS TO SELECT MODELS

# ==================================================
# UNIFIED MODEL GENERATOR
# ==================================================
@component
class BoschUnifiedGenerator:
    """Unified generator supporting OpenAI and Anthropic formats"""
    
    def __init__(self, model_key: str, api_key: str, temperature: float = 0.1, max_tokens: int = 256):
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        
        self.model_key = model_key
        self.config = MODELS[model_key]
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @component.output_types(replies=list)
    def run(self, prompt: str):
        if self.config["type"] == "openai":
            return self._run_openai(prompt)
        elif self.config["type"] == "anthropic":
            return self._run_anthropic(prompt)
        else:
            raise ValueError(f"Unknown API type: {self.config['type']}")
    
    def _run_openai(self, prompt: str):
        """Handle OpenAI-compatible APIs (GPT, Gemini)"""
        headers = {
            "genaiplatform-farm-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                self.config["endpoint"],
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            return {"replies": [text]}
            
        except Exception as e:
            raise Exception(f"Failed to generate response from {self.model_key}: {str(e)}")
    
    def _run_anthropic(self, prompt: str):
        """Handle Anthropic Claude APIs"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                self.config["endpoint"],
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")
            
            result = response.json()
            text = result["content"][0]["text"]
            return {"replies": [text]}
            
        except Exception as e:
            raise Exception(f"Failed to generate response from {self.model_key}: {str(e)}")

# ==================================================
# BUILD RAG PIPELINE FOR A SPECIFIC MODEL
# ==================================================
def create_rag_pipeline(model_key: str):
    """Create a RAG pipeline for the specified model"""
    
    # Connect to Qdrant
    document_store = QdrantDocumentStore(
        url="http://localhost:7333",
        index="rag_database_384_ARAI",
        embedding_dim=384,
        similarity="cosine",
        recreate_index=False,
    )
    
    # RAG components
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
    
    # Create model-specific generator
    llm = BoschUnifiedGenerator(
        model_key=model_key,
        api_key=API_KEY,
        temperature=0.1,
        max_tokens=256
    )
    
    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_component("query_embedder", query_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)
    
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    
    return pipeline

# ==================================================
# RUN RAG WITH SELECTED MODELS
# ==================================================
if __name__ == "__main__":
    QUESTION = "Explain Cultural Control in detail."
    
    print("=" * 80)
    print("MULTI-MODEL RAG PIPELINE")
    print("=" * 80)
    print(f"Question: {QUESTION}")
    print(f"Selected Models: {SELECTED_MODELS}")
    print("=" * 80)
    
    results = {}
    
    for model_key in SELECTED_MODELS:
        print(f"\n{'=' * 80}")
        print(f"Running with: {model_key}")
        print("=" * 80)
        
        try:
            # Create pipeline for this model
            pipeline = create_rag_pipeline(model_key)
            
            # Run query
            result = pipeline.run({
                "query_embedder": {"text": QUESTION},
                "prompt_builder": {"question": QUESTION},
            })
            
            answer = result["llm"]["replies"][0]
            results[model_key] = answer
            
            print(f"\n? SUCCESS with {model_key}")
            print(f"\nANSWER:\n{answer}")
            
        except Exception as e:
            print(f"\n? FAILED with {model_key}")
            print(f"Error: {e}")
            results[model_key] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL RESULTS")
    print("=" * 80)
    
    for model_key, answer in results.items():
        print(f"\n{model_key}:")
        print("-" * 80)
        if answer.startswith("ERROR:"):
            print(f"? {answer}")
        else:
            print(f"? {answer[:200]}..." if len(answer) > 200 else f"? {answer}")