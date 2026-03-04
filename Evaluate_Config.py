"""
CONFIGURATION FILE
==================
Central configuration for the advanced RAG system
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    use_ollama: bool = True
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "bge-m3"
    fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 1024  # BGE-M3: 1024, MiniLM: 384


@dataclass
class ChunkingConfig:
    """Chunking strategy configuration"""
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    max_chunk_size: int = 1024
    enable_section_aware: bool = True
    respect_sentence_boundaries: bool = True
    
    # Section detection patterns
    section_patterns: List[str] = None
    
    def __post_init__(self):
        if self.section_patterns is None:
            self.section_patterns = [
                r'^#{1,6}\s+(.+)$',  # Markdown headers
                r'^([A-Z][^.!?]*):$',  # Title case with colon
                r'^\d+\.\s+([A-Z].+)$',  # Numbered sections
                r'^([A-Z\s]{3,})$',  # All caps headers (min 3 chars)
            ]


@dataclass
class RetrievalConfig:
    """Retrieval strategy configuration"""
    # Search parameters
    dense_top_k: int = 50
    sparse_top_k: int = 50
    hybrid_top_k: int = 30
    final_top_k: int = 10
    
    # Fusion weights
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    
    # Reranking
    use_cross_encoder: bool = True
    use_ollama_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Query expansion
    enable_query_expansion: bool = True
    expansion_synonyms: Dict[str, List[str]] = None
    
    # Caching
    enable_cache: bool = True
    cache_size: int = 1000
    
    def __post_init__(self):
        if self.expansion_synonyms is None:
            self.expansion_synonyms = {
                "ai": ["artificial intelligence", "machine learning", "deep learning"],
                "ml": ["machine learning", "ai", "artificial intelligence"],
                "llm": ["large language model", "language model", "transformer"],
                "nlp": ["natural language processing", "language understanding"],
                "cv": ["computer vision", "image recognition"],
                "dl": ["deep learning", "neural networks"],
            }


@dataclass
class QdrantConfig:
    """Qdrant database configuration"""
    url: str = "http://localhost:7333"
    collection_name: str = "rag_hybrid_bge_m3"
    enable_sparse_vectors: bool = True
    sparse_vector_name: str = "bm25"


@dataclass
class IngestionConfig:
    """Document ingestion configuration"""
    data_dir: str = "DATASET"
    supported_extensions: List[str] = None
    batch_size: int = 100
    skip_already_indexed: bool = True
    
    # Metadata extraction
    extract_metadata: bool = True
    extract_tables: bool = True
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [".pdf", ".docx", ".txt", ".md", ".html"]


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    k_values: List[int] = None
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]
        
        if self.metrics is None:
            self.metrics = [
                "precision",
                "recall",
                "mrr",
                "ndcg",
                "map",
            ]


class SystemConfig:
    """Main system configuration"""
    
    def __init__(self):
        self.embedding = EmbeddingConfig()
        self.chunking = ChunkingConfig()
        self.retrieval = RetrievalConfig()
        self.qdrant = QdrantConfig()
        self.ingestion = IngestionConfig()
        self.evaluation = EvaluationConfig()
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "embedding": self.embedding.__dict__,
            "chunking": {k: v for k, v in self.chunking.__dict__.items()},
            "retrieval": {k: v for k, v in self.retrieval.__dict__.items()},
            "qdrant": self.qdrant.__dict__,
            "ingestion": {k: v for k, v in self.ingestion.__dict__.items()},
            "evaluation": {k: v for k, v in self.evaluation.__dict__.items()},
        }
    
    def summary(self) -> str:
        """Get configuration summary"""
        lines = [
            "="*80,
            "SYSTEM CONFIGURATION SUMMARY",
            "="*80,
            "",
            "EMBEDDING:",
            f"  Model: {'Ollama BGE-M3' if self.embedding.use_ollama else self.embedding.fallback_model}",
            f"  Dimension: {self.embedding.embedding_dimension}",
            "",
            "CHUNKING:",
            f"  Size: {self.chunking.chunk_size} (overlap: {self.chunking.chunk_overlap})",
            f"  Section-aware: {self.chunking.enable_section_aware}",
            "",
            "RETRIEVAL:",
            f"  Strategy: Hybrid (dense: {self.retrieval.dense_weight}, sparse: {self.retrieval.sparse_weight})",
            f"  Reranking: {self.retrieval.use_cross_encoder}",
            f"  Query expansion: {self.retrieval.enable_query_expansion}",
            f"  Cache: {self.retrieval.enable_cache} (size: {self.retrieval.cache_size})",
            "",
            "QDRANT:",
            f"  Collection: {self.qdrant.collection_name}",
            f"  URL: {self.qdrant.url}",
            f"  Sparse vectors: {self.qdrant.enable_sparse_vectors}",
            "",
            "="*80,
        ]
        return "\n".join(lines)


# Default configuration instance
default_config = SystemConfig()


if __name__ == "__main__":
    config = SystemConfig()
    print(config.summary())
    
    # Save to file
    import json
    with open("system_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print("\n✓ Configuration saved to system_config.json")