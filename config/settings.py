"""
Centralized configuration management for EvenementsRAG.

This module uses Pydantic Settings to manage all configuration with:
- Type safety and validation
- Environment variable loading from .env file
- Sensible defaults for development
- Easy access throughout the application

Usage:
    from config.settings import settings

    # Access configuration
    api_key = settings.ANTHROPIC_API_KEY
    top_k = settings.DEFAULT_TOP_K
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ========================================================================
    # Project Paths
    # ========================================================================
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Root directory of the project",
    )

    @property
    def DATA_DIR(self) -> Path:
        """Data directory path."""
        return self.PROJECT_ROOT / "data"

    @property
    def LOGS_DIR(self) -> Path:
        """Logs directory path."""
        return self.PROJECT_ROOT / "logs"

    @property
    def CACHE_DIR_PATH(self) -> Path:
        """Cache directory path."""
        return self.PROJECT_ROOT / self.CACHE_DIR

    # ========================================================================
    # LLM API Keys
    # ========================================================================
    ANTHROPIC_API_KEY: str = Field(
        default="",
        description="Anthropic API key for Claude models",
    )

    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key for GPT models",
    )

    OPENROUTER_API_KEY: str = Field(
        default="",
        description="OpenRouter API key for accessing various models",
    )

    # ========================================================================
    # Qdrant Vector Database
    # ========================================================================
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Qdrant server host",
    )

    QDRANT_PORT: int = Field(
        default=6333,
        description="Qdrant server port",
    )

    QDRANT_API_KEY: str = Field(
        default="",
        description="Qdrant API key (empty for local instance)",
    )

    QDRANT_COLLECTION_NAME: str = Field(
        default="ww2_historical_events",
        description="Name of the Qdrant collection",
    )

    QDRANT_URL: str = Field(
        default="",
        description="Qdrant Cloud URL (alternative to host/port)",
    )

    @property
    def QDRANT_LOCATION(self) -> str | tuple[str, int]:
        """Get Qdrant location (URL or host/port tuple)."""
        if self.QDRANT_URL:
            return self.QDRANT_URL
        return (self.QDRANT_HOST, self.QDRANT_PORT)

    # ========================================================================
    # Embedding Configuration
    # ========================================================================
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model identifier",
    )

    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="Embedding vector dimension",
    )

    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation",
    )

    # ========================================================================
    # LLM Configuration
    # ========================================================================
    LLM_PROVIDER: Literal["anthropic", "openai", "openrouter"] = Field(
        default="openrouter",
        description="LLM provider to use",
    )

    ANTHROPIC_MODEL: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model identifier",
    )

    OPENAI_MODEL: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model identifier",
    )

    OPENROUTER_MODEL: str = Field(
        default="mistralai/mistral-small-3.1-24b-instruct:free",
        description="OpenRouter model identifier",
    )

    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    LLM_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation",
    )

    LLM_MAX_TOKENS: int = Field(
        default=2000,
        ge=1,
        le=8000,
        description="Maximum tokens for LLM generation",
    )

    @property
    def CURRENT_LLM_MODEL(self) -> str:
        """Get the current LLM model based on provider."""
        if self.LLM_PROVIDER == "anthropic":
            return self.ANTHROPIC_MODEL
        elif self.LLM_PROVIDER == "openai":
            return self.OPENAI_MODEL
        else:  # openrouter
            return self.OPENROUTER_MODEL

    # ========================================================================
    # Graph Database (Phase 4)
    # ========================================================================
    GRAPH_BACKEND: Literal["networkx", "neo4j"] = Field(
        default="networkx",
        description="Graph database backend",
    )

    NEO4J_URI: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI",
    )

    NEO4J_USER: str = Field(
        default="neo4j",
        description="Neo4j username",
    )

    NEO4J_PASSWORD: str = Field(
        default="",
        description="Neo4j password",
    )

    # ========================================================================
    # Data Processing
    # ========================================================================
    CHUNK_SIZE: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Chunk size in tokens",
    )

    CHUNK_OVERLAP: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens",
    )

    MAX_CHUNKS_PER_DOC: int = Field(
        default=100,
        ge=1,
        description="Maximum chunks per document",
    )

    # Wikipedia settings
    WIKIPEDIA_LANGUAGE: str = Field(
        default="en",
        description="Wikipedia language code",
    )

    WIKIPEDIA_MAX_ARTICLES: int = Field(
        default=500,
        ge=1,
        description="Maximum Wikipedia articles to fetch",
    )

    WIKIPEDIA_MIN_ARTICLE_LENGTH: int = Field(
        default=500,
        ge=0,
        description="Minimum article length in words",
    )

    # ========================================================================
    # Retrieval Configuration
    # ========================================================================
    DEFAULT_TOP_K: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Default number of documents to retrieve",
    )

    RERANK_TOP_K: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of documents to retrieve before reranking",
    )

    # Hybrid search weights (Phase 3)
    SEMANTIC_WEIGHT: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in hybrid retrieval",
    )

    BM25_WEIGHT: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 search in hybrid retrieval",
    )

    @field_validator("BM25_WEIGHT")
    @classmethod
    def validate_hybrid_weights(cls, v: float, info) -> float:
        """Ensure semantic + BM25 weights sum to 1.0."""
        semantic_weight = info.data.get("SEMANTIC_WEIGHT", 0.7)
        if abs(semantic_weight + v - 1.0) > 0.01:
            raise ValueError(
                f"SEMANTIC_WEIGHT ({semantic_weight}) + BM25_WEIGHT ({v}) must sum to 1.0"
            )
        return v

    # ========================================================================
    # Evaluation
    # ========================================================================
    EVAL_QUESTIONS_PATH: str = Field(
        default="data/evaluation/test_questions.json",
        description="Path to evaluation questions",
    )

    EVAL_GROUND_TRUTH_PATH: str = Field(
        default="data/evaluation/ground_truth.json",
        description="Path to ground truth answers",
    )

    COMPUTE_RAGAS_METRICS: bool = Field(
        default=True,
        description="Whether to compute RAGAS metrics",
    )

    COMPUTE_BERT_SCORE: bool = Field(
        default=True,
        description="Whether to compute BERTScore",
    )

    # Automated question generation settings
    EVALUATION_NUM_SAMPLES: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of articles to sample for question generation",
    )

    EVALUATION_QUESTIONS_PER_ARTICLE: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Questions to generate per article",
    )

    EVALUATION_K_VALUES: list[int] = Field(
        default=[1, 3, 5, 10],
        description="K values for Recall@K metric",
    )

    EVALUATION_MIN_RECALL_AT_5: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable Recall@5 threshold",
    )

    # Question generation LLM settings
    QUESTION_GEN_MODEL: str = Field(
        default="mistralai/mistral-small-3.1-24b-instruct:free",
        description="Model for automated question generation",
    )

    QUESTION_GEN_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for question generation",
    )

    QUESTION_GEN_MAX_TOKENS: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Max tokens for question generation",
    )

    # ========================================================================
    # Logging
    # ========================================================================
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    LOG_FILE: str = Field(
        default="logs/evenementsrag.log",
        description="Log file path",
    )

    BENCHMARK_RESULTS_DIR: str = Field(
        default="results/benchmarks",
        description="Root directory for auto-saved benchmark result JSON files.",
    )

    # ========================================================================
    # Cache
    # ========================================================================
    CACHE_DIR: str = Field(
        default=".cache",
        description="Cache directory name",
    )

    ENABLE_CACHE: bool = Field(
        default=True,
        description="Whether to enable caching",
    )

    # ========================================================================
    # Development
    # ========================================================================
    DEBUG: bool = Field(
        default=False,
        description="Debug mode flag",
    )

    ENV: Literal["development", "production"] = Field(
        default="development",
        description="Environment name",
    )

    RANDOM_SEED: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    # ========================================================================
    # Methods
    # ========================================================================

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.DATA_DIR / "raw" / "wikipedia_articles",
            self.DATA_DIR / "raw" / "metadata",
            self.DATA_DIR / "processed" / "chunks",
            self.DATA_DIR / "processed" / "embeddings",
            self.DATA_DIR / "processed" / "entities",
            self.DATA_DIR / "graph" / "nodes",
            self.DATA_DIR / "graph" / "edges",
            self.DATA_DIR / "evaluation" / "benchmarks",
            self.LOGS_DIR,
            self.CACHE_DIR_PATH,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_api_keys(self) -> None:
        """Validate that required API keys are set."""
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set when using LLM_PROVIDER='anthropic'"
            )
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY must be set when using LLM_PROVIDER='openai'"
            )
        if self.LLM_PROVIDER == "openrouter" and not self.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY must be set when using LLM_PROVIDER='openrouter'"
            )

    def get_qdrant_config(self) -> dict:
        """Get Qdrant configuration as a dictionary."""
        config = {
            "location": self.QDRANT_LOCATION,
            "collection_name": self.QDRANT_COLLECTION_NAME,
        }
        if self.QDRANT_API_KEY:
            config["api_key"] = self.QDRANT_API_KEY
        return config

    def get_embedding_config(self) -> dict:
        """Get embedding configuration as a dictionary."""
        return {
            "model_name": self.EMBEDDING_MODEL,
            "dimension": self.EMBEDDING_DIMENSION,
            "batch_size": self.EMBEDDING_BATCH_SIZE,
        }

    def get_chunking_config(self) -> dict:
        """Get chunking configuration as a dictionary."""
        return {
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "max_chunks_per_doc": self.MAX_CHUNKS_PER_DOC,
        }

    def __repr__(self) -> str:
        """String representation hiding sensitive values."""
        return (
            f"Settings("
            f"ENV={self.ENV}, "
            f"LLM_PROVIDER={self.LLM_PROVIDER}, "
            f"QDRANT_HOST={self.QDRANT_HOST}:{self.QDRANT_PORT}, "
            f"EMBEDDING_MODEL={self.EMBEDDING_MODEL}"
            f")"
        )


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()


# Helper function to get settings
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


if __name__ == "__main__":
    # Print configuration when run directly
    print("EvenementsRAG Configuration")
    print("=" * 60)
    print(f"Environment: {settings.ENV}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"Current LLM Model: {settings.CURRENT_LLM_MODEL}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"Qdrant Location: {settings.QDRANT_LOCATION}")
    print(f"Qdrant Collection: {settings.QDRANT_COLLECTION_NAME}")
    print(f"Graph Backend: {settings.GRAPH_BACKEND}")
    print(f"Chunk Size: {settings.CHUNK_SIZE} tokens")
    print(f"Default Top-K: {settings.DEFAULT_TOP_K}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print("=" * 60)

    # Validate API keys
    try:
        settings.validate_api_keys()
        print("✓ API keys validated successfully")
    except ValueError as e:
        print(f"⚠ API key validation warning: {e}")
