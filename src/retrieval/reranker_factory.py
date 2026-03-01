"""
RerankerFactory — registry-based factory for reranker implementations.

Pattern mirrors VectorStoreFactory: a string registry maps type names to
dotted class paths, and from_config() / create() instantiate them lazily.

Usage:
    from src.retrieval.reranker_factory import RerankerFactory

    reranker = RerankerFactory.create("none")
    bge = RerankerFactory.create("bge", model_name="BAAI/bge-reranker-v2-m3")
"""

import importlib
from typing import Any

from src.retrieval.reranker import BaseReranker

_RERANKER_REGISTRY: dict[str, str] = {
    "none": "src.retrieval.reranker.NoOpReranker",
    "cohere": "src.retrieval.reranker.CohereReranker",
    "bge": "src.retrieval.reranker.BGEReranker",
    "cross_encoder": "src.retrieval.reranker.CrossEncoderReranker",
}


class RerankerFactory:
    """Factory for creating reranker instances from a type name or config."""

    @staticmethod
    def from_config(reranker_config) -> BaseReranker:
        """Create a reranker from a ``RerankerConfig`` pydantic model.

        Args:
            reranker_config: A ``RerankerConfig`` with ``type`` and optional
                             ``model_name`` fields.

        Returns:
            A concrete ``BaseReranker`` instance.
        """
        kwargs: dict[str, Any] = {}
        if reranker_config.model_name:
            kwargs["model_name"] = reranker_config.model_name
        return RerankerFactory.create(reranker_config.type, **kwargs)

    @staticmethod
    def create(type_: str, **kwargs: Any) -> BaseReranker:
        """Create a reranker by type name.

        Args:
            type_: One of ``_RERANKER_REGISTRY`` keys
                   (``"none"``, ``"cohere"``, ``"bge"``, ``"cross_encoder"``).
            **kwargs: Constructor kwargs forwarded to the reranker class
                      (e.g. ``model_name="BAAI/bge-reranker-v2-m3"``).

        Returns:
            A concrete ``BaseReranker`` instance.

        Raises:
            ValueError: If *type_* is not in the registry.
        """
        target = _RERANKER_REGISTRY.get(type_)
        if target is None:
            raise ValueError(
                f"Unknown reranker type '{type_}'. "
                f"Available: {sorted(_RERANKER_REGISTRY)}"
            )
        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    @staticmethod
    def available_types() -> list[str]:
        """Return the list of registered reranker type names."""
        return sorted(_RERANKER_REGISTRY)
