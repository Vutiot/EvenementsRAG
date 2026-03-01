"""
src.vector_store — public API for vector store backends.

Exposes:
    BaseVectorStore    (E2-F2-T1)
    DistanceMetric     (E2-F2-T1)
    VectorStoreFactory (E2-F2-T1)
"""

from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.vector_store.factory import VectorStoreFactory

__all__ = [
    "BaseVectorStore",
    "DistanceMetric",
    "VectorStoreFactory",
]
