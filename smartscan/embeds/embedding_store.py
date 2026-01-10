from abc import ABC, abstractmethod
from typing import Generic, List, Optional
from smartscan.embeds.types import TData, TMetadata , FilterType, ItemEmbedding, GetResult, QueryResult, Include, ItemEmbeddingUpdate
import numpy as np



class EmbeddingStore(ABC, Generic[TData, TMetadata]):
    """Generic embedding store interface using ItemEmbedding objects."""

    @abstractmethod
    def add(self, items: List[ItemEmbedding[TData, TMetadata]]) -> None:
        """Add embeddings with optional data and metadata."""
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[FilterType] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Include = ["metadatas"],
    ) -> GetResult:
        """Retrieve embeddings and their data/metadata by IDs or filter."""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embeds: List[np.ndarray],
        filter: Optional[FilterType] = None,
        limit: int = 10,
        include: Include = ["metadatas"],
    ) -> QueryResult:
        """Return nearest neighbors for given query embeddings with optional filters."""
        raise NotImplementedError

    @abstractmethod
    def update(self, items: List[ItemEmbeddingUpdate[TData, TMetadata]]) -> None:
        """Update embeddings, data, or metadata using ItemEmbedding objects."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[FilterType] = None) -> None:
        """Delete embeddings and associated data/metadata by IDs or filter."""
        raise NotImplementedError

    @abstractmethod
    def count(self, filter: Optional[FilterType] = None) -> int:
        """Return the total number of embeddings, optionally filtered."""
        raise NotImplementedError
