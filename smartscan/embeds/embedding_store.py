from abc import ABC, abstractmethod
from typing import List, Optional
from smartscan.embeds.types import QueryResult, StoredEmbedding
from numpy.typing import NDArray


class EmbeddingStore(ABC):
    @abstractmethod
    async def add(self, items: List[StoredEmbedding]) -> None: ...
    
    @abstractmethod
    async def get(self, ids: Optional[List[str]] = None) -> List[StoredEmbedding]: ...
    
    @abstractmethod
    async def query(self, query_embed: NDArray, topK: int, ids: Optional[List[str]] = None, include_sims: bool = False, threshold: Optional[float] = None) -> QueryResult: ...
    
    @abstractmethod
    async def update(self, items: List[StoredEmbedding]) -> None: ...

    @abstractmethod
    async def upsert(self, items: List[StoredEmbedding]) -> None: ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def clear(self) -> None: ...