from numpy import ndarray
from typing import Literal, Generic, TypeVar, Optional, Any, Dict, Union, Callable, List
from dataclasses import dataclass, field


TData = TypeVar("TData", bound=Any)
TMetadata = TypeVar("TMetadata", bound=Dict)

FilterType = Union[Dict[str, Any], Callable[[Any], bool]] 

__all__ = [
    "ItemEmbedding",
    "GetResult",
    "QueryResult",
    "EncoderType",
    "FilterType",
]

@dataclass
class ItemEmbedding(Generic[TData, TMetadata]):
    item_id: str
    embedding: ndarray
    data: Optional[TData] = None
    metadata: Optional[TMetadata] = None

@dataclass(frozen=True)
class GetResult(Generic[TData, TMetadata]):
    ids: List[str] = field(default_factory=list)
    embeddings: List[ndarray] = field(default_factory=list)
    metadatas: List[Optional[TMetadata]] = field(default_factory=list)
    datas: List[Optional[TData]] = field(default_factory=list)

@dataclass(frozen=True)
class QueryResult(GetResult[TData, TMetadata]):
    sims: List[float] = field(default_factory=list)

EncoderType = Literal["image_encoder", "text_encoder", "face_encoder"]
