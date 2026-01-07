from numpy import ndarray
from typing import Literal, Generic, TypeVar, Optional, Any, Dict, Union, Callable
from dataclasses import dataclass


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

@dataclass
class GetResult(ItemEmbedding[TData, TMetadata]):
    embedding: Optional[ndarray] = None

@dataclass
class QueryResult(GetResult[TData, TMetadata]):
    sim: Optional[float] = None


EncoderType = Literal["image_encoder", "text_encoder", "face_encoder"]
