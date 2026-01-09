from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from PIL import Image
import numpy as np

T = TypeVar("T")

class EmbeddingProvider(ABC, Generic[T]):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def embed(self, data: T) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def embed_batch(self, data: list[T]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def init(self):
        raise NotImplementedError

    @abstractmethod
    def is_initialized(self) -> bool: 
        raise NotImplementedError
    
    @abstractmethod
    def close_session(self):
        raise NotImplementedError


class TextEmbeddingProviderBase(EmbeddingProvider[str], ABC):
    @property
    @abstractmethod
    def max_tokenizer_length(self) -> int:
        raise NotImplementedError


ImageEmbeddingProvider = EmbeddingProvider[Image.Image]
TextEmbeddingProvider = TextEmbeddingProviderBase