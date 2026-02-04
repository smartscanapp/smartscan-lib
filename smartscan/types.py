from typing import Literal, TypeAlias, TypedDict, Optional, List
from smartscan.embeds.types import *
from smartscan.classify.types import *
from smartscan.media.types import VideoSource, ImageSource
from smartscan.providers import TextEmbeddingProvider, ImageEmbeddingProvider, EmbeddingProvider


LocalTextEmbeddingModel: TypeAlias = Literal["all-minilm-l6-v2", "clip-vit-b-32-text"]
LocalImageEmbeddingModel: TypeAlias = Literal["clip-vit-b-32-image", "dinov2-small"]
LocalFaceEmbeddingModel: TypeAlias = Literal["inception-resnet-v1"]

ModelName = Literal[LocalTextEmbeddingModel,LocalImageEmbeddingModel, LocalFaceEmbeddingModel]

class ModelInfo(TypedDict):
    url: str
    path: str
    dependencies_paths: Optional[List[str]] = None
    file_hash: Optional[str] = None