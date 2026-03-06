from typing import Literal, TypeAlias, TypedDict, Optional, List

LocalTextEmbeddingModel: TypeAlias = Literal["all-minilm-l6-v2", "clip-vit-b-32-text", "all-distilroberta-v1"]
LocalImageEmbeddingModel: TypeAlias = Literal["clip-vit-b-32-image", "dinov2-small"]
LocalFaceEmbeddingModel: TypeAlias = Literal["inception-resnet-v1"]

ModelName = Literal[LocalTextEmbeddingModel,LocalImageEmbeddingModel, LocalFaceEmbeddingModel]

class ModelInfo(TypedDict):
    url: str
    model_path: str             
    resource_files: Optional[List[str]]
    file_hash: Optional[str] = None
