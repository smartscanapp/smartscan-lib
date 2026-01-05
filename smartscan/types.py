from numpy import ndarray
from typing import Literal, TypeVar
from dataclasses import dataclass

@dataclass
class ItemEmbedding:
    item_id: str
    embedding: ndarray

ImageSource = str | ndarray

VideoSource = str | ndarray

EncoderType = Literal["image_encoder", "text_encoder", "face_encoder"]

ModelName = Literal[
    "clip-vit-b-32-image",
    "clip-vit-b-32-text",
    "dinov2-small",
    "inception-resnet-v1",
    "all-minilm-l6-v2",
]

Input = TypeVar("Input")
Output = TypeVar("Output")