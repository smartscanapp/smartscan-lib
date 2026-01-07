from numpy import ndarray
from typing import Literal, TypeVar
from dataclasses import dataclass
from smartscan.cluster.types import *
from smartscan.embeds.types import *


@dataclass
class ClassificationResult:
    item_id: str
    label: str | None = None
    similarity: float = 0.0


ImageSource = str | ndarray

VideoSource = str | ndarray


ModelName = Literal[
    "clip-vit-b-32-image",
    "clip-vit-b-32-text",
    "dinov2-small",
    "inception-resnet-v1",
    "all-minilm-l6-v2",
]