from typing import Literal
from smartscan.embeds.types import *
from smartscan.classify.types import *
from smartscan.media.types import VideoSource, ImageSource

ModelName = Literal[
    "clip-vit-b-32-image",
    "clip-vit-b-32-text",
    "dinov2-small",
    "inception-resnet-v1",
    "all-minilm-l6-v2",
]