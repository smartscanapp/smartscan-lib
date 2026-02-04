import os
from typing import Dict
from smartscan.models.types import ModelName, ModelInfo
from pathlib import Path


LOCAL_BASE_DIR = Path.home() / ".cache" / "smartscan"
BASE_DIR = Path(os.environ.get("SMARTSCAN_BASE_DIR", LOCAL_BASE_DIR))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")

MINILM_MAX_TOKENS = 512
MINILM_MODEL_PATH = 'minilm_sentence_transformer_quant.onnx'
MINILM_MODEL_URL = "https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/minilm_sentence_transformer_quant.onnx"

CLIP_IMAGE_MODEL_URL = "https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/clip_image_encoder_quant.onnx"
CLIP_IMAGE_MODEL_PATH = "clip_image_encoder_quant.onnx"

CLIP_TEXT_MODEL_URL = "https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/clip_text_encoder_quant.onnx"
CLIP_TEXT_MODEL_PATH = "clip_text_encoder_quant.onnx"

DINO_SMALL_MODEL_URL = "https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/dinov2_small_quant.onnx"
DINO_SMALL_MODEL_PATH = "dinov2_small_quant.onnx"

INCEPTION_RESNET_ZIP_URL = "https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/facial_recognition_inception_resnet_v1.zip"
INCEPTION_RESNET_ZIP_PATH = "inception_resnet_v1.zip"
ULTRA_FACE_DETECTION_MODEL_PATH = "ultra_face_detector_model.onnx"
INCEPTION_RESNET_MODEL_PATH = "inception_resnet_v1.onnx"

MODEL_REGISTRY: Dict[ModelName, ModelInfo] = {
    'all-minilm-l6-v2': ModelInfo(url=MINILM_MODEL_URL, path=MINILM_MODEL_PATH),
    'clip-vit-b-32-image': ModelInfo(url=CLIP_IMAGE_MODEL_URL, path=CLIP_IMAGE_MODEL_PATH),
    'clip-vit-b-32-text': ModelInfo(url=CLIP_TEXT_MODEL_URL, path=CLIP_TEXT_MODEL_PATH),
    'dinov2-small': ModelInfo(url=DINO_SMALL_MODEL_URL, path=DINO_SMALL_MODEL_PATH),
}
