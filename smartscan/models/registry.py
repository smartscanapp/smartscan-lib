from typing import Dict
from smartscan.models.types import ModelInfo, ModelName

MODEL_REGISTRY: Dict[ModelName, ModelInfo] = {
    'all-minilm-l6-v2': ModelInfo(
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.1/minilm_sentence_transformer_quant.zip", 
        model_path='all_minilm_l6_v2',
        resource_files=["minilm_sentence_transformer_quant.onnx", "vocab.txt"]
        ),
    'all-distilroberta-v1': ModelInfo(
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.1/all_distilroberta_v1_quant.zip", 
        model_path='all_distilroberta_v1',
        resource_files=["sentence-transformers_all-distilroberta-v1_quant.onnx", "vocab.json", "merges.txt"]
        ),
    'clip-vit-b-32-text': ModelInfo(
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.1/clip_text_encoder_quant.zip", 
        model_path='clip_vit_b_32_text',
        resource_files=["clip_text_encoder_quant.onnx", "vocab.json", "merges.txt"]
        ),
    'clip-vit-b-32-image': ModelInfo(
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/clip_image_encoder_quant.onnx", 
        model_path="clip_image_encoder_quant.onnx"
        ),
    'dinov2-small': ModelInfo(
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/dinov2_small_quant.onnx", 
        model_path="dinov2_small_quant.onnx"
    ),
}
