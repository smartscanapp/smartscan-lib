import numpy as np
from PIL import Image
from smartscan.providers import ImageEmbeddingProvider
from smartscan.models.onnx_model import OnnxModel
from smartscan.errors import SmartScanError, ErrorCode


class DinoSmallV2ImageEmbedder(ImageEmbeddingProvider):
    def __init__(self, model_path: str):
        self._model = OnnxModel(model_path)

    @property
    def embedding_dim(self) -> int:
        return 384

    def embed(self, data: Image.Image)-> np.ndarray:
        """Create vector embeddings for text or image files using an ONNX model."""

        if not self.is_initialized(): raise SmartScanError("Model not loaded", code=ErrorCode.MODEL_NOT_LOADED, details="Call init method first")
        input_name = self._model.get_inputs()[0].name
        image_input = self._preprocess(data)
        outputs = self._model.run({input_name: image_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[Image.Image])-> np.ndarray:
        """Create vector embeddings for text or image files using an ONNX model."""

        if not self.is_initialized(): raise SmartScanError("Model not loaded", code=ErrorCode.MODEL_NOT_LOADED, details="Call init method first")
        input_name = self._model.get_inputs()[0].name
        images = [self._preprocess(item) for item in data]
        image_inputs = np.concatenate(images, axis=0)
        outputs = self._model.run({input_name: image_inputs})
        embeddings = outputs[0]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def close_session(self):
        self._model.close()

    def init(self):
        self._model.load()
    
    def is_initialized(self):
        return self._model.is_load()
    
    @staticmethod
    def _preprocess(image: Image.Image) -> np.ndarray:
        MODE = 'RGB'
        SIZE = 224
        RESAMPLE=Image.BICUBIC
        MEAN=(0.485, 0.456, 0.406)
        STD=(0.229, 0.224, 0.225)
        
        image = image.convert(MODE)

        # Resize shorter side to output_size / crop_pct (224/0.875 = 256)
        crop_pct = SIZE / (SIZE / 0.875) 
        scale_size = int(SIZE / crop_pct + 0.5) 

        w, h = image.size
        if h < w:
            new_h = scale_size
            new_w = int(w * (scale_size / h))
        else:
            new_w = scale_size
            new_h = int(h * (scale_size / w))
        image = image.resize((new_w, new_h), resample=RESAMPLE)

        left = (new_w - SIZE) // 2
        top  = (new_h - SIZE) // 2
        image = image.crop((left, top, left + SIZE, top + SIZE))

        arr = np.asarray(image, dtype=np.float32) / 255.0

        mean = np.array(MEAN, dtype=np.float32)
        std  = np.array(STD,  dtype=np.float32)
        arr = (arr - mean) / std

        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)
        return arr.astype(dtype=np.float32)