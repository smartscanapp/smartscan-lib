import numpy as np
from PIL import Image
from smartscan.providers import  ImageEmbeddingProvider
from smartscan.models.onnx_model import OnnxModel
from smartscan.errors import SmartScanError, ErrorCode


class ClipImageEmbedder(ImageEmbeddingProvider):
    def __init__(self, model_path: str):
        self._model = OnnxModel(model_path)
        self._embedding_dim = 512

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, data: Image.Image)-> np.ndarray:
        if not self.is_initialized(): raise SmartScanError("Model not loaded", code=ErrorCode.MODEL_NOT_LOADED, details="Call init method first")
        input_name = self._model.get_inputs()[0].name
        image_input = self._preprocess(data)
        outputs = self._model.run({input_name: image_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[Image.Image])-> np.ndarray:
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
    def _preprocess(image: Image.Image):
        SIZE = 224
        MODE = 'RGB'
        MEAN = (0.48145466, 0.4578275, 0.40821073)
        STD = (0.26862954, 0.26130258, 0.27577711)
        INTERPOLATION = Image.BICUBIC

        image = image.convert(MODE)
        
        # Resize based on the shortest edge
        w, h = image.size
        scale = SIZE / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        image = image.resize((new_w, new_h), INTERPOLATION)
        
        left = (new_w - SIZE) // 2
        top = (new_h - SIZE) // 2
        image = image.crop((left, top, left + SIZE, top + SIZE))
        
        img_array = np.array(image).astype(np.float32) / 255.0
        
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # channel-first normalization tensors so normalization must happen after transposing
        mean = np.array(MEAN).reshape(3, 1, 1)
        std = np.array(STD).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        return img_array.astype(dtype=np.float32)