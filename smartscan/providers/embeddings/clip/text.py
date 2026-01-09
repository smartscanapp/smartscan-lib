import numpy as np
from smartscan.providers import TextEmbeddingProvider
from smartscan.models.onnx_model import OnnxModel
from smartscan.providers.embeddings.clip.tokenizer import load_clip_tokenizer
from importlib import resources
from smartscan.errors import SmartScanError, ErrorCode

class ClipTextEmbedder(TextEmbeddingProvider):
    def __init__(self, model_path: str):
        self._model = OnnxModel(model_path)
        self._embedding_dim = 512
        self._max_len = 77
        with resources.path("smartscan.providers.embeddings.clip", "vocab.json") as vocab_path, \
             resources.path("smartscan.providers.embeddings.clip", "merges.txt") as merges_path:
            self.tokenizer = load_clip_tokenizer(str(vocab_path), str(merges_path))

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def max_tokenizer_length(self) -> int:
        return self._max_len

    def embed(self, data: str)-> np.ndarray:
        if not self.is_initialized(): raise SmartScanError("Model not loaded", code=ErrorCode.MODEL_NOT_LOADED, details="Call init method first")                
        input_name = self._model.get_inputs()[0].name
        token_ids = self._tokenize(data)
        token_input = np.array([token_ids], dtype=np.int64)
        outputs = self._model.run({input_name: token_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[str])-> np.ndarray:
        if not self.is_initialized(): raise SmartScanError("Model not loaded", code=ErrorCode.MODEL_NOT_LOADED, details="Call init method first")                
        
        input_name = self._model.get_inputs()[0].name
        token_ids_batch = [self._tokenize(item) for item in data]
        token_inputs = np.array(token_ids_batch, dtype=np.int64)
        outputs = self._model.run({input_name: token_inputs})
        embeddings = outputs[0]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def close_session(self):
        self._model.close()

    def init(self):
        self._model.load()
    
    def is_initialized(self):
        return self._model.is_load()
    
    def _tokenize(self, text: str):
        token_ids = self.tokenizer.encode(text).ids
        return token_ids[:self._max_len] + [0] * (self._max_len - len(token_ids))