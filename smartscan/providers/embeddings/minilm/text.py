import numpy as np
from importlib import resources
from smartscan.providers import TextEmbeddingProvider
from smartscan.models.onnx_model import OnnxModel
from smartscan.providers.embeddings.minilm.tokenizer import load_minilm_tokenizer
from smartscan.errors import SmartScanError, ErrorCode


class MiniLmTextEmbedder(TextEmbeddingProvider):
    def __init__(self, model_path: str,  max_tokenizer_length: int):
        self._model = OnnxModel(model_path)
        self._embedding_dim = 384
        self._max_len = max_tokenizer_length
        with resources.path("smartscan.providers.embeddings.minilm", "vocab.txt") as vocab_path:
                self.tokenizer = load_minilm_tokenizer(str(vocab_path))
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def max_tokenizer_length(self) -> int:
        return self._max_len
        
    def embed(self, data: str) -> np.ndarray:
        if not self.is_initialized(): raise SmartScanError("Model not loaded", code=ErrorCode.MODEL_NOT_LOADED, details="Call init method first")
        input_name = self._model.get_inputs()[0].name
        token_ids = self._tokenize(data)
        attention_mask = [1 if id != 0 else 0 for id in token_ids]
        token_input = np.array([token_ids], dtype=np.int64)
        mask_input = np.array([attention_mask], dtype=np.int64)

        token_input = np.array([token_ids], dtype=np.int64)
        outputs = self._model.run({input_name: token_input, "attention_mask": mask_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[str])-> np.ndarray:
        if not self.is_initialized(): raise SmartScanError("Model not loaded", code=ErrorCode.MODEL_NOT_LOADED, details="Call init method first")
        input_names = self._model.get_inputs()
        token_ids_batch = [self._tokenize(item) for item in data]
        attention_mask_batch = [[1 if id != 0 else 0 for id in token_ids] for token_ids in token_ids_batch]

        token_inputs = np.array(token_ids_batch, dtype=np.int64)
        mask_inputs = np.array(attention_mask_batch, dtype=np.int64)

        outputs = self._model.run({input_names[0].name: token_inputs, input_names[1].name: mask_inputs})
        embeddings = outputs[0]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def close_session(self):
        self._model.close()

    def init(self):
        self._model.load()
    
    def is_initialized(self):
        return self._model.is_load()
    
    def _tokenize(self, text):
        token_ids = self.tokenizer.encode(text).ids
        return token_ids[:self.max_tokenizer_length] + [0] * (self.max_tokenizer_length - len(token_ids))