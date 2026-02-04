import shutil
import tempfile
import urllib.request
from pathlib import Path

from smartscan import TextEmbeddingProvider, ImageEmbeddingProvider
from smartscan.providers import  MiniLmTextEmbedder, ClipTextEmbedder, ClipImageEmbedder, DinoSmallV2ImageEmbedder
from smartscan.types import LocalTextEmbeddingModel, LocalImageEmbeddingModel, ModelName
from smartscan.errors import SmartScanError, ErrorCode
from smartscan.constants import DEFAULT_MODEL_DIR, MODEL_REGISTRY, MINILM_MAX_TOKENS

class ModelManager:
    def __init__(self, root_dir: str = DEFAULT_MODEL_DIR):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, name: ModelName, timeout: int = 30) -> Path:
        """
        Download a file from `url` into the manager's root_dir.
        - Writes to a temp file and atomically moves into place.
        - Returns the Path to the downloaded file.
        """

        target = self.get_model_path(name)
        if not str(target).startswith(str(self.root_dir)):
            raise SmartScanError("Target path is outside the configured root_dir", code=ErrorCode.INVALID_MODEL_PATH)

        # Create parent directories if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        # Stream download to a temp file then move atomically
        with tempfile.NamedTemporaryFile(delete=False, dir=str(self.root_dir)) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(self.get_model_download_url(name), timeout=timeout) as resp:
                chunk_size = 64 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    tmp.write(chunk)

        tmp_path.replace(target)
        return target

    def delete_model(self, name: ModelName) -> None:
        """
        Delete a file or directory at `path`. `path` may be an absolute path or
        relative to the manager's root_dir. Safety: disallow deletion outside root_dir.
        """
        path = self.get_model_path(name)
        if not str(path).startswith(str(self.root_dir)):
            raise SmartScanError("Cannot delete file outside of root_dir", code=ErrorCode.INVALID_MODEL_PATH)

        if not path.exists():
            return

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def model_exists(self, name: ModelName) -> bool:
        """
        Return True if a file or directory exists at `path`.
        `path` may be absolute or relative to root_dir.
        """
        resolved = self.get_model_path(name)
        return resolved.exists()
    
    def get_model_path(self, name: ModelName) -> Path:
        model_info = MODEL_REGISTRY[name]
        return (self.root_dir / model_info['path']).resolve() if not Path(model_info['path']).is_absolute() else Path(model_info["path"]).resolve()

    def get_model_download_url(self, name: ModelName) -> str:
        return MODEL_REGISTRY[name]['url']


    def get_text_embedder(self,model: LocalTextEmbeddingModel) -> TextEmbeddingProvider:
        if model == "clip-vit-b-32-text":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
                return ClipTextEmbedder(path)
            path = self.get_model_path(model)
            return ClipTextEmbedder(path)

        elif model == "all-minilm-l6-v2":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
                return MiniLmTextEmbedder(path, MINILM_MAX_TOKENS)
            path = self.get_model_path(model)
            return MiniLmTextEmbedder(path, MINILM_MAX_TOKENS)
        else:
            raise SmartScanError("Model not supported", code=ErrorCode.UNSUPPORTED_MODEL)
        
    def get_image_embedder(self,model: LocalImageEmbeddingModel) -> ImageEmbeddingProvider:
        if model == "clip-vit-b-32-image":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
                return ClipImageEmbedder(path)
            path = self.get_model_path(model)
            return ClipImageEmbedder(path)

        elif model == "dinov2-small":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
                return DinoSmallV2ImageEmbedder(path)
            path = self.get_model_path(model)
            return DinoSmallV2ImageEmbedder(path)
        else:
            raise SmartScanError("Model not supported", code=ErrorCode.UNSUPPORTED_MODEL)

