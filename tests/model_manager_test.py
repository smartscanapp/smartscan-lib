from smartscan import LocalImageEmbeddingModel, LocalTextEmbeddingModel
from smartscan.models.model_manager import ModelManager

from pathlib import Path
from typing import get_args

class TestModelManager:
    TEXT_MODELS = get_args(LocalTextEmbeddingModel)
    IMAGE_MODELS = get_args(LocalImageEmbeddingModel)

    def test_model_manager_all_text_embedders(self):
        model_manager = ModelManager()

        for model_name in TestModelManager.TEXT_MODELS:
            self._run_model_round_trip(model_manager, model_name, True)
    
    def test_model_manager_all_image_embedders(self):
        model_manager = ModelManager()

        for model_name in TestModelManager.IMAGE_MODELS:
            self._run_model_round_trip(model_manager, model_name, False)

    @staticmethod
    def _run_model_round_trip(model_manager: ModelManager, model_name: str, is_text_model: bool):
        # Ensure model is removed
        model_manager.delete_model(model_name)
        assert model_manager.model_exists(model_name) is False

        # Download model
        path = model_manager.download_model(model_name)
        assert isinstance(path, Path)

        # Verify existence
        assert model_manager.model_exists(model_name) is True

        # Load and initialize
        if is_text_model:
            model = model_manager.get_text_embedder(model_name)
        else:
            model = model_manager.get_image_embedder(model_name)

        model.init()
