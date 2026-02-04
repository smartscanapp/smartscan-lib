from smartscan.utils.models import ModelManager
from pathlib import Path

def test_model_manager():
    model_manager = ModelManager()
    exists = model_manager.model_exists('all-minilm-l6-v2')
    model_manager.delete_model('all-minilm-l6-v2')
    exists = model_manager.model_exists('all-minilm-l6-v2')

    assert(exists == False)
    exists = model_manager.model_exists('all-minilm-l6-v2')
    assert(exists == False)

    path = model_manager.download_model('all-minilm-l6-v2')
    assert(isinstance(path, Path) == True)

    exists = model_manager.model_exists('all-minilm-l6-v2')
    assert(exists == True)
    print(exists)