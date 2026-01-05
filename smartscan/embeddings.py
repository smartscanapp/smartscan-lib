
import numpy as np
import pickle
from smartscan.utils import  video_source_to_pil_images, image_source_to_pil_image
from smartscan.providers import ImageEmbeddingProvider, TextEmbeddingProvider
from smartscan.types import VideoSource, ImageSource

# embeddings (b, dim)
def generate_prototype_embedding(embeddings: np.ndarray) -> np.ndarray:    
    embeddings_tensor = np.stack(embeddings, axis=0)
    prototype = np.mean(embeddings_tensor, axis=0)
    prototype /= np.linalg.norm(prototype)
    return prototype

def update_prototype_embedding(current_prototype: np.ndarray, new_embeddings: np.ndarray, current_n: int) -> np.ndarray:
    new_embeddings = np.asarray(new_embeddings)
    if new_embeddings.ndim == 1:
        new_embeddings = new_embeddings[np.newaxis, :]
    batch_sum = np.sum(new_embeddings, axis=0)
    updated_n = current_n + new_embeddings.shape[0]
    updated_prototype = (current_prototype * current_n + batch_sum) / updated_n
    updated_prototype /= np.linalg.norm(updated_prototype)
    return updated_prototype

def embed_video(embedder: ImageEmbeddingProvider, source: VideoSource, n_frames: int):
    batch = embedder.embed_batch(video_source_to_pil_images(source, n_frames))
    return generate_prototype_embedding(batch)


def embed_videos(embedder: ImageEmbeddingProvider, sources: list[VideoSource], n_frames: int):
    return np.stack([embed_video(embedder, source, n_frames) for source in sources], axis=0)


def embed_image(embedder: ImageEmbeddingProvider, source: ImageSource,):
    return embedder.embed(image_source_to_pil_image(source))


def embed_images(embedder: ImageEmbeddingProvider, sources: list[ImageSource],):
    return embedder.embed_batch([image_source_to_pil_image(source) for source in sources])


def embed_text(embedder: TextEmbeddingProvider, text: str):
    return embedder.embed(text)


def embed_texts(embedder: TextEmbeddingProvider, texts: list[str], ):
    return embedder.embed_batch(texts)


def save_embedding(filepath: str, embedding: np.ndarray):
    """Saves embedding to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)


def load_embedding(filepath: str) -> np.ndarray:
    """Loads embedding from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
