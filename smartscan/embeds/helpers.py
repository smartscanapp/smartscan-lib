
import numpy as np
import pickle

from smartscan.types import VideoSource, ImageSource
from smartscan.utils import  video_source_to_pil_images, image_source_to_pil_image, doc_source_to_text_chunks
from smartscan.providers import ImageEmbeddingProvider, TextEmbeddingProvider

_all__ = [
    "generate_prototype_embedding",
    "update_prototype_embedding",
    "embed_video",
    "embed_image",
    "embed_text",
    "save_embedding",
    "load_embedding",
    "calculate_cohesion_score",
    "update_cohesion_score",
]

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
    """Embed video from url or file"""
    batch = embedder.embed_batch(video_source_to_pil_images(source, n_frames))
    return generate_prototype_embedding(batch)

def embed_image(embedder: ImageEmbeddingProvider, source: ImageSource):
    """Embed image from url or file"""
    return embedder.embed(image_source_to_pil_image(source))

def embed_text(embedder: TextEmbeddingProvider, source: str, tokenizer_max_length: int = 128, max_chunks: int | None = None ):
    """Embed doc from url or file.
    Returns ndarray with shape (batch, dim)
    """
    chunks = doc_source_to_text_chunks(source, tokenizer_max_length, max_chunks)
    return embedder.embed_batch(chunks)

def save_embedding(filepath: str, embedding: np.ndarray):
    """Saves embedding to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)


def load_embedding(filepath: str) -> np.ndarray:
    """Loads embedding from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_cohesion_score(prototype: np.ndarray, sample_batch: np.ndarray) -> float:
    return np.mean(np.dot(sample_batch, prototype))

def update_cohesion_score(current_score: float, n: int, prototype_embedding: np.ndarray, new_samples: np.ndarray) -> float:
    if new_samples.ndim == 1:
        new_samples = new_samples[np.newaxis, :] 

    m = new_samples.shape[0]
    if m == 0:
        return current_score

    new_sum = np.sum(np.dot(new_samples, prototype_embedding))
    return (current_score * n + new_sum) / (n + m)

