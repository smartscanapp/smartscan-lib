
import numpy as np
import pickle
from PIL import Image
from smartscan.utils import get_frames_from_video, read_text_file
from smartscan.providers import ImageEmbeddingProvider, TextEmbeddingProvider

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

def embed_video_file(path: str, n_frames: int, embedder: ImageEmbeddingProvider):
    frame_arrs = get_frames_from_video(path, n_frames)
    frame_images = [Image.fromarray(frame) for frame in frame_arrs]
    batch = embedder.embed_batch(frame_images)
    return generate_prototype_embedding(batch)


def embed_video_files(paths: list[str], n_frames: int, embedder: ImageEmbeddingProvider):
    return np.stack([embed_video_file(path, n_frames, embedder) for path in paths], axis=0)


def embed_image_file(path: str, embedder: ImageEmbeddingProvider):
    return embedder.embed(Image.open(path))


def embed_image_files(paths: list[str], embedder: ImageEmbeddingProvider):
    return embedder.embed_batch([Image.open(path) for path in paths])


def embed_text_file(path: str, embedder: TextEmbeddingProvider, max_tokenizer_length=128, max_chunks=5):
    chunks = chunk_text(read_text_file(path), max_tokenizer_length, max_chunks)
    chunk_embeddings = embedder.embed_batch(chunks)
    return generate_prototype_embedding(chunk_embeddings)


def embed_text_files(paths: list[str], embedder: TextEmbeddingProvider, max_tokenizer_length=128, max_chunks=5):
    return np.stack([embed_text_file(path, embedder, max_tokenizer_length, max_chunks) for path in paths], axis=0)


def chunk_text(s: str, tokenizer_max_length: int, limit: int = 10):
    max_chunks = len(s) // 4 * tokenizer_max_length
    n_chunks = min(limit, max_chunks)
    chunks = []
    start = 0

    while len(chunks) < n_chunks:
        end = start + tokenizer_max_length
        if end >= len(s):
            chunk = s[start:]
        else:
            space_index = s.rfind(" ", start, end)
            if space_index == -1: 
                space_index = end
            chunk = s[start:space_index]
            end = space_index
        if not chunk:
            break
        chunks.append(chunk)
        start = end + 1

    return chunks


def save_embedding(filepath: str, embedding: np.ndarray):
    """Saves embedding to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)


def load_embedding(filepath: str) -> np.ndarray:
    """Loads embedding from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
