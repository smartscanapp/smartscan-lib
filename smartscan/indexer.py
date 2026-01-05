from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.embeddings import embed_video, embed_text, embed_image
from smartscan.providers import ImageEmbeddingProvider, TextEmbeddingProvider
from smartscan.types import ItemEmbedding


class ImageIndexer(BatchProcessor[str, ItemEmbedding]):
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                listener = ProcessorListener[str, ItemEmbedding],
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.image_encoder = image_encoder

    def on_process(self, item):
        embedding = embed_image(self.image_encoder, item)
        return ItemEmbedding(item, embedding)
             
    # delegate to lister e.g to handle storage
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


        
class VideoIndexer(BatchProcessor[str, ItemEmbedding]):
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                n_frames: int = 10,
                listener = ProcessorListener[str, ItemEmbedding],
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.image_encoder = image_encoder
        self.n_frames = n_frames

    def on_process(self, item):
        embedding = embed_video(self.image_encoder, item, self.n_frames)
        return ItemEmbedding(item, embedding)
             
    # delegate to lister e.g to handle storage
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


class DocIndexer(BatchProcessor[str, list[ItemEmbedding]]):
    def __init__(self, 
                text_encoder: TextEmbeddingProvider,
                listener = ProcessorListener[str, ItemEmbedding],
                max_chunks: int | None = None,
                tokenizer_max_length: int = 128,
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.text_encoder = text_encoder
        self.max_chunks = max_chunks
        self.tokenizer_max_length = tokenizer_max_length

    def on_process(self, item):
        chunk_embeddings = embed_text(self.text_encoder, item, self.tokenizer_max_length, self.max_chunks)
        return [ItemEmbedding(item, embedding) for embedding in chunk_embeddings]
             
    # delegate to lister e.g to handle storage
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)