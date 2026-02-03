# SmartScan Python Library

Python library providing tools for ML inference, embeddings, indexing, semantic search, clustering, few-shot classification, and efficient batch processing. This library powers the SmartScan Server used by the Desktop App.

---

## Supported Embedding Providers

### Image

* CLIP ViT-B-32
* DINOv2 Small
* Inception ResNet V2 (facial recognition)

### Text

* CLIP ViT-B-32
* all-MiniLM-L6-v2

---

## Installation

### Prerequisites

* Python 3.10+

```bash
pip install git+https://github.com/smartscanapp/smartscan-lib.git
```

---

## Quick Start

### Embeddings

#### Embed images

```python
from smartscan.providers import ClipImageEmbedder, DinoSmallV2ImageEmbedder
from PIL import Image

image_embedder = ClipImageEmbedder("path-to-clip-model")
# or
image_embedder = DinoSmallV2ImageEmbedder("path-to-dino-model")

image_embedder.init()

image_embedder.embed(Image.open("image.jpg"))
image_embedder.embed_batch([
    Image.open("image1.jpg"),
    Image.open("image2.jpg")
])
```

#### Embed text

```python
from smartscan.providers import ClipTextEmbedder, MiniLmTextEmbedder

text_embedder = ClipTextEmbedder("path-to-clip-model")
text_embedder.init()

text_embedder.embed("text to embed")
text_embedder.embed_batch(["text1", "text2", "text3"])
```

```python
text_embedder = MiniLmTextEmbedder("path-to-minilm-model", 512)
text_embedder.init()

text_embedder.embed("text to embed")
text_embedder.embed_batch(["text1", "text2", "text3"])
```

---

### Indexing

Indexers are implemented using the `BatchProcessor` abstraction. Default indexers are provided for common data types.
All indexers optionally accept a `ProcessorListener` for progress and batch callbacks.

#### Images

```python
from smartscan.indexer import ImageIndexer
from smartscan.providers import DinoSmallV2ImageEmbedder

image_urls = [...]
image_paths = [...]

image_embedder = DinoSmallV2ImageEmbedder("path-to-dino-model")
image_embedder.init()

indexer = ImageIndexer(
    image_encoder=image_embedder,
    listener=listener  # optional
)

await indexer.run(image_urls)
await indexer.run(image_paths)
```

#### Videos

```python
from smartscan.indexer import VideoIndexer
from smartscan.providers import DinoSmallV2ImageEmbedder

video_urls = [...]
video_paths = [...]

image_embedder = DinoSmallV2ImageEmbedder("path-to-dino-model")
image_embedder.init()

indexer = VideoIndexer(
    image_encoder=image_embedder,
    listener=listener  # optional
)

await indexer.run(video_urls)
await indexer.run(video_paths)
```

#### Documents

```python
from smartscan.indexer import DocIndexer
from smartscan.providers import MiniLmTextEmbedder

doc_paths = [...]

text_embedder = MiniLmTextEmbedder("path-to-minilm-model", 512)
text_embedder.init()

indexer = DocIndexer(
    text_encoder=text_embedder,
    listener=listener  # optional
)

await indexer.run(doc_paths)
```

---

### Clustering

Incrementally groups embeddings into clusters based on similarity. Supports existing clusters, adaptive thresholds, and optional auto-merging.

```python
from smartscan.cluster import IncrementalClusterer

clusterer = IncrementalClusterer(
    default_threshold=initial_threshold,
    merge_threshold=auto_merge_threshold,
    existing_assignments=existing_assignments,
    existing_clusters=existing_clusters,
)

result = clusterer.cluster(ids, embeddings)
```

---

### Few-Shot Classification

Assigns a label to an embedding by comparing it against pre-labelled cluster centroids.
Supports batch processing and an optional `ProcessorListener`.

#### Single item

```python
from smartscan.classify.fewshot import few_shot_classify

result = few_shot_classify(
    item=item_embedding,
    labelled_clusters=clusters,
    sim_factor=1.0
)

print(result.label, result.similarity)
```

#### Batch processing

```python
from smartscan.classify.fewshot import FewShotClassifier

classifier = FewShotClassifier(
    labelled_clusters=clusters,
    listener=listener,  # optional
    sim_factor=1.0,
    batch_size=32
)

await classifier.run(item_embeddings)
```

---
