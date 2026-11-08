## v1.1.2 - 08/11/2026

### Added
* Added register and remove listener methods to `BatchProcessor`

### Changed
* Allow null cluster labels

### Removed
* Removed FewShotClassifier BatchProcessor implementation
* Removed Output from ProcessorListener

___


## v1.1.1 - 07/09/2026

### Removed
* Removed index package

### Changed
* Improved robustness and type safety of resource file usage in `ModelManager` and `ModelInfo` by using `TypedDict` for resource files instead of `List`

___

## v1.1.0 - 08/07/2026

### Changed
* Made `EmbeddingStore` async
* Replace all usage of `ItemEmbedding` with `StoredEmbedding` (breaking)
* Updated `cluster` method on `IncrementalClusterer` to accept dict[ItemId, NDArray] (breaking)
* Remove `label` from Cluster dataclass (use from metadata) and make `ClusterMetadata` dataclass ✅
* Removed sim factor from `FewShotClassifier` (breaking)
* Made `embedding_dim` a class attribute
* Renamed `max_tokenizer_length` to `max_tokens` and made it a class attribute (breaking)
* Removed `embed_batch` from embedding providers and update `embed` to handle list. It now returns embeddings in format [b,dim] (breaking)

___

## v1.0.2 - 12/04/2026

### Fixed
* Fixed bug caused by incorrect model path being passed for clip text embedder

### Added
* Added model registry module ( move from constants)
* Added model manager test

### Changed
* Assigned default values for max token length on relevant text embedder

___

## v1.0.1 - 08/03/2026

### Removed
* Removed benchmarking param for `Incremental Clusterer` (breaking)
* Removed merge-threshold param (breaking) for `Incremental Clusterer` and replaced with dynamic merge-threshold based on stats across clusters (breaking)

### Changed
* Adjusted how dynamic threshold is calculated (less restrictive)

___

## v1.0.0 - 06/03/2026

Initial release
