## v1.0.2 - 12/04/2026

### Fixed
* Fixed bug caused by incorrect model path being passed for clip text embedder

### Added
* Added model registry module ( move from constants)
* Added model manager test

### Changed
* Assigned default values for max token length on relevant text embedder




## v1.0.1 - 08/03/2026

### Removed
* Removed benchmarking param for `Incremental Clusterer` (breaking)
* Removed merge-threshold param (breaking) for `Incremental Clusterer` and replaced with dynamic merge-threshold based on stats across clusters (breaking)

### Changed

* Adjusted how dynamic threshold is calculated (less restrictive)

## v1.0.0 - 06/03/2026

Initial release
