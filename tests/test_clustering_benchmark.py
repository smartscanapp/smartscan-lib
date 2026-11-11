
import pytest
import time
import numpy as np

from smartscan.cluster import IncrementalClusterer


def test_incremental_clusterer_benchmark(request):
    n_items = request.config.getoption("--items")
    dim = request.config.getoption("--dim")

    embeddings = {str(i): np.random.rand(dim).astype(np.float32) for i in range(n_items)}
    clusterer = IncrementalClusterer(default_threshold=0.3, top_k=5)

    start = time.perf_counter()
    result = clusterer.cluster(embeddings)
    elapsed = time.perf_counter() - start

    print(f"\nItems: {n_items} | Time: {elapsed:.4f}s")

    assert result is not None
