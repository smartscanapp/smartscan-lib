import pytest
import numpy as np
from smartscan.classify import IncrementalClusterer
from smartscan.classify.types import Cluster

class TestIncrementalClusterer:
    @pytest.fixture
    def embeddings(self):
        # 5 simple 3D embeddings for deterministic behavior
        return {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.9, 0.1, 0.0]),
            "c": np.array([0.0, 1.0, 0.0]),
            "d": np.array([0.0, 0.9, 0.1]),
            "e": np.array([0.5, 0.5, 0.0]),
        }

    def test_basic_clustering_integrity(self, embeddings):
        clusterer = IncrementalClusterer(default_threshold=0.1, top_k=2, benchmarking=True)

        ids = list(embeddings.keys())
        vectors = list(embeddings.values())

        result = clusterer.cluster(ids, vectors)

        # Cluster and assignments counts should match expected
        assert len(result.assignments) == len(ids)
        for item_id in ids:
            assert item_id in result.assignments
            cid = result.assignments[item_id]
            assert cid in result.clusters
            cluster = result.clusters[cid]
            assert isinstance(cluster, Cluster)
            assert cluster.metadata.prototype_size >= 1

        # Counts consistency
        for cid, cluster in result.clusters.items():
            assigned_items = [i for i, c in result.assignments.items() if c == cid]
            assert cluster.metadata.prototype_size == len(assigned_items)

    def test_nearest_neighbor_assignment(self, embeddings):
        clusterer = IncrementalClusterer(default_threshold=0.0, top_k=1, benchmarking=True)

        # Cluster all items at once
        ids = ["a", "b", "c", "d"]
        vectors = [embeddings[i] for i in ids]
        clusterer.cluster(ids, vectors)

        # Check that similar embeddings "a" and "b" are in same cluster
        assert clusterer.assignments["a"] == clusterer.assignments["b"]
        assert clusterer.assignments["c"] == clusterer.assignments["d"]

        # Ensure different clusters for distant embeddings
        assert clusterer.assignments["a"] != clusterer.assignments["c"]


    def test_incremental_update_counts(self, embeddings):
        clusterer = IncrementalClusterer(default_threshold=0.0, top_k=1, benchmarking=True)

        # Add first embedding
        clusterer.cluster(["a"], [embeddings["a"]])
        first_cid = clusterer.assignments["a"]
        assert clusterer._counts[first_cid] == 1

        # Add similar embedding
        clusterer.cluster(["b"], [embeddings["b"]])
        second_cid = clusterer.assignments["b"]

        # Should merge into same cluster due to similarity
        assert first_cid == second_cid
        assert clusterer._counts[first_cid] == 2

    def test_clear_resets_clusterer(self, embeddings):
        clusterer = IncrementalClusterer(default_threshold=0.0, top_k=1, benchmarking=True)
        clusterer.cluster(list(embeddings.keys()), list(embeddings.values()))

        assert clusterer.clusters
        assert clusterer.assignments
        assert clusterer._counts

        clusterer.clear()

        # Everything should be empty
        assert clusterer.clusters == {}
        assert clusterer.assignments == {}
        assert clusterer._counts == {}

    def test_no_stale_clusters_or_assignments(self, embeddings):
        clusterer = IncrementalClusterer(default_threshold=0.0, top_k=1, benchmarking=True)

        ids = list(embeddings.keys())
        vectors = list(embeddings.values())
        result = clusterer.cluster(ids, vectors)

        # Every assignment points to a valid cluster
        for item_id, cid in result.assignments.items():
            assert cid in result.clusters, f"Assignment of {item_id} points to missing cluster {cid}"

        # No cluster should have zero assigned items
        for cid, cluster in result.clusters.items():
            assigned_items = [i for i, c in result.assignments.items() if c == cid]
            assert len(assigned_items) > 0, f"Cluster {cid} has no assigned items"

