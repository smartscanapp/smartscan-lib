import numpy as np
import random
import uuid

from typing import Dict, List, Optional
from smartscan.classify.helpers import merge_similar_clusters
from smartscan.classify.types import (
    BaseCluster,
    Assignments,
    UnLabelledCluster,
    ClusterMetadata,
    ClusterResult,
)
from smartscan.classify.metrics import ClusterMetricTracker
from smartscan.embeds import update_prototype_embedding


class IncrementalClusterer:
    """
    IncrementalClusterer incrementally clusters items using cluster-based centroids.
    """

    def __init__(
        self,
        existing_clusters: Dict[str, UnLabelledCluster] = {},
        default_threshold: float = 0.3,
        merge_threshold: Optional[float] = None,
        sim_factor: float = 1.0,
        min_cluster_size: int = 5,
        top_k: int = 3,
    ):
        self.default_threshold = default_threshold
        self.sim_factor = sim_factor
        self.clusters: Dict[str, UnLabelledCluster] = existing_clusters
        self.assignments: Assignments = {}
        self.min_cluster_size = min_cluster_size
        self.top_k = top_k
        self.max_batch_size = 10_000
        self.min_batch_size = 10
        self.merge_threshold = merge_threshold

    def cluster(self, ids: List[str], embeddings: List[np.ndarray]) -> ClusterResult:
        items = list(zip(ids, embeddings))
        random.shuffle(items)

        batch_size = min(
            self.max_batch_size,
            max(self.min_batch_size, int(len(items) * 0.01)),
        )
        max_singleton_clusters = max(1000, batch_size * 5)

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            self._assign_batch(batch)

            check_clusters_to_purge = (
                len(self.clusters) > 0
                and len(self.assignments) % int(max_singleton_clusters * 0.1) == 0
            )
            if check_clusters_to_purge:
                weak_cluster_ids, weak_assignment_ids = self._get_weak_clusters(2)
                if len(weak_assignment_ids) >= max_singleton_clusters:
                    self._remove_clusters(weak_cluster_ids)
                    self._remove_assignments(weak_assignment_ids)

        weak_cluster_ids, weak_assignment_ids = self._get_weak_clusters(self.min_cluster_size)
        self._remove_clusters(weak_cluster_ids)

        retry_items = [(i, e) for i, e in items if i in weak_assignment_ids]
        for item_id, embedding in retry_items:
            self._assign_item(item_id, embedding, self.sim_factor * 0.9)

        weak_cluster_ids, weak_assignment_ids = self._get_weak_clusters(self.min_cluster_size)
        self._remove_clusters(weak_cluster_ids)
        self._remove_assignments(weak_assignment_ids)

        cluster_merges = None
        if self.merge_threshold:
            cluster_merges = merge_similar_clusters(self.clusters, self.merge_threshold)

        return ClusterResult(self.clusters, self.assignments, cluster_merges)

    def _assign_batch(self, batch: List[tuple[str, np.ndarray]]):
        if not self.clusters:
            for item_id, embedding in batch:
                self._set_and_assign(item_id, embedding)
            return

        cluster_ids, cluster_embeds = zip(
            *[(c.prototype_id, c.embedding) for c in self.clusters.values()]
        )
        cluster_embeds = np.stack(cluster_embeds, axis=0)

        for item_id, embedding in batch:
            sims = np.dot(cluster_embeds, embedding)
            top_k_indices = np.argsort(sims)[-self.top_k :][::-1]
            assigned = False

            for idx in top_k_indices:
                cluster = self.clusters[cluster_ids[idx]]
                if sims[idx] >= self.sim_factor * cluster.metadata.mean_similarity:
                    self._update_and_assign(item_id, embedding, cluster)
                    assigned = True
                    break

            if not assigned:
                self._set_and_assign(item_id, embedding)

    def clear(self):
        self.clusters.clear()
        self.assignments.clear()

    def _assign_item(
        self,
        item_id: str,
        embedding: np.ndarray,
        retry_sim_factor: Optional[float] = None,
    ):
        if not self.clusters:
            self._set_and_assign(item_id, embedding)
            return

        cluster_ids, cluster_embeds = zip(
            *[(c.prototype_id, c.embedding) for c in self.clusters.values()]
        )
        cluster_embeds = np.stack(cluster_embeds, axis=0)

        sims = np.dot(cluster_embeds, embedding)
        best_idx = np.argmax(sims)
        cluster = self.clusters[cluster_ids[best_idx]]

        factor = retry_sim_factor if retry_sim_factor else self.sim_factor
        if sims[best_idx] >= factor * cluster.metadata.mean_similarity:
            self._update_and_assign(item_id, embedding, cluster)
        else:
            self._set_and_assign(item_id, embedding)

    def _get_weak_clusters(self, n: int):
        weak_cluster_ids = {
            c.prototype_id
            for c in self.clusters.values()
            if c.metadata.prototype_size < n
        }
        weak_assignment_ids = {
            item_id
            for item_id, proto_id in self.assignments.items()
            if proto_id in weak_cluster_ids
        }
        return weak_cluster_ids, weak_assignment_ids

    def _set_and_assign(self, item_id: str, embedding: np.ndarray):
        prototype_id = uuid.uuid4().hex
        cluster = UnLabelledCluster(
            prototype_id,
            embedding,
            ClusterMetadata(prototype_size=1),
        )
        self.clusters[prototype_id] = cluster
        self.assignments[item_id] = prototype_id

    def _update_and_assign(
        self,
        item_id: str,
        embedding: np.ndarray,
        cluster: BaseCluster,
    ):
        new_embedding = update_prototype_embedding(
            cluster.embedding,
            embedding,
            cluster.metadata.prototype_size,
        )

        metrics_tracker = ClusterMetricTracker(
            cluster,
            np.stack([c.embedding for c in self.clusters.values()], axis=0),
        )
        metrics_tracker.add_samples(embedding)

        updated = UnLabelledCluster(
            cluster.prototype_id,
            new_embedding,
            metadata=metrics_tracker.get_metrics(),
        )

        self.clusters[cluster.prototype_id] = updated
        self.assignments[item_id] = cluster.prototype_id

    def _remove_clusters(self, cluster_ids: set[str] | list[str]):
        for pid in cluster_ids:
            self.clusters.pop(pid, None)

    def _remove_assignments(self, assignment_ids: set[str] | list[str]):
        for item_id in assignment_ids:
            self.assignments.pop(item_id, None)
