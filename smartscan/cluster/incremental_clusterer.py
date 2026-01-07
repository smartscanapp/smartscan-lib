import numpy as np
import random
import uuid

from smartscan.embeds.types import ItemEmbedding, Prototype
from smartscan.embeddings import update_prototype_embedding, update_cohesion_score

class IncrementalClusterer():
    """
    IncrementalClusterer incrementally clusters items using prototype-based centroids.

    Items are assigned to the most similar existing cluster or form a new cluster if no
    sufficient match exists. Small/weak clusters are periodically purged, and items from
    removed clusters are retried. Batch size and purge thresholds are computed dynamically.
    """

    def __init__(
        self,
        existing_prototypes: dict[str, Prototype] = {},
        default_threshold: float = 0.3,
        sim_factor: float = 1.0,
        min_cluster_size: int = 5,
        top_k: int = 3
    ):
        self.default_threshold = default_threshold
        self.sim_factor = sim_factor
        self.prototypes: dict[str, Prototype] = existing_prototypes
        self.assignments: dict[str, str] = {}
        self.min_cluster_size = min_cluster_size
        self.top_k = top_k
        self.max_batch_size = 10_000
        self.min_batch_size = 10


    def cluster(self, items: list[ItemEmbedding]):
        random.shuffle(items)
        
        batch_size = min(self.max_batch_size, max(self.min_batch_size, int(len(items) * 0.01)))
        max_singleton_clusters = max(1000, batch_size * 5)

        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            self._assign_batch(batch)

            # Periodically purge weak clusters to avoid buildup
            check_clusters_to_purge = len(self.prototypes) > 0 and len(self.assignments) % int(max_singleton_clusters * 0.1) == 0
            if check_clusters_to_purge:
                weak_clusters_ids, weak_assignments_ids = self._get_weak_clusters(2)
                if len(weak_assignments_ids) >= max_singleton_clusters:
                    self._remove_clusters(weak_clusters_ids)
                    self._remove_assignments(weak_assignments_ids)

        # Retry weakly assigned items
        weak_clusters_ids, weak_assignments_ids = self._get_weak_clusters(self.min_cluster_size)
        self._remove_clusters(weak_clusters_ids)

        retry_items = [item for item in items if item.item_id in weak_assignments_ids]
        for item in retry_items:
            self._assign_items(item, self.sim_factor * 0.9)

        # Final cleanup
        weak_clusters_ids, weak_assignments_ids = self._get_weak_clusters(self.min_cluster_size)
        self._remove_clusters(weak_clusters_ids)
        self._remove_assignments(weak_assignments_ids)

        return self.prototypes, self.assignments

    def _assign_batch(self, batch: list[ItemEmbedding]):
        if not self.prototypes:
            for item in batch:
                self._set_and_assign(item)
            return

        cluster_ids, prototype_embeds = zip(*(((p.prototype_id, p.embedding) for p in self.prototypes.values())))

        for item in batch:
            sims = np.dot(prototype_embeds, item.embedding)
            # Get top-k candidate clustersn to reduce order bias from early assignments
            top_k_indices = np.argsort(sims)[-self.top_k:][::-1]  # descending
            assigned = False
            for idx in top_k_indices:
                prototype = self.prototypes[cluster_ids[idx]]
                if sims[idx] >= self.sim_factor * prototype.cohesion_score:
                    self._update_and_assign(item, prototype)
                    assigned = True
                    break
            if not assigned:
                self._set_and_assign(item)

    def clear(self):
        self.prototypes.clear()
        self.assignments.clear()

    def _assign_items(self, item: ItemEmbedding, retry_sim_factor: float | None = None):
        if len(self.prototypes) == 0:
            self._set_and_assign(item)
            return

        cluster_ids, prototype_embeds = zip(*(((p.prototype_id, p.embedding) for p in self.prototypes.values())))
        sims = np.dot(prototype_embeds, item.embedding)
        best_idx = np.argmax(sims)
        prototype = self.prototypes[cluster_ids[best_idx]]
        factor = retry_sim_factor if retry_sim_factor else self.sim_factor

        if sims[best_idx] >= factor * prototype.cohesion_score:
            self._update_and_assign(item, prototype)
        else:
            self._set_and_assign(item)

    def _get_weak_clusters(self, n: int):
        weak_cluster_ids = {c.prototype_id for c in self.prototypes.values() if c.prototype_size < n}
        weak_assignments_ids = {item_id for item_id, proto_id in self.assignments.items() if proto_id in weak_cluster_ids}
        return weak_cluster_ids, weak_assignments_ids

    def _set_and_assign(self, item: ItemEmbedding):
        prototype_id = uuid.uuid4().hex
        prototype = Prototype(prototype_id, item.embedding, self.default_threshold, prototype_size=1)
        self.prototypes[prototype_id] = prototype
        self.assignments[item.item_id] = prototype_id

    def _update_and_assign(self, item: ItemEmbedding, prototype: Prototype):
        new_embedding = update_prototype_embedding(prototype.embedding, item.embedding, prototype.prototype_size)
        new_cohesion = update_cohesion_score(prototype.cohesion_score, prototype.prototype_size, prototype.embedding, item.embedding)
        updated = Prototype(prototype.prototype_id, new_embedding, new_cohesion, prototype.prototype_size + 1)
        self.prototypes[prototype.prototype_id] = updated
        self.assignments[item.item_id] = prototype.prototype_id

    def _remove_clusters(self, cluster_ids: set[str] | list[str]):
        for pid in cluster_ids:
            self.prototypes.pop(pid, None)

    def _remove_assignments(self, assignment_ids: set[str] | list[str]):
        for item_id in assignment_ids:
            self.assignments.pop(item_id, None)