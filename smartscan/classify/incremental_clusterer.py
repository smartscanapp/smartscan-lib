import numpy as np
import random
import uuid

from typing import Dict, List, Optional
from smartscan.classify.helpers import merge_similar_clusters
from smartscan.classify.types import (
    Cluster,
    Assignments,
    ClusterMetadata,
    ClusterResult,
)
from smartscan.embeds import update_prototype_embedding


class IncrementalClusterer:
    def __init__(
        self,
        existing_clusters: Optional[Dict[str, Cluster]] = None,
        existing_assignments: Optional[Assignments] = None,
        default_threshold: float = 0.3,
        merge_threshold: Optional[float] = None,
        min_cluster_size: int = 8,
        top_k: int = 3,
        benchmarking: bool = False,
    ):
        self.default_threshold = default_threshold
        self.clusters: Dict[str, Cluster] = existing_clusters or {}
        self.assignments: Assignments = existing_assignments or {}
        self.min_cluster_size = min_cluster_size
        self.top_k = top_k
        self.merge_threshold = merge_threshold
        self.benchmarking = benchmarking

    def _get_threshold(self, cluster: Cluster) -> float:
        mean_sim = getattr(cluster.metadata, "mean_similarity", 0.0)
        std_sim = getattr(cluster.metadata, "std_similarity", 0.0)
        base_threshold = (
            mean_sim
            if (mean_sim > 0 and cluster.metadata.prototype_size > self.min_cluster_size)
            else self.default_threshold
        )
        return max(base_threshold - std_sim, 0.0)

    def cluster(self, ids: List[str], embeddings: List[np.ndarray]) -> ClusterResult:
        items: Dict[str, np.ndarray] = {i: e for i, e in zip(ids, embeddings)}
        all_items = items.copy()

        while items:
            iid = sorted(items.keys())[0]
            emb = items.pop(iid)

            if not self.clusters:
                self._set_and_assign(iid, emb)
                continue

            cluster_ids = list(self.clusters.keys())

            # Compute cosine similarity to each cluster
            cos_sims = np.array([np.dot(emb, self.clusters[cid].embedding) for cid in cluster_ids])
            thresholds = np.array([self._get_threshold(self.clusters[cid]) for cid in cluster_ids])

            valid_idx = np.where(cos_sims >= thresholds)[0]
            if valid_idx.size > 0:
                best_idx = valid_idx[np.argmax(cos_sims[valid_idx])]
                self._update_and_assign(iid, emb, self.clusters[cluster_ids[best_idx]])
            else:
                self._set_and_assign(iid, emb)

        # --- Merge or reassign small clusters ---
        small_clusters = {cid: c for cid, c in self.clusters.items() if c.metadata.prototype_size < self.min_cluster_size}
        large_clusters = {cid: c for cid, c in self.clusters.items() if c.metadata.prototype_size >= self.min_cluster_size}

        for sid, small_cluster in small_clusters.items():
            reassigned_items = [item_id for item_id, cid in self.assignments.items() if cid == sid]
            cluster_deleted = False  # flag to track deletion

            if large_clusters:
                large_ids = list(large_clusters.keys())
                cos_sims = np.array([np.dot(small_cluster.embedding, large_clusters[cid].embedding) for cid in large_ids])
                best_idx = np.argmax(cos_sims)
                target_cid = large_ids[best_idx]
                target_cluster = large_clusters[target_cid]

                for item_id in reassigned_items:
                    item_emb = all_items[item_id]
                    other_ids = [oid for oid in all_items if oid != item_id and oid in self.assignments]

                    if not other_ids:
                        self._update_and_assign(item_id, item_emb, target_cluster)
                        cluster_deleted = True
                        continue

                    # partial sort for efficiency
                    other_embeds = np.array([all_items[oid] for oid in other_ids])
                    nn_dists = np.dot(other_embeds, item_emb)
                    top_k = 5

                    if len(nn_dists) > top_k:
                        nn_idx = np.argpartition(-nn_dists, top_k - 1)[:top_k] 
                        nn_idx = nn_idx[np.argsort(-nn_dists[nn_idx])]  # sort only top k
                    else:
                        nn_idx = np.argsort(-nn_dists)


                    votes = {}
                    sims = {}

                    for idx in nn_idx:
                        nid = other_ids[idx]
                        cid = self.assignments.get(nid)
                        if cid and cid in self.clusters:
                            votes[cid] = votes.get(cid, 0) + 1
                            sims.setdefault(cid, []).append(float(np.dot(item_emb, self.clusters[cid].embedding)))
                    
                    # keep in original cluster
                    if not votes:
                        self.assignments[item_id] = sid
                        continue

                    top_clusters = [cid for cid, v in votes.items() if v == max(votes.values())]
                    if len(top_clusters) == 1:
                        chosen_cid = top_clusters[0]
                    else:
                        chosen_cid = max(top_clusters, key=lambda cid: np.mean(sims.get(cid, [0.0])))
                  
                    self._update_and_assign(item_id, item_emb, self.clusters[chosen_cid])
                    
                    # mark cluster as having been reassigned
                    if chosen_cid != sid:
                        cluster_deleted = True
                # delete cluster only if at least one item was reassigned to another cluster
                if cluster_deleted:
                    del self.clusters[sid]

            else:
                for item_id in reassigned_items:
                    self.assignments[item_id] = sid


        cluster_merges = None
        if self.merge_threshold:
            cluster_merges = merge_similar_clusters(self.clusters, self.merge_threshold)

        return ClusterResult(self.clusters, self.assignments, cluster_merges)


    def clear(self):
        self.clusters.clear()
        self.assignments.clear()

    def _set_and_assign(self, item_id: str, embedding: np.ndarray):
        prototype_id = self._generate_id()
        metadata = ClusterMetadata(prototype_size=1, mean_similarity=self.default_threshold, std_similarity=0.0, label=Cluster.UNLABELLED)
        cluster = Cluster(prototype_id, embedding, metadata, label=Cluster.UNLABELLED)
        self.clusters[prototype_id] = cluster
        self.assignments[item_id] = prototype_id

    def _update_and_assign(self, item_id: str, embedding: np.ndarray, cluster: Cluster):
        old_meta = cluster.metadata
        old_size = getattr(old_meta, "prototype_size", 1)
        old_mean = getattr(old_meta, "mean_similarity", 0.0)
        old_std = getattr(old_meta, "std_similarity", 0.0)

        new_embedding = update_prototype_embedding(cluster.embedding, embedding, old_size)
        sim_new = float(np.dot(new_embedding, embedding))
        new_mean = (old_mean * old_size + sim_new) / (old_size + 1)

        if old_size > 1:
            new_std = np.sqrt(((old_size - 1) * old_std**2 + (sim_new - old_mean) * (sim_new - new_mean)) / old_size)
        else:
            new_std = 0.0

        updated = Cluster(
            cluster.prototype_id,
            new_embedding,
            metadata=ClusterMetadata(prototype_size=old_size + 1, mean_similarity=new_mean, std_similarity=new_std, label=cluster.label),
            label=cluster.label,
        )

        self.clusters[cluster.prototype_id] = updated
        self.assignments[item_id] = cluster.prototype_id

    # seed randomness during benchmarking for reproduceability
    def _generate_id(self):
        return random.randbytes(8).hex() if self.benchmarking else uuid.uuid4().hex