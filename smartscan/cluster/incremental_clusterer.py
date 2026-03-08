import numpy as np
import uuid
import hnswlib

from typing import Optional

from smartscan.embeds.helpers import update_prototype_embedding
from smartscan.cluster.helpers import merge_similar_clusters
from smartscan.cluster.types import Cluster, Assignments, ClusterMetadata, ClusterResult, ClusterId


class IncrementalClusterer:
    def __init__(
        self,
        existing_clusters: Optional[dict[ClusterId, Cluster]] = None,
        default_threshold: float = 0.3,
        min_cluster_size: int = 2,
        top_k: int = 5,
        ann_max_elements: int = 1_000_000,
        ann_ef_construction: int = 200,
        ann_max_neighbors: int = 16,
        ann_ef_search: int = 50,
        ):
        self.clusters: dict[ClusterId, Cluster] = existing_clusters or {}
        self.assignments: Assignments = {}
        self.default_threshold = default_threshold
        self.min_cluster_size = min_cluster_size
        self.top_k = top_k
        self.ann_max_elements=ann_max_elements
        self.ann_ef_construction = ann_ef_construction
        self.ann_max_neighbors=ann_max_neighbors
        self.ann_ef_search=ann_ef_search
        self._ann_index = None
        self._ann_initialized = False
        self._id_map: dict[int, str] = {}
        self._rev_id_map: dict[str, int] = {}
        self._next_int_id = 0


    def cluster(self, ids: list[str], embeddings: list[np.ndarray]) -> ClusterResult:
        if not ids:
            return ClusterResult(self.clusters, self.assignments, None)

        all_items: dict[str, np.ndarray] = {i: e for i, e in zip(ids, embeddings)}
        embed_dim = embeddings[0].shape[0]
        self._init_ann(embed_dim)

        min_cluster_size = self._compute_min_cluster_size(len(all_items))

        for item_id in ids:
            emb = all_items[item_id]

            if not self.clusters:
                self._set_and_assign(item_id, emb)
                self._add_to_ann(item_id, emb)
                continue

            cluster_ids = list(self.clusters.keys())
            cos_sims = np.array([np.dot(emb, self.clusters[cid].embedding) for cid in cluster_ids])
            best_idx = np.argmax(cos_sims)
            best_cid = cluster_ids[best_idx]
            best_sim = cos_sims[best_idx]

            avg_cohesion, _, _ = self._compute_average_cluster_stats()
            threshold = self._get_threshold(self.clusters[best_cid], avg_cohesion, min_cluster_size)
            
            if best_sim >= threshold:
                self._update_and_assign(item_id, emb, best_cid)
            else:
                # fallback to voting apporach for weaker/immature clusters
                chosen_cid= self._assign_by_votes(emb, avg_cohesion, min_cluster_size)
                if chosen_cid:
                    self._update_and_assign(item_id, emb, chosen_cid)
                else:
                    self._set_and_assign(item_id, emb)

            self._add_to_ann(item_id, emb)

        self._remove_singletons()
        
        avg_cohesion, _, avg_std = self._compute_average_cluster_stats()
        merge_threshold = max(self.default_threshold, avg_cohesion - avg_std)
        cluster_merges = merge_similar_clusters(self.clusters, merge_threshold)
        return ClusterResult(self.clusters, self.assignments, cluster_merges)

    def clear(self):
        self.clusters.clear()
        self.assignments.clear()
        self._ann_index = None
        self._ann_initialized = False
        self._id_map.clear()
        self._rev_id_map.clear()
        self._next_int_id = 0

    def _init_ann(self, dim: int) -> None:
        self._ann_index = hnswlib.Index(space="cosine", dim=dim)
        self._ann_index.init_index(
            max_elements=self.ann_max_elements,
            ef_construction=self.ann_ef_construction,
            M=self.ann_max_neighbors,
        )
        self._ann_index.set_ef(self.ann_ef_search)
        self._ann_initialized = True

    def _add_to_ann(self, item_id: str, embedding: np.ndarray) -> None:
        int_id = self._next_int_id
        self._next_int_id += 1
        self._id_map[int_id] = item_id
        self._rev_id_map[item_id] = int_id
        self._ann_index.add_items(embedding.reshape(1, -1), [int_id])

    def _query_ann(self, embedding: np.ndarray) -> list[str]:
        k = min(self.top_k, self._next_int_id)
        if k == 0:
            return []
        labels, _ = self._ann_index.knn_query(embedding, k=k)
        return [self._id_map[l] for l in labels[0] if l in self._id_map]

    def _get_threshold(self, cluster: Cluster, avg_cohesion: float,min_cluster_size: int) -> float:
        mean_sim = cluster.metadata.mean_similarity
        std_sim = cluster.metadata.std_similarity
        cluster_size = cluster.metadata.prototype_size
        size_factor = 1 - np.exp(-cluster.metadata.prototype_size)
        baseline = (self.default_threshold) * size_factor
        if cluster_size < min_cluster_size or avg_cohesion <= 0:
            return baseline
        # prevents rejecting reasonable items or accepting poor matches too abruptly
        x = (mean_sim - avg_cohesion) / max(1e-6, avg_cohesion)
        alpha = 1.0 / (1.0 + np.exp(-x))
        cohesion_score = mean_sim
        adaptive_threshold = max(cohesion_score, baseline)
        adaptive_threshold = alpha * adaptive_threshold + (1.0 - alpha) * baseline
        return adaptive_threshold

    def _set_and_assign(self, item_id: str, embedding: np.ndarray) -> None:
        prototype_id = self._generate_id()
        metadata = ClusterMetadata(prototype_size=1, mean_similarity=self.default_threshold, std_similarity=0.0, label=Cluster.UNLABELLED)
        cluster = Cluster(prototype_id, embedding, metadata, label=Cluster.UNLABELLED)
        self.clusters[prototype_id] = cluster
        self.assignments[item_id] = prototype_id

    def _update_and_assign(self, item_id: str, embedding: np.ndarray, cluster_id: ClusterId) -> None:
        cluster = self.clusters[cluster_id]
        old_size = cluster.metadata.prototype_size
        old_meta = cluster.metadata
        new_embedding = update_prototype_embedding(cluster.embedding, embedding, old_size)
        sim_new = float(np.dot(new_embedding, embedding))
        new_mean = (old_meta.mean_similarity * old_size + sim_new) / (old_size + 1) if old_size >= 1 else sim_new
        new_std = (np.sqrt(((old_size - 1) * old_meta.std_similarity**2 + (sim_new - old_meta.mean_similarity) * (sim_new - new_mean)) / old_size)
                   if old_size > 1 else 0.0)

        cluster.embedding = new_embedding
        cluster.metadata.prototype_size = old_size + 1
        cluster.metadata.mean_similarity = new_mean
        cluster.metadata.std_similarity = new_std
        self.assignments[item_id] = cluster_id

    def _assign_by_votes(self, emb: np.ndarray, avg_cohesion: float, min_cluster_size: int) -> Optional[ClusterId]:
        if not (self._ann_initialized and self._next_int_id > 0):
            return None
        nn_ids = self._query_ann(emb)
        vote_counts, vote_sims = self._tally_votes(nn_ids, emb)
        if not vote_counts:
            return None
        voted_cid = self._select_top_cluster(vote_counts, vote_sims)
        voted_cluster = self.clusters[voted_cid]
        n_votes = vote_counts[voted_cid]
        required_votes = (self.top_k // 2)
        if n_votes < required_votes:
            return None
        sim_to_voted = float(np.dot(emb, voted_cluster.embedding))
        voted_threshold = self._get_threshold(voted_cluster, avg_cohesion, min_cluster_size)
        if sim_to_voted >= voted_threshold:
            return voted_cid
        return None

    def _tally_votes(self, neighbour_ids: list[str], embedding: np.ndarray) -> tuple[dict[ClusterId, int], dict[ClusterId, list[float]]]:
        vote_counts: dict[ClusterId, int] = {}
        vote_sims: dict[ClusterId, list[float]] = {}
        for nid in neighbour_ids:
            cid = self.assignments.get(nid)
            if cid and cid in self.clusters:
                vote_counts[cid] = vote_counts.get(cid, 0) + 1
                vote_sims.setdefault(cid, []).append(
                    float(np.dot(embedding, self.clusters[cid].embedding))
                )
        return vote_counts, vote_sims

    def _select_top_cluster(self, vote_counts: dict[ClusterId, int], vote_sims: dict[ClusterId, list[float]]) -> ClusterId:
        top_value = max(vote_counts.values())
        top_cids = [cid for cid, v in vote_counts.items() if v == top_value]
        if len(top_cids) == 1:
            return top_cids[0]
        return max(top_cids, key=lambda cid: float(np.mean(vote_sims.get(cid, [0.0]))))

    def _generate_id(self) -> str:
        return uuid.uuid4().hex

    def _compute_min_cluster_size(self, total_items: int) -> int:
        if total_items <= 0:
            return max(2, self.min_cluster_size)
        adaptive = max(2, int(np.sqrt(total_items)))
        return max(adaptive, self.min_cluster_size)

    def _compute_average_cluster_stats(self) -> tuple[float, float, float]:
        cohesions, cluster_sizes, stds = [], [], []
        for c in self.clusters.values():
            size = c.metadata.prototype_size
            cluster_sizes.append(size)
            if size > 1:
                cohesions.append(c.metadata.mean_similarity)
                stds.append(c.metadata.std_similarity)
        avg_cohesion = np.mean(cohesions) if cohesions else 0.0
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0
        avg_std = np.mean(stds) if stds else 0.0
        return avg_cohesion, avg_cluster_size, avg_std
    
    def _remove_singletons(self) -> None:
        singleton_items = {
            item_id: cid
            for item_id, cid in self.assignments.items()
            if self.clusters[cid].metadata.prototype_size == 1
        }
        for item_id, cid in singleton_items.items():
            del self.clusters[cid]
            self.assignments.pop(item_id, None)