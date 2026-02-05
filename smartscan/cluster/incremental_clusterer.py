import numpy as np
import random
import uuid

from typing import Dict, List, Optional

from smartscan.cluster.helpers import merge_similar_clusters
from smartscan.cluster.types import Cluster, Assignments, ClusterMetadata, ClusterResult

class IncrementalClusterer:
    def __init__(
        self,
        existing_clusters: Optional[Dict[str, Cluster]] = None,
        existing_assignments: Optional[Assignments] = None,
        default_threshold: float = 0.3,
        merge_threshold: Optional[float] = None,
        min_cluster_size: int = 2,
        top_k: int = 3,
        benchmarking: bool = False,
    ):
        self.clusters: Dict[str, Cluster] = existing_clusters or {}
        self.assignments: Assignments = existing_assignments or {}        
        self._counts: Dict[str, int] = {cid: 0 for cid in self.clusters}
        for cid in self.assignments.values():
            if cid in self._counts:
                self._counts[cid] += 1
        cohesions = [c.metadata.mean_similarity for c in self.clusters.values()]
        min_cohesion = np.percentile(cohesions, 1) if cohesions else 0.0
        self.default_threshold = max(default_threshold, min_cohesion)
        self.min_cluster_size = min_cluster_size
        self.top_k = top_k
        self.merge_threshold = merge_threshold
        self.benchmarking = benchmarking


    def cluster(self, ids: List[str], embeddings: List[np.ndarray]) -> ClusterResult:
        all_items: Dict[str, np.ndarray] = {i: e for i, e in zip(ids, embeddings)}
        min_cluster_size = self._compute_min_cluster_size(len(all_items))
        processed_count = 0

        while processed_count < len(all_items):
            item_id = ids[processed_count]
            emb = all_items[item_id]

            if not self.clusters:
                self._set_and_assign(item_id, emb)
                continue

            cluster_ids = list(self.clusters.keys())
            cos_sims = np.array([np.dot(emb, self.clusters[cid].embedding) for cid in cluster_ids])            
            avg_cohesion, avg_cluster_size = self._compute_average_cluster_stats()
            thresholds = np.array([self._get_threshold(self.clusters[cid], avg_cohesion, avg_cluster_size, min_cluster_size) for cid in cluster_ids])
            valid_idx = np.where(cos_sims >= thresholds)[0]
            
            if valid_idx.size > 0:
                best_idx = valid_idx[np.argmax(cos_sims[valid_idx])]
                self._update_and_assign(item_id, emb, self.clusters[cluster_ids[best_idx]])
            else:
                prev_cid = self.assignments.get(item_id)
                if prev_cid and prev_cid in self.clusters:
                    pass
                else:
                    self._set_and_assign(item_id, emb)
            processed_count += 1

        # Reassignment phase.
        # Termination conditions: loop until either (a) no progressed changes in an iteration, or
        # every item's assigned cluster equals the cluster with maximum similarity (i.e. stable).
        while True:
            if not self.clusters:
                break

            avg_cohesion, _ = self._compute_average_cluster_stats()
            candidate_clusters = {cid: c for cid, c in self.clusters.items()}
            progressed = False

            for cluster in candidate_clusters.values():
                other_item_ids = [item_id for item_id in all_items.keys() if item_id not in self.assignments or self.assignments.get(item_id) != cluster.prototype_id]
                if not other_item_ids:
                    continue

                other_embeds = np.stack([all_items[other_item_id] for other_item_id in other_item_ids])
                nn_indices = self._get_top_k(cluster.embedding, other_embeds)
                if len(nn_indices) == 0:
                    continue
                nn_ids = [other_item_ids[idx] for idx in nn_indices]
                chosen_target_cid = self._reassign_by_nn_votes(nn_ids, cluster)
                target_cluster = self.clusters.get(chosen_target_cid) if chosen_target_cid else None
                if not target_cluster:
                    continue

                reassigned_items = [item_id for item_id, cid in self.assignments.items() if cid == cluster.prototype_id]

                for item_id in reassigned_items:
                    item_emb = all_items[item_id]
                    other_ids = [oid for oid in all_items if oid != item_id and oid in self.assignments]
                    if not other_ids:
                        self._update_and_assign(item_id, item_emb, target_cluster)
                        progressed = True
                        continue

                    other_embs = np.stack([all_items[oid] for oid in other_ids])
                    nn_indices = self._get_top_k(item_emb, other_embs)
                    vote_counts, vote_sims = self._tally_votes([other_ids[idx] for idx in nn_indices], item_emb)
                 
                    if not vote_counts:
                        continue

                    chosen_cid = self._select_top_cluster(vote_counts, vote_sims)
                    self._update_and_assign(item_id, item_emb, self.clusters[chosen_cid])
                    if chosen_cid != cluster.prototype_id:
                        progressed = True
                        
            self._remove_empty_clusters()
            stable = self._are_clusters_stable(all_items)
            # Check global stability: every item should be assigned to the cluster with highest similarity
            # If that holds, clusters are stable and can stop. Also stop if no changes were made in this iteration.
            if stable or not progressed:
                break

        cluster_merges = None
        if self.merge_threshold:
            cluster_merges = merge_similar_clusters(self.clusters, self.merge_threshold)

        return ClusterResult(self.clusters, self.assignments, cluster_merges)

    def clear(self):
        self.clusters.clear()
        self.assignments.clear()
        self._counts.clear()

    def _set_and_assign(self, item_id: str, embedding: np.ndarray):
        prototype_id = self._generate_id()
        metadata = ClusterMetadata(prototype_size=1, mean_similarity=self.default_threshold, std_similarity=0.0, label=Cluster.UNLABELLED)
        cluster = Cluster(prototype_id, embedding, metadata, label=Cluster.UNLABELLED)
        self.clusters[prototype_id] = cluster
        self.assignments[item_id] = prototype_id
        self._counts[prototype_id] = 1

    def _update_and_assign(self, item_id: str, embedding: np.ndarray, cluster: Cluster):
        target_cid = cluster.prototype_id
        prev_cid = self.assignments.get(item_id)

        if prev_cid == target_cid:
            return

        old_size = self._counts.get(target_cid, 0)
        old_meta = cluster.metadata
        new_embedding = self._update_prototype_embedding(cluster.embedding, embedding, old_size)
        sim_new = float(np.dot(new_embedding, embedding))
        new_mean = (old_meta.mean_similarity * old_size + sim_new) / (old_size + 1) if old_size >= 1 else sim_new
        new_std = (np.sqrt(((old_size - 1) * old_meta.std_similarity**2 + (sim_new - old_meta.mean_similarity) * (sim_new - new_mean)) / old_size)
                   if old_size > 1 else 0.0)

        self.clusters[target_cid] = Cluster(
            target_cid,
            new_embedding,
            metadata=ClusterMetadata(
                prototype_size=old_size + 1,
                mean_similarity=new_mean,
                std_similarity=new_std,
                label=cluster.label,
            ),
            label=cluster.label,
        )
        self.assignments[item_id] = target_cid
        self._counts[target_cid] = old_size + 1

        if prev_cid and prev_cid != target_cid:
            prev_count = self._counts.get(prev_cid, 0) - 1
            if prev_count <= 0:
                if prev_cid in self.clusters:
                    del self.clusters[prev_cid]
                self._counts.pop(prev_cid, None)
            else:
                self._counts[prev_cid] = prev_count
                prev_cluster = self.clusters[prev_cid]
                prev_meta = prev_cluster.metadata
                self.clusters[prev_cid] = Cluster(
                    prev_cluster.prototype_id,
                    self._update_prototype_embedding(prev_cluster.embedding, embedding, prev_count + 1, -1),
                    metadata=ClusterMetadata(
                        prototype_size=prev_count,
                        mean_similarity=prev_meta.mean_similarity,
                        std_similarity=prev_meta.std_similarity,
                        label=prev_meta.label,
                    ),
                    label=prev_cluster.label,
                )
    
    def _generate_id(self):
        return random.randbytes(8).hex() if self.benchmarking else uuid.uuid4().hex

    def _compute_min_cluster_size(self, total_items: int) -> int:
        if total_items <= 0:
            return max(2, self.min_cluster_size)
        adaptive = max(2, int(np.sqrt(total_items)))
        return max(adaptive, self.min_cluster_size)

    def _get_threshold(self, cluster: Cluster, avg_cohesion: float, avg_cluster_size: float, min_cluster_size: int) -> float:
        mean_sim = cluster.metadata.mean_similarity
        std_sim = cluster.metadata.std_similarity
        cluster_size = cluster.metadata.prototype_size
        size_factor = cluster_size / max(1, avg_cluster_size)        
        baseline = self.default_threshold + std_sim

         # immature clusters: always loose
        if cluster_size < min_cluster_size or avg_cohesion <= 0:
            return max(baseline - std_sim, 0.0)
        x = (mean_sim - avg_cohesion) / max(1e-6, avg_cohesion)
        alpha = 1.0 / (1.0 + np.exp(-x))
        mean_candidate = mean_sim if mean_sim > baseline else baseline
        base_candidate = alpha * mean_candidate + (1.0 - alpha) * baseline
        return max(base_candidate - std_sim, 0.0) * size_factor

    def _compute_average_cluster_stats(self) -> tuple[float, float]:
        cohesions, cluster_sizes = [], []

        for c in self.clusters.values():
            size = c.metadata.prototype_size
            cluster_sizes.append(size)
            if size > 1:
                cohesions.append(c.metadata.mean_similarity)
        avg_cohesion = np.mean(cohesions) if cohesions else 0.0
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0
        return avg_cohesion, avg_cluster_size
    
    def _remove_empty_clusters(self):
        empty_clusters = [cluster_id for cluster_id in self.clusters if cluster_id not in self.assignments.values()]
        for cluster_id in empty_clusters:
             del self.clusters[cluster_id]

    def _are_clusters_stable(self, all_items: Dict[str, np.ndarray]) -> bool:
        if not self.clusters:
            return True

        cluster_ids = list(self.clusters.keys())
        cluster_embs = np.array([c.embedding for c in self.clusters.values()]) 
        item_ids = list(all_items.keys())
        item_embs = np.array([all_items[i] for i in item_ids]) 
        sims = item_embs @ cluster_embs.T
        best_idx = np.argmax(sims, axis=1)
        best_cids = [cluster_ids[i] for i in best_idx]

        for item_id, best_cid in zip(item_ids, best_cids):
            if self.assignments.get(item_id) != best_cid:
                return False
        return True

    def _tally_votes(self, neighbour_ids: list[str], embedding: np.ndarray) -> tuple[dict[str, int], dict[str, list[float]]]:
        vote_counts: dict[str, int] = {}
        vote_sims: dict[str, list[float]] = {}
        for nid in neighbour_ids:
            cid = self.assignments.get(nid)
            if cid and cid in self.clusters:
                vote_counts[cid] = vote_counts.get(cid, 0) + 1
                vote_sims.setdefault(cid, []).append(float(np.dot(embedding, self.clusters[cid].embedding)))
        return vote_counts, vote_sims

    def _reassign_by_nn_votes(self, nearest_neighbour_ids: list[str], cluster: Cluster) -> Optional[str]:
        vote_counts, vote_sims = self._tally_votes(nearest_neighbour_ids, cluster.embedding)
        total_votes = sum(vote_counts.values())
        majority_needed = (len(nearest_neighbour_ids) // 2) + 1
        other_votes = total_votes - vote_counts.get(cluster.prototype_id, 0)

        if other_votes < majority_needed:
            return None

        chosen_target_cid = self._select_top_cluster(vote_counts, vote_sims)

        if chosen_target_cid == cluster.prototype_id:
            return None

        return chosen_target_cid

    def _select_top_cluster(self, vote_counts: dict[str, int], vote_sims: dict[str, list[float]]) -> str:
        top_value = max(vote_counts.values())
        top_cids = [cid for cid, v in vote_counts.items() if v == top_value]
        if len(top_cids) == 1:
            return top_cids[0]
        return max(top_cids, key=lambda cid: float(np.mean(vote_sims.get(cid, [0.0]))))

    @staticmethod
    def _get_top_k(embedding: np.ndarray, other_embeds: np.ndarray, k=5):
        if len(other_embeds) == 0:
            return []
        sims = np.dot(other_embeds, embedding)
        k = min(k, len(sims))
        if k == 0:
            return []
        top_idx = np.argpartition(-sims, k - 1)[:k] if len(sims) > k else np.argsort(-sims)
        if len(sims) > k:
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        return top_idx
    

    @staticmethod
    def _update_prototype_embedding(current_prototype: np.ndarray, batch_embeddings: np.ndarray, current_n: int, sign: int = 1) -> np.ndarray:
        if sign not in (1, -1):
            raise ValueError("sign must be 1 (add) or -1 (remove)")
        
        batch_embeddings = np.asarray(batch_embeddings)
        if batch_embeddings.ndim == 1:
            batch_embeddings = batch_embeddings[np.newaxis, :]
        
        batch_sum = np.sum(batch_embeddings, axis=0) * sign
        updated_n = current_n + batch_embeddings.shape[0] * sign
        if updated_n <= 0:
            raise ValueError("Prototype count cannot be <= 0")
        
        updated_prototype = (current_prototype * current_n + batch_sum) / updated_n
        return updated_prototype / np.linalg.norm(updated_prototype)