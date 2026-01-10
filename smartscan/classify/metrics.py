import numpy as np

from typing import Optional, Dict
from smartscan.classify.types import ClusterAccuracy, ClusterMetrics, Assignments, ItemId, ClusterId, Cluster

class ClusterMetricTracker:
    def __init__(self, cluster: Cluster, other_prototypes: Optional[np.ndarray] = None):
        self.cluster = cluster
        self.other_prototypes = other_prototypes

        self.n = cluster.metadata.prototype_size
        self.mean_similarity = cluster.metadata.mean_similarity or 0.0
        self.min_similarity = cluster.metadata.min_similarity
        self.max_similarity = cluster.metadata.max_similarity

        self.mean_nearest_other = cluster.metadata.nearest_other_similarity or 0.0
        self.separation_margin = cluster.metadata.separation_margin

    def add_samples(self, new_samples: np.ndarray):
        if new_samples.ndim == 1:
            new_samples = new_samples[np.newaxis, :]
        m = new_samples.shape[0]
        if m == 0:
            return

        proto = self.cluster.embedding
        sims = new_samples @ proto

        # update intra-cluster stats
        total = self.n + m
        self.mean_similarity = (self.mean_similarity * self.n + sims.sum()) / total
        self.min_similarity = np.min(sims) if self.min_similarity is None else min(self.min_similarity, np.min(sims))
        self.max_similarity = np.max(sims) if self.max_similarity is None else max(self.max_similarity, np.max(sims))
        self.n = total

        # update nearest other (sample-aware, averaged)
        if self.other_prototypes is not None and len(self.other_prototypes) > 0:
            nearest_other = np.max(new_samples @ self.other_prototypes.T, axis=1).mean()
            self.mean_nearest_other = (
                (self.mean_nearest_other * (self.n - m) + nearest_other * m) / self.n
            )
        self.separation_margin = self.mean_similarity - self.mean_nearest_other

    def get_metrics(self) -> ClusterMetrics:
        return ClusterMetrics(
            prototype_size=self.n,
            mean_similarity=self.mean_similarity,
            min_similarity=self.min_similarity,
            max_similarity=self.max_similarity,
            nearest_other_similarity=self.mean_nearest_other,
            separation_margin=self.separation_margin,
        )


def calculate_cluster_accuracy(true_labels: Dict[ItemId, str], predicted_clusters: Assignments) -> ClusterAccuracy:
    """
    Compute per-cluster and overall accuracy given true labels and predicted clusters.

    Args:
        true_labels: Mapping from item ID to true label (e.g., "btc", "forex").
        predicted_clusters: Mapping from item ID to predicted cluster ID.

    Returns:
        ClusterAccuracy
    """
    # Build contingency matrix
    contingency: Dict[str, Dict[ClusterId, int]] = {}
    for item, tlabel in true_labels.items():
        pcluster = predicted_clusters.get(item)
        if tlabel not in contingency:
            contingency[tlabel] = {}
        if pcluster is not None:
            contingency[tlabel][pcluster] = contingency[tlabel].get(pcluster, 0) + 1

    # Greedy matching for per-cluster accuracy
    matched_pred = set()
    per_cluster_acc: Dict[str, float] = {}
    for tlabel, counts in contingency.items():
        # Pick predicted cluster with largest overlap not already matched
        best_p = max(
            ((p, cnt) for p, cnt in counts.items() if p not in matched_pred),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]
        matched_pred.add(best_p)
        total = sum(counts.values())
        correct = counts.get(best_p, 0)
        per_cluster_acc[tlabel] = correct / total if total > 0 else 0.0

    # Overall accuracy
    total_correct = sum(int(sum(counts.values()) * per_cluster_acc[tlabel]) for tlabel, counts in contingency.items())
    total_items = sum(sum(counts.values()) for counts in contingency.values())
    overall_acc = total_correct / total_items if total_items > 0 else 0.0
    return ClusterAccuracy(per_cluster_acc, overall_acc)