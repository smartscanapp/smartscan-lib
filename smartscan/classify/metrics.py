import numpy as np

from typing import Optional, Dict
from smartscan.classify.types import ClusterAccuracy, ClusterMetrics, Assignments, ItemId, ClusterId, BaseCluster

class ClusterMetricTracker:
    def __init__(self, cluster: BaseCluster, other_prototypes: Optional[np.ndarray] = None):
        self.cluster = cluster
        self.other_prototypes = other_prototypes
        self.n = cluster.metadata.prototype_size
        self.mean_similarity: float = cluster.metadata.mean_similarity
        self.min_similarity: float = cluster.metadata.min_similarity
        self.max_similarity: float = cluster.metadata.max_similarity
        self.nearest_other_similarity: Optional[float] = cluster.metadata.nearest_other_similarity
        self.separation_margin:  Optional[float] = cluster.metadata.separation_margin

    def add_samples(self, new_samples: np.ndarray):
        if new_samples.ndim == 1:
            new_samples = new_samples[np.newaxis, :]
        m = new_samples.shape[0]
        if m == 0 or self.n <= 0:
            return

        sims = np.dot(new_samples, self.cluster.embedding)
        # update mean
        self.mean_similarity = (self.mean_similarity * self.n + np.sum(sims)) / (self.n + m)
        self.n += m
        # update min/max
        self.min_similarity = min(self.min_similarity, np.min(sims)) if self.min_similarity is not None else np.min(sims)
        self.max_similarity = max(self.max_similarity, np.max(sims)) if self.max_similarity is not None else np.max(sims)
        # update nearest other cluster similarity
        if self.other_prototypes is not None and len(self.other_prototypes) > 0:
            nearest = np.max(np.dot(self.other_prototypes, self.cluster.embedding))
            self.nearest_other_similarity = max(self.nearest_other_similarity or nearest, nearest)
        # update separation
        self.separation_margin = self.mean_similarity - (self.nearest_other_similarity or 0.0)

    def get_metrics(self) -> ClusterMetrics:
        return ClusterMetrics(
            prototype_size=self.n,
            mean_similarity=self.mean_similarity,
            min_similarity=self.min_similarity,
            max_similarity=self.max_similarity,
            nearest_other_similarity=self.nearest_other_similarity,
            separation_margin=self.separation_margin
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