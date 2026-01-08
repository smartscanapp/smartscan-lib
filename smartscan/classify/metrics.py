import numpy as np

from smartscan.classify.types import ClusterAccuracy, ClusterMetrics

from typing import Optional
import numpy as np
from smartscan.classify.types import ClusterMetrics, BaseCluster

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
            self.nearest_other_similarity = max(self.nearest_other_similarity, nearest)
        # update separation
        self.separation_margin = self.mean_similarity - (self.nearest_other_similarity or 0.0)

    def get_metrics(self) -> ClusterMetrics:
        return ClusterMetrics(
            mean_similarity=self.mean_similarity,
            min_similarity=self.min_similarity,
            max_similarity=self.max_similarity,
            nearest_other_similarity=self.nearest_other_similarity,
            separation_margin=self.separation_margin
        )


def calculate_cluster_accuracy(labelled_cluster_counts: dict[str, int], predicted_cluster_counts: dict[str, int]) -> ClusterAccuracy:
    per_cluster_acc = {cluster_id: 0 for cluster_id in labelled_cluster_counts}
    labelled_cluster_counts = dict(sorted(labelled_cluster_counts.items()))
    predicted_cluster_counts = dict(sorted(predicted_cluster_counts.items()))
    running_acc = 0.0

    for ((label, count), (_, predict_count)) in zip(labelled_cluster_counts.items(), predicted_cluster_counts.items()):
        prediction_acc = float(predict_count / count)
        per_cluster_acc[label] = prediction_acc
        running_acc += prediction_acc
        
    return ClusterAccuracy(per_cluster_acc, running_acc/len(per_cluster_acc))

